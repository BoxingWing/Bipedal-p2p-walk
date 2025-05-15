import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import copy
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Import extensions to set up environment tasks
import AzureLoong.tasks  # noqa: F401

import configparser # export config to ini files

import torch
import tvm
from tvm import relay

def export_policy_as_jit_hw(actor_critic, path, filename, input_nums):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, filename)
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    traced_script_module = torch.jit.trace(model,torch.randn(1,input_nums))
    traced_script_module.save(path)

env_cfg = parse_env_cfg(
    args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
)
agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

# specify directory for logging experiments
log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
log_root_path = os.path.abspath(log_root_path)
print(f"[INFO] Loading experiment from directory: {log_root_path}")
resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
log_dir = os.path.dirname(resume_path)

# create isaac environment
env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

# convert to single-agent instance if required by the RL algorithm
if isinstance(env.unwrapped, DirectMARLEnv):
    env = multi_agent_to_single_agent(env)

# wrap around environment for rsl-rl
env = RslRlVecEnvWrapper(env)

print(f"[INFO]: Loading model checkpoint from: {resume_path}")

# load previously trained model
ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
ppo_runner.load(resume_path)

# obtain the trained policy for inference
policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

# export policy to onnx/jit
export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
export_policy_as_jit_hw(
    ppo_runner.alg.actor_critic, export_model_dir, "policy_hw.pt", env_cfg.observation_space
)
# export_policy_as_onnx(
#     ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
# )

print('Exported policy as jit script to: ', export_model_dir)

inputSize=env_cfg.observation_space
path_to_policy = os.path.join(export_model_dir, "policy_hw.pt")
print(path_to_policy)

model_trace=torch.jit.load(path_to_policy)
model_trace=torch.jit.trace(model_trace,torch.randn(1,inputSize)).eval()
shape_list = [("input0",((inputSize,),"float32"))]
mod,param=relay.frontend.from_pytorch(model_trace,shape_list)

# ==x64=======
path_to_policy_hw=os.path.join(export_model_dir, "policy_x64_cpu.so")

target = tvm.target.Target("llvm", host="llvm")
with tvm.transform.PassContext(opt_level=2):
    lib = relay.build(mod, target=target, params=param)
lib.export_library(path_to_policy_hw)

#==arm64=======
path_to_policy_hw_arm=os.path.join(export_model_dir, "policy_a64_cpu.so")
target = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu")
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=param)
lib.export_library(path_to_policy_hw_arm, cc='/usr/bin/aarch64-linux-gnu-g++-9')

# 创建 ConfigParser 对象
config = configparser.ConfigParser()
config.optionxform = str
path_to_policy_ini=os.path.join(export_model_dir, "policyConfig.ini")


# 添加配置段落和键值对
config.write(";[ctrl_para]")
config['ctrl_para'] = {
    'frame_stack': env_cfg.frame_stack,
    'num_single_obs': env_cfg.num_single_obs,
    'num_commands': 3, 
    'commands_stack': 2, 
    'num_commands_input': 10,
    'num_actions': env_cfg.action_space,
    'control_dt': env_cfg.sim.dt*env_cfg.decimation,
    'cycle_time': env_cfg.commands.nominal_gait_cycle_time,
    'nominal_base_height': 1.115,#env_cfg.init_state.pos[2],
    'scales_action': env_cfg.action_scale,
    'scales_lin_vel': env_cfg.normalization.obs_scales.lin_vel,
    'scales_ang_vel': env_cfg.normalization.obs_scales.ang_vel,
    'scales_quat': env_cfg.normalization.obs_scales.quat,
    'scales_dof_pos': env_cfg.normalization.obs_scales.dof_pos,
    'scales_dof_vel': env_cfg.normalization.obs_scales.dof_vel,
    'scales_feet_pos': env_cfg.normalization.obs_scales.feet_pos,
    'scales_feet_vel': env_cfg.normalization.obs_scales.feet_vel,
    'scales_feet_eul': env_cfg.normalization.obs_scales.feet_eul,
    'clip_actions': env_cfg.normalization.clip_actions,
    'clip_observations': env_cfg.normalization.clip_observations
}

# config['default_joint_angles'] = {key: str(value) for key, value in env.robot.data.default_joint_pos.items()}

# config['default_joint_stiffness'] = {f"{key}-kp": str(value) for key, value in env.robot.data.default_joint_stiffness.items()}

# config['default_joint_damping'] = {f"{key}-kd": str(value) for key, value in env.robot.data.default_joint_damping.items()}

# 导出到 ini 文件
with open(path_to_policy_ini, 'w') as configfile:
    config.write(configfile)

configfile.close()

# comment segment titles
input_file = path_to_policy_ini
output_file = input_file

with open(input_file, 'r') as infile:
    lines = infile.readlines()

with open(output_file, 'w') as configfile:
    for line in lines:
        stripped_line = line.strip()
        # 如果是段落标题，即以 [ 开始并以 ] 结束，则进行注释
        if stripped_line.startswith('[') and stripped_line.endswith(']'):
            configfile.write(f";{line}")
        else:
            configfile.write(line)

print("配置已导出到 ini 文件")



## to compare the output between jit trace and jit script
# path_to_policy = os.path.join(path, "policy_hw.pt")
# print(path_to_policy)
# model_trace=torch.jit.load(path_to_policy)
# model_trace=torch.jit.trace(model_trace,torch.randn(1,inputSize)).eval()

# path_to_policy = os.path.join(path, "policy_1.pt")
# print(path_to_policy)
# model_script=torch.jit.load(path_to_policy)
# model_script=torch.jit.script(model_script,torch.randn(1,inputSize)).eval()

# input_1 = torch.randn(1, inputSize)
# input_2 = torch.zeros(1, inputSize)

# # 使用 torch.jit.trace
# traced_model = torch.jit.trace(model_trace, input_1)
# print("Trace (input_1):", traced_model(input_1))
# print("Trace (input_2):", traced_model(input_2))

# # 使用 torch.jit.script
# scripted_model = torch.jit.script(model_script)
# print("Script (input_1):", scripted_model(input_1))
# print("Script (input_2):", scripted_model(input_2))









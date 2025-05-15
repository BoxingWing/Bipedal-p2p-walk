# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn a cart-pole and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_articulation.py

"""

"""Launch Isaac Sim Simulator first."""


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
# args_cli.headless = False
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.terrains import HfRandomUniformTerrainCfg,TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_rotate_inverse, quat_inv, quat_mul
# from isaaclab.utils.math import matrix_from_quat, euler_xyz_from_quat, quat_from_matrix, quat_inv, quat_mul

import gymnasium as gym
import os
from datetime import datetime
import torch
import numpy as np
from cmd_generators import cmd_scheduler, cmd_scheduler_tensor
from logger_text_file import TxtFileWriter

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab.utils import CircularBuffer
##
# Pre-defined configs
##
# from isaaclab_assets import CARTPOLE_CFG  # isort:skip
from AzureLoong.assets.AzureLoong import AZURELOONG_FLOAT_CFG, AZURELOONG_CFG

@torch.jit.script
def torch_rand_float_ranges(lower1:float, upper1:float, lower2:float, upper2:float, shape: tuple[int,int], device:str):
    # 确保 a < b 和 c < d
    assert lower1 < upper1 and lower2 < upper2, "Invalid ranges: a must be less than b, and c must be less than d"
    
    # 计算两个区间的长度
    range1_length = upper1 - lower1
    range2_length = upper2 - lower2
    
    # 计算总长度
    total_length = range1_length + range2_length
    
    # 生成 [0, total_length) 范围内的均匀分布随机数
    random_values = torch.rand(tuple(shape), device=device) * total_length
    
    # 将随机数映射到 [a, b) 和 [c, d) 的并集范围内
    random_vector = torch.where(
        random_values < range1_length,
        lower1 + random_values,  # 映射到 [a, b)
        lower2 + (random_values - range1_length)  # 映射到 [c, d)
    )
    
    return random_vector


COBBLESTONE_TERRAIN_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
        seed=0,
        size=(3.0, 3.0),
        border_width=0.0,
        num_rows=1,
        num_cols=1,
        sub_terrains={
            "random_rough": HfRandomUniformTerrainCfg(
                proportion=1.0, noise_range=(0.0, 0.05), noise_step=0.01, border_width=0.25
            ),
        },
    ),
)

@configclass
class RobotGymSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

    # ground plane
    # ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    # terrain = COBBLESTONE_TERRAIN_CFG

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    robot = AZURELOONG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_sensors = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=6,
        track_air_time=True,
        debug_vis=False,
    )

    # def __init__()
    #         self.txt_writer = TxtFileWriter(file_dir="text")
    #         self.txt_writer.add_item_info(name='pos_ref',length=2)
    #         self.txt_writer.add_item_info(name='pos',length=2)
    #         self.txt_writer.add_item_info(name='pos_bias',length=2)
    #         self.txt_writer.add_item_info(name='yaw_ref',length=1)
    #         self.txt_writer.add_item_info(name='eul',length=3)
    #         self.txt_writer.add_item_info(name='yaw_bias',length=1)
    #         self.txt_writer.add_item_info(name='vel_ref',length=2)
    #         self.txt_writer.add_item_info(name='base_vel',length=2)
    #         self.txt_writer.add_item_info(name='command_real',length=8)
    #         self.txt_writer.add_item_info(name='pos_ref_b',length=2)
    #         self.txt_writer.add_item_info(name='reach_time',length=1)
    #         self.txt_writer.add_item_info(name='phase_track',length=1)
    #         self.txt_writer.add_item_info(name='reach',length=1)
    #         self.txt_writer.add_item_info(name='first_reach',length=1)
    #         self.txt_writer.add_item_info(name='update_cmd',length=1)
    #         self.txt_writer.add_item_info(name='command_stack',length=8)
    #         self.txt_writer.add_item_info(name='walk',length=1)
    #         self.txt_writer.add_item_info(name='divid_time_cmd',length=1)
    #         self.txt_writer.add_item_info(name='stand_next_flag',length=1)
    #         self.txt_writer.add_item_info(name='commands',length=4)
    #         self.txt_writer.add_item_info(name='stand_rate',length=1)
    #         self.txt_writer.add_item_info(name='reach_time_wt',length=1)
    #         self.cnt_txt = 0
    #         self.txt_writer.finish_item_adding()
    #         self.enable_txt_writer = True

    #   if self.enable_txt_writer:
    #         self.txt_writer.rec_item_data('yaw_ref',yaw_des[0].cpu().numpy())
    #         self.txt_writer.rec_item_data('eul',self.base_euler_xyz[0,:].cpu().numpy())
    #         self.txt_writer.rec_item_data('yaw_bias',self.yaw_bias[0].cpu().numpy())



class RealTimeDataSaver:
    def __init__(self, filename):
        """初始化时创建一个文件或打开现有文件"""
        self.filename = filename
        self.current_line = []  # 用来暂存当前行的数据
        # 打开文件进行追加写入
        self.file = open(self.filename, 'w')

    def append_data(self, data):
        """向当前行添加数据"""
        self.current_line.append(str(data))  # 将数据转换为字符串并追加到当前行
    
    def append_data_array(self, data_array):
        """向当前行添加数据"""
        for data in data_array:
            self.current_line.append(str(data))  # 将数据转换为字符串并追加到当前行

    def write_line_to_file(self):
        """将当前行的数据写入文件并清空当前行"""
        if self.current_line:  # 确保当前行不为空
            data_line = ' '.join(self.current_line) + '\n'
            self.file.write(data_line)
            self.file.flush()  # 确保数据立即写入磁盘
            self.current_line.clear()  # 清空当前行，为下次写入做准备

    def close(self):
        """关闭文件"""
        self.file.close()


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, policy, env_cfg, sim_cfg):
    """Run the simulator."""
    # dataRecoder = RealTimeDataSaver("data_sim.txt")

    current_file_path = os.path.abspath(__file__)           # 当前文件的绝对路径
    current_dir = os.path.dirname(current_file_path)        # 当前文件所在目录
    txt_writer = TxtFileWriter(file_dir=current_dir)

    txt_writer.add_item_info(name='count',length=1)
    txt_writer.add_item_info(name='phi',length=scene.num_envs)
    txt_writer.add_item_info(name='base_pX',length=scene.num_envs)
    txt_writer.add_item_info(name='base_pY',length=scene.num_envs)
    txt_writer.add_item_info(name='base_thetaZ',length=scene.num_envs)
    txt_writer.add_item_info(name='cmd_cur_pX',length=scene.num_envs)
    txt_writer.add_item_info(name='cmd_cur_pY',length=scene.num_envs)
    txt_writer.add_item_info(name='cmd_cur_thetaZ',length=scene.num_envs)
    txt_writer.add_item_info(name='cmd_next_pX',length=scene.num_envs)
    txt_writer.add_item_info(name='cmd_next_pY',length=scene.num_envs)
    txt_writer.add_item_info(name='cmd_next_thetaZ',length=scene.num_envs)
    txt_writer.finish_item_adding()

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    robot =  scene["robot"]
    # contact_sensor_l = scene["contact_sensor_l"]
    # contact_sensor_r = scene["contact_sensor_r"]
    contact_sensors = scene["contact_sensors"]
    print("=====joint names=====")
    for i, name in enumerate(robot.data.joint_names):
        print(f"{i}: {name}")
    print("=====body names=====")
    for i, name in enumerate(robot.data.body_names):
        print(f"{i}: {name}")
    print("=====body names in contact sensors=====")
    for i, name in enumerate(contact_sensors.body_names):
        print(f"{i}: {name}")

    action = torch.zeros((scene.num_envs,env_cfg.action_space), device=sim.device,dtype=torch.float)
    dof_pos = robot.data.joint_pos
    dof_vel = robot.data.joint_vel
    root_states = robot.data.root_state_w # w.r.t the world frame !
    rigid_state = robot.data.body_state_w
    base_ang_vel = robot.data.root_ang_vel_b
    base_quat = root_states[:, 3:7]
    r, p, w = euler_xyz_from_quat(root_states[:, 3:7])
    base_euler_xyz = torch.stack((r, p, w), dim=1)
    base_euler_xyz[base_euler_xyz > torch.pi] -= 2 * torch.pi
    obs_history = CircularBuffer(max_len=env_cfg.frame_stack, batch_size=scene.num_envs, device = sim.device)
    target_q = robot.data.default_joint_pos
    target_q_filtered = robot.data.default_joint_pos

    cmd = cmd_scheduler_tensor(run_dt=sim_cfg.dt, gait_cycle_time=0.9, max_cmd_len=10, env_nums=scene.num_envs, device=sim.device)
    period = torch.ones((scene.num_envs,1), device=sim.device) * 1.
    px = torch.ones((scene.num_envs,1), device=sim.device) * 0.
    py = torch.ones((scene.num_envs,1), device=sim.device) * 0.
    thetaZ = torch.ones((scene.num_envs,1), device=sim.device) * 0.
    isStand = torch.full((scene.num_envs, 1), True, dtype=torch.bool, device=sim.device)
    cmd.add_mode(period=period,px=px,py=py,thetaZ=thetaZ,isStand=isStand)

    period = torch.ones((scene.num_envs,1), device=sim.device) * 1.8
    # px = torch.ones((scene.num_envs,1), device=sim.device) * 0.3 * 1.8
    # py = torch.ones((scene.num_envs,1), device=sim.device) * 0.
    # thetaZ = torch.ones((scene.num_envs,1), device=sim.device) * 0.
    lin_vel_x = [-0.4, 0.0, 0.0, 0.4]   # min max min max [m/s]
    lin_vel_y = [-0.4, 0.0, 0.0, 0.4]   # min max min max [m/s]
    ang_vel_yaw = [-0.3, 0.0, 0.0, 0.3] # min max min max [rad/s]
    vx_rand = torch_rand_float_ranges(lin_vel_x[0], lin_vel_x[1],lin_vel_x[2],lin_vel_x[3],(scene.num_envs, 1), device=sim.device)
    vy_rand = torch_rand_float_ranges(lin_vel_y[0], lin_vel_y[1],lin_vel_y[2],lin_vel_y[3],(scene.num_envs, 1), device=sim.device)
    wz_rand = torch_rand_float_ranges(ang_vel_yaw[0], ang_vel_yaw[1],ang_vel_yaw[2],ang_vel_yaw[3],(scene.num_envs, 1), device=sim.device)
    px = vx_rand * period
    py = vy_rand * period
    thetaZ = wz_rand * period
    isStand = torch.full((scene.num_envs, 1), False, dtype=torch.bool, device=sim.device)
    cmd.add_mode(period=period,px=px,py=py,thetaZ=thetaZ,isStand=isStand)

    # cmd.add_mode(period=2,px=0,py=0,thetaZ=0,isStand=True)
    # cmd.add_mode(period=1.8,px=0.3*1.8,py=0,thetaZ=0,isStand=False)
    # cmd.add_mode(period=1,px=0,py=0,thetaZ=0,isStand=True)
    # cmd.add_mode(period=4,px=0.3*4,py=0.1*0.4,thetaZ=0,isStand=False)
    # cmd.add_mode(period=4,px=0.3*4,py=0.1*0.4,thetaZ=1.5,isStand=False)
    # cmd.add_mode(period=2,px=0,py=0,thetaZ=0,isStand=True)

    # Simulate physics
    start_ctr = 200
    #reset robots
    root_state_ini = robot.data.default_root_state.clone()
    root_state_ini[:, :3] += scene.env_origins
    root_state_ini[:, 2] += 0.01
    robot.write_root_pose_to_sim(root_state_ini[:, :7])
    robot.write_root_velocity_to_sim(root_state_ini[:, 7:])
    # set joint positions with some noise
    joint_pos, joint_vel = (
        robot.data.default_joint_pos.clone(),
        robot.data.default_joint_vel.clone(),
    )
    # joint_pos += torch.rand_like(joint_pos) * 0.1
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    # clear internal buffers
    scene.reset()
    robot.set_joint_position_target(robot.data.default_joint_pos)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim_dt)
    # create variables
    sin_pos_tensor = torch.zeros((scene.num_envs,1), device=sim.device,dtype=torch.float)
    cos_pos_tensor = torch.zeros((scene.num_envs,1), device=sim.device,dtype=torch.float)
    sin_pos_gait_tensor = torch.zeros((scene.num_envs,1), device=sim.device,dtype=torch.float)
    cos_pos_gait_tensor = torch.zeros((scene.num_envs,1), device=sim.device,dtype=torch.float)
    cmd_cur_tensor = torch.zeros((scene.num_envs,3), device=sim.device,dtype=torch.float)
    cmd_next_tensor = torch.zeros((scene.num_envs,3), device=sim.device,dtype=torch.float)

    while simulation_app.is_running():
        # update observations
        if count>start_ctr:
            cmd.step()
        
        dof_pos = robot.data.joint_pos
        dof_vel = robot.data.joint_vel
        root_states = robot.data.root_state_w # w.r.t the world frame !
        rigid_state = robot.data.body_state_w
        base_ang_vel = robot.data.root_ang_vel_b
        base_quat = robot.data.root_link_quat_w
        r, p, w = euler_xyz_from_quat(base_quat)
        base_euler_xyz = torch.stack((r, p, w), dim=1)
        base_euler_xyz[base_euler_xyz > torch.pi] -= 2 * torch.pi

        feet_l_quat_w2f = rigid_state[:, 11,3:7]
        feet_r_quat_w2f = rigid_state[:, 12,3:7]
        quat_w2b_inv = quat_inv(base_quat)
        quat_b2f_l = quat_mul(quat_w2b_inv, feet_l_quat_w2f)
        quat_b2f_r = quat_mul(quat_w2b_inv, feet_r_quat_w2f)
        r, p, w = euler_xyz_from_quat(quat_b2f_l)
        eul_b2f_l = torch.stack((r, p, w), dim=1)
        eul_b2f_l[eul_b2f_l > torch.pi] -= 2 * torch.pi
        r, p, w = euler_xyz_from_quat(quat_b2f_r)
        eul_b2f_r = torch.stack((r, p, w), dim=1)
        eul_b2f_r[eul_b2f_r > torch.pi] -= 2 * torch.pi

        base_pos_cur = root_states[:,:3]
        feet_l_pos_w = rigid_state[:, 11,:3] - base_pos_cur
        feet_r_pos_w = rigid_state[:, 12,:3] - base_pos_cur
        feet_l_pos_b = quat_rotate_inverse(base_quat, feet_l_pos_w) - torch.tensor(env_cfg.init_state.feet_l_default_pos_b, device=sim.device)
        feet_r_pos_b = quat_rotate_inverse(base_quat, feet_r_pos_w) - torch.tensor(env_cfg.init_state.feet_r_default_pos_b, device=sim.device)

        if count % env_cfg.decimation == 0 :
            if count % 100 ==0:
                print(count)
            # sin_pos = np.sin(2 * np.pi * cmd.phi)
            # cos_pos = np.cos(2 * np.pi * cmd.phi)
            # sin_pos_gait = np.sin(2 * np.pi * cmd.phi_gait)
            # cos_pos_gait = np.cos(2 * np.pi * cmd.phi_gait)
            # cmd_cur = np.zeros((1,3))
            # cmd_next = np.zeros((1,3))
            # cmd_cur[0,0] = cmd.cmd_cur[0] * env_cfg.normalization.obs_scales.lin_vel
            # cmd_cur[0,1] = cmd.cmd_cur[1] * env_cfg.normalization.obs_scales.lin_vel
            # cmd_cur[0,2] = cmd.cmd_cur[2] * env_cfg.normalization.obs_scales.ang_vel
            # cmd_next[0,0] = cmd.cmd_next[0] * env_cfg.normalization.obs_scales.lin_vel
            # cmd_next[0,1] = cmd.cmd_next[1] * env_cfg.normalization.obs_scales.lin_vel
            # cmd_next[0,2] = cmd.cmd_next[2] * env_cfg.normalization.obs_scales.ang_vel

            # sin_pos_tensor = torch.tensor(sin_pos, device=sim.device,dtype=torch.float).expand(scene.num_envs, 1)
            # cos_pos_tensor = torch.tensor(cos_pos, device=sim.device,dtype=torch.float).expand(scene.num_envs, 1)
            # sin_pos_gait_tensor = torch.tensor(sin_pos_gait, device=sim.device,dtype=torch.float).expand(scene.num_envs, 1)
            # cos_pos_gait_tensor = torch.tensor(cos_pos_gait, device=sim.device,dtype=torch.float).expand(scene.num_envs, 1)
            # cmd_cur_tensor = torch.tensor(cmd_cur, device=sim.device,dtype=torch.float).expand(scene.num_envs, 3)
            # cmd_next_tensor = torch.tensor(cmd_next, device=sim.device,dtype=torch.float).expand(scene.num_envs, 3)

            sin_pos_tensor = torch.sin( cmd.phi * 2 * torch.pi)
            cos_pos_tensor = torch.cos( cmd.phi * 2 * torch.pi)
            sin_pos_gait_tensor = torch.sin( cmd.phi_gait * 2 * torch.pi)
            cos_pos_gait_tensor = torch.cos( cmd.phi_gait * 2 * torch.pi)
            cmd_cur_tensor[:,0] = cmd.cmd_cur[:,0] * env_cfg.normalization.obs_scales.lin_vel
            cmd_cur_tensor[:,1] = cmd.cmd_cur[:,1] * env_cfg.normalization.obs_scales.lin_vel
            cmd_cur_tensor[:,2] = cmd.cmd_cur[:,2] * env_cfg.normalization.obs_scales.ang_vel
            cmd_next_tensor[:,0] = cmd.cmd_next[:,0] * env_cfg.normalization.obs_scales.lin_vel
            cmd_next_tensor[:,1] = cmd.cmd_next[:,1] * env_cfg.normalization.obs_scales.lin_vel
            cmd_next_tensor[:,2] = cmd.cmd_next[:,2] * env_cfg.normalization.obs_scales.ang_vel

            # print(cmd.phi)
            # print(cmd._phi_pseudo)

            # update policy obs
            command_input = torch.cat(
                (sin_pos_tensor, cos_pos_tensor, sin_pos_gait_tensor, cos_pos_gait_tensor, 
                cmd_cur_tensor, cmd_next_tensor), dim=1)
            
            q = (dof_pos - robot.data.default_joint_pos) * env_cfg.normalization.obs_scales.dof_pos
            dq = dof_vel * env_cfg.normalization.obs_scales.dof_vel


            obs_buf = torch.cat((
                command_input,  # 10 = 4D(sin cos) + 6
                q,    # 12D
                dq,  # 12D
                action,   # 12D
                base_ang_vel * env_cfg.normalization.obs_scales.ang_vel,  # 3
                base_euler_xyz[:, :2] * env_cfg.normalization.obs_scales.quat,  # 2
                feet_l_pos_b * env_cfg.normalization.obs_scales.feet_pos, # 3
                feet_r_pos_b * env_cfg.normalization.obs_scales.feet_pos, # 3
                eul_b2f_l * env_cfg.normalization.obs_scales.feet_eul,
                eul_b2f_r * env_cfg.normalization.obs_scales.feet_eul  
            ), dim=-1)
            obs_history.append(obs_buf.clone())
            # print(obs_history.dtype)
    
            obs_buf_all = torch.cat([obs_history.buffer[:,i,:] for i in range(obs_history.max_length)], dim=1)
            # print(obs_buf_all)
            # get the obs of the frist robot
            action[:] = policy(obs_buf_all).detach()
            # action_numpy=action[0].detach().numpy()
            
            target_q = robot.data.default_joint_pos
            if count > start_ctr:
                target_q =  action * env_cfg.action_scale + robot.data.default_joint_pos
        
        # -- apply action to the robot
        target_q_filtered = 0.9 * target_q_filtered + 0.1 * target_q
        robot.set_joint_position_target(target_q_filtered)
        # -- write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)
        # record variables
        # txt_writer.add_item_info(name='count',length=scene.num_envs)
        # txt_writer.add_item_info(name='phi',length=scene.num_envs)
        # txt_writer.add_item_info(name='base_pX',length=scene.num_envs)
        # txt_writer.add_item_info(name='base_pY',length=scene.num_envs)
        # txt_writer.add_item_info(name='base_thetaZ',length=scene.num_envs)
        # txt_writer.add_item_info(name='cmd_cur_pX',length=scene.num_envs)
        # txt_writer.add_item_info(name='cmd_cur_pY',length=scene.num_envs)
        # txt_writer.add_item_info(name='cmd_cur_thetaZ',length=scene.num_envs)
        # txt_writer.add_item_info(name='cmd_next_pX',length=scene.num_envs)
        # txt_writer.add_item_info(name='cmd_next_pY',length=scene.num_envs)
        # txt_writer.add_item_info(name='cmd_next_thetaZ',length=scene.num_envs)

        txt_writer.rec_item_data('count',count)
        txt_writer.rec_item_data('phi',cmd.phi.squeeze().cpu().numpy())
        txt_writer.rec_item_data('base_pX',base_pos_cur[:,0].squeeze().cpu().numpy())
        txt_writer.rec_item_data('base_pY',base_pos_cur[:,1].squeeze().cpu().numpy())
        txt_writer.rec_item_data('base_thetaZ',base_euler_xyz[:,2].squeeze().cpu().numpy())
        txt_writer.rec_item_data('cmd_cur_pX',cmd.cmd_cur[:,0].squeeze().cpu().numpy())
        txt_writer.rec_item_data('cmd_cur_pY',cmd.cmd_cur[:,1].squeeze().cpu().numpy())
        txt_writer.rec_item_data('cmd_cur_thetaZ',cmd.cmd_cur[:,2].squeeze().cpu().numpy())
        txt_writer.rec_item_data('cmd_next_pX',cmd.cmd_next[:,0].squeeze().cpu().numpy())
        txt_writer.rec_item_data('cmd_next_pY',cmd.cmd_next[:,1].squeeze().cpu().numpy())
        txt_writer.rec_item_data('cmd_next_thetaZ',cmd.cmd_next[:,2].squeeze().cpu().numpy())
        txt_writer.finish_line()

        if count > 4000:
            break


        ## print infos
        # print("========")
        # print(robot.data.body_link_state_w[0, 0, :3])
        # print(robot.data.body_link_state_w[0, 11, :3])
        # print(robot.data.body_link_state_w[0, 12, :3])
        # air_time_l = contact_sensor_l.data.current_air_time
        # air_time_r = contact_sensor_r.data.current_air_time
        # contact_time_l = contact_sensor_l.data.current_contact_time
        # contact_time_r = contact_sensor_r.data.current_contact_time
        # forces_l = contact_sensor_l.data.net_forces_w
        # forces_r = contact_sensor_r.data.net_forces_w
        # print(air_time_l[0,:])
        # print(air_time_r[0,:])
        # print(forces_l[0,:])
        # print(forces_r[0,:])
        # print(air_time[0,:])
        # print(contact_time[0,:])
        # print(forces[0,:])


def main():
    """Main function."""
    # parse configuration
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
    path_to_policy = "exported/policy.pt"
    path_to_policy_full = os.path.join(log_dir,path_to_policy)
    print("================")
    print(path_to_policy_full)
    print(env_cfg.num_single_obs)

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    grandparent_dir = os.path.dirname(os.path.dirname(current_script_dir))
    logs_sim2sim_dir = os.path.join(grandparent_dir, 'logs', 'Isaac_sim')
    current_time = datetime.now().strftime("rec_%m-%d_%H-%M-%S")
    new_folder_path = os.path.join(logs_sim2sim_dir, current_time)

    try:
        os.makedirs(new_folder_path, exist_ok=True)
        print(f"文件夹已成功创建: {new_folder_path}")
    except Exception as e:
        print(f"创建文件夹时出错: {e}")

    policy = torch.jit.load(path_to_policy_full,map_location="cuda:0")

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.002, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
    # design scene
    scene_cfg = RobotGymSceneCfg(num_envs=1000, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    #pint(sim.device) # cuda 0 is default
    run_simulator(sim, scene, policy, env_cfg, sim_cfg)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
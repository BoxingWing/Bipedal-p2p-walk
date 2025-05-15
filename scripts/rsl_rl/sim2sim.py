# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
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
args_cli.headless = True
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""
import math
import numpy as np
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
import torch
from torch.utils.tensorboard import SummaryWriter # type: ignore
from termcolor import colored
import os
from datetime import datetime
import imageio
import glfw

import mujoco, mujoco_viewer
from OpenGL import GL

import matplotlib.pyplot as plt

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab.assets import Articulation

# Import extensions to set up environment tasks
import AzureLoong.tasks  # noqa: F401

isaaclab_ordered_joint_names=[
"J_hip_l_roll",
"J_hip_r_roll",
"J_hip_l_yaw",
"J_hip_r_yaw",
"J_hip_l_pitch",
"J_hip_r_pitch",
"J_knee_l_pitch",
"J_knee_r_pitch",
"J_ankle_l_pitch",
"J_ankle_r_pitch",
"J_ankle_l_roll",
"J_ankle_r_roll",
]


def arrayReorder(arrayIn, sorted_indices):
    return arrayIn[sorted_indices]

class TensorBoard_recorder:
    def __init__(self, log_dir_in):
        # self.tb_process = subprocess.Popen(['tensorboard', '--logdir', os.path.dirname(log_dir_in), '--port', '6007'])
        # time.sleep(2)
        self.writer = SummaryWriter(log_dir=log_dir_in, flush_secs=1)
        # print(colored(f'Train record dir: {self.log_dir}','green',attrs=['bold']))
        print(colored('Tensorboard activated!!! Chech http://localhost:6007/','green',attrs=['bold']))
        print(colored(f'Start command: tensorboard --port 6007 --logdir={log_dir_in}','green'))
        # 注册信号处理函数，当按下 Ctrl+C 时会调用 signal_handler
        # signal.signal(signal.SIGINT, self.signal_handler)

    # def signal_handler(self, sig, frame):
    #     print(colored(" Ctrl+C detected, terminating TensorBoard...",'red'))
    #     self.tb_process.terminate()  # 优雅地终止 TensorBoard
    #     self.tb_process.wait()       # 等待 TensorBoard 完全退出
    #     sys.exit(0)             # 退出脚本
def get_obs(data, model):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)

    base_pos = data.sensor('position').data[[0, 1, 2]].astype(np.double)
    base_lin_vel = data.sensor('linear-velocity').data[[0, 1, 2]].astype(np.double)

    feet_index_l = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"Link_ankle_l_roll")
    feet_index_r = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"Link_ankle_r_roll")
    # print("feet_index_l:", feet_index_l, ", feet_index_r:", feet_index_r)
    feet_pos = data.xpos[[feet_index_l, feet_index_r],:].astype(np.double)
    # print("xpos: ", data.xpos)
    feet_vel = data.cvel[[feet_index_l, feet_index_r],3:6].astype(np.double)
    # print("cvel: ", data.cvel[[6+feet_index_l, 6+feet_index_r],:])
    feet_quat = data.xquat[[feet_index_l, feet_index_r],:].astype(np.double)    
    # print("xquat: ", data.xquat)  

    return (q, dq, quat, v, omega, gvec, base_pos, base_lin_vel, feet_pos, feet_vel, feet_quat)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd

def quat_from_euler_xyz(roll, pitch, yaw):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return np.array([qx, qy, qz, qw])

def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])


def quat_rotate_inverse(q, v):
    """
    用四元数的逆旋转向量（q⁻¹ * v * q）
    
    Args:
        q: 单位四元数 [w, x, y, z]
        v: 3D向量 [vx, vy, vz]
    
    Returns:
        旋转后的向量 [vx', vy', vz']
    """
    q_inv = np.array([q[0], -q[1], -q[2], -q[3]])  # 四元数的逆 = [w, -x, -y, -z]
    return quat_rotate(q_inv, v)  # 用逆四元数旋转

def quat_rotate(q, v):
    """
    用四元数旋转向量（q * v * q⁻¹）
    """
    v_quat = np.array([0, v[0], v[1], v[2]])  # 向量转四元数 [0, vx, vy, vz]
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])  # 共轭四元数 q⁻¹
    # 计算 q * v * q⁻¹
    rotated = quat_multiply(quat_multiply(q, v_quat), q_conj)
    return rotated[1:]  # 返回旋转后的向量部分 [vx', vy', vz']

def quat_multiply(q1, q2):
    """
    四元数乘法 (Hamilton 乘法)
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

def quat_inv(q):
    """计算单位四元数的逆（共轭）"""
    return np.array([q[0], -q[1], -q[2], -q[3]])

class cmd_scheduler:
    def __init__(self, run_dt, gait_cycle_time):
        self._start_count = []
        self._period = []
        self._px = []
        self._py = []
        self._thetaZ = []
        self._delta_phi = []
        self._delta_phi_gait = []
        self._run_dt = run_dt
        self._gait_cycle_time = gait_cycle_time

        self.phi = 0.0
        self.phi_gait = 0.0
        self._phi_pseudo = 0.0
        self._current_idx = 0

        self.cmd_cur = [0.0, 0.0, 0.0]
        self.cmd_next = [0.0, 0.0, 0.0]
    
    def add_mode(self, period, px, py, thetaZ, isStand):
        # self._start_count.append(start_count)
        self._period.append(np.round(period/self._gait_cycle_time) * self._gait_cycle_time)
        self._px.append(px)
        self._py.append(py)
        self._thetaZ.append(thetaZ)
        if not(isStand):
            self._delta_phi.append(self._run_dt / period)
            self._delta_phi_gait.append(self._run_dt / self._gait_cycle_time)
        else:
            self._delta_phi.append(0.0)
            self._delta_phi_gait.append(0.0)
    
    def step(self):

        if self._current_idx >= len(self._px):
            self._phi_pseudo = 0.0
            self.phi = 0.0
            self.phi_gait = 0.0

            self.cmd_cur[0] = 0.0
            self.cmd_cur[1] = 0.0
            self.cmd_cur[2] = 0.0
        else:
            self._phi_pseudo += self._run_dt / self._period[self._current_idx]
            self.phi += self._delta_phi[self._current_idx]
            self.phi_gait += self._delta_phi_gait[self._current_idx]

            self.cmd_cur[0] = self._px[self._current_idx]
            self.cmd_cur[1] = self._py[self._current_idx]
            self.cmd_cur[2] = self._thetaZ[self._current_idx]

        if self._current_idx+1 >= len(self._px):
            self.cmd_next[0] = 0.0
            self.cmd_next[1] = 0.0
            self.cmd_next[2] = 0.0
        else:
            self.cmd_next[0] = self._px[self._current_idx+1]
            self.cmd_next[1] = self._py[self._current_idx+1]
            self.cmd_next[2] = self._thetaZ[self._current_idx+1]
        
        if self._phi_pseudo>=1:
            self._current_idx += 1
            self._phi_pseudo = 0.0
            self.phi = 0.0
            self.phi_gait = 0.0

def run_mujoco(policy, sim_cfg, train_cfg, recorderIn):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """

    frames=[]
    video_rec = False
    window_width = 1920
    window_height = 1080

    model = mujoco.MjModel.from_xml_path(sim_cfg.sim_config.mujoco_model_path) # type: ignore
    model.opt.timestep = sim_cfg.sim_config.dt
    data = mujoco.MjData(model) # type: ignore

    # set ini state same to the one used in training
    data.qpos[0]=0 #train_cfg.init_state.pos[0]
    data.qpos[1]=0 #train_cfg.init_state.pos[1]
    data.qpos[2]=1.13 #train_cfg.init_state.pos[2]
    quat = quat_from_euler_xyz(-0.02, 0, 0)
    data.qpos[3] = quat[3]
    data.qpos[4] = quat[0]
    data.qpos[5] = quat[1]
    data.qpos[6] = quat[2]
    target_q_pre = np.zeros(12)

    # ordered_joint_names_isaaclab = isaaclab_ordered_joint_names
    map_mujoco2lab = np.zeros((train_cfg.action_space,1), dtype=np.int32)
    map_lab2mujoco = np.zeros((train_cfg.action_space,1), dtype=np.int32)
    for i in range(train_cfg.action_space):
        map_mujoco2lab[i] = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,isaaclab_ordered_joint_names[i]) # type: ignore
    map_lab2mujoco = np.argsort(map_mujoco2lab, axis=0)
    
    default_pos_dict = train_cfg.robot_cfg.init_state.joint_pos
    for jointName, jointPos in default_pos_dict.items():
        mujoco_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,jointName) # type: ignore
        data.qpos[mujoco_id+7]=jointPos
        target_q_pre[mujoco_id]=jointPos
    for i in range(0,model.nbody):
        print("[", i, "]: ", mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i))
    # print("target_q_pre: ", target_q_pre)

    mujoco.mj_step(model, data) # type: ignore
    viewer = mujoco_viewer.MujocoViewer(model, data, width=window_width, height=window_height)
    viewer.cam.type = 1
    viewer.cam.trackbodyid = 1
    viewer.cam.distance = 4.0
    viewer.cam.azimuth = 135.
    viewer.cam.elevation = -30.

    target_q = np.zeros((train_cfg.action_space), dtype=np.double)
    action = np.zeros((train_cfg.action_space), dtype=np.double)
    target_q = data.qpos[-train_cfg.action_space:].copy()
    default_q = data.qpos[-train_cfg.action_space:].copy()

    hist_obs = deque()
    for _ in range(train_cfg.frame_stack):
        hist_obs.append(np.zeros([1, train_cfg.num_single_obs], dtype=np.double))

    cmd_deque = deque(maxlen=2)
    cmd_deque.append(np.zeros([1,3],dtype=np.double))
    cmd_deque.append(np.zeros([1,3],dtype=np.double))

    count_lowlevel = 0
    stepCount=0.
    start_ctr = 500
    
    cmd = cmd_scheduler(run_dt=sim_cfg.sim_config.dt, gait_cycle_time=0.9)
    cmd.add_mode(period=2,px=0,py=0,thetaZ=0,isStand=True)
    cmd.add_mode(period=1.8,px=0.1*1.8,py=0,thetaZ=0,isStand=False)
    cmd.add_mode(period=1,px=0,py=0,thetaZ=0,isStand=True)
    cmd.add_mode(period=1.8,px=0.0*1.8,py=-0.2*1.8,thetaZ=0,isStand=False)
    cmd.add_mode(period=1,px=0,py=0,thetaZ=0,isStand=True)
    cmd.add_mode(period=1.8,px=0.3*1.8,py=0,thetaZ=0,isStand=False)
    cmd.add_mode(period=1,px=0,py=0,thetaZ=0,isStand=True)
    cmd.add_mode(period=4,px=0.3*4,py=0.0*0.4,thetaZ=0,isStand=False)
    cmd.add_mode(period=4,px=0.3*4,py=0.0*0.4,thetaZ=1.0,isStand=False)
    cmd.add_mode(period=2,px=0,py=0,thetaZ=0,isStand=True)

    for _ in tqdm(range(int(sim_cfg.sim_config.sim_duration / sim_cfg.sim_config.dt)), desc="Simulating..."):

        # Obtain an observation
        q, dq, quat, v, omega, gvec, base_pos, base_lin_vel, feet_pos, feet_vel, feet_quat = get_obs(data, model)
        quat_mj = quat[[3, 0, 1, 2]]
        # print("q: ", q)
        q = q[-train_cfg.action_space:]
        dq = dq[-train_cfg.action_space:]

        if stepCount>start_ctr:
            cmd.step()

        # 1000hz -> 100hz
        if count_lowlevel % sim_cfg.sim_config.decimation == 0 :
            obs = np.zeros([1, train_cfg.num_single_obs], dtype=np.float32)
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi
            
            feet_l_pos_w = feet_pos[0,:] - base_pos
            feet_r_pos_w = feet_pos[1,:] - base_pos
            # print("base_pos: ", base_pos)
            # print("feet_pos: ", feet_pos)
            # print("feet_posl: ", feet_pos[0,:])
            # print("feet_posr: ", feet_pos[1,:])
            # print("feet_l_pos_w: ", feet_l_pos_w)
            # print("feet_r_pos_w: ", feet_r_pos_w)

            # print("quat_mj: ", quat_mj)
            # print("feet_l_pos_w: ", feet_l_pos_w)
            # print("feet_r_pos_w: ", feet_r_pos_w)

            feet_l_pos_b = quat_rotate_inverse(quat_mj, feet_l_pos_w) - train_cfg.init_state.feet_l_default_pos_b
            feet_r_pos_b = quat_rotate_inverse(quat_mj, feet_r_pos_w) - train_cfg.init_state.feet_r_default_pos_b

            # print("feet_l_pos_b: ", feet_l_pos_b)
            # print("feet_r_pos_b: ", feet_r_pos_b)

            # print("base_lin_vel: ", base_lin_vel)
            # print("feet_vel: ", feet_vel[0,:])
            # print("feet_vel: ", feet_vel[1,:])
            # print("feet_l_vel_w: ", feet_l_vel_w, "\n, feet_r_vel_w:", feet_r_vel_w)
            feet_l_vel_w = feet_vel[0,:] - base_lin_vel
            feet_r_vel_w = feet_vel[1,:] - base_lin_vel
            feet_l_vel_b = quat_rotate_inverse(quat_mj, feet_l_vel_w)
            feet_r_vel_b = quat_rotate_inverse(quat_mj, feet_r_vel_w)
            # print("feet_l_vel_b: ", feet_l_vel_b)
            # print("feet_r_vel_b: ", feet_r_vel_b)

            # print("feet_quat[0,:]: ", feet_quat[0,:])
            # print("feet_quat[1,:]: ", feet_quat[1,:])
            quat_w2b_inv = quat_inv(quat_mj)
            # print("quat_w2b_inv: ", quat_w2b_inv)
            # print("feet_quat: ", feet_quat[0,:])
            # print("feet_quat: ", feet_quat[1,:])
            quat_b2f_l = quat_multiply(quat_w2b_inv, feet_quat[0,:])
            quat_b2f_r = quat_multiply(quat_w2b_inv, feet_quat[1,:])
            # print("quat_b2f_l: ", quat_b2f_l)
            # print("quat_b2f_r: ", quat_b2f_r)
            eul_b2f_l = quaternion_to_euler_array(quat_b2f_l[[1,2,3,0]])
            eul_b2f_r = quaternion_to_euler_array(quat_b2f_r[[1,2,3,0]])
            # print("eul_b2f_l: ", eul_b2f_l)
            # print("eul_b2f_r: ", eul_b2f_r)

            # print("\n")

            obs[0, 0] = math.sin(2 * math.pi * cmd.phi )
            obs[0, 1] = math.cos(2 * math.pi * cmd.phi )
            obs[0, 2] = math.sin(2 * math.pi * cmd.phi_gait )
            obs[0, 3] = math.cos(2 * math.pi * cmd.phi_gait )
            obs[0, 4] = cmd.cmd_cur[0] * train_cfg.normalization.obs_scales.lin_vel
            obs[0, 5] = cmd.cmd_cur[1] * train_cfg.normalization.obs_scales.lin_vel
            obs[0, 6] = cmd.cmd_cur[2] * train_cfg.normalization.obs_scales.ang_vel
            obs[0, 7] = cmd.cmd_next[0] * train_cfg.normalization.obs_scales.lin_vel
            obs[0, 8] = cmd.cmd_next[1] * train_cfg.normalization.obs_scales.lin_vel
            obs[0, 9] = cmd.cmd_next[2] * train_cfg.normalization.obs_scales.ang_vel
            obs[0, 10:22] = arrayReorder(q - default_q, map_mujoco2lab).flatten() * train_cfg.normalization.obs_scales.dof_pos
            obs[0, 22:34] = arrayReorder(dq, map_mujoco2lab).flatten() * train_cfg.normalization.obs_scales.dof_vel
            obs[0, 34:46] = action
            obs[0, 46:49] = omega * train_cfg.normalization.obs_scales.ang_vel
            obs[0, 49:51] = eu_ang[:2] * train_cfg.normalization.obs_scales.quat

            obs[0, 51:54] = feet_l_pos_b * train_cfg.normalization.obs_scales.feet_pos
            obs[0, 54:57] = feet_r_pos_b * train_cfg.normalization.obs_scales.feet_pos
            obs[0, 57:60] = eul_b2f_l * train_cfg.normalization.obs_scales.feet_eul
            obs[0, 60:63] = eul_b2f_r * train_cfg.normalization.obs_scales.feet_eul            
            # obs[0, 57:60] = feet_l_vel_b * train_cfg.normalization.obs_scales.feet_vel
            # obs[0, 60:63] = feet_r_vel_b * train_cfg.normalization.obs_scales.feet_vel
            # obs[0, 63:66] = eul_b2f_l * train_cfg.normalization.obs_scales.feet_eul
            # obs[0, 66:69] = eul_b2f_r * train_cfg.normalization.obs_scales.feet_eul            
            # print(f"obs: \n {obs}")
            obs = np.clip(obs, -train_cfg.normalization.clip_observations, train_cfg.normalization.clip_observations)

            hist_obs.append(obs)
            hist_obs.popleft()

            policy_input = np.zeros([1, train_cfg.observation_space], dtype=np.float32)

            for i in range(train_cfg.frame_stack):
                policy_input[0, i * train_cfg.num_single_obs : (i + 1) * train_cfg.num_single_obs] = hist_obs[i][0, :]
            
            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
            action = np.clip(action, -train_cfg.normalization.clip_actions, train_cfg.normalization.clip_actions)
            # action = lpf.run(action) # for now not good, no use to supress feet oscillation
            if count_lowlevel>start_ctr:
                target_q = (arrayReorder(action, map_lab2mujoco).flatten() * train_cfg.action_scale + default_q)   # * cfg.action_scale
            else:
                target_q = default_q   # * cfg.action_scale
            
        target_dq = np.zeros((train_cfg.action_space), dtype=np.double)
        # Generate PD control
        # target_q_pre = 0.05*target_q + 0.95*target_q_pre
        target_q_pre = target_q
        
        tau = pd_control(target_q_pre, q, sim_cfg.robot_config.kps,
                        target_dq, dq, sim_cfg.robot_config.kds)  # Calc torques
        tau = np.clip(tau, -sim_cfg.robot_config.tau_limit, sim_cfg.robot_config.tau_limit)  # Clamp torques
        data.ctrl = tau

        # for i in range(12):
        #     # 单独记录目标值
        #     recorder.writer.add_scalar(f"joint_{i+1}/target", target_q[i], stepCount)
        #     # 单独记录当前值
        #     recorder.writer.add_scalar(f"joint_{i+1}/current", q[i], stepCount)

        for i in range(12):
            recorderIn.writer.add_scalars(
            f"joint pos {i+1}",  # 为每个关节创建一个单独的标签，但不会创建多个文件夹
            {'target': target_q[i], 'current': q[i], 'torque': tau[i]}, 
            stepCount
            )

        mujoco.mj_step(model, data) # type: ignore
        viewer.render()
        if video_rec and count_lowlevel% ( (1.0 / sim_cfg.sim_config.dt) // 60)==0 :
            width, height = glfw.get_framebuffer_size(viewer.window)
            img = np.zeros((height, width, 3), dtype=np.uint8)
            GL.glReadPixels(0, 0, width, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, img)
            frames.append(np.flipud(img))
        count_lowlevel += 1
        stepCount+=1

        # phi += delta_phi
        # phi_gait += delta_phi_gait

        # if phi>1:
        #     phi = 0
        #     delta_phi = 0
        euler=quaternion_to_euler_array(quat)  
    video_filename = 'sim.mp4'
    if video_rec:
        imageio.mimsave(video_filename, frames, fps=60)
    recorderIn.writer.close()
    viewer.close()

def main():
    """Play with RSL-RL agent."""
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
    logs_sim2sim_dir = os.path.join(grandparent_dir, 'logs', 'sim2sim')
    current_time = datetime.now().strftime("rec_%m-%d_%H-%M-%S")
    new_folder_path = os.path.join(logs_sim2sim_dir, current_time)

    try:
        os.makedirs(new_folder_path, exist_ok=True)
        print(f"文件夹已成功创建: {new_folder_path}")
    except Exception as e:
        print(f"创建文件夹时出错: {e}")

    recorder=TensorBoard_recorder(new_folder_path)

    policy = torch.jit.load(path_to_policy_full)

    class Sim2simCfg_Loong():
        class sim_config:
            mujoco_model_path = os.path.abspath("source/AzureLoong/AzureLoong/assets/Robots/AzureLoong_scene.xml")
            print(mujoco_model_path)
            sim_duration = 30.0
            dt = 0.001
            decimation = 10
        class robot_config:
            kps = np.array([400, 300, 400, 400, 100, 100,  400, 300, 400, 400, 100, 100], dtype=np.double)
            kds = np.array([1., 1., 1., 1., 0.25, 0.25,      1., 1., 1., 1., 0.25, 0.25], dtype=np.double)
            # kps = np.array([0, 0, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0], dtype=np.double)
            # kds = np.array([0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0], dtype=np.double)
            tau_limit = np.array([400, 50, 400, 400, 60, 60, 400, 50, 400, 400, 60, 60], dtype=np.double)
    
    run_mujoco(policy, Sim2simCfg_Loong(), env_cfg, recorderIn=recorder)

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

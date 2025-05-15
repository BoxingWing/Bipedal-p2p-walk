import numpy as np
import torch
from isaaclab.utils.math import euler_xyz_from_quat, quat_rotate_inverse, quat_apply, wrap_to_pi

# generate p2p commands for 1 robot
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
        period_new = np.round(period/self._gait_cycle_time) * self._gait_cycle_time
        self._period.append(period_new)
        self._px.append(px)
        self._py.append(py)
        self._thetaZ.append(thetaZ)
        if not(isStand):
            self._delta_phi.append(self._run_dt / period_new)
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

# generate p2p cmds for parallel envs, tensor is used
class cmd_scheduler_tensor:
    def __init__(self, run_dt, gait_cycle_time, max_cmd_len, env_nums, device):
        self._env_nums = env_nums
        self._device = device
        self._store_start_idx = 0
        self._store_end_idx = 0
        self._period = torch.zeros((self._env_nums,max_cmd_len), device=self._device)
        self._px = torch.zeros((self._env_nums,max_cmd_len), device=self._device)
        self._py = torch.zeros((self._env_nums,max_cmd_len), device=self._device)
        self._thetaZ = torch.zeros((self._env_nums,max_cmd_len), device=self._device)
        self._delta_phi = torch.zeros((self._env_nums,max_cmd_len), device=self._device)
        self._delta_phi_gait = torch.zeros((self._env_nums,max_cmd_len), device=self._device)
        self._run_dt = run_dt
        self._gait_cycle_time = gait_cycle_time

        self.phi = torch.zeros((self._env_nums,1), device=self._device)
        self.phi_gait = torch.zeros((self._env_nums,1), device=self._device)
        self._phi_pseudo = torch.zeros((self._env_nums,1), device=self._device)
        self._read_current_idx = torch.zeros(self._env_nums, device=self._device, dtype=torch.int)

        self.cmd_cur = torch.zeros((self._env_nums,3), device=self._device)
        self.cmd_next = torch.zeros((self._env_nums,3), device=self._device)
    
    # all inputs must share the same tensor shape
    def add_mode(self, period, px, py, thetaZ, isStand):
        col_add = period.shape[1]
        # self._start_count.append(start_count)
        period_new = torch.round(period / self._gait_cycle_time) * self._gait_cycle_time
        self._period[:,self._store_start_idx:self._store_start_idx+col_add] = period_new
        self._px[:,self._store_start_idx:self._store_start_idx+col_add] = px
        self._py[:,self._store_start_idx:self._store_start_idx+col_add] = py
        self._thetaZ[:,self._store_start_idx:self._store_start_idx+col_add] = thetaZ

        delta_phi = self._run_dt / period_new
        delta_phi_gait = self._run_dt / self._gait_cycle_time

        self._delta_phi[:,self._store_start_idx:self._store_start_idx+col_add] = torch.where(isStand, 0.0, delta_phi)
        self._delta_phi_gait[:,self._store_start_idx:self._store_start_idx+col_add] = torch.where(isStand, 0.0, delta_phi_gait)
        
        self._store_start_idx += col_add
        # print(self._store_start_idx)
        # print(self._period)
        # print(self._px)
        # print(self._py)
        # print(self._thetaZ)
        # print(self._delta_phi)
        # print(self._delta_phi_gait)
    
    def step(self):
        idx_stand = self._read_current_idx >= self._store_start_idx
        idx_walk = self._read_current_idx < self._store_start_idx
        # print('read_current_idx')
        # print(self._read_current_idx)

        # print(idx_walk)
        if idx_walk.any():
            self._phi_pseudo[idx_walk,0] += self._run_dt / self._period[idx_walk,self._read_current_idx]
            self.phi[idx_walk,0] += self._delta_phi[idx_walk,self._read_current_idx]
            self.phi_gait[idx_walk,0] += self._delta_phi_gait[idx_walk,self._read_current_idx]

            self.cmd_cur[idx_walk,0] = self._px[idx_walk,self._read_current_idx]
            self.cmd_cur[idx_walk,1] = self._py[idx_walk,self._read_current_idx]
            self.cmd_cur[idx_walk,2] = self._thetaZ[idx_walk,self._read_current_idx]

        if idx_stand.any():
            self._phi_pseudo[idx_stand,0] = 0.
            self.phi[idx_stand,0] = 0.
            self.phi_gait[idx_stand,0] = 0.

            self.cmd_cur[idx_stand,0] = 0.
            self.cmd_cur[idx_stand,1] = 0.
            self.cmd_cur[idx_stand,2] = 0.
        
        idx_stand_next = self._read_current_idx+1 >= self._store_start_idx
        idx_walk_next = self._read_current_idx+1 < self._store_start_idx
        

        if idx_stand_next.any():
            self.cmd_next[idx_stand_next,0] = 0.
            self.cmd_next[idx_stand_next,1] = 0.
            self.cmd_next[idx_stand_next,2] = 0.

        if idx_walk_next.any():
            self.cmd_next[idx_walk_next,0] = self._px[idx_walk_next,self._read_current_idx+1]
            self.cmd_next[idx_walk_next,1] = self._py[idx_walk_next,self._read_current_idx+1]
            self.cmd_next[idx_walk_next,2] = self._thetaZ[idx_walk_next,self._read_current_idx+1]

        finish_idx = (self._phi_pseudo >=1).squeeze(-1)
        self._read_current_idx[finish_idx] += 1
        self._phi_pseudo[finish_idx] = 0.0
        self.phi[finish_idx,0] = 0.0
        self.phi_gait[finish_idx,0] = 0.0

        # if self._current_idx+1 >= len(self._px):
        #     self.cmd_next[0] = 0.0
        #     self.cmd_next[1] = 0.0
        #     self.cmd_next[2] = 0.0
        # else:
        #     self.cmd_next[0] = self._px[self._current_idx+1]
        #     self.cmd_next[1] = self._py[self._current_idx+1]
        #     self.cmd_next[2] = self._thetaZ[self._current_idx+1]
        
        # if self._phi_pseudo>=1:
        #     self._current_idx += 1
        #     self._phi_pseudo = 0.0
        #     self.phi = 0.0
        #     self.phi_gait = 0.0
    
    @property
    def cmd_count(self):
        """Number of joints in articulation."""
        return self._read_current_idx
    

class euler_expander:
    def __init__(self, num_envs, device):
        self._num_envs = num_envs
        self._device = device
        self.euler_xyz = torch.zeros((self._num_envs,3), device=self._device)
        self.euler_xyz_multiloop = torch.zeros((self._num_envs,3), device=self._device)
        self.base_yaw_loopCount = torch.zeros((self._num_envs,1), device=self._device)
    
    def reset_loop_count(self):
        self.base_yaw_loopCount *= 0.0
    
    def cal(self, quat):
        base_euler_cur = self.get_euler_xyz(quat)
        
        ids = (base_euler_cur[:,2] - self.euler_xyz[:,2]) < -6.0
        self.base_yaw_loopCount[ids] += 1
        ids = (base_euler_cur[:,2] - self.euler_xyz[:,2]) > 6.0
        self.base_yaw_loopCount[ids] -= 1

        self.euler_xyz = base_euler_cur.clone()
        self.euler_xyz_multiloop = base_euler_cur.clone()
        self.euler_xyz_multiloop[:,2] = self.euler_xyz_multiloop[:,2] + self.base_yaw_loopCount * torch.pi * 2.0
    

    def get_euler_xyz(self, quat):
        r, p, w = euler_xyz_from_quat(quat)
        # stack r, p, w in dim1
        euler_xyz = torch.stack((r, p, w), dim=1)
        euler_xyz[euler_xyz > torch.pi] -= 2 * torch.pi
        return euler_xyz
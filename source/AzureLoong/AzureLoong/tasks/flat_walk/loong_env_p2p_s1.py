from .base_scripts.loong_env_base import LoongBaseEnv, torch_rand_float, torch_rand_float_ranges
from .base_scripts.loong_cfg_base import LoongEnvBaseCfg
import torch
from isaaclab.utils.math import euler_xyz_from_quat, quat_rotate_inverse, quat_apply, wrap_to_pi, quat_rotate, quat_apply_yaw, quat_from_euler_xyz
from isaaclab.utils.math import matrix_from_quat, euler_xyz_from_quat, quat_from_matrix, quat_inv, quat_mul, yaw_quat
from isaaclab.utils import CircularBuffer

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

class LoongEnvP2PS1(LoongBaseEnv):
    cfg: LoongEnvBaseCfg

    def __init__(self, cfg: LoongEnvBaseCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.last_feet_z = self.cfg.init_state.ankle_height
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.phi = torch.zeros(self.num_envs, dtype=torch.float, device=self.device) # phase signal for each env
        self.last_phi = torch.zeros(self.num_envs, dtype=torch.float, device=self.device) # phase signal for each env
        self.delta_phi = torch.zeros(self.num_envs, dtype=torch.float, device=self.device) # delta phase signal for each env in each time step
        self.phi_gait = torch.zeros(self.num_envs, dtype=torch.float, device=self.device) # phase signal for each env
        self.last_phi_gait = torch.zeros(self.num_envs, dtype=torch.float, device=self.device) # phase signal for each env
        self.delta_phi_gait = torch.zeros(self.num_envs, dtype=torch.float, device=self.device) # delta phase signal for each env in each time step
        self.time_count = torch.zeros(self.num_envs, dtype=torch.float, device=self.device) # time count for mode transition
        self.mode_period_next = torch.zeros(self.num_envs, dtype=torch.float, device=self.device) # next mode period
        self.stand_flag = torch.ones(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False) # flag indicates whether in stand mode
        self.base_euler_xyz_old = torch.zeros((self.num_envs,3), dtype=torch.float, device=self.device) # base euler angle of last sim step
        self.start_base_cmd = torch.zeros((self.num_envs,3), dtype=torch.float, device=self.device)
        self.start_base_quat = torch.zeros((self.num_envs,4), dtype=torch.float, device=self.device)
        self.base_vel_cmd_last = torch.zeros((self.num_envs,3), dtype=torch.float, device=self.device)
        self.vel_mask = torch.ones((self.num_envs, 3), device=self.device)
        self.focus_on_stepping = False
        if cfg.focus_on_stepping == True :
            self.focus_on_stepping = True

        self.feZ_swing = torch.zeros((self.num_envs, 2), device=self.device)
        self.dfeZ_swing = torch.zeros((self.num_envs, 2), device=self.device)
        self.root_pos_ini = torch.zeros((self.num_envs,3), dtype=torch.float, device=self.device)
        self.stance_mask_scheduled = torch.zeros((self.num_envs, 2), device=self.device)
        self.last_stance_mask_scheduled = torch.zeros((self.num_envs, 2), device=self.device)
        self.contact_mask = torch.zeros((self.num_envs, 2), device=self.device)
        self.commands_last = torch.zeros((self.num_envs, 3), device=self.device)
        
        # self.dof_pos_l_LF = torch.zeros((self.num_envs,6), device=self.device)
        # self.dof_pos_r_LF = torch.zeros((self.num_envs,6), device=self.device)
        self.virtual_leg_l_LF = torch.zeros((self.num_envs,3), device=self.device)
        self.virtual_leg_r_LF = torch.zeros((self.num_envs,3), device=self.device)
        self.virtual_leg_l_TD = torch.zeros((self.num_envs,3), device=self.device)
        self.virtual_leg_r_TD = torch.zeros((self.num_envs,3), device=self.device)
        self.virtual_leg_l_Mid = torch.zeros((self.num_envs,3), device=self.device) # virtual leg in mid phase, update both in swing and stance
        self.virtual_leg_r_Mid = torch.zeros((self.num_envs,3), device=self.device) 
        # self.dof_knee_swing = torch.zeros((self.num_envs,2), device=self.device)
        self.event_TD = torch.zeros((self.num_envs,2), dtype=torch.bool, device=self.device) # only True when the touch-down event happens
        self.event_TD_real = torch.zeros((self.num_envs,2), dtype=torch.bool, device=self.device) # only True when the touch-down event happens
        self.event_LF = torch.zeros((self.num_envs,2), dtype=torch.bool, device=self.device) # only True when the leave-off event happens
        self.first_leg = torch.ones(self.num_envs, dtype=torch.bool, device=self.device) # only True when the leave-off event happens
        self.base_pos_des_end = self.root_states[:, :2]
        self.base_pos_des = self.root_states[:, :2]
        self.base_pos_cur = self.root_states[:, :2]
        self.base_lin_vel_des_w = torch.zeros((self.num_envs,3), device=self.device)
        self.base_thetaZ_des_end = self.base_euler_xyz_multiLoop[:, 2]
        all_env = torch.ones(self.num_envs, device=self.device)
        self.vxy_w_history = CircularBuffer(max_len=3, batch_size=self.num_envs,device=self.device)
        self.vxy_w_des_history = CircularBuffer(max_len=3, batch_size=self.num_envs,device=self.device)
        for _ in range(self.vxy_w_history.max_length):
            self.vxy_w_history.append(torch.zeros(
                self.num_envs, 3, dtype=torch.float, device=self.device))
            self.vxy_w_des_history.append(torch.zeros(
                self.num_envs, 3, dtype=torch.float, device=self.device))

        self._reset_idx(all_env)
        self._check_and_resample_commands(all_env,True)
        self._get_observations()
        if self.num_envs<20: # enable log when play with fewer envs
            self.dataRecoder = RealTimeDataSaver("data.txt")
    
    def  _get_phase(self):
        '''
        update phase, stance_mask, last_stance_mask, dof_pos_LF, and events relevent things
        '''
        # cycle_time = self.cfg.rewards.cycle_time
        # phase = self.episode_length_buf * self.dt / cycle_time
        self.last_phi[:] = self.phi[:]
        self.phi += self.delta_phi * torch.logical_not(self.stand_flag)
        self.time_count += self.delta_phi
        self.last_phi_gait[:] = self.phi_gait[:]
        self.phi_gait += self.delta_phi_gait * torch.logical_not(self.stand_flag)
        
        sin_pos = torch.sin(2 * torch.pi * self.phi_gait)
        self.last_stance_mask_scheduled[:] = self.stance_mask_scheduled[:]
        # print("self.commands: ", self.commands[:,1])
        # print("self.first_leg: ", self.first_leg)
        # print("\n")
        # left foot stance
        self.stance_mask_scheduled[:, 0] = torch.where(self.first_leg, sin_pos >= 0, sin_pos < 0)
        # right foot stance
        self.stance_mask_scheduled[:, 1] = torch.where(self.first_leg, sin_pos < 0, sin_pos >= 0)
        # Double support phase
        self.stance_mask_scheduled[torch.abs(sin_pos) < 0.01, :] = 1

        l_hip_yaw_id = self.cfg.init_state.l_hip_yaw_id
        l_ankle_roll_id = self.cfg.init_state.l_ankle_roll_id
        r_hip_yaw_id = self.cfg.init_state.r_hip_yaw_id
        r_ankle_roll_id = self.cfg.init_state.r_ankle_roll_id
        '''
        =====joint names=====
        0: J_hip_l_roll
        1: J_hip_r_roll
        2: J_hip_l_yaw
        3: J_hip_r_yaw
        4: J_hip_l_pitch
        5: J_hip_r_pitch
        6: J_knee_l_pitch
        7: J_knee_r_pitch
        8: J_ankle_l_pitch
        9: J_ankle_r_pitch
        10: J_ankle_l_roll
        11: J_ankle_r_roll
        '''
        indices = torch.logical_and(self.last_stance_mask_scheduled, torch.logical_not(self.stance_mask_scheduled))
        # self.dof_pos_l_LF[indices[:,0]] = self.dof_pos[indices[:,0],[0,2,4,6,8,10]]
        # self.dof_pos_r_LF[indices[:,1]] = self.dof_pos[indices[:,1],[1,3,5,7,9,11]]
        self.virtual_leg_l_LF[indices[:,0],:] = quat_rotate_inverse(self.base_quat[indices[:,0],:], self.rigid_state[indices[:,0], l_ankle_roll_id, :3]-self.rigid_state[indices[:,0], l_hip_yaw_id, :3])
        self.virtual_leg_r_LF[indices[:,1],:] = quat_rotate_inverse(self.base_quat[indices[:,1],:], self.rigid_state[indices[:,1], r_ankle_roll_id, :3]-self.rigid_state[indices[:,1], r_hip_yaw_id, :3])
        self.event_LF[indices] = True
        self.event_LF[torch.logical_not(indices)] = False

        indices = torch.logical_and(self.stance_mask_scheduled, torch.logical_not(self.last_stance_mask_scheduled))
        self.virtual_leg_l_TD[indices[:,0],:] = quat_rotate_inverse(self.base_quat[indices[:,0],:], self.rigid_state[indices[:,0], l_ankle_roll_id, :3]-self.rigid_state[indices[:,0], l_hip_yaw_id, :3])
        self.virtual_leg_r_TD[indices[:,1],:] = quat_rotate_inverse(self.base_quat[indices[:,1],:], self.rigid_state[indices[:,1], r_ankle_roll_id, :3]-self.rigid_state[indices[:,1], r_hip_yaw_id, :3])
        self.event_TD[indices] = True
        self.event_TD[torch.logical_not(indices)] = False

        phi_Tmp = torch.fmod(self.phi_gait*2,1)
        phi_last_Tmp = torch.fmod(self.last_phi_gait*2,1)
        indices = torch.logical_and(phi_Tmp>0.5, phi_last_Tmp<=0.5)
        self.virtual_leg_l_Mid[indices,:] = quat_rotate_inverse(self.base_quat[indices,:], self.rigid_state[indices, l_ankle_roll_id, :3]-self.rigid_state[indices, l_hip_yaw_id, :3])
        self.virtual_leg_r_Mid[indices,:] = quat_rotate_inverse(self.base_quat[indices,:], self.rigid_state[indices, r_ankle_roll_id, :3]-self.rigid_state[indices, r_hip_yaw_id, :3])

    def compute_feZ_swing(self):
        phiTmp = torch.fmod(self.phi_gait * 2, 1)
        y0 = torch.zeros(self.num_envs, device=self.device)
        yM = torch.ones_like(y0) * self.cfg.rewards.target_feet_height
        ye = torch.zeros_like(y0)
        phiM = torch.ones_like(y0) * 0.5

        # Precompute powers of phiM
        phiM_sq = torch.pow(phiM, 2)
        phiM_cubed = torch.pow(phiM, 3)
        denominator = 3 * phiM - 3 * phiM_sq + phiM_cubed - 1

        # Compute coefficients
        a0 = y0.clone()
        a1 = torch.zeros_like(a0)
        a2 = -(3.0 * (y0 - yM)) / phiM_sq
        a3 = (2.0 * (y0 - yM)) / phiM_cubed
        b0 = -(yM - 3 * phiM * yM + 3 * phiM_sq * ye - phiM_cubed * ye) / denominator
        b1 = -(6 * (phiM * yM - phiM * ye)) / denominator
        b2 = (3 * (yM - ye) * (phiM + 1)) / denominator
        b3 = -(2 * (yM - ye)) / denominator

        # Compute y1, y2, dy1, dy2
        y1 = a0 + a1 * phiTmp + a2 * torch.pow(phiTmp, 2) + a3 * torch.pow(phiTmp, 3)
        y2 = b0 + b1 * phiTmp + b2 * torch.pow(phiTmp, 2) + b3 * torch.pow(phiTmp, 3)
        dy1 = a1 + 2 * a2 * phiTmp + 3 * a3 * torch.pow(phiTmp, 2)
        dy2 = b1 + 2 * b2 * phiTmp + 3 * b3 * torch.pow(phiTmp, 2)

        # Use torch.where for efficient indexing
        y = torch.where(phiTmp <= phiM, y1, y2)
        dy = torch.where(phiTmp <= phiM, dy1, dy2) * (1.0 / self.cfg.commands.nominal_gait_cycle_time * 2)

        # Apply swing mask
        swing_mask = 1 - self.stance_mask_scheduled
        self.feZ_swing = y.unsqueeze(1) * swing_mask
        self.dfeZ_swing = dy.unsqueeze(1) * swing_mask
    
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(63, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        dof_pos_noise = torch.ones(12, device=self.device) * noise_scales.dof_pos * self.obs_scales.dof_pos
        dof_vel_noise = torch.ones(12, device=self.device) * noise_scales.dof_vel * self.obs_scales.dof_vel
        # increase noise for ankle
        dof_pos_noise[8:12] *= 3.
        dof_vel_noise[8:12] *= 3.

        noise_vec[0: 10] = 0.  # commands
        noise_vec[10: 22] = dof_pos_noise
        noise_vec[22: 34] = dof_vel_noise
        noise_vec[34: 46] = 0.  # previous actions
        noise_vec[46: 49] = noise_scales.ang_vel * self.obs_scales.ang_vel   # ang vel
        noise_vec[49: 51] = noise_scales.quat * self.obs_scales.quat         # euler x,y
        noise_vec[51: 57] = noise_scales.feet_pos * self.obs_scales.feet_pos         # euler x,y
        noise_vec[57: 63] = noise_scales.feet_eul * self.obs_scales.feet_eul
        # increase noise for ankle

        # noise_vec[57: 63] = noise_scales.feet_vel * self.obs_scales.feet_vel
        # noise_vec[63: 69] = noise_scales.feet_eul * self.obs_scales.feet_eul

        return noise_vec
    
    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        
        self.start_base_cmd[env_ids, :2] = self.terrain.env_origins[env_ids,:2]
        self.start_base_cmd[env_ids, 2] = 0.0
        self.start_base_quat[env_ids, :] = self.root_states[env_ids,3:7]
        self.base_pos_des_end[env_ids, :2] = self.terrain.env_origins[env_ids,:2]
        self.base_thetaZ_des_end[env_ids] = 0.0
        self.base_vel_cmd_last[env_ids, :] = 0.0

        self.commands_last[env_ids,:] *= 0.0

        # self.commands[env_ids,0]=4.0
        # self.commands[env_ids,1]=0.0
        # self.commands[env_ids,2]=0.0

        # self.phi[env_ids] = 0.0
        # self.last_phi[env_ids] = 0.0
        # self.phi[env_ids] = 1.5 # 
        # self.delta_phi[env_ids] = 1.0 * self.dt / 10.0

        # will cause right leg wont lift, dont know why
        # random_values = torch.rand(len(env_ids), device=self.device)
        # selected_env_ids = env_ids[random_values > 0.5]
        # unselected_env_ids = env_ids[random_values <= 0.5]
        # self.stand_flag[selected_env_ids] = False
        # self.stand_flag[unselected_env_ids] = True
        # self.delta_phi[unselected_env_ids] = 0
        # self.delta_phi[selected_env_ids] = 1.0 * self.dt / self.cfg.rewards.cycle_time

        self.base_euler_xyz_old[env_ids,:]=self.base_euler_xyz[env_ids,:]
        self.root_pos_ini[env_ids,:3] = self.root_states[env_ids,:3]

        # print("self.contact_mask shape:", self.contact_mask.shape)
        # print("env_ids:", env_ids)

        self.contact_mask[env_ids,:] = False
        
        for i in range(self.obs_history.max_length):
            self.obs_history._buffer[i, env_ids, :] = 0.0
        for i in range(self.critic_history.max_length):
            self.critic_history._buffer[i, env_ids, :] = 0.0
        for i in range(self.vxy_w_history.max_length):
            self.vxy_w_history._buffer[i, env_ids, :] = 0.0
            self.vxy_w_des_history._buffer[i, env_ids, :] = 0.0
        '''
        =====joint names=====
        0: J_hip_l_roll
        1: J_hip_r_roll
        2: J_hip_l_yaw
        3: J_hip_r_yaw
        4: J_hip_l_pitch
        5: J_hip_r_pitch
        6: J_knee_l_pitch
        7: J_knee_r_pitch
        8: J_ankle_l_pitch
        9: J_ankle_r_pitch
        10: J_ankle_l_roll
        11: J_ankle_r_roll
        =====body names=====
        0: base_link
        1: Link_hip_l_roll
        2: Link_hip_r_roll
        3: Link_hip_l_yaw
        4: Link_hip_r_yaw
        5: Link_hip_l_pitch
        6: Link_hip_r_pitch
        7: Link_knee_l_pitch
        8: Link_knee_r_pitch
        9: Link_ankle_l_pitch
        10: Link_ankle_r_pitch
        11: Link_ankle_l_roll
        12: Link_ankle_r_roll
        '''
        l_hip_yaw_id = self.cfg.init_state.l_hip_yaw_id
        l_ankle_roll_id = self.cfg.init_state.l_ankle_roll_id
        r_hip_yaw_id = self.cfg.init_state.r_hip_yaw_id
        r_ankle_roll_id = self.cfg.init_state.r_ankle_roll_id

        self.virtual_leg_l_LF[env_ids,:] = quat_rotate_inverse(self.base_quat[env_ids,:], self.rigid_state[env_ids, l_ankle_roll_id, :3]-self.rigid_state[env_ids, l_hip_yaw_id, :3])
        self.virtual_leg_l_TD[env_ids,:] = quat_rotate_inverse(self.base_quat[env_ids,:], self.rigid_state[env_ids, l_ankle_roll_id, :3]-self.rigid_state[env_ids, l_hip_yaw_id, :3])
        self.virtual_leg_l_Mid[env_ids,:] = quat_rotate_inverse(self.base_quat[env_ids,:], self.rigid_state[env_ids, l_ankle_roll_id, :3]-self.rigid_state[env_ids, l_hip_yaw_id, :3])
        self.virtual_leg_r_LF[env_ids,:] = quat_rotate_inverse(self.base_quat[env_ids,:], self.rigid_state[env_ids, r_ankle_roll_id, :3]-self.rigid_state[env_ids, r_hip_yaw_id, :3])
        self.virtual_leg_r_TD[env_ids,:] = quat_rotate_inverse(self.base_quat[env_ids,:], self.rigid_state[env_ids, r_ankle_roll_id, :3]-self.rigid_state[env_ids, r_hip_yaw_id, :3])
        self.virtual_leg_r_Mid[env_ids,:] = quat_rotate_inverse(self.base_quat[env_ids,:], self.rigid_state[env_ids, r_ankle_roll_id, :3]-self.rigid_state[env_ids, r_hip_yaw_id, :3])
        self.event_TD[env_ids,:] = False # only True when the touch-down event happens
        self.event_LF[env_ids,:] = False # only True when the leave-off event happens
        self.event_TD_real[env_ids,:] = False

    
    def _get_observations(self) -> dict:
        # 更新历史观测变量
        super()._update_obs_history_variables()

        # 缓存中间结果
        contact_forces_his = self._contact_sensor.data.net_forces_w_history
        is_contact = (torch.max(torch.norm(contact_forces_his[:, :, self.feet_contact_indices], dim=-1), dim=1)[0] > 5.0)
        index = torch.logical_and(is_contact==True, self.contact_mask==False)
        self.event_TD_real[index]=True
        self.event_TD_real[torch.logical_not(index)]=False
        self.contact_mask = is_contact.clone()

        # 计算相位和摆动高度
        self._get_phase()
        self.compute_feZ_swing()

        # 计算 sin 和 cos
        sin_pos = torch.sin(2 * torch.pi * self.phi).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * self.phi).unsqueeze(1)
        sin_pos_gait = torch.sin(2 * torch.pi * self.phi_gait).unsqueeze(1)
        cos_pos_gait = torch.cos(2 * torch.pi * self.phi_gait).unsqueeze(1)

        # 组装命令输入
        self.command_input = torch.cat(
            (sin_pos, cos_pos, sin_pos_gait, cos_pos_gait, self.commands[:, :6] * self.commands_scale), dim=1
        )
        # print("self.command_input: \n", self.command_input)
        # 缓存关节位置和速度
        q = (self.dof_pos - self.robot.data.default_joint_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel
        base_pos_cur = self.root_states[:,:3]

        # base_pos_cur[:,:3] = torch.tensor([0.03631497, 0.01387502, 1.19055806])
        # self.base_quat[:,:4] = torch.tensor([0.99853865, -0.01528902, -0.034874,    0.03834846])

        feet_l_pos_w = self.rigid_state[:, self.feet_indices[0],:3] - base_pos_cur
        feet_r_pos_w = self.rigid_state[:, self.feet_indices[1],:3] - base_pos_cur

        # feet_l_pos_w[:,:3] = torch.tensor([-0.09811187,  0.09555374, -1.05195679])
        # feet_r_pos_w[:,:3] = torch.tensor([-0.04072365, -0.08532687, -1.05093651])


        feet_l_pos_b = quat_rotate_inverse(self.base_quat, feet_l_pos_w) - torch.tensor(self.cfg.init_state.feet_l_default_pos_b, device=self.device)
        feet_r_pos_b = quat_rotate_inverse(self.base_quat, feet_r_pos_w) - torch.tensor(self.cfg.init_state.feet_r_default_pos_b, device=self.device)
        # print("feet_l_pos_b: ", feet_l_pos_b)
        # print("feet_r_pos_b: ", feet_r_pos_b) 
        feet_l_vel_w = self.rigid_state[:, self.feet_indices[0],7:10] - self.base_lin_vel
        feet_r_vel_w = self.rigid_state[:, self.feet_indices[1],7:10] - self.base_lin_vel
        # print("feet_l_vel_w: ", feet_l_vel_w, "\n, feet_r_vel_w:", feet_r_vel_w)
        feet_l_vel_b = quat_rotate_inverse(self.base_quat, feet_l_vel_w)
        feet_r_vel_b = quat_rotate_inverse(self.base_quat, feet_r_vel_w)

        feet_l_quat_w2f = self.rigid_state[:, self.feet_indices[0],3:7]
        feet_r_quat_w2f = self.rigid_state[:, self.feet_indices[1],3:7]

        # feet_l_quat_w2f[:,:4] = torch.tensor([0.95936791,  0.10320974, -0.21913836,  0.14470431])
        # feet_r_quat_w2f[:,:4] = torch.tensor([0.9419193,   0.23834952, -0.13483252,  0.19441638])
        # feet_l_quat_w2f[:,:4] = torch.tensor([1.0, 0.0, 0.0,  0.0])
        # feet_r_quat_w2f[:,:4] = torch.tensor([1.0, 0.0, 0.0,  0.0])
        # print("feet_l_eul_w2f: ", feet_l_quat_w2f)
        # print("feet_r_eul_w2f: ", feet_r_quat_w2f)

        # feet_l_eul_w2f = euler_xyz_from_quat(feet_l_quat_w2f)
        # feet_r_eul_w2f = euler_xyz_from_quat(feet_r_quat_w2f)
        # print("feet_l_eul_w2f: ", feet_l_eul_w2f)
        # print("feet_r_eul_w2f: ", feet_r_eul_w2f)

        # base_euler_xyz_wo = torch.zeros_like(self.base_euler_xyz)
        # base_euler_xyz_wo[:,:2] = self.base_euler_xyz[:,:2]
        # quat_w2b_inv = quat_inv(quat_from_euler_xyz(base_euler_xyz_wo))

        quat_w2b_inv = quat_inv(self.base_quat)
        # print("quat_w2b_inv: ", quat_w2b_inv)
        quat_b2f_l = quat_mul(quat_w2b_inv, feet_l_quat_w2f)
        quat_b2f_r = quat_mul(quat_w2b_inv, feet_r_quat_w2f)
        # print("quat_b2f_l: ", quat_b2f_l)
        # print("quat_b2f_r: ", quat_b2f_r)
        eul_b2f_l = torch.column_stack(euler_xyz_from_quat(quat_b2f_l))
        eul_b2f_r = torch.column_stack(euler_xyz_from_quat(quat_b2f_r))
        eul_b2f_l = torch.where(eul_b2f_l > torch.pi, eul_b2f_l - 2*torch.pi, eul_b2f_l)
        eul_b2f_l = torch.where(eul_b2f_l < -torch.pi, eul_b2f_l + 2*torch.pi, eul_b2f_l)
        eul_b2f_r = torch.where(eul_b2f_r > torch.pi, eul_b2f_r - 2*torch.pi, eul_b2f_r)
        eul_b2f_r = torch.where(eul_b2f_r < -torch.pi, eul_b2f_r + 2*torch.pi, eul_b2f_r)        

        # feet_l_default_pos_b = [-0.04, 0.0945, -1.04]
        # feet_r_default_pos_b = [-0.04, -0.0945, -1.04]
        # print("feet_l_vel_b: ", feet_l_vel_b)
        # print("feet_r_vel_b: ", feet_r_vel_b)       
        # print("eul_b2f_l: ", eul_b2f_l)
        # print("eul_b2f_r: ", eul_b2f_r)
        # print("\n")

        # 组装观测值
        obs_buf = torch.cat(
            (   self.command_input,  # 10 = 4D(sin cos) + 6
                q,  # 12D
                dq,  # 12D
                self.actions[:],  # 12D
                self.base_ang_vel * self.obs_scales.ang_vel,  # 3
                self.base_euler_xyz[:, :2] * self.obs_scales.quat,  # 2
                feet_l_pos_b * self.obs_scales.feet_pos,
                feet_r_pos_b * self.obs_scales.feet_pos,
                # feet_l_vel_b * self.obs_scales.feet_vel,
                # feet_r_vel_b * self.obs_scales.feet_vel,   
                eul_b2f_l * self.obs_scales.feet_eul,
                eul_b2f_r * self.obs_scales.feet_eul                               
            ),
            dim=-1,
        )

        # 组装特权观测值
        base_height = torch.clamp(self.root_states[:, 2] - torch.mean(self._height_scanner.data.ray_hits_w[..., 2]), min=0., max=3.)
        privileged_obs_buf = torch.cat(
            (
                self.command_input,  # 4 + 6
                q,  # 12 (复用已计算的 q)
                dq,  # 12 (复用已计算的 dq)
                self.actions[:],  # 12
                # (self.start_base_cmd[:, :2] - self.terrain.env_origins[:, :2]) * self.obs_scales.pos_xy,  # 2
                # (self.root_states[:, :2] - self.terrain.env_origins[:, :2]) * self.obs_scales.pos_xy,  # 2
                # base_height.unsqueeze(1),
                (self.base_pos_des[:,:2] - self.base_pos_cur[:,:2]) * self.obs_scales.pos_xy, # 2
                (self.base_lin_vel_des_w[:,:2] - self.base_lin_vel_w[:, :2]) * self.obs_scales.lin_vel, #2
                self.start_base_cmd[:,2].unsqueeze(1) * self.obs_scales.pos_theta, #1
                self.base_lin_vel * self.obs_scales.lin_vel,  # 3
                self.base_ang_vel * self.obs_scales.ang_vel,  # 3 (复用已计算的结果)
                self.base_euler_xyz_multiLoop * self.obs_scales.quat,  # 3
                self.body_mass / 60.0,  # 1
                self.stance_mask_scheduled[:], # 2
                self.contact_mask[:],  # 2
                self.contact_forces[:, self.feet_contact_indices[0], :3] / 2000.0,  # 3
                self.contact_forces[:, self.feet_contact_indices[1], :3] / 2000.0,  # 3
                self.env_stiffness_ratio[:],  # 12
                self.env_damping_ratio[:],  # 12
                feet_l_pos_b * self.obs_scales.feet_pos, # 3
                feet_r_pos_b * self.obs_scales.feet_pos, # 3
                # feet_l_vel_b * self.obs_scales.feet_vel, #3
                # feet_r_vel_b * self.obs_scales.feet_vel, #3
                eul_b2f_l * self.obs_scales.feet_eul, #3
                eul_b2f_r * self.obs_scales.feet_eul,  #3
                self.static_friction[:,0, 0].unsqueeze(1) # 1, friction
            ),
            dim=-1,
        )

        self.static_friction[:,0, 0].unsqueeze(1)

        # 添加噪声
        if self.add_noise:
            obs_now = obs_buf + torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf

        # 更新历史缓冲区
        self.obs_history.append(obs_now.clone())
        self.critic_history.append(privileged_obs_buf.clone())

        # 组装所有历史观测值
        obs_buf_all = torch.cat([self.obs_history.buffer[:, i, :] for i in range(self.obs_history.max_length)], dim=1)
        privileged_obs_buf_all = torch.cat(
            [self.critic_history.buffer[:, i, :] for i in range(self.critic_history.max_length)], dim=1
        )

        # 返回观测值
        observations = {"policy": obs_buf_all, "critic": privileged_obs_buf_all}
        self.obs_rec = obs_buf.clone()
        return observations

    
    def _check_and_resample_commands(self, env_ids_extra, isIni):
        
        env_ids = self.time_count >= 1.0
        env_ids = torch.logical_or(env_ids, env_ids_extra)
        env_ids_extra = torch.logical_and(torch.ones(self.num_envs, device=self.device), env_ids_extra)

        if isIni:
            vx_rand = torch_rand_float_ranges(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], 
                                              self.command_ranges["lin_vel_x"][2], self.command_ranges["lin_vel_x"][3],
                                       (self.num_envs, 1), device=self.device).squeeze(1)
            vy_rand = torch_rand_float_ranges(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], 
                                              self.command_ranges["lin_vel_y"][2], self.command_ranges["lin_vel_y"][3],
                                       (self.num_envs, 1), device=self.device).squeeze(1)
            wz_rand = torch_rand_float_ranges(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], 
                                              self.command_ranges["ang_vel_yaw"][2], self.command_ranges["ang_vel_yaw"][3], 
                                        (self.num_envs, 1), device=self.device).squeeze(1)

            period_rand = torch_rand_float(self.command_ranges["period"][0], self.command_ranges["period"][1], (self.num_envs, 1), device=self.device).squeeze(1)
            period_rand = torch.round(period_rand/self.cfg.commands.nominal_gait_cycle_time) * self.cfg.commands.nominal_gait_cycle_time

            self.commands[env_ids, 3] = vx_rand[env_ids] * period_rand[env_ids]
            self.commands[env_ids, 4] = vy_rand[env_ids] * period_rand[env_ids]
            self.commands[env_ids, 5] = wz_rand[env_ids] * period_rand[env_ids]
            self.mode_period_next[env_ids] = period_rand[env_ids]

        
        # reset env won stop at the first mode
        if (env_ids_extra > 0).sum() > 0 and ~isIni:
            vx_rand = torch_rand_float_ranges(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], 
                                                self.command_ranges["lin_vel_x"][2], self.command_ranges["lin_vel_x"][3],
                                        (self.num_envs, 1), device=self.device).squeeze(1)
            vy_rand = torch_rand_float_ranges(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], 
                                                self.command_ranges["lin_vel_y"][2], self.command_ranges["lin_vel_y"][3],
                                        (self.num_envs, 1), device=self.device).squeeze(1)
            wz_rand = torch_rand_float_ranges(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], 
                                                self.command_ranges["ang_vel_yaw"][2], self.command_ranges["ang_vel_yaw"][3], 
                                        (self.num_envs, 1), device=self.device).squeeze(1)

            period_rand = torch_rand_float(self.command_ranges["period"][0], self.command_ranges["period"][1], (self.num_envs, 1), device=self.device).squeeze(1)
            period_rand = torch.round(period_rand/self.cfg.commands.nominal_gait_cycle_time) * self.cfg.commands.nominal_gait_cycle_time

            self.commands[env_ids_extra, 3] = vx_rand[env_ids_extra] * period_rand[env_ids_extra]
            self.commands[env_ids_extra, 4] = vy_rand[env_ids_extra] * period_rand[env_ids_extra]
            self.commands[env_ids_extra, 5] = wz_rand[env_ids_extra] * period_rand[env_ids_extra]
            self.mode_period_next[env_ids_extra] = period_rand[env_ids_extra]
        
        #######
        
        if (env_ids > 0).sum() > 0:
            # vx_rand = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (self.num_envs, 1), device=self.device).squeeze(1)
            # vy_rand = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (self.num_envs, 1), device=self.device).squeeze(1)
            # wz_rand = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (self.num_envs, 1), device=self.device).squeeze(1)
            self.commands_last[env_ids,0] = self.commands[env_ids, 0]
            self.commands_last[env_ids,1] = self.commands[env_ids, 1]
            self.commands_last[env_ids,2] = self.commands[env_ids, 2]
            self.commands[env_ids, 0] = self.commands[env_ids, 3]
            self.commands[env_ids, 1] = self.commands[env_ids, 4]
            self.commands[env_ids, 2] = self.commands[env_ids, 5]
            
            mode_period_now = self.mode_period_next[env_ids]

            vx_rand = torch_rand_float_ranges(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], 
                                              self.command_ranges["lin_vel_x"][2], self.command_ranges["lin_vel_x"][3],
                                       (self.num_envs, 1), device=self.device).squeeze(1)
            vy_rand = torch_rand_float_ranges(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], 
                                              self.command_ranges["lin_vel_y"][2], self.command_ranges["lin_vel_y"][3],
                                       (self.num_envs, 1), device=self.device).squeeze(1)
            wz_rand = torch_rand_float_ranges(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], 
                                              self.command_ranges["ang_vel_yaw"][2], self.command_ranges["ang_vel_yaw"][3], 
                                        (self.num_envs, 1), device=self.device).squeeze(1)

            period_rand = torch_rand_float(self.command_ranges["period"][0], self.command_ranges["period"][1], (self.num_envs, 1), device=self.device).squeeze(1)
            period_rand = torch.round(period_rand/self.cfg.commands.nominal_gait_cycle_time) * self.cfg.commands.nominal_gait_cycle_time

            self.commands[env_ids, 3] = vx_rand[env_ids] * period_rand[env_ids]
            self.commands[env_ids, 4] = vy_rand[env_ids] * period_rand[env_ids]
            self.commands[env_ids, 5] = wz_rand[env_ids] * period_rand[env_ids]
            self.mode_period_next[env_ids] = period_rand[env_ids]

            # set small commands to zero
            # normal_cmds = torch.norm(self.commands[env_ids, 3:5], dim=1) > 0.01
            # self.commands[env_ids, 3:6] *= normal_cmds.unsqueeze(1)

            # pick envs to stop
            stopFlag = torch.logical_and(torch.rand(self.num_envs, device = self.device) > 0.8, env_ids)
            self.commands[stopFlag, 3:6] = 0.0

            # self.commands[env_ids,0]=4.0
            # self.commands[env_ids,1]=0.0
            # self.commands[env_ids,2]=0.0
            # period_rand[env_ids] = 10

            self.phi[env_ids] = 0.0
            self.time_count[env_ids] = 0.0
            self.delta_phi[env_ids] = 1.0 * self.dt / mode_period_now
            self.phi_gait[env_ids] = 0.0
            self.delta_phi_gait[env_ids] = 1.0 * self.dt / self.cfg.commands.nominal_gait_cycle_time

            # start with random leg
            # flag = torch.logical_and(torch.logical_and(torch.rand(self.num_envs, device = self.device) > 0.5, env_ids), ~stopFlag)
            # N = 0.5/self.delta_phi_gait
            # self.phi[flag] = self.delta_phi[flag] * N[flag]
            # self.time_count[flag] = self.delta_phi[flag] * N[flag]
            # self.phi_gait[flag] = self.delta_phi_gait[flag] * N[flag]

            # ===================================================================
            self.start_base_cmd[env_ids, :2] = self.root_states[env_ids, :2]
            self.start_base_cmd[env_ids, 2] = self.base_euler_xyz_multiLoop[env_ids, 2]
            self.start_base_quat[env_ids,:] = self.root_states[env_ids,3:7]

            # self.start_base_cmd[env_ids, :2] = self.base_pos_des_end[env_ids, :2]
            # self.start_base_cmd[env_ids, 2] = self.base_thetaZ_des_end[env_ids]
            # self.start_base_quat[env_ids,:] = self.root_states[env_ids,3:7]

            # last_stand = torch.norm(self.commands_last[:,:3], dim=1) < 0.001
            # self.first_leg[env_ids] = torch.logical_and(self.commands[env_ids,1] <= 0, last_stand[env_ids])
            delta_y = torch.abs(self.commands_last[:,1]- self.commands[:,1])
            # self.first_leg[env_ids] = torch.logical_and(self.commands[env_ids,1] <= -0.01, delta_y[env_ids] >= 0.2) 
            self.first_leg[env_ids] = self.commands[env_ids,1] <= -0.01

            # self.first_leg[env_ids] = torch.where(
            #     torch.logical_and(
            #         self.commands_last[env_ids, 1] * self.commands[env_ids, 1] < -0.0001, 
            #         abs(self.commands[env_ids, 1]) > 0.01
            #         ),
            #     self.commands[env_ids, 1] <= 0, 
            #     self.first_leg[env_ids]
            #     )

        self.stand_flag = torch.norm(self.commands[:, :3], dim=1) < 0.001


    


#================================================= rewards ====================================================================
    def _prepare_rewards_variables(self):
        self.base_pos_cur = self.root_states[:,:2]
        self.base_thetaZ_cur = self.base_euler_xyz[:,2]

        commands_w = torch.zeros((self.num_envs,3), device=self.device)
        commands_w[:,:2] = self.commands[:,:2]
        commands_w = quat_apply_yaw(self.start_base_quat, commands_w)
        commands_w[self.stand_flag,:] *= 0.

        self.base_pos_des = commands_w[:,:2] * (self.phi + self.delta_phi).unsqueeze(1) + self.start_base_cmd[:,:2]
        self.base_pos_des_end = commands_w[:,:2] + self.start_base_cmd[:,:2]
        norminal_lin_vel = commands_w[:,:2] / ( self.dt / self.delta_phi).unsqueeze(1)

        ## self.delta_phi[env_ids] = 1.0 * self.dt / mode_period_now
        # time_remain = ( self.dt / self.delta_phi) * ( 1.0-self.phi)
        # base_lin_vel_des = (self.base_pos_des_end-self.base_pos_cur) / ( time_remain.unsqueeze(1) + 1e-3 )
        # max_lin_vel = 1.5
        vel_correction = torch.clamp((self.base_pos_des-self.base_pos_cur), max=0.4, min=-0.4)
        base_lin_vel_des = norminal_lin_vel + vel_correction

        # base_lin_vel_des = torch.clip(base_lin_vel_des, -max_lin_vel, max_lin_vel)
        # base_lin_vel_des[torch.norm(base_lin_vel_des,dim=1)<0.001] *= 0.0``
        base_lin_vel_des[self.stand_flag] *= 0.0

        # base_lin_vel_des = commands_w[:,:2] / ( self.dt / self.delta_phi).unsqueeze(1)
        k = 0.55
        base_lin_vel_des_filtered = k * self.base_vel_cmd_last[:,:2] + (1 - k)*base_lin_vel_des
        self.base_vel_cmd_last[:,:2] = base_lin_vel_des_filtered[:,:2]

        self.base_pos_des_v2 = self.base_vel_cmd_last[:,:2] * self.dt + self.base_pos_cur


        self.base_thetaZ_cur = self.base_euler_xyz_multiLoop[:,2]

        self.base_thetaZ_des = self.commands[:,2] * (self.phi + self.delta_phi)  + self.start_base_cmd[:,2]
        self.base_thetaZ_des_end = self.commands[:,2] + self.start_base_cmd[:,2]
        self.base_wz_des = self.commands[:,2] / ( self.dt / self.delta_phi)

        max_ang_vel = 2.
        # base_ang_vel_des = (self.base_thetaZ_des_end-self.base_thetaZ_cur) / ( time_remain + 1e-3)
        base_ang_vel_des = self.base_wz_des + 1. * (self.base_thetaZ_des-self.base_thetaZ_cur)
        base_ang_vel_des = torch.clip(base_ang_vel_des, -max_ang_vel, max_ang_vel)
        # base_ang_vel_des[torch.abs(base_ang_vel_des)<0.0015] *= 0.0
        base_ang_vel_des[self.stand_flag] *= 0.0

        base_ang_vel_des_filtered = k * self.base_vel_cmd_last[:,2] + (1 - k)*base_ang_vel_des
        self.base_vel_cmd_last[:,2] = base_ang_vel_des_filtered
        self.base_thetaZ_des_v2 = self.base_thetaZ_cur + base_ang_vel_des_filtered * self.dt

        self.quat_des_next = quat_from_euler_xyz(torch.zeros(self.num_envs, device=self.device), torch.zeros(self.num_envs, device=self.device),
                                                 self.base_thetaZ_des_v2)
        
        self.base_lin_vel_des_w = torch.cat( (base_lin_vel_des_filtered,torch.zeros((self.num_envs,1), device=self.device)), dim=1)
        self.base_lin_vel_des_b = quat_rotate_inverse(self.quat_des_next, self.base_lin_vel_des_w)

         # 更新历史缓冲区
        self.vxy_w_des_history.append(self.base_lin_vel_des_w.clone())
        self.vxy_w_history.append(self.base_lin_vel_w.clone())

        if self.num_envs<20:
            ref_feZ=self.feZ_swing + self.cfg.init_state.ankle_height        
            feet_z = self.rigid_state[:, self.feet_indices, 2]
            base_thetaZ_des = self.commands[:,2] * (self.phi + self.delta_phi)  + self.start_base_cmd[:,2]
            vel_yaw = quat_rotate_inverse(yaw_quat(self.base_quat), self.base_lin_vel_w)

            self.dataRecoder.append_data(self.episode_length_buf[0].item())
            self.dataRecoder.append_data(self.phi[0].item())
            self.dataRecoder.append_data(self.phi_gait[0].item())
            self.dataRecoder.append_data(self.time_count[0].item())
            self.dataRecoder.append_data_array(self.commands[0,:].tolist())
            self.dataRecoder.append_data_array(self.base_pos_des[0,:2].tolist())
            self.dataRecoder.append_data_array(self.base_pos_cur[0,:2].tolist())
            self.dataRecoder.append_data_array(base_lin_vel_des_filtered[0,:2].tolist())
            self.dataRecoder.append_data_array(self.base_lin_vel_w[0,:2].tolist())
            self.dataRecoder.append_data_array(self.start_base_cmd[0,:3].tolist())
            self.dataRecoder.append_data(ref_feZ[0,0].item())
            self.dataRecoder.append_data(ref_feZ[0,1].item())
            self.dataRecoder.append_data(feet_z[0,0].item())
            self.dataRecoder.append_data(feet_z[0,1].item())
            self.dataRecoder.append_data_array(self.feet_euler[0,0,:3].tolist())
            self.dataRecoder.append_data_array(self.feet_euler[0,1,:3].tolist())
            self.dataRecoder.append_data(self.mode_period_next[0].item())
            self.dataRecoder.append_data_array(self.base_euler_xyz_multiLoop[0,:].tolist())
            self.dataRecoder.append_data_array(self.base_euler_xyz[0,:].tolist())
            self.dataRecoder.append_data(base_thetaZ_des[0].item())
            self.dataRecoder.append_data_array(self.base_vel_cmd_last[0,:3].tolist())
            self.dataRecoder.append_data_array(self.base_lin_vel[0,:3].tolist())
            self.dataRecoder.append_data_array(self.base_lin_vel_w[0,:3].tolist())
            self.dataRecoder.append_data_array(vel_yaw[0,:3].tolist())
            self.dataRecoder.append_data(self.stand_flag[0].to(torch.float).item())
            self.dataRecoder.append_data_array(self.base_ang_vel[0,:3].tolist())
            self.dataRecoder.write_line_to_file()

    """
    # reference position trace tracking
    pos_trace = 0.1
    pos_start = 0.1
    pos_end = 0.1

    # gait
    feet_air_time = 0 #1.
    feet_orientation = 2 #2.
    feet_clearence = 0.1
    feet_slip = -0.05
    feet_distance = 0
    knee_distance = 0
    
    # contact
    feet_contact_forces = -0.003

    # base pos
    orientation = 4.
    base_height = 3
    base_acc = 1

    # energy
    action_smoothness = -0.01
    torques_max = -5e-3
    torques = -5e-7
    dof_vel = -4e-4
    dof_acc = -1e-8
    collision = -1.
    """
    def _reward_pos_trace_pXY(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """

        # pos_err = torch.norm(base_pos_cur - base_pos_des, dim=1)
        pos_err = torch.cat((self.base_pos_des - self.base_pos_cur, torch.zeros((self.num_envs,1), device=self.device)),dim=1)
        pos_err_end = torch.abs(self.base_pos_des_end - self.base_pos_cur)
        # pos_err = quat_rotate_inverse(self.quat_des_next, pos_err) # not good
        
        err_px = torch.square(pos_err[:,0])
        err_py = torch.square(pos_err[:,1])
        err_px_abs = torch.abs(pos_err[:,0])
        err_py_abs = torch.abs(pos_err[:,1])
        err_pxy = torch.sum(torch.square(pos_err[:,:2]), dim=1)
        err_pxy_norm = torch.norm(pos_err[:,:2], dim=1)
        err_pxy_end_norm = torch.norm(pos_err_end[:,:2], dim=1)

        # rew_pos_px = torch.exp( -err_px*self.cfg.rewards.tracking_sigma_pxy)
        # rew_pos_py = torch.exp( -err_py*self.cfg.rewards.tracking_sigma_pxy)
        # rew_pos_pxy = torch.exp( -err_pxy*self.cfg.rewards.tracking_sigma_pxy)

        rew_pos_px = torch.exp( -err_px_abs*self.cfg.rewards.tracking_sigma_pxy) - 0.5 * err_px_abs
        rew_pos_py = torch.exp( -err_py_abs*self.cfg.rewards.tracking_sigma_pxy) - 0.5 * err_py_abs
        # rew_pos_pxy = torch.exp( -err_pxy_norm*self.cfg.rewards.tracking_sigma_pxy) - 0.5 * err_pxy_norm + 2.0 * torch.exp(-err_pxy_end_norm * 10.0)
        err_log_use = torch.clamp(err_pxy_end_norm*2.0, min=1e-3, max=1.0)
        rew_pos_pxy = torch.exp( -err_pxy_norm*self.cfg.rewards.tracking_sigma_pxy) - 0.5 * err_pxy_norm - 1.3 * torch.log(err_log_use + 1e-5)

        rewv1 = rew_pos_pxy
        rewv2 = (rew_pos_px + rew_pos_py) * 0.5

        rewv1[self.stand_flag] *= 0.0

        return rewv1
    
    
    def _reward_pos_trace_pXY_vel(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        # in local frame
        vel_yaw = quat_rotate_inverse(yaw_quat(self.base_quat), self.base_lin_vel_w)
        # lin_vel_error = torch.sum(torch.square(self.base_lin_vel_des_b[:,:2] - vel_yaw[:, :2]), dim=1)
        # lin_vel_error = torch.norm(self.base_lin_vel_des_b[:,:2] - vel_yaw[:, :2], dim=1)
        # lin_vel_error = torch.norm(self.base_lin_vel_des_b[:,:2] - self.base_lin_vel[:, :2], dim=1)
        # rew_pos = torch.exp( -lin_vel_error*self.cfg.rewards.tracking_sigma) - 0.5 * lin_vel_error

        # # single buffer
        # # in world frame
        # lin_vel_error = torch.norm(self.base_lin_vel_des_w[:,:2] - self.base_lin_vel_w[:, :2], dim=1)
        # rew_pos = torch.exp( -lin_vel_error*self.cfg.rewards.tracking_sigma) - 0.5 * lin_vel_error

        # rew = rew_pos
        # # rew[self.stand_flag] *= 0.0
        

        # use buffer
        vxy_his_all = torch.cat([self.vxy_w_history.buffer[:, i, :] for i in range(self.vxy_w_history.max_length)], dim=1)
        vxy_des_all = torch.cat([self.vxy_w_des_history.buffer[:, i, :] for i in range(self.vxy_w_des_history.max_length)], dim=1)
        # print(vxy_des_all[0,:])
        # print(vxy_his_all[0,:])
        # Reshape 为 [num_envs, N, 3]，再取前两列(x, y)：shape [num_envs, N, 2]
        his_xy = vxy_his_all.view(self.num_envs, -1, 3)[..., :2]
        des_xy = vxy_des_all.view(self.num_envs, -1, 3)[..., :2]

        # 差的模长（按时间步）：[num_envs, N]
        diff_norm = torch.norm(des_xy - his_xy, dim=2)

        # 每个环境的平均误差：[num_envs]
        mean_error = diff_norm.mean(dim=1)

        # 最终奖励计算
        rew_vel = torch.exp(-mean_error * self.cfg.rewards.tracking_sigma) - 0.7 * mean_error

        return rew_vel
    
    def _reward_pos_trace_thetaZ(self):

        yaw_error_square = torch.square(self.base_thetaZ_cur-self.base_thetaZ_des)
        yaw_error_abs = torch.abs(self.base_thetaZ_cur-self.base_thetaZ_des)
        # rew_thetaZ = torch.exp(-yaw_error_square * self.cfg.rewards.tracking_sigma_thetaZ)
        err_end_log_use = torch.clamp(torch.abs(self.base_thetaZ_des_end - self.base_thetaZ_cur) * 2, min=1e-3, max=1.0)
        err_end = torch.abs(self.base_thetaZ_des_end - self.base_thetaZ_cur)
        rew_thetaZ = torch.exp(-yaw_error_abs * self.cfg.rewards.tracking_sigma_thetaZ) - 1.5 * yaw_error_abs + 0.5 * torch.exp(-err_end * 10)   # - 0.1 * torch.log(err_end_log_use + 1e-6)

        rew = rew_thetaZ # - 0.25 * ang_vel_error
        return rew
    
    def _reward_pos_trace_thetaZ_vel(self):
        # ang_vel_error = torch.square(self.base_vel_cmd_last[:,2] - self.base_ang_vel_w[:, 2])
        ang_vel_error = torch.abs(self.base_vel_cmd_last[:,2] - self.base_ang_vel_w[:, 2])
        rew_thetaZ = torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma ) - 0.5 * ang_vel_error

        # rew = rew_thetaZ - 0.01 * yaw_error   - 0.25 * ang_vel_error
        rew = rew_thetaZ
        # rew[self.stand_flag] *= 0.0
        return rew
    

    def _reward_pos_stop(self):
        """
        force the robot to stop when receive the 'stop' command
        """
        rew_lin = torch.exp(-torch.norm(self.base_lin_vel[:,:3], dim=1) * 20)
        rew_ang = torch.exp(-torch.norm(self.base_ang_vel[:,:3], dim=1) * 20)
        rew_joint_diff_norm = torch.norm(self.dof_pos - self.robot.data.default_joint_pos, dim=1)
        rew_joint_diff = torch.sum(torch.abs(self.dof_pos - self.robot.data.default_joint_pos), dim=1)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 40)
        # rew = ((rew_lin + rew_ang) + torch.exp(-rew_joint_diff_norm * 10) - 0.25 * rew_joint_diff_norm) * self.stand_flag
        rew = ((rew_lin + rew_ang) * 0.5 + torch.exp(-rew_joint_diff_norm * 10) + 1.5 * orientation) * self.stand_flag
        return rew
    
    # need modification about the frame
    def _reward_pos_trace_pY_stand(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        foot_pos = self.rigid_state[:, self.knee_indices, :2]
        pos_err_y = (foot_pos[:,0,1] + foot_pos[:,1,1])*0.5 - self.base_pos_cur[:,1]
        err_py = torch.abs(pos_err_y)
        rewpy = torch.exp( -err_py*self.cfg.rewards.tracking_sigma_pxy)
        rewpy[torch.logical_not(self.stand_flag)] *= 0.0
        return rewpy


    def _reward_feet_orientation(self):
        """
        Try to regulate swing feet orientation through feet euler angle.
        keep the feet flat all the time

        =====joint names=====
        0: J_hip_l_roll
        1: J_hip_r_roll
        2: J_hip_l_yaw
        3: J_hip_r_yaw
        4: J_hip_l_pitch
        5: J_hip_r_pitch
        6: J_knee_l_pitch
        7: J_knee_r_pitch
        8: J_ankle_l_pitch
        9: J_ankle_r_pitch
        10: J_ankle_l_roll
        11: J_ankle_r_roll
        """
        # quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
        quat_feet_l = self.rigid_state[:, self.feet_indices[0], 3:7]
        quat_feet_r = self.rigid_state[:, self.feet_indices[1], 3:7]
        feet_checkpoints = torch.tensor([[0.5,0.,0.],[-0.5,0.,0,],[0.,0.5,0.],[0.,-0.5,0.]], device=self.device)

        feet_checkpoints_l_w = quat_rotate(quat_feet_l.unsqueeze(1), feet_checkpoints.unsqueeze(0))
        feet_checkpoints_r_w = quat_rotate(quat_feet_r.unsqueeze(1), feet_checkpoints.unsqueeze(0))
        # err_l = torch.sum(torch.square(feet_checkpoints_l_w[:,2]), dim=1)
        # err_r = torch.sum(torch.square(feet_checkpoints_r_w[:,2]), dim=1)

        err_l = torch.norm(feet_checkpoints_l_w[:,:,2], dim=1)
        err_r = torch.norm(feet_checkpoints_r_w[:,:,2], dim=1)

        # rew_swingz = torch.mean(self._reward_feet_swingZ())

        scale = torch.where(self.contact_mask, 0.3, 1.5)

        rew1 =  (torch.exp(-err_l * 50) - 5. * torch.square(err_l) ) * scale[:,0]
        rew2 =  (torch.exp(-err_r * 50) - 5. * torch.square(err_r) ) * scale[:,1]

        # swing_mask_real = torch.logical_not(self.contact_mask)
        # swing_mask_schedule = 1.0 -  self.stance_mask_scheduled
        # swing_mask = torch.logical_and(swing_mask_real, swing_mask_schedule)

        # rew1 *= swing_mask[:,0]
        # rew2 *= swing_mask[:,1]
        
        # return torch.exp(-yaw_roll * 200)  - 0.4 * yaw_roll * ( rew_swingz > 0.3 )

        return rew1 + rew2
    
    
    def _reward_feet_orientation_swingSpeed(self):
        """
        Try to regulate swing feet orientation through feet euler angle.
        keep the feet flat all the time

        =====joint names=====
        0: J_hip_l_roll
        1: J_hip_r_roll
        2: J_hip_l_yaw
        3: J_hip_r_yaw
        4: J_hip_l_pitch
        5: J_hip_r_pitch
        6: J_knee_l_pitch
        7: J_knee_r_pitch
        8: J_ankle_l_pitch
        9: J_ankle_r_pitch
        10: J_ankle_l_roll
        11: J_ankle_r_roll
        """
        # quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
        feet_eul = self.feet_euler
        # print(feet_eul.size())
        # Compute swing mask
        # swing_mask = 1 - self._get_gait_phase()
        swing_mask = torch.logical_not(self.contact_mask)
        rew1 = torch.exp(-torch.sum(torch.abs(feet_eul[:, 0, :2]), dim=1)  * 60) - torch.sum(torch.abs(feet_eul[:, 0, :2]), dim=1) * 5
        rew2 = torch.exp(-torch.sum(torch.abs(feet_eul[:, 1, :2]), dim=1)  * 60) - torch.sum(torch.abs(feet_eul[:, 1, :2]), dim=1) * 5
        rew1 *= self.contact_mask[:,0]
        rew2 *= self.contact_mask[:,1]

        feet_joint_vel_l = self.dof_vel[:, [8,10]]
        feet_joint_vel_r = self.dof_vel[:, [9,11]]
        err_l = torch.exp(-torch.norm(feet_joint_vel_l, dim=1) * 20 ) * swing_mask[:,0]
        err_r = torch.exp(-torch.norm(feet_joint_vel_r, dim=1) * 20 ) * swing_mask[:,1]
        tmp = torch.square(feet_joint_vel_l) + torch.square(feet_joint_vel_r)
        err_all = torch.sum(tmp, dim=1)

        return err_all

    def _reward_feet_clearance(self):
        """
        Calculates reward based on the clearance of the swing leg from the ground during movement.
        Encourages appropriate lift of the feet during the swing phase of the gait.
        """
        # Compute feet contact mask
        # contact = self.contact_forces[:, self.feet_contact_indices, 2] > 5.

        # contact_forces_his = self._contact_sensor.data.net_forces_w_history
        # is_contact = (torch.max(torch.norm(contact_forces_his[:,:,self.feet_contact_indices],dim=-1),dim=1)[0]>5.0)

        # Get the z-position of the feet and compute the change in z-position
        # feet_z = self.rigid_state[:, self.feet_indices, 2]
        # delta_z = feet_z - self.last_feet_z
        # self.feet_height += delta_z
        # self.last_feet_z = feet_z

        # # feet height should be closed to target feet height at the peak
        # rew_pos = torch.abs(self.feet_height - self.cfg.rewards.target_feet_height) < 0.01
        # rew_pos = torch.sum(rew_pos * (~self.contact_mask), dim=1)
        # self.feet_height *= ~self.contact_mask
        # return rew_pos

        foot_z_target_error = torch.abs(self.cfg.rewards.target_feet_height - (self.rigid_state[:, self.feet_indices, 2] - self.cfg.init_state.ankle_height))
        foot_velocity_tanh = torch.tanh( 5 * torch.norm(self.rigid_state[:, self.feet_indices, 7:9], dim=2))
        # reward = foot_z_target_error 
        # rew = torch.sum(torch.exp(-reward* 50.0) * (~self.contact_mask), dim=1)
        reward = foot_z_target_error * foot_velocity_tanh
        rew = torch.sum(torch.exp(-reward* 50.0) * foot_velocity_tanh, dim=1)
        return rew
    
    def _reward_feet_swingZ(self):
        """
        Calculates reward based on the referenced swing leg motion along z direction. Note: no rewards or ref during stance phase
        """
        # Compute swing mask
        # swing_mask = 1 - self._get_gait_phase()
        swing_mask = 1 - self.stance_mask_scheduled
        ref_feZ=self.feZ_swing  * swing_mask + self.cfg.init_state.ankle_height

        # Get the z-position of the feet
        feet_z = self.rigid_state[:, self.feet_indices, 2]
        dfeet_z = self.rigid_state[:, self.feet_indices, 9]
        # feet height should be closed to ref feet motion

        err_abs = torch.abs(feet_z-ref_feZ)
        err_square =  torch.square(ref_feZ - feet_z)

        # rew_pos = torch.exp(-torch.square(feet_z - ref_feZ)*100) - 2 * torch.abs(feet_z-ref_feZ)
        rew_pos = torch.exp(-err_abs * 80.) - 20. * err_square
        rew_pos = torch.sum(rew_pos, dim=1)

        return rew_pos

    
    def _reward_feet_dswingZ(self):
        """
        Calculates reward based on the referenced swing leg motion along z direction. Note: no rewards or ref during stance phase
        The buffer has shape (num_rigid_bodies, 13). State for each rigid body contains position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
        """
        swing_mask = 1 - self.stance_mask_scheduled
        ref_feZ=self.feZ_swing * swing_mask + self.cfg.init_state.ankle_height
        ref_dfeZ=self.dfeZ_swing * swing_mask
        # Compute swing mask
        # swing_mask = 1 - self._get_gait_phase()
        
        # Get the z-position of the feet
        feet_z = self.rigid_state[:, self.feet_indices, 2]
        dfeet_z = self.rigid_state[:, self.feet_indices, 9]

        err_abs =  torch.abs((ref_dfeZ - dfeet_z))
        err_square =  torch.square((ref_dfeZ - dfeet_z))
        # err_square =  torch.clamp(torch.square((ref_dfeZ - dfeet_z)), min=-0.3, max=0.3)
        # print('err_abs=', err_abs)

        rew_dpos = torch.exp(-err_abs * 5) - 0.1 * err_square
        rew_dpos = torch.sum(rew_dpos, dim=1)

        return rew_dpos
    
    def _reward_feet_dswingXY(self):
        """
        Calculates reward based on the referenced swing leg motion along z direction. Note: no rewards or ref during stance phase
        The buffer has shape (num_rigid_bodies, 13). State for each rigid body contains position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
        """
        swing_mask = 1 - self.stance_mask_scheduled
        
        dfeet_XY_l = quat_rotate_inverse(self.base_quat, self.rigid_state[:, self.feet_indices[0], 7:10])
        dfeet_XY_r = quat_rotate_inverse(self.base_quat, self.rigid_state[:, self.feet_indices[1], 7:10])

        err_1 =  torch.sum(torch.square(dfeet_XY_l[:,:2]),dim=1) * torch.logical_not(self.contact_mask[:,0])
        err_2 =  torch.sum(torch.square(dfeet_XY_r[:,:2]),dim=1) * torch.logical_not(self.contact_mask[:,1])
        
        rew_pos = torch.exp(-(err_1+err_2)*20)

        return rew_pos


    def _reward_feet_swingZ_dz(self):
        """
        Calculates reward based on the referenced swing leg motion along z direction. Note: no rewards or ref during stance phase
        The buffer has shape (num_rigid_bodies, 13). State for each rigid body contains position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
        """
        swing_mask = 1 - self.stance_mask_scheduled
        ref_feZ=self.feZ_swing + self.cfg.init_state.ankle_height
        ref_dfeZ=self.dfeZ_swing * swing_mask
        # Compute swing mask
        # swing_mask = 1 - self._get_gait_phase()
        
        # Get the z-position of the feet
        feet_z = self.rigid_state[:, self.feet_indices, 2]
        dfeet_z = self.rigid_state[:, self.feet_indices, 9]
    
        # rew_pos[good_indices] = torch.exp(-torch.square(feet_z[good_indices] - ref_feZ[good_indices])*150)
        errZ =  torch.square((ref_feZ - feet_z))*100
        rew_pos = torch.sum(torch.exp(-errZ), dim=1)

        errDz =  torch.square((ref_dfeZ - dfeet_z))*50
        rew_dpos = torch.sum(torch.exp(-errDz), dim=1)

        # rew_pos = torch.exp(-torch.square(feet_z - ref_feZ)*100) - 2 * torch.abs(feet_z-ref_feZ)
        rew_pos = rew_pos * rew_dpos

        return rew_pos
    
    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase. 
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.contact_mask
        # stance_mask = self._get_gait_phase()
        stance_mask = self.stance_mask_scheduled
        reward = torch.where(contact == stance_mask, 2, -0.3)
        return torch.mean(reward, dim=1)


    def _reward_virtual_leg_sym(self):
        """
        virtual leg symmetric along x direction in body frame
        """
        indices = self.event_LF == True
        rewl = torch.zeros(self.num_envs,device=self.device)
        rewl[indices[:,0]] = torch.exp(-torch.abs(self.virtual_leg_l_LF[indices[:,0],0] + self.virtual_leg_l_TD[indices[:,0],0])*10)
        rewr = torch.zeros(self.num_envs,device=self.device)
        rewr[indices[:,1]] = torch.exp(-torch.abs(self.virtual_leg_r_LF[indices[:,1],0] + self.virtual_leg_r_TD[indices[:,1],0])*10)

        indices = self.event_TD == True
        rewl_m = torch.zeros(self.num_envs,device=self.device)
        rewr_m = torch.zeros(self.num_envs,device=self.device)
        rew_bal = torch.zeros(self.num_envs,device=self.device)
        rewl_m[indices[:,0]] = torch.exp(-torch.abs(self.virtual_leg_l_Mid[indices[:,0],0])*30)
        rewr_m[indices[:,1]] = torch.exp(-torch.abs(self.virtual_leg_r_Mid[indices[:,1],0])*30)
        rew_bal[indices[:,0]] = torch.exp(-torch.abs(self.virtual_leg_l_TD[indices[:,0],0]-self.virtual_leg_r_TD[indices[:,0],0])*10)
        rew_bal[indices[:,1]] = torch.exp(-torch.abs(self.virtual_leg_l_TD[indices[:,1],0]-self.virtual_leg_r_TD[indices[:,1],0])*10)

        rew = rewl + rewr + 2.*rewl_m + 2.*rewr_m + 3.*rew_bal
        # scale = torch.exp(-torch.abs(self.base_vel_cmd_last[:,2]) * 10.0) # no use for thetaZ, and bad for pstop
        
        return rew
    
    def _reward_virtual_leg_sym_continuous(self):
        l_hip_yaw_id = self.cfg.init_state.l_hip_yaw_id
        l_ankle_roll_id = self.cfg.init_state.l_ankle_roll_id
        r_hip_yaw_id = self.cfg.init_state.r_hip_yaw_id
        r_ankle_roll_id = self.cfg.init_state.r_ankle_roll_id
        virtual_leg_l = quat_rotate_inverse(self.base_quat, self.rigid_state[:, l_ankle_roll_id, :3]-self.rigid_state[:, l_hip_yaw_id, :3])
        virtual_leg_r = quat_rotate_inverse(self.base_quat, self.rigid_state[:, r_ankle_roll_id, :3]-self.rigid_state[:, r_hip_yaw_id, :3])
        phiTmp = torch.fmod(self.phi_gait * 2, 1)
        # self.stance_mask_scheduled
        first_half_swing = phiTmp <= 0.5
        second_half_swing = phiTmp >= 0.5
        rew = torch.zeros((self.num_envs,2), device=self.device)
        rew[first_half_swing,0] = torch.where(virtual_leg_l[first_half_swing,0]<=0, 0., -1.) # 2.0, -1.
        rew[second_half_swing,0] = torch.where(virtual_leg_l[second_half_swing,0]>=0, 0., -1.)
        rew[:,0] *= (1-self.stance_mask_scheduled[:,0])
        rew[first_half_swing,1] = torch.where(virtual_leg_r[first_half_swing,0]<=0, 0., -1.)
        rew[second_half_swing,1] = torch.where(virtual_leg_r[second_half_swing,0]>=0, 0., -1.)
        rew[:,1] *= (1-self.stance_mask_scheduled[:,1])
        return torch.sum(rew, dim=1)

    
    def _reward_feet_clearance_joint(self):
        """
        Calculates reward based on the clearance of the swing leg from the ground during movement.
        Encourages appropriate lift of the feet during the swing phase of the gait.
        """
        swing_knee_joint = -1.1587
        l_knee_error = (torch.clip(torch.square(self.dof_pos[:,self.cfg.init_state.l_knee_id] - swing_knee_joint), max = (10.0/180.0*3.1415)**2)).unsqueeze(1)
        r_knee_error = (torch.clip(torch.square(self.dof_pos[:,self.cfg.init_state.r_knee_id] - swing_knee_joint), max = (10.0/180.0*3.1415)**2)).unsqueeze(1)
        err = torch.cat((l_knee_error,r_knee_error),dim=1)
        # reward = foot_z_target_error 
        rew = torch.sum(torch.exp(-err* 100.0) * (~self.contact_mask), dim=1)
        return rew

    def _reward_feet_slip(self):
        """
        Calculates the reward for minimizing feet slip. The reward is based on the contact forces 
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact 
        with the ground. The speed of the foot is calculated and scaled by the contact condition.
        """
        contact = self.contact_mask
        feet_data = self.rigid_state[:, self.feet_indices]  # [batch, 2, ...]
        speed_components = feet_data[:, :, [7,8,9,10,11,12]]
        foot_speed_norm = torch.norm(speed_components, dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        # rew *= self.stance_mask
        return (torch.sum(rew, dim=1))


    def _reward_knee_distance(self):
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        """
        foot_pos = self.rigid_state[:, self.knee_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist / 2.0  # why /2 ?
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
        """
        foot_pos = self.rigid_state[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2
    
    def _reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus 
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
                =====joint names=====
        0: J_hip_l_roll
        1: J_hip_r_roll
        2: J_hip_l_yaw
        3: J_hip_r_yaw
        4: J_hip_l_pitch
        5: J_hip_r_pitch
        6: J_knee_l_pitch
        7: J_knee_r_pitch
        8: J_ankle_l_pitch
        9: J_ankle_r_pitch
        10: J_ankle_l_roll
        11: J_ankle_r_roll
        """
        joint_diff = self.dof_pos - self.robot.data.default_joint_pos
        left_yaw_roll = joint_diff[:, [0,2]]
        right_yaw_roll = joint_diff[:, [1,3]]
        yaw_roll_all = joint_diff[:,[0,1,2,3]]
        
        # yaw_roll = torch.sum(torch.abs(yaw_roll_all), dim=1)
        # yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        # rew_swingz = torch.mean(self._reward_feet_swingZ())
        # return torch.exp(-yaw_roll * 200)  - 0.4 * yaw_roll * (rew_swingz>0.4)

        scale = torch.tanh(torch.abs(self.base_vel_cmd_last[:,2]) * 10.0)
    
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1 - 0.2 * scale, 0, 50)
        return torch.exp(-yaw_roll * 100) #- 0.01 * torch.norm(joint_diff, dim=1)

        # scale = torch.exp(-torch.abs(self.base_vel_cmd_last[:,2]) * 10.0)
        # err = torch.sum(torch.abs(yaw_roll_all),dim=1)

        # rew = (torch.norm(left_yaw_roll,dim=1)+torch.norm(right_yaw_roll,dim=1))*0.5
        # return err


    """
    # contact
    feet_contact_forces = -0.003

    # base pos
    orientation = 4.
    base_height = 3
    base_acc = 1

    # energy
    action_smoothness = -0.01
    torques_max = -5e-3
    torques = -5e-7
    dof_vel = -4e-4
    dof_acc = -1e-8
    collision = -1.
    """
    
    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_contact_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(0, 2000), dim=1)

    def _reward_touch_down_velocity(self):
        tmp = (self.event_TD_real == True)
        vel_left = self.last_rigid_state[:, self.feet_indices[0],7:10]
        vel_right = self.last_rigid_state[:, self.feet_indices[1],7:10]
        # err_left=torch.sum(torch.abs(vel_left),dim=1)
        # err_right=torch.sum(torch.abs(vel_right),dim=1)

        err_left=torch.norm(vel_left,dim=1)
        err_right=torch.norm(vel_right,dim=1)

        rew1 = err_left * tmp[:,0]
        rew2 = err_right * tmp[:,1]
    #     indices = torch.logical_or(tmp[:,0],tmp[:,1])
    #     pos_err = torch.sum(torch.square(self.base_pos_cur - self.base_pos_des), dim=1)
    #     # rew_pos = torch.exp( -pos_err*self.cfg.rewards.tracking_sigma_pxy)
    #     rew_pos = -torch.log( pos_err + 1e-5)
    #     rew_pos[torch.logical_not(indices)] *= 0.0

        return rew1+rew2

    def _reward_orientation(self):
        """
        Calculates the reward for maintaining a flat base orientation. It penalizes deviation 
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 30)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 40)
        return (quat_mismatch + orientation) / 2.
    
    def _reward_flat_orientation_l2(self):
        """Penalize non-flat base orientation using L2 squared kernel.

        This is computed by penalizing the xy-components of the projected gravity vector.
        """
        # extract the used quantities (to enable type-hinting)
        return torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1)
    
    def _reward_vel_mismatch_exp(self):
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities. 
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """
        # lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 200)
        # ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.)

        # c_update = (lin_mismatch + ang_mismatch) / 2.
        des_ang_vel = -self.base_euler_xyz[:, :2]
        lin_mismatch = torch.square(self.base_lin_vel[:, 2]) 
        ang_mismatch = torch.sum(torch.square(des_ang_vel*1.-self.base_ang_vel[:, :2]),dim=1)

        c_update = (lin_mismatch + ang_mismatch)

        return c_update

    def _reward_base_height(self):
        """
        Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
        The reward is computed based on the height difference between the robot's base and the average height 
        of its feet when they are in contact with the ground.
        """
        contact =  self.stance_mask_scheduled # self.contact_mask # Note scheduled contact is used NOT real contact state !
        row_sums = torch.sum(contact, dim=1)
        # zero_indices = row_sums < 0.5
        # row_sums[zero_indices] = 1
        measured_bottom_heights = torch.sum(
            self.rigid_state[:, self.feet_indices, 2] * contact, dim=1) / row_sums - self.cfg.init_state.ankle_height
        # measured_bottom_heights[zero_indices] *= 0.0
        # base_height = self.root_states[:, 2]  # - measured_bottom_heights
        base_height = torch.clamp(self.root_states[:, 2] - torch.mean(self._height_scanner.data.ray_hits_w[..., 2]), min=0.,max=3.)

        # return torch.exp(-torch.abs((base_height - self.cfg.rewards.base_height_target)**2) * 100)
        reward = torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 20) - 15 * torch.square(base_height - self.cfg.rewards.base_height_target)
        
        # reward = torch.exp(-torch.square(base_height - self.cfg.rewards.base_height_target) * 150)

        # reward = torch.where(reward < 0.5, torch.tensor(-0.1, device=self.device), reward)
        # reward[zero_indices] = -0.5

        return reward
    
    

    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 6)
        return rew
    
    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        return term_1 + 0.1 * term_2
    
    def _reward_action_smoothness_minimum(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_3
    
    def _reward_action_smoothness_minimum_filter(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """

        # term_4 = 0.05 * torch.sum(torch.abs((self.processed_actions - self.processed_actions_used)), dim=1)
        term_4 = torch.exp(-torch.sum(torch.abs(self.processed_actions - self.processed_actions_used), dim=1))
        return term_4

    def _reward_torques(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        # stance_mask = self._get_gait_phase()
        # stance_mask = self.stance_mask.clone()
        # weight=torch.ones_like(self.torques)
        # weight[:,self.cfg.init_state.l_ankle_pitch_id]= 0.1 * stance_mask[:,0]+ 1 * ( 1 - stance_mask[:,0])
        # weight[:,self.cfg.init_state.l_ankle_roll_id]= 0.1 * stance_mask[:,0]+ 1 * ( 1 - stance_mask[:,0])
        # weight[:,self.cfg.init_state.r_ankle_pitch_id]= 0.1 * stance_mask[:,1]+ 1 * ( 1 - stance_mask[:,1])
        # weight[:,self.cfg.init_state.r_ankle_roll_id]= 0.1 * stance_mask[:,1]+ 1 * ( 1 - stance_mask[:,1])
        # weight=torch.zeros_like(self.torques)
        # weight[:, :4] = 1
        return torch.sum(torch.square(self.torques), dim=1)
    
    def _reward_torques_max(self):
        """
        Penalizes the use of high torques in the robot's ankles joints.
        =====joint names=====
        0: J_hip_l_roll
        1: J_hip_r_roll
        2: J_hip_l_yaw
        3: J_hip_r_yaw
        4: J_hip_l_pitch
        5: J_hip_r_pitch
        6: J_knee_l_pitch
        7: J_knee_r_pitch
        8: J_ankle_l_pitch
        9: J_ankle_r_pitch
        10: J_ankle_l_roll
        11: J_ankle_r_roll
        """
        torques_ankles_pitch=torch.abs(self.torques[:,[8,9]])
        torques_ankles_roll=torch.abs(self.torques[:,[10,11]])
        torques_ankles_pitch_max=50
        torques_ankles_roll_max=40
        rew_tor_pitch=torch.sum(torch.clamp(torques_ankles_pitch - torques_ankles_pitch_max, min=0.0, max=None), dim=1)
        rew_tor_roll=torch.sum(torch.clamp(torques_ankles_roll - torques_ankles_roll_max, min=0.0, max=None),dim=1)
        rew = rew_tor_pitch + rew_tor_roll
        rew[self.stand_flag] *= 0.
        return rew
    
    def _reward_dof_vel(self):
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and 
        more controlled movements.
        =====joint names=====
        0: J_hip_l_roll
        1: J_hip_r_roll
        2: J_hip_l_yaw
        3: J_hip_r_yaw
        4: J_hip_l_pitch
        5: J_hip_r_pitch
        6: J_knee_l_pitch
        7: J_knee_r_pitch
        8: J_ankle_l_pitch
        9: J_ankle_r_pitch
        10: J_ankle_l_roll
        11: J_ankle_r_roll
        """
        # weight=torch.ones_like(self.torques)
        # weight[:, 0] = 100
        # weight[:, 1] = 100
        # weight=torch.zeros_like(self.torques)
        # weight[:, :4] = 1
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_pow(self):
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and 
        more controlled movements.
        """
        
        epsilon = 1e-8
        reciprocal_tensor = 1.0 / (self.env_actuator_effort_limit + epsilon)

        # 归一化，每行元素的和为 1
        normalized_tensor = reciprocal_tensor / reciprocal_tensor.sum(dim=1, keepdim=True)
        pow = self.dof_vel * self.torques
        return torch.sum(torch.square(pow), dim=1) # normalized_tensor

    def _reward_fe_pow(self):
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and 
        more controlled movements.
        """
        
        fe_vel_l = self.rigid_state[:,self.feet_indices[0],7:10]
        fe_vel_r = self.rigid_state[:,self.feet_indices[1],7:10]
        fe_forces_l = self.contact_forces[:, self.feet_contact_indices[0],:3]
        fe_forces_r = self.contact_forces[:, self.feet_contact_indices[1],:3]
        fe_pow = torch.abs(torch.sum(fe_vel_l*fe_forces_l, dim=1)) + torch.abs(torch.sum(fe_vel_r*fe_forces_r, dim=1))
        return fe_pow

    def _reward_dof_acc(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        epsilon = 1e-8
        reciprocal_tensor = 1.0 / (self.env_actuator_effort_limit + epsilon)

        # 归一化，每行元素的和为 1
        normalized_tensor = reciprocal_tensor / reciprocal_tensor.sum(dim=1, keepdim=True)
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt)*normalized_tensor, dim=1)
    
    def _reward_collision(self):
        """
        Penalizes collisions of the robot with the environment, specifically focusing on selected body parts.
        This encourages the robot to avoid undesired contact with objects or surfaces.
        """
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

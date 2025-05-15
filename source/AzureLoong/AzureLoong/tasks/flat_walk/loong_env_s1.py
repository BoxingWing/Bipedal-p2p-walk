from .base_scripts.loong_env_base import LoongBaseEnv, torch_rand_float
from .base_scripts.loong_cfg_base import LoongEnvBaseCfg
import torch
from isaaclab.utils.math import euler_xyz_from_quat, quat_rotate_inverse, quat_apply, wrap_to_pi

class LoongEnvS1(LoongBaseEnv):
    cfg: LoongEnvBaseCfg

    def __init__(self, cfg: LoongEnvBaseCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.last_feet_z = self.cfg.init_state.ankle_height
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.phi = torch.zeros(self.num_envs, dtype=torch.float, device=self.device) # phase signal for each env
        self.last_phi = torch.zeros(self.num_envs, dtype=torch.float, device=self.device) # phase signal for each env
        self.delta_phi = torch.zeros(self.num_envs, dtype=torch.float, device=self.device) # delta phase signal for each env in each time step
        self.stand_flag = torch.ones(self.num_envs, dtype=torch.bool, device=self.device) # flag indicates whether in stand mode
        self.base_euler_xyz_old = torch.zeros((self.num_envs,3), dtype=torch.float, device=self.device) # base euler angle of last sim step
        
        self.vel_mask = torch.ones((self.num_envs, 3), device=self.device)
        self.focus_on_stepping = False
        if cfg.focus_on_stepping == True :
            self.focus_on_stepping = True

        self.feZ_swing = torch.zeros((self.num_envs, 2), device=self.device)
        self.dfeZ_swing = torch.zeros((self.num_envs, 2), device=self.device)
        self.root_pos_ini = torch.zeros((self.num_envs,3), dtype=torch.float, device=self.device)
        self.stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        self.contact_mask = torch.zeros((self.num_envs, 2), device=self.device)
        self.last_stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
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
        self.event_LF = torch.zeros((self.num_envs,2), dtype=torch.bool, device=self.device) # only True when the leave-off event happens
        self._reset_idx(torch.ones(self.num_envs, device=self.device))
        self._get_observations()

    
    def  _get_phase(self):
        '''
        update phase, stance_mask, last_stance_mask, dof_pos_LF, and events relevent things
        '''
        # cycle_time = self.cfg.rewards.cycle_time
        # phase = self.episode_length_buf * self.dt / cycle_time
        self.last_phi[:] = self.phi[:]
        self.phi += self.delta_phi
        sin_pos = torch.sin(2 * torch.pi * self.phi)
        
        self.last_stance_mask[:] = self.stance_mask[:]
        # left foot stance
        self.stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        self.stance_mask[:, 1] = sin_pos < 0
        # Double support phase
        self.stance_mask[torch.abs(sin_pos) < 0.01, :] = 1

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
        indices = torch.logical_and(self.last_stance_mask, torch.logical_not(self.stance_mask))
        # self.dof_pos_l_LF[indices[:,0]] = self.dof_pos[indices[:,0],[0,2,4,6,8,10]]
        # self.dof_pos_r_LF[indices[:,1]] = self.dof_pos[indices[:,1],[1,3,5,7,9,11]]
        self.virtual_leg_l_LF[indices[:,0],:] = quat_rotate_inverse(self.base_quat[indices[:,0],:], self.rigid_state[indices[:,0], l_ankle_roll_id, :3]-self.rigid_state[indices[:,0], l_hip_yaw_id, :3])
        self.virtual_leg_r_LF[indices[:,1],:] = quat_rotate_inverse(self.base_quat[indices[:,1],:], self.rigid_state[indices[:,1], r_ankle_roll_id, :3]-self.rigid_state[indices[:,1], r_hip_yaw_id, :3])
        self.event_LF[indices] = True
        self.event_LF[torch.logical_not(indices)] = False

        indices = torch.logical_and(self.stance_mask, torch.logical_not(self.last_stance_mask))
        self.virtual_leg_l_TD[indices[:,0],:] = quat_rotate_inverse(self.base_quat[indices[:,0],:], self.rigid_state[indices[:,0], l_ankle_roll_id, :3]-self.rigid_state[indices[:,0], l_hip_yaw_id, :3])
        self.virtual_leg_r_TD[indices[:,1],:] = quat_rotate_inverse(self.base_quat[indices[:,1],:], self.rigid_state[indices[:,1], r_ankle_roll_id, :3]-self.rigid_state[indices[:,1], r_hip_yaw_id, :3])
        self.event_TD[indices] = True
        self.event_TD[torch.logical_not(indices)] = False

        phi_Tmp = torch.fmod(self.phi*2,1)
        phi_last_Tmp = torch.fmod(self.last_phi*2,1)
        indices = torch.logical_and(phi_Tmp>0.5, phi_last_Tmp<=0.5)
        self.virtual_leg_l_Mid[indices,:] = quat_rotate_inverse(self.base_quat[indices,:], self.rigid_state[indices, l_ankle_roll_id, :3]-self.rigid_state[indices, l_hip_yaw_id, :3])
        self.virtual_leg_r_Mid[indices,:] = quat_rotate_inverse(self.base_quat[indices,:], self.rigid_state[indices, r_ankle_roll_id, :3]-self.rigid_state[indices, r_hip_yaw_id, :3])
    
    def compute_feZ_swing(self):
        # add_double_support_phase = 0.1
        # phase = self.phi.detach()
        # phase = torch.fmod(2*phase,1)*(1+add_double_support_phase)
        # idx = phase < add_double_support_phase
        # idx_rest = phase >= add_double_support_phase
        # phase[idx] = 0
        # phase[idx_rest] -= add_double_support_phase
        phiTmp = torch.fmod(self.phi*2,1)
        y0=torch.zeros(self.num_envs, device=self.device)
        yM=torch.ones_like(y0)* self.cfg.rewards.target_feet_height
        ye=torch.ones_like(y0)*0
        phiM=torch.ones_like(y0)*0.4

        a0=y0.clone()
        a1=torch.zeros_like(a0)
        a2=-(3.0*(y0 - yM))/torch.pow(phiM,2)
        a3=(2.0*(y0 - yM))/torch.pow(phiM,3)
        b0=-(yM - 3*phiM*yM + 3*torch.pow(phiM,2)*ye - torch.pow(phiM,3)*ye)/(3*phiM - 3*torch.pow(phiM,2) + torch.pow(phiM,3) - 1)
        b1=-(6*(phiM*yM - phiM*ye))/(3*phiM - 3*torch.pow(phiM,2) + torch.pow(phiM,3) - 1)
        b2=(3*(yM - ye)*(phiM + 1))/(3*phiM - 3*torch.pow(phiM,2) + torch.pow(phiM,3) - 1)
        b3=-(2*(yM - ye))/(3*phiM - 3*torch.pow(phiM,2) + torch.pow(phiM,3) - 1)
        y1=a0+ a1 * phiTmp + a2 * torch.pow(phiTmp,2) + a3 * torch.pow(phiTmp,3)
        y2=b0+ b1 * phiTmp + b2 * torch.pow(phiTmp,2) + b3 * torch.pow(phiTmp,3)
        dy1 = a1 + 2 * a2 *phiTmp + 3 * a3 * torch.pow(phiTmp,2)
        dy2 = b1 + 2 * b2 *phiTmp + 3 * b3 * torch.pow(phiTmp,2)

        index_sg1 = phiTmp<=phiM
        index_sg2 = phiTmp>phiM

        y = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        y[index_sg1]=y1[index_sg1]
        y[index_sg2]=y2[index_sg2]
        dy = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        dy[index_sg1]=dy1[index_sg1] * ( 1.0 / self.cfg.rewards.cycle_time * 2)
        dy[index_sg2]=dy2[index_sg2] * ( 1.0 / self.cfg.rewards.cycle_time * 2)

        swing_mask = 1 - self.stance_mask
        self.feZ_swing[:,0] = y[:] * swing_mask[:,0]
        self.feZ_swing[:,1] = y[:] * swing_mask[:,1]
        self.dfeZ_swing[:,0] = dy[:] * swing_mask[:,0]
        self.dfeZ_swing[:,1] = dy[:] * swing_mask[:,1]
    
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(
            self.cfg.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_vec[0: 5] = 0.  # commands
        noise_vec[5: 17] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[17: 29] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[29: 41] = 0.  # previous actions
        noise_vec[41: 44] = noise_scales.ang_vel * self.obs_scales.ang_vel   # ang vel
        noise_vec[44: 46] = noise_scales.quat * self.obs_scales.quat         # euler x,y
        return noise_vec
    
    def _reset_idx(self, env_ids: torch.Tensor ):
        super()._reset_idx(env_ids)
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.phi[env_ids] = 0.0
        self.last_phi[env_ids] = 0.0

        # will cause right leg wont lift, dont know why
        # random_values = torch.rand(len(env_ids), device=self.device)
        # selected_env_ids = env_ids[random_values > 0.5]
        # unselected_env_ids = env_ids[random_values <= 0.5]
        # self.stand_flag[selected_env_ids] = False
        # self.stand_flag[unselected_env_ids] = True
        # self.delta_phi[unselected_env_ids] = 0
        # self.delta_phi[selected_env_ids] = 1.0 * self.dt / self.cfg.rewards.cycle_time

        if self.cfg.focus_stand:
                self.stand_flag[env_ids] = True
                self.delta_phi[env_ids] =  0.0
        else:
            self.stand_flag[env_ids] = False
            self.delta_phi[env_ids] = 1.0 * self.dt / self.cfg.rewards.cycle_time

        
        self.base_euler_xyz_old[env_ids,:]=self.base_euler_xyz[env_ids,:]
        self.root_pos_ini[env_ids,:] = self.root_states[env_ids,:3]
        self.stance_mask[env_ids, :] = True
        self.last_stance_mask[env_ids, :] = True
        self.contact_mask[env_ids,:] = False
        
        # expanded_target_tmp = self.robot.data.default_joint_pos.squeeze(0)
        # expanded_target = expanded_target_tmp.unsqueeze(0).expand(self.num_envs, -1)
        # print("=================")
        # print(expanded_target_tmp.size())
        # print(self.dof_pos_l_LF[env_ids,:].size())
        # print(env_ids.size())
        # self.dof_pos_l_LF[env_ids,:6] = expanded_target_tmp[env_ids][:,[0,2,4,6,8,10]]
        # self.dof_pos_r_LF[env_ids,:6] = expanded_target_tmp[env_ids][:,[1,3,5,7,9,11]]
        # self.dof_knee_swing[env_ids,0] = expanded_target[env_ids,3]
        # self.dof_knee_swing[env_ids,1] = expanded_target[env_ids,9]
        for i in range(self.obs_history.max_length):
            self.obs_history._buffer[i, env_ids, :] = 0.0
        for i in range(self.critic_history.max_length):
            self.critic_history._buffer[i, env_ids, :] = 0.0

        '''
        =====body names=====
        0: base_link
        1: Link_arm_l_01
        2: Link_arm_r_01
        3: Link_head_yaw
        4: Link_waist_pitch
        5: Link_hip_l_roll
        6: Link_hip_r_roll
        7: Link_hip_l_yaw
        8: Link_hip_r_yaw
        9: Link_hip_l_pitch
        10: Link_hip_r_pitch
        11: Link_knee_l_pitch
        12: Link_knee_r_pitch
        13: Link_ankle_l_pitch
        14: Link_ankle_r_pitch
        15: Link_ankle_l_roll
        16: Link_ankle_r_roll
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
    
    def _get_observations(self) -> dict:
        super()._update_obs_history_variables()
        # resample phase to generate stand mode
        if self.cfg.enable_stand:
            env_ids = ( torch.logical_and( self.episode_length_buf % int(self.cfg.commands.stand_resampling_time / self.dt) == 0, 
                                                  self.episode_length_buf > 200 ) ).nonzero(as_tuple=False).flatten()
            random_values = torch.rand(len(env_ids), device=self.device)
            selected_env_ids = env_ids[random_values > 0.5]
            unselected_env_ids = env_ids[random_values <= 0.5]
            self.stand_flag[selected_env_ids] = True
            self.delta_phi[selected_env_ids] *= 0.0
            self.phi[selected_env_ids] *= 0.0
            self.last_phi[selected_env_ids] *= 0.0

            self.stand_flag[unselected_env_ids] = False
            self.delta_phi[unselected_env_ids] =  1.0 * self.dt / self.cfg.rewards.cycle_time
        elif self.cfg.focus_stand:
            self.stand_flag[:] = True
            self.delta_phi[:] =  0.0
        else:
            self.stand_flag[:] = False
            self.delta_phi[:] =  1.0 * self.dt / self.cfg.rewards.cycle_time

        self._get_phase()

        # phase = self._get_phase()
        self.compute_feZ_swing()
        # self.compute_knee_joint_swing()

        sin_pos = torch.sin(2 * torch.pi * self.phi).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * self.phi).unsqueeze(1)

        stance_mask = self.stance_mask.clone()
        # self.contact_mask = self.contact_forces[:, self.feet_contact_indices, 2] > 5.
        contact_forces_his = self._contact_sensor.data.net_forces_w_history
        is_contact = (torch.max(torch.norm(contact_forces_his[:,:,self.feet_contact_indices],dim=-1),dim=1)[0]>1.0)
        # print("contact size=============")
        # print(is_contact.size())
        self.contact_mask = is_contact

        self.commands[:, :3] = self.commands[:, :3] * self.vel_mask
        self.command_input = torch.cat(
            (sin_pos, cos_pos, self.commands[:, :3] * torch.logical_not(self.stand_flag.unsqueeze(1)) * self.commands_scale), dim=1)
        
        q = (self.dof_pos - self.robot.data.default_joint_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel

        # print((self.body_mass).size())
        privileged_obs_buf = torch.cat((
            self.command_input,  # 2 + 3
            (self.dof_pos - self.robot.data.default_joint_pos) * self.obs_scales.dof_pos,  # 12
            self.dof_vel * self.obs_scales.dof_vel,  # 12
            self.actions,  # 12
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz * self.obs_scales.quat,  # 3
            self.body_mass / 30.,  # 1
            stance_mask,  # 2
            self.contact_mask[:],  # 2
            self.contact_forces[:, self.feet_contact_indices[0], :3] / 2000., #3
            self.contact_forces[:, self.feet_contact_indices[1], :3] / 2000.,#3
            self.env_stiffness_ratio[:], #12
            self.env_damping_ratio[:] #12
        ), dim=-1)

        obs_buf = torch.cat((
            self.command_input,  # 5 = 2D(sin cos) + 3D(vel_x, vel_y, aug_vel_yaw)
            q,    # 12D
            dq,  # 12D
            self.actions,   # 12D
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz[:, :2] * self.obs_scales.quat,  # 2
        ), dim=-1)

        # nan_mask = torch.isnan(obs_buf)

        # # 获取NaN所在的行列索引
        # nan_indices = torch.nonzero(nan_mask)

        # # 判断是否存在NaN值
        # if nan_indices.numel() > 0:  # 如果存在NaN
        #     print("found NaN in obs_buf")
        #     print("NaN indices (row, column):")
        #     print(nan_indices)
        
        # nan_mask = torch.isnan(privileged_obs_buf)

        # # 获取NaN所在的行列索引
        # nan_indices = torch.nonzero(nan_mask)

        # # 判断是否存在NaN值
        # if nan_indices.numel() > 0:  # 如果存在NaN
        #     print("found NaN in privileged_obs_buf")
        #     print("NaN indices (row, column):")
        #     print(nan_indices)
        
        if self.add_noise:  
            obs_now = obs_buf.clone() + torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()
        self.obs_history.append(obs_now.clone())
        self.critic_history.append(privileged_obs_buf.clone())

        # self.cmd_stack.append((self.commands[:, :3] * torch.logical_not(self.stand_flag.unsqueeze(1))).clone() )
        # self.lin_vel_stack.append(self.base_lin_vel.clone())
        # self.ang_vel_stack.append(self.base_ang_vel.clone())
        # self.contact_forces_l_stack.append(self.contact_forces[:, self.feet_contact_indices[0], : ].clone())
        # self.contact_forces_r_stack.append(self.contact_forces[:, self.feet_contact_indices[1], : ].clone())

        obs_buf_all = torch.cat([self.obs_history.buffer[:,i,:] for i in range(self.obs_history.max_length)], dim=1)  # Note for CirculBuffer index 0 is the latest data
        privileged_obs_buf_all = torch.cat([self.critic_history.buffer[:,i,:] for i in range(self.critic_history.max_length)], dim=1)

        observations = {"policy": obs_buf_all, "critic" : privileged_obs_buf_all}
        return observations

#================================================= rewards ====================================================================
    def _reward_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = joint_pos - pos_target
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        return r

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


    def _reward_foot_slip(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces 
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact 
        with the ground. The speed of the foot is calculated and scaled by the contact condition.
        """
        contact = self.contact_mask
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 7:9], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        # rew *= self.stance_mask
        return (torch.sum(rew, dim=1))

    def _reward_feet_air_time(self):
        """
        Calculates the reward for feet air time, promoting longer steps. This is achieved by
        checking the first contact with the ground after being in the air. The air time is
        limited to a maximum value for reward calculation.
        """
        contact = self.contact_mask
        # stance_mask = self._get_gait_phase()
        stance_mask = self.stance_mask.clone()
        self.contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        air_time = self.feet_air_time.clamp(0, self.cfg.rewards.cycle_time/2.0) * first_contact
        self.feet_air_time *= ~self.contact_filt
        return air_time.sum(dim=1)

    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase. 
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.contact_mask
        # stance_mask = self._get_gait_phase()
        stance_mask = self.stance_mask.clone()
        reward = torch.where(contact == stance_mask, 1, -0.3)
        return torch.mean(reward, dim=1)

    def _reward_orientation(self):
        """
        Calculates the reward for maintaining a flat base orientation. It penalizes deviation 
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 30)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 40)
        return (quat_mismatch + orientation) / 2.
    
    def _reward_feet_contact_forces_swing(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Encourage low contact forces during swing phase
        """
        swing_mask = 1 - self.stance_mask
        rew_l = torch.exp(-torch.norm(self.contact_forces[:, self.feet_contact_indices[0], :3], dim=-1)*0.1) * swing_mask[:,0]
        rew_r = torch.exp(-torch.norm(self.contact_forces[:, self.feet_contact_indices[1], :3], dim=-1)*0.1) * swing_mask[:,1]
        return rew_l+rew_r

    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_contact_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(0, 2000), dim=1)
    
    def _reward_feet_double_stance_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        # indices = torch.logical_and(self.stance_mask[:,0], self.stance_mask[:,1])
        indices = self.stand_flag
        rew = torch.zeros(self.num_envs, device=self.device)
        rew[indices] = torch.exp(-(torch.norm(self.contact_forces[indices, self.feet_contact_indices[0], :2], dim=-1) \
            + torch.norm(self.contact_forces[indices, self.feet_contact_indices[1], :2], dim=-1))*0.1) 
        return rew
    
    def _reward_feet_TD_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        rewl = torch.zeros(self.num_envs, device=self.device)
        rewr = torch.zeros(self.num_envs, device=self.device)
        rewl[self.event_TD[:,0]] = torch.exp(-torch.norm(self.contact_forces[self.event_TD[:,0], self.feet_contact_indices[0], :], dim=-1)*0.01) 
        rewr[self.event_TD[:,1]] = torch.exp(-torch.norm(self.contact_forces[self.event_TD[:,1], self.feet_contact_indices[1], :], dim=-1)*0.01)
        return (rewl+rewr)* (1-self.stand_flag.float())
    
    def _reward_feet_contact_forces_v2(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        contact_forces_error = torch.zeros(self.num_envs, device=self.device)
        # print('forces stack size')
        # print(self.contact_forces_l_stack[-1][:,2].size())
        # print(self.contact_forces_r_stack[-1].size())
        for i in range(self.cfg.contact_forces_frame_stack):
            contact_forces_error += torch.abs(self.contact_forces_l_stack[-1-i][:,2]) + torch.abs(self.contact_forces_r_stack[-1-i][:,2])


        return contact_forces_error / self.cfg.contact_forces_frame_stack
    

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
        # yaw_roll = torch.sum(torch.square(left_yaw_roll), dim=1) + torch.sum(torch.square(right_yaw_roll), dim=1)
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        # return torch.exp(-yaw_roll * 150) - 0.01 * torch.norm(joint_diff, dim=1)

        rew = torch.exp(-yaw_roll * 150) - 0.05 * yaw_roll

        # rew_prerequire=self._reward_feet_swingZ_constant() * self.cfg.rewards.scales.feet_swingZ_constant
        # indices = rew_prerequire > 2
        # rew[indices] *= 1.5
        
        return rew
    
    def _reward_default_joint_pos_swing(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus 
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        left_yaw_roll = self.dof_pos[:, [0,2]]
        right_yaw_roll = self.dof_pos[:, [1,3]]
        swing_mask = 1-self.stance_mask
        rew_l = torch.exp(-(torch.sum(torch.square(left_yaw_roll), dim=1))*150) * swing_mask[:,0]
        rew_r = torch.exp(-(torch.sum(torch.square(right_yaw_roll), dim=1))*150) * swing_mask[:,1]
        
        return rew_l+rew_r

    def _reward_base_height(self):
        """
        Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
        The reward is computed based on the height difference between the robot's base and the average height 
        of its feet when they are in contact with the ground.
        """
        # stance_mask = self._get_gait_phase()
        contact = self.contact_mask
        row_sums = torch.sum(contact, dim=1)
        zero_indices = row_sums < 0.5
        row_sums[zero_indices] = 1
        measured_bottom_heights = torch.sum(
            self.rigid_state[:, self.feet_indices, 2] * contact, dim=1) / row_sums - self.cfg.init_state.ankle_height
        base_height = self.root_states[:, 2] - measured_bottom_heights

        # return torch.exp(-torch.abs((base_height - self.cfg.rewards.base_height_target)**2) * 100)
        reward = torch.exp(-torch.square(base_height - self.cfg.rewards.base_height_target) * 150) - 1.5 * torch.abs(base_height - self.cfg.rewards.base_height_target)
        # reward = torch.where(reward < 0.5, torch.tensor(-0.1, device=self.device), reward)
        reward[zero_indices] = 0.0
        return reward * (1.0 - self.stand_flag.float() )

    def _reward_base_height_stand(self):
        """
        Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
        The reward is computed based on the height difference between the robot's base and the average height 
        of its feet when they are in contact with the ground.
        """
        # stance_mask = self._get_gait_phase()
        contact = self.contact_mask
        row_sums = torch.sum(contact, dim=1)
        zero_indices = row_sums < 0.5
        row_sums[zero_indices] = 1
        measured_bottom_heights = torch.sum(
            self.rigid_state[:, self.feet_indices, 2] * contact, dim=1) / row_sums - self.cfg.init_state.ankle_height
        base_height = self.root_states[:, 2] - measured_bottom_heights

        # return torch.exp(-torch.abs((base_height - self.cfg.rewards.base_height_target)**2) * 100)
        reward = torch.exp(-torch.square(base_height - self.cfg.rewards.base_height_target) * 150) - 1.5 * torch.abs(base_height - self.cfg.rewards.base_height_target)
        # reward = torch.where(reward < 0.5, torch.tensor(-0.1, device=self.device), reward)
        reward[zero_indices] = 0.0
        return reward * self.stand_flag.float()                                                                    
    
    def _reward_base_xy(self):
        """
        Calculates the reward based on the robot's base xy pos, specifically for stepping.
        """
        if self.focus_on_stepping == True:
            base_xy = self.root_states[:, :2] - self.root_pos_ini[:, :2]
        else:
            base_xy = self.root_states[:, :2]*0
        # return torch.exp(-torch.abs((base_height - self.cfg.rewards.base_height_target)**2) * 100)
        return torch.exp(-torch.sum(torch.abs(base_xy),dim=1)*10)


    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 6)
        return rew
    
    def _reward_base_acc_VxyWz(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        The buffer has shape (num_rigid_bodies, 13). State for each rigid body contains position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
        """
        root_acc = self.last_root_vel[:, [0,1,5]] - self.root_states[:, [7,8,12]]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 6)
        return rew
    
    def _reward_base_acc_VzWxy(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        The buffer has shape (num_rigid_bodies, 13). State for each rigid body contains position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
        """
        root_acc = self.last_root_vel[:, [2,4,5]] - self.root_states[:, [9,11,12]]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 6)
        return rew


    def _reward_vel_mismatch_exp(self):
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities. 
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 100)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.)

        c_update = (lin_mismatch + ang_mismatch) / 2.

        return c_update

    def _reward_track_vel_hard(self):
        """
        Calculates a reward for accurately tracking both linear and angular velocity commands.
        Penalizes deviations from specified linear and angular velocity targets.
        """
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.norm(
            self.commands[:, :2] * torch.logical_not(self.stand_flag.unsqueeze(1)) - self.base_lin_vel[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * self.cfg.rewards.tracking_sigma)

        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(
            self.commands[:, 2] * torch.logical_not(self.stand_flag) - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma)

        linear_error = 0.25 * (lin_vel_error + ang_vel_error) # 0.05 -> 0.5

        rew = (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error

        # ids = self.episode_length_buf > 1000
        # rew[ids] -= linear_error[ids]

        return rew
    
    def _reward_track_vel_y_hard(self):
        """
        Calculates a reward for accurately tracking both linear and angular velocity commands.
        Penalizes deviations from specified linear and angular velocity targets.
        """
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.abs(
            self.commands[:, 1] * torch.logical_not(self.stand_flag) - self.base_lin_vel[:, 1])
        lin_vel_error_exp = torch.exp(-lin_vel_error * 50)

        return lin_vel_error_exp

    def _reward_tracking_lin_vel(self):
        """
        Tracks linear velocity commands along the xy axes. 
        Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
        """
        lin_vel_error = torch.sum(torch.square(
            self.commands[:, :2]*torch.logical_not(self.stand_flag.unsqueeze(1)) - self.base_lin_vel[:, :2]), dim=1)
        # lowpass_error = torch.abs( \
        #     self.commands[:, :2]*torch.logical_not(self.stand_flag.unsqueeze(1))- self.lin_vel_filtered[:, :2])
        return torch.exp(-lin_vel_error * self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        """
        Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
        """   
        
        ang_vel_error = torch.square(
            self.commands[:, 2] * torch.logical_not(self.stand_flag) - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma)
    
    def _reward_ang_vel(self):
        """
        supress angular velocity along roll and pitch
        """   
        ang_vel_error = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        return torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma/5.0) - 0.5 * torch.sum(torch.abs(self.base_ang_vel[:,:2]), dim=1)
    
    # def _reward_feet_clearance(self):
    #     """
    #     Calculates reward based on the clearance of the swing leg from the ground during movement.
    #     Encourages appropriate lift of the feet during the swing phase of the gait.
    #     """
    #     # Compute feet contact mask
    #     contact = self.contact_forces[:, self.feet_contact_indices, 2] > 5.

    #     # Get the z-position of the feet and compute the change in z-position
    #     feet_z = self.rigid_state[:, self.feet_contact_indices, 2]
    #     delta_z = feet_z - self.last_feet_z
    #     self.feet_height += delta_z
    #     self.last_feet_z = feet_z

    #     # Compute swing mask
    #     swing_mask = 1 - self._get_gait_phase()

    #     # feet height should be closed to target feet height at the peak
    #     rew_pos = torch.abs(self.feet_height - self.cfg.rewards.target_feet_height) < 0.01
    #     rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
    #     self.feet_height *= ~contact
    #     return rew_pos
    
    def _reward_feet_swingZ(self):
        """
        Calculates reward based on the referenced swing leg motion along z direction. Note: no rewards or ref during stance phase
        """
        ref_feZ=self.feZ_swing + self.cfg.init_state.ankle_height
        # Compute swing mask
        # swing_mask = 1 - self._get_gait_phase()
        swing_mask = 1 - self.stance_mask
        # Get the z-position of the feet
        feet_z = self.rigid_state[:, self.feet_indices, 2]
        dfeet_z = self.rigid_state[:, self.feet_indices, 9]
        # feet height should be closed to ref feet motion
        rew_pos = torch.zeros_like(feet_z)
        good_indices = feet_z >= ref_feZ
        # rew_pos[good_indices] = torch.exp(-torch.square(feet_z[good_indices] - ref_feZ[good_indices])*150)
        err =  torch.square((ref_feZ - feet_z))*100

        # rew_pos = torch.exp(-torch.square(feet_z - ref_feZ)*100) - 2 * torch.abs(feet_z-ref_feZ)
        rew_pos = torch.exp(-err) - 10 * torch.abs(feet_z-ref_feZ)
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        return rew_pos
    
    def _reward_feet_dswingZ(self):
        """
        Calculates reward based on the referenced swing leg motion along z direction. Note: no rewards or ref during stance phase
        The buffer has shape (num_rigid_bodies, 13). State for each rigid body contains position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
        """
        swing_mask = 1 - self.stance_mask
        ref_feZ=self.feZ_swing + self.cfg.init_state.ankle_height
        ref_dfeZ=self.dfeZ_swing * swing_mask
        # Compute swing mask
        # swing_mask = 1 - self._get_gait_phase()
        
        # Get the z-position of the feet
        feet_z = self.rigid_state[:, self.feet_indices, 2]
        dfeet_z = self.rigid_state[:, self.feet_indices, 9]
        dfeet_z_v2 = self.robot.data.body_state_w[:, self.feet_indices, 9]
        # feet height should be closed to ref feet motion
        rew_pos = torch.zeros_like(feet_z)
        # rew_pos[good_indices] = torch.exp(-torch.square(feet_z[good_indices] - ref_feZ[good_indices])*150)
        # print("========")
        # print(ref_dfeZ[0])
        # print(dfeet_z[0])
        # print(dfeet_z_v2[0])
        # print(self.dof_pos[0])
        # print(self.root_states[0,:])
        # print(self.robot.data.root_state_w[0,:])
        # print(self.contact_forces[0,:])
        # print(self.torques[0,:])
        err =  torch.square((ref_dfeZ - dfeet_z))*50

        # rew_pos = torch.exp(-torch.square(feet_z - ref_feZ)*100) - 2 * torch.abs(feet_z-ref_feZ)
        rew_pos = torch.sum(torch.exp(-err), dim=1)

        return rew_pos
    
    def _reward_feet_swingZ_constant(self):
        """
        Calculates reward based on the referenced swing leg motion along z direction. Note: no rewards or ref during stance phase
        """
        
        # des_feZ = self.cfg.rewards.target_feet_height + self.cfg.init_state.ankle_height
        feet_z = self.rigid_state[:, self.feet_indices, 2]
        swing_mask = torch.logical_not(self.stance_mask)
        des_feZ_envs = torch.zeros_like(feet_z)
        des_feZ_envs[swing_mask] = self.cfg.rewards.target_feet_height + self.cfg.init_state.ankle_height
        des_feZ_envs[self.stance_mask.to(torch.bool)] = self.cfg.init_state.ankle_height
        # Get the z-position of the feet
        # feet height should be closed to ref feet motion
        rew_pos = torch.zeros_like(feet_z)
        err =  torch.square((des_feZ_envs - feet_z))*20

        # rew_pos = torch.exp(-torch.square(feet_z - ref_feZ)*100) - 2 * torch.abs(feet_z-ref_feZ)
        rew_pos = torch.exp(-err) - 10 * torch.abs(feet_z-des_feZ_envs)
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)

        # rew_prerequire=self._reward_default_joint_pos() * self.cfg.rewards.scales.default_joint_pos
        # indices = rew_prerequire < 1
        # rew_pos[indices] *= 0.8

        return rew_pos
    
    def _reward_feet_swingZ_constant_v2(self):
        """
        Calculates reward based on the referenced swing leg motion along z direction. Note: no rewards or ref during stance phase
        """
        
        # des_feZ = self.cfg.rewards.target_feet_height + self.cfg.init_state.ankle_height
        feet_z = self.rigid_state[:, self.feet_indices, 2]
        dfeet_z = self.rigid_state[:, self.feet_indices, 9]
        swing_mask = torch.logical_not(self.stance_mask)
        des_feZ_envs = torch.zeros_like(feet_z)
        des_feZ_envs[swing_mask] = self.cfg.rewards.target_feet_height + self.cfg.init_state.ankle_height
        des_feZ_envs[self.stance_mask.to(torch.bool)] = self.cfg.init_state.ankle_height
        des_dfeZ= torch.zeros_like(dfeet_z)
        # Get the z-position of the feet
        # feet height should be closed to ref feet motion
        rew_pos = torch.zeros_like(feet_z)
        err =  torch.square((des_feZ_envs - feet_z))*20

        # rew_pos = torch.exp(-torch.square(feet_z - ref_feZ)*100) - 2 * torch.abs(feet_z-ref_feZ)
        rew_pos = (torch.exp(-err) - 10 * torch.abs(feet_z-des_feZ_envs)) * swing_mask
        rew_dpos = torch.clamp(torch.square((des_dfeZ - dfeet_z)*50) * swing_mask, 0, 1)

        ids = self.episode_length_buf < 400
        rew_dpos[ids] *= 0.0

        rew = torch.sum(rew_pos - 0.6 * rew_dpos, dim=1)

        return rew
    
    def _reward_feet_dswingZ_constant(self):
        """
        Calculates reward based on the referenced swing leg motion along z direction. Note: no rewards or ref during stance phase
        The buffer has shape (num_rigid_bodies, 13). State for each rigid body contains position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
        """
        swing_mask = torch.logical_not(self.stance_mask)
        dfeet_z = self.rigid_state[:, self.feet_indices, 9]
        ref_dfeZ= torch.zeros_like(dfeet_z)
        # feet height should be closed to ref feet motion
        rew_pos = torch.zeros_like(dfeet_z)
        err =  torch.square((ref_dfeZ - dfeet_z))*20
        # rew_pos = torch.exp(-torch.square(feet_z - ref_feZ)*100) - 2 * torch.abs(feet_z-ref_feZ)
        rew_pos = torch.sum(torch.exp(-err) * swing_mask, dim=1)

        rew_prerequire=self._reward_feet_swingZ_constant() * self.cfg.rewards.scales.feet_swingZ_constant
        indices = rew_prerequire < 2
        rew_pos[indices] *= 0

        # rew_prerequire=self._reward_default_joint_pos() * self.cfg.rewards.scales.default_joint_pos
        # indices = rew_prerequire < 2
        # rew_pos[indices] = 0
        return rew_pos
    
    def _reward_feet_dswingZ_constant_v2(self):
        """
        Calculates reward based on the referenced swing leg motion along z direction. Note: no rewards or ref during stance phase
        The buffer has shape (num_rigid_bodies, 13). State for each rigid body contains position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
        """
        swing_mask = torch.logical_not(self.stance_mask)
        dfeet_z = self.rigid_state[:, self.feet_indices, 9]
        ref_dfeZ= torch.zeros_like(dfeet_z)
        # feet height should be closed to ref feet motion
        rew_pos = torch.zeros_like(dfeet_z)
        err =  torch.sum(torch.square((ref_dfeZ - dfeet_z)) * swing_mask * 5, dim=1)
        # rew_pos = torch.exp(-torch.square(feet_z - ref_feZ)*100) - 2 * torch.abs(feet_z-ref_feZ)
        rew_pos = torch.clamp(err, 0, 1)

        rew_prerequire=self._reward_feet_swingZ_constant() * self.cfg.rewards.scales.feet_swingZ_constant
        indices = rew_prerequire < 4.0
        rew_pos[indices] *= 0

        # rew_prerequire=self._reward_default_joint_pos() * self.cfg.rewards.scales.default_joint_pos
        # indices = rew_prerequire < 2
        # rew_pos[indices] = 0
        return rew_pos
    
    def _reward_feet_swingZ_positive(self):
        """
        Calculates reward based on the referenced swing leg motion along z direction.
        """
        ref_feZ=self.feZ_swing + self.cfg.init_state.ankle_height
        # Compute swing mask
        # swing_mask = 1 - self._get_gait_phase()
        swing_mask = 1 - self.stance_mask.clone()
        # Get the z-position of the feet
        feet_z = self.rigid_state[:, self.feet_indices, 2]
        # feet height should be closed to ref feet motion
        rew_pos = torch.zeros_like(feet_z)
        indices = torch.logical_and(feet_z - ref_feZ>=0, feet_z - ref_feZ<=0.02)
        rew_pos[indices] = 1
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        return rew_pos
    
    # def _reward_feet_swingZ_joint(self):
    #     """
    #     Calculates reward based on the referenced swing leg motion along z direction.
    #     """
    #         # dof_knee_swing
    #     cur_knee_joint=self.dof_pos[:,[3,9]]
        
    #     # swing_mask = 1 - self._get_gait_phase()
    #     swing_mask = 1 - self.stance_mask.clone()
        
    #     rew_pos = torch.exp(-torch.square(cur_knee_joint - self.dof_knee_swing)*100)
    #     rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
    #     return rew_pos
    
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
        rewl_m[indices[:,0]] = torch.exp(-torch.abs(self.virtual_leg_l_Mid[indices[:,0],0])*20)
        rewr_m[indices[:,1]] = torch.exp(-torch.abs(self.virtual_leg_r_Mid[indices[:,1],0])*20)
        rew_bal[indices[:,0]] = torch.exp(-torch.abs(self.virtual_leg_l_TD[indices[:,0],0]-self.virtual_leg_r_TD[indices[:,0],0])*10)
        rew_bal[indices[:,1]] = torch.exp(-torch.abs(self.virtual_leg_l_TD[indices[:,1],0]-self.virtual_leg_r_TD[indices[:,1],0])*10)

        return rewl + rewr + rewl_m + rewr_m + 3*rew_bal
    
    # list of rigit_state:
    #   position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13])
    #   rotation: (Quat: x, y, z, w)
    def _reward_feet_orientation(self):
        """
        Try to regulate swing feet orientation through feet euler angle.
        keep the feet flat all the time
        """
        # quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
        feet_eul = self.feet_euler
        # print(feet_eul.size())
        # Compute swing mask
        # swing_mask = 1 - self._get_gait_phase()
        rew1 = torch.exp(-torch.sum(torch.abs(feet_eul[:, 0, :2]), dim=1)  * 30) - torch.sum(torch.abs(feet_eul[:, 0, :2]), dim=1) * 5
        rew2 = torch.exp(-torch.sum(torch.abs(feet_eul[:, 1, :2]), dim=1)  * 30) - torch.sum(torch.abs(feet_eul[:, 1, :2]), dim=1) * 5
        return rew1+rew2
    
    def _reward_feet_stance_orientation(self):
        """
        Try to regulate stance feet orientation through feet euler angle.
        """
        # quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
        feet_eul = self.feet_euler
        # print(feet_eul.size())
        # Compute swing mask
        # stance_mask = self._get_gait_phase()
        stance_mask = self.stance_mask.clone()
        rew1 = torch.exp(-torch.sum(torch.abs(feet_eul[:, 0, :2]), dim=1)  * 10) * stance_mask[:,0]
        rew2 = torch.exp(-torch.sum(torch.abs(feet_eul[:, 1, :2]), dim=1)  * 10) * stance_mask[:,1]
        return rew1+rew2


        # feet_quat_0 = self.rigid_state[:, self.feet_indices[0], 3:7].view(-1,4)
        # feet_quat_1 = self.rigid_state[:, self.feet_indices[1], 3:7].view(-1,4)
        # feet_rpy_0 = self.quaternion_to_euler(feet_quat_0)
        # feet_rpy_1 = self.quaternion_to_euler(feet_quat_1)
        # rew_0 = torch.sum(feet_rpy_0[:,:2]**2, dim=1)
        # rew_1 = torch.sum(feet_rpy_1[:,:2]**2, dim=1)
        # # Compute swing mask
        # swing_mask = 1 - self._get_gait_phase()
        # swing_mask_0 = swing_mask[:,0]
        # swing_mask_1 = swing_mask[:,1]
        # rew= rew_0 * swing_mask_0 + rew_1 * swing_mask_1
        # return rew

    def _reward_feet_orientation_v2(self):
        """
        try to keep the ankle joint as the default ankle joints
        activate until the end of the swing phase, when try to set the feet euler angle flat
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
        no_contact = 1 - self.contact_mask.float()

        phiTmp=torch.fmod(self.phi * 2, 1)

        joint_cur = self.last_actions
        # swing_mask = 1 - self._get_gait_phase()
        swing_mask = 1 - self.stance_mask.clone()
        swing_mask_l = swing_mask[:,0]
        swing_mask_r = swing_mask[:,1]

        left_ankle = joint_cur[:,[4,5]]
        right_ankle = joint_cur[:,[10,11]]
        default_ankle_l = torch.zeros_like(left_ankle)
        default_ankle_r = torch.zeros_like(right_ankle)
        default_ankle_l[:,0]=self.robot.data.default_joint_pos[0,8]
        default_ankle_l[:,1]=self.robot.data.default_joint_pos[0,10]
        default_ankle_r[:,0]=self.robot.data.default_joint_pos[0,9]
        default_ankle_r[:,1]=self.robot.data.default_joint_pos[0,11]

        # default_ankle_l = [self.cfg.init_state.default_joint_angles['left-ankle-pitch'], 
        #                             self.cfg.init_state.default_joint_angles['left-ankle-roll']]
        # default_ankle_r = [self.cfg.init_state.default_joint_angles['right-ankle-pitch'], 
        #                             self.cfg.init_state.default_joint_angles['right-ankle-roll']]
        rew_left = torch.sum(torch.square(left_ankle-default_ankle_l), dim=1) 
        rew_right = torch.sum(torch.square(right_ankle-default_ankle_r), dim=1)

        rew_left= (torch.exp(-rew_left*50) - 0.2* torch.sum(torch.abs(left_ankle-default_ankle_l), dim=1) )*swing_mask_l * (1-self.stand_flag.float())
        rew_right= (torch.exp(-rew_right*50) - 0.2* torch.sum(torch.abs(right_ankle-default_ankle_r), dim=1))*swing_mask_r * (1-self.stand_flag.float())

        feet_eul = self.feet_euler
        ankle = torch.zeros((self.num_envs,2), device=self.device)
        if self.cfg.rewards.enable_td_pitch_angle:
            ankle[:,1] = -5./180.*3.1415
        rew1 = torch.exp(-torch.sum(torch.square(feet_eul[:, 0, :2]-ankle * no_contact[:,0].unsqueeze(1)), dim=1)  * 40) * swing_mask[:,0] * (1-self.stand_flag.float())
        rew2 = torch.exp(-torch.sum(torch.square(feet_eul[:, 1, :2]-ankle * no_contact[:,1].unsqueeze(1)), dim=1)  * 40) * swing_mask[:,1] * (1-self.stand_flag.float())

        scale = 10
        weight = 0.4
        swing_seg1 = (phiTmp<=0.6)*weight*scale
        swing_seg2 = (phiTmp>0.6)*(1-weight)*scale

        return rew_left * swing_seg1 + rew_right * swing_seg1 + rew1 * swing_seg2 + rew2 * swing_seg2

    def _reward_low_speed_vxy(self):
        """
        Rewards or penalizes the linear velocity along x and y diretion, based on base's speed relative to the commanded speed. 
        This function checks if the robot is moving too slow, too fast, or at the desired speed, 
        and if the movement direction matches the command.
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_lin_vel[:, :2])
        absolute_command = torch.abs(self.commands[:, :2] * torch.logical_not(self.stand_flag).unsqueeze(1))

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.1 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(
            self.base_lin_vel[:, :2]) != torch.sign(self.commands[:, :2] * torch.logical_not(self.stand_flag).unsqueeze(1))

        # Initialize reward tensor
        reward = torch.zeros_like(self.base_lin_vel[:, :2])

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.
        # Speed within desired range
        reward[speed_desired] = 1.2
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        # return reward * (self.commands[:, 0].abs() > 0.1)
        return torch.sum(reward, dim=1)
    
    def _reward_low_speed_wz(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed. 
        This function checks if the robot is moving too slow, too fast, or at the desired speed, 
        and if the movement direction matches the command.
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_ang_vel[:, 2])
        absolute_command = torch.abs(self.commands[:, 2] * torch.logical_not(self.stand_flag))

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = (torch.sign(
            self.base_ang_vel[:, 2]) != torch.sign(self.commands[:, 2] * torch.logical_not(self.stand_flag)))

        # Initialize reward tensor
        reward = torch.zeros_like(absolute_speed)

        # print('reward  ', reward.size())

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.
        # Speed within desired range
        reward[speed_desired] = 1.2
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        # return reward * (self.commands[:, 0].abs() > 0.1)
        # print('wz reward  ', reward.size())
        return reward
    
    def _reward_torques(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        # stance_mask = self._get_gait_phase()
        stance_mask = self.stance_mask.clone()
        weight=torch.ones_like(self.torques)
        weight[:,4]= 0.1 * stance_mask[:,0]+ 1 * ( 1 - stance_mask[:,0])
        weight[:,5]= 0.1 * stance_mask[:,0]+ 1 * ( 1 - stance_mask[:,0])
        weight[:,10]= 0.1 * stance_mask[:,1]+ 1 * ( 1 - stance_mask[:,1])
        weight[:,11]= 0.1 * stance_mask[:,1]+ 1 * ( 1 - stance_mask[:,1])
        return torch.sum(torch.square(self.torques)*weight, dim=1)
    
    def _reward_torques_max(self):
        """
        Penalizes the use of high torques in the robot's ankles joints.
        """
        # stance_mask = self._get_gait_phase()
        stance_mask = self.stance_mask.clone()
        rew = torch.zeros((self.num_envs, 4), device=self.device)
        torques_ankles_pitch=torch.abs(self.torques[:,[4,10]])
        torques_ankles_roll=torch.abs(self.torques[:,[5,11]])
        torques_ankles_pitch_max=50
        torques_ankles_roll_max=40
        rew_tor_pitch=torch.sum(torch.clamp(torques_ankles_pitch - torques_ankles_pitch_max, min=0.0, max=None), dim=1)
        rew_tor_roll=torch.sum(torch.clamp(torques_ankles_roll - torques_ankles_roll_max, min=0.0, max=None),dim=1)
        return (rew_tor_roll+rew_tor_pitch)
    
    # def _reward_ankle_torques(self):
    #     """
    #     no use, bad results
    #     """
    #     stance_mask = self._get_gait_phase()
    #     rew_pitch = torch.zeros((self.num_envs, 2), device=self.device)
    #     rew_roll = torch.zeros((self.num_envs, 2), device=self.device)
    #     torques_ankles_pitch=torch.abs(self.torques[:,[4,10]])
    #     torques_ankles_roll=torch.abs(self.torques[:,[5,11]])
    #     torques_ankles_pitch_rew_min=20
    #     torques_ankles_pitch_rew_max=50
    #     torques_ankles_roll_rew_min=20
    #     torques_ankles_roll_rew_max=50
    #     torques_ankles_pitch_too_low = torques_ankles_pitch < torques_ankles_pitch_rew_min
    #     torques_ankles_pitch_too_high = torques_ankles_pitch > torques_ankles_pitch_rew_max
    #     pitch_desired = ~(torques_ankles_pitch_too_low | torques_ankles_pitch_too_high)
    #     torques_ankles_roll_too_low = torques_ankles_roll < torques_ankles_roll_rew_min
    #     torques_ankles_roll_too_high = torques_ankles_roll > torques_ankles_roll_rew_max
    #     roll_desired = ~(torques_ankles_roll_too_low | torques_ankles_roll_too_high)

    #     rew_pitch[pitch_desired] = 1.2
    #     rew_pitch[pitch_desired] *= stance_mask[pitch_desired]
    #     rew_roll[roll_desired] = 1.2 
    #     rew_roll[roll_desired] *= stance_mask[roll_desired]

    #     return torch.sum(rew_roll+rew_pitch, dim=1)


    def _reward_dof_vel(self):
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and 
        more controlled movements.
        """
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_ankle_vel_added(self):
        """
        DOFs of asset:
        0: 'J_hip_l_roll' (Rotation)
        1: 'J_hip_l_yaw' (Rotation)
        2: 'J_hip_l_pitch' (Rotation)
        3: 'J_knee_l_pitch' (Rotation)
        4: 'J_ankle_l_pitch' (Rotation)
        5: 'J_ankle_l_roll' (Rotation)
        6: 'J_hip_r_roll' (Rotation)
        7: 'J_hip_r_yaw' (Rotation)
        8: 'J_hip_r_pitch' (Rotation)
        9: 'J_knee_r_pitch' (Rotation)
        10: 'J_ankle_r_pitch' (Rotation)
        11: 'J_ankle_r_roll' (Rotation)
        """
        dof_ankle_vel = self.dof_vel[:,[4,5,10,11]]
        return torch.sum(torch.square(dof_ankle_vel), dim=1)
    
    def _reward_dof_vel_swing(self):
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and 
        more controlled movements.
        """
        """
        DOFs of asset:
        0: 'left-hip-roll' (Rotation)
        1: 'left-hip-yaw' (Rotation)
        2: 'left-hip-pitch' (Rotation)
        3: 'left-knee-pitch' (Rotation)
        4: 'left-ankle-pitch' (Rotation)
        5: 'left-ankle-roll' (Rotation)
        6: 'right-hip-roll' (Rotation)
        7: 'right-hip-yaw' (Rotation)
        8: 'right-hip-pitch' (Rotation)
        9: 'right-knee-pitch' (Rotation)
        10: 'right-ankle-pitch' (Rotation)
        11: 'right-ankle-roll' (Rotation)
        """
        # swing_mask = 1 - self._get_gait_phase()
        swing_mask = 1 - self.stance_mask.clone()
        left_ankle=[4,5]
        right_ankle=[10,11]
        rew1 = torch.sum(torch.square(self.dof_vel[:,left_ankle]), dim=1) * swing_mask[:,0]
        rew2 = torch.sum(torch.square(self.dof_vel[:,right_ankle]), dim=1) * swing_mask[:,1]
        return rew1+rew2
    
    
    def _reward_dof_acc(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_dof_acc_swing(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        DOFs of asset:
        0: 'left-hip-roll' (Rotation)
        1: 'left-hip-yaw' (Rotation)
        2: 'left-hip-pitch' (Rotation)
        3: 'left-knee-pitch' (Rotation)
        4: 'left-ankle-pitch' (Rotation)
        5: 'left-ankle-roll' (Rotation)
        6: 'right-hip-roll' (Rotation)
        7: 'right-hip-yaw' (Rotation)
        8: 'right-hip-pitch' (Rotation)
        9: 'right-knee-pitch' (Rotation)
        10: 'right-ankle-pitch' (Rotation)
        11: 'right-ankle-roll' (Rotation)
        """
        swing_mask = 1 - self.stance_mask
        phiTmp=torch.fmod(self.phi * 2, 1)
        indices = phiTmp > 0.6
        rew_l = torch.exp(-(torch.abs(self.dof_vel[:,3])) * 10) * swing_mask[:,0] * indices
        rew_r = torch.exp(-(torch.abs(self.dof_vel[:,9])) * 10) * swing_mask[:,1] * indices
        return rew_l + rew_r
    
    def _reward_collision(self):
        """
        Penalizes collisions of the robot with the environment, specifically focusing on selected body parts.
        This encourages the robot to avoid undesired contact with objects or surfaces.
        """
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        return term_1 + term_2
    
    def _reward_action_smoothness_minimum(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_3
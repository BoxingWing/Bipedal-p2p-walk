from __future__ import annotations

import torch
from typing import Tuple
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sensors import Imu

from .cfg_base import BaseEnvCfg

from isaaclab.utils.math import euler_xyz_from_quat, quat_rotate_inverse, quat_apply, wrap_to_pi, quat_from_euler_xyz
from isaaclab.utils import CircularBuffer
from isaaclab.utils.dict import class_to_dict

from typing import TYPE_CHECKING, Literal
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

class BaseEnv(DirectRLEnv):
    cfg: BaseEnvCfg

    def __init__(self, cfg: BaseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        body_names = self.robot.data.body_names
        feet_names = [s for s in body_names if any(foot_name in s for foot_name in self.cfg.asset.foot_name)]
        knee_names = [s for s in body_names if self.cfg.asset.knee_name in s]

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            indices, _ = self.robot.find_bodies(feet_names[i])
            self.feet_indices[i] = indices[0]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            indices, _ = self.robot.find_bodies(knee_names[i])
            self.knee_indices[i] = indices[0]
        # print('feet_indices\n',self.feet_indices.view(-1).tolist())

        self.feet_contact_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            indices, _ = self._contact_sensor.find_bodies(feet_names[i])
            self.feet_contact_indices[i] = indices[0]

        self.num_dofs = len(self.robot.data.joint_names)
        self.num_bodies = len(self.robot.data.body_names)
        print("=====joint names=====")
        for i, name in enumerate(self.robot.data.joint_names):
            print(f"{i}: {name}")
        print("=====body names=====")
        for i, name in enumerate(self.robot.data.body_names):
            print(f"{i}: {name}")

        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
        
        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            indices, _ = self._contact_sensor.find_bodies(penalized_contact_names[i])
            self.penalised_contact_indices[i] = indices[0]

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            indices, _ = self._contact_sensor.find_bodies(termination_contact_names[i])
            self.termination_contact_indices[i] = indices[0]

        self.body_mass = self.robot.data.default_mass[:, 0].to(self.device).unsqueeze(-1)

        self.dt = cfg.decimation * cfg.sim.dt
        self.obs_scales = cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(cfg.rewards.scales)
        self.command_ranges = class_to_dict(cfg.commands.ranges)
        
        self.action_scale = self.cfg.action_scale

        self.dof_pos = self.robot.data.joint_pos
        self.dof_vel = self.robot.data.joint_vel
        self.contact_forces = self._contact_sensor.data.net_forces_w

        # rigid-body-related variables, due to the lazy-update design for RigidObjectData in Isaaclab, these values need to update before used or after env reset

        self.root_states = self.robot.data.root_state_w # w.r.t the world frame !
        self.rigid_state = self.robot.data.body_state_w
        self.base_quat = self.root_states[:, 3:7]
        self.base_euler_xyz = torch.zeros((self.num_envs,3),dtype=torch.float, device=self.device)
        self.base_yaw_loopCount = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.base_euler_xyz_multiLoop = torch.zeros((self.num_envs,3),dtype=torch.float, device=self.device)
        self.feet_quat = torch.zeros((self.num_envs,2,4),dtype=torch.float, device=self.device)
        self.feet_euler = torch.zeros(self.num_envs, len(self.feet_indices), 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_lin_vel = self.robot.data.root_lin_vel_b
        self.base_ang_vel = self.robot.data.root_ang_vel_b
        self.base_lin_vel_w = self.robot.data.root_lin_vel_w
        self.base_ang_vel_w = self.robot.data.root_ang_vel_w
        self.projected_gravity = self.robot.data.projected_gravity_b

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = torch.tensor([0., 0., -1.], device=self.device).repeat(self.num_envs, 1)
        self.forward_vec = torch.tensor([1., 0., 0.], device=self.device).repeat(self.num_envs, 1)
        self.torques = self.robot.data.applied_torque
        self.last_actions = torch.zeros(self.num_envs, self.cfg.action_space, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.cfg.action_space, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_rigid_state = torch.zeros_like(self.rigid_state)
        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.pos_xy, self.obs_scales.pos_xy, self.obs_scales.pos_theta, self.obs_scales.pos_xy, self.obs_scales.pos_xy, self.obs_scales.pos_theta], 
                                           device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        # self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        
        self.measured_heights = 0
        self.obs_history = CircularBuffer(max_len=self.cfg.frame_stack, batch_size=self.num_envs, device=self.device)
        self.critic_history = CircularBuffer(max_len=self.cfg.c_frame_stack, batch_size=self.num_envs,device=self.device)
        for _ in range(self.cfg.frame_stack):
            self.obs_history.append(torch.zeros(
                self.num_envs, self.cfg.num_single_obs, dtype=torch.float, device=self.device))
        for _ in range(self.cfg.c_frame_stack):
            self.critic_history.append(torch.zeros(
                self.num_envs, self.cfg.single_num_privileged_obs, dtype=torch.float, device=self.device))

        material = self.robot.root_physx_view.get_material_properties().to(self.device)

        self.fritions = material[:, -1, 0] # only extract one of three body shapes' static friction, still dont know which body uses this friction
        self.feet_applied_forces = torch.zeros((self.num_envs,2,3), device=self.device)
        self.feet_applied_torques = torch.zeros((self.num_envs,2,3), device=self.device)
        self.base_apply_torque = torch.zeros((self.num_envs,1,3), device=self.device)
        self.base_apply_torque[:,0,:2] = torch_rand_float(-5, 5, (self.num_envs,2), device=self.device)

        self.env_stiffness=torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        self.env_damping=torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        self.env_actuator_effort_limit=torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        self.env_stiffness_ratio=torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        self.env_damping_ratio=torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        for actuator in self.robot.actuators.values():
            self.env_stiffness[:, actuator.joint_indices] = actuator.stiffness[:]
            self.env_damping[:, actuator.joint_indices] = actuator.damping[:]
            self.env_actuator_effort_limit[:, actuator.joint_indices] = actuator.effort_limit[:]
        self.env_stiffness_ratio = self.env_stiffness / self.robot.data.default_joint_stiffness
        self.env_damping_ratio = self.env_damping / self.robot.data.default_joint_damping
        self.processed_actions_used = torch.zeros(self.num_envs, self.cfg.action_space, dtype=torch.float, device=self.device, requires_grad=False)
        self.processed_actions_used = self.robot.data.default_joint_pos
        self.push_velocity_w = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device)

        # random friction
        static_friction = torch.FloatTensor(self.num_envs, 3, 1).uniform_(self.cfg.domain_rand.static_friction_range[0], self.cfg.domain_rand.static_friction_range[1])
        self.static_friction = static_friction.clone().to(self.device)
        dynamic_friction = static_friction.clone()
        restitution = torch.FloatTensor(self.num_envs, 3, 1).uniform_(0.0, 0.0)
        materials = torch.cat([static_friction, dynamic_friction, restitution], dim=-1)
        indices = torch.tensor(range(3 * self.num_envs), dtype=torch.int)
        # Add friction to robot
        self.scene["robot"].root_physx_view.set_material_properties(materials, indices)

        # random base link com
        # get the current com of the bodies (num_assets, num_bodies)
        coms = self.scene["robot"].root_physx_view.get_coms().clone()[:, 0, :3]

        # Randomize the com in range -max displacement to max displacement
        coms += torch.rand_like(coms) * 2 * self.cfg.domain_rand.com_rand_range -  self.cfg.domain_rand.com_rand_range

        # Set the new coms
        if self.cfg.domain_rand.enable_base_com_rand:
            new_coms = self.scene["robot"].root_physx_view.get_coms().clone()
            new_coms[:, 0, 0:3] = coms
            all_env_idx = torch.tensor([x for x in range(self.num_envs)])
            self.scene["robot"].root_physx_view.set_coms(new_coms, all_env_idx)
        

        self._prepare_reward_function()


    def _setup_scene(self): # Will be called in super().__init__()
        self.robot = Articulation(self.cfg.robot_cfg)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self._imu_sensor_base = Imu(self.cfg.imu_base)
        self._height_scanner = RayCaster(self.cfg.height_scanner)

        # add articultion and sensors to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.scene.sensors["imu_base"] = self._imu_sensor_base
        self.scene.sensors["height_scanner"] = self._height_scanner
        
        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        self.env_origins = self.scene.env_origins
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def push_by_setting_velocity(self, env_ids: torch.Tensor):
        """Push the asset by setting the root velocity to a random value within the given ranges.

        This creates an effect similar to pushing the asset with a random impulse that changes the asset's velocity.
        It samples the root velocity from the given ranges and sets the velocity into the physics simulation.

        The function takes a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
        are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form ``(min, max)``.
        If the dictionary does not contain a key, the velocity is set to zero for that axis.
        """
        # extract the used quantities (to enable type-hinting)
        asset: RigidObject | Articulation = self.scene.articulations["robot"]

        # velocities
        vel_w = asset.data.root_vel_w[env_ids]
        # sample random velocities
        range_list = [self.cfg.normalization.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        self.push_velocity_w = torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device)
        random_vel = math_utils.sample_uniform(
            ranges[:, 0],
            ranges[:, 1],
            (len(env_ids), 6),
            device=self.device
        )

        self.push_velocity_w[env_ids] = random_vel
        vel_w += self.push_velocity_w[env_ids,:]

        # set the velocities into the physics simulation
        asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()
        self.actions = torch.clip(self.actions, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)
        actions_noise_scale = torch.rand_like(self.actions) * self.cfg.noise.action_out_noise
        actions_noised = self.actions * (1.0 + actions_noise_scale) 
        self.processed_actions = self.cfg.action_scale * actions_noised + self.robot.data.default_joint_pos

        env_ids = (self.episode_length_buf % int(self.cfg.normalization.push_interval / self.dt)==0).nonzero(as_tuple=False).flatten()
        if self.cfg.normalization.push_robots :
            self.push_by_setting_velocity(env_ids)

    def _apply_action(self):
        k = 0.9
        self.processed_actions_used = (1-k)*self.processed_actions + k*self.processed_actions_used
        self.robot.set_joint_position_target(self.processed_actions_used)
        
        self.robot.set_joint_position_target(self.processed_actions)

        self.feet_applied_forces[:,0,:3] = torch_rand_float(-1.5, 1.5, (self.num_envs,3), device=self.device)
        self.feet_applied_forces[:,1,:3] = torch_rand_float(-1.5, 1.5, (self.num_envs,3), device=self.device)

        self.robot.set_external_force_and_torque(self.feet_applied_forces, self.feet_applied_torques, body_ids=self.feet_indices)
        if self.cfg.enable_base_torques:
            self.robot.set_external_force_and_torque(self.base_apply_torque * 0.0, self.base_apply_torque, body_ids=0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute and return the done flags for the environment.
            must return two types of flags: reset_terminated, reset_time_outs
        """
        # update variables
        self.root_states = self.robot.data.root_state_w # w.r.t the world frame !
        self.rigid_state = self.robot.data.body_state_w

        self.update_base_euler()

        self.feet_quat = self.rigid_state[:, self.feet_indices, 3:7]
        self.feet_euler[:, 0, :] = self.get_euler_xyz(self.feet_quat[:,0,:].view(self.num_envs,4))
        self.feet_euler[:, 1, :] = self.get_euler_xyz(self.feet_quat[:,1,:].view(self.num_envs,4))
        self.base_lin_vel = self.robot.data.root_lin_vel_b
        self.base_ang_vel = self.robot.data.root_ang_vel_b
        self.base_lin_vel_w = self.robot.data.root_lin_vel_w
        self.base_ang_vel_w = self.robot.data.root_ang_vel_w
        self.projected_gravity = self.robot.data.projected_gravity_b

        # check termination
        reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        # reset_buf_2 = torch.any(torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) > 80. / 180.0 * 3.1415, dim=0)
        # reset_buf = torch.logical_or(reset_buf_1, reset_buf_2)
        time_out_buf = self.episode_length_buf >= self.max_episode_length - 1 # no terminal reward for time-outs

        # resample commands
        self._check_and_resample_commands(torch.logical_or(reset_buf, time_out_buf), False)
        
        return reset_buf, time_out_buf
    
    def _get_rewards(self) -> torch.Tensor:
        self._prepare_rewards_variables()

        rew_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            # print(f'{name}  ', rew)
            rew_buf += rew
            self._episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            rew_buf[:] = torch.clip(rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            rew_buf += rew
            self._episode_sums["termination"] += rew
        return rew_buf


    def _reset_idx(self, env_ids: torch.Tensor ):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # reset lazy update variables
        self.root_states[env_ids,:] = self.robot.data.root_state_w[env_ids,:] # w.r.t the world frame !
        self.rigid_state[env_ids,:,:] = self.robot.data.body_state_w[env_ids,:,:]

        # reset buffers
        self.last_last_actions[env_ids,:] *= 0.
        self.last_actions[env_ids,:] *= 0.
        self.last_rigid_state[env_ids,:,:] = self.rigid_state[env_ids,:,:]
        self.last_dof_vel[env_ids,:] *= 0.
        self.last_dof_pos[env_ids,:] *= 0.
        self.last_root_vel[env_ids,:] *= 0.0

        # self.base_quat = self.root_states[:, 3:7]
        # self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.reset_base_euler(env_ids)

        self.feet_quat[env_ids,:2,:] = self.rigid_state[env_ids.unsqueeze(1), self.feet_indices, 3:7]
        self.feet_euler[env_ids, 0, :] = self.get_euler_xyz(self.feet_quat[env_ids,0,:])
        self.feet_euler[env_ids, 1, :] = self.get_euler_xyz(self.feet_quat[env_ids,1,:])
        self.base_lin_vel[env_ids,:] = self.robot.data.root_lin_vel_b[env_ids,:]
        self.base_ang_vel[env_ids,:] = self.robot.data.root_ang_vel_b[env_ids,:]
        self.base_lin_vel_w[env_ids,:] = self.robot.data.root_lin_vel_w[env_ids,:]
        self.base_ang_vel_w[env_ids,:] = self.robot.data.root_ang_vel_w[env_ids,:]
        self.projected_gravity[env_ids,:] = self.robot.data.projected_gravity_b[env_ids,:]
        self.processed_actions_used[env_ids,:] = self.robot.data.default_joint_pos[env_ids,:]
        self.push_velocity_w[env_ids,:] = 0.0   

        # Reset robot state

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        pos_rand_add = torch_rand_float(lower=-0.01, upper=0.01, shape=joint_pos.size(), device=self.device) * 0.0
        vel_rand_add = torch_rand_float(lower=-0.01, upper=0.01, shape=joint_vel.size(), device=self.device) * 0.0

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.terrain.env_origins[env_ids]
        default_root_state[:, 2] += 0.04

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos+pos_rand_add, joint_vel+vel_rand_add, None, env_ids)

        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)
        
    
    def _get_observations(self) -> dict:
        raise NotImplementedError

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, which will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self._episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                                for name in self.reward_scales.keys()}
    
    def _check_and_resample_commands(self, env_ids_extra, isIni):
        """ Randommly select commands of some environments
        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        
        # if env_ids_extra is None or len(env_ids_extra) == self.num_envs:
        #     env_ids_extra = self.robot._ALL_INDICES
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        env_ids = torch.logical_or(env_ids, env_ids_extra)
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
    
    def _prepare_rewards_variables(self):
        pass

    def _update_obs_history_variables(self):
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_dof_pos[:] = self.dof_pos[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_rigid_state[:] = self.rigid_state[:]

    def get_euler_xyz(self, quat):
        r, p, w = euler_xyz_from_quat(quat)
        # stack r, p, w in dim1
        euler_xyz = torch.stack((r, p, w), dim=1)
        euler_xyz[euler_xyz > torch.pi] -= 2 * torch.pi
        return euler_xyz

    def get_quat_yaw(self, yaw):
        roll = torch.zeros_like(yaw)
        pitch = torch.zeros_like(yaw)
        quat = quat_from_euler_xyz(roll, pitch, yaw)
        return quat
    
    def update_base_euler(self):
        base_euler_cur = self.get_euler_xyz(self.robot.data.root_state_w[:, 3:7])
        
        ids = (base_euler_cur[:,2] - self.base_euler_xyz[:,2]) < -6.0
        self.base_yaw_loopCount[ids] += 1
        ids = (base_euler_cur[:,2] - self.base_euler_xyz[:,2]) > 6.0
        self.base_yaw_loopCount[ids] -= 1

        self.base_euler_xyz = base_euler_cur.clone()
        self.base_euler_xyz_multiLoop = base_euler_cur.clone()
        self.base_euler_xyz_multiLoop[:,2] = self.base_euler_xyz_multiLoop[:,2] + self.base_yaw_loopCount * torch.pi * 2.0

    def reset_base_euler(self, env_ids):
        base_euler_cur = self.get_euler_xyz(self.robot.data.root_state_w[env_ids, 3:7])
        self.base_euler_xyz[env_ids,:] = base_euler_cur.clone()
        self.base_euler_xyz_multiLoop[env_ids,:] = base_euler_cur.clone()
        self.base_yaw_loopCount[env_ids] *= 0.0

@torch.jit.script
def torch_rand_float(lower: float, upper: float, shape: Tuple[int, int], device: str):
    """ type: (float, float, Tuple[int, int], str) -> Tensor """
    return (upper - lower) * torch.rand(tuple(shape), device=device) + lower

@torch.jit.script
def torch_rand_float_ranges(lower1: float, upper1: float, lower2: float, upper2: float, shape: Tuple[int, int], device: str):
    range1_valid = upper1 > lower1
    range2_valid = upper2 > lower2
    
    if not (range1_valid or range2_valid):
        raise ValueError("At least one range must be valid (upper > lower)")
    
    range1_length = upper1 - lower1 if range1_valid else 0.0
    range2_length = upper2 - lower2 if range2_valid else 0.0
    
    total_length = range1_length + range2_length
    
    random_values = torch.rand(shape, device=device) * total_length
    
    if range1_valid and range2_valid:
        random_vector = torch.where(
            random_values < range1_length,
            lower1 + random_values,
            lower2 + (random_values - range1_length)
        )
    elif range1_valid:
        random_vector = lower1 + random_values
    else:
        random_vector = lower2 + random_values
    
    return random_vector

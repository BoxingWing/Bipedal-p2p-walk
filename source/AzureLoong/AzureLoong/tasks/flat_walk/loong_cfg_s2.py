
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors import ImuCfg
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

from AzureLoong.assets.AzureLoong import AZURELOONG_CFG

from .base_scripts.loong_cfg_base import LoongEnvBaseCfg

@configclass
class EventCfg:
  # startup
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.15, 1.3),
            "dynamic_friction_range": (0.2, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.5, 1.5),
            "damping_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (0.0, 5.0),
            "operation": "add",
        },
    )
    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-0.5, 0.5), 
                "y": (-0.5, 0.5), 
                "roll": (-0.02, 0.02),
                "pitch": (-0.02, 0.02),
                "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0., 0.),
                "y": (-0., 0.),
                "z": (-0., 0.),
                "roll": (-0., 0.),
                "pitch": (-0., 0.),
                "yaw": (-0., 0.),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.0, 0.0),
            "velocity_range": (-0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 9.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        },
    )

@configclass
class LoongEnvS2Cfg(LoongEnvBaseCfg):
    # env
    # vel_frame_stack = 10  # stored frame num for vel reward evaluation
    # contact_forces_frame_stack = 5  # stored frame num for contact forces reward evaluation
    
    num_single_obs = 46 # defined in compute_observations(), command 5D, q 12D, dq 12D, actions 12D, base_ang_vel 3D, base_euler_xy 2D
    frame_stack = 15  # stored frame num for observations
    observation_space = int(frame_stack * num_single_obs)

    single_num_privileged_obs = 85 # defined in compute_observations()
    c_frame_stack = 3 # stored frame num for privileged (or critic) observations
    state_space = int(c_frame_stack * single_num_privileged_obs)

    action_space = 12 # actuated joint numbers
    num_envs = 4096

    episode_length_s = 40     # episode length in seconds
    decimation = 10
    action_scale = 0.25

    use_ref_actions = False   # speed up training by using reference actions
    enable_stand = True
    focus_stand = False
    enable_imu_filter = False
    enable_imu_offset_disturbance = False
    enable_imu_rand_reset = False
    focus_on_stepping = False
    
    class asset:
        foot_name = ["ankle_l_roll", "ankle_r_roll"] # any body name contains this string segment will recognized as foot
        knee_name = "knee"
        terminate_after_contacts_on = ['base_link']
        penalize_contacts_on = ["base_link"]

    # simulation  # use_fabric=True the GUI will not update
    sim: SimulationCfg = SimulationCfg(
        device = "cuda:0", # can be "cpu", "cuda", "cuda:<device_id>"
        dt= 0.001,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx= PhysxCfg(
            solver_type=1,
            max_position_iteration_count=4,
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.1,
            gpu_max_rigid_contact_count=2**23
        )
    )
    
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

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=3.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # reset
    class init_state:
        '''
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
        pos = [0.0, 0.0, 1.1]  # height from base to the ground 
        ankle_height = 0.07 # the distance between the ankle and the foot bottom
        l_hip_yaw_id = 3
        l_ankle_roll_id = 11
        r_hip_yaw_id = 4
        r_ankle_roll_id = 12

    class commands:
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.  # time before command are changed[s]
        stand_resampling_time = 8. # time before stand resample command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-0.3, 0.6]   # min max [m/s]
            lin_vel_y = [-0.3, 0.3]   # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5] # min max [rad/s]
            heading = [-3.14, 3.14]
    
    class noise:
        add_noise = True
        noise_level = 0.6    # scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 0.5
            ang_vel = 0.1
            lin_vel = 0.05
            quat = 0.03
            height_measurements = 0.1

    # reward scales
    class rewards:
        base_height_target = 1.08
        min_dist = 0.25 # minimum distance between two feet
        max_dist = 0.6
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.17    # rad, scale for amplifying ref joint angle
        target_feet_height = 0.1        # m, desired feet height during swing phase
        cycle_time = 0.9                # sec, gait period including swing and stance
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(error*sigma)
        tracking_sigma = 30
        max_contact_force = 700  # Forces above this value are penalized
        enable_td_pitch_angle = False

        class scales:
            # reference motion tracking
            joint_pos = 0 # 1.6
            feet_clearance = 0 # 0.6
            feet_contact_number = 1
            feet_swingZ = 8
            feet_dswingZ = 3
            
            # feet_swingZ_constant = 0 # 6
            # feet_swingZ_constant_v2 = 7
            # feet_dswingZ_constant = 0 # 5
            # feet_dswingZ_constant_v2 = 0

            virtual_leg_sym = 60
            feet_orientation = 2 #2.
            feet_orientation_v2 = 0
            feet_stance_orientation = 0 
            # gait
            feet_air_time = 0 #1.
            foot_slip = -0.1
            feet_distance = 0
            knee_distance = 0
            # contact
            feet_contact_forces = -0.003
            # vel tracking
            tracking_lin_vel = 2 #2 # use cmd histroy
            tracking_ang_vel = 2 #2 # use cmd history
            vel_mismatch_exp = 2  # lin_z; ang x,y
            low_speed_vxy = 0
            low_speed_wz = 0
            track_vel_hard = 6 # 0.5 # no cmd history
            track_vel_y_hard = 0.0 #1 
            ang_vel = 0
            # base pos
            default_joint_pos = 4
            orientation = 4.
            base_height = 3
            base_height_stand = 4
            base_acc = 1
            # energy
            action_smoothness = -0.01
            torques_max = 0
            torques = -5e-7
            dof_vel = -4e-4
            dof_ankle_vel_added = 0
            dof_vel_swing = 0 #-5e-2
            dof_acc = -1e-8
            collision = -1.

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 0.25
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        clip_observations = 18.
        clip_actions = 18.
    
    # seems less flexiable than the old fashion
    # # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    # action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
    #     noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.01, operation="add"),
    #     bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.005, operation="abs"),
    # )

    # # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    # observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
    #     noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
    #     bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    # )

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensorCfg,  RayCasterCfg, patterns
from isaaclab.sensors import ImuCfg
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

from AzureLoong.assets.AzureLoong import AZURELOONG_CFG

# @configclass
# class EventCfg:
  # startup
    # robot_physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (0.7, 1.3),
    #         "dynamic_friction_range": (1.0, 1.0),
    #         "restitution_range": (1.0, 1.0),
    #         "num_buckets": 250,
    #     },
    # )
    # robot_joint_stiffness_and_damping = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    #         "stiffness_distribution_params": (0.75, 1.5),
    #         "damping_distribution_params": (0.3, 3.0),
    #         "operation": "scale",
    #         "distribution": "log_uniform",
    #     },
    # )
    # add_base_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
    #         "mass_distribution_params": (0.0, 5.0),
    #         "operation": "add",
    #     },
    # )
    # # reset
    # reset_base = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "pose_range": {
    #             "x": (-0.5, 0.5), 
    #             "y": (-0.5, 0.5), 
    #             "roll": (-0.09, 0.09),
    #             "pitch": (-0.09, 0.09),
    #             "yaw": (-3.14, 3.14)},
    #         "velocity_range": {
    #             "x": (-0.2, 0.2),
    #             "y": (-0.2, 0.2),
    #             "z": (-0.5, 0.5),
    #             "roll": (-0.2, 0.2),
    #             "pitch": (-0.2, 0.2),
    #             "yaw": (-0.2, 0.2),
    #         },
    #     },
    # )
    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "position_range": (-0.2, 0.2),
    #         "velocity_range": (-0.2, 0.2),
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )
    # # interval
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(8.0, 10.0),
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "velocity_range": {"x": (-0.3, 0.3), "y": (-0.2, 0.2)},
    #     },
    # )

@configclass
class LoongEnvBaseCfg(DirectRLEnvCfg):
    # env
    # vel_frame_stack = 10  # stored frame num for vel reward evaluation
    # contact_forces_frame_stack = 5  # stored frame num for contact forces reward evaluation
    
    num_single_obs = 46 # defined in compute_observations(), command 5D, q 12D, dq 12D, actions 12D, base_ang_vel 3D, base_euler_xy 2D
    frame_stack = 15  # stored frame num for observations
    observation_space = int(frame_stack * num_single_obs)

    single_num_privileged_obs = 73 # defined in compute_observations()
    c_frame_stack = 3 # stored frame num for privileged (or critic) observations
    state_space = int(c_frame_stack * single_num_privileged_obs)

    action_space = 12 # actuated joint numbers
    num_envs = 4096
    env_spacing = 2.0

    episode_length_s = 40     # episode length in seconds
    decimation = 5
    action_scale = 0.25

    use_ref_actions = False   # speed up training by using reference actions
    enable_stand = True
    focus_stand = False
    enable_imu_filter = False
    enable_imu_offset_disturbance = False
    enable_imu_rand_reset = False
    focus_on_stepping = False
    enable_base_torques = False

    # robot
    robot_cfg: ArticulationCfg = AZURELOONG_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # contact_sensor_Left: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/Link_ankle_l_roll", update_period=0.0, history_length=6, debug_vis=False
    # )
    # contact_sensor_Right: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/Link_ankle_r_roll", update_period=0.0, history_length=6, debug_vis=False
    # )
    # contact_sensor_base: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/base_link", update_period=0.0, history_length=6, debug_vis=False
    # )
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", update_period=0.0, history_length=3, debug_vis=False, track_air_time=True,
        # force_threshold = 5.0
    )


    imu_base: ImuCfg = ImuCfg(prim_path="/World/envs/env_.*/Robot/base_link", debug_vis=False)
    
    class asset:
        foot_name = ["ankle_l_roll", "ankle_r_roll"] # any body name contains this string segment will recognized as foot
        knee_name = "knee"
        terminate_after_contacts_on = ['base_link']
        penalize_contacts_on = ["base_link"]

    # simulation  # use_fabric=True the GUI will not update
    sim: SimulationCfg = SimulationCfg(
        device = "cuda:0", # can be "cpu", "cuda", "cuda:<device_id>"
        dt= 0.002,
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
            gpu_max_rigid_contact_count=2**24
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

    # height scanner
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        update_period=0.01,
        # offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.1, 0.1]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=env_spacing, replicate_physics=True)

    # events
    # events: EventCfg = EventCfg()

    # reset
    class init_state:
        pos = [0.0, 0.0, 1.1]  # height from base to the ground 
        ankle_height = 0.07 # the distance between the ankle and the foot bottom
        l_hip_yaw_id = 6
        l_ankle_roll_id = 10
        r_hip_yaw_id = 12
        r_ankle_roll_id = 16

        # # nominal height base to ankle 1.01
        # default_joint_angles = {  # = target angles [rad] when action = 0.0, is defined in the asset cfg
        #     'J_hip_l_roll': 0.0303,
        #     'J_hip_l_yaw': 0.,
        #     'J_hip_l_pitch': 0.3452,
        #     'J_knee_l_pitch': -0.7817,
        #     'J_ankle_l_pitch': 0.4365,
        #     'J_ankle_l_roll': -0.0303,

        #     'J_hip_r_roll': -0.0303,
        #     'J_hip_r_yaw': 0.,
        #     'J_hip_r_pitch': 0.3452,
        #     'J_knee_r_pitch': -0.7817,
        #     'J_ankle_r_pitch': 0.4365,
        #     'J_ankle_r_roll': 0.0303
        # }

    class commands:
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.  # time before command are changed[s]
        stand_resampling_time = 8. # time before stand resample command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-0.3, 0.3]   # min max [m/s]
            lin_vel_y = [-0.3, 0.3]   # min max [m/s]
            ang_vel_yaw = [-0.3, 0.3] # min max [rad/s]
            heading = [-3.14, 3.14]
    
    class domain_rand:
        static_friction_range = [0.1, 3.0] # dynamic friction is set to the same
        enable_base_com_rand = True
        com_rand_range = 0.1 # x, y, z

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
        tracking_sigma = 20
        max_contact_force = 700  # Forces above this value are penalized
        enable_td_pitch_angle = False
        enable_base_com_rand = False

        class scales:
            # reference motion tracking
            joint_pos = 0 # 1.6
            feet_clearance = 0 # 0.6
            feet_contact_number = 0
            feet_swingZ = 0
            feet_dswingZ = 0
            virtual_leg_sym = 0
            feet_orientation = 0 #2.
            feet_orientation_v2 = 0
            feet_stance_orientation = 0 
            # gait
            feet_air_time = 0 #1.
            foot_slip = 0
            feet_distance = 0
            knee_distance = 0
            # contact
            feet_contact_forces = 0
            # vel tracking
            tracking_lin_vel = 0 #2 # use cmd histroy
            tracking_ang_vel = 0 #2 # use cmd history
            vel_mismatch_exp = 0  # lin_z; ang x,y
            low_speed_vxy = 0
            low_speed_wz = 0
            track_vel_hard = 0 # 0.5 # no cmd history
            track_vel_y_hard = 0.0 #1 
            ang_vel = 0
            # base pos
            default_joint_pos = 0
            orientation = 0
            base_height = 0
            base_height_stand = 0
            base_acc = 0
            # energy
            action_smoothness = 0
            torques_max = 0
            torques = 0
            dof_vel = 0
            dof_ankle_vel_added = 0
            dof_vel_swing = 0 #-5e-2
            dof_acc = 0
            collision = 0
    
    class noise:
        add_noise = True
        action_out_noise = 0.02 # scales to the output action
        noise_level = 0.3    # scales other values
        class noise_scales:
            dof_pos = 0.05
            dof_vel = 0.5
            ang_vel = 0.1
            lin_vel = 0.05
            quat = 0.03
            height_measurements = 0.1
            feet_pos = 0.2

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            pos_xy = 1.
            pos_theta = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
            feet_pos = 0.2
            feet_vel = 0.2
            feet_eul = 0.2
        clip_observations = 18.
        clip_actions = 18.
    
    # action noise
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    # action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
    #     noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
    #     bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    # )
    # # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    # observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
    #     noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
    #     bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    # )

# from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
# from isaaclab.utils import configclass

# @configclass
# class BipedPPORunnerBaseCfg(RslRlOnPolicyRunnerCfg):
#     num_steps_per_env = 60  # The number of steps per environment per update.
#     max_iterations = 3000
#     save_interval = 100
#     experiment_name = "loong_walk"
#     empirical_normalization = False
#     policy = RslRlPpoActorCriticCfg(
#         init_noise_std=1.0,
#         actor_hidden_dims = [512, 256, 128],
#         critic_hidden_dims = [512, 256, 128],
#         activation="elu",
#     )
#     algorithm = RslRlPpoAlgorithmCfg(
#         value_loss_coef=1.0,
#         use_clipped_value_loss=True,
#         clip_param=0.2,
#         entropy_coef=0.005,
#         num_learning_epochs=5,
#         num_mini_batches=4,
#         learning_rate=1.0e-3,
#         schedule="adaptive",
#         gamma=0.99,
#         lam=0.95,
#         desired_kl=0.01,
#         max_grad_norm=1.0,
#     )
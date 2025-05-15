
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
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg, UniformNoiseCfg

from AzureLoong.assets.AzureLoong import AZURELOONG_CFG

from .base_scripts.loong_cfg_base import LoongEnvBaseCfg

@configclass
class EventCfg:
  # startup
    # robot_physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (0.05, 3.0),
    #         "dynamic_friction_range": (0.03, 2.0),
    #         "restitution_range": (1.0, 1.0),
    #         "num_buckets": 250,
    #     },
    # )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
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
    robot_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.1, 2.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

@configclass
class LoongEnvP2PS1Cfg(LoongEnvBaseCfg):
    num_single_obs = 63 # defined in compute_observations(), command 10D, q 12D, dq 12D, actions 12D, base_ang_vel 3D, base_euler_xy 2D
    frame_stack = 15  # stored frame num for observations
    observation_space = int(frame_stack * num_single_obs)

    single_num_privileged_obs = 108 # defined in compute_observations()
    c_frame_stack = 5 # stored frame num for privileged (or critic) observations
    state_space = int(c_frame_stack * single_num_privileged_obs)

    action_space = 12 # actuated joint numbers
    num_envs = 4096

    episode_length_s = 25     # episode length in seconds
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
        #pos = [0.0, 0.0, 1.12]  # height from base to the ground 
        ankle_height = 0.07 # the distance between the ankle and the foot bottom
        feet_l_default_pos_b = [-0.04, 0.0945, -1.04]
        feet_r_default_pos_b = [-0.04, -0.0945, -1.04]
        l_hip_yaw_id = 3
        l_hip_roll_id = 1
        l_knee_id = 7
        l_ankle_pitch_id = 9
        l_ankle_roll_id = 11
        r_hip_yaw_id = 4
        r_hip_roll_id = 2
        r_knee_id = 8
        r_ankle_pitch_id = 10
        r_ankle_roll_id = 12
        

    class commands:
        # Vers: px, py, theta_z, 2 * sequenced
        num_commands = 6
        nominal_gait_cycle_time = 0.9

        class ranges:
            period = [0.45, 4.05]  # position to position period [s]
            lin_vel_x = [-0.4, 0.0, 0.0, 0.6]   # min max min max [m/s]
            lin_vel_y = [-0.4, 0.0, 0.0, 0.4]   # min max min max [m/s]
            ang_vel_yaw = [-0.6, 0.0, 0.0, 0.6] # min max min max [rad/s]
    
    class domain_rand:
        static_friction_range = [0.1, 3.0] # dynamic friction is set to the same
        enable_base_com_rand = True
        com_rand_range = 0.025 # x, y, z

    # reward scales
    class rewards:
        base_height_target = 1.11
        min_dist = 0.3 # minimum distance between two feet
        max_dist = 0.6
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.17    # rad, scale for amplifying ref joint angle
        target_feet_height = 0.1        # m, desired feet height during swing phase
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(error*sigma)
        tracking_sigma = 20 # mainly for velocity tracking
        tracking_sigma_pxy = 10
        tracking_sigma_thetaZ = 10
        max_contact_force = 900  # Forces above this value are penalized

        class scales:
            pos_trace_pXY = 4 #7
            pos_trace_pXY_vel = 6
            pos_trace_thetaZ = 5 #7
            pos_trace_thetaZ_vel = 5 #7
            pos_stop = 5 #5

            feet_contact_number = 3 #2
            feet_swingZ = 6 #10
            feet_dswingZ = 3 #3
            feet_dswingXY = 1.5 #1.5
            virtual_leg_sym = 55 #50
            virtual_leg_sym_continuous = 1.2 # 1.2

            feet_orientation = 2.5 #2.
            feet_slip = -0.1 #-0.1
            
            # contact
            feet_contact_forces = -0.005
            touch_down_velocity = 0

            # base pos
            orientation = 3 #4.
            flat_orientation_l2 = -6 # -5
            vel_mismatch_exp = -2  # lin_z; ang x,y
            base_height = 4.
            base_acc = 0.2 # 1
            default_joint_pos = 4 # 4

            # energy
            action_smoothness = -0.2
            action_smoothness_minimum = -0.02
            action_smoothness_minimum_filter = 4 #4
            torques_max = -1.5e-1
            torques = -5e-7
            dof_vel = -5e-4 
            dof_pow = -0.25 * 1e-5
            fe_pow = 0 #-0.4 * 1e-2
            dof_acc = -3e-8
            collision = -1.

    class noise:
        add_noise = True
        noise_level = 0.5    # scales other values
        action_out_noise = 0.02 # scales to the output action
        class noise_scales:
            dof_pos = 0.03
            dof_vel = 0.2
            ang_vel = 0.15
            lin_vel = 0.02
            quat = 0.015
            height_measurements = 0.1
            feet_pos = 0.01
            feet_vel = 0.01
            feet_eul = 0.01

    class normalization:
        push_robots = False
        push_interval = 1.0
        velocity_range = {
            "x": (-0.2, 0.2),  
            "y": (-0.2, 0.2),  
        }
        class obs_scales:
            lin_vel = 2.
            ang_vel = 0.2
            pos_xy = 2.
            pos_theta = 0.2
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
            feet_pos = 1
            feet_vel = 1
            feet_eul = 1
        clip_observations = 18.
        clip_actions = 18.
    
    
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    # action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
    #     noise_cfg=UniformNoiseCfg(n_min=-0.03, n_max=0.03, operation="add"),
    #     bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    # )

    # seems less flexiable than the old fashion
    # # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    # observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
    #     noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
    #     bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    # )
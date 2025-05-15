import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg

from AzureLoong.assets import ASSET_DIR

HIP_ROLL_ACTUATOR_CFG = DelayedPDActuatorCfg(
    joint_names_expr=[".*hip.*roll"],
    stiffness={".*": 400.0},
    damping={".*": 1.}, # 1.5
    effort_limit=320.0,
    velocity_limit=20.94,
    # armature={".*": 6.9e-5 * 81},
    friction=0.02,
    min_delay=0,
    max_delay=20,
)

HIP_YAW_ACTUATOR_CFG = DelayedPDActuatorCfg(
    joint_names_expr=[".*hip.*yaw"],
    stiffness={".*": 300.0},
    damping={".*": 1.},
    effort_limit=160.0,
    velocity_limit=19.63,
    # armature={".*": 6.9e-5 * 81},
    friction=0.02,
    min_delay=0,
    max_delay=20,
)

HIP_PITCH_ACTUATOR_CFG = DelayedPDActuatorCfg(
    joint_names_expr=[".*hip.*pitch"],
    stiffness={".*": 400.0},
    damping={".*": 1.},
    effort_limit=396.0,
    velocity_limit=19.16,
    # armature={".*": 6.9e-5 * 81},
    friction=0.02,
    min_delay=0,
    max_delay=20,
)

KNEE_PITCH_ACTUATOR_CFG = DelayedPDActuatorCfg(
    joint_names_expr=[".*knee.*pitch"],
    stiffness={".*": 400.0},
    damping={".*": 2.},
    effort_limit=396.0,
    velocity_limit=19.16,
    # armature={".*": 6.9e-5 * 81},
    friction=0.02,
    min_delay=0,
    max_delay=20,
)

ANKLE_PITCH_ACTUATOR_CFG = DelayedPDActuatorCfg(
    joint_names_expr=[".*ankle.*pitch"],
    stiffness={".*": 200.0},
    damping={".*": 0.5},
    effort_limit=58.5,
    velocity_limit=48.8,
    # armature={".*": 6.9e-5 * 81},
    friction=0.02,
    min_delay=0,
    max_delay=20,
)

ANKLE_ROLL_ACTUATOR_CFG = DelayedPDActuatorCfg(
    joint_names_expr=[".*ankle.*roll"],
    stiffness={".*": 200.0},
    damping={".*": 0.5},
    effort_limit=58.5,
    velocity_limit=48.8,
    # armature={".*": 6.9e-5 * 81},
    friction=0.02,
    min_delay=0,
    max_delay=20,
)


AZURELOONG_FLOAT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/Robots/AzureLoong_shortFeet.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            stabilization_threshold=0.001,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.11), # nominal height 1.11, including ankle height of 0.07
        joint_pos={
            "J_hip_l_roll": 0.0,
            "J_hip_l_yaw": 0.0,
            "J_hip_l_pitch": 0.2151,
            "J_knee_l_pitch": -0.5403,
            "J_ankle_l_pitch": 0.3252,
            "J_ankle_l_roll": 0.0,
            "J_hip_r_roll": 0.0,
            "J_hip_r_yaw": 0.0,
            "J_hip_r_pitch": 0.2151,
            "J_knee_r_pitch": -0.5403,
            "J_ankle_r_pitch": 0.3252,
            "J_ankle_r_roll": 0.0,
        },
    ),
    actuators={
        ".*hip.*roll": HIP_ROLL_ACTUATOR_CFG,
        ".*hip.*pitch": HIP_PITCH_ACTUATOR_CFG,
        ".*hip.*yaw": HIP_YAW_ACTUATOR_CFG,
        ".*knee.*pitch": KNEE_PITCH_ACTUATOR_CFG,
        ".*ankle.*pitch": ANKLE_PITCH_ACTUATOR_CFG,
        ".*ankle.*roll": ANKLE_ROLL_ACTUATOR_CFG,
    },
    soft_joint_pos_limit_factor=0.95,
)

AZURELOONG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/Robots/AzureLoong_shortFeet.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.115),
        joint_pos={
            "J_hip_l_roll": 0.0,
            "J_hip_l_yaw": 0.0,
            "J_hip_l_pitch": 0.2151,
            "J_knee_l_pitch": -0.5403,
            "J_ankle_l_pitch": 0.3252,
            "J_ankle_l_roll": 0.0,
            "J_hip_r_roll": 0.0,
            "J_hip_r_yaw": 0.0,
            "J_hip_r_pitch": 0.2151,
            "J_knee_r_pitch": -0.5403,
            "J_ankle_r_pitch": 0.3252,
            "J_ankle_r_roll": 0.0,
        },
    ),
    actuators={
        ".*hip.*roll": HIP_ROLL_ACTUATOR_CFG,
        ".*hip.*pitch": HIP_PITCH_ACTUATOR_CFG,
        ".*hip.*yaw": HIP_YAW_ACTUATOR_CFG,
        ".*knee.*pitch": KNEE_PITCH_ACTUATOR_CFG,
        ".*ankle.*pitch": ANKLE_PITCH_ACTUATOR_CFG,
        ".*ankle.*roll": ANKLE_ROLL_ACTUATOR_CFG,
    },
    soft_joint_pos_limit_factor=0.95,
)
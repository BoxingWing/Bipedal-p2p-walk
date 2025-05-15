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

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab.sensors import ContactSensor, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import HfRandomUniformTerrainCfg,TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# Pre-defined configs
##
# from isaaclab_assets import CARTPOLE_CFG  # isort:skip
from AzureLoong.assets.AzureLoong import AZURELOONG_FLOAT_CFG, AZURELOONG_CFG

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

ROUGH_TERRAINS_CFG_DIY = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=1, # 10
    num_cols=2, # 20
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        # ),
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=1.0, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        # ),
        "random_rough": terrain_gen.HfWaveTerrainCfg(
            proportion=1.0, amplitude_range=(0.05,0.08),num_waves=4, border_width=0.25
        ),
        # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        # ),
        # "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        # ),
    },
)

@configclass
class ContactSensorSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

    # ground plane
    # ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="plane",
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #         restitution=0.0,
    #     ),
    #     debug_vis=False,
    # )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG_DIY,
        max_init_terrain_level=0,
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

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=0.01,
        # offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.1, 0.1]),
        debug_vis=not args_cli.headless,
        mesh_prim_paths=["/World/ground"],
    )

    # Rigid Object
    # contact_sensor_l = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/.*ankle_l_roll",
    #     update_period=0.0,
    #     history_length=6,
    #     track_air_time=True,
    #     debug_vis=False,
    # )
    
    # contact_sensor_r = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/.*ankle_r_roll",
    #     update_period=0.0,
    #     history_length=6,
    #     track_air_time=True,
    #     debug_vis=False,
    # )
    contact_sensors = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=6,
        track_air_time=True,
        debug_vis=False,
    )
    """
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
    """
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
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
        lower = robot.data.default_joint_pos_limits[0, i, 0].item()
        upper = robot.data.default_joint_pos_limits[0, i, 1].item()
        lower2 = robot.data.joint_pos_limits[0,i,0].item()
        upper2 = robot.data.joint_pos_limits[0,i,1].item()
        print(f"{i}: {name}, lim: [{lower:.3f}, {upper:.3f}] lim: [{lower2:.3f}, {upper2:.3f}]")
    print("=====body names=====")
    for i, name in enumerate(robot.data.body_names):
        print(f"{i}: {name}")
    print("=====body names in contact sensors=====")
    for i, name in enumerate(contact_sensors.body_names):
        print(f"{i}: {name}")

     # Set material properties
    # static_friction = torch.FloatTensor(num_envs, num_cubes, 1).uniform_(0.4, 0.8)
    # dynamic_friction = torch.FloatTensor(num_envs, num_cubes, 1).uniform_(0.4, 0.8)
    # restitution = torch.FloatTensor(num_envs, num_cubes, 1).uniform_(0.0, 0.2)

    # materials = torch.cat([static_friction, dynamic_friction, restitution], dim=-1)

    # indices = torch.tensor(range(num_cubes * num_envs), dtype=torch.int)
    # # Add friction to cube
    # object_collection.root_physx_view.set_material_properties(
    #     object_collection.reshape_data_to_view(materials), indices
    # )

    static_friction = torch.FloatTensor(scene.num_envs, 3, 1).uniform_(0.7, 0.7)
    dynamic_friction = torch.FloatTensor(scene.num_envs, 3, 1).uniform_(0.4, 0.4)
    restitution = torch.FloatTensor(scene.num_envs, 3, 1).uniform_(0.0, 0.0)

    materials = torch.cat([static_friction, dynamic_friction, restitution], dim=-1)

    indices = torch.tensor(range(3 * scene.num_envs), dtype=torch.int)

    # Add friction to cube
    scene["robot"].root_physx_view.set_material_properties(
        materials, indices
    )

    materials_to_check = scene["robot"].root_physx_view.get_material_properties()
    print("materials check: ", materials_to_check)
    # Simulate physics
    while simulation_app.is_running():

        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            root_state[:, 0] += 0.1
            root_state[:, 2] += 0.05
            scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions with some noise
            joint_pos, joint_vel = (
                scene["robot"].data.default_joint_pos.clone(),
                scene["robot"].data.default_joint_vel.clone(),
            )
            # joint_pos += torch.rand_like(joint_pos) * 0.1
            scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
        # Apply default actions to the robot
        # -- generate actions/commands
        targets = scene["robot"].data.default_joint_pos
        # -- apply action to the robot
        scene["robot"].set_joint_position_target(targets)
        # -- write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)
        ## print infos
        print("========")
        print(robot.data.body_link_state_w[0, 0, :3])
        print(robot.data.body_link_state_w[0, 11, :3])
        print(robot.data.body_link_state_w[0, 12, :3])
        # print(scene["height_scanner"])

        # print("Received mean height value: ", torch.mean(scene["height_scanner"].data.ray_hits_w[..., -1]).item())
        print("scanner height:", torch.mean(scene["height_scanner"].data.pos_w[:, 2].unsqueeze(1) - scene["height_scanner"].data.ray_hits_w[..., 2]))

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

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
    # design scene
    scene_cfg = ContactSensorSceneCfg(num_envs=2, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
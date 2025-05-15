# Bipedal point-to-point walk training code

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

## Overview
This repository shows the training code for paper "Learning  Point-to-Point Bipedal Walking Without Global Navigation" with Isaaclab.



## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). **Please use Isaaclab v2.1.0 with IsaacSim 4.5.0**

- Using a python interpreter that has IsaacLab installed, install the library


```bash
cd Bipedal-p2p-walk
python -m pip install -e source/AzureLoong
```



## Run

Training and play an agent with RSL-RL on a bipedal robot AzureLoong:

```
python scripts/rsl_rl/train.py --task=loong_walk_p2p_s1 --headless
python scripts/rsl_rl/play.py --task=loong_walk_p2p_s1 --num_envs 5
```



## Project Structure

The project is mainly organized as follows:

```
Bipedal-p2p-walk/
├── cmd.txt
├── scripts/
│   └── rsl_rl/
│       ├── cli_args.py
│       ├── export.py
│       ├── play.py
│       └── train.py
└── source/
    └── AzureLoong/
        ├── AzureLoong/
            ├── assets/
            │   ├── AzureLoong.py
            │   ├── __init__.py
            │   └── Robots/
            │       ├── AzureLoong_shortFeet.usd
            │       └── configuration/
            ├── tasks/
                └── flat_walk/
                    ├── agents/
                    │   └── rsl_rl_ppo_cfg.py
                    ├── base_scripts/
                    │   ├── loong_cfg_base.py
                    │   └── loong_env_base.py
                    ├── loong_cfg_p2p_s1.py
                    ├── loong_cfg_p2p_s2.py
                    └── loong_env_p2p_s1.py
```

**The robot asset file** is store as AzureLoong_shortFeet.usd.

**Joint configurations** such as stiffness and damping are configured in AzureLoong.py.

**Reward functions, environment settings and corresponding scales** are defined in loong_env_p2p_s1.py and loong_cfg_p2p_s1.py . loong_cfg_p2p_s2.py is the second stage training config with more domain randomization. 

**PPO parameters** are configured in rsl_rl_ppo_cfg.py.



## References:

Humanoid-Gym: Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer. https://github.com/roboterax/humanoid-gym

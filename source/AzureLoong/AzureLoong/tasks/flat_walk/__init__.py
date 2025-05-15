import gymnasium as gym

from . import agents
from .loong_cfg_s2 import LoongEnvS2Cfg
from .loong_cfg_s1 import LoongEnvS1Cfg
from .loong_cfg_p2p_s1 import LoongEnvP2PS1Cfg

gym.register(
    id="loong_walk_s1",
    entry_point=f"{__name__}.loong_env_s1:LoongEnvS1",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loong_cfg_s1:LoongEnvS1Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BipedPPORunnerCfg",
    },
)

gym.register(
    id="loong_walk_s2",
    entry_point=f"{__name__}.loong_env_s1:LoongEnvS1",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loong_cfg_s2:LoongEnvS2Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BipedPPORunnerCfgS2",
    },
)

gym.register(
    id="loong_walk_p2p_s1",
    entry_point=f"{__name__}.loong_env_p2p_s1:LoongEnvP2PS1",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loong_cfg_p2p_s1:LoongEnvP2PS1Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BipedPPORunnerCfg",
    },
)

gym.register(
    id="loong_walk_p2p_s2",
    entry_point=f"{__name__}.loong_env_p2p_s1:LoongEnvP2PS1",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loong_cfg_p2p_s2:LoongEnvP2PS2Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BipedPPORunnerCfgS2",
    },
)
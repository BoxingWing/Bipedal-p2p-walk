import gymnasium as gym
from . import agents

gym.register(
    id="walk_p2p_s1",
    entry_point=f"{__name__}.env_p2p_s1:EnvP2PS1",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg_p2p_s1:EnvP2PS1Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BipedPPORunnerCfg",
    },
)

gym.register(
    id="walk_p2p_s2",
    entry_point=f"{__name__}.env_p2p_s1:EnvP2PS1",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg_p2p_s2:EnvP2PS2Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BipedPPORunnerCfgS2",
    },
)
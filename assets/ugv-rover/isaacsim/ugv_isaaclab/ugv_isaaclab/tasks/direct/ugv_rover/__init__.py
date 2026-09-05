"""Goal-navigation task for the UGV rover in a rectangular arena."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-UGV-Rover-Empty-v0",
    entry_point=f"{__name__}.ugv_rover_env:UGVRoverEmptyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ugv_rover_env_cfg:UGVRoverEmptyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UGVRoverEmptyPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Isaac-UGV-Rover-City-v0",
    entry_point=f"{__name__}.ugv_rover_env:UGVRoverEmptyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ugv_rover_env_cfg:UGVRoverCityEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UGVRoverCityPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

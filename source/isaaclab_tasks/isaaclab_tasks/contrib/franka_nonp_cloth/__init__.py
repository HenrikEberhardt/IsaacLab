# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka non-prehensile cloth task: move a rigid cube by pulling the cloth it rests on.

a flat cloth sheet lies on the table with a PLA cube on top, and the cube is simulated by
the same VBD solver as the cloth not connected over the SolverCoupling.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="IsaacContrib-Franka-nonp-cloth",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_nonp_cloth_env_cfg:FrankaNonpClothEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaNonpClothPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="IsaacContrib-Franka-nonp-cloth-WristCam",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_nonp_cloth_env_cfg:FrankaNonpClothWristCameraEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaNonpClothPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="IsaacContrib-Franka-nonp-cloth-HeadCam",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_nonp_cloth_env_cfg:FrankaNonpClothHeadCameraEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaNonpClothPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka sock task: lift a sleeve-shaped surface deformable off the table.

The task derives its MDP from ``Isaac-Lift-Cloth-Franka``. What differs is the asset, which is a
capped tube rather than a flat sheet, and the scene, which drops the cloth's support rails and adds
visual dividers between environments. The ``-WristCam`` and ``-HeadCam`` variants add a wrist or head
camera to the scene and keep the state task's observations, agent, and checkpoints.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="IsaacContrib-Lift-Sock-Franka",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_sock_env_cfg:FrankaSockEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaSockPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="IsaacContrib-Lift-Sock-Franka-WristCam",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_sock_env_cfg:FrankaSockWristCameraEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaSockPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="IsaacContrib-Lift-Sock-Franka-HeadCam",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_sock_env_cfg:FrankaSockHeadCameraEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaSockPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

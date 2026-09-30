# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.core.lift.config.franka_soft.agents.rsl_rl_ppo_cfg import ALGO_CFG, FrankaDeformablePPORunnerCfg


@configclass
class FrankaNonpClothPPORunnerCfg(FrankaDeformablePPORunnerCfg):
    experiment_name = "nonprehensile_cloth"
    # long horizontal task, change the PPO defaults, 0.98 => 0.99, 1e-4 => 1e-3
    algorithm = ALGO_CFG.replace(learning_rate=1.0e-3, gamma=0.99)

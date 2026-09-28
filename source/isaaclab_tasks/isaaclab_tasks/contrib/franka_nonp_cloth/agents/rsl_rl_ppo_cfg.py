# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.core.lift.config.franka_soft.agents.rsl_rl_ppo_cfg import FrankaDeformablePPORunnerCfg


@configclass
class FrankaNonpClothPPORunnerCfg(FrankaDeformablePPORunnerCfg):
    experiment_name = "franka_nonp_cloth"

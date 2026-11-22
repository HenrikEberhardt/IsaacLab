# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visual distillation of the Franka cloth-drag teacher into a joint-delta student."""

import gymnasium as gym

from . import agents

# joint-delta actions without the camera: checks of the teacher-label conversion (no agent)
gym.register(
    id="IsaacContrib-Drag-Cloth-Franka-Joint",
    entry_point="isaaclab_tasks.contrib.franka_cloth_drag.franka_cloth_drag_env:FrankaClothDragEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.franka_cloth_drag_visual_env_cfg:FrankaClothDragJointEnvCfg"},
)

# the RGB student: wrist camera, visual randomization and the DAgger runner
gym.register(
    id="IsaacContrib-Drag-Cloth-Franka-Visual",
    entry_point="isaaclab_tasks.contrib.franka_cloth_drag.franka_cloth_drag_env:FrankaClothDragEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_cloth_drag_visual_env_cfg:FrankaClothDragVisualEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:FrankaClothDragDistillationRunnerCfg",
    },
)

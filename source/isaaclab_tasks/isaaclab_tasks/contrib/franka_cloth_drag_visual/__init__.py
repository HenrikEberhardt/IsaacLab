# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visual distillation of the Franka cloth-drag teacher into a joint-delta wrist-camera student."""

import gymnasium as gym

ENTRY_POINT = "isaaclab_tasks.contrib.franka_cloth_drag.franka_cloth_drag_env:FrankaClothDragEnv"

gym.register(  # no camera: label conversion checks
    id="IsaacContrib-Drag-Cloth-Franka-Joint",
    entry_point=ENTRY_POINT,
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.franka_cloth_drag_visual_env_cfg:FrankaClothDragJointEnvCfg"},
)

gym.register(  # the student: wrist camera, visual randomization, DAgger runner
    id="IsaacContrib-Drag-Cloth-Franka-Visual",
    entry_point=ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_cloth_drag_visual_env_cfg:FrankaClothDragVisualEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{__name__}.student.agent_cfg:FrankaClothDragDistillationRunnerCfg",
    },
)

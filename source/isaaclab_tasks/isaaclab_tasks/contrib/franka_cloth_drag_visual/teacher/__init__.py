# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Everything pinned to the teacher checkpoint: its arm action, actor network and curriculum state."""

from .ik_action import TeacherTopDownIKAction, TeacherTopDownIKActionCfg, top_down_quat
from .teacher_cfg import (
    TEACHER_ACTOR_CFG,
    TEACHER_ARM_CFG,
    TEACHER_CHECKPOINT,
    TEACHER_CURRICULUM_PROGRESS,
    TEACHER_NUM_ACTIONS,
    TEACHER_SCALE_ROT,
    TEACHER_SCALE_Z,
    apply_final_curriculum_values,
)

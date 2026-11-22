# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The IK teacher's arm action and the conversion of its actions into joint-delta labels."""

from .ik_action import TeacherTopDownIKAction, TeacherTopDownIKActionCfg
from .ik_label import IK_STATE_DIM, IKLabelCfg, teacher_actions_to_joint_deltas, teacher_joint_step

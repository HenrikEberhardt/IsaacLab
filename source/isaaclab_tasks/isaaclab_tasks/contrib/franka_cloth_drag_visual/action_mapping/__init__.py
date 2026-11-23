# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mapping of teacher actions into the student's action space (the algorithm is in .distillation)."""

from .base import ActionMapping, ActionMappingCfg
from .identity import IdentityMapping, IdentityMappingCfg
from .ik_to_joint_delta import (
    IK_STATE_DIM,
    IKToJointDeltaMapping,
    IKToJointDeltaMappingCfg,
    ik_to_joint_delta_mapping_cfg,
    teacher_actions_to_joint_deltas,
    teacher_joint_step,
)

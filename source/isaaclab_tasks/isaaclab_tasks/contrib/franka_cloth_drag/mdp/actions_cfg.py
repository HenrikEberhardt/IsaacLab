# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations for the action terms of the Franka cloth dragging task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .actions import TopDownDeltaEEAction


@configclass
class TopDownDeltaEEActionCfg(DifferentialInverseKinematicsActionCfg):
    """Configuration for a 4-DOF delta end-effector action.

    The action is limited to ``(dx, dy, dz, dyaw)`` in the robot base frame.
    
    The IK controller must take relative poses (``command_type="pose"`` and ``use_relative_mode=True``).
    """

    class_type: type[TopDownDeltaEEAction] | str = "{DIR}.actions:TopDownDeltaEEAction"

    scale_trans: float = 0.01
    """
    Scale of the ``(dx, dy, dz)`` actions per policy step [m].
    """

    scale_rot: float = 0.05
    """
    Scale of the ``dyaw`` action per policy step [rad].
    """

    yaw_limits: tuple[float, float] = (-0.5 * math.pi, 0.5 * math.pi)
    """
    Limit yaw range to [-pi/2, pi/2] to avoid wrist joint limits. Symmetry of gripper allows to limit to +- quarter rotation.
    """

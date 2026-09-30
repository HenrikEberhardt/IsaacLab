# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of the arm and gripper actions for the Franka non-prehensile cloth task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import math

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from .cloth_grasp import GRASP_PROXIMITY

if TYPE_CHECKING:
    from .actions import ClothGraspAction, TopDownDifferentialIKAction


@configclass
class ClothGraspActionCfg(BinaryJointPositionActionCfg):
    """Configuration of a binary gripper action that grasps cloth, see :class:`ClothGraspAction`."""

    class_type: type[ClothGraspAction] | str = "{DIR}.actions:ClothGraspAction"

    ee_frame_name: str = "ee_frame"
    """Name of the frame transformer whose first target is the fingertip point between the fingers."""

    attach: bool = True
    """Whether to lock the grasped cloth vertices to the gripper. Defaults to True.

    If False, the gripper holds the cloth with friction alone.
    """

    grasp_opening: float = 0.015
    """Opening of each finger at or below which commanded-closed fingers grasp the cloth [m].

    Closing from fully open (0.04 m), the fingers reach it on the third step at the task's 30 Hz, so one or two noisy
    close actions do not grasp.
    """

    release_opening: float = 0.032
    """Opening of each finger at or above which the gripper releases the cloth and may grasp again [m].

    Opening from closed, the fingers reach it on the third step at the task's 30 Hz, so one or two noisy open actions
    do not drop the cloth.
    """

    grasp_proximity: float = GRASP_PROXIMITY
    """Distance from a finger anchor within which a cloth vertex is grasped [m]."""

    contact_clearance: float = 0.05
    """Distance from the fingertip point to the nearest cloth vertex at which a gripper that released the cloth
    collides with it again [m]."""



@configclass
class TopDownDifferentialIKActionCfg(DifferentialInverseKinematicsActionCfg):
    """Configuration of a differential IK action for a hand that points straight down, see
    :class:`TopDownDifferentialIKAction`.

    The controller must take relative poses (``command_type="pose"``, ``use_relative_mode=True``), and
    :attr:`scale` has four entries, for ``(dx, dy, dz, dyaw)``.
    """

    class_type: type[TopDownDifferentialIKAction] | str = "{DIR}.actions:TopDownDifferentialIKAction"

    yaw_limits: tuple[float, float] = (-0.5 * math.pi, 0.5 * math.pi)
    """Range of the hand's yaw about the robot base z-axis [rad]. Defaults to a quarter turn either way.

    A half-turn range covers every grasp direction of a two-finger gripper, and limiting it keeps the wrist away from
    its joint limits.
    """

    min_height: float | None = None
    """Lowest height of the end-effector frame above the robot base that the action commands [m].

    Defaults to None, in which case the height is not limited.
    """

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations for the action terms of the Franka cloth dragging task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
    RelativeJointPositionActionCfg,
)
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .actions import (
        LimitedRelativeJointPositionAction,
        RateLimitedBinaryJointPositionAction,
        TopDownDeltaEEAction,
    )


@configclass
class LimitedRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    """Configuration for relative joint position targets with range and PD position-error bounds."""

    class_type: type[LimitedRelativeJointPositionAction] | str = "{DIR}.actions:LimitedRelativeJointPositionAction"

    joint_range_margin: float | None = None
    """Distance kept from each hard joint limit [rad]. Defaults to None (no range bound)."""

    joint_vel_factor: float | None = None
    """Velocity-limit fraction used to cap PD position error. Defaults to None (no cap)."""


@configclass
class TopDownDeltaEEActionCfg(DifferentialInverseKinematicsActionCfg):
    """Configuration for a 4-DOF delta end-effector action ``(dx, dy, dz, dyaw)`` in the robot base frame."""

    class_type: type[TopDownDeltaEEAction] | str = "{DIR}.actions:TopDownDeltaEEAction"

    body_name: str = ""
    """Body resolved from ``ee_frame_name`` when the action is initialized."""

    ee_frame_name: str = "ee_frame"
    """Scene frame transformer that defines the end-effector frame."""

    scale_trans: float = 0.02
    """Scale of the ``(dx, dy, dz)`` actions per policy step [m]."""

    scale_rot: float = 0.05
    """Scale of the ``dyaw`` action per policy step [rad]."""

    wrist_joint_limits: tuple[float, float] = (-2.0 * math.pi / 3.0, 2.0 * math.pi / 3.0)
    """Range of the last arm joint, which turns the hand, that ``dyaw`` may command [rad]. Half a turn would cover every
    orientation of the symmetric gripper; ±120° adds 60° of overlap, so a jaw orientation near the end of a half turn
    can be reached from either side instead of only by the long way round. Inside the hardware range (±166°)."""

    limit_operation_space: bool = False
    """Whether to clamp commanded EE positions to the workspace planes below."""

    x_min: float = 0.05
    """Plane towards the robot base: smallest end-effector x-value [m]."""

    y_limits: tuple[float, float] = (-0.65, 0.65)
    """Planes at the table sides: end-effector y-range [m]."""

    z_min: float = 0.0085
    """Floor: lowest end-effector height. The fingertips are 8.65 mm below the
    end-effector frame."""

    joint_vel_factor: float | None = None
    """Velocity-limit fraction used to cap PD position error by uniformly scaling the IK step."""

    joint_range_margin: float | None = None
    """Margin inside hard joint limits [rad]. The IK step scales uniformly to keep targets inside this range."""


@configclass
class RateLimitedBinaryJointPositionActionCfg(BinaryJointPositionActionCfg):
    """Configuration for a binary joint position action with a speed-limited target."""

    class_type: type[RateLimitedBinaryJointPositionAction] | str = "{DIR}.actions:RateLimitedBinaryJointPositionAction"

    joint_vel_factor: float | None = None
    """Fraction of the hardware joint velocity limit the target moves at."""

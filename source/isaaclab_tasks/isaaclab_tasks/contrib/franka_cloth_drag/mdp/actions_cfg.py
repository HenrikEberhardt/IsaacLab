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
    from .actions import TopDownDeltaEEAction, TopDownStabilizedDeltaEEAction


@configclass
class TopDownDeltaEEActionCfg(DifferentialInverseKinematicsActionCfg):
    """Configuration for a 4-DOF delta end-effector action ``(dx, dy, dz, dyaw)`` in the robot base frame."""

    class_type: type[TopDownDeltaEEAction] | str = "{DIR}.actions:TopDownDeltaEEAction"

    body_name: str = ""
    """Body resolved from ``ee_frame_name`` when the action is initialized."""

    ee_frame_name: str = "ee_frame"
    """Scene frame transformer that defines the end-effector frame."""

    scale_trans: tuple[float, float, float] = (0.02, 0.02, 0.02)
    """Scale of the ``(dx, dy, dz)`` actions per policy step, one per axis of the robot base frame [m]."""

    scale_rot: float = 0.05
    """Scale of the ``dyaw`` action per policy step [rad]."""

    wrist_joint_limits: tuple[float, float] = (-2.0 * math.pi / 3.0, 2.0 * math.pi / 3.0)
    """Range of the wrist joint, in [rad]. ±120° to include overlap. Inside the hardware range (±166°)."""

    joint_vel_factor: float | None = None
    """Velocity-limit fraction used to cap PD position error by uniformly scaling the IK step."""

    joint_range_margin: float | None = None
    """Margin inside hard joint limits [rad]. The IK step scales uniformly to keep targets inside this range."""

    command_delay_range: tuple[int, int] = (0, 0)
    """Range of the delay of the IK target in physics steps, drawn uniformly per environment at reset.

    It models the latency between the policy and the real robot's controller. The policy's commanded target pose
    (start pose plus delta) reaches the IK only after the drawn number of physics steps. (0, 0) disables the delay.
    """


@configclass
class TopDownStabilizedDeltaEEActionCfg(TopDownDeltaEEActionCfg):
    """Top-down delta-IK action with bounded operational-space roll/pitch stabilization."""

    class_type: type[TopDownStabilizedDeltaEEAction] | str = "{DIR}.actions:TopDownStabilizedDeltaEEAction"

    tilt_stiffness: float = 150.0
    """Operational-space roll/pitch stiffness [N·m/rad]."""

    tilt_damping: float = 25.0
    """Operational-space roll/pitch damping [N·m·s/rad]."""

    tilt_effort_limit_factor: float = 0.25
    """Largest stabilizing torque per joint as a fraction of its effort limit."""

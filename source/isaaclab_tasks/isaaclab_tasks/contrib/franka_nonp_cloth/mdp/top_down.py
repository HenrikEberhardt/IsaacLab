# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Yaw of a gripper that points straight down."""

from __future__ import annotations

import math

import torch

import isaaclab.utils.math as math_utils

# (x, y, z, w) half turn about x, which points the Franka hand straight down with its fingers closing along -y
TOP_DOWN_QUAT = (1.0, 0.0, 0.0, 0.0)


def top_down_yaw(quat: torch.Tensor) -> torch.Tensor:
    """Return the yaw of a hand orientation: the heading of its x-axis about the z-axis [rad].

    For an orientation ``Rz(yaw) * TOP_DOWN_QUAT`` this recovers ``yaw``. It stays well defined for a hand that is
    tilted away from pointing straight down.

    Args:
        quat: Hand orientations (x, y, z, w), shape [N, 4].

    Returns:
        The yaws in (-pi, pi], shape [N].
    """
    x_axis = math_utils.quat_apply(quat, torch.tensor((1.0, 0.0, 0.0), device=quat.device).expand(len(quat), 3))
    return torch.atan2(x_axis[:, 1], x_axis[:, 0])


def top_down_quat(yaw: torch.Tensor) -> torch.Tensor:
    """Return the orientations (x, y, z, w) that point the hand straight down at the given yaws [rad], shape [N, 4]."""
    zeros = torch.zeros_like(yaw)
    top_down = torch.tensor(TOP_DOWN_QUAT, device=yaw.device).expand(len(yaw), 4)
    return math_utils.quat_mul(math_utils.quat_from_euler_xyz(zeros, zeros, yaw), top_down)


def gripper_yaw_error(yaw: torch.Tensor, target_yaw: torch.Tensor) -> torch.Tensor:
    """Return the yaw of a two-finger gripper relative to a target yaw [rad], in [-pi/2, pi/2).

    The gripper looks the same after a half turn, so yaws that differ by pi close the fingers along the same line.
    """
    return torch.remainder(yaw - target_yaw + 0.5 * math.pi, math.pi) - 0.5 * math.pi

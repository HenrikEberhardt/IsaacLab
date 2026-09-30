# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU tests for the top-down gripper yaw of the Franka non-prehensile cloth task."""

import math

import torch

import isaaclab.utils.math as math_utils

from isaaclab_tasks.contrib.franka_nonp_cloth.mdp.top_down import gripper_yaw_error, top_down_quat, top_down_yaw


def test_top_down_quat_points_down_and_round_trips_yaw():
    """The top-down orientation points the hand's z-axis down, and its yaw is recovered from it."""
    yaw = torch.linspace(-3.0, 3.0, 13)
    quat = top_down_quat(yaw)

    z_axis = math_utils.quat_apply(quat, torch.tensor((0.0, 0.0, 1.0)).expand(len(yaw), 3))
    torch.testing.assert_close(z_axis, torch.tensor((0.0, 0.0, -1.0)).expand(len(yaw), 3), rtol=0.0, atol=1e-6)
    torch.testing.assert_close(top_down_yaw(quat), yaw, rtol=0.0, atol=1e-6)


def test_gripper_yaw_error_is_half_turn_symmetric():
    """Yaws a half turn apart close the fingers along the same line, so they have the same error."""
    yaw = torch.linspace(-3.0, 3.0, 25)
    target = torch.full_like(yaw, -0.25 * math.pi)
    error = gripper_yaw_error(yaw, target)

    assert (error >= -0.5 * math.pi).all() and (error < 0.5 * math.pi).all()
    torch.testing.assert_close(gripper_yaw_error(yaw + math.pi, target), error, rtol=0.0, atol=1e-5)
    # the error is the plain difference while it is less than a quarter turn
    near = (yaw - target).abs() < 0.49 * math.pi
    torch.testing.assert_close(error[near], (yaw - target)[near], rtol=0.0, atol=1e-6)

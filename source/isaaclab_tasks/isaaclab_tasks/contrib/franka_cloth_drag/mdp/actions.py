# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms for the Franka cloth dragging task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .actions_cfg import TopDownDeltaEEActionCfg

# (x, y, z, w) half turn about x, which points the Franka hand straight down
TOP_DOWN_QUAT = (1.0, 0.0, 0.0, 0.0)


def _top_down_yaw(quat: torch.Tensor) -> torch.Tensor:
    """Return the heading of the hand's x-axis about the z-axis [rad], shape [N].

    For an orientation ``Rz(yaw) * TOP_DOWN_QUAT`` this recovers ``yaw``, and it stays defined for a tilted hand.

    Args:
        quat: Hand orientations (x, y, z, w), shape [N, 4].
    """
    x_axis = math_utils.quat_apply(quat, torch.tensor((1.0, 0.0, 0.0), device=quat.device).expand(len(quat), 3))
    return torch.atan2(x_axis[:, 1], x_axis[:, 0])


def _top_down_quat(yaw: torch.Tensor) -> torch.Tensor:
    """Return the orientations (x, y, z, w) that point the hand straight down at the given yaws [rad], shape [N, 4]."""
    zeros = torch.zeros_like(yaw)
    top_down = torch.tensor(TOP_DOWN_QUAT, device=yaw.device).expand(len(yaw), 4)
    return math_utils.quat_mul(math_utils.quat_from_euler_xyz(zeros, zeros, yaw), top_down)


class TopDownDeltaEEAction(DifferentialInverseKinematicsAction):
    """4-DOF delta end-effector action: the hand moves in 3D and turns only about the vertical axis.

    The action ``(dx, dy, dz, dyaw)`` is scaled by :attr:`TopDownDeltaEEActionCfg.scale_trans` and
    :attr:`TopDownDeltaEEActionCfg.scale_rot`. Each step the term commands
    the relative IK controller to move the end-effector by ``(dx, dy, dz)`` in the robot base frame and to rotate it
    to point straight down at the current yaw plus ``dyaw``, clamped to :attr:`TopDownDeltaEEActionCfg.yaw_limits`.
    """

    cfg: TopDownDeltaEEActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: TopDownDeltaEEActionCfg, env: ManagerBasedEnv):
        if cfg.controller.command_type != "pose" or not cfg.controller.use_relative_mode:
            raise ValueError("TopDownDeltaEEAction needs an IK controller with relative pose commands.")
        super().__init__(cfg, env)
        # per-axis scale of (dx, dy, dz, dyaw)
        self._scale[:] = torch.tensor((cfg.scale_trans,) * 3 + (cfg.scale_rot,), device=self.device)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 4

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = self.raw_actions * self._scale
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        yaw = (_top_down_yaw(ee_quat_curr) + self._processed_actions[:, 3]).clamp(*self.cfg.yaw_limits)
        # rotation from the current orientation to pointing straight down at the new yaw
        _, rotation = math_utils.compute_pose_error(
            ee_pos_curr, ee_quat_curr, ee_pos_curr, _top_down_quat(yaw), rot_error_type="axis_angle"
        )
        pose_delta = torch.cat([self._processed_actions[:, :3], rotation], dim=-1)
        self._ik_controller.set_command(pose_delta, ee_pos_curr, ee_quat_curr)

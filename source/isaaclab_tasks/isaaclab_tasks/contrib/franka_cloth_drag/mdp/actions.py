# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms for the Franka cloth dragging task."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction

from .frames import top_down_yaw

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv

    from .actions_cfg import (
        LimitedRelativeJointPositionActionCfg,
        RateLimitedBinaryJointPositionActionCfg,
        TopDownDeltaEEActionCfg,
        TopDownStabilizedDeltaEEActionCfg,
    )

TOP_DOWN_QUAT = (1.0, 0.0, 0.0, 0.0)


def _top_down_quat(yaw: torch.Tensor) -> torch.Tensor:
    """Orientations (x, y, z, w) that point the hand straight down at the given yaws [rad], shape [N, 4]."""
    zeros = torch.zeros_like(yaw)
    top_down = torch.tensor(TOP_DOWN_QUAT, device=yaw.device).expand(len(yaw), 4)
    return math_utils.quat_mul(math_utils.quat_from_euler_xyz(zeros, zeros, yaw), top_down)


def _reduced_joint_pos_limits(asset: Articulation, joint_ids, margin: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Hard joint limits moved inwards by ``margin`` [rad] on each side, each of shape [num_envs, num_joints]."""
    limits = asset.data.joint_pos_limits.torch[:, joint_ids]
    lower, upper = limits[..., 0] + margin, limits[..., 1] - margin
    default_pos = asset.data.default_joint_pos.torch[:, joint_ids]
    if torch.any((default_pos < lower) | (default_pos > upper)):
        raise ValueError(f"The default joint positions lie outside the joint ranges reduced by {margin} rad.")
    return lower.clone(), upper.clone()


def _max_joint_pos_offset(asset: Articulation, joint_ids, vel_factor: float) -> torch.Tensor:
    """Approximate a speed bound through the PD position-error cap [rad], without limiting measured velocity."""
    stiffness = asset.data.joint_stiffness.torch[:, joint_ids]
    damping = asset.data.joint_damping.torch[:, joint_ids]
    return (vel_factor * asset.data.joint_vel_limits.torch[:, joint_ids] * damping / stiffness).clone()


def _commanded_joint_vel_limits(asset: Articulation, joint_ids, vel_factor: float | None) -> torch.Tensor:
    """Velocity reference for observations [rad/s]: scaled joint limits, or the limits when no factor is set."""
    factor = 1.0 if vel_factor is None else vel_factor
    return (factor * asset.data.joint_vel_limits.torch[:, joint_ids]).clone()


class LimitedRelativeJointPositionAction(RelativeJointPositionAction):
    """Relative joint position action with target-range and PD position-error bounds."""

    cfg: LimitedRelativeJointPositionActionCfg

    def __init__(self, cfg: LimitedRelativeJointPositionActionCfg, env: ManagerBasedEnv):
        """Initialize joint target limits and the PD position-error cap."""
        super().__init__(cfg, env)
        self._pos_lower = self._pos_upper = self._max_offset = None
        self._joint_pos_limits = _reduced_joint_pos_limits(self._asset, self._joint_ids, cfg.joint_range_margin or 0.0)
        if cfg.joint_range_margin is not None:
            self._pos_lower, self._pos_upper = self._joint_pos_limits
        if cfg.joint_vel_factor is not None:
            self._max_offset = _max_joint_pos_offset(self._asset, self._joint_ids, cfg.joint_vel_factor)
        self._joint_vel_limits = _commanded_joint_vel_limits(self._asset, self._joint_ids, cfg.joint_vel_factor)

    @property
    def joint_pos_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Joint position reference bounds [rad], each of shape [num_envs, num_joints]."""
        return self._joint_pos_limits

    @property
    def joint_vel_limits(self) -> torch.Tensor:
        """Joint velocity reference for observation normalization [rad/s], shape [num_envs, num_joints]."""
        return self._joint_vel_limits

    def apply_actions(self):
        """Apply joint position deltas with configured target-range and PD position-error bounds."""
        offset = self.processed_actions
        if self._max_offset is not None:
            offset = torch.clamp(offset, -self._max_offset, self._max_offset)
        target = self._asset.data.joint_pos.torch[:, self._joint_ids] + offset
        if self._pos_lower is not None:
            target = torch.clamp(target, self._pos_lower, self._pos_upper)
        self._asset.set_joint_position_target_index(target=target, joint_ids=self._joint_ids)


class TopDownDeltaEEAction(DifferentialInverseKinematicsAction):
    """4-DOF delta end-effector action: the hand moves in 3D, points straight down and turns about the vertical axis."""

    cfg: TopDownDeltaEEActionCfg

    def __init__(self, cfg: TopDownDeltaEEActionCfg, env: ManagerBasedEnv):
        """Resolve the EE frame and initialize workspace, yaw and joint target bounds."""
        if cfg.controller.command_type != "pose" or not cfg.controller.use_relative_mode:
            raise ValueError("TopDownDeltaEEAction needs an IK controller with relative pose commands.")
        ee_frame = env.scene[cfg.ee_frame_name].cfg.target_frames[0]
        cfg = cfg.replace(
            body_name=ee_frame.prim_path.rsplit("/", 1)[-1],
            body_offset=cfg.OffsetCfg(pos=ee_frame.offset.pos, rot=ee_frame.offset.rot),
        )
        super().__init__(cfg, env)
        self._scale[:] = torch.tensor((cfg.scale_trans,) * 3 + (cfg.scale_rot,), device=self.device)
        self._space_lower = torch.tensor((cfg.x_min, cfg.y_limits[0], cfg.z_min), device=self.device)
        self._space_upper = torch.tensor((math.inf, cfg.y_limits[1], math.inf), device=self.device)
        self._wrist_id = self._asset.find_joints(self._joint_names[-1])[0][0]
        self._pos_lower = self._pos_upper = self._max_offset = None
        lower, upper = _reduced_joint_pos_limits(self._asset, self._joint_ids, cfg.joint_range_margin or 0.0)
        if cfg.joint_range_margin is not None:
            self._pos_lower, self._pos_upper = lower, upper
        if cfg.joint_vel_factor is not None:
            self._max_offset = _max_joint_pos_offset(self._asset, self._joint_ids, cfg.joint_vel_factor)
        lower, upper = lower.clone(), upper.clone()
        lower[:, -1] = lower[:, -1].clamp(min=cfg.wrist_joint_limits[0])
        upper[:, -1] = upper[:, -1].clamp(max=cfg.wrist_joint_limits[1])
        self._joint_pos_limits = (lower, upper)
        self._joint_vel_limits = _commanded_joint_vel_limits(self._asset, self._joint_ids, cfg.joint_vel_factor)

    @property
    def action_dim(self) -> int:
        """Return the four action dimensions: translation and yaw."""
        return 4

    @property
    def joint_pos_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Joint reference bounds [rad], including the wrist yaw window, shape [num_envs, num_joints]."""
        return self._joint_pos_limits

    @property
    def joint_vel_limits(self) -> torch.Tensor:
        """Joint velocity reference for observation normalization [rad/s], shape [num_envs, num_joints]."""
        return self._joint_vel_limits

    def process_actions(self, actions: torch.Tensor):
        """Convert translation and yaw deltas into a top-down IK command."""
        self._raw_actions[:] = actions
        self._processed_actions[:] = self.raw_actions * self._scale
        ee_pos, ee_quat = self._compute_frame_pose()
        if self.cfg.limit_operation_space:
            target = torch.clamp(ee_pos + self._processed_actions[:, :3], self._space_lower, self._space_upper)
            self._processed_actions[:, :3] = target - ee_pos
        wrist_pos = self._asset.data.joint_pos.torch[:, self._wrist_id]
        lower, upper = self.cfg.wrist_joint_limits
        self._processed_actions[:, 3] = torch.clamp(self._processed_actions[:, 3], wrist_pos - upper, wrist_pos - lower)
        yaw = top_down_yaw(ee_quat) + self._processed_actions[:, 3]
        _, rotation = math_utils.compute_pose_error(
            ee_pos, ee_quat, ee_pos, _top_down_quat(yaw), rot_error_type="axis_angle"
        )
        self._ik_controller.set_command(torch.cat([self._processed_actions[:, :3], rotation], dim=-1), ee_pos, ee_quat)

    def apply_actions(self):
        """Uniformly scale the IK step to respect configured joint target and PD position-error bounds."""
        ee_pos, ee_quat = self._compute_frame_pose()
        if (self._max_offset is None and self._pos_lower is None) or ee_quat.norm() == 0:
            super().apply_actions()
            return
        joint_pos = self._asset.data.joint_pos.torch[:, self._joint_ids]
        jacobian = self._compute_frame_jacobian()
        step = self._ik_controller.compute(ee_pos, ee_quat, jacobian, joint_pos) - joint_pos
        ratio = torch.ones_like(step)
        if self._max_offset is not None:
            ratio = torch.maximum(ratio, step.abs() / self._max_offset)
        if self._pos_lower is not None:
            room = torch.where(step > 0, self._pos_upper - joint_pos, joint_pos - self._pos_lower).clamp(min=0.0)
            ratio = torch.maximum(ratio, step.abs() / room.clamp(min=1e-9))
        ratio = ratio.amax(dim=1, keepdim=True)
        self._asset.set_joint_position_target_index(target=joint_pos + step / ratio, joint_ids=self._joint_ids)


class TopDownStabilizedDeltaEEAction(TopDownDeltaEEAction):
    """Delta-IK action reinforced by a bounded roll/pitch operational-space torque."""

    cfg: TopDownStabilizedDeltaEEActionCfg

    def __init__(self, cfg: TopDownStabilizedDeltaEEActionCfg, env: ManagerBasedEnv):
        """Initialize the tilt-feedback gains and per-joint stabilizing torque bound."""
        super().__init__(cfg, env)
        if cfg.tilt_stiffness < 0.0 or cfg.tilt_damping < 0.0:
            raise ValueError("Tilt stiffness and damping must be non-negative.")
        if not 0.0 <= cfg.tilt_effort_limit_factor <= 1.0:
            raise ValueError("tilt_effort_limit_factor must lie in [0, 1].")
        self._tilt_stiffness = torch.full((self.num_envs, 1), cfg.tilt_stiffness, device=self.device)
        self._tilt_damping = torch.full((self.num_envs, 1), cfg.tilt_damping, device=self.device)
        effort_limits = self._asset.data.joint_effort_limits.torch[:, self._joint_ids]
        self._tilt_effort_limits = cfg.tilt_effort_limit_factor * effort_limits

    def apply_actions(self):
        """Apply the IK position target and a bounded torque that corrects actual roll/pitch error."""
        super().apply_actions()
        ee_pos, ee_quat = self._compute_frame_pose()
        jacobian = self._compute_frame_jacobian()
        joint_vel = self._asset.data.joint_vel.torch[:, self._joint_ids]
        angular_vel = (jacobian[:, 3:, :] @ joint_vel.unsqueeze(-1)).squeeze(-1)
        _, orientation_error = math_utils.compute_pose_error(
            ee_pos,
            ee_quat,
            self._ik_controller.ee_pos_des,
            self._ik_controller.ee_quat_des,
            rot_error_type="axis_angle",
        )
        correction = torch.zeros_like(orientation_error)
        correction[:, :2] = self._tilt_stiffness * orientation_error[:, :2] - self._tilt_damping * angular_vel[:, :2]
        effort = (jacobian[:, 3:, :].transpose(1, 2) @ correction.unsqueeze(-1)).squeeze(-1)
        effort = torch.clamp(effort, -self._tilt_effort_limits, self._tilt_effort_limits)
        self._asset.set_joint_effort_target_index(target=effort, joint_ids=self._joint_ids)


class RateLimitedBinaryJointPositionAction(BinaryJointPositionAction):
    """Binary joint position action whose target moves to the open or close command at a limited speed."""

    cfg: RateLimitedBinaryJointPositionActionCfg

    def __init__(self, cfg: RateLimitedBinaryJointPositionActionCfg, env: ManagerBasedEnv):
        """Initialize finger targets and their maximum change per physics step."""
        super().__init__(cfg, env)
        self._torch_joint_ids = self._asset.find_joints(cfg.joint_names)[0]
        vel_limits = self._asset.data.joint_vel_limits.torch[:, self._torch_joint_ids]
        factor = math.inf if cfg.joint_vel_factor is None else cfg.joint_vel_factor
        self._max_step = factor * vel_limits * env.physics_dt
        self._target = self._open_command.expand(self.num_envs, -1).clone()

    def apply_actions(self):
        """Move finger targets toward the binary command at the configured target speed."""
        self._target += torch.clamp(self._processed_actions - self._target, -self._max_step, self._max_step)
        self._asset.set_joint_position_target_index(target=self._target, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset finger targets to the open command for the selected environments."""
        super().reset(env_ids)
        self._target[slice(None) if env_ids is None else env_ids] = self._open_command

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
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils.buffers import DelayBuffer

from .frames import top_down_yaw

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv

    from .actions_cfg import TopDownDeltaEEActionCfg, TopDownStabilizedDeltaEEActionCfg

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


class TopDownDeltaEEAction(DifferentialInverseKinematicsAction):
    """4-DOF delta end-effector action: the hand moves in 3D and points straight"""

    cfg: TopDownDeltaEEActionCfg

    def __init__(self, cfg: TopDownDeltaEEActionCfg, env: ManagerBasedEnv):
        """Resolve the EE frame and initialize yaw and joint target bounds."""
        if cfg.controller.command_type != "pose" or not cfg.controller.use_relative_mode:
            raise ValueError("TopDownDeltaEEAction needs an IK controller with relative pose commands.")
        ee_frame = env.scene[cfg.ee_frame_name].cfg.target_frames[0]
        cfg = cfg.replace(
            body_name=ee_frame.prim_path.rsplit("/", 1)[-1],
            body_offset=cfg.OffsetCfg(pos=ee_frame.offset.pos, rot=ee_frame.offset.rot),
        )
        super().__init__(cfg, env)
        self._scale[:] = torch.tensor((*cfg.scale_trans, cfg.scale_rot), device=self.device)
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
        # delay line of the commanded target pose (position, quaternion), advanced once per physics step
        min_delay, max_delay = cfg.command_delay_range
        if not 0 <= min_delay <= max_delay:
            raise ValueError(f"Expected 0 <= min <= max in command_delay_range, got {cfg.command_delay_range}.")
        self._command_delay = None
        if max_delay > 0:
            self._command_delay = DelayBuffer(max_delay, self.num_envs, self.device)
            self._commanded_pose = torch.zeros(self.num_envs, 7, device=self.device)

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

    @property
    def scale_trans(self) -> tuple[float, float, float]:
        """Translation step (dx, dy, dz) per unit action [m]."""
        return tuple(self._scale[0, :3].tolist())

    @scale_trans.setter
    def scale_trans(self, scale: tuple[float, float, float]) -> None:
        """Set the translation step per unit action [m], e.g. from a curriculum; applies from the next action."""
        self._scale[:, :3] = torch.tensor(scale, device=self.device)

    def process_actions(self, actions: torch.Tensor):
        """Convert translation and yaw deltas into a top-down IK command."""
        self._raw_actions[:] = actions
        self._processed_actions[:] = self.raw_actions * self._scale
        ee_pos, ee_quat = self._compute_frame_pose()
        wrist_pos = self._asset.data.joint_pos.torch[:, self._wrist_id]
        lower, upper = self.cfg.wrist_joint_limits
        self._processed_actions[:, 3] = torch.clamp(self._processed_actions[:, 3], wrist_pos - upper, wrist_pos - lower)
        yaw = top_down_yaw(ee_quat) + self._processed_actions[:, 3]
        _, rotation = math_utils.compute_pose_error(
            ee_pos, ee_quat, ee_pos, _top_down_quat(yaw), rot_error_type="axis_angle"
        )
        self._ik_controller.set_command(torch.cat([self._processed_actions[:, :3], rotation], dim=-1), ee_pos, ee_quat)
        if self._command_delay is not None:
            self._commanded_pose[:] = torch.cat([self._ik_controller.ee_pos_des, self._ik_controller.ee_quat_des], -1)

    def apply_actions(self):
        """Uniformly scale the IK step to respect configured joint target and PD position-error bounds."""
        if self._command_delay is not None:
            delayed = self._command_delay.compute(self._commanded_pose)
            self._ik_controller.ee_pos_des[:] = delayed[:, :3]
            self._ik_controller.ee_quat_des[:] = delayed[:, 3:]
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

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the raw actions, empty the delay line and draw new command delays."""
        super().reset(env_ids)
        if self._command_delay is None:
            return
        ids = torch.arange(self.num_envs, device=self.device)[slice(None) if env_ids is None else env_ids]
        # an emptied delay line returns the newest command, so no target of the previous episode is applied
        self._command_delay.reset(ids)
        min_delay, max_delay = self.cfg.command_delay_range
        lags = torch.randint(min_delay, max_delay + 1, (len(ids),), dtype=torch.int, device=self.device)
        self._command_delay.set_time_lag(lags, ids)

    @property
    def command_delay(self) -> torch.Tensor:
        """Delay of the IK target in physics steps, shape [num_envs]."""
        if self._command_delay is None:
            return torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        return self._command_delay.time_lags


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

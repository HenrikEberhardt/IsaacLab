# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Frozen copy of the teacher's top-down delta IK arm action.

The cloth-drag task keeps changing its arm action, while a teacher checkpoint was trained with one fixed version. This
module keeps the version of the teacher run (``TopDownDeltaEEAction`` at commit 2918b254f, with workspace clamp,
joint-speed cap and joint-range margin), so the student env can

* compute the IK state (EE pose, Jacobian, joint bounds) that :mod:`.ik_label` turns teacher actions into joint-delta
  labels with, and
* run the teacher natively for the reference success rate of the label conversion.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils import configclass
from isaaclab.utils.buffers import DelayBuffer

from isaaclab_tasks.contrib.franka_cloth_drag.mdp.frames import top_down_yaw

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

TOP_DOWN_QUAT = (1.0, 0.0, 0.0, 0.0)


def top_down_quat(yaw: torch.Tensor) -> torch.Tensor:
    """Orientations (x, y, z, w) that point the hand straight down at the given yaws [rad], shape [N, 4]."""
    zeros = torch.zeros_like(yaw)
    top_down = torch.tensor(TOP_DOWN_QUAT, device=yaw.device).expand(len(yaw), 4)
    return math_utils.quat_mul(math_utils.quat_from_euler_xyz(zeros, zeros, yaw), top_down)


class TeacherTopDownIKAction(DifferentialInverseKinematicsAction):
    """4-DOF delta end-effector action ``(dx, dy, dz, dyaw)`` of the teacher run; the hand points straight down."""

    cfg: TeacherTopDownIKActionCfg

    def __init__(self, cfg: TeacherTopDownIKActionCfg, env: ManagerBasedEnv):
        """Resolve the EE frame and initialize workspace, yaw and joint target bounds."""
        if cfg.controller.command_type != "pose" or not cfg.controller.use_relative_mode:
            raise ValueError("TeacherTopDownIKAction needs an IK controller with relative pose commands.")
        ee_frame = env.scene[cfg.ee_frame_name].cfg.target_frames[0]
        cfg = cfg.replace(
            body_name=ee_frame.prim_path.rsplit("/", 1)[-1],
            body_offset=cfg.OffsetCfg(pos=ee_frame.offset.pos, rot=ee_frame.offset.rot),
        )
        super().__init__(cfg, env)
        self._scale[:] = torch.tensor((*cfg.scale_trans, cfg.scale_rot), device=self.device)
        self._space_lower = torch.tensor((cfg.x_limits[0], cfg.y_limits[0], cfg.z_min), device=self.device)
        self._space_upper = torch.tensor((cfg.x_limits[1], cfg.y_limits[1], math.inf), device=self.device)
        self._wrist_id = self._asset.find_joints(self._joint_names[-1])[0][0]
        # joint targets stay this far inside the hard limits [rad]
        limits = self._asset.data.joint_pos_limits.torch[:, self._joint_ids]
        self._pos_lower = (limits[..., 0] + cfg.joint_range_margin).clone()
        self._pos_upper = (limits[..., 1] - cfg.joint_range_margin).clone()
        # PD position-error cap that approximates a joint-speed bound [rad]
        stiffness = self._asset.data.joint_stiffness.torch[:, self._joint_ids]
        damping = self._asset.data.joint_damping.torch[:, self._joint_ids]
        vel_limits = self._asset.data.joint_vel_limits.torch[:, self._joint_ids]
        self._max_offset = (cfg.joint_vel_factor * vel_limits * damping / stiffness).clone()
        # delay line of the commanded target pose (position, quaternion), advanced once per physics step
        min_delay, max_delay = cfg.command_delay_range
        if not 0 <= min_delay <= max_delay:
            raise ValueError(f"Expected 0 <= min <= max in command_delay_range, got {cfg.command_delay_range}.")
        self._command_delay = None
        if max_delay > 0:
            self._command_delay = DelayBuffer(max_delay, self.num_envs, self.device)
            self._commanded_pose = torch.zeros(self.num_envs, 7, device=self.device)
        self._first_step = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)
        self._substep = 0

    @property
    def action_dim(self) -> int:
        """Return the four action dimensions: translation and yaw."""
        return 4

    @property
    def first_step(self) -> torch.Tensor:
        """Joint target offset of the first physics substep after the last action [rad], shape [N, 7]."""
        return self._first_step

    def ik_state(self) -> torch.Tensor:
        """Current inputs of :func:`.ik_label.teacher_actions_to_joint_deltas`, shape [N, IK_STATE_DIM].

        Joint positions [rad], EE position [m] and orientation (x, y, z, w) in the base frame, the EE Jacobian in the
        base frame, the lower and upper joint target bounds [rad] and the PD position-error cap [rad].
        """
        ee_pos, ee_quat = self._compute_frame_pose()
        jacobian = self._compute_frame_jacobian()
        joint_pos = self._asset.data.joint_pos.torch[:, self._joint_ids]
        return torch.cat(
            (joint_pos, ee_pos, ee_quat, jacobian.flatten(1), self._pos_lower, self._pos_upper, self._max_offset),
            dim=-1,
        )

    def frame_jacobian(self) -> torch.Tensor:
        """Geometric Jacobian of the EE frame in the base frame, shape [N, 6, 7]."""
        return self._compute_frame_jacobian().clone()

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
            ee_pos, ee_quat, ee_pos, top_down_quat(yaw), rot_error_type="axis_angle"
        )
        self._ik_controller.set_command(torch.cat([self._processed_actions[:, :3], rotation], dim=-1), ee_pos, ee_quat)
        if self._command_delay is not None:
            self._commanded_pose[:] = torch.cat([self._ik_controller.ee_pos_des, self._ik_controller.ee_quat_des], -1)
        self._substep = 0

    def apply_actions(self):
        """Uniformly scale the IK step to respect the joint target and PD position-error bounds."""
        if self._command_delay is not None:
            delayed = self._command_delay.compute(self._commanded_pose)
            self._ik_controller.ee_pos_des[:] = delayed[:, :3]
            self._ik_controller.ee_quat_des[:] = delayed[:, 3:]
        ee_pos, ee_quat = self._compute_frame_pose()
        if ee_quat.norm() == 0:
            super().apply_actions()
            return
        joint_pos = self._asset.data.joint_pos.torch[:, self._joint_ids]
        jacobian = self._compute_frame_jacobian()
        step = self._ik_controller.compute(ee_pos, ee_quat, jacobian, joint_pos) - joint_pos
        ratio = torch.maximum(torch.ones_like(step), step.abs() / self._max_offset)
        room = torch.where(step > 0, self._pos_upper - joint_pos, joint_pos - self._pos_lower).clamp(min=0.0)
        ratio = torch.maximum(ratio, step.abs() / room.clamp(min=1e-9))
        step = step / ratio.amax(dim=1, keepdim=True)
        if self._substep == 0:
            self._first_step[:] = step
        self._substep += 1
        self._asset.set_joint_position_target_index(target=joint_pos + step, joint_ids=self._joint_ids)

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


@configclass
class TeacherTopDownIKActionCfg(DifferentialInverseKinematicsActionCfg):
    """Configuration of :class:`TeacherTopDownIKAction`; the defaults are those of the teacher run."""

    class_type: type = TeacherTopDownIKAction

    body_name: str = ""
    """Body resolved from ``ee_frame_name`` when the action is initialized."""

    ee_frame_name: str = "ee_frame"
    """Scene frame transformer that defines the end-effector frame."""

    scale_trans: tuple[float, float, float] = (0.015, 0.015, 0.015)
    """Scale of the ``(dx, dy, dz)`` actions per policy step [m]."""

    scale_rot: float = 0.2
    """Scale of the ``dyaw`` action per policy step [rad]."""

    wrist_joint_limits: tuple[float, float] = (-2.0 * math.pi / 3.0, 2.0 * math.pi / 3.0)
    """Range of the wrist joint [rad]."""

    limit_operation_space: bool = True
    """Whether to clamp commanded EE positions to the workspace below."""

    x_limits: tuple[float, float] = (0.30, 0.80)
    """End-effector x-range [m]."""

    y_limits: tuple[float, float] = (-0.40, 0.40)
    """End-effector y-range [m]."""

    z_min: float = 0.0193
    """Lowest end-effector height [m]."""

    joint_vel_factor: float = 0.9
    """Velocity-limit fraction that caps the PD position error by uniformly scaling the IK step."""

    joint_range_margin: float = 0.05
    """Margin inside the hard joint limits [rad] that the IK step is scaled to respect."""

    command_delay_range: tuple[int, int] = (0, 4)
    """Range of the delay of the IK target in physics steps, drawn uniformly per environment at reset."""

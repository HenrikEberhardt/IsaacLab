# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Frozen copy of the teacher run's top-down delta IK arm action (TopDownDeltaEEAction at 1fb4d0c93)."""

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
UNBOUNDED = 1.0e6  # [rad] joint bound or PD error cap that never binds


def top_down_quat(yaw: torch.Tensor) -> torch.Tensor:
    """Hand-down orientations (x, y, z, w) at the given yaws [rad], shape [N, 4]."""
    zeros = torch.zeros_like(yaw)
    top_down = torch.tensor(TOP_DOWN_QUAT, device=yaw.device).expand(len(yaw), 4)
    return math_utils.quat_mul(math_utils.quat_from_euler_xyz(zeros, zeros, yaw), top_down)


class TeacherTopDownIKAction(DifferentialInverseKinematicsAction):
    """4-DOF delta EE action (dx, dy, dz, dyaw) with the hand pointing down."""

    cfg: TeacherTopDownIKActionCfg

    def __init__(self, cfg: TeacherTopDownIKActionCfg, env: ManagerBasedEnv):
        """Resolve the EE frame and the workspace, yaw and joint target bounds."""
        if cfg.controller.command_type != "pose" or not cfg.controller.use_relative_mode:
            raise ValueError("TeacherTopDownIKAction needs an IK controller with relative pose commands.")
        ee_frame = env.scene[cfg.ee_frame_name].cfg.target_frames[0]
        cfg = cfg.replace(
            body_name=ee_frame.prim_path.rsplit("/", 1)[-1],
            body_offset=cfg.OffsetCfg(pos=ee_frame.offset.pos, rot=ee_frame.offset.rot),
        )
        super().__init__(cfg, env)
        self._scale[:] = torch.tensor((*cfg.scale_trans, cfg.scale_rot), device=self.device)
        self._wrist_id = self._asset.find_joints(self._joint_names[-1])[0][0]
        limits = self._asset.data.joint_pos_limits.torch[:, self._joint_ids]
        self._pos_lower = torch.full_like(limits[..., 0], -UNBOUNDED)  # [rad]
        self._pos_upper = torch.full_like(limits[..., 1], UNBOUNDED)  # [rad]
        if cfg.joint_range_margin is not None:
            self._pos_lower = (limits[..., 0] + cfg.joint_range_margin).clone()
            self._pos_upper = (limits[..., 1] - cfg.joint_range_margin).clone()
        self._max_offset = torch.full_like(limits[..., 0], UNBOUNDED)  # PD error cap [rad]
        if cfg.joint_vel_factor is not None:
            stiffness = self._asset.data.joint_stiffness.torch[:, self._joint_ids]
            damping = self._asset.data.joint_damping.torch[:, self._joint_ids]
            vel_limits = self._asset.data.joint_vel_limits.torch[:, self._joint_ids]
            self._max_offset = (cfg.joint_vel_factor * vel_limits * damping / stiffness).clone()
        min_delay, max_delay = cfg.command_delay_range
        if not 0 <= min_delay <= max_delay:
            raise ValueError(f"Expected 0 <= min <= max in command_delay_range, got {cfg.command_delay_range}.")
        self._command_delay = None
        if max_delay > 0:
            self._command_delay = DelayBuffer(max_delay, self.num_envs, self.device)
            self._commanded_pose = torch.zeros(self.num_envs, 7, device=self.device)  # position, quaternion
        self._first_step = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)
        self._substep = 0

    @property
    def action_dim(self) -> int:
        """Translation and yaw."""
        return 4

    @property
    def first_step(self) -> torch.Tensor:
        """Joint target offset of the first physics substep after the last action [rad], shape [N, 7]."""
        return self._first_step

    def ik_state(self) -> torch.Tensor:
        """Joint pos, EE pose, EE Jacobian, joint target bounds and PD error cap, shape [N, IK_STATE_DIM]."""
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
        if self.cfg.z_min is not None:  # shorten the z step so the target ends on the floor at the lowest
            target_z = (ee_pos[:, 2] + self._processed_actions[:, 2]).clamp(min=self.cfg.z_min)
            self._processed_actions[:, 2] = target_z - ee_pos[:, 2]
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
        """Uniformly scale the IK step to the joint target and PD error bounds (no-op when unbounded)."""
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
        self._command_delay.reset(ids)  # an emptied line returns the newest command
        min_delay, max_delay = self.cfg.command_delay_range
        lags = torch.randint(min_delay, max_delay + 1, (len(ids),), dtype=torch.int, device=self.device)
        self._command_delay.set_time_lag(lags, ids)


@configclass
class TeacherTopDownIKActionCfg(DifferentialInverseKinematicsActionCfg):
    """Configuration of :class:`TeacherTopDownIKAction`; the defaults are the teacher run's."""

    class_type: type = TeacherTopDownIKAction

    body_name: str = ""
    """Resolved from ``ee_frame_name`` at init."""

    ee_frame_name: str = "ee_frame"
    """Frame transformer that defines the EE frame."""

    scale_trans: tuple[float, float, float] = (0.015, 0.015, 0.015)
    """(dx, dy, dz) scale per policy step [m]."""

    scale_rot: float = 0.2
    """dyaw scale per policy step [rad]."""

    wrist_joint_limits: tuple[float, float] = (-2.0 * math.pi / 3.0, 2.0 * math.pi / 3.0)
    """Wrist joint window [rad]."""

    z_min: float | None = 0.0193
    """Floor of the EE target height [m]; None for no floor."""

    joint_vel_factor: float | None = None
    """Velocity-limit fraction that caps the PD position error; None for no cap."""

    joint_range_margin: float | None = None
    """Margin inside the hard joint limits [rad]; None for no range scaling."""

    command_delay_range: tuple[int, int] = (0, 4)
    """IK target delay in physics steps, drawn per env at reset."""

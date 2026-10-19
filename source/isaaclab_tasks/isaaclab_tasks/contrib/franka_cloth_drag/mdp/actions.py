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

from .cloth_grasp import ClothGraspConstraint
from .frames import top_down_yaw

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, DeformableObject
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

    from .actions_cfg import (
        ClothGraspActionCfg,
        LimitedRelativeJointPositionActionCfg,
        RateLimitedBinaryJointPositionActionCfg,
        TopDownDeltaEEActionCfg,
    )

TOP_DOWN_QUAT = (1.0, 0.0, 0.0, 0.0)

PAD_HALF_WIDTH = 0.0088
PAD_THICKNESS = 0.0152
PAD_HALF_LENGTH = 0.0082
HAND_HALF_EXTENTS = (0.04, 0.10, 0.04)
HAND_CENTER_Z = -0.0737


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


class ClothGraspAction(RateLimitedBinaryJointPositionAction):
    """Rate-limited gripper that locks selected cloth nodes to the EE frame after the fingers settle."""

    cfg: ClothGraspActionCfg

    def __init__(self, cfg: ClothGraspActionCfg, env: ManagerBasedRLEnv):
        """Initialize the gripper command and optional cloth grasp constraint."""
        super().__init__(cfg, env)
        self._closed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.constraint: ClothGraspConstraint | None = None
        if not cfg.attach:
            return
        command_cfg = env.command_manager.get_term(cfg.command_name).cfg
        self._cloth: DeformableObject = env.scene[command_cfg.object_name]
        self._ee_frame = env.scene[command_cfg.ee_frame_name]
        self._finger_id = self._asset.find_joints(command_cfg.finger_joint_name)[0][0]
        layer_half_thickness = cfg.layer_half_thickness
        if layer_half_thickness is None:
            material = getattr(self._cloth.cfg.spawn, "physics_material", None)
            layer_half_thickness = getattr(material, "particle_radius", 0.002)
        self._pinch_opening = 2.0 * layer_half_thickness
        self.constraint = ClothGraspConstraint(
            env,
            command_cfg,
            layer_half_thickness=layer_half_thickness,
            squeeze_lift=cfg.squeeze_lift,
            squeeze_steps=cfg.squeeze_steps * env.cfg.decimation,
        )
        self._grasp_tried = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._held_opening = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._any_active = False
        self._targets_dirty = True

    @property
    def closed(self) -> torch.Tensor:
        """Whether each gripper is commanded closed, shape [num_envs]."""
        return self._closed

    @property
    def is_grasping(self) -> torch.Tensor:
        """Whether each gripper holds cloth nodes locked to the EE frame. Shape is (num_envs,)."""
        if self.constraint is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self.constraint.active

    @property
    def num_grasped(self) -> torch.Tensor:
        """Number of cloth nodes locked to the EE frame. Shape is (num_envs,)."""
        if self.constraint is None:
            return torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        return self.constraint.num_grasped

    def process_actions(self, actions: torch.Tensor):
        """Update the gripper command, grasp/release transitions, holding targets and cloth contacts."""
        super().process_actions(actions)
        action = self._raw_actions[:, 0]
        threshold = self.cfg.latch_threshold
        if threshold is None:
            self._closed = action < 0.0
        else:
            self._closed = torch.where(action < -threshold, True, torch.where(action > threshold, False, self._closed))
            self._processed_actions = torch.where(self._closed.unsqueeze(1), self._close_command, self._open_command)
            if self.cfg.clip is not None:
                self._processed_actions = torch.clamp(
                    self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
                )
        if self.constraint is None:
            return
        constraint = self.constraint
        close = self._closed
        joint_pos = self._asset.data.joint_pos.torch
        opening = joint_pos[:, self._finger_id]

        released = constraint.active & ~close & (constraint.squeeze_progress <= 0.0)
        if released.any():
            constraint.deactivate(released.nonzero(as_tuple=False).squeeze(-1))
            self._targets_dirty = True
        opened = ~close & ~constraint.active & (opening >= self._held_opening[:, 0] + self.cfg.release_travel)
        self._grasp_tried &= ~(released | opened)

        ramp_done = torch.all((self._target - self._close_command).abs() <= 1e-6, dim=1)
        settled = self._asset.data.joint_vel.torch[:, self._finger_id].abs() <= self.cfg.settle_speed
        grasping = close & ~self._grasp_tried & ramp_done & settled
        if grasping.any():
            grasp_ids = grasping.nonzero(as_tuple=False).squeeze(-1)
            self._grasp_tried[grasp_ids] = True
            self._held_opening[grasp_ids] = joint_pos[grasp_ids][:, self._torch_joint_ids]
            holding_ids = constraint.activate(grasp_ids)
            self._target[holding_ids] = self._held_opening[holding_ids]

        self._processed_actions = torch.where(
            constraint.active.unsqueeze(-1), self._hold_target(), self._processed_actions
        )

        clearing = constraint.contacts_disabled & ~constraint.active
        if clearing.any():
            clearing_ids = clearing.nonzero(as_tuple=False).squeeze(-1)
            constraint.restore_contacts(clearing_ids[self._clear_of_cloth(clearing_ids)])

        self._any_active = bool(constraint.active.any())

    def _hold_target(self) -> torch.Tensor:
        """Interpolate finger targets with the squeeze progress [m], shape [num_envs, num_joints]."""
        alpha = self.constraint.squeeze_progress.unsqueeze(-1)
        pinched = self._held_opening.clamp(max=self._pinch_opening)
        return self._held_opening + alpha * (pinched - self._held_opening)

    def _clear_of_cloth(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Check cloth-node clearance from the pads, finger gap and hand, shape [len(env_ids)]."""
        ee_pos_w = self._ee_frame.data.target_pos_w.torch[env_ids, 0]
        ee_quat_w = self._ee_frame.data.target_quat_w.torch[env_ids, 0]
        nodal_pos_w = self._cloth.data.nodal_pos_w.torch[env_ids]
        ee_quat_w = ee_quat_w.unsqueeze(1).expand(-1, nodal_pos_w.shape[1], -1)
        rel = math_utils.quat_apply_inverse(ee_quat_w, nodal_pos_w - ee_pos_w.unsqueeze(1))
        margin = self.cfg.contact_clearance
        opening = self._asset.data.joint_pos.torch[env_ids, self._finger_id].unsqueeze(1)
        near_pads = (
            (rel[..., 0].abs() <= PAD_HALF_WIDTH + margin)
            & (rel[..., 1].abs() <= opening + PAD_THICKNESS + margin)
            & (rel[..., 2].abs() <= PAD_HALF_LENGTH + margin)
        )
        near_hand = (
            (rel[..., 0].abs() <= HAND_HALF_EXTENTS[0] + margin)
            & (rel[..., 1].abs() <= HAND_HALF_EXTENTS[1] + margin)
            & ((rel[..., 2] - HAND_CENTER_Z).abs() <= HAND_HALF_EXTENTS[2] + margin)
        )
        return ~(near_pads | near_hand).any(dim=1)

    def apply_actions(self):
        """Advance the fingers and squeeze, then update attached cloth-node targets."""
        super().apply_actions()
        if self.constraint is not None and (self._any_active or self._targets_dirty):
            self.constraint.advance_squeeze(self._closed)
            self._cloth.write_nodal_kinematic_target_to_sim_index(self.constraint.kinematic_targets())
            self._targets_dirty = False

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset selected grippers, release attached nodes and restore cloth contacts."""
        super().reset(env_ids)
        self._closed[slice(None) if env_ids is None else env_ids] = False
        if self.constraint is None:
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.constraint.deactivate(env_ids)
        self.constraint.restore_contacts(env_ids)
        self._grasp_tried[env_ids] = False
        self._any_active = bool(self.constraint.active.any())
        self._cloth.write_nodal_kinematic_target_to_sim_index(self.constraint.kinematic_targets())
        self._targets_dirty = False

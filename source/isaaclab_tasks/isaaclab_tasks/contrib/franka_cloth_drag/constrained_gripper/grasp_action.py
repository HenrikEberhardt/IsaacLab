# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gripper action term of the constrained gripper: the Franka fingers plus the lock of the pinched cloth nodes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

from ..mdp.actions import RateLimitedBinaryJointPositionAction
from .constraint import ClothGraspConstraint

if TYPE_CHECKING:
    from isaaclab.assets import DeformableObject
    from isaaclab.envs import ManagerBasedRLEnv

    from .grasp_action_cfg import ClothGraspActionCfg


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
        self._finger_id = self._asset.find_joints(cfg.finger_joint_name)[0][0]
        layer_half_thickness = cfg.layer_half_thickness
        if layer_half_thickness is None:
            material = getattr(self._cloth.cfg.spawn, "physics_material", None)
            layer_half_thickness = getattr(material, "particle_radius", 0.002)
        self._pinch_opening = 2.0 * layer_half_thickness
        self.constraint = ClothGraspConstraint(
            env,
            command_cfg,
            grasp_min_nodes=cfg.grasp_min_nodes,
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
    def grasp_tried(self) -> torch.Tensor:
        """Whether the current close has attempted the lock (it locked, or found too few nodes); cleared when the
        gripper reopens. Shape is (num_envs,)."""
        if self.constraint is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self._grasp_tried

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
        cfg = self.cfg
        margin = cfg.contact_clearance
        hand = cfg.contact_hand_half_extents
        opening = self._asset.data.joint_pos.torch[env_ids, self._finger_id].unsqueeze(1)
        near_pads = (
            (rel[..., 0].abs() <= cfg.contact_pad_half_width + margin)
            & (rel[..., 1].abs() <= opening + cfg.contact_pad_thickness + margin)
            & (rel[..., 2].abs() <= cfg.contact_pad_half_length + margin)
        )
        near_hand = (
            (rel[..., 0].abs() <= hand[0] + margin)
            & (rel[..., 1].abs() <= hand[1] + margin)
            & ((rel[..., 2] - cfg.contact_hand_center_z).abs() <= hand[2] + margin)
        )
        return ~(near_pads | near_hand).any(dim=1)

    def apply_actions(self):
        """Advance the fingers and squeeze, then update attached cloth-node targets."""
        super().apply_actions()
        if self.constraint is not None and (self._any_active or self._targets_dirty):
            self.constraint.advance_squeeze(self._closed)
            self._cloth.write_nodal_kinematic_target_to_sim_index(self.constraint.kinematic_targets())
            self._targets_dirty = False

    def get_grasp_state(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return copies of the gripper and grasp state of the given environments, see :meth:`set_grasp_state`."""
        if self.constraint is None:
            raise RuntimeError("The grasp state needs the lock constraint (attach=True).")
        state = {
            "closed": self._closed[env_ids].clone(),
            "finger_target": self._target[env_ids].clone(),
            "grasp_tried": self._grasp_tried[env_ids].clone(),
            "held_opening": self._held_opening[env_ids].clone(),
        }
        return state | self.constraint.get_state(env_ids)

    def set_grasp_state(self, env_ids: torch.Tensor, state: dict[str, torch.Tensor]) -> None:
        """Continue a stored gripper and grasp state in the given environments, after their reset.

        Restores the closing latch, the rate-limited finger targets and the lock constraint (see
        :meth:`~.constraint.ClothGraspConstraint.set_state`), and locks the grasped nodes where they are now. The next
        :meth:`apply_actions` moves them with the hand.

        Args:
            env_ids: The environments, shape [N].
            state: The state of :meth:`get_grasp_state`, each entry with the environments as first dimension.
        """
        if self.constraint is None:
            raise RuntimeError("The grasp state needs the lock constraint (attach=True).")
        closed = state["closed"]
        self._closed[env_ids] = closed
        self._raw_actions[env_ids, 0] = torch.where(closed, -1.0, 1.0)
        self._target[env_ids] = state["finger_target"]
        self._grasp_tried[env_ids] = state["grasp_tried"]
        self._held_opening[env_ids] = state["held_opening"]
        self.constraint.set_state(env_ids, state)
        self._any_active = bool(self.constraint.active.any())
        nodal_pos_w = self._cloth.data.nodal_pos_w.torch[env_ids]
        locked = self.constraint.grasped[env_ids] & self.constraint.active[env_ids].unsqueeze(1)
        targets = torch.cat([nodal_pos_w, (~locked).float().unsqueeze(-1)], dim=-1)
        self._cloth.write_nodal_kinematic_target_to_sim_index(targets, env_ids=env_ids)
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

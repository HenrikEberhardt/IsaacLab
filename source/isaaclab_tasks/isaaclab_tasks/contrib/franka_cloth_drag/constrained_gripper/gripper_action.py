# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action term of the constrained gripper: the Robotiq fingers plus the lock of the pinched cloth nodes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction

from .constraint_node_lock import ClothNodeLock
from .robotiq import joint_from_pad_gap, pad_gap_from_joint, robotiq_collision_corners

if TYPE_CHECKING:
    from isaaclab.assets import DeformableObject
    from isaaclab.envs import ManagerBasedRLEnv

    from .gripper_action_cfg import ConstrainedGripperActionCfg


class ConstrainedGripperAction(BinaryJointPositionAction):
    """Gripper with a hysteresis that locks the pinched cloth nodes to the EE frame once the closing fingers settle.

    The fingers get a step open or close target, as the real gripper does; the finger actuator sets their speed and
    command delay. Per step (:meth:`process_actions`):

    1. Hysteresis: an action below ``-hysteresis_threshold`` closes, above ``hysteresis_threshold`` opens, otherwise
       the command stays. At 0 it is the plain sign of the action.
    2. Release: commanded open, the lock lets go once the pads are ``release_gap`` wider than the pinch.
    3. Grasp: closed, once the finger has moved in the closing direction and settled again (or ``close_timeout`` after
       the close command, if it never moved), the gripper tries the lock once (:attr:`grasp_tried`); it re-arms after
       opening by ``release_travel``.
    4. Hold: while locked and closed, the fingers track the pinch gap instead of the close command.
    5. Clearance: after a release, the gripper-cloth contacts come back once the cloth is ``contact_clearance`` away.
    """

    cfg: ConstrainedGripperActionCfg

    def __init__(self, cfg: ConstrainedGripperActionCfg, env: ManagerBasedRLEnv):
        """Initialize the finger command, the node lock and the collider boxes of the contact clearance."""
        super().__init__(cfg, env)
        command_cfg = env.command_manager.get_term(cfg.command_name).cfg
        self._cloth: DeformableObject = env.scene[command_cfg.object_name]
        self._finger_id = self._asset.find_joints(cfg.finger_joint_name)[0][0]
        self._torch_joint_ids = self._asset.find_joints(cfg.joint_names)[0]
        self._step_dt = env.step_dt

        # finger command
        self._closed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._grasp_tried = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # whether the finger has moved in the closing direction since the close command
        self._moved = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # time [s] since the close command, zero while open
        self._close_time = torch.zeros(self.num_envs, device=self.device)
        # finger joint angle [rad] at the last grasp attempt, shape [num_envs, 1]
        self._held_joint = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        # node lock
        layer_half_thickness = cfg.layer_half_thickness
        if layer_half_thickness is None:
            material = getattr(self._cloth.cfg.spawn, "physics_material", None)
            layer_half_thickness = getattr(material, "particle_radius", 0.002)
        # gap [m] the squeeze closes the pads to, at most: two cloth layers
        self._pinch_gap = 4.0 * layer_half_thickness
        self.node_lock = ClothNodeLock(
            env,
            command_cfg,
            grasp_min_nodes=cfg.grasp_min_nodes,
            layer_half_thickness=layer_half_thickness,
            squeeze_steps=cfg.squeeze_steps * env.cfg.decimation,
        )
        self._any_active = False
        self._targets_dirty = True

        # box of every gripper collider in its body's frame [m], for the contact clearance
        corners = robotiq_collision_corners()
        self._collider_body_ids = self._asset.find_bodies(list(corners), preserve_order=True)[0]
        points = [torch.tensor(corner, device=self.device) for corner in corners.values()]
        self._collider_lower = torch.stack([point.amin(dim=0) for point in points])
        self._collider_upper = torch.stack([point.amax(dim=0) for point in points])

    @property
    def closed(self) -> torch.Tensor:
        """Whether each gripper is commanded closed."""
        return self._closed

    @property
    def is_grasping(self) -> torch.Tensor:
        """Whether each gripper holds cloth nodes locked to the EE frame."""
        return self.node_lock.active

    @property
    def grasp_tried(self) -> torch.Tensor:
        """Whether the current close has tried the lock (it locked, or found too few nodes); cleared on reopening."""
        return self._grasp_tried

    @property
    def num_grasped(self) -> torch.Tensor:
        """Number of cloth nodes locked to the EE frame."""
        return self.node_lock.num_grasped

    def process_actions(self, actions: torch.Tensor):
        """Update the hysteresis, the release and grasp transitions, the finger targets and the cloth contacts."""
        super().process_actions(actions)
        lock = self.node_lock
        action = self._raw_actions[:, 0]
        threshold = self.cfg.hysteresis_threshold
        was_closed = self._closed
        self._closed = torch.where(action < -threshold, True, torch.where(action > threshold, False, self._closed))
        close = self._closed
        new_close = close & ~was_closed
        self._processed_actions = torch.where(close.unsqueeze(1), self._close_command, self._open_command)
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

        # release: commanded open and the pads have left the pinch; re-arm after a failed grasp once opened
        gap = self._pad_gap()
        released = lock.active & ~close & (gap - self._hold_gap() >= self.cfg.release_gap)
        if released.any():
            lock.deactivate(released.nonzero(as_tuple=False).squeeze(-1))
            self._targets_dirty = True
        opened = ~close & ~lock.active & (gap >= self._held_gap()[:, 0] + self.cfg.release_travel)
        self._grasp_tried &= ~(released | opened)

        # grasp: once per close, when the finger has closed and come to rest (on the cloth or fully closed). The finger
        # angle grows while closing; the command delay keeps it still for a moment after the close command, so the
        # lock waits for closing motion first, or for the timeout if the finger cannot move.
        finger_vel = self._asset.data.joint_vel.torch[:, self._finger_id]
        self._close_time = torch.where(new_close | ~close, 0.0, self._close_time + self._step_dt)
        self._moved = (self._moved & ~new_close) | (close & (finger_vel > self.cfg.settle_speed))
        settled = finger_vel.abs() <= self.cfg.settle_speed
        closed_down = self._moved | (self._close_time >= self.cfg.close_timeout)
        grasping = close & ~self._grasp_tried & closed_down & settled
        if grasping.any():
            grasp_ids = grasping.nonzero(as_tuple=False).squeeze(-1)
            self._grasp_tried[grasp_ids] = True
            self._held_joint[grasp_ids] = self._asset.data.joint_pos.torch[grasp_ids][:, self._torch_joint_ids]
            lock.activate(grasp_ids)

        # hold: a closed gripper holds the pinch; an open one follows the open command until the release
        self._processed_actions = torch.where(
            (lock.active & close).unsqueeze(-1), self._hold_target(), self._processed_actions
        )

        # clearance: restore the contacts once the released cloth is clear of the gripper
        clearing = lock.contacts_disabled & ~lock.active
        if clearing.any():
            clearing_ids = clearing.nonzero(as_tuple=False).squeeze(-1)
            lock.restore_contacts(clearing_ids[self._clear_of_cloth(clearing_ids)])

        self._any_active = bool(lock.active.any())

    def apply_actions(self):
        """Advance the fingers and the squeeze, then move the locked cloth nodes with the hand."""
        super().apply_actions()
        if self._any_active or self._targets_dirty:
            self.node_lock.advance_squeeze(self._squeeze_ceiling())
            self._cloth.write_nodal_kinematic_target_to_sim_index(self.node_lock.kinematic_targets())
            self._targets_dirty = False

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the selected grippers, release their locked nodes and restore their cloth contacts."""
        super().reset(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._closed[env_ids] = False
        self._grasp_tried[env_ids] = False
        self._moved[env_ids] = False
        self._close_time[env_ids] = 0.0
        self._processed_actions[env_ids] = self._open_command
        self.node_lock.deactivate(env_ids)
        self.node_lock.restore_contacts(env_ids)
        self._any_active = bool(self.node_lock.active.any())
        self._cloth.write_nodal_kinematic_target_to_sim_index(self.node_lock.kinematic_targets())
        self._targets_dirty = False

    def get_grasp_state(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return copies of the gripper and lock state of the given environments, see :meth:`set_grasp_state`.

        ``finger_target`` is the current finger command, kept so that stored reset banks keep their layout.
        """
        state = {
            "closed": self._closed[env_ids].clone(),
            "finger_target": self._processed_actions[env_ids].clone(),
            "grasp_tried": self._grasp_tried[env_ids].clone(),
            "held_finger_pos": self._held_joint[env_ids].clone(),
        }
        return state | self.node_lock.get_state(env_ids)

    def set_grasp_state(self, env_ids: torch.Tensor, state: dict[str, torch.Tensor]) -> None:
        """Continue a stored gripper and lock state in the given environments, after their reset.

        The locked nodes are held where they are now; the next :meth:`apply_actions` moves them with the hand.

        The stored ``finger_target`` is not needed: the next :meth:`process_actions` recomputes the finger command from
        the restored hysteresis state and lock. A restored close that has not tried the lock yet tries it once the
        finger moved and settled, or after ``close_timeout``.

        Args:
            env_ids: The environments, shape [N].
            state: The state of :meth:`get_grasp_state`, each entry with the environments as first dimension.
        """
        closed = state["closed"]
        self._closed[env_ids] = closed
        self._raw_actions[env_ids, 0] = torch.where(closed, -1.0, 1.0)
        self._processed_actions[env_ids] = torch.where(closed.unsqueeze(1), self._close_command, self._open_command)
        self._moved[env_ids] = False
        self._close_time[env_ids] = 0.0
        self._grasp_tried[env_ids] = state["grasp_tried"]
        self._held_joint[env_ids] = state["held_finger_pos"]
        self.node_lock.set_state(env_ids, state)
        self._any_active = bool(self.node_lock.active.any())
        nodal_pos_w = self._cloth.data.nodal_pos_w.torch[env_ids]
        locked = self.node_lock.grasped[env_ids] & self.node_lock.active[env_ids].unsqueeze(1)
        targets = torch.cat([nodal_pos_w, (~locked).float().unsqueeze(-1)], dim=-1)
        self._cloth.write_nodal_kinematic_target_to_sim_index(targets, env_ids=env_ids)
        self._targets_dirty = False

    def _pad_gap(self) -> torch.Tensor:
        """Measured distance [m] between the pad faces."""
        return pad_gap_from_joint(self._asset.data.joint_pos.torch[:, self._finger_id], self.cfg.gap_curve)

    def _held_gap(self) -> torch.Tensor:
        """Distance [m] between the pad faces at the last grasp attempt, shape [num_envs, 1]."""
        return pad_gap_from_joint(self._held_joint, self.cfg.gap_curve)

    def _hold_gap(self) -> torch.Tensor:
        """Distance [m] between the pad faces while the lock holds the pinch."""
        return self._held_gap()[:, 0].clamp(max=self._pinch_gap)

    def _hold_target(self) -> torch.Tensor:
        """Finger targets [rad] from the grasp gap to the pinch gap by the squeeze progress, shape [num_envs, 1]."""
        alpha = self.node_lock.squeeze_progress.unsqueeze(-1)
        held_gap = self._held_gap()
        gap = held_gap + alpha * (self._hold_gap().unsqueeze(-1) - held_gap)
        return joint_from_pad_gap(gap, self.cfg.gap_curve)

    def _squeeze_ceiling(self) -> torch.Tensor:
        """Highest squeeze progress [0, 1] each gripper allows: full while closed, relaxing as the pads open."""
        relaxed = (1.0 - (self._pad_gap() - self._hold_gap()) / self.cfg.release_gap).clamp(0.0, 1.0)
        return torch.where(self._closed, torch.ones_like(relaxed), relaxed)

    def _clear_of_cloth(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Whether every cloth node is ``contact_clearance`` away from every gripper collider, shape [len(env_ids)]."""
        nodal_pos_w = self._cloth.data.nodal_pos_w.torch[env_ids]
        body_pos_w = self._asset.data.body_link_pos_w.torch[env_ids][:, self._collider_body_ids]
        body_quat_w = self._asset.data.body_link_quat_w.torch[env_ids][:, self._collider_body_ids]
        num_envs, num_nodes, num_bodies = len(env_ids), nodal_pos_w.shape[1], len(self._collider_body_ids)
        # nodes in the frame of every body: [num_envs, num_nodes, num_bodies, 3]
        rel_w = nodal_pos_w.unsqueeze(2) - body_pos_w.unsqueeze(1)
        quat = body_quat_w.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        rel = math_utils.quat_apply_inverse(quat.reshape(-1, 4), rel_w.reshape(-1, 3)).view(
            num_envs, num_nodes, num_bodies, 3
        )
        margin = self.cfg.contact_clearance
        inside = ((rel >= self._collider_lower - margin) & (rel <= self._collider_upper + margin)).all(dim=-1)
        return ~inside.any(dim=(1, 2))

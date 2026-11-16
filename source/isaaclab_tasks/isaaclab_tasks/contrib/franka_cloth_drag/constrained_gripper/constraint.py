# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cloth grasp that locks the cloth nodes pinched by the Franka gripper to the end-effector (EE) frame.

The constrained based gripper follows the implementation of Nayak et al.,
"A Simple Gripper Interface for Simulator-Agnostic Cloth Manipulation" (arXiv 2609.29340).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from ..mdp.frames import cloth_edges, cloth_triangles
from .geometry import pad_box_mask

if TYPE_CHECKING:
    from isaaclab.assets import DeformableObject
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.sensors import FrameTransformer

    from ..mdp.commands_cfg import ClothTargetCommandCfg


class ClothGraspConstraint:
    """Lock grasped cloth nodes to the end-effector (EE) frame with position-level bilateral constraints.

    A grasp locks every node inside the picker box, plus the nodes of every edge whose midpoint and of every
    triangle whose centroid lies inside it.

    Over ``squeeze_steps`` calls of :meth:`advance_squeeze` after the grasp, the two flanks of the pinched fold are
    pressed onto each other, a squeeze like Nayak Eq. 7-8 that stops at the cloth thickness. It relaxes again as the
    gripper opens.

    While the constraint holds cloth, the gripper stops colliding with the cloth, since the fingers would otherwise
    press against the immovable locked nodes and blow up the solver.
    """

    def __init__(
        self,
        env: ManagerBasedEnv,
        command_cfg: ClothTargetCommandCfg,
        grasp_min_nodes: int,
        layer_half_thickness: float,
        squeeze_lift: float,
        squeeze_steps: int,
    ):
        """Set up the empty grasp state for the cloth and EE frame"""
        self._env = env
        self._command_cfg = command_cfg
        self.grasp_min_nodes = grasp_min_nodes
        self.layer_half_thickness = layer_half_thickness
        self.squeeze_lift = squeeze_lift
        self._squeeze_rate = 1.0 / max(squeeze_steps, 1)
        self._set_gripper_contacts = getattr(env.sim.physics_manager, "set_proxy_particle_contacts", None)
        self._ee_frame: FrameTransformer = env.scene[command_cfg.ee_frame_name]
        self._cloth: DeformableObject = env.scene[command_cfg.object_name]
        cloth_cfg = SceneEntityCfg(command_cfg.object_name)
        self._edges = cloth_edges(env, cloth_cfg)
        self._triangles = cloth_triangles(env, cloth_cfg)
        num_envs, device = env.num_envs, env.device
        num_nodes = self._cloth.max_sim_vertices_per_body
        self.grasp_offsets = torch.zeros((num_envs, num_nodes, 3), device=device)
        self.grasped = torch.zeros((num_envs, num_nodes), dtype=torch.bool, device=device)
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.contacts_disabled = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.squeeze_progress = torch.zeros(num_envs, device=device)

    @property
    def num_grasped(self) -> torch.Tensor:
        """Number of nodes locked to the EE frame. Shape is (num_envs,)."""
        return (self.grasped & self.active.unsqueeze(1)).sum(dim=1)

    def select(self) -> torch.Tensor:
        """Return the nodes a grasp would lock now. Shape is (num_envs, num_nodes)."""
        nodal_pos_w = self._cloth.data.nodal_pos_w.torch
        selected = pad_box_mask(self._env, self._command_cfg, nodal_pos_w)
        for cells in (self._edges, self._triangles):
            inside = pad_box_mask(self._env, self._command_cfg, nodal_pos_w[:, cells].mean(dim=2))
            hits = torch.zeros_like(nodal_pos_w[..., 0])
            hits.index_add_(1, cells.flatten(), inside.float().repeat_interleave(cells.shape[1], dim=1))
            selected |= hits > 0
        return selected

    def activate(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Lock the selected nodes of the given environments to the EE frame if ``grasp_min_nodes`` is reached."""
        selected = self.select()[env_ids]
        holding = selected.sum(dim=1) >= max(self.grasp_min_nodes, 1)
        env_ids, selected = env_ids[holding], selected[holding]
        if len(env_ids) == 0:
            return env_ids
        ee_pos_w, ee_quat_w = self._ee_pose_w()
        nodal_pos_w = self._cloth.data.nodal_pos_w.torch[env_ids]
        ee_quat_w = ee_quat_w[env_ids].unsqueeze(1).expand(-1, nodal_pos_w.shape[1], -1)
        self.grasp_offsets[env_ids] = math_utils.quat_apply_inverse(
            ee_quat_w, nodal_pos_w - ee_pos_w[env_ids].unsqueeze(1)
        )
        self.grasped[env_ids] = selected
        self.active[env_ids] = True
        self.squeeze_progress[env_ids] = 0.0
        if self._set_gripper_contacts is not None:
            self._set_gripper_contacts(env_ids.tolist(), enabled=False)
            self.contacts_disabled[env_ids] = True
        return env_ids

    def deactivate(self, env_ids: torch.Tensor):
        """Release the locked nodes of the given environments."""
        self.active[env_ids] = False
        self.grasped[env_ids] = False

    def restore_contacts(self, env_ids: torch.Tensor):
        """Let the gripper of the given environments collide with the cloth again."""
        env_ids = env_ids[self.contacts_disabled[env_ids]]
        if len(env_ids) > 0 and self._set_gripper_contacts is not None:
            self._set_gripper_contacts(env_ids.tolist(), enabled=True)
            self.contacts_disabled[env_ids] = False

    def get_state(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return copies of the grasp state of the given environments."""
        return {
            "active": self.active[env_ids].clone(),
            "grasped": self.grasped[env_ids].clone(),
            "grasp_offsets": self.grasp_offsets[env_ids].clone(),
            "squeeze_progress": self.squeeze_progress[env_ids].clone(),
            "contacts_disabled": self.contacts_disabled[env_ids].clone(),
        }

    def set_state(self, env_ids: torch.Tensor, state: dict[str, torch.Tensor]):
        """Continue a grasp state of :meth:`get_state` in the given environments."""
        self.active[env_ids] = state["active"]
        self.grasped[env_ids] = state["grasped"]
        self.grasp_offsets[env_ids] = state["grasp_offsets"]
        self.squeeze_progress[env_ids] = state["squeeze_progress"]
        disable = env_ids[state["contacts_disabled"] & ~self.contacts_disabled[env_ids]]
        if len(disable) > 0 and self._set_gripper_contacts is not None:
            self._set_gripper_contacts(disable.tolist(), enabled=False)
            self.contacts_disabled[disable] = True

    def advance_squeeze(self, ceiling: torch.Tensor):
        """Ramp the squeeze of the holding environments in by one step"""
        self.squeeze_progress = torch.where(
            self.active,
            torch.minimum(self.squeeze_progress + self._squeeze_rate, ceiling).clamp(0.0, 1.0),
            self.squeeze_progress,
        )

    def kinematic_targets(self) -> torch.Tensor:
        """Return nodal targets [m] and flags (0 locked, 1 free) that move the grasped nodes with the EE frame."""
        nodal_pos_w = self._cloth.data.nodal_pos_w.torch
        targets = torch.cat([nodal_pos_w, torch.ones_like(nodal_pos_w[..., :1])], dim=-1)
        locked = self.grasped & self.active.unsqueeze(1)
        if not locked.any():
            return targets
        alpha = self.squeeze_progress.view(-1, 1)
        offsets = self.grasp_offsets.clone()
        pressed_y = offsets[..., 1].clamp(-self.layer_half_thickness, self.layer_half_thickness)
        offsets[..., 1] += alpha * (pressed_y - offsets[..., 1])
        offsets[..., 2] -= self.squeeze_lift * alpha
        ee_pos_w, ee_quat_w = self._ee_pose_w()
        ee_quat_w = ee_quat_w.unsqueeze(1).expand(-1, offsets.shape[1], -1)
        points = ee_pos_w.unsqueeze(1) + math_utils.quat_apply(ee_quat_w, offsets)
        targets[..., :3] = torch.where(locked.unsqueeze(-1), points, nodal_pos_w)
        targets[..., 3] = torch.where(locked, 0.0, 1.0)
        return targets

    def _ee_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the EE positions [m] and orientations (x, y, z, w) in the world frame."""
        return self._ee_frame.data.target_pos_w.torch[:, 0], self._ee_frame.data.target_quat_w.torch[:, 0]

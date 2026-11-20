# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Constraint that locks the cloth nodes pinched by the gripper to the end-effector (EE) frame.

Follows Nayak et al., "A Simple Gripper Interface for Simulator-Agnostic Cloth Manipulation" (arXiv 2609.29340).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from ..mdp.frames import cloth_edges, cloth_triangles
from .gripper_state import pad_box_mask

if TYPE_CHECKING:
    from isaaclab.assets import DeformableObject
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.sensors import FrameTransformer

    from ..mdp.commands_cfg import ClothTargetCommandCfg


class ClothNodeLock:
    """Lock grasped cloth nodes to the EE frame as kinematic targets.

    A grasp locks every node inside the pad box, plus the nodes of every edge whose midpoint and of every triangle
    whose centroid lies inside it. Over ``squeeze_steps`` calls of :meth:`advance_squeeze`, the two flanks of the
    pinched fold are pressed onto each other down to the cloth thickness (Nayak Eq. 7-8); the squeeze relaxes again as
    the gripper opens. While the lock holds cloth, the gripper stops colliding with the cloth, since the fingers would
    otherwise press against the immovable locked nodes and blow up the solver.
    """

    def __init__(
        self,
        env: ManagerBasedEnv,
        command_cfg: ClothTargetCommandCfg,
        grasp_min_nodes: int,
        layer_half_thickness: float,
        squeeze_steps: int,
    ):
        """Set up the empty lock state of the cloth and the EE frame."""
        self._env = env
        self._command_cfg = command_cfg
        self.grasp_min_nodes = grasp_min_nodes
        self.layer_half_thickness = layer_half_thickness
        self._squeeze_rate = 1.0 / max(squeeze_steps, 1)
        # provided by the task's coupler manager, see physics/coupler.py
        self._set_gripper_contacts = getattr(env.sim.physics_manager, "set_proxy_particle_contacts", None)
        self._ee_frame: FrameTransformer = env.scene[command_cfg.ee_frame_name]
        self._cloth: DeformableObject = env.scene[command_cfg.object_name]
        cloth_cfg = SceneEntityCfg(command_cfg.object_name)
        self._edges = cloth_edges(env, cloth_cfg)
        self._triangles = cloth_triangles(env, cloth_cfg)
        num_envs, device = env.num_envs, env.device
        num_nodes = self._cloth.max_sim_vertices_per_body
        self.grasp_offsets = torch.zeros((num_envs, num_nodes, 3), device=device)
        """Locked node positions [m] in the EE frame at the grasp, shape [num_envs, num_nodes, 3]."""
        self.grasped = torch.zeros((num_envs, num_nodes), dtype=torch.bool, device=device)
        """Nodes of the last grasp, shape [num_envs, num_nodes]; locked only while :attr:`active`."""
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        """Whether the lock holds cloth."""
        self.contacts_disabled = torch.zeros(num_envs, dtype=torch.bool, device=device)
        """Whether the gripper-cloth contacts are off."""
        self.squeeze_progress = torch.zeros(num_envs, device=device)
        """Squeeze progress, from 0 at the grasp to 1 when the fold is pressed together."""

    @property
    def num_grasped(self) -> torch.Tensor:
        """Number of nodes locked to the EE frame."""
        return (self.grasped & self.active.unsqueeze(1)).sum(dim=1)

    def select(self) -> torch.Tensor:
        """Return the nodes a grasp would lock now, shape [num_envs, num_nodes]."""
        nodal_pos_w = self._cloth.data.nodal_pos_w.torch
        selected = pad_box_mask(self._env, self._command_cfg, nodal_pos_w)
        for cells in (self._edges, self._triangles):
            inside = pad_box_mask(self._env, self._command_cfg, nodal_pos_w[:, cells].mean(dim=2))
            hits = torch.zeros_like(nodal_pos_w[..., 0])
            hits.index_add_(1, cells.flatten(), inside.float().repeat_interleave(cells.shape[1], dim=1))
            selected |= hits > 0
        return selected

    def activate(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Lock the selected nodes of the environments that select at least ``grasp_min_nodes``; return those ids."""
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
        self._set_contacts(env_ids, enabled=False)
        return env_ids

    def deactivate(self, env_ids: torch.Tensor):
        """Release the locked nodes of the given environments."""
        self.active[env_ids] = False
        self.grasped[env_ids] = False

    def restore_contacts(self, env_ids: torch.Tensor):
        """Let the gripper of the given environments collide with the cloth again."""
        self._set_contacts(env_ids, enabled=True)

    def get_state(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return copies of the lock state of the given environments."""
        return {
            "active": self.active[env_ids].clone(),
            "grasped": self.grasped[env_ids].clone(),
            "grasp_offsets": self.grasp_offsets[env_ids].clone(),
            "squeeze_progress": self.squeeze_progress[env_ids].clone(),
            "contacts_disabled": self.contacts_disabled[env_ids].clone(),
        }

    def set_state(self, env_ids: torch.Tensor, state: dict[str, torch.Tensor]):
        """Continue a lock state of :meth:`get_state` in the given environments."""
        self.active[env_ids] = state["active"]
        self.grasped[env_ids] = state["grasped"]
        self.grasp_offsets[env_ids] = state["grasp_offsets"]
        self.squeeze_progress[env_ids] = state["squeeze_progress"]
        self._set_contacts(env_ids[state["contacts_disabled"]], enabled=False)

    def advance_squeeze(self, ceiling: torch.Tensor):
        """Ramp the squeeze of the holding environments in by one step, up to ``ceiling`` [0, 1]."""
        self.squeeze_progress = torch.where(
            self.active,
            torch.minimum(self.squeeze_progress + self._squeeze_rate, ceiling).clamp(0.0, 1.0),
            self.squeeze_progress,
        )

    def kinematic_targets(self) -> torch.Tensor:
        """Return nodal targets [m] and flags (0 locked, 1 free) that move the locked nodes with the EE frame."""
        nodal_pos_w = self._cloth.data.nodal_pos_w.torch
        targets = torch.cat([nodal_pos_w, torch.ones_like(nodal_pos_w[..., :1])], dim=-1)
        locked = self.grasped & self.active.unsqueeze(1)
        if not locked.any():
            return targets
        # press the fold's flanks towards the EE frame's xz-plane, down to the layer thickness
        alpha = self.squeeze_progress.view(-1, 1)
        offsets = self.grasp_offsets.clone()
        pressed_y = offsets[..., 1].clamp(-self.layer_half_thickness, self.layer_half_thickness)
        offsets[..., 1] += alpha * (pressed_y - offsets[..., 1])
        ee_pos_w, ee_quat_w = self._ee_pose_w()
        ee_quat_w = ee_quat_w.unsqueeze(1).expand(-1, offsets.shape[1], -1)
        points = ee_pos_w.unsqueeze(1) + math_utils.quat_apply(ee_quat_w, offsets)
        targets[..., :3] = torch.where(locked.unsqueeze(-1), points, nodal_pos_w)
        targets[..., 3] = torch.where(locked, 0.0, 1.0)
        return targets

    def _set_contacts(self, env_ids: torch.Tensor, enabled: bool):
        """Switch the gripper-cloth contacts of the given environments whose contacts are not yet in that state."""
        if self._set_gripper_contacts is None:
            return
        env_ids = env_ids[self.contacts_disabled[env_ids] == enabled]
        if len(env_ids) > 0:
            self._set_gripper_contacts(env_ids.tolist(), enabled=enabled)
            self.contacts_disabled[env_ids] = not enabled

    def _ee_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the EE positions [m] and orientations (x, y, z, w) in the world frame."""
        return self._ee_frame.data.target_pos_w.torch[:, 0], self._ee_frame.data.target_quat_w.torch[:, 0]

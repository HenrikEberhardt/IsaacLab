# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cloth grasp that locks the cloth nodes pinched by the Franka gripper to the end-effector (EE) frame.

Adapted from ``GripperConstraint`` on the ``non-prehensile-cloth`` branch. The node selection and the squeeze follow
Nayak et al., "A Simple Gripper Interface for Simulator-Agnostic Cloth Manipulation" (arXiv 2609.29340).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from .frames import cloth_edges, cloth_triangles
from .stages import pad_box_mask

if TYPE_CHECKING:
    from isaaclab.assets import DeformableObject
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.sensors import FrameTransformer

    from .commands_cfg import ClothTargetCommandCfg


class ClothGraspConstraint:
    """Lock grasped cloth nodes to the end-effector (EE) frame with position-level bilateral constraints.

    every node inside the picker box, plus the nodes of every edge whose midpoint and of every triangle whose centroid lies inside it. 
    
    Each grasped node keeps its offset in the EE frame from the moment of the grasp.

    Over ``squeeze_steps`` calls of :meth:`advance_squeeze` after the grasp, the two flanks of the pinched fold are
    pressed onto each other, a squeeze like Nayak Eq. 7-8 that stops at the cloth thickness: the y component of each
    offset moves to ``clamp(p0_iy, -layer_half_thickness, layer_half_thickness)`` and the offset is lifted by
    ``squeeze_lift``, ``p_i = p0_i + alpha * ((clamp(p0_iy) - p0_iy) * e_y - squeeze_lift * e_z)``, with ``alpha``
    ramping from 0 to 1. Nodes already within one layer half thickness of the plane between the fingers stay where they
    are, so the layers end stacked on top of each other. The EE point lies between the finger pads, the EE frame's
    y-axis is the closing direction and its z-axis points down toward the table, so ``-e_z`` lifts the nodes.

    While the constraint holds cloth, the gripper stops colliding with the cloth in that environment, since the
    fingers would otherwise press against the immovable locked nodes and destabilize the coupling between the robot
    and the cloth solver. The caller should hold the fingers at the opening where they grasped. The gripper collides
    with the cloth again only once :meth:`restore_contacts` is called after it has left the cloth; turning the contacts
    on with the fingers inside the cloth blows up the coupling the same way.
    """

    def __init__(
        self,
        env: ManagerBasedEnv,
        command_cfg: ClothTargetCommandCfg,
        layer_half_thickness: float,
        squeeze_lift: float,
        squeeze_steps: int,
    ):
        """Initialize the constraint.

        Args:
            env: The environment.
            command_cfg: The command configuration with the robot, the cloth, the pad box and ``grasp_min_nodes``.
            layer_half_thickness: Half thickness of one cloth layer, the distance from the plane between the fingers
                to which the squeeze presses the grasped nodes [m].
            squeeze_lift: Distance the grasped nodes are lifted along the EE frame's -z axis while squeezed [m].
            squeeze_steps: Number of :meth:`advance_squeeze` calls over which the squeeze ramps in or out.
        """
        self._env = env
        self._command_cfg = command_cfg
        self.layer_half_thickness = layer_half_thickness
        self.squeeze_lift = squeeze_lift
        self._squeeze_rate = 1.0 / max(squeeze_steps, 1)
        # the task's coupler manager can switch the gripper's cloth contacts per environment
        self._set_gripper_contacts = getattr(env.sim.physics_manager, "set_proxy_particle_contacts", None)
        self._ee_frame: FrameTransformer = env.scene[command_cfg.ee_frame_name]
        self._cloth: DeformableObject = env.scene[command_cfg.object_name]
        cloth_cfg = SceneEntityCfg(command_cfg.object_name)
        self._edges = cloth_edges(env, cloth_cfg)
        self._triangles = cloth_triangles(env, cloth_cfg)

        num_envs, device = env.num_envs, env.device
        num_nodes = self._cloth.max_sim_vertices_per_body
        # grasp point of each node in the EE frame at the moment of the grasp [m]
        self.grasp_offsets = torch.zeros((num_envs, num_nodes, 3), device=device)
        self.grasped = torch.zeros((num_envs, num_nodes), dtype=torch.bool, device=device)
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.contacts_disabled = torch.zeros(num_envs, dtype=torch.bool, device=device)
        # squeeze progress alpha of each environment, from 0 at the grasp to 1
        self.squeeze_progress = torch.zeros(num_envs, device=device)

    @property
    def num_grasped(self) -> torch.Tensor:
        """Number of nodes locked to the EE frame. Shape is (num_envs,)."""
        return (self.grasped & self.active.unsqueeze(1)).sum(dim=1)

    def select(self) -> torch.Tensor:
        """Return the nodes a grasp would lock now, see the class docstring. Shape is (num_envs, num_nodes)."""
        nodal_pos_w = self._cloth.data.nodal_pos_w.torch
        selected = pad_box_mask(self._env, self._command_cfg, nodal_pos_w)
        for cells in (self._edges, self._triangles):
            inside = pad_box_mask(self._env, self._command_cfg, nodal_pos_w[:, cells].mean(dim=2))
            # every node of a cell whose center is inside the box
            hits = torch.zeros_like(nodal_pos_w[..., 0])
            hits.index_add_(1, cells.flatten(), inside.float().repeat_interleave(cells.shape[1], dim=1))
            selected |= hits > 0
        return selected

    def activate(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Lock the selected nodes of the given environments to the EE frame.

        Environments with fewer than the command's ``grasp_min_nodes`` selected nodes grasp nothing and keep colliding
        with the cloth.

        Args:
            env_ids: Environments that grasp now.

        Returns:
            The environments that locked cloth nodes.
        """
        selected = self.select()[env_ids]
        holding = selected.sum(dim=1) >= max(self._command_cfg.grasp_min_nodes, 1)
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
        """Release the locked nodes of the given environments.

        The gripper keeps passing through the cloth until :meth:`restore_contacts` is called.
        """
        self.active[env_ids] = False
        self.grasped[env_ids] = False

    def restore_contacts(self, env_ids: torch.Tensor):
        """Let the gripper collide with the cloth again, once it has left the cloth."""
        env_ids = env_ids[self.contacts_disabled[env_ids]]
        if len(env_ids) > 0 and self._set_gripper_contacts is not None:
            self._set_gripper_contacts(env_ids.tolist(), enabled=True)
            self.contacts_disabled[env_ids] = False

    def advance_squeeze(self, pressing: torch.Tensor):
        """Ramp the squeeze of the holding environments in or out by one step.

        Args:
            pressing: Whether each environment presses the fold, else the squeeze ramps back out, shape [num_envs].
        """
        step = torch.where(pressing, self._squeeze_rate, -self._squeeze_rate)
        self.squeeze_progress = torch.where(
            self.active, (self.squeeze_progress + step).clamp(0.0, 1.0), self.squeeze_progress
        )

    def kinematic_targets(self) -> torch.Tensor:
        """Return nodal kinematic targets that lock the grasped nodes to the EE frame at its current pose.

        Returns:
            Target positions [m] and flags (0 locked, 1 free) of all nodes, shape [num_envs, num_nodes, 4].
        """
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
        """EE positions [m], shape [num_envs, 3], and orientations (x, y, z, w), shape [num_envs, 4], in the world
        frame."""
        return self._ee_frame.data.target_pos_w.torch[:, 0], self._ee_frame.data.target_quat_w.torch[:, 0]

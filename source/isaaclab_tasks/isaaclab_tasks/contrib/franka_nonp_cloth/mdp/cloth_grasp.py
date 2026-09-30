# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cloth grasp that locks the cloth vertices pinched by the Franka gripper to the gripper."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Gripper anchors along each finger, as offsets from the fingertip point (0.1034 m below the hand) toward the
# fingertips [m]. A Franka finger runs from its joint 0.0584 m below the hand to its tip at about 0.112 m, so
# its middle lies 1.8 cm above the fingertip point and its tip 0.7 cm below it.
FINGER_ANCHOR_DEPTHS = {"middle": -0.018, "tip": 0.007}
# cloth vertices within this distance of a gripper anchor are grasped when the gripper closes [m]
GRASP_PROXIMITY = 0.015


class GripperConstraint:
    """Lock grasped cloth vertices to the gripper with position-level bilateral constraints.

    This follows the cloth gripping model of FLASH. When the gripper closes, every cloth vertex within
    :attr:`proximity` of one of the gripper's anchors joins the grasped set. While the constraint is active, each
    grasped vertex is driven as a kinematic node at the gripper point where it was grasped, which enforces the
    equality constraint ``x_i - p_grip,i(t) = 0``, a zero-length rigid link. Opening the gripper deactivates the
    constraint and returns the vertices to the cloth's internal elastic forces and gravity.

    The gripper points move rigidly with the hand. Locking each vertex to its own grasp point, instead of pulling
    all grasped vertices onto the anchors themselves, keeps the grasped patch intact: collapsing up to 1.5 cm of
    cloth onto a few points degenerates its triangles, and VBD's elastic forces then blow up.

    While the constraint holds cloth, the gripper stops colliding with the cloth in that environment, as the
    virtual gripper of FLASH does: the constraint is then its whole interaction with the grasped cloth. The
    fingers would otherwise press against the immovable locked vertices, and that contact destabilizes the
    coupling between the robot and the cloth solver. The caller should hold the fingers at the opening they closed
    to on the fold, so they keep resting on it instead of closing through it. The gripper collides with the cloth
    again only once :meth:`restore_contacts` is called after it has left the cloth; turning the contacts on with the
    fingers inside the cloth blows up the coupling the same way.

    Each finger carries one anchor per entry of :data:`FINGER_ANCHOR_DEPTHS`, at its middle and at its tip, on the
    inner pad surface, so the grasped vertices are the cloth pinched against the pads along both fingers and the
    grasped cloth cannot rotate about the grasp.

    When the constraint is disabled, the grasped set is still recorded, to report how far the cloth slips in a
    friction-only grasp.
    """

    def __init__(self, env: ManagerBasedEnv, enabled: bool, particle_radius: float, proximity: float = GRASP_PROXIMITY):
        """Initialize the constraint.

        Args:
            env: The environment, with a ``robot`` Franka and a ``deformable`` cloth in its scene.
            enabled: Whether to enforce the constraint on the grasped vertices.
            particle_radius: Collision radius of the cloth vertices [m].
            proximity: Distance from an anchor within which a vertex is grasped [m].
        """
        self.enabled = enabled
        self.proximity = proximity
        self.particle_radius = particle_radius
        # the Newton coupler manager of the task can switch the gripper's cloth contacts per environment
        self._set_gripper_contacts = getattr(env.sim.physics_manager, "set_proxy_particle_contacts", None)
        self._robot = env.scene["robot"]
        self._cloth = env.scene["deformable"]
        self._finger_ids, _ = self._robot.find_bodies(["panda_leftfinger", "panda_rightfinger"], preserve_order=True)
        self._finger_joint_ids, _ = self._robot.find_joints(
            ["panda_finger_joint1", "panda_finger_joint2"], preserve_order=True
        )
        self._hand_id = self._robot.find_bodies("panda_hand")[0][0]

        num_envs, device = env.num_envs, env.device
        # one anchor per finger and depth: left middle, left tip, right middle, right tip
        depths = torch.tensor(list(FINGER_ANCHOR_DEPTHS.values()), device=device)
        self.anchor_finger = torch.arange(2, device=device).repeat_interleave(len(depths))
        self.anchor_depth = depths.repeat(2)
        num_vertices = self._cloth.max_sim_vertices_per_body
        # grasp point of each vertex in the hand frame [m]
        self.grasp_offsets = torch.zeros((num_envs, num_vertices, 3), device=device)
        self.grasped = torch.zeros((num_envs, num_vertices), dtype=torch.bool, device=device)
        self.grasp_vertex_pos = torch.zeros((num_envs, num_vertices, 3), device=device)
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.contacts_disabled = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def activate(self, env_ids: torch.Tensor, tcp_pos_w: torch.Tensor):
        """Grasp the cloth vertices near the gripper anchors.

        The gripper stops colliding with the cloth only in environments that grasped at least one vertex.

        Args:
            env_ids: Environments that just closed the gripper.
            tcp_pos_w: World position of the fingertip point between the fingers [m], shape [E, 3].
        """
        hand = self._robot.data.body_link_pose_w.torch[env_ids, self._hand_id]
        hand_pos, hand_quat = hand[:, :3], hand[:, 3:7]
        # the fingers close along the hand's y axis and point along its z axis
        hand_axes = torch.eye(3, device=hand.device)
        closing_axis = math_utils.quat_apply(hand_quat, hand_axes[1].expand_as(tcp_pos_w))
        approach_axis = math_utils.quat_apply(hand_quat, hand_axes[2].expand_as(tcp_pos_w))
        # each finger's inner pad surface lies its joint opening away from the fingertip point, on its side
        finger_pos = self._robot.data.body_link_pose_w.torch[env_ids][:, self._finger_ids, :3]
        side = torch.sign(((finger_pos - tcp_pos_w.unsqueeze(1)) * closing_axis.unsqueeze(1)).sum(dim=-1))
        opening = self._robot.data.joint_pos.torch[env_ids][:, self._finger_joint_ids]
        lateral = side * (opening - self.particle_radius).clamp_min(0.0)
        anchors = (
            tcp_pos_w.unsqueeze(1)
            + lateral[:, self.anchor_finger].unsqueeze(-1) * closing_axis.unsqueeze(1)
            + self.anchor_depth.view(1, -1, 1) * approach_axis.unsqueeze(1)
        )

        # grasp every vertex within the proximity threshold of an anchor, at its current position on the gripper
        vertices = self._cloth.data.nodal_pos_w.torch[env_ids]
        distance = (vertices.unsqueeze(1) - anchors.unsqueeze(2)).norm(dim=-1).amin(dim=1)
        num_vertices = vertices.shape[1]
        self.grasped[env_ids] = distance <= self.proximity
        self.grasp_offsets[env_ids] = math_utils.quat_apply_inverse(
            hand_quat.unsqueeze(1).expand(-1, num_vertices, -1), vertices - hand_pos.unsqueeze(1)
        )
        self.grasp_vertex_pos[env_ids] = vertices
        self.active[env_ids] = True
        if self.enabled and self._set_gripper_contacts is not None:
            # a gripper that closed on no cloth keeps colliding with it
            holding_ids = env_ids[self.grasped[env_ids].any(dim=1)]
            if len(holding_ids) > 0:
                self._set_gripper_contacts(holding_ids.tolist(), enabled=False)
                self.contacts_disabled[holding_ids] = True

    def deactivate(self, env_ids: torch.Tensor):
        """Release the grasped vertices of the given environments.

        The gripper keeps passing through the cloth until :meth:`restore_contacts` is called.
        """
        self.active[env_ids] = False

    def restore_contacts(self, env_ids: torch.Tensor):
        """Let the gripper collide with the cloth again, once it has left the cloth."""
        env_ids = env_ids[self.contacts_disabled[env_ids]]
        if len(env_ids) > 0 and self._set_gripper_contacts is not None:
            self._set_gripper_contacts(env_ids.tolist(), enabled=True)
            self.contacts_disabled[env_ids] = False

    def grasp_points(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Return the current world position of each vertex's grasp point on the gripper [m], shape [E, P, 3]."""
        hand = self._robot.data.body_link_pose_w.torch[env_ids, self._hand_id]
        num_vertices = self.grasp_offsets.shape[1]
        hand_quat = hand[:, 3:7].unsqueeze(1).expand(-1, num_vertices, -1)
        return hand[:, :3].unsqueeze(1) + math_utils.quat_apply(hand_quat, self.grasp_offsets[env_ids])

    def slip(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Return the mean distance of the grasped vertices from their grasp points [m], shape [E]."""
        grasped = self.grasped[env_ids]
        vertices = self._cloth.data.nodal_pos_w.torch[env_ids]
        distance = (vertices - self.grasp_points(env_ids)).norm(dim=-1)
        return (distance * grasped).sum(dim=1) / grasped.sum(dim=1).clamp_min(1)

    def kinematic_targets(self) -> torch.Tensor:
        """Return nodal kinematic targets that lock the grasped vertices to the gripper, shape [N, P, 4]."""
        vertices = self._cloth.data.nodal_pos_w.torch
        targets = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
        locked = self.grasped & self.active.unsqueeze(1)
        if locked.any():
            env_ids = torch.arange(vertices.shape[0], device=vertices.device)
            targets[..., :3] = torch.where(locked.unsqueeze(-1), self.grasp_points(env_ids), vertices)
            targets[..., 3] = torch.where(locked, 0.0, 1.0)
        return targets

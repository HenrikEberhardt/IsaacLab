# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task stages of the Franka cloth dragging task and the state checks that define them."""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from ..constrained_gripper.gripper_state import gripper_cfg
from .frames import cloth_center_pos_b, cloth_center_vel_b, points_to_base, raised_node_ids, vectors_to_base

# without raised nodes, grasp the node this far from the central node towards the target [m]
FALLBACK_GRASP_OFFSET = 0.06

if TYPE_CHECKING:
    from isaaclab.assets import DeformableObject, RigidObject
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.sensors import FrameTransformer

    from .commands_cfg import ClothTargetCommandCfg


class ClothDragStage(IntEnum):
    """Chronological stages of the cloth dragging task"""

    APPROACH = 0
    """The EE frame origin is away from the cloth."""
    GRASP = 1
    """The EE frame origin is near the grasp node, see :attr:`~.commands.ClothTargetCommand.grasp_node_id`."""
    DRAG = 2
    """The gripper holds the cloth by locked nodes that include a raised node, see :func:`gripper_holds_raised_node`."""
    GOAL = 3
    """The gripper holds the cloth by a raised node and its central node is near the target."""


START_STAGE_NAMES = ("approach", "grasp", "drag", "goal")
"""Names of the stages an episode can start in, in :class:`ClothDragStage` order."""

START_STAGE_ALIASES = {"default": "approach"}
"""Legacy input names mapped to their canonical start-stage names."""


def canonical_start_stage_name(name: str) -> str:
    """Return the canonical start-stage name, accepting legacy input aliases.

    Args:
        name: Start-stage name. ``default`` is accepted as a compatibility alias for ``approach``.

    Returns:
        The canonical name. Unknown names are returned unchanged for the caller to validate.
    """
    return START_STAGE_ALIASES.get(name, name)


def ee_cloth_distance(env: ManagerBasedEnv, cfg: ClothTargetCommandCfg, height_offset: float = 0.0) -> torch.Tensor:
    """Distance of the EE frame origin to the nearest cloth node [m]. Shape is (num_envs,)."""
    cloth: DeformableObject = env.scene[cfg.object_name]
    ee_frame: FrameTransformer = env.scene[cfg.ee_frame_name]
    tip_w = ee_frame.data.target_pos_w.torch[:, 0].clone()
    tip_w[:, 2] -= height_offset
    return torch.linalg.norm(cloth.data.nodal_pos_w.torch - tip_w.unsqueeze(1), dim=2).min(dim=1).values


def grasp_node_ids(
    nodes_b: torch.Tensor,
    center_b: torch.Tensor,
    target_b: torch.Tensor,
    candidate_ids: torch.Tensor | None,
) -> torch.Tensor:
    """Return the cloth node to grasp: the candidate closest to the segment from the central node to the target.

    Without candidates (no raised nodes), the node closest to the point :data:`FALLBACK_GRASP_OFFSET` from the central
    node towards the target; the squeeze creases the flat cloth there.

    Args:
        nodes_b: Cloth node positions in the robot base frame [m], shape [N, P, 3].
        center_b: Position of the cloth's central node in the robot base frame [m], shape [N, 3].
        target_b: Target of the central node in the robot base frame [m], shape [N, 3].
        candidate_ids: Ids of the candidate nodes, shape [N, K], or None.

    Returns:
        The node ids, shape [N].
    """
    rows = torch.arange(len(nodes_b), device=nodes_b.device)
    to_target = (target_b - center_b)[:, :2]
    if candidate_ids is None or candidate_ids.shape[1] == 0:
        point = center_b[:, :2] + FALLBACK_GRASP_OFFSET * torch.nn.functional.normalize(to_target, dim=-1)
        return (nodes_b[..., :2] - point.unsqueeze(1)).norm(dim=-1).argmin(dim=1)
    rel = nodes_b[rows.unsqueeze(1), candidate_ids, :2] - center_b[:, None, :2]
    t = (rel * to_target.unsqueeze(1)).sum(dim=-1) / to_target.square().sum(dim=-1, keepdim=True).clamp(min=1e-12)
    distance = (rel - t.clamp(0.0, 1.0).unsqueeze(-1) * to_target.unsqueeze(1)).norm(dim=-1)
    return candidate_ids[rows, distance.argmin(dim=1)]


def grasp_node_pos_w(env: ManagerBasedEnv, cfg: ClothTargetCommandCfg, grasp_node: torch.Tensor) -> torch.Tensor:
    """World position of the grasp node of every environment [m], shape [num_envs, 3].

    Args:
        env: The environment.
        cfg: The cloth target command configuration.
        grasp_node: Grasp node id of every environment (:attr:`~.commands.ClothTargetCommand.grasp_node_id`),
            shape [num_envs].
    """
    cloth: DeformableObject = env.scene[cfg.object_name]
    return cloth.data.nodal_pos_w.torch.gather(1, grasp_node.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)


def ee_grasp_node_distance(
    env: ManagerBasedEnv,
    cfg: ClothTargetCommandCfg,
    grasp_node: torch.Tensor,
    target_height: float | None = None,
) -> torch.Tensor:
    """Distance of the EE frame origin to a point at the grasp node [m]. Shape is (num_envs,).

    The point lies at the grasp node, or with ``target_height`` straight above or below it at that height above the
    table top, so it does not move with the height of the deforming cloth.

    Args:
        env: The environment.
        cfg: The cloth target command configuration.
        grasp_node: Grasp node id of every environment, shape [num_envs].
        target_height: Height of the point above the table top [m], or None for the node's own height.
    """
    point_w = grasp_node_pos_w(env, cfg, grasp_node)
    if target_height is not None:
        point_w[:, 2] = env.scene.env_origins[:, 2] + target_height
    ee_frame: FrameTransformer = env.scene[cfg.ee_frame_name]
    return torch.linalg.norm(point_w - ee_frame.data.target_pos_w.torch[:, 0], dim=1)


def gripper_holds_raised_node(env: ManagerBasedEnv, cfg: ClothTargetCommandCfg) -> torch.Tensor:
    """Whether the gripper holds the cloth with a raised node among the nodes locked to the EE frame.

    A grasp that locks only other nodes, such as a pinch of the cloth's edge, does not count. Without raised nodes
    (see :attr:`ClothTargetCommandCfg.raised_nodes_event`), this is
    :func:`~..constrained_gripper.gripper_state.gripper_holds_cloth`. Shape is (num_envs,).
    """
    term = env.action_manager.get_term(cfg.gripper_action_name)
    ids = raised_node_ids(env, cfg.raised_nodes_event)
    if ids is None:
        return term.is_grasping
    return term.node_lock.active & term.node_lock.grasped.gather(1, ids).any(dim=1)


def grasp_node_capturable(
    env: ManagerBasedEnv, cfg: ClothTargetCommandCfg, grasp_node: torch.Tensor, require_min_nodes: bool = True
) -> torch.Tensor:
    """Whether closing now captures the grasp node. Shape is (num_envs,).

    The lock selector (:meth:`~..constrained_gripper.constraint_node_lock.ClothNodeLock.select`) picks the grasp node,
    so it or a mesh edge or triangle through it lies between the pads, and with ``require_min_nodes`` at least
    :attr:`~..constrained_gripper.gripper_action_cfg.ConstrainedGripperActionCfg.grasp_min_nodes` nodes, so the lock
    forms; while a lock is active, whether the locked nodes include the grasp node.

    Args:
        env: The environment.
        cfg: The cloth target command configuration.
        grasp_node: Grasp node id of every environment, shape [num_envs].
        require_min_nodes: Whether the selection must also reach the lock's minimum node count.
    """
    node_lock = env.action_manager.get_term(cfg.gripper_action_name).node_lock
    node = grasp_node.unsqueeze(1)
    selected = node_lock.select()
    ready = selected.gather(1, node).squeeze(1)
    if require_min_nodes:
        ready &= selected.sum(dim=1) >= max(gripper_cfg(env, cfg).grasp_min_nodes, 1)
    return torch.where(node_lock.active, node_lock.grasped.gather(1, node).squeeze(1), ready)


def goal_point_pos_b(env: ManagerBasedEnv, cfg: ClothTargetCommandCfg) -> torch.Tensor:
    """Position of the point that must reach the target in the robot's root frame [m], shape [num_envs, 3].

    The root of :attr:`ClothTargetCommandCfg.goal_object_name` if set, else the cloth's central node.
    """
    if cfg.goal_object_name is None:
        return cloth_center_pos_b(env, SceneEntityCfg(cfg.object_name), SceneEntityCfg(cfg.asset_name))
    goal_object: RigidObject = env.scene[cfg.goal_object_name]
    return points_to_base(env, goal_object.data.root_pos_w.torch, SceneEntityCfg(cfg.asset_name))


def goal_point_vel_b(env: ManagerBasedEnv, cfg: ClothTargetCommandCfg) -> torch.Tensor:
    """Velocity of the point of :func:`goal_point_pos_b` in the robot's root frame [m/s], shape [num_envs, 3]."""
    if cfg.goal_object_name is None:
        return cloth_center_vel_b(env, SceneEntityCfg(cfg.object_name), SceneEntityCfg(cfg.asset_name))
    goal_object: RigidObject = env.scene[cfg.goal_object_name]
    return vectors_to_base(env, goal_object.data.root_lin_vel_w.torch, SceneEntityCfg(cfg.asset_name))


def cloth_goal_xy_distance(
    env: ManagerBasedEnv, target_pos_b: torch.Tensor, cfg: ClothTargetCommandCfg
) -> torch.Tensor:
    """Planar distance between the goal point (:func:`goal_point_pos_b`) and the target [m]."""
    goal_pos_b = goal_point_pos_b(env, cfg)
    return torch.linalg.norm(target_pos_b[:, :2] - goal_pos_b[:, :2], dim=1)

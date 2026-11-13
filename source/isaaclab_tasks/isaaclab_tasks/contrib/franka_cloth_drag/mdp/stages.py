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
from isaaclab.utils.math import quat_apply_inverse

from .frames import cloth_center_pos_b, cloth_center_vel_b, points_to_base, raised_node_ids, vectors_to_base

# without raised nodes, grasp the node this far from the central node towards the target [m]
FALLBACK_GRASP_OFFSET = 0.06

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, DeformableObject, RigidObject
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


def gripper_closing(env: ManagerBasedEnv, cfg: ClothTargetCommandCfg) -> torch.Tensor:
    """Whether the binary gripper action commands closing. Shape is (num_envs,)."""
    term = env.action_manager.get_term(cfg.gripper_action_name)
    if hasattr(term, "closed"):
        return term.closed
    return torch.any(term.raw_actions < 0.0, dim=1)


def finger_opening(env: ManagerBasedEnv, cfg: ClothTargetCommandCfg) -> torch.Tensor:
    """Measured half the distance [m] between the finger joint positions."""
    robot: Articulation = env.scene[cfg.asset_name]
    finger_ids, _ = robot.find_joints(cfg.finger_joint_name)
    return robot.data.joint_pos.torch[:, finger_ids[0]]


def gripper_closure(env: ManagerBasedEnv, cfg: ClothTargetCommandCfg) -> torch.Tensor:
    """Measured closure of the gripper, zero at its open command and one at its close command. Shape is (num_envs,)."""
    action_cfg = env.action_manager.get_term(cfg.gripper_action_name).cfg
    open_opening = action_cfg.open_command_expr[cfg.finger_joint_name]
    close_opening = action_cfg.close_command_expr[cfg.finger_joint_name]
    return ((open_opening - finger_opening(env, cfg)) / (open_opening - close_opening)).clamp(0.0, 1.0)


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
    choice: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the cloth node to grasp: the candidate closest to the segment from the central node to the target.

    With ``choice``, the chosen candidate of every environment instead. Without candidates (no raised nodes), the node
    closest to the point :data:`FALLBACK_GRASP_OFFSET` from the central node towards the target; the squeeze creases
    the flat cloth there.

    Args:
        nodes_b: Cloth node positions in the robot base frame [m], shape [N, P, 3].
        center_b: Position of the cloth's central node in the robot base frame [m], shape [N, 3].
        target_b: Target of the central node in the robot base frame [m], shape [N, 3].
        candidate_ids: Ids of the candidate nodes, shape [N, K], or None.
        choice: Index of the candidate to grasp in every environment, taken modulo K, shape [N], or None.

    Returns:
        The node ids, shape [N].
    """
    rows = torch.arange(len(nodes_b), device=nodes_b.device)
    to_target = (target_b - center_b)[:, :2]
    if candidate_ids is None or candidate_ids.shape[1] == 0:
        point = center_b[:, :2] + FALLBACK_GRASP_OFFSET * torch.nn.functional.normalize(to_target, dim=-1)
        return (nodes_b[..., :2] - point.unsqueeze(1)).norm(dim=-1).argmin(dim=1)
    if choice is not None:
        return candidate_ids[rows, choice % candidate_ids.shape[1]]
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


def pad_box_mask(env: ManagerBasedEnv, cfg: ClothTargetCommandCfg, points_w: torch.Tensor) -> torch.Tensor:
    """Whether each point lies between the finger pads. Shape is (num_envs, num_points)."""
    ee_frame: FrameTransformer = env.scene[cfg.ee_frame_name]
    ee_pos_w = ee_frame.data.target_pos_w.torch[:, 0]
    ee_quat_w = ee_frame.data.target_quat_w.torch[:, 0].unsqueeze(1).expand(-1, points_w.shape[1], -1)
    # points relative to the EE point, in the EE frame
    rel_ee = quat_apply_inverse(ee_quat_w, points_w - ee_pos_w.unsqueeze(1))
    half_gap = finger_opening(env, cfg).unsqueeze(1) + cfg.finger_gap_margin
    return (
        (rel_ee[..., 0].abs() <= cfg.finger_pad_half_width)
        & (rel_ee[..., 1].abs() <= half_gap)
        & (rel_ee[..., 2].abs() <= cfg.finger_pad_half_length)
    )


def nodes_between_fingers(env: ManagerBasedEnv, cfg: ClothTargetCommandCfg) -> torch.Tensor:
    """Number of cloth nodes between the finger pads."""
    cloth: DeformableObject = env.scene[cfg.object_name]
    return pad_box_mask(env, cfg, cloth.data.nodal_pos_w.torch).sum(dim=1)


def nodes_in_grasp(env: ManagerBasedEnv, cfg: ClothTargetCommandCfg) -> torch.Tensor:
    """Number of cloth nodes a grasp would lock now. Shape is (num_envs,)."""
    constraint = getattr(env.action_manager.get_term(cfg.gripper_action_name), "constraint", None)
    if constraint is None:
        return nodes_between_fingers(env, cfg)
    return constraint.select().sum(dim=1)


def gripper_holds_cloth(env: ManagerBasedEnv, cfg: ClothTargetCommandCfg) -> torch.Tensor:
    """Whether the gripper holds the cloth with constraints or friction."""
    return env.action_manager.get_term(cfg.gripper_action_name).is_grasping


def gripper_holds_raised_node(env: ManagerBasedEnv, cfg: ClothTargetCommandCfg) -> torch.Tensor:
    """Whether the gripper holds the cloth with a raised node among the nodes locked to the EE frame.

    A grasp that locks only other nodes, such as a pinch of the cloth's edge, does not count. Without raised nodes
    (see :attr:`ClothTargetCommandCfg.raised_nodes_event`) or without the lock constraint, this is
    :func:`gripper_holds_cloth`. Shape is (num_envs,).
    """
    term = env.action_manager.get_term(cfg.gripper_action_name)
    ids = raised_node_ids(env, cfg.raised_nodes_event)
    constraint = getattr(term, "constraint", None)
    if ids is None or constraint is None:
        return term.is_grasping
    return constraint.active & constraint.grasped.gather(1, ids).any(dim=1)


def grasp_node_capturable(
    env: ManagerBasedEnv, cfg: ClothTargetCommandCfg, grasp_node: torch.Tensor, require_min_nodes: bool = True
) -> torch.Tensor:
    """Whether closing now captures the grasp node. Shape is (num_envs,).

    With the lock constraint: the attachment selector (:meth:`~.cloth_grasp.ClothGraspConstraint.select`) picks the
    grasp node, so it or a mesh edge or triangle through it lies between the pads, and with ``require_min_nodes`` at
    least :attr:`ClothTargetCommandCfg.grasp_min_nodes` nodes, so the lock forms; while a lock is active, whether the
    locked nodes include the grasp node. Without the constraint: the grasp node lies between the pads.

    Args:
        env: The environment.
        cfg: The cloth target command configuration.
        grasp_node: Grasp node id of every environment, shape [num_envs].
        require_min_nodes: Whether the selection must also reach the lock's minimum node count.
    """
    constraint = getattr(env.action_manager.get_term(cfg.gripper_action_name), "constraint", None)
    if constraint is None:
        return pad_box_mask(env, cfg, grasp_node_pos_w(env, cfg, grasp_node).unsqueeze(1)).squeeze(1)
    node = grasp_node.unsqueeze(1)
    selected = constraint.select()
    ready = selected.gather(1, node).squeeze(1)
    if require_min_nodes:
        ready &= selected.sum(dim=1) >= max(cfg.grasp_min_nodes, 1)
    return torch.where(constraint.active, constraint.grasped.gather(1, node).squeeze(1), ready)


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

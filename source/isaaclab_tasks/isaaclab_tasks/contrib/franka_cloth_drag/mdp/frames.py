# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Frame and cloth geometry helpers of the Franka cloth dragging task.

Task quantities are expressed in one of two frames: the robot's root frame (suffix ``_b``) or the end-effector (EE)
frame, the first target of the ``ee_frame`` frame transformer, centered between the finger pads.

The cloth's reference point is its central node: the landmark at the center of the rest mesh, see
:func:`cloth_landmark_ids`. A material point stays well defined when the cloth crumples, unlike a center of mass.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, DeformableObject
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.sensors import FrameTransformer


def top_down_yaw(quat: torch.Tensor) -> torch.Tensor:
    """Return the heading of the hand's x-axis about the z-axis [rad], shape [N].

    For a hand pointing straight down and turned by ``yaw`` about the z-axis this recovers ``yaw``, and it stays
    defined for a tilted hand.

    Args:
        quat: Hand orientations (x, y, z, w), shape [N, 4].
    """
    x_axis = quat_apply(quat, torch.tensor((1.0, 0.0, 0.0), device=quat.device).expand(len(quat), 3))
    return torch.atan2(x_axis[:, 1], x_axis[:, 0])


def points_to_base(
    env: ManagerBasedEnv, points_w: torch.Tensor, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Express world positions in the robot's root frame.

    Args:
        env: The environment.
        points_w: Positions in the world frame [m], shape [num_envs, ..., 3].
        robot_cfg: The robot whose root frame is used.

    Returns:
        Positions in the robot's root frame [m], same shape as ``points_w``.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    points = points_w.reshape(len(points_w), -1, 3)
    num_points = points.shape[1]
    root_pos_w = robot.data.root_pos_w.torch.unsqueeze(1).expand(-1, num_points, -1)
    root_quat_w = robot.data.root_quat_w.torch.unsqueeze(1).expand(-1, num_points, -1)
    points_b, _ = subtract_frame_transforms(
        root_pos_w.reshape(-1, 3), root_quat_w.reshape(-1, 4), points.reshape(-1, 3)
    )
    return points_b.view(points_w.shape)


def vectors_to_base(
    env: ManagerBasedEnv, vectors_w: torch.Tensor, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Rotate world vectors, such as velocities, into the robot's root frame.

    Args:
        env: The environment.
        vectors_w: Vectors in the world frame, shape [num_envs, ..., 3].
        robot_cfg: The robot whose root frame is used.

    Returns:
        Vectors in the robot's root frame, same shape and unit as ``vectors_w``.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    vectors = vectors_w.reshape(len(vectors_w), -1, 3)
    root_quat_w = robot.data.root_quat_w.torch.unsqueeze(1).expand(-1, vectors.shape[1], -1)
    return quat_apply_inverse(root_quat_w, vectors).view(vectors_w.shape)


CLOTH_CENTER_LANDMARK = 4
"""Index of the central node among the landmarks of :func:`cloth_landmark_ids`."""


@functools.cache
def _landmark_ids(asset: DeformableObject) -> torch.Tensor:
    """Nodes nearest to a 3 x 3 grid over the xy bounding box of the rest mesh, shape [9]."""
    rest_xy = asset.data.default_nodal_state_w.torch[0, :, :2]
    lower, upper = rest_xy.amin(dim=0), rest_xy.amax(dim=0)
    fractions = rest_xy.new_tensor((0.0, 0.5, 1.0))
    targets = lower + torch.cartesian_prod(fractions, fractions) * (upper - lower)
    return torch.cdist(targets, rest_xy).argmin(dim=1)


@functools.cache
def _rest_triangles(asset: DeformableObject) -> torch.Tensor:
    """Node ids of the cloth's triangles, read from its USD file, shape [num_triangles, 3]."""
    # imported here: pxr must not load before the simulation app starts, and this module is imported with the env cfg
    from pxr import Usd, UsdGeom

    usd_path = getattr(asset.cfg.spawn, "usd_path", None)
    if usd_path is None:
        raise ValueError("The cloth triangles need a cloth spawned from a USD file (spawn.usd_path).")
    stage = Usd.Stage.Open(usd_path)
    mesh = next((UsdGeom.Mesh(prim) for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)), None)
    if mesh is None:
        raise ValueError(f"No mesh found in '{usd_path}'.")
    if any(count != 3 for count in mesh.GetFaceVertexCountsAttr().Get()):
        raise ValueError(f"The cloth mesh in '{usd_path}' must consist of triangles only.")
    rest_pos = asset.data.default_nodal_state_w.torch[0, :, :3]
    if len(mesh.GetPointsAttr().Get()) != len(rest_pos):
        raise ValueError(f"The cloth mesh in '{usd_path}' does not match the {len(rest_pos)} simulated nodes.")
    return torch.tensor(list(mesh.GetFaceVertexIndicesAttr().Get()), device=rest_pos.device).view(-1, 3)


@functools.cache
def _rest_edges(asset: DeformableObject) -> torch.Tensor:
    """Node ids of the cloth's unique edges, lower id first, shape [num_edges, 2]."""
    faces = _rest_triangles(asset)
    edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return torch.unique(edges.sort(dim=1).values, dim=0)


@functools.cache
def _rest_node_areas(asset: DeformableObject) -> torch.Tensor:
    """Cloth area around each node of the rest mesh [m^2], a third of each adjacent triangle, shape [num_nodes]."""
    rest_pos = asset.data.default_nodal_state_w.torch[0, :, :3]
    faces = _rest_triangles(asset)
    corners = rest_pos[faces]
    triangle_areas = 0.5 * torch.linalg.norm(
        torch.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0], dim=-1), dim=-1
    )
    return torch.zeros(len(rest_pos), device=rest_pos.device).index_add_(
        0, faces.flatten(), (triangle_areas / 3.0).repeat_interleave(3)
    )


def cloth_landmark_ids(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable")) -> torch.Tensor:
    """Ids of the 9 landmark nodes of the cloth, shape [9].

    The landmarks are the nodes nearest to a 3 x 3 grid over the xy bounding box of the rest mesh: the corners, the
    edge midpoints and the center, ordered by the grid fractions (0, 0.5, 1) along x first. Index
    :data:`CLOTH_CENTER_LANDMARK` is the central node. The ids are the same for every environment and episode.
    """
    return _landmark_ids(env.scene[asset_cfg.name])


def cloth_node_areas(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable")) -> torch.Tensor:
    """Cloth area around each node of the rest mesh [m^2], shape [num_nodes].

    Each node gets a third of the area of every adjacent triangle, the way Newton distributes the cloth mass. The
    triangles are read from the cloth's USD file.
    """
    return _rest_node_areas(env.scene[asset_cfg.name])


def cloth_triangles(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable")) -> torch.Tensor:
    """Node ids of the cloth's triangles, read from its USD file, shape [num_triangles, 3]."""
    return _rest_triangles(env.scene[asset_cfg.name])


def cloth_edges(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable")) -> torch.Tensor:
    """Node ids of the cloth's unique edges, lower id first, shape [num_edges, 2]."""
    return _rest_edges(env.scene[asset_cfg.name])


def raised_node_ids(env: ManagerBasedEnv, event_name: str | None = "reset_deformable") -> torch.Tensor | None:
    """Ids of the cloth nodes raised in the current episode, shape [num_envs, K], or None.

    The ids come from the reset event term ``event_name`` (:class:`~.events.ResetClothWithRaisedNodes`), which
    redraws them, sorted, at every reset. Reset events run before the observation and command resets, so those see
    the ids of the new episode.

    Args:
        env: The environment.
        event_name: Name of the reset event term that raises the nodes. None => no raised nodes.

    Returns:
        The live tensor of node ids, or None if the term is missing, raises no nodes, or ``event_name`` is None.
    """
    if event_name is None:
        return None
    try:
        term = env.event_manager.get_term_cfg(event_name).func
    except ValueError:
        return None
    ids = getattr(term, "raised_node_ids", None)
    if ids is None or ids.shape[1] == 0:
        return None
    return ids


def cloth_center_pos_b(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position of the cloth's central node in the robot's root frame [m], shape [num_envs, 3].

    Every term of the task that needs the cloth's position uses this function.
    """
    asset: DeformableObject = env.scene[asset_cfg.name]
    center_id = _landmark_ids(asset)[CLOTH_CENTER_LANDMARK]
    pos_b = points_to_base(env, asset.data.nodal_pos_w.torch[:, center_id], robot_cfg)
    return torch.nan_to_num(pos_b, nan=0.0, posinf=0.0, neginf=0.0)


def cloth_center_vel_b(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Velocity of the cloth's central node in the robot's root frame [m/s], shape [num_envs, 3]."""
    asset: DeformableObject = env.scene[asset_cfg.name]
    center_id = _landmark_ids(asset)[CLOTH_CENTER_LANDMARK]
    vel_b = vectors_to_base(env, asset.data.nodal_vel_w.torch[:, center_id], robot_cfg)
    return torch.nan_to_num(vel_b, nan=0.0, posinf=0.0, neginf=0.0)


def ee_pose_in_base(
    env: ManagerBasedEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """End-effector pose in the robot's root frame.

    The end effector (EE) is the first target of the frame transformer, the point centered between the finger pads.

    Returns:
        Position [m], shape [num_envs, 3], and orientation (x, y, z, w), shape [num_envs, 4].
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return subtract_frame_transforms(
        robot.data.root_pos_w.torch,
        robot.data.root_quat_w.torch,
        ee_frame.data.target_pos_w.torch[:, 0],
        ee_frame.data.target_quat_w.torch[:, 0],
    )


def ee_twist_in_base(
    env: ManagerBasedEnv, hand_cfg: SceneEntityCfg, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear velocity of the end effector and angular velocity of the hand in the robot's root frame.

    The frame transformer has no velocities, so the end-effector velocity is the hand body velocity moved along the
    lever arm from the hand body to the end effector.

    Args:
        env: The environment.
        hand_cfg: The robot, with ``body_names`` selecting the hand body that carries the end-effector frame.
        ee_frame_cfg: The frame transformer whose first target is the end effector.

    Returns:
        Linear velocity [m/s], shape [num_envs, 3], and angular velocity [rad/s], shape [num_envs, 3].
    """
    robot: Articulation = env.scene[hand_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    hand_id = hand_cfg.body_ids[0]
    ang_vel_w = robot.data.body_link_ang_vel_w.torch[:, hand_id]
    lever_w = ee_frame.data.target_pos_w.torch[:, 0] - robot.data.body_link_pos_w.torch[:, hand_id]
    lin_vel_w = robot.data.body_link_lin_vel_w.torch[:, hand_id] + torch.cross(ang_vel_w, lever_w, dim=-1)
    return vectors_to_base(env, lin_vel_w, hand_cfg), vectors_to_base(env, ang_vel_w, hand_cfg)

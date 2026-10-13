# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Frame helpers of the Franka cloth dragging task.

Task quantities are expressed in the robot's root frame (suffix ``_b``) or in the cloth COM frame (suffix ``_com``),
whose origin is the cloth's center of mass and whose axes are those of the robot's root frame.
"""

from __future__ import annotations

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


def points_to_com(points_b: torch.Tensor, com_pos_b: torch.Tensor) -> torch.Tensor:
    """Express positions in the robot's root frame in the cloth COM frame.

    Args:
        points_b: Positions in the robot's root frame [m], shape [num_envs, ..., 3].
        com_pos_b: Cloth center of mass in the robot's root frame [m], shape [num_envs, 3].

    Returns:
        Positions relative to the cloth's center of mass along the root frame axes [m], same shape as ``points_b``.
    """
    return points_b - com_pos_b.view(len(com_pos_b), *([1] * (points_b.dim() - 2)), 3)


def cloth_com_pos_b(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Cloth center of mass in the robot's root frame [m], shape [num_envs, 3].

    The center of mass is the mean of the cloth nodes. Every term of the task that needs it uses this function.
    """
    asset: DeformableObject = env.scene[asset_cfg.name]
    return points_to_base(env, asset.data.root_pos_w.torch, robot_cfg)


def ee_pose_in_base(
    env: ManagerBasedEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """End-effector pose (first target of the frame transformer, the fingertip point) in the robot's root frame.

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
    """Linear velocity of the fingertip point and angular velocity of the hand in the robot's root frame.

    The frame transformer has no velocities, so the fingertip velocity is the hand body velocity moved along the lever
    arm from the hand body to the fingertip point.

    Args:
        env: The environment.
        hand_cfg: The robot, with ``body_names`` selecting the hand body that carries the end-effector frame.
        ee_frame_cfg: The frame transformer whose first target is the fingertip point.

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

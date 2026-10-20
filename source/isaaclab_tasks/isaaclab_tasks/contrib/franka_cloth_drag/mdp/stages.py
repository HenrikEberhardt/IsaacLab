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

from .frames import cloth_center_pos_b, raised_node_ids

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, DeformableObject
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.sensors import FrameTransformer

    from .commands_cfg import ClothTargetCommandCfg


class ClothDragStage(IntEnum):
    """Chronological stages of the cloth dragging task"""

    APPROACH = 0
    """The EE frame origin is away from the cloth."""
    GRASP = 1
    """The EE frame origin is near a raised cloth node (any cloth node if none is raised)."""
    DRAG = 2
    """The gripper holds the cloth by nodes locked to the EE frame, wherever it grasped."""
    GOAL = 3
    """The gripper holds the cloth and its central node is near the target."""


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


def ee_cloth_distance(env: ManagerBasedEnv, cfg: ClothTargetCommandCfg, height_offset: float = 0.0) -> torch.Tensor:
    """Distance of the EE frame origin to the nearest cloth node [m]. Shape is (num_envs,)."""
    cloth: DeformableObject = env.scene[cfg.object_name]
    ee_frame: FrameTransformer = env.scene[cfg.ee_frame_name]
    tip_w = ee_frame.data.target_pos_w.torch[:, 0].clone()
    tip_w[:, 2] -= height_offset
    return torch.linalg.norm(cloth.data.nodal_pos_w.torch - tip_w.unsqueeze(1), dim=2).min(dim=1).values


def ee_raised_node_distance(env: ManagerBasedEnv, cfg: ClothTargetCommandCfg) -> torch.Tensor:
    """Distance of the EE frame origin to the nearest raised cloth node [m]. Shape is (num_envs,).

    The raised nodes are those of the reset event :attr:`ClothTargetCommandCfg.raised_nodes_event`. Without raised
    nodes, this is the distance to the nearest cloth node, see :func:`ee_cloth_distance`.
    """
    ids = raised_node_ids(env, cfg.raised_nodes_event)
    if ids is None:
        return ee_cloth_distance(env, cfg)
    cloth: DeformableObject = env.scene[cfg.object_name]
    ee_frame: FrameTransformer = env.scene[cfg.ee_frame_name]
    raised_w = cloth.data.nodal_pos_w.torch.gather(1, ids.unsqueeze(-1).expand(-1, -1, 3))
    tip_w = ee_frame.data.target_pos_w.torch[:, 0]
    return torch.linalg.norm(raised_w - tip_w.unsqueeze(1), dim=2).min(dim=1).values


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


def cloth_goal_xy_distance(
    env: ManagerBasedEnv, target_pos_b: torch.Tensor, cfg: ClothTargetCommandCfg
) -> torch.Tensor:
    """Planar distance between the cloth's central node and the target [m]."""
    center_pos_b = cloth_center_pos_b(env, SceneEntityCfg(cfg.object_name), SceneEntityCfg(cfg.asset_name))
    return torch.linalg.norm(target_pos_b[:, :2] - center_pos_b[:, :2], dim=1)

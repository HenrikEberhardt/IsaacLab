# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task stages of the Franka cloth dragging task."""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from .rewards import _cloth_goal_xy_distance, _fingertips_near_cloth, _gripper_closing

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, DeformableObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer


class ClothDragStage(IntEnum):
    """Stages of the cloth dragging task, in the order the task passes through them."""

    APPROACH = 0
    """The fingertips are away from the cloth."""
    GRASP = 1
    """The fingertips are near the cloth."""
    DRAG = 2
    """Both finger pads touch the cloth while the gripper closes."""
    GOAL = 3
    """The cloth's center of mass is near the target."""


def cloth_drag_stage_index(
    env: ManagerBasedRLEnv,
    command_name: str,
    near_radius: float = 0.03,
    touch_distance: float = 0.008,
    goal_radius: float = 0.05,
    action_name: str = "gripper_action",
    finger_joint_name: str = "panda_finger_joint1",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Current :class:`ClothDragStage` of every environment. Shape is (num_envs,).

    The stage is evaluated from the current state on every call, checking the later stages first, so it falls back
    when, for example, the cloth slips out of the gripper:

    * GOAL: the cloth's center of mass is within ``goal_radius`` [m] of the target in the table plane.
    * DRAG: the gripper closes and both finger pads are within ``touch_distance`` [m] of a cloth node.
    * GRASP: the fingertip point is within ``near_radius`` [m] of a cloth node.
    * APPROACH: otherwise.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    asset: DeformableObject = env.scene[asset_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # finger pads: the fingertip point moved by the finger opening along the hand's y-axis, the closing direction
    finger_ids, _ = robot.find_joints(finger_joint_name)
    opening = robot.data.joint_pos.torch[:, finger_ids[0]].unsqueeze(-1)
    tip_w = ee_frame.data.target_pos_w.torch[:, 0]
    hand_y = math_utils.quat_apply(
        ee_frame.data.target_quat_w.torch[:, 0],
        torch.tensor((0.0, 1.0, 0.0), device=env.device).expand(env.num_envs, 3),
    )
    nodal_pos_w = asset.data.nodal_pos_w.torch
    pads_touch = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for side in (1.0, -1.0):
        pad_w = tip_w + side * opening * hand_y
        distance = torch.linalg.norm(nodal_pos_w - pad_w.unsqueeze(1), dim=2).min(dim=1).values
        pads_touch &= distance < touch_distance

    stage = torch.full((env.num_envs,), int(ClothDragStage.APPROACH), dtype=torch.long, device=env.device)
    stage[_fingertips_near_cloth(env, near_radius, asset_cfg, ee_frame_cfg)] = int(ClothDragStage.GRASP)
    stage[pads_touch & _gripper_closing(env, action_name)] = int(ClothDragStage.DRAG)
    stage[_cloth_goal_xy_distance(env, command_name, robot_cfg, asset_cfg) < goal_radius] = int(ClothDragStage.GOAL)
    return stage

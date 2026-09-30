# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observations for the Franka non-prehensile cloth task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from .top_down import gripper_yaw_error, top_down_yaw

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer

    from .actions import ClothGraspAction
    from .commands import CreaseDragCommand


def _in_robot_root_frame(env: ManagerBasedRLEnv, pos_w: torch.Tensor, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Express world positions [m] in the robot's root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, pos_w)
    return pos_b


def ee_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position of the fingertip point between the fingers in the robot's root frame [m]."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return _in_robot_root_frame(env, ee_frame.data.target_pos_w.torch[..., 0, :], robot_cfg)


def crease_position_in_robot_root_frame(
    env: ManagerBasedRLEnv, command_name: str, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Position of the cloth's crease in the robot's root frame [m]."""
    command: CreaseDragCommand = env.command_manager.get_term(command_name)
    return _in_robot_root_frame(env, command.crease_pos_w, robot_cfg)


def crease_relative_to_ee(
    env: ManagerBasedRLEnv,
    command_name: str,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Vector from the fingertip point to the crease in the robot's root frame [m]."""
    return crease_position_in_robot_root_frame(env, command_name, robot_cfg) - ee_position_in_robot_root_frame(
        env, ee_frame_cfg, robot_cfg
    )


def target_relative_to_crease(
    env: ManagerBasedRLEnv, command_name: str, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Vector from the crease to its drag target in the robot's root frame [m]."""
    command: CreaseDragCommand = env.command_manager.get_term(command_name)
    return command.command[:, :3] - crease_position_in_robot_root_frame(env, command_name, robot_cfg)


def ee_yaw_error_to_crease(
    env: ManagerBasedRLEnv,
    command_name: str,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Yaw of the gripper relative to the yaw that closes its fingers across the crease [rad], in [-pi/2, pi/2).

    Shape is (num_envs,).
    """
    command: CreaseDragCommand = env.command_manager.get_term(command_name)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    _, ee_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w.torch,
        robot.data.root_quat_w.torch,
        ee_frame.data.target_pos_w.torch[..., 0, :],
        ee_frame.data.target_quat_w.torch[..., 0, :],
    )
    return gripper_yaw_error(top_down_yaw(ee_quat_b), command.crease_yaw_b)


def crease_yaw_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gripper yaw error to the crease as the sine and cosine of twice the error, which are continuous across the
    gripper's half-turn symmetry. Shape is (num_envs, 2)."""
    error = 2.0 * ee_yaw_error_to_crease(env, command_name, ee_frame_cfg, robot_cfg)
    return torch.stack([torch.sin(error), torch.cos(error)], dim=-1)


def grasp_state(env: ManagerBasedRLEnv, action_name: str = "gripper_action") -> torch.Tensor:
    """Whether the gripper holds cloth vertices, as 1.0 or 0.0. Shape is (num_envs, 1)."""
    gripper: ClothGraspAction = env.action_manager.get_term(action_name)
    return gripper.is_grasping.float().unsqueeze(-1)

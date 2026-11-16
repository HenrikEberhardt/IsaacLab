# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""State of the constrained gripper: its command, finger opening and the cloth nodes between its pads.

Every function takes the cloth target command configuration for the scene names and reads the gripper settings
(finger joint, pad box) from the gripper action term it names, see :func:`gripper_cfg`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, DeformableObject
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.sensors import FrameTransformer

    from ..mdp.commands_cfg import ClothTargetCommandCfg
    from .grasp_action_cfg import ClothGraspActionCfg


def gripper_cfg(env: ManagerBasedEnv, cfg: ClothTargetCommandCfg) -> ClothGraspActionCfg:
    """Configuration of the gripper action term named by the command, which holds the gripper settings."""
    return env.action_manager.get_term(cfg.gripper_action_name).cfg


def gripper_closing(env: ManagerBasedEnv, cfg: ClothTargetCommandCfg) -> torch.Tensor:
    """Whether the binary gripper action commands closing. Shape is (num_envs,)."""
    term = env.action_manager.get_term(cfg.gripper_action_name)
    if hasattr(term, "closed"):
        return term.closed
    return torch.any(term.raw_actions < 0.0, dim=1)


def finger_opening(env: ManagerBasedEnv, cfg: ClothTargetCommandCfg) -> torch.Tensor:
    """Measured half the distance [m] between the finger joint positions."""
    robot: Articulation = env.scene[cfg.asset_name]
    finger_ids, _ = robot.find_joints(gripper_cfg(env, cfg).finger_joint_name)
    return robot.data.joint_pos.torch[:, finger_ids[0]]


def gripper_closure(env: ManagerBasedEnv, cfg: ClothTargetCommandCfg) -> torch.Tensor:
    """Measured closure of the gripper, zero at its open command and one at its close command. Shape is (num_envs,)."""
    action_cfg = gripper_cfg(env, cfg)
    open_opening = action_cfg.open_command_expr[action_cfg.finger_joint_name]
    close_opening = action_cfg.close_command_expr[action_cfg.finger_joint_name]
    return ((open_opening - finger_opening(env, cfg)) / (open_opening - close_opening)).clamp(0.0, 1.0)


def pad_box_mask(env: ManagerBasedEnv, cfg: ClothTargetCommandCfg, points_w: torch.Tensor) -> torch.Tensor:
    """Whether each point lies between the finger pads. Shape is (num_envs, num_points)."""
    action_cfg = gripper_cfg(env, cfg)
    ee_frame: FrameTransformer = env.scene[cfg.ee_frame_name]
    ee_pos_w = ee_frame.data.target_pos_w.torch[:, 0]
    ee_quat_w = ee_frame.data.target_quat_w.torch[:, 0].unsqueeze(1).expand(-1, points_w.shape[1], -1)
    # points relative to the EE point, in the EE frame
    rel_ee = quat_apply_inverse(ee_quat_w, points_w - ee_pos_w.unsqueeze(1))
    half_gap = finger_opening(env, cfg).unsqueeze(1) + action_cfg.finger_gap_margin
    return (
        (rel_ee[..., 0].abs() <= action_cfg.finger_pad_half_width)
        & (rel_ee[..., 1].abs() <= half_gap)
        & (rel_ee[..., 2].abs() <= action_cfg.finger_pad_half_length)
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

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read-outs of the constrained gripper for the MDP terms: its command, pad gap and the cloth nodes between its pads.

Every function takes the cloth target command configuration for the scene names and reads the gripper settings from
the gripper action term it names. The EE frame sits at the center of the closed pads, so an open pad sits behind it
along the frame's z-axis by its shift.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_apply_inverse

from .robotiq import pad_gap_from_joint, pad_shift_from_joint

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.sensors import FrameTransformer

    from ..mdp.commands_cfg import ClothTargetCommandCfg
    from .gripper_action_cfg import ConstrainedGripperActionCfg


def gripper_cfg(env: ManagerBasedEnv, cfg: ClothTargetCommandCfg) -> ConstrainedGripperActionCfg:
    """Configuration of the gripper action term named by the command."""
    return env.action_manager.get_term(cfg.gripper_action_name).cfg


def gripper_closing(env: ManagerBasedEnv, cfg: ClothTargetCommandCfg) -> torch.Tensor:
    """Whether the gripper action commands closing."""
    return env.action_manager.get_term(cfg.gripper_action_name).closed


def gripper_holds_cloth(env: ManagerBasedEnv, cfg: ClothTargetCommandCfg) -> torch.Tensor:
    """Whether the gripper holds cloth nodes locked to the EE frame."""
    return env.action_manager.get_term(cfg.gripper_action_name).is_grasping


def nodes_in_grasp(env: ManagerBasedEnv, cfg: ClothTargetCommandCfg) -> torch.Tensor:
    """Number of cloth nodes a grasp would lock now."""
    return env.action_manager.get_term(cfg.gripper_action_name).node_lock.select().sum(dim=1)


def finger_joint_pos(env: ManagerBasedEnv, cfg: ClothTargetCommandCfg) -> torch.Tensor:
    """Measured angle [rad] of the driven finger joint."""
    robot: Articulation = env.scene[cfg.asset_name]
    finger_ids, _ = robot.find_joints(gripper_cfg(env, cfg).finger_joint_name)
    return robot.data.joint_pos.torch[:, finger_ids[0]]


def pad_gap(env: ManagerBasedEnv, cfg: ClothTargetCommandCfg) -> torch.Tensor:
    """Measured distance [m] between the pad faces."""
    return pad_gap_from_joint(finger_joint_pos(env, cfg), gripper_cfg(env, cfg).gap_curve)


def pad_shift(env: ManagerBasedEnv, cfg: ClothTargetCommandCfg) -> torch.Tensor:
    """Measured position [m] of the pads along the EE frame's z-axis, zero when closed."""
    action_cfg = gripper_cfg(env, cfg)
    closed_q = action_cfg.close_command_expr[action_cfg.finger_joint_name]
    return pad_shift_from_joint(
        finger_joint_pos(env, cfg), action_cfg.gap_curve, closed_q, action_cfg.pad_path_direction
    )


def gripper_closure(env: ManagerBasedEnv, cfg: ClothTargetCommandCfg) -> torch.Tensor:
    """Measured closure of the gripper, zero at its open command and one at its close command."""
    action_cfg = gripper_cfg(env, cfg)
    open_q = action_cfg.open_command_expr[action_cfg.finger_joint_name]
    close_q = action_cfg.close_command_expr[action_cfg.finger_joint_name]
    return ((finger_joint_pos(env, cfg) - open_q) / (close_q - open_q)).clamp(0.0, 1.0)


def pad_box_mask(env: ManagerBasedEnv, cfg: ClothTargetCommandCfg, points_w: torch.Tensor) -> torch.Tensor:
    """Whether each point [m] lies between the finger pads, shape [num_envs, num_points].

    The box follows the pads: it is centered on the EE frame's origin shifted by :func:`pad_shift` along its z-axis.
    """
    action_cfg = gripper_cfg(env, cfg)
    ee_frame: FrameTransformer = env.scene[cfg.ee_frame_name]
    ee_pos_w = ee_frame.data.target_pos_w.torch[:, 0]
    ee_quat_w = ee_frame.data.target_quat_w.torch[:, 0].unsqueeze(1).expand(-1, points_w.shape[1], -1)
    rel_ee = quat_apply_inverse(ee_quat_w, points_w - ee_pos_w.unsqueeze(1))
    half_gap = 0.5 * pad_gap(env, cfg).unsqueeze(1) + action_cfg.finger_gap_margin
    return (
        (rel_ee[..., 0].abs() <= action_cfg.finger_pad_half_width)
        & (rel_ee[..., 1].abs() <= half_gap)
        & ((rel_ee[..., 2] - pad_shift(env, cfg).unsqueeze(1)).abs() <= action_cfg.finger_pad_half_length)
    )

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Conversion of the IK teacher's actions into the student's joint-delta labels.

The teacher acts in ``(dx, dy, dz, dyaw, gripper)``; :class:`~.ik_action.TeacherTopDownIKAction` turns the arm part into
a top-down pose target and, on every physics substep, into a joint target ``q + step`` with a damped-least-squares IK
step that is uniformly scaled to the joint-speed and joint-range bounds. The student commands joint targets
``q + scale * a`` (re-anchored on every substep as well), so its label is the teacher's first-substep step divided by
the student's joint-delta scale. The functions here are pure torch: the distillation algorithm calls them on the
``ik_state`` observation, without access to the environment.
"""

from __future__ import annotations

import math

import torch

import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.franka_cloth_drag.mdp.frames import top_down_yaw

from .ik_action import top_down_quat

NUM_ARM_JOINTS = 7
IK_STATE_DIM = NUM_ARM_JOINTS + 3 + 4 + 6 * NUM_ARM_JOINTS + 3 * NUM_ARM_JOINTS
"""Joint positions, EE position and orientation, Jacobian, lower and upper joint bounds and PD error cap."""


@configclass
class IKLabelCfg:
    """Teacher action parameters and the student's joint-delta scale used to convert teacher actions to labels."""

    scale_trans: tuple[float, float, float] = (0.015, 0.015, 0.015)
    """Teacher ``(dx, dy, dz)`` scale [m]."""

    scale_rot: float = 0.2
    """Teacher ``dyaw`` scale [rad]."""

    wrist_joint_limits: tuple[float, float] = (-2.0 * math.pi / 3.0, 2.0 * math.pi / 3.0)
    """Teacher wrist joint window [rad]."""

    limit_operation_space: bool = True
    """Whether the teacher clamps its position targets to the workspace."""

    x_limits: tuple[float, float] = (0.30, 0.80)
    """Teacher workspace x-range [m]."""

    y_limits: tuple[float, float] = (-0.40, 0.40)
    """Teacher workspace y-range [m]."""

    z_min: float = 0.0193
    """Teacher workspace floor [m]."""

    ik_lambda: float = 0.1
    """Damping of the teacher's damped-least-squares IK."""

    joint_delta_scale: float | tuple[float, ...] = 0.05
    """Student joint-delta scale [rad], one value or one per arm joint: an action of 1 commands a target this far
    from the current position."""

    preserve_direction: bool = False
    """Whether labels beyond [-1, 1] are scaled down uniformly (keeping the joint-space direction of the teacher's
    step) instead of being clipped per joint."""


def unpack_ik_state(ik_state: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split the ``ik_state`` observation, shape [N, IK_STATE_DIM], into its named parts."""
    n = NUM_ARM_JOINTS
    sizes = (n, 3, 4, 6 * n, n, n, n)
    names = ("joint_pos", "ee_pos", "ee_quat", "jacobian", "pos_lower", "pos_upper", "max_offset")
    parts = dict(zip(names, torch.split(ik_state, sizes, dim=-1)))
    parts["jacobian"] = parts["jacobian"].reshape(-1, 6, n)
    return parts


def teacher_joint_step(teacher_arm: torch.Tensor, ik_state: torch.Tensor, cfg: IKLabelCfg) -> torch.Tensor:
    """The teacher's first-substep joint target offset for its raw arm actions [rad], shape [N, 7].

    Args:
        teacher_arm: Raw teacher arm actions ``(dx, dy, dz, dyaw)``, shape [N, 4].
        ik_state: The ``ik_state`` observation, shape [N, IK_STATE_DIM].
        cfg: The teacher action parameters.
    """
    state = unpack_ik_state(ik_state)
    joint_pos, ee_pos, ee_quat = state["joint_pos"], state["ee_pos"], state["ee_quat"]
    scale = torch.tensor((*cfg.scale_trans, cfg.scale_rot), device=teacher_arm.device)
    delta = teacher_arm * scale
    # pose target, as TeacherTopDownIKAction.process_actions
    if cfg.limit_operation_space:
        lower = torch.tensor((cfg.x_limits[0], cfg.y_limits[0], cfg.z_min), device=delta.device)
        upper = torch.tensor((cfg.x_limits[1], cfg.y_limits[1], math.inf), device=delta.device)
        delta[:, :3] = torch.clamp(ee_pos + delta[:, :3], lower, upper) - ee_pos
    wrist_pos = joint_pos[:, -1]
    wrist_lower, wrist_upper = cfg.wrist_joint_limits
    delta[:, 3] = torch.clamp(delta[:, 3], wrist_pos - wrist_upper, wrist_pos - wrist_lower)
    yaw = top_down_yaw(ee_quat) + delta[:, 3]
    _, rotation = math_utils.compute_pose_error(ee_pos, ee_quat, ee_pos, top_down_quat(yaw), rot_error_type="axis_angle")
    pos_des, quat_des = math_utils.apply_delta_pose(ee_pos, ee_quat, torch.cat((delta[:, :3], rotation), dim=-1))
    # damped-least-squares step, as DifferentialIKController.compute
    pos_error, rot_error = math_utils.compute_pose_error(ee_pos, ee_quat, pos_des, quat_des, rot_error_type="axis_angle")
    jacobian = state["jacobian"]
    jacobian_t = jacobian.transpose(1, 2)
    damping = (cfg.ik_lambda**2) * torch.eye(6, device=jacobian.device)
    error = torch.cat((pos_error, rot_error), dim=-1).unsqueeze(-1)
    step = (jacobian_t @ torch.inverse(jacobian @ jacobian_t + damping) @ error).squeeze(-1)
    # uniform scaling to the PD error cap and the joint target bounds, as TeacherTopDownIKAction.apply_actions
    ratio = torch.maximum(torch.ones_like(step), step.abs() / state["max_offset"])
    room = torch.where(step > 0, state["pos_upper"] - joint_pos, joint_pos - state["pos_lower"]).clamp(min=0.0)
    ratio = torch.maximum(ratio, step.abs() / room.clamp(min=1e-9))
    return step / ratio.amax(dim=1, keepdim=True)


def teacher_actions_to_joint_deltas(
    teacher_actions: torch.Tensor, ik_state: torch.Tensor, cfg: IKLabelCfg
) -> tuple[torch.Tensor, torch.Tensor]:
    """Student labels for raw teacher actions, and which arm entries were clipped.

    Args:
        teacher_actions: Raw teacher actions ``(dx, dy, dz, dyaw, gripper)``, shape [N, 5].
        ik_state: The ``ik_state`` observation, shape [N, IK_STATE_DIM].
        cfg: The teacher action parameters and the student's joint-delta scale.

    Returns:
        The labels ``(7 joint deltas in [-1, 1], gripper)``, shape [N, 8], and a mask of the joint deltas clipped to
        [-1, 1], shape [N, 7].
    """
    step = teacher_joint_step(teacher_actions[:, :4], ik_state, cfg)
    scale = torch.as_tensor(cfg.joint_delta_scale, dtype=step.dtype, device=step.device)
    arm = torch.nan_to_num(step / scale, nan=0.0, posinf=1.0, neginf=-1.0)
    clipped = arm.abs() > 1.0
    if cfg.preserve_direction:
        arm = arm / arm.abs().amax(dim=1, keepdim=True).clamp(min=1.0)
    return torch.cat((arm.clamp(-1.0, 1.0), teacher_actions[:, 4:5]), dim=-1), clipped

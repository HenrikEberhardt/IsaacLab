# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events for the Franka cloth dragging task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, DeformableObject
    from isaaclab.envs import ManagerBasedEnv


def reset_franka_top_down(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    joint_offset_ranges: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset a Franka to a random arm pose with its hand still pointing the way it points in the default pose.

    All joints, the fingers included, are reset to their default positions with their default velocities, and offsets
    drawn uniformly from ``joint_offset_ranges`` [rad] are added to the default arm joint positions. With joints 3
    and 5 at zero, joints 2, 4 and 6 turn about parallel axes, and the hand's pitch is ``-q2 + q4 + q6`` plus a
    constant, so joint 6 takes up the pitch that the offsets of joints 2 and 4 add. Joints 1 and 7 only turn the hand
    about the vertical axis. Offsets of joints 3, 5, or 6 would tilt the hand, so they are rejected.

    Adapted from ``reset_franka_top_down`` on the ``non-prehensile-cloth`` branch.

    Args:
        env: The environment instance.
        env_ids: The environment indices to reset.
        joint_offset_ranges: Offset bounds [rad] keyed by the names of ``panda_joint1``, ``panda_joint2``,
            ``panda_joint4``, and ``panda_joint7``. Missing joints keep their default positions.
        asset_cfg: Scene entity of the Franka arm.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    tilting = set(joint_offset_ranges) & {"panda_joint3", "panda_joint5", "panda_joint6"}
    if tilting:
        raise ValueError(f"Offsets of {sorted(tilting)} would tilt the hand away from its default direction.")

    joint_pos = robot.data.default_joint_pos.torch[env_ids].clone()
    offset_ids, _ = robot.find_joints(list(joint_offset_ranges), preserve_order=True)
    ranges = torch.tensor(list(joint_offset_ranges.values()), device=joint_pos.device)
    offsets = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), len(offset_ids)), device=joint_pos.device)
    joint_pos[:, offset_ids] += offsets
    # keep the pitch of the hand: dq6 = dq2 - dq4
    (j2, j4, j6), _ = robot.find_joints(["panda_joint2", "panda_joint4", "panda_joint6"], preserve_order=True)
    default_pos = robot.data.default_joint_pos.torch[env_ids]
    joint_pos[:, j6] += (joint_pos[:, j2] - default_pos[:, j2]) - (joint_pos[:, j4] - default_pos[:, j4])

    limits = robot.data.soft_joint_pos_limits.torch[env_ids]
    joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])
    robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(
        velocity=robot.data.default_joint_vel.torch[env_ids].clone(), env_ids=env_ids
    )


def reset_cloth_planar_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: dict[str, tuple[float, float]],
    yaw_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> None:
    """Reset a cloth to its default shape, turned about the vertical axis and shifted in the table plane.

    The nodes of the default nodal state are rotated about the cloth's default center (the mean of its nodes) by a
    yaw drawn uniformly from ``yaw_range``, then shifted by an offset drawn uniformly from ``position_range``. The
    velocities are zero. The cloth keeps its shape, so the bending rest shape and all per-node quantities still apply.

    Args:
        env: The environment instance.
        env_ids: The environment indices to reset.
        position_range: Planar offset bounds [m] keyed by ``x`` and ``y``. Missing keys => no offset.
        yaw_range: Bounds of the rotation about the vertical axis [rad].
        asset_cfg: Scene entity of the cloth.
    """
    cloth: DeformableObject = env.scene[asset_cfg.name]
    nodal_state = cloth.data.default_nodal_state_w.torch[env_ids].clone()
    device = nodal_state.device
    num = len(env_ids)

    yaw = sample_uniform(yaw_range[0], yaw_range[1], (num, 1), device=device)
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    center_xy = nodal_state[..., :2].mean(dim=1, keepdim=True)
    rel = nodal_state[..., :2] - center_xy
    rotated = torch.stack([cos * rel[..., 0] - sin * rel[..., 1], sin * rel[..., 0] + cos * rel[..., 1]], dim=-1)

    ranges = torch.tensor([position_range.get(key, (0.0, 0.0)) for key in ("x", "y")], device=device)
    offset = sample_uniform(ranges[:, 0], ranges[:, 1], (num, 2), device=device)
    nodal_state[..., :2] = center_xy + rotated + offset.unsqueeze(1)
    nodal_state[..., 3:] = 0.0
    cloth.write_nodal_state_to_sim_index(nodal_state, env_ids=env_ids)

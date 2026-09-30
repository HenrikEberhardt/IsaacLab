# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events for the Franka non-prehensile cloth task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, DeformableObject, RigidObject
    from isaaclab.envs import ManagerBasedEnv


def reset_deformable_with_object(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: dict[str, tuple[float, float]],
    object_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> None:
    """Reset a deformable object and a rigid object resting on it by one shared random offset.

    => box is placed with same offset on the cloth

    Args:
        env: The environment instance.
        env_ids: The environment indices to reset.
        position_range: Shared displacement bounds [m] keyed by ``x``, ``y``, ``z``. Missing keys => defaults to
            no displacement.
        object_cfg: Scene entity of the rigid object that moves with the deformable.
        asset_cfg: Scene entity of the deformable object to reset.
    """
    deformable: DeformableObject = env.scene[asset_cfg.name]
    rigid_object: RigidObject = env.scene[object_cfg.name]

    ranges = torch.tensor([position_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z")], device=deformable.device)
    offset = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=deformable.device)

    nodal_state = deformable.data.default_nodal_state_w.torch[env_ids].clone()
    nodal_state[..., :3] += offset.unsqueeze(1)
    deformable.write_nodal_state_to_sim_index(nodal_state, env_ids=env_ids)

    root_pose = rigid_object.data.default_root_pose.torch[env_ids].clone()
    root_pose[:, :3] += env.scene.env_origins[env_ids] + offset
    rigid_object.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
    rigid_object.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros_like(rigid_object.data.default_root_vel.torch[env_ids]), env_ids=env_ids
    )


def reset_franka_top_down(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    joint_offset_ranges: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset a Franka arm to a random pose with its hand still pointing the way it points in the default pose.

    Offsets drawn uniformly from ``joint_offset_ranges`` [rad] are added to the default joint positions. With joints 3
    and 5 at zero, joints 2, 4 and 6 turn about parallel axes, and the hand's pitch is ``-q2 + q4 + q6`` plus a
    constant, so joint 6 takes up the pitch that the offsets of joints 2 and 4 add. Joints 1 and 7 only turn the hand
    about the vertical axis. Offsets of joints 3, 5, or 6 would tilt the hand, so they are rejected.

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
    arm_ids, _ = robot.find_joints("panda_joint[1-7]")
    robot.write_joint_position_to_sim_index(position=joint_pos[:, arm_ids], joint_ids=arm_ids, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(
        velocity=torch.zeros_like(joint_pos[:, arm_ids]), joint_ids=arm_ids, env_ids=env_ids
    )

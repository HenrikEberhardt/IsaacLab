# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture the state of environments of the Franka cloth dragging task as reset states for the bank."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..reset_strategies.restore import GRASP_STATE_KEYS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def capture_reset_states(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    robot_name: str = "robot",
    cloth_name: str = "deformable",
    cube_name: str | None = "cube",
    command_name: str = "cloth_target",
    gripper_action_name: str = "gripper_action",
    raised_nodes_event: str = "reset_deformable",
) -> dict[str, dict[str, torch.Tensor]]:
    """Capture the state of the given environments as reset states, see :mod:`~..reset_strategies.bank` for the layout.

    The ``scene`` group holds the robot's joint positions [rad or m] and velocities, the cloth's nodal positions [m]
    and velocities [m/s], and, with ``cube_name``, the cube's root pose (position [m], quaternion x, y, z, w) and
    velocity [m/s, rad/s]. Positions are relative to the environment origin. The ``task`` group holds the raised nodes,
    the target in the robot's root frame [m], the stage and the gripper and grasp state
    (:data:`~..reset_strategies.restore.GRASP_STATE_KEYS`).

    Args:
        env: The environment.
        env_ids: The environments to capture, shape [N].
        robot_name: Scene name of the robot.
        cloth_name: Scene name of the cloth.
        cube_name: Scene name of the cube standing on the cloth, or None without one.
        command_name: Name of the cloth target command.
        gripper_action_name: Name of the cloth grasp action term.
        raised_nodes_event: Name of the reset event that raises the cloth nodes.

    Returns:
        The ``scene`` and ``task`` groups, every tensor with N rows.
    """
    from ..mdp.frames import raised_node_ids

    robot, cloth = env.scene[robot_name], env.scene[cloth_name]
    origins = env.scene.env_origins[env_ids]
    scene = {
        "joint_pos": robot.data.joint_pos.torch[env_ids].clone(),
        "joint_vel": robot.data.joint_vel.torch[env_ids].clone(),
        "nodal_pos": cloth.data.nodal_pos_w.torch[env_ids] - origins.unsqueeze(1),
        "nodal_vel": cloth.data.nodal_vel_w.torch[env_ids].clone(),
    }
    if cube_name is not None:
        cube = env.scene[cube_name]
        pose = cube.data.root_pose_w.torch[env_ids].clone()
        pose[:, :3] -= origins
        scene["cube_root_pose"] = pose
        scene["cube_root_vel"] = cube.data.root_vel_w.torch[env_ids].clone()

    raised = raised_node_ids(env, raised_nodes_event)
    if raised is None:
        raise RuntimeError(f"Reset states need the raised nodes of the reset event '{raised_nodes_event}'.")
    command = env.command_manager.get_term(command_name)
    grasp = env.action_manager.get_term(gripper_action_name).get_grasp_state(env_ids)
    task = {
        "raised_node_ids": raised[env_ids].clone(),
        "target_pos_b": command.command[env_ids].clone(),
        "stage": command.stage[env_ids].clone(),
    }
    task |= {key: grasp[key] for key in GRASP_STATE_KEYS}
    return {"scene": scene, "task": task}

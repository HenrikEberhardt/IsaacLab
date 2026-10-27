# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Restore stored reset states into environments of the Franka cloth dragging task.

A reset runs the reset events, then resets the observation, action, reward, curriculum and command managers, then the
event manager. The action and command resets clear the grasp and draw a new target, so a restore has two parts:

1. :func:`restore_physical`, from a reset event: joint states, cloth nodes, cube, and the cloth rest shape around the
   stored raised nodes (before the observation reset, which reads the raised nodes).
2. :func:`restore_task_state`, from the event term's reset hook after the manager resets: gripper and grasp state,
   target and stage, and the gripper entry of the action buffers.

Both write only the given environments. :class:`~..mdp.reset_bank.ResetFromBank` calls them in that order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .capture import GRASP_STATE_KEYS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .bank import ResetBank


def restore_physical(
    env: ManagerBasedRLEnv,
    bank: ResetBank,
    rows: torch.Tensor,
    env_ids: torch.Tensor,
    robot_name: str = "robot",
    cloth_name: str = "deformable",
    cube_name: str | None = "cube",
    raised_nodes_event: str = "reset_deformable",
) -> None:
    """Write the physical state of the bank rows ``rows`` into the environments ``env_ids``.

    Uses the same writes as the task's reset events, so the solvers see a reset like any other. The cloth rest shape
    is set around the stored raised nodes through the reset event's
    :meth:`~..mdp.events.ResetClothWithRaisedNodes.set_raised_nodes`.

    Args:
        env: The environment.
        bank: The bank, on the environment's device.
        rows: Rows of the bank, shape [N].
        env_ids: Environments to write, shape [N].
        robot_name: Scene name of the robot.
        cloth_name: Scene name of the cloth.
        cube_name: Scene name of the cube standing on the cloth, or None without one.
        raised_nodes_event: Name of the reset event that raises the cloth nodes.
    """
    # the asset writes pass the indices to warp kernels as they are, which take int32 like the env's reset indices
    env_ids = env_ids.to(dtype=torch.int32)
    scene = bank.data["scene"]
    robot, cloth = env.scene[robot_name], env.scene[cloth_name]
    origins = env.scene.env_origins[env_ids]
    robot.write_joint_position_to_sim_index(position=scene["joint_pos"][rows], env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=scene["joint_vel"][rows], env_ids=env_ids)
    nodal_state = torch.cat([scene["nodal_pos"][rows] + origins.unsqueeze(1), scene["nodal_vel"][rows]], dim=-1)
    cloth.write_nodal_state_to_sim_index(nodal_state, env_ids=env_ids)
    if cube_name is not None:
        cube = env.scene[cube_name]
        pose = scene["cube_root_pose"][rows].clone()
        pose[:, :3] += origins
        cube.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        cube.write_root_velocity_to_sim_index(root_velocity=scene["cube_root_vel"][rows], env_ids=env_ids)
    raise_term = env.event_manager.get_term_cfg(raised_nodes_event).func
    raise_term.set_raised_nodes(env_ids, bank.data["task"]["raised_node_ids"][rows])


def restore_task_state(
    env: ManagerBasedRLEnv,
    bank: ResetBank,
    rows: torch.Tensor,
    env_ids: torch.Tensor,
    command_name: str = "cloth_target",
    gripper_action_name: str = "gripper_action",
) -> None:
    """Continue the task state of the bank rows ``rows`` in the environments ``env_ids``, after the manager resets.

    Restores the gripper and grasp state (:meth:`~..mdp.actions.ClothGraspAction.set_grasp_state`), the target and the
    stage (:meth:`~..mdp.commands.ClothTargetCommand.set_restored`), and the gripper entry of the action buffers, so the
    first observation's last action and the first action rate describe the stored gripper command.

    Args:
        env: The environment.
        bank: The bank, on the environment's device.
        rows: Rows of the bank, shape [N].
        env_ids: Environments to write, shape [N].
        command_name: Name of the cloth target command.
        gripper_action_name: Name of the cloth grasp action term.
    """
    env_ids = env_ids.to(dtype=torch.int32)
    task = bank.data["task"]
    gripper = env.action_manager.get_term(gripper_action_name)
    gripper.set_grasp_state(env_ids, {key: task[key][rows] for key in GRASP_STATE_KEYS})
    env.command_manager.get_term(command_name).set_restored(env_ids, task["target_pos_b"][rows], task["stage"][rows])
    terms = env.action_manager.active_terms
    start = sum(env.action_manager.action_term_dim[: terms.index(gripper_action_name)])
    command = torch.where(task["closed"][rows], -1.0, 1.0)
    env.action_manager.action[env_ids, start] = command
    env.action_manager.prev_action[env_ids, start] = command

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture the reset state of environments of the Franka cloth dragging task, and the task's fingerprint."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .bank import SCHEMA_VERSION, stable_hash

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

GRASP_STATE_KEYS = (
    "closed",
    "finger_target",
    "grasp_tried",
    "held_opening",
    "active",
    "grasped",
    "grasp_offsets",
    "squeeze_progress",
    "contacts_disabled",
)
"""Entries of :meth:`~..mdp.actions.ClothGraspAction.get_grasp_state` that a reset state stores."""


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
    """Capture the reset state of the given environments, see :mod:`.bank` for the layout.

    The ``scene`` group holds the robot's joint positions [rad or m] and velocities, the cloth's nodal positions [m]
    and velocities [m/s], and, with ``cube_name``, the cube's root pose (position [m], quaternion x, y, z, w) and
    velocity [m/s, rad/s]. Positions are relative to the environment origin. The ``task`` group holds the raised nodes,
    the target in the robot's root frame [m], the stage and the gripper and grasp state (:data:`GRASP_STATE_KEYS`).

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


def task_fingerprint(
    env: ManagerBasedRLEnv,
    robot_name: str = "robot",
    cloth_name: str = "deformable",
    cube_name: str | None = "cube",
    command_name: str = "cloth_target",
    gripper_action_name: str = "gripper_action",
    raised_nodes_event: str = "reset_deformable",
) -> dict[str, Any]:
    """Return what the stored states depend on, to check a bank against the task it is loaded into.

    Strict entries: the schema version, the number of cloth nodes, the robot's joint names, the scene's entity names and
    the cloth's default nodes relative to the environment origin. Soft entries: hashes of the physics settings (solver,
    time step, decimation, robot, cloth, cube) and of the task settings (gripper action, command, raised nodes).

    Args:
        env: The environment.
        robot_name: Scene name of the robot.
        cloth_name: Scene name of the cloth.
        cube_name: Scene name of the cube standing on the cloth, or None without one.
        command_name: Name of the cloth target command.
        gripper_action_name: Name of the cloth grasp action term.
        raised_nodes_event: Name of the reset event that raises the cloth nodes.

    Returns:
        The JSON-serializable fingerprint.
    """
    cfg = env.cfg
    robot, cloth = env.scene[robot_name], env.scene[cloth_name]
    mesh = cloth.data.default_nodal_state_w.torch[0, :, :3] - env.scene.env_origins[0]
    physics = {
        "sim_dt": cfg.sim.dt,
        "decimation": cfg.decimation,
        "physics": _without_resolved_bodies(cfg.sim.physics.to_dict()),
        "robot_spawn": getattr(cfg.scene, robot_name).spawn,
        "robot_actuators": getattr(cfg.scene, robot_name).actuators,
        "cloth_spawn": getattr(cfg.scene, cloth_name).spawn,
        "cube_spawn": getattr(cfg.scene, cube_name).spawn if cube_name is not None else None,
    }
    command_cfg = env.command_manager.get_term(command_name).cfg.to_dict()
    for key in ("debug_vis", "goal_visualizer_cfg", "resampling_time_range"):
        command_cfg.pop(key, None)
    task = {
        "gripper_action": env.action_manager.get_term(gripper_action_name).cfg,
        "command": command_cfg,
        "raised_nodes": env.event_manager.get_term_cfg(raised_nodes_event).params,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "num_nodes": int(mesh.shape[0]),
        "joint_names": list(robot.joint_names),
        "scene_entities": sorted(env.scene.keys()),
        "mesh_default_local": [[round(float(x), 6) for x in node] for node in mesh.tolist()],
        "physics_hash": stable_hash(physics),
        "task_hash": stable_hash(task),
    }


def _without_resolved_bodies(tree: Any) -> Any:
    """Drop the ``bodies`` entries that hold body indices instead of patterns from a physics config dictionary.

    The coupler replaces the body patterns of its proxies with the model body indices of every environment at runtime,
    so they depend on the number of environments, not on the physics.
    """
    if isinstance(tree, dict):
        return {
            key: _without_resolved_bodies(value)
            for key, value in tree.items()
            if not (key == "bodies" and isinstance(value, list) and all(isinstance(v, int) for v in value))
        }
    if isinstance(tree, list):
        return [_without_resolved_bodies(value) for value in tree]
    return tree

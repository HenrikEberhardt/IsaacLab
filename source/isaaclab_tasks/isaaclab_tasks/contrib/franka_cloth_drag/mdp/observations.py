# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the Franka cloth dragging task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, quat_inv, quat_mul, quat_unique

from ..constrained_gripper.gripper_state import nodes_in_grasp
from ..constrained_gripper.robotiq import pad_gap_from_joint
from .frames import (
    cloth_center_pos_b,
    cloth_landmark_ids,
    cloth_node_areas,
    ee_pose_in_base,
    ee_twist_in_base,
    points_to_base,
    raised_node_ids,
    top_down_yaw,
    vectors_to_base,
)
from .stages import ClothDragStage, grasp_node_pos_w
from .stages import grasp_node_capturable as _grasp_node_capturable

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, DeformableObject, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg
    from isaaclab.sensors import FrameTransformer

    from ..constrained_gripper.gripper_action import ConstrainedGripperAction


"""
Actor (and critic) terms.
"""


def ee_pose_b(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """EE position [m] and jaw yaw as (cos, sin) in the robot base frame, shape [N, 5]."""
    pos_b, quat_b = ee_pose_in_base(env, ee_frame_cfg, robot_cfg)
    yaw = top_down_yaw(quat_b).unsqueeze(-1)
    pose = torch.cat([pos_b, torch.cos(yaw), torch.sin(yaw)], dim=-1)
    return torch.nan_to_num(pose, nan=0.0, posinf=0.0, neginf=0.0)


def ee_velocity_b(
    env: ManagerBasedRLEnv, hand_cfg: SceneEntityCfg, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    """EE linear velocity [m/s] and yaw rate [rad/s] in the robot base frame, shape [N, 4]."""
    lin_vel_b, ang_vel_b = ee_twist_in_base(env, hand_cfg, ee_frame_cfg)
    velocity = torch.cat([lin_vel_b, ang_vel_b[:, 2:3]], dim=-1)
    return torch.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)


def gripper_width(env: ManagerBasedRLEnv, action_name: str = "gripper_action") -> torch.Tensor:
    """Gap between the finger pads [m] from the finger joint angle, shape [N, 1]."""
    action: ConstrainedGripperAction = env.action_manager.get_term(action_name)
    robot: Articulation = env.scene[action.cfg.asset_name]
    finger_id = robot.find_joints(action.cfg.finger_joint_name)[0][0]
    angle = robot.data.joint_pos.torch[:, finger_id]
    width = pad_gap_from_joint(angle, action.cfg.gap_curve).unsqueeze(1)
    return torch.nan_to_num(width, nan=0.0, posinf=0.0, neginf=0.0)

class ClothPointsInBaseFrame(ManagerTermBase):
    """Positions [m] of 9 landmarks, the raised nodes (grasp node first) and random nodes in the robot base frame."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        """Set up the landmarks and the per-episode node buffers."""
        super().__init__(cfg, env)
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("deformable"))
        self._landmark_ids = cloth_landmark_ids(env, asset_cfg)
        self._raised_event = cfg.params.get("raised_nodes_event", "reset_deformable")
        raised = raised_node_ids(env, self._raised_event)
        num_raised = 0 if raised is None else raised.shape[1]
        self._num_random = cfg.params.get("num_random_nodes", 11)
        self._draw_weights = cloth_node_areas(env, asset_cfg).clone()
        self._draw_weights[self._landmark_ids] = 0.0
        if not 0 <= self._num_random <= int((self._draw_weights > 0).sum()) - num_raised:
            raise ValueError("num_random_nodes must fit the positive-area nodes that are no landmark or raised node.")
        self._raised_ids = torch.zeros(env.num_envs, num_raised, dtype=torch.long, device=env.device)
        self._random_ids = torch.empty(env.num_envs, self._num_random, dtype=torch.long, device=env.device)
        self._generator = torch.Generator(device=env.device)
        self._generator.manual_seed(env.cfg.seed if env.cfg.seed is not None else torch.initial_seed())
        self.reset()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Take the episode's raised nodes and draw new area-weighted random nodes."""
        env_ids = slice(None) if env_ids is None else env_ids
        weights = self._draw_weights.expand(self.num_envs, -1)[env_ids].clone()
        if weights.shape[0] == 0:
            return
        if self._raised_ids.shape[1] > 0:
            raised = raised_node_ids(self._env, self._raised_event)
            self._raised_ids[env_ids] = raised[env_ids]
            weights.scatter_(1, self._raised_ids[env_ids], 0.0)
        if self._num_random > 0:
            ids = torch.multinomial(weights, self._num_random, replacement=False, generator=self._generator)
            self._random_ids[env_ids] = ids.sort(dim=1).values

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        num_random_nodes: int = 11,
        raised_nodes_event: str | None = "reset_deformable",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        command_name: str | None = None,
    ) -> torch.Tensor:
        """Current positions of the selected nodes [m], shape [N, (9 + raised + random) * 3]."""
        asset: DeformableObject = env.scene[asset_cfg.name]
        raised = self._raised_ids
        if command_name is not None and raised.shape[1] > 1:
            grasp_node = env.command_manager.get_term(command_name).grasp_node_id.unsqueeze(1)
            raised = raised.gather(1, torch.sort((raised != grasp_node).int(), dim=1, stable=True).indices)
        ids = torch.cat([self._landmark_ids.expand(env.num_envs, -1), raised, self._random_ids], dim=1)
        points_w = asset.data.nodal_pos_w.torch.gather(1, ids.unsqueeze(-1).expand(-1, -1, 3))
        points_b = points_to_base(env, points_w, robot_cfg).reshape(env.num_envs, -1)
        return torch.nan_to_num(points_b, nan=0.0, posinf=0.0, neginf=0.0)


def object_pose_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Rigid object position [m] and quaternion (x, y, z, w; w >= 0) in the robot base frame, shape [N, 7]."""
    return torch.cat(
        (object_position_b(env, asset_cfg, robot_cfg), object_quat_b(env, asset_cfg, robot_cfg)), dim=-1
    )


def object_position_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Rigid object position [m] in the robot base frame, shape [N, 3]."""
    asset: RigidObject = env.scene[asset_cfg.name]
    pos_b = points_to_base(env, asset.data.root_pos_w.torch, robot_cfg)
    return torch.nan_to_num(pos_b, nan=0.0, posinf=0.0, neginf=0.0)


def object_quat_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Rigid object quaternion (x, y, z, w; w >= 0) in the robot base frame, shape [N, 4]."""
    asset: RigidObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    quat_b = quat_unique(quat_mul(quat_inv(robot.data.root_quat_w.torch), asset.data.root_quat_w.torch))
    return torch.nan_to_num(quat_b, nan=0.0, posinf=0.0, neginf=0.0)


def cloth_drag_stage(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """One-hot task stage, shape [N, 4]."""
    stage = env.command_manager.get_term(command_name).stage
    one_hot = torch.nn.functional.one_hot(stage, num_classes=len(ClothDragStage)).float()
    return torch.nan_to_num(one_hot, nan=0.0, posinf=0.0, neginf=0.0)


def cloth_grasped(env: ManagerBasedRLEnv, action_name: str = "gripper_action") -> torch.Tensor:
    """+1 while cloth nodes are locked to the hand, else -1, shape [N, 1]."""
    action: ConstrainedGripperAction = env.action_manager.get_term(action_name)
    return action.is_grasping.float().unsqueeze(1) * 2.0 - 1.0


"""
Critic-only terms.
"""


def gripper_force(
    env: ManagerBasedRLEnv, termination_name: str = "gripper_force_limit", scale: float = 20.0
) -> torch.Tensor:
    """Push of the gripper into the table and the cloth divided by ``scale`` [N], shape [N, 1]; one at a ``scale``
    safety stop.

    Reads the push of the :class:`~.terminations.GripperForceLimit` term ``termination_name``, which the
    termination manager computes before the observations; reset environments read zero. Zero while the managers are
    built: the observation manager evaluates its terms once before the termination manager exists.
    """
    termination_manager = getattr(env, "termination_manager", None)
    if termination_manager is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    force = termination_manager.get_term_cfg(termination_name).func.force
    return (force / scale).unsqueeze(1)


def cloth_nodal_pos_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Positions of all cloth nodes [m] in the robot base frame, shape [N, num_nodes * 3]."""
    asset: DeformableObject = env.scene[asset_cfg.name]
    pos_b = points_to_base(env, asset.data.nodal_pos_w.torch, robot_cfg).reshape(env.num_envs, -1)
    return torch.nan_to_num(pos_b, nan=0.0, posinf=0.0, neginf=0.0)


def cloth_nodal_vel_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Velocities of all cloth nodes [m/s] in the robot base frame, shape [N, num_nodes * 3]."""
    asset: DeformableObject = env.scene[asset_cfg.name]
    vel_b = vectors_to_base(env, asset.data.nodal_vel_w.torch, robot_cfg).reshape(env.num_envs, -1)
    return torch.nan_to_num(vel_b, nan=0.0, posinf=0.0, neginf=0.0)


def grasp_node_pos_b(env: ManagerBasedRLEnv, command_name: str = "cloth_target") -> torch.Tensor:
    """Position of the episode's grasp node [m] in the robot base frame, shape [N, 3]."""
    command = env.command_manager.get_term(command_name)
    pos_w = grasp_node_pos_w(env, command.cfg, command.grasp_node_id)
    pos_b = points_to_base(env, pos_w, SceneEntityCfg(command.cfg.asset_name))
    return torch.nan_to_num(pos_b, nan=0.0, posinf=0.0, neginf=0.0)


def object_velocity_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Rigid object linear [m/s] and angular [rad/s] velocity in the robot base frame, shape [N, 6]."""
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_b = vectors_to_base(env, asset.data.root_lin_vel_w.torch, robot_cfg)
    ang_b = vectors_to_base(env, asset.data.root_ang_vel_w.torch, robot_cfg)
    return torch.nan_to_num(torch.cat((lin_b, ang_b), dim=-1), nan=0.0, posinf=0.0, neginf=0.0)


def object_center_offset_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    cloth_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Rigid object position relative to the cloth's central node [m] in the robot base frame, shape [N, 3]."""
    asset: RigidObject = env.scene[asset_cfg.name]
    pos_b = points_to_base(env, asset.data.root_pos_w.torch, robot_cfg)
    return torch.nan_to_num(pos_b - cloth_center_pos_b(env, cloth_cfg, robot_cfg), nan=0.0, posinf=0.0, neginf=0.0)


def ee_object_distance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Distance from the EE frame origin to the surface of a cuboid rigid object [m], zero inside, shape [N, 1]."""
    asset: RigidObject = env.scene[asset_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    half_size = 0.5 * torch.tensor(asset.cfg.spawn.size, device=env.device)
    pose_w = asset.data.root_pose_w.torch
    rel = quat_apply_inverse(pose_w[:, 3:7], ee_frame.data.target_pos_w.torch[:, 0] - pose_w[:, :3])
    outside = rel - rel.clamp(-half_size, half_size)
    return torch.nan_to_num(torch.linalg.norm(outside, dim=-1, keepdim=True), nan=0.0, posinf=0.0, neginf=0.0)


def cloth_center_to_target_b(env: ManagerBasedRLEnv, command_name: str = "cloth_target") -> torch.Tensor:
    """Planar displacement from the cloth's central node to the target [m] in the robot base frame, shape [N, 2]."""
    command = env.command_manager.get_term(command_name)
    center_b = cloth_center_pos_b(env, SceneEntityCfg(command.cfg.object_name), SceneEntityCfg(command.cfg.asset_name))
    return torch.nan_to_num(command.command[:, :2] - center_b[:, :2], nan=0.0, posinf=0.0, neginf=0.0)


def object_to_target_b(
    env: ManagerBasedRLEnv,
    command_name: str = "cloth_target",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Planar displacement from a rigid object to the target [m] in the robot base frame, shape [N, 2]."""
    asset: RigidObject = env.scene[asset_cfg.name]
    pos_b = points_to_base(env, asset.data.root_pos_w.torch, robot_cfg)
    target_pos_b = env.command_manager.get_command(command_name)[:, :2]
    return torch.nan_to_num(target_pos_b - pos_b[:, :2], nan=0.0, posinf=0.0, neginf=0.0)


def locked_node_count(env: ManagerBasedRLEnv, action_name: str = "gripper_action") -> torch.Tensor:
    """Number of cloth nodes locked to the hand, shape [N, 1]."""
    action: ConstrainedGripperAction = env.action_manager.get_term(action_name)
    return action.num_grasped.float().unsqueeze(1)


def nodes_between_pads(env: ManagerBasedRLEnv, command_name: str = "cloth_target") -> torch.Tensor:
    """Number of cloth nodes a close would lock now, shape [N, 1]."""
    command = env.command_manager.get_term(command_name)
    return nodes_in_grasp(env, command.cfg).float().unsqueeze(1)


def grasp_node_capturable(env: ManagerBasedRLEnv, command_name: str = "cloth_target") -> torch.Tensor:
    """1 if closing now captures the grasp node (or the lock holds it), else 0, shape [N, 1]."""
    command = env.command_manager.get_term(command_name)
    return _grasp_node_capturable(env, command.cfg, command.grasp_node_id).float().unsqueeze(1)


def grasp_tried(env: ManagerBasedRLEnv, action_name: str = "gripper_action") -> torch.Tensor:
    """1 if the current close has already tried the lock, else 0, shape [N, 1]."""
    action: ConstrainedGripperAction = env.action_manager.get_term(action_name)
    return action.grasp_tried.float().unsqueeze(1)


def domain_randomization_params(
    env: ManagerBasedRLEnv,
    mass_event: str = "cube_mass",
    material_event: str = "cloth_material",
    friction_event: str = "cloth_friction",
) -> torch.Tensor:
    """Drawn task parameters as log-ratios to their authored values (0 at the authored value), shape [N, 3 + G].

    Columns: cube mass, cloth density, cloth bending stiffness, then the cloth friction factor of each shape group of
    ``friction_event`` (e.g. table, cube). The terms are the event terms of these names: a
    :func:`~isaaclab.envs.mdp.randomize_rigid_body_mass` on the cube, :class:`.randomization.randomize_cloth_material`
    and :class:`.randomization.randomize_cloth_friction`.
    """
    mass_term = env.event_manager.get_term_cfg(mass_event).func
    material = env.event_manager.get_term_cfg(material_event).func
    friction = env.event_manager.get_term_cfg(friction_event).func
    # the mass term records the authored masses at its first call
    mass = mass_term.asset.data.body_mass.torch
    mass_scale = mass / mass_term.default_mass if mass_term.default_mass is not None else torch.ones_like(mass)
    columns = [material.SCALE_NAMES.index("density"), material.SCALE_NAMES.index("edge_ke")]
    scales = torch.cat((mass_scale[:, :1], material.scales[:, columns], friction.scales), dim=-1)
    return torch.log(scales)

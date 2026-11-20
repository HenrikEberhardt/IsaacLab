# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the Franka cloth dragging task.

Quantities are expressed in one of two frames: the robot's root frame (suffix ``_b``) or the end-effector (EE) frame,
the first target of the ``ee_frame`` frame transformer, centered between the finger pads. The EE frame turns with the
hand, like a wrist camera. The cloth's reference point is its central node, see :func:`~.frames.cloth_landmark_ids`.

Every term replaces NaN and infinite values with zeros: a diverged cloth simulation must never reach the policy or the
observation history, and rsl_rl aborts training on NaN.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, quat_inv, quat_mul, quat_unique

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

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, DeformableObject, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg
    from isaaclab.sensors import FrameTransformer

    from ..constrained_gripper.gripper_action import ConstrainedGripperAction
    from .actions import TopDownDeltaEEAction


def joint_pos_ratio(env: ManagerBasedRLEnv, action_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Joint positions as a ratio of the range the arm action keeps them in, shape [num_envs, num_joints].

    Minus one and one are the lower and upper limits of the action's ``joint_pos_limits``, where the controller stops
    the joint.

    Args:
        env: The environment.
        action_name: Name of the arm action term, which provides ``joint_pos_limits``.
        asset_cfg: The robot, with ``joint_names`` selecting the joints of the action in the same order.
    """
    action: TopDownDeltaEEAction = env.action_manager.get_term(action_name)
    lower, upper = action.joint_pos_limits
    robot = _robot_with_action_joints(env, asset_cfg, action_name, lower.shape[1])
    ratio = 2.0 * (robot.data.joint_pos.torch[:, asset_cfg.joint_ids] - lower) / (upper - lower) - 1.0
    return torch.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)


def joint_vel_ratio(env: ManagerBasedRLEnv, action_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Joint velocities as a ratio of the largest speeds the arm action commands, shape [num_envs, num_joints].

    The ratio is not clipped; in normal operation the measured speeds stay below the commanded ones.

    Args:
        env: The environment.
        action_name: Name of the arm action term, which provides ``joint_vel_limits``.
        asset_cfg: The robot, with ``joint_names`` selecting the joints of the action in the same order.
    """
    action: TopDownDeltaEEAction = env.action_manager.get_term(action_name)
    vel_limits = action.joint_vel_limits
    robot = _robot_with_action_joints(env, asset_cfg, action_name, vel_limits.shape[1])
    ratio = robot.data.joint_vel.torch[:, asset_cfg.joint_ids] / vel_limits
    return torch.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)


def _robot_with_action_joints(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, action_name: str, num_action_joints: int
) -> Articulation:
    """Return the robot after checking that ``asset_cfg`` selects as many joints as the action ``action_name``."""
    robot: Articulation = env.scene[asset_cfg.name]
    num_joints = robot.data.joint_pos.torch[:, asset_cfg.joint_ids].shape[1]
    if num_joints != num_action_joints:
        raise ValueError(
            f"The observation selects {num_joints} joints, but action '{action_name}' limits {num_action_joints}."
        )
    return robot


def gripper_width(env: ManagerBasedRLEnv, action_name: str = "gripper_action") -> torch.Tensor:
    """Distance between the finger pads [m], from the finger joint angle, shape [num_envs, 1].

    The real Robotiq reports its finger position, from which the gap follows.

    Args:
        env: The environment.
        action_name: Name of the :class:`~..constrained_gripper.gripper_action.ConstrainedGripperAction` term.
    """
    action: ConstrainedGripperAction = env.action_manager.get_term(action_name)
    robot: Articulation = env.scene[action.cfg.asset_name]
    finger_id = robot.find_joints(action.cfg.finger_joint_name)[0][0]
    angle = robot.data.joint_pos.torch[:, finger_id]
    width = pad_gap_from_joint(angle, action.cfg.gap_curve).unsqueeze(1)
    return torch.nan_to_num(width, nan=0.0, posinf=0.0, neginf=0.0)


def cloth_grasped(
    env: ManagerBasedRLEnv, action_name: str = "gripper_action", hide_curriculum: str | None = None
) -> torch.Tensor:
    """One while the gripper holds cloth nodes locked to the hand, else minus one, shape [num_envs, 1].

    Privileged: the real Franka Hand's ``is_grasped`` flag cannot tell thin cloth from an empty close, because
    ``grasp(width=0)`` also counts the closed fingers as grasped. See
    :class:`~..constrained_gripper.gripper_action.ConstrainedGripperAction`.

    With ``hide_curriculum``, the flag is zero in the episodes the curriculum term of that name hides it in
    (:attr:`~.curriculums.ObservationHidingCurriculum.hidden`), distinct from both grasp states. Without that term,
    e.g. with the curriculum disabled, the flag is always visible.

    Args:
        env: The environment.
        action_name: Name of the cloth grasp action term.
        hide_curriculum: Name of the curriculum term that hides the flag, or None.
    """
    action: ConstrainedGripperAction = env.action_manager.get_term(action_name)
    flag = action.is_grasping.float().unsqueeze(1) * 2.0 - 1.0
    if hide_curriculum is not None:
        hidden = _curriculum_hidden(env, hide_curriculum)
        if hidden is not None:
            flag = torch.where(hidden.unsqueeze(1), 0.0, flag)
    return flag


def grasped_node_ratio(env: ManagerBasedRLEnv, action_name: str = "gripper_action") -> torch.Tensor:
    """Share of the cloth nodes locked to the hand, shape [num_envs, 1]. See
    :class:`~..constrained_gripper.gripper_action.ConstrainedGripperAction`.

    Args:
        env: The environment.
        action_name: Name of the cloth grasp action term.
    """
    action: ConstrainedGripperAction = env.action_manager.get_term(action_name)
    return (action.num_grasped.float() / action.node_lock.grasped.shape[1]).unsqueeze(1)


def ee_pose_b(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """End-effector position [m] and jaw yaw as (cos, sin) in the robot's root frame, shape [num_envs, 5].

    The yaw is the rotation of the gripper jaws about the root frame's z-axis: the heading of the EE x-axis in the root
    frame's xy-plane. With the hand pointing down, the EE y-axis is the closing direction of the fingers, so a yaw of
    zero closes the jaws along the root frame's y-axis. The yaw describes the orientation of the hand as a whole, which
    all arm joints change, not the angle of the last arm joint. It is observed as its cosine and sine, which stay
    continuous where the heading wraps at ±180°: the wrist window lets the hand turn past it.
    """
    pos_b, quat_b = ee_pose_in_base(env, ee_frame_cfg, robot_cfg)
    yaw = top_down_yaw(quat_b).unsqueeze(-1)
    pose = torch.cat([pos_b, torch.cos(yaw), torch.sin(yaw)], dim=-1)
    return torch.nan_to_num(pose, nan=0.0, posinf=0.0, neginf=0.0)


def ee_velocity_b(
    env: ManagerBasedRLEnv, hand_cfg: SceneEntityCfg, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    """End-effector linear velocity [m/s] and yaw rate [rad/s] in the robot's root frame, shape [num_envs, 4].

    The yaw rate is the hand's angular velocity about the root frame z-axis, the rate of the yaw in :func:`ee_pose_b`
    while the hand points down.

    Args:
        env: The environment.
        hand_cfg: The robot, with ``body_names`` selecting the hand body that carries the end-effector frame.
        ee_frame_cfg: The frame transformer whose first target is the end effector.
    """
    lin_vel_b, ang_vel_b = ee_twist_in_base(env, hand_cfg, ee_frame_cfg)
    velocity = torch.cat([lin_vel_b, ang_vel_b[:, 2:3]], dim=-1)
    return torch.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)


class ClothPointsInEEFrame(ManagerTermBase):
    """Cloth nodes relative to the end effector, in the EE frame [m], shape [num_envs, num_points * 3].

    The EE frame turns with the hand, like a wrist camera. The term observes ``round(ratio * num_nodes)`` distinct
    nodes, so the observation size scales with the mesh, in three blocks:

    * the 9 landmarks of :func:`~.frames.cloth_landmark_ids` (corners, edge midpoints, central node), fixed;
    * random nodes, drawn without replacement and with a probability proportional to their cloth area when an
      environment resets, kept for the whole episode and sorted by id;
    * the highest remaining nodes at every step, sorted from the highest down: the crests of the wrinkles.

    ``height_share`` is the share of the non-landmark nodes chosen by height. The random draw uses its own generator,
    seeded with the environment seed. With ``fixed_random_nodes``, the random nodes are drawn once when the term is
    created, the same for every environment, and kept for the whole run instead of being drawn again at every reset.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("deformable"))
        asset: DeformableObject = env.scene[asset_cfg.name]
        num_nodes = asset.data.nodal_pos_w.torch.shape[1]
        self._landmark_ids = cloth_landmark_ids(env, asset_cfg)
        num_landmarks = len(self._landmark_ids)
        num_points = round(cfg.params["ratio"] * num_nodes)
        if not num_landmarks < num_points <= num_nodes:
            raise ValueError(
                f"The ratio {cfg.params['ratio']} selects {num_points} cloth nodes, but must select more than the"
                f" {num_landmarks} landmarks and at most the {num_nodes} nodes."
            )
        height_share = cfg.params.get("height_share", 0.5)
        if not 0.0 <= height_share <= 1.0:
            raise ValueError(f"The height share must be in [0, 1], got {height_share}.")
        self._num_highest = round(height_share * (num_points - num_landmarks))
        self._num_random = num_points - num_landmarks - self._num_highest
        # the landmarks are always observed, so the random draw skips them
        self._draw_weights = cloth_node_areas(env, asset_cfg).clone()
        self._draw_weights[self._landmark_ids] = 0.0
        self._random_ids = torch.empty(env.num_envs, self._num_random, dtype=torch.long, device=env.device)
        self._generator = torch.Generator(device=env.device)
        self._generator.manual_seed(env.cfg.seed if env.cfg.seed is not None else torch.initial_seed())
        self._fixed_random_nodes = cfg.params.get("fixed_random_nodes", False)
        if self._fixed_random_nodes and self._num_random > 0:
            ids = torch.multinomial(
                self._draw_weights.unsqueeze(0), self._num_random, replacement=False, generator=self._generator
            )
            self._random_ids[:] = ids.sort(dim=1).values
        else:
            self.reset()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Draw new random nodes for the selected environments, unless they are fixed for the run."""
        if self._fixed_random_nodes:
            return
        if env_ids is None:
            env_ids = slice(None)
            num_envs = self.num_envs
        else:
            num_envs = len(env_ids)
        if self._num_random == 0 or num_envs == 0:
            return
        weights = self._draw_weights.expand(num_envs, -1)
        ids = torch.multinomial(weights, self._num_random, replacement=False, generator=self._generator)
        self._random_ids[env_ids] = ids.sort(dim=1).values

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        ratio: float,
        height_share: float = 0.5,
        fixed_random_nodes: bool = False,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
        """Return the landmarks, random and highest nodes in the EE frame; ``ratio``, ``height_share`` and
        ``fixed_random_nodes`` are read once in the constructor."""
        asset: DeformableObject = env.scene[asset_cfg.name]
        ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
        nodal_pos_w = asset.data.nodal_pos_w.torch
        # the highest nodes among those that are neither landmarks nor random nodes
        height = nodal_pos_w[..., 2].clone()
        height[:, self._landmark_ids] = -math.inf
        height.scatter_(1, self._random_ids, -math.inf)
        highest_ids = height.topk(self._num_highest, dim=1).indices
        ids = torch.cat([self._landmark_ids.expand(env.num_envs, -1), self._random_ids, highest_ids], dim=1)
        points_w = nodal_pos_w.gather(1, ids.unsqueeze(-1).expand(-1, -1, 3))
        ee_pos_w = ee_frame.data.target_pos_w.torch[:, 0].unsqueeze(1)
        ee_quat_w = ee_frame.data.target_quat_w.torch[:, 0].unsqueeze(1).expand(-1, ids.shape[1], -1)
        points_e = quat_apply_inverse(ee_quat_w, points_w - ee_pos_w).reshape(env.num_envs, -1)
        return torch.nan_to_num(points_e, nan=0.0, posinf=0.0, neginf=0.0)


class ClothPointsInBaseFrame(ManagerTermBase):
    """Observe nine fixed landmarks, the episode's raised nodes and episode-sampled nodes in the robot base frame.

    Positions [m], shape [num_envs, (9 + K + num_random_nodes) * 3], in the slot order landmarks, raised nodes,
    random nodes:

    * the 9 landmarks of :func:`~.frames.cloth_landmark_ids`, fixed;
    * the K nodes raised by the reset event ``raised_nodes_event`` (:func:`~.frames.raised_node_ids`), sorted by id.
      With ``command_name``, the episode's grasp node (:attr:`~.commands.ClothTargetCommand.grasp_node_id`) comes
      first and the others follow by id. K is that event's ``num_raised_nodes``, read once at initialization; without
      raised nodes, K is 0;
    * ``num_random_nodes`` nodes sampled without replacement, weighted by cloth area, excluding the landmarks and the
      episode's raised nodes, sorted by id.

    All ids are taken at episode start (reset events run before the observation reset) and retained until the next
    episode, so every observation slot tracks the same material point through history. No cloth velocities are
    exposed.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        """Initialize the permanent landmarks and episode sampling buffers."""
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
        """Take the raised nodes and sample material node IDs for the selected episodes, preserving all others."""
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
        """Return current node positions [m]; ``num_random_nodes`` and ``raised_nodes_event`` are read once during
        initialization. With ``command_name``, the grasp node of that cloth target command leads the raised nodes."""
        asset: DeformableObject = env.scene[asset_cfg.name]
        raised = self._raised_ids
        if command_name is not None and raised.shape[1] > 1:
            grasp_node = env.command_manager.get_term(command_name).grasp_node_id.unsqueeze(1)
            # stable sort on "is not the grasp node" moves it to the front and keeps the others in id order
            raised = raised.gather(1, torch.sort((raised != grasp_node).int(), dim=1, stable=True).indices)
        ids = torch.cat([self._landmark_ids.expand(env.num_envs, -1), raised, self._random_ids], dim=1)
        points_w = asset.data.nodal_pos_w.torch.gather(1, ids.unsqueeze(-1).expand(-1, -1, 3))
        points_b = points_to_base(env, points_w, robot_cfg).reshape(env.num_envs, -1)
        return torch.nan_to_num(points_b, nan=0.0, posinf=0.0, neginf=0.0)


def grasp_node_pos_b(env: ManagerBasedRLEnv, command_name: str = "cloth_target") -> torch.Tensor:
    """Position of the episode's grasp node in the robot base frame [m], shape [num_envs, 3].

    See :attr:`~.commands.ClothTargetCommand.grasp_node_id` for the grasp node.
    """
    command = env.command_manager.get_term(command_name)
    pos_w = grasp_node_pos_w(env, command.cfg, command.grasp_node_id)
    pos_b = points_to_base(env, pos_w, SceneEntityCfg(command.cfg.asset_name))
    return torch.nan_to_num(pos_b, nan=0.0, posinf=0.0, neginf=0.0)


def cloth_target_displacement_b(env: ManagerBasedRLEnv, command_name: str = "cloth_target") -> torch.Tensor:
    """Planar central-node-to-target displacement in the robot base frame [m], shape [num_envs, 2]."""
    command = env.command_manager.get_term(command_name)
    center_b = cloth_center_pos_b(env, SceneEntityCfg(command.cfg.object_name), SceneEntityCfg(command.cfg.asset_name))
    return torch.nan_to_num(command.command[:, :2] - center_b[:, :2], nan=0.0, posinf=0.0, neginf=0.0)


def target_pos_ee(
    env: ManagerBasedRLEnv,
    command_name: str,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Commanded target relative to the end effector, along the robot's root frame axes [m], shape [num_envs, 3].

    The displacement the hand still has to make. Its axes do not turn with the hand's yaw, like the deltas of the EE
    action.
    """
    target_pos_b = env.command_manager.get_command(command_name)[:, :3]
    pos_b, _ = ee_pose_in_base(env, ee_frame_cfg, robot_cfg)
    return torch.nan_to_num(target_pos_b - pos_b, nan=0.0, posinf=0.0, neginf=0.0)


def object_pose_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Root position [m] and orientation (x, y, z, w) of a rigid object in the robot's root frame, shape
    [num_envs, 7].

    The quaternion is standardized to a non-negative real part, so ``q`` and ``-q`` do not alternate.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    pos_b = points_to_base(env, asset.data.root_pos_w.torch, robot_cfg)
    quat_b = quat_unique(quat_mul(quat_inv(robot.data.root_quat_w.torch), asset.data.root_quat_w.torch))
    return torch.nan_to_num(torch.cat((pos_b, quat_b), dim=-1), nan=0.0, posinf=0.0, neginf=0.0)


def object_velocity_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Root linear [m/s] and angular [rad/s] velocity of a rigid object along the robot's root frame axes, shape
    [num_envs, 6]."""
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
    """Rigid object's root relative to the cloth's central node, along the robot's root frame axes [m], shape
    [num_envs, 3]. Shows how far the object has slid on the cloth."""
    asset: RigidObject = env.scene[asset_cfg.name]
    pos_b = points_to_base(env, asset.data.root_pos_w.torch, robot_cfg)
    return torch.nan_to_num(pos_b - cloth_center_pos_b(env, cloth_cfg, robot_cfg), nan=0.0, posinf=0.0, neginf=0.0)


def target_object_displacement_b(
    env: ManagerBasedRLEnv,
    command_name: str = "cloth_target",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Commanded target relative to a rigid object's root, along the robot's root frame axes [m], shape
    [num_envs, 3]."""
    asset: RigidObject = env.scene[asset_cfg.name]
    pos_b = points_to_base(env, asset.data.root_pos_w.torch, robot_cfg)
    target_pos_b = env.command_manager.get_command(command_name)[:, :3]
    return torch.nan_to_num(target_pos_b - pos_b, nan=0.0, posinf=0.0, neginf=0.0)


def ee_object_distance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Distance of the EE frame origin to the surface of a cuboid rigid object [m], zero inside, shape
    [num_envs, 1].

    The object must be spawned as a cuboid (``spawn.size``). The fingers reach a few centimeters beyond the EE frame
    origin, so this is a coarse proximity measure; touching is detected by the solver contacts.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    half_size = 0.5 * torch.tensor(asset.cfg.spawn.size, device=env.device)
    pose_w = asset.data.root_pose_w.torch
    rel = quat_apply_inverse(pose_w[:, 3:7], ee_frame.data.target_pos_w.torch[:, 0] - pose_w[:, :3])
    outside = rel - rel.clamp(-half_size, half_size)
    return torch.nan_to_num(torch.linalg.norm(outside, dim=-1, keepdim=True), nan=0.0, posinf=0.0, neginf=0.0)


def termination_flag(env: ManagerBasedRLEnv, term_name: str) -> torch.Tensor:
    """Whether the termination term ``term_name`` fired at the current step, as 1 or 0, shape [num_envs, 1].

    Zeros before the termination manager exists (the observation manager computes every term once at initialization).
    """
    termination_manager = getattr(env, "termination_manager", None)
    if termination_manager is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    return termination_manager.get_term(term_name).float().unsqueeze(1)


def cloth_nodal_pos_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Positions of all cloth nodes, in a fixed order, in the robot's root frame [m], shape
    [num_envs, num_nodes * 3]."""
    asset: DeformableObject = env.scene[asset_cfg.name]
    pos_b = points_to_base(env, asset.data.nodal_pos_w.torch, robot_cfg).reshape(env.num_envs, -1)
    return torch.nan_to_num(pos_b, nan=0.0, posinf=0.0, neginf=0.0)


def cloth_nodal_vel_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Velocities of all cloth nodes, in a fixed order, in the robot's root frame [m/s], shape
    [num_envs, num_nodes * 3]."""
    asset: DeformableObject = env.scene[asset_cfg.name]
    vel_b = vectors_to_base(env, asset.data.nodal_vel_w.torch, robot_cfg).reshape(env.num_envs, -1)
    return torch.nan_to_num(vel_b, nan=0.0, posinf=0.0, neginf=0.0)


def cloth_drag_stage(env: ManagerBasedRLEnv, command_name: str, hide_curriculum: str | None = None) -> torch.Tensor:
    """Current :class:`ClothDragStage` of the command ``command_name`` as a one-hot vector, see
    :class:`~.commands.ClothTargetCommand`.

    With ``hide_curriculum``, the vector is all zeros in the episodes the curriculum term of that name hides the stage
    in (:attr:`~.curriculums.ObservationHidingCurriculum.hidden`). All zeros is distinct from every one-hot stage,
    APPROACH included. Without that term, e.g. with the curriculum disabled, the stage is always visible.

    Shape is (num_envs, 4).
    """
    stage = env.command_manager.get_term(command_name).stage
    one_hot = torch.nn.functional.one_hot(stage, num_classes=len(ClothDragStage)).float()
    if hide_curriculum is not None:
        hidden = _curriculum_hidden(env, hide_curriculum)
        if hidden is not None:
            one_hot = torch.where(hidden.unsqueeze(1), 0.0, one_hot)
    return torch.nan_to_num(one_hot, nan=0.0, posinf=0.0, neginf=0.0)


def _curriculum_hidden(env: ManagerBasedRLEnv, term_name: str) -> torch.Tensor | None:
    """Per-episode mask of the curriculum term ``term_name``, or None if there is no such term.

    The observation manager computes every term once at initialization, before the curriculum manager exists.
    """
    curriculum_manager = getattr(env, "curriculum_manager", None)
    term_cfg = getattr(curriculum_manager.cfg, term_name, None) if curriculum_manager is not None else None
    return getattr(term_cfg.func, "hidden", None) if term_cfg is not None else None

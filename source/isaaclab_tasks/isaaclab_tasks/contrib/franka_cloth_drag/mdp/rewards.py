# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms specific to the Franka cloth dragging task.

Stage terms take the names of the stages (``"approach"``, ``"grasp"``, ``"drag"``, ``"goal"``) they are active in,
read from the stage of the :class:`~.commands.ClothTargetCommand`. Outside these stages a term pays nothing, except
that a positive term pays its maximum (one) in every stage after its last active stage. A passed stage therefore keeps
paying its full reward, so advancing never lowers the reward, and dropping back loses the reward of the dropped
stages. Penalties never saturate.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .frames import cloth_center_pos_b, cloth_center_vel_b, ee_pose_in_base, ee_twist_in_base
from .observations import _robot_with_action_joints
from .stages import (
    ClothDragStage,
    cloth_goal_xy_distance,
    ee_cloth_distance,
    ee_raised_node_distance,
    gripper_closing,
    gripper_holds_cloth,
    nodes_in_grasp,
)

if TYPE_CHECKING:
    from isaaclab.assets import DeformableObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg
    from isaaclab.sensors import FrameTransformer

    from .actions import LimitedRelativeJointPositionAction, TopDownDeltaEEAction
    from .commands import ClothTargetCommand


def _command(env: ManagerBasedRLEnv, command_name: str) -> ClothTargetCommand:
    return env.command_manager.get_term(command_name)


def _stage_gate(value: torch.Tensor, stage: torch.Tensor, stages: Sequence[str], saturate: bool) -> torch.Tensor:
    """Keep ``value`` in the active ``stages``; pay one after the last active stage if ``saturate``, else zero."""
    try:
        stage_ids = [int(ClothDragStage[name.upper()]) for name in stages]
    except KeyError as err:
        names = [s.name.lower() for s in ClothDragStage]
        raise ValueError(f"Unknown stage {err} in {list(stages)}, expected names from {names}.") from None
    active = torch.zeros_like(stage, dtype=torch.bool)
    for stage_id in stage_ids:
        active |= stage == stage_id
    gated = torch.where(active, value, torch.zeros_like(value))
    if saturate:
        gated = torch.where(stage > max(stage_ids), torch.ones_like(value), gated)
    return gated


"""
General rewards and penalties, active in every stage.
"""


def cloth_held_at_target(env: ManagerBasedRLEnv, command_name: str = "cloth_target") -> torch.Tensor:
    """Return one while the cloth has rested at the target for the command's hold time, see
    :attr:`~.commands.ClothTargetCommand.holding`."""
    return _command(env, command_name).holding.float()


def cloth_lifted(
    env: ManagerBasedRLEnv,
    max_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return one when the cloth's central node is higher than ``max_height`` [m] above the robot's root frame."""
    return (cloth_center_pos_b(env, asset_cfg, robot_cfg)[:, 2] > max_height).float()


def fingertip_below_table(
    env: ManagerBasedRLEnv, min_height: float, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    """Depth of the fingertip point below ``min_height`` [m] above the table top [m]. Zero above it.

    The table top is at the height of the environment origin. The robot does not collide with the table, and the
    finger ends reach about 9 mm below the fingertip point, so a lower fingertip point pushes the fingers into the
    table and squeezes the cloth against it.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    tip_height = ee_frame.data.target_pos_w.torch[:, 0, 2] - env.scene.env_origins[:, 2]
    return (min_height - tip_height).clamp(min=0.0)


"""
Stage rewards.
"""


def fingertip_cloth_proximity(
    env: ManagerBasedRLEnv,
    std: float,
    stages: Sequence[str],
    hover_height: float = 0.0,
    command_name: str = "cloth_target",
) -> torch.Tensor:
    """Reward the fingertip point for approaching the nearest cloth node with a tanh kernel (``std`` [m]).

    With ``hover_height`` [m] the reward peaks that high above the nearest node, so the hand comes down onto the
    cloth from above instead of sweeping into it from the side.
    """
    command = _command(env, command_name)
    reward = 1.0 - torch.tanh(ee_cloth_distance(env, command.cfg, hover_height) / std)
    return _stage_gate(reward, command.stage, stages, saturate=True)


def fingertip_low(
    env: ManagerBasedRLEnv,
    min_height: float,
    std: float,
    stages: Sequence[str],
    xy_std: float | None = None,
    command_name: str = "cloth_target",
) -> torch.Tensor:
    """Reward a low fingertip point with a tanh kernel (``std`` [m]) on its height above ``min_height`` [m] over the
    table top; full reward at or below ``min_height``.

    Closing low puts the finger pads around the crease instead of its top, so the pinch holds when the hand moves.

    If ``xy_std`` [m] is set, the reward is scaled by a tanh kernel on the planar distance of the fingertip point to the
    nearest cloth node, so it is only paid over the cloth, not for pressing the hand onto the table beside it.
    """
    command = _command(env, command_name)
    ee_frame: FrameTransformer = env.scene[command.cfg.ee_frame_name]
    tip_w = ee_frame.data.target_pos_w.torch[:, 0]
    tip_height = tip_w[:, 2] - env.scene.env_origins[:, 2]
    reward = 1.0 - torch.tanh((tip_height - min_height).clamp(min=0.0) / std)
    if xy_std is not None:
        cloth: DeformableObject = env.scene[command.cfg.object_name]
        xy_offsets = cloth.data.nodal_pos_w.torch[..., :2] - tip_w[:, :2].unsqueeze(1)
        xy_distance = torch.linalg.norm(xy_offsets, dim=2).min(dim=1).values
        reward = reward * (1.0 - torch.tanh(xy_distance / xy_std))
    return _stage_gate(reward, command.stage, stages, saturate=True)


def gripper_open(
    env: ManagerBasedRLEnv,
    stages: Sequence[str],
    max_distance: float | None = None,
    saturate: bool = True,
    command_name: str = "cloth_target",
) -> torch.Tensor:
    """Return one while the gripper action commands opening.

    If ``max_distance`` [m] is set, it is only paid while the fingertip point is within that distance of the nearest
    cloth node, so holding the gripper open far from the cloth earns nothing. With ``saturate`` False, the term pays
    nothing after its active stages, so keeping the gripper open is not rewarded once the hand should close.
    """
    command = _command(env, command_name)
    reward = ~gripper_closing(env, command.cfg)
    if max_distance is not None:
        reward &= ee_cloth_distance(env, command.cfg) < max_distance
    return _stage_gate(reward.float(), command.stage, stages, saturate=saturate)


def ee_elevated_node_proximity(
    env: ManagerBasedRLEnv,
    std: float,
    stages: Sequence[str],
    flat_height: float = 0.005,
    elevation_scale: float = 0.01,
    command_name: str = "cloth_target",
) -> torch.Tensor:
    """Reward the EE frame origin for approaching raised cloth nodes, such as the crest of a fold.

    Each node's proximity, a tanh kernel (``std`` [m]) on its distance to the EE frame origin, is weighted by its
    elevation: its height above ``flat_height`` [m] over the table top, divided by ``elevation_scale`` [m] and clipped
    to [0, 1]. The reward is the largest weighted proximity. Flat cloth earns nothing, so the term only steers the hand
    towards folds and leaves a grasp of flat cloth to the other grasp terms.

    Args:
        env: The environment.
        std: Distance scale of the proximity kernel [m].
        stages: Names of the stages the term is active in.
        flat_height: Height of a node of flat cloth resting on the table, below which a node counts as not raised [m].
        elevation_scale: Elevation above ``flat_height`` at which a node gets the full weight [m].
        command_name: Name of the cloth target command.
    """
    command = _command(env, command_name)
    cloth: DeformableObject = env.scene[command.cfg.object_name]
    ee_frame: FrameTransformer = env.scene[command.cfg.ee_frame_name]
    nodal_pos_w = cloth.data.nodal_pos_w.torch
    ee_pos_w = ee_frame.data.target_pos_w.torch[:, 0]
    elevation = nodal_pos_w[..., 2] - env.scene.env_origins[:, 2:3] - flat_height
    weight = (elevation / elevation_scale).clamp(0.0, 1.0)
    proximity = 1.0 - torch.tanh(torch.linalg.norm(nodal_pos_w - ee_pos_w.unsqueeze(1), dim=2) / std)
    reward = (weight * proximity).amax(dim=1)
    return _stage_gate(reward, command.stage, stages, saturate=True)


def cloth_between_fingers(
    env: ManagerBasedRLEnv,
    num_nodes_full: int,
    stages: Sequence[str],
    command_name: str = "cloth_target",
    saturate: bool = True,
) -> torch.Tensor:
    """Reward cloth nodes between the finger pads, linear up to ``num_nodes_full`` nodes.

    The nodes are counted like the lock of the grasp counts them, see :func:`~.stages.nodes_in_grasp`.
    Set ``saturate`` False to stop paying after the active stages instead of paying the maximum.
    """
    command = _command(env, command_name)
    reward = (nodes_in_grasp(env, command.cfg).float() / num_nodes_full).clamp(max=1.0)
    return _stage_gate(reward, command.stage, stages, saturate=saturate)


def gripper_closed_on_cloth(
    env: ManagerBasedRLEnv, stages: Sequence[str], command_name: str = "cloth_target"
) -> torch.Tensor:
    """Return one when the gripper closes with at least the command's ``grasp_min_nodes`` nodes between the pads.

    The nodes are counted like the lock of the grasp counts them, see :func:`~.stages.nodes_in_grasp`.
    """
    command = _command(env, command_name)
    on_cloth = nodes_in_grasp(env, command.cfg) >= command.cfg.grasp_min_nodes
    reward = (gripper_closing(env, command.cfg) & on_cloth).float()
    return _stage_gate(reward, command.stage, stages, saturate=True)


def gripper_closed_on_nothing(
    env: ManagerBasedRLEnv, stages: Sequence[str], command_name: str = "cloth_target"
) -> torch.Tensor:
    """Return one when the gripper closes without any cloth node between the pads. A penalty: it never saturates.

    The nodes are counted like the lock of the grasp counts them, see :func:`~.stages.nodes_in_grasp`.
    """
    command = _command(env, command_name)
    empty = nodes_in_grasp(env, command.cfg) == 0
    penalty = (gripper_closing(env, command.cfg) & empty).float()
    return _stage_gate(penalty, command.stage, stages, saturate=False)


def joint_limit_barrier(
    env: ManagerBasedRLEnv, action_name: str, margin: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize arm joints within ``margin`` [rad] of the range the arm action keeps them in. A penalty.

    Per joint ``clamp((margin - room) / margin, 0, 1) ** 2``, summed over the joints, where ``room`` is the distance to
    the nearer bound of the action's ``joint_pos_limits`` (the reduced hardware range, and for the wrist the yaw
    window). It is exactly zero with at least ``margin`` of room and one per joint at the bound, where the controller
    stops the joint.

    Args:
        env: The environment.
        action_name: Name of the arm action term, which provides ``joint_pos_limits``.
        margin: Room to a bound below which the penalty starts [rad].
        asset_cfg: The robot, with ``joint_names`` selecting the joints of the action in the same order.
    """
    if margin <= 0.0:
        raise ValueError("margin must be positive.")
    action: LimitedRelativeJointPositionAction | TopDownDeltaEEAction = env.action_manager.get_term(action_name)
    lower, upper = action.joint_pos_limits
    robot = _robot_with_action_joints(env, asset_cfg, action_name, lower.shape[1])
    joint_pos = robot.data.joint_pos.torch[:, asset_cfg.joint_ids]
    room = torch.minimum(joint_pos - lower, upper - joint_pos)
    penalty = ((margin - room) / margin).clamp(0.0, 1.0).square().sum(dim=1)
    return torch.nan_to_num(penalty, nan=0.0, posinf=0.0, neginf=0.0)


def cloth_node_speed(env: ManagerBasedRLEnv, stages: Sequence[str], command_name: str = "cloth_target") -> torch.Tensor:
    """Return the mean planar speed of the cloth nodes [m/s] in the active stages. A penalty: it never saturates.

    Gate it on the stages before the grasp: pushing the cloth bunches it into a heap, which packs more nodes between
    the pads and inflates the grasp rewards, and the push carries the cloth off the table. Dragging moves the cloth on
    purpose, so the term should not be active then.
    """
    command = _command(env, command_name)
    cloth: DeformableObject = env.scene[command.cfg.object_name]
    speed = torch.linalg.norm(cloth.data.nodal_vel_w.torch[..., :2], dim=2).mean(dim=1)
    return _stage_gate(speed, command.stage, stages, saturate=False)


def ee_height_above(
    env: ManagerBasedRLEnv, max_height: float, stages: Sequence[str], command_name: str = "cloth_target"
) -> torch.Tensor:
    """Return the height of the EE frame origin above ``max_height`` over the table top [m], zero below it, in the
    active stages. A penalty: it never saturates.

    The EE frame origin lies between the finger pads. While the gripper holds the cloth, the cloth hangs from it, so
    no node is more than about 1 cm above it: gated on the holding stages, this bounds how far the cloth is lifted, so
    the policy drags the cloth instead of carrying it.

    Args:
        env: The environment.
        max_height: Height of the EE frame origin above the table top up to which there is no penalty [m].
        stages: Names of the stages the term is active in.
        command_name: Name of the cloth target command.
    """
    command = _command(env, command_name)
    ee_frame: FrameTransformer = env.scene[command.cfg.ee_frame_name]
    ee_height = ee_frame.data.target_pos_w.torch[:, 0, 2] - env.scene.env_origins[:, 2]
    return _stage_gate((ee_height - max_height).clamp(min=0.0), command.stage, stages, saturate=False)


def cloth_goal_proximity(
    env: ManagerBasedRLEnv, std: float, stages: Sequence[str], command_name: str = "cloth_target"
) -> torch.Tensor:
    """Reward the cloth's central node for approaching the target in the table plane with a tanh kernel
    (``std`` [m])."""
    command = _command(env, command_name)
    distance = cloth_goal_xy_distance(env, command.command, command.cfg)
    return _stage_gate(1.0 - torch.tanh(distance / std), command.stage, stages, saturate=True)


def cloth_goal_progress(
    env: ManagerBasedRLEnv, max_distance: float, stages: Sequence[str], command_name: str = "cloth_target"
) -> torch.Tensor:
    """Reward the cloth's central node for approaching the target in the table plane, linear in the distance.

    Returns ``1 - distance / max_distance`` clipped to [0, 1], so every meter of progress pays the same, unlike the
    flat tail of the tanh kernel of :func:`cloth_goal_proximity` far from the target.

    Args:
        env: The environment.
        max_distance: Planar distance at and beyond which the reward is zero [m].
        stages: Names of the stages the term is active in.
        command_name: Name of the cloth target command.
    """
    command = _command(env, command_name)
    distance = cloth_goal_xy_distance(env, command.command, command.cfg)
    reward = (1.0 - distance / max_distance).clamp(0.0, 1.0)
    return _stage_gate(reward, command.stage, stages, saturate=True)


def fingertip_goal_progress(
    env: ManagerBasedRLEnv, max_distance: float, stages: Sequence[str], command_name: str = "cloth_target"
) -> torch.Tensor:
    """Reward the fingertip point for approaching the target in the table plane, linear in the distance.

    Returns ``1 - distance / max_distance`` clipped to [0, 1]. The policy moves the fingertip directly, so while the
    cloth is pinched this pulls the hand, and with it the cloth, towards the target without waiting for the cloth's
    central node to follow.

    Args:
        env: The environment.
        max_distance: Planar distance at and beyond which the reward is zero [m].
        stages: Names of the stages the term is active in.
        command_name: Name of the cloth target command.
    """
    command = _command(env, command_name)
    tip_b, _ = ee_pose_in_base(env, SceneEntityCfg(command.cfg.ee_frame_name), SceneEntityCfg(command.cfg.asset_name))
    distance = torch.linalg.norm(command.command[:, :2] - tip_b[:, :2], dim=1)
    reward = (1.0 - distance / max_distance).clamp(0.0, 1.0)
    return _stage_gate(reward, command.stage, stages, saturate=True)


def fingertip_velocity_to_goal(
    env: ManagerBasedRLEnv,
    speed_scale: float,
    stages: Sequence[str],
    hand_cfg: SceneEntityCfg,
    command_name: str = "cloth_target",
) -> torch.Tensor:
    """Reward the planar fingertip velocity along the direction from the cloth's central node to the target,
    divided by ``speed_scale`` [m/s] and clipped to [-1, 1].

    The fingertip is driven directly by the policy, so this is a cleaner drag signal than the velocity of the central
    node. Gate it on a stage that requires the pinch, so moving the hand without the cloth earns nothing.

    Args:
        env: The environment.
        speed_scale: Fingertip speed towards the target that pays the full reward [m/s].
        stages: Names of the stages the term is active in.
        hand_cfg: The robot, with ``body_names`` selecting the hand body that carries the end-effector frame.
        command_name: Name of the cloth target command.
    """
    command = _command(env, command_name)
    cloth_cfg, robot_cfg = SceneEntityCfg(command.cfg.object_name), SceneEntityCfg(command.cfg.asset_name)
    to_goal = command.command[:, :2] - cloth_center_pos_b(env, cloth_cfg, robot_cfg)[:, :2]
    direction = to_goal / torch.linalg.norm(to_goal, dim=1, keepdim=True).clamp(min=1e-6)
    tip_vel_b, _ = ee_twist_in_base(env, hand_cfg, SceneEntityCfg(command.cfg.ee_frame_name))
    reward = ((tip_vel_b[:, :2] * direction).sum(dim=1) / speed_scale).clamp(-1.0, 1.0)
    return _stage_gate(reward, command.stage, stages, saturate=True)


def cloth_velocity_to_goal(
    env: ManagerBasedRLEnv, speed_scale: float, stages: Sequence[str], command_name: str = "cloth_target"
) -> torch.Tensor:
    """Reward the planar velocity of the cloth's central node towards the target, divided by ``speed_scale`` [m/s]
    and clipped to [-1, 1].

    Moving away from the target is penalized by the same amount, so going back and forth earns nothing.
    """
    command = _command(env, command_name)
    cloth_cfg, robot_cfg = SceneEntityCfg(command.cfg.object_name), SceneEntityCfg(command.cfg.asset_name)
    to_goal = command.command[:, :2] - cloth_center_pos_b(env, cloth_cfg, robot_cfg)[:, :2]
    direction = to_goal / torch.linalg.norm(to_goal, dim=1, keepdim=True).clamp(min=1e-6)
    velocity = cloth_center_vel_b(env, cloth_cfg, robot_cfg)[:, :2]
    reward = ((velocity * direction).sum(dim=1) / speed_scale).clamp(-1.0, 1.0)
    return _stage_gate(reward, command.stage, stages, saturate=True)


def fingertip_settled_at_target(
    env: ManagerBasedRLEnv,
    std_speed: float,
    stages: Sequence[str],
    hand_cfg: SceneEntityCfg,
    command_name: str = "cloth_target",
) -> torch.Tensor:
    """Reward a still fingertip while the cloth's central node is inside the command's success radius, with a tanh
    kernel on the planar fingertip speed (``std_speed`` [m/s]).

    The policy drives the fingertip directly, while the cloth's central node only follows it and also moves when the
    cloth deforms. A still pinching hand stops the light cloth, so the cloth comes to rest at the target.

    Args:
        env: The environment.
        std_speed: Planar fingertip speed at which the reward has dropped to about a quarter [m/s].
        stages: Names of the stages the term is active in.
        hand_cfg: The robot, with ``body_names`` selecting the hand body that carries the end-effector frame.
        command_name: Name of the cloth target command.
    """
    command = _command(env, command_name)
    in_radius = cloth_goal_xy_distance(env, command.command, command.cfg) < command.cfg.success_radius
    tip_vel_b, _ = ee_twist_in_base(env, hand_cfg, SceneEntityCfg(command.cfg.ee_frame_name))
    reward = in_radius.float() * (1.0 - torch.tanh(torch.linalg.norm(tip_vel_b[:, :2], dim=1) / std_speed))
    return _stage_gate(reward, command.stage, stages, saturate=True)


def cloth_settled_at_target(
    env: ManagerBasedRLEnv, std_speed: float, stages: Sequence[str], command_name: str = "cloth_target"
) -> torch.Tensor:
    """Reward a slow cloth inside the command's success radius with a tanh kernel on the planar speed of its central
    node (``std_speed`` [m/s]), so the cloth stops at the target instead of overshooting it."""
    command = _command(env, command_name)
    cloth_cfg, robot_cfg = SceneEntityCfg(command.cfg.object_name), SceneEntityCfg(command.cfg.asset_name)
    in_radius = cloth_goal_xy_distance(env, command.command, command.cfg) < command.cfg.success_radius
    speed = torch.linalg.norm(cloth_center_vel_b(env, cloth_cfg, robot_cfg)[:, :2], dim=1)
    reward = in_radius.float() * (1.0 - torch.tanh(speed / std_speed))
    return _stage_gate(reward, command.stage, stages, saturate=True)


class ClothDistanceProgress(ManagerTermBase):
    """Signed distance improvement for approach or attached cloth transport.

    ``objective="approach"`` tracks the distance of the EE frame origin to the nearest raised cloth node before
    attachment, the distance of the GRASP stage (:func:`~.stages.ee_raised_node_distance`; the nearest cloth node if
    none is raised). ``objective="drag"`` tracks planar center-to-target distance while attached. The first eligible
    sample establishes a reference; inactive samples still update the reference. Negative increments are retained. No
    stage-completion reward is paid. Attachment boundaries intentionally restart progress and do not imply a globally
    telescoping return.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize per-environment progress references."""
        super().__init__(cfg, env)
        self._previous_distance = torch.zeros(self.num_envs, device=self.device)
        self._was_eligible = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Invalidate distance references for the selected environments."""
        self._was_eligible[slice(None) if env_ids is None else env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        objective: str,
        distance_scale: float = 0.1,
        command_name: str = "cloth_target",
    ) -> torch.Tensor:
        """Return signed improvement per control second, normalized by ``distance_scale`` [m]."""
        if distance_scale <= 0.0:
            raise ValueError("distance_scale must be positive.")
        command = _command(env, command_name)
        attached = gripper_holds_cloth(env, command.cfg)
        if objective == "approach":
            distance = ee_raised_node_distance(env, command.cfg)
            eligible = ~attached
        elif objective == "drag":
            distance = cloth_goal_xy_distance(env, command.command, command.cfg)
            eligible = attached
        else:
            raise ValueError(f"Unknown progress objective {objective!r}; expected 'approach' or 'drag'.")
        eligible = eligible & torch.isfinite(distance)
        progress = (self._previous_distance - distance) / distance_scale
        reward = torch.where(eligible & self._was_eligible, progress, 0.0) / env.step_dt
        self._previous_distance.copy_(distance)
        self._was_eligible.copy_(eligible)
        return reward


def cloth_drag_success_bonus(env: ManagerBasedRLEnv, termination_name: str = "success") -> torch.Tensor:
    """Pay one on successful termination, compensating for reward-manager time integration."""
    return env.termination_manager.get_term(termination_name).float() / env.step_dt

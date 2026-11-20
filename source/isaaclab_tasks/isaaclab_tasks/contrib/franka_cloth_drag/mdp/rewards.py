# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms specific to the Franka cloth dragging task.

Stage terms take the names of the stages (``"approach"``, ``"grasp"``, ``"drag"``, ``"goal"``) they are active in,
read from the stage of the :class:`~.commands.ClothTargetCommand`. Outside these stages a term pays nothing, except
that saturating positive terms pay their maximum (one) in every stage after their last active stage. A passed stage
therefore keeps paying its full saturating reward, and dropping back loses it. Penalties and the closing incentives
never saturate.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from ..constrained_gripper.gripper_state import gripper_cfg, gripper_closing, gripper_closure, nodes_in_grasp
from .frames import (
    cloth_center_pos_b,
    cloth_center_vel_b,
    ee_pose_in_base,
    ee_twist_in_base,
    vectors_to_base,
)
from .stages import (
    ClothDragStage,
    cloth_goal_xy_distance,
    ee_cloth_distance,
    ee_grasp_node_distance,
    grasp_node_capturable,
)

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, DeformableObject, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg
    from isaaclab.sensors import FrameTransformer

    from .actions import TopDownDeltaEEAction
    from .commands import ClothTargetCommand


def _command(env: ManagerBasedRLEnv, command_name: str) -> ClothTargetCommand:
    return env.command_manager.get_term(command_name)


def _robot_with_action_joints(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, action_name: str, num_action_joints: int
) -> Articulation:
    """Return the robot after checking that ``asset_cfg`` selects as many joints as the action ``action_name``."""
    robot: Articulation = env.scene[asset_cfg.name]
    num_joints = robot.data.joint_pos.torch[:, asset_cfg.joint_ids].shape[1]
    if num_joints != num_action_joints:
        raise ValueError(
            f"The reward selects {num_joints} joints, but action '{action_name}' limits {num_action_joints}."
        )
    return robot


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

    The nodes are counted like the lock of the grasp counts them, see
    :func:`~..constrained_gripper.gripper_state.nodes_in_grasp`. Set ``saturate`` False to stop paying after the active
    stages instead of paying the maximum.
    """
    command = _command(env, command_name)
    reward = (nodes_in_grasp(env, command.cfg).float() / num_nodes_full).clamp(max=1.0)
    return _stage_gate(reward, command.stage, stages, saturate=saturate)


def gripper_closed(env: ManagerBasedRLEnv, stages: Sequence[str], command_name: str = "cloth_target") -> torch.Tensor:
    """Return one while commanded closed in the active stages, and zero otherwise.

    Use a negative weight in APPROACH to discourage premature closing without penalizing grasp attempts.

    Args:
        env: The environment.
        stages: Names of the stages in which closing is penalized.
        command_name: Name of the cloth target command.
    """
    command = _command(env, command_name)
    return _stage_gate(gripper_closing(env, command.cfg).float(), command.stage, stages, saturate=False)


def gripper_closed_on_grasp_node(
    env: ManagerBasedRLEnv,
    stages: Sequence[str],
    num_nodes_full: int | None = None,
    first_grasp_only: bool = False,
    command_name: str = "cloth_target",
) -> torch.Tensor:
    """Reward closing while the closing captures the grasp node, in the active stages.

    The grasp node is :attr:`~.commands.ClothTargetCommand.grasp_node_id`; capture is
    :func:`~.stages.grasp_node_capturable`: the attachment selector, including edge midpoints and triangle centers,
    picks at least ``grasp_min_nodes`` nodes with the grasp node among them, or an active lock holds it. The pad box
    bounds the EE height above the node by the pad half length. It never saturates into later stages.

    Only a close before its lock attempt pays
    (:attr:`~..constrained_gripper.gripper_action.ConstrainedGripperAction.grasp_tried`): once the fingers settle, the
    lock either forms or finds too few nodes, and a gripper held closed after a failed attempt earns nothing until it
    reopens and closes again.

    Without ``num_nodes_full`` the reward is one while closing captures the grasp node. With it, the reward grows with
    the number of nodes the grasp would lock (:func:`~..constrained_gripper.gripper_state.nodes_in_grasp`),
    ``min(n, num_nodes_full) / num_nodes_full``: a larger pinch pays more, and the cap keeps pushing the cloth into a
    heap from paying extra.

    Args:
        env: The environment.
        stages: Names of the stages in which closing is rewarded.
        num_nodes_full: Number of nodes in the grasp at which the reward reaches one, or None for one at any count.
        first_grasp_only: Whether to pay only until the episode first reaches DRAG
            (:attr:`~.commands.ClothTargetCommand.time_to_stage`), so releasing and grasping again does not pay twice.
        command_name: Name of the cloth target command.
    """
    if num_nodes_full is not None and num_nodes_full < 1:
        raise ValueError("num_nodes_full must be at least one.")
    command = _command(env, command_name)
    capturable = grasp_node_capturable(env, command.cfg, command.grasp_node_id)
    reward = (gripper_closing(env, command.cfg) & capturable).float()
    grasp_tried = getattr(env.action_manager.get_term(command.cfg.gripper_action_name), "grasp_tried", None)
    if grasp_tried is not None:
        reward *= (~grasp_tried).float()
    if num_nodes_full is not None:
        reward *= (nodes_in_grasp(env, command.cfg).float() / num_nodes_full).clamp(max=1.0)
    if first_grasp_only:
        reward *= (command.time_to_stage[:, int(ClothDragStage.DRAG)] < 0.0).float()
    return _stage_gate(reward, command.stage, stages, saturate=False)


def gripper_closed_off_grasp_node(
    env: ManagerBasedRLEnv, stages: Sequence[str], command_name: str = "cloth_target"
) -> torch.Tensor:
    """Return one while the gripper is commanded closed without the grasp node between the pads, in the active stages.

    Between the pads as the lock selects nodes (:func:`~.stages.grasp_node_capturable` without the minimum node
    count): the grasp node, or a mesh edge or triangle through it, lies in the pad box. Closing away from the grasp
    node, a grasp node that escapes the closing pads, or a lock without it (e.g. an edge pinch) costs. A penalty: it
    never saturates.

    Args:
        env: The environment.
        stages: Names of the stages in which closing is penalized.
        command_name: Name of the cloth target command.
    """
    command = _command(env, command_name)
    between = grasp_node_capturable(env, command.cfg, command.grasp_node_id, require_min_nodes=False)
    penalty = (gripper_closing(env, command.cfg) & ~between).float()
    return _stage_gate(penalty, command.stage, stages, saturate=False)


def gripper_closed_on_cloth(
    env: ManagerBasedRLEnv, stages: Sequence[str], command_name: str = "cloth_target"
) -> torch.Tensor:
    """Return one when the gripper closes with at least the gripper's ``grasp_min_nodes`` nodes between the pads.

    The nodes are counted like the lock of the grasp counts them, see
    :func:`~..constrained_gripper.gripper_state.nodes_in_grasp`.
    """
    command = _command(env, command_name)
    on_cloth = nodes_in_grasp(env, command.cfg) >= gripper_cfg(env, command.cfg).grasp_min_nodes
    reward = (gripper_closing(env, command.cfg) & on_cloth).float()
    return _stage_gate(reward, command.stage, stages, saturate=True)


def gripper_closed_on_nothing(
    env: ManagerBasedRLEnv, stages: Sequence[str], command_name: str = "cloth_target"
) -> torch.Tensor:
    """Return one when the gripper closes without any cloth node between the pads. A penalty: it never saturates.

    The nodes are counted like the lock of the grasp counts them, see
    :func:`~..constrained_gripper.gripper_state.nodes_in_grasp`.
    """
    command = _command(env, command_name)
    empty = nodes_in_grasp(env, command.cfg) == 0
    penalty = (gripper_closing(env, command.cfg) & empty).float()
    return _stage_gate(penalty, command.stage, stages, saturate=False)


def action_term_l2(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """Squared magnitude of one action term's actions, summed over its dimensions.

    Like :func:`isaaclab.envs.mdp.action_l2`, restricted to the term ``action_name``. For the arm term, this leaves out
    the binary gripper action and penalizes sustained full-range motion in addition to changes between actions.

    Args:
        env: The environment.
        action_name: Name of the action term.
    """
    manager = env.action_manager
    index = manager.active_terms.index(action_name)
    start = sum(manager.action_term_dim[:index])
    end = start + manager.action_term_dim[index]
    return torch.sum(torch.square(manager.action[:, start:end]), dim=1)


def action_term_rate_l2(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """Squared change of one action term's actions between the last two steps, summed over its dimensions.

    Like :func:`isaaclab.envs.mdp.action_rate_l2`, restricted to the term ``action_name``. For the arm term, this leaves
    out the binary gripper action, whose raw value changes freely without moving anything until it crosses the
    hysteresis threshold.

    Args:
        env: The environment.
        action_name: Name of the action term.
    """
    manager = env.action_manager
    index = manager.active_terms.index(action_name)
    start = sum(manager.action_term_dim[:index])
    end = start + manager.action_term_dim[index]
    return torch.sum(torch.square(manager.action[:, start:end] - manager.prev_action[:, start:end]), dim=1)


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
    action: TopDownDeltaEEAction = env.action_manager.get_term(action_name)
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


def ee_speed_above(
    env: ManagerBasedRLEnv,
    max_speed: float,
    stages: Sequence[str],
    hand_cfg: SceneEntityCfg,
    near_node_radius: float | None = None,
    only_while_closing: bool = False,
    command_name: str = "cloth_target",
) -> torch.Tensor:
    """Return the EE frame's speed above ``max_speed`` relative to ``max_speed``, clipped to [0, 1], in the active
    stages. A penalty: it never saturates.

    Near the grasp node it makes the approach arrive as slowly as the bank's grasp states were captured, and while
    closing it keeps the EE still, so the grasp the policy learns from bank starts also works on its own arrivals.

    Args:
        env: The environment.
        max_speed: EE speed up to which there is no penalty [m/s]; the penalty is one from twice this speed.
        stages: Names of the stages the term is active in.
        hand_cfg: The robot, with ``body_names`` selecting the hand body that carries the end-effector frame.
        near_node_radius: Only count steps with the EE frame origin this close to the grasp node [m]; None counts
            all.
        only_while_closing: Only count steps on which the gripper is commanded closed.
        command_name: Name of the cloth target command.
    """
    command = _command(env, command_name)
    ee_vel_b, _ = ee_twist_in_base(env, hand_cfg, SceneEntityCfg(command.cfg.ee_frame_name))
    penalty = ((torch.linalg.norm(ee_vel_b, dim=1) - max_speed) / max_speed).clamp(0.0, 1.0)
    if near_node_radius is not None:
        near = ee_grasp_node_distance(env, command.cfg, command.grasp_node_id) < near_node_radius
        penalty = penalty * near.float()
    if only_while_closing:
        penalty = penalty * gripper_closing(env, command.cfg).float()
    return _stage_gate(penalty, command.stage, stages, saturate=False)


def object_slip_speed(
    env: ManagerBasedRLEnv,
    threshold: float,
    stages: Sequence[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    command_name: str = "cloth_target",
) -> torch.Tensor:
    """Return the planar speed of a rigid object relative to the cloth's central node above ``threshold`` [m/s],
    zero below it, in the active stages. A penalty: it never saturates.

    The object slides on the cloth when the drag is too jerky. VBD smooths friction below a slip speed
    (``friction_epsilon``), so a resting object creeps slowly even without sliding; the threshold lies above that
    creep, so only real sliding is paid for.

    Args:
        env: The environment.
        threshold: Relative planar speed up to which there is no penalty [m/s].
        stages: Names of the stages the term is active in.
        asset_cfg: Scene entity of the rigid object on the cloth.
        command_name: Name of the cloth target command.
    """
    command = _command(env, command_name)
    asset: RigidObject = env.scene[asset_cfg.name]
    robot_cfg = SceneEntityCfg(command.cfg.asset_name)
    object_vel = vectors_to_base(env, asset.data.root_lin_vel_w.torch, robot_cfg)
    center_vel = cloth_center_vel_b(env, SceneEntityCfg(command.cfg.object_name), robot_cfg)
    slip = torch.linalg.norm(object_vel[:, :2] - center_vel[:, :2], dim=-1)
    slip = torch.nan_to_num(slip, nan=0.0, posinf=0.0, neginf=0.0)
    return _stage_gate((slip - threshold).clamp(min=0.0), command.stage, stages, saturate=False)


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
    """Signed distance improvement toward the grasp node or of the goal point toward the target, in the active stages.

    ``stages`` within APPROACH and GRASP tracks the distance of the EE frame origin to the grasp node
    (:attr:`~.commands.ClothTargetCommand.grasp_node_id`, see :func:`~.stages.ee_grasp_node_distance`), or to the
    point straight above or below it at ``target_height`` above the table.
    ``stages`` within DRAG and GOAL tracks the planar distance of the goal point to the target
    (:func:`~.stages.cloth_goal_xy_distance`: the cloth's central node, or the object on the cloth that must reach the
    target); DRAG and GOAL hold a raised node. The first active sample of an episode establishes a reference. While the
    stages are inactive the reference keeps the last active sample, so the first sample back pays or charges the change
    made meanwhile: the return telescopes over the episode, and leaving and re-entering the stages, e.g. lifting and
    dropping the cloth before re-grasping it, cannot pay the same distance twice. Negative increments are retained. No
    stage-completion reward is paid.

    With ``kernel_std`` [m], the tracked quantity is ``tanh(distance / kernel_std)`` instead of the distance: the
    progress over the whole approach still sums to at most one, but nearly all of it is paid within a few
    ``kernel_std`` of the target, where a precise final alignment is decided. Hovering still pays nothing.
    """

    _APPROACH_STAGES = frozenset({"approach", "grasp"})
    _DRAG_STAGES = frozenset({"drag", "goal"})

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize per-environment progress references and check the stages."""
        super().__init__(cfg, env)
        stages = {name.lower() for name in cfg.params.get("stages", ())}
        if stages and stages <= self._APPROACH_STAGES:
            self._approach = True
        elif stages and stages <= self._DRAG_STAGES:
            self._approach = False
            if cfg.params.get("target_height") is not None:
                raise ValueError("target_height only applies to the approach and grasp stages.")
        else:
            raise ValueError(
                f"Progress stages must lie within {sorted(self._APPROACH_STAGES)} or {sorted(self._DRAG_STAGES)},"
                f" got {cfg.params.get('stages')!r}."
            )
        self._previous_distance = torch.zeros(self.num_envs, device=self.device)
        self._has_reference = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Invalidate distance references for the selected environments."""
        self._has_reference[slice(None) if env_ids is None else env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        stages: Sequence[str],
        distance_scale: float = 0.1,
        kernel_std: float | None = None,
        target_height: float | None = None,
        command_name: str = "cloth_target",
    ) -> torch.Tensor:
        """Return signed improvement per control second, normalized by ``distance_scale`` [m], or of the tanh kernel
        of width ``kernel_std`` [m] if given (``distance_scale`` is then unused).

        Args:
            env: The environment.
            stages: Names of the stages the term is active in, within APPROACH and GRASP or within DRAG and GOAL.
            distance_scale: Distance improvement that pays one [m].
            kernel_std: Width of the tanh kernel on the distance [m], or None.
            target_height: Approach only: height of the tracked point above the table top [m], or None for the grasp
                node's own height.
            command_name: Name of the cloth target command.
        """
        if distance_scale <= 0.0:
            raise ValueError("distance_scale must be positive.")
        if kernel_std is not None and kernel_std <= 0.0:
            raise ValueError("kernel_std must be positive.")
        command = _command(env, command_name)
        if self._approach:
            distance = ee_grasp_node_distance(env, command.cfg, command.grasp_node_id, target_height)
        else:
            distance = cloth_goal_xy_distance(env, command.command, command.cfg)
        active = _stage_gate(torch.ones_like(distance), command.stage, stages, saturate=False) > 0.0
        eligible = active & torch.isfinite(distance)
        if kernel_std is not None:
            distance, distance_scale = torch.tanh(distance / kernel_std), 1.0
        progress = (self._previous_distance - distance) / distance_scale
        reward = torch.where(eligible & self._has_reference, progress, 0.0) / env.step_dt
        # inactive steps keep the reference, so re-entering the stages settles what changed meanwhile
        self._previous_distance.copy_(torch.where(eligible, distance, self._previous_distance))
        self._has_reference |= eligible
        return reward


class GripperCloseProgress(ManagerTermBase):
    """Signed progress of closing the gripper on the grasp node, in the active stages.

    The tracked quantity is the measured closure (:func:`~..constrained_gripper.gripper_state.gripper_closure`). With
    ``num_nodes_full`` it is multiplied by ``min(n, num_nodes_full) / num_nodes_full`` for the ``n`` nodes the grasp
    would lock (:func:`~..constrained_gripper.gripper_state.nodes_in_grasp`): a larger pinch pays more, and the cap
    keeps pushing the cloth into a heap from paying extra.

    Only steps in the active stages on which closing captures the grasp node
    (:func:`~.stages.grasp_node_capturable`) sample it. The first sample of an episode establishes a reference. Other
    steps keep the reference, so the next sample pays or charges the change made meanwhile. The return telescopes over
    the episode: a full close pays one, holding pays nothing, and reopening takes back what closing paid, also when the
    fingers open away from the grasp node. Opening and closing the fingers repeatedly therefore cannot pay twice.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize per-environment progress references."""
        super().__init__(cfg, env)
        self._previous_closure = torch.zeros(self.num_envs, device=self.device)
        self._has_reference = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Invalidate closure references for the selected environments."""
        self._has_reference[slice(None) if env_ids is None else env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        stages: Sequence[str],
        num_nodes_full: int | None = None,
        command_name: str = "cloth_target",
    ) -> torch.Tensor:
        """Return the signed closure improvement per control second.

        Args:
            env: The environment.
            stages: Names of the stages the term is active in.
            num_nodes_full: Number of nodes in the grasp at which a full close pays one, or None for one at any count.
            command_name: Name of the cloth target command.
        """
        if num_nodes_full is not None and num_nodes_full < 1:
            raise ValueError("num_nodes_full must be at least one.")
        command = _command(env, command_name)
        closure = gripper_closure(env, command.cfg)
        if num_nodes_full is not None:
            closure = closure * (nodes_in_grasp(env, command.cfg).float() / num_nodes_full).clamp(max=1.0)
        active = _stage_gate(torch.ones_like(closure), command.stage, stages, saturate=False) > 0.0
        eligible = active & grasp_node_capturable(env, command.cfg, command.grasp_node_id)
        progress = closure - self._previous_closure
        reward = torch.where(eligible & self._has_reference, progress, 0.0) / env.step_dt
        # other steps keep the reference, so the next eligible step settles what changed meanwhile
        self._previous_closure.copy_(torch.where(eligible, closure, self._previous_closure))
        self._has_reference |= eligible
        return reward


class StageProgress(ManagerTermBase):
    """Signed progress of the task stage (:class:`~.stages.ClothDragStage` index), paid per stage step.

    The first call of an episode sets the reference at the stage the episode starts in, so a bank start pays only for
    the stages it reaches itself. Entering the next stage pays one, falling back (leaving the grasp radius, dropping the
    cloth, leaving the goal zone) takes back what the stages paid, so the return telescopes and cannot be farmed. Every
    start stage gets the same dense "reach the next stage" signal.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize per-environment stage references."""
        super().__init__(cfg, env)
        self._previous_stage = torch.zeros(self.num_envs, device=self.device)
        self._has_reference = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Invalidate the stage references of the selected environments."""
        self._has_reference[slice(None) if env_ids is None else env_ids] = False

    def __call__(self, env: ManagerBasedRLEnv, command_name: str = "cloth_target") -> torch.Tensor:
        """Return the stage change since the last call per control second.

        Args:
            env: The environment.
            command_name: Name of the cloth target command.
        """
        stage = _command(env, command_name).stage.float()
        reward = torch.where(self._has_reference, stage - self._previous_stage, 0.0) / env.step_dt
        self._previous_stage.copy_(stage)
        self._has_reference[:] = True
        return reward


def cloth_drag_success_bonus(env: ManagerBasedRLEnv, termination_name: str = "success") -> torch.Tensor:
    """Pay one on successful termination, compensating for reward-manager time integration."""
    return env.termination_manager.get_term(termination_name).float() / env.step_dt


def cloth_drag_termination_penalty(env: ManagerBasedRLEnv, term_keys: Sequence[str]) -> torch.Tensor:
    """Pay one when any of the termination terms ``term_keys`` ends the episode, compensating for reward-manager time
    integration like :func:`cloth_drag_success_bonus`. Use a negative weight: the weight is the cost per termination.

    Several terms firing in the same step still count once.

    Args:
        env: The environment.
        term_keys: Names of the termination terms that are penalized.
    """
    terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in term_keys:
        terminated |= env.termination_manager.get_term(name)
    return terminated.float() / env.step_dt


def cloth_held(env: ManagerBasedRLEnv, stages: Sequence[str], command_name: str = "cloth_target") -> torch.Tensor:
    """Return one in the active stages, zero otherwise. It never saturates.

    Gated on DRAG and GOAL, which exist only while the gripper holds a raised node locked, this pays per second of
    holding, so forming the lock pays and releasing it stops the payment. Keep its weight times the episode length below
    the drag and goal progress rewards, so holding still never beats dragging.

    Args:
        env: The environment.
        stages: Names of the stages the term pays in.
        command_name: Name of the cloth target command.
    """
    command = _command(env, command_name)
    return _stage_gate(torch.ones_like(command.stage, dtype=torch.float), command.stage, stages, saturate=False)


def cloth_drag_drop_penalty(
    env: ManagerBasedRLEnv, stages: Sequence[str], command_name: str = "cloth_target"
) -> torch.Tensor:
    """Pay one when the gripper releases the cloth in one of ``stages`` (:attr:`~.commands.ClothTargetCommand.dropped`),
    compensating for reward-manager time integration like :func:`cloth_drag_success_bonus`. Use a negative weight: the
    weight is the cost per drop.

    A drop already returns the stage to APPROACH, so ``stages`` is checked against the stage before the drop
    (:attr:`~.commands.ClothTargetCommand.previous_stage`). Only DRAG and GOAL hold the cloth, so other stages never
    pay. Rewards are computed before the command update, so a release is paid on the step after it, with the same
    one-step lag as the stage gates.

    Args:
        env: The environment.
        stages: Names of the stages a drop is penalized in, within DRAG and GOAL.
        command_name: Name of the cloth target command.
    """
    command = _command(env, command_name)
    return _stage_gate(command.dropped.float(), command.previous_stage, stages, saturate=False) / env.step_dt

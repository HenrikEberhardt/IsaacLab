# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations for the command terms of the Franka cloth dragging task."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import SPHERE_MARKER_CFG
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .commands import ClothTargetCommand


@configclass
class ClothTargetCommandCfg(CommandTermCfg):
    """Configuration for a target position of the cloth's central node, see :class:`ClothTargetCommand`.

    The goal point is the cloth's central node, or the root of the rigid object :attr:`goal_object_name` (e.g. a cube
    standing on the cloth) if set; "central node" below means that goal point.

    The goal is reached when the cloth's central node stays within :attr:`success_radius` of the target in the table
    plane for :attr:`hold_time` in a row while the gripper holds the cloth by a raised node (stage DRAG or GOAL), so a
    cloth pushed into the target or dragged by its edge does not count. With :attr:`max_ee_height` set, the EE frame
    origin must also stay at most that high above the table, so a cloth carried over the target does not count.

    The command also tracks the task stage (:class:`~.stages.ClothDragStage`):

    * GRASP: the EE frame origin is within :attr:`approach_radius` of the episode's grasp node
      (:attr:`~.commands.ClothTargetCommand.grasp_node_id`): the node raised by the reset event
      :attr:`raised_nodes_event` closest to the segment from the central node to the target.
    * DRAG: the gripper holds the cloth by nodes locked to the EE frame that include a raised node, see
      :func:`~.stages.gripper_holds_raised_node` and
      :class:`~..constrained_gripper.gripper_action.ConstrainedGripperAction`. The nodes lock once the closing fingers
      have come to rest with at least
      :attr:`~..constrained_gripper.gripper_action_cfg.ConstrainedGripperActionCfg.grasp_min_nodes` nodes selected; a
      lock without a raised node (e.g. a pinch of the edge) stays in the earlier stage. Without raised nodes any lock
      counts.
    * GOAL: DRAG, and the cloth's central node is within :attr:`success_radius` + :attr:`goal_zone_margin` of the
      target.

    The locked nodes cannot slip, so the gripper only lets go of the cloth when it is commanded open. Releasing the
    cloth in DRAG or GOAL drops the stage back to APPROACH for one update.
    """

    class_type: type[ClothTargetCommand] | str = "{DIR}.commands:ClothTargetCommand"

    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)  # no resampling based on time
    """Time between resamples during an episode [s]. The target is only resampled when the episode resets."""

    cmd_kind: str | None = "command/body/target_position"
    """LEAPP input kind of the exported target: on the robot, Isaac ROS Deploy reads it from the position of the first
    pose of an ``isaac_ros_deploy_interfaces/msg/BodyCommand``."""

    element_names: list[str] | list[list[str]] | None = ["x", "y", "z"]
    """Target position in the robot's root frame [m]."""

    asset_name: str = MISSING
    """Name of the robot whose root frame the target is expressed in."""

    object_name: str = MISSING
    """Name of the cloth whose central node is tracked."""

    goal_object_name: str | None = None
    """Name of a rigid object on the cloth that must reach the target instead of the cloth's central node. Optional.

    With it set, the goal point of the target distance (GOAL stage, success, drag progress) is the object's root
    position. Defaults to None, the cloth's central node.
    """

    @configclass
    class Ranges:
        """Uniform ranges of the target position in the robot's root frame."""

        pos_x: tuple[float, float] = MISSING
        """Range of the target x-position [m]."""

        pos_y: tuple[float, float] = MISSING
        """Range of the target y-position [m]."""

        pos_z: tuple[float, float] = MISSING
        """Range of the target z-position [m]."""

    ranges: Ranges = MISSING
    """Ranges from which the target position is sampled."""

    success_radius: float = 0.05
    """Planar distance of the cloth's central node to the target within which the target counts as reached [m]."""

    hold_time: float = 0.5
    """Time the cloth's central node must stay within :attr:`success_radius` in a row to succeed [s]."""

    max_ee_height: float | None = 0.06
    """Highest the EE frame origin may be above the table top while the target counts as reached [m]. Optional.

    The held cloth hangs from the EE frame, so this keeps the cloth dragged on the table instead of carried. Defaults
    to 0.06, 4 cm above the lowest EE height. None disables the condition.
    """

    speed_window: int = 10
    """Number of last steps over which the speed of the central node is averaged for the ``center_speed_at_target``
    metric. The metric is logged only; the success does not depend on the speed."""

    ee_frame_name: str = "ee_frame"
    """Name of the frame transformer whose first target is the EE frame, centered between the finger pads."""

    gripper_action_name: str = "gripper_action"
    """Name of the :class:`~..constrained_gripper.gripper_action.ConstrainedGripperAction` term; a negative action
    closes the gripper. Its configuration holds the gripper settings (finger joint, pad box, ``grasp_min_nodes``)."""

    approach_radius: float = 0.05
    """Distance of the EE frame origin to the grasp node below which APPROACH turns into GRASP [m].

    See :attr:`~.commands.ClothTargetCommand.grasp_node_id` for the grasp node.
    """

    raised_nodes_event: str | None = "reset_deformable"
    """Name of the reset event term whose ``raised_node_ids`` are the raised nodes, see
    :class:`~.events.ResetClothWithRaisedNodes`. None, a missing term, or a term without raised nodes => the grasp
    node is the node a few centimeters from the central node towards the target, and any lock counts as a grasp."""

    goal_zone_margin: float = 0.05
    """Margin added to :attr:`success_radius` for the planar distance below which DRAG turns into GOAL [m]."""

    success_window_steps: int = 480
    """Number of environment steps the smoothed success rates per start stage average over (20 iterations at 24
    steps/iteration), see :class:`~.commands.StartStageSuccess`."""

    goal_visualizer_cfg: VisualizationMarkersCfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Command/cloth_target")
    """Configuration of the marker drawn at the target when :attr:`debug_vis` is enabled."""

    goal_visualizer_cfg.markers["sphere"].radius = 0.02

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of the gripper action term of the constrained gripper."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from ..mdp.actions_cfg import RateLimitedBinaryJointPositionActionCfg
from .robotiq import ROBOTIQ_DRIVE_JOINT, ROBOTIQ_GAP_CURVE

if TYPE_CHECKING:
    from .grasp_action import ClothGraspAction


@configclass
class ClothGraspActionCfg(RateLimitedBinaryJointPositionActionCfg):
    """Configuration for the constraint-based cloth gripper.

    Holds every setting of the Robotiq 2F-85 gripper: the finger command, the closing latch, the pad box that selects
    the nodes to lock, the lock itself and its release by opening, and the clearance before the gripper collides with
    the cloth again.
    """

    class_type: type[ClothGraspAction] | str = "{DIR}.grasp_action:ClothGraspAction"

    command_name: str = "cloth_target"
    """Name of the :class:`~..mdp.commands.ClothTargetCommand` term whose scene names (robot, cloth, EE frame) the
    grasp uses."""

    finger_joint_name: str = ROBOTIQ_DRIVE_JOINT
    """Name of the driven finger joint, whose angle [rad] grows while closing."""

    gap_curve: tuple[float, float, float] = ROBOTIQ_GAP_CURVE
    """``(a, b, c)`` of the distance between the pad faces, ``a - b * sin(q + c)`` [m], over the finger joint angle
    ``q`` [rad]."""

    pad_path_direction: float = 1.0
    """Direction of the pads' motion along the EE frame's z-axis while the gripper opens: 1.0 if the open pads sit
    towards the fingertips (``+z``) of the closed ones, -1.0 if they sit behind them."""

    latch_threshold: float = 0.5
    """Action threshold: close below its negative, open above it, otherwise retain the command."""

    settle_speed: float = 0.05
    """Maximum finger joint speed [rad/s] counted as settled when attempting a grasp."""

    release_travel: float = 0.004
    """Growth of the pad gap [m] that the gripper has to make after a failed grip before rearming the grasp."""

    release_gap: float = 0.002
    """Growth of the pad gap [m] from the pinch, once commanded open, at which the lock lets go of the cloth. The
    squeeze relaxes over this distance. A close command before it keeps the lock."""

    grasp_min_nodes: int = 1
    """Number of cloth nodes the gripper must lock to hold the cloth.

    The count is of the selected nodes: those in the pad box plus every node of an edge or triangle whose center lies
    in it (:meth:`~.constraint.ClothGraspConstraint.select`). A grasp with fewer selected nodes locks nothing, so the
    stage stays in GRASP. A lock counts for DRAG only if a raised node is among the locked nodes.
    """

    finger_pad_half_width: float = 0.0125
    """Half width of the pad box along the hand's x-axis [m].

    Wider than the physical pad (about 8.5 mm), so at least one node of the coarse cloth mesh (25 mm node spacing) can
    fall between the pads.
    """

    finger_pad_half_length: float = 0.0085
    """Half length of the pad box along the hand's z-axis, centered on the EE frame origin [m].

    The menagerie Panda pads span about ±8.5 mm around the fingertip point, so cloth lying under the closed fingertips
    (at least a particle radius below the pad) stays outside.
    """

    finger_gap_margin: float = 0.003
    """Tolerance added to the gap between the finger pads [m]."""

    layer_half_thickness: float | None = None
    """Half thickness of one cloth layer [m]. Defaults to None, in which case the cloth's particle radius is used.
    """

    squeeze_lift: float = 0.0
    """Distance by which the grasped nodes are lifted along the EE frame's -z axis while squeezed, epsilon of Nayak et
    al. [m]."""

    squeeze_steps: int = 5
    """Number of environment steps over which the press of the fold and the fingers ramp in after the grasp."""

    contact_clearance: float = 0.01
    """Distance [m] from the gripper's colliders that every cloth node needs before the contacts are restored."""

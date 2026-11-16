# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of the gripper action term of the constrained gripper."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from ..mdp.actions_cfg import RateLimitedBinaryJointPositionActionCfg

if TYPE_CHECKING:
    from .grasp_action import ClothGraspAction


@configclass
class ClothGraspActionCfg(RateLimitedBinaryJointPositionActionCfg):
    """Configuration for the constraint-based cloth gripper.

    Holds every setting of the gripper: the finger command, the closing latch, the pad box that selects the nodes to
    lock, the lock itself and the clearance before the gripper collides with the cloth again.
    """

    class_type: type[ClothGraspAction] | str = "{DIR}.grasp_action:ClothGraspAction"

    attach: bool = True
    """Whether to attach grasped cloth nodes. False uses friction alone."""

    command_name: str = "cloth_target"
    """Name of the :class:`~..mdp.commands.ClothTargetCommand` term whose scene names (robot, cloth, EE frame) the
    grasp uses."""

    finger_joint_name: str = "panda_finger_joint1"
    """Name of the actuated finger joint, whose position is the half gap between the finger pads."""

    latch_threshold: float | None = None
    """Action threshold: close below its negative, open above it, otherwise retain the command.
    """

    settle_speed: float = 0.005
    """Maximum finger speed [m/s] counted as settled when attempting a grasp."""

    release_travel: float = 0.002
    """Opening distance [m] that the gripper has to make after a failed grip before rearming the grasp."""

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
    """Number of environment steps over which the press of the fold and the fingers ramp in after the grasp, and back
    out before the release when the gripper is commanded open."""

    contact_clearance: float = 0.01
    """Cloth-node clearance [m] required to restore gripper contacts after release."""

    contact_pad_half_width: float = 0.0088
    """Half width of a physical finger pad along the hand's x-axis, for the contact clearance [m]."""

    contact_pad_thickness: float = 0.0152
    """Thickness of a physical finger pad along the closing direction, for the contact clearance [m]."""

    contact_pad_half_length: float = 0.0082
    """Half length of a physical finger pad along the hand's z-axis, for the contact clearance [m]."""

    contact_hand_half_extents: tuple[float, float, float] = (0.04, 0.10, 0.04)
    """Half extents of the box around the hand in the EE frame, for the contact clearance [m]."""

    contact_hand_center_z: float = -0.0737
    """Position of the hand box's center along the EE frame's z-axis, for the contact clearance [m]."""

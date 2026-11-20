# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of the action term of the constrained gripper."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.utils import configclass

from .robotiq import ROBOTIQ_DRIVE_JOINT, ROBOTIQ_GAP_CURVE, ROBOTIQ_PAD_LENGTH

if TYPE_CHECKING:
    from .gripper_action import ConstrainedGripperAction


@configclass
class ConstrainedGripperActionCfg(BinaryJointPositionActionCfg):
    """Configuration of the constrained gripper: finger command, hysteresis, lock, squeeze, release and contact
    clearance.

    The finger command is a step to the open or close position, as the real gripper receives it; the finger actuator
    sets the speed and delay.
    """

    class_type: type[ConstrainedGripperAction] | str = "{DIR}.gripper_action:ConstrainedGripperAction"

    command_name: str = "cloth_target"
    """Name of the :class:`~..mdp.commands.ClothTargetCommand` whose scene names (robot, cloth, EE frame) are used."""

    # finger command

    finger_joint_name: str = ROBOTIQ_DRIVE_JOINT
    """Name of the driven finger joint, whose angle [rad] grows while closing."""

    gap_curve: tuple[float, float, float] = ROBOTIQ_GAP_CURVE
    """``(a, b, c)`` of the distance between the pad faces, ``a - b * sin(q + c)`` [m], over the finger angle ``q``."""

    pad_path_direction: float = 1.0
    """Direction of the pads' motion along the EE frame's z-axis while opening: 1.0 if the open pads sit towards the
    fingertips (``+z``) of the closed ones, -1.0 if they sit behind them."""

    hysteresis_threshold: float = 0.5
    """Action threshold: close below its negative, open above it, otherwise keep the command."""

    # lock

    settle_speed: float = 0.05
    """Finger joint speed [rad/s] below which a closing gripper counts as settled and tries the lock, once the finger
    has moved faster than this since the close command."""

    close_timeout: float = 1.0
    """Time [s] after the close command from which a settled finger tries the lock even if it never moved (e.g. it was
    already pinching). Longer than the delay plus the full stroke (about 0.9 s)."""

    grasp_min_nodes: int = 1
    """Number of selected cloth nodes a grasp needs to lock; with fewer it locks nothing."""

    finger_pad_half_width: float = 0.0125
    """Half width [m] of the pad box that selects the nodes, along the EE frame's x-axis (1.5 mm wider than the
    physical pad on each side)."""

    finger_pad_half_length: float = 0.5 * ROBOTIQ_PAD_LENGTH
    """Half length [m] of the pad box along the EE frame's z-axis, centered on the pads."""

    finger_gap_margin: float = 0.003
    """Tolerance [m] added to the gap between the pads of the pad box."""

    layer_half_thickness: float | None = None
    """Half thickness [m] of one cloth layer; None uses the cloth's particle radius."""

    squeeze_steps: int = 5
    """Number of environment steps over which the squeeze of the fold and the fingers ramps in after the grasp."""

    # release

    release_gap: float = 0.002
    """Growth of the pad gap [m] from the pinch, once commanded open, at which the lock lets go of the cloth."""

    release_travel: float = 0.004
    """Growth of the pad gap [m] after a failed grasp that re-arms the next grasp attempt."""

    contact_clearance: float = 0.01
    """Distance [m] from the gripper's colliders that every cloth node needs before the contacts are restored."""

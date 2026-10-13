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
    """Configuration for a target position of the cloth's center of mass, see :class:`ClothTargetCommand`.

    The goal is reached when the cloth's center of mass rests (slower than :attr:`max_speed`) within
    :attr:`success_radius` of the target in the table plane for :attr:`hold_time` in a row.
    """

    class_type: type[ClothTargetCommand] | str = "{DIR}.commands:ClothTargetCommand"

    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)  # no resampling based on time
    """Time between resamples during an episode [s]. The target is only resampled when the episode resets."""

    asset_name: str = MISSING
    """Name of the robot whose root frame the target is expressed in."""

    object_name: str = MISSING
    """Name of the cloth whose center of mass is tracked."""

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
    """Planar distance of the cloth's center of mass to the target within which the target counts as reached [m]."""

    max_speed: float = 0.02
    """Speed of the cloth's center of mass below which the cloth counts as at rest [m/s]."""

    hold_time: float = 0.5
    """Time the cloth must rest at the target in a row to succeed [s]."""

    speed_window: int = 10
    """Number of last steps over which the speed of the COM is averaged for the ``com_speed_at_target`` metric."""

    goal_visualizer_cfg: VisualizationMarkersCfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Command/cloth_target")
    """Configuration of the marker drawn at the target when :attr:`debug_vis` is enabled."""

    goal_visualizer_cfg.markers["sphere"].radius = 0.02

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
    """Configuration for a target position of the cloth's center of mass, see :class:`ClothTargetCommand`."""

    class_type: type[ClothTargetCommand] | str = "{DIR}.commands:ClothTargetCommand"

    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    """Time between resamples during an episode [s]. Defaults to far longer than any episode, so the target only
    changes when the episode resets."""

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

    goal_visualizer_cfg: VisualizationMarkersCfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Command/cloth_target")
    """Configuration of the marker drawn at the target when :attr:`debug_vis` is enabled."""

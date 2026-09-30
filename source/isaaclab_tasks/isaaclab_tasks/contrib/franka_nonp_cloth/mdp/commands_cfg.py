# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of the crease drag command for the Franka non-prehensile cloth task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass

from isaaclab_tasks.core.lift.mdp.commands.pose_commands_cfg import DeformableUniformPoseCommandCfg

if TYPE_CHECKING:
    from .commands import CreaseDragCommand


def _crease_marker_cfg(prim_path: str, radius: float) -> VisualizationMarkersCfg:
    """Return a sphere marker that is red (index 0) away from the goal and green (index 1) at it."""
    return VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "position_far": sim_utils.SphereCfg(
                radius=radius, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
            ),
            "position_near": sim_utils.SphereCfg(
                radius=radius, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
            ),
        },
    )


@configclass
class CreaseDragCommandCfg(DeformableUniformPoseCommandCfg):
    """Configuration of the target point to drag the crease of the cloth to, see :class:`CreaseDragCommand`."""

    class_type: type[CreaseDragCommand] | str = "{DIR}.commands:CreaseDragCommand"

    crest_band: float = 0.005
    """Height below the highest node of the cloth's rest shape within which its nodes form the crest [m]."""

    grasp_depth: float = 0.012
    """Depth of the grasp point below the crease [m], where the fingertip point sits when the fingers straddle the fold.
    """

    crease_radius: float = 0.02
    """Horizontal distance from the center of the crest within which crest nodes form the tracked crease [m].
    """

    success_threshold: float = 0.03
    """Horizontal distance of the crease from the target within which the drag succeeded [m]."""

    drag_fraction: float = 1.0
    """Fraction of the way from the crease to the target drawn on the line at which the target is placed.

    A curriculum can raise it from a short drag to the full one. Defaults to 1.0, the target on the line.
    """

    min_drag_distance: float = 0.06
    """Shortest horizontal drag that :attr:`drag_fraction` can shorten a target to [m].

    No episode succeeds without a drag as min distance is larger then success threshold.
    """

    # the drag target is drawn larger than the crease, so both show when they meet
    goal_pose_visualizer_cfg: VisualizationMarkersCfg = _crease_marker_cfg("/Visuals/Command/drag_target", 0.02)
    """Marker of the drag target."""

    curr_pose_visualizer_cfg: VisualizationMarkersCfg = _crease_marker_cfg("/Visuals/Command/crease", 0.012)
    """Marker of the crease."""

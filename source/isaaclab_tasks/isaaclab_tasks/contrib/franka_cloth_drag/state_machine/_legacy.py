# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated names kept for one compatibility period."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab_visualizers.viser import ViserVisualizerCfg

from .cloth_drag_sm import DragSmState

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnvCfg

HOVER_MARGIN = 0.01
MAX_GRASP_TRIES = 1
STATE_NAMES = [state.name for state in DragSmState]
YAW_ACTION_SCALE = 0.5
YAW_TOLERANCE = 0.1


def keep_requested_visualizers(env_cfg: ManagerBasedRLEnvCfg, visualizers: Sequence[str] | None):
    """Configure fixed Viser output on port 8760.

    .. deprecated:: 3.0
       Interactive scripts now configure their visualizer directly.
    """
    del visualizers
    warnings.warn(
        "keep_requested_visualizers() is deprecated; configure ViserVisualizerCfg(port=8760) directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    env_cfg.sim.visualizer_cfgs = [ViserVisualizerCfg(port=8760)]

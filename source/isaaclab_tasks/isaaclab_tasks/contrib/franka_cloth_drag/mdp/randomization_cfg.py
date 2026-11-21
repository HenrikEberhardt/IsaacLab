# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations of the cloth-drag observation noise models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseCfg, NoiseModelCfg

if TYPE_CHECKING:
    from .randomization import ClothPointsNoiseModel


@configclass
class ClothPointsNoiseModelCfg(NoiseModelCfg):
    """Perception error of the observed cloth points: a rigid offset of the whole cloth and a less certain crease.

    The term's points are laid out as ``num_landmarks`` landmarks, then ``num_raised`` raised nodes, then any further
    points, each as (x, y, z). Every episode draws one planar offset that shifts all points together, like a camera
    calibration error, and a second one shared by the raised nodes, like a crease seen slightly off; every frame adds
    noise to the raised nodes. The cloth's shape between non-raised points is kept.
    """

    class_type: type[ClothPointsNoiseModel] | str = "{DIR}.randomization:ClothPointsNoiseModel"

    noise_cfg: NoiseCfg = GaussianNoiseCfg(std=0.0)
    """Unused; the noise is set by the fields below."""

    num_landmarks: int = 9
    """Number of landmark points before the raised nodes."""

    num_raised: int = 3
    """Number of raised-node points after the landmarks."""

    cloth_offset_range: tuple[float, float] = (-0.003, 0.003)
    """Range [m] of the per-episode planar (x, y) offset of all points."""

    raised_offset_range: tuple[float, float] = (-0.004, 0.004)
    """Range [m] of the per-episode planar (x, y) offset added to the raised nodes."""

    raised_noise_std: float = 0.0025
    """Standard deviation [m] of the per-frame noise on each raised node, in x, y and z."""

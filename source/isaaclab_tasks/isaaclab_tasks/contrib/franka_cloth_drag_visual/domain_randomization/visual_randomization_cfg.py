# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""All camera-side randomization values and the events that apply them."""

from __future__ import annotations

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

from .appearance import randomize_light, randomize_shape_colors
from .camera_mount import randomize_camera_offset
from .image_augmentation import RgbAugmentationCfg

##
# Image stream (applied by the student's image term)
##

IMAGE_LATENCY_STEPS = (2, 3)  # policy steps: ~66 ms capture-to-publish at 30 Hz + up to one frame of waiting
IMAGE_HOLD_PROB = 0.1  # a dropped frame repeats the previous one
IMAGE_AUGMENTATION_CFG = RgbAugmentationCfg()

CLOTH_COLORS = {
    "palette": [
        (0.67, 0.74, 0.55),  # the task's light-green microfiber cloth (20 x 20 cm)
        (0.70, 0.78, 0.52),  # slightly more saturated
        (0.71, 0.75, 0.55),  # towards yellow
        (0.64, 0.73, 0.57),  # towards blue-green
    ],
    "brightness": (1.0, 1.3),  # under the randomized light this centers the rendered cloth on the photo's color
    "jitter": 0.05,
}
"""Per-episode cloth color (sRGB), light variations of the task's green cloth. The cloth's color is the median of five
cloth patches in a phone photo on the beech table; the orange cube in the same photo measured (0.99, 0.48, 0.14)
against its official (1.0, 0.40, 0.12), so the photo's white balance is taken as neutral. Under the nominal light the
recolored cloth renders at this color (0.671, 0.741, 0.549). Refit it from a wrist-camera frame of the cloth when one is
available."""

##
# Scene appearance and camera mount (events)
##

NEUTRAL_TINTS = [(1.0, 1.0, 1.0), (1.0, 0.97, 0.93), (0.95, 0.97, 1.0)]

SHAPE_COLORS = {
    "Robot": {"palette": NEUTRAL_TINTS, "brightness": (0.85, 1.05), "jitter": 0.02, "tint": True},
    "Cube": {  # Bambu PETG Basic Orange 30301, #FF671F; glossy, slightly translucent
        "palette": [
            (1.00, 0.40, 0.12),  # official #FF671F
            (0.93, 0.33, 0.08),  # deeper / redder: translucent body, faces away from the light
            (1.00, 0.47, 0.17),  # lighter / yellower: warm light, print-to-print variation
            (1.00, 0.55, 0.33),  # glossy sheen washing toward the light color (no specular in the renderer)
        ],
        "brightness": (0.75, 1.1),  # > 1 clips red at 1.0 and shifts the hue toward yellow
        "jitter": 0.04,
    },
    "Table": {  # beech-like around the real table's sRGB (0.68, 0.61, 0.51)
        "palette": [(0.68, 0.61, 0.51), (0.72, 0.62, 0.48), (0.64, 0.58, 0.50), (0.70, 0.64, 0.56)],
        "brightness": (0.9, 1.2),  # the real table is light wood; darker episodes looked too dark
        "jitter": 0.05,
    },
}
"""Per-episode shape colors (sRGB) of the rendered scene prims."""

CAMERA_POS_RANGE = 0.01  # [m] per axis
CAMERA_ROT_RANGE = math.radians(1.0)  # [rad] per axis
LIGHT_INTERVAL_S = (2.0, 5.0)  # [s]
LIGHT_ELEVATION = (math.radians(35.0), math.radians(88.0))  # [rad]

##
# Nominal values (play mode): the first palette entry of each color, no jitter, a fixed light
##


def _nominal(spec: dict) -> dict:
    """The first palette entry of a color distribution, at brightness 1 and without jitter."""
    return {**spec, "palette": spec["palette"][:1], "brightness": (1.0, 1.0), "jitter": 0.0}


NOMINAL_CLOTH_COLOR = _nominal(CLOTH_COLORS)
"""The task's green cloth."""

NOMINAL_SHAPE_COLORS = {name: _nominal(spec) for name, spec in SHAPE_COLORS.items()}
"""Untinted robot, official cube orange, the real table's beech."""

NOMINAL_LIGHT_ELEVATION = math.radians(70.0)  # [rad], inside LIGHT_ELEVATION; not measured
NOMINAL_LIGHT_AZIMUTH = 0.0  # [rad]
NOMINAL_IMAGE_LATENCY_STEPS = (2, 2)  # policy steps, ~66 ms capture-to-publish at 30 Hz


def add_nominal_visuals(env_cfg, sensor_name: str = "wrist_camera") -> None:
    """Add a fixed light and the nominal shape colors to ``env_cfg.events``; the camera keeps its nominal mount."""
    sensor_cfg = SceneEntityCfg(sensor_name)
    env_cfg.events.nominal_light = EventTerm(
        func=randomize_light,
        mode="reset",  # re-set to the same direction at every reset
        params={
            "sensor_cfg": sensor_cfg,
            "elevation": (NOMINAL_LIGHT_ELEVATION, NOMINAL_LIGHT_ELEVATION),
            "azimuth": (NOMINAL_LIGHT_AZIMUTH, NOMINAL_LIGHT_AZIMUTH),
        },
    )
    env_cfg.events.nominal_colors = EventTerm(
        func=randomize_shape_colors,
        mode="reset",
        params={"sensor_cfg": sensor_cfg, "colors": NOMINAL_SHAPE_COLORS},
    )


def add_visual_randomization(env_cfg, sensor_name: str = "wrist_camera") -> None:
    """Add the camera mount, light and shape color events to ``env_cfg.events``."""
    sensor_cfg = SceneEntityCfg(sensor_name)
    env_cfg.events.randomize_wrist_camera = EventTerm(
        func=randomize_camera_offset,
        mode="reset",
        params={
            "sensor_cfg": sensor_cfg,
            "parent_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "pos_range": CAMERA_POS_RANGE,
            "rot_range": CAMERA_ROT_RANGE,
        },
    )
    env_cfg.events.randomize_light = EventTerm(
        func=randomize_light,
        mode="interval",
        interval_range_s=LIGHT_INTERVAL_S,
        is_global_time=True,  # one light shared by all envs
        params={"sensor_cfg": sensor_cfg, "elevation": LIGHT_ELEVATION},
    )
    env_cfg.events.randomize_colors = EventTerm(
        func=randomize_shape_colors,
        mode="reset",
        params={"sensor_cfg": sensor_cfg, "colors": SHAPE_COLORS},
    )

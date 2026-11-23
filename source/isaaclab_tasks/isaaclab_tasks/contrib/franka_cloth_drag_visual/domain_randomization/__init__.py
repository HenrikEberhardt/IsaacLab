# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera-side randomization: mount, light, colors, cloth recolor, photometric augmentation, latency and drops."""

from .appearance import ClothRecolor, randomize_light, randomize_shape_colors
from .camera_mount import randomize_camera_offset
from .image_augmentation import RgbAugmentation, RgbAugmentationCfg
from .visual_randomization_cfg import (
    CLOTH_COLORS,
    IMAGE_AUGMENTATION_CFG,
    IMAGE_HOLD_PROB,
    IMAGE_LATENCY_STEPS,
    NOMINAL_CLOTH_COLOR,
    NOMINAL_IMAGE_LATENCY_STEPS,
    SHAPE_COLORS,
    add_nominal_visuals,
    add_visual_randomization,
)

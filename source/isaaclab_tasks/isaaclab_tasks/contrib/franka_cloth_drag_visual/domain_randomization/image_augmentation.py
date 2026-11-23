# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# follows srl-ethz/IsaacLab-Cloth (cloth-fold-environments), src/Cloth_Manipulation/visual/lab_scene.py; + grayscale

"""Photometric camera-vs-renderer augmentation of RGB images, one fused Warp kernel."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg, NoiseModel, NoiseModelCfg


@wp.func
def _graded_pixel(image: wp.array4d(dtype=float), params: wp.array2d(dtype=float), i: int, y: int, x: int) -> wp.vec3:
    """Contrast, saturation, exposure, white balance, vignetting and gamma of one pixel."""
    height = image.shape[2]
    width = image.shape[3]
    mean = wp.vec3(params[i, 0])
    rgb = (wp.vec3(image[i, 0, y, x], image[i, 1, y, x], image[i, 2, y, x]) - mean) * params[i, 1] + mean
    gray = wp.vec3((rgb[0] + rgb[1] + rgb[2]) / 3.0)
    rgb = (rgb - gray) * params[i, 2] + gray
    rgb = wp.cw_mul(rgb * params[i, 3], wp.vec3(params[i, 4], params[i, 5], params[i, 6]))
    fy = (-1.0 + 2.0 * float(y) / float(height - 1)) * float(height)
    fx = (-1.0 + 2.0 * float(x) / float(width - 1)) * float(width)
    rgb = rgb * (1.0 - params[i, 7] * (fx * fx + fy * fy) / float(width * width + height * height))  # vignette
    gamma = params[i, 8]
    return wp.vec3(
        wp.pow(wp.clamp(rgb[0], 0.0, 1.0), gamma),
        wp.pow(wp.clamp(rgb[1], 0.0, 1.0), gamma),
        wp.pow(wp.clamp(rgb[2], 0.0, 1.0), gamma),
    )


@wp.func
def _defocused_pixel(
    image: wp.array4d(dtype=float), params: wp.array2d(dtype=float), i: int, y: int, x: int
) -> wp.vec3:
    """Graded pixel, 3x3 box-blurred for defocused images."""
    if params[i, 9] == 0.0:
        return _graded_pixel(image, params, i, y, x)
    total = wp.vec3(0.0)
    count = float(0.0)  # noqa: UP018 (Warp needs a typed variable to mutate in a loop)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if y + dy >= 0 and y + dy < image.shape[2] and x + dx >= 0 and x + dx < image.shape[3]:
                total += _graded_pixel(image, params, i, y + dy, x + dx)
                count += 1.0
    return total / count


@wp.kernel
def _rgb_augmentation_kernel(
    image: wp.array4d(dtype=float),
    params: wp.array2d(dtype=float),
    along_y: int,
    seed: int,
    out: wp.array4d(dtype=float),
):
    i, y, x = wp.tid()
    rgb = wp.vec3(0.0)
    if params[i, 10] == 0.0:
        rgb = _defocused_pixel(image, params, i, y, x)
    else:
        count = float(0.0)  # noqa: UP018
        for t in range(-2, 3):  # 5 px motion blur along one axis
            yy = y + t * along_y
            xx = x + t * (1 - along_y)
            if yy >= 0 and yy < image.shape[2] and xx >= 0 and xx < image.shape[3]:
                rgb += _defocused_pixel(image, params, i, yy, xx)
                count += 1.0
        rgb = rgb / count
    state = wp.rand_init(seed, (i * image.shape[2] + y) * image.shape[3] + x)
    for c in range(3):
        noise_std = params[i, 11] + params[i, 12] * wp.sqrt(rgb[c])
        out[i, c, y, x] = wp.clamp(rgb[c] + noise_std * wp.randn(state), 0.0, 1.0)


class RgbAugmentation(NoiseModel):
    """Augments [N, 3, H, W] images in [0, 1]; grading/defocus/noise level per episode, noise and motion blur per step."""

    def __init__(self, cfg: RgbAugmentationCfg, num_envs: int, device: str):
        super().__init__(cfg, num_envs, device)
        # mean, contrast, saturation, exposure, 3 gains, vignette, gamma, defocus, motion blur, read noise, shot noise
        self._params = torch.zeros(num_envs, 13, device=device)
        self.reset()

    def reset(self, env_ids: Sequence[int] | None = None):
        cfg = self._noise_model_cfg
        ids = slice(None) if env_ids is None else env_ids
        n = self._params[ids].shape[0]

        def uniform(bounds: tuple[float, float], size: int = 1) -> torch.Tensor:
            return math_utils.sample_uniform(*bounds, (n, size), self._device)

        defocus = (torch.rand(n, 1, device=self._device) < cfg.blur_prob).float()
        color = (torch.rand(n, 1, device=self._device) >= cfg.grayscale_prob).float()
        self._params[ids, 1:10] = torch.cat(
            [
                uniform(cfg.contrast_range),
                uniform(cfg.saturation_range) * color,
                uniform(cfg.brightness_range),
                uniform(cfg.channel_gain_range, 3),
                uniform(cfg.vignette_range),
                uniform(cfg.gamma_range),
                defocus,
            ],
            dim=1,
        )
        self._params[ids, 11:13] = torch.cat([uniform(cfg.noise_std_range), uniform(cfg.shot_noise_range)], dim=1)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        params = self._params.clone()
        params[:, 0] = data.mean(dim=(1, 2, 3))
        params[:, 10] = (torch.rand(len(data), device=data.device) < self._noise_model_cfg.motion_blur_prob).float()
        image = data.contiguous()
        out = torch.empty_like(image)
        along_y, seed = int(torch.rand(1) < 0.5), int(torch.randint(0, 2**31 - 1, (1,)))  # blur axis shared per step
        wp.launch(
            _rgb_augmentation_kernel,
            dim=image.shape[:1] + image.shape[2:],
            inputs=[wp.from_torch(image), wp.from_torch(params), along_y, seed],
            outputs=[wp.from_torch(out)],
            stream=wp.stream_from_torch(torch.cuda.current_stream(data.device)),
        )
        return out


@configclass
class RgbAugmentationCfg(NoiseModelCfg):
    """Configuration for :class:`RgbAugmentation` (Joran's ranges)."""

    class_type: type = RgbAugmentation

    noise_cfg: NoiseCfg | None = None
    """Unused."""

    brightness_range: tuple[float, float] = (0.65, 1.35)
    """Multiplicative exposure."""

    contrast_range: tuple[float, float] = (0.7, 1.3)
    """Scale of the deviation from the image mean."""

    saturation_range: tuple[float, float] = (0.6, 1.3)
    """Scale of the deviation from the per-pixel gray value."""

    grayscale_prob: float = 0.05
    """Probability of a grayscale episode."""

    channel_gain_range: tuple[float, float] = (0.87, 1.13)
    """Per-channel gain (white balance)."""

    vignette_range: tuple[float, float] = (0.0, 0.4)
    """Brightness loss in the corners."""

    gamma_range: tuple[float, float] = (0.8, 1.25)
    """Exponent after clamping to [0, 1]."""

    blur_prob: float = 0.2
    """Probability of a 3x3 defocus blur per episode."""

    motion_blur_prob: float = 0.1
    """Probability of a 5 px axis blur per step."""

    noise_std_range: tuple[float, float] = (0.0, 0.015)
    """Per-episode std of additive read noise."""

    shot_noise_range: tuple[float, float] = (0.0, 0.03)
    """Per-episode shot-noise scale (std grows with sqrt of intensity)."""

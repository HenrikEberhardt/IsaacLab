# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# randomize_light, randomize_shape_colors and RgbAugmentation follow srl-ethz/IsaacLab-Cloth (branch
# cloth-fold-environments), src/Cloth_Manipulation/visual/lab_scene.py; RgbAugmentation gained a grayscale probability.

"""Visual randomization of the Newton Warp renderer: light direction, shape colors and photometric augmentation.

The renderer has no HDRI, light intensity or light color and no roughness, metallic or specular parameters. The light
direction and shadows, the albedo of every shape and the camera-side image statistics are what can be randomized.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg, NoiseModel, NoiseModelCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import EventTermCfg


def _newton_sensor(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg):
    """The Newton tiled camera behind a camera's Warp renderer (private: Isaac Lab exposes no handle)."""
    return env.scene.sensors[sensor_cfg.name]._renderer.newton_sensor


def randomize_light(
    env: ManagerBasedEnv, env_ids: torch.Tensor | slice, sensor_cfg: SceneEntityCfg, elevation: tuple[float, float]
):
    """Point the shadow-casting directional light from a random elevation [rad] and azimuth.

    The light is shared by every environment, so use it as a global-time interval event.
    """
    angle = math_utils.sample_uniform(*elevation, (1,), "cpu").item()
    azimuth = math_utils.sample_uniform(-math.pi, math.pi, (1,), "cpu").item()
    to_light = (math.cos(angle) * math.cos(azimuth), math.cos(angle) * math.sin(azimuth), math.sin(angle))
    _newton_sensor(env, sensor_cfg).utils.create_default_light(
        enable_shadows=True, direction=wp.vec3f(*(-c for c in to_light))
    )


class randomize_shape_colors(ManagerTermBase):
    """Per environment, recolor every rendered shape of the named scene prims.

    ``colors`` maps a prim name under the environment namespace (e.g. ``"Cube"``) to a ``palette`` of sRGB tuples (one
    picked per environment), a ``brightness`` (lo, hi) scale, a relative per-channel ``jitter`` and an optional
    ``tint`` (True multiplies the authored colors instead of replacing them).
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        model = _newton_sensor(env, cfg.params["sensor_cfg"]).model
        self._shape_colors = wp.to_torch(model.shape_color)
        pattern = re.compile(r"^/World/envs/env_(\d+)/([^/]+)")
        shapes: dict[str, list[list[int]]] = {name: [[] for _ in range(env.num_envs)] for name in cfg.params["colors"]}
        for shape_index, label in enumerate(model.shape_label):
            match = pattern.match(label)
            if match and match.group(2) in shapes:
                shapes[match.group(2)][int(match.group(1))].append(shape_index)
        self._shapes = {}
        for name, per_env in shapes.items():
            if not per_env[0]:
                raise ValueError(f"No rendered shapes found under '/World/envs/env_0/{name}'.")
            self._shapes[name] = torch.tensor(per_env, dtype=torch.long, device=self._shape_colors.device)
        self._authored = {name: self._shape_colors[ids].clone() for name, ids in self._shapes.items()}

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice,
        sensor_cfg: SceneEntityCfg,
        colors: dict[str, dict],
    ):
        device = self._shape_colors.device
        env_ids = torch.arange(env.num_envs, device=device)[env_ids]
        n = len(env_ids)
        for name, spec in colors.items():
            palette = torch.tensor(spec["palette"], dtype=torch.float32, device=device)
            color = palette[torch.randint(len(palette), (n,), device=device)]
            color = color * math_utils.sample_uniform(*spec["brightness"], (n, 1), device)
            color = color * math_utils.sample_uniform(1.0 - spec["jitter"], 1.0 + spec["jitter"], (n, 3), device)
            shapes = self._shapes[name][env_ids]
            if spec.get("tint", False):
                values = self._authored[name][env_ids] * color[:, None]
            else:
                values = color[:, None].expand(-1, shapes.shape[1], -1)
            self._shape_colors[shapes.flatten()] = values.reshape(-1, 3).clamp(0.0, 1.0)
        # the renderer reads the colors on Warp's stream; finish the writes first
        torch.cuda.current_stream(device).synchronize()


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
    # radial falloff, 1 at the center and 1 - strength in the corners
    fy = (-1.0 + 2.0 * float(y) / float(height - 1)) * float(height)
    fx = (-1.0 + 2.0 * float(x) / float(width - 1)) * float(width)
    rgb = rgb * (1.0 - params[i, 7] * (fx * fx + fy * fy) / float(width * width + height * height))
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
    """Graded pixel, 3x3 box-blurred for the images drawn to be blurred (borders average the valid pixels)."""
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
        # 5 px motion blur along one image axis
        count = float(0.0)  # noqa: UP018
        for t in range(-2, 3):
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
    """Photometric augmentation of ``(N, 3, H, W)`` images in [0, 1].

    Covers exposure, contrast, saturation (zero for grayscale episodes), white balance, gamma and lens vignetting
    differences between renderer and camera, plus signal-dependent sensor noise, slight defocus and motion blur. The
    grading, defocus and noise levels are drawn per environment at reset and held for the episode; the noise itself
    and the motion blur are drawn every step. One fused Warp kernel applies them.
    """

    def __init__(self, cfg: RgbAugmentationCfg, num_envs: int, device: str):
        super().__init__(cfg, num_envs, device)
        # kernel layout: image mean, contrast, saturation, exposure, 3 channel gains, vignette, gamma, defocus,
        # motion blur, read noise, shot noise
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
        # the motion-blur axis is shared by the images of this step
        along_y, seed = int(torch.rand(1) < 0.5), int(torch.randint(0, 2**31 - 1, (1,)))
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
    """Configuration for :class:`RgbAugmentation`."""

    class_type: type = RgbAugmentation

    noise_cfg: NoiseCfg | None = None
    """Unused: :class:`RgbAugmentation` draws its own parameters."""

    brightness_range: tuple[float, float] = (0.65, 1.35)
    """Multiplicative exposure."""

    contrast_range: tuple[float, float] = (0.7, 1.3)
    """Scale of the deviation from the image mean."""

    saturation_range: tuple[float, float] = (0.6, 1.3)
    """Scale of the deviation from the per-pixel gray value."""

    grayscale_prob: float = 0.05
    """Probability of a grayscale episode (saturation zero)."""

    channel_gain_range: tuple[float, float] = (0.87, 1.13)
    """Per-channel gain, i.e. white balance."""

    vignette_range: tuple[float, float] = (0.0, 0.4)
    """Brightness loss in the image corners."""

    gamma_range: tuple[float, float] = (0.8, 1.25)
    """Exponent applied after clamping to [0, 1]."""

    blur_prob: float = 0.2
    """Probability of a 3x3 box blur (defocus) for an episode."""

    motion_blur_prob: float = 0.1
    """Probability of a 5 px blur along one image axis, per step."""

    noise_std_range: tuple[float, float] = (0.0, 0.015)
    """Range of the per-episode standard deviation of additive Gaussian read noise."""

    shot_noise_range: tuple[float, float] = (0.0, 0.03)
    """Range of the per-episode shot-noise scale; the noise std grows with the square root of intensity."""

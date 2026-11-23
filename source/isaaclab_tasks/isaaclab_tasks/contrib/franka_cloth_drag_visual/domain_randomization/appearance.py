# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# follows srl-ethz/IsaacLab-Cloth (cloth-fold-environments), src/Cloth_Manipulation/visual/lab_scene.py and mdp.py

"""Scene appearance in the Newton Warp renderer: light direction, shape colors and cloth color."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import EventTermCfg

CLOTH_SHAPE_ID = -3
"""Newton's TRIANGLE_MESH_SHAPE_ID (0xFFFFFFFD) read as int32: a hit on a deformable's triangles."""


def _newton_sensor(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg):
    """The Newton tiled camera behind a camera's Warp renderer (private handle)."""
    return env.scene.sensors[sensor_cfg.name]._renderer.newton_sensor


def srgb_to_linear(srgb: torch.Tensor) -> torch.Tensor:
    """sRGB to linear RGB, both in [0, 1]."""
    return torch.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(linear: torch.Tensor) -> torch.Tensor:
    """Linear RGB to sRGB, both in [0, 1]."""
    return torch.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear.clamp_min(1.0e-12) ** (1.0 / 2.4) - 0.055)


def sample_colors(spec: dict, n: int, device) -> torch.Tensor:
    """n sRGB colors: a palette entry times a brightness and a per-channel jitter, shape [n, 3]."""
    palette = torch.tensor(spec["palette"], dtype=torch.float32, device=device)
    color = palette[torch.randint(len(palette), (n,), device=device)]
    color = color * math_utils.sample_uniform(*spec["brightness"], (n, 1), device)
    return color * math_utils.sample_uniform(1.0 - spec["jitter"], 1.0 + spec["jitter"], (n, 3), device)


def randomize_light(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice,
    sensor_cfg: SceneEntityCfg,
    elevation: tuple[float, float],
    azimuth: tuple[float, float] = (-math.pi, math.pi),
):
    """Point the shared shadow-casting light from a random elevation and azimuth [rad] (global-time event)."""
    angle = math_utils.sample_uniform(*elevation, (1,), "cpu").item()
    azimuth = math_utils.sample_uniform(*azimuth, (1,), "cpu").item()
    to_light = (math.cos(angle) * math.cos(azimuth), math.cos(angle) * math.sin(azimuth), math.sin(angle))
    _newton_sensor(env, sensor_cfg).utils.create_default_light(
        enable_shadows=True, direction=wp.vec3f(*(-c for c in to_light))
    )


class randomize_shape_colors(ManagerTermBase):
    """Per env, recolor the rendered shapes of named scene prims; colors[name] = palette, brightness, jitter, tint."""

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
        for name, spec in colors.items():
            color = sample_colors(spec, len(env_ids), device)
            shapes = self._shapes[name][env_ids]
            if spec.get("tint", False):
                values = self._authored[name][env_ids] * color[:, None]  # tint the authored colors
            else:
                values = color[:, None].expand(-1, shapes.shape[1], -1)
            self._shape_colors[shapes.flatten()] = values.reshape(-1, 3).clamp(0.0, 1.0)
        torch.cuda.current_stream(device).synchronize()  # the renderer reads on Warp's stream


class ClothRecolor:
    """Per-episode cloth color, multiplied onto the cloth's pixels in linear RGB (Newton renders cloth white)."""

    def __init__(self, colors: dict, num_envs: int, device):
        self._colors = colors
        self._albedo = torch.ones(num_envs, 3, device=device)  # linear RGB

    def reset(self, env_ids: torch.Tensor) -> None:
        """Draw new cloth colors."""
        color = sample_colors(self._colors, len(env_ids), self._albedo.device)
        self._albedo[env_ids] = srgb_to_linear(color.clamp(0.0, 1.0))

    def __call__(self, env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, raw: torch.Tensor) -> torch.Tensor:
        """Recolor the uint8 sRGB frames [N, H, W, 3]; needs a segmentation output on the camera."""
        shape_index = env.scene.sensors[sensor_cfg.name]._render_data.outputs.shape_index_image  # private
        if shape_index is None:
            raise RuntimeError("Cloth recoloring needs a segmentation output in the camera's data_types.")
        cloth = wp.to_torch(shape_index)[:, 0, :, :, None] == CLOTH_SHAPE_ID  # (world, camera, H, W) as int32
        linear = srgb_to_linear(raw.float() / 255.0) * self._albedo[:, None, None]
        recolored = (linear_to_srgb(linear) * 255.0).round().to(raw.dtype)
        return torch.where(cloth, recolored, raw)

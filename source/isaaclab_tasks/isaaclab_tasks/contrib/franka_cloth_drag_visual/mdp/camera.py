# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# randomize_camera_offset and image_rgb_deploy follow srl-ethz/IsaacLab-Cloth (branch cloth-fold-environments),
# src/Cloth_Manipulation/visual/mdp.py, which copies them from mmichelis/IsaacLab-Soft (branch liftcube-rgb-lab).

"""Wrist camera terms: the mount randomization and the delayed, recolored RGB observation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.buffers import DelayBuffer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import ObservationTermCfg


def randomize_camera_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice,
    sensor_cfg: SceneEntityCfg,
    parent_cfg: SceneEntityCfg,
    pos_range: float,
    rot_range: float,
):
    """Perturb a camera's mount on its parent body per environment, to tolerate calibration error.

    The configured offset is moved by a uniform position offset in the parent frame within ``pos_range`` [m] and
    turned by uniform XYZ Euler angles in the camera frame within ``rot_range`` [rad].
    """
    env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    camera, parent = env.scene.sensors[sensor_cfg.name], env.scene[parent_cfg.name]
    offset, n = camera.cfg.offset, len(env_ids)
    pos = torch.tensor(offset.pos, device=env.device) + math_utils.sample_uniform(
        -pos_range, pos_range, (n, 3), env.device
    )
    euler = math_utils.sample_uniform(-rot_range, rot_range, (n, 3), env.device)
    rot = math_utils.quat_mul(
        torch.tensor(offset.rot, device=env.device).expand(n, 4),
        math_utils.quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2]),
    )
    body = parent_cfg.body_ids[0]
    world_pos, world_rot = math_utils.combine_frame_transforms(
        parent.data.body_pos_w.torch[env_ids, body], parent.data.body_quat_w.torch[env_ids, body], pos, rot
    )
    camera.set_world_poses(world_pos, world_rot, env_ids, convention=offset.convention)


_CLOTH_SHAPE_ID = -3
"""Shape index of a hit on a deformable's simulation triangles, Newton's ``TRIANGLE_MESH_SHAPE_ID`` (0xFFFFFFFD)
read as int32."""


def _srgb_to_linear(srgb: torch.Tensor) -> torch.Tensor:
    return torch.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(linear: torch.Tensor) -> torch.Tensor:
    return torch.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear.clamp_min(1.0e-12) ** (1.0 / 2.4) - 0.055)


class image_rgb_deploy(ManagerTermBase):
    """Wrist RGB image as the real camera delivers it: [0, 1] minus ``mean``, shape [N, 3, H, W] (channel first).

    * **Latency.** The frame is delayed by ``latency_steps`` = (min, max) policy steps, drawn per environment at
      reset. The real camera publishes 30 fps frames about 66 ms after capture, and the policy reads the latest one up
      to a frame period later. After a reset the delay line holds only new frames.
    * **Frame drops.** With probability ``hold_prob`` per step and environment, the previous frame is repeated.
    * **Cloth color (workaround).** Newton's tiled camera shades the cloth's triangles white and ignores
      ``Model.tri_color``. With ``cloth_colors``, each episode draws a cloth color (sRGB) and the cloth's pixels are
      multiplied by it in linear RGB. The cloth is found through the renderer's private shape-index buffer, which the
      camera fills only when a segmentation output is requested.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._latency = cfg.params.get("latency_steps", (0, 0))
        self._delay = DelayBuffer(self._latency[1], env.num_envs, env.device) if self._latency[1] > 0 else None
        self._previous: torch.Tensor | None = None
        self._cloth_colors = cfg.params.get("cloth_colors")
        # linear-RGB cloth albedo per environment; white is the renderer's own
        self._cloth_albedo = torch.ones(env.num_envs, 3, device=env.device)
        self._fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.reset(None)

    def reset(self, env_ids=None):
        env_ids = torch.arange(self.num_envs, device=self.device)[slice(None) if env_ids is None else env_ids]
        self._fresh[env_ids] = True
        if self._delay is not None:
            self._delay.reset(env_ids)
            lags = torch.randint(self._latency[0], self._latency[1] + 1, (len(env_ids),), device=self.device)
            self._delay.set_time_lag(lags.int(), env_ids)
        if self._cloth_colors is not None:
            spec, n = self._cloth_colors, len(env_ids)
            palette = torch.tensor(spec["palette"], device=self.device)
            color = palette[torch.randint(len(palette), (n,), device=self.device)]
            color = color * math_utils.sample_uniform(*spec["brightness"], (n, 1), self.device)
            color = color * math_utils.sample_uniform(1.0 - spec["jitter"], 1.0 + spec["jitter"], (n, 3), self.device)
            self._cloth_albedo[env_ids] = _srgb_to_linear(color.clamp(0.0, 1.0))

    def _recolor_cloth(self, env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, raw: torch.Tensor) -> torch.Tensor:
        """Multiply the cloth's pixels of the uint8 sRGB frame by each environment's cloth albedo."""
        shape_index = env.scene.sensors[sensor_cfg.name]._render_data.outputs.shape_index_image
        if shape_index is None:
            raise RuntimeError("cloth_colors needs a segmentation output in the camera's data_types.")
        # (world, camera, height, width) uint32, which Warp hands to torch as int32
        cloth = wp.to_torch(shape_index)[:, 0, :, :, None] == _CLOTH_SHAPE_ID
        linear = _srgb_to_linear(raw.float() / 255.0) * self._cloth_albedo[:, None, None]
        recolored = (_linear_to_srgb(linear) * 255.0).round().to(raw.dtype)
        return torch.where(cloth, recolored, raw)

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg,
        mean: float = 0.0,
        latency_steps: tuple[int, int] = (0, 0),
        hold_prob: float = 0.0,
        cloth_colors: dict | None = None,
    ) -> torch.Tensor:
        raw = env.scene.sensors[sensor_cfg.name].data.output["rgb"].torch[..., :3]
        if self._cloth_colors is not None:
            raw = self._recolor_cloth(env, sensor_cfg, raw)
        if self._delay is not None:
            raw = self._delay.compute(raw)
        image = (raw.float() / 255.0 - mean).permute(0, 3, 1, 2)
        if self._previous is None:
            self._previous = image.clone()
        hold = (torch.rand(self.num_envs, device=self.device) < hold_prob) & ~self._fresh
        image = torch.where(hold[:, None, None, None], self._previous, image)
        self._previous.copy_(image)
        self._fresh[:] = False
        return image

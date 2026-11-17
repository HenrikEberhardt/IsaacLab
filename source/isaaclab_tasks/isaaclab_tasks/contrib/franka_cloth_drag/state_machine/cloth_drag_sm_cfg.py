# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration and per-rollout parameters for the cloth-drag state machine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from isaaclab.utils import configclass

ClothDragSmProfile = Literal["nominal", "randomized"]
"""Named state-machine profiles."""

PROFILE_NAMES: tuple[ClothDragSmProfile, ...] = ("nominal", "randomized")
"""Valid profile names."""


@configclass
class ClothDragSmCfg:
    """Fixed controller settings."""

    arm_action_limit: float = 0.85
    """Largest absolute normalized arm-action component, applied by uniformly scaling the command."""

    hover_height: float = 0.03
    """Hover height above the grasp node [m]."""

    grasp_height: float = 0.0195
    """End-effector height during the grasp [m]."""

    drag_height: float = 0.031
    """Default end-effector height during the drag [m]."""

    drag_speed: float = 0.08
    """Default planar drag speed [m/s]."""

    goal_tolerance: float = 0.02
    """Planar goal distance at which the controller starts holding [m]."""

    approach_threshold: float = 0.01
    """Position tolerance for the pre-grasp and hover targets [m]."""

    grasp_threshold: float = 0.005
    """Position tolerance for the grasp target [m]."""

    yaw_tolerance: float = 0.1
    """Yaw tolerance before descending [rad]."""

    pregrasp_hold_time: float = 0.5
    """Time the reached pre-grasp pose is held [s]."""

    hover_hold_time: float = 0.3
    """Time the reached hover pose is held [s]."""

    descend_hold_time: float = 0.3
    """Time the reached grasp pose is held before closing [s]."""

    grasp_squeeze_time: float = 0.3
    """Minimum squeeze time before starting the drag [s]."""

    grasp_timeout: float = 2.0
    """Time allowed for one grasp attempt [s]."""

    hold_timeout: float = 2.0
    """Time allowed for task success while holding at the target [s]."""

    drag_max_lead: float = 0.03
    """Maximum distance by which the drag target leads the end effector [m]."""

    drag_yaw_tolerance: float = 0.2
    """Cloth yaw error that starts a correction during dragging [rad]."""

    drag_yaw_rate: float = 0.5
    """Maximum yaw correction rate during dragging [rad/s]."""

    stall_step: float = 0.0001
    """Largest per-step end-effector displacement that counts as stalled [m]."""

    stall_time: float = 0.5
    """Continuous stalled time that fails a rollout [s]."""

    def __post_init__(self):
        if not 0.0 < self.arm_action_limit <= 1.0:
            raise ValueError(f"arm_action_limit must lie in (0, 1], got {self.arm_action_limit}.")


@configclass
class ClothDragSmRandomizationCfg:
    """Ranges sampled independently for each randomized rollout."""

    drag_speed_range: tuple[float, float] = (0.05, 0.12)
    """Drag-speed range [m/s]."""

    drag_height_range: tuple[float, float] = (0.015, 0.030)
    """Drag-height range [m]."""

    yaw_offset_range: tuple[float, float] = (-0.3, 0.3)
    """Grasp-yaw offset range [rad]."""

    pregrasp_height_range: tuple[float, float] = (0.005, 0.025)
    """Pre-grasp height above the grasp node [m]."""

    pregrasp_radius_range: tuple[float, float] = (0.020, 0.025)
    """Maximum pre-grasp planar radius at the lowest and highest sampled heights [m]."""

    pregrasp_yaw_error_range: tuple[float, float] = (-0.79, 0.79)
    """Pre-grasp yaw-error range [rad]."""

    def __post_init__(self):
        for name in (
            "drag_speed_range",
            "drag_height_range",
            "yaw_offset_range",
            "pregrasp_height_range",
            "pregrasp_radius_range",
            "pregrasp_yaw_error_range",
        ):
            low, high = getattr(self, name)
            if low > high:
                raise ValueError(f"{name} must be increasing, got {(low, high)}.")


@dataclass
class ClothDragSmParams:
    """Per-environment controller parameters, with first dimension ``num_envs``.

    Heights and offsets are in meters, speeds in meters per second, and angles in radians.
    """

    drag_height: torch.Tensor
    drag_speed: torch.Tensor
    yaw_offset: torch.Tensor
    pregrasp_offset: torch.Tensor
    pregrasp_yaw_error: torch.Tensor

    def select(self, env_ids: torch.Tensor) -> ClothDragSmParams:
        """Select the parameters of the given environments."""
        return ClothDragSmParams(
            drag_height=self.drag_height[env_ids],
            drag_speed=self.drag_speed[env_ids],
            yaw_offset=self.yaw_offset[env_ids],
            pregrasp_offset=self.pregrasp_offset[env_ids],
            pregrasp_yaw_error=self.pregrasp_yaw_error[env_ids],
        )


def make_sm_profile(
    profile: ClothDragSmProfile,
) -> tuple[ClothDragSmCfg, ClothDragSmRandomizationCfg | None]:
    """Create the fixed and randomized settings of a named profile."""
    if profile == "nominal":
        return ClothDragSmCfg(pregrasp_hold_time=0.0), None
    if profile == "randomized":
        return ClothDragSmCfg(grasp_height=0.0205), ClothDragSmRandomizationCfg()
    raise ValueError(f"Unknown state-machine profile {profile!r}; expected one of {PROFILE_NAMES}.")


def sample_sm_params(
    cfg: ClothDragSmCfg,
    randomization: ClothDragSmRandomizationCfg | None,
    num_envs: int,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> ClothDragSmParams:
    """Sample one parameter row per environment, or return the fixed profile values."""
    if randomization is None:
        return ClothDragSmParams(
            drag_height=torch.full((num_envs,), cfg.drag_height, device=device),
            drag_speed=torch.full((num_envs,), cfg.drag_speed, device=device),
            yaw_offset=torch.zeros(num_envs, device=device),
            pregrasp_offset=torch.tensor((0.0, 0.0, cfg.hover_height), device=device).expand(num_envs, 3).clone(),
            pregrasp_yaw_error=torch.zeros(num_envs, device=device),
        )

    height = _uniform(randomization.pregrasp_height_range, num_envs, device, generator)
    h_low, h_high = randomization.pregrasp_height_range
    r_low, r_high = randomization.pregrasp_radius_range
    share = (height - h_low) / max(h_high - h_low, 1.0e-9)
    radius = (r_low + (r_high - r_low) * share) * _uniform((0.0, 1.0), num_envs, device, generator)
    angle = _uniform((0.0, 2.0 * math.pi), num_envs, device, generator)
    offset = torch.stack((radius * torch.cos(angle), radius * torch.sin(angle), height), dim=1)
    return ClothDragSmParams(
        drag_height=_uniform(randomization.drag_height_range, num_envs, device, generator),
        drag_speed=_uniform(randomization.drag_speed_range, num_envs, device, generator),
        yaw_offset=_uniform(randomization.yaw_offset_range, num_envs, device, generator),
        pregrasp_offset=offset,
        pregrasp_yaw_error=_uniform(randomization.pregrasp_yaw_error_range, num_envs, device, generator),
    )


def _uniform(
    bounds: tuple[float, float],
    num: int,
    device: torch.device | str,
    generator: torch.Generator | None,
) -> torch.Tensor:
    low, high = bounds
    return low + (high - low) * torch.rand(num, device=device, generator=generator)

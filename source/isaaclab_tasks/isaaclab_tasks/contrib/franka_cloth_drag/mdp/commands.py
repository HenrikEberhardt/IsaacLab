# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command terms for the Franka cloth dragging task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import combine_frame_transforms

from .frames import cloth_com_pos_b

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, DeformableObject
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import ClothTargetCommandCfg


class ClothTargetCommand(CommandTerm):
    """Target position of the cloth's center of mass in the robot's root frame.

    The target is sampled uniformly from :attr:`ClothTargetCommandCfg.ranges` whenever the command is resampled, by
    default only when the episode resets. The goal is reached when the cloth's center of mass rests at the target, see
    :class:`ClothTargetCommandCfg`.

    Metrics, logged at reset:

    * ``position_error_xy``: planar distance of the cloth's center of mass to the target [m].
    * ``success``: one if the goal was reached during the episode, so its mean is the episode success rate.
    * ``com_speed_at_target``: speed of the cloth's center of mass [m/s], averaged over the steps of the last
      :attr:`ClothTargetCommandCfg.speed_window` steps in which it was inside the success radius. Environments whose
      cloth was not inside in that window are left out; the metric is omitted when none was inside.
    """

    cfg: ClothTargetCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: ClothTargetCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.cloth: DeformableObject = env.scene[cfg.object_name]
        self._robot_cfg = SceneEntityCfg(cfg.asset_name)
        self._cloth_cfg = SceneEntityCfg(cfg.object_name)

        # target position in the robot's root frame [m]
        self.target_pos_b = torch.zeros(self.num_envs, 3, device=self.device)

        # consecutive steps the cloth has rested at the target
        self._hold_steps = round(cfg.hold_time / env.step_dt)
        self._steps_held = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Attributes for logging metrics at reset
        self.metrics["position_error_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["success"] = torch.zeros(self.num_envs, device=self.device)

        # ring buffers of the last speeds and whether the cloth was inside the success radius; all envs step together,
        # so one write index serves every env
        self._speed_hist = torch.zeros(self.num_envs, cfg.speed_window, device=self.device)
        self._at_target_hist = torch.zeros(self.num_envs, cfg.speed_window, dtype=torch.bool, device=self.device)
        self._hist_idx = 0

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Target position of the cloth's center of mass in the robot's root frame [m]. Shape is (num_envs, 3)."""
        return self.target_pos_b

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)
        # read the speed window before the resample clears it
        at_target = self._at_target_hist[env_ids]
        num_at_target = at_target.sum(dim=1)
        speed_sum = (self._speed_hist[env_ids] * at_target).sum(dim=1)

        extras = super().reset(env_ids)
        # a NaN or placeholder would distort the mean over all resets in the logger, so leave the key out instead
        entered = num_at_target > 0
        if entered.any():
            extras["com_speed_at_target"] = (speed_sum[entered] / num_at_target[entered]).mean().item()
        return extras

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        com_pos_b = cloth_com_pos_b(self._env, self._cloth_cfg, self._robot_cfg)
        position_error = torch.linalg.norm(self.target_pos_b[:, :2] - com_pos_b[:, :2], dim=-1)
        speed = torch.linalg.norm(self.cloth.data.root_vel_w.torch, dim=-1)
        in_radius = position_error < self.cfg.success_radius
        # count consecutive steps at rest at the target, restarting whenever the cloth leaves or moves too fast
        at_rest = in_radius & (speed < self.cfg.max_speed)
        self._steps_held = torch.where(at_rest, self._steps_held + 1, 0)

        self._speed_hist[:, self._hist_idx] = speed
        self._at_target_hist[:, self._hist_idx] = in_radius
        self._hist_idx = (self._hist_idx + 1) % self.cfg.speed_window

        self.metrics["position_error_xy"] = position_error
        # sticky until the reset, so the logged mean is the episode success rate
        self.metrics["success"] = torch.maximum(self.metrics["success"], (self._steps_held >= self._hold_steps).float())

    def _resample_command(self, env_ids: Sequence[int]):
        # a new target restarts the hold and the speed window
        self._steps_held[env_ids] = 0
        self._at_target_hist[env_ids] = False
        r = torch.empty(len(env_ids), device=self.device)
        self.target_pos_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.target_pos_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        self.target_pos_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Create the target marker on first use and show or hide it when :attr:`debug_vis` is toggled."""
        if debug_vis:
            if not hasattr(self, "goal_visualizer"):
                self.goal_visualizer = VisualizationMarkers(self.cfg.goal_visualizer_cfg)
            self.goal_visualizer.set_visibility(True)
        elif hasattr(self, "goal_visualizer"):
            self.goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Move the target marker to the target in the world frame on every rendered frame."""
        # the robot data cannot be accessed while it is de-initialized
        if not self.robot.is_initialized:
            return
        target_pos_w, _ = combine_frame_transforms(
            self.robot.data.root_pos_w.torch, self.robot.data.root_quat_w.torch, self.target_pos_b
        )
        self.goal_visualizer.visualize(target_pos_w, environment_ids=self._env.scene._ALL_INDICES)

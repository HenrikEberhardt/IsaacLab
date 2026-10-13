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
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import ClothTargetCommandCfg


class ClothTargetCommand(CommandTerm):
    """Target position of the cloth's center of mass in the robot's root frame.

    The target is sampled uniformly from :attr:`ClothTargetCommandCfg.ranges` whenever the command is resampled, by
    default only when the episode resets.
    """

    cfg: ClothTargetCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: ClothTargetCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self._robot_cfg = SceneEntityCfg(cfg.asset_name)
        self._cloth_cfg = SceneEntityCfg(cfg.object_name)

        # target position in the robot's root frame [m]
        self.target_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Target position of the cloth's center of mass in the robot's root frame [m]. Shape is (num_envs, 3)."""
        return self.target_pos_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        com_pos_b = cloth_com_pos_b(self._env, self._cloth_cfg, self._robot_cfg)
        self.metrics["position_error"] = torch.linalg.norm(self.target_pos_b - com_pos_b, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.target_pos_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.target_pos_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        self.target_pos_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_visualizer"):
                self.goal_visualizer = VisualizationMarkers(self.cfg.goal_visualizer_cfg)
            self.goal_visualizer.set_visibility(True)
        elif hasattr(self, "goal_visualizer"):
            self.goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # the robot data cannot be accessed while it is de-initialized
        if not self.robot.is_initialized:
            return
        target_pos_w, _ = combine_frame_transforms(
            self.robot.data.root_pos_w.torch, self.robot.data.root_quat_w.torch, self.target_pos_b
        )
        self.goal_visualizer.visualize(target_pos_w, environment_ids=self._env.scene._ALL_INDICES)

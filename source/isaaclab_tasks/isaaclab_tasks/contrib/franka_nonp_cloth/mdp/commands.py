# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Crease drag command for the Franka non-prehensile cloth task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

from isaaclab_tasks.core.lift.mdp.commands.pose_commands import DeformableUniformPoseCommand

from ..cloth_mesh import crease_node_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import CreaseDragCommandCfg


class CreaseDragCommand(DeformableUniformPoseCommand):
    """Target point in the robot base frame to drag the crease of the cloth to.

    The crease is tracked by the mean position of the middle of its crest in the cloth's rest shape: the nodes within
    :attr:`CreaseDragCommandCfg.crest_band` of the highest node and within :attr:`CreaseDragCommandCfg.crease_radius`
    of the crest's center, which is where the gripper pinches the crease. The node set is fixed when the term is
    created, so the same cloth is tracked after the drag has flattened the crest. Only the horizontal distance
    between the crease and the target counts; the target height only places its marker.

    With :attr:`CreaseDragCommandCfg.drag_fraction` below one, the target drawn on the line is moved toward the crease,
    to that fraction of the way from the crease's reset position, but no closer than
    :attr:`CreaseDragCommandCfg.min_drag_distance`.

    The goal marker shows the target and the current marker the crease; both turn green once the crease is within
    :attr:`CreaseDragCommandCfg.success_threshold` of the target. The markers are also updated with the metrics
    every step, since a headless visualizer that only records video never runs the debug visualization callback.
    """

    cfg: CreaseDragCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: CreaseDragCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # every environment spawns the same creased mesh
        rest_nodes = self.object.data.default_nodal_state_w.torch[0, :, :3]
        crest = crease_node_mask(rest_nodes, cfg.crest_band)
        crest_center = rest_nodes[crest, :2].mean(dim=0)
        near_center = torch.linalg.norm(rest_nodes[:, :2] - crest_center, dim=-1) <= cfg.crease_radius
        self.crease_node_ids = (crest & near_center).nonzero(as_tuple=False).squeeze(-1)

    """
    Properties
    """

    @property
    def crease_pos_w(self) -> torch.Tensor:
        """Position of the crease in the world frame [m]. Shape is (num_envs, 3)."""
        return self.object.data.nodal_pos_w.torch[:, self.crease_node_ids].mean(dim=1)

    @property
    def grasp_pos_w(self) -> torch.Tensor:
        """Position in the world frame at which the fingertip point grasps the crease [m]. Shape is (num_envs, 3).

        It lies :attr:`CreaseDragCommandCfg.grasp_depth` below the crease, so the fingers straddle the fold.
        """
        grasp_pos_w = self.crease_pos_w
        grasp_pos_w[:, 2] -= self.cfg.grasp_depth
        return grasp_pos_w

    @property
    def crease_pos_b(self) -> torch.Tensor:
        """Position of the crease in the robot base frame [m]. Shape is (num_envs, 3)."""
        crease_pos_b, _ = subtract_frame_transforms(
            self.robot.data.root_pos_w.torch, self.robot.data.root_quat_w.torch, self.crease_pos_w
        )
        return crease_pos_b

    @property
    def crease_yaw_b(self) -> torch.Tensor:
        """Yaw about the robot base z-axis at which a top-down gripper closes across the crease [rad].

        The fingers close along the line from the center of the cloth to the crease, see
        :func:`~.top_down.top_down_yaw`. Shape is (num_envs,).
        """
        center_w = self.object.data.nodal_pos_w.torch.mean(dim=1)
        center_b, _ = subtract_frame_transforms(
            self.robot.data.root_pos_w.torch, self.robot.data.root_quat_w.torch, center_w
        )
        across = self.crease_pos_b[:, :2] - center_b[:, :2]
        # the top-down hand closes its fingers along -y at zero yaw
        return torch.atan2(across[:, 0], -across[:, 1])

    @property
    def target_pos_w(self) -> torch.Tensor:
        """Position of the drag target in the world frame [m]. Shape is (num_envs, 3).

        Unlike :attr:`pose_command_w`, it is current right after a reset resampled the target.
        """
        target_pos_w, _ = combine_frame_transforms(
            self.robot.data.root_pos_w.torch, self.robot.data.root_quat_w.torch, self.pose_command_b[:, :3]
        )
        return target_pos_w

    @property
    def crease_error(self) -> torch.Tensor:
        """Horizontal distance of the crease from the drag target [m]. Shape is (num_envs,)."""
        return torch.linalg.norm(self.target_pos_w[:, :2] - self.crease_pos_w[:, :2], dim=-1)

    """
    Implementation specific functions.
    """

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        if self.cfg.drag_fraction >= 1.0:
            return
        # the reset events have already placed the cloth, so the crease is at its reset position
        crease_xy = self.crease_pos_b[env_ids, :2]
        drag = self.pose_command_b[env_ids, :2] - crease_xy
        distance = torch.linalg.norm(drag, dim=-1, keepdim=True)
        shortened = (self.cfg.drag_fraction * distance).clamp_min(distance.clamp_max(self.cfg.min_drag_distance))
        self.pose_command_b[env_ids, :2] = crease_xy + drag * shortened / distance.clamp_min(1e-6)

    def _update_metrics(self):
        self.pose_command_w[:, :3] = self.target_pos_w
        self.pose_command_w[:, 3:] = self.robot.data.root_quat_w.torch
        self.metrics["position_error"] = self.crease_error
        if hasattr(self, "goal_visualizer") and self.goal_visualizer.is_visible():
            self._visualize_crease_and_target()

        if self.success_vis_asset is None:
            return
        success_id = (self.metrics["position_error"] < self.cfg.success_threshold).int()
        self.success_visualizer.visualize(
            self._get_success_vis_pos_w(),
            marker_indices=success_id,
            environment_ids=self._env.scene._ALL_INDICES,
        )

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        self._visualize_crease_and_target()

    def _visualize_crease_and_target(self):
        """Draw the drag target and the crease, green once the crease is at the target and red otherwise."""
        marker_indices = (self.crease_error < self.cfg.success_threshold).int()
        environment_ids = self._env.scene._ALL_INDICES
        self.goal_visualizer.visualize(
            self.target_pos_w, marker_indices=marker_indices, environment_ids=environment_ids
        )
        self.curr_visualizer.visualize(
            self.crease_pos_w, marker_indices=marker_indices, environment_ids=environment_ids
        )

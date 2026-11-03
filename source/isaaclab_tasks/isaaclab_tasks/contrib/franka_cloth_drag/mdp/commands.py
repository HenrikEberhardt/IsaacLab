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

from .frames import cloth_center_pos_b, cloth_center_vel_b, points_to_base, raised_node_ids
from .stages import (
    ClothDragStage,
    cloth_goal_xy_distance,
    ee_grasp_node_distance,
    goal_point_pos_b,
    goal_point_vel_b,
    grasp_node_ids,
    gripper_holds_cloth,
    gripper_holds_raised_node,
)

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, DeformableObject
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.sensors import FrameTransformer

    from .commands_cfg import ClothTargetCommandCfg


class ClothTargetCommand(CommandTerm):
    """Target position of the cloth's central node in the robot's root frame, and the task stage.

    The target is sampled uniformly from :attr:`ClothTargetCommandCfg.ranges` whenever the command is resampled, by
    default only when the episode resets. The goal is reached when the cloth's central node stays at the target while
    the gripper holds the cloth by a raised node, see :class:`ClothTargetCommandCfg`.

    The term also tracks the :class:`~.stages.ClothDragStage` of every environment, see :class:`ClothTargetCommandCfg`
    for the transitions. The stage is updated once per environment step, after the rewards and before the
    observations, so the stage the policy observes is the stage that gates the reward of its next action.

    Metrics, logged at reset:

    * ``position_error_xy``: planar distance of the goal point (the cloth's central node, or the root of
      :attr:`ClothTargetCommandCfg.goal_object_name`) to the target [m].
    * ``success``: one if the goal was reached during the episode, so its mean is the episode success rate.
    * ``center_speed_at_target``: planar speed of the goal point [m/s], averaged over the steps of the last
      :attr:`ClothTargetCommandCfg.speed_window` steps in which it was inside the success radius. Environments whose
      cloth was not inside in that window are left out; the metric is omitted when none was inside.
    * ``max_stage``: highest stage index reached during the episode.
    * ``reached_grasp``, ``reached_drag``, ``reached_goal``: one if the stage was reached during the episode.
    * ``time_in_approach``, ``time_in_grasp``, ``time_in_drag``, ``time_in_goal``: time spent in the stage [s].
    * ``drops``: number of times the gripper released the cloth in DRAG or GOAL.
    * ``time_in_wrong_grasp``: time the gripper held the cloth without a raised node among the locked nodes [s].
    * ``time_to_grasp``, ``time_to_drag``, ``time_to_goal``: episode time at which the stage was first reached [s],
      averaged over the environments that reached it; omitted when none did.

    With :attr:`ClothTargetCommandCfg.goal_object_name` set, also:

    * ``object_center_offset``: planar distance of the object's root to the cloth's central node at the end of the
      episode [m].
    * ``object_slip_speed``: planar speed of the object relative to the cloth's central node [m/s], averaged over the
      episode.
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

        # consecutive steps the cloth's central node has stayed at the target
        self._hold_steps = round(cfg.hold_time / env.step_dt)
        self._steps_held = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._last_hold_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)

        # grasp node of the episode, chosen on first use after a reset, once the target and the cloth are final
        self._grasp_node = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._grasp_node_stale = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        # stage machine
        self._stage = torch.full((self.num_envs,), int(ClothDragStage.APPROACH), dtype=torch.long, device=self.device)
        self._previous_stage = self._stage.clone()
        self._time_in_stage = torch.zeros(self.num_envs, device=self.device)
        self._dropped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # episode time at which each stage was first reached [s]; negative while not reached
        self._time_to_stage = torch.full((self.num_envs, len(ClothDragStage)), -1.0, device=self.device)

        # Attributes for logging metrics at reset
        self.metrics["position_error_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["success"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["max_stage"] = torch.zeros(self.num_envs, device=self.device)
        for stage in list(ClothDragStage)[1:]:
            self.metrics[f"reached_{stage.name.lower()}"] = torch.zeros(self.num_envs, device=self.device)
        for stage in ClothDragStage:
            self.metrics[f"time_in_{stage.name.lower()}"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["drops"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["time_in_wrong_grasp"] = torch.zeros(self.num_envs, device=self.device)
        if cfg.goal_object_name is not None:
            self.metrics["object_center_offset"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["object_slip_speed"] = torch.zeros(self.num_envs, device=self.device)

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
        """Target position of the cloth's central node in the robot's root frame [m]. Shape is (num_envs, 3)."""
        return self.target_pos_b

    @property
    def grasp_node_id(self) -> torch.Tensor:
        """Cloth node the gripper should grasp in this episode. Shape is (num_envs,).

        The raised node (see :attr:`ClothTargetCommandCfg.raised_nodes_event`) closest to the segment from the cloth's
        central node to the target, the node the scripted state machine grasps (:func:`~.stages.grasp_node_ids`).
        Without raised nodes, the node a few centimeters from the central node towards the target. It is chosen on the
        first use after a reset, after a restored target is set, and kept for the episode.
        """
        if self._grasp_node_stale.any():
            nodes_b = points_to_base(self._env, self.cloth.data.nodal_pos_w.torch, self._robot_cfg)
            center_b = cloth_center_pos_b(self._env, self._cloth_cfg, self._robot_cfg)
            ids = raised_node_ids(self._env, self.cfg.raised_nodes_event)
            node = grasp_node_ids(nodes_b, center_b, self.target_pos_b, ids)
            self._grasp_node = torch.where(self._grasp_node_stale, node, self._grasp_node)
            self._grasp_node_stale[:] = False
        return self._grasp_node

    @property
    def stage(self) -> torch.Tensor:
        """Current :class:`~.stages.ClothDragStage` index. Shape is (num_envs,)."""
        return self._stage

    @property
    def previous_stage(self) -> torch.Tensor:
        """:class:`~.stages.ClothDragStage` index before the last update, e.g. the stage a drop happened in.
        Shape is (num_envs,)."""
        return self._previous_stage

    @property
    def time_in_stage(self) -> torch.Tensor:
        """Time since the current stage was entered [s]. Shape is (num_envs,)."""
        return self._time_in_stage

    @property
    def dropped(self) -> torch.Tensor:
        """Whether the gripper released the cloth in DRAG or GOAL at the last update. Shape is (num_envs,)."""
        return self._dropped

    @property
    def time_to_stage(self) -> torch.Tensor:
        """Episode time at which each :class:`~.stages.ClothDragStage` was first reached [s], negative while not
        reached. Shape is (num_envs, num_stages)."""
        return self._time_to_stage

    @property
    def holding(self) -> torch.Tensor:
        """Whether the cloth's central node has stayed within :attr:`ClothTargetCommandCfg.success_radius` of the
        target for :attr:`ClothTargetCommandCfg.hold_time` in a row while the gripper holds the cloth by a raised node,
        with the EE frame origin at most :attr:`ClothTargetCommandCfg.max_ee_height` above the table.
        Shape is (num_envs,)."""
        return self._steps_held >= self._hold_steps

    """
    Operations.
    """

    def update_goal_status(self) -> None:
        """Update current center error and consecutive hold steps once per episode step.

        Success termination runs before command computation. Both call this method so termination, reward and
        logged success agree on the current physical state without counting the same control step twice.
        """
        distance = cloth_goal_xy_distance(self._env, self.target_pos_b, self.cfg)
        step = self._env.episode_length_buf
        fresh = step != self._last_hold_step
        at_target = distance < self.cfg.success_radius
        # the hold counts only while a raised node is locked: a pushed cloth or one held by its edge does not count
        at_target &= gripper_holds_raised_node(self._env, self.cfg)
        if self.cfg.max_ee_height is not None:
            # a cloth carried over the target does not count: the held cloth hangs from the EE frame
            ee_frame: FrameTransformer = self._env.scene[self.cfg.ee_frame_name]
            ee_height = ee_frame.data.target_pos_w.torch[:, 0, 2] - self._env.scene.env_origins[:, 2]
            at_target &= ee_height <= self.cfg.max_ee_height
        held = torch.where(at_target, self._steps_held + 1, 0)
        self._steps_held = torch.where(fresh, held, self._steps_held)
        self._last_hold_step.copy_(step)
        self.metrics["position_error_xy"] = distance
        self.metrics["success"] = torch.maximum(self.metrics["success"], self.holding.float())

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)
        # read the speed window before the resample clears it
        at_target = self._at_target_hist[env_ids]
        num_at_target = at_target.sum(dim=1)
        speed_sum = (self._speed_hist[env_ids] * at_target).sum(dim=1)
        time_to_stage = self._time_to_stage[env_ids]

        extras = super().reset(env_ids)
        # a NaN or placeholder would distort the mean over all resets in the logger, so leave the key out instead
        entered = num_at_target > 0
        if entered.any():
            extras["center_speed_at_target"] = (speed_sum[entered] / num_at_target[entered]).mean().item()
        for stage in list(ClothDragStage)[1:]:
            reached = time_to_stage[:, int(stage)] >= 0.0
            if reached.any():
                extras[f"time_to_{stage.name.lower()}"] = time_to_stage[reached, int(stage)].mean().item()

        self._grasp_node_stale[env_ids] = True
        self._stage[env_ids] = int(ClothDragStage.APPROACH)
        self._previous_stage[env_ids] = int(ClothDragStage.APPROACH)
        self._time_in_stage[env_ids] = 0.0
        self._dropped[env_ids] = False
        self._time_to_stage[env_ids] = -1.0
        return extras

    def set_restored(self, env_ids: torch.Tensor, target_pos_b: torch.Tensor, stage: torch.Tensor) -> None:
        """Continue a stored reset state in the given environments, after their reset.

        Sets the stored target and the stored stage. The stage holds until the first update recomputes it from the
        restored state, e.g. for the first observation after :meth:`~isaaclab.envs.ManagerBasedEnv.reset`.

        Args:
            env_ids: The environments, shape [N].
            target_pos_b: Targets in the robot's root frame [m], shape [N, 3].
            stage: :class:`~.stages.ClothDragStage` indices, shape [N].
        """
        self.target_pos_b[env_ids] = target_pos_b
        self._stage[env_ids] = stage
        self._grasp_node_stale[env_ids] = True

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        self.update_goal_status()
        in_radius = self.metrics["position_error_xy"] < self.cfg.success_radius
        # planar speed, for the metric only: the pinched fold bobbing up and down moves the cloth without it sliding
        speed = torch.linalg.norm(goal_point_vel_b(self._env, self.cfg)[:, :2], dim=-1)
        if self.cfg.goal_object_name is not None:
            center_pos = cloth_center_pos_b(self._env, self._cloth_cfg, self._robot_cfg)
            offset = goal_point_pos_b(self._env, self.cfg)[:, :2] - center_pos[:, :2]
            self.metrics["object_center_offset"] = torch.linalg.norm(offset, dim=-1)
            center_vel = cloth_center_vel_b(self._env, self._cloth_cfg, self._robot_cfg)
            slip = torch.linalg.norm(goal_point_vel_b(self._env, self.cfg)[:, :2] - center_vel[:, :2], dim=-1)
            # running mean over the episode's steps
            steps = self._env.episode_length_buf.clamp(min=1).float()
            self.metrics["object_slip_speed"] += (slip - self.metrics["object_slip_speed"]) / steps

        self._speed_hist[:, self._hist_idx] = speed
        self._at_target_hist[:, self._hist_idx] = in_radius
        self._hist_idx = (self._hist_idx + 1) % self.cfg.speed_window

    def _resample_command(self, env_ids: Sequence[int]):
        # a new target restarts the hold and the speed window
        self._steps_held[env_ids] = 0
        self._last_hold_step[env_ids] = -1
        self._at_target_hist[env_ids] = False
        r = torch.empty(len(env_ids), device=self.device)
        self.target_pos_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.target_pos_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        self.target_pos_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)

    def _update_command(self):
        """Advance the stage machine on the current state."""
        cfg = self.cfg
        dt = self._env.step_dt
        # GRASP: near the grasp node; DRAG: holding a raised node; GOAL: DRAG and the center near the target
        near = ee_grasp_node_distance(self._env, cfg, self.grasp_node_id) < cfg.approach_radius
        # locked nodes cannot slip: the grasp holds until the policy opens the gripper, so no hysteresis is needed
        pinched = gripper_holds_raised_node(self._env, cfg)
        wrong_grasp = gripper_holds_cloth(self._env, cfg) & ~pinched
        goal_radius = cfg.success_radius + cfg.goal_zone_margin
        in_goal_zone = pinched & (cloth_goal_xy_distance(self._env, self.target_pos_b, cfg) < goal_radius)
        stage = torch.where(pinched, int(ClothDragStage.DRAG) + in_goal_zone.long(), near.long())
        # releasing the cloth while dragging falls back to APPROACH; the next update re-applies the conditions
        self._dropped = (self._stage >= int(ClothDragStage.DRAG)) & ~pinched
        stage[self._dropped] = int(ClothDragStage.APPROACH)

        self._time_in_stage = torch.where(stage == self._stage, self._time_in_stage + dt, 0.0)
        self._previous_stage = self._stage
        self._stage = stage

        # episode bookkeeping
        episode_time = self._env.episode_length_buf.float() * dt
        for s in ClothDragStage:
            first = (stage >= s) & (self._time_to_stage[:, int(s)] < 0.0)
            self._time_to_stage[:, int(s)] = torch.where(first, episode_time, self._time_to_stage[:, int(s)])
            self.metrics[f"time_in_{s.name.lower()}"] += (stage == s).float() * dt
            if s > ClothDragStage.APPROACH:
                reached = self.metrics[f"reached_{s.name.lower()}"]
                self.metrics[f"reached_{s.name.lower()}"] = torch.maximum(reached, (stage >= s).float())
        self.metrics["max_stage"] = torch.maximum(self.metrics["max_stage"], stage.float())
        self.metrics["drops"] += self._dropped.float()
        self.metrics["time_in_wrong_grasp"] += wrong_grasp.float() * dt

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

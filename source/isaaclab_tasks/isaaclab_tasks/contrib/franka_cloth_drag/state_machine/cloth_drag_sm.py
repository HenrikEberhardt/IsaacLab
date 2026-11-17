# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched Torch state machine for one cloth-drag attempt."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import torch

from .cloth_drag_sm_cfg import ClothDragSmCfg, ClothDragSmParams, sample_sm_params

OPEN_GRIPPER = 1.0
CLOSE_GRIPPER = -1.0


class DragSmState(IntEnum):
    """Controller phases."""

    PREGRASP = 0
    HOVER = 1
    DESCEND = 2
    GRASP = 3
    DRAG = 4
    HOLD = 5
    FAILED = 6


@dataclass
class ClothDragSmObservation:
    """Physical controller inputs for all environments.

    Positions are in the robot base frame [m], yaw is in radians, and tensors have first dimension ``num_envs``.
    """

    ee_pos: torch.Tensor
    ee_yaw: torch.Tensor
    grasp_node_pos: torch.Tensor
    cloth_center_pos: torch.Tensor
    goal_pos: torch.Tensor
    target_pos: torch.Tensor
    valid_grasp: torch.Tensor
    success: torch.Tensor


@dataclass
class ClothDragSmCommand:
    """Desired end-effector pose and gripper command for all environments."""

    ee_pos: torch.Tensor
    ee_yaw: torch.Tensor
    gripper: torch.Tensor
    pregrasp_ready: torch.Tensor
    failed: torch.Tensor


def wrap_half_turn(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angles to ``[-pi / 2, pi / 2)`` for the symmetric gripper."""
    return torch.remainder(angle + 0.5 * math.pi, math.pi) - 0.5 * math.pi


def planar_heading(from_pos: torch.Tensor, to_pos: torch.Tensor) -> torch.Tensor:
    """Return planar headings from ``from_pos`` to ``to_pos`` [rad]."""
    delta = to_pos[..., :2] - from_pos[..., :2]
    return torch.atan2(delta[..., 1], delta[..., 0])


class DragClothSm:
    """Generate one pre-grasp, grasp, drag and hold attempt in parallel environments."""

    def __init__(
        self,
        dt: float,
        num_envs: int,
        device: torch.device | str,
        cfg: ClothDragSmCfg,
        params: ClothDragSmParams | None = None,
    ):
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.cfg = cfg

        self.sm_state = torch.full((num_envs,), int(DragSmState.PREGRASP), dtype=torch.long, device=device)
        self.sm_wait_time = torch.zeros(num_envs, device=device)
        self.grasp_pos = torch.zeros((num_envs, 3), device=device)
        self.grasp_yaw = torch.zeros(num_envs, device=device)
        self.hold_pos = torch.zeros((num_envs, 3), device=device)
        self.stall_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._previous_ee_pos = torch.zeros((num_envs, 3), device=device)

        self.des_ee_pos = torch.zeros((num_envs, 3), device=device)
        self.des_yaw = torch.zeros(num_envs, device=device)
        self.des_gripper_state = torch.full((num_envs,), OPEN_GRIPPER, device=device)
        self.pregrasp_ready = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.failed = torch.zeros(num_envs, dtype=torch.bool, device=device)

        self.params = params or sample_sm_params(cfg, None, num_envs, device)

    def reset_idx(self, env_ids: torch.Tensor | None = None):
        """Restart the selected environments at the pre-grasp phase."""
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = int(DragSmState.PREGRASP)
        self.sm_wait_time[env_ids] = 0.0
        self.stall_steps[env_ids] = 0
        self.pregrasp_ready[env_ids] = False
        self.failed[env_ids] = False

    def set_params(self, env_ids: torch.Tensor, params: ClothDragSmParams):
        """Set per-rollout parameters for the selected environments."""
        self.params.drag_height[env_ids] = params.drag_height
        self.params.drag_speed[env_ids] = params.drag_speed
        self.params.yaw_offset[env_ids] = params.yaw_offset
        self.params.pregrasp_offset[env_ids] = params.pregrasp_offset
        self.params.pregrasp_yaw_error[env_ids] = params.pregrasp_yaw_error

    def compute(self, observation: ClothDragSmObservation) -> ClothDragSmCommand:
        """Advance every environment by one controller step."""
        state = self.sm_state.clone()
        self.pregrasp_ready.zero_()

        still = torch.linalg.vector_norm(observation.ee_pos - self._previous_ee_pos, dim=1) < self.cfg.stall_step
        dragging = state == int(DragSmState.DRAG)
        self.stall_steps = torch.where(dragging & still, self.stall_steps + 1, 0)

        self._step_pregrasp(state == int(DragSmState.PREGRASP), observation)
        self._step_hover(state == int(DragSmState.HOVER), observation)
        self._step_descend(state == int(DragSmState.DESCEND), observation)
        self._step_grasp(state == int(DragSmState.GRASP), observation)
        self._step_drag(state == int(DragSmState.DRAG), observation)
        self._step_hold(state == int(DragSmState.HOLD), observation)
        self._step_failed(state == int(DragSmState.FAILED), observation)

        self.sm_wait_time += self.dt
        self._previous_ee_pos.copy_(observation.ee_pos)
        self.failed = self.sm_state == int(DragSmState.FAILED)
        return ClothDragSmCommand(
            ee_pos=self.des_ee_pos,
            ee_yaw=self.des_yaw,
            gripper=self.des_gripper_state,
            pregrasp_ready=self.pregrasp_ready,
            failed=self.failed,
        )

    def _step_pregrasp(self, mask: torch.Tensor, observation: ClothDragSmObservation):
        self.des_ee_pos[mask] = (observation.grasp_node_pos + self.params.pregrasp_offset)[mask]
        yaw = (
            planar_heading(observation.grasp_node_pos, observation.cloth_center_pos)
            + self.params.yaw_offset
            + self.params.pregrasp_yaw_error
        )
        self.des_yaw[mask] = yaw[mask]
        self.des_gripper_state[mask] = OPEN_GRIPPER

        ready = self._pose_reached(observation, self.cfg.approach_threshold, self.cfg.yaw_tolerance)
        self.pregrasp_ready = mask & ready
        self.sm_wait_time[mask & ~ready] = 0.0
        self._transition(mask & ready & (self.sm_wait_time >= self.cfg.pregrasp_hold_time), DragSmState.HOVER)

    def _step_hover(self, mask: torch.Tensor, observation: ClothDragSmObservation):
        target = observation.grasp_node_pos + observation.grasp_node_pos.new_tensor((0.0, 0.0, self.cfg.hover_height))
        self.des_ee_pos[mask] = target[mask]
        yaw = planar_heading(observation.grasp_node_pos, observation.cloth_center_pos) + self.params.yaw_offset
        self.des_yaw[mask] = yaw[mask]
        self.des_gripper_state[mask] = OPEN_GRIPPER

        ready = self._pose_reached(observation, self.cfg.approach_threshold, self.cfg.yaw_tolerance)
        self.sm_wait_time[mask & ~ready] = 0.0
        transition = mask & ready & (self.sm_wait_time >= self.cfg.hover_hold_time)
        self.grasp_pos[transition, :2] = observation.grasp_node_pos[transition, :2]
        self.grasp_pos[transition, 2] = self.cfg.grasp_height
        self.grasp_yaw[transition] = yaw[transition]
        self._transition(transition, DragSmState.DESCEND)

    def _step_descend(self, mask: torch.Tensor, observation: ClothDragSmObservation):
        self.des_ee_pos[mask] = self.grasp_pos[mask]
        self.des_yaw[mask] = self.grasp_yaw[mask]
        self.des_gripper_state[mask] = OPEN_GRIPPER

        ready = torch.linalg.vector_norm(observation.ee_pos - self.des_ee_pos, dim=1) < self.cfg.grasp_threshold
        self.sm_wait_time[mask & ~ready] = 0.0
        self._transition(mask & ready & (self.sm_wait_time >= self.cfg.descend_hold_time), DragSmState.GRASP)

    def _step_grasp(self, mask: torch.Tensor, observation: ClothDragSmObservation):
        self.des_ee_pos[mask] = self.grasp_pos[mask]
        self.des_yaw[mask] = self.grasp_yaw[mask]
        self.des_gripper_state[mask] = CLOSE_GRIPPER

        grasped = mask & observation.valid_grasp & (self.sm_wait_time >= self.cfg.grasp_squeeze_time)
        timed_out = mask & ~observation.valid_grasp & (self.sm_wait_time >= self.cfg.grasp_timeout)
        self._transition(grasped, DragSmState.DRAG)
        self._transition(timed_out, DragSmState.FAILED)

    def _step_drag(self, mask: torch.Tensor, observation: ClothDragSmObservation):
        to_target = observation.target_pos - observation.goal_pos
        to_target[:, 2] = 0.0
        distance = torch.linalg.vector_norm(to_target, dim=1)
        step = torch.minimum(distance, self.params.drag_speed * self.dt)
        direction = to_target / distance.clamp(min=1.0e-6).unsqueeze(1)
        target = self.des_ee_pos + direction * step.unsqueeze(1)
        target[:, 2] = self.params.drag_height
        lead = target - observation.ee_pos
        lead_norm = torch.linalg.vector_norm(lead, dim=1)
        scale = torch.minimum(torch.ones_like(lead_norm), self.cfg.drag_max_lead / lead_norm.clamp(min=1.0e-6))
        target = observation.ee_pos + lead * scale.unsqueeze(1)
        self.des_ee_pos[mask] = target[mask]
        self.des_gripper_state[mask] = CLOSE_GRIPPER

        desired = planar_heading(observation.ee_pos, observation.cloth_center_pos) + self.params.yaw_offset
        error = wrap_half_turn(desired - self.des_yaw)
        correction = error.clamp(min=-self.cfg.drag_yaw_rate * self.dt, max=self.cfg.drag_yaw_rate * self.dt)
        turn = mask & (error.abs() > self.cfg.drag_yaw_tolerance)
        self.des_yaw[turn] += correction[turn]

        reached = mask & (distance < self.cfg.goal_tolerance)
        stalled = mask & ~reached & (self.stall_steps * self.dt >= self.cfg.stall_time)
        self.hold_pos[reached] = self.des_ee_pos[reached]
        self._transition(reached, DragSmState.HOLD)
        self._transition(stalled, DragSmState.FAILED)

    def _step_hold(self, mask: torch.Tensor, observation: ClothDragSmObservation):
        self.des_ee_pos[mask] = self.hold_pos[mask]
        self.des_gripper_state[mask] = CLOSE_GRIPPER
        timed_out = mask & ~observation.success & (self.sm_wait_time >= self.cfg.hold_timeout)
        self._transition(timed_out, DragSmState.FAILED)

    def _step_failed(self, mask: torch.Tensor, observation: ClothDragSmObservation):
        self.des_ee_pos[mask] = observation.ee_pos[mask]
        self.des_gripper_state[mask] = OPEN_GRIPPER

    def _pose_reached(
        self, observation: ClothDragSmObservation, position_threshold: float, yaw_threshold: float
    ) -> torch.Tensor:
        position_ready = torch.linalg.vector_norm(observation.ee_pos - self.des_ee_pos, dim=1) < position_threshold
        yaw_ready = wrap_half_turn(self.des_yaw - observation.ee_yaw).abs() < yaw_threshold
        return position_ready & yaw_ready

    def _transition(self, mask: torch.Tensor, state: DragSmState):
        self.sm_state[mask] = int(state)
        self.sm_wait_time[mask] = 0.0

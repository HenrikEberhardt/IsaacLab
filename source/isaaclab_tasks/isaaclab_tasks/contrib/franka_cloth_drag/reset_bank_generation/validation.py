# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate stored reset states of the Franka cloth dragging task by restoring and holding them."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from ..mdp.frames import ee_pose_in_base
from ..mdp.stages import START_STAGE_NAMES
from ..state_machine import ee_down_tilt
from .capture import capture_reset_states

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from ..reset_strategies.bank import ResetBank


class ResetStateValidator:
    """Check restored reset states while they are held: arm still, gripper as stored.

    A state is rejected when its restore does not reproduce the stored state, when a termination other than success
    fires (a gripper-cube contact included), when a state leaves the task stage it was stored in, when a stored grasp is
    lost or its locked nodes change, when a grasp (pre-grasp) state closes or locks, when a cloth node is faster than
    ``max_node_speed``, when the cube tilts more than ``max_cube_tilt_deg``, or when the end-effector tilts more than
    ``max_ee_down_tilt_deg`` from base-frame down. A success ends the check of drag and goal states as valid.
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        max_node_speed: float,
        max_cube_tilt_deg: float,
        cube_name: str | None = "cube",
        max_ee_down_tilt_deg: float = 1.0,
    ):
        """Look up the task terms.

        Args:
            env: The unwrapped environment.
            max_node_speed: Fastest cloth node speed [m/s].
            max_cube_tilt_deg: Largest cube tilt from upright [deg].
            cube_name: Scene name of the cube, or None without one.
            max_ee_down_tilt_deg: Largest end-effector tilt from base-frame down [deg].
        """
        self.env = env
        self.max_node_speed = max_node_speed
        self.min_cube_up = math.cos(math.radians(max_cube_tilt_deg))
        self.max_ee_down_tilt = math.radians(max_ee_down_tilt_deg)
        self.cube_name = cube_name
        self.command = env.command_manager.get_term("cloth_target")
        self.gripper = env.action_manager.get_term("gripper_action")
        self.failure_terms = [
            name for name in env.termination_manager.active_terms if name not in ("task_success", "task_timeout")
        ]
        self.reasons = [
            "restore_mismatch",
            "stage_change",
            "grasp_changed",
            "node_speed",
            "cube_tilt",
            "ee_not_down_restore",
            "ee_not_down_hold",
        ]
        self.reasons += ["success_early", "time_out"] + [f"termination:{name}" for name in self.failure_terms]
        """Rejection reasons; a reason code is its index plus one, zero is no rejection."""

    def start(self, bank: ResetBank, rows: torch.Tensor, env_ids: torch.Tensor) -> None:
        """Begin the check of the bank rows ``rows``, just restored into ``env_ids`` by a reset.

        Args:
            bank: The bank the rows belong to.
            rows: Rows of the bank, shape [N].
            env_ids: Environments they were restored into, shape [N].
        """
        env = self.env
        self.env_ids = env_ids
        self.start_stage = bank.stage[rows]
        task = bank.data["task"]
        self.closed = task["closed"][rows]
        self.grasped = task["grasped"][rows] & task["active"][rows].unsqueeze(1)
        self.holds = task["active"][rows]
        self.alive = torch.ones(len(rows), dtype=torch.bool, device=env.device)
        self.ended_ok = torch.zeros_like(self.alive)
        self.reason = torch.zeros(len(rows), dtype=torch.long, device=env.device)

        # the restore must reproduce the stored state
        stored = bank.select(rows).data
        now = capture_reset_states(env, env_ids, cube_name=self.cube_name)
        mismatch = torch.zeros_like(self.alive)
        num = len(rows)
        for key in ("joint_pos", "nodal_pos"):
            mismatch |= (now["scene"][key] - stored["scene"][key]).reshape(num, -1).abs().amax(dim=1) > 1.0e-4
        if self.cube_name is not None:
            pos_error = (now["scene"]["cube_root_pose"][:, :3] - stored["scene"]["cube_root_pose"][:, :3]).abs()
            quat_dot = (now["scene"]["cube_root_pose"][:, 3:] * stored["scene"]["cube_root_pose"][:, 3:]).sum(dim=1)
            mismatch |= (pos_error.amax(dim=1) > 1.0e-4) | (quat_dot.abs() < 1.0 - 1.0e-6)
        for key in ("raised_node_ids", "stage", "active", "grasped", "closed"):
            mismatch |= (now["task"][key] != stored["task"][key]).reshape(num, -1).any(dim=1)
        self._reject(mismatch, "restore_mismatch")
        self._check_ee_down("ee_not_down_restore")

    def hold_actions(self) -> torch.Tensor:
        """Actions of all environments: zero arm deltas, gripper as stored (others open), shape [num_envs, 5]."""
        actions = torch.zeros(self.env.action_space.shape, device=self.env.device)
        actions[:, 4] = 1.0
        actions[self.env_ids, 4] = torch.where(self.closed, -1.0, 1.0)
        return actions

    def update(self, terminated: torch.Tensor, truncated: torch.Tensor) -> None:
        """Check the states after an environment step.

        Args:
            terminated: Terminated environments of the step, shape [num_envs].
            truncated: Timed-out environments of the step, shape [num_envs].
        """
        env, ids = self.env, self.env_ids
        manager = env.termination_manager
        for name in self.failure_terms:
            self._reject(self.alive & manager.get_term(name)[ids], f"termination:{name}")
        self._reject(self.alive & truncated[ids], "time_out")
        self._check_ee_down("ee_not_down_hold")
        if "task_success" in manager.active_terms:
            success = self.alive & manager.get_term("task_success")[ids] & (self.reason == 0)
            late = self.start_stage >= START_STAGE_NAMES.index("drag")
            self.ended_ok |= success & late
            self._reject(success & ~late, "success_early")
        self.alive &= ~(terminated | truncated)[ids]
        # the environments that did not end show the state that followed the stored one: every state keeps the stage
        # it was stored in, drag and goal states their grasp, pre-grasp states stay open and unlocked
        self._reject(self.alive & (self.command.stage[ids] != self.start_stage), "stage_change")
        node_lock = self.gripper.node_lock
        locked = node_lock.grasped[ids] & node_lock.active[ids].unsqueeze(1)
        changed = self.holds & ((locked != self.grasped).any(dim=1) | ~node_lock.active[ids])
        pregrasp = self.start_stage == START_STAGE_NAMES.index("grasp")
        changed |= pregrasp & (node_lock.active[ids] | self.gripper.closed[ids])
        self._reject(self.alive & changed, "grasp_changed")
        node_speed = env.scene["deformable"].data.nodal_vel_w.torch[ids].norm(dim=-1).amax(dim=1)
        self._reject(self.alive & ~(node_speed <= self.max_node_speed), "node_speed")
        if self.cube_name is not None:
            from isaaclab.utils.math import quat_apply

            quat = env.scene[self.cube_name].data.root_quat_w.torch[ids]
            up = quat_apply(quat, torch.tensor((0.0, 0.0, 1.0), device=env.device).expand(len(ids), 3))[:, 2]
            self._reject(self.alive & ~(up >= self.min_cube_up), "cube_tilt")

    def result(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return whether each state is valid and its rejection reason code (0 if valid), each of shape [N]."""
        return (self.reason == 0) & (self.alive | self.ended_ok), self.reason

    def _reject(self, mask: torch.Tensor, reason: str) -> None:
        """Record ``reason`` for the states of ``mask`` that have no reason yet."""
        self.reason = torch.where(mask & (self.reason == 0), self.reasons.index(reason) + 1, self.reason)

    def _check_ee_down(self, reason: str) -> None:
        """Reject live states whose end-effector does not point sufficiently down."""
        tilt = ee_down_tilt(ee_pose_in_base(self.env)[1][self.env_ids])
        self._reject(self.alive & ~(tilt <= self.max_ee_down_tilt), reason)

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate reset states with the randomized state machine, validate them by replay, and fill a bank.

Used by ``scripts/environments/state_machine/nonprehensile_cloth_collect_resets.py`` (generation) and
``nonprehensile_cloth_reset_sm.py`` (viewer). See ``implementation_log/reset_states.md``.
"""

from __future__ import annotations

import math
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import torch

from .bank import BUCKET_NAMES, SCHEMA_VERSION, ResetBank, assign_tree, empty_like_tree, subsample_to_quotas
from .capture import capture_reset_states, task_fingerprint

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from ..mdp.reset_bank import ResetFromBank
    from ..state_machine import ClothDragSmRunner

SAMPLED_BUCKETS = ("grasp", "drag", "goal")
"""Buckets that training samples from and the collector stores."""


@dataclass
class CollectCfg:
    """Settings of the reset-state generation."""

    num_states: int = 5000
    """Number of stored grasp, drag and goal states, split evenly."""

    validation_steps: int = 30
    """Steps each candidate is held after its restore before it counts as valid."""

    randomize: bool = True
    """Whether the state machine is randomized per rollout with the ranges below."""

    random_grasp_node: bool = False
    """Whether each randomized rollout grasps a random raised node instead of the task's grasp node
    (:attr:`~..mdp.commands.ClothTargetCommand.grasp_node_id`), which the approach rewards target."""

    drag_speed_range: tuple[float, float] = (0.05, 0.12)
    """Range of the drag speed [m/s]."""

    drag_height_range: tuple[float, float] = (0.015, 0.030)
    """Range of the end-effector height while dragging [m]."""

    yaw_offset_range: tuple[float, float] = (-0.3, 0.3)
    """Range of the offset added to the grasp yaw [rad]."""

    pregrasp_height_range: tuple[float, float] = (0.005, 0.03)
    """Range of the height of the pre-grasp pose (grasp states) above the grasped raised node [m]. With
    :attr:`pregrasp_radius_range`, the pose stays within 3.4 cm of the node, inside the GRASP stage's approach radius
    (4 cm) with margin for the tracking error."""

    pregrasp_radius_range: tuple[float, float] = (0.005, 0.015)
    """Largest planar offset of the pre-grasp pose from the raised node at the lowest and at the highest pre-grasp
    height [m]: it grows linearly with the height (a cone, approach from the top), and the offset is drawn biased to the
    centre (radius uniform, not area)."""

    pregrasp_yaw_error_range: tuple[float, float] = (-0.26, 0.26)
    """Range of the yaw of the pre-grasp pose relative to the grasp yaw [rad] (±15°)."""

    pregrasp_max_speed: float = 0.01
    """Fastest end-effector speed of a captured pre-grasp state [m/s]."""

    pregrasp_min_hold: float = 0.2
    """Time the state machine must have held the pre-grasp pose before a grasp state is captured [s]."""

    drag_window: float = 0.5
    """Drag states are captured at most this long after the squeeze completed [s]."""

    drag_max_progress: float = 0.1
    """Drag states are captured only while the goal point has covered at most this share of its distance."""

    goal_distance_window: tuple[float, float] = (0.05, 0.09)
    """Planar distance of the goal point to the target of the goal states [m]: inside the goal zone (success radius +
    goal zone margin, 0.10 m) and outside the success disc (0.04 m)."""

    action_noise_std: float = 0.05
    """Standard deviation of the noise on the normalized arm actions; the gripper command gets none."""

    max_node_speed: float = 2.0
    """Fastest cloth node speed [m/s] while a candidate is held."""

    max_cube_tilt_deg: float = 20.0
    """Largest cube tilt from upright [deg] while a candidate is held."""

    candidate_margin: float = 1.3
    """Candidates collected per missing state, on top of the expected rejections."""

    max_rounds: int = 20
    """Most rounds of rollouts and validation before giving up."""

    max_round_steps: int = 3000
    """Most environment steps of the rollouts of one round."""

    seed: int = 0
    """Seed of the state machine randomization and the final pick."""

    def __post_init__(self):
        if self.num_states < len(SAMPLED_BUCKETS):
            raise ValueError(f"num_states must be at least {len(SAMPLED_BUCKETS)}, one per bucket.")
        if self.validation_steps <= 0:
            raise ValueError("validation_steps must be positive.")
        for name in (
            "drag_speed_range",
            "drag_height_range",
            "yaw_offset_range",
            "pregrasp_height_range",
            "pregrasp_radius_range",
            "pregrasp_yaw_error_range",
            "goal_distance_window",
        ):
            low, high = getattr(self, name)
            if not low <= high:
                raise ValueError(f"{name} must be an increasing range, got {(low, high)}.")

    def quotas(self) -> dict[str, int]:
        """Number of stored states per sampled bucket."""
        base, extra = divmod(self.num_states, len(SAMPLED_BUCKETS))
        return {name: base + (i < extra) for i, name in enumerate(SAMPLED_BUCKETS)}


class ResetValidator:
    """Check restored reset states while they are held: arm still, gripper as stored.

    A state is rejected when its restore does not reproduce the stored state, when a termination other than success
    fires (a gripper-cube contact included), when a state leaves the task stage of its bucket, when a stored grasp is
    lost or its locked nodes change, when a grasp (pre-grasp) state closes or locks, when a cloth node is faster than
    ``max_node_speed`` or when the cube tilts more than ``max_cube_tilt_deg``. A success ends the check of drag and goal
    states as valid.
    """

    def __init__(self, env: ManagerBasedRLEnv, max_node_speed: float, max_cube_tilt_deg: float, cube_name="cube"):
        """Look up the task terms.

        Args:
            env: The unwrapped environment.
            max_node_speed: Fastest cloth node speed [m/s].
            max_cube_tilt_deg: Largest cube tilt from upright [deg].
            cube_name: Scene name of the cube, or None without one.
        """
        self.env = env
        self.max_node_speed = max_node_speed
        self.min_cube_up = math.cos(math.radians(max_cube_tilt_deg))
        self.cube_name = cube_name
        self.command = env.command_manager.get_term("cloth_target")
        self.gripper = env.action_manager.get_term("gripper_action")
        self.failure_terms = [
            name for name in env.termination_manager.active_terms if name not in ("success", "time_out")
        ]
        self.reasons = ["restore_mismatch", "stage_change", "grasp_changed", "node_speed", "cube_tilt"]
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
        self.bucket = bank.data["labels"]["bucket"][rows]
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
        if "success" in manager.active_terms:
            success = self.alive & manager.get_term("success")[ids] & (self.reason == 0)
            late = self.bucket >= BUCKET_NAMES.index("drag")
            self.ended_ok |= success & late
            self._reject(success & ~late, "success_early")
        self.alive &= ~(terminated | truncated)[ids]
        # the environments that did not end show the state that followed the stored one: every state keeps the stage
        # of its bucket, drag and goal states their grasp, pre-grasp states stay open and unlocked
        self._reject(self.alive & (self.command.stage[ids] != self.bucket), "stage_change")
        constraint = self.gripper.constraint
        locked = constraint.grasped[ids] & constraint.active[ids].unsqueeze(1)
        changed = self.holds & ((locked != self.grasped).any(dim=1) | ~constraint.active[ids])
        pregrasp = self.bucket == BUCKET_NAMES.index("grasp")
        changed |= pregrasp & (constraint.active[ids] | self.gripper.closed[ids])
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


class ResetCollector:
    """Run the randomized state machine, capture candidate reset states, validate them and fill a bank.

    Rollouts stream in all environments: every success hands its captured grasp, drag and goal states to the candidate
    pool, and every failure is dropped. Each sampled window has one reservoir slot per environment, so the capture
    time is uniform over the window. Candidates are then restored in chunks of the environments through
    :class:`~..mdp.reset_bank.ResetFromBank` and checked by :class:`ResetValidator`. Rounds of rollouts and validation
    alternate until every bucket of :meth:`CollectCfg.quotas` is full.
    """

    def __init__(self, env: ManagerBasedRLEnv, runner: ClothDragSmRunner, bank_term: ResetFromBank, cfg: CollectCfg):
        """Set up the collection.

        Args:
            env: The unwrapped environment, with the success termination and the reset event ``bank_term``.
            runner: The state machine runner of all environments.
            bank_term: The reset event term that restores the candidates.
            cfg: The generation settings.
        """
        if "success" not in env.termination_manager.active_terms:
            raise ValueError("The collection needs the task's success termination to end successful rollouts.")
        self.env = env
        self.runner = runner
        self.sm = runner.sm
        self.bank_term = bank_term
        self.cfg = cfg
        self.command = runner.command
        self.gripper = runner.gripper
        self.validator = ResetValidator(env, cfg.max_node_speed, cfg.max_cube_tilt_deg)
        self.num_envs, self.device = env.num_envs, env.device
        self._rows = torch.arange(self.num_envs, device=self.device)
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(cfg.seed)
        self.stats: Counter = Counter()

    def run(self) -> ResetBank:
        """Collect until every bucket is full and return the bank, with its meta information."""
        cfg = self.cfg
        quotas = cfg.quotas()
        start_time = time.time()
        accepted: list[ResetBank] = []
        acceptance = dict.fromkeys(SAMPLED_BUCKETS, 1.0)
        for round_id in range(cfg.max_rounds):
            counts = ResetBank.concat(accepted).bucket_counts() if accepted else dict.fromkeys(BUCKET_NAMES, 0)
            missing = {name: max(0, quotas[name] - counts[name]) for name in SAMPLED_BUCKETS}
            if not any(missing.values()):
                break
            targets = {
                name: math.ceil(missing[name] * cfg.candidate_margin / max(acceptance[name], 0.2))
                for name in SAMPLED_BUCKETS
            }
            candidates = self.collect_candidates(targets)
            if candidates is None:
                print(f"[INFO]: Round {round_id}: no successful rollout.")
                continue
            valid, reasons = self.validate(candidates)
            accepted.append(candidates.select(valid.nonzero(as_tuple=False).squeeze(-1)))
            self._record_validation(candidates, valid, reasons, acceptance)
            print(f"[INFO]: Round {round_id}: {self._summary(candidates, valid, quotas, accepted)}")
        else:
            counts = ResetBank.concat(accepted).bucket_counts() if accepted else dict.fromkeys(BUCKET_NAMES, 0)
            if any(counts[name] < quotas[name] for name in SAMPLED_BUCKETS):
                raise RuntimeError(f"Could not fill the buckets in {cfg.max_rounds} rounds: {counts} of {quotas}.")

        pool = ResetBank.concat(accepted)
        generator = torch.Generator().manual_seed(cfg.seed)
        bank = pool.select(subsample_to_quotas(pool, quotas, generator))
        off_stage = int((bank.data["task"]["stage"].long() != bank.data["labels"]["bucket"].long()).sum())
        if off_stage > 0:
            raise RuntimeError(f"{off_stage} states are not in the task stage of their bucket.")
        self.stats["wall_time_s"] = round(time.time() - start_time, 1)
        bank.meta = {
            "schema_version": SCHEMA_VERSION,
            "fingerprint": task_fingerprint(self.env),
            "task": self.env.spec.id if getattr(self.env, "spec", None) is not None else None,
            "created": datetime.now().isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "generator": asdict(cfg) | {"num_envs": self.num_envs},
            "statistics": dict(self.stats),
            "bucket_counts": bank.bucket_counts(),
        }
        return bank

    def collect_candidates(self, targets: dict[str, int]) -> ResetBank | None:
        """Run rollouts in all environments until every sampled bucket has ``targets[name]`` candidates.

        Returns:
            The candidates of the successful rollouts, or None if no rollout succeeded.
        """
        from ..mdp.frames import ee_pose_in_base
        from ..mdp.stages import cloth_goal_xy_distance

        env, runner, cfg = self.env, self.runner, self.cfg
        num_envs, device = self.num_envs, self.device
        env.reset()
        runner.reset_idx(self._rows)
        self._draw_sm_params(self._rows)
        num_levels = len(BUCKET_NAMES)
        seen = torch.zeros(num_levels, num_envs, dtype=torch.long, device=device)
        slots: list[dict | None] = [None] * num_levels
        progress = torch.zeros(num_levels, num_envs, device=device)
        episode_time = torch.zeros(num_levels, num_envs, device=device)
        start_distance = torch.ones(num_envs, device=device)
        # episode time at which the squeeze completed, negative while it has not
        squeeze_time = torch.full((num_envs,), -1.0, device=device)
        pieces: list[ResetBank] = []
        counts = Counter()
        for _ in range(cfg.max_round_steps):
            actions = runner.compute().clone()
            if cfg.randomize and cfg.action_noise_std > 0.0:
                noise = torch.randn(num_envs, 4, device=device, generator=self._generator) * cfg.action_noise_std
                actions[:, :4] = (actions[:, :4] + noise).clamp(-1.0, 1.0)
            _, _, terminated, truncated, _ = env.step(actions)
            dones = terminated | truncated
            self.stats["rollout_steps"] += 1

            # rollouts that ended in this step: hand the slots of the successful ones to the pool
            if dones.any():
                manager = env.termination_manager
                success = dones & manager.get_term("success")
                for name in self.validator.failure_terms + ["time_out"]:
                    self.stats[f"rollouts_failed/{name}"] += int((dones & ~success & manager.get_term(name)).sum())
                self.stats["rollouts_succeeded"] += int(success.sum())
                done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                success_ids = success.nonzero(as_tuple=False).squeeze(-1)
                if len(success_ids) > 0:
                    piece = self._finalize(success_ids, seen, slots, progress, episode_time)
                    if piece is not None:
                        pieces.append(piece)
                        counts.update({name: n for name, n in piece.bucket_counts().items() if n})
                seen[:, done_ids] = 0
                squeeze_time[done_ids] = -1.0
                runner.reset_idx(done_ids)
                self._draw_sm_params(done_ids)
                if all(counts[name] >= targets[name] for name in SAMPLED_BUCKETS):
                    break

            # capture: one reservoir slot per window and environment
            step = env.episode_length_buf
            ongoing = ~dones & (step >= 1)
            distance = cloth_goal_xy_distance(env, self.command.command, self.command.cfg)
            start_distance = torch.where(ongoing & (step == 1), distance.clamp(min=1.0e-6), start_distance)
            covered = 1.0 - distance / start_distance
            time = step.float() * env.step_dt
            squeezed = self._squeezed()
            squeeze_time = torch.where(squeezed, torch.where(squeeze_time < 0.0, time, squeeze_time), -1.0)
            since_squeeze = torch.where(squeezed, time - squeeze_time, math.inf)
            # speed over the step, from the pose the state machine acted on
            ee_speed = (ee_pose_in_base(env)[0] - runner.ee_pos).norm(dim=-1) / env.step_dt
            windows = self._windows(ongoing, step, covered, distance, ee_speed, since_squeeze)
            for level, eligible in enumerate(windows):
                if BUCKET_NAMES[level] not in SAMPLED_BUCKETS:
                    continue
                seen[level] += eligible.long()
                draw = torch.rand(num_envs, device=device, generator=self._generator)
                replace = eligible & (draw * seen[level] < 1.0)
                ids = replace.nonzero(as_tuple=False).squeeze(-1)
                if len(ids) == 0:
                    continue
                state = capture_reset_states(env, ids)
                if slots[level] is None:
                    slots[level] = empty_like_tree(state, num_envs)
                assign_tree(slots[level], ids, state)
                progress[level, ids] = covered[ids]
                episode_time[level, ids] = time[ids]
        else:
            print(f"[WARN]: Rollouts stopped after {cfg.max_round_steps} steps at {dict(counts)} of {targets}.")
        return ResetBank.concat(pieces) if pieces else None

    def validate(self, candidates: ResetBank) -> tuple[torch.Tensor, torch.Tensor]:
        """Restore the candidates in chunks of the environments and hold them, see :class:`ResetValidator`.

        Returns:
            Whether each candidate is valid and its rejection reason code, each of shape [len(candidates)].
        """
        env, validator = self.env, self.validator
        self.bank_term.set_bank(candidates)
        valid = torch.zeros(len(candidates), dtype=torch.bool, device=self.device)
        reasons = torch.zeros(len(candidates), dtype=torch.long, device=self.device)
        for start in range(0, len(candidates), self.num_envs):
            rows = torch.arange(start, min(start + self.num_envs, len(candidates)), device=self.device)
            env_ids = self._rows[: len(rows)]
            self.bank_term.force_rows(env_ids, rows)
            env.reset()
            validator.start(candidates, rows, env_ids)
            for _ in range(self.cfg.validation_steps):
                _, _, terminated, truncated, _ = env.step(validator.hold_actions())
                validator.update(terminated, truncated)
            valid[rows], reasons[rows] = validator.result()
        self.bank_term.set_bank(None)
        return valid, reasons

    def _squeezed(self) -> torch.Tensor:
        """Whether the gripper holds locked nodes with the squeeze complete, shape [num_envs]."""
        constraint = self.gripper.constraint
        return constraint.active & constraint.contacts_disabled & (constraint.squeeze_progress >= 1.0)

    def _windows(
        self,
        ongoing: torch.Tensor,
        step: torch.Tensor,
        progress: torch.Tensor,
        distance: torch.Tensor,
        ee_speed: torch.Tensor,
        since_squeeze: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Whether each environment is in the capture window of each level, see :data:`~.bank.BUCKET_NAMES`.

        Every window also requires the task stage of its level, so each state is stored in the stage of its bucket.

        * approach: the first step of the episode;
        * grasp (pre-grasp): the state machine holds its pre-grasp pose (for at least ``pregrasp_min_hold``), the arm
          still, the gripper open and nothing locked;
        * drag: a raised node held with the squeeze complete for at most ``drag_window``, the goal point moved by at
          most ``drag_max_progress`` of its distance;
        * goal: a raised node held with the squeeze complete, the goal point within ``goal_distance_window`` of the
          target, the success hold not complete.

        Args:
            ongoing: Environments whose episode runs on after this step, past its first step, shape [N].
            step: Episode step of each environment, shape [N].
            progress: Share of the goal point's start distance to the target covered, shape [N].
            distance: Planar distance of the goal point to the target [m], shape [N].
            ee_speed: End-effector speed over the last step [m/s], shape [N].
            since_squeeze: Time since the squeeze completed [s], infinite while not squeezed, shape [N].
        """
        from ..state_machine import DragSmState

        cfg = self.cfg
        constraint = self.gripper.constraint
        unlocked = ~constraint.active & ~constraint.contacts_disabled
        holding_pose = (self.sm.sm_state == DragSmState.PREGRASP_HOLD) & (self.sm.sm_wait_time >= cfg.pregrasp_min_hold)
        pregrasp = holding_pose & (ee_speed <= cfg.pregrasp_max_speed) & ~self.gripper.closed & unlocked
        low, high = cfg.goal_distance_window
        in_goal_window = (distance >= low) & (distance <= high) & (since_squeeze < math.inf)
        windows = [
            step == 1,
            pregrasp,
            (since_squeeze <= cfg.drag_window) & (progress <= cfg.drag_max_progress),
            in_goal_window & ~self.command.holding,
        ]
        # the bucket index is the stage index
        return [ongoing & window & (self.command.stage == level) for level, window in enumerate(windows)]

    def _finalize(self, env_ids, seen, slots, progress, episode_time) -> ResetBank | None:
        """Collect the captured states of the given environments' finished, successful rollouts."""
        # grasp choice, drag speed, drag height, grasp yaw offset, pre-grasp offset (x, y, z), pre-grasp yaw error
        sm = self.sm
        sm_params = torch.cat(
            [
                torch.stack([self._grasp_choice_label(), sm.drag_speed, sm.drag_height, sm.yaw_offset], dim=1),
                sm.pregrasp_offset,
                sm.pregrasp_yaw_error.unsqueeze(1),
            ],
            dim=1,
        )
        pieces = []
        for level, slot in enumerate(slots):
            filled = seen[level, env_ids] > 0
            ids = env_ids[filled]
            if slot is None or len(ids) == 0:
                continue
            data = {group: {key: value[ids] for key, value in slot[group].items()} for group in slot}
            data["labels"] = {
                "bucket": torch.full((len(ids),), level, dtype=torch.long, device=self.device),
                "progress": progress[level, ids],
                "episode_time": episode_time[level, ids],
                "sm_params": sm_params[ids],
            }
            pieces.append(ResetBank(data))
        return ResetBank.concat(pieces) if pieces else None

    def _draw_sm_params(self, env_ids: torch.Tensor) -> None:
        """Draw the state machine randomization of the rollouts starting in the given environments."""
        cfg, num = self.cfg, len(env_ids)
        if num == 0:
            return
        if not cfg.randomize:
            # the pre-grasp pose straight above the node, at the middle height
            offset = torch.zeros(num, 3, device=self.device)
            offset[:, 2] = sum(cfg.pregrasp_height_range) / 2.0
            self.sm.set_params(env_ids, pregrasp_offset=offset)
            return
        if cfg.random_grasp_node and self.runner.raised_node_ids is not None:
            if self.runner.grasp_choice is None:
                self.runner.grasp_choice = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            num_raised = self.runner.raised_node_ids.shape[1]
            self.runner.grasp_choice[env_ids] = torch.randint(
                num_raised, (num,), device=self.device, generator=self._generator
            )
        self.sm.set_params(
            env_ids,
            drag_height=self._uniform(cfg.drag_height_range, num),
            drag_speed=self._uniform(cfg.drag_speed_range, num),
            yaw_offset=self._uniform(cfg.yaw_offset_range, num),
            pregrasp_offset=self._pregrasp_offsets(num),
            pregrasp_yaw_error=self._uniform(cfg.pregrasp_yaw_error_range, num),
        )

    def _pregrasp_offsets(self, num: int) -> torch.Tensor:
        """Draw pre-grasp positions relative to the grasp node [m], shape [num, 3]: a height in
        ``pregrasp_height_range``, a planar offset in a cone whose radius grows with the height over
        ``pregrasp_radius_range``, its radius uniform (biased to the centre) and its direction uniform."""
        cfg = self.cfg
        (h_low, h_high), (r_low, r_high) = cfg.pregrasp_height_range, cfg.pregrasp_radius_range
        height = self._uniform(cfg.pregrasp_height_range, num)
        share = (height - h_low) / max(h_high - h_low, 1.0e-9)
        radius = (r_low + (r_high - r_low) * share) * self._uniform((0.0, 1.0), num)
        angle = self._uniform((0.0, 2.0 * math.pi), num)
        return torch.stack([radius * torch.cos(angle), radius * torch.sin(angle), height], dim=1)

    def _uniform(self, bounds: tuple[float, float], num: int) -> torch.Tensor:
        low, high = bounds
        return low + (high - low) * torch.rand(num, device=self.device, generator=self._generator)

    def _grasp_choice_label(self) -> torch.Tensor:
        """Index of the grasped raised node per environment, -1 for the state machine's own choice."""
        if self.runner.grasp_choice is None:
            return torch.full((self.num_envs,), -1.0, device=self.device)
        return self.runner.grasp_choice.float()

    def _record_validation(self, candidates, valid, reasons, acceptance) -> None:
        """Add the validation outcome to the statistics and update the acceptance estimates."""
        buckets = candidates.data["labels"]["bucket"]
        for index, name in enumerate(BUCKET_NAMES):
            in_bucket = buckets == index
            total = int(in_bucket.sum())
            self.stats[f"candidates/{name}"] += total
            self.stats[f"validated/{name}"] += int((in_bucket & valid).sum())
            for code in reasons[in_bucket & ~valid].unique().tolist():
                count = int((reasons[in_bucket & ~valid] == code).sum())
                reason = self.validator.reasons[code - 1] if code > 0 else "unknown"
                self.stats[f"rejected/{name}/{reason}"] += count
            if name in acceptance and self.stats[f"candidates/{name}"] > 0:
                acceptance[name] = self.stats[f"validated/{name}"] / self.stats[f"candidates/{name}"]

    def _summary(self, candidates, valid, quotas, accepted) -> str:
        """One line on the round: candidates, valid ones and the filled buckets."""
        total = ResetBank.concat(accepted).bucket_counts()
        parts = [f"{len(candidates)} candidates, {int(valid.sum())} valid"]
        parts += [f"{name} {total[name]}/{quotas[name]}" for name in SAMPLED_BUCKETS]
        return ", ".join(parts)


def _git_commit() -> str | None:
    """Commit hash of the working tree, or None outside a git checkout."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None

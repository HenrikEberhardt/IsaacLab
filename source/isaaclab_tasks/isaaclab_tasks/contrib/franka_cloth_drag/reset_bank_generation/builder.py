# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build a bank of reset states with the randomized state machine.

Rollouts of the state machine (:mod:`..state_machine.cloth_drag_sm`) capture candidates in one window per
start stage (:mod:`.capture`), the candidates are validated by replay (:mod:`.validation`), and the valid ones fill
the bank. Used by ``scripts/generate_reset_bank.py``. See ``implementation_log/reset_states.md``.
"""

from __future__ import annotations

import math
import subprocess
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import torch

from ..mdp.stages import START_STAGE_NAMES
from ..reset_strategies.bank import SCHEMA_VERSION, ResetBank, task_fingerprint
from ..state_machine import ClothDragSmRandomizationCfg, ee_down_tilt, limit_arm_action, sample_sm_params
from .capture import capture_reset_states
from .validation import ResetStateValidator

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

    from ..reset_strategies import ResetFromBank
    from ..state_machine import ClothDragSmRunner


@dataclass
class ResetBankGenerationCfg:
    """Settings of the reset-state generation."""

    num_states: int = 4000
    """Number of stored states, split evenly over the start stages (:data:`~..mdp.stages.START_STAGE_NAMES`): the
    approach, pre-grasp, fresh-grasp and goal-zone states. Not used with :attr:`stage_quotas`."""

    stage_quotas: dict[str, int] | None = None
    """Number of stored states per start stage name, e.g. ``{"approach": 8000, "grasp": 1000, "drag": 100, "goal":
    100}``; every stage needs at least one. None splits :attr:`num_states` evenly."""

    approach_start_share: float = 0.5
    """Share of the rollouts whose approach state is the first step of the episode; the others store a state drawn
    uniformly over the steps before the GRASP stage is first reached."""

    validation_steps: int = 30
    """Steps each candidate is held after its restore before it counts as valid."""

    sm_randomization: ClothDragSmRandomizationCfg = field(default_factory=ClothDragSmRandomizationCfg)
    """Per-rollout state-machine randomization."""

    pregrasp_max_speed: float = 0.05
    """Fastest end-effector speed of a captured pre-grasp state [m/s]; states are also captured on the way into the
    pose."""

    pregrasp_min_hold: float = 0.0
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

    max_ee_down_tilt_deg: float = 1.0
    """Largest end-effector tilt from base-frame down [deg] while a restored candidate is held."""

    candidate_margin: float = 1.5
    """Candidates collected per missing state, on top of the expected rejections."""

    max_rounds: int = 20
    """Most rounds of rollouts and validation before giving up."""

    max_round_steps: int = 3000
    """Most environment steps of the rollouts of one round."""

    seed: int = 0
    """Seed of the state machine randomization and the final pick."""

    def __post_init__(self):
        if self.num_states < len(START_STAGE_NAMES):
            raise ValueError(f"num_states must be at least {len(START_STAGE_NAMES)}, one per stage.")
        if self.stage_quotas is not None:
            if set(self.stage_quotas) != set(START_STAGE_NAMES):
                raise ValueError(f"stage_quotas needs exactly the stages {START_STAGE_NAMES}, got {self.stage_quotas}.")
            if any(quota < 1 for quota in self.stage_quotas.values()):
                raise ValueError(f"Every stage quota must be at least one, got {self.stage_quotas}.")
        if not 0.0 <= self.approach_start_share <= 1.0:
            raise ValueError(f"approach_start_share must lie in [0, 1], got {self.approach_start_share}.")
        if self.validation_steps <= 0:
            raise ValueError("validation_steps must be positive.")
        if self.goal_distance_window[0] > self.goal_distance_window[1]:
            raise ValueError(f"goal_distance_window must be increasing, got {self.goal_distance_window}.")
        if not 0.0 <= self.max_ee_down_tilt_deg <= 180.0:
            raise ValueError(f"max_ee_down_tilt_deg must lie in [0, 180], got {self.max_ee_down_tilt_deg}.")

    def quotas(self) -> dict[str, int]:
        """Number of stored states per start stage."""
        if self.stage_quotas is not None:
            return {name: int(self.stage_quotas[name]) for name in START_STAGE_NAMES}
        base, extra = divmod(self.num_states, len(START_STAGE_NAMES))
        return {name: base + (i < extra) for i, name in enumerate(START_STAGE_NAMES)}


def add_bank_term(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Add the bank reset event ``reset_from_bank`` after the default reset events of an environment configuration.

    The term loads no bank and draws nothing: it restores the rows forced on it (candidates, see
    :meth:`~..reset_strategies.ResetFromBank.force_rows`) and leaves the other environments as the default resets
    made them. The bank generation and its probes use it.

    Args:
        env_cfg: The environment configuration, with the default resets as its events.

    Raises:
        ValueError: If the events are not the default resets.
    """
    from ..reset_strategies import ResetFromBankCfg

    events = env_cfg.events
    if getattr(events, "reset_deformable", None) is None:
        raise ValueError(
            "The bank generation starts its rollouts with the default resets: use env.events=default_resets."
        )
    events.reset_from_bank = ResetFromBankCfg()


class ResetBankBuilder:
    """Run the randomized state machine, capture candidate reset states, validate them and fill a bank.

    Rollouts stream in all environments, started by the task's default reset events: every success hands its captured
    task-start, grasp, drag and goal states to the candidate pool, and every failure is dropped. Each window has
    one reservoir slot per environment, so the capture time is uniform over its EE-down-aligned steps. Candidates are
    then restored in chunks of the environments through :class:`~..reset_strategies.ResetFromBank` and checked by
    :class:`~.validation.ResetStateValidator`. Rounds of rollouts and validation alternate until every stage of
    :meth:`ResetBankGenerationCfg.quotas` is full.
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        runner: ClothDragSmRunner,
        bank_term: ResetFromBank,
        cfg: ResetBankGenerationCfg,
    ):
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
        self.validator = ResetStateValidator(
            env,
            cfg.max_node_speed,
            cfg.max_cube_tilt_deg,
            max_ee_down_tilt_deg=cfg.max_ee_down_tilt_deg,
        )
        self.num_envs, self.device = env.num_envs, env.device
        self._rows = torch.arange(self.num_envs, device=self.device)
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(cfg.seed)
        self._cpu_generator = torch.Generator().manual_seed(cfg.seed)
        # whether the rollout of every environment stores its first step as its approach state
        self._approach_at_start = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.stats: Counter = Counter()

    def run(self) -> ResetBank:
        """Collect until every stage is full and return the bank, with its meta information."""
        from ..mdp.frames import raised_nodes_term

        cfg = self.cfg
        quotas = cfg.quotas()
        start_time = time.time()
        accepted: list[ResetBank] = []
        acceptance = dict.fromkeys(START_STAGE_NAMES, 1.0)
        for round_id in range(cfg.max_rounds):
            counts = ResetBank.concat(accepted).stage_counts() if accepted else dict.fromkeys(START_STAGE_NAMES, 0)
            missing = {name: max(0, quotas[name] - counts[name]) for name in START_STAGE_NAMES}
            if not any(missing.values()):
                break
            targets = {
                name: math.ceil(missing[name] * cfg.candidate_margin / max(acceptance[name], 0.2))
                for name in START_STAGE_NAMES
            }
            candidates = self.collect_candidates(targets)
            if candidates is None:
                print(f"[INFO]: Round {round_id}: no successful rollout.")
                continue
            # every rollout gives a candidate of each stage: validate only as many as a stage still needs
            found = candidates.stage_counts()
            caps = {name: min(targets[name], found[name]) for name in START_STAGE_NAMES}
            candidates = candidates.select(subsample_to_quotas(candidates, caps, self._cpu_generator).to(self.device))
            valid, reasons = self.validate(candidates)
            accepted.append(candidates.select(valid.nonzero(as_tuple=False).squeeze(-1)))
            self._record_validation(candidates, valid, reasons, acceptance)
            print(f"[INFO]: Round {round_id}: {self._summary(candidates, valid, quotas, accepted)}")
        else:
            counts = ResetBank.concat(accepted).stage_counts() if accepted else dict.fromkeys(START_STAGE_NAMES, 0)
            if any(counts[name] < quotas[name] for name in START_STAGE_NAMES):
                raise RuntimeError(f"Could not fill the stages in {cfg.max_rounds} rounds: {counts} of {quotas}.")

        pool = ResetBank.concat(accepted)
        generator = torch.Generator().manual_seed(cfg.seed)
        bank = pool.select(subsample_to_quotas(pool, quotas, generator))
        self.stats["wall_time_s"] = round(time.time() - start_time, 1)
        bank.meta = {
            "schema_version": SCHEMA_VERSION,
            "fingerprint": task_fingerprint(self.env),
            "node_raise_height": float(raised_nodes_term(self.env).node_raise_height),
            "task": self.env.spec.id if getattr(self.env, "spec", None) is not None else None,
            "created": datetime.now().isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "generator": asdict(cfg) | {"num_envs": self.num_envs},
            "statistics": dict(self.stats),
        }
        return bank

    def collect_candidates(self, targets: dict[str, int]) -> ResetBank | None:
        """Run rollouts in all environments until every stage has ``targets[name]`` candidates.

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
        num_stages = len(START_STAGE_NAMES)
        seen = torch.zeros(num_stages, num_envs, dtype=torch.long, device=device)
        slots: list[dict | None] = [None] * num_stages
        start_distance = torch.ones(num_envs, device=device)
        # episode time at which the squeeze completed, negative while it has not
        squeeze_time = torch.full((num_envs,), -1.0, device=device)
        pieces: list[ResetBank] = []
        counts = Counter()
        for _ in range(cfg.max_round_steps):
            actions = runner.compute().clone()
            if cfg.action_noise_std > 0.0:
                noise = torch.randn(num_envs, 4, device=device, generator=self._generator) * cfg.action_noise_std
                actions[:, :4] += noise
                actions[:, :4] = limit_arm_action(actions[:, :4], self.sm.cfg.arm_action_limit)
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
                    piece = self._finalize(success_ids, seen, slots)
                    if piece is not None:
                        pieces.append(piece)
                        counts.update({name: n for name, n in piece.stage_counts().items() if n})
                seen[:, done_ids] = 0
                squeeze_time[done_ids] = -1.0
                runner.reset_idx(done_ids)
                self._draw_sm_params(done_ids)
                if all(counts[name] >= targets[name] for name in START_STAGE_NAMES):
                    break

            # capture: one reservoir slot per window and environment
            step = env.episode_length_buf
            ongoing = ~dones & (step >= 1)
            distance = cloth_goal_xy_distance(env, self.command.command, self.command.cfg)
            start_distance = torch.where(ongoing & (step == 1), distance.clamp(min=1.0e-6), start_distance)
            covered = 1.0 - distance / start_distance
            episode_time = step.float() * env.step_dt
            squeezed = self._squeezed()
            squeeze_time = torch.where(squeezed, torch.where(squeeze_time < 0.0, episode_time, squeeze_time), -1.0)
            since_squeeze = torch.where(squeezed, episode_time - squeeze_time, math.inf)
            # speed over the step, from the pose the state machine acted on
            ee_pos, ee_quat = ee_pose_in_base(env)
            ee_speed = (ee_pos - runner.ee_pos).norm(dim=-1) / env.step_dt
            windows = self._windows(ongoing, step, covered, distance, ee_speed, since_squeeze)
            tilt_ok = torch.rad2deg(ee_down_tilt(ee_quat)) <= cfg.max_ee_down_tilt_deg
            windows = [window & tilt_ok for window in windows]
            for stage, eligible in enumerate(windows):
                seen[stage] += eligible.long()
                draw = torch.rand(num_envs, device=device, generator=self._generator)
                replace = eligible & (draw * seen[stage] < 1.0)
                ids = replace.nonzero(as_tuple=False).squeeze(-1)
                if len(ids) == 0:
                    continue
                state = capture_reset_states(env, ids)
                if slots[stage] is None:
                    slots[stage] = _map_tree(lambda value: value.new_zeros((num_envs, *value.shape[1:])), state)
                _assign_tree(slots[stage], ids, state)
        else:
            print(f"[WARN]: Rollouts stopped after {cfg.max_round_steps} steps at {dict(counts)} of {targets}.")
        return ResetBank.concat(pieces) if pieces else None

    def validate(self, candidates: ResetBank) -> tuple[torch.Tensor, torch.Tensor]:
        """Restore the candidates in chunks and hold them, see :class:`~.validation.ResetStateValidator`.

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
        """Whether each environment is in the capture window of each start stage
        (:data:`~..mdp.stages.START_STAGE_NAMES`).

        Every window also requires its task stage, so each state is stored in the stage it starts in.

        * approach: the first step of the episode, or, for the rollouts drawn without ``approach_start_share``, any step
          before the GRASP stage is first reached;
        * grasp (pre-grasp): the state machine holds or moves into its pre-grasp pose (in it for at least
          ``pregrasp_min_hold``), the arm at most ``pregrasp_max_speed`` fast, the gripper open and nothing locked;
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
        from ..mdp.stages import ClothDragStage

        cfg = self.cfg
        constraint = self.gripper.constraint
        unlocked = ~constraint.active & ~constraint.contacts_disabled
        holding_pose = self.sm.pregrasp_ready & (self.sm.sm_wait_time >= cfg.pregrasp_min_hold)
        pregrasp = holding_pose & (ee_speed <= cfg.pregrasp_max_speed) & ~self.gripper.closed & unlocked
        # the stage is not monotonic (a failed try can leave the approach radius again): only before the first GRASP
        before_grasp = self.command.time_to_stage[:, int(ClothDragStage.GRASP)] < 0.0
        low, high = cfg.goal_distance_window
        in_goal_window = (distance >= low) & (distance <= high) & (since_squeeze < math.inf)
        windows = [
            torch.where(self._approach_at_start, step == 1, before_grasp),
            pregrasp,
            (since_squeeze <= cfg.drag_window) & (progress <= cfg.drag_max_progress),
            in_goal_window & ~self.command.holding,
        ]
        return [ongoing & window & (self.command.stage == stage) for stage, window in enumerate(windows)]

    def _finalize(self, env_ids: torch.Tensor, seen: torch.Tensor, slots: list[dict | None]) -> ResetBank | None:
        """Collect the captured states of the given environments' finished, successful rollouts."""
        pieces = []
        for stage, slot in enumerate(slots):
            ids = env_ids[seen[stage, env_ids] > 0]
            if slot is not None and len(ids) > 0:
                pieces.append(ResetBank(_map_tree(lambda value: value[ids], slot)))
        return ResetBank.concat(pieces) if pieces else None

    def _draw_sm_params(self, env_ids: torch.Tensor) -> None:
        """Draw the state machine randomization of the rollouts starting in the given environments."""
        cfg, num = self.cfg, len(env_ids)
        if num == 0:
            return
        draw = torch.rand(num, device=self.device, generator=self._generator)
        self._approach_at_start[env_ids] = draw < cfg.approach_start_share
        params = sample_sm_params(self.sm.cfg, cfg.sm_randomization, num, self.device, self._generator)
        self.sm.set_params(env_ids, params)

    def _record_validation(self, candidates, valid, reasons, acceptance) -> None:
        """Add the validation outcome to the statistics and update the acceptance estimates."""
        for index, name in enumerate(START_STAGE_NAMES):
            in_stage = candidates.stage == index
            self.stats[f"candidates/{name}"] += int(in_stage.sum())
            self.stats[f"validated/{name}"] += int((in_stage & valid).sum())
            for code in reasons[in_stage & ~valid].unique().tolist():
                count = int((reasons[in_stage & ~valid] == code).sum())
                reason = self.validator.reasons[code - 1] if code > 0 else "unknown"
                self.stats[f"rejected/{name}/{reason}"] += count
            if name in acceptance and self.stats[f"candidates/{name}"] > 0:
                acceptance[name] = self.stats[f"validated/{name}"] / self.stats[f"candidates/{name}"]

    def _summary(self, candidates, valid, quotas, accepted) -> str:
        """One line on the round: candidates, valid ones and the filled stages."""
        total = ResetBank.concat(accepted).stage_counts()
        parts = [f"{len(candidates)} candidates, {int(valid.sum())} valid"]
        parts += [f"{name} {total[name]}/{quotas[name]}" for name in START_STAGE_NAMES]
        return ", ".join(parts)


def _git_commit() -> str | None:
    """Commit hash of the working tree, or None outside a git checkout."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def subsample_to_quotas(
    bank: ResetBank, quotas: dict[str, int], generator: torch.Generator | None = None
) -> torch.Tensor:
    """Pick the rows of a bank to keep: ``quotas[name]`` random rows of each named start stage.

    Args:
        bank: The bank to pick from.
        quotas: Number of rows to keep per start stage name, e.g. ``{"approach": 1000, "grasp": 1000}``.
        generator: Random generator on the CPU, for a reproducible pick.

    Returns:
        The kept rows, sorted, on the CPU.

    Raises:
        ValueError: If a stage has fewer rows than its quota.
    """
    kept = []
    for name, quota in quotas.items():
        rows = bank.stage_rows(name).cpu()
        if len(rows) < quota:
            raise ValueError(f"Stage '{name}' has {len(rows)} states, fewer than its quota of {quota}.")
        kept.append(rows[torch.randperm(len(rows), generator=generator)[:quota]])
    return (torch.cat(kept) if kept else torch.zeros(0, dtype=torch.long)).sort().values


def _map_tree(fn: Callable[[torch.Tensor], torch.Tensor], tree: dict) -> dict:
    """Apply ``fn`` to every tensor of a nested dictionary and return the new dictionary."""
    return {key: _map_tree(fn, value) if isinstance(value, dict) else fn(value) for key, value in tree.items()}


def _assign_tree(dst: dict, ids: torch.Tensor, src: dict) -> None:
    """Write ``src`` into the rows ``ids`` of ``dst``, two nested dictionaries of the same layout."""
    for key, value in src.items():
        if isinstance(value, dict):
            _assign_tree(dst[key], ids, value)
        else:
            dst[key][ids] = value

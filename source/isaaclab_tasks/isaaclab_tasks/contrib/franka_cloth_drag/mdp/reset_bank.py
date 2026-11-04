# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset event that continues stored reset states of the Franka cloth dragging task."""

from __future__ import annotations

import os
import tempfile
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import EventTermCfg, ManagerTermBase

from ..reset_states import (
    ResetBank,
    restore_physical,
    restore_task_state,
    task_fingerprint,
)
from ..reset_states.bank import (
    BUCKET_NAMES,
    RESET_MODES,
    cascade_shares,
    check_min_probabilities,
    check_mode_probabilities,
    sample_reset_rows,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ResetFromBank(ManagerTermBase):
    """Reset environments to stored reset states of a :class:`~..reset_states.ResetBank`.

    Every reset draws a mode per environment by ``mode_probabilities`` (:data:`~..reset_states.bank.RESET_MODES`): the
    task's default reset, or a uniformly drawn stored state of the ``grasp``, ``drag`` or ``goal`` bucket. Rows given
    with :meth:`force_rows` take precedence for the next reset. Without a bank, every reset is the default one.

    With ``min_probabilities``, the shares adapt to the success of the buckets: ``mode_probabilities`` are the start
    shares, and a solved bucket gives up its share to the stage before it, down to its minimum (see
    :func:`~..reset_states.bank.cascade_shares`). The success of a source is its smoothed task success, see below. The
    shares keep their start values for the first ``hold_steps`` environment steps.

    The term must come after the other reset events, so the default resets run first and a stored state overwrites
    them. It restores the physical state when called, and the task state that the action and command resets clear in
    its :meth:`reset`, which the event manager runs after those resets (see :mod:`~..reset_states.restore`).

    Every reset after the first also logs, in ``extras["log"]``, per source the episodes that just ended started from
    (``default`` or the bucket name):

    * ``Reset/task_success_<source>_raw``: their success rate.
    * ``Reset/stage_success_<source>_raw``: the share that reached the stage after their start stage (GRASP for default
      resets, the success for ``goal``).
    * ``Reset/task_success_<source>_smoothed``, ``Reset/stage_success_<source>_smoothed``: the same rates over all
      episodes of the source that ended in the last ``window_steps`` environment steps. At the start of training, the
      window holds the steps so far.

    A value without an episode to average is left out. The reset also logs the share of each reset mode among the
    episodes that start now (``Reset/share_<mode>_raw``), with or without a bank. For every stored reset state, it also
    tracks an EMA of reaching the next stage and of completing the task, along with the number of finished episodes
    observed from that state.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        """Load the bank, if any. It is checked against the task at the first reset, once all managers exist."""
        super().__init__(cfg, env)
        self.bank: ResetBank | None = None
        """The stored states, on the environment's device; None for default resets only."""
        bank_path = cfg.params.get("bank_path")
        if bank_path is not None:
            self.bank = ResetBank.load(_resolve_bank_path(bank_path), env.device)
        self._row_ema_episodes = cfg.params.get("row_ema_episodes", 8)
        if self._row_ema_episodes <= 0:
            raise ValueError(f"row_ema_episodes must be positive, got {self._row_ema_episodes}.")
        self._row_stats_export_interval_steps = int(cfg.params.get("row_stats_export_interval_steps", 0))
        if self._row_stats_export_interval_steps < 0:
            raise ValueError(
                f"row_stats_export_interval_steps must be non-negative, got {self._row_stats_export_interval_steps}."
            )
        log_dir = getattr(getattr(env, "cfg", None), "log_dir", None)
        self._row_stats_dir = (
            Path(log_dir) / "reset_state_success"
            if self._row_stats_export_interval_steps > 0 and log_dir is not None and int(os.getenv("RANK", "0")) == 0
            else None
        )
        self._next_row_stats_export_step = self._row_stats_export_interval_steps
        self._reset_row_estimates()
        self._fingerprint_checked = bank_path is None
        self._mode_probabilities = cfg.params.get("mode_probabilities") or {"default": 1.0}
        self._shares = check_mode_probabilities(self._mode_probabilities, self.bank)
        self._window_steps = cfg.params.get("window_steps", 480)
        if self._window_steps <= 0:
            raise ValueError(f"window_steps must be positive, got {self._window_steps}.")
        # per step of the moving window, the episodes, task and stage successes of every source, and their sum
        self._window: deque[tuple[int, torch.Tensor]] = deque()
        self._window_totals = torch.zeros(3, 1 + len(BUCKET_NAMES), dtype=torch.float64)
        # adaptive shares: start shares, minimum shares (None for fixed shares) and smoothed task success of the modes
        self._start_shares = self._shares
        self._min_shares: torch.Tensor | None = None
        self._task_success = torch.zeros(len(RESET_MODES))
        self._success_range = cfg.params.get("success_range", (0.85, 0.95))
        self._hold_steps = cfg.params.get("hold_steps", 0)
        if self._hold_steps < 0:
            raise ValueError(f"hold_steps must be non-negative, got {self._hold_steps}.")
        min_probabilities = cfg.params.get("min_probabilities")
        if min_probabilities is not None:
            self._min_shares = check_min_probabilities(min_probabilities, self._start_shares, self._success_range)
        self.rows = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        """Bank row the current episode of every environment started from, -1 for a default reset."""
        self._forced_rows = torch.full_like(self.rows, -1)
        self._forced = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        # environments whose task state the reset hook still has to restore
        self._pending = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._command_name = cfg.params.get("command_name", "cloth_target")
        # log of the current reset, written by the reset hook once the environment has recreated extras["log"]
        self._log: dict[str, float] = {}

    def set_bank(self, bank: ResetBank | None) -> None:
        """Use another bank, captured in this environment (e.g. candidates during generation), so it is not checked
        against the task; the mode shares keep applying to it."""
        self.bank = None if bank is None else bank.to(self.device)
        self._shares = check_mode_probabilities(self._mode_probabilities, self.bank)
        self._reset_row_estimates()
        self._fingerprint_checked = True

    def force_rows(self, env_ids: torch.Tensor, rows: torch.Tensor) -> None:
        """Reset the given environments to the given bank rows at their next reset, regardless of the mode shares.

        Args:
            env_ids: The environments, shape [N].
            rows: Rows of the bank, shape [N]; -1 forces the default reset.
        """
        if self.bank is None and (rows >= 0).any():
            raise RuntimeError("Rows can only be forced with a bank, see set_bank.")
        self._forced_rows[env_ids] = rows.to(self.device)
        self._forced[env_ids] = True

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        bank_path: str | None = None,
        mode_probabilities: dict[str, float] | None = None,
        allow_settings_mismatch: bool = False,
        command_name: str = "cloth_target",
        min_probabilities: dict[str, float] | None = None,
        success_range: tuple[float, float] = (0.85, 0.95),
        window_steps: int = 480,
        hold_steps: int = 0,
        row_ema_episodes: int = 8,
        row_stats_export_interval_steps: int = 0,
    ) -> None:
        """Draw the reset of every environment and restore the physical state of the stored ones.

        Args:
            env: The environment.
            env_ids: The environments to reset.
            bank_path: Path of the bank (``.pt``) to load at initialization, or None for default resets only. A
                relative path is looked up in the working directory, then in the repository root.
            mode_probabilities: Share of the resets per mode, e.g. ``{"default": 0.25, "grasp": 0.25, "drag": 0.25,
                "goal": 0.25}``; normalized. Defaults to default resets only.
            allow_settings_mismatch: Whether a bank made with other physics or task settings may be loaded.
            command_name: Name of the cloth target command, whose ``success`` still holds the outcome of the finished
                episodes (the command reset clears it later in the same reset).
            min_probabilities: Minimum share of the resets per mode. Enables the adaptive shares, with
                ``mode_probabilities`` as the start shares. Defaults to fixed shares.
            success_range: Smoothed task success (low, high) between which a bucket gives up its share.
            window_steps: Number of environment steps the smoothed success rates span.
            hold_steps: Number of environment steps the adaptive shares keep their start values.
            row_ema_episodes: Number of finished episodes from a stored reset state that its success EMA effectively
                spans.
            row_stats_export_interval_steps: Number of environment steps between atomic snapshots of all per-state
                estimates under the run's ``reset_state_success`` directory. Zero disables snapshots.
        """
        if not self._fingerprint_checked:
            # the event terms are built before the action and command managers that the fingerprint reads
            self.bank.check_fingerprint(task_fingerprint(env), allow_settings_mismatch)
            self._fingerprint_checked = True
        ids = self._resolve(env_ids)
        # the reset at the start of training ends no episode
        finished = self.rows[ids] if env.common_step_counter > 0 else None
        drawn = ~self._forced[ids]
        rows = torch.where(drawn, -1, self._forced_rows[ids])
        self._forced[ids] = False
        if self.bank is not None and drawn.any() and self._shares[0] < 1.0:
            rows[drawn] = sample_reset_rows(self.bank, self._shares, int(drawn.sum()))
        self._log = {} if finished is None else self._source_log(env, ids, finished, rows)
        self.rows[ids] = rows
        stored = rows >= 0
        self._pending[ids] = stored
        if stored.any():
            restore_physical(env, self.bank, rows[stored], ids[stored])

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Restore the task state of the environments reset to stored states, after the action and command resets,
        and log the reset."""
        ids = self._resolve(env_ids)
        ids = ids[self._pending[ids]]
        if len(ids) > 0:
            restore_task_state(self._env, self.bank, self.rows[ids], ids)
            self._pending[ids] = False
        if self._log:
            self._env.extras.setdefault("log", {}).update(self._log)
            self._log = {}

    def _source_log(
        self, env: ManagerBasedRLEnv, env_ids: torch.Tensor, finished: torch.Tensor, started: torch.Tensor
    ) -> dict[str, float]:
        """Raw and smoothed success of the finished episodes per source, and share of each reset mode among the
        started ones.

        Args:
            env: The environment.
            env_ids: The environments being reset, shape [N].
            finished: Bank rows the finished episodes started from, -1 for a default reset, shape [N].
            started: Bank rows the new episodes start from, -1 for a default reset, shape [N].
        """
        names = ("default", *BUCKET_NAMES)
        command = env.command_manager.get_term(self._command_name)
        success = command.success[env_ids].float()
        finished_source, started_source = self._source(finished), self._source(started)
        # the buckets are named after their stage, so a bucket's index is its start stage; default resets start in
        # APPROACH. Column k of reached is the stage after stage k, the success after GOAL.
        start_stage = (finished_source - 1).clamp(min=0).unsqueeze(1)
        reached = (command.time_to_stage[env_ids, 1:] >= 0.0).float()
        reached = torch.cat([reached, success.unsqueeze(1)], dim=1).gather(1, start_stage).squeeze(1)
        self._update_row_estimates(finished, reached, success)
        self._maybe_export_row_estimates(env.common_step_counter)
        counts = torch.bincount(finished_source, minlength=len(names))
        task_sums = torch.bincount(finished_source, weights=success, minlength=len(names))
        stage_sums = torch.bincount(finished_source, weights=reached, minlength=len(names))
        shares = torch.bincount(started_source, minlength=len(names)) / max(len(started), 1)
        # one transfer to the host for all values
        values = torch.stack([counts.float(), task_sums, stage_sums, shares]).cpu()
        window = self._update_window(env.common_step_counter, values[:3].double()).tolist()
        log = self._success_log(names, "raw", *values[:3].tolist()) | self._success_log(names, "smoothed", *window)
        log |= {
            f"Reset/share_{name}_raw": share for name, share in zip(names, values[3].tolist()) if name in RESET_MODES
        }
        if self._min_shares is not None:
            self._adapt_shares(env.common_step_counter, dict(zip(names, window[0])), dict(zip(names, window[1])))
        return log

    def _update_window(self, step: int, episodes: torch.Tensor) -> torch.Tensor:
        """Add the episodes that ended in this step to the moving window and drop the steps that left it.

        Args:
            step: The environment step.
            episodes: Number of episodes, task successes and stage successes per source, shape [3, S].

        Returns:
            The totals over the window, shape [3, S].
        """
        self._window.append((step, episodes))
        self._window_totals += episodes
        while self._window[0][0] <= step - self._window_steps:
            self._window_totals -= self._window.popleft()[1]
        return self._window_totals

    @staticmethod
    def _success_log(
        names: Sequence[str], suffix: str, counts: list[float], task_sums: list[float], stage_sums: list[float]
    ) -> dict[str, float]:
        """Task and stage success rate of the sources with episodes.

        A source without an episode is left out: a placeholder would distort the logger's mean.

        Args:
            names: Names of the sources.
            suffix: Suffix of the log keys.
            counts: Number of episodes per source.
            task_sums: Number of episodes that completed the task per source.
            stage_sums: Number of episodes that reached the stage after their start stage per source.
        """
        log = {}
        for name, num, task, stage in zip(names, counts, task_sums, stage_sums):
            if num > 0:
                log[f"Reset/task_success_{name}_{suffix}"] = task / num
                log[f"Reset/stage_success_{name}_{suffix}"] = stage / num
        return log

    def _reset_row_estimates(self) -> None:
        """Clear the success estimates of every stored reset state."""
        num_rows = 0 if self.bank is None else len(self.bank)
        self.row_next_stage_success: torch.Tensor = torch.zeros(num_rows, device=self.device)
        """EMA of reaching the stage after each stored reset state, shape [N]."""
        self.row_full_task_success: torch.Tensor = torch.zeros(num_rows, device=self.device)
        """EMA of completing the task from each stored reset state, shape [N]."""
        self.row_episode_count: torch.Tensor = torch.zeros(num_rows, dtype=torch.long, device=self.device)
        """Number of finished episodes observed from every stored reset state, shape [N]."""

    def _update_row_estimates(
        self, finished: torch.Tensor, reached_next: torch.Tensor, full_success: torch.Tensor
    ) -> None:
        """Fold finished stored-state episodes into their per-row success estimates.

        Args:
            finished: Bank rows the episodes started from, with -1 for default resets, shape [N].
            reached_next: Whether each episode reached the stage after its starting stage, shape [N].
            full_success: Whether each episode completed the task, shape [N].
        """
        stored = finished >= 0
        if self.bank is None or not stored.any():
            return
        rows = finished[stored]
        num_rows = len(self.bank)
        counts = torch.bincount(rows, minlength=num_rows)
        next_sums = torch.bincount(rows, weights=reached_next[stored], minlength=num_rows)
        full_sums = torch.bincount(rows, weights=full_success[stored], minlength=num_rows)
        observed = counts > 0
        observed_rows = observed.nonzero(as_tuple=False).squeeze(-1)
        batch_counts = counts[observed].float()
        alpha = batch_counts / (batch_counts + self._row_ema_episodes)
        next_rate = next_sums[observed] / batch_counts
        full_rate = full_sums[observed] / batch_counts
        self.row_next_stage_success[observed_rows] += alpha * (next_rate - self.row_next_stage_success[observed_rows])
        self.row_full_task_success[observed_rows] += alpha * (full_rate - self.row_full_task_success[observed_rows])
        self.row_episode_count += counts

    def _maybe_export_row_estimates(self, env_step: int) -> None:
        """Atomically snapshot all per-state estimates when the configured environment-step interval elapses."""
        if self._row_stats_dir is None or self.bank is None or env_step < self._next_row_stats_export_step:
            return
        self._next_row_stats_export_step = (
            env_step // self._row_stats_export_interval_steps + 1
        ) * self._row_stats_export_interval_steps
        payload = {
            "schema_version": 1,
            "env_step": int(env_step),
            "row_ema_episodes": self._row_ema_episodes,
            "bank_fingerprint": self.bank.meta.get("fingerprint", {}),
            "bucket": self.bank.data["labels"]["bucket"].detach().cpu(),
            "next_stage_success": self.row_next_stage_success.detach().cpu(),
            "full_task_success": self.row_full_task_success.detach().cpu(),
            "episode_count": self.row_episode_count.detach().cpu(),
        }
        self._save_row_estimates(self._row_stats_dir / f"step_{env_step:08d}.pt", payload)
        self._save_row_estimates(self._row_stats_dir / "latest.pt", payload)

    @staticmethod
    def _save_row_estimates(path: Path, payload: dict) -> None:
        """Save one per-state estimate snapshot, replacing an existing path only after the write completes."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as stream:
                temporary = Path(stream.name)
                torch.save(payload, stream)
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _adapt_shares(self, step: int, counts: dict[str, float], successes: dict[str, float]) -> None:
        """Set the shares of the next resets from the smoothed task success of every mode, once the hold has ended.

        A mode without an episode in the window keeps its last value, so a bucket that gave up all its share stays
        solved.

        Args:
            step: The environment step.
            counts: Number of episodes in the window per source.
            successes: Number of successful ones per source.
        """
        for mode_id, mode in enumerate(RESET_MODES):
            if counts[mode] > 0:
                self._task_success[mode_id] = successes[mode] / counts[mode]
        if step >= self._hold_steps:
            self._shares = cascade_shares(self._start_shares, self._min_shares, self._task_success, self._success_range)

    def _source(self, rows: torch.Tensor) -> torch.Tensor:
        """Index of the source of each bank row in ``("default", *BUCKET_NAMES)``: 0 for -1, else its bucket + 1."""
        source = torch.zeros_like(rows)
        stored = rows >= 0
        if stored.any():
            source[stored] = self.bank.data["labels"]["bucket"][rows[stored]].to(rows.dtype) + 1
        return source

    def _resolve(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        """Environment indices as a long tensor on the environment's device."""
        if env_ids is None or isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)[env_ids if env_ids is not None else slice(None)]
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)


def _resolve_bank_path(path: str | Path) -> Path:
    """The bank file: ``path`` if it is absolute or exists in the working directory, else relative to the repository
    root, so runs from another working directory find the banks under the repository's ``logs``.

    Raises:
        FileNotFoundError: If the bank exists in neither place.
    """
    import isaaclab_tasks

    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    # isaaclab_tasks/__init__.py -> source/isaaclab_tasks -> source -> repository root
    in_repo = Path(isaaclab_tasks.__file__).resolve().parents[3] / path
    if in_repo.exists():
        return in_repo
    raise FileNotFoundError(f"Reset bank '{path}' not found in the working directory or under '{in_repo.parent}'.")

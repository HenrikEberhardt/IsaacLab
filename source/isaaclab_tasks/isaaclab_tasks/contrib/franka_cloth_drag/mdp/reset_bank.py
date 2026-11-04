# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset event that continues stored reset states of the Franka cloth dragging task."""

from __future__ import annotations

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
from .stages import ClothDragStage

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ResetFromBank(ManagerTermBase):
    """Reset environments to stored reset states of a :class:`~..reset_states.ResetBank`.

    Every reset draws a mode per environment by ``mode_probabilities`` (:data:`~..reset_states.bank.RESET_MODES`): the
    task's default reset, or a uniformly drawn stored state of the ``grasp``, ``drag`` or ``goal`` bucket. Rows given
    with :meth:`force_rows` take precedence for the next reset. Without a bank, every reset is the default one.

    With ``min_probabilities``, the shares adapt to the success of the buckets: ``mode_probabilities`` are the start
    shares, and a solved bucket gives up its share to the stage before it, down to its minimum (see
    :func:`~..reset_states.bank.cascade_shares`). The success of a source is a moving average over its finished
    episodes that spans about ``ema_episodes`` of them, logged as ``Reset/success_ema_<mode>``.

    The term must come after the other reset events, so the default resets run first and a stored state overwrites
    them. It restores the physical state when called, and the task state that the action and command resets clear in
    its :meth:`reset`, which the event manager runs after those resets (see :mod:`~..reset_states.restore`).

    Every reset after the first also logs, in ``extras["log"]``, per source the episodes that just ended started from
    (``default`` or the bucket name):

    * ``Reset/success_<source>``: their success rate.
    * ``Reset/next_stage_<source>``: the share that reached the stage after their start stage (GRASP for default
      resets), for every source but ``goal``.
    * ``Reset/time_to_next_<source>``: the episode time [s] until that stage, over the episodes that reached it; for
      ``goal``, until the success.

    A value without an episode to average is left out. The reset also logs the share of each reset mode among the
    episodes that start now (``Reset/share_<mode>``), with or without a bank.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        """Load the bank, if any. It is checked against the task at the first reset, once all managers exist."""
        super().__init__(cfg, env)
        self.bank: ResetBank | None = None
        """The stored states, on the environment's device; None for default resets only."""
        bank_path = cfg.params.get("bank_path")
        if bank_path is not None:
            self.bank = ResetBank.load(_resolve_bank_path(bank_path), env.device)
        self._fingerprint_checked = bank_path is None
        self._mode_probabilities = cfg.params.get("mode_probabilities") or {"default": 1.0}
        self._shares = check_mode_probabilities(self._mode_probabilities, self.bank)
        # adaptive shares: start shares, minimum shares (None for fixed shares) and success average of the modes
        self._start_shares = self._shares
        self._min_shares: torch.Tensor | None = None
        self._success_ema = torch.zeros(len(RESET_MODES))
        self._success_range = cfg.params.get("success_range", (0.85, 0.95))
        self._ema_episodes = cfg.params.get("ema_episodes", 256)
        min_probabilities = cfg.params.get("min_probabilities")
        if min_probabilities is not None:
            self._min_shares = check_min_probabilities(
                min_probabilities, self._start_shares, self._success_range, self._ema_episodes
            )
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
        ema_episodes: int = 256,
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
            command_name: Name of the cloth target command, whose ``success`` metric still holds the outcome of the
                finished episodes (the command reset clears it later in the same reset).
            min_probabilities: Minimum share of the resets per mode. Enables the adaptive shares, with
                ``mode_probabilities`` as the start shares. Defaults to fixed shares.
            success_range: Success average (low, high) between which a bucket gives up its share.
            ema_episodes: Number of finished episodes the success average of a source effectively spans.
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
        """Success and progress of the finished episodes per source, and share of each reset mode among the started
        ones.

        Args:
            env: The environment.
            env_ids: The environments being reset, shape [N].
            finished: Bank rows the finished episodes started from, -1 for a default reset, shape [N].
            started: Bank rows the new episodes start from, -1 for a default reset, shape [N].
        """
        names = ("default", *BUCKET_NAMES)
        command = env.command_manager.get_term(self._command_name)
        success = command.metrics["success"][env_ids].float()
        finished_source, started_source = self._source(finished), self._source(started)
        # the buckets are named after their stage, so a bucket's index is its start stage; default resets start in
        # APPROACH. Column k of reached and time is the stage after stage k, the success after GOAL.
        start_stage = (finished_source - 1).clamp(min=0).unsqueeze(1)
        reached = [command.metrics[f"reached_{stage.name.lower()}"][env_ids] for stage in list(ClothDragStage)[1:]]
        reached = torch.stack([*reached, success], dim=1).gather(1, start_stage).squeeze(1).float()
        # the success terminates the episode, whose length the environment clears after the reset
        episode_time = env.episode_length_buf[env_ids].float() * env.step_dt
        time = torch.cat([command.time_to_stage[env_ids, 1:], episode_time.unsqueeze(1)], dim=1)
        time = time.gather(1, start_stage).squeeze(1) * reached
        counts = torch.bincount(finished_source, minlength=len(names))
        sums = torch.bincount(finished_source, weights=success, minlength=len(names))
        reached_counts = torch.bincount(finished_source, weights=reached, minlength=len(names))
        time_sums = torch.bincount(finished_source, weights=time, minlength=len(names))
        shares = torch.bincount(started_source, minlength=len(names)) / max(len(started), 1)
        # one transfer to the host for all values
        values = torch.stack([counts.float(), sums, reached_counts, time_sums, shares]).cpu().tolist()
        counts, sums, reached_counts, time_sums, shares = values
        # a source without an episode to average is left out: a placeholder would distort the logger's mean
        log = {f"Reset/success_{name}": total / num for name, num, total in zip(names, counts, sums) if num > 0}
        log |= {
            f"Reset/next_stage_{name}": total / num
            for name, num, total in zip(names, counts, reached_counts)
            if num > 0 and name != "goal"
        }
        log |= {
            f"Reset/time_to_next_{name}": total / num
            for name, num, total in zip(names, reached_counts, time_sums)
            if num > 0
        }
        log |= {f"Reset/share_{name}": share for name, share in zip(names, shares) if name in RESET_MODES}
        if self._min_shares is not None:
            log |= self._adapt_shares(dict(zip(names, counts)), dict(zip(names, sums)))
        return log

    def _adapt_shares(self, counts: dict[str, float], successes: dict[str, float]) -> dict[str, float]:
        """Fold the finished episodes into the success average of every mode and set the shares of the next resets.

        Args:
            counts: Number of finished episodes per source.
            successes: Number of successful ones per source.

        Returns:
            The success averages to log.
        """
        for mode_id, mode in enumerate(RESET_MODES):
            if counts[mode] > 0:
                alpha = counts[mode] / (counts[mode] + self._ema_episodes)
                self._success_ema[mode_id] += alpha * (successes[mode] / counts[mode] - self._success_ema[mode_id])
        self._shares = cascade_shares(self._start_shares, self._min_shares, self._success_ema, self._success_range)
        return {f"Reset/success_ema_{mode}": value for mode, value in zip(RESET_MODES, self._success_ema.tolist())}

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

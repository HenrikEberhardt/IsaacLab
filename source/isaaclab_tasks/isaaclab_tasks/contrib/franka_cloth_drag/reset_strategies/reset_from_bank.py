# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset event that starts the episodes of the Franka cloth dragging task from stored reset states."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase

from ..mdp.stages import START_STAGE_NAMES
from .bank import ResetBank, save_atomic, task_fingerprint
from .frontier import ResetStateFrontierSampler
from .restore import restore_physical, restore_task_state
from .stage_ratios import cascade_shares, check_min_stage_ratios, check_stage_ratios, sample_reset_rows

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from ..mdp.events import ClothRaisedNodes
    from .reset_from_bank_cfg import ResetFromBankCfg


class ResetFromBank(ManagerTermBase):
    """Start the episodes from stored reset states of a :class:`~.bank.ResetBank`.

    Every reset draws a stage per environment by ``start_stage_ratios`` (:data:`~..mdp.stages.START_STAGE_NAMES`) and a
    stored state that starts in that stage: a state of the ``approach``, ``grasp``, ``drag`` or ``goal`` stage. The
    term is the only reset event of its strategy: it writes the whole state, so no
    default reset event runs before it.

    With ``min_stage_ratios``, the ratios adapt to the success of the stages: ``start_stage_ratios`` are the ratios at
    the start of training, and a solved stage hands its ratio to the stage before it, down to its minimum (see
    :func:`~.stage_ratios.cascade_shares`). The success of a stage is the smoothed task success of the episodes that
    started in it, which the cloth target command logs (:class:`~..mdp.commands.StartStageSuccess`). The ratios keep
    their start values for the first ``ratio_hold_steps`` environment steps.

    With ``use_frontier_sampling``, the states of a stage are not drawn uniformly but by an SGS beta kernel over a
    linearly weighted history of their recent next-stage outcomes.

    The term restores the physical state when called, and the task state that the action and command resets clear in
    its :meth:`reset`, which the event manager runs after those resets (see :mod:`~.restore`). The stored raised nodes
    are restored through the cloth reset event if the strategy has one, else through the term's own
    :class:`~..mdp.events.ClothRaisedNodes`, which then also serves :func:`~..mdp.frames.raised_node_ids`.

    For the bank generation, the term runs after the default reset events without ``start_stage_ratios``: it then
    restores only the rows given with :meth:`force_rows` and leaves the other environments as the default resets made
    them (row -1, source ``default``).

    The term logs the share of each stage among the episodes that start now (``Reset/share_<stage>_raw``) and how often
    the stored states of every stage were visited, and saves snapshots of the visit counts and frontier data under the
    run's ``reset_state_success`` directory every ``state_success_export_interval_steps``.
    """

    cfg: ResetFromBankCfg

    def __init__(self, cfg: ResetFromBankCfg, env: ManagerBasedRLEnv):
        """Load the bank, if any. It is checked against the task at the first reset, once all managers exist."""
        super().__init__(cfg, env)
        self.bank: ResetBank | None = None
        """The stored states, on the environment's device; None until :meth:`set_bank` without a ``bank_path``."""
        if cfg.bank_path is not None:
            self.bank = ResetBank.load(_resolve_bank_path(cfg.bank_path), env.device)
        if not isinstance(cfg.use_frontier_sampling, bool):
            raise TypeError("use_frontier_sampling must be a bool.")
        cfg.frontier_sampling_parameters.validate_values()
        if cfg.use_frontier_sampling and self.bank is None:
            raise ValueError("use_frontier_sampling requires a reset-state bank.")
        self._export_interval_steps = int(cfg.state_success_export_interval_steps)
        if self._export_interval_steps < 0:
            raise ValueError(
                f"state_success_export_interval_steps must be non-negative, got {self._export_interval_steps}."
            )
        if cfg.ratio_hold_steps < 0:
            raise ValueError(f"ratio_hold_steps must be non-negative, got {cfg.ratio_hold_steps}.")
        log_dir = getattr(getattr(env, "cfg", None), "log_dir", None)
        self._state_stats_dir = (
            Path(log_dir) / "reset_state_success"
            if self._export_interval_steps > 0 and log_dir is not None and int(os.getenv("RANK", "0")) == 0
            else None
        )
        self._next_export_step = self._export_interval_steps
        self._reset_state_statistics()
        self._fingerprint_checked = cfg.bank_path is None
        # ratios of the stages among the resets; None draws nothing (only forced rows are restored)
        self._ratios = None if cfg.start_stage_ratios is None else check_stage_ratios(cfg.start_stage_ratios, self.bank)
        # adaptive ratios: start ratios, minimum ratios (None for fixed ratios) and smoothed task success of the stages
        self._start_ratios = self._ratios
        self._min_ratios: torch.Tensor | None = None
        self._task_success = torch.zeros(len(START_STAGE_NAMES))
        if cfg.min_stage_ratios is not None:
            if self._start_ratios is None:
                raise ValueError("min_stage_ratios needs start_stage_ratios as the start ratios.")
            self._min_ratios = check_min_stage_ratios(
                cfg.min_stage_ratios, self._start_ratios, cfg.ratio_shift_success_range
            )
        self.rows = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        """Bank row the current episode of every environment started from, -1 without a stored state."""
        self._forced_rows = torch.full_like(self.rows, -1)
        self._forced = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        # environments whose task state the reset hook still has to restore
        self._pending = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        # the cloth rest shape around the stored raised nodes, built at its first use
        self._raised_nodes: ClothRaisedNodes | None = None
        # log of the current reset, written by the reset hook once the environment has recreated extras["log"]
        self._log: dict[str, float] = {}

    @property
    def raised_node_ids(self) -> torch.Tensor | None:
        """Nodes raised in the current episode of every environment, sorted, shape [num_envs, K]; None without a
        bank. For a strategy without the cloth reset event, see :func:`~..mdp.frames.raised_node_ids`."""
        raised_nodes = self._own_raised_nodes()
        return None if raised_nodes is None else raised_nodes.raised_node_ids

    def set_raised_nodes(self, env_ids: torch.Tensor, raised: torch.Tensor) -> None:
        """Set the cloth rest shape of the given environments around the given raised nodes, at the height the bank
        was made with (see :meth:`~..mdp.events.ClothRaisedNodes.set_raised_nodes`).

        Args:
            env_ids: The environment indices, shape [N].
            raised: Ids of the nodes to raise, sorted, shape [N, K].
        """
        self._own_raised_nodes().set_raised_nodes(env_ids, raised)

    def set_bank(self, bank: ResetBank | None) -> None:
        """Use another bank, captured in this environment (e.g. candidates during generation), so it is not checked
        against the task; the stage ratios keep applying to it."""
        self.bank = None if bank is None else bank.to(self.device)
        if self.cfg.start_stage_ratios is not None:
            self._ratios = check_stage_ratios(self.cfg.start_stage_ratios, self.bank)
        self._reset_state_statistics()
        self._fingerprint_checked = True

    def force_rows(self, env_ids: torch.Tensor, rows: torch.Tensor) -> None:
        """Reset the given environments to the given bank rows at their next reset, regardless of the stage ratios.

        Args:
            env_ids: The environments, shape [N].
            rows: Rows of the bank, shape [N]; -1 restores nothing.
        """
        if self.bank is None and (rows >= 0).any():
            raise RuntimeError("Rows can only be forced with a bank, see set_bank.")
        self._forced_rows[env_ids] = rows.to(self.device)
        self._forced[env_ids] = True

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

    def __call__(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> None:
        """Take the outcome of the finished episodes, draw the stored state of every environment and restore its
        physical state.

        Args:
            env: The environment.
            env_ids: The environments to reset.
        """
        env_ids = self._resolve(env_ids)
        if not self._fingerprint_checked:
            # the event terms are built before the action and command managers that the fingerprint reads
            self.bank.check_fingerprint(task_fingerprint(env), self.cfg.allow_settings_mismatch)
            self._fingerprint_checked = True
        command = env.command_manager.get_term(self.cfg.command_name)
        finished = self.rows[env_ids]
        # the reset at the start of training ends no episode; the command reset clears the outcome later in this reset
        ended = env.common_step_counter > 0
        if ended:
            success, reached = command.episode_outcomes(env_ids)
            if self._frontier is not None:
                stored = finished >= 0
                self._frontier.record(finished[stored], reached[stored])
            self._count_visits(finished)
            self._maybe_export_state_statistics(env.common_step_counter)
            if self._min_ratios is not None:
                self._adapt_ratios(env.common_step_counter, command.start_stage_success.window_totals)
        drawn = ~self._forced[env_ids]
        rows = torch.where(drawn, -1, self._forced_rows[env_ids])
        self._forced[env_ids] = False
        if self._ratios is not None and drawn.any():
            if self._frontier is None:
                rows[drawn] = sample_reset_rows(self.bank, self._ratios, int(drawn.sum()))
            else:
                rows[drawn] = sample_reset_rows(
                    self.bank, self._ratios, int(drawn.sum()), self._frontier.sampling_weights()
                )
        self.rows[env_ids] = rows
        stored = rows >= 0
        self._pending[env_ids] = stored
        if stored.any():
            restore_physical(env, self.bank, rows[stored], env_ids[stored])
        self._log = self._share_log(rows) | self._state_log() if ended else {}

    def _share_log(self, rows: torch.Tensor) -> dict[str, float]:
        """Share of each stage among the episodes that start from the given bank rows."""
        num = len(START_STAGE_NAMES)
        shares = (torch.bincount(self._source(rows), minlength=num) / max(len(rows), 1)).tolist()
        return {f"Reset/share_{name}_raw": share for name, share in zip(START_STAGE_NAMES, shares)}

    def _state_log(self) -> dict[str, float]:
        """Visit fractions and, with frontier sampling, frontier diagnostics of the stored states of every stage."""
        if self.bank is None:
            return {}
        keys: list[str] = []
        values: list[torch.Tensor] = []
        for stage, rows in self._stage_rows.items():
            if rows.numel() == 0:
                continue
            counts = self.state_visit_count[rows]
            visit_masks = (counts == 0, counts == 1, counts == 2, counts == 3, counts >= 4)
            for label, mask in zip(("0", "1", "2", "3", "4_plus"), visit_masks):
                keys.append(f"Reset/state_visit_fraction_{label}_{stage}")
                values.append(mask.float().mean())
            if self._frontier is not None:
                for name, value in self._frontier.metrics(rows).items():
                    keys.append(f"Reset/{name}_{stage}")
                    values.append(value)
        return dict(zip(keys, torch.stack(values).cpu().tolist())) if values else {}

    def _reset_state_statistics(self) -> None:
        """Clear the visit counts and the frontier histories of the stored states."""
        num_rows = 0 if self.bank is None else len(self.bank)
        self.state_visit_count: torch.Tensor = torch.zeros(num_rows, dtype=torch.long, device=self.device)
        """Number of finished episodes that started from every stored state, shape [N]."""
        self._stage_rows: dict[str, torch.Tensor] = (
            {}
            if self.bank is None
            else {stage: self.bank.stage_rows(stage_id) for stage_id, stage in enumerate(START_STAGE_NAMES)}
        )
        self._frontier = (
            ResetStateFrontierSampler(num_rows, self.device, self.cfg.frontier_sampling_parameters)
            if self.cfg.use_frontier_sampling and num_rows > 0
            else None
        )

    def _count_visits(self, finished: torch.Tensor) -> None:
        """Count the finished episodes per stored state they started from.

        Args:
            finished: Bank rows the episodes started from, -1 without a stored state, shape [N].
        """
        stored = finished >= 0
        if self.bank is not None and stored.any():
            self.state_visit_count += torch.bincount(finished[stored], minlength=len(self.bank))

    def _maybe_export_state_statistics(self, env_step: int) -> None:
        """Save a snapshot of the visit counts and frontier data once the configured interval has elapsed."""
        if self._state_stats_dir is None or self.bank is None or env_step < self._next_export_step:
            return
        self._next_export_step = (env_step // self._export_interval_steps + 1) * self._export_interval_steps
        payload = {
            "schema_version": 3,
            "env_step": int(env_step),
            "bank_fingerprint": self.bank.meta.get("fingerprint", {}),
            "stage": self.bank.stage.detach().cpu(),
            "visit_count": self.state_visit_count.detach().cpu(),
            "frontier": self._frontier_snapshot(),
        }
        self._state_stats_dir.mkdir(parents=True, exist_ok=True)
        save_atomic(self._state_stats_dir / f"step_{env_step:08d}.pt", payload)
        save_atomic(self._state_stats_dir / "latest.pt", payload)

    def _frontier_snapshot(self) -> dict | None:
        """Serializable frontier histories, estimates and per-stage sampling probabilities."""
        if self._frontier is None:
            return None
        sampling_weight = self._frontier.sampling_weights()
        sampling_probability = torch.zeros_like(sampling_weight)
        for rows in self._stage_rows.values():
            if rows.numel() > 0:
                sampling_probability[rows] = self._frontier.probabilities(rows)
        return {
            "cfg": self.cfg.frontier_sampling_parameters.to_dict(),
            "history": self._frontier.history.detach().cpu(),
            "history_pointer": self._frontier.history_pointer.detach().cpu(),
            "history_size": self._frontier.history_size.detach().cpu(),
            "success_estimate": self._frontier.success_rate.detach().cpu(),
            "sampling_weight": sampling_weight.detach().cpu(),
            "sampling_probability": sampling_probability.detach().cpu(),
        }

    def _adapt_ratios(self, step: int, window_totals: torch.Tensor) -> None:
        """Set the ratios of the resets from the smoothed task success of every stage, once the hold has ended.

        A stage without an episode in the window keeps its last value, so a stage that handed over all its ratio stays
        solved.

        Args:
            step: The environment step.
            window_totals: Number of episodes and task successes per start stage over the command's window (rows 0 and
                1), up to the last reset, shape [3, 4].
        """
        counts, successes = window_totals[0].tolist(), window_totals[1].tolist()
        for stage_id in range(len(START_STAGE_NAMES)):
            if counts[stage_id] > 0:
                self._task_success[stage_id] = successes[stage_id] / counts[stage_id]
        if step >= self.cfg.ratio_hold_steps:
            self._ratios = cascade_shares(
                self._start_ratios, self._min_ratios, self._task_success, self.cfg.ratio_shift_success_range
            )

    def _source(self, rows: torch.Tensor) -> torch.Tensor:
        """Stage each bank row starts in, an index into :data:`~..mdp.stages.START_STAGE_NAMES`; 0 for -1."""
        source = torch.zeros_like(rows)
        stored = rows >= 0
        if stored.any():
            source[stored] = self.bank.stage[rows[stored]].to(rows.dtype)
        return source

    def _own_raised_nodes(self) -> ClothRaisedNodes | None:
        """The cloth rest shape around the stored raised nodes, built from the bank at the first use."""
        if self._raised_nodes is None and self.bank is not None:
            from ..mdp.events import ClothRaisedNodes

            num_raised_nodes = self.bank.data["task"]["raised_node_ids"].shape[1]
            self._raised_nodes = ClothRaisedNodes(self._env, num_raised_nodes, self.bank.meta["node_raise_height"])
        return self._raised_nodes

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

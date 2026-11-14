# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of the reset event term of the uniform and adaptive reset strategies."""

from __future__ import annotations

from collections.abc import Callable

from isaaclab.managers import EventTermCfg
from isaaclab.utils import configclass

from .frontier import FrontierSamplerCfg


@configclass
class ResetFromBankCfg(EventTermCfg):
    """Configuration of :class:`~.reset_from_bank.ResetFromBank`: starts the episodes from stored reset states.

    With :attr:`start_stage_ratios` only, the ratios are fixed (uniform resets). With :attr:`min_stage_ratios`, they
    follow the success of the stages (adaptive resets).
    """

    func: Callable | str = "{DIR}.reset_from_bank:ResetFromBank"

    mode: str = "reset"

    command_name: str = "cloth_target"
    """Name of the cloth target command, which holds the outcome of the finished episodes and their success per start
    stage (:attr:`~..mdp.commands.ClothTargetCommand.start_stage_success`)."""

    bank_path: str | None = None
    """Path of the bank (``.pt``). A relative path is looked up in the working directory, then in the repository root.
    None loads no bank: the term then only restores rows forced on it (bank generation)."""

    start_stage_ratios: dict[str, float] | None = None
    """Ratio of the resets per stage the episodes start in, keyed by ``approach``, ``grasp``, ``drag`` and ``goal``;
    normalized. ``default`` is accepted as a legacy alias for ``approach``. Missing stages get none. The ratios stay
    fixed (uniform resets), or, with :attr:`min_stage_ratios`, are the ratios at the start of training (adaptive
    resets). None draws nothing."""

    allow_settings_mismatch: bool = False
    """Whether a bank made with other physics or task settings may be loaded."""

    use_frontier_sampling: bool = False
    """Whether the states of a stage are drawn by their recent next-stage success instead of uniformly."""

    frontier_sampling_parameters: FrontierSamplerCfg = FrontierSamplerCfg()
    """Settings of the frontier sampling, used with :attr:`use_frontier_sampling`."""

    frontier_stages: tuple[str, ...] = ("grasp", "drag", "goal")
    """Stages whose states are drawn by the frontier sampling, used with :attr:`use_frontier_sampling`; the states of
    the other stages are drawn uniformly. The approach stage is left out by default: its bank is large, so each state
    is visited too rarely for a success estimate, and uniform draws keep its starts diverse."""

    state_success_export_interval_steps: int = 120
    """Number of environment steps between the snapshots of the per-state visit counts and frontier data under the
    run's ``reset_state_success`` directory (5 iterations at 24 steps/iteration). Zero disables the snapshots."""

    min_stage_ratios: dict[str, float] | None = None
    """Smallest ratio of the resets per stage, keyed like :attr:`start_stage_ratios`. Enables the adaptive ratios, with
    :attr:`start_stage_ratios` as the start ratios: a solved grasp, drag or goal stage hands its ratio over to the
    approach stage, see :func:`~.stage_ratios.handover_stage_ratios`. The approach stage never gives up ratio, so its
    entry is not used. None keeps the ratios fixed."""

    ratio_shift_success_range: tuple[float, float] = (0.80, 0.92)
    """Success (low, high) of a stage between which it hands over its ratio: all of it is kept below ``low``, only the
    minimum from ``high``. The success of a stage in :attr:`frontier_stages` with frontier sampling is the mean of its
    states' frontier success estimates (every state weighted equally, unvisited states count as failed); otherwise it
    is the smoothed share of the episodes started in the stage that reach the next one (the task success for the goal
    stage). Used with :attr:`min_stage_ratios`."""

    ratio_max_shift_per_step: float = 0.0008
    """Largest change of any stage's ratio per environment step (0.02 per iteration at 24 steps/iteration), so a
    handover is spread over more than an episode's length. Used with :attr:`min_stage_ratios`."""

    ratio_hold_steps: int = 0
    """Number of environment steps the adaptive ratios keep their start values. Used with :attr:`min_stage_ratios`."""

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ratios of the stages among the resets of the Franka cloth dragging task.

Uniform resets draw with fixed ratios (:func:`sample_reset_rows`); adaptive resets hand the ratios of solved stages over
to the approach stage (:func:`handover_stage_ratios`), at a limited rate (:func:`limit_ratio_shift`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from ..mdp.stages import START_STAGE_NAMES, canonical_start_stage_name

if TYPE_CHECKING:
    from .bank import ResetBank


def check_stage_ratios(stage_ratios: dict[str, float], bank: ResetBank | None) -> torch.Tensor:
    """Check the ratios of the stages and return them in the order of :data:`~..mdp.stages.START_STAGE_NAMES`,
    normalized.

    Args:
        stage_ratios: Ratio of the resets per stage name (:data:`~..mdp.stages.START_STAGE_NAMES`); missing stages get
            none.
        bank: The bank the states are drawn from, or None without one.

    Raises:
        ValueError: On an unknown stage, negative or all-zero ratios, or a stage without a bank or stored states.
    """
    stage_ratios = _canonical_stage_dict(stage_ratios)
    unknown = set(stage_ratios) - set(START_STAGE_NAMES)
    if unknown:
        raise ValueError(f"Unknown stages {sorted(unknown)}, expected names from {START_STAGE_NAMES}.")
    ratios = torch.tensor([float(stage_ratios.get(stage, 0.0)) for stage in START_STAGE_NAMES])
    if not torch.isfinite(ratios).all() or (ratios < 0.0).any() or ratios.sum() <= 0.0:
        raise ValueError(f"The stage ratios must be finite, non-negative and not all zero: {stage_ratios}.")
    for stage_id, (stage, ratio) in enumerate(zip(START_STAGE_NAMES, ratios)):
        if ratio > 0.0 and (bank is None or len(bank.stage_rows(stage_id)) == 0):
            raise ValueError(f"Stage '{stage}' has a ratio of {ratio.item()} but no stored states.")
    return ratios / ratios.sum()


def check_min_stage_ratios(
    min_stage_ratios: dict[str, float],
    start_ratios: torch.Tensor,
    success_range: Sequence[float],
) -> torch.Tensor:
    """Check the settings of the adaptive ratios and return the minimum ratios in the order of
    :data:`~..mdp.stages.START_STAGE_NAMES`.

    Args:
        min_stage_ratios: Minimum ratio of the resets per stage name; missing stages get none.
        start_ratios: Start ratios of the stages, from :func:`check_stage_ratios`, shape [M].
        success_range: Success (low, high) between which a stage hands over its ratio.

    Raises:
        ValueError: On an unknown stage, a minimum outside 0 and its start ratio, or a ``success_range`` that is not
            increasing inside [0, 1].
    """
    min_stage_ratios = _canonical_stage_dict(min_stage_ratios)
    unknown = set(min_stage_ratios) - set(START_STAGE_NAMES)
    if unknown:
        raise ValueError(f"Unknown stages {sorted(unknown)}, expected names from {START_STAGE_NAMES}.")
    minimum = torch.tensor([float(min_stage_ratios.get(stage, 0.0)) for stage in START_STAGE_NAMES])
    if (minimum < 0.0).any() or (minimum > start_ratios + 1e-6).any():
        raise ValueError(f"Every minimum must lie between 0 and its start ratio {start_ratios.tolist()}: {minimum}.")
    low, high = success_range
    if not 0.0 <= low < high <= 1.0:
        raise ValueError(f"success_range must increase inside [0, 1], got {success_range}.")
    return minimum


def handover_stage_ratios(
    start: torch.Tensor, minimum: torch.Tensor, success: torch.Tensor, success_range: Sequence[float]
) -> torch.Tensor:
    """Hand the ratios of solved bank stages over to the approach stage.

    The grasp, drag and goal stages keep their start ratio up to the low end of ``success_range``, only their minimum
    from the high end, and linearly between. The approach stage, the start of the task and the only source of real
    arrivals at the cloth, takes the rest: it never gives up any ratio, so it never falls below its start ratio. With
    every stage unsolved the ratios are the start ratios.

    Args:
        start: Start ratios in the order of :data:`~..mdp.stages.START_STAGE_NAMES`, shape [M].
        minimum: Minimum ratios in the same order, shape [M]. The approach stage's is not used.
        success: Success of each stage in [0, 1] in the same order, shape [M]. The approach stage's is not used.
        success_range: Success (low, high) between which a stage hands over its ratio.

    Returns:
        The ratios in the same order, shape [M].
    """
    low, high = success_range
    deficit = ((high - success) / (high - low)).clamp(0.0, 1.0)
    ratios = minimum + (start - minimum) * deficit
    ratios[0] = 1.0 - ratios[1:].sum()
    return ratios


def focus_stage_ratios(focus_id: int, minimum: torch.Tensor) -> torch.Tensor:
    """Put every stage at its minimum and give the rest to the focus stage (reverse curriculum).

    Args:
        focus_id: Index of the focus stage in :data:`~..mdp.stages.START_STAGE_NAMES`.
        minimum: Minimum ratios in the same order, shape [M]; the focus stage's is not used.

    Returns:
        The ratios in the same order, shape [M].
    """
    ratios = minimum.clone()
    ratios[focus_id] = 0.0
    ratios[focus_id] = 1.0 - ratios.sum()
    return ratios


def learning_progress_ratios(progress: torch.Tensor, minimum: torch.Tensor, eps: float = 0.01) -> torch.Tensor:
    """Put every stage at its minimum and split the rest in proportion to the stages' learning progress.

    Args:
        progress: Absolute change of each stage's success over the learning-progress window, shape [M].
        minimum: Minimum ratios in the same order, shape [M].
        eps: Progress added to every stage, so no progress anywhere splits the rest evenly.

    Returns:
        The ratios in the same order, shape [M].
    """
    weights = progress.clamp(min=0.0) + eps
    return minimum + (1.0 - minimum.sum()) * weights / weights.sum()


def limit_ratio_shift(current: torch.Tensor, target: torch.Tensor, max_shift: float) -> torch.Tensor:
    """Move the ratios towards the target by at most ``max_shift`` in any stage.

    The whole difference is scaled, so the ratios keep their sum.

    Args:
        current: Current ratios, shape [M].
        target: Target ratios, shape [M].
        max_shift: Largest change of any stage's ratio, non-negative.

    Returns:
        The new ratios, shape [M].
    """
    difference = target - current
    largest = difference.abs().max()
    if largest <= max_shift:
        return target.clone()
    return current + difference * (max_shift / largest)


def sample_reset_rows(
    bank: ResetBank, ratios: torch.Tensor, num: int, row_weights: torch.Tensor | None = None
) -> torch.Tensor:
    """Draw the reset of ``num`` environments: a stage by its ratio, then a stored state of that stage.

    Args:
        bank: The bank to draw from.
        ratios: Ratios of the stages in the order of :data:`~..mdp.stages.START_STAGE_NAMES`, from
            :func:`check_stage_ratios`.
        num: Number of environments.
        row_weights: Optional non-negative weight of every bank row, shape [N]. Weights are normalized independently
            inside each selected stage. Without them, rows are drawn uniformly.

    Returns:
        Bank rows, shape [num], on the bank's device.
    """
    device = bank.device
    if row_weights is not None:
        if row_weights.shape != (len(bank),):
            raise ValueError(f"row_weights must have shape ({len(bank)},), got {tuple(row_weights.shape)}.")
        if not torch.isfinite(row_weights).all() or (row_weights < 0.0).any():
            raise ValueError("row_weights must be finite and non-negative.")
        row_weights = row_weights.to(device)
    stages = torch.multinomial(ratios.to(device), num, replacement=True)
    rows = torch.full((num,), -1, dtype=torch.long, device=device)
    for stage_id, stage in enumerate(START_STAGE_NAMES):
        selected = stages == stage_id
        count = int(selected.sum())
        if count > 0:
            stage_rows = bank.stage_rows(stage_id)
            if row_weights is None:
                positions = torch.randint(len(stage_rows), (count,), device=device)
            else:
                weights = row_weights[stage_rows]
                if weights.sum() <= 0.0:
                    raise ValueError(f"row_weights sum to zero inside the '{stage}' stage.")
                positions = torch.multinomial(weights, count, replacement=True)
            rows[selected] = stage_rows[positions]
    return rows


def _canonical_stage_dict(values: dict[str, float]) -> dict[str, float]:
    """Return stage values with legacy names replaced by canonical names."""
    normalized: dict[str, float] = {}
    for name, value in values.items():
        canonical = canonical_start_stage_name(name)
        if canonical in normalized:
            raise ValueError(f"Stage '{canonical}' was provided more than once through canonical or legacy names.")
        normalized[canonical] = value
    return normalized

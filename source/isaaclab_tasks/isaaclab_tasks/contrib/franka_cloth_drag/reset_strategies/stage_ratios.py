# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ratios of the stages among the resets of the Franka cloth dragging task.

Uniform resets draw with fixed ratios (:func:`sample_reset_rows`); adaptive resets move the ratios of solved stages
to the stage before them (:func:`cascade_shares`).
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


def cascade_shares(
    start: torch.Tensor, minimum: torch.Tensor, success: torch.Tensor, success_range: Sequence[float]
) -> torch.Tensor:
    """Move the ratios of solved stages to the stage before them.

    Resets of an earlier stage run through the later ones, so a stage hands its ratio to the stage before it in
    :data:`~..mdp.stages.START_STAGE_NAMES`: goal to drag, drag to grasp, grasp to approach. A stage keeps all it holds,
    its start ratio and what it received, up to the low end of ``success_range``, only its minimum from the high end,
    and linearly between. The approach stage hands over nothing, so it never falls below its start ratio.

    Args:
        start: Start ratios in the order of :data:`~..mdp.stages.START_STAGE_NAMES`, shape [M].
        minimum: Minimum ratios in the same order, shape [M].
        success: Success of each stage in [0, 1] in the same order, shape [M]. The approach stage's is not used.
        success_range: Success (low, high) between which a stage hands over its ratio.

    Returns:
        The ratios in the same order, shape [M].
    """
    low, high = success_range
    given_up = ((success - low) / (high - low)).clamp(0.0, 1.0)
    ratios = start.clone()
    for stage_id in range(len(ratios) - 1, 0, -1):
        released = (ratios[stage_id] - minimum[stage_id]) * given_up[stage_id]
        ratios[stage_id] -= released
        ratios[stage_id - 1] += released
    return ratios


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

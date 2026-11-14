# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success-guided frontier sampling over stored cloth-drag reset states."""

from __future__ import annotations

import math

import torch

from isaaclab.utils import configclass


@configclass
class FrontierSamplerCfg:
    """Configuration for per-reset-state success-guided sampling."""

    window_size: int = 4
    """Number of recent next-stage outcomes per reset state that its success estimate averages over."""

    weight_increment: float = 1.0
    """Linear weight increase from one outcome to the next, oldest to newest."""

    target_success_estimate: float = 0.5
    """Success estimate of a reset state at which the beta sampling kernel peaks: states near it are drawn most."""

    kappa: float = 1.0
    """Concentration of the beta sampling kernel around :attr:`target_success_estimate`."""

    temperature: float = 2.0
    """Temperature applied to the beta-kernel logits, at least one."""

    epsilon: float = 1.0e-4
    """Positive beta-weight floor that prevents reset-state starvation."""

    def __post_init__(self) -> None:
        """Validate configuration values."""
        self.validate_values()

    def validate_values(self) -> None:
        """Validate values after construction or runtime overrides."""
        if isinstance(self.window_size, bool) or not isinstance(self.window_size, int) or self.window_size < 1:
            raise ValueError("window_size must be a positive integer.")
        for name in ("weight_increment", "kappa", "temperature", "epsilon", "target_success_estimate"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")
        if self.weight_increment < 0.0:
            raise ValueError("weight_increment must be non-negative.")
        if not 0.0 < self.target_success_estimate < 1.0:
            raise ValueError("target_success_estimate must lie strictly between zero and one.")
        if self.kappa <= 0.0:
            raise ValueError("kappa must be positive.")
        if self.temperature < 1.0:
            raise ValueError("temperature must be at least one.")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")


class ResetStateFrontierSampler:
    """Track recent next-stage outcomes and produce beta-kernel reset-state weights."""

    def __init__(self, num_rows: int, device: str | torch.device, cfg: FrontierSamplerCfg):
        """Allocate the outcome histories.

        Args:
            num_rows: Number of reset states.
            device: Device on which histories and weights are stored.
            cfg: Frontier sampling configuration.
        """
        if isinstance(num_rows, bool) or not isinstance(num_rows, int) or num_rows < 1:
            raise ValueError("num_rows must be a positive integer.")
        cfg.validate_values()
        self.cfg = cfg
        self.history = torch.zeros((num_rows, cfg.window_size), dtype=torch.bool, device=device)
        """Circular next-stage outcome histories, shape [N, H]."""
        self.history_pointer = torch.zeros(num_rows, dtype=torch.long, device=device)
        """Next history column written for every reset state, shape [N]."""
        self.history_size = torch.zeros(num_rows, dtype=torch.long, device=device)
        """Number of valid outcomes in every history, shape [N]."""
        self.success_rate = torch.zeros(num_rows, device=device)
        """Linearly weighted next-stage success estimate, shape [N]."""
        self._linear_weights = 1.0 + cfg.weight_increment * torch.arange(
            cfg.window_size, dtype=torch.float32, device=device
        )

    def record(self, rows: torch.Tensor, success: torch.Tensor) -> None:
        """Append Boolean outcomes to their reset-state histories.

        Args:
            rows: Reset-state rows, shape [N].
            success: Whether each rollout reached its next stage, shape [N].
        """
        if rows.numel() == 0:
            return
        rows = rows.long()
        success = success.bool()
        order = torch.argsort(rows, stable=True)
        sorted_rows = rows[order]
        sorted_success = success[order]
        unique_rows, counts = torch.unique_consecutive(sorted_rows, return_counts=True)
        starts = counts.cumsum(0) - counts

        # A reset batch can contain the same row more than once. Preserve its stable environment order while
        # appending, including when more than one complete history is supplied in one batch.
        for offset in range(int(counts.max())):
            present = counts > offset
            update_rows = unique_rows[present]
            value_indices = starts[present] + offset
            columns = (self.history_pointer[update_rows] + offset) % self.cfg.window_size
            self.history[update_rows, columns] = sorted_success[value_indices]

        self.history_pointer[unique_rows] = (self.history_pointer[unique_rows] + counts) % self.cfg.window_size
        self.history_size[unique_rows] = (self.history_size[unique_rows] + counts).clamp(max=self.cfg.window_size)
        self._refresh_success_rates(unique_rows)

    def sampling_weights(self) -> torch.Tensor:
        """Return positive SGS beta-kernel weights for every reset state."""
        target = self.cfg.target_success_estimate
        rate = self.success_rate
        weights = (rate + self.cfg.epsilon).pow(self.cfg.kappa * target)
        weights *= (1.0 - rate + self.cfg.epsilon).pow(self.cfg.kappa * (1.0 - target))
        weights.clamp_min_(self.cfg.epsilon).add_(self.cfg.epsilon)
        return weights.pow_(1.0 / self.cfg.temperature)

    def metrics(self, rows: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return frontier success and probability-concentration metrics for one stage buffer."""
        success = self.success_rate[rows]
        weights = self.sampling_weights()[rows]
        probabilities = weights / weights.sum()
        top_count = max(1, math.ceil(0.01 * len(rows)))
        return {
            "frontier_success_mean": success.mean(),
            "frontier_success_median": success.median(),
            "frontier_hard_fraction": (success < 0.25).float().mean(),
            "frontier_medium_fraction": ((success >= 0.25) & (success <= 0.75)).float().mean(),
            "frontier_solved_fraction": (success > 0.75).float().mean(),
            "frontier_effective_pool_fraction": probabilities.square().sum().reciprocal() / len(rows),
            "frontier_top_1_percent_mass": torch.topk(probabilities, top_count, sorted=False).values.sum(),
        }

    def probabilities(self, rows: torch.Tensor) -> torch.Tensor:
        """Return normalized sampling probabilities for one stage buffer."""
        weights = self.sampling_weights()[rows]
        return weights / weights.sum()

    def _refresh_success_rates(self, rows: torch.Tensor) -> None:
        """Recompute the weighted estimates of the given rows in chronological history order."""
        size = self.history_size[rows]
        start = torch.where(size == self.cfg.window_size, self.history_pointer[rows], 0)
        offsets = torch.arange(self.cfg.window_size, device=rows.device)
        columns = (start.unsqueeze(1) + offsets.unsqueeze(0)) % self.cfg.window_size
        values = self.history[rows.unsqueeze(1), columns].float()
        valid = offsets.unsqueeze(0) < size.unsqueeze(1)
        weights = self._linear_weights.unsqueeze(0) * valid
        self.success_rate[rows] = (values * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

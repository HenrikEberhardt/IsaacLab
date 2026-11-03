# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Policy output distribution of the Franka cloth dragging task, for rsl_rl."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from rsl_rl.modules.distribution import GaussianDistribution
from torch.distributions import Normal


class SigmoidBoundedGaussianDistribution(GaussianDistribution):
    """Gaussian with a state-independent standard deviation smoothly bounded to ``std_range``.

    The standard deviation is ``min + (max - min) * sigmoid(raw)`` of one learnable parameter per action dimension.
    Unlike a hard parameter clamp, the sigmoid keeps a gradient near either bound.

    With ``mean_bound_coef > 0``, every backward pass through the mean also descends the bounds loss
    ``mean_bound_coef * mean_batch(sum_actions(relu(|mean| - mean_bound) ** 2))`` (rl_games' ``bounds_loss_coef``).
    A runner action clip hides the excess from the environment and its rewards, so without this loss nothing pulls a
    saturated mean back inside the clip. The gradient is added by a hook on the mean, which only exists in PPO's
    update; rollouts and exported policies are unaffected.
    """

    def __init__(
        self,
        output_dim: int,
        init_std: float = 0.5,
        std_range: tuple[float, float] = (0.05, 1.0),
        std_type: str = "scalar",
        learn_std: bool = True,
        mean_bound: float = 1.0,
        mean_bound_coef: float = 0.0,
    ) -> None:
        """Initialize the distribution.

        Args:
            output_dim: Dimension of the action space.
            init_std: Initial standard deviation, strictly inside ``std_range``.
            std_range: Bounds (min, max) of the standard deviation.
            std_type: Standard-deviation parameterization. Only ``"scalar"`` is supported.
            learn_std: Whether the standard deviation is learnable.
            mean_bound: Magnitude of the mean above which the bounds loss acts.
            mean_bound_coef: Weight of the bounds loss on the mean. Zero disables it.

        Raises:
            ValueError: If ``std_type`` is unsupported, ``init_std`` is outside ``std_range``, ``mean_bound`` is not
                positive or ``mean_bound_coef`` is negative.
        """
        if std_type != "scalar":
            raise ValueError(f"{type(self).__name__} supports std_type 'scalar' only, got '{std_type}'.")
        low, high = float(std_range[0]), float(std_range[1])
        if not 0.0 < low < init_std < high:
            raise ValueError(f"init_std {init_std} must lie strictly inside std_range ({low}, {high}).")
        if mean_bound <= 0.0 or mean_bound_coef < 0.0:
            raise ValueError(f"Need mean_bound > 0 and mean_bound_coef >= 0, got {mean_bound}, {mean_bound_coef}.")
        super().__init__(output_dim, learn_std=False)
        share = (init_std - low) / (high - low)
        self.std_range = [low, high]
        self.std_raw = nn.Parameter(torch.full((output_dim,), math.log(share / (1.0 - share))), learn_std)
        self.mean_bound = mean_bound
        self.mean_bound_coef = mean_bound_coef

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the Gaussian from the MLP output (the mean), shape [..., A]."""
        low, high = self.std_range
        std = low + (high - low) * torch.sigmoid(self.std_raw)
        self._distribution = Normal(mlp_output, std)
        if self.mean_bound_coef > 0.0 and mlp_output.requires_grad:
            mlp_output.register_hook(self._add_bounds_loss_grad(mlp_output.detach()))

    def bounds_loss(self, mean: torch.Tensor) -> torch.Tensor:
        """Bounds loss of a batch of means, shape [..., A], as weighted by ``mean_bound_coef``."""
        excess = (mean.abs() - self.mean_bound).clamp(min=0.0)
        return self.mean_bound_coef * excess.square().sum(dim=-1).mean()

    def _add_bounds_loss_grad(self, mean: torch.Tensor):
        """Gradient hook adding the bounds-loss gradient of ``mean`` to the incoming gradient."""
        num_samples = mean.numel() // mean.shape[-1]
        grad = 2.0 * self.mean_bound_coef / num_samples * (mean.abs() - self.mean_bound).clamp(min=0.0) * mean.sign()
        return lambda incoming: incoming + grad

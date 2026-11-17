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


def _init_raw(init_std: float, low: float, high: float, std_type: str) -> float:
    """Raw value of the sigmoid parameterization that yields ``init_std``. ``"log"`` interpolates in log space."""
    if std_type == "log":
        init_std, low, high = math.log(init_std), math.log(low), math.log(high)
    share = (init_std - low) / (high - low)
    return math.log(share / (1.0 - share))


def _std_from_raw(raw: torch.Tensor, low: float, high: float, std_type: str) -> torch.Tensor:
    """Standard deviation inside ``(low, high)`` from raw values: linear (``"scalar"``) or log-space (``"log"``)."""
    share = torch.sigmoid(raw)
    if std_type == "log":
        return torch.exp(math.log(low) + (math.log(high) - math.log(low)) * share)
    return low + (high - low) * share


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

    _STD_TYPES = ("scalar",)
    """Supported ``std_type`` values: ``"scalar"`` bounds the std linearly, ``"log"`` in log space."""

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
            std_type: Standard-deviation parameterization, ``"scalar"`` (linear sigmoid between the bounds) or, where
                supported, ``"log"`` (sigmoid between the log bounds).
            learn_std: Whether the standard deviation is learnable.
            mean_bound: Magnitude of the mean above which the bounds loss acts.
            mean_bound_coef: Weight of the bounds loss on the mean. Zero disables it.

        Raises:
            ValueError: If ``std_type`` is unsupported, ``init_std`` is outside ``std_range``, ``mean_bound`` is not
                positive or ``mean_bound_coef`` is negative.
        """
        if std_type not in self._STD_TYPES:
            raise ValueError(f"{type(self).__name__} supports std_type {self._STD_TYPES}, got '{std_type}'.")
        low, high = float(std_range[0]), float(std_range[1])
        if not 0.0 < low < init_std < high:
            raise ValueError(f"init_std {init_std} must lie strictly inside std_range ({low}, {high}).")
        if mean_bound <= 0.0 or mean_bound_coef < 0.0:
            raise ValueError(f"Need mean_bound > 0 and mean_bound_coef >= 0, got {mean_bound}, {mean_bound_coef}.")
        super().__init__(output_dim, learn_std=False)
        self.std_type = std_type
        self.std_range = [low, high]
        self.init_raw = _init_raw(init_std, low, high, std_type)
        self.std_raw = nn.Parameter(torch.full((output_dim,), self.init_raw), learn_std)
        self.mean_bound = mean_bound
        self.mean_bound_coef = mean_bound_coef
        self._kl_batch_means: list[torch.Tensor] = []

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the Gaussian from the MLP output (the mean), shape [..., A]."""
        self._set_distribution(mlp_output, _std_from_raw(self.std_raw, *self.std_range, self.std_type))

    def _set_distribution(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Build the Normal and attach the bounds-loss gradient hook to the mean."""
        self._distribution = Normal(mean, std)
        if self.mean_bound_coef > 0.0 and mean.requires_grad:
            mean.register_hook(self._add_bounds_loss_grad(mean.detach()))

    def bounds_loss(self, mean: torch.Tensor) -> torch.Tensor:
        """Bounds loss of a batch of means, shape [..., A], as weighted by ``mean_bound_coef``."""
        excess = (mean.abs() - self.mean_bound).clamp(min=0.0)
        return self.mean_bound_coef * excess.square().sum(dim=-1).mean()

    def kl_divergence(self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Compute KL(old || new) and retain its minibatch mean for training diagnostics."""
        kl = super().kl_divergence(old_params, new_params)
        self._kl_batch_means.append(kl.detach().mean())
        return kl

    def reset_kl_metrics(self):
        """Clear the retained minibatch KL means before a PPO update."""
        self._kl_batch_means.clear()

    def kl_batch_means(self) -> torch.Tensor:
        """Return the retained minibatch KL means, shape [num PPO minibatch updates]."""
        if not self._kl_batch_means:
            return torch.empty(0, device=self.std_raw.device)
        return torch.stack(self._kl_batch_means)

    def _add_bounds_loss_grad(self, mean: torch.Tensor):
        """Gradient hook adding the bounds-loss gradient of ``mean`` to the incoming gradient."""
        num_samples = mean.numel() // mean.shape[-1]
        grad = 2.0 * self.mean_bound_coef / num_samples * (mean.abs() - self.mean_bound).clamp(min=0.0) * mean.sign()
        return lambda incoming: incoming + grad


class SigmoidBoundedHeteroscedasticGaussianDistribution(SigmoidBoundedGaussianDistribution):
    """Gaussian whose standard deviation depends on the state and is smoothly bounded to ``std_range``.

    The standard deviation is the sigmoid-bounded value of a raw output ``std_raw`` of shape [..., A] that the actor
    model supplies next to the mean, see :class:`~.models.DetachedStdMLPModel`. Unlike the parent it holds no
    standard-deviation parameter of its own. The bounds loss, KL diagnostics and deterministic output are inherited.

    With ``std_type="log"`` the sigmoid acts between ``log(min)`` and ``log(max)``: equal steps of ``std_raw`` change
    the standard deviation by equal factors, and the entropy gradient is symmetric at both bounds.
    """

    _STD_TYPES = ("scalar", "log")

    def __init__(self, *args, **kwargs) -> None:
        """Initialize like :class:`SigmoidBoundedGaussianDistribution`; ``learn_std`` has no effect."""
        super().__init__(*args, **kwargs)
        del self.std_raw

    def update(self, mlp_output: torch.Tensor, std_raw: torch.Tensor) -> None:  # type: ignore[override]
        """Update the Gaussian from the mean and the raw standard deviation, both of shape [..., A]."""
        self._set_distribution(mlp_output, _std_from_raw(std_raw, *self.std_range, self.std_type))

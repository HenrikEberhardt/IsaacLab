# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Policy output distributions of the Franka cloth dragging task, for rsl_rl."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from rsl_rl.modules.distribution import Distribution, GaussianDistribution, HeteroscedasticGaussianDistribution
from torch.distributions import Normal


def _bounded_std_logit(name: str, init_std: float, std_range: tuple[float, float], std_type: str) -> float:
    """Check the bounded standard deviation settings and return the raw value (logit) that yields ``init_std``.

    Raises:
        ValueError: If ``std_type`` is not "scalar" or ``init_std`` does not lie strictly inside ``std_range``.
    """
    if std_type != "scalar":
        raise ValueError(f"{name} supports std_type 'scalar' only, got '{std_type}'.")
    low, high = float(std_range[0]), float(std_range[1])
    if not 0.0 < low < init_std < high:
        raise ValueError(f"init_std {init_std} must lie strictly inside std_range ({low}, {high}).")
    share = (init_std - low) / (high - low)
    return math.log(share / (1.0 - share))


class SigmoidBoundedGaussianDistribution(GaussianDistribution):
    """Gaussian with a state-independent standard deviation bounded smoothly to ``std_range``.

    The standard deviation is ``min + (max - min) * sigmoid(raw)`` of a learnable parameter ``raw`` per action. Unlike
    the hard clamp of :class:`~rsl_rl.modules.distribution.GaussianDistribution`, whose gradient vanishes once the
    parameter leaves the range, so it stays stuck at a bound, the bounded value always passes a gradient: the entropy
    bonus and the rewards can move it back from either bound.

    Kept for policies trained before :class:`SigmoidBoundedHeteroscedasticGaussianDistribution`.
    """

    def __init__(
        self,
        output_dim: int,
        init_std: float = 0.5,
        std_range: tuple[float, float] = (0.2, 1.0),
        std_type: str = "scalar",
        learn_std: bool = True,
    ) -> None:
        """Initialize the bounded standard deviation.

        Args:
            output_dim: Dimension of the action space.
            init_std: Initial standard deviation, strictly inside ``std_range``.
            std_range: Bounds (min, max) of the standard deviation.
            std_type: Only "scalar"; the bounded parameterization replaces the "log" one.
            learn_std: Whether the standard deviation is learnable.
        """
        # the base Gaussian would create its own clamped std parameter, so initialize the plain distribution module
        Distribution.__init__(self, output_dim)
        raw = _bounded_std_logit(type(self).__name__, init_std, std_range, std_type)
        self.std_type = std_type
        self.std_range = [float(std_range[0]), float(std_range[1])]
        self.std_raw = nn.Parameter(raw * torch.ones(output_dim), requires_grad=learn_std)
        self._distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the Gaussian from the MLP output (the mean) and the bounded standard deviation."""
        low, high = self.std_range
        std = low + (high - low) * torch.sigmoid(self.std_raw)
        self._distribution = Normal(mlp_output, std)


class SigmoidBoundedHeteroscedasticGaussianDistribution(HeteroscedasticGaussianDistribution):
    """Gaussian whose standard deviation is predicted per state by the MLP and bounded smoothly to ``std_range``.

    The MLP outputs the mean and a raw value per action; the standard deviation is ``min + (max - min) * sigmoid(raw)``.
    The policy can so explore differently per stage (e.g. searching while approaching, precisely while dragging), where
    a single standard deviation is a compromise between the stages. The sigmoid replaces the hard clamp of
    :class:`~rsl_rl.modules.distribution.HeteroscedasticGaussianDistribution`, whose gradient vanishes outside the
    range. The standard deviation head starts with zero weights, so every state starts at ``init_std``.
    """

    def __init__(
        self,
        output_dim: int,
        init_std: float = 0.5,
        std_range: tuple[float, float] = (0.2, 1.0),
        std_type: str = "scalar",
    ) -> None:
        """Initialize the bounded standard deviation head.

        Args:
            output_dim: Dimension of the action space.
            init_std: Initial standard deviation of every state, strictly inside ``std_range``.
            std_range: Bounds (min, max) of the standard deviation.
            std_type: Only "scalar"; the bounded parameterization replaces the "log" one.
        """
        self._raw_init = _bounded_std_logit(type(self).__name__, init_std, std_range, std_type)
        super().__init__(output_dim, init_std, (float(std_range[0]), float(std_range[1])), std_type)

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the Gaussian from the MLP output: the mean and the raw standard deviation, shape [..., 2, A]."""
        mean, raw = torch.unbind(mlp_output, dim=-2)
        low, high = self.std_range
        self._distribution = Normal(mean, low + (high - low) * torch.sigmoid(raw))

    def init_mlp_weights(self, mlp: nn.Module) -> None:
        """Start the standard deviation head at ``init_std`` in every state: zero weights, bias at its raw value."""
        nn.init.zeros_(mlp[-2].weight[self.output_dim :])  # type: ignore
        nn.init.constant_(mlp[-2].bias[self.output_dim :], self._raw_init)  # type: ignore

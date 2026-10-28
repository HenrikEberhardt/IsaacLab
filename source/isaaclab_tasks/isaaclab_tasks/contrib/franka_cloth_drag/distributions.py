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
    """

    def __init__(
        self,
        output_dim: int,
        init_std: float = 0.5,
        std_range: tuple[float, float] = (0.05, 1.0),
        std_type: str = "scalar",
        learn_std: bool = True,
    ) -> None:
        """Initialize the distribution.

        Args:
            output_dim: Dimension of the action space.
            init_std: Initial standard deviation, strictly inside ``std_range``.
            std_range: Bounds (min, max) of the standard deviation.
            std_type: Standard-deviation parameterization. Only ``"scalar"`` is supported.
            learn_std: Whether the standard deviation is learnable.

        Raises:
            ValueError: If ``std_type`` is unsupported or ``init_std`` is outside ``std_range``.
        """
        if std_type != "scalar":
            raise ValueError(f"{type(self).__name__} supports std_type 'scalar' only, got '{std_type}'.")
        low, high = float(std_range[0]), float(std_range[1])
        if not 0.0 < low < init_std < high:
            raise ValueError(f"init_std {init_std} must lie strictly inside std_range ({low}, {high}).")
        super().__init__(output_dim, learn_std=False)
        share = (init_std - low) / (high - low)
        self.std_range = [low, high]
        self.std_raw = nn.Parameter(torch.full((output_dim,), math.log(share / (1.0 - share))), learn_std)

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the Gaussian from the MLP output (the mean), shape [..., A]."""
        low, high = self.std_range
        std = low + (high - low) * torch.sigmoid(self.std_raw)
        self._distribution = Normal(mlp_output, std)

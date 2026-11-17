# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations of the action distribution and of the PPO of the Franka cloth dragging task."""

from typing import Literal

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlPpoAlgorithmCfg


@configclass
class BoundedGaussianDistributionCfg(RslRlMLPModelCfg.GaussianDistributionCfg):
    """Gaussian with one smoothly bounded, learned standard deviation per action dimension."""

    class_name: str = (
        "isaaclab_tasks.contrib.franka_cloth_drag.action_distribution.distributions:SigmoidBoundedGaussianDistribution"
    )

    std_range: tuple[float, float] = (0.05, 1.0)
    """Bounds (min, max) of the standard deviation."""

    mean_bound: float = 1.0
    """Magnitude of the mean action above which the bounds loss acts."""

    mean_bound_coef: float = 0.0
    """Weight of the bounds loss pulling the mean action back inside ``mean_bound``. Zero disables it."""

    std_range_per_dim: list[tuple[float, float]] | None = None
    """Bounds (min, max) of the standard deviation per action dimension, replacing ``std_range``. None uses
    ``std_range`` for every dimension."""

    init_std_per_dim: list[float] | None = None
    """Initial standard deviation per action dimension, replacing ``init_std``. None uses ``init_std`` for every
    dimension."""


@configclass
class BoundedHeteroscedasticGaussianDistributionCfg(BoundedGaussianDistributionCfg):
    """Gaussian with a smoothly bounded standard deviation per action dimension and state.

    Use it with :class:`DetachedStdMLPModelCfg`, which supplies the raw standard deviation.
    """

    class_name: str = (
        "isaaclab_tasks.contrib.franka_cloth_drag.action_distribution.distributions:"
        "SigmoidBoundedHeteroscedasticGaussianDistribution"
    )

    std_type: Literal["scalar", "log"] = "log"
    """Sigmoid between the standard-deviation bounds: ``"scalar"`` linear, ``"log"`` in log space."""


@configclass
class DetachedStdMLPModelCfg(RslRlMLPModelCfg):
    """MLP actor with a standard-deviation head on the detached features of its last hidden layer."""

    class_name: str = "isaaclab_tasks.contrib.franka_cloth_drag.action_distribution.models:DetachedStdMLPModel"

    detach_std_features: bool = True
    """Whether the standard-deviation head's gradient stops at the features, so it trains only the head."""


@configclass
class KLMetricsPPOCfg(RslRlPpoAlgorithmCfg):
    """PPO configuration that logs the adaptive-scheduler KL statistics."""

    class_name: str = "isaaclab_tasks.contrib.franka_cloth_drag.action_distribution.ppo:KLMetricsPPO"

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO diagnostics for the Franka cloth dragging task."""

import torch
from rsl_rl.algorithms import PPO

from .distributions import SigmoidBoundedGaussianDistribution

ACTION_DIM_NAMES = ("dx", "dy", "dz", "dyaw", "gripper")
"""Names of the action dimensions, in action order."""


class KLMetricsPPO(PPO):
    """PPO that reports the KL of the adaptive learning-rate scheduler (``kl_mean``, ``kl_max``) in each update."""

    def update(self) -> dict[str, float]:
        """Update the policy and add KL statistics to the logged loss metrics."""
        distribution = self._raw_actor.distribution
        if not isinstance(distribution, SigmoidBoundedGaussianDistribution):
            raise TypeError(f"{type(self).__name__} requires SigmoidBoundedGaussianDistribution.")

        distribution.reset_kl_metrics()
        loss_dict = super().update()
        kl_batch_means = distribution.kl_batch_means()
        if kl_batch_means.numel() == 0:
            return loss_dict

        if self.is_multi_gpu:
            torch.distributed.all_reduce(kl_batch_means, op=torch.distributed.ReduceOp.SUM)
            kl_batch_means /= self.gpu_world_size
        loss_dict["kl_mean"] = kl_batch_means.mean().item()
        loss_dict["kl_max"] = kl_batch_means.max().item()
        return loss_dict

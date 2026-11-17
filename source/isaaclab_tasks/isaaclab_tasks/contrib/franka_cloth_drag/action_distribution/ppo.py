# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO diagnostics for the Franka cloth dragging task."""

import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from ..mdp.stages import START_STAGE_NAMES
from .distributions import SigmoidBoundedGaussianDistribution

ACTION_DIM_NAMES = ("dx", "dy", "dz", "dyaw", "gripper")
"""Names of the action dimensions, in action order."""


class KLMetricsPPO(PPO):
    """PPO that reports the adaptive-scheduler KL and the action standard deviation per stage in each update.

    The per-stage standard deviation is logged as ``std_<stage>_<dimension>`` (``Loss/...``) when the environment has
    a ``stage`` observation group with the one-hot stage of every environment.
    """

    _std_sum: torch.Tensor | None = None
    _std_count: torch.Tensor | None = None

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and accumulate the standard deviation of the sampled states per stage."""
        actions = super().act(obs)
        if "stage" in obs.keys():
            one_hot = obs["stage"].float()
            std = self._raw_actor.output_std.detach().reshape(len(one_hot), -1)
            if self._std_sum is None:
                self._std_sum = torch.zeros(one_hot.shape[1], std.shape[1], device=std.device)
                self._std_count = torch.zeros(one_hot.shape[1], device=std.device)
            self._std_sum += one_hot.T @ std
            self._std_count += one_hot.sum(dim=0)
        return actions

    def update(self) -> dict[str, float]:
        """Update the policy and add KL statistics to the logged loss metrics."""
        distribution = self._raw_actor.distribution
        if not isinstance(distribution, SigmoidBoundedGaussianDistribution):
            raise TypeError(f"{type(self).__name__} requires SigmoidBoundedGaussianDistribution.")

        distribution.reset_kl_metrics()
        loss_dict = super().update()
        self._add_stage_std(loss_dict)
        kl_batch_means = distribution.kl_batch_means()
        if kl_batch_means.numel() == 0:
            return loss_dict

        if self.is_multi_gpu:
            torch.distributed.all_reduce(kl_batch_means, op=torch.distributed.ReduceOp.SUM)
            kl_batch_means /= self.gpu_world_size
        loss_dict["kl_mean"] = kl_batch_means.mean().item()
        loss_dict["kl_max"] = kl_batch_means.max().item()
        return loss_dict

    def _add_stage_std(self, loss_dict: dict[str, float]) -> None:
        """Add the mean standard deviation per visited stage and action dimension, then reset the accumulators."""
        if self._std_sum is None or self._std_count is None:
            return
        mean = (self._std_sum / self._std_count.clamp(min=1.0).unsqueeze(1)).cpu()
        visited = (self._std_count > 0).cpu()
        for stage, name in enumerate(START_STAGE_NAMES):
            if visited[stage]:
                for dim, dim_name in enumerate(ACTION_DIM_NAMES):
                    loss_dict[f"std_{name}_{dim_name}"] = mean[stage, dim].item()
        self._std_sum.zero_()
        self._std_count.zero_()

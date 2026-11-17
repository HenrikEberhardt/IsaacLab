# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actor model of the Franka cloth dragging task with a state-dependent standard deviation, for rsl_rl."""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.utils import unpad_trajectories
from tensordict import TensorDict

from .distributions import SigmoidBoundedHeteroscedasticGaussianDistribution


class DetachedStdMLPModel(MLPModel):
    """MLP actor whose state-dependent standard deviation comes from a separate head on detached features.

    The mean is the usual output layer of the MLP. The standard-deviation head is one linear layer on the features of
    the last hidden layer. With ``detach_std_features`` its gradient stops at those features, so the noisy
    standard-deviation gradient trains only the head and the hidden layers learn from the mean alone. The head
    starts with zero weights, so the standard deviation starts equal in every state. The exported deterministic policy
    is the mean network, unchanged.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        detach_std_features: bool = True,
    ) -> None:
        """Initialize the model.

        Args:
            obs: Observation dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set of this model, e.g. ``"actor"``.
            output_dim: Dimension of the action space.
            hidden_dims: Hidden dimensions of the MLP.
            activation: Activation function of the MLP.
            obs_normalization: Whether to normalize the observations.
            distribution_cfg: Configuration of a :class:`SigmoidBoundedHeteroscedasticGaussianDistribution`.
            detach_std_features: Whether the standard-deviation head reads detached features.

        Raises:
            TypeError: If the distribution is not a :class:`SigmoidBoundedHeteroscedasticGaussianDistribution`.
        """
        super().__init__(
            obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, distribution_cfg
        )
        if not isinstance(self.distribution, SigmoidBoundedHeteroscedasticGaussianDistribution):
            raise TypeError(f"{type(self).__name__} requires a SigmoidBoundedHeteroscedasticGaussianDistribution.")
        self.detach_std_features = detach_std_features
        self.std_head = nn.Linear(self.mlp[-1].in_features, output_dim)
        nn.init.zeros_(self.std_head.weight)
        with torch.no_grad():
            self.std_head.bias.copy_(self.distribution.init_raw)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Return the mean, or a sample of the Gaussian with the state-dependent standard deviation."""
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        # Run the layers one by one: slicing rsl_rl's MLP is not possible, and its activation is one module registered
        # under several names, which children() would skip, so index by position.
        layers = [self.mlp[i] for i in range(len(self.mlp))]
        features = self.get_latent(obs, masks, hidden_state)
        for layer in layers[:-1]:
            features = layer(features)
        mean = layers[-1](features)
        if not stochastic_output:
            return self.distribution.deterministic_output(mean)
        std_raw = self.std_head(features.detach() if self.detach_std_features else features)
        self.distribution.update(mean, std_raw)
        return self.distribution.sample()

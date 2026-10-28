# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the bounded action standard deviation of the Franka cloth dragging actor. No Kit/GPU required."""

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from isaaclab_tasks.contrib.franka_cloth_drag.distributions import SigmoidBoundedGaussianDistribution

STD_MIN = 0.05
STD_MAX = 1.0
INIT_STD = 0.5
NUM_ACTIONS, POLICY_DIM = 5, 8


def _obs(num_envs: int = 4) -> TensorDict:
    """Create policy observations."""
    torch.manual_seed(0)
    return TensorDict({"policy": torch.randn(num_envs, POLICY_DIM)}, batch_size=[num_envs])


def _model(obs: TensorDict) -> MLPModel:
    distribution_cfg = {
        "class_name": SigmoidBoundedGaussianDistribution,
        "init_std": INIT_STD,
        "std_range": (STD_MIN, STD_MAX),
        "std_type": "scalar",
    }
    return MLPModel(
        obs, {"actor": ["policy"]}, "actor", NUM_ACTIONS, hidden_dims=[16], distribution_cfg=distribution_cfg
    )


def test_std_starts_at_configured_value_and_stays_in_global_range():
    """The std starts at the configured value and stays within the global bounds."""
    obs = _obs()
    model = _model(obs)
    model(obs, stochastic_output=True)
    torch.testing.assert_close(model.output_std, torch.full((len(obs), NUM_ACTIONS), INIT_STD))

    distribution = model.distribution
    for raw, expected in ((50.0, STD_MAX), (-50.0, STD_MIN)):
        with torch.no_grad():
            distribution.std_raw.fill_(raw)
        distribution.update(torch.zeros(len(obs), NUM_ACTIONS))
        torch.testing.assert_close(distribution.std, torch.full((len(obs), NUM_ACTIONS), expected))


def test_std_is_state_independent_and_mean_uses_only_policy_observations():
    """The std is shared across states while the mean retains the ordinary policy input contract."""
    obs = _obs()
    model = _model(obs)
    with torch.no_grad():
        model.distribution.std_raw.normal_()
    model(obs, stochastic_output=True)
    std = model.output_std
    assert not torch.equal(obs["policy"][0], obs["policy"][1])
    torch.testing.assert_close(std[0], std[1])
    assert model.obs_dim == POLICY_DIM
    assert model.distribution.input_dim == NUM_ACTIONS


def test_entropy_keeps_a_gradient_near_both_std_bounds():
    """The smooth parameterization retains a gradient near the floor and ceiling."""
    distribution = SigmoidBoundedGaussianDistribution(
        NUM_ACTIONS, init_std=INIT_STD, std_range=(STD_MIN, STD_MAX)
    )
    for raw in (-4.0, 4.0):
        distribution.std_raw.grad = None
        with torch.no_grad():
            distribution.std_raw.fill_(raw)
        distribution.update(torch.zeros(2, NUM_ACTIONS))
        distribution.entropy.sum().backward()
        assert distribution.std_raw.grad is not None
        assert (distribution.std_raw.grad > 0.0).all()

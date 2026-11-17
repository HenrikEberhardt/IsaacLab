# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the bounded action standard deviation of the Franka cloth dragging actor. No Kit/GPU required."""

from types import SimpleNamespace

import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from isaaclab_tasks.contrib.franka_cloth_drag.action_distribution.distributions import (
    SigmoidBoundedGaussianDistribution,
    SigmoidBoundedHeteroscedasticGaussianDistribution,
)
from isaaclab_tasks.contrib.franka_cloth_drag.action_distribution.models import DetachedStdMLPModel
from isaaclab_tasks.contrib.franka_cloth_drag.action_distribution.ppo import ACTION_DIM_NAMES, KLMetricsPPO

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
    distribution = SigmoidBoundedGaussianDistribution(NUM_ACTIONS, init_std=INIT_STD, std_range=(STD_MIN, STD_MAX))
    for raw in (-4.0, 4.0):
        distribution.std_raw.grad = None
        with torch.no_grad():
            distribution.std_raw.fill_(raw)
        distribution.update(torch.zeros(2, NUM_ACTIONS))
        distribution.entropy.sum().backward()
        assert distribution.std_raw.grad is not None
        assert (distribution.std_raw.grad > 0.0).all()


def test_bounds_loss_gradient_pulls_only_an_excess_mean_back():
    """The hook adds exactly the bounds-loss gradient, which is zero for a mean inside the bound."""
    coef, bound = 0.5, 1.0
    distribution = SigmoidBoundedGaussianDistribution(
        NUM_ACTIONS, init_std=INIT_STD, std_range=(STD_MIN, STD_MAX), mean_bound=bound, mean_bound_coef=coef
    )
    mean = torch.tensor([[0.3, -0.9, 1.6, -2.5, 1.0], [-1.2, 0.0, 0.5, 3.0, -0.4]])
    actions = torch.zeros_like(mean)

    def log_prob_grad(with_bounds_loss: bool) -> torch.Tensor:
        distribution.mean_bound_coef = coef if with_bounds_loss else 0.0
        leaf = mean.clone().requires_grad_(True)
        distribution.update(leaf)
        distribution.log_prob(actions).sum().backward()
        return leaf.grad

    hooked, plain = log_prob_grad(True), log_prob_grad(False)
    # independent reference: autograd of the explicit loss
    leaf = mean.clone().requires_grad_(True)
    (coef * (leaf.abs() - bound).clamp(min=0.0).square().sum(dim=-1).mean()).backward()
    torch.testing.assert_close(hooked - plain, leaf.grad)
    inside = mean.abs() <= bound
    torch.testing.assert_close(hooked[inside], plain[inside])
    assert (torch.sign(hooked - plain)[~inside] == torch.sign(mean)[~inside]).all()


def test_kl_minibatch_means_are_retained_and_reset():
    """KL diagnostics retain one scalar per scheduler call and reset between PPO updates."""
    distribution = SigmoidBoundedGaussianDistribution(NUM_ACTIONS, init_std=INIT_STD, std_range=(STD_MIN, STD_MAX))
    old_std = torch.full((3, NUM_ACTIONS), INIT_STD)
    old_params = (torch.zeros_like(old_std), old_std)

    expected = []
    for shift in (0.1, 0.3):
        new_params = (torch.full_like(old_std, shift), old_std)
        kl = distribution.kl_divergence(old_params, new_params)
        expected.append(kl.mean())

    torch.testing.assert_close(distribution.kl_batch_means(), torch.stack(expected))
    distribution.reset_kl_metrics()
    assert distribution.kl_batch_means().shape == (0,)


def test_ppo_adds_mean_and_max_kl_to_loss_metrics(monkeypatch):
    """The task PPO adds scheduler-compatible KL summaries to the standard loss dictionary."""
    distribution = SigmoidBoundedGaussianDistribution(NUM_ACTIONS, init_std=INIT_STD, std_range=(STD_MIN, STD_MAX))
    old_std = torch.full((3, NUM_ACTIONS), INIT_STD)
    old_params = (torch.zeros_like(old_std), old_std)
    algorithm = object.__new__(KLMetricsPPO)
    algorithm._raw_actor = SimpleNamespace(distribution=distribution)
    algorithm.is_multi_gpu = False

    def update(_algorithm):
        for shift in (0.1, 0.3):
            new_params = (torch.full_like(old_std, shift), old_std)
            distribution.kl_divergence(old_params, new_params)
        return {"value": 2.0}

    monkeypatch.setattr(PPO, "update", update)
    metrics = algorithm.update()

    assert metrics["value"] == 2.0
    assert metrics["kl_mean"] == pytest.approx(0.5)
    assert metrics["kl_max"] == pytest.approx(0.9)


def _detached_model(obs: TensorDict, std_type: str = "log", detach: bool = True) -> DetachedStdMLPModel:
    distribution_cfg = {
        "class_name": SigmoidBoundedHeteroscedasticGaussianDistribution,
        "init_std": INIT_STD,
        "std_range": (STD_MIN, STD_MAX),
        "std_type": std_type,
    }
    return DetachedStdMLPModel(
        obs,
        {"actor": ["policy"]},
        "actor",
        NUM_ACTIONS,
        hidden_dims=[16, 16],
        distribution_cfg=distribution_cfg,
        detach_std_features=detach,
    )


@pytest.mark.parametrize("std_type", ["scalar", "log"])
def test_heteroscedastic_std_starts_equal_everywhere_and_then_differs_by_state(std_type):
    """The std head starts at the configured value in every state and can then learn to depend on the state."""
    obs = _obs()
    model = _detached_model(obs, std_type)
    model(obs, stochastic_output=True)
    torch.testing.assert_close(model.output_std, torch.full((len(obs), NUM_ACTIONS), INIT_STD))

    with torch.no_grad():
        model.std_head.weight.normal_()
    model(obs, stochastic_output=True)
    assert not torch.allclose(model.output_std[0], model.output_std[1])
    assert (model.output_std >= STD_MIN).all() and (model.output_std <= STD_MAX).all()


def test_log_sigmoid_std_is_geometric_between_the_bounds():
    """The log parameterization bounds the std in log space, and the linear one keeps its previous formula."""
    raw = torch.tensor([[-50.0, 0.0, 50.0]])
    for std_type, middle in (("log", (STD_MIN * STD_MAX) ** 0.5), ("scalar", 0.5 * (STD_MIN + STD_MAX))):
        distribution = SigmoidBoundedHeteroscedasticGaussianDistribution(
            3, init_std=0.3, std_range=(STD_MIN, STD_MAX), std_type=std_type
        )
        distribution.update(torch.zeros(1, 3), raw)
        torch.testing.assert_close(distribution.std, torch.tensor([[STD_MIN, middle, STD_MAX]]))


def test_global_distribution_rejects_log_std_type_and_model_rejects_global_distribution():
    """Only the heteroscedastic distribution supports the log type, and the detached model needs it."""
    with pytest.raises(ValueError):
        SigmoidBoundedGaussianDistribution(NUM_ACTIONS, init_std=INIT_STD, std_type="log")
    obs = _obs()
    with pytest.raises(TypeError):
        DetachedStdMLPModel(
            obs,
            {"actor": ["policy"]},
            "actor",
            NUM_ACTIONS,
            hidden_dims=[16],
            distribution_cfg={"class_name": SigmoidBoundedGaussianDistribution, "init_std": INIT_STD},
        )


@pytest.mark.parametrize("detach", [True, False])
def test_std_head_gradient_reaches_the_hidden_layers_only_without_detaching(detach):
    """Entropy and log-probability gradients of the std reach the head, and the trunk only when not detached."""
    obs = _obs()
    model = _detached_model(obs, detach=detach)
    with torch.no_grad():
        model.std_head.weight.normal_()
    # the mean is held fixed through a zero output layer, so any trunk gradient can only come from the std
    with torch.no_grad():
        model.mlp[-1].weight.zero_()
        model.mlp[-1].bias.zero_()
    action = model(obs, stochastic_output=True).detach()
    (model.get_output_log_prob(action).sum() + model.output_entropy.sum()).backward()

    assert model.std_head.weight.grad.abs().sum() > 0.0
    trunk_grad = sum(
        p.grad.abs().sum() for layer in list(model.mlp)[:-1] for p in layer.parameters() if p.grad is not None
    )
    assert (trunk_grad > 0.0) == (not detach)


def test_exported_policy_is_the_mean_network():
    """The JIT export holds the mean network only and returns the deterministic action."""
    obs = _obs()
    model = _detached_model(obs)
    with torch.no_grad():
        model.std_head.weight.normal_()
    exported = model.as_jit()
    torch.testing.assert_close(exported(obs["policy"]), model(obs))
    assert not any(name.startswith("std_head") for name, _ in exported.named_parameters())


def test_ppo_logs_the_mean_std_per_visited_stage(monkeypatch):
    """The std of the sampled states is averaged per stage and action dimension, and cleared after the update."""
    distribution = SigmoidBoundedGaussianDistribution(NUM_ACTIONS, init_std=INIT_STD, std_range=(STD_MIN, STD_MAX))
    std = torch.tensor([[0.1] * NUM_ACTIONS, [0.3] * NUM_ACTIONS, [0.5] * NUM_ACTIONS])
    actor = SimpleNamespace(distribution=distribution, output_std=std)
    algorithm = object.__new__(KLMetricsPPO)
    algorithm._raw_actor = actor
    algorithm.is_multi_gpu = False
    monkeypatch.setattr(PPO, "act", lambda _algorithm, obs: torch.zeros(len(obs), NUM_ACTIONS))
    monkeypatch.setattr(PPO, "update", lambda _algorithm: {})

    stage = torch.nn.functional.one_hot(torch.tensor([0, 0, 2]), num_classes=4).float()
    obs = TensorDict({"policy": torch.zeros(3, POLICY_DIM), "stage": stage}, batch_size=[3])
    algorithm.act(obs)
    metrics = algorithm.update()

    for name in ACTION_DIM_NAMES:
        assert metrics[f"std_approach_{name}"] == pytest.approx(0.2)
        assert metrics[f"std_drag_{name}"] == pytest.approx(0.5)
        assert f"std_grasp_{name}" not in metrics
    assert not any(key.startswith("std_") for key in algorithm.update())

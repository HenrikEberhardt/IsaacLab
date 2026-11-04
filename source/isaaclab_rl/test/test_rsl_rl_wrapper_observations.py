# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-free checks for the observation contract of :class:`RslRlVecEnvWrapper`.

The wrapper reads the environment-owned ``obs_buf`` instead of calling the environment's
private ``_get_observations``. These tests pin that contract without a simulator, so a
regression is caught before the Kit-dependent wrapper tests run.
"""

import inspect
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from isaaclab.envs import DirectRLEnv

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class _FakeEnv:
    """Minimal stand-in exposing only what :meth:`get_observations` reads."""

    def __init__(self):
        self.unwrapped = self
        self.obs_buf = {"policy": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}


class _StepEnv(_FakeEnv):
    """Minimal environment recording the actions received by :meth:`RslRlVecEnvWrapper.step`."""

    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(is_finite_horizon=True)
        self.received_actions = None

    def step(self, actions: torch.Tensor):
        self.received_actions = actions.clone()
        zeros = torch.zeros(len(actions))
        return self.obs_buf, zeros, zeros.bool(), zeros.bool(), {}


class _UnsupportedEnv:
    pass


class _OuterEnv:
    def __init__(self):
        self.unwrapped = _UnsupportedEnv()


def _make_wrapper(env: _FakeEnv, num_envs: int = 2) -> RslRlVecEnvWrapper:
    """Build a wrapper without ``__init__``, which requires a real environment and a live sim."""
    wrapper = object.__new__(RslRlVecEnvWrapper)
    wrapper.env = env
    wrapper.num_envs = num_envs
    wrapper.clip_actions = None
    wrapper.num_actions = 2
    wrapper.action_dim_names = ["dim_0", "dim_1"]
    wrapper._diagnostics_policy = None
    return wrapper


def test_get_observations_returns_the_environment_buffer():
    """The wrapper should hand back the environment's own observation buffer."""
    env = _FakeEnv()
    wrapper = _make_wrapper(env)

    observations = wrapper.get_observations()

    assert isinstance(observations, TensorDict)
    torch.testing.assert_close(observations["policy"], env.obs_buf["policy"])


def test_get_observations_tracks_buffer_updates():
    """Successive reads should reflect the latest reset/step observations."""
    env = _FakeEnv()
    wrapper = _make_wrapper(env)

    env.obs_buf = {"policy": torch.tensor([[5.0, 6.0], [7.0, 8.0]])}

    torch.testing.assert_close(wrapper.get_observations()["policy"], env.obs_buf["policy"])


def test_get_observations_does_not_use_private_environment_methods():
    """The wrapper must not reach into the environment's private observation API."""
    env = _FakeEnv()

    def _private_call():
        raise AssertionError("get_observations() must not call the private _get_observations()")

    env._get_observations = _private_call
    wrapper = _make_wrapper(env)

    wrapper.get_observations()


def test_direct_rl_env_stores_the_observation_buffer():
    """The environment side of the contract: reset and step both publish ``obs_buf``."""
    assert "self.obs_buf" in inspect.getsource(DirectRLEnv.reset)
    assert "self.obs_buf" in inspect.getsource(DirectRLEnv.step)


def test_invalid_environment_error_reports_unwrapped_type():
    """The validation error should identify the object that failed the type check."""
    with pytest.raises(ValueError) as exc_info:
        RslRlVecEnvWrapper(_OuterEnv())

    assert "_UnsupportedEnv" in str(exc_info.value)
    assert "_OuterEnv" not in str(exc_info.value)


def test_action_diagnostics_log_policy_distribution_and_commanded_magnitude_per_named_dimension():
    """With named dimensions and a policy, the diagnostics should log its distribution and the commanded magnitude."""
    wrapper = _make_wrapper(_FakeEnv())
    wrapper.action_dim_names = ["dx", "gripper"]
    policy = SimpleNamespace(
        output_mean=torch.tensor([[0.2, -1.0], [0.6, 1.0]]), output_std=torch.tensor([[0.5, 0.1], [0.5, 0.1]])
    )
    wrapper.set_diagnostics_policy(policy)
    extras = {}

    wrapper._log_action_diagnostics(torch.tensor([[1.5, -0.5], [-0.5, 2.0]]), extras)

    log = extras["log"]
    assert len(log) == 6
    torch.testing.assert_close(log["Action/policy_mean_dx"], torch.tensor(0.4))
    torch.testing.assert_close(log["Action/policy_mean_gripper"], torch.tensor(0.0))
    torch.testing.assert_close(log["Action/policy_std_dx"], torch.tensor(0.5))
    torch.testing.assert_close(log["Action/policy_std_gripper"], torch.tensor(0.1))
    torch.testing.assert_close(log["Action/commanded_abs_dx"], torch.tensor(1.0))
    torch.testing.assert_close(log["Action/commanded_abs_gripper"], torch.tensor(1.25))


def test_action_diagnostics_are_opt_in():
    """The wrapper should not add metrics without a diagnostics policy."""
    wrapper = _make_wrapper(_FakeEnv())
    extras = {}

    wrapper._log_action_diagnostics(torch.ones(2, 2), extras)

    assert extras == {}


def test_step_logs_the_clipped_action_sent_to_the_environment():
    """The commanded magnitude should describe what the environment received, not the raw sample."""
    env = _StepEnv()
    wrapper = _make_wrapper(env)
    wrapper.clip_actions = 2.0
    wrapper.set_diagnostics_policy(SimpleNamespace(output_mean=torch.zeros(2, 2), output_std=torch.ones(2, 2)))
    actions = torch.tensor([[3.0, -1.5], [-4.0, 0.5]])

    _, _, _, extras = wrapper.step(actions)

    torch.testing.assert_close(env.received_actions, torch.tensor([[2.0, -1.5], [-2.0, 0.5]]))
    torch.testing.assert_close(extras["log"]["Action/commanded_abs_dim_0"], torch.tensor(2.0))
    torch.testing.assert_close(extras["log"]["Action/commanded_abs_dim_1"], torch.tensor(1.0))

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
    wrapper.action_clip_diagnostics = None
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


def test_action_clip_diagnostics_measure_raw_action_exceedances():
    """Diagnostics should describe the raw policy samples relative to the configured threshold."""
    wrapper = _make_wrapper(_FakeEnv())
    wrapper.action_clip_diagnostics = 1.0
    actions = torch.tensor([[0.0, 1.5], [-2.0, 0.5]])
    extras = {}

    wrapper._log_action_clip_diagnostics(actions, extras)

    log = extras["log"]
    torch.testing.assert_close(log["Action/would_clip_fraction"], torch.tensor(0.5))
    torch.testing.assert_close(log["Action/would_clip_env_fraction"], torch.tensor(1.0))
    torch.testing.assert_close(log["Action/would_clip_correction_l1_mean"], torch.tensor(0.375))
    torch.testing.assert_close(log["Action/would_clip_excess_mean"], torch.tensor(0.75))
    torch.testing.assert_close(log["Action/would_clip_excess_max"], torch.tensor(1.0))
    torch.testing.assert_close(log["Action/would_clip_fraction_dim_0"], torch.tensor(0.5))
    torch.testing.assert_close(log["Action/would_clip_fraction_dim_1"], torch.tensor(0.5))


def test_action_clip_diagnostics_are_opt_in():
    """The wrapper should not add metrics unless a diagnostic threshold is configured."""
    wrapper = _make_wrapper(_FakeEnv())
    extras = {}

    wrapper._log_action_clip_diagnostics(torch.ones(2, 2), extras)

    assert extras == {}

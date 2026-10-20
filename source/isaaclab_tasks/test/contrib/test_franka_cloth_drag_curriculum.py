# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the success-gated curriculum blocks of the Franka cloth dragging task. No Kit/GPU required."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import CurriculumTermCfg

from isaaclab_tasks.contrib.franka_cloth_drag.mdp.curriculums import ActionRateCurriculum, StageHidingCurriculum

START_WEIGHT = -1e-3
FINAL_WEIGHT = -1e-1


class _RewardManager:
    """Holds one reward term cfg, like the reward manager's get/set term cfg."""

    def __init__(self, weight: float):
        self.cfgs = {"pen_action_rate": SimpleNamespace(weight=weight)}

    def get_term_cfg(self, name):
        return self.cfgs[name]

    def set_term_cfg(self, name, cfg):
        self.cfgs[name] = cfg


def _env(num_envs: int):
    """A fake environment whose episode successes are set by the test."""
    success = torch.zeros(num_envs)
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        cfg=SimpleNamespace(seed=0),
        common_step_counter=0,
        reward_manager=_RewardManager(START_WEIGHT),
        command_manager=SimpleNamespace(get_term=lambda name: SimpleNamespace(metrics={"success": success})),
    )
    return env, success


def _params(**overrides):
    # ema_episodes 0: the average is the success of the last reset batch, which keeps the tests exact
    params = {"command_name": "cloth_target", "success_threshold": 0.8, "ramp_steps": 100, "ema_episodes": 0}
    return params | overrides


def _hiding(env, **overrides):
    params = _params(**overrides)
    return StageHidingCurriculum(CurriculumTermCfg(func=StageHidingCurriculum, params=params), env), params


def _action_rate(env, **overrides):
    params = _params(**({"reward_term": "pen_action_rate", "final_weight": FINAL_WEIGHT} | overrides))
    return ActionRateCurriculum(CurriculumTermCfg(func=ActionRateCurriculum, params=params), env), params


def _call(term, env, params, step, env_ids=slice(None)):
    env.common_step_counter = step
    return term(env, env_ids, **params)


def test_progress_is_gated_linear_and_held():
    """No progress at the initial reset or below the threshold; linear above it; held after a drop."""
    env, success = _env(num_envs=8)
    term, params = _hiding(env)

    success[:] = 1.0
    state = _call(term, env, params, step=0)  # initial reset: no finished episode counts
    assert state["success_ema"] == 0.0 and state["progress"] == 0.0 and state["hidden_fraction"] == 0.0

    success[:] = 0.0
    assert _call(term, env, params, step=50)["progress"] == 0.0

    success[:] = 1.0
    assert _call(term, env, params, step=100)["progress"] == pytest.approx(0.5)  # 50 of 100 ramp steps

    success[:] = 0.0
    assert _call(term, env, params, step=180)["progress"] == pytest.approx(0.5)  # held, not rolled back

    success[:] = 1.0
    state = _call(term, env, params, step=400)
    assert state["progress"] == 1.0 and state["hidden_fraction"] == 1.0


def test_action_rate_weight_ramps_geometrically():
    env, success = _env(num_envs=4)
    term, params = _action_rate(env, initial_progress=0.5)
    weight = lambda: env.reward_manager.get_term_cfg("pen_action_rate").weight  # noqa: E731

    assert _call(term, env, params, step=0)["weight"] == pytest.approx(-1e-2)  # halfway in log space
    assert weight() == pytest.approx(-1e-2)
    success[:] = 1.0
    assert _call(term, env, params, step=100)["weight"] == pytest.approx(FINAL_WEIGHT)


def test_action_rate_rejects_weights_that_cannot_ramp_geometrically():
    env, _ = _env(num_envs=2)
    with pytest.raises(ValueError, match="same sign"):
        _action_rate(env, final_weight=1e-1)


def test_stage_hiding_redraws_only_reset_episodes_with_the_progress_share():
    num_envs = 4000
    env, _ = _env(num_envs)
    term, params = _hiding(env, initial_progress=0.5)
    first, second = torch.arange(num_envs // 2), torch.arange(num_envs // 2, num_envs)

    # success below the threshold: the progress stays at 0.5
    _call(term, env, params, step=10, env_ids=first)
    assert not term.hidden[second].any()
    assert term.hidden[first].float().mean().item() == pytest.approx(0.5, abs=0.05)

    kept = term.hidden[first].clone()
    _call(term, env, params, step=20, env_ids=second)
    assert torch.equal(term.hidden[first], kept)
    assert term.hidden[second].float().mean().item() == pytest.approx(0.5, abs=0.05)


def test_blocks_advance_independently():
    """Each block has its own gate: with a success of 0.85 only the block gated at 0.8 advances."""
    env, success = _env(num_envs=20)
    hiding, hiding_params = _hiding(env, success_threshold=0.8)
    action_rate, action_rate_params = _action_rate(env, success_threshold=0.9)
    success[:17] = 1.0  # 17 of 20 episodes succeeded
    for step in (1, 51):
        hiding_state = _call(hiding, env, hiding_params, step)
        action_rate_state = _call(action_rate, env, action_rate_params, step)
    assert hiding_state["progress"] == pytest.approx(0.51)
    assert action_rate_state["progress"] == 0.0 and action_rate_state["weight"] == pytest.approx(START_WEIGHT)

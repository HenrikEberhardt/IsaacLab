# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU tests for the cloth-drag state machine."""

import math

import pytest
import torch

from isaaclab_tasks.contrib.franka_cloth_drag.state_machine import (
    ClothDragSmCfg,
    ClothDragSmObservation,
    DragClothSm,
    DragSmState,
    ee_down_tilt,
    feasible_yaw_step,
    limit_arm_action,
    make_sm_profile,
    sample_sm_params,
    wrap_half_turn,
)


def _observation(num_envs: int = 1) -> ClothDragSmObservation:
    zeros = torch.zeros((num_envs, 3))
    return ClothDragSmObservation(
        ee_pos=zeros.clone(),
        ee_yaw=torch.zeros(num_envs),
        grasp_node_pos=zeros.clone(),
        cloth_center_pos=torch.tensor((1.0, 0.0, 0.0)).expand(num_envs, 3).clone(),
        goal_pos=zeros.clone(),
        target_pos=torch.tensor((0.3, 0.0, 0.0)).expand(num_envs, 3).clone(),
        valid_grasp=torch.zeros(num_envs, dtype=torch.bool),
        success=torch.zeros(num_envs, dtype=torch.bool),
    )


def test_state_machine_runs_one_attempt_without_retry():
    cfg = ClothDragSmCfg(
        hover_height=0.03,
        grasp_height=0.02,
        pregrasp_hold_time=0.0,
        hover_hold_time=0.0,
        descend_hold_time=0.0,
        grasp_timeout=0.2,
    )
    sm = DragClothSm(0.1, 1, "cpu", cfg)
    obs = _observation()

    obs.ee_pos[:, 2] = cfg.hover_height
    sm.compute(obs)
    assert sm.sm_state.item() == int(DragSmState.HOVER)
    sm.compute(obs)
    assert sm.sm_state.item() == int(DragSmState.DESCEND)

    obs.ee_pos[:, 2] = cfg.grasp_height
    sm.compute(obs)
    assert sm.sm_state.item() == int(DragSmState.GRASP)
    sm.compute(obs)
    sm.compute(obs)
    sm.compute(obs)
    assert sm.sm_state.item() == int(DragSmState.FAILED)


def test_valid_grasp_reaches_drag_and_hold():
    cfg = ClothDragSmCfg(
        hover_height=0.03,
        grasp_height=0.02,
        pregrasp_hold_time=0.0,
        hover_hold_time=0.0,
        descend_hold_time=0.0,
        grasp_squeeze_time=0.0,
    )
    sm = DragClothSm(0.1, 1, "cpu", cfg)
    obs = _observation()
    obs.ee_pos[:, 2] = cfg.hover_height
    sm.compute(obs)
    sm.compute(obs)
    obs.ee_pos[:, 2] = cfg.grasp_height
    sm.compute(obs)
    obs.valid_grasp[:] = True
    sm.compute(obs)
    assert sm.sm_state.item() == int(DragSmState.DRAG)

    obs.target_pos[:] = obs.goal_pos
    sm.compute(obs)
    assert sm.sm_state.item() == int(DragSmState.HOLD)


def test_feasible_yaw_step_respects_symmetry_and_wrist_window():
    desired = torch.tensor((-0.5, 0.5))
    current = torch.zeros(2)
    wrist = torch.tensor((2.0, 0.0))
    limits = (-2.0 * math.pi / 3.0, 2.0 * math.pi / 3.0)
    step = feasible_yaw_step(desired, current, wrist, limits)

    predicted_wrist = wrist - step
    assert ((predicted_wrist >= limits[0]) & (predicted_wrist <= limits[1])).all()
    torch.testing.assert_close(wrap_half_turn(step), wrap_half_turn(desired - current))


def test_randomized_profile_is_seeded():
    cfg, randomization = make_sm_profile("randomized")
    first = sample_sm_params(cfg, randomization, 8, "cpu", torch.Generator().manual_seed(7))
    second = sample_sm_params(cfg, randomization, 8, "cpu", torch.Generator().manual_seed(7))
    torch.testing.assert_close(first.drag_height, second.drag_height)
    torch.testing.assert_close(first.drag_speed, second.drag_speed)
    torch.testing.assert_close(first.pregrasp_offset, second.pregrasp_offset)


def test_ee_down_tilt_ignores_yaw_and_measures_down_alignment():
    quaternions = torch.tensor(
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    tilt = ee_down_tilt(quaternions)

    torch.testing.assert_close(tilt[:2], torch.zeros(2))
    torch.testing.assert_close(tilt[2], torch.tensor(math.pi))


def test_arm_action_limit_uniformly_scales_each_row():
    actions = torch.tensor(((1.0, -0.5, 0.25, 0.75), (0.2, -0.3, 0.1, 0.0)))
    limited = limit_arm_action(actions, 0.7)

    torch.testing.assert_close(limited[0], actions[0] * 0.7)
    torch.testing.assert_close(limited[1], actions[1])
    assert limited.abs().amax().item() == pytest.approx(0.7)
    with pytest.raises(ValueError, match="limit"):
        limit_arm_action(actions, 0.0)


@pytest.mark.parametrize("limit", [0.0, -0.1, 1.1])
def test_arm_action_limit_setting_is_checked(limit):
    with pytest.raises(ValueError, match="arm_action_limit"):
        ClothDragSmCfg(arm_action_limit=limit)

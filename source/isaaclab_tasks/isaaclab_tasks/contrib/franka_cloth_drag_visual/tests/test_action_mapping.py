# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teacher-to-student action mappings on a synthetic IK state. No Kit/GPU required."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from isaaclab_tasks.contrib.franka_cloth_drag_visual.action_mapping import (
    IdentityMapping,
    IdentityMappingCfg,
    IKToJointDeltaMapping,
    IKToJointDeltaMappingCfg,
)
from isaaclab_tasks.contrib.franka_cloth_drag_visual.teacher.ik_action import TOP_DOWN_QUAT


def _ik_state(num_envs: int = 2, ee_z: float = 0.2) -> torch.Tensor:
    """Hand down at (0.5, 0, ee_z) m, Jacobian [I6 | 0], wide joint bounds."""
    joint_pos = torch.zeros(num_envs, 7)
    ee_pos = torch.tensor([0.5, 0.0, ee_z]).expand(num_envs, 3)
    ee_quat = torch.tensor(TOP_DOWN_QUAT).expand(num_envs, 4)
    jacobian = torch.cat((torch.eye(6), torch.zeros(6, 1)), dim=1).expand(num_envs, 6, 7)
    lower, upper, cap = torch.full((num_envs, 7), -3.0), torch.full((num_envs, 7), 3.0), torch.full((num_envs, 7), 1.0)
    return torch.cat((joint_pos, ee_pos, ee_quat, jacobian.flatten(1), lower, upper, cap), dim=-1)


def _mapping(**kwargs) -> IKToJointDeltaMapping:
    """Build the mapping from its dict form, as the distillation algorithm does."""
    cfg = IKToJointDeltaMappingCfg(**kwargs).to_dict()
    cfg.pop("class_name")
    return IKToJointDeltaMapping(**cfg)


def _obs(num_envs: int = 2, ee_z: float = 0.2) -> TensorDict:
    return TensorDict({"ik_state": _ik_state(num_envs, ee_z)}, batch_size=[num_envs])


def test_identity_passes_teacher_actions():
    actions = torch.randn(3, 8)
    labels, stats = IdentityMapping(**{k: v for k, v in IdentityMappingCfg().to_dict().items() if k != "class_name"})(
        actions, TensorDict({}, batch_size=[3])
    )
    assert torch.equal(labels, actions) and stats == {}


def test_ik_label_matches_reference_dls_step():
    mapping = _mapping(ik_lambda=0.1, joint_delta_scale=0.05)
    actions = torch.tensor([[1.0, 0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0, 1.0]])
    labels, stats = mapping(actions, _obs())
    # J = [I6 | 0]: step = e / (1 + lambda^2) on joint 1, label = step / scale
    expected = 0.015 / (1.0 + 0.1**2) / 0.05
    assert labels.shape == (2, 8)
    assert labels[0, 0].item() == pytest.approx(expected, rel=1e-5)
    assert labels[0, 1:7].abs().max().item() < 1e-5
    assert labels[1, :7].abs().max().item() < 1e-5  # zero action, zero label
    assert torch.equal(labels[:, 7], actions[:, 4])  # gripper passes through
    assert stats["label_clip_fraction"].item() == 0.0


@pytest.mark.parametrize("preserve_direction, expected", [(True, (1.0, 0.5)), (False, (1.0, 1.0))])
def test_over_range_labels(preserve_direction, expected):
    mapping = _mapping(ik_lambda=0.0, joint_delta_scale=0.001, preserve_direction=preserve_direction)
    labels, stats = mapping(torch.tensor([[1.0, 0.5, 0.0, 0.0, 1.0]]), _obs(1))
    assert labels[0, :2].tolist() == pytest.approx(expected, rel=1e-5)
    assert stats["label_clip_fraction"].item() == pytest.approx(2 / 7)


def test_z_floor_shortens_a_downward_step():
    mapping = _mapping(ik_lambda=0.1, joint_delta_scale=0.05, z_min=0.0193)
    labels, _ = mapping(torch.tensor([[0.0, 0.0, -1.0, 0.0, 1.0]]), _obs(1, ee_z=0.02))
    # target 0.02 - 0.015 lies below the floor, so the step ends on it: dz = 0.0193 - 0.02
    expected = (0.0193 - 0.02) / (1.0 + 0.1**2) / 0.05
    assert labels[0, 2].item() == pytest.approx(expected, rel=1e-4)


def test_label_scale_is_read_from_the_student_action():
    term = SimpleNamespace(cfg=SimpleNamespace(scale=0.1))
    env = SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda name: term))
    cfg = IKToJointDeltaMappingCfg().to_dict()
    assert IKToJointDeltaMapping.resolve_cfg(cfg, env)["joint_delta_scale"] == 0.1
    assert IKToJointDeltaMapping.resolve_cfg({**cfg, "joint_delta_scale": 0.05}, env)["joint_delta_scale"] == 0.05

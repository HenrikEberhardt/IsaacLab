# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Staged rewards for the Franka non-prehensile cloth task.

The rewards follow the task's stages: reach the crease's grasp point, align the gripper with the crease and close it
there, grasp the crease, then drag it to its target. Once the gripper holds the crease, the terms of the first stage pay
their held value, their maximum, so grasping never costs reward, and only then do the drag terms pay.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .observations import ee_yaw_error_to_crease

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg
    from isaaclab.sensors import FrameTransformer

    from .actions import ClothGraspAction
    from .commands import CreaseDragCommand

HELD_DISTANCE = 0.04
"""Distance of the fingertip point from the crease's grasp point within which a grasp holds the crease [m]."""


def _ee_grasp_distance(env: ManagerBasedRLEnv, command_name: str, ee_frame_cfg: SceneEntityCfg) -> torch.Tensor:
    """Distance of the fingertip point from the crease's grasp point [m], shape [N]."""
    command: CreaseDragCommand = env.command_manager.get_term(command_name)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return torch.linalg.norm(command.grasp_pos_w - ee_frame.data.target_pos_w.torch[..., 0, :], dim=-1)


def _crease_held(
    env: ManagerBasedRLEnv, command_name: str, held_distance: float, action_name: str, ee_frame_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Whether the gripper holds cloth within ``held_distance`` [m] of the crease's grasp point, shape [N]."""
    gripper: ClothGraspAction = env.action_manager.get_term(action_name)
    return gripper.is_grasping & (_ee_grasp_distance(env, command_name, ee_frame_cfg) < held_distance)


def crease_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    held_value: float | None = 1.0,
    held_distance: float = HELD_DISTANCE,
    action_name: str = "gripper_action",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward end-effector proximity to the crease's grasp point with a tanh kernel (``std`` [m]).

    While the gripper holds the crease, return ``held_value`` instead, unless it is None.
    """
    reward = 1.0 - torch.tanh(_ee_grasp_distance(env, command_name, ee_frame_cfg) / std)
    if held_value is None:
        return reward
    return torch.where(_crease_held(env, command_name, held_distance, action_name, ee_frame_cfg), held_value, reward)


def crease_yaw_alignment(
    env: ManagerBasedRLEnv,
    command_name: str,
    held_value: float | None = 1.0,
    held_distance: float = HELD_DISTANCE,
    action_name: str = "gripper_action",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the gripper for turning to close its fingers across the crease: the squared cosine of its yaw error.

    While the gripper holds the crease, return ``held_value`` instead, unless it is None.
    """
    reward = torch.cos(ee_yaw_error_to_crease(env, command_name, ee_frame_cfg)).square()
    if held_value is None:
        return reward
    return torch.where(_crease_held(env, command_name, held_distance, action_name, ee_frame_cfg), held_value, reward)


def crease_closing(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    open_opening: float = 0.04,
    held_value: float | None = 1.0,
    held_distance: float = HELD_DISTANCE,
    action_name: str = "gripper_action",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward closing the gripper at the crease: how far the fingers have closed from ``open_opening`` [m], times a
    tanh kernel (``std`` [m]) on the distance of the end-effector from the crease's grasp point.

    While the gripper holds the crease, return ``held_value`` instead, unless it is None.
    """
    gripper: ClothGraspAction = env.action_manager.get_term(action_name)
    closed = (1.0 - gripper.finger_opening / open_opening).clamp(0.0, 1.0)
    reward = closed * (1.0 - torch.tanh(_ee_grasp_distance(env, command_name, ee_frame_cfg) / std))
    if held_value is None:
        return reward
    return torch.where(_crease_held(env, command_name, held_distance, action_name, ee_frame_cfg), held_value, reward)


def crease_closing_away(
    env: ManagerBasedRLEnv,
    margin: float,
    std: float,
    command_name: str,
    open_opening: float = 0.04,
    held_distance: float = HELD_DISTANCE,
    action_name: str = "gripper_action",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Measure closing the gripper away from the crease, for use as a penalty: how far the fingers have closed from
    ``open_opening`` [m], times a tanh ramp (``std`` [m]) of the end-effector's distance from the crease's grasp point
    beyond ``margin`` [m].

    It is zero within ``margin`` of the grasp point and while the gripper holds the crease.
    """
    gripper: ClothGraspAction = env.action_manager.get_term(action_name)
    closed = (1.0 - gripper.finger_opening / open_opening).clamp(0.0, 1.0)
    excess = (_ee_grasp_distance(env, command_name, ee_frame_cfg) - margin).clamp_min(0.0)
    held = _crease_held(env, command_name, held_distance, action_name, ee_frame_cfg)
    return closed * torch.tanh(excess / std) * ~held


def crease_grasped(
    env: ManagerBasedRLEnv,
    command_name: str,
    held_distance: float = HELD_DISTANCE,
    action_name: str = "gripper_action",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Return one while the gripper holds cloth within ``held_distance`` [m] of the crease's grasp point and zero
    otherwise."""
    return _crease_held(env, command_name, held_distance, action_name, ee_frame_cfg).float()


def crease_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    require_held: bool = True,
    held_distance: float = HELD_DISTANCE,
    action_name: str = "gripper_action",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the horizontal proximity of the crease to its drag target with a tanh kernel (``std`` [m]).

    With ``require_held``, pay it only while the gripper holds the crease.
    """
    command: CreaseDragCommand = env.command_manager.get_term(command_name)
    reward = 1.0 - torch.tanh(command.crease_error / std)
    if not require_held:
        return reward
    return reward * _crease_held(env, command_name, held_distance, action_name, ee_frame_cfg)


class CreaseGoalReached(ManagerTermBase):
    """Reward the crease for reaching its drag target, and log the episode success rate.

    An episode succeeds once the crease has come within ``success_threshold`` [m] of the target. The sticky
    :attr:`succeeded` flag drives the lift tasks' :class:`~isaaclab_tasks.core.lift.mdp.DifficultyScheduler`.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        """Whether the crease has reached its target during the current episode. Shape is (num_envs,)."""

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._env.extras.setdefault("log", {})["Metrics/success_rate"] = self.succeeded[env_ids].float().mean().item()
        self.succeeded[env_ids] = False

    def __call__(self, env: ManagerBasedRLEnv, command_name: str, success_threshold: float) -> torch.Tensor:
        command: CreaseDragCommand = env.command_manager.get_term(command_name)
        reached = command.crease_error < success_threshold
        self.succeeded |= reached
        return reached.float()

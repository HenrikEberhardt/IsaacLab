# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms specific to the Franka cloth dragging task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from .frames import cloth_com_pos_b

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, DeformableObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer


def _cloth_goal_xy_distance(
    env: ManagerBasedRLEnv, command_name: str, robot_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Planar distance between the cloth's center of mass and the commanded target [m]. Shape is (num_envs,)."""
    command = env.command_manager.get_command(command_name)
    return torch.linalg.norm(command[:, :2] - cloth_com_pos_b(env, asset_cfg, robot_cfg)[:, :2], dim=1)


def _gripper_closing(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """Whether the binary gripper action commands closing. Shape is (num_envs,)."""
    return torch.any(env.action_manager.get_term(action_name).raw_actions < 0.0, dim=1)


def _fingertips_near_cloth(
    env: ManagerBasedRLEnv, grasp_radius: float, asset_cfg: SceneEntityCfg, ee_frame_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Whether the fingertip point is within ``grasp_radius`` [m] of the nearest cloth node. Shape is (num_envs,)."""
    asset: DeformableObject = env.scene[asset_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w.torch[..., 0, :]
    distance = torch.linalg.norm(asset.data.nodal_pos_w.torch - ee_w.unsqueeze(1), dim=2).min(dim=1).values
    return distance < grasp_radius


def _cloth_grasped(
    env: ManagerBasedRLEnv,
    grasp_radius: float,
    min_finger_opening: float,
    action_name: str,
    finger_joint_name: str,
    robot_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Whether the gripper closes at the cloth and the cloth holds the finger open by more than
    ``min_finger_opening`` [m]. Shape is (num_envs,)."""
    robot: Articulation = env.scene[robot_cfg.name]
    finger_ids, _ = robot.find_joints(finger_joint_name)
    # the closed target is 0.0, so a finger that stays open is blocked by the cloth between the fingers
    blocked = robot.data.joint_pos.torch[:, finger_ids[0]] > min_finger_opening
    closing = _gripper_closing(env, action_name)
    return closing & blocked & _fingertips_near_cloth(env, grasp_radius, asset_cfg, ee_frame_cfg)


def cloth_goal_distance_xy(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    grasp_radius: float | None = None,
    min_finger_opening: float = 0.001,
    action_name: str = "gripper_action",
    finger_joint_name: str = "panda_finger_joint1",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the cloth's center of mass for approaching the target in the table plane with a tanh kernel
    (``std`` [m]).

    If ``grasp_radius`` [m] is set, the reward is only paid while the cloth is grasped.
    """
    reward = 1.0 - torch.tanh(_cloth_goal_xy_distance(env, command_name, robot_cfg, asset_cfg) / std)
    if grasp_radius is not None:
        reward *= _cloth_grasped(
            env, grasp_radius, min_finger_opening, action_name, finger_joint_name, robot_cfg, asset_cfg, ee_frame_cfg
        ).float()
    return reward


def cloth_settled_at_target(
    env: ManagerBasedRLEnv,
    std_speed: float,
    success_radius: float,
    command_name: str,
    grasp_radius: float | None = None,
    min_finger_opening: float = 0.001,
    action_name: str = "gripper_action",
    finger_joint_name: str = "panda_finger_joint1",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward a slow cloth inside the target radius (``success_radius`` [m]) with a tanh kernel on the speed of its
    center of mass (``std_speed`` [m/s]), so the cloth stops at the target instead of overshooting it.

    If ``grasp_radius`` [m] is set, the reward is only paid while the cloth is grasped.
    """
    asset: DeformableObject = env.scene[asset_cfg.name]
    in_radius = _cloth_goal_xy_distance(env, command_name, robot_cfg, asset_cfg) < success_radius
    speed = torch.linalg.norm(asset.data.root_vel_w.torch, dim=1)
    reward = in_radius.float() * (1.0 - torch.tanh(speed / std_speed))
    if grasp_radius is not None:
        reward *= _cloth_grasped(
            env, grasp_radius, min_finger_opening, action_name, finger_joint_name, robot_cfg, asset_cfg, ee_frame_cfg
        ).float()
    return reward


def gripper_closed_near_cloth(
    env: ManagerBasedRLEnv,
    grasp_radius: float,
    action_name: str = "gripper_action",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Return one when the gripper closes with the fingertip point within ``grasp_radius`` [m] of the cloth."""
    near = _fingertips_near_cloth(env, grasp_radius, asset_cfg, ee_frame_cfg)
    return (_gripper_closing(env, action_name) & near).float()


def cloth_pinched(
    env: ManagerBasedRLEnv,
    grasp_radius: float,
    min_finger_opening: float,
    action_name: str = "gripper_action",
    finger_joint_name: str = "panda_finger_joint1",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Return one when the gripper closes at the cloth and the cloth holds the finger open by more than
    ``min_finger_opening`` [m]."""
    return _cloth_grasped(
        env, grasp_radius, min_finger_opening, action_name, finger_joint_name, robot_cfg, asset_cfg, ee_frame_cfg
    ).float()


def gripper_closed_away_from_cloth(
    env: ManagerBasedRLEnv,
    grasp_radius: float,
    action_name: str = "gripper_action",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Return one when the gripper closes with the fingertip point farther than ``grasp_radius`` [m] from the
    cloth."""
    near = _fingertips_near_cloth(env, grasp_radius, asset_cfg, ee_frame_cfg)
    return (_gripper_closing(env, action_name) & ~near).float()


class ClothHeldAtTarget(ManagerTermBase):
    """Reward every step once the cloth has been held at the target, and log the episode success rate.

    The cloth counts as held when its center of mass has stayed inside ``success_radius`` [m] of the target in the
    table plane, slower than ``max_speed`` [m/s], for ``hold_time`` [s] in a row.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._hold_steps = round(cfg.params["hold_time"] / env.step_dt)
        self._steps_held = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._env.extras.setdefault("log", {})["Metrics/success_rate"] = self._succeeded[env_ids].float().mean().item()
        self._steps_held[env_ids] = 0
        self._succeeded[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        success_radius: float,
        max_speed: float,
        hold_time: float,
        command_name: str,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    ) -> torch.Tensor:
        asset: DeformableObject = env.scene[asset_cfg.name]
        in_radius = _cloth_goal_xy_distance(env, command_name, robot_cfg, asset_cfg) < success_radius
        slow = torch.linalg.norm(asset.data.root_vel_w.torch, dim=1) < max_speed
        # count consecutive steps at the target, restarting whenever the cloth leaves or moves too fast
        self._steps_held = torch.where(in_radius & slow, self._steps_held + 1, 0)
        held = self._steps_held >= self._hold_steps
        self._succeeded |= held
        return held.float()


def cloth_lifted(
    env: ManagerBasedRLEnv, max_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable")
) -> torch.Tensor:
    """Return one when the cloth's center of mass is higher than ``max_height`` [m] above the environment origin."""
    asset: DeformableObject = env.scene[asset_cfg.name]
    return (asset.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2] > max_height).float()

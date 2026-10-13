# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the Franka cloth dragging task.

Positions are expressed in the robot's root frame (suffix ``_b``) or in the cloth COM frame (suffix ``_com``), whose
origin is the cloth's center of mass and whose axes are those of the robot's root frame.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .frames import (
    cloth_com_pos_b,
    ee_pose_in_base,
    ee_twist_in_base,
    points_to_base,
    points_to_com,
    top_down_yaw,
    vectors_to_base,
)
from .stages import ClothDragStage, cloth_drag_stage_index

if TYPE_CHECKING:
    from isaaclab.assets import DeformableObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg


def ee_pose_b(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Fingertip position [m] and yaw [rad] of the end effector in the robot's root frame, shape [num_envs, 4]."""
    pos_b, quat_b = ee_pose_in_base(env, ee_frame_cfg, robot_cfg)
    return torch.cat([pos_b, top_down_yaw(quat_b).unsqueeze(-1)], dim=-1)


def ee_velocity_b(
    env: ManagerBasedRLEnv, hand_cfg: SceneEntityCfg, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    """Fingertip linear velocity [m/s] and yaw rate [rad/s] of the end effector in the robot's root frame, shape
    [num_envs, 4].

    The yaw rate is the hand's angular velocity about the root frame z-axis, the rate of the yaw in :func:`ee_pose_b`
    while the hand points down.

    Args:
        env: The environment.
        hand_cfg: The robot, with ``body_names`` selecting the hand body that carries the end-effector frame.
        ee_frame_cfg: The frame transformer whose first target is the fingertip point.
    """
    lin_vel_b, ang_vel_b = ee_twist_in_base(env, hand_cfg, ee_frame_cfg)
    return torch.cat([lin_vel_b, ang_vel_b[:, 2:3]], dim=-1)


class ClothSampledPointsInComFrame(ManagerTermBase):
    """Randomly sampled cloth nodes in the cloth COM frame [m], shape [num_envs, num_points * 3].

    The term observes a fixed number of distinct nodes, so the observation size does not depend on the mesh resolution.
    The node ids are sorted ascending, so each slot follows the mesh layout instead of a random order. Each environment
    observes the same nodes for a whole episode and samples new ones when it resets. The sampling generator is seeded
    with the environment seed (the training seed), so two terms, for example in the actor and critic groups, observe
    the same nodes.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset: DeformableObject = env.scene[cfg.params.get("asset_cfg", SceneEntityCfg("deformable")).name]
        self.num_nodes: int = asset.data.nodal_pos_w.shape[1]
        self.num_points: int = cfg.params.get("num_points", 20)
        if not 1 <= self.num_points <= self.num_nodes:
            raise ValueError(f"Number of observed cloth nodes must be in [1, {self.num_nodes}], got {self.num_points}.")
        self.node_ids = torch.empty(env.num_envs, self.num_points, dtype=torch.long, device=env.device)
        self._generator = torch.Generator(device=env.device)
        self._generator.manual_seed(env.cfg.seed if env.cfg.seed is not None else torch.initial_seed())
        self.reset()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Resample the observed cloth nodes for the selected environments."""
        if env_ids is None:
            env_ids = slice(None)
            num_envs = self.num_envs
        else:
            num_envs = len(env_ids)
        scores = torch.rand((num_envs, self.num_nodes), generator=self._generator, device=self.device)
        # sorted ids keep the slot order tied to the mesh layout
        self.node_ids[env_ids] = scores.topk(self.num_points, dim=1).indices.sort(dim=1).values

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        num_points: int = 20,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Return the sampled nodes in the cloth COM frame; ``num_points`` is read once in the constructor."""
        asset: DeformableObject = env.scene[asset_cfg.name]
        points_w = asset.data.nodal_pos_w.torch.gather(1, self.node_ids.unsqueeze(-1).expand(-1, -1, 3))
        points_b = points_to_base(env, points_w, robot_cfg)
        return points_to_com(points_b, cloth_com_pos_b(env, asset_cfg, robot_cfg)).reshape(env.num_envs, -1)


def ee_pos_com(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Fingertip position of the end effector in the cloth COM frame [m], shape [num_envs, 3]."""
    pos_b, _ = ee_pose_in_base(env, ee_frame_cfg, robot_cfg)
    return points_to_com(pos_b, cloth_com_pos_b(env, asset_cfg, robot_cfg))


def target_pos_com(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Commanded target in the cloth COM frame [m], the displacement the cloth still has to make, shape
    [num_envs, 3]."""
    target_pos_b = env.command_manager.get_command(command_name)[:, :3]
    return points_to_com(target_pos_b, cloth_com_pos_b(env, asset_cfg, robot_cfg))


def cloth_com_vel_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Velocity of the cloth's center of mass (mean nodal velocity) in the robot's root frame [m/s], shape
    [num_envs, 3]."""
    asset: DeformableObject = env.scene[asset_cfg.name]
    return vectors_to_base(env, asset.data.root_vel_w.torch, robot_cfg)


def cloth_nodal_pos_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Positions of all cloth nodes, in a fixed order, in the robot's root frame [m], shape
    [num_envs, num_nodes * 3]."""
    asset: DeformableObject = env.scene[asset_cfg.name]
    return points_to_base(env, asset.data.nodal_pos_w.torch, robot_cfg).reshape(env.num_envs, -1)


def cloth_nodal_vel_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Velocities of all cloth nodes, in a fixed order, in the robot's root frame [m/s], shape
    [num_envs, num_nodes * 3]."""
    asset: DeformableObject = env.scene[asset_cfg.name]
    return vectors_to_base(env, asset.data.nodal_vel_w.torch, robot_cfg).reshape(env.num_envs, -1)


def cloth_drag_stage(
    env: ManagerBasedRLEnv,
    command_name: str,
    near_radius: float = 0.03,
    touch_distance: float = 0.008,
    goal_radius: float = 0.05,
    action_name: str = "gripper_action",
    finger_joint_name: str = "panda_finger_joint1",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Current :class:`ClothDragStage` as a one-hot vector, see :func:`cloth_drag_stage_index`.

    Shape is (num_envs, 4).
    """
    stage = cloth_drag_stage_index(
        env,
        command_name,
        near_radius,
        touch_distance,
        goal_radius,
        action_name,
        finger_joint_name,
        robot_cfg,
        asset_cfg,
        ee_frame_cfg,
    )
    return torch.nn.functional.one_hot(stage, num_classes=len(ClothDragStage)).float()

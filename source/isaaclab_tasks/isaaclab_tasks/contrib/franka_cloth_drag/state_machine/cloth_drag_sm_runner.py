# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment adapter for the cloth-drag state machine."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from ..mdp.frames import cloth_center_pos_b, ee_pose_in_base, points_to_base, top_down_yaw
from ..mdp.stages import goal_point_pos_b, gripper_holds_raised_node
from .cloth_drag_sm import ClothDragSmObservation, DragClothSm, wrap_half_turn

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feasible_yaw_step(
    desired_yaw: torch.Tensor,
    current_yaw: torch.Tensor,
    wrist_pos: torch.Tensor,
    wrist_limits: tuple[float, float],
) -> torch.Tensor:
    """Choose the shortest equivalent yaw change allowed by the wrist window [rad]."""
    base = wrap_half_turn(desired_yaw - current_yaw)
    candidates = torch.stack((base, base - math.pi, base + math.pi), dim=1)
    predicted_wrist = wrist_pos.unsqueeze(1) - candidates
    feasible = (predicted_wrist >= wrist_limits[0]) & (predicted_wrist <= wrist_limits[1])
    cost = torch.where(feasible, candidates.abs(), torch.inf)
    selected = candidates.gather(1, cost.argmin(dim=1, keepdim=True)).squeeze(1)
    return torch.where(feasible.any(dim=1), selected, base)


def limit_arm_action(actions: torch.Tensor, limit: float) -> torch.Tensor:
    """Uniformly scale normalized arm actions to an absolute component limit.

    Args:
        actions: Normalized arm actions, shape [N, action_dim].
        limit: Largest absolute action component in ``(0, 1]``.

    Returns:
        Limited actions with the same shape. Rows already inside the limit are unchanged.
    """
    if not 0.0 < limit <= 1.0:
        raise ValueError(f"limit must lie in (0, 1], got {limit}.")
    largest = actions.abs().amax(dim=1, keepdim=True)
    scale = (limit / largest.clamp(min=1.0e-9)).clamp(max=1.0)
    return actions * scale


class ClothDragSmRunner:
    """Read task state, run the controller, and produce normalized environment actions."""

    def __init__(self, env: ManagerBasedRLEnv, sm: DragClothSm):
        self.env = env
        self.sm = sm
        self.command = env.command_manager.get_term("cloth_target")
        self.gripper = env.action_manager.get_term("gripper_action")
        self.arm = env.action_manager.get_term("arm_action")
        self.cloth = env.scene[self.command.cfg.object_name]
        self.robot = env.scene[self.command.cfg.asset_name]
        self.wrist_joint_id = self.robot.find_joints("panda_joint7")[0][0]
        self._rows = torch.arange(env.num_envs, device=env.device)

        if sm.cfg.hover_height >= self.command.cfg.approach_radius:
            raise ValueError("The state-machine hover height must lie inside the task approach radius.")
        if sm.cfg.goal_tolerance > self.command.cfg.success_radius:
            raise ValueError("The state-machine goal tolerance must not exceed the task success radius.")
        for name in ("scale_trans", "scale_rot", "wrist_joint_limits"):
            if not hasattr(self.arm.cfg, name):
                raise TypeError(f"The arm action must define {name}; use the ee_delta action preset.")

        self.previous_state = sm.sm_state.clone()
        self.actions = torch.zeros(env.action_space.shape, device=env.device)
        self.actions[:, self.arm.action_dim :] = 1.0

    def reset_idx(self, env_ids: torch.Tensor):
        """Reset controller state for the selected environments."""
        self.sm.reset_idx(env_ids)

    def compute(self) -> torch.Tensor:
        """Compute the next normalized action of every environment."""
        env = self.env
        self.ee_pos, self.ee_quat = ee_pose_in_base(env)
        self.ee_yaw = top_down_yaw(self.ee_quat)
        self.center_pos = cloth_center_pos_b(env)
        self.goal_pos = goal_point_pos_b(env, self.command.cfg)
        self.nodes_b = points_to_base(env, self.cloth.data.nodal_pos_w.torch)
        self.grasp_node = self.command.grasp_node_id

        observation = ClothDragSmObservation(
            ee_pos=self.ee_pos,
            ee_yaw=self.ee_yaw,
            grasp_node_pos=self.nodes_b[self._rows, self.grasp_node],
            cloth_center_pos=self.center_pos,
            goal_pos=self.goal_pos,
            target_pos=self.command.command,
            valid_grasp=gripper_holds_raised_node(env, self.command.cfg),
            success=self.command.holding,
        )
        self.previous_state = self.sm.sm_state.clone()
        command = self.sm.compute(observation)

        self.actions[:, :3] = ((command.ee_pos - self.ee_pos) / self.arm.cfg.scale_trans).clamp(-1.0, 1.0)
        yaw_step = feasible_yaw_step(
            command.ee_yaw,
            self.ee_yaw,
            self.robot.data.joint_pos.torch[:, self.wrist_joint_id],
            self.arm.cfg.wrist_joint_limits,
        )
        self.actions[:, 3] = (yaw_step / self.arm.cfg.scale_rot).clamp(-1.0, 1.0)
        arm_actions = self.actions[:, : self.arm.action_dim]
        self.actions[:, : self.arm.action_dim] = limit_arm_action(arm_actions, self.sm.cfg.arm_action_limit)
        self.actions[:, self.arm.action_dim :] = command.gripper.unsqueeze(1)
        return self.actions

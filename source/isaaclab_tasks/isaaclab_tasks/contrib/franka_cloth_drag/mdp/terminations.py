# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the Franka cloth dragging task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, DeformableObject
    from isaaclab.envs import ManagerBasedRLEnv


def nonfinite_state(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cloth_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Terminate when a robot joint or a cloth node has a non-finite position or velocity."""

    robot: Articulation = env.scene[robot_cfg.name]
    cloth: DeformableObject = env.scene[cloth_cfg.name]
    finite = torch.isfinite(robot.data.joint_pos.torch).all(dim=1)
    finite &= torch.isfinite(robot.data.joint_vel.torch).all(dim=1)
    finite &= torch.isfinite(cloth.data.nodal_pos_w.torch).all(dim=(1, 2))
    finite &= torch.isfinite(cloth.data.nodal_vel_w.torch).all(dim=(1, 2))
    return ~finite


def cloth_drag_success(env: ManagerBasedRLEnv, command_name: str = "cloth_target") -> torch.Tensor:
    """Terminate after the central cloth node satisfies the command's target radius and consecutive hold time."""
    command = env.command_manager.get_term(command_name)
    command.update_goal_status()
    return command.holding

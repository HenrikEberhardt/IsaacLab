# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terminations for the Franka non-prehensile cloth task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, DeformableObject, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def nonfinite_state(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cloth_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Terminate environments whose robot joint state, cloth nodes, or cube position is no longer finite.

    A simulation that blew up leaves NaN or infinite values behind, which the other terminations do not catch: every
    comparison with NaN is false.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    cloth: DeformableObject = env.scene[cloth_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]
    finite = torch.isfinite(robot.data.joint_pos.torch).all(dim=1)
    finite &= torch.isfinite(robot.data.joint_vel.torch).all(dim=1)
    finite &= torch.isfinite(cloth.data.nodal_pos_w.torch).flatten(1).all(dim=1)
    finite &= torch.isfinite(cube.data.root_pos_w.torch).all(dim=1)
    return ~finite

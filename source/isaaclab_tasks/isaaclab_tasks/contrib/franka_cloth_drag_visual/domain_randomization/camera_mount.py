# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# follows srl-ethz/IsaacLab-Cloth (cloth-fold-environments), src/Cloth_Manipulation/visual/mdp.py

"""Per-episode perturbation of the wrist camera mount (calibration error)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_camera_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice,
    sensor_cfg: SceneEntityCfg,
    parent_cfg: SceneEntityCfg,
    pos_range: float,
    rot_range: float,
):
    """Shift the camera mount by ±pos_range [m] in the parent frame and turn it by ±rot_range [rad] per axis."""
    env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    camera, parent = env.scene.sensors[sensor_cfg.name], env.scene[parent_cfg.name]
    offset, n = camera.cfg.offset, len(env_ids)
    pos = torch.tensor(offset.pos, device=env.device) + math_utils.sample_uniform(
        -pos_range, pos_range, (n, 3), env.device
    )
    euler = math_utils.sample_uniform(-rot_range, rot_range, (n, 3), env.device)
    rot = math_utils.quat_mul(
        torch.tensor(offset.rot, device=env.device).expand(n, 4),
        math_utils.quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2]),
    )
    body = parent_cfg.body_ids[0]
    world_pos, world_rot = math_utils.combine_frame_transforms(
        parent.data.body_pos_w.torch[env_ids, body], parent.data.body_quat_w.torch[env_ids, body], pos, rot
    )
    camera.set_world_poses(world_pos, world_rot, env_ids, convention=offset.convention)

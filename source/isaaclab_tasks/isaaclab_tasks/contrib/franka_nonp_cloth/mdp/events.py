# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events for the Franka non-prehensile cloth task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.assets import DeformableObject, RigidObject
    from isaaclab.envs import ManagerBasedEnv


def reset_deformable_with_object(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: dict[str, tuple[float, float]],
    object_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> None:
    """Reset a deformable object and a rigid object resting on it by one shared random offset.

    => box is placed with same offset on the cloth

    Args:
        env: The environment instance.
        env_ids: The environment indices to reset.
        position_range: Shared displacement bounds [m] keyed by ``x``, ``y``, ``z``. Missing keys => defaults to
            no displacement.
        object_cfg: Scene entity of the rigid object that moves with the deformable.
        asset_cfg: Scene entity of the deformable object to reset.
    """
    deformable: DeformableObject = env.scene[asset_cfg.name]
    rigid_object: RigidObject = env.scene[object_cfg.name]

    ranges = torch.tensor([position_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z")], device=deformable.device)
    offset = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=deformable.device)

    nodal_state = deformable.data.default_nodal_state_w.torch[env_ids].clone()
    nodal_state[..., :3] += offset.unsqueeze(1)
    deformable.write_nodal_state_to_sim_index(nodal_state, env_ids=env_ids)

    root_pose = rigid_object.data.default_root_pose.torch[env_ids].clone()
    root_pose[:, :3] += env.scene.env_origins[env_ids] + offset
    rigid_object.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
    rigid_object.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros_like(rigid_object.data.default_root_vel.torch[env_ids]), env_ids=env_ids
    )

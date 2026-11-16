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
    """Terminate after the goal point (central cloth node or object on the cloth) satisfies the command's target radius
    and consecutive hold time."""
    command = env.command_manager.get_term(command_name)
    command.update_goal_status()
    return command.holding


def physics_contact(env: ManagerBasedRLEnv, contact_name: str) -> torch.Tensor:
    """Terminate when the monitored contact pair ``contact_name`` touched during the last step.

    The physics manager latches the pair after every physics substep (see
    :meth:`~..constrained_gripper.coupler.NewtonClothGraspCouplerManager.contact_latch`), so a touch within a step is
    not missed. This term reads the latch and clears it, so it must be the only reader of that pair.
    """
    physics = env.sim.physics_manager
    touched = physics.contact_latch(contact_name).clone()
    physics.clear_contact_latch(contact_name)
    return touched


def bank_start_advanced(
    env: ManagerBasedRLEnv,
    start_stages: tuple[str, ...] = ("grasp", "drag"),
    steps_after_entry: int = 0,
    command_name: str = "cloth_target",
) -> torch.Tensor:
    """End an episode that started in one of ``start_stages`` of a reset bank once it reaches the next stage.

    Use as a truncation (``time_out=True``) so the value of the reached stage is bootstrapped: bank starts become short,
    stage-local episodes and the critic chains the stages. Approach and goal starts are never ended by it.

    The time-out bootstrap uses the value of the state before the last action. With ``steps_after_entry`` zero that
    state is the one before the stage entry, whose value already anticipates the reward paid for the entry; one or more
    steps end the episode only that many steps after the entry, so the bootstrapped state lies past it.

    Args:
        env: The environment.
        start_stages: Bank start stages whose episodes are ended, out of ``"grasp"`` and ``"drag"``.
        steps_after_entry: Environment steps the episode keeps running in the reached stage before it ends.
        command_name: Name of the cloth target command.
    """
    from .stages import START_STAGE_NAMES, ClothDragStage

    command = env.command_manager.get_term(command_name)
    start = command.start_stage
    bank_start = torch.zeros_like(start, dtype=torch.bool)
    for name in start_stages:
        stage_id = START_STAGE_NAMES.index(name)
        if stage_id not in (int(ClothDragStage.GRASP), int(ClothDragStage.DRAG)):
            raise ValueError(f"start_stages must be out of 'grasp' and 'drag', got {start_stages}.")
        bank_start |= start == stage_id
    advanced = bank_start & (command.stage > start)
    if steps_after_entry > 0:
        advanced &= command.time_in_stage >= (steps_after_entry - 0.5) * env.step_dt
    return advanced

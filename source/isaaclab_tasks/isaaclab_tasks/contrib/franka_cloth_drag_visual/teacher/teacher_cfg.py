# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teacher checkpoint pins: run 2026-11-22_18-53-59_gentle_contact_adaptive, commit 1fb4d0c93 + adaptive resets."""

from __future__ import annotations

from isaaclab.controllers import DifferentialIKControllerCfg

from isaaclab_tasks.contrib.franka_cloth_drag.config.franka.agents.rsl_rl_ppo_cfg import FrankaClothDragPPORunnerCfg
from isaaclab_tasks.contrib.franka_cloth_drag.config.franka.franka_cloth_drag_env_cfg import ActionsCfg, CurriculumCfg

from .ik_action import TeacherTopDownIKActionCfg

TEACHER_CHECKPOINT = "logs/rsl_rl/cloth_drag_task/2026-11-23_16-10-18_slow_curriculum/model_749.pt"
"""Teacher PPO checkpoint, relative to the Isaac Lab root (model_999 = end of the 1000-iteration run)."""

TEACHER_CURRICULUM_PROGRESS = 1.0
"""AvoidTableCollisionCurriculum progress p at the checkpoint (tests/read_teacher_curriculum.py)."""

TEACHER_YAW_RAMP = True
"""Whether the checkpoint was trained with the curriculum's yaw-step ramp (slow_curriculum model_749: yes, 0.05 rad)."""

_CURRICULUM = CurriculumCfg().avoid_table_collision.params
_TASK_ARM = ActionsCfg().arm_action


def _ramp(start: float, final: float) -> float:
    """Curriculum value at the pinned progress."""
    return start + TEACHER_CURRICULUM_PROGRESS * (final - start)


TEACHER_SCALE_Z = _ramp(_TASK_ARM.scale_trans[2], _CURRICULUM["final_values"]["scale_z"])  # [m] curriculum z step
TEACHER_SCALE_ROT = (
    _ramp(_TASK_ARM.scale_rot, _CURRICULUM["final_values"]["scale_rot"]) if TEACHER_YAW_RAMP else _TASK_ARM.scale_rot
)  # [rad] curriculum yaw step

TEACHER_ARM_CFG = TeacherTopDownIKActionCfg(
    asset_name="robot",
    joint_names=["panda_joint.*"],
    controller=DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=True, ik_method="dls", ik_params={"lambda_val": 0.1}
    ),
    scale_trans=(0.015, 0.015, TEACHER_SCALE_Z),  # [m]
    scale_rot=TEACHER_SCALE_ROT,  # [rad]
    z_min=0.0193,  # [m]
    joint_vel_factor=None,
    joint_range_margin=None,
    command_delay_range=(0, 4),  # physics steps
)
"""The teacher run's arm action (params/env.yaml) at the pinned curriculum progress."""

TEACHER_ACTOR_CFG = FrankaClothDragPPORunnerCfg().actor
"""The teacher's actor network; a mismatch fails the strict checkpoint load."""

TEACHER_NUM_ACTIONS = 5
"""Teacher actions (dx, dy, dz, dyaw, gripper)."""


def apply_final_curriculum_values(env_cfg) -> None:
    """Set the curriculum's final safety stops and penalties: distillation runs under them from the start."""
    final = _CURRICULUM["final_values"]
    env_cfg.terminations.gripper_force_limit.params["max_force"] = _CURRICULUM["termination_force"][1]
    # joint speed stop, and the penalty band moved with it as the curriculum does
    speed = env_cfg.terminations.joint_velocity_limit.params
    band = env_cfg.rewards.pen_joint_speed.params
    band_ratio = band["min_fraction"] / band["max_fraction"]
    speed["max_fraction"] = final["joint_speed_fraction"]
    band["min_fraction"], band["max_fraction"] = band_ratio * speed["max_fraction"], speed["max_fraction"]
    # reward weights, only logged in distillation
    env_cfg.rewards.pen_force_over_10.weight = final["pen_force_over_10"]
    for name, weight in _CURRICULUM.get("gate_values", {}).items():
        getattr(env_cfg.rewards, name).weight = weight

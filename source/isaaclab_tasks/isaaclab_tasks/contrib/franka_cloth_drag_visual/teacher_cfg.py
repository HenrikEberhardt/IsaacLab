# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Everything the distillation needs to know about the teacher checkpoint.

The teacher is trained in ``IsaacContrib-Drag-Cloth-Franka``, whose configuration keeps changing. A checkpoint only
works with the arm action, actor network and MDP it was trained with, so these are pinned here. Retraining the
teacher means updating this module, and nothing else.

Current teacher: run ``2026-11-21_15-56-41_one_node_dr``, ``model_750.pt``, commit 2918b254f (clean tree).
"""

from __future__ import annotations

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

from isaaclab_tasks.contrib.franka_cloth_drag import mdp as drag_mdp
from isaaclab_tasks.contrib.franka_cloth_drag.config.franka.agents.rsl_rl_ppo_cfg import FrankaClothDragPPORunnerCfg

from .teacher.ik_action import TeacherTopDownIKActionCfg
from .teacher.ik_label import IKLabelCfg

TEACHER_CHECKPOINT = "logs/rsl_rl/cloth_drag_task/2026-11-21_15-56-41_one_node_dr/model_750.pt"
"""Teacher checkpoint, relative to the Isaac Lab root. Pass it to ``isaaclab train --checkpoint``."""

STUDENT_JOINT_DELTA_SCALE = 0.05
"""Joint-delta scale of the student's arm action [rad], the one of the joint-delta deploy stack."""

TEACHER_ARM_CFG = TeacherTopDownIKActionCfg(
    asset_name="robot",
    joint_names=["panda_joint.*"],
    controller=DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=True, ik_method="dls", ik_params={"lambda_val": 0.1}
    ),
    scale_trans=(0.015, 0.015, 0.015),  # [m]
    scale_rot=0.2,  # [rad]
    limit_operation_space=True,
    x_limits=(0.30, 0.80),  # [m]
    y_limits=(-0.40, 0.40),  # [m]
    z_min=0.0193,  # [m]
    joint_vel_factor=0.9,
    joint_range_margin=0.05,  # [rad]
    command_delay_range=(0, 4),  # physics steps
)
"""The teacher run's arm action (``params/env.yaml`` of the run)."""

IK_LABEL_CFG = IKLabelCfg(
    scale_trans=TEACHER_ARM_CFG.scale_trans,
    scale_rot=TEACHER_ARM_CFG.scale_rot,
    wrist_joint_limits=TEACHER_ARM_CFG.wrist_joint_limits,
    limit_operation_space=TEACHER_ARM_CFG.limit_operation_space,
    x_limits=TEACHER_ARM_CFG.x_limits,
    y_limits=TEACHER_ARM_CFG.y_limits,
    z_min=TEACHER_ARM_CFG.z_min,
    ik_lambda=TEACHER_ARM_CFG.controller.ik_params["lambda_val"],
    joint_delta_scale=STUDENT_JOINT_DELTA_SCALE,
    # Clipping each joint at 0.05 rad bends the motion: teacher through its labels 84.9 % vs 99.5 % native. Uniform
    # scaling keeps the direction: 96.7 %, at 70 % of the teacher's EE speed (teacher_joint_replay.py, 2026-11-22).
    preserve_direction=True,
)
"""Conversion of the teacher's actions into the student's joint-delta labels."""

TEACHER_ACTOR_CFG = FrankaClothDragPPORunnerCfg().actor
"""The teacher's actor network (``params/agent.yaml`` of the run); a mismatch fails the strict checkpoint load."""

TEACHER_NUM_ACTIONS = 5
"""Teacher actions ``(dx, dy, dz, dyaw, gripper)``."""


def apply_teacher_terminations(env_cfg) -> None:
    """Restore the terminations the teacher was trained with where the task has changed them since.

    The teacher run ended episodes on the end-effector 10 cm below the table and on joint speeds above the
    simulation limits, and had no joint-range termination. Delete this once a teacher is trained with the current task.
    """
    terminations = env_cfg.terminations
    terminations.ee_below_table = DoneTerm(
        func=drag_mdp.ee_below_minimum, params={"minimum_height": -0.1, "ee_frame_cfg": SceneEntityCfg("ee_frame")}
    )
    terminations.joint_velocity_limit = DoneTerm(
        func=drag_mdp.joint_vel_out_of_sim_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
    )
    terminations.joint_range_limit = None
    term_keys = env_cfg.rewards.pen_termination.params["term_keys"]
    env_cfg.rewards.pen_termination.params["term_keys"] = [key for key in term_keys if key != "joint_range_limit"]

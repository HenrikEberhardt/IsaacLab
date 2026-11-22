# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms of the distillation env: the teacher's IK state and its last action, and the wrist image."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase

from ..teacher.ik_action import TeacherTopDownIKAction, TeacherTopDownIKActionCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import ObservationTermCfg


class ik_state(ManagerTermBase):
    """Inputs of the label conversion, shape [N, IK_STATE_DIM] (see :meth:`TeacherTopDownIKAction.ik_state`).

    Evaluated by a private copy of the teacher's arm action term, so the EE frame, its Jacobian and the joint bounds
    are exactly those of the teacher.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._teacher_arm = TeacherTopDownIKAction(cfg.params["teacher_arm_cfg"], env)

    def __call__(self, env: ManagerBasedEnv, teacher_arm_cfg: TeacherTopDownIKActionCfg) -> torch.Tensor:
        return self._teacher_arm.ik_state()


class teacher_last_action(ManagerTermBase):
    """The executed action expressed in the teacher's action space, shape [N, 5].

    The teacher observes its own previous raw action ``(dx, dy, dz, dyaw, gripper)``. In the distillation env the
    student acts, so the term maps the student's commanded joint offset through the teacher's EE Jacobian into an EE
    displacement and divides it by the teacher's scales; the gripper part is the gripper term's raw action. Both are
    zero after a reset (the gripper part as restored by the reset), as for the teacher.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        teacher_arm_cfg: TeacherTopDownIKActionCfg = cfg.params["teacher_arm_cfg"]
        self._teacher_arm = TeacherTopDownIKAction(teacher_arm_cfg, env)
        self._scale = torch.tensor((*teacher_arm_cfg.scale_trans, teacher_arm_cfg.scale_rot), device=env.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        teacher_arm_cfg: TeacherTopDownIKActionCfg,
        arm_action_name: str = "arm_action",
        gripper_action_name: str = "gripper_action",
    ) -> torch.Tensor:
        joint_delta = env.action_manager.get_term(arm_action_name).processed_actions
        twist = (self._teacher_arm.frame_jacobian() @ joint_delta.unsqueeze(-1)).squeeze(-1)
        arm = torch.cat((twist[:, :3], twist[:, 5:6]), dim=-1) / self._scale
        gripper = env.action_manager.get_term(gripper_action_name).raw_actions
        return torch.cat((arm, gripper), dim=-1)

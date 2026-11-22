# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DAgger distillation from the IK teacher into a joint-delta student."""

from __future__ import annotations

from dataclasses import MISSING

import torch
from rsl_rl.algorithms import Distillation
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_class, resolve_obs_groups
from tensordict import TensorDict

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlDistillationAlgorithmCfg

from ..teacher.ik_label import IKLabelCfg, teacher_actions_to_joint_deltas


class IKTeacherDistillation(Distillation):
    """:class:`~rsl_rl.algorithms.Distillation` with a teacher in another action space.

    The student acts in the environment's action space (7 joint deltas and the gripper). The teacher keeps its own
    action space (``teacher_num_actions`` outputs, the IK arm action and the gripper): its deterministic action is
    converted into the student's action space with :func:`teacher_actions_to_joint_deltas` on the ``ik_state``
    observation group, and that label is stored as the privileged action of the MSE loss.
    """

    def __init__(self, *args, label_cfg: dict, ik_state_group: str = "ik_state", **kwargs):
        super().__init__(*args, **kwargs)
        self.label_cfg = IKLabelCfg(**label_cfg)
        self.ik_state_group = ik_state_group
        self._label_stats = torch.zeros(4, device=self.device)

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample the student's action, and store the teacher's action converted to the student's action space."""
        self.transition.actions = self.student(obs, stochastic_output=True).detach()
        teacher_actions = self.teacher(obs).detach()
        labels, clipped = teacher_actions_to_joint_deltas(teacher_actions, obs[self.ik_state_group], self.label_cfg)
        self.transition.privileged_actions = labels
        self.transition.observations = obs
        # label statistics: clipped arm entries, steps, and the squared error of the student's mean on arm and gripper
        error = (self.student.output_mean.detach() - labels).square()
        self._label_stats += torch.stack((
            clipped.float().mean(),
            torch.ones((), device=self.device),
            error[:, :-1].mean(),
            error[:, -1].mean(),
        ))
        return self.transition.actions  # type: ignore

    def update(self) -> dict[str, float]:
        """Run the distillation update and add the label statistics of the rollout."""
        loss_dict = super().update()
        clipped, steps, arm_error, gripper_error = (self._label_stats / self._label_stats[1].clamp(min=1.0)).tolist()
        self._label_stats.zero_()
        loss_dict.update({
            "label_clip_fraction": clipped,
            "rollout_arm_mse": arm_error,
            "rollout_gripper_mse": gripper_error,
        })
        return loss_dict

    @staticmethod
    def construct_algorithm(obs: TensorDict, env, cfg: dict, device: str) -> IKTeacherDistillation:
        """Construct the algorithm with a teacher of ``algorithm.teacher_num_actions`` outputs."""
        alg_class, alg_cfg = resolve_class(cfg["algorithm"])
        student_class, student_cfg = resolve_class(cfg["student"])
        teacher_class, teacher_cfg = resolve_class(cfg["teacher"])
        teacher_num_actions = alg_cfg.pop("teacher_num_actions")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["student", "teacher"])
        if alg_cfg.get("rnd_cfg") is not None or alg_cfg.get("symmetry_cfg") is not None:
            raise ValueError("The RND and symmetry extensions are not compatible with distillation.")
        alg_cfg["rnd_cfg"] = alg_cfg["symmetry_cfg"] = None

        student = student_class(obs, cfg["obs_groups"], "student", env.num_actions, **student_cfg).to(device)
        print(f"Student Model: {student}")
        teacher = teacher_class(obs, cfg["obs_groups"], "teacher", teacher_num_actions, **teacher_cfg).to(device)
        print(f"Teacher Model: {teacher}")
        storage = RolloutStorage("distillation", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        alg = alg_class(student, teacher, storage, device=device, **alg_cfg, multi_gpu_cfg=cfg["multi_gpu"])
        alg.compile(cfg.get("torch_compile_mode"))
        return alg


@configclass
class IKTeacherDistillationAlgorithmCfg(RslRlDistillationAlgorithmCfg):
    """Configuration of :class:`IKTeacherDistillation`."""

    class_name: str = f"{__name__}:IKTeacherDistillation"

    teacher_num_actions: int = MISSING
    """Number of teacher actions."""

    label_cfg: IKLabelCfg = MISSING
    """Conversion of the teacher's actions into the student's action space."""

    ik_state_group: str = "ik_state"
    """Observation group with the conversion's inputs."""

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL DAgger distillation whose teacher labels pass through an action mapping."""

from __future__ import annotations

from dataclasses import MISSING

import torch
from rsl_rl.algorithms import Distillation
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_class, resolve_obs_groups
from tensordict import TensorDict

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlDistillationAlgorithmCfg

from .base import ActionMappingCfg


class MappedDistillation(Distillation):
    """Distillation with a teacher in its own action space; identity mapping = stock behaviour."""

    def __init__(self, *args, mapping_cfg: dict, **kwargs):
        super().__init__(*args, **kwargs)
        mapping_class, mapping_kwargs = resolve_class(mapping_cfg)
        self.mapping = mapping_class(**mapping_kwargs)
        self._stats: dict[str, torch.Tensor] = {}
        self._num_steps = 0

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample the student's action and store the mapped teacher label."""
        self.transition.actions = self.student(obs, stochastic_output=True).detach()
        labels, stats = self.mapping(self.teacher(obs).detach(), obs)
        self.transition.privileged_actions = labels
        self.transition.observations = obs
        error = (self.student.output_mean.detach() - labels).square()
        stats = {**stats, "rollout_arm_mse": error[:, :-1].mean(), "rollout_gripper_mse": error[:, -1].mean()}
        for key, value in stats.items():
            self._stats[key] = self._stats.get(key, 0.0) + value
        self._num_steps += 1
        return self.transition.actions  # type: ignore

    def update(self) -> dict[str, float]:
        """Distillation update plus the mean label statistics of the rollout."""
        loss_dict = super().update()
        loss_dict.update({key: float(value) / max(self._num_steps, 1) for key, value in self._stats.items()})
        self._stats, self._num_steps = {}, 0
        return loss_dict

    @staticmethod
    def construct_algorithm(obs: TensorDict, env, cfg: dict, device: str) -> MappedDistillation:
        """Build the algorithm with a teacher of ``mapping_cfg.teacher_num_actions`` outputs."""
        alg_class, alg_cfg = resolve_class(cfg["algorithm"])
        student_class, student_cfg = resolve_class(cfg["student"])
        teacher_class, teacher_cfg = resolve_class(cfg["teacher"])
        mapping_class = resolve_callable(alg_cfg["mapping_cfg"]["class_name"])
        alg_cfg["mapping_cfg"] = mapping_class.resolve_cfg(alg_cfg["mapping_cfg"], env.unwrapped)
        teacher_num_actions = alg_cfg["mapping_cfg"].get("teacher_num_actions") or env.num_actions
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
class MappedDistillationAlgorithmCfg(RslRlDistillationAlgorithmCfg):
    """Configuration of :class:`MappedDistillation`."""

    class_name: str = f"{__name__}:MappedDistillation"

    mapping_cfg: ActionMappingCfg = MISSING
    """Teacher-to-student action mapping."""

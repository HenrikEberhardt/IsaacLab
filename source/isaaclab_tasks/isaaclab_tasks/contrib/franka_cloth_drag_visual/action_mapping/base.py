# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interface between the teacher's and the student's action spaces."""

from __future__ import annotations

from dataclasses import MISSING

import torch
from tensordict import TensorDict

from isaaclab.utils import configclass


@configclass
class ActionMappingCfg:
    """Base configuration of an :class:`ActionMapping`."""

    class_name: str = MISSING
    """Mapping class as ``"module:Class"``, resolved by the distillation algorithm."""

    teacher_num_actions: int | None = None
    """Teacher outputs; None for the student's action count."""

    def apply_to_env(self, env_cfg, teacher_arm_cfg) -> None:
        """Add the observations the mapping needs to the env config."""
        pass


class ActionMapping:
    """Converts the teacher's deterministic actions into labels in the student's action space."""

    cfg_type: type[ActionMappingCfg] = ActionMappingCfg

    def __init__(self, **cfg):
        self.cfg = self.cfg_type(**cfg)

    @staticmethod
    def resolve_cfg(cfg: dict, env) -> dict:
        """Fill settings that come from the env (e.g. the student's action scale) before construction."""
        return cfg

    def __call__(self, teacher_actions: torch.Tensor, obs: TensorDict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Labels of shape [N, student actions] and scalar statistics to log."""
        raise NotImplementedError

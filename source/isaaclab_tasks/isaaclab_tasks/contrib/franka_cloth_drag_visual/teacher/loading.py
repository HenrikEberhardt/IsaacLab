# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load the teacher actor outside the distillation runner (evaluation and inspection scripts)."""

from __future__ import annotations

import importlib.metadata as metadata

import torch
from tensordict import TensorDict


def load_teacher(obs: TensorDict, checkpoint: str, device: str, group: str = "teacher") -> torch.nn.Module:
    """The teacher actor as the distillation runner builds it, loaded strictly from a PPO checkpoint.

    Args:
        obs: Observations of the distillation env, with the teacher's group.
        checkpoint: Path of the teacher's PPO checkpoint.
        device: Torch device of the model.
        group: Observation group of the teacher.

    Returns:
        The teacher in eval mode; calling it returns its deterministic actions, shape [N, 5].
    """
    from rsl_rl.utils import resolve_class

    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

    from ..agents.rsl_rl_distillation_cfg import FrankaClothDragDistillationRunnerCfg
    from ..teacher_cfg import TEACHER_NUM_ACTIONS

    agent_cfg = handle_deprecated_rsl_rl_cfg(FrankaClothDragDistillationRunnerCfg(), metadata.version("rsl-rl-lib"))
    model_class, model_cfg = resolve_class(agent_cfg.to_dict()["teacher"])
    teacher = model_class(obs, {"teacher": [group]}, "teacher", TEACHER_NUM_ACTIONS, **model_cfg).to(device)
    teacher.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["actor_state_dict"])
    return teacher.eval()

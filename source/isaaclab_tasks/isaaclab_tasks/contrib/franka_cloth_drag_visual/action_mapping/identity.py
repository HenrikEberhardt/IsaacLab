# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Identity mapping: teacher and student share the action space and controller."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from isaaclab.utils import configclass

from .base import ActionMapping, ActionMappingCfg


@configclass
class IdentityMappingCfg(ActionMappingCfg):
    """Configuration of :class:`IdentityMapping`."""

    class_name: str = f"{__name__}:IdentityMapping"


class IdentityMapping(ActionMapping):
    """The teacher's action is the label."""

    cfg_type = IdentityMappingCfg

    def __call__(self, teacher_actions: torch.Tensor, obs: TensorDict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return teacher_actions, {}

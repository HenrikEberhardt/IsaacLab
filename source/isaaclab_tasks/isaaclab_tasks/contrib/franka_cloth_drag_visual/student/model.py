# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Student network: RSL-RL CNNModel on uint8 image histories."""

from __future__ import annotations

import torch
from rsl_rl.models import CNNModel
from tensordict import TensorDict
from torch import nn


class ToUnitRange(nn.Module):
    """Image input: a [B, T, C, H, W] history folded into [B, T * C, H, W] (oldest frame first), 0-255 to [0, 1]."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            x = x.flatten(1, 2)
        return x.float() / 255.0


class Uint8ImageCNNModel(CNNModel):
    """CNNModel on 0-255 image histories, so storage and exports stay in camera units.

    An image group may be a single frame [B, C, H, W] or a history [B, T, C, H, W]; a history enters the default CNN
    as T * C channels.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for group in self.obs_groups_2d:
            self.cnns[group] = nn.Sequential(ToUnitRange(), self.cnns[group])

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Size the CNNs as if each image history were already folded into channels."""
        folded = {
            group: obs[group].flatten(1, 2) if obs[group].dim() == 5 else obs[group] for group in obs_groups[obs_set]
        }
        return super()._get_obs_dim(folded, obs_groups, obs_set)

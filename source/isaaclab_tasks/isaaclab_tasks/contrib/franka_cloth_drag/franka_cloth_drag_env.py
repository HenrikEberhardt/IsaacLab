# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment class for the Franka cloth dragging task."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn


def _nan_to_zero(value: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


class FrankaClothDragEnv(ManagerBasedRLEnv):
    """Manager-based RL environment for the Franka cloth dragging task.

    The coupled cloth simulation can blow up into NaN. The ``nonfinite_state`` termination then resets the env, but
    the step's reward is computed on the broken state, and some observations after the reset still read it, since the
    end-effector frame is only recomputed by the next simulation step. This class replaces non-finite rewards and
    observations with zeros, so one broken env does not stop training of all others. The number of affected envs is
    logged as ``Metrics/nonfinite_envs``.

    Adapted from ``FrankaNonpClothEnv`` on the ``non-prehensile-cloth`` branch.
    """

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        obs, rew, terminated, truncated, extras = super().step(action)
        broken = ~torch.isfinite(rew)
        for value in obs.values():
            terms = [value] if isinstance(value, torch.Tensor) else value.values()
            for term_value in terms:
                broken |= ~torch.isfinite(term_value).flatten(1).all(dim=1)
        if broken.any():
            rew = _nan_to_zero(rew)
            self.reward_buf = rew
            for group, value in obs.items():
                if isinstance(value, torch.Tensor):
                    obs[group] = _nan_to_zero(value)
                else:
                    for term, term_value in value.items():
                        value[term] = _nan_to_zero(term_value)
        # set every step: the log dict is only cleared on resets, so a stale count would otherwise persist
        extras.setdefault("log", {})["Metrics/nonfinite_envs"] = broken.sum().item()
        return obs, rew, terminated, truncated, extras

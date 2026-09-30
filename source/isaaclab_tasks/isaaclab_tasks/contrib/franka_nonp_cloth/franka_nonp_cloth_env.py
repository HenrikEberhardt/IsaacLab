# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment class for the Franka non-prehensile cloth task."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn


class FrankaNonpClothEnv(ManagerBasedRLEnv):
    """Manager-based RL environment for training Non-Prehensile Cloth tasks.

    The cloth simulation can occasionally blow up. The ``nonfinite_state`` termination then resets it, but the step's reward is computed on the broken state. Further some observations after the reset still read it, since the end-effector frame is only recomputed by the next simulation step. 
    
    This class replaces the non-finite (NaN) rewards and observations values with zeros, so they never reach the learning algorithm.
    
    # It logs the number of affected environments as ``Metrics/nonfinite_envs``.
    """

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        obs, rew, terminated, truncated, extras = super().step(action)
        broken = ~torch.isfinite(rew)
        for group, value in obs.items():
            if isinstance(value, torch.Tensor):
                broken |= ~torch.isfinite(value).flatten(1).all(dim=1)
            else:
                for term_value in value.values():
                    broken |= ~torch.isfinite(term_value).flatten(1).all(dim=1)
        if broken.any():
            rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)
            self.reward_buf = rew
            for group, value in obs.items():
                if isinstance(value, torch.Tensor):
                    obs[group] = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    for term, term_value in value.items():
                        value[term] = torch.nan_to_num(term_value, nan=0.0, posinf=0.0, neginf=0.0)
            extras.setdefault("log", {})["Metrics/nonfinite_envs"] = broken.sum().item()
        return obs, rew, terminated, truncated, extras

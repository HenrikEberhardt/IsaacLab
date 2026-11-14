# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment class for the Franka cloth dragging task."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab_newton.physics import NewtonManager

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.sensors import BaseFrameTransformer


def _nan_to_zero(value: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


class FrankaClothDragEnv(ManagerBasedRLEnv):
    """Manager-based RL environment for the Franka cloth dragging task.

    The coupled cloth simulation can blow up into NaN. The ``nonfinite_state`` termination then resets the env, but
    the step's reward is computed on the broken state, and some observations after the reset still read it, since the
    end-effector frame is only recomputed by the next simulation step. This class replaces non-finite rewards and
    observations with zeros, so one broken env does not stop training of all others.

    After every reset, the class refreshes the frame transformer sensors, so they show the poses after the reset. On
    Newton, a frame transformer copies the transforms of a Newton frame-transform sensor, which Newton only updates at
    the end of a physics step: :meth:`NewtonManager.forward` refreshes the body poses after a state write, but not that
    sensor. Until the next step, the frame transformers of reset environments would keep the previous episode's poses,
    in the first observation, the first stage update and the grasp selection of the first gripper action (see
    ``implementation_log/reset_states.md``).

    Adapted from ``FrankaNonpClothEnv`` on the ``non-prehensile-cloth`` branch.
    """

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset the environments, then update the frame transformers from the reset state."""
        super()._reset_idx(env_ids)
        # private: Newton has no public way to update its sensors outside a step; empty without Newton sensors
        newton_sensors = NewtonManager._newton_frame_transform_sensors
        if not newton_sensors:
            return
        # forward kinematics of the written joint states, then the sensors from the refreshed body poses
        NewtonManager.forward()
        state = NewtonManager.get_state_0()
        for sensor in newton_sensors:
            sensor.update(state)
        # the frame transformers recompute lazily from the updated transforms at their next read
        for sensor in self.scene.sensors.values():
            if isinstance(sensor, BaseFrameTransformer):
                sensor.reset(env_ids)

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
        return obs, rew, terminated, truncated, extras

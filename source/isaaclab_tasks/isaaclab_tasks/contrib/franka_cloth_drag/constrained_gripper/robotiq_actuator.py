# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Ported from srl-ethz/IsaacLab-Cloth (branch cloth-fold-environments, commit c445370), common/gripper_actuator.py,
# which follows mmichelis/IsaacLab-Soft (branch liftsoft-robotiq, commit 1ee1ab3); see assets/robotiq/README.md.

"""Finger drive of the Robotiq 2F-85 with the command delay of the real gripper."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.actuators import ImplicitActuator, ImplicitActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class RobotiqGripperActuator(ImplicitActuator):
    """Robotiq 2F-85: the solver's PD drive tracks a delayed copy of the commanded position.

    The real gripper starts moving some time after the command. The commanded target is delayed by ``min_delay`` to
    ``max_delay`` physics steps, sampled per environment at reset. After a reset the delay line is filled with the
    measured position, so the gripper holds still until the first command comes through. The PD gains and effort
    limit stay the drive's own, so the squeeze force is unchanged; the drive's effort limit and viscous friction set
    the finger speed.
    """

    cfg: RobotiqGripperActuatorCfg

    def __init__(self, cfg: RobotiqGripperActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if not 0 <= cfg.min_delay <= cfg.max_delay:
            raise ValueError(f"Expected 0 <= min_delay <= max_delay, got {cfg.min_delay}, {cfg.max_delay}")
        self._history = torch.zeros(cfg.max_delay + 1, self._num_envs, self.num_joints, device=self._device)
        self._write_index = 0
        self._lag = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        self._fresh = torch.ones(self._num_envs, dtype=torch.bool, device=self._device)
        self._env_index = torch.arange(self._num_envs, device=self._device)
        self.reset(None)

    def reset(self, env_ids: Sequence[int] | None = None):
        ids = slice(None) if env_ids is None else env_ids
        num = self._lag[ids].shape[0]
        self._lag[ids] = torch.randint(self.cfg.min_delay, self.cfg.max_delay + 1, (num,), device=self._device)
        self._fresh[ids] = True

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # after a reset, the delay line starts from the measured position
        self._history = torch.where(self._fresh.view(1, -1, 1), joint_pos.unsqueeze(0), self._history)
        self._fresh[:] = False

        self._history[self._write_index] = control_action.joint_positions
        read_index = (self._write_index - self._lag) % self._history.shape[0]
        control_action.joint_positions = self._history[read_index, self._env_index].clone()
        self._write_index = (self._write_index + 1) % self._history.shape[0]
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class RobotiqGripperActuatorCfg(ImplicitActuatorCfg):
    """Configuration for :class:`RobotiqGripperActuator`."""

    class_type: type = RobotiqGripperActuator

    min_delay: int = 0
    """Minimum command delay [physics steps]."""

    max_delay: int = 0
    """Maximum command delay [physics steps]."""

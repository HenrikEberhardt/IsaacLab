# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Student arm action: relative joint position targets with command latency and dropped commands."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions import RelativeJointPositionAction, RelativeJointPositionActionCfg
from isaaclab.utils import configclass
from isaaclab.utils.buffers import DelayBuffer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DelayedRelativeJointPositionAction(RelativeJointPositionAction):
    """Relative joint position targets ``q + scale * a``, delayed and occasionally dropped like the real commands.

    * **Latency.** The commanded offset reaches the joints after a per-environment delay of
      :attr:`~DelayedRelativeJointPositionActionCfg.command_delay_range` physics steps, drawn at reset, like the
      teacher's IK target delay.
    * **Dropped commands.** With probability :attr:`~DelayedRelativeJointPositionActionCfg.hold_prob` per policy step,
      an environment ignores the new action and keeps the previous offset.
    * **Export.** The current joint positions are read with int64 indices: the base term's int32 indices export an
      int32 ``GatherND``, which ONNX does not allow (from Joran's ``RelativeJointPositionInt64Action``).
    """

    cfg: DelayedRelativeJointPositionActionCfg

    def __init__(self, cfg: DelayedRelativeJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        ids = self._joint_ids
        self._read_ids = ids.long() if isinstance(ids, torch.Tensor) else ids
        min_delay, max_delay = cfg.command_delay_range
        if not 0 <= min_delay <= max_delay:
            raise ValueError(f"Expected 0 <= min <= max in command_delay_range, got {cfg.command_delay_range}.")
        if not 0.0 <= cfg.hold_prob < 1.0:
            raise ValueError(f"Expected hold_prob in [0, 1), got {cfg.hold_prob}.")
        self._delay = DelayBuffer(max_delay, self.num_envs, self.device) if max_delay > 0 else None

    @property
    def command_delay(self) -> torch.Tensor:
        """Delay of the joint target offset in physics steps, shape [num_envs]."""
        if self._delay is None:
            return torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        return self._delay.time_lags

    def process_actions(self, actions: torch.Tensor):
        """Scale the actions; dropped environments keep their previous offset."""
        previous = self._processed_actions.clone()
        super().process_actions(actions)
        if self.cfg.hold_prob > 0.0:
            hold = torch.rand(self.num_envs, device=self.device) < self.cfg.hold_prob
            self._processed_actions[hold] = previous[hold]

    def apply_actions(self):
        """Command the (delayed) offset relative to the current joint positions."""
        offset = self._processed_actions if self._delay is None else self._delay.compute(self._processed_actions)
        target = offset + self._asset.data.joint_pos.torch[:, self._read_ids]
        self._asset.set_joint_position_target_index(target=target, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Zero the offsets, empty the delay line and draw new delays."""
        super().reset(env_ids)
        ids = torch.arange(self.num_envs, device=self.device)[slice(None) if env_ids is None else env_ids]
        self._processed_actions[ids] = 0.0
        if self._delay is None:
            return
        # an emptied delay line returns the newest offset, so no offset of the previous episode is applied
        self._delay.reset(ids)
        min_delay, max_delay = self.cfg.command_delay_range
        lags = torch.randint(min_delay, max_delay + 1, (len(ids),), dtype=torch.int, device=self.device)
        self._delay.set_time_lag(lags, ids)


@configclass
class DelayedRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    """Configuration for :class:`DelayedRelativeJointPositionAction`."""

    class_type: type = DelayedRelativeJointPositionAction

    command_delay_range: tuple[int, int] = (0, 0)
    """Range of the delay of the joint target offset in physics steps, drawn uniformly per environment at reset.
    (0, 0) disables the delay."""

    hold_prob: float = 0.0
    """Probability per policy step and environment that the new action is ignored and the previous offset kept."""

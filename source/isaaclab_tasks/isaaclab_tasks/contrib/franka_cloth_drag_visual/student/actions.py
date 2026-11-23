# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Student action space: relative joint targets on the arm and the binary Robotiq gripper (8 actions)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions import RelativeJointPositionAction, RelativeJointPositionActionCfg
from isaaclab.utils import configclass
from isaaclab.utils.buffers import DelayBuffer

from isaaclab_tasks.contrib.franka_cloth_drag.config.franka.franka_cloth_drag_env_cfg import GRIPPER_ACTION_CFG
from isaaclab_tasks.contrib.franka_cloth_drag.constrained_gripper.gripper_action import ConstrainedGripperAction

from .export import leapp_exporting

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

STUDENT_JOINT_DELTA_SCALE = 0.05
"""Joint-delta scale of the deploy stack [rad] per action unit."""


class DelayedRelativeJointPositionAction(RelativeJointPositionAction):
    """Relative joint targets q + scale * a with per-env command latency and dropped commands (int64 export).

    Re-anchored on the current q every physics step, or with ``anchor_per_policy_step`` once per action and held.
    """

    cfg: DelayedRelativeJointPositionActionCfg

    def __init__(self, cfg: DelayedRelativeJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        ids = self._joint_ids
        self._read_ids = ids.long() if isinstance(ids, torch.Tensor) else ids  # int32 GatherND is not valid ONNX
        min_delay, max_delay = cfg.command_delay_range
        if not 0 <= min_delay <= max_delay:
            raise ValueError(f"Expected 0 <= min <= max in command_delay_range, got {cfg.command_delay_range}.")
        if not 0.0 <= cfg.hold_prob < 1.0:
            raise ValueError(f"Expected hold_prob in [0, 1), got {cfg.hold_prob}.")
        self._delay = DelayBuffer(max_delay, self.num_envs, self.device) if max_delay > 0 else None
        self._target = torch.zeros(self.num_envs, self.action_dim, device=self.device)  # anchored joint target [rad]

    @property
    def command_delay(self) -> torch.Tensor:
        """Delay of the joint target offset in physics steps, shape [N]."""
        if self._delay is None:
            return torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        return self._delay.time_lags

    def process_actions(self, actions: torch.Tensor):
        """Scale the actions; dropped envs keep their previous offset (or anchored target)."""
        previous = self._processed_actions.clone()
        super().process_actions(actions)
        if leapp_exporting():  # dropped commands are sim-only; apply_actions exports q + scale * a
            return
        hold = torch.rand(self.num_envs, device=self.device) < self.cfg.hold_prob
        self._processed_actions[hold] = previous[hold]
        if self.cfg.anchor_per_policy_step:
            target = self._processed_actions + self._asset.data.joint_pos.torch[:, self._read_ids]
            self._target = torch.where(hold[:, None], self._target, target)

    def apply_actions(self):
        """Command the (delayed) offset relative to the current q, or the (delayed) anchored target.

        A LEAPP export traces the first physics step of a policy step, so it exports q + scale * a without the
        sim-only delay and drops: the deployed target, which the robot holds until the next policy step.
        """
        if leapp_exporting():
            target = self._processed_actions + self._asset.data.joint_pos.torch[:, self._read_ids]
        elif self.cfg.anchor_per_policy_step:
            target = self._target if self._delay is None else self._delay.compute(self._target)
        else:
            offset = self._processed_actions if self._delay is None else self._delay.compute(self._processed_actions)
            target = offset + self._asset.data.joint_pos.torch[:, self._read_ids]
        self._asset.set_joint_position_target_index(target=target, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Zero the offsets, empty the delay line and draw new delays."""
        super().reset(env_ids)
        ids = torch.arange(self.num_envs, device=self.device)[slice(None) if env_ids is None else env_ids]
        self._processed_actions[ids] = 0.0
        self._target[ids] = self._asset.data.joint_pos.torch[ids][:, self._read_ids]  # a dropped first command holds
        if self._delay is None:
            return
        self._delay.reset(ids)  # an emptied line returns the newest offset
        min_delay, max_delay = self.cfg.command_delay_range
        lags = torch.randint(min_delay, max_delay + 1, (len(ids),), dtype=torch.int, device=self.device)
        self._delay.set_time_lag(lags, ids)


@configclass
class DelayedRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    """Configuration for :class:`DelayedRelativeJointPositionAction`."""

    class_type: type = DelayedRelativeJointPositionAction

    command_delay_range: tuple[int, int] = (0, 0)
    """Offset delay in physics steps, drawn per env at reset; (0, 0) disables it."""

    hold_prob: float = 0.0
    """Probability per policy step and env that the previous offset is kept."""

    anchor_per_policy_step: bool = False
    """Anchor the target on q once per action and hold it over the substeps, instead of re-anchoring every substep."""


class StudentGripperAction(ConstrainedGripperAction):
    """Constrained gripper that a LEAPP export traces as the plain binary step target on the finger joint.

    The cloth-node lock is simulation-only; the real gripper receives the open/close step and closes on the cloth.
    """

    def process_actions(self, actions: torch.Tensor):
        if leapp_exporting():
            super(ConstrainedGripperAction, self).process_actions(actions)
        else:
            super().process_actions(actions)

    def apply_actions(self):
        if leapp_exporting():
            super(ConstrainedGripperAction, self).apply_actions()
        else:
            super().apply_actions()


STUDENT_GRIPPER_ACTION_CFG = GRIPPER_ACTION_CFG.replace(class_type=StudentGripperAction)
"""The task's constrained gripper with the export path."""


@configclass
class JointDeltaActionsCfg:
    """Relative joint targets on the arm and the binary Robotiq gripper."""

    arm_action = DelayedRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=STUDENT_JOINT_DELTA_SCALE,  # [rad] per action unit
        command_delay_range=(0, 4),  # physics steps at 120 Hz, as the teacher's IK delay
        hold_prob=0.1,  # dropped command
        anchor_per_policy_step=True,  # one target per action, as the IK teacher's labels mean it
    )
    gripper_action = STUDENT_GRIPPER_ACTION_CFG

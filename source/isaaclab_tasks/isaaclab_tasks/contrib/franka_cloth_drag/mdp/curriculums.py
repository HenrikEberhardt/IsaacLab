# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms for the Franka cloth dragging task.

Each term is a block with its own success gate: it tracks the success rate of the finished episodes as an exponential
moving average and, once the average has stayed at least ``success_threshold`` for ``dwell_steps``, grows its progress
linearly with the environment steps from ``initial_progress`` to one over ``ramp_steps``. Below the threshold the
progress holds; it never goes back. After such a pause, the average must stay above the threshold for
``resume_dwell_steps`` (by default ``dwell_steps``) before the progress grows again. With ``after``, a block waits
until the named block is complete, so the blocks run one after another. Only the episodes that started at the
beginning of the task count (:attr:`~.commands.ClothTargetCommand.start_stage` APPROACH); episodes started from stored
reset states of later stages (uniform and adaptive resets) begin mid-task and succeed more easily.

The curriculum manager calls the terms at the start of every reset, with the environments being reset. The state is
not stored in policy checkpoints: a resumed run starts again from ``initial_progress``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase

from .stages import ClothDragStage

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import CurriculumTermCfg


class _SuccessGatedRamp(ManagerTermBase):
    """Progress in [0, 1] that grows with the environment steps while the success average is above a threshold."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        """Initialize the progress and the success average."""
        super().__init__(cfg, env)
        self.progress = float(cfg.params.get("initial_progress", 0.0))
        """Curriculum progress in [0, 1]."""
        if not 0.0 <= self.progress <= 1.0:
            raise ValueError(f"initial_progress must be in [0, 1], got {self.progress}.")
        self.success_ema = 0.0
        """Moving average of the success of the finished episodes."""
        self.gate_open = False
        """Whether the progress grew at the last update."""
        self._last_step = env.common_step_counter
        # step from which the block is ready and the average has stayed above the threshold, None while it is not
        self._above_since: int | None = None
        # whether the gate has opened before, so a reopening after a pause uses the resume wait
        self._has_opened = False

    def _update(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int] | slice,
        command_name: str,
        success_threshold: float,
        ramp_steps: int,
        ema_episodes: int,
        dwell_steps: int,
        after: str | None,
        resume_dwell_steps: int | None = None,
    ) -> None:
        """Fold the finished episodes into the success average and advance the progress.

        Args:
            env: The environment.
            env_ids: The environments being reset.
            command_name: Name of the :class:`~.commands.ClothTargetCommand`, whose ``success`` and ``start_stage``
                still hold the finished episode's: the command reset clears them later in the same reset.
            success_threshold: Success average from which the progress grows.
            ramp_steps: Environment steps (policy steps per environment) above the threshold from zero to full
                progress. With 24 steps per iteration, 7200 steps are 300 iterations.
            ema_episodes: Number of finished episodes the success average effectively spans.
            dwell_steps: Environment steps the block must be ready with the average above the threshold before the
                progress grows for the first time. Every drop below the threshold starts the wait again.
            after: Name of the curriculum block that must be at full progress before this one is ready. A disabled
                block (``None`` in the curriculum cfg) counts as complete.
            resume_dwell_steps: Environment steps the average must stay above the threshold before the progress grows
                again after a pause, once the gate has opened before. None => ``dwell_steps``.
        """
        if ramp_steps <= 0:
            raise ValueError("ramp_steps must be positive.")
        step = env.common_step_counter
        # the reset at the start of training ends no episode
        if step > 0:
            command = env.command_manager.get_term(command_name)
            # only the episodes that started at the beginning of the task
            ids = torch.arange(env.num_envs, device=command.start_stage.device)[env_ids]
            success = command.success[ids[command.start_stage[ids] == int(ClothDragStage.APPROACH)]]
            if success.numel() > 0:
                alpha = success.numel() / (success.numel() + ema_episodes)
                self.success_ema += alpha * (success.float().mean().item() - self.success_ema)
        if self.success_ema >= success_threshold and _block_complete(env, after):
            if self._above_since is None:
                self._above_since = step
        else:
            self._above_since = None
        wait = dwell_steps if not self._has_opened or resume_dwell_steps is None else resume_dwell_steps
        self.gate_open = self._above_since is not None and step - self._above_since >= wait
        self._has_opened |= self.gate_open
        if self.gate_open:
            self.progress = min(1.0, self.progress + (step - self._last_step) / ramp_steps)
        self._last_step = step


def _block_complete(env: ManagerBasedRLEnv, name: str | None) -> bool:
    """Whether the curriculum block ``name`` is at full progress; True without a name or for a disabled block."""
    if name is None:
        return True
    if not hasattr(env.curriculum_manager.cfg, name):
        raise ValueError(f"The curriculum has no block '{name}' to wait for.")
    term_cfg = getattr(env.curriculum_manager.cfg, name)
    return term_cfg is None or term_cfg.func.progress >= 1.0


class ActionRateCurriculum(_SuccessGatedRamp):
    """Success-gated ramp of the weight of an action-rate reward term.

    The weight goes geometrically with the progress from the term's configured weight to ``final_weight``:
    ``w = w_start * (final_weight / w_start) ** progress``, so each part of the ramp scales it by the same factor.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        """Read the start weight of the reward term."""
        super().__init__(cfg, env)
        self._reward_cfg = env.reward_manager.get_term_cfg(cfg.params["reward_term"])
        self._start_weight = float(self._reward_cfg.weight)
        final = float(cfg.params["final_weight"])
        if self._start_weight == 0.0 or final == 0.0 or (self._start_weight > 0.0) != (final > 0.0):
            raise ValueError(
                f"The start ({self._start_weight}) and final ({final}) weights must be nonzero and have the same sign"
                " for a geometric ramp."
            )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int] | slice,
        command_name: str,
        success_threshold: float,
        ramp_steps: int,
        reward_term: str,
        final_weight: float,
        ema_episodes: int = 2048,
        dwell_steps: int = 0,
        after: str | None = None,
        initial_progress: float = 0.0,
        resume_dwell_steps: int | None = None,
    ) -> dict[str, float]:
        """Advance the progress and set the reward weight.

        Args:
            env: The environment.
            env_ids: The environments being reset.
            command_name: See :meth:`_SuccessGatedRamp._update`.
            success_threshold: See :meth:`_SuccessGatedRamp._update`.
            ramp_steps: See :meth:`_SuccessGatedRamp._update`.
            reward_term: Name of the action-rate reward term whose weight is ramped.
            final_weight: Weight of that term at full progress.
            ema_episodes: See :meth:`_SuccessGatedRamp._update`.
            dwell_steps: See :meth:`_SuccessGatedRamp._update`.
            after: See :meth:`_SuccessGatedRamp._update`.
            initial_progress: Progress at the start of training, read once at initialization.
            resume_dwell_steps: See :meth:`_SuccessGatedRamp._update`.

        Returns:
            The state to log: success average, gate, progress and weight.
        """
        self._update(
            env,
            env_ids,
            command_name,
            success_threshold,
            ramp_steps,
            ema_episodes,
            dwell_steps,
            after,
            resume_dwell_steps,
        )
        weight = self._start_weight * (final_weight / self._start_weight) ** self.progress
        self._reward_cfg.weight = weight
        env.reward_manager.set_term_cfg(reward_term, self._reward_cfg)
        return {
            "success_ema": self.success_ema,
            "gate_open": float(self.gate_open),
            "progress": self.progress,
            "weight": weight,
        }


class ObservationHidingCurriculum(_SuccessGatedRamp):
    """Success-gated share of episodes in which the actor does not see an observation.

    :attr:`hidden` is drawn once per episode, with probability equal to the progress, so an episode sees the
    observation in every step or in none. The observation terms that name this block in ``hide_curriculum``
    (:func:`~.observations.cloth_drag_stage`, :func:`~.observations.cloth_grasped`) are all zeros in hidden episodes,
    a value distinct from every visible one.

    The term runs at the start of a reset, before the observation manager resets the observation history of the same
    environments and backfills it with the first observation of the new episode. The history of an episode therefore
    holds either the real observation or zeros in all of its slots.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        """Initialize the per-episode mask."""
        super().__init__(cfg, env)
        self.hidden = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        """Whether the actor does not see the observation in the current episode, shape [num_envs]."""
        self._generator = torch.Generator(device=env.device)
        self._generator.manual_seed(env.cfg.seed if env.cfg.seed is not None else torch.initial_seed())

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int] | slice,
        command_name: str,
        success_threshold: float,
        ramp_steps: int,
        ema_episodes: int = 2048,
        dwell_steps: int = 0,
        after: str | None = None,
        initial_progress: float = 0.0,
        resume_dwell_steps: int | None = None,
    ) -> dict[str, float]:
        """Advance the progress and draw the mask of the episodes that start now.

        Args:
            env: The environment.
            env_ids: The environments being reset.
            command_name: See :meth:`_SuccessGatedRamp._update`.
            success_threshold: See :meth:`_SuccessGatedRamp._update`.
            ramp_steps: See :meth:`_SuccessGatedRamp._update`.
            ema_episodes: See :meth:`_SuccessGatedRamp._update`.
            dwell_steps: See :meth:`_SuccessGatedRamp._update`.
            after: See :meth:`_SuccessGatedRamp._update`.
            initial_progress: Progress at the start of training, read once at initialization.
            resume_dwell_steps: See :meth:`_SuccessGatedRamp._update`.

        Returns:
            The state to log: progress and success average.
        """
        self._update(
            env,
            env_ids,
            command_name,
            success_threshold,
            ramp_steps,
            ema_episodes,
            dwell_steps,
            after,
            resume_dwell_steps,
        )
        num_reset = self.hidden[env_ids].shape[0]
        draw = torch.rand(num_reset, generator=self._generator, device=self.hidden.device)
        self.hidden[env_ids] = draw < self.progress
        return {"progress": self.progress, "success_ema": self.success_ema}

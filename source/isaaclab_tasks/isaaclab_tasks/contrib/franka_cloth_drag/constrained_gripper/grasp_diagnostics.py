# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-episode diagnostics of the grasp of the cloth dragging task, logged under ``Grasp/*``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from ..mdp.stages import (
    START_STAGE_NAMES,
    ClothDragStage,
    ee_grasp_node_distance,
    grasp_node_capturable,
    grasp_node_pos_w,
    gripper_holds_raised_node,
)
from .geometry import gripper_closing, gripper_holds_cloth

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.sensors import FrameTransformer

    from ..mdp.commands_cfg import ClothTargetCommandCfg

GRASP_OUTCOMES = ("right_lock", "wrong_lock", "closed_no_lock", "never_closed")
"""Outcome classes of an episode, see :class:`GraspDiagnostics`."""

# per-event means logged as sum over count; the count key of every mean
_MEANS = {
    "closes_per_episode": "episodes",
    "attempts_per_episode": "episodes",
    "attempt_lock_rate": "attempts",
    "attempt_right_lock_rate": "attempts",
    "close_ee_node_distance_mm": "closes",
    "close_node_between_pads": "closes",
    "attempt_ee_node_distance_mm": "attempts",
    "attempt_node_between_pads": "attempts",
    "close_ee_travel_mm": "timed_attempts",
    "close_node_travel_mm": "timed_attempts",
    "right_lock_nodes": "right_locks",
    "right_lock_on_grasp_node": "right_locks",
    "right_lock_time_s": "right_episodes",
    "wrong_lock_nodes": "wrong_locks",
    "wrong_lock_nearest_node_mm": "wrong_locks",
    "wrong_lock_time_s": "wrong_episodes",
    "wrong_lock_released": "wrong_episodes",
}
_COUNTS = ("episodes", "closes", "attempts", "timed_attempts", "right_locks", "wrong_locks")


class GraspDiagnostics:
    """Grasp events of every episode, logged at its end under ``Grasp/*``.

    Only episodes that start without a lock, in APPROACH or GRASP, are counted. A lock is right when it holds a raised
    node (the condition of DRAG, see :func:`~..mdp.stages.gripper_holds_raised_node`) and wrong otherwise, e.g. flat
    cloth next to the raised node. A close is the gripper command switching to closed; a lock attempt is the gripper
    trying to lock once the fingers settled (:attr:`~.grasp_action.ClothGraspAction.grasp_tried`), which forms a lock
    only with at least ``grasp_min_nodes`` nodes between the pads. Events are read once per environment step, after the
    physics, so values "at" an attempt or lock are one control step after the gripper tried.

    Logged keys, each left out when no event of its kind ended in the step:

    * ``outcome_<outcome>_<stage>``: share of the episodes that started in ``<stage>`` (approach or grasp) by outcome:
      ``right_lock`` (reached DRAG, also after a wrong lock), ``wrong_lock`` (locked only without a raised node),
      ``closed_no_lock`` (closed, but no lock formed) and ``never_closed``. The four sum to one.
    * ``closes_per_episode``, ``attempts_per_episode``: closes and lock attempts per episode.
    * ``attempt_lock_rate``, ``attempt_right_lock_rate``: share of the lock attempts that formed any lock, or a right
      lock.
    * ``close_ee_node_distance_mm``, ``attempt_ee_node_distance_mm``: distance of the EE frame origin to the grasp node
      at the close and at the attempt [mm].
    * ``close_node_between_pads``, ``attempt_node_between_pads``: share of the closes and attempts with the grasp node
      between the pads, as the lock selects nodes (:func:`~..mdp.stages.grasp_node_capturable` without the minimum
      count).
    * ``close_ee_travel_mm``, ``close_node_travel_mm``: distance the EE frame origin and the grasp node moved from the
      close to its attempt [mm]. Large EE travel with the node not between the pads at the attempt is a sweep past it.
    * ``right_lock_nodes``, ``wrong_lock_nodes``: number of locked nodes of right and wrong locks.
    * ``right_lock_on_grasp_node``: share of the right locks that hold the episode's grasp node, not another raised
      node.
    * ``right_lock_time_s``: episode time of the first right lock [s].
    * ``wrong_lock_nearest_node_mm``: distance of the grasp node to the nearest node of a wrong lock [mm].
    * ``wrong_lock_time_s``: time spent in a wrong lock, over the episodes with any wrong lock [s].
    * ``wrong_lock_released``: share of the episodes with any wrong lock that released one.
    """

    def __init__(self, env: ManagerBasedEnv, cfg: ClothTargetCommandCfg):
        """Set up empty per-environment accumulators.

        Args:
            env: The environment.
            cfg: The cloth target command configuration.
        """
        self._env = env
        self._cfg = cfg
        self._ee_frame: FrameTransformer = env.scene[cfg.ee_frame_name]
        num_envs, device = env.num_envs, env.device
        zeros = lambda: torch.zeros(num_envs, device=device)  # noqa: E731
        flag = lambda: torch.zeros(num_envs, dtype=torch.bool, device=device)  # noqa: E731
        self._sums = {name: zeros() for name in _MEANS}
        self._counts = {name: zeros() for name in _COUNTS if name != "episodes"}
        # state of the last update, to detect transitions; the first update of an episode only takes it
        self._fresh = torch.ones(num_envs, dtype=torch.bool, device=device)
        self._prev_closed, self._prev_tried, self._prev_active, self._prev_wrong = flag(), flag(), flag(), flag()
        # positions at the last close [m], and whether the episode had a close before its next attempt
        self._close_ee_pos_w = torch.zeros(num_envs, 3, device=device)
        self._close_node_pos_w = torch.zeros(num_envs, 3, device=device)
        self._close_pending = flag()
        # episode flags
        self._closed, self._wrong, self._wrong_released = flag(), flag(), flag()

    def update(self, grasp_node: torch.Tensor) -> None:
        """Record the grasp events since the last update. Call once per environment step.

        Args:
            grasp_node: Grasp node id of every environment, shape [num_envs].
        """
        env, cfg = self._env, self._cfg
        term = env.action_manager.get_term(cfg.gripper_action_name)
        constraint = term.constraint
        dt = env.step_dt
        closed = gripper_closing(env, cfg)
        tried = term.grasp_tried
        active = constraint.active
        holds = gripper_holds_cloth(env, cfg)
        right = gripper_holds_raised_node(env, cfg)
        wrong = holds & ~right

        live = ~self._fresh
        close_event = live & closed & ~self._prev_closed
        attempt_event = live & tried & ~self._prev_tried
        lock_event = live & active & ~self._prev_active
        right_lock, wrong_lock = lock_event & right, lock_event & wrong

        ee_pos_w = self._ee_frame.data.target_pos_w.torch[:, 0]
        node_pos_w = grasp_node_pos_w(env, cfg, grasp_node)
        distance_mm = ee_grasp_node_distance(env, cfg, grasp_node) * 1000.0
        between = grasp_node_capturable(env, cfg, grasp_node, require_min_nodes=False).float()
        s, c = self._sums, self._counts

        # closes
        s["closes_per_episode"] += close_event.float()
        c["closes"] += close_event.float()
        s["close_ee_node_distance_mm"] += torch.where(close_event, distance_mm, 0.0)
        s["close_node_between_pads"] += close_event.float() * between
        self._close_ee_pos_w = torch.where(close_event.unsqueeze(1), ee_pos_w, self._close_ee_pos_w)
        self._close_node_pos_w = torch.where(close_event.unsqueeze(1), node_pos_w, self._close_node_pos_w)
        self._close_pending |= close_event

        # lock attempts
        s["attempts_per_episode"] += attempt_event.float()
        c["attempts"] += attempt_event.float()
        s["attempt_lock_rate"] += (attempt_event & active).float()
        s["attempt_right_lock_rate"] += (attempt_event & right).float()
        s["attempt_ee_node_distance_mm"] += torch.where(attempt_event, distance_mm, 0.0)
        s["attempt_node_between_pads"] += attempt_event.float() * between
        timed = attempt_event & self._close_pending
        c["timed_attempts"] += timed.float()
        ee_travel_mm = torch.linalg.norm(ee_pos_w - self._close_ee_pos_w, dim=1) * 1000.0
        node_travel_mm = torch.linalg.norm(node_pos_w - self._close_node_pos_w, dim=1) * 1000.0
        s["close_ee_travel_mm"] += torch.where(timed, ee_travel_mm, 0.0)
        s["close_node_travel_mm"] += torch.where(timed, node_travel_mm, 0.0)
        self._close_pending &= ~attempt_event

        # locks
        num_locked = constraint.num_grasped.float()
        c["right_locks"] += right_lock.float()
        s["right_lock_nodes"] += torch.where(right_lock, num_locked, 0.0)
        on_node = constraint.grasped.gather(1, grasp_node.unsqueeze(1)).squeeze(1)
        s["right_lock_on_grasp_node"] += (right_lock & on_node).float()
        c["wrong_locks"] += wrong_lock.float()
        s["wrong_lock_nodes"] += torch.where(wrong_lock, num_locked, 0.0)
        if wrong_lock.any():
            nodal_pos_w = env.scene[cfg.object_name].data.nodal_pos_w.torch
            to_node = torch.linalg.norm(nodal_pos_w - node_pos_w.unsqueeze(1), dim=2)
            nearest_mm = torch.where(constraint.grasped, to_node, torch.inf).min(dim=1).values * 1000.0
            s["wrong_lock_nearest_node_mm"] += torch.where(wrong_lock, nearest_mm, 0.0)

        # episode flags
        self._closed |= live & closed
        self._wrong |= live & wrong
        self._wrong_released |= live & self._prev_wrong & ~holds
        s["wrong_lock_time_s"] += (live & wrong).float() * dt

        # the gripper changes these in place
        self._prev_closed, self._prev_tried, self._prev_active = closed.clone(), tried.clone(), active.clone()
        self._prev_wrong = wrong
        self._fresh[:] = False

    def reset(
        self, env_ids: Sequence[int] | slice, start_stage: torch.Tensor, time_to_drag: torch.Tensor
    ) -> dict[str, float]:
        """Return the log of the finished episodes of the given environments and clear their accumulators.

        Args:
            env_ids: The environments.
            start_stage: :class:`~..mdp.stages.ClothDragStage` index the finished episodes started in, shape [N].
            time_to_drag: Episode time at which the finished episodes first reached DRAG, negative if never [s],
                shape [N].

        Returns:
            The ``Grasp/*`` log, see the class docstring.
        """
        counted = start_stage <= int(ClothDragStage.GRASP)
        right = time_to_drag >= 0.0
        wrong = self._wrong[env_ids] & ~right
        closed = self._closed[env_ids]
        outcomes = torch.stack([right, wrong, closed & ~right & ~wrong, ~closed & ~right & ~wrong]).float()

        sums = {name: value[env_ids] for name, value in self._sums.items()}
        sums["right_lock_time_s"] = torch.where(right, time_to_drag, 0.0)
        counts = {name: value[env_ids] for name, value in self._counts.items()}
        counts["episodes"] = torch.ones_like(time_to_drag)
        counts["right_episodes"] = right.float()
        # any wrong lock, also one followed by a right lock: the outcome counts only the first of the two
        any_wrong = self._wrong[env_ids]
        counts["wrong_episodes"] = any_wrong.float()
        sums["wrong_lock_released"] = (any_wrong & self._wrong_released[env_ids]).float()

        # per start stage the episodes and their outcomes [2, 5], per mean its sum and count [M, 2]
        stage_rows = []
        for stage in (ClothDragStage.APPROACH, ClothDragStage.GRASP):
            in_stage = (counted & (start_stage == int(stage))).float()
            stage_rows.append(torch.cat([in_stage.sum().view(1), (outcomes * in_stage).sum(dim=1)]))
        weight = counted.float()
        mean_rows = [torch.stack([(sums[n] * weight).sum(), (counts[k] * weight).sum()]) for n, k in _MEANS.items()]
        # one transfer to the host for all values
        values = torch.cat([torch.stack(stage_rows).flatten(), torch.stack(mean_rows).flatten()]).tolist()
        stage_values = [values[i * 5 : (i + 1) * 5] for i in range(2)]
        mean_values = [values[10 + 2 * i : 12 + 2 * i] for i in range(len(_MEANS))]

        log = {}
        for stage, (num, *per_outcome) in zip(START_STAGE_NAMES[:2], stage_values):
            if num > 0:
                for outcome, value in zip(GRASP_OUTCOMES, per_outcome):
                    log[f"Grasp/outcome_{outcome}_{stage}"] = value / num
        for name, (total, count) in zip(_MEANS, mean_values):
            if count > 0:
                log[f"Grasp/{name}"] = total / count

        for value in (*self._sums.values(), *self._counts.values()):
            value[env_ids] = 0.0
        for flag in (self._closed, self._wrong, self._wrong_released, self._close_pending):
            flag[env_ids] = False
        self._fresh[env_ids] = True
        return log

Added
^^^^^

* Added three reset strategies to ``IsaacContrib-Drag-Cloth-Franka``, selected with the ``events`` preset:
  ``env.events=default_resets`` (the default) starts every episode at the beginning of the task with the randomized
  reset events; ``env.events=uniform_resets`` starts every episode from a stored reset state, drawn with fixed stage
  ratios; ``env.events=adaptive_resets`` draws the same states, but a solved stage hands its ratio of the resets to
  the stage before it (goal to drag, drag to grasp, grasp to the task start), down to a minimum ratio. The ratios
  follow the smoothed task success, after holding their start values for the first ``ratio_hold_steps`` environment
  steps. Play mode always uses the default resets.
* Added the offline bank of reset states behind the uniform and adaptive resets: task-start, pre-grasp (open gripper
  above a raised node), fresh-grasp and goal-zone states, generated and validated with the randomized state machine
  (``franka_cloth_drag/reset_bank_generation/scripts/generate_reset_bank.py``, inspector
  ``inspect_reset_bank.py``), and the ``ResetFromBank`` reset event (``franka_cloth_drag.reset_strategies``,
  configured by ``ResetFromBankCfg``) that restores them. Every stored state starts in the task stage of its bucket
  (APPROACH, GRASP, DRAG or GOAL). The two strategies load ``logs/reset_banks/bank_4000_v7.pt``, which must be
  generated first.
* Added frontier sampling as an option of the uniform and adaptive resets
  (``env.events.reset_from_bank.use_frontier_sampling=true``). It keeps the stage ratios and samples independently
  inside the states of every stage with an SGS beta kernel over a four-rollout, linearly weighted next-stage-success
  history (``FrontierSamplerCfg``). It also logs per-stage visit-frequency and frontier diagnostics and includes the
  frontier history, estimates, weights and probabilities in the periodic per-state snapshots.
* Added the success of the finished episodes per stage they started in to the metrics of the cloth target command,
  under every reset strategy: ``Metrics/cloth_target/task_success_<stage>_raw`` and
  ``Metrics/cloth_target/stage_success_<stage>_raw`` (the share that reached the stage after the start stage, the
  success for goal starts), and their moving averages over 20 iterations (``..._smoothed``,
  ``ClothTargetCommandCfg.success_window_steps``). ``task_success_approach_*``, the episodes that started at the
  beginning of the task, compares the reset strategies. The command tracks the start stage of every episode
  (``ClothTargetCommand.start_stage``). The uniform and adaptive resets also log ``Reset/share_<stage>_raw``, the share
  of each start stage.
* Added an eight-episode EMA of next-stage and full-task success for every stored reset state. The uniform and
  adaptive resets periodically save atomic per-state snapshots under the training run's ``reset_state_success``
  directory.

* Added the ``drag_hold`` reward to ``IsaacContrib-Drag-Cloth-Franka`` (:func:`cloth_held`, weight 0.5 per second in
  the drag and goal stages), so forming and keeping the lock on a raised node pays.

Changed
^^^^^^^

* Removed the ``pen_drop`` reward term from ``IsaacContrib-Drag-Cloth-Franka``. With default resets the one-time drop
  cost made the policy stop just short of the lock. Drop/re-grasp laps still earn nothing, because the progress terms
  telescope over the episode, and a drop now costs the hold reward. :func:`cloth_drag_drop_penalty` remains available.

* Renamed the stored task-start stage from ``default`` to ``approach`` to distinguish it from the default randomized
  reset strategy and match :class:`ClothDragStage`. Reset ratio inputs and bank lookup still accept ``default`` as a
  compatibility alias. Metrics now use ``approach`` in their stage suffix.
* Changed the success-gated curriculum of ``IsaacContrib-Drag-Cloth-Franka`` to count only episodes that started at
  the beginning of the task (``ClothTargetCommand.start_stage``). Runs with the default resets are unchanged.
* Reduced the training logs of ``IsaacContrib-Drag-Cloth-Franka``. ``Metrics/cloth_target`` keeps
  ``position_error_xy``, ``center_speed_at_target``, ``time_in_wrong_grasp`` and ``object_center_offset``; the stage
  outcomes it logged are in the ``Reset/*`` logs per source. ``Metrics/nonfinite_envs`` is gone; the ``nonfinite_state``
  termination still reports these resets. :class:`ClothTargetCommand` keeps the episode outcome in
  :attr:`ClothTargetCommand.success` instead of ``metrics["success"]``.
* **Breaking:** Changed reset banks to schema version 6: a bank holds exactly the requested number of states, split
  evenly over the task-start, grasp, drag and goal buckets, and the height of its raised cloth nodes. Regenerate
  older banks with ``generate_reset_bank.py``; older schemas are rejected when loaded.

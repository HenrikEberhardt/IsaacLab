Added
^^^^^

* Added reset states to ``IsaacContrib-Drag-Cloth-Franka``: an offline bank of pre-grasp (open gripper above a raised
  node), fresh-grasp and goal-zone states generated and validated with the randomized state machine
  (``scripts/environments/state_machine/nonprehensile_cloth_collect_resets.py``, viewer
  ``nonprehensile_cloth_reset_sm.py``), and the ``ResetFromBank`` reset event that restores them. Every stored state
  starts in the task stage of its bucket (GRASP, DRAG or GOAL). The bank presets load
  ``logs/reset_banks/bank_5000_v5.pt``, which must be generated first;
  ``env.events.reset_from_bank=off`` restores default resets only. Play mode always starts from default resets.
* Added ``Reset/task_success_<source>_raw`` and ``Reset/stage_success_<source>_raw`` training logs, the success of
  finished episodes per reset source and the share that reached the stage after their start stage (the success for goal
  starts), their moving averages over 20 iterations ``Reset/task_success_<source>_smoothed`` and
  ``Reset/stage_success_<source>_smoothed``, and ``Reset/share_<mode>_raw``, the share of each reset mode.
* Added the ``adaptive`` reset preset (``env.events.reset_from_bank=adaptive``) to ``IsaacContrib-Drag-Cloth-Franka``:
  the bank and start shares of ``uniform``, but a solved stage hands its share of the resets to the stage before it
  (goal to drag, drag to grasp, grasp to default), down to a minimum share. ``ResetFromBank`` takes the new optional
  ``min_probabilities``, ``success_range``, ``window_steps`` and ``hold_steps`` parameters. The shares follow the
  smoothed task success, after holding their start values for the first ``hold_steps`` environment steps.
* Added an eight-episode EMA of next-stage and full-task success for every stored reset state for later frontier
  sampling. The adaptive reset preset periodically saves atomic per-state snapshots under the training run's
  ``reset_state_success`` directory.
* Added the ``frontier`` reset preset (``env.events.reset_from_bank=frontier``). It keeps the fixed ``uniform`` stage
  shares and samples independently inside the grasp, drag and goal buffers with an SGS beta kernel over a four-rollout,
  linearly weighted next-stage-success history. It also logs per-buffer visit-frequency and frontier diagnostics and
  includes the frontier history, estimates, weights and probabilities in the periodic per-state snapshots.

Changed
^^^^^^^

* Changed the success-gated curriculum of ``IsaacContrib-Drag-Cloth-Franka`` to count only episodes started from the
  default reset (``reset_term`` parameter). Runs without reset states are unchanged.
* Reduced the training logs of ``IsaacContrib-Drag-Cloth-Franka``. ``Metrics/cloth_target`` keeps
  ``position_error_xy``, ``center_speed_at_target``, ``time_in_wrong_grasp`` and ``object_center_offset``; the stage
  outcomes it logged are in the ``Reset/*`` logs per source. ``Metrics/nonfinite_envs`` is gone; the ``nonfinite_state``
  termination still reports these resets. :class:`ClothTargetCommand` keeps the episode outcome in
  :attr:`ClothTargetCommand.success` instead of ``metrics["success"]``.
* **Breaking:** Changed reset banks to schema version 4, containing exactly the requested number of grasp, drag and
  goal reset states without approach states or the unused ``level`` and ``context_id`` labels. Regenerate older banks
  with ``nonprehensile_cloth_collect_resets.py``; older schemas are rejected when loaded.

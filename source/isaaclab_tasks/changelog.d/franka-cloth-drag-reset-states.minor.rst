Added
^^^^^

* Added reset states to ``IsaacContrib-Drag-Cloth-Franka``: an offline bank of pre-grasp (open gripper above a raised
  node), fresh-grasp and goal-zone states generated and validated with the randomized state machine
  (``scripts/environments/state_machine/nonprehensile_cloth_collect_resets.py``, viewer
  ``nonprehensile_cloth_reset_sm.py``), and the ``ResetFromBank`` reset event that restores them. Every stored state
  starts in the task stage of its bucket (GRASP, DRAG or GOAL); banks of an older schema no longer load and must be
  regenerated. The default preset ``uniform`` draws 70 % default resets and 15/10/5 % pre-grasp, drag and goal states
  from ``logs/reset_banks/bank_5000_v3.pt``, which must be generated first; ``env.events.reset_from_bank=off`` restores
  default resets only. Play mode always starts from default resets.
* Added ``Reset/success_<source>`` and ``Reset/share_<mode>`` training logs, the success of finished episodes per
  reset source and the share of each reset mode, and ``Reset/next_stage_<source>`` and ``Reset/time_to_next_<source>``,
  the share of finished episodes per source that reached the stage after their start stage and the time to it (to the
  success for goal starts).

Changed
^^^^^^^

* Changed the success-gated curriculum of ``IsaacContrib-Drag-Cloth-Franka`` to count only episodes started from the
  default reset (``reset_term`` parameter). Runs without reset states are unchanged.

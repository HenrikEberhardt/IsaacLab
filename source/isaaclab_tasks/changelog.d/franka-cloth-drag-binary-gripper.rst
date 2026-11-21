Changed
^^^^^^^

* **Breaking:** Changed the gripper of ``IsaacContrib-Drag-Cloth-Franka`` to a binary command, as on the real
  Robotiq gripper: a negative action closes, a positive one opens.
* **Breaking:** Removed the ``stage`` and ``cloth_grasped`` observations from the actor of
  ``IsaacContrib-Drag-Cloth-Franka``, because the real robot observes neither. The critic keeps both.
* **Breaking:** Removed the ``gripper_hysteresis_command`` observation from the actor and the critic of
  ``IsaacContrib-Drag-Cloth-Franka``, together with the ``gripper_command`` term: with the binary gripper it equals the
  sign of the gripper entry of ``last_action``. The actor observation shrinks from 675 to 645 and the critic from 794
  to 793, so policies trained before this change do not load.
* Added noise to the actor's gripper width of ``IsaacContrib-Drag-Cloth-Franka``: 0.4 mm per step (the Robotiq
  position resolution) and an offset of up to ±5 mm per episode. The offset spans the gap between a pinch and an empty
  close, so the width hints at a grasp only weakly, and the real gripper's readings lie inside the trained range. The
  critic keeps the true width.

Removed
^^^^^^^

* Removed ``hysteresis_threshold`` from ``ConstrainedGripperActionCfg``. Training with the binary command from the
  start reached the same success as a curriculum that ramped the hysteresis out. Reset banks made before this change
  fail the settings fingerprint check: re-stamp them or allow the settings mismatch. The stored grasp state is
  unchanged.
* Removed the curriculum of ``IsaacContrib-Drag-Cloth-Franka``, ``DeployCurriculum``, ``ObservationHidingCurriculum``
  and the ``hide_curriculum`` argument of ``cloth_drag_stage`` and ``cloth_grasped``. ``ActionRateCurriculum`` remains
  available. Resume or play overrides of ``env.curriculum.deploy.params.*`` no longer apply; drop them.

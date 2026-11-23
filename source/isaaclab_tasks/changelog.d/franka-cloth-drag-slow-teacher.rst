Added
^^^^^

* Added a ``scale_rot`` property and setter to ``TopDownDeltaEEAction`` of ``IsaacContrib-Drag-Cloth-Franka``.

Changed
^^^^^^^

* Changed ``AvoidTableCollisionCurriculum`` of ``IsaacContrib-Drag-Cloth-Franka`` to also ramp the arm's yaw step
  (``final_values["scale_rot"]``, 0.2 to 0.05 rad) and the joint speed stop (``final_values["joint_speed_fraction"]``,
  0.6 to 0.3 of the joint velocity limits, with the ``pen_joint_speed`` band). The force stop now ends at 15 N.
* **Breaking:** Changed the distillation envs of ``IsaacContrib-Drag-Cloth-Franka-Visual`` to apply the curriculum's
  final force and joint speed stops and penalties from the start. ``apply_teacher_curriculum`` is replaced by
  ``apply_final_curriculum_values`` and ``TEACHER_FORCE_LIMIT`` is removed. The teacher's arm action stays pinned to
  ``TEACHER_CURRICULUM_PROGRESS``. Set ``TEACHER_YAW_RAMP = True`` for a teacher checkpoint trained with the yaw-step
  ramp.

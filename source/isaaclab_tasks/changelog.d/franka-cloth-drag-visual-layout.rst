Added
^^^^^

* Added a 5-step proprioception history and a 5-frame uint8 wrist-image stack to the student of
  ``IsaacContrib-Drag-Cloth-Franka-Visual``, and an action-mapping interface (``IdentityMappingCfg``,
  ``IKToJointDeltaMappingCfg``) for the distillation algorithm ``MappedDistillation``.

Changed
^^^^^^^

* **Breaking:** Reorganized ``isaaclab_tasks.contrib.franka_cloth_drag_visual`` into ``teacher``, ``student``,
  ``action_mapping``, ``domain_randomization`` and ``tests``. ``IKLabelCfg`` is now ``IKToJointDeltaMappingCfg``,
  ``IKTeacherDistillation`` is now ``MappedDistillation``, and ``image_rgb_deploy`` is now ``wrist_rgb``.
  Student checkpoints of the previous single-frame layout do not load.
* Changed the distillation envs of ``IsaacContrib-Drag-Cloth-Franka-Visual`` to the task's adaptive bank resets.
* Changed the student's joint-delta action to anchor its target once per policy step
  (``anchor_per_policy_step=True``). The labels now take their scale from the student's action. Set
  ``anchor_per_policy_step=False`` for the previous per-physics-step re-anchoring.
* Changed the distillation teacher to the run ``gentle_contact_adaptive``. Its frozen arm action has only a z target
  floor and no joint-speed cap or range scaling. Its curriculum progress pins the z action scale and the gripper force
  stop. ``apply_teacher_terminations`` is replaced by ``apply_teacher_curriculum``.
* Changed the reset-bank fingerprint check of ``IsaacContrib-Drag-Cloth-Franka`` to accept sensors that the bank's
  task did not have, so banks made without the wrist camera load in the visual env.

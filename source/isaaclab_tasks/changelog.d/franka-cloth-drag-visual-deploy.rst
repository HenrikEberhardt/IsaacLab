Added
^^^^^

* Added LEAPP export paths to the student of ``IsaacContrib-Drag-Cloth-Franka-Visual``: the wrist image is exported as
  a float32 0-255 ``state/camera/image`` input, and the joint-delta and gripper actions export ``q + scale * a`` and
  the binary finger target without the simulation-only latency, dropped commands and cloth-node lock.
* Added the LEAPP command kind ``command/body/target_position`` to ``ClothTargetCommandCfg`` and
  ``deploy/send_target.py``, a terminal that publishes the cloth target to Isaac ROS Deploy.

Changed
^^^^^^^

* Changed the student's wrist image to the observation manager's history (``history_length=5``,
  ``flatten_history_dim=False``), which LEAPP exports as graph state; ``wrist_rgb_stack`` is now the single-frame
  term ``wrist_rgb``, and the model folds the ``[N, 5, 3, H, W]`` history into 15 channels. Checkpoints of the 5-frame
  stack load unchanged.
* Changed the play mode of the distillation envs to nominal values: no physics randomization, a fixed light and
  colors, the nominal camera mount, a 2-step image latency, and no dropped frames or commands.
* Changed the distillation teacher to label from uncorrupted observations.
* Changed the distillation teacher to the run ``slow_curriculum`` (``model_749``), trained with the yaw-step and
  joint-speed ramps of the curriculum.
* Changed the cloth color of ``IsaacContrib-Drag-Cloth-Franka`` and of the student's cloth recolor to the task's
  light-green cloth, sRGB (0.67, 0.74, 0.55).

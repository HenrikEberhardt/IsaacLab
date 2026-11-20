Added
^^^^^

* Added relative-physics diversity reporting and deterministic, stage-progress-stratified reset-bank selection for
  ``IsaacContrib-Drag-Cloth-Franka``.

Changed
^^^^^^^

* Changed cloth-drag frontier sampling to reserve a bounded coverage mixture for reset states with fewer than
  ``min_history`` outcomes and to admit states to the SGS beta kernel only after that history is reached. Existing
  configurations retain the weighted outcome history; set ``min_history=0``, ``coverage_mix_min=0`` and
  ``coverage_mix_max=0`` to recover the previous target-prior SGS behavior.

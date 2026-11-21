Added
^^^^^

* Added ``sample_targets`` and ``preset_targets`` to the cloth target command of ``IsaacContrib-Drag-Cloth-Franka``,
  so a reset event can draw the episode's target before the command resamples and hand it over.

Changed
^^^^^^^

* **Breaking:** ``ResetClothWithRaisedNodes`` of ``IsaacContrib-Drag-Cloth-Franka`` now raises exactly one node
  between the cube and the episode's target. Its distance from the cube (``grasp_distance_range``, 8–10 cm) and its
  sideways offset (``grasp_lateral_range``, ±1 cm) are drawn per episode. This replaces 3 random nodes. The node is a
  training aid where the real lower-and-squeeze grasp happens, so an image-based student can predict the grasp spot
  from the cube and the target. Migration:

  * Drop ``num_raised_nodes`` and ``min_spacing`` from the event params, and set ``grasp_distance_range``.
  * The actor observation shrinks from 645 to 615, so policies trained before this change do not load.
  * Reset banks with 3 raised nodes no longer match the task. The bank presets now read
    ``logs/reset_banks/bank_v13_one_raised_node.pt``.

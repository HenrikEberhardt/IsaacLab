Added
^^^^^

* Added a startup randomization of the arm joint armature, ×U(0.8, 1.2), to ``IsaacContrib-Drag-Cloth-Franka``.
* Added :class:`~isaaclab_tasks.contrib.franka_cloth_drag.mdp.ClothPointsNoiseModelCfg`, a perception-error noise
  model for the actor's cloth points of ``IsaacContrib-Drag-Cloth-Franka``. Per episode it shifts all points by
  one planar offset (±3 mm) and the raised nodes by another (±4 mm); per frame the raised nodes jitter by 2.5 mm. The
  teacher then aims at creases with a margin, which a later image-based student can imitate. The critic stays clean.

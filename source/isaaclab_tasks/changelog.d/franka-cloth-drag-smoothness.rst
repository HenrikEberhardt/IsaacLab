Changed
^^^^^^^

* Added a success-gated arm action-magnitude penalty to ``IsaacContrib-Drag-Cloth-Franka`` to discourage sustained
  full-range end-effector commands. Retrain cloth-drag policies to apply the new reward; existing checkpoints remain
  loadable but retain their previously learned motion.
* Changed the ``IsaacContrib-Drag-Cloth-Franka`` PPO actor to learn one smoothly bounded action standard deviation per
  action dimension and removed the runner action clip. The action term still enforces workspace, joint-position, and
  velocity safety limits. Retrain cloth-drag policies; play older checkpoints from the commit they were trained with.
* Added W&B metrics under ``Action/would_clip_*`` for raw policy samples exceeding magnitude 1.0.

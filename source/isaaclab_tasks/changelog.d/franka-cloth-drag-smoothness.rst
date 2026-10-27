Changed
^^^^^^^

* Added a success-gated arm action-magnitude penalty to ``IsaacContrib-Drag-Cloth-Franka`` to discourage sustained
  full-range end-effector commands. Retrain cloth-drag policies to apply the new reward; existing checkpoints remain
  loadable but retain their previously learned motion.
* Changed the ``IsaacContrib-Drag-Cloth-Franka`` PPO actor to predict a state-dependent, smoothly bounded action
  standard deviation, so each task stage can learn its own exploration. Retrain cloth-drag policies; play checkpoints
  trained before with ``agent.actor.distribution_cfg.class_name=isaaclab_tasks.contrib.franka_cloth_drag.distributions:SigmoidBoundedGaussianDistribution``.

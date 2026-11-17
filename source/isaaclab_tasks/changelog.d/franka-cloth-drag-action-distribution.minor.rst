Added
^^^^^

* Added the agent entry point ``rsl_rl_detached_std_cfg_entry_point`` (``--agent rsl_rl_detached_std_cfg_entry_point``)
  to ``IsaacContrib-Drag-Cloth-Franka``: an actor with a learned, state-dependent action standard deviation. A linear
  head on the detached features of the last hidden layer predicts it, so its gradient does not reach the layers that
  compute the mean, and a log-space sigmoid bounds it to ``std_range``. The default agent is unchanged.
* Added W&B metrics ``Loss/std_<stage>_<dimension>``: the mean action standard deviation of the states of each start
  stage. They need the new ``stage`` observation group, which neither the actor nor the critic reads.

Changed
^^^^^^^

* Moved the cloth-drag action distribution, its configuration and its PPO into the package
  ``franka_cloth_drag.action_distribution``: ``franka_cloth_drag.distributions`` and ``franka_cloth_drag.ppo`` became
  ``franka_cloth_drag.action_distribution.distributions`` and ``franka_cloth_drag.action_distribution.ppo``, and
  ``BoundedGaussianDistributionCfg`` and ``KLMetricsPPOCfg`` moved out of the runner configuration module. Checkpoints
  stay loadable. Update ``class_name`` overrides on the command line, e.g.
  ``agent.actor.distribution_cfg.class_name=isaaclab_tasks.contrib.franka_cloth_drag.action_distribution.distributions:SigmoidBoundedGaussianDistribution``.

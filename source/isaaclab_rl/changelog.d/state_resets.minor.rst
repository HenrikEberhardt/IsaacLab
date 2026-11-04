Added
^^^^^

* Added opt-in action diagnostics to the RSL-RL environment wrapper and runner configuration. With
  :attr:`~isaaclab_rl.rsl_rl.RslRlBaseRunnerCfg.action_dim_names` set, training logs per named action dimension the mean
  and standard deviation of the policy's action distribution (``Action/policy_mean_<name>``,
  ``Action/policy_std_<name>``) and the magnitude of the action sent to the environment
  (``Action/commanded_abs_<name>``).

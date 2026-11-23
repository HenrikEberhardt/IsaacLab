Added
^^^^^

* Added LEAPP input semantics for individual camera output buffers (``CameraData.output``), and resolved dotted
  input paths such as ``output.rgb`` in :class:`~isaaclab.envs.leapp_deployment_env.LeappDeploymentEnv`.

Changed
^^^^^^^

* Changed the ``leapp`` extra to LEAPP 0.7.2.

Fixed
^^^^^

* Fixed LEAPP export of observation history: the buffer's push count is now feedback state, so the exported graph
  backfills the history with the first frame at inference time instead of starting from the frames seen during
  export.

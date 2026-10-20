Fixed
^^^^^

* Fixed :func:`~isaaclab.app.launch_simulation` starting Isaac Sim / Kit for a kitless configuration whose
  ``sim.visualizer_cfgs`` lists a Kit visualizer although ``--visualizer`` explicitly selects other visualizers,
  e.g. ``--viz viser`` or ``--viz none``. An explicit selection already replaced the configured visualizers at
  runtime, so the run now stays kitless unless the selection includes ``kit``.

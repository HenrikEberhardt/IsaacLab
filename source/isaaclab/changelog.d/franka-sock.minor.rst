Added
^^^^^

* Added :class:`~isaaclab.sim.MeshTubeCfg` and :meth:`~isaaclab.sim.spawn_mesh_tube` for spawning a
  hollow tube mesh. Unlike :class:`~isaaclab.sim.MeshCylinderCfg`, which is a closed solid, the tube
  is an open-ended shell whose ends can each be closed with a hemispherical cap, which suits
  sleeve-shaped surface deformables. Its resolution is set by ``num_segments`` rather than
  ``edge_refinement``, because a slender tube's bounding-box diagonal stays above its
  circumferential edge length.

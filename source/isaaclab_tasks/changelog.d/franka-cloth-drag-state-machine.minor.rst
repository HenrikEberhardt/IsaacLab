Changed
^^^^^^^

* Refactored the Franka cloth-drag scripted expert into a typed Torch state machine, configuration profiles, an
  environment runner, and optional diagnostics. The expert now made one grasp attempt per rollout, used the task's
  configured action scales unchanged, and ran interactively with Viser on port 8760.

* Migrated custom callers to create ``ClothDragSmCfg`` and ``ClothDragSmParams`` explicitly. The former module path
  and constants remained available as compatibility imports, while ``keep_requested_visualizers`` was deprecated in
  favor of configuring ``ViserVisualizerCfg(port=8760)`` directly.

* Added periodic end-effector down-tilt diagnostics to distinguish tilt from yaw-tracking error.

* Limited scripted arm commands to 85 percent of the normalized action range without changing the task action
  scales. Added bounded operational-space roll/pitch stabilization for state-machine tools without changing the
  policy action or actuator configuration. Captured and validated reset-bank states only while the restored
  end-effector stayed within 1 degree of down.

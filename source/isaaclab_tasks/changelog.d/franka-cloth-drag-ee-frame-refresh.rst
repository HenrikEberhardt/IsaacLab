Fixed
^^^^^

* Fixed the end-effector frame sensor of ``IsaacContrib-Drag-Cloth-Franka`` showing the previous episode's pose
  until the first physics step after a reset. Newton updates its frame-transform sensors only at the end of a step,
  so the first observation of every episode and the grasp selection of the first gripper action used a pose up to
  0.4 m off. The new ``RefreshFrameSensors`` reset event updates the sensor from the reset state. The stage no
  longer is forced to APPROACH at the first step of an episode, and the success hold counts that step, both now
  computed from the fresh sensor. Existing checkpoints still load; their first observation per episode now differs
  from what they were trained on.

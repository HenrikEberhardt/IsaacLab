Added
^^^^^

* Added ``IsaacContrib-Franka-nonp-cloth``, ``IsaacContrib-Franka-nonp-cloth-WristCam``, and
  ``IsaacContrib-Franka-nonp-cloth-HeadCam``, the scene for a non-prehensile Franka Panda task in which the
  robot moves a rigid cube by pulling the cloth it rests on. A 0.40 m x 0.30 m cloth sheet with the sock's
  cloth material lies on the table, and a 5 cm solid PLA cube (0.155 kg) sits on top of it. The sheet
  carries a 3 cm crease across the corner nearest the robot on its left, which gives the gripper a fold to
  pinch. Each reset
  moves the cloth and the cube by one shared random offset, so the cube starts centred on the cloth.
  The tasks reuse the room, lighting, cameras, visualizers, and MDP of the ``IsaacContrib-Lift-Sock-Franka``
  tasks; task-specific rewards, commands, and observations are still to come.
* Simulated the cube of the ``IsaacContrib-Franka-nonp-cloth`` tasks with the same Newton VBD solver as the
  cloth, so cloth and cube contacts are solved together while MJWarp still simulates the robot. A coupler
  manager derives the cube's free-joint coordinates from the VBD body state after every substep, so the
  cube's pose and velocity read through :class:`~isaaclab.assets.RigidObject` stay current.
* Kept the crease of the ``IsaacContrib-Franka-nonp-cloth`` cloth in its rest shape. The sheet is spawned
  creased, and the tasks' coupler manager restores the bending rest angles that the Newton deformable asset
  zeroes when it initializes, so the crease holds under gravity instead of relaxing flat.

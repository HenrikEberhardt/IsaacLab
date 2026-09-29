Added
^^^^^

* Added ``IsaacContrib-Franka-nonp-cloth``, ``IsaacContrib-Franka-nonp-cloth-WristCam``, and
  ``IsaacContrib-Franka-nonp-cloth-HeadCam``, the scene for a non-prehensile Franka Panda task in which the
  robot moves a rigid cube by pulling the cloth it rests on. A 0.30 m x 0.30 m cloth sheet with the sock's
  cloth material lies on the table 0.60 m in front of the robot, and a 5 cm solid PLA cube (0.155 kg) sits on
  top of it. The sheet carries a 3 cm crease across the corner nearest the robot on its right, which gives the
  gripper a fold to pinch. Each reset moves the cloth and the cube by one shared random offset, so the cube
  starts centred on the cloth. The tasks reuse the room, lighting, cameras, visualizers, and MDP of the
  ``IsaacContrib-Lift-Sock-Franka`` tasks; task-specific rewards, commands, and observations are still to
  come.
* Simulated the cube of the ``IsaacContrib-Franka-nonp-cloth`` tasks with the same Newton VBD solver as the
  cloth, so cloth and cube contacts are solved together while MJWarp still simulates the robot. A coupler
  manager derives the cube's free-joint coordinates from the VBD body state after every substep, so the
  cube's pose and velocity read through :class:`~isaaclab.assets.RigidObject` stay current.
* Kept the crease of the ``IsaacContrib-Franka-nonp-cloth`` cloth in its rest shape. The sheet is spawned
  creased, and the tasks' coupler manager restores the bending rest angles that the Newton deformable asset
  zeroes when it initializes, so the crease holds under gravity instead of relaxing flat.
* Let kinematic cloth nodes work under the ``IsaacContrib-Franka-nonp-cloth`` coupler. The coupler manager
  copies the particles' active flags and inverse masses into the VBD solver view before every substep, which
  :meth:`~isaaclab.assets.DeformableObject.write_nodal_kinematic_target_to_sim_index` needs to pin nodes at
  runtime. It can also switch off the contacts between the proxied gripper and the cloth per environment,
  for grasps that lock cloth vertices to the gripper.
* Compensated gravity in the arm actuators of ``IsaacContrib-Franka-nonp-cloth`` on Newton, where MuJoCo
  ignores the inherited ``disable_gravity`` setting. Without it the low-gain arm sagged about 4 cm below its
  IK targets under full gravity. The IK action preset now also closes the gripper to a 1.4 cm gap, about the
  width of the pinched crease, instead of the 2 cm of the soft lift tasks, so the fingers close onto the fold
  without pushing through it.
* Gave visualizers that ``--viz`` requests but the ``IsaacContrib-Franka-nonp-cloth`` tasks do not configure,
  such as the OVRTX ``newton_rtx`` viewport, the camera of the tasks' configured visualizers, which is the
  ``IsaacContrib-Lift-Sock-Franka`` view, so videos show the same view whichever visualizer records them.
* Added the ``scripts/environments/state_machine/franka_nonp_cloth_sm.py`` state machine, which pinches the
  crease of ``IsaacContrib-Franka-nonp-cloth``, drags the cloth and the cube to a random point on a line in
  front of the robot, and releases it. By default it holds the cloth with position-level bilateral
  constraints that lock the grasped cloth vertices to the gripper, and holds the fingers at the opening where
  they closed on the fold; ``--grasp_mode friction`` squeezes the fold with the fully closed gripper and relies
  on friction alone.

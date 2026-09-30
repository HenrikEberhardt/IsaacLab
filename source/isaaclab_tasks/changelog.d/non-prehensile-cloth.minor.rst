Added
^^^^^

* Added ``IsaacContrib-Franka-nonp-cloth``, ``IsaacContrib-Franka-nonp-cloth-WristCam``, and
  ``IsaacContrib-Franka-nonp-cloth-HeadCam``, the scene for a non-prehensile Franka Panda task in which the
  robot moves a rigid cube by pulling the cloth it rests on. A 0.30 m x 0.30 m cloth sheet with the sock's
  cloth material lies on the table 0.60 m in front of the robot, and a 5 cm solid PLA cube (0.155 kg) sits on
  top of it. The sheet carries a 3 cm crease across the corner nearest the robot on its right, which gives the
  gripper a fold to pinch. Each reset moves the cloth and the cube by one shared random offset, so the cube
  starts centred on the cloth. The tasks reuse the room, lighting, cameras, and visualizers of the
  ``IsaacContrib-Lift-Sock-Franka`` tasks.

* Simulated the cube of the ``IsaacContrib-Franka-nonp-cloth`` tasks with the same Newton VBD solver as the
  cloth, so cloth and cube contacts are solved together while MJWarp still simulates the robot. A coupler
  manager derives the cube's free-joint coordinates from the VBD body state after every substep, so the
  cube's pose and velocity read through :class:`~isaaclab.assets.RigidObject` stay current.
* Kept the crease of the ``IsaacContrib-Franka-nonp-cloth`` cloth in its rest shape. The sheet is spawned
  creased, and the tasks' coupler manager restores the bending rest angles that the Newton deformable asset
  zeroes when it initializes, so the crease holds under gravity instead of relaxing flat.

* Let kinematic cloth nodes work under the 
``IsaacContrib-Franka-nonp-cloth`` coupler. The coupler manager
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

* Added the reinforcement learning MDP of the ``IsaacContrib-Franka-nonp-cloth`` tasks, which replaces the one
  inherited from the ``IsaacContrib-Lift-Sock-Franka`` tasks. The policy pinches the crease and drags it to a
  target that each episode draws on the line the state machine uses, 0.35 m in front of the robot base with
  y between -0.30 m and 0.30 m. The target and the crease are drawn as spheres that turn green, together with the
  table rim, once the crease is within 3 cm of the target. The rewards are staged. Before the grasp they pay for
  reaching the point 1.2 cm below the crest where the fingers straddle the fold, turning the gripper to close across
  the crease, and closing the fingers at that point, and a small penalty discourages closing them farther than
  2.5 cm from it; while the gripper holds the crease these terms pay their maximum, and a grasp term and the
  crease's horizontal distance to the target pay on top. A bonus pays once the crease is within 3 cm of the
  target, and ``Metrics/success_rate`` logs the share of episodes in which it got there. A drag-distance curriculum
  starts the targets 20 % of the way from the crease to the line, but at least 6 cm from it, and moves them out as
  episodes succeed; ``play`` always uses the full drag, since the curriculum is not saved with the policy. The
  action-rate penalty rises to -0.01 after 15,000 steps, instead of the soft lift tasks'
  -0.1, which would outweigh the approach rewards. Episodes last 10 s under full gravity, since the gravity
  curriculum of the soft lift tasks would leave the cube too light to ride on the cloth.

* Added ``ClothGraspActionCfg``, a binary gripper action that grasps the cloth inside the environment as the state
  machine does: once the closing fingers reach 1.5 cm each, it locks the cloth vertices pinched between them to the
  gripper, and once the opening fingers reach 3.2 cm each, it releases them. The fingers take three steps to cover
  either way, so single noisy gripper actions neither grasp nor drop the cloth. The state machine now imports the
  shared ``GripperConstraint`` from the task and drags to the task's target command, so the target marker shows
  where it drags.

* Gave the ``IsaacContrib-Franka-nonp-cloth`` tasks two action presets: the default ``ik_rel`` for training and
  ``ik``, an absolute end-effector pose, for the state machine. ``ik_rel`` combines the cloth grasp gripper with
  ``TopDownDifferentialIKActionCfg``, a differential inverse kinematics action that keeps the hand pointing straight
  down and only moves it and turns it about the vertical axis, a 4-dimensional arm action.

* Started the ``IsaacContrib-Franka-nonp-cloth`` robot with its hand pointing straight down. The soft lift tasks'
  default arm pose tilts the hand by 46 deg; the tasks turn joints 6 and 7 to point it down at zero yaw, and each
  reset draws a random arm pose that keeps it pointing down.

Added
^^^^^

* Added ``--viser_port`` to the Franka cloth-drag reset-bank inspector so multiple viewers can run concurrently.

Changed
^^^^^^^

* Changed ``IsaacContrib-Drag-Cloth-Franka`` from the Franka Panda hand to a Robotiq 2F-85 on the Panda arm (asset and
  drive from ``srl-ethz/IsaacLab-Cloth``). The gripper joint is ``finger_joint`` (0 rad open, 0.8 rad closed), driven
  with the real gripper's command delay, squeeze force and speed. Its pads move on a circle while closing, so the
  gripper observation (``gripper_width``), the pad box that selects the nodes to lock and the closure reward are
  computed from the pad gap curve. The end-effector frame is at the center of the closed pads, 19 mm above the
  fingertips, so the lowest end-effector height is 19.5 mm (``z_min``), and the reward, success and state machine
  heights moved up by about 11 mm. The gripper still locks the pinched cloth nodes; the lock now lets go once the
  commanded-open pads are 2 mm (``release_gap``) wider than at the pinch, instead of after a fixed number of steps.
  Retrain policies and regenerate reset banks: checkpoints and banks of the Panda hand do not fit the new joints,
  observations and stored grasp state.
* Changed the Robotiq colliders of ``IsaacContrib-Drag-Cloth-Franka`` to 11 boxes, loaded through
  ``assets/robotiq/franka_panda_robotiq_simplified.usda``: the pads stay exact, and every other gripper collider is
  replaced by an enclosing box per rigid unit (base, inner-finger backing in two boxes, outer finger with outer
  knuckle), grown by 2 mm. The inner knuckles retain their diagonal collider orientation with the same margin. The
  mechanism still touches the cube and cloth wherever it did before; the arm and camera-assembly colliders, which do
  not participate in task contact, are removed. The simulation runs about 1.5 times faster (982 to about 645 ms per
  step at 2048 environments). Reset banks made with the previous colliders fail the physics fingerprint check:
  regenerate them, or allow the settings mismatch.
* Moved the constraint gripper into ``franka_cloth_drag/constrained_gripper``: the coupler, the lock constraint,
  ``ClothGraspAction`` and its configuration, the gripper helpers of ``mdp.stages`` and the grasp diagnostics. The
  settings ``grasp_min_nodes``, ``finger_joint_name``, ``finger_pad_half_width``, ``finger_pad_half_length`` and
  ``finger_gap_margin`` moved from ``ClothTargetCommandCfg`` to ``ClothGraspActionCfg``
  (``env.actions.gripper_action.<name>`` on the command line).
* Changed the cloth-drag state-machine script to always use fresh default resets and serve its Viser viewer on port
  8752.

Removed
^^^^^^^

* Removed the friction-only gripper mode (``ClothGraspActionCfg.attach``) of ``IsaacContrib-Drag-Cloth-Franka``: the
  gripper always locks cloth nodes. Removed the Panda hand's pad and hand box settings (``contact_pad_*``,
  ``contact_hand_*``); the contact clearance uses the Robotiq's collider boxes.

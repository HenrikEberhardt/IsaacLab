Changed
^^^^^^^

* Restructured the constrained gripper of ``IsaacContrib-Drag-Cloth-Franka``. The behavior is unchanged. Migration
  from the old to the new names:

  * ``constrained_gripper.grasp_action.ClothGraspAction`` became
    ``constrained_gripper.gripper_action.ConstrainedGripperAction``, and its attribute ``constraint`` became
    ``node_lock``.
  * ``constrained_gripper.grasp_action_cfg.ClothGraspActionCfg`` became
    ``constrained_gripper.gripper_action_cfg.ConstrainedGripperActionCfg``.
  * ``constrained_gripper.constraint.ClothGraspConstraint`` became
    ``constrained_gripper.constraint_node_lock.ClothNodeLock``.
  * ``constrained_gripper.geometry`` became ``constrained_gripper.gripper_state``. The gap curve functions
    ``pad_gap_from_joint``, ``joint_from_pad_gap`` and ``pad_shift_from_joint`` moved to
    ``constrained_gripper.robotiq``.
  * ``RobotiqGripperActuator`` and its configuration moved to ``constrained_gripper.robotiq_actuator``.
  * ``constrained_gripper.coupler.NewtonClothGraspCouplerManager`` and
    ``constrained_gripper.coupler_cfg.ClothGraspCouplerProxyCfg`` moved to the new ``franka_cloth_drag.physics``
    package as ``NewtonClothDragCouplerManager`` and ``ClothDragCouplerCfg``.

  Reset banks made before this change fail the settings fingerprint check: re-stamp them or allow the settings
  mismatch. The stored grasp state is unchanged.

Removed
^^^^^^^

* Removed settings of ``IsaacContrib-Drag-Cloth-Franka`` that no configuration used. Each was a no-op at its default:
  ``squeeze_lift`` of the gripper action, ``rest_angle_scale`` of the coupler (the raised nodes set the rest shape per
  environment), ``stroke_time_range``, ``stroke`` and ``physics_dt`` of ``RobotiqGripperActuatorCfg`` (the drive's
  effort limit and viscous friction set the finger speed), the ``finger_opening`` helper (use ``0.5 * pad_gap``), and
  the unused Robotiq constants ``ROBOTIQ_OPEN``, ``ROBOTIQ_CLOSED``, ``ROBOTIQ_BASE_BODY`` and ``ROBOTIQ_PAD_*``.

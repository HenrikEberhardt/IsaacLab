Changed
^^^^^^^

* Added fixed quadratic arm action-magnitude and action-rate penalties to ``IsaacContrib-Drag-Cloth-Franka`` from the
  first training step, replacing their success-gated curricula. Retrain cloth-drag policies to apply the new rewards;
  existing checkpoints remain loadable but retain their previously learned motion.
* Changed the ``IsaacContrib-Drag-Cloth-Franka`` PPO actor to learn one smoothly bounded action standard deviation per
  action dimension. Retrain cloth-drag policies; play older checkpoints from the commit they were trained with.
* Added separate W&B metrics under ``Action/outside_nominal_*`` for raw policy samples exceeding magnitude 1.0 and
  ``Action/safety_clip_*`` for samples clipped by a runner action clip, if set, and per action dimension (``dx``,
  ``dy``, ``dz``, ``dyaw``, ``gripper``) the share of those samples and the policy's mean and standard deviation.
* Added an ``approach_precision`` reward to ``IsaacContrib-Drag-Cloth-Franka``: approach progress measured on
  ``tanh(distance / 2 cm)`` to the state machine's squeeze pose, the grasp node's x-y at 9.5 mm above the table, which
  pays nearly all of its reward in the last 3 cm before the grasp. :class:`ClothDistanceProgress` takes the new
  optional ``kernel_std`` and ``target_height`` parameters for this.
* Changed the ``IsaacContrib-Drag-Cloth-Franka`` approach to target one grasp node per episode,
  :attr:`ClothTargetCommand.grasp_node_id`: the raised node closest to the segment from the cloth center to the target,
  which the scripted state machine also grasps, instead of the raised node nearest the end effector. APPROACH turns into
  GRASP near that node, ``approach_progress`` targets the hover point 4 cm above the table over it, the actor's
  ``cloth_points`` list it first among the raised nodes, and the critic observes its position (``grasp_node_pos``).
  Retrain cloth-drag policies.
* Changed :class:`ClothDistanceProgress` to take ``stages`` like the other stage terms instead of ``objective``.
  Replace ``objective="approach"`` with ``stages=["approach", "grasp"]`` and ``objective="drag"`` with
  ``stages=["drag", "goal"]``. The approach progress now pays in the APPROACH and GRASP stages instead of while the
  gripper holds nothing.
* Replaced ``gripper_closed_on_raised_node`` and ``gripper_closed_without_raised_node`` with
  :func:`gripper_closed_on_grasp_node` and :func:`gripper_closed_off_grasp_node`. ``IsaacContrib-Drag-Cloth-Franka``
  rewards a close that locks the grasp node in GRASP (``grasp_close_on_node``, 4.0 per second for the first grasp of an
  episode, scaled by the nodes in the grasp up to ``num_nodes_full=6``) and penalizes a closed gripper without the grasp
  node between the pads in APPROACH and GRASP (``pen_close_off_node``, 1.0 per second) instead of any closing in
  APPROACH. A close pays only until its lock attempt (:attr:`ClothGraspAction.grasp_tried`), so a gripper held closed
  after a failed attempt earns nothing.
* Added the ``random_grasp_node`` reset-state collection option (``--random_grasp_node``). Collected banks now grasp
  the task's grasp node by default; recollect cloth-drag reset banks.
* Split the end-effector height penalty of ``IsaacContrib-Drag-Cloth-Franka`` into ``pen_ee_height_drag`` and
  ``pen_ee_height_goal``, one per stage, and softened it to a weight of -15 above 4 cm (was -50 above 3 cm).
* Rebalanced the ``IsaacContrib-Drag-Cloth-Franka`` stages: ``drag_progress`` pays 3 per 10 cm (was 10) and the success
  bonus is 75 (was 100), so approach, grasp and drag shaping earn about 4, 1.4 and 7 per episode and success about six
  times their sum.
* Added a ``goal_precision`` reward to ``IsaacContrib-Drag-Cloth-Franka``: progress of the cube toward the target
  measured on ``tanh(distance / 2 cm)`` in DRAG and GOAL, which pays nearly all of its 2.0 within the last 4 cm.
* Changed the ``IsaacContrib-Drag-Cloth-Franka`` termination penalty to :func:`cloth_drag_termination_penalty`, paid once
  per failed episode like the success bonus: -75 per termination. ``is_terminated_term`` was integrated over one control
  step, so its weight of -200 cost only about -6.7 per termination. The penalty now covers every termination except
  success and time out, including ``nonfinite_state`` and ``joint_vel_out_of_limit``, which ended episodes for free.
* Changed :class:`ClothDistanceProgress` to keep its reference distance while its stages are inactive: the first step
  back pays or charges what changed meanwhile, so the return sums to the start minus the end distance over the episode.
  Previously the progress restarted on re-entry, and ``IsaacContrib-Drag-Cloth-Franka`` policies learned to lift,
  drop and re-grasp the cloth to collect the approach reward again on every descent. Retrain cloth-drag policies.
* Added a one-time drop penalty to ``IsaacContrib-Drag-Cloth-Franka`` (``pen_drop``, -3 per release of the cloth in
  DRAG or GOAL, :func:`cloth_drag_drop_penalty`), and :attr:`ClothTargetCommand.previous_stage`, the stage before the
  last update, which it checks its ``stages`` against.
* Added a bounds loss on the mean action to the ``IsaacContrib-Drag-Cloth-Franka`` PPO actor
  (``mean_bound=1.0``, ``mean_bound_coef=0.01`` on ``BoundedGaussianDistributionCfg``), which pulls a saturated mean
  back toward the nominal range of 1.0.
* Removed the runner action clip of ``IsaacContrib-Drag-Cloth-Franka`` (``clip_actions=None``). PPO learned from samples
  beyond 1.0 that the environment never executed, which pushed the mean past the bound. The action penalties now see the
  raw action. Play policies trained with the clip with ``agent.clip_actions=1.0``; retrain for the new behavior.
* Changed the hiding curricula of ``IsaacContrib-Drag-Cloth-Franka`` to ramp together: both open at a success average of
  0.75 from default resets after 25 iterations and ramp over 150 iterations. They ran one after the other from 0.8.

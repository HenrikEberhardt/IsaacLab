# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Constrained gripper of the Franka cloth dragging task: the gripper locks the cloth nodes it pinches to its EE frame.

* :mod:`.gripper_action` / :mod:`.gripper_action_cfg`: the action term :class:`ConstrainedGripperAction` (latch,
  grasp, hold, release, contact clearance) and its configuration :class:`ConstrainedGripperActionCfg`.
* :mod:`.constraint_node_lock`: :class:`ClothNodeLock`, which selects the pinched nodes, moves them with the hand and
  squeezes the fold.
* :mod:`.gripper_state`: read-outs for the MDP terms (command, closure, pad gap, nodes between the pads).
* :mod:`.robotiq`: the Robotiq 2F-85 asset, its constants and pad kinematics, and the linkage stiffening.
* :mod:`.robotiq_actuator`: the finger drive with its command delay.
* :mod:`.grasp_diagnostics`: per-episode grasp statistics logged under ``Grasp/*``.

The lock needs the coupler manager of :mod:`..physics`, which keeps locked nodes kinematic and switches the
gripper-cloth contacts. The task's values (finger drives, ``GRIPPER_ACTION_CFG``) are set in
``config/franka/franka_cloth_drag_env_cfg.py``.
"""

from isaaclab.utils.module import lazy_export

lazy_export()

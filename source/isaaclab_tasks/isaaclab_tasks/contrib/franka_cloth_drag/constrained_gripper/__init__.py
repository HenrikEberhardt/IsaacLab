# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Constrained gripper of the Franka cloth dragging task: the gripper locks the cloth nodes it pinches to its EE frame.

* :mod:`.grasp_action` / :mod:`.grasp_action_cfg`: the gripper action term :class:`ClothGraspAction` and its
  configuration :class:`ClothGraspActionCfg`.
* :mod:`.constraint`: the lock :class:`ClothGraspConstraint`, which selects the pinched nodes and moves them with the
  hand.
* :mod:`.geometry`: gripper state for the MDP terms (command, finger opening, nodes between the pads).
* :mod:`.coupler` / :mod:`.coupler_cfg`: the Newton coupler manager that the lock needs, and its configuration
  :class:`ClothGraspCouplerProxyCfg`.
* :mod:`.grasp_diagnostics`: per-episode grasp statistics logged under ``Grasp/*``.

The task's values (finger drives, cloth-solver proxy, ``GRIPPER_ACTION_CFG``) are set in
``config/franka/franka_cloth_drag_env_cfg.py``.
"""

from isaaclab.utils.module import lazy_export

lazy_export()

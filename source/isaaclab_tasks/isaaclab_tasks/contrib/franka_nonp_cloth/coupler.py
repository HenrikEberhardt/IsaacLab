# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton coupler manager for a creased cloth sheet with a VBD-owned rigid cube on top."""

from __future__ import annotations

import numpy as np
import warp as wp
from isaaclab_newton.physics import VBDSolverCfg
from isaaclab_newton.physics.newton_manager import NewtonManager
from newton import Contacts, Control, Model, State, eval_ik

from isaaclab_contrib.coupling.coupler import NewtonCouplerManager
from isaaclab_contrib.coupling.coupler_cfg import CouplerCfg


class NewtonClothCubeCouplerManager(NewtonCouplerManager):
    """Coupler manager that keeps a creased cloth creased and VBD-owned rigid bodies readable.

    It adds two things to :class:`~isaaclab_contrib.coupling.coupler NewtonCouplerManager`:

    * The Newton deformable asset zeroes every cloth bending rest angle when it initializes, so cloth
      always bends back toward flat. Before building the solver, this manager restores the rest angles
      the model builder computed from the spawned meshes, so a cloth spawned with a crease keeps it.

    * VBD integrates rigid bodies in world coordinates and writes only ``body_q`` and ``body_qd``,while Isaac Lab reads a floating body's root pose and velocity from its free joint's ``joint_q``and ``joint_qd``. 
        
        After each substep, this manager runs Newton's inverse kinematics on the
        articulations whose joints a VBD entry owns. => MuJoCo computees "joint" and derives "body_q". VBD does it the other way round.

        Update also the "joint" so the Isaac Lab also showes the dispalycment in simulation.
    """

    _vbd_articulation_mask: wp.array | None = None
    """Per-articulation mask of the articulations owned by a VBD entry, or None if there are none."""

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: CouplerCfg) -> None:
        """Restore the authored cloth rest angles, build the coupled solver, and record which articulations VBD owns."""
        builder = NewtonManager._builder
        if builder is not None and model.edge_count:
            rest_angles = np.asarray(builder.edge_rest_angle, dtype=np.float32)
            if rest_angles.shape[0] != model.edge_count:
                raise RuntimeError(
                    f"Builder has {rest_angles.shape[0]} edge rest angles, but the model has {model.edge_count} edges."
                )
            model.edge_rest_angle.assign(rest_angles)

        super()._build_solver(model, solver_cfg)

        vbd_joints = [
            joint
            for entry_cfg in solver_cfg.entries
            if isinstance(entry_cfg.solver_cfg, VBDSolverCfg)
            for joint in cls._resolve_entry(model, entry_cfg).joints
        ]
        if not vbd_joints or not model.articulation_count:
            cls._vbd_articulation_mask = None
            return
        articulations = model.joint_articulation.numpy()[vbd_joints]
        mask = np.zeros(model.articulation_count, dtype=bool)
        mask[articulations[articulations >= 0]] = True
        cls._vbd_articulation_mask = wp.array(mask, dtype=wp.bool, device=model.device)

    @classmethod
    def _step_solver(
        cls, state_0: State, state_1: State, control: Control, contacts: Contacts | None, substep_dt: float
    ) -> None:
        """Run one coupled substep, then derive the VBD-owned joint coordinates from the body state."""
        super()._step_solver(state_0, state_1, control, contacts, substep_dt)
        if cls._vbd_articulation_mask is not None:
            eval_ik(NewtonManager._model, state_1, state_1.joint_q, state_1.joint_qd, mask=cls._vbd_articulation_mask)

    @classmethod
    def _solver_specific_clear(cls) -> None:
        """Release the articulation mask with the rest of the solver state."""
        super()._solver_specific_clear()
        cls._vbd_articulation_mask = None

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton coupler manager for a creased cloth sheet with a VBD-owned rigid cube on top."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import warp as wp
from isaaclab_newton.physics import VBDSolverCfg
from isaaclab_newton.physics.newton_manager import NewtonManager
from newton import Contacts, Control, Model, ParticleFlags, ShapeFlags, State, eval_ik

from isaaclab_contrib.coupling.coupler import NewtonCouplerManager
from isaaclab_contrib.coupling.coupler_cfg import CouplerCfg

_PARTICLE_ACTIVE = wp.constant(int(ParticleFlags.ACTIVE))


@wp.kernel
def _sync_particle_activity(
    particle_indices: wp.array(dtype=wp.int32),
    particle_global_to_local: wp.array(dtype=wp.int32),
    model_flags: wp.array(dtype=wp.int32),
    model_inv_mass: wp.array(dtype=wp.float32),
    view_flags: wp.array(dtype=wp.int32),
    view_inv_mass: wp.array(dtype=wp.float32),
):
    """Copy the active flag and inverse mass of owned particles from the model into a solver view."""
    tid = wp.tid()
    particle = particle_indices[tid]
    local = particle_global_to_local[particle]
    view_flags[local] = (view_flags[local] & ~_PARTICLE_ACTIVE) | (model_flags[particle] & _PARTICLE_ACTIVE)
    view_inv_mass[local] = model_inv_mass[particle]


class NewtonClothCubeCouplerManager(NewtonCouplerManager):
    """Coupler manager that keeps a creased cloth creased and VBD-owned rigid bodies readable.

    It adds three things to :class:`~isaaclab_contrib.coupling.coupler.NewtonCouplerManager`:

    * The Newton deformable asset zeroes every cloth bending rest angle when it initializes, so cloth
      always bends back toward flat. Before building the solver, this manager restores the rest angles
      the model builder computed from the spawned meshes, so a cloth spawned with a crease keeps it.

    * VBD integrates rigid bodies in world coordinates and writes only ``body_q`` and ``body_qd``, while
      Isaac Lab reads a floating body's root pose and velocity from its free joint's ``joint_q`` and
      ``joint_qd``.

      After each substep, this manager runs Newton's inverse kinematics on the articulations whose joints
      a VBD entry owns. => MuJoCo computees "joint" and derives "body_q". VBD does it the other way round.

      Update also the "joint" so the Isaac Lab also showes the dispalycment in simulation.

    * The coupler's solver views keep their own copies of per-particle properties and never refresh them,
      so a particle made kinematic at runtime (e.g. through
      :meth:`~isaaclab.assets.DeformableObject.write_nodal_kinematic_target_to_sim_index`) would stay
      dynamic in VBD. Before each substep, this manager copies the active flag and inverse mass of the
      particles each entry owns from the model into that entry's view.
    """

    _vbd_articulation_mask: wp.array | None = None
    """Per-articulation mask of the articulations owned by a VBD entry, or None if there are none."""

    _particle_entries: list = []
    """Coupled-solver entries that own particles, whose views get the model's particle activity each substep."""

    _default_shape_flags: list = []
    """Shape flags of each particle entry's view when the solver was built, to restore proxy contacts."""

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

        cls._particle_entries = [
            entry for entry in NewtonManager._solver._entries.values() if entry.particle_indices.shape[0] > 0
        ]
        cls._default_shape_flags = [entry.view.shape_flags.numpy() for entry in cls._particle_entries]

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
        """Sync particle activity into the solver views, run one coupled substep, then derive the VBD-owned
        joint coordinates from the body state."""
        model = NewtonManager._model
        for entry in cls._particle_entries:
            wp.launch(
                _sync_particle_activity,
                dim=entry.particle_indices.shape[0],
                inputs=[
                    entry.particle_indices,
                    entry.particle_global_to_local,
                    model.particle_flags,
                    model.particle_inv_mass,
                ],
                outputs=[entry.view.particle_flags, entry.view.particle_inv_mass],
                device=model.device,
            )
        super()._step_solver(state_0, state_1, control, contacts, substep_dt)
        if cls._vbd_articulation_mask is not None:
            eval_ik(NewtonManager._model, state_1, state_1.joint_q, state_1.joint_qd, mask=cls._vbd_articulation_mask)

    @classmethod
    def set_proxy_particle_contacts(cls, world_ids: Sequence[int], enabled: bool) -> None:
        """Enable or disable contacts between the proxied rigid bodies and the particles of the given worlds.

        Disabling them makes the proxied bodies, e.g. a gripper, a virtual one for the cloth. A position-level
        constraint that locks grasped cloth vertices to the gripper needs this: the gripper's contacts with the
        immovable, locked vertices would otherwise fight the constraint and destabilize the lagged coupling.
        The proxied bodies keep colliding with other rigid bodies.

        Args:
            world_ids: Worlds (environments) to change.
            enabled: Whether the proxied bodies collide with particles.
        """
        worlds = np.asarray(world_ids, dtype=np.int64).reshape(-1)
        collide_particles = int(ShapeFlags.COLLIDE_PARTICLES)
        for entry, default_flags in zip(cls._particle_entries, cls._default_shape_flags):
            proxy_bodies = entry.proxy_body_local_indices.numpy()
            if proxy_bodies.size == 0:
                continue
            view = entry.view
            selected = np.isin(view.shape_body.numpy(), proxy_bodies) & np.isin(view.shape_world.numpy(), worlds)
            flags = view.shape_flags.numpy()
            if enabled:
                flags[selected] = (flags[selected] & ~collide_particles) | (default_flags[selected] & collide_particles)
            else:
                flags[selected] &= ~collide_particles
            view.shape_flags.assign(flags)

    @classmethod
    def _solver_specific_clear(cls) -> None:
        """Release the articulation mask and the particle entries with the rest of the solver state."""
        super()._solver_specific_clear()
        cls._vbd_articulation_mask = None
        cls._particle_entries = []
        cls._default_shape_flags = []

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton coupler manager that supports locking cloth nodes to the gripper.

Adapted from ``NewtonClothCubeCouplerManager`` on the ``non-prehensile-cloth`` branch.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import warp as wp
from isaaclab_newton.physics.newton_manager import NewtonManager
from newton import Contacts, Control, Model, ParticleFlags, ShapeFlags, State

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


class NewtonClothGraspCouplerManager(NewtonCouplerManager):
    """Coupler manager for a gripper that grasps cloth by locking cloth nodes to it.

    The class name must start with ``Newton``: Isaac Lab picks the asset backend from the physics manager's name.

    It adds three things to :class:`~isaaclab_contrib.coupling.coupler.NewtonCouplerManager`:

    * The coupler's solver views keep their own copies of per-particle properties and never refresh them, so a node
      made kinematic at runtime through
      :meth:`~isaaclab.assets.DeformableObject.write_nodal_kinematic_target_to_sim_index` would stay dynamic in VBD.
      Before each substep, this manager copies the active flag and inverse mass of the particles each entry owns from
      the model into that entry's view.
    * :meth:`set_proxy_particle_contacts` switches the contacts between the proxied gripper and the cloth per
      environment, so a gripper holding locked nodes does not press against them.
    * Before building the solver, it restores :attr:`~.coupler_cfg.ClothGraspCouplerProxyCfg.rest_angle_scale` of the
      bending rest angles that the model builder computed from the spawned cloth and the Newton deformable asset
      zeroed, so the wrinkles of the spawned cloth stay partly set.
    """

    _particle_entries: list = []
    """Coupled-solver entries that own particles, whose views get the model's particle activity each substep."""

    _default_shape_flags: list = []
    """Shape flags of each particle entry's view when the solver was built, to restore proxy contacts."""

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: CouplerCfg) -> None:
        """Restore the scaled cloth rest angles, build the coupled solver and record the entries that own particles."""
        cls._restore_rest_angles(model, getattr(solver_cfg, "rest_angle_scale", 0.0))
        super()._build_solver(model, solver_cfg)
        cls._particle_entries = [
            entry for entry in NewtonManager._solver._entries.values() if entry.particle_indices.shape[0] > 0
        ]
        cls._default_shape_flags = [entry.view.shape_flags.numpy() for entry in cls._particle_entries]

    @staticmethod
    def _restore_rest_angles(model: Model, scale: float) -> None:
        """Set the bending rest angles to ``scale`` times those the model builder computed from the spawned meshes."""
        builder = NewtonManager._builder
        if scale == 0.0 or builder is None or not model.edge_count:
            return
        rest_angles = np.asarray(builder.edge_rest_angle, dtype=np.float32)
        if rest_angles.shape[0] != model.edge_count:
            raise RuntimeError(
                f"Builder has {rest_angles.shape[0]} edge rest angles, but the model has {model.edge_count} edges."
            )
        model.edge_rest_angle.assign(scale * rest_angles)

    @classmethod
    def _step_solver(
        cls, state_0: State, state_1: State, control: Control, contacts: Contacts | None, substep_dt: float
    ) -> None:
        """Sync the particle activity into the solver views, then run one coupled substep."""
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

    @classmethod
    def set_proxy_particle_contacts(cls, world_ids: Sequence[int], enabled: bool) -> None:
        """Enable or disable contacts between the proxied rigid bodies and the particles of the given worlds.

        Disabling them makes the proxied gripper a virtual one for the cloth. A grasp that locks cloth nodes to the
        gripper needs this: the gripper's contacts with the immovable locked nodes would otherwise fight the lock and
        destabilize the lagged coupling. The proxied bodies keep colliding with other rigid bodies.

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
        """Release the particle entries with the rest of the solver state."""
        super()._solver_specific_clear()
        cls._particle_entries = []
        cls._default_shape_flags = []

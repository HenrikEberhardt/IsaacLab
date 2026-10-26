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
import torch
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


class NewtonClothGraspCouplerManager(NewtonCouplerManager):
    """Coupler manager for a gripper that grasps cloth by locking cloth nodes to it.

    The class name must start with ``Newton``: Isaac Lab picks the asset backend from the physics manager's name.

    It adds five things to :class:`~isaaclab_contrib.coupling.coupler.NewtonCouplerManager`:

    * VBD integrates rigid bodies, such as the cube on the cloth, in world coordinates and writes only ``body_q`` and
      ``body_qd``, while Isaac Lab reads a floating body's root pose and velocity from its free joint's ``joint_q`` and
      ``joint_qd``. After each substep, this manager runs Newton's inverse kinematics on the articulations whose
      joints a VBD entry owns, so Isaac Lab sees their motion.
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
    * :meth:`set_cloth_rest` changes the cloth rest shape (bending rest angles and lengths, triangle rest shapes and
      areas) at runtime in the model and the solver views.
    """

    _vbd_articulation_mask: wp.array | None = None
    """Per-articulation mask of the articulations owned by a VBD entry, or None if there are none."""

    _particle_entries: list = []
    """Coupled-solver entries that own particles, whose views get the model's particle activity each substep."""

    _default_shape_flags: list = []
    """Shape flags of each particle entry's view when the solver was built, to restore proxy contacts."""

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: CouplerCfg) -> None:
        """Restore the scaled cloth rest angles, build the coupled solver, and record the entries that own particles
        and the articulations that VBD owns."""
        cls._restore_rest_angles(model, getattr(solver_cfg, "rest_angle_scale", 0.0))
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
        cls._vbd_articulation_mask = None
        if vbd_joints and model.articulation_count:
            articulations = model.joint_articulation.numpy()[vbd_joints]
            mask = np.zeros(model.articulation_count, dtype=bool)
            mask[articulations[articulations >= 0]] = True
            if mask.any():
                cls._vbd_articulation_mask = wp.array(mask, dtype=wp.bool, device=model.device)

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
        """Sync the particle activity into the solver views, run one coupled substep, then derive the joint
        coordinates of the VBD-owned articulations from their body state."""
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
            eval_ik(model, state_1, state_1.joint_q, state_1.joint_qd, mask=cls._vbd_articulation_mask)

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

    CLOTH_REST_ARRAYS = ("edge_rest_angle", "edge_rest_length", "tri_poses", "tri_areas")
    """Cloth rest-shape arrays of the model that :meth:`get_cloth_rest` and :meth:`set_cloth_rest` handle."""

    @classmethod
    def get_cloth_rest(cls, name: str, ids: torch.Tensor) -> torch.Tensor:
        """Rest values the cloth solver uses for the given model edges or triangles.

        Args:
            name: One of :attr:`CLOTH_REST_ARRAYS`.
            ids: Model edge ids (``edge_*``) or triangle ids (``tri_*``), any shape.

        Returns:
            A copy of the values, shape ``ids.shape`` plus ``(2, 2)`` for ``tri_poses``.
        """
        return wp.to_torch(cls._rest_arrays(name)[-1])[ids].clone()

    @classmethod
    def set_cloth_rest(cls, name: str, ids: torch.Tensor, values: torch.Tensor) -> None:
        """Set rest values of the given model edges or triangles at runtime.

        The solver views keep their own copies of model arrays, made when the solver was built, so writing the model
        alone would not reach the cloth solver. This writes the model and every view that owns particles.

        Args:
            name: One of :attr:`CLOTH_REST_ARRAYS`: bending rest angles [rad], bending edge rest lengths [m],
                inverse triangle rest matrices [1/m], or triangle rest areas [m^2].
            ids: Model edge ids (``edge_*``) or triangle ids (``tri_*``), any shape.
            values: Rest values, shape ``ids.shape`` plus ``(2, 2)`` for ``tri_poses``.
        """
        for array in cls._rest_arrays(name):
            wp.to_torch(array)[ids] = values

    @classmethod
    def _rest_arrays(cls, name: str) -> list[wp.array]:
        """The named array of the model and of the particle-owning solver views, without duplicates.

        The model comes first, so the last array is the one a cloth solver reads.
        """
        if name not in cls.CLOTH_REST_ARRAYS:
            raise ValueError(f"Unknown cloth rest array '{name}', expected one of {cls.CLOTH_REST_ARRAYS}.")
        model_array = getattr(NewtonManager._model, name)
        arrays = {model_array.ptr: model_array}
        for entry in cls._particle_entries:
            array = getattr(entry.view, name)
            if array.shape != model_array.shape:
                raise RuntimeError(f"Solver view '{entry.view.name}' does not hold every model element of '{name}'.")
            arrays[array.ptr] = array
        return list(arrays.values())

    @classmethod
    def _solver_specific_clear(cls) -> None:
        """Release the articulation mask and the particle entries with the rest of the solver state."""
        super()._solver_specific_clear()
        cls._vbd_articulation_mask = None
        cls._particle_entries = []
        cls._default_shape_flags = []

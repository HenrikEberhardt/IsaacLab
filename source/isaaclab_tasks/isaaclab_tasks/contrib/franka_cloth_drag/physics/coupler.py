# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton coupler manager of the Franka cloth dragging task."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import torch
import warp as wp
from isaaclab_newton.physics import VBDSolverCfg
from isaaclab_newton.physics.newton_manager import NewtonManager
from newton import Contacts, Control, Model, ModelBuilder, ParticleFlags, ShapeFlags, State, eval_ik

from isaaclab.physics import PhysicsManager

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


@wp.kernel
def _latch_touching_pairs(
    contact_count: wp.array(dtype=wp.int32),
    contact_shape0: wp.array(dtype=wp.int32),
    contact_shape1: wp.array(dtype=wp.int32),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_margin0: wp.array(dtype=wp.float32),
    contact_margin1: wp.array(dtype=wp.float32),
    shape_body: wp.array(dtype=wp.int32),
    shape_world: wp.array(dtype=wp.int32),
    shape_groups: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    tolerance: float,
    num_pairs: int,
    latch: wp.array2d(dtype=wp.int32),
):
    """Latch every monitored pair (shape group bits ``2 i`` / ``2 i + 1``) whose contact touches in ``body_q``."""
    tid = wp.tid()
    if tid >= contact_count[0]:
        return
    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]
    if shape_a < 0 or shape_b < 0:
        return
    groups_a = shape_groups[shape_a]
    groups_b = shape_groups[shape_b]
    if groups_a == 0 or groups_b == 0:
        return
    # contact points are in the body frames, static shapes in the world frame
    point_a = contact_point0[tid]
    if shape_body[shape_a] >= 0:
        point_a = wp.transform_point(body_q[shape_body[shape_a]], point_a)
    point_b = contact_point1[tid]
    if shape_body[shape_b] >= 0:
        point_b = wp.transform_point(body_q[shape_body[shape_b]], point_b)
    gap = wp.dot(contact_normal[tid], point_b - point_a) - contact_margin0[tid] - contact_margin1[tid]
    if gap > tolerance:
        return
    world = shape_world[shape_a]
    if world < 0:
        world = shape_world[shape_b]
    if world < 0:
        return
    for i in range(num_pairs):
        first = 1 << (2 * i)
        second = 1 << (2 * i + 1)
        if ((groups_a & first) != 0 and (groups_b & second) != 0) or (
            (groups_a & second) != 0 and (groups_b & first) != 0
        ):
            latch[world, i] = 1


def _shapes_matching(pattern: str, shape_labels, shape_body, body_labels) -> np.ndarray:
    """Return the ids of the shapes whose label, or whose body's label, fully matches ``pattern``."""
    regex = re.compile(pattern)
    return np.array(
        [
            shape
            for shape, label in enumerate(shape_labels)
            if regex.fullmatch(label) or (shape_body[shape] >= 0 and regex.fullmatch(body_labels[shape_body[shape]]))
        ],
        dtype=np.int64,
    )


class NewtonClothDragCouplerManager(NewtonCouplerManager):
    """Coupler manager of the cloth dragging task.

    The class name must start with ``Newton``: Isaac Lab picks the asset backend from the physics manager's name.

    Additions to :class:`~isaaclab_contrib.coupling.coupler.NewtonCouplerManager`:

    * Gripper lock: the particle activity is copied into the solver views before each substep, so locked cloth nodes
      stay kinematic in VBD; :meth:`set_proxy_particle_contacts` switches the gripper-cloth contacts per environment.
    * Cube: inverse kinematics after each substep, so Isaac Lab sees the motion of VBD-integrated bodies.
    * ``collision_filter_pairs``: shape pairs that never collide.
    * ``monitored_contacts``: per-world touch latch of shape pairs, see :meth:`contact_latch`.
    * :meth:`get_cloth_rest` / :meth:`set_cloth_rest`: read and change the cloth rest shape at runtime.
    """

    _vbd_articulation_mask: wp.array | None = None
    """Per-articulation mask of the articulations owned by a VBD entry, or None if there are none."""

    _particle_entries: list = []
    """Coupled-solver entries that own particles, whose views get the model's particle activity each substep."""

    _default_shape_flags: list = []
    """Shape flags of each particle entry's view when the solver was built, to restore proxy contacts."""

    _contact_names: list[str] = []
    """Names of the monitored contact pairs, in latch column order."""

    _contact_shape_groups: wp.array | None = None
    """Per-shape bit mask of the monitored pair sides the shape belongs to, or None without monitored pairs."""

    _contact_latch: wp.array | None = None
    """Per-world, per-pair touch flags since the last :meth:`clear_contact_latch`, shape [num_worlds, num_pairs]."""

    _contact_buffers: list | None = None
    """Contacts of the proxy pipelines, looked up at the first substep; None until then."""

    @classmethod
    def _prepare_builder_for_finalize(cls, builder: ModelBuilder) -> None:
        """Exclude the configured shape pairs from collision in every world before the model is finalized."""
        super()._prepare_builder_for_finalize(builder)
        pairs = getattr(PhysicsManager._cfg.solver_cfg, "collision_filter_pairs", [])
        if not pairs:
            return
        shape_world = np.asarray(builder.shape_world)
        for pattern_a, pattern_b in pairs:
            shapes_a, shapes_b = (
                _shapes_matching(pattern, builder.shape_label, builder.shape_body, builder.body_label)
                for pattern in (pattern_a, pattern_b)
            )
            if len(shapes_a) == 0 or len(shapes_b) == 0:
                raise ValueError(f"Collision filter pair ({pattern_a!r}, {pattern_b!r}) matches no shape.")
            for shape_a in shapes_a:
                for shape_b in shapes_b[shape_world[shapes_b] == shape_world[shape_a]]:
                    builder.add_shape_collision_filter_pair(int(shape_a), int(shape_b))

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: CouplerCfg) -> None:
        """Build the coupled solver and record the particle entries, the VBD bodies and the monitored contacts."""
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
        cls._build_contact_monitor(model, solver_cfg)

    @classmethod
    def _build_contact_monitor(cls, model: Model, solver_cfg: CouplerCfg) -> None:
        """Mark the shapes of the monitored pairs and allocate the per-world latch before any graph capture."""
        monitored = getattr(solver_cfg, "monitored_contacts", {})
        cls._contact_names = list(monitored)
        cls._contact_buffers = None
        if not monitored:
            cls._contact_shape_groups = cls._contact_latch = None
            return
        if len(monitored) > 15:
            raise ValueError("At most 15 contact pairs can be monitored.")
        shape_body = model.shape_body.numpy()
        groups = np.zeros(model.shape_count, dtype=np.int32)
        for i, (name, patterns) in enumerate(monitored.items()):
            for side, pattern in enumerate(patterns):
                shapes = _shapes_matching(pattern, model.shape_label, shape_body, model.body_label)
                if len(shapes) == 0:
                    raise ValueError(f"Monitored contact '{name}': pattern {pattern!r} matches no shape.")
                groups[shapes] |= 1 << (2 * i + side)
        cls._contact_shape_groups = wp.array(groups, dtype=wp.int32, device=model.device)
        cls._contact_latch = wp.zeros((model.world_count, len(monitored)), dtype=wp.int32, device=model.device)

    @classmethod
    def _step_solver(
        cls, state_0: State, state_1: State, control: Control, contacts: Contacts | None, substep_dt: float
    ) -> None:
        """Sync the particle activity, run one coupled substep, then update VBD bodies and the contact latch."""
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
        if cls._contact_latch is not None:
            cls._latch_contacts(model, state_1)

    @classmethod
    def _latch_contacts(cls, model: Model, state: State) -> None:
        """Latch the monitored pairs whose detected contacts still touch with the body poses of ``state``."""
        if cls._contact_buffers is None:
            solver = NewtonManager._solver
            proxies = PhysicsManager._cfg.solver_cfg.proxies
            buffers = [solver.get_proxy_contacts(proxy.source, proxy.destination) for proxy in proxies]
            cls._contact_buffers = [buffer for buffer in buffers if buffer is not None]
            if not cls._contact_buffers:
                raise RuntimeError("Monitored contacts need a proxy with its own collision pipeline.")
        tolerance = PhysicsManager._cfg.solver_cfg.contact_touch_tolerance
        for buffer in cls._contact_buffers:
            wp.launch(
                _latch_touching_pairs,
                dim=buffer.rigid_contact_max,
                inputs=[
                    buffer.rigid_contact_count,
                    buffer.rigid_contact_shape0,
                    buffer.rigid_contact_shape1,
                    buffer.rigid_contact_point0,
                    buffer.rigid_contact_point1,
                    buffer.rigid_contact_normal,
                    buffer.rigid_contact_margin0,
                    buffer.rigid_contact_margin1,
                    model.shape_body,
                    model.shape_world,
                    cls._contact_shape_groups,
                    state.body_q,
                    tolerance,
                    len(cls._contact_names),
                ],
                outputs=[cls._contact_latch],
                device=model.device,
            )

    @classmethod
    def contact_latch(cls, name: str) -> torch.Tensor:
        """Return per world whether the monitored pair ``name`` touched since its last :meth:`clear_contact_latch`."""
        if name not in cls._contact_names:
            raise KeyError(f"Contact pair '{name}' is not monitored; monitored: {cls._contact_names}.")
        return wp.to_torch(cls._contact_latch)[:, cls._contact_names.index(name)].bool()

    @classmethod
    def clear_contact_latch(cls, name: str) -> None:
        """Clear the touch flags of the monitored pair ``name`` in every world."""
        wp.to_torch(cls._contact_latch)[:, cls._contact_names.index(name)] = 0

    @classmethod
    def set_proxy_particle_contacts(cls, world_ids: Sequence[int], enabled: bool) -> None:
        """Enable or disable the contacts between the proxied gripper and the cloth particles of the given worlds."""
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
        """Return a copy of the rest values ``name`` the cloth solver uses for the given edge or triangle ids."""
        return wp.to_torch(cls._rest_arrays(name)[-1])[ids].clone()

    @classmethod
    def set_cloth_rest(cls, name: str, ids: torch.Tensor, values: torch.Tensor) -> None:
        """Write the rest values ``name`` of the given edge or triangle ids into the model and every solver view."""
        for array in cls._rest_arrays(name):
            wp.to_torch(array)[ids] = values

    @classmethod
    def _rest_arrays(cls, name: str) -> list[wp.array]:
        """Return the named array of the model, then of each particle-owning solver view, without duplicates."""
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
        """Release the articulation mask, particle entries and contact monitor with the rest of the solver state."""
        super()._solver_specific_clear()
        cls._vbd_articulation_mask = None
        cls._particle_entries = []
        cls._default_shape_flags = []
        cls._contact_names = []
        cls._contact_shape_groups = None
        cls._contact_latch = None
        cls._contact_buffers = None

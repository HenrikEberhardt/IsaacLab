# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import Literal

from isaaclab.sim.spawners import materials
from isaaclab.sim.spawners.spawner_cfg import DeformableObjectSpawnerCfg, RigidObjectSpawnerCfg
from isaaclab.utils import configclass


@configclass
class MeshCfg(RigidObjectSpawnerCfg, DeformableObjectSpawnerCfg):
    """Configuration parameters for a USD Geometry or Geom prim.

    This class is similar to :class:`ShapeCfg` but is specifically for meshes.

    Meshes support both rigid and deformable properties. However, their schemas are applied at
    different levels in the USD hierarchy based on the type of the object. These are described below:

    - Deformable body properties: Applied to the parent prim: ``{prim_path}``.
    - Collision properties: Applied to the simulation mesh ``{prim_path}/sim_mesh`` for deformable bodies,
      and to the mesh prim ``{prim_path}/geometry/mesh`` otherwise.
    - Rigid body properties: Applied to the parent prim: ``{prim_path}``.

    where ``{prim_path}`` is the path to the prim in the USD stage and ``{prim_path}/geometry/mesh``
    is the path to the mesh prim.

    .. note::
        There are mututally exclusive parameters for rigid and deformable properties. If both are set,
        then an error will be raised. If :attr:`collision_props` is set alongside deformable properties,
        it must be given as collision fragments, since legacy cfgs cannot target the simulation mesh.

    """

    visual_material_path: str = "material"
    """Path to the visual material to use for the prim. Defaults to "material".

    If the path is relative, then it will be relative to the prim's path.
    This parameter is ignored if `visual_material` is not None.
    """

    visual_material: materials.VisualMaterialCfg | None = None
    """Visual material properties.

    Note:
        If None, then no visual material will be added.
    """

    physics_material_path: str = "material"
    """Path to the physics material to use for the prim. Defaults to "material".

    If the path is relative, then it will be relative to the prim's path.
    This parameter is ignored if `physics_material` is not None.
    """

    physics_material: (
        materials.PhysicsMaterialCfg
        | materials.RigidBodyMaterialFragment
        | list[materials.RigidBodyMaterialFragment]
        | None
    ) = None
    """Physics material properties.

    Accepts either a legacy material cfg, a single
    :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialFragment`, or a list of such
    single-namespace fragments.

    Note:
        If None, then no physics material will be added.
    """

    edge_refinement: float = 4.0
    """Mesh edge refinement factor for deformable bodies.

    The maximum surface edge length is the bounding-box diagonal divided by this value. Volume deformables use the
    same normalized target for automatic tetrahedralization. The factor must be at least ``1.0``. For volume
    deformables, values near ``1.0`` should be avoided because they can make TetWild tetrahedralization significantly
    slower. Defaults to ``4.0``.
    """


@configclass
class MeshSphereCfg(MeshCfg):
    """Configuration parameters for a sphere mesh prim with deformable properties.

    See :meth:`spawn_mesh_sphere` for more information.
    """

    func: Callable | str = "{DIR}.meshes:spawn_mesh_sphere"

    radius: float = MISSING
    """Radius of the sphere (in m)."""


@configclass
class MeshCuboidCfg(MeshCfg):
    """Configuration parameters for a cuboid mesh prim with deformable properties.

    See :meth:`spawn_mesh_cuboid` for more information.
    """

    func: Callable | str = "{DIR}.meshes:spawn_mesh_cuboid"

    size: tuple[float, float, float] = MISSING
    """Size of the cuboid [m]."""


@configclass
class MeshCylinderCfg(MeshCfg):
    """Configuration parameters for a cylinder mesh prim with deformable properties.

    See :meth:`spawn_cylinder` for more information.
    """

    func: Callable | str = "{DIR}.meshes:spawn_mesh_cylinder"

    radius: float = MISSING
    """Radius of the cylinder (in m)."""
    height: float = MISSING
    """Height of the cylinder (in m)."""
    axis: Literal["X", "Y", "Z"] = "Z"
    """Axis of the cylinder. Defaults to "Z"."""


@configclass
class MeshCapsuleCfg(MeshCfg):
    """Configuration parameters for a capsule mesh prim.

    See :meth:`spawn_capsule` for more information.
    """

    func: Callable | str = "{DIR}.meshes:spawn_mesh_capsule"

    radius: float = MISSING
    """Radius of the capsule (in m)."""
    height: float = MISSING
    """Height of the capsule (in m)."""
    axis: Literal["X", "Y", "Z"] = "Z"
    """Axis of the capsule. Defaults to "Z"."""


@configclass
class MeshConeCfg(MeshCfg):
    """Configuration parameters for a cone mesh prim.

    See :meth:`spawn_cone` for more information.
    """

    func: Callable | str = "{DIR}.meshes:spawn_mesh_cone"

    radius: float = MISSING
    """Radius of the cone (in m)."""
    height: float = MISSING
    """Height of the v (in m)."""
    axis: Literal["X", "Y", "Z"] = "Z"
    """Axis of the cone. Defaults to "Z"."""


@configclass
class MeshRectangleCfg(MeshCfg):
    """Configuration parameters for a 2D rectangle mesh prim.

    See :meth:`spawn_mesh_rectangle` for more information.
    """

    func: Callable | str = "{DIR}.meshes:spawn_mesh_rectangle"

    size: tuple[float, float] = MISSING
    """Edge lengths of the rectangle along the X and Y axes [m]."""


@configclass
class MeshTubeCfg(MeshCfg):
    """Configuration parameters for a hollow tube mesh prim.

    Unlike :class:`MeshCylinderCfg`, which is a closed solid, this is an open-ended shell whose
    ends can be capped independently. See :meth:`spawn_mesh_tube` for more information.
    """

    func: Callable | str = "{DIR}.meshes:spawn_mesh_tube"

    radius: float = MISSING
    """Radius of the tube [m]."""
    height: float = MISSING
    """Length of the cylindrical section, excluding the caps [m].

    A cap extends the tube by :attr:`radius` beyond this length, as it does for
    :class:`MeshCapsuleCfg`.
    """
    axis: Literal["X", "Y", "Z"] = "Z"
    """Axis of the tube. Defaults to "Z"."""
    num_segments: int = 16
    """Number of circumferential segments. Defaults to 16.

    This sets the resolution of the whole shell, since the axial and latitudinal spacing follow the
    circumferential edge length. For a slender tube, :attr:`~MeshCfg.edge_refinement` is usually
    ineffective, because the bounding-box diagonal it divides is already close to the tube length
    while the circumferential edges are much shorter. Prefer this parameter, and note that a
    volume deformable still needs :attr:`~MeshCfg.edge_refinement` for tetrahedralization.
    """
    cap_start: bool = False
    """Whether to close the negative-axis end with a hemisphere. Defaults to False."""
    cap_end: bool = False
    """Whether to close the positive-axis end with a hemisphere. Defaults to False."""

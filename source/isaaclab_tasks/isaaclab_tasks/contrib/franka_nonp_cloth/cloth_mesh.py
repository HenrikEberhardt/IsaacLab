# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cloth sheet mesh with a crease across one corner, for the Franka non-prehensile cloth task."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from pxr import Usd


def corner_ridge_positions(
    positions: torch.Tensor,
    corner_signs: torch.Tensor,
    height: torch.Tensor,
    width: torch.Tensor,
    distance: torch.Tensor,
    num_samples: int = 64,
) -> torch.Tensor:
    """Raise a ridge across one corner of flat, rectangular cloth sheets without stretching them.

    Nodes in the ridge follow a ``sin^2`` bump, placed along its arc length.

    Args:
        positions: Node positions of flat sheets lying in the xy plane [m], shape [E, N, 3].
        corner_signs: Signs (+-1) of the chosen corner along x and y in the sheet frame, shape [E, 2].
        height: Ridge height above the sheet [m], shape [E].
        width: Footprint width of the ridge along the diagonal [m], shape [E].
        distance: Distance of the ridge center from the corner, along the diagonal [m], shape [E].
        num_samples: Number of samples used to integrate the arc length of the bump.

    Returns:
        The wrinkled node positions [m], shape [E, N, 3].
    """
    center_xy = positions[..., :2].mean(dim=1, keepdim=True)
    local_xy = positions[..., :2] - center_xy
    corner = corner_signs * local_xy.abs().amax(dim=1)
    corner_dist = corner.norm(dim=-1)
    direction = corner / corner_dist.unsqueeze(-1)
    t = (local_xy * direction.unsqueeze(1)).sum(dim=-1)
    t_in = (corner_dist - distance - 0.5 * width).unsqueeze(-1)

    # bump profile over its footprint and its cumulative arc length, per env
    u = torch.linspace(0.0, 1.0, num_samples, device=positions.device)
    x_k = u * width.unsqueeze(-1)
    z_k = height.unsqueeze(-1) * torch.sin(torch.pi * u).square()
    s_k = torch.cat(
        (torch.zeros_like(x_k[:, :1]), torch.hypot(x_k.diff(dim=-1), z_k.diff(dim=-1)).cumsum(dim=-1)), dim=-1
    )
    arc = s_k[:, -1:]

    # map each node's arc-length coordinate in the ridge onto the bump
    s = t - t_in
    s_ridge = torch.minimum(s.clamp_min(0.0), arc).contiguous()
    idx = torch.searchsorted(s_k.contiguous(), s_ridge).clamp(1, num_samples - 1)
    s0, s1 = s_k.gather(-1, idx - 1), s_k.gather(-1, idx)
    frac = (s_ridge - s0) / (s1 - s0).clamp_min(1e-12)
    x = torch.lerp(x_k.gather(-1, idx - 1), x_k.gather(-1, idx), frac)
    z = torch.lerp(z_k.gather(-1, idx - 1), z_k.gather(-1, idx), frac)

    new_t = torch.where(s <= 0.0, t, torch.where(s < arc, t_in + x, t - (arc - width.unsqueeze(-1))))
    wrinkled = positions.clone()
    wrinkled[..., :2] += (new_t - t).unsqueeze(-1) * direction.unsqueeze(1)
    wrinkled[..., 2] += z
    return wrinkled


@sim_utils.clone
def spawn_creased_mesh_rectangle(
    prim_path: str,
    cfg: CreasedMeshRectangleCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a rectangle mesh that carries a ridge across one corner in its rest shape.

    The rectangle is refined as for :class:`~isaaclab.sim.MeshRectangleCfg`, and its vertices are then raised with :func:`corner_ridge_positions`.

    Returns:
        The created prim.
    """
    # imported at spawn time: pxr must not load before Kit starts, and trimesh pulls in scipy, whose BLAS
    # threads crash the fork Kit makes at startup
    import trimesh

    from isaaclab.sim.spawners.meshes.meshes import _refine_surface_mesh, _spawn_mesh_geom_from_mesh

    half_x, half_y = cfg.size[0] / 2, cfg.size[1] / 2
    vertices = np.array(
        [(-half_x, -half_y, 0.0), (half_x, -half_y, 0.0), (half_x, half_y, 0.0), (-half_x, half_y, 0.0)],
        dtype=np.float32,
    )

    mesh = _refine_surface_mesh(trimesh.Trimesh(vertices=vertices, faces=((0, 1, 2), (0, 2, 3)), process=False), cfg)

    positions = torch.from_numpy(np.asarray(mesh.vertices, dtype=np.float64)).unsqueeze(0)
    creased = corner_ridge_positions(
        positions,
        corner_signs=torch.tensor([cfg.ridge_corner], dtype=torch.float64),
        height=torch.tensor([cfg.ridge_height], dtype=torch.float64),
        width=torch.tensor([cfg.ridge_width], dtype=torch.float64),
        distance=torch.tensor([cfg.ridge_distance], dtype=torch.float64),
    )
    mesh = trimesh.Trimesh(vertices=creased[0].numpy(), faces=mesh.faces, process=False)

    stage = sim_utils.get_current_stage()
    # the mesh is already refined; the ridge would otherwise change the bounding box the refinement uses
    _spawn_mesh_geom_from_mesh(
        prim_path, cfg.replace(edge_refinement=1.0), mesh, translation, orientation, None, stage=stage
    )
    return stage.GetPrimAtPath(prim_path)


@configclass
class CreasedMeshRectangleCfg(sim_utils.MeshRectangleCfg):
    """Rectangle mesh with a ridge across one corner in its rest shape, see :func:`spawn_creased_mesh_rectangle`."""

    func: Callable = spawn_creased_mesh_rectangle

    ridge_corner: tuple[float, float] = (-1.0, 1.0)
    """Signs (+-1) of the creased corner along x and y. Defaults to the -x, +y corner."""

    ridge_height: float = 0.03
    """Ridge height above the sheet [m]."""

    ridge_width: float = 0.04
    """Footprint width of the ridge along the corner diagonal [m]."""

    ridge_distance: float = 0.065
    """Distance of the ridge center from the corner, along the diagonal [m]."""

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events for the Franka cloth dragging task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from isaaclab_newton.physics import NewtonManager

from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, DeformableObject
    from isaaclab.envs import ManagerBasedEnv

# Node sets of 120_nodes_optimized_mesh.usd that may be raised
# (plot: implementation_log/test_scripts/plot_raised_nodes.py)
# The interior of the refined U along the mesh's -x edge, without the border and the area around the central node 87
interior_nodes = [
    16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41, 43,
    44, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 64, 65, 68, 69, 72, 74, 76, 80,
]  # fmt: skip
# The interior of the U plus its border: the -x edge and the two side edges up to x = 0
edge_included = sorted(interior_nodes + [
    85, 73, 63, 51, 42, 25, 15,  # side edge at +y
    14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,  # -x edge
    20, 34, 48, 60, 71, 82,  # side edge at -y
])  # fmt: skip


def reset_franka_top_down(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    joint_offset_ranges: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset a Franka to a random arm pose with its hand still pointing the way it points in the default pose.

    All joints, the fingers included, are reset to their default positions with their default velocities, and offsets
    drawn uniformly from ``joint_offset_ranges`` [rad] are added to the default arm joint positions. With joints 3
    and 5 at zero, joints 2, 4 and 6 turn about parallel axes, and the hand's pitch is ``-q2 + q4 + q6`` plus a
    constant, so joint 6 takes up the pitch that the offsets of joints 2 and 4 add. Joints 1 and 7 only turn the hand
    about the vertical axis. Offsets of joints 3, 5, or 6 would tilt the hand, so they are rejected.

    Adapted from ``reset_franka_top_down`` on the ``non-prehensile-cloth`` branch.

    Args:
        env: The environment instance.
        env_ids: The environment indices to reset.
        joint_offset_ranges: Offset bounds [rad] keyed by the names of ``panda_joint1``, ``panda_joint2``,
            ``panda_joint4``, and ``panda_joint7``. Missing joints keep their default positions.
        asset_cfg: Scene entity of the Franka arm.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    tilting = set(joint_offset_ranges) & {"panda_joint3", "panda_joint5", "panda_joint6"}
    if tilting:
        raise ValueError(f"Offsets of {sorted(tilting)} would tilt the hand away from its default direction.")

    joint_pos = robot.data.default_joint_pos.torch[env_ids].clone()
    offset_ids, _ = robot.find_joints(list(joint_offset_ranges), preserve_order=True)
    ranges = torch.tensor(list(joint_offset_ranges.values()), device=joint_pos.device)
    offsets = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), len(offset_ids)), device=joint_pos.device)
    joint_pos[:, offset_ids] += offsets
    # keep the pitch of the hand: dq6 = dq2 - dq4
    (j2, j4, j6), _ = robot.find_joints(["panda_joint2", "panda_joint4", "panda_joint6"], preserve_order=True)
    default_pos = robot.data.default_joint_pos.torch[env_ids]
    joint_pos[:, j6] += (joint_pos[:, j2] - default_pos[:, j2]) - (joint_pos[:, j4] - default_pos[:, j4])

    limits = robot.data.soft_joint_pos_limits.torch[env_ids]
    joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])
    robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(
        velocity=robot.data.default_joint_vel.torch[env_ids].clone(), env_ids=env_ids
    )


def reset_cloth_planar_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: dict[str, tuple[float, float]],
    yaw_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> None:
    """Reset a cloth to its default shape, turned about the vertical axis and shifted in the table plane.

    The nodes of the default nodal state are rotated about the cloth's default center (the mean of its nodes) by a
    yaw drawn uniformly from ``yaw_range``, then shifted by an offset drawn uniformly from ``position_range``. The
    velocities are zero. The cloth keeps its shape, so the bending rest shape and all per-node quantities still apply.

    Args:
        env: The environment instance.
        env_ids: The environment indices to reset.
        position_range: Planar offset bounds [m] keyed by ``x`` and ``y``. Missing keys => no offset.
        yaw_range: Bounds of the rotation about the vertical axis [rad].
        asset_cfg: Scene entity of the cloth.
    """
    cloth: DeformableObject = env.scene[asset_cfg.name]
    nodal_state = _planar_pose_nodal_state(cloth, env_ids, position_range, yaw_range)
    cloth.write_nodal_state_to_sim_index(nodal_state, env_ids=env_ids)


class ResetClothWithRaisedNodes(ManagerTermBase):
    """Reset a cloth like :func:`reset_cloth_planar_pose` and raise a few nodes that keep sticking out.

    At every reset, ``num_raised_nodes`` nodes are drawn anew from :data:`interior_nodes`, or from
    :data:`edge_included` if ``include_edge_nodes``, in random order and at least ``min_spacing`` apart where
    possible, and start lifted by ``node_raise_height``. The triangles and bending edges around them take the lifted
    shape as their full rest shape (triangle rest shapes and areas, bending rest lengths and angles), and every other
    rest value is restored. The bump is stress-free, so it keeps its height and its neighbours stay flat. It adds a
    little material, so the cloth has local slack there, and it acts like a set crease: pressed flat, it springs
    back, and it travels with the cloth. The drawn nodes are kept, sorted, in :attr:`raised_node_ids`.

    The shape is built from the default nodes and then turned and shifted like :func:`reset_cloth_planar_pose`; the
    rest values do not change under that planar motion. The cloth must be the only particle body of each environment,
    and its physics manager must provide ``set_cloth_rest`` (see :mod:`..coupler`).
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Collect the cloth triangles and bending edges of every environment."""
        super().__init__(cfg, env)
        params = cfg.params
        cloth: DeformableObject = env.scene[params.get("asset_cfg", SceneEntityCfg("deformable")).name]
        num_nodes = cloth.max_sim_vertices_per_body
        node_set = edge_included if params.get("include_edge_nodes", False) else interior_nodes
        self._node_set = torch.tensor(node_set, dtype=torch.long, device=env.device)
        if ((self._node_set < 0) | (self._node_set >= num_nodes)).any():
            raise ValueError(f"The raised-node candidates must be cloth node ids in [0, {num_nodes}).")
        if not 0 <= params["num_raised_nodes"] <= len(self._node_set):
            raise ValueError(f"num_raised_nodes must be between 0 and the {len(self._node_set)} candidate nodes.")

        # default nodes about their mean, the frame the shape is built in
        rest_pos = cloth.data.default_nodal_state_w.torch[0, :, :3]
        self._rest_pos = rest_pos - rest_pos.mean(dim=0)
        candidate_xy = self._rest_pos[self._node_set, :2]
        self._candidate_dist = torch.cdist(candidate_xy, candidate_xy)
        self.raised_node_ids = torch.zeros(
            env.num_envs, params["num_raised_nodes"], dtype=torch.long, device=env.device
        )
        """Nodes raised in the current episode of every environment, sorted, shape [num_envs, num_raised_nodes]."""

        # the solver views copy the rest values, so they are set through the physics manager (the task's coupler)
        self._physics = env.sim.physics_manager
        if not hasattr(self._physics, "set_cloth_rest"):
            raise TypeError(f"The physics manager {self._physics.__name__} cannot set the cloth rest shape at runtime.")

        # model ids and local node ids of the bending edges (i, j, k, l with both opposite nodes) and the triangles
        model = NewtonManager.get_model()
        world = model.particle_world.numpy()
        first_particle = np.full(env.num_envs, np.iinfo(np.int64).max)
        in_env = world >= 0
        np.minimum.at(first_particle, world[in_env], np.flatnonzero(in_env))
        quads = model.edge_indices.numpy()
        bending = np.flatnonzero((quads[:, 0] >= 0) & (quads[:, 1] >= 0))
        self._edge_ids, self._quads = _cloth_elements_per_env(
            bending, quads[bending], world, first_particle, num_nodes, env.device
        )
        tris = model.tri_indices.numpy()
        self._tri_ids, self._tris = _cloth_elements_per_env(
            np.arange(len(tris)), tris, world, first_particle, num_nodes, env.device
        )
        # rest values the cloth was built with, read at the first reset once every model setup has run
        self._base_rest: dict[str, torch.Tensor] | None = None

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        position_range: dict[str, tuple[float, float]],
        yaw_range: tuple[float, float],
        include_edge_nodes: bool,
        num_raised_nodes: int,
        node_raise_height: float,
        min_spacing: float = 0.03,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    ) -> None:
        """Reset the cloth pose, draw the raised nodes and set the cloth rest shape around them.

        Args:
            env: The environment instance.
            env_ids: The environment indices to reset.
            position_range: Planar offset bounds [m] keyed by ``x`` and ``y``. Missing keys => no offset.
            yaw_range: Bounds of the rotation about the vertical axis [rad].
            include_edge_nodes: Whether the nodes on the U's border may be raised too (:data:`edge_included`), or only
                its interior (:data:`interior_nodes`).
            num_raised_nodes: Number of nodes raised in every episode.
            node_raise_height: Height of the raised nodes above the rest of the cloth [m].
            min_spacing: Minimum planar distance between raised nodes in the rest mesh [m].
            asset_cfg: Scene entity of the cloth.
        """
        cloth: DeformableObject = env.scene[asset_cfg.name]
        ids = {"edge_rest_angle": self._edge_ids, "edge_rest_length": self._edge_ids}
        ids |= {"tri_poses": self._tri_ids, "tri_areas": self._tri_ids}
        if self._base_rest is None:
            self._base_rest = {name: self._physics.get_cloth_rest(name, ids[name]) for name in ids}
        base = {name: values[env_ids] for name, values in self._base_rest.items()}
        rest_pos = self._rest_pos.expand(len(env_ids), -1, -1)
        raised = self._node_set[self._draw_spread(len(env_ids), num_raised_nodes, min_spacing)]
        self.raised_node_ids[env_ids] = raised
        is_raised = torch.zeros(len(env_ids), len(self._rest_pos), dtype=torch.bool, device=env.device)
        is_raised.scatter_(1, raised, True)
        shaped = rest_pos.clone()
        shaped[..., 2] += node_raise_height * is_raised

        # the elements around a raised node take the lifted shape as their rest shape, bending and in-plane
        angle_change = _dihedral_angles(shaped, self._quads) - _dihedral_angles(rest_pos, self._quads)
        rest = {"edge_rest_angle": base["edge_rest_angle"] + angle_change}
        tri_raised = is_raised[:, self._tris].any(dim=-1)
        poses, areas = _triangle_rest_shapes(shaped, self._tris)
        rest["tri_poses"] = torch.where(tri_raised[..., None, None], poses, base["tri_poses"])
        rest["tri_areas"] = torch.where(tri_raised, areas, base["tri_areas"])
        edge_raised = is_raised[:, self._quads[:, 2:]].any(dim=-1)
        lengths = (shaped[:, self._quads[:, 3]] - shaped[:, self._quads[:, 2]]).norm(dim=-1)
        rest["edge_rest_length"] = torch.where(edge_raised, lengths, base["edge_rest_length"])
        for name, values in rest.items():
            self._physics.set_cloth_rest(name, ids[name][env_ids], values)

        nodal_state = _planar_pose_nodal_state(cloth, env_ids, position_range, yaw_range, shaped - rest_pos)
        cloth.write_nodal_state_to_sim_index(nodal_state, env_ids=env_ids)

    def _draw_spread(self, num_envs: int, num_drawn: int, min_spacing: float) -> torch.Tensor:
        """Draw ``num_drawn`` candidate indices per environment, at least ``min_spacing`` [m] apart where possible.

        Candidates are visited in random order; the spacing is dropped for the rest if it cannot be met.

        Returns:
            Indices into the candidate nodes, sorted, shape [num_envs, num_drawn].
        """
        order = torch.rand(num_envs, len(self._node_set), device=self._node_set.device).argsort(dim=1)
        rows = torch.arange(num_envs, device=order.device)
        chosen = torch.zeros(num_envs, len(self._node_set), dtype=torch.bool, device=order.device)
        blocked = torch.zeros_like(chosen)
        for relax in (False, True):
            for candidate in order.T:
                take = (chosen.sum(dim=1) < num_drawn) & ~chosen[rows, candidate]
                if not relax:
                    take &= ~blocked[rows, candidate]
                chosen[rows, candidate] |= take
                blocked |= take.unsqueeze(1) & (self._candidate_dist[candidate] < min_spacing)
        return chosen.nonzero()[:, 1].view(num_envs, num_drawn)


def _cloth_elements_per_env(
    element_ids: np.ndarray,
    element_nodes: np.ndarray,
    world: np.ndarray,
    first_particle: np.ndarray,
    num_nodes: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group cloth elements (edges or triangles) of the model by environment.

    Args:
        element_ids: Model ids of the elements, shape [M].
        element_nodes: Global particle ids of their nodes, shape [M, K]; the last column is always a valid node.
        world: Environment of every particle, shape [num_particles].
        first_particle: First particle of every environment, shape [num_envs].
        num_nodes: Number of cloth nodes per environment.
        device: Device of the returned tensors.

    Returns:
        Model ids per environment, shape [num_envs, M / num_envs], and the local node ids, the same in every
        environment, shape [M / num_envs, K].
    """
    element_env = world[element_nodes[:, -1]]
    local_nodes = element_nodes - first_particle[element_env][:, None]
    per_env_ids, per_env_nodes = [], []
    for env_id in range(len(first_particle)):
        mask = element_env == env_id
        per_env_ids.append(element_ids[mask])
        per_env_nodes.append(local_nodes[mask])
        if not np.array_equal(per_env_nodes[-1], per_env_nodes[0]):
            raise RuntimeError("The cloth elements differ between environments.")
    if per_env_nodes[0].min(initial=0) < 0 or per_env_nodes[0].max(initial=0) >= num_nodes:
        raise RuntimeError("The cloth must be the only particle body of each environment.")
    return (
        torch.tensor(np.stack(per_env_ids), dtype=torch.long, device=device),
        torch.tensor(per_env_nodes[0], dtype=torch.long, device=device),
    )


def _triangle_rest_shapes(pos: torch.Tensor, tris: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse rest matrices and rest areas of triangles, as Newton computes them (``ModelBuilder.add_triangles``).

    Args:
        pos: Node positions [m], shape [N, num_nodes, 3].
        tris: Triangle node ids, shape [T, 3].

    Returns:
        Inverse rest matrices [1/m], shape [N, T, 2, 2], and rest areas [m^2], shape [N, T].
    """
    p, q, r = (pos[:, tris[:, col]] for col in range(3))
    qp, rp = q - p, r - p
    normal = torch.nn.functional.normalize(torch.cross(qp, rp, dim=-1), dim=-1)
    e1 = torch.nn.functional.normalize(qp, dim=-1)
    e2 = torch.nn.functional.normalize(torch.cross(normal, e1, dim=-1), dim=-1)
    rest = torch.stack(
        [
            torch.stack([(e1 * qp).sum(dim=-1), (e1 * rp).sum(dim=-1)], dim=-1),
            torch.stack([(e2 * qp).sum(dim=-1), (e2 * rp).sum(dim=-1)], dim=-1),
        ],
        dim=-2,
    )
    return torch.linalg.inv(rest), torch.linalg.det(rest) / 2.0


def _planar_pose_nodal_state(
    cloth: DeformableObject,
    env_ids: torch.Tensor,
    position_range: dict[str, tuple[float, float]],
    yaw_range: tuple[float, float],
    displacement: torch.Tensor | None = None,
) -> torch.Tensor:
    """Default nodal state, optionally displaced, turned about the cloth center and shifted in the plane.

    Args:
        cloth: The cloth.
        env_ids: The environment indices.
        position_range: Planar offset bounds [m] keyed by ``x`` and ``y``. Missing keys => no offset.
        yaw_range: Bounds of the rotation about the vertical axis [rad].
        displacement: Displacements of the default nodes before the turn [m], shape [len(env_ids), num_nodes, 3].

    Returns:
        Nodal state with zero velocities [m, m/s], shape [len(env_ids), num_nodes, 6].
    """
    nodal_state = cloth.data.default_nodal_state_w.torch[env_ids].clone()
    device = nodal_state.device
    num = len(env_ids)

    yaw = sample_uniform(yaw_range[0], yaw_range[1], (num, 1), device=device)
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    center_xy = nodal_state[..., :2].mean(dim=1, keepdim=True)
    if displacement is not None:
        nodal_state[..., :3] += displacement
    rel = nodal_state[..., :2] - center_xy
    rotated = torch.stack([cos * rel[..., 0] - sin * rel[..., 1], sin * rel[..., 0] + cos * rel[..., 1]], dim=-1)

    ranges = torch.tensor([position_range.get(key, (0.0, 0.0)) for key in ("x", "y")], device=device)
    offset = sample_uniform(ranges[:, 0], ranges[:, 1], (num, 2), device=device)
    nodal_state[..., :2] = center_xy + rotated + offset.unsqueeze(1)
    nodal_state[..., 3:] = 0.0
    return nodal_state


def _dihedral_angles(pos: torch.Tensor, quads: torch.Tensor) -> torch.Tensor:
    """Signed dihedral angles [rad] across bending edges, as Newton computes rest angles (``ModelBuilder.add_edges``).

    Args:
        pos: Node positions [m], shape [N, num_nodes, 3].
        quads: Edge node ids (i, j, k, l): edge k-l, opposite nodes i and j, shape [E, 4].

    Returns:
        Angles [rad], shape [N, E].
    """
    x1, x2, x3, x4 = (pos[:, quads[:, col]] for col in range(4))
    n1 = torch.nn.functional.normalize(torch.cross(x3 - x1, x4 - x1, dim=-1), dim=-1)
    n2 = torch.nn.functional.normalize(torch.cross(x4 - x2, x3 - x2, dim=-1), dim=-1)
    edge = torch.nn.functional.normalize(x4 - x3, dim=-1)
    cos = (n1 * n2).sum(dim=-1).clamp(-1.0, 1.0)
    sin = (torch.cross(n1, n2, dim=-1) * edge).sum(dim=-1)
    return torch.atan2(sin, cos)

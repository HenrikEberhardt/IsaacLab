# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the cloth rest-shape helpers of the Franka cloth dragging reset event. No Kit/GPU required."""

import newton
import numpy as np
import torch
import warp as wp

from isaaclab_tasks.contrib.franka_cloth_drag.mdp.events import _dihedral_angles, _triangle_rest_shapes


def _grid(size: int, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Flat square grid of ``size`` x ``size`` nodes in the xy-plane, centered at the origin, and its triangles."""
    coords = (np.arange(size) - (size - 1) / 2) * spacing
    x, y = np.meshgrid(coords, coords, indexing="ij")
    vertices = np.stack([x.ravel(), y.ravel(), np.zeros(size * size)], axis=-1)
    tris = []
    for i in range(size - 1):
        for j in range(size - 1):
            a, b, c, d = i * size + j, (i + 1) * size + j, (i + 1) * size + j + 1, i * size + j + 1
            tris += [(a, b, c), (a, c, d)]
    return vertices, np.array(tris)


def test_rest_shape_helpers_match_newton_builder():
    """Rest angles, rest lengths and triangle rest shapes match what Newton computes for a lifted, uneven cloth."""
    vertices, tris = _grid(5, 0.02)
    rng = np.random.default_rng(0)
    vertices[:, :2] += rng.uniform(-0.003, 0.003, (len(vertices), 2))
    vertices[12, 2] = 0.012  # raised center node: the rest angles around it must point up, not down
    vertices[3, 2] = -0.004

    builder = newton.ModelBuilder()
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=[wp.vec3(*v) for v in vertices],
        indices=tris.ravel().tolist(),
        density=0.2,
    )
    pos = torch.tensor(np.array(builder.particle_q), dtype=torch.float64).unsqueeze(0)

    quads = np.array(builder.edge_indices)
    bending = (quads[:, 0] >= 0) & (quads[:, 1] >= 0)
    quads_t = torch.tensor(quads[bending])
    angles = _dihedral_angles(pos, quads_t)[0].numpy()
    np.testing.assert_allclose(angles, np.array(builder.edge_rest_angle)[bending], atol=1e-6)
    assert angles.max() > 0.1  # the raised node bends the cloth
    lengths = (pos[0, quads_t[:, 3]] - pos[0, quads_t[:, 2]]).norm(dim=-1).numpy()
    np.testing.assert_allclose(lengths, np.array(builder.edge_rest_length)[bending], rtol=1e-6)

    poses, areas = _triangle_rest_shapes(pos, torch.tensor(np.array(builder.tri_indices)))
    np.testing.assert_allclose(poses[0].numpy(), np.array(builder.tri_poses), rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(areas[0].numpy(), np.array(builder.tri_areas), rtol=1e-5)

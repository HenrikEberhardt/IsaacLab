# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU tests for the Franka non-prehensile cloth crease geometry."""

import torch

from isaaclab_tasks.contrib.franka_nonp_cloth.cloth_mesh import corner_ridge_positions

# flat 0.40 m x 0.30 m sheet on a 1 mm grid, centered away from the origin
_SIZE = (0.40, 0.30)
_SPACING = 0.001
_CENTER = (0.45, -0.1, 0.004)


def _flat_sheet(num_envs: int) -> tuple[torch.Tensor, int, int]:
    """Return a batch of identical flat sheets, shape [E, NX * NY, 3], and the grid dimensions."""
    nx, ny = round(_SIZE[0] / _SPACING) + 1, round(_SIZE[1] / _SPACING) + 1
    xs = torch.linspace(-0.5 * _SIZE[0], 0.5 * _SIZE[0], nx, dtype=torch.float64)
    ys = torch.linspace(-0.5 * _SIZE[1], 0.5 * _SIZE[1], ny, dtype=torch.float64)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
    points = torch.stack((grid_x, grid_y, torch.zeros_like(grid_x)), dim=-1).reshape(-1, 3)
    points = points + torch.tensor(_CENTER, dtype=torch.float64)
    return points.expand(num_envs, -1, -1).clone(), nx, ny


def _ridge_inputs() -> dict[str, torch.Tensor]:
    """Return ridge parameters for the two corners facing the robot."""
    return {
        "corner_signs": torch.tensor([[-1.0, 1.0], [-1.0, -1.0]], dtype=torch.float64),
        "height": torch.tensor([0.03, 0.02], dtype=torch.float64),
        "width": torch.tensor([0.05, 0.04], dtype=torch.float64),
        "distance": torch.tensor([0.07, 0.05], dtype=torch.float64),
    }


def _diagonal_coordinate(points: torch.Tensor, corner_signs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return each point's coordinate along the center-to-corner diagonal and the corner's coordinate."""
    corner = corner_signs * torch.tensor(_SIZE, dtype=torch.float64) * 0.5
    corner_dist = corner.norm(dim=-1)
    direction = corner / corner_dist.unsqueeze(-1)
    local_xy = points[..., :2] - torch.tensor(_CENTER[:2], dtype=torch.float64)
    return (local_xy * direction.unsqueeze(1)).sum(dim=-1), corner_dist


def test_ridge_preserves_edge_lengths():
    """Neighboring grid nodes keep their spacing, so the ridge bends the sheet without stretching it."""
    inputs = _ridge_inputs()
    flat, nx, ny = _flat_sheet(2)
    wrinkled = corner_ridge_positions(flat, **inputs).reshape(2, nx, ny, 3)

    for axis in (1, 2):
        lengths = wrinkled.diff(dim=axis).norm(dim=-1)
        torch.testing.assert_close(lengths, torch.full_like(lengths, _SPACING), rtol=0.02, atol=0.0)


def test_ridge_leaves_center_side_unchanged():
    """Nodes between the sheet center and the ridge, including under the cube, do not move."""
    inputs = _ridge_inputs()
    flat, _, _ = _flat_sheet(2)
    wrinkled = corner_ridge_positions(flat, **inputs)

    t, corner_dist = _diagonal_coordinate(flat, inputs["corner_signs"])
    ridge_start = (corner_dist - inputs["distance"] - 0.5 * inputs["width"]).unsqueeze(-1)
    center_side = t < ridge_start
    assert center_side.float().mean() > 0.9
    torch.testing.assert_close(wrinkled[center_side], flat[center_side], rtol=0.0, atol=1e-12)


def test_ridge_peak_height_and_location():
    """The ridge peaks at the requested height, at the requested distance from the chosen corner."""
    inputs = _ridge_inputs()
    flat, _, _ = _flat_sheet(2)
    wrinkled = corner_ridge_positions(flat, **inputs)

    rise = wrinkled[..., 2] - flat[..., 2]
    torch.testing.assert_close(rise.amax(dim=-1), inputs["height"], rtol=0.0, atol=1e-4)

    peak = rise.argmax(dim=-1)
    t_wrinkled, corner_dist = _diagonal_coordinate(wrinkled, inputs["corner_signs"])
    peak_t = t_wrinkled.gather(-1, peak.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(peak_t, corner_dist - inputs["distance"], rtol=0.0, atol=2 * _SPACING)

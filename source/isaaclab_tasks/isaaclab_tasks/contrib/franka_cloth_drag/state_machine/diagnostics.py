# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional diagnostics for the cloth-drag state machine."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_apply, quat_apply_inverse

from .cloth_drag_sm import DragSmState

if TYPE_CHECKING:
    from .cloth_drag_sm_runner import ClothDragSmRunner


def ee_down_tilt(ee_quat_b: torch.Tensor) -> torch.Tensor:
    """Return EE-frame tilt away from base-frame down [rad], shape [N].

    A zero value means the EE frame's local positive z-axis points along the
    base frame's negative z-axis. Rotation about that axis does not affect the
    result.

    Args:
        ee_quat_b: EE orientations in the robot base frame (x, y, z, w), shape [N, 4].
    """
    local_z = ee_quat_b.new_tensor((0.0, 0.0, 1.0)).expand(len(ee_quat_b), 3)
    z_axis_b = quat_apply(ee_quat_b, local_z)
    down_alignment = -z_axis_b[:, 2]
    return torch.acos(down_alignment.clamp(-1.0, 1.0))


def pad_cloth_gaps(
    ee_pos_w: torch.Tensor,
    ee_quat_w: torch.Tensor,
    nodes_w: torch.Tensor,
    opening: torch.Tensor,
    pad_shift: torch.Tensor,
    pad_half_width: float,
    pad_thickness: float,
    pad_half_length: float,
    layer_half_thickness: float,
) -> torch.Tensor:
    """Return finger-pad gaps to the cloth surface [m], shape [N, 2]."""
    quat = ee_quat_w.unsqueeze(1).expand(-1, nodes_w.shape[1], -1)
    rel = quat_apply_inverse(quat, nodes_w - ee_pos_w.unsqueeze(1))
    surface_gap = rel[..., 2] - layer_half_thickness - pad_half_length - pad_shift.unsqueeze(1)
    in_width = rel[..., 0].abs() <= pad_half_width
    inner = opening.unsqueeze(1)
    gaps = []
    for side in (1.0, -1.0):
        lateral = side * rel[..., 1]
        under = in_width & (lateral >= inner) & (lateral <= inner + pad_thickness)
        gaps.append(torch.where(under, surface_gap, torch.inf).amin(dim=1))
    gaps = torch.stack(gaps, dim=1)
    return torch.where(torch.isinf(gaps), torch.nan, gaps)


class ClothDragSmDiagnostics:
    """Print state transitions, failed attempts, and EE down-alignment."""

    def __init__(self, tilt_report_interval: int = 30, tilt_warn_deg: float = 2.0):
        """Configure periodic EE-tilt reporting.

        Args:
            tilt_report_interval: Controller steps between tilt summaries. Set to zero to disable summaries.
            tilt_warn_deg: Tilt from base-frame down that counts as misaligned [deg].
        """
        if tilt_report_interval < 0:
            raise ValueError("tilt_report_interval must be non-negative.")
        self.tilt_report_interval = tilt_report_interval
        self.tilt_warn_deg = tilt_warn_deg
        self._step = 0

    def report(self, runner: ClothDragSmRunner):
        """Report controller transitions from the last computation."""
        self._step += 1
        state = runner.sm.sm_state
        changed = runner.previous_state != state
        for env_id in changed.nonzero(as_tuple=False).squeeze(-1).tolist():
            previous = DragSmState(int(runner.previous_state[env_id])).name
            current = DragSmState(int(state[env_id])).name
            print(f"[INFO]: env {env_id}: {previous} -> {current}")
        failed = changed & (state == int(DragSmState.FAILED))
        for env_id in failed.nonzero(as_tuple=False).squeeze(-1).tolist():
            print(f"[WARN]: env {env_id}: state-machine attempt failed; waiting for the episode reset.")
        if self.tilt_report_interval and self._step % self.tilt_report_interval == 0:
            tilt_deg = torch.rad2deg(ee_down_tilt(runner.ee_quat))
            misaligned = tilt_deg > self.tilt_warn_deg
            print(
                f"[INFO]: EE down tilt: mean={tilt_deg.mean():.3f} deg, max={tilt_deg.max():.3f} deg,"
                f" above {self.tilt_warn_deg:g} deg={int(misaligned.sum())}/{len(tilt_deg)}"
            )

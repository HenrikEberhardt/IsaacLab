# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# follows srl-ethz/IsaacLab-Cloth (cloth-fold-environments), src/Cloth_Manipulation/visual/mdp.py

"""LEAPP export detection: sim-only randomization in the student's terms is skipped while the policy is traced."""

from __future__ import annotations

import functools


@functools.cache
def _leapp_export_manager():
    """LEAPP's ExportManager class, or None without the optional LEAPP dependency (imported once)."""
    try:
        from leapp.export_manager import ExportManager
    except ImportError:
        return None
    return ExportManager


def leapp_exporting() -> bool:
    """Whether a LEAPP export is tracing the policy."""
    manager = _leapp_export_manager()
    return manager is not None and manager.is_interpret_graph_enabled()

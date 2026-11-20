# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of the Newton coupler manager of the Franka cloth dragging task."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from isaaclab_contrib.coupling import CouplerProxyCfg

if TYPE_CHECKING:
    from .coupler import NewtonClothDragCouplerManager


@configclass
class ClothDragCouplerCfg(CouplerProxyCfg):
    """Proxy coupler of the cloth dragging task, see :class:`~.coupler.NewtonClothDragCouplerManager`."""

    class_type: type[NewtonClothDragCouplerManager] | str = "{DIR}.coupler:NewtonClothDragCouplerManager"
    """Coupler implementation class."""

    collision_filter_pairs: list[tuple[str, str]] = []
    """Pairs of prim-path regexes whose shapes never collide with each other within a world."""

    monitored_contacts: dict[str, tuple[str, str]] = {}
    """Named pairs of prim-path regexes whose touching is latched per world, seen only by a proxy pipeline."""

    contact_touch_tolerance: float = 1.0e-3
    """Largest gap [m] between two contact surfaces, after their margins, that counts as touching."""

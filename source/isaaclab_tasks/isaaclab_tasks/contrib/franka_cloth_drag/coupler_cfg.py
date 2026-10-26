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
    from .coupler import NewtonClothGraspCouplerManager


@configclass
class ClothGraspCouplerProxyCfg(CouplerProxyCfg):
    """Proxy coupler whose manager supports the constraint cloth grasp.

    See :class:`~.coupler.NewtonClothGraspCouplerManager`.
    """

    class_type: type[NewtonClothGraspCouplerManager] | str = "{DIR}.coupler:NewtonClothGraspCouplerManager"
    """Coupler implementation class."""

    rest_angle_scale: float = 0.0
    """Share of the cloth's authored bending rest angles to restore. Defaults to 0.0 (flat rest shape).

    The Newton deformable asset zeroes the bending rest angles, so bending pulls every fold of a spawned cloth flat. A
    positive value keeps that share of the folds of the spawned mesh as rest shape, like a partly set crease, which
    keeps the wrinkles of a light cloth from flattening.
    """

    collision_filter_pairs: list[tuple[str, str]] = []
    """Pairs of label regexes whose shapes never collide with each other within a world. Defaults to none.

    A shape matches a regex if its label or its body's label (the USD prim paths) matches in full, e.g.
    ``(r"/World/envs/env_[^/]+/Cube", r"/World/envs/env_[^/]+/Table(/.*)?")``. Every regex must match a shape.
    """

    monitored_contacts: dict[str, tuple[str, str]] = {}
    """Named pairs of label regexes (matched like :attr:`collision_filter_pairs`) whose touching is latched per world
    after every substep, see :meth:`~.coupler.NewtonClothGraspCouplerManager.contact_latch`. Defaults to none.

    Only contacts that a proxy's own collision pipeline detects are seen, so both sides must collide in that proxy's
    destination entry.
    """

    contact_touch_tolerance: float = 1.0e-3
    """Largest gap [m] between two contact surfaces, after their margins, that counts as touching."""

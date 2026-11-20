# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton physics of the Franka cloth dragging task: the MJWarp robot coupled to the VBD cloth and cube.

* :mod:`.coupler` / :mod:`.coupler_cfg`: :class:`NewtonClothDragCouplerManager` and its configuration
  :class:`ClothDragCouplerCfg`.
"""

from isaaclab.utils.module import lazy_export

lazy_export()

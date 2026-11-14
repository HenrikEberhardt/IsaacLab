# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Uniform and adaptive reset strategies of the Franka cloth dragging task: training starts from stored reset states.

* :mod:`.reset_from_bank`: the reset event :class:`ResetFromBank`, run at every reset: it draws, restores and logs.
* :mod:`.reset_from_bank_cfg`: its configuration :class:`ResetFromBankCfg`, which the env cfg fills in.
* :mod:`.bank`: the stored states (:class:`ResetBank`), their file and the task fingerprint they are checked against.
* :mod:`.restore`: writes stored states into environments.
* :mod:`.stage_ratios`: the ratios of the start stages, fixed (uniform resets) or adapted to the success (adaptive).
* :mod:`.frontier`: success-guided sampling of the states inside a stage, an option of both strategies.

The bank is made offline, see :mod:`..reset_bank_generation`.
"""

from isaaclab.utils.module import lazy_export

lazy_export()

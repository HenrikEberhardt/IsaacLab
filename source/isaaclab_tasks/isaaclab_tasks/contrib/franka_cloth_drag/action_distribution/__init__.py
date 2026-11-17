# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action distribution of the Franka cloth dragging policy: bounded standard deviation and mean, for rsl_rl.

* :mod:`.distributions`: the sigmoid-bounded Gaussians, with one standard deviation per action dimension
  (:class:`SigmoidBoundedGaussianDistribution`) or one per state (:class:`SigmoidBoundedHeteroscedasticGaussianDistribution`),
  and the bounds loss that keeps the mean inside the action range.
* :mod:`.distributions_cfg`: their configurations, and the one of :class:`KLMetricsPPO`, which the runner configuration uses.
* :mod:`.models`: the actor model :class:`DetachedStdMLPModel`, whose standard-deviation head reads detached features.
* :mod:`.ppo`: :class:`KLMetricsPPO`, which logs the adaptive-scheduler KL and the standard deviation per stage.
"""

from isaaclab.utils.module import lazy_export

lazy_export()

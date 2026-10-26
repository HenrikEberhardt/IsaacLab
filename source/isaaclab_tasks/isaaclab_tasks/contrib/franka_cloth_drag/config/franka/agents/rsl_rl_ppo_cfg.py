# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configuration for Franka cloth dragging."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class BoundedGaussianDistributionCfg(RslRlMLPModelCfg.GaussianDistributionCfg):
    """Gaussian action distribution whose standard deviation is clamped to a range."""

    std_range: tuple[float, float] = (1.0e-6, 1.0e6)
    """Range (min, max) the standard deviation is clamped to, passed to rsl_rl's ``GaussianDistribution``."""


@configclass
class FrankaClothDragPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Cloth-drag policy with five-frame actor inputs and a privileged critic."""

    seed = 42
    experiment_name = "cloth_drag_task"
    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 50
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    clip_actions = 1.0

    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        # Std floor keeps exploring (0.001 entropy let it collapse to 0.15), ceiling at the action clip avoids
        # bang-bang training (0.006 entropy grew it to 3-6 past the clip)
        distribution_cfg=BoundedGaussianDistributionCfg(init_std=0.5, std_type="scalar", std_range=(0.2, 1.0)),
    )

    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,  # exploration as in the lift tasks; the std range bounds its growth
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

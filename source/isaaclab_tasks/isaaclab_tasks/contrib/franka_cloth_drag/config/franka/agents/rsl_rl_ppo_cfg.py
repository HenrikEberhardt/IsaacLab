# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configuration for Franka cloth dragging."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg

from ....action_distribution import (
    ACTION_DIM_NAMES,
    BoundedGaussianDistributionCfg,
    BoundedHeteroscedasticGaussianDistributionCfg,
    DetachedStdMLPModelCfg,
    KLMetricsPPOCfg,
)


@configclass
class FrankaClothDragPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Non-Prehensile Cloth Drag Task"""

    seed = 42
    experiment_name = "cloth_drag_task"
    num_steps_per_env = 24
    max_iterations = 500
    save_interval = 50
    logger = "wandb"
    wandb_project = "non-prehensile_cloth_task"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}

    # Action clipping setup
    clip_actions = None
    action_dim_names = list(ACTION_DIM_NAMES)

    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128, 64],
        activation="elu",
        obs_normalization=True,
        # The bounds loss pulls a saturated mean back toward the nominal range
        distribution_cfg=BoundedGaussianDistributionCfg(
            init_std=0.5,
            std_type="scalar",
            std_range=(0.05, 1.0),
            mean_bound=1.0,
            mean_bound_coef=0.01,
            # dx, dy, dz, dyaw, gripper: the binary gripper toggles on every sign flip and no penalty acts on its
            # action, so without a lower ceiling its std grows and flips the gripper at random
            std_range_per_dim=[(0.05, 1.0)] * 4 + [(0.05, 0.35)],
            init_std_per_dim=[0.5] * 4 + [0.3],
        ),
    )

    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128, 64],
        activation="elu",
        obs_normalization=True,
    )

    algorithm = KLMetricsPPOCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=4,
        num_mini_batches=8,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.015,
        max_grad_norm=1.0,
    )


@configclass
class FrankaClothDragDetachedStdPPORunnerCfg(FrankaClothDragPPORunnerCfg):
    """Cloth drag with a learned, state-dependent action std from a head on detached actor features.

    Differs from :class:`FrankaClothDragPPORunnerCfg` only in the actor's standard deviation.
    """

    actor = DetachedStdMLPModelCfg(
        hidden_dims=[512, 256, 128, 64],
        activation="elu",
        obs_normalization=True,
        detach_std_features=True,
        distribution_cfg=BoundedHeteroscedasticGaussianDistributionCfg(
            init_std=0.5,
            std_type="log",
            std_range=(0.05, 1.0),
            mean_bound=1.0,
            mean_bound_coef=0.01,
            # dx, dy, dz, dyaw, gripper: the binary gripper toggles on every sign flip and no penalty acts on its
            # action, so without a lower ceiling its std grows and flips the gripper at random
            std_range_per_dim=[(0.05, 1.0)] * 4 + [(0.05, 0.35)],
            init_std_per_dim=[0.5] * 4 + [0.3],
        ),
    )

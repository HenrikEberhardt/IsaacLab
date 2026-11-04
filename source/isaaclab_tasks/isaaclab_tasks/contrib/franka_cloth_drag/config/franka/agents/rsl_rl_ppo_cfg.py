# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configuration for Franka cloth dragging."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class BoundedGaussianDistributionCfg(RslRlMLPModelCfg.GaussianDistributionCfg):
    """Gaussian with one smoothly bounded, learned standard deviation per action dimension."""

    class_name: str = "isaaclab_tasks.contrib.franka_cloth_drag.distributions:SigmoidBoundedGaussianDistribution"

    std_range: tuple[float, float] = (0.05, 1.0)
    """Bounds (min, max) of the standard deviation."""

    mean_bound: float = 1.0
    """Magnitude of the mean action above which the bounds loss acts."""

    mean_bound_coef: float = 0.0
    """Weight of the bounds loss pulling the mean action back inside ``mean_bound``. Zero disables it."""


@configclass
class FrankaClothDragPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Cloth-drag policy with five-frame actor inputs and a privileged critic."""

    seed = 42
    experiment_name = "cloth_drag_task"
    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 50
    logger = "wandb"
    wandb_project = "non-prehensile_cloth_task"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    # No runner clip: PPO would learn from samples the environment never executed, which pushes the mean past the
    # bound. The action penalties and the bounds loss keep actions near the nominal range, whose excess is only logged.
    clip_actions = None
    action_clip_diagnostics = 1.0
    # Names of the delta EE action dimensions in the action diagnostics (the joint_delta preset falls back to indices)
    action_dim_names = ["dx", "dy", "dz", "dyaw", "gripper"]

    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        # The bounds loss pulls a saturated mean back toward the nominal range
        distribution_cfg=BoundedGaussianDistributionCfg(
            init_std=0.5, std_type="scalar", std_range=(0.05, 1.0), mean_bound=1.0, mean_bound_coef=0.01
        ),
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
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

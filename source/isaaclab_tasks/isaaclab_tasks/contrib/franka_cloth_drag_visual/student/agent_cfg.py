# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL DAgger runner: frozen teacher actor to CNN student (values from Joran's cloth visual distillation)."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlCNNModelCfg, RslRlDistillationRunnerCfg, RslRlMLPModelCfg

from ..action_mapping.distillation import MappedDistillationAlgorithmCfg
from ..franka_cloth_drag_visual_env_cfg import ACTION_MAPPING_CFG
from ..teacher import TEACHER_ACTOR_CFG
from . import model


@configclass
class FrankaClothDragDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Train with --checkpoint set to the teacher's PPO checkpoint (teacher_cfg.TEACHER_CHECKPOINT)."""

    seed = 42
    experiment_name = "cloth_drag_distillation"
    num_steps_per_env = 64
    max_iterations = 750
    save_interval = 25
    logger = "wandb"
    wandb_project = "non-prehensile_cloth_task"
    clip_actions = 20.0
    action_dim_names = [f"joint{i}" for i in range(1, 8)] + ["gripper"]
    obs_groups = {"student": ["student_proprio", "student_image"], "teacher": ["teacher"]}

    student = RslRlCNNModelCfg(
        class_name=f"{model.__name__}:Uint8ImageCNNModel",
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.05),  # fixed DAgger exploration noise
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[32, 64, 64, 64],
            kernel_size=[5, 3, 3, 3],
            stride=2,
            padding="zeros",
            activation="elu",
            flatten=True,  # 15x120x160 -> 64x8x10
        ),
    )
    teacher = TEACHER_ACTOR_CFG

    algorithm = MappedDistillationAlgorithmCfg(
        num_learning_epochs=4,
        learning_rate=5.0e-4,
        gradient_length=16,
        max_grad_norm=1.0,
        loss_type="mse",
        mapping_cfg=ACTION_MAPPING_CFG,
    )

    def __post_init__(self):
        # the LEAPP exporter only traces the student observations
        self.obs_groups["actor"] = self.obs_groups["student"]

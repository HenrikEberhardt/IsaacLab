# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL DAgger configuration: IK teacher to joint-delta RGB student (values from Joran's cloth visual distillation)."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlCNNModelCfg, RslRlDistillationRunnerCfg, RslRlMLPModelCfg

from ..teacher_cfg import IK_LABEL_CFG, TEACHER_ACTOR_CFG, TEACHER_NUM_ACTIONS
from .distillation import IKTeacherDistillationAlgorithmCfg


@configclass
class FrankaClothDragDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """The student acts with 7 joint deltas and the gripper; the teacher labels every visited state.

    Train with ``--checkpoint`` pointing at the teacher's PPO checkpoint (``teacher_cfg.TEACHER_CHECKPOINT``).
    """

    seed = 42
    experiment_name = "cloth_drag_distillation"
    num_steps_per_env = 64
    max_iterations = 1000
    save_interval = 50
    logger = "wandb"
    wandb_project = "non-prehensile_cloth_task"
    clip_actions = 20.0
    action_dim_names = [f"joint{i}" for i in range(1, 8)] + ["gripper"]
    obs_groups = {"student": ["student_proprio", "student_image"], "teacher": ["teacher"]}

    # the CNN encodes the image group; the MLP head maps [CNN latent, proprioception] to the actions
    student = RslRlCNNModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.05),
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[32, 64, 64, 64],
            kernel_size=[5, 3, 3, 3],
            stride=2,
            padding="zeros",
            activation="elu",
            flatten=True,
        ),
    )
    teacher = TEACHER_ACTOR_CFG

    algorithm = IKTeacherDistillationAlgorithmCfg(
        num_learning_epochs=4,
        learning_rate=5.0e-4,
        gradient_length=16,
        max_grad_norm=1.0,
        loss_type="mse",
        teacher_num_actions=TEACHER_NUM_ACTIONS,
        label_cfg=IK_LABEL_CFG,
    )

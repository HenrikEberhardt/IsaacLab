# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.core.lift.config.franka_soft.agents.rsl_rl_ppo_cfg import FrankaClothPPORunnerCfg


@configclass
class FrankaClothDragPPORunnerCfg(FrankaClothPPORunnerCfg):
    experiment_name = "franka_cloth_drag"

    # the observation preset (env.observations=joint_symm|joint_asymm|ee_symm|ee_asymm) decides what the critic group
    # contains
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}

    # For a recurrent policy instead of the 5-step observation history, set history_length = 0 in the observation
    # groups and use isaaclab_rl.rsl_rl.RslRlRNNModelCfg for the actor and critic, for example
    # RslRlRNNModelCfg(rnn_type="lstm", rnn_hidden_dim=256, rnn_num_layers=1, hidden_dims=[256, 128, 64],
    # activation="elu", obs_normalization=True) with rnn_type="lstm" or "gru".

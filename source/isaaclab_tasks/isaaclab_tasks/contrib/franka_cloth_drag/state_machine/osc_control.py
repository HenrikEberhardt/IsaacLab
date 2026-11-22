# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Roll/pitch operational-space stabilization used only by the cloth-drag state-machine tools."""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg

from ..mdp import TopDownDeltaEEActionCfg, TopDownStabilizedDeltaEEActionCfg


def configure_state_machine_osc(env_cfg: ManagerBasedRLEnvCfg):
    """Add bounded roll/pitch operational-space stabilization to the scripted arm action.

    The registered task configuration is not changed. Call this function on the resolved configuration used by a
    state-machine tool before constructing its environment.

    Args:
        env_cfg: Resolved cloth-drag environment configuration, changed in place.
    """
    action_cfg = env_cfg.actions.arm_action
    if not isinstance(action_cfg, TopDownDeltaEEActionCfg):
        raise TypeError("State-machine OSC requires the task's TopDownDeltaEEActionCfg arm action.")

    env_cfg.actions.arm_action = TopDownStabilizedDeltaEEActionCfg(
        asset_name=action_cfg.asset_name,
        joint_names=action_cfg.joint_names,
        controller=action_cfg.controller,
        ee_frame_name=action_cfg.ee_frame_name,
        scale_trans=action_cfg.scale_trans,
        scale_rot=action_cfg.scale_rot,
        wrist_joint_limits=action_cfg.wrist_joint_limits,
        # the policy's action has no limiters (it learns the safety stops); the scripted demonstrator keeps them
        joint_vel_factor=0.9,
        joint_range_margin=0.05,  # [rad]
    )

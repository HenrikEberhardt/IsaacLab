# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the cloth-drag state machine with fixed Viser output on port 8760."""

import argparse
import sys

import gymnasium as gym
import torch
from isaaclab_visualizers.viser import ViserVisualizerCfg

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.franka_cloth_drag.config.franka.franka_cloth_drag_env_cfg import DefaultResetsCfg
from isaaclab_tasks.contrib.franka_cloth_drag.mdp import TopDownStabilizedDeltaEEActionCfg
from isaaclab_tasks.contrib.franka_cloth_drag.state_machine import (
    PROFILE_NAMES,
    ClothDragSmDiagnostics,
    ClothDragSmRunner,
    DragClothSm,
    configure_state_machine_osc,
    make_sm_profile,
    sample_sm_params,
)
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

TASK_ID = "IsaacContrib-Drag-Cloth-Franka"
VISER_PORT = 8760

parser = argparse.ArgumentParser(description="Run the Franka cloth-drag state machine.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--num_steps", type=int, default=None, help="Steps to run; defaults to running until Ctrl+C.")
parser.add_argument("--seed", type=int, default=0, help="Random seed.")
parser.add_argument("--profile", choices=PROFILE_NAMES, default="nominal", help="State-machine profile.")
add_launcher_args(parser)
parser.set_defaults(device=None)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra_args]


def main():
    env_cfg, _ = resolve_task_config(TASK_ID, "")
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    args_cli.device = env_cfg.sim.device
    args_cli.visualizer = ["viser"]
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.events = DefaultResetsCfg()
    configure_state_machine_osc(env_cfg)
    env_cfg.sim.visualizer_cfgs = [
        ViserVisualizerCfg(
            eye=(3.68, -3.18, 7.59),
            lookat=(0.5, 0.0, -0.2),
            focal_length=23.5,
            port=VISER_PORT,
        )
    ]
    if not isinstance(env_cfg.actions.arm_action, TopDownStabilizedDeltaEEActionCfg):
        raise SystemExit("The state machine requires its roll/pitch-stabilized arm action.")

    sm_cfg, randomization = make_sm_profile(args_cli.profile)
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(TASK_ID, cfg=env_cfg)
        unwrapped = env.unwrapped
        env.reset(seed=args_cli.seed)

        generator = torch.Generator(device=unwrapped.device).manual_seed(args_cli.seed)
        params = sample_sm_params(sm_cfg, randomization, unwrapped.num_envs, unwrapped.device, generator)
        sm = DragClothSm(unwrapped.step_dt, unwrapped.num_envs, unwrapped.device, sm_cfg, params)
        runner = ClothDragSmRunner(unwrapped, sm)
        diagnostics = ClothDragSmDiagnostics()
        actions = runner.compute()

        step = 0
        while args_cli.num_steps is None or step < args_cli.num_steps:
            _, _, terminated, truncated, _ = env.step(actions)
            step += 1
            dones = terminated | truncated
            if dones.any():
                env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                runner.reset_idx(env_ids)
                params = sample_sm_params(sm_cfg, randomization, len(env_ids), unwrapped.device, generator)
                sm.set_params(env_ids, params)
            actions = runner.compute()
            diagnostics.report(runner)

        env.close()


if __name__ == "__main__":
    main()

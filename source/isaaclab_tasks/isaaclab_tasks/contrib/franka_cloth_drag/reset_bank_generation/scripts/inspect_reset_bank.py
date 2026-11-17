# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to inspect the reset states of a bank of ``IsaacContrib-Drag-Cloth-Franka``.

Loads a bank of ``generate_reset_bank.py`` through the task's reset event
(:class:`~isaaclab_tasks.contrib.franka_cloth_drag.reset_strategies.ResetFromBank`, with its fingerprint check), as the
only reset event like the uniform and adaptive reset strategies, restores its states chunk by chunk into the
environments of ``IsaacContrib-Drag-Cloth-Franka``, holds each for ``--hold_steps`` steps (arm still, gripper as
stored), and prints the start stage and the outcome of the validation checks of every state. The visualizer's Reset
Episode button skips to the next stored states.

.. code-block:: bash

    GEN=source/isaaclab_tasks/isaaclab_tasks/contrib/franka_cloth_drag/reset_bank_generation/scripts

    # Four environments in the Viser web viewer on port 8760; open http://localhost:8760.
    uv run python $GEN/inspect_reset_bank.py --bank logs/reset_banks/debug.pt

    # Only the drag states, once.
    uv run python $GEN/inspect_reset_bank.py --bank logs/reset_banks/debug.pt --stage drag --num_cycles 1

"""

import argparse
import sys

import gymnasium as gym
import torch
import warp as wp
from isaaclab_visualizers.viser import ViserVisualizerCfg

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.franka_cloth_drag.mdp.stages import START_STAGE_ALIASES, START_STAGE_NAMES
from isaaclab_tasks.contrib.franka_cloth_drag.reset_bank_generation import (
    ResetBankGenerationCfg,
    ResetStateValidator,
)
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

defaults = ResetBankGenerationCfg()
# add argparse arguments
parser = argparse.ArgumentParser(description="Inspect the reset states of a bank of the cloth dragging task.")
parser.add_argument("--bank", type=str, required=True, help="Path of the bank (.pt).")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="IsaacContrib-Drag-Cloth-Franka", help="The task to run.")
parser.add_argument(
    "--stage",
    type=str,
    default="all",
    choices=["all", *START_STAGE_NAMES, *START_STAGE_ALIASES],
    help="Start stage of the states to inspect; 'default' is a legacy alias for 'approach'.",
)
parser.add_argument("--hold_steps", type=int, default=90, help="Steps each state is held.")
parser.add_argument("--num_cycles", type=int, default=None, help="Passes over the states. Defaults to endless.")
parser.add_argument(
    "--allow_settings_mismatch", action="store_true", help="Load a bank made with other physics or task settings."
)
parser.add_argument(
    "--max_node_speed", type=float, default=defaults.max_node_speed, help="Fastest valid cloth node speed [m/s]."
)
parser.add_argument(
    "--max_cube_tilt_deg", type=float, default=defaults.max_cube_tilt_deg, help="Largest valid cube tilt [deg]."
)
add_launcher_args(parser)
# the task runs on Newton, so default to the Viser web viewer and let the task pick the device
parser.set_defaults(device=None, visualizer=["viser"])
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

# initialize warp
wp.init()


def main():
    # parse configuration via Hydra, so presets can be selected on the CLI
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    args_cli.device = env_cfg.sim.device
    env_cfg.scene.num_envs = args_cli.num_envs
    args_cli.visualizer = ["viser"]
    # a held state may reach the target: let it hold there instead of resetting
    env_cfg.terminations.success = None
    env_cfg.rewards.success = None
    env_cfg.sim.visualizer_cfgs = [ViserVisualizerCfg(port=8760)]

    with launch_simulation(env_cfg, args_cli):
        from isaaclab_tasks.contrib.franka_cloth_drag.config.franka.franka_cloth_drag_env_cfg import UniformResetsCfg

        # load the bank through the training reset event, as the only reset event; it draws nothing by itself
        env_cfg.events = UniformResetsCfg()
        env_cfg.events.reset_from_bank.bank_path = args_cli.bank
        env_cfg.events.reset_from_bank.start_stage_ratios = None
        env_cfg.events.reset_from_bank.allow_settings_mismatch = args_cli.allow_settings_mismatch
        env = gym.make(args_cli.task, cfg=env_cfg)
        unwrapped = env.unwrapped
        device = unwrapped.device
        bank_term = unwrapped.event_manager.get_term_cfg("reset_from_bank").func
        bank = bank_term.bank
        validator = ResetStateValidator(unwrapped, args_cli.max_node_speed, args_cli.max_cube_tilt_deg)
        if args_cli.stage == "all":
            rows = torch.arange(len(bank), device=device)
        else:
            rows = bank.stage_rows(args_cli.stage)
        print(f"[INFO]: Inspecting {len(rows)} of {len(bank)} states {bank.stage_counts()} from {args_cli.bank}.")

        cycle = 0
        while args_cli.num_cycles is None or cycle < args_cli.num_cycles:
            for start in range(0, len(rows), unwrapped.num_envs):
                chunk = rows[start : start + unwrapped.num_envs]
                env_ids = torch.arange(len(chunk), device=device)
                bank_term.force_rows(env_ids, chunk)
                unwrapped.reset()
                validator.start(bank, chunk, env_ids)
                skipped = False
                for _ in range(args_cli.hold_steps):
                    _, _, terminated, truncated, _ = unwrapped.step(validator.hold_actions())
                    # the visualizer's Reset Episode button makes env.step reset every environment with the default
                    # reset (terminated, but by no termination term): go on to the next stored states instead
                    manager = unwrapped.termination_manager
                    fired = torch.stack([manager.get_term(name) for name in manager.active_terms]).any(dim=0)
                    if (terminated & ~fired).any():
                        print(f"[INFO]: Reset Episode: skipping rows {chunk.tolist()}, showing the next states.")
                        skipped = True
                        break
                    validator.update(terminated, truncated)
                if skipped:
                    continue
                valid, reasons = validator.result()
                for env_id, row in enumerate(chunk.tolist()):
                    code = int(reasons[env_id])
                    reason = "valid" if valid[env_id] else validator.reasons[code - 1] if code > 0 else "ended"
                    print(f"[INFO]: env {env_id}: row {row} ({START_STAGE_NAMES[int(bank.stage[row])]}): {reason}.")
            cycle += 1

        # close the environment
        env.close()


if __name__ == "__main__":
    main()

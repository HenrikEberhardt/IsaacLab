# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to generate a bank of reset states of ``IsaacContrib-Drag-Cloth-Franka`` with the randomized state machine.

The state machine (:mod:`isaaclab_tasks.contrib.franka_cloth_drag.state_machine`) runs in all environments, started
by the task's default resets and randomized per rollout (drag speed, drag height, grasp yaw offset, pre-grasp pose,
arm action noise). Each successful rollout contributes one state per start stage, captured at a uniformly random time
inside the stage's window: the start of the episode or a step before the GRASP stage, the pre-grasp pose the state
machine holds or moves into (open gripper), the fresh grasp before the drag, and the goal point at the edge of the goal
zone. Every candidate is restored through the training reset path and held for a few steps; only valid states are
kept. The bank holds exactly ``--stage_quotas`` states per stage, or ``--num_states`` states split evenly over the four
stages, and is saved to ``--output``. All other generation settings are those of
:class:`~isaaclab_tasks.contrib.franka_cloth_drag.reset_bank_generation.ResetBankGenerationCfg`; the bank stores them
in its meta. See ``franka_cloth_drag/implementation_log/reset_states.md``.

.. code-block:: bash

    GEN=source/isaaclab_tasks/isaaclab_tasks/contrib/franka_cloth_drag/reset_bank_generation/scripts

    # Debug bank: 12 states (3 per stage) in 16 envs.
    uv run python $GEN/generate_reset_bank.py --num_envs 16 --num_states 12 --output logs/reset_banks/debug.pt

    # Full bank.
    uv run python $GEN/generate_reset_bank.py --num_envs 256 --num_states 4000 --output logs/reset_banks/bank_4000_v7.pt

    # Bank sized per stage: large approach bank, frontier-sized grasp, drag and goal banks.
    uv run python $GEN/generate_reset_bank.py --num_envs 256 --stage_quotas approach=8000,grasp=1000,drag=100,goal=100 \
        --output logs/reset_banks/bank_v8.pt

"""

import argparse
import sys
from datetime import datetime

import gymnasium as gym

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.franka_cloth_drag.mdp import TopDownStabilizedDeltaEEActionCfg
from isaaclab_tasks.contrib.franka_cloth_drag.reset_bank_generation import ResetBankGenerationCfg
from isaaclab_tasks.contrib.franka_cloth_drag.state_machine import (
    ClothDragSmRunner,
    DragClothSm,
    configure_state_machine_osc,
    make_sm_profile,
)
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

defaults = ResetBankGenerationCfg()
# add argparse arguments
parser = argparse.ArgumentParser(description="Generate a bank of reset states of the cloth dragging task.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="IsaacContrib-Drag-Cloth-Franka", help="The task to run.")
parser.add_argument(
    "--num_states",
    type=int,
    default=defaults.num_states,
    help="Number of stored states, split evenly over the four stages. Not used with --stage_quotas.",
)
parser.add_argument(
    "--stage_quotas",
    type=str,
    default=None,
    help="Number of stored states per stage, e.g. 'approach=8000,grasp=1000,drag=100,goal=100'.",
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Path of the bank (.pt). Defaults to logs/reset_banks/reset_bank_<date>_<time>.pt.",
)
parser.add_argument("--seed", type=int, default=defaults.seed, help="Seed of the environment and the randomization.")
add_launcher_args(parser)
parser.set_defaults(device=None)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def main():
    # the generation settings (state machine randomization, capture windows, validation) use one shared config
    stage_quotas = None
    if args_cli.stage_quotas is not None:
        stage_quotas = {
            name.strip(): int(count) for name, count in (item.split("=") for item in args_cli.stage_quotas.split(","))
        }
    sm_cfg, sm_randomization = make_sm_profile("randomized")
    generation_cfg = ResetBankGenerationCfg(
        num_states=args_cli.num_states,
        stage_quotas=stage_quotas,
        seed=args_cli.seed,
        sm_randomization=sm_randomization,
    )
    output = args_cli.output or f"logs/reset_banks/reset_bank_{datetime.now():%Y-%m-%d_%H-%M-%S}.pt"

    # parse configuration via Hydra, so presets can be selected on the CLI
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    args_cli.device = env_cfg.sim.device
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.visualizer_cfgs = []
    args_cli.visualizer = None
    configure_state_machine_osc(env_cfg)
    # the training physics and terminations; the success termination ends successful rollouts
    if not isinstance(env_cfg.actions.arm_action, TopDownStabilizedDeltaEEActionCfg):
        raise SystemExit("Reset-bank generation requires the state-machine roll/pitch-stabilized arm action.")

    with launch_simulation(env_cfg, args_cli):
        from isaaclab_tasks.contrib.franka_cloth_drag.reset_bank_generation import ResetBankBuilder, add_bank_term

        # the rollouts start with the default resets; the candidates are restored through the training reset path
        add_bank_term(env_cfg)
        env = gym.make(args_cli.task, cfg=env_cfg)
        unwrapped = env.unwrapped
        unwrapped.reset(seed=args_cli.seed)

        drag_sm = DragClothSm(
            unwrapped.step_dt,
            unwrapped.num_envs,
            unwrapped.device,
            sm_cfg,
        )
        runner = ClothDragSmRunner(unwrapped, drag_sm)
        bank_term = unwrapped.event_manager.get_term_cfg("reset_from_bank").func
        builder = ResetBankBuilder(unwrapped, runner, bank_term, generation_cfg)
        print(f"[INFO]: Collecting {generation_cfg.quotas()} reset states in {unwrapped.num_envs} env(s).")

        bank = builder.run()
        bank.save(output)
        statistics = bank.meta["statistics"]
        print(f"[INFO]: Saved {len(bank)} states {bank.stage_counts()} to {output}.")
        for key in sorted(statistics):
            print(f"[INFO]:   {key}: {statistics[key]}")

        # close the environment
        env.close()


if __name__ == "__main__":
    main()

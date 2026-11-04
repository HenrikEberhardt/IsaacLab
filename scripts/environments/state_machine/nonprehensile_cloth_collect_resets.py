# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to generate the reset states of the non-prehensile cloth task with the randomized state machine.

The state machine of :mod:`isaaclab_tasks.contrib.franka_cloth_drag.state_machine` runs in all environments of
``IsaacContrib-Drag-Cloth-Franka``, randomized per rollout (drag speed, drag height, grasp yaw offset, arm action
noise, and with ``--random_grasp_node`` the grasped raised node instead of the task's grasp node). Each successful
rollout contributes one state per training reset window (grasp, drag, goal), captured at a uniformly random time
inside the window: the pre-grasp pose the state machine holds (open
gripper, offset and yaw error per rollout), the fresh grasp before the drag, and the goal point at the edge of the goal
zone. Every candidate is restored through the training reset
path into a freshly reset environment and held for ``--validation_steps`` steps; only valid states are kept. The bank
holds exactly ``--num_states`` grasp, drag and goal states, split evenly, and is saved to ``--output``. See
``franka_cloth_drag/implementation_log/reset_states.md``.

.. code-block:: bash

    # Debug bank: 12 states (4 per bucket) in 16 envs.
    uv run python scripts/environments/state_machine/nonprehensile_cloth_collect_resets.py \
        --num_envs 16 --num_states 12 --output logs/reset_banks/debug.pt

    # Full bank.
    uv run python scripts/environments/state_machine/nonprehensile_cloth_collect_resets.py \
        --num_envs 256 --num_states 5000 --output logs/reset_banks/bank_5000.pt

"""

import argparse
import sys
from datetime import datetime

import gymnasium as gym
import warp as wp

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.franka_cloth_drag.mdp import TopDownDeltaEEActionCfg
from isaaclab_tasks.contrib.franka_cloth_drag.reset_states.collect import CollectCfg
from isaaclab_tasks.contrib.franka_cloth_drag.state_machine import (
    HOVER_MARGIN,
    YAW_ACTION_SCALE,
    ClothDragSmRunner,
    DragClothSm,
    keep_requested_visualizers,
)
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

defaults = CollectCfg()
# add argparse arguments
parser = argparse.ArgumentParser(description="Generate the reset states of the non-prehensile cloth task.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="IsaacContrib-Drag-Cloth-Franka", help="The task to run.")
parser.add_argument(
    "--num_states", type=int, default=defaults.num_states, help="Number of grasp, drag and goal states, split evenly."
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Path of the bank (.pt). Defaults to logs/reset_banks/reset_bank_<date>_<time>.pt.",
)
parser.add_argument("--seed", type=int, default=defaults.seed, help="Seed of the environment and the randomization.")
parser.add_argument(
    "--validation_steps", type=int, default=defaults.validation_steps, help="Steps each candidate is held."
)
parser.add_argument("--no_randomize", action="store_true", help="Run the state machine without randomization.")
parser.add_argument(
    "--random_grasp_node",
    action="store_true",
    help="Grasp a random raised node per rollout instead of the task's grasp node.",
)
parser.add_argument(
    "--drag_speed_range", type=float, nargs=2, default=defaults.drag_speed_range, help="Drag speed range [m/s]."
)
parser.add_argument(
    "--drag_height_range", type=float, nargs=2, default=defaults.drag_height_range, help="Drag height range [m]."
)
parser.add_argument(
    "--yaw_offset_range", type=float, nargs=2, default=defaults.yaw_offset_range, help="Grasp yaw offset range [rad]."
)
parser.add_argument(
    "--pregrasp_height_range",
    type=float,
    nargs=2,
    default=defaults.pregrasp_height_range,
    help="Height range of the pre-grasp pose above the raised node [m].",
)
parser.add_argument(
    "--pregrasp_radius_range",
    type=float,
    nargs=2,
    default=defaults.pregrasp_radius_range,
    help="Largest planar pre-grasp offset at the lowest and highest pre-grasp height [m].",
)
parser.add_argument(
    "--pregrasp_yaw_error_range",
    type=float,
    nargs=2,
    default=defaults.pregrasp_yaw_error_range,
    help="Yaw range of the pre-grasp pose relative to the grasp yaw [rad].",
)
parser.add_argument(
    "--drag_window", type=float, default=defaults.drag_window, help="Latest drag capture after the squeeze [s]."
)
parser.add_argument(
    "--drag_max_progress", type=float, default=defaults.drag_max_progress, help="Largest progress of a drag state."
)
parser.add_argument(
    "--goal_distance_window",
    type=float,
    nargs=2,
    default=defaults.goal_distance_window,
    help="Distance range of the goal point to the target of the goal states [m].",
)
parser.add_argument(
    "--action_noise", type=float, default=defaults.action_noise_std, help="Std of the normalized arm action noise."
)
parser.add_argument(
    "--max_node_speed", type=float, default=defaults.max_node_speed, help="Fastest valid cloth node speed [m/s]."
)
parser.add_argument(
    "--max_cube_tilt_deg", type=float, default=defaults.max_cube_tilt_deg, help="Largest valid cube tilt [deg]."
)
parser.add_argument("--max_rounds", type=int, default=defaults.max_rounds, help="Most rounds of rollouts.")
parser.add_argument(
    "--grasp_height",
    type=float,
    default=0.0095,
    help="End-effector height commanded for the squeeze [m], at least the action's floor.",
)
add_launcher_args(parser)
# the task runs on Newton; let the task pick the device. Without --viz the task's visualizers are dropped (headless);
# a "none" default would not count as an explicit choice
parser.set_defaults(device=None)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

# initialize warp
wp.init()


def main():
    collect_cfg = CollectCfg(
        num_states=args_cli.num_states,
        validation_steps=args_cli.validation_steps,
        randomize=not args_cli.no_randomize,
        random_grasp_node=args_cli.random_grasp_node,
        drag_speed_range=tuple(args_cli.drag_speed_range),
        drag_height_range=tuple(args_cli.drag_height_range),
        yaw_offset_range=tuple(args_cli.yaw_offset_range),
        pregrasp_height_range=tuple(args_cli.pregrasp_height_range),
        pregrasp_radius_range=tuple(args_cli.pregrasp_radius_range),
        pregrasp_yaw_error_range=tuple(args_cli.pregrasp_yaw_error_range),
        drag_window=args_cli.drag_window,
        drag_max_progress=args_cli.drag_max_progress,
        goal_distance_window=tuple(args_cli.goal_distance_window),
        action_noise_std=args_cli.action_noise,
        max_node_speed=args_cli.max_node_speed,
        max_cube_tilt_deg=args_cli.max_cube_tilt_deg,
        max_rounds=args_cli.max_rounds,
        seed=args_cli.seed,
    )
    output = args_cli.output or f"logs/reset_banks/reset_bank_{datetime.now():%Y-%m-%d_%H-%M-%S}.pt"

    # parse configuration via Hydra, so presets can be selected on the CLI
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    args_cli.device = env_cfg.sim.device
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    # the training physics and terminations; the success termination ends successful rollouts
    env_cfg.actions.gripper_action.attach = True
    if not isinstance(env_cfg.actions.arm_action, TopDownDeltaEEActionCfg):
        raise SystemExit("The state machine commands the delta end-effector action: use the 'ee_delta' actions preset.")
    # turn the hand faster than the task's yaw scale allows (changes the action mapping only, not the states)
    env_cfg.actions.arm_action.scale_rot = YAW_ACTION_SCALE
    keep_requested_visualizers(env_cfg, args_cli.visualizer)

    with launch_simulation(env_cfg, args_cli):
        from isaaclab.managers import EventTermCfg

        from isaaclab_tasks.contrib.franka_cloth_drag.mdp import ResetFromBank
        from isaaclab_tasks.contrib.franka_cloth_drag.reset_states.collect import ResetCollector

        # the candidates are restored through the training reset path; it must run after the other reset events
        env_cfg.events.reset_from_bank = EventTermCfg(func=ResetFromBank, mode="reset", params={"bank_path": None})
        env = gym.make(args_cli.task, cfg=env_cfg)
        unwrapped = env.unwrapped
        unwrapped.reset(seed=args_cli.seed)

        command = unwrapped.command_manager.get_term("cloth_target")
        arm_cfg = unwrapped.action_manager.get_term("arm_action").cfg
        drag_sm = DragClothSm(
            unwrapped.step_dt,
            unwrapped.num_envs,
            unwrapped.device,
            hover_height=command.cfg.approach_radius - HOVER_MARGIN,
            grasp_height=max(args_cli.grasp_height, arm_cfg.z_min),
            drag_height=0.02,
            drag_speed=0.08,
            goal_tolerance=min(0.02, command.cfg.success_radius),
            wrist_limits=arm_cfg.wrist_joint_limits,
            attach=True,
            pregrasp_hold=True,
        )
        runner = ClothDragSmRunner(unwrapped, drag_sm)
        bank_term = unwrapped.event_manager.get_term_cfg("reset_from_bank").func
        collector = ResetCollector(unwrapped, runner, bank_term, collect_cfg)
        print(f"[INFO]: Collecting {collect_cfg.quotas()} reset states in {unwrapped.num_envs} env(s).")

        bank = collector.run()
        bank.save(output)
        statistics = bank.meta["statistics"]
        print(f"[INFO]: Saved {len(bank)} states {bank.meta['bucket_counts']} to {output}.")
        for key in sorted(statistics):
            print(f"[INFO]:   {key}: {statistics[key]}")

        # close the environment
        env.close()


if __name__ == "__main__":
    main()

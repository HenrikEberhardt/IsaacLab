# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Save the student's wrist images over an episode and time the visual env.

Runs ``IsaacContrib-Drag-Cloth-Franka-Visual`` with the teacher acting through its joint-delta labels, so the arm
does the task while the camera records. Writes one PNG grid (rows: environments, columns: time) of the student's
image observation, as the network sees it (after latency, frame drops, recolor and augmentation), and prints the
time per environment step.

Usage (from the IsaacLab root, one GPU):
    uv run python source/isaaclab_tasks/isaaclab_tasks/contrib/franka_cloth_drag_visual/scripts/save_image_strip.py \\
        --num_envs 64 --num_steps 300 --out /tmp/strip.png --viz none
"""

import argparse
import sys
import time

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

TASK = "IsaacContrib-Drag-Cloth-Franka-Visual"

parser = argparse.ArgumentParser(description="Wrist image strip and timing of the cloth-drag visual env.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_steps", type=int, default=300)
parser.add_argument("--rows", type=int, default=8, help="Environments shown.")
parser.add_argument("--columns", type=int, default=10, help="Time steps shown, evenly spaced.")
parser.add_argument("--out", default="image_strip.png")
parser.add_argument("--checkpoint", default=None, help="Teacher checkpoint (default: teacher_cfg.TEACHER_CHECKPOINT).")
parser.add_argument("--clean", action="store_true", help="Disable the corruption of the image (no augmentation).")
parser.add_argument("--seed", type=int, default=0)
add_launcher_args(parser)
parser.set_defaults(device=None)
args_cli, hydra_args = setup_preset_cli(parser, sys.argv[1:])
sys.argv = [sys.argv[0]] + hydra_args


def main():
    env_cfg, _ = resolve_task_config(TASK, None)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    args_cli.device = env_cfg.sim.device
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.clean:
        env_cfg.observations.student_image.enable_corruption = False

    with launch_simulation(env_cfg, args_cli):
        from PIL import Image

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        from isaaclab_tasks.contrib.franka_cloth_drag_visual.teacher import teacher_actions_to_joint_deltas
        from isaaclab_tasks.contrib.franka_cloth_drag_visual.teacher.loading import load_teacher
        from isaaclab_tasks.contrib.franka_cloth_drag_visual.teacher_cfg import IK_LABEL_CFG, TEACHER_CHECKPOINT

        gym_env = gym.make(TASK, cfg=env_cfg)
        env = gym_env.unwrapped
        wrapped = RslRlVecEnvWrapper(gym_env)
        obs = wrapped.get_observations()
        teacher = load_teacher(obs, args_cli.checkpoint or TEACHER_CHECKPOINT, env.device)
        print(f"[STRIP] image {tuple(obs['student_image'].shape[1:])}, proprio {obs['student_proprio'].shape[-1]}")

        keep = set(torch.linspace(0, args_cli.num_steps - 1, args_cli.columns).round().long().tolist())
        frames = []
        successes = episodes = 0
        torch.cuda.synchronize()
        start = time.perf_counter()
        for step in range(args_cli.num_steps):
            with torch.inference_mode():
                labels, _ = teacher_actions_to_joint_deltas(teacher(obs), obs["ik_state"], IK_LABEL_CFG)
            obs, _, dones, _ = wrapped.step(labels)
            if step in keep:
                frames.append(obs["student_image"][: args_cli.rows].clamp(0.0, 1.0).cpu())
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            episodes += len(done_ids)
            successes += int(env.termination_manager.get_term("task_success")[done_ids].sum())
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        print(
            f"[STRIP] {1e3 * elapsed / args_cli.num_steps:.1f} ms/step at {args_cli.num_envs} envs,"
            f" success {successes}/{episodes}"
        )

        # grid: rows are environments, columns are time
        grid = torch.stack(frames, dim=1)  # [rows, columns, 3, H, W]
        rows, columns, _, height, width = grid.shape
        grid = grid.permute(0, 3, 1, 4, 2).reshape(rows * height, columns * width, 3)
        Image.fromarray((grid * 255).round().byte().numpy()).save(args_cli.out)
        print(f"[STRIP] saved {args_cli.out}")
        gym_env.close()


if __name__ == "__main__":
    main()

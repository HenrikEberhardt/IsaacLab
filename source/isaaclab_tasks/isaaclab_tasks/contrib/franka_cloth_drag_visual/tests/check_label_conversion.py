# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teacher-label conversion check: --mode native (reference), labels (teacher through its labels), check (float diff)."""

import argparse
import sys

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

TASK = "IsaacContrib-Drag-Cloth-Franka-Joint"

parser = argparse.ArgumentParser(description="Teacher-label conversion check of the cloth-drag distillation env.")
parser.add_argument("--mode", choices=("native", "labels", "check"), default="labels")
parser.add_argument("--checkpoint", default=None, help="Teacher checkpoint (default: teacher_cfg.TEACHER_CHECKPOINT).")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--num_steps", type=int, default=900, help="Environment steps.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--resets", choices=("default", "adaptive"), default="default", help="Task-start or bank resets.")
parser.add_argument("--reanchor", action="store_true", help="Student target re-anchored every physics step.")
parser.add_argument("--clean_teacher_obs", action="store_true", help="Disable the corruption of the teacher's group.")
parser.add_argument(
    "--joint_scale", type=float, nargs="+", default=None, help="Student joint-delta scale [rad], one or seven values."
)
parser.add_argument(
    "--per_joint_clip", action="store_true", help="Clip over-range labels per joint instead of scaling them uniformly."
)
add_launcher_args(parser)
parser.set_defaults(device=None)
args_cli, hydra_args = setup_preset_cli(parser, sys.argv[1:])
sys.argv = [sys.argv[0]] + hydra_args


def configure(env_cfg):
    """Apply the mode's action space and the reset strategy."""
    from isaaclab.envs import mdp as isaaclab_mdp
    from isaaclab.managers import ObservationTermCfg as ObsTerm

    from isaaclab_tasks.contrib.franka_cloth_drag.config.franka.franka_cloth_drag_env_cfg import DefaultResetsCfg
    from isaaclab_tasks.contrib.franka_cloth_drag_visual.teacher import TEACHER_ARM_CFG

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.resets == "default":
        env_cfg.events = DefaultResetsCfg()
    if args_cli.mode in ("native", "check"):
        env_cfg.actions.arm_action = TEACHER_ARM_CFG.copy()
        env_cfg.observations.teacher.last_action = ObsTerm(func=isaaclab_mdp.last_action)
        if args_cli.mode == "check":
            env_cfg.actions.arm_action.command_delay_range = (0, 0)
    if args_cli.clean_teacher_obs:
        env_cfg.observations.teacher.enable_corruption = False
    if args_cli.mode == "labels":
        env_cfg.actions.arm_action.anchor_per_policy_step = not args_cli.reanchor
    scale = joint_scale()
    if scale is not None and args_cli.mode == "labels":
        if isinstance(scale, float):
            env_cfg.actions.arm_action.scale = scale
        else:
            env_cfg.actions.arm_action.scale = {f"panda_joint{i + 1}": value for i, value in enumerate(scale)}


def joint_scale() -> float | tuple[float, ...] | None:
    """The --joint_scale override: one float, seven values or None."""
    if args_cli.joint_scale is None:
        return None
    if len(args_cli.joint_scale) not in (1, 7):
        raise ValueError("--joint_scale takes one or seven values.")
    return args_cli.joint_scale[0] if len(args_cli.joint_scale) == 1 else tuple(args_cli.joint_scale)


def main():
    env_cfg, _ = resolve_task_config(TASK, None)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    args_cli.device = env_cfg.sim.device
    configure(env_cfg)

    with launch_simulation(env_cfg, args_cli):
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        from isaaclab_tasks.contrib.franka_cloth_drag_visual.action_mapping import (
            teacher_actions_to_joint_deltas,
            teacher_joint_step,
        )
        from isaaclab_tasks.contrib.franka_cloth_drag_visual.franka_cloth_drag_visual_env_cfg import ACTION_MAPPING_CFG
        from isaaclab_tasks.contrib.franka_cloth_drag_visual.student.actions import STUDENT_JOINT_DELTA_SCALE
        from isaaclab_tasks.contrib.franka_cloth_drag_visual.teacher import TEACHER_CHECKPOINT
        from isaaclab_tasks.contrib.franka_cloth_drag_visual.teacher.loading import load_teacher

        gym_env = gym.make(TASK, cfg=env_cfg)
        env = gym_env.unwrapped
        wrapped = RslRlVecEnvWrapper(gym_env)
        obs = wrapped.get_observations()
        teacher = load_teacher(obs, args_cli.checkpoint or TEACHER_CHECKPOINT, env.device)
        label_cfg = ACTION_MAPPING_CFG.replace(
            preserve_direction=not args_cli.per_joint_clip, joint_delta_scale=joint_scale() or STUDENT_JOINT_DELTA_SCALE
        )
        print(f"[REPLAY] scale {label_cfg.joint_delta_scale}, preserve direction {label_cfg.preserve_direction}, reanchor {args_cli.reanchor}")
        arm = env.action_manager.get_term("arm_action")
        print(f"[REPLAY] mode {args_cli.mode}, teacher obs {obs['teacher'].shape[-1]}, ik_state {obs['ik_state'].shape[-1]}")

        term_names = [name for name in env.termination_manager.active_terms]
        term_counts = dict.fromkeys(term_names, 0)
        episodes = 0
        clipped = torch.zeros(7, device=env.device)
        abs_label = torch.zeros(7, device=env.device)
        label_steps = 0
        ee_travel, ee_steps = 0.0, 0
        check_error = torch.zeros((), device=env.device)
        for _ in range(args_cli.num_steps):
            with torch.inference_mode():
                teacher_actions = teacher(obs)
                labels, clip_mask = teacher_actions_to_joint_deltas(teacher_actions, obs["ik_state"], label_cfg)
                step = teacher_joint_step(teacher_actions[:, :4], obs["ik_state"], label_cfg)
            clipped += clip_mask.float().sum(0)
            abs_label += labels[:, :7].abs().sum(0)
            label_steps += len(labels)
            actions = labels if args_cli.mode == "labels" else teacher_actions
            ee_before = obs["ik_state"][:, 7:10].clone()
            obs, _, dones, _ = wrapped.step(actions)
            if args_cli.mode == "check":
                error = (arm.first_step - step).abs().amax(dim=1)
                if len(error):
                    check_error = torch.maximum(check_error, error.max())
            alive = ~dones.bool()
            ee_travel += float((obs["ik_state"][:, 7:10] - ee_before).norm(dim=1)[alive].sum())
            ee_steps += int(alive.sum())
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if len(done_ids):
                episodes += len(done_ids)
                for name in term_names:
                    term_counts[name] += int(env.termination_manager.get_term(name)[done_ids].sum())

        print(f"[REPLAY] episodes {episodes}, success {term_counts.get('task_success', 0) / max(episodes, 1):.3f}")
        print("[REPLAY] terminations: " + ", ".join(f"{k} {v}" for k, v in term_counts.items() if v))
        print(f"[REPLAY] label clip fraction per joint: {(clipped / label_steps).cpu().numpy().round(4)}")
        print(f"[REPLAY] mean |label| per joint:       {(abs_label / label_steps).cpu().numpy().round(3)}")
        print(f"[REPLAY] mean EE travel per step: {1e3 * ee_travel / max(ee_steps, 1):.2f} mm")
        if args_cli.mode == "check":
            print(f"[REPLAY] max |first IK step - converted step|: {float(check_error):.3e} rad")
        gym_env.close()


if __name__ == "__main__":
    main()

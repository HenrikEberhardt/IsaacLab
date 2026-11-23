# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a distilled student from the task start: success, gripper forces, width and raised-node reliance.

Tests (``--test``), each a list of variants of ``--num_episodes`` first episodes. They run in rounds of ``--num_envs``
envs (one first episode per env; fewer envs, less GPU memory), with the same seeds in every variant:

    rollout: the student as trained (reference).
    width:   the observed gripper width lies around grasp events or is shifted:
             true (reference), empty (empty-close width while the lock holds cloth), real (--real_pinch while locked),
             fake_closed (--fake_lock_width while closed without a lock), fake_failed (the same only after a failed
             close), offset<X>mm (constant shift, the training bias is +-5 mm).
    flat:    the raised node at --raise_heights_mm (10 mm in training, 0 mm = flat cloth).

The student acts with its mean action. Training randomization stays on unless ``--clean``.

Usage (from the IsaacLab root, one GPU):
    uv run python source/isaaclab_tasks/isaaclab_tasks/contrib/franka_cloth_drag_visual/tests/eval_student.py \\
        --checkpoint logs/rsl_rl/cloth_drag_distillation/<run>/model_999.pt [--test all] [--out result.json]
"""

import argparse
import json
import math
import sys

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

TASK = "IsaacContrib-Drag-Cloth-Franka-Visual"
PHYSICS_DR_EVENTS = ("arm_gains", "arm_friction", "arm_armature", "cloth_material", "cloth_friction", "cube_mass")
CONTACT_FORCE = 0.5  # [N] push counted as contact, as GripperForceLimit._CONTACT_FORCE
BUMP_MEASURE_STEP = 15  # policy step (0.5 s) at which the grasp node height above the cloth is measured

parser = argparse.ArgumentParser(description="Evaluation of a distilled cloth-drag student.")
parser.add_argument("--checkpoint", required=True, help="Distillation checkpoint (model_N.pt).")
parser.add_argument("--test", choices=("rollout", "width", "flat", "all"), default="all")
parser.add_argument("--num_episodes", type=int, default=100, help="First episodes per variant.")
parser.add_argument("--num_envs", type=int, default=100, help="Parallel envs; the episodes run in rounds of this size.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--clean", action="store_true", help="No augmentation, obs noise, action lag, physics DR.")
parser.add_argument("--no_force_stop", action="store_true", help="No gripper force termination (still measured).")
parser.add_argument("--real_pinch", type=float, default=0.002, help="Width [m] of the real variant (caliper pinch).")
parser.add_argument("--fake_lock_width", type=float, default=0.008, help="Width [m] of the fake variants (sim pinch).")
parser.add_argument("--offsets_mm", type=float, nargs="*", default=[-10.0, -5.0, 5.0, 10.0])
parser.add_argument("--raise_heights_mm", type=float, nargs="*", default=[10.0, 5.0, 2.5, 0.0])
parser.add_argument("--out", default=None, help="Write all variant results to this JSON file.")
add_launcher_args(parser)
parser.set_defaults(device=None)
args_cli, hydra_args = setup_preset_cli(parser, sys.argv[1:])
sys.argv = [sys.argv[0]] + hydra_args


def configure(env_cfg):
    """Task-start resets with the camera events, optional clean mode and no force stop."""
    from isaaclab_tasks.contrib.franka_cloth_drag.config.franka.franka_cloth_drag_env_cfg import DefaultResetsCfg
    from isaaclab_tasks.contrib.franka_cloth_drag_visual.domain_randomization import add_visual_randomization

    env_cfg.scene.num_envs = min(args_cli.num_envs, args_cli.num_episodes)
    env_cfg.seed = args_cli.seed
    env_cfg.events = DefaultResetsCfg()
    add_visual_randomization(env_cfg)  # replacing the events dropped the camera events
    if args_cli.clean:
        for name in PHYSICS_DR_EVENTS:
            setattr(env_cfg.events, name, None)
        env_cfg.observations.student_image.rgb.params["augmentation"] = None
        env_cfg.observations.student_proprio.enable_corruption = False
        env_cfg.actions.arm_action.command_delay_range = (0, 0)
        env_cfg.actions.arm_action.hold_prob = 0.0
    if args_cli.no_force_stop:
        env_cfg.terminations.gripper_force_limit.params["max_force"] = math.inf


def variants() -> list[tuple[str, str, float | None]]:
    """(name, kind, value) of every variant of the selected tests."""
    tests = ("rollout", "width", "flat") if args_cli.test == "all" else (args_cli.test,)
    out = []
    if "rollout" in tests:
        out.append(("rollout", "none", None))
    if "width" in tests:
        out += [("width_true", "none", None), ("width_empty", "empty", None), ("width_real", "real", None)]
        out += [("width_fake_closed", "fake_closed", None), ("width_fake_failed", "fake_failed", None)]
        out += [(f"width_offset{o:+g}mm", "offset", o * 1e-3) for o in args_cli.offsets_mm]
    if "flat" in tests:
        out += [(f"bump_{h:g}mm", "bump", h * 1e-3) for h in args_cli.raise_heights_mm]
    return out


def load_student(obs, agent_cfg, checkpoint: str, num_actions: int, device: str) -> torch.nn.Module:
    """The student in eval mode, strictly loaded from a distillation checkpoint; returns mean actions."""
    from rsl_rl.utils import resolve_class, resolve_obs_groups

    cfg = agent_cfg.to_dict()
    obs_groups = resolve_obs_groups(obs, cfg["obs_groups"], ["student", "teacher"])
    model_class, model_cfg = resolve_class(cfg["student"])
    student = model_class(obs, obs_groups, "student", num_actions, **model_cfg).to(device)
    student.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["student_state_dict"])
    return student.eval()


class EpisodeStats:
    """Per-env statistics of the first episode of every env."""

    def __init__(self, env, command, gripper):
        self.env, self.command, self.gripper = env, command, gripper
        n, dev = env.num_envs, env.device
        z = lambda: torch.zeros(n, device=dev)  # noqa: E731
        b = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
        self.active = torch.ones(n, dtype=torch.bool, device=dev)
        self.length, self.success, self.term = z(), b(), torch.full((n,), -1, dtype=torch.long, device=dev)
        # forces [N], recorded inside the termination manager so the terminating step counts
        self.force_max, self.peak_max = z(), z()
        self.contact_sum, self.contact_steps = z(), z()
        self.stage_sum, self.stage_steps, self.stage_contact = {}, {}, {}
        for stage in ("grasp", "drag"):
            self.stage_sum[stage], self.stage_steps[stage], self.stage_contact[stage] = z(), z(), z()
        # gripper events
        self.closes, self.close_dist_sum = z(), z()
        self.failed_closes, self.failed_reopens = z(), z()
        self.lock_losses, self.lock_reopens = z(), z()
        self.right_any, self.wrong_any = b(), b()
        self.first_right_step = torch.full((n,), math.inf, device=dev)
        self.grasp_steps, self.min_ee_z = z(), torch.full((n,), math.inf, device=dev)
        self.bump_height = torch.full((n,), math.nan, device=dev)  # [m] grasp node above the median node
        self._prev = None

    def record_force(self, force_term) -> None:
        """Called right after the force term, before any reset."""
        from isaaclab_tasks.contrib.franka_cloth_drag.mdp.stages import ClothDragStage

        a = self.active
        force, peak = force_term.force, force_term.force_peak
        self.force_max = torch.where(a, torch.maximum(self.force_max, force), self.force_max)
        self.peak_max = torch.where(a, torch.maximum(self.peak_max, peak), self.peak_max)
        contact = a & (force > CONTACT_FORCE)
        self.contact_sum += torch.where(contact, force, 0.0)
        self.contact_steps += contact.float()
        stage = self.command.stage
        for name, index in (("grasp", ClothDragStage.GRASP), ("drag", ClothDragStage.DRAG)):
            in_stage = a & (stage == int(index))
            self.stage_sum[name] += torch.where(in_stage, force, 0.0)
            self.stage_steps[name] += in_stage.float()
            self.stage_contact[name] += (in_stage & (force > CONTACT_FORCE)).float()

    def _gripper_state(self) -> dict[str, torch.Tensor]:
        g = self.gripper
        closed, locked = g.closed.clone(), g.is_grasping.clone()
        return {"closed": closed, "locked": locked, "failed": closed & g.grasp_tried & ~locked}

    def start(self) -> None:
        """Take the gripper state after the reset."""
        self._prev = self._gripper_state()

    def record_step(self, step: int, dones: torch.Tensor) -> None:
        """Gripper events of the envs whose first episode continues; outcome of those that ended."""
        from isaaclab_tasks.contrib.franka_cloth_drag.mdp.stages import (
            ClothDragStage,
            ee_grasp_node_distance,
            gripper_holds_raised_node,
        )

        env, cfg = self.env, self.command.cfg
        done = dones.bool()
        live = self.active & ~done  # the state of done envs is already reset
        now, prev = self._gripper_state(), self._prev

        close = live & now["closed"] & ~prev["closed"]
        self.closes += close.float()
        if close.any():
            dist = ee_grasp_node_distance(env, cfg, self.command.grasp_node_id)
            self.close_dist_sum += torch.where(close, dist, 0.0)
        self.failed_closes += (live & now["failed"] & ~prev["failed"]).float()
        self.failed_reopens += (live & prev["failed"] & ~now["closed"]).float()
        lost = live & prev["locked"] & ~now["locked"]
        self.lock_losses += lost.float()
        self.lock_reopens += (lost & ~now["closed"]).float()
        right = live & gripper_holds_raised_node(env, cfg)
        self.right_any |= right
        self.wrong_any |= live & now["locked"] & ~right
        first = right & torch.isinf(self.first_right_step)
        self.first_right_step = torch.where(first, float(step + 1), self.first_right_step)
        self.grasp_steps += (live & (self.command.stage == int(ClothDragStage.GRASP))).float()
        robot = env.scene["robot"]
        ee_z = env.scene[cfg.ee_frame_name].data.target_pos_w.torch[:, 0, 2] - robot.data.root_pos_w.torch[:, 2]
        self.min_ee_z = torch.where(live, torch.minimum(self.min_ee_z, ee_z), self.min_ee_z)
        if step == BUMP_MEASURE_STEP:
            nodes_z = env.scene[cfg.object_name].data.nodal_pos_w.torch[..., 2]
            node_z = nodes_z.gather(1, self.command.grasp_node_id.unsqueeze(1)).squeeze(1)
            self.bump_height = torch.where(live, node_z - nodes_z.median(dim=1).values, self.bump_height)

        ended = self.active & done
        if ended.any():
            tm = env.termination_manager
            self.length = torch.where(ended, float(step + 1), self.length)
            self.success |= ended & tm.get_term("task_success")
            for k, name in enumerate(tm.active_terms):
                fired = ended & tm.get_term(name) & (self.term < 0)
                self.term = torch.where(fired, k, self.term)
            self.active &= ~ended
        self._prev = now

    @staticmethod
    def concat(rounds: list["EpisodeStats"], num_episodes: int) -> "EpisodeStats":
        """The first ``num_episodes`` envs of the rounds as one statistics object."""
        out = object.__new__(EpisodeStats)
        out.env, out.command, out.gripper, out._prev = rounds[0].env, rounds[0].command, rounds[0].gripper, None
        for key, value in vars(rounds[0]).items():
            if isinstance(value, torch.Tensor):
                setattr(out, key, torch.cat([vars(r)[key] for r in rounds])[:num_episodes])
            elif isinstance(value, dict) and key != "_prev" and value:
                setattr(out, key, {k: torch.cat([vars(r)[key][k] for r in rounds])[:num_episodes] for k in value})
        return out

    def summary(self) -> dict:
        """Means and quantiles over the finished first episodes."""
        f = ~self.active
        n = max(int(f.sum()), 1)
        dt = self.env.step_dt

        def q(x: torch.Tensor) -> dict[str, float]:
            x = x[f & torch.isfinite(x)]
            if x.numel() == 0:
                return {"median": math.nan, "p90": math.nan, "max": math.nan}
            return {"median": x.median().item(), "p90": x.quantile(0.9).item(), "max": x.max().item()}

        def ratio(num: torch.Tensor, den: torch.Tensor) -> float:
            d = den[f].sum().item()
            return num[f].sum().item() / d if d > 0 else math.nan

        names = self.env.termination_manager.active_terms
        terms = {name: (self.term[f] == k).float().sum().item() / n for k, name in enumerate(names)}
        right_time = self.first_right_step[f & self.right_any] * dt
        return {
            "episodes": int(f.sum()),
            "success": self.success[f].float().mean().item(),
            "terminations": {k: v for k, v in terms.items() if v > 0},
            "episode_length_s": (self.length[f] * dt).mean().item(),
            "force_episode_max_N": q(self.force_max),
            "force_substep_peak_N": q(self.peak_max),
            "episodes_over_10N": (self.force_max[f] > 10.0).float().mean().item(),
            "episodes_over_20N": (self.force_max[f] > 20.0).float().mean().item(),
            "force_contact_mean_N": ratio(self.contact_sum, self.contact_steps),
            "force_grasp_mean_N": ratio(self.stage_sum["grasp"], self.stage_steps["grasp"]),
            "force_grasp_contact_mean_N": ratio(self.stage_sum["grasp"], self.stage_contact["grasp"]),
            "force_drag_mean_N": ratio(self.stage_sum["drag"], self.stage_steps["drag"]),
            "force_drag_contact_mean_N": ratio(self.stage_sum["drag"], self.stage_contact["drag"]),
            "closes_per_episode": self.closes[f].mean().item(),
            "close_ee_node_distance_mm": 1e3 * ratio(self.close_dist_sum, self.closes),
            "right_lock_episodes": self.right_any[f].float().mean().item(),
            "wrong_lock_episodes": self.wrong_any[f].float().mean().item(),
            "first_right_lock_s": right_time.mean().item() if right_time.numel() else math.nan,
            "lock_losses_per_episode": self.lock_losses[f].mean().item(),
            "lock_losses_reopened_per_episode": self.lock_reopens[f].mean().item(),
            "failed_closes_per_episode": self.failed_closes[f].mean().item(),
            "failed_close_reopens_per_episode": self.failed_reopens[f].mean().item(),
            "grasp_stage_time_s": (self.grasp_steps[f] * dt).mean().item(),
            "min_ee_z_mm": q(1e3 * self.min_ee_z),
            "grasp_node_height_mm": q(1e3 * self.bump_height),
        }


def print_summary(name: str, s: dict) -> None:
    fm, fp, z = s["force_episode_max_N"], s["force_substep_peak_N"], s["min_ee_z_mm"]
    terms = ", ".join(f"{k} {v:.2f}" for k, v in s["terminations"].items())
    print(f"\n[EVAL] === {name}: {s['episodes']} episodes, success {s['success']:.3f}")
    print(f"[EVAL]   episode length {s['episode_length_s']:.2f} s")
    print(f"[EVAL]   terminations: {terms}")
    print(
        f"[EVAL]   force episode max [N] median {fm['median']:.1f} p90 {fm['p90']:.1f} max {fm['max']:.1f} |"
        f" substep peak median {fp['median']:.1f} p90 {fp['p90']:.1f} max {fp['max']:.1f} |"
        f" >10 N {s['episodes_over_10N']:.2f}, >20 N {s['episodes_over_20N']:.2f}"
    )
    print(
        f"[EVAL]   force mean [N] contact {s['force_contact_mean_N']:.2f} | grasp {s['force_grasp_mean_N']:.2f}"
        f" (contact {s['force_grasp_contact_mean_N']:.2f}) | drag {s['force_drag_mean_N']:.2f}"
        f" (contact {s['force_drag_contact_mean_N']:.2f})"
    )
    print(
        f"[EVAL]   grasp: closes/ep {s['closes_per_episode']:.2f} at {s['close_ee_node_distance_mm']:.1f} mm from the"
        f" node, right lock {s['right_lock_episodes']:.2f} (first at {s['first_right_lock_s']:.2f} s), wrong lock"
        f" {s['wrong_lock_episodes']:.2f}, lock losses/ep {s['lock_losses_per_episode']:.2f}"
        f" (re-opened {s['lock_losses_reopened_per_episode']:.2f}), failed closes/ep"
        f" {s['failed_closes_per_episode']:.2f} (re-opened {s['failed_close_reopens_per_episode']:.2f})"
    )
    print(
        f"[EVAL]   time in GRASP {s['grasp_stage_time_s']:.2f} s, min EE z [mm] median {z['median']:.1f}"
        f" p90 {z['p90']:.1f}, grasp node above the cloth at 0.5 s [mm]"
        f" median {s['grasp_node_height_mm']['median']:.1f}",
        flush=True,
    )


def print_table(results: dict[str, dict]) -> None:
    cols = ("success", "force>20N", "Fmax med", "peak med", "closes", "right", "wrong", "losses", "fail cl", "reopen")
    print("\n[EVAL] " + f"{'variant':24s}" + "".join(f"{c:>10s}" for c in cols))
    for name, s in results.items():
        row = (
            s["success"],
            s["episodes_over_20N"],
            s["force_episode_max_N"]["median"],
            s["force_substep_peak_N"]["median"],
            s["closes_per_episode"],
            s["right_lock_episodes"],
            s["wrong_lock_episodes"],
            s["lock_losses_per_episode"],
            s["failed_closes_per_episode"],
            s["failed_close_reopens_per_episode"],
        )
        print("[EVAL] " + f"{name:24s}" + "".join(f"{v:10.2f}" for v in row))


def main():
    env_cfg, agent_cfg = resolve_task_config(TASK, "rsl_rl_cfg_entry_point")
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    args_cli.device = env_cfg.sim.device
    configure(env_cfg)

    with launch_simulation(env_cfg, args_cli):
        import importlib.metadata as metadata

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

        from isaaclab_tasks.contrib.franka_cloth_drag.constrained_gripper.robotiq import (
            ROBOTIQ_CLOSED,
            pad_gap_from_joint,
        )

        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        gym_env = gym.make(TASK, cfg=env_cfg)
        env = gym_env.unwrapped
        wrapped = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
        obs = wrapped.get_observations()
        student = load_student(obs, agent_cfg, args_cli.checkpoint, wrapped.num_actions, env.device)
        command = env.command_manager.get_term("cloth_target")
        gripper = env.action_manager.get_term("gripper_action")
        stats_ref: dict[str, EpisodeStats | None] = {"stats": None}

        # record the force inside the termination manager, before the resets zero it
        force_cfg = env.termination_manager.get_term_cfg("gripper_force_limit")
        force_term = force_cfg.func

        class ForceWithRecord:
            """The force term, recording its force after every call; the manager also calls its reset."""

            def __call__(self, env_, **kwargs):
                out = force_term(env_, **kwargs)
                if stats_ref["stats"] is not None:
                    stats_ref["stats"].record_force(force_term)
                return out

            def __getattr__(self, name):
                return getattr(force_term, name)

        force_cfg.func = ForceWithRecord()

        # the student's width reading, overridable per variant
        names = env.observation_manager._group_obs_term_names["student_proprio"]
        width_cfg = env.observation_manager._group_obs_term_cfgs["student_proprio"][names.index("gripper_width")]
        true_width = width_cfg.func
        empty_close = float(pad_gap_from_joint(torch.tensor(ROBOTIQ_CLOSED), gripper.cfg.gap_curve))
        lie = {"kind": "none", "value": None}

        def width(env_, **kwargs):
            value = true_width(env_, **kwargs)
            kind = lie["kind"]
            if kind == "offset":
                return value + lie["value"]
            if kind in ("empty", "real"):
                mask = gripper.is_grasping
                forced = empty_close if kind == "empty" else args_cli.real_pinch
            elif kind == "fake_closed":
                mask = gripper.closed & ~gripper.is_grasping
                forced = args_cli.fake_lock_width
            elif kind == "fake_failed":
                mask = gripper.closed & gripper.grasp_tried & ~gripper.is_grasping
                forced = args_cli.fake_lock_width
            else:
                return value
            return torch.where(mask.unsqueeze(1), torch.full_like(value, forced), value)

        width_cfg.func = width
        bump_params = env.event_manager.get_term_cfg("reset_deformable").params
        train_height = bump_params["node_raise_height"]
        print(f"[EVAL] checkpoint {args_cli.checkpoint}")
        print(
            f"[EVAL] clean {args_cli.clean}, force stop {force_cfg.params['max_force']} N,"
            f" empty-close width {1e3 * empty_close:.2f} mm, training bump {1e3 * train_height:.1f} mm"
        )

        num_rounds = math.ceil(args_cli.num_episodes / env.num_envs)
        print(f"[EVAL] {args_cli.num_episodes} episodes per variant in {num_rounds} rounds of {env.num_envs} envs")
        results = {}
        for name, kind, value in variants():
            lie["kind"], lie["value"] = (kind, value) if kind != "bump" else ("none", None)
            bump_params["node_raise_height"] = value if kind == "bump" else train_height
            rounds = []
            for round_id in range(num_rounds):
                env.seed(args_cli.seed + round_id)  # same draws in every variant
                obs, _ = wrapped.reset()
                stats = EpisodeStats(env, command, gripper)
                stats_ref["stats"] = stats
                stats.start()
                for step in range(env.max_episode_length + 2):
                    with torch.inference_mode():
                        actions = student(obs)
                    obs, _, dones, _ = wrapped.step(actions)
                    stats.record_step(step, dones)
                    if not stats.active.any():
                        break
                stats_ref["stats"] = None
                rounds.append(stats)
            results[name] = EpisodeStats.concat(rounds, args_cli.num_episodes).summary()
            print_summary(name, results[name])

        print_table(results)
        if args_cli.out:
            with open(args_cli.out, "w") as f:
                meta = {"checkpoint": args_cli.checkpoint, "clean": args_cli.clean}
                json.dump({**meta, "variants": results}, f, indent=2)
            print(f"[EVAL] wrote {args_cli.out}")
        gym_env.close()


if __name__ == "__main__":
    main()

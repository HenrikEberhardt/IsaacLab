#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deep analysis of one cloth-drag training run from its tfevents file.

Usage:
    uv run python analyze_run.py <run_dir> [--out analysis.json] [--compare <run_dir> ...] [--md]
    uv run python analyze_run.py <run_dir> --series '<regex>' [--every 10]

The default mode writes ``analysis.json`` (see ``analyze``) and with ``--md`` prints a compact markdown
summary; ``--compare`` adds a side-by-side table of the key numbers of other runs (champion, baseline).
``--series`` prints raw curves of matching tags, every N iterations, for follow-up questions.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

STAGES = ("approach", "grasp", "drag", "goal")  # start stages; "approach" = default task start
TASK_ORDER = ("goal", "drag", "grasp", "approach")  # expected order in which task success should rise
THRESHOLDS = (0.2, 0.5, 0.8, 0.9, 0.95)
PRIMARY = "Metrics/cloth_target/task_success_approach_smoothed"
FINAL_WINDOW = 50  # iterations
CHECKPOINT_EVERY = 50  # iterations of the compact curve table


def load_with_wall_times(
    run_dir: Path, until: int | None = None
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, np.ndarray]]:
    """Load scalar values and wall times; later event files win on overlapping iterations.

    ``until`` drops iterations after it, so a longer run compares fairly with a shorter one.
    """
    files = sorted(glob.glob(str(run_dir / "events.out.tfevents.*")))
    if not files:
        raise FileNotFoundError(f"no tfevents file in {run_dir}")
    merged: dict[str, dict[int, tuple[float, float]]] = {}
    for f in files:
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags()["scalars"]:
            merged.setdefault(tag, {}).update({e.step: (e.value, e.wall_time) for e in ea.Scalars(tag)})
    series = {}
    wall_times = {}
    for tag, points in merged.items():
        steps = np.array(sorted(s for s in points if until is None or s <= until), dtype=int)
        if len(steps) == 0:
            continue
        series[tag] = (steps, np.array([points[s][0] for s in steps], dtype=float))
        wall_times[tag] = np.array([points[s][1] for s in steps], dtype=float)
    return series, wall_times


def load(run_dir: Path, until: int | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """All scalar series of a run as ``tag -> (iterations, values)``."""
    return load_with_wall_times(run_dir, until)[0]


def rolling(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) < n:
        return x.copy()
    kernel = np.ones(n) / n
    pad = np.pad(x, (n // 2, n - 1 - n // 2), mode="edge")
    return np.convolve(pad, kernel, mode="valid")


def first_crossing(steps: np.ndarray, values: np.ndarray, threshold: float) -> int | None:
    """First iteration at which the 10-iteration rolling mean reaches ``threshold``."""
    smooth = rolling(values, 10)
    idx = np.nonzero(smooth >= threshold)[0]
    return int(steps[idx[0]]) if len(idx) else None


def crossing_minutes(values: np.ndarray, wall_times: np.ndarray, threshold: float) -> float | None:
    """Elapsed training minutes to a crossing, including an estimate for the first logged iteration."""
    idx = np.nonzero(rolling(values, 10) >= threshold)[0]
    if len(idx) == 0:
        return None
    intervals = np.diff(wall_times)
    first_iteration = float(np.median(intervals[intervals > 0])) if np.any(intervals > 0) else 0.0
    return (float(wall_times[idx[0]] - wall_times[0]) + first_iteration) / 60.0


def run_dimensions(run_dir: Path) -> tuple[int, int]:
    """Return ``(num_envs, num_steps_per_env)`` from the generated parameter files."""
    env_text = (run_dir / "params/env.yaml").read_text()
    agent_text = (run_dir / "params/agent.yaml").read_text()
    num_envs = re.search(r"^\s+num_envs:\s+(\d+)\s*$", env_text, re.MULTILINE)
    num_steps = re.search(r"^num_steps_per_env:\s+(\d+)\s*$", agent_text, re.MULTILINE)
    if num_envs is None or num_steps is None:
        raise ValueError(f"could not read rollout dimensions from {run_dir / 'params'}")
    return int(num_envs.group(1)), int(num_steps.group(1))


def r(x: float | None, digits: int = 4) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), digits)


def curve_summary(steps: np.ndarray, values: np.ndarray, thresholds=THRESHOLDS) -> dict:
    final = values[steps > steps[-1] - FINAL_WINDOW]
    return {
        "final_mean": r(final.mean()),
        "auc": r(values.mean()),  # mean over all logged iterations
        "max": r(values.max()),
        "max_it": int(steps[values.argmax()]),
        "cross": {str(t): first_crossing(steps, values, t) for t in thresholds},
    }


def oscillation(steps: np.ndarray, values: np.ndarray) -> dict:
    """Amplitude and dominant period of the residual around a 21-iteration moving average."""
    if len(values) < 40:
        return {}
    resid = values - rolling(values, 21)
    spec = np.abs(np.fft.rfft(resid - resid.mean()))
    freqs = np.fft.rfftfreq(len(resid), d=1.0)
    k = int(spec[1:].argmax()) + 1
    half = len(values) // 2
    return {
        "residual_std": r(resid.std()),
        "residual_std_second_half": r(resid[half:].std()),
        "peak_to_peak_second_half": r(np.ptp(values[half:])),
        "dominant_period_it": r(1.0 / freqs[k], 1) if freqs[k] > 0 else None,
    }


def lagged_corr(a: np.ndarray, b: np.ndarray, max_lag: int = 60) -> dict:
    """Lag (in iterations) at which changes of ``a`` best correlate with later changes of ``b``."""
    da, db = np.diff(rolling(a, 5)), np.diff(rolling(b, 5))
    best = (0, 0.0)
    for lag in range(0, min(max_lag, len(da) - 20)):
        x, y = da[: len(da) - lag], db[lag:]
        if x.std() < 1e-9 or y.std() < 1e-9:
            continue
        c = float(np.corrcoef(x, y)[0, 1])
        if abs(c) > abs(best[1]):
            best = (lag, c)
    return {"lag_it": best[0], "corr": r(best[1], 3)}


def aligned(series: dict, a: str, b: str) -> tuple[np.ndarray, np.ndarray] | None:
    if a not in series or b not in series:
        return None
    sa, va = series[a]
    sb, vb = series[b]
    common, ia, ib = np.intersect1d(sa, sb, return_indices=True)
    return (va[ia], vb[ib]) if len(common) > 30 else None


def stats(steps: np.ndarray, values: np.ndarray) -> dict:
    last = values[steps > steps[-1] - FINAL_WINDOW]
    med = rolling(np.abs(values), 21)
    spikes = int(np.sum(np.abs(values) > 5 * np.maximum(med, 1e-8))) if len(values) > 21 else 0
    slope = np.polyfit(steps[-100:], values[-100:], 1)[0] if len(values) >= 20 else np.nan
    return {
        "first": r(values[0]),
        "mean": r(values.mean()),
        "final_mean": r(last.mean()),
        "min": r(values.min()),
        "max": r(values.max()),
        "max_it": int(steps[values.argmax()]),
        "spikes_5x_median": spikes,
        "slope_last_100_per_100it": r(slope * 100),
    }


def at_iterations(steps: np.ndarray, values: np.ndarray, every: int = CHECKPOINT_EVERY) -> dict:
    out = {}
    for it in range(0, int(steps[-1]) + 1, every):
        idx = np.nonzero(steps <= it)[0]
        if len(idx):
            lo = max(0, idx[-1] - 4)
            out[str(it)] = r(values[lo : idx[-1] + 1].mean(), 3)  # 5-iteration mean ending at it
    out[str(int(steps[-1]))] = r(values[-5:].mean(), 3)
    return out


def analyze(run_dir: Path, until: int | None = None) -> dict:
    s, wall_times = load_with_wall_times(run_dir, until)
    num_envs, num_steps_per_env = run_dimensions(run_dir)
    out: dict = {
        "run_dir": str(run_dir),
        "num_tags": len(s),
        "num_envs": num_envs,
        "num_steps_per_env": num_steps_per_env,
    }
    if "Train/mean_reward" in s:
        out["last_iteration"] = int(s["Train/mean_reward"][0][-1])
    if "Perf/total_fps" in s:
        out["fps_mean"] = r(s["Perf/total_fps"][1].mean(), 0)

    if PRIMARY in s:
        primary = curve_summary(*s[PRIMARY])
        primary["cross_transitions"] = {
            threshold: crossing * num_envs * num_steps_per_env if crossing is not None else None
            for threshold, crossing in primary["cross"].items()
        }
        primary["cross_minutes"] = {
            str(threshold): r(crossing_minutes(s[PRIMARY][1], wall_times[PRIMARY], threshold), 2)
            for threshold in THRESHOLDS
        }
        out["primary"] = {"tag": PRIMARY, **primary}
    else:
        out["primary"] = None

    # Stage and task success per start stage.
    stages = {}
    for st in STAGES:
        entry = {}
        for kind in ("task", "stage"):
            tag = f"Metrics/cloth_target/{kind}_success_{st}_smoothed"
            if tag in s:
                entry[kind] = curve_summary(*s[tag])
        if entry:
            stages[st] = entry
    out["stages"] = stages

    # Shape: do stage successes rise together, does task success rise goal -> drag -> grasp -> approach.
    shape = {}
    for t in ("0.5", "0.8"):
        stage_cross = {st: v["stage"]["cross"][t] for st, v in stages.items() if "stage" in v}
        reached = [x for x in stage_cross.values() if x is not None]
        shape[f"stage_cross_{t}"] = stage_cross
        shape[f"stage_cross_{t}_spread_it"] = (max(reached) - min(reached)) if len(reached) > 1 else None
        task_cross = {st: v["task"]["cross"][t] for st, v in stages.items() if "task" in v}
        order = sorted((x, st) for st, x in task_cross.items() if x is not None)
        shape[f"task_cross_{t}"] = task_cross
        shape[f"task_order_{t}"] = [st for _, st in order]
        expected = [st for st in TASK_ORDER if task_cross.get(st) is not None]
        shape[f"task_order_{t}_as_expected"] = [st for _, st in order] == expected if order else None
    out["shape"] = shape

    # Stage ratios: level, movement, oscillation, coupling to stage success.
    shares = {}
    for st in STAGES:
        tag = f"Reset/share_{st}_raw"
        if tag not in s:
            continue
        steps, values = s[tag]
        moved = np.nonzero(np.abs(values - values[0]) > 0.02)[0]
        entry = {
            "first": r(values[0]),
            "final_mean": r(values[steps > steps[-1] - FINAL_WINDOW].mean()),
            "min": r(values.min()),
            "max": r(values.max()),
            "first_move_it": int(steps[moved[0]]) if len(moved) else None,
            "oscillation": oscillation(steps, values),
            "curve": at_iterations(steps, values),
        }
        pair = aligned(s, tag, f"Metrics/cloth_target/stage_success_{st}_smoothed")
        if pair is not None:
            entry["share_to_own_stage_success"] = lagged_corr(*pair)
        pair = aligned(s, f"Metrics/cloth_target/task_success_{st}_smoothed", tag)
        if pair is not None:
            entry["own_task_success_to_share"] = lagged_corr(*pair)
        shares[st] = entry
    out["shares"] = shares
    out["frontier"] = {
        tag.removeprefix("Reset/"): {"mean": r(v.mean()), "final_mean": r(v[st > st[-1] - FINAL_WINDOW].mean())}
        for tag, (st, v) in s.items()
        if tag.startswith("Reset/frontier_")
    }

    # Terminations.
    term = {}
    for tag, (steps, values) in s.items():
        if tag.startswith("Episode_Termination/"):
            above = np.nonzero(values > 0.01)[0]
            term[tag.split("/", 1)[1]] = {
                "final_mean": r(values[steps > steps[-1] - FINAL_WINDOW].mean()),
                "peak": r(values.max()),
                "peak_it": int(steps[values.argmax()]),
                "first_it_above_1pct": int(steps[above[0]]) if len(above) else None,
            }
    out["terminations"] = term

    # Grasp quality.
    out["grasp"] = {
        tag.split("/", 1)[1]: {**stats(*s[tag]), "curve": at_iterations(*s[tag], every=100)}
        for tag in s
        if tag.startswith("Grasp/")
    }

    # PPO health.
    ppo_tags = [t for t in s if t.startswith(("Loss/", "Policy/", "Train/")) and not t.endswith("/time")]
    ppo_tags += [t for t in s if t.startswith("Action/policy_std_")]
    out["ppo"] = {t: stats(*s[t]) for t in ppo_tags}

    out["rewards_final"] = {
        t.split("/", 1)[1]: r(v[st > st[-1] - FINAL_WINDOW].mean())
        for t, (st, v) in s.items()
        if t.startswith("Episode_Reward/")
    }
    out["curves"] = {
        t: at_iterations(*s[t])
        for t in [PRIMARY]
        + [f"Metrics/cloth_target/{k}_success_{st}_smoothed" for k in ("task", "stage") for st in STAGES]
        if t in s
    }
    return out


def key_numbers(a: dict) -> dict:
    """Flat dict of the numbers used for run-to-run comparison."""
    k = {}
    if a.get("primary"):
        p = a["primary"]
        k.update(
            {
                "primary_final": p["final_mean"],
                "primary_auc": p["auc"],
                "primary_cross_0.5": p["cross"]["0.5"],
                "primary_cross_0.8": p["cross"]["0.8"],
                "primary_cross_0.95": p["cross"]["0.95"],
                "primary_cross_0.95_transitions": p["cross_transitions"]["0.95"],
                "primary_cross_0.95_minutes": p["cross_minutes"]["0.95"],
            }
        )
    for st, v in a["stages"].items():
        for kind in ("stage", "task"):
            if kind in v:
                k[f"{kind}_{st}_final"] = v[kind]["final_mean"]
                k[f"{kind}_{st}_cross_0.5"] = v[kind]["cross"]["0.5"]
    k["stage_cross_0.5_spread"] = a["shape"].get("stage_cross_0.5_spread_it")
    for st, v in a["shares"].items():
        k[f"share_{st}_osc_2nd_half"] = v["oscillation"].get("residual_std_second_half")
    for name in ("attempt_right_lock_rate", "attempt_lock_rate", "wrong_lock_nodes", "attempts_per_episode"):
        if name in a["grasp"]:
            k[f"grasp_{name}"] = a["grasp"][name]["final_mean"]
    for name, v in a["terminations"].items():
        if name not in ("success", "time_out") and (v["peak"] or 0) > 0.005:
            k[f"term_{name}_final"] = v["final_mean"]
    for t in ("Policy/mean_std", "Loss/value", "Loss/entropy", "Loss/learning_rate", "Loss/surrogate"):
        if t in a["ppo"]:
            k[f"{t}_final"] = a["ppo"][t]["final_mean"]
    k["last_iteration"] = a.get("last_iteration")
    return k


def to_markdown(a: dict, others: list[dict]) -> str:
    runs = [a, *others]
    names = [Path(x["run_dir"]).name for x in runs]
    keys = list(dict.fromkeys(k for x in runs for k in key_numbers(x)))
    rows = [key_numbers(x) for x in runs]
    lines = ["| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(runs)]
    for key in keys:
        lines.append(f"| {key} | " + " | ".join(str(row.get(key, "-")) for row in rows) + " |")
    sh = a["shape"]
    lines += [
        "",
        f"task success order (0.5): {sh.get('task_order_0.5')} expected {list(TASK_ORDER)} -> "
        f"{sh.get('task_order_0.5_as_expected')}",
        f"stage success crossing 0.5: {sh.get('stage_cross_0.5')} (spread {sh.get('stage_cross_0.5_spread_it')} it)",
    ]
    return "\n".join(lines)


def print_series(run_dir: Path, pattern: str, every: int) -> None:
    s = load(run_dir)
    tags = [t for t in sorted(s) if re.search(pattern, t)]
    if not tags:
        print(f"no tag matches {pattern!r}")
        return
    last = max(int(s[t][0][-1]) for t in tags)
    print("it\t" + "\t".join(tags))
    for it in range(0, last + 1, every):
        row = []
        for t in tags:
            steps, values = s[t]
            idx = np.nonzero(steps == it)[0]
            row.append(f"{values[idx[0]]:.4g}" if len(idx) else "-")
        print(f"{it}\t" + "\t".join(row))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, help="Write the full analysis as JSON here.")
    parser.add_argument("--compare", type=Path, nargs="*", default=[], help="Other runs for the comparison table.")
    parser.add_argument("--md", action="store_true", help="Print the markdown summary.")
    parser.add_argument("--series", help="Print raw curves of tags matching this regex instead.")
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--until", type=int, help="Ignore iterations after this one (all runs).")
    args = parser.parse_args()

    if args.series:
        print_series(args.run_dir, args.series, args.every)
        return 0
    a = analyze(args.run_dir, args.until)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(a, indent=1))
        print(f"wrote {args.out}")
    if args.md or not args.out:
        print(to_markdown(a, [analyze(d, args.until) for d in args.compare]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

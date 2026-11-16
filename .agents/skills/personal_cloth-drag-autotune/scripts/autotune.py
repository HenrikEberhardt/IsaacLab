#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""State, launch, watch, kill, promote and report for the overnight cloth-drag autotune.

All state lives in ``logs/autotune/<night>/state.json``; ``logs/autotune/current`` names the active night.

Subcommands (run from anywhere with ``uv run python autotune.py <cmd> ...``):

    init     --goal TEXT --stop HH:MM [--iterations 400] [--base-override K=V ...]
    status
    launch   --run-id g1a --ref <commit|tag> --slug SLUG --hypothesis TEXT
    watch    [--interval 60]          (long running; exits with one ``EVENT ...`` line)
    update   --run-id g1a KEY=VALUE ...
    promote  --run-id g1a
    kill     --run-id g1a --reason TEXT
    report   [--final]

Exit codes of ``launch``: 0 launched, 3 no free GPU, 4 machine load too high, 5 smoke run failed,
6 guardrail violation, 7 full run did not start.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/hdd1/home/heberhardt/IsaacLab")
SKILL = REPO / ".agents/skills/personal_cloth-drag-autotune"
PICK_GPU = REPO / ".agents/skills/personal_gpu-preflight/scripts/pick_gpu.sh"
AUTOTUNE_DIR = REPO / "logs/autotune"
TASK = "IsaacContrib-Drag-Cloth-Franka"
SEED = 42  # fixed for every run of every night; seeds are never varied
SMOKE_ITERATIONS = 5
MIN_RAM_GIB = 16
MIN_DISK_GIB = 20
STALL_MIN = 15  # minutes without log output before a run counts as stalled
CONTENTION_FREE_MIB = 2048  # free GPU memory below which a foreign process on our GPU forces us to yield
GPU_FREE_EVENT_EVERY_MIN = 30

sys.path.insert(0, str(SKILL / "scripts"))
import check_guardrails  # noqa: E402

ANSI = re.compile(r"\x1b\[[0-9;]*m")


# ----------------------------------------------------------------------------------------------- helpers
def now() -> dt.datetime:
    return dt.datetime.now().replace(microsecond=0)


def sh(*cmd: str, check: bool = True, **kw) -> str:
    return subprocess.run(cmd, cwd=REPO, check=check, capture_output=True, text=True, **kw).stdout


def git(*args: str, check: bool = True) -> str:
    return sh("git", *args, check=check).strip()


def night_dir() -> Path:
    current = AUTOTUNE_DIR / "current"
    if not current.exists():
        sys.exit("no active autotune night (logs/autotune/current missing); run init first")
    return AUTOTUNE_DIR / current.read_text().strip()


def load_state() -> dict:
    return json.loads((night_dir() / "state.json").read_text())


def _write_state(state: dict) -> None:
    path = night_dir() / "state.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(path)


@contextlib.contextmanager
def edit_state():
    """Locked read-modify-write of ``state.json``; the watcher and launch may run at the same time."""
    with (night_dir() / "state.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = load_state()
        yield state
        _write_state(state)


def edit_run(run_id: str, **fields) -> dict:
    with edit_state() as state:
        run = find_run(state, run_id)
        run.update(fields)
        return dict(run)


def find_run(state: dict, run_id: str) -> dict:
    for run in state["runs"]:
        if run["id"] == run_id:
            return run
    sys.exit(f"unknown run id {run_id}")


def log(msg: str) -> None:
    print(msg, flush=True)


def tmux_alive(session: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", f"={session}"], capture_output=True).returncode == 0


def tmux_pane_pid(session: str) -> int | None:
    out = subprocess.run(
        ["tmux", "list-panes", "-t", f"={session}", "-F", "#{pane_pid}"], capture_output=True, text=True
    ).stdout.split()
    return int(out[0]) if out else None


def process_tree(root: int) -> set[int]:
    """``root`` and all its descendants."""
    children: dict[int, list[int]] = {}
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
            children.setdefault(int(fields[1]), []).append(int(stat.parent.name))
        except (OSError, IndexError, ValueError):
            continue
    tree, todo = set(), [root]
    while todo:
        pid = todo.pop()
        if pid not in tree:
            tree.add(pid)
            todo += children.get(pid, [])
    return tree


def gpu_table() -> tuple[dict[int, dict], list[tuple[int, int, int]]]:
    """``{index: {mem_used, mem_total}}`` and compute apps as ``(gpu_index, pid, mem_mib)``."""
    gpus, uuid_to_idx = {}, {}
    for line in sh(
        "nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total", "--format=csv,noheader,nounits"
    ).splitlines():
        idx, uuid, used, total = (x.strip() for x in line.split(","))
        gpus[int(idx)] = {"mem_used": int(used), "mem_total": int(total)}
        uuid_to_idx[uuid] = int(idx)
    apps = []
    for line in sh(
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits", check=False
    ).splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 3 and parts[0] in uuid_to_idx:
            apps.append((uuid_to_idx[parts[0]], int(parts[1]), int(parts[2]) if parts[2].isdigit() else 0))
    return gpus, apps


def pick_gpu() -> tuple[int | None, str]:
    """Fresh preflight; returns (gpu index or None, preflight output)."""
    proc = subprocess.run(["bash", str(PICK_GPU), "--need", "1"], capture_output=True, text=True)
    match = re.search(r"FREE_GPU=(\d+)", proc.stdout)
    return (int(match.group(1)) if proc.returncode == 0 and match else None), proc.stdout


def machine_ok() -> tuple[bool, str]:
    load1 = float(Path("/proc/loadavg").read_text().split()[0])
    cores = os.cpu_count() or 1
    meminfo = dict(re.findall(r"(\w+):\s+(\d+)", Path("/proc/meminfo").read_text()))
    ram_gib = int(meminfo["MemAvailable"]) / 2**20
    disk_gib = shutil.disk_usage(REPO).free / 2**30
    problems = []
    if load1 > cores:
        problems.append(f"load {load1:.1f} > {cores} cores")
    if ram_gib < MIN_RAM_GIB:
        problems.append(f"available RAM {ram_gib:.1f} GiB < {MIN_RAM_GIB}")
    if disk_gib < MIN_DISK_GIB:
        problems.append(f"free disk {disk_gib:.0f} GiB < {MIN_DISK_GIB}")
    return not problems, "; ".join(
        problems
    ) or f"load {load1:.1f}/{cores}, RAM {ram_gib:.0f} GiB, disk {disk_gib:.0f} GiB"


def find_run_dir(run_name: str) -> Path | None:
    hits = sorted((REPO / "logs/rsl_rl").glob(f"*/*_{run_name}"))
    return hits[-1] if hits else None


def last_iteration(text: str) -> tuple[int | None, int | None]:
    hits = re.findall(r"Learning iteration (\d+)/(\d+)", text)
    return (int(hits[-1][0]), int(hits[-1][1])) if hits else (None, None)


def run_dimensions(run_dir: Path) -> tuple[int, int]:
    """Return ``(num_envs, num_steps_per_env)`` from a generated run's parameter files."""
    env_text = (run_dir / "params/env.yaml").read_text()
    agent_text = (run_dir / "params/agent.yaml").read_text()
    num_envs = re.search(r"^\s+num_envs:\s+(\d+)\s*$", env_text, re.MULTILINE)
    num_steps = re.search(r"^num_steps_per_env:\s+(\d+)\s*$", agent_text, re.MULTILINE)
    if num_envs is None or num_steps is None:
        raise ValueError(f"could not read rollout dimensions from {run_dir / 'params'}")
    return int(num_envs.group(1)), int(num_steps.group(1))


def transitions(iterations: int | None, run: dict) -> int | None:
    """Environment transitions for ``iterations`` of a registered run."""
    if iterations is None or "num_envs" not in run or "num_steps_per_env" not in run:
        return None
    return iterations * run["num_envs"] * run["num_steps_per_env"]


def tail(path: Path, nbytes: int = 200_000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - nbytes))
        return ANSI.sub("", f.read().decode(errors="replace"))


def train_args(state: dict, run_name: str, iterations: int) -> list[str]:
    """Training command. Variants differ only by their commit; the base overrides are the same for the night."""
    return [
        "uv", "run", "isaaclab", "train",
        "--rl_library", "rsl_rl", "--task", TASK, "--seed", str(SEED),
        "--max_iterations", str(iterations), "--run_name", run_name,
        *state["base_overrides"],
    ]  # fmt: skip


def ensure_clean_tree() -> None:
    dirty = [line for line in git("status", "--porcelain", "--untracked-files=no").splitlines() if line.strip()]
    if dirty:
        sys.exit(
            "working tree has uncommitted changes to tracked files; commit them on the autotune branch first:\n"
            + "\n".join(dirty)
        )


# ----------------------------------------------------------------------------------------------- commands
def cmd_init(args) -> int:
    current = AUTOTUNE_DIR / "current"
    if current.exists():
        state_file = AUTOTUNE_DIR / current.read_text().strip() / "state.json"
        if state_file.exists() and json.loads(state_file.read_text())["status"] == "running":
            log(f"RESUME {current.read_text().strip()} (night still running; use status, not init)")
            return 0
    start_branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if start_branch.startswith("autotune/"):
        sys.exit(f"already on {start_branch}; start from your working branch")
    stamp = now().strftime("%Y-%m-%d_%H%M")
    branch = f"autotune/{stamp}"
    git("switch", "-c", branch)

    # Snapshot uncommitted work, so every run starts from a commit. Untracked files outside the task are left alone.
    snapshot = None
    tracked_dirty = git("status", "--porcelain", "--untracked-files=no")
    untracked = git("ls-files", "--others", "--exclude-standard", "source").split()
    if tracked_dirty or untracked:
        git("add", "-u")
        if untracked:
            git("add", "--", *untracked)
        git("commit", "-m", "Snapshot uncommitted work before autotune")
        snapshot = git("rev-parse", "HEAD")
    start_commit = git("rev-parse", "HEAD")

    d = AUTOTUNE_DIR / stamp
    (d / "runs").mkdir(parents=True, exist_ok=True)
    stop = dt.datetime.combine(now().date(), dt.time.fromisoformat(args.stop))
    if stop <= now():
        stop += dt.timedelta(days=1)
    state = {
        "night": stamp,
        "status": "running",
        "goal": args.goal,
        "started_at": now().isoformat(),
        "stop_at": stop.isoformat(),
        "interim_report_written": False,
        "iterations": args.iterations,
        "base_overrides": args.base_override,
        "start_branch": start_branch,
        "branch": branch,
        "snapshot_commit": snapshot,
        "start_commit": start_commit,
        "champion": None,  # {"run_id", "commit"}
        "ref_params_dir": None,  # smoke run of the baseline; frozen params are compared against it
        "last_gpu_free_event": None,
        "runs": [],
    }
    (d / "state.json").write_text(json.dumps(state, indent=1))
    current.write_text(stamp)
    lessons = AUTOTUNE_DIR / "lessons.md"
    if not lessons.exists():
        lessons.write_text(
            "# Autotune lessons across nights\n\nOne bullet per durable finding, with night and run id.\n"
        )
    (d / "journal.md").write_text(
        f"# Autotune {stamp}\n\nGoal: {args.goal}\n\nStop at: {stop:%Y-%m-%d %H:%M}\n\n"
        f"Start branch `{start_branch}` at `{start_commit[:9]}`"
        + (f" (snapshot commit `{snapshot[:9]}` holds uncommitted work)" if snapshot else "")
        + f"\n\nBase overrides: `{' '.join(args.base_override) or '-'}`\n\n## Runs\n"
    )
    log(f"NIGHT {stamp} branch {branch} start {start_commit[:9]} stop {stop:%Y-%m-%d %H:%M} dir {d}")
    return 0


def cmd_status(args) -> int:
    state = load_state()
    log(f"night {state['night']} status {state['status']} goal: {state['goal']}")
    log(f"stop at {state['stop_at']}  branch {state['branch']}  champion {state['champion']}")
    for run in state["runs"]:
        text = tail(Path(run["log"]), 50_000) if run.get("log") else ""
        it, total = last_iteration(text)
        done = transitions(it, run)
        budget = transitions(run.get("iterations", total), run)
        transition_status = f" transitions {done}/{budget}" if budget is not None else ""
        log(
            f"{run['id']:5s} {run['status']:10s} gpu {run.get('gpu')} it {it}/{total}{transition_status} "
            f"{run['slug']}  "
            f"decision={run.get('decision', '-')}"
        )
    return 0


def cmd_launch(args) -> int:
    state = load_state()
    if any(r["id"] == args.run_id for r in state["runs"]):
        sys.exit(f"run id {args.run_id} already exists")
    ensure_clean_tree()
    ok, info = machine_ok()
    if not ok:
        log(f"MACHINE_BUSY {info}")
        return 4
    gpu, preflight = pick_gpu()
    if gpu is None:
        log(preflight)
        log("NO_FREE_GPU nothing started")
        return 3
    log(f"preflight: GPU {gpu} free; {info}")

    d = night_dir() / "runs" / args.run_id
    d.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.fromisoformat(state["started_at"]).strftime("%m%d")
    run_name = f"autotune_{stamp}_{args.run_id}_{args.slug}_s{SEED}"
    session = f"autotune-{state['night']}-{args.run_id}"
    commit = git("rev-parse", f"{args.ref}^{{commit}}")
    branch = state["branch"]

    git("switch", "--detach", commit)
    try:
        # Guardrails on the source of this variant.
        violations = check_guardrails.check_source(state["start_commit"])
        if violations:
            log("\n".join(f"VIOLATION {v}" for v in violations))
            return 6

        # Smoke run on the chosen GPU.
        smoke_name = f"{run_name}_smoke"
        smoke_cmd = train_args(state, smoke_name, SMOKE_ITERATIONS) + [
            "agent.logger=tensorboard", "agent.experiment_name=autotune_smoke",
        ]  # fmt: skip
        log(f"smoke run on GPU {gpu}: {shlex.join(smoke_cmd)}")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"}
        with (d / "smoke.log").open("w") as f:
            proc = subprocess.Popen(
                smoke_cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                rc = proc.wait(timeout=25 * 60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)  # our own child process group only
                rc = -1
        it, _ = last_iteration(tail(d / "smoke.log"))
        if rc != 0 or it is None or it < SMOKE_ITERATIONS - 1:
            log(f"SMOKE_FAILED rc={rc} last_iteration={it}; see {d / 'smoke.log'}")
            log(tail(d / "smoke.log", 4000))
            return 5
        smoke_dir = find_run_dir(smoke_name)
        with edit_state() as fresh:
            ref_params = fresh["ref_params_dir"]
            if ref_params is None:
                fresh["ref_params_dir"] = str(smoke_dir)
        if ref_params is None:
            log(f"reference params set from this smoke run: {smoke_dir}")
        else:
            violations = check_guardrails.check_params(smoke_dir, Path(ref_params))
            if violations:
                log("\n".join(f"VIOLATION {v}" for v in violations))
                return 6
        num_envs, num_steps_per_env = run_dimensions(smoke_dir)

        # The GPU must still be free after the smoke run. Our own smoke run leaves utilization above the
        # threshold for a few seconds after it exits, so re-check a few times (thresholds unchanged).
        for attempt in range(4):
            if attempt:
                time.sleep(15)
            gpu2, preflight = pick_gpu()
            free_now = re.search(rf"^{gpu}\s+\S+\s+\S+\s+free", preflight, re.M)
            if free_now:
                break
        if not free_now:
            log(preflight)
            log(f"NO_FREE_GPU GPU {gpu} was taken during the smoke run; nothing started")
            return 3

        # Register before starting, so the session is always known to kill/watch.
        run = {
            "id": args.run_id, "slug": args.slug, "hypothesis": args.hypothesis, "ref": args.ref,
            "commit": commit, "parent_champion": (state["champion"] or {}).get("commit"),
            "run_name": run_name, "session": session, "gpu": gpu,
            "log": str(d / "train.log"), "exit_file": str(d / "train.exit"), "run_dir": None,
            "status": "starting", "launched_at": now().isoformat(), "pids": [],
            "iterations": args.iterations or state["iterations"],
            "num_envs": num_envs, "num_steps_per_env": num_steps_per_env,
        }  # fmt: skip
        with edit_state() as fresh:
            if any(r["id"] == args.run_id for r in fresh["runs"]):
                sys.exit(f"run id {args.run_id} already exists")
            fresh["runs"].append(run)

        full_cmd = train_args(state, run_name, run["iterations"])
        inner = (
            f"set -o pipefail; cd {shlex.quote(str(REPO))}; "
            f"CUDA_VISIBLE_DEVICES={gpu} PYTHONUNBUFFERED=1 {shlex.join(full_cmd)} 2>&1 "
            f"| tee {shlex.quote(run['log'])}; "
            f"echo $? > {shlex.quote(run['exit_file'])}; exec bash"
        )
        subprocess.run(["tmux", "new-session", "-d", "-s", session, "-c", str(REPO), "bash", "-c", inner], check=True)
        (d / "command.txt").write_text(shlex.join(full_cmd) + "\n")
        log(f"started tmux session {session} on GPU {gpu}: {shlex.join(full_cmd)}")

        # Keep this commit checked out until the run has imported its code and logged an iteration.
        deadline = time.time() + 25 * 60
        while time.time() < deadline:
            it, _ = last_iteration(tail(Path(run["log"])))
            if it is not None and it >= 1:
                break
            if Path(run["exit_file"]).exists():
                break
            time.sleep(15)
        it, _ = last_iteration(tail(Path(run["log"])))
        pane = tmux_pane_pid(session)
        pids = sorted(process_tree(pane)) if pane else []
        run_dir = str(find_run_dir(run_name) or "")
        edit_run(args.run_id, pids=pids, run_dir=run_dir)
        if it is None or it < 1:
            edit_run(args.run_id, status="failed_start")
            log(f"FULL_RUN_NOT_STARTED; see {run['log']}")
            log(tail(Path(run["log"]), 4000))
            return 7
        _, apps = gpu_table()
        on_gpu = sorted({a[0] for a in apps if a[1] in set(pids)})
        edit_run(args.run_id, status="running")
        log(f"LAUNCHED {args.run_id} session={session} gpu={gpu} (nvidia-smi sees it on {on_gpu}) run_dir={run_dir}")
        log(f"attach: tmux attach -t {session}")
        return 0
    finally:
        git("switch", branch, check=False)


def run_status_line(run: dict) -> tuple[str, str | None]:
    """One status line for ``status.log`` and an event type if the run needs attention."""
    text = tail(Path(run["log"]))
    it, total = last_iteration(text)

    def metric(name: str) -> str:
        hits = re.findall(rf"{re.escape(name)}:\s+(-?[\d.]+(?:e-?\d+)?|nan|inf)", text)
        return hits[-1] if hits else "-"

    line = (
        f"{run['id']} it {it}/{total} task_appr {metric('Metrics/cloth_target/task_success_approach_smoothed')} "
        f"stage g/d/G {metric('Metrics/cloth_target/stage_success_grasp_smoothed')}/"
        f"{metric('Metrics/cloth_target/stage_success_drag_smoothed')}/"
        f"{metric('Metrics/cloth_target/stage_success_goal_smoothed')} "
        f"std {metric('Mean action std')} sps {metric('Steps per second')} gpu {run['gpu']}"
    )
    exit_file = Path(run["exit_file"])
    if exit_file.exists():
        rc = exit_file.read_text().strip()
        finished = rc == "0" and it is not None and total is not None and it >= total - 1
        return line + f" exit={rc}", "finished" if finished else "crashed"
    if not tmux_alive(run["session"]):
        return line + " session gone", "crashed"
    if Path(run["log"]).exists() and time.time() - Path(run["log"]).stat().st_mtime > STALL_MIN * 60:
        return line + f" no output for {STALL_MIN} min", "stalled"
    if re.search(r"(task_success_approach_smoothed|Mean reward):\s+nan", text[-20000:]):
        return line + " NaN", "nonfinite"
    return line + " ok", None


def cmd_watch(args) -> int:
    status_log = night_dir() / "status.log"
    last_gpu_check = 0.0
    while True:
        state = load_state()
        stamp = now().strftime("%H:%M")
        active = [r for r in state["runs"] if r["status"] == "running"]
        lines, event = [], None
        for run in active:
            line, ev = run_status_line(run)
            lines.append(f"{stamp} {line}")
            if ev and event is None:
                event = f"{ev} {run['id']}"
                if ev in ("finished", "crashed"):
                    edit_run(run["id"], status=ev, ended_at=now().isoformat())
        # Yield: foreign process on a GPU we use while its memory is tight.
        if event is None and active:
            gpus, apps = gpu_table()
            for run in active:
                pane = tmux_pane_pid(run["session"])
                ours = process_tree(pane) if pane else set()
                foreign = [a for a in apps if a[0] == run["gpu"] and a[1] not in ours]
                g = gpus.get(run["gpu"])
                free_mib = g["mem_total"] - g["mem_used"] if g else None
                if foreign and free_mib is not None and free_mib < CONTENTION_FREE_MIB:
                    event = f"contention {run['id']} foreign pids {[a[1] for a in foreign]} free {free_mib} MiB"
                    break
        if not lines:
            lines.append(f"{stamp} no active runs")
        with status_log.open("a") as f:
            f.write("\n".join(lines) + "\n")
        if (
            event is None
            and not state["interim_report_written"]
            and now() >= dt.datetime.fromisoformat(state["stop_at"])
        ):
            event = "stop_time"
        if event is None and len(active) < 4 and time.time() - last_gpu_check > 600:
            last_gpu_check = time.time()
            last = state.get("last_gpu_free_event")
            recent = last and (now() - dt.datetime.fromisoformat(last)).total_seconds() < GPU_FREE_EVENT_EVERY_MIN * 60
            if not recent or not active:
                gpu, _ = pick_gpu()
                if gpu is not None:
                    event = f"gpu_free {gpu}"
                    with edit_state() as fresh:
                        fresh["last_gpu_free_event"] = now().isoformat()
        if event:
            with status_log.open("a") as f:
                f.write(f"{stamp} EVENT {event}\n")
            log("\n".join(lines))
            log(f"EVENT {event}")
            return 0
        time.sleep(args.interval)


def cmd_update(args) -> int:
    fields = {}
    for kv in args.pairs:
        key, _, value = kv.partition("=")
        try:
            fields[key] = json.loads(value)
        except json.JSONDecodeError:
            fields[key] = value
    edit_run(args.run_id, **fields)
    log(f"updated {args.run_id}: {', '.join(args.pairs)}")
    return 0


def cmd_promote(args) -> int:
    state = load_state()
    run = find_run(state, args.run_id)
    ensure_clean_tree()
    head = git("rev-parse", "HEAD")
    if git("rev-parse", "--abbrev-ref", "HEAD") != state["branch"]:
        sys.exit(f"not on {state['branch']}")
    if subprocess.run(["git", "merge-base", "--is-ancestor", head, run["commit"]], cwd=REPO).returncode == 0:
        git("merge", "--ff-only", run["commit"])
        how = "fast-forward"
    else:
        # The champion moved since this variant was branched: replay only the variant's own commits.
        base = run["parent_champion"] or state["start_commit"]
        proc = subprocess.run(
            ["git", "cherry-pick", f"{base}..{run['commit']}"], cwd=REPO, capture_output=True, text=True
        )
        if proc.returncode != 0:
            subprocess.run(["git", "cherry-pick", "--abort"], cwd=REPO)
            log(f"PROMOTE_CONFLICT {args.run_id}: resolve by hand as a new variant on top of {head[:9]}\n{proc.stderr}")
            return 1
        how = "cherry-pick onto the newer champion (combination not yet trained)"
    champion = {"run_id": args.run_id, "commit": git("rev-parse", "HEAD"), "how": how}
    with edit_state() as fresh:
        fresh["champion"] = champion
        find_run(fresh, args.run_id)["decision"] = "promoted"
    log(f"PROMOTED {args.run_id} by {how}; branch tip {champion['commit'][:9]}")
    return 0


def cmd_kill(args) -> int:
    """Stop one of OUR runs: only the registered tmux session, only processes of that run."""
    state = load_state()
    run = find_run(state, args.run_id)
    session = run["session"]
    if not session.startswith(f"autotune-{state['night']}-"):
        sys.exit(f"refusing: {session} is not an autotune session of this night")
    me = os.getuid()
    pane = tmux_pane_pid(session)
    pids = process_tree(pane) if pane else set()
    subprocess.run(["tmux", "kill-session", "-t", f"={session}"], capture_output=True)
    time.sleep(10)
    # Survivors: only processes that were in the session's tree, owned by us, and carry this run's name.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        survivors = []
        for pid in pids:
            try:
                if os.stat(f"/proc/{pid}").st_uid != me:
                    continue
                if run["run_name"] in Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace"):
                    survivors.append(pid)
            except OSError:
                continue
        for pid in survivors:
            os.kill(pid, sig)
        if not survivors:
            break
        time.sleep(15)
    with edit_state() as fresh:
        r = find_run(fresh, args.run_id)
        if r["status"] in ("running", "starting"):
            r["status"] = "killed"
        r["kill_reason"] = args.reason
    log(f"KILLED session {session} ({args.reason})")
    return 0


def cmd_report(args) -> int:
    state = load_state()
    d = night_dir()
    rows = []
    for run in state["runs"]:
        a_file = d / "runs" / run["id"] / "analysis.json"
        k = {}
        if a_file.exists():
            a = json.loads(a_file.read_text())
            p = a.get("primary") or {}
            k = {
                "final": p.get("final_mean"), "auc": p.get("auc"),
                "c50": (p.get("cross") or {}).get("0.5"), "c80": (p.get("cross") or {}).get("0.8"),
                "c95_transitions": (p.get("cross_transitions") or {}).get("0.95"),
                "c95_minutes": (p.get("cross_minutes") or {}).get("0.95"),
                "lock": (a.get("grasp", {}).get("attempt_right_lock_rate") or {}).get("final_mean"),
            }  # fmt: skip
        it, total = last_iteration(tail(Path(run["log"]), 50_000)) if run.get("log") else (None, None)
        budget = transitions(run.get("iterations", total), run)
        rows.append(
            f"| {run['id']} | {run['slug']} | {run['status']} | {it}/{total} | {budget or '-'} | "
            f"{k.get('c95_transitions', '-')} | {k.get('c95_minutes', '-')} | {k.get('final', '-')} | "
            f"{k.get('auc', '-')} | {k.get('c50', '-')} | {k.get('c80', '-')} | {k.get('lock', '-')} | "
            f"{run.get('decision', '-')} | `{run['run_name']}` |"
        )
    start, branch = state["start_commit"], state["branch"]
    diffstat = git("diff", "--stat", start, branch)
    commits = git("log", "--oneline", f"{start}..{branch}")
    running = [r for r in state["runs"] if r["status"] == "running"]
    journal = (d / "journal.md").read_text()
    kind = "Final" if args.final else "Interim"
    lines = [
        f"# Autotune report {state['night']} ({kind}, written {now():%Y-%m-%d %H:%M})", "",
        f"**Goal:** {state['goal']}", "",
        f"Branch `{branch}` from `{state['start_branch']}` at `{start[:9]}`. "
        f"Champion: {state['champion']}.", "",
        "Score = task success from default (approach) starts, `task_success_approach_smoothed`: final = mean of the "
        "last 50 iterations, auc = mean over the run, c50/c80 = iteration of reaching 0.5/0.8, c95 = first "
        "0.95 crossing in environment transitions and elapsed training minutes; lock = right-lock rate per grasp "
        "attempt.", "",
        "| run | change | status | iteration | transition budget | c95 transitions | c95 minutes | final | auc | "
        "c50 | c80 | lock | decision | wandb / run name |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        *rows, "",
        "## Still running", "",
        *([f"- {r['id']} `{r['session']}` on GPU {r['gpu']}: `tmux attach -t {r['session']}`; stop it with "
           f"`uv run python {SKILL}/scripts/autotune.py kill --run-id {r['id']} --reason user`" for r in running]
          or ["- none"]), "",
        "## Code changes on the branch", "", "```", commits or "(none)", "", diffstat or "(no diff)", "```", "",
        "## Merge in the morning", "", "```bash",
        f"git switch {state['start_branch']}",
        f"git diff {state['start_branch']}...{branch}   # review",
        f"git merge {branch}",
        f"git tag -l 'autotune-{state['night']}-*' | xargs -r git tag -d   # drop variant tags when done",
        "```", "",
        "## Journal", "", journal,
    ]  # fmt: skip
    if state.get("snapshot_commit"):
        lines.insert(
            6,
            f"The first branch commit `{state['snapshot_commit'][:9]}` is a snapshot of work that was "
            "uncommitted at the start.\n",
        )
    (d / "report.md").write_text("\n".join(lines) + "\n")
    with edit_state() as fresh:
        if args.final:
            fresh["status"] = "stopped"
        else:
            fresh["interim_report_written"] = True
    log(f"REPORT {d / 'report.md'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init")
    p.add_argument("--goal", required=True)
    p.add_argument("--stop", required=True, help="Local stop time HH:MM (next occurrence).")
    p.add_argument("--iterations", type=int, default=400)
    p.add_argument("--base-override", action="append", default=[], help="Hydra override applied to every run.")
    sub.add_parser("status")
    p = sub.add_parser("launch")
    p.add_argument("--run-id", required=True, help="g<generation><slot>, e.g. g0a, g1b.")
    p.add_argument("--ref", required=True, help="Commit or tag to train.")
    p.add_argument("--slug", required=True, help="Short name of the change, [a-z0-9_].")
    p.add_argument("--hypothesis", required=True)
    p.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Iterations of this run; default: the night's. Use it to keep the"
        " transitions equal when num_steps_per_env changes.",
    )
    p = sub.add_parser("watch")
    p.add_argument("--interval", type=int, default=60)
    p = sub.add_parser("update")
    p.add_argument("--run-id", required=True)
    p.add_argument("pairs", nargs="+")
    p = sub.add_parser("promote")
    p.add_argument("--run-id", required=True)
    p = sub.add_parser("kill")
    p.add_argument("--run-id", required=True)
    p.add_argument("--reason", required=True)
    p = sub.add_parser("report")
    p.add_argument("--final", action="store_true")
    args = parser.parse_args()
    if args.cmd == "launch" and not re.fullmatch(r"[a-z0-9_]+", args.slug):
        sys.exit("slug must match [a-z0-9_]+")
    return globals()[f"cmd_{args.cmd}"](args)


if __name__ == "__main__":
    sys.exit(main())

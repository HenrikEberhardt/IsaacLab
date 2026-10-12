---
name: gpu-preflight
description: Use before running any Isaac Lab simulation, test, smoke run (zero_agent/random_agent), play or training on this shared multi-user GPU workstation. Checks GPU, CPU, RAM and disk load, picks a free GPU, refuses to run when none is free, and launches training runs inside a tmux session the user can attach to.
---

# GPU Preflight

The workstation is a shared 4x RTX 3090 machine. Other users run long jobs, so GPU usage changes all the time. Run this check fresh before every GPU command; never reuse an earlier result.

## 1. Check and pick a GPU

```bash
bash /hdd1/home/heberhardt/IsaacLab/.claude/skills/gpu-preflight/scripts/pick_gpu.sh            # one GPU
bash /hdd1/home/heberhardt/IsaacLab/.claude/skills/gpu-preflight/scripts/pick_gpu.sh --need 2   # multi-GPU runs
```

The script prints a GPU table (memory, utilization, process owners), CPU load, RAM, disk, and ends with:

- `FREE_GPU=<idx>[,<idx>]` and exit 0, or
- `NO_FREE_GPU ...` and exit 1.

Thresholds are overridable via env vars `GPU_MAX_MEM_MIB` (default 1000), `GPU_MAX_UTIL` (default 10), `PREFERRED_GPU` (default 1).

## 2. Decide

- **Exit 1: stop.** Do not run the command, not even on a "less busy" GPU. Tell the user which GPUs are busy, by whom (user:pid:MiB from the table), and that nothing was started.
- **Load warning:** if the 1-min load average exceeds the core count, or available RAM is below 16 GiB, tell the user and ask before launching.
- **Exit 0:** use the printed index.

## 3. Run

Shell state does not persist between Bash calls, so put `CUDA_VISIBLE_DEVICES=<idx>` in front of every command instead of a lone `export`.

**Short commands** (smoke runs, tests, quick scripts) run directly:

```bash
CUDA_VISIBLE_DEVICES=<idx> uv run isaaclab zero_agent --task <Task> --headless
```

**Training runs always go into a detached tmux session**, started from the IsaacLab repo root so logs land in `IsaacLab/logs`:

```bash
SESSION=train-<short-task-name>-$(date +%H%M)
tmux new-session -d -s "$SESSION" -c /hdd1/home/heberhardt/IsaacLab \
  "CUDA_VISIBLE_DEVICES=<idx> uv run isaaclab train --task <Task> --headless <args> 2>&1 | tee /tmp/$SESSION.log; exec bash"
```

`exec bash` keeps the session open after training ends so the user can read the final output. Long `play` or evaluation runs follow the same pattern.

After launching, tell the user:

- the session name and GPU index,
- `tmux attach -t <session>` to watch it (detach with `Ctrl-b d`),
- `tmux ls` to list sessions.

## 4. Monitor

- Read progress with `tmux capture-pane -pt <session> -S -50` or `tail -n 50 /tmp/<session>.log`.
- Confirm the run is on the right GPU with `nvidia-smi` shortly after launch.
- Never `tmux kill-session` or kill a process without asking the user, and never touch processes owned by other users.

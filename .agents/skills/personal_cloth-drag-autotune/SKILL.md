---
name: cloth-drag-autotune
description: Runs an unattended overnight tuning loop for the Franka cloth-drag RL task on the shared GPU workstation. Trains variants on every free GPU, analyses stage and task success, stage ratios, terminations, grasp quality and PPO health, changes reset strategies, rewards, terminations and PPO algorithm settings, and leaves a report plus an autotune branch to merge. Use when the user starts an autotune night with a goal and a stop time, asks for its status, or says to stop it.
---

# Cloth-drag autotune (orchestrator)

You are the orchestrator of a night of training runs for `IsaacContrib-Drag-Cloth-Franka`. You launch,
watch, evaluate and decide; subagents analyse (evaluator) and change code (tuner). The user starts this in
the evening, leaves, and reads `report.md` in the morning.

```bash
S=/hdd1/home/heberhardt/IsaacLab/.agents/skills/personal_cloth-drag-autotune/scripts
A="uv run python $S/autotune.py"        # state, launch, watch, kill, promote, report
```

Run every command from `/hdd1/home/heberhardt/IsaacLab`. `$S` and `$A` are shorthand in this file: write
the expanded command, `uv run python .agents/skills/personal_cloth-drag-autotune/scripts/autotune.py ...`,
so it matches the user's permission allow-rules and no prompt blocks the night.

## Hard rules

1. **Shared workstation: other people's jobs must never be harmed.** This outranks every goal of the
   night. Only `autotune.py launch` starts GPU work; it runs a fresh GPU preflight (strict thresholds,
   never changed) and refuses busy GPUs. One run per GPU. If no GPU is free, wait; never squeeze in.
2. **Kill only what you started**, and only with `$A kill --run-id <id> --reason ...`. It closes the
   registered tmux session of that run and nothing else. Never use `kill`, `pkill`, `killall`,
   `tmux kill-server`, `nvidia-smi` resets, or any pattern match. Never touch another user's process.
3. **Yield** when the watcher reports `contention`: kill that run of ours, log it, relaunch later.
4. **Git:** commits only on `autotune/<night>` and detached variant commits tagged
   `autotune-<night>-<id>`. Never commit on another branch, never merge into it, never push.
5. **Fixed run protocol:** seed 42 for every run (never vary seeds), full iteration count for every run
   (no early stop of a healthy run), `num_envs` and everything else frozen stays as is. Only crashed,
   stalled, non-finite or contended runs are ended.
6. **Scope:** observations and actions never change; `check_guardrails.py` enforces the allowlist in
   `agents/tuner.md`. A variant that fails it is dropped, not forced through.
7. The stop time does not stop the loop. Runs and generations continue until the user stops the skill.

## Roles

| Role | How | Model | When |
|---|---|---|---|
| Watcher | `$A watch` as a background Bash command | none (script) | always while runs are active; logs one line per run per minute to `status.log`, exits with one `EVENT` |
| Triage | Agent, prompt = `agents/triage.md` + log path | `haiku` | only after `crashed` / `stalled` / `nonfinite` |
| Evaluator | Agent, prompt = `agents/evaluator.md` + inputs | `opus` | after each finished run |
| Tuner | Agent, prompt = `agents/tuner.md` + inputs | session model (omit `model`) | whenever variants are needed |

Pass role prompts by reading the file and putting its full text plus the concrete inputs into the Agent
prompt. Run the evaluator and the tuner in the foreground; one evaluator per finished run.

## Start (user present)

1. Parse goal, stop time and options from the invocation, for example
   `/cloth-drag-autotune improve adaptive ratios so grasp success rises earlier until 07:30`.
   Ask only if the goal or the reset preset is unclear; the user is about to leave.
2. If `logs/autotune/current` names a night with `"status": "running"`, resume it: read `state.json`,
   `journal.md`, latest verdicts, then continue at the loop.
3. `$A init --goal "<goal>" --stop HH:MM [--base-override env.events=adaptive_resets ...]`.
   Base overrides are night-wide (the reset preset the user wants); variants never add overrides.
   `init` creates `autotune/<night>` from the current branch and commits any uncommitted work there as a
   snapshot, so the user's branch is untouched.
4. In `journal.md`, under the goal, write the goal metric you will judge by (default: auc and 0.8 crossing
   of `task_success_approach_smoothed`; a goal about stage balance or grasping names its own metric).
5. Launch the baseline: `$A launch --run-id g0a --ref HEAD --slug baseline --hypothesis "unchanged baseline"`
   as a background Bash command; wait for its notification. Exactly one baseline per night.
6. If more GPUs are free, run the tuner for generation 0 (variants `g0b`, `g0c`, ...), then launch each.
7. Tell the user: night id, branch, sessions and GPUs, `tmux attach -t <session>`, where `status.log` and
   `report.md` live, and how to stop (say "stop autotune").

## Launch

`$A launch --run-id <id> --ref <tag|commit> --slug <slug> --hypothesis "<text>"` (run in background; it
takes 5-15 min). It checks machine load, picks a free GPU, checks guardrails, runs a 5-iteration smoke
run, re-checks the GPU, starts the tmux session, keeps the variant checked out until iteration 1, and
switches back to the night branch. Exit codes: 3 no free GPU (keep the variant queued), 4 machine busy,
5 smoke failed (triage; fix via tuner or drop), 6 guardrail violation (drop), 7 run did not start.
Launch variants one at a time; never edit the working tree while a launch is running.

Run ids: `g<generation><slot>`. A generation is one tuner call; its variants run in parallel when GPUs
allow, otherwise queue them in `journal.md` and launch them as GPUs free up.

## Loop

Keep one watcher running in the background whenever no launch is in progress, and handle its `EVENT`:

- `finished <id>`:
  1. `uv run python $S/analyze_run.py <run_dir> --out <night>/runs/<id>/analysis.json --until <iterations>`
  2. Evaluator agent with: run id, analysis path, run dir, champion and baseline run dirs, hypothesis,
     `git diff <parent_champion or start_commit>..<commit>`, goal, `verdict.md` path.
  3. Decide: `promote` → `$A promote --run-id <id>` (g0a is promoted as the first champion);
     otherwise `$A update --run-id <id> decision=rejected` or `decision=inconclusive`.
     Use the evaluator's recommendation unless the numbers contradict it; write why in the journal.
  4. Close the finished session with `$A kill --run-id <id> --reason finished` (the log stays in
     `<night>/runs/<id>/train.log`).
  5. Append a journal entry: id, changes, hypothesis, key numbers vs champion, decision, one line why.
     Durable findings that would help a future night go to `logs/autotune/lessons.md`.
  6. Launch the next queued variant on the freed GPU, or call the tuner for the next generation first.
- `crashed <id>`, `stalled <id>`, `nonfinite <id>`: triage agent → `$A kill` if the session still runs →
  journal → `CODE_BUG`: tuner fixes it as a new variant; `GPU_MEMORY` or `INFRA`: relaunch once under a
  new id; `NONFINITE`: reject.
- `contention <id> ...`: `$A kill --run-id <id> --reason contention`, journal, relaunch later.
- `gpu_free <idx>`: launch a queued variant, or call the tuner if the queue is empty.
- `stop_time`: `$A report`, then add a short summary at the top of `report.md` (see Report), and continue.
  If a push-notification tool is available, send one line that the interim report is ready.

After compaction or any confusion, re-read this file, `state.json` (`$A status`) and `journal.md`.

## Report

`$A report` writes `<night>/report.md`: run table (scores, decisions, wandb names), running sessions with
attach and kill commands, commits and diff stat of the branch, merge commands, and the journal. Then edit
the top of the report with a short summary for the user:

- what was tried, which bundles were kept and why (numbers), which were rejected and why;
- what remains open (bundles not yet split, inconclusive results, ideas for the next night);
- new reset banks and their generation commands.

## Stop (user says "stop autotune" or similar)

1. Stop the background watcher (TaskStop). Start no new launch.
2. Do not kill running runs. List them with progress (`$A status`) and ask whether to keep them.
3. `$A report --final`, add the summary, and give the user the path and the merge commands:
   `git switch <start branch> && git merge autotune/<night>`.
4. Leave the working tree on `autotune/<night>`, clean. The user merges.

## Unattended operation

Before leaving, the user must keep this session open and allow without prompts: `uv run python
.../autotune.py *`, `uv run python .../analyze_run.py *`, `uv run python .../check_guardrails.py *`,
`uv run python -m pre_commit *`, `git switch/add/commit/tag/diff/log` and edits in the task folder.
If a permission prompt blocks the loop, nothing else breaks: the running trainings continue.

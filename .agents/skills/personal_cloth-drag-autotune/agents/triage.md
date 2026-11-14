# Triage: classify a failed autotune run

You read the end of one failed training log and classify the failure. You are read-only: run no command
other than reading files, edit nothing, start and stop nothing.

The orchestrator gives you: the log path, the smoke or train exit code if any, and the event
(`crashed`, `stalled`, `nonfinite`, `contention`).

Read the last ~200 lines of the log (more only if the traceback starts earlier).

Classes:

- `CODE_BUG`: Python traceback raised in task code (`franka_cloth_drag/...`) or a config error.
- `NONFINITE`: NaN/inf detected (rsl_rl `check_for_nan`, `nonfinite_state`, NaN losses).
- `GPU_MEMORY`: CUDA out of memory, cuBLAS/cuDNN allocation failures, illegal memory access.
- `INFRA`: wandb/network errors, disk full, the process was killed by a signal, tmux session gone without a
  traceback.
- `UNKNOWN`: none of the above fits.

Reply with exactly five lines:

```
CLASS: <class>
EVIDENCE: <the single most telling log line, verbatim, max 200 chars>
LOCATION: <file:line of the innermost task-code frame, or ->
ACTION: fix-code | relaunch-once | drop-variant
OTHERS: <yes if the log suggests another user's process on the GPU was involved, else no>
```

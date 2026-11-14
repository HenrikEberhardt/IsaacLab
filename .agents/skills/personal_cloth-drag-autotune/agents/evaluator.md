# Evaluator: judge one finished autotune run

You analyse one finished training run of the Franka cloth-drag task and write a verdict. You are read-only:
do not edit code, do not commit, do not start simulations or anything on a GPU. Use `uv run python` for
Python. Take your time; the night has hours, and a careful reading of the curves is the point.

The orchestrator gives you: run id, its `analysis.json`, the run dir, the champion's and baseline's run dirs,
the run's hypothesis, the commit range of its change, the night's goal, and the path of `verdict.md` to write.

## Tools

```bash
S=/hdd1/home/heberhardt/IsaacLab/.agents/skills/personal_cloth-drag-autotune/scripts
uv run python $S/analyze_run.py <run_dir> --md --compare <champion_dir> <baseline_dir> --until 400
uv run python $S/analyze_run.py <run_dir> --series 'share_|stage_success_.*_smoothed' --every 10
git -C /hdd1/home/heberhardt/IsaacLab diff <parent>..<commit>     # what the variant changed
```

`--series` takes any regex over the 138 logged tags (`Metrics/cloth_target/*`, `Reset/*`, `Grasp/*`,
`Episode_Termination/*`, `Episode_Reward/*`, `Loss/*`, `Policy/mean_std`, `Action/policy_std_*`).
Compare curves against the champion iteration by iteration, not just the summary numbers.

## What the metrics mean

- Start stages: `approach` (the default task start), `grasp`, `drag`, `goal`. Runs with bank resets start
  episodes in all four; default-reset runs only in `approach`.
- `stage_success_<s>`: reached the next stage from start stage `s`. `task_success_<s>`: finished the whole
  task from start `s`. `_smoothed` = windowed mean. Primary score: `task_success_approach_smoothed`.
- `Reset/share_<s>_raw`: share of episodes started in stage `s` (adaptive resets move these from success).
- `Grasp/*`: per grasp attempt. `attempt_right_lock_rate` (locked the intended node), `wrong_lock_*`,
  `outcome_closed_no_lock_*` (closed but nothing locked), `attempts_per_episode` (re-grasp loops),
  `close_ee_node_distance_mm` and `close_node_between_pads` (alignment at closing).
- All runs use seed 42. One seed cannot resolve small effects: treat auc differences below 0.03 and
  crossing differences below about 20 iterations as noise unless the whole curve shape agrees.

## What to examine (one section each in the verdict)

1. **Score.** final mean, auc, crossing iterations of 0.5/0.8/0.9 for the primary metric vs champion and
   baseline. At 400 iterations good runs saturate near 0.99, so learning speed (auc, crossings) decides.
2. **Stage success balance.** Ideally all four stage successes rise at the same speed. Report the spread of
   their crossing iterations and which stage lags, and whether the change narrowed or widened it.
3. **Task success order.** It should rise goal, then drag, then grasp, then approach, each after the other.
   Report the order and the gaps; a stage that rises late or stalls is the bottleneck.
4. **Stage ratios.** For adaptive resets: when shares start moving, how fast, oscillations (amplitude,
   period; a limit cycle of ~14 iterations happened before with short success windows), whether a share
   change is followed by the matching stage success rising (`share_to_own_stage_success` lag/corr, and look
   at the curves). Judge whether ratios should shift earlier, later, faster or slower, and what minimum
   shares would help. Frontier tags (`Reset/frontier_*`) when frontier sampling is on.
5. **Terminations.** Anything new or larger than in the champion (`gripper_cube_contact`, `cube_fell`,
   `cloth_out_of_bounds`, `ee_below_table`, `nonfinite_state`, `joint_vel_out_of_limit`), when it peaks.
6. **Grasp quality.** Did reward changes improve alignment and reduce wrong locks, closed-no-lock and
   re-grasp attempts? Give numbers at iterations 100/200/300/400 vs champion.
7. **PPO health.** `Loss/value` (spikes, level), `Loss/surrogate` (sustained positive values were seen
   before with a learning-rate collapse), `Loss/learning_rate` (the adaptive schedule collapsing to its
   floor means KL is too high), `Loss/entropy`, `Policy/mean_std` and per-action std (late rise = std
   blow-up history; collapse = lost exploration).
8. **Verdict.** Did the hypothesis hold? Which part of a 2-3 change bundle likely caused which effect
   (state the confidence)? What should be split out and tested separately?
9. **Suggestions.** Two to four concrete next changes for the tuner, each with the evidence behind it.

## Decision rule

Recommend `promote` only if, at 400 iterations against the champion:

- primary auc is higher by at least 0.03, or the 0.8 crossing is at least 25 iterations earlier with auc
  not lower, or the night's goal metric improves clearly by the same standard; and
- primary final mean is not lower by more than 0.01; and
- no termination grows above 2 % final mean, right-lock rate does not drop by more than 0.03, and PPO
  shows no new pathology.

Otherwise `reject`, or `inconclusive` when the effect is inside the noise but the direction is promising.

## Output

Write `verdict.md` with the sections above. End it with exactly this block:

```
VERDICT: promote | reject | inconclusive
PRIMARY_AUC: <run> vs <champion>
PRIMARY_CROSS_0.8: <run> vs <champion>
ONE_LINE: <one sentence for the journal>
```

Reply to the orchestrator with only that block and the verdict path.

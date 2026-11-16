# Tuner: propose and implement the next autotune variants

You change the cloth-drag training setup to reach the night's goal. Each variant you make is trained for the
full run length on its own GPU, about 1.5 h. Your output is the input to that run, so be bold but correct.

The orchestrator gives you: the goal, the night dir, the generation number, the run ids to fill (for
example `g2a`, `g2b`), the champion commit, the start commit, and the base overrides of the night.

## Read first

1. `logs/autotune/lessons.md` (findings of earlier nights) and `<night>/journal.md` (this night so far).
2. The latest `<night>/runs/*/verdict.md`, especially their suggestions.
3. Background as needed:
   `source/isaaclab_tasks/isaaclab_tasks/contrib/franka_cloth_drag/implementation_log/reset_states.md`,
   `rewards.md`, `curriculum.md`, and the memory index
   `/hdd1/home/heberhardt/.claude/projects/-hdd1-home-heberhardt-IsaacLab/memory/MEMORY.md`
   (files named `project_cloth_drag_*` and `project_reset_state_plan.md` hold past experiment results).
4. The code you will change.

## What you may change

Task dir `T=source/isaaclab_tasks/isaaclab_tasks/contrib/franka_cloth_drag`.

- `$T/reset_strategies/**`, `$T/reset_bank_generation/**`: reset strategy logic, sampling, ratio
  adaptation, frontier sampling, bank generation.
- In `$T/config/franka/franka_cloth_drag_env_cfg.py`: `UniformResetsCfg`, `AdaptiveResetsCfg`,
  `ResetStrategiesCfg` (which preset is default), `FRONTIER_SAMPLER_CFG`, `RewardsCfg`, `TerminationsCfg`.
- `$T/mdp/rewards.py`, `$T/mdp/terminations.py`, `$T/mdp/events.py`, and the matching export lines in
  `$T/mdp/__init__.pyi`.
- In `$T/config/franka/agents/rsl_rl_ppo_cfg.py`: only the `algorithm=RslRlPpoAlgorithmCfg(...)` block
  (learning rate, schedule, desired_kl, epochs, mini-batches, gamma, lam, clip, entropy, value loss,
  grad norm) and `num_steps_per_env`.
- Tests `source/isaaclab_tasks/test/contrib/test_franka_cloth_drag*.py` when a changed contract needs it.

## What stays frozen (checked by `check_guardrails.py`; a violation drops the variant)

Observations, actions, the curriculum, `DefaultResetsCfg` (it defines the task start distribution the score
is measured on), the `success` and `time_out` terminations and `cloth_drag_success`, the command and its
metrics (`mdp/commands.py`, `stages.py`, `grasp_diagnostics.py`), actor/critic networks, std settings,
`num_envs`, `seed`, cloth material, physics and assets. Never vary the seed: every run uses 42. The orchestrator may
vary a run's iteration count only to keep environment transitions equal when `num_steps_per_env` changes. A new
reset bank must keep the approach bucket drawn from the default-reset distribution; never narrow what the task
starts look like.

## How to design variants

- Explore: each variant is a bundle of 2-3 changes that together look promising. Make variants of one
  generation different from each other (for example one reset-strategy bundle, one reward bundle, one PPO
  bundle), so a generation tests several directions at once.
- When a bundle was promoted earlier, a later variant may split it to attribute the effect; the journal
  says which bundles still need that.
- Do not repeat a change the journal already rejected unless the new evidence says why it would now work.
- Base changes on the verdicts' evidence (stage success balance, task success order, ratio oscillations,
  terminations, grasp quality, PPO health) and on the goal. Write down the expected effect in numbers.

## How to implement a variant

All git work is local. Never push, never merge, never touch any branch other than detached commits and
`autotune/<night>`. No AI attribution lines in commit messages (repo rule).

```bash
cd /hdd1/home/heberhardt/IsaacLab
S=.agents/skills/personal_cloth-drag-autotune/scripts
git switch --detach <champion_commit>
# ... edit files ...
uv run python -m pre_commit run --files <changed files>    # format; re-run until it passes
uv run python -m py_compile <changed .py files>
uv run python $S/check_guardrails.py source --base <start_commit>
git add <changed files> && git commit -m "Autotune <run_id>: <imperative summary>"
git tag autotune-<night>-<run_id>
git switch autotune/<night>                                  # always end on the night branch, clean tree
```

Parameter-only changes are still commits (edit the cfg), so a promoted variant brings its change to the
branch the user merges. Do not use per-run Hydra overrides.

A new reset bank needs a GPU for ~5-10 min. Run the GPU preflight first and use only a free GPU:
`bash .agents/skills/personal_gpu-preflight/scripts/pick_gpu.sh`, then
`CUDA_VISIBLE_DEVICES=<idx> uv run python $T/reset_bank_generation/scripts/generate_reset_bank.py ...
--output logs/reset_banks/autotune_<night>_<run_id>.pt`. Never overwrite an existing bank. If no GPU is
free, choose a variant without a new bank. Commit the cfg that points to the new bank path and record
the exact generation command in the hypothesis.

## Output

Reply with one JSON object per variant, nothing else besides one short paragraph of reasoning:

```json
{"run_id": "g2a", "ref": "autotune-<night>-g2a", "slug": "ratio_hold_kl",
 "hypothesis": "what should change in which metric by how much, and why",
 "changes": ["AdaptiveResetsCfg.ratio_hold_steps 300 -> 600", "desired_kl 0.01 -> 0.008"], "iterations": 300,
 "splits_bundle": null, "new_bank": null}
```

`slug` matches `[a-z0-9_]+`, at most 24 characters.

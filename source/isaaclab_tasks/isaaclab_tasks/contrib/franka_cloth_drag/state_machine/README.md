# Cloth-drag state machine

This folder contains the scripted state machine controller used to create the reset state bank.

## Structure

```text
scripts/run_state_machine.py
        |
        +--> cloth_drag_sm_cfg.py       define profiles and parameters of SM
        +--> cloth_drag_sm.py           state transitions and target commands
        +--> cloth_drag_sm_runner.py    environment state -> controller -> actions
```

The controller follows this sequence:

```text
PREGRASP -> HOVER -> DESCEND -> GRASP -> DRAG -> HOLD
                                  |        |
                                  +-> FAILED <-+
```

There is one grasp attempt per rollout. A failed grasp or stalled drag enters `FAILED` and waits for the environment
to reset. These controller phases are motion steps; they are separate from the task's four progress stages.
Every arm command is uniformly scaled to keep its largest normalized component at or below `0.85`; the task's action
scales and IK settings remain unchanged.

## Profiles

The `--profile` flag selects one of two profiles:

| Profile | Purpose | Behavior |
| --- | --- | --- |
| `nominal` | Reproducible debugging and demonstrations | Uses fixed controller parameters |
| `randomized` | Reset-bank generation and robustness checks | Samples drag speed and height, grasp yaw, and pre-grasp position and yaw independently per environment. |

## Run

Run commands from the Isaac Lab repository root:

```bash
# Deterministic controller; run until Ctrl+C.
uv run python source/isaaclab_tasks/isaaclab_tasks/contrib/franka_cloth_drag/state_machine/scripts/run_state_machine.py \
    --profile nominal --num_envs 1 --seed 42

# Randomized controller; stop after 1,200 environment steps.
uv run python source/isaaclab_tasks/isaaclab_tasks/contrib/franka_cloth_drag/state_machine/scripts/run_state_machine.py \
    --profile randomized --num_envs 16 --num_steps 1200 --seed 42
```

Open `http://localhost:8760` while the process is running.
The diagnostics print EE-frame down-tilt statistics every 30 controller steps; `0 deg` means that the gripper points
straight down, independently of its yaw.
Reset-bank generation captures only states within `0.5 deg` of down, and validation checks the limit again after
restoration and throughout the hold.

## Command-line flags

State-machine flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `-h`, `--help` | — | Print all available options. |
| `--profile {nominal,randomized}` | `nominal` | Select the controller profile. |
| `--num_envs N` | `1` | Number of parallel environments. |
| `--num_steps N` | unlimited | Stop after `N` environment steps. |
| `--seed N` | `0` | Environment and profile-sampling seed. |

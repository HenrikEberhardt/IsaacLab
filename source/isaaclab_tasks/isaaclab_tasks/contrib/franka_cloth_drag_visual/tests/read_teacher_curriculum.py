# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Print a teacher run's AvoidTableCollisionCurriculum state at a checkpoint iteration (no sim): sets the teacher pins."""

import argparse
import re
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PREFIX = "Curriculum/avoid_table_collision/"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", required=True, help="Teacher run directory (logs/rsl_rl/cloth_drag_task/<run>).")
parser.add_argument("--iteration", type=int, default=None, help="Checkpoint iteration (default: newest model_N.pt).")
args = parser.parse_args()

run = Path(args.run)
iteration = args.iteration
if iteration is None:
    iteration = max(int(re.match(r"model_(\d+)\.pt", p.name).group(1)) for p in run.glob("model_*.pt"))
events = EventAccumulator(str(run), size_guidance={"scalars": 0})
events.Reload()
print(f"run {run.name}, iteration {iteration}")
for key in (
    "progress",
    "gate_open",
    "success_ema",
    "scale_z",
    "scale_rot",
    "joint_speed_fraction",
    "termination_force",
):
    values = [e for e in events.Scalars(PREFIX + key) if e.step <= iteration]
    print(f"  {key:18s} {values[-1].value:.4f}" if values else f"  {key:18s} not logged yet")
print("  -> teacher_cfg.TEACHER_CURRICULUM_PROGRESS = progress; TEACHER_YAW_RAMP = True if scale_rot is logged")

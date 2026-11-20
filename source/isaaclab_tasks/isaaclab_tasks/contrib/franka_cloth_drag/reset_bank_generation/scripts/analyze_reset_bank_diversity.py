# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Save a physical-diversity report for a cloth-drag reset bank."""

import argparse
import json
from pathlib import Path

from isaaclab_tasks.contrib.franka_cloth_drag.reset_bank_generation.diversity import diversity_report
from isaaclab_tasks.contrib.franka_cloth_drag.reset_strategies import ResetBank

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bank", required=True, help="Input reset bank (.pt).")
parser.add_argument("--output", required=True, help="Output report (.json).")
args = parser.parse_args()

bank = ResetBank.load(args.bank)
report = diversity_report(bank)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(f"[INFO]: Saved diversity report for {len(bank)} states to {output}.")

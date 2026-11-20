# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Select diverse frontier stages and combine them with an existing approach bank."""

import argparse
from datetime import datetime

from isaaclab_tasks.contrib.franka_cloth_drag.reset_bank_generation.diversity import (
    ResetBankDiversityCfg,
    select_diverse_rows,
)
from isaaclab_tasks.contrib.franka_cloth_drag.reset_strategies import ResetBank

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--candidates", required=True, help="Validated candidate-pool bank (.pt).")
parser.add_argument("--approach_bank", required=True, help="Bank providing unchanged approach states (.pt).")
parser.add_argument("--output", required=True, help="Output reset bank (.pt).")
parser.add_argument("--grasp", type=int, default=1000)
parser.add_argument("--drag", type=int, default=100)
parser.add_argument("--goal", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

candidates = ResetBank.load(args.candidates)
approach_bank = ResetBank.load(args.approach_bank)
cfg = ResetBankDiversityCfg(seed=args.seed)
pieces = [approach_bank.select(approach_bank.stage_rows("approach"))]
selected = {}
for stage, count in (("grasp", args.grasp), ("drag", args.drag), ("goal", args.goal)):
    rows = select_diverse_rows(candidates, stage, count, cfg)
    selected[stage] = rows.tolist()
    piece = candidates.select(rows)
    # Candidate-only hand position makes grasp diversity measurable but is not needed by reset restore or training.
    piece.data["task"].pop("ee_pos_b", None)
    pieces.append(piece)

meta = dict(approach_bank.meta)
meta["created"] = datetime.now().isoformat(timespec="seconds")
meta["diversity_selection"] = {
    "candidate_bank": args.candidates,
    "approach_bank": args.approach_bank,
    "candidate_counts": candidates.stage_counts(),
    "selected_counts": {"approach": len(pieces[0]), "grasp": args.grasp, "drag": args.drag, "goal": args.goal},
    "config": vars(cfg),
    "selected_rows": selected,
}
bank = ResetBank.concat(pieces, meta=meta)
bank.save(args.output)
print(f"[INFO]: Saved {bank.stage_counts()} diverse states to {args.output}.")

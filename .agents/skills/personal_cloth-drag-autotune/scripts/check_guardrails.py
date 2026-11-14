#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Guardrails of the cloth-drag autotune: what the tuner may change.

Usage:
    uv run python check_guardrails.py source --base <commit>
    uv run python check_guardrails.py params --run <run_dir> --ref <run_dir>

``source`` checks the working tree (tracked changes since ``base`` plus untracked files under ``source/``):
every changed file must be on the allowlist, and in the env cfg and PPO cfg only the allowed top-level
definitions or fields may differ (AST comparison, comments and formatting are ignored).

``params`` compares the frozen sections of the resolved ``params/env.yaml`` and ``params/agent.yaml`` of a
run against a reference run (the night's baseline smoke run).

Exit code 0 when everything passes, 1 on a violation (each one printed as ``VIOLATION ...``).
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

REPO = Path("/hdd1/home/heberhardt/IsaacLab")
TASK = "source/isaaclab_tasks/isaaclab_tasks/contrib/franka_cloth_drag"
ENV_CFG = f"{TASK}/config/franka/franka_cloth_drag_env_cfg.py"
PPO_CFG = f"{TASK}/config/franka/agents/rsl_rl_ppo_cfg.py"
MDP_STUB = f"{TASK}/mdp/__init__.pyi"
TERMINATIONS = f"{TASK}/mdp/terminations.py"

# Whole directories and files the tuner may change freely.
ALLOWED_PREFIXES = (
    f"{TASK}/reset_strategies/",
    f"{TASK}/reset_bank_generation/",
)
ALLOWED_FILES = {
    f"{TASK}/mdp/rewards.py",
    f"{TASK}/mdp/terminations.py",
    f"{TASK}/mdp/events.py",
    ENV_CFG,  # restricted further by ENV_CFG_EDITABLE
    PPO_CFG,  # restricted further by PPO_EDITABLE
    MDP_STUB,  # only the exports of rewards, terminations and events may change
}
ALLOWED_TEST_PATTERN = re.compile(r"^source/isaaclab_tasks/test/contrib/test_franka_cloth_drag[^/]*\.py$")

# Top-level definitions of the env cfg that may change: bank reset strategies, rewards, terminations.
# DefaultResetsCfg stays frozen: it defines the task start distribution the primary score is measured on.
ENV_CFG_EDITABLE = {
    "FRONTIER_SAMPLER_CFG",
    "UniformResetsCfg",
    "AdaptiveResetsCfg",
    "ResetStrategiesCfg",
    "RewardsCfg",
    "TerminationsCfg",
}
# Terms inside TerminationsCfg that stay frozen (task success and episode length define the task).
TERMINATIONS_FROZEN_TERMS = {"success", "time_out"}
# Functions of terminations.py that stay frozen.
TERMINATIONS_FROZEN_FUNCS = {"cloth_drag_success"}
# Modules of mdp/__init__.pyi whose exports may change.
STUB_EDITABLE_MODULES = {"rewards", "terminations", "events"}

# Fields of the PPO runner cfg that may change. Everything else in the file is frozen.
PPO_RUNNER_CLASS = "FrankaClothDragPPORunnerCfg"
PPO_EDITABLE = {"algorithm", "num_steps_per_env"}

# Frozen top-level sections of the resolved run params.
ENV_FROZEN_SECTIONS = (
    "sim",
    "decimation",
    "scene",
    "observations",
    "actions",
    "commands",
    "curriculum",
    "episode_length_s",
    "seed",
)
AGENT_FROZEN_SECTIONS = (
    "seed",
    "obs_groups",
    "clip_actions",
    "action_dim_names",
    "empirical_normalization",
    "actor",
    "critic",
    "experiment_name",
)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout


def _key(node: ast.stmt, index: int) -> str | None:
    """Name of a top-level (or class-body) statement, ``None`` for imports and docstrings."""
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return None
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name
    if isinstance(node, ast.Assign):
        return ",".join(ast.unparse(t) for t in node.targets)
    if isinstance(node, ast.AnnAssign):
        return ast.unparse(node.target)
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
        return None  # docstring or attribute docstring
    return f"<statement {index}: {ast.unparse(node)[:60]}>"


def _definitions(body: list[ast.stmt]) -> dict[str, str]:
    out = {}
    for i, node in enumerate(body):
        key = _key(node, i)
        if key is not None:
            out[key] = ast.dump(node)
    return out


def _compare(base: dict[str, str], new: dict[str, str], editable: set[str], where: str) -> list[str]:
    violations = []
    for key, dump in base.items():
        if key in editable:
            continue
        if key not in new:
            violations.append(f"{where}: frozen definition '{key}' was removed")
        elif new[key] != dump:
            violations.append(f"{where}: frozen definition '{key}' was changed")
    for key in new.keys() - base.keys():
        if key.startswith("<statement"):
            violations.append(f"{where}: new top-level statement {key}")
    return violations


def _class_body(tree: ast.Module, name: str) -> dict[str, str] | None:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return _definitions(node.body)
    return None


def check_env_cfg(base_src: str, new_src: str) -> list[str]:
    base_tree, new_tree = ast.parse(base_src), ast.parse(new_src)
    violations = _compare(_definitions(base_tree.body), _definitions(new_tree.body), ENV_CFG_EDITABLE, ENV_CFG)
    base_terms, new_terms = (
        _class_body(base_tree, "TerminationsCfg") or {},
        _class_body(new_tree, "TerminationsCfg") or {},
    )
    for term in TERMINATIONS_FROZEN_TERMS:
        if base_terms.get(term) != new_terms.get(term):
            violations.append(f"{ENV_CFG}::TerminationsCfg: frozen term '{term}' was changed or removed")
    return violations


def check_terminations(base_src: str, new_src: str) -> list[str]:
    base, new = _definitions(ast.parse(base_src).body), _definitions(ast.parse(new_src).body)
    return [
        f"{TERMINATIONS}: frozen function '{name}' was changed or removed"
        for name in TERMINATIONS_FROZEN_FUNCS
        if base.get(name) != new.get(name)
    ]


def check_mdp_stub(base_src: str, new_src: str) -> list[str]:
    def imports(src: str) -> dict[str, str]:
        out = {}
        for node in ast.parse(src).body:
            if isinstance(node, ast.ImportFrom):
                key = node.module or ""
                out[key] = out.get(key, "") + ast.dump(node)
            elif not (isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "__all__"):
                out[f"<{ast.unparse(node)[:60]}>"] = ast.dump(node)
        return out

    base, new = imports(base_src), imports(new_src)
    return [
        f"{MDP_STUB}: exports of '{key}' may not change"
        for key in base.keys() | new.keys()
        if key not in STUB_EDITABLE_MODULES and base.get(key) != new.get(key)
    ]


def check_ppo_cfg(base_src: str, new_src: str) -> list[str]:
    base_tree, new_tree = ast.parse(base_src), ast.parse(new_src)
    violations = _compare(_definitions(base_tree.body), _definitions(new_tree.body), {PPO_RUNNER_CLASS}, PPO_CFG)

    base_body, new_body = _class_body(base_tree, PPO_RUNNER_CLASS), _class_body(new_tree, PPO_RUNNER_CLASS)
    if new_body is None:
        return violations + [f"{PPO_CFG}: class {PPO_RUNNER_CLASS} was removed"]
    violations += _compare(base_body, new_body, PPO_EDITABLE, f"{PPO_CFG}::{PPO_RUNNER_CLASS}")
    for key in new_body.keys() - base_body.keys() - PPO_EDITABLE:
        violations.append(f"{PPO_CFG}::{PPO_RUNNER_CLASS}: new field '{key}' is not allowed")
    return violations


def file_allowed(path: str) -> bool:
    return path in ALLOWED_FILES or path.startswith(ALLOWED_PREFIXES) or bool(ALLOWED_TEST_PATTERN.match(path))


def check_source(base: str) -> list[str]:
    changed = set(git("diff", "--name-only", base).split())
    changed |= {p for p in git("ls-files", "--others", "--exclude-standard", "source").split()}
    violations = [
        f"{path}: file is not on the autotune allowlist" for path in sorted(changed) if not file_allowed(path)
    ]
    checkers = ((ENV_CFG, check_env_cfg), (PPO_CFG, check_ppo_cfg), (TERMINATIONS, check_terminations),
                (MDP_STUB, check_mdp_stub))  # fmt: skip
    for path, checker in checkers:
        if path in changed:
            violations += checker(git("show", f"{base}:{path}"), (REPO / path).read_text())
    return violations


def yaml_sections(path: Path) -> dict[str, str]:
    """Split a resolved params yaml into its top-level sections (raw text, so no python tags get loaded)."""
    sections: dict[str, list[str]] = {}
    current = None
    for line in path.read_text().splitlines():
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):", line)
        if match:
            current = match.group(1)
            sections[current] = []
        if current is not None:
            sections[current].append(line)
    return {k: "\n".join(v) for k, v in sections.items()}


def check_params(run: Path, ref: Path) -> list[str]:
    violations = []
    for name, frozen in (("env.yaml", ENV_FROZEN_SECTIONS), ("agent.yaml", AGENT_FROZEN_SECTIONS)):
        run_file, ref_file = run / "params" / name, ref / "params" / name
        if not run_file.exists() or not ref_file.exists():
            violations.append(f"params/{name} missing in {run if not run_file.exists() else ref}")
            continue
        new, old = yaml_sections(run_file), yaml_sections(ref_file)
        for section in frozen:
            if new.get(section) != old.get(section):
                violations.append(f"params/{name}: frozen section '{section}' differs from the reference run")
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_src = sub.add_parser("source")
    p_src.add_argument("--base", required=True, help="Commit the night started from.")
    p_par = sub.add_parser("params")
    p_par.add_argument("--run", required=True, type=Path)
    p_par.add_argument("--ref", required=True, type=Path)
    args = parser.parse_args()

    violations = check_source(args.base) if args.cmd == "source" else check_params(args.run, args.ref)
    for v in violations:
        print(f"VIOLATION {v}")
    print("GUARDRAILS_OK" if not violations else f"GUARDRAILS_FAILED ({len(violations)})")
    return 0 if not violations else 1


if __name__ == "__main__":
    sys.exit(main())

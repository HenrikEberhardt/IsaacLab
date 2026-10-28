# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the reset-state bank of the Franka cloth dragging task. No Kit/GPU required."""

import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import EventTermCfg

from isaaclab_tasks.contrib.franka_cloth_drag.mdp import reset_bank
from isaaclab_tasks.contrib.franka_cloth_drag.mdp.stages import ClothDragStage
from isaaclab_tasks.contrib.franka_cloth_drag.reset_states.bank import (
    BUCKET_NAMES,
    SCHEMA_VERSION,
    ResetBank,
    check_mode_probabilities,
    sample_reset_rows,
    stable_hash,
    subsample_to_quotas,
)
from isaaclab_tasks.contrib.franka_cloth_drag.reset_states.capture import _without_resolved_bodies
from isaaclab_tasks.contrib.franka_cloth_drag.reset_states.collect import CollectCfg, ResetCollector
from isaaclab_tasks.contrib.franka_cloth_drag.state_machine import grasp_node_ids

FINGERPRINT = {
    "schema_version": SCHEMA_VERSION,
    "num_nodes": 2,
    "joint_names": ["a", "b"],
    "scene_entities": ["cube", "deformable", "robot"],
    "mesh_default_local": [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
    "physics_hash": "p",
    "task_hash": "t",
}


def _bank(buckets, contexts):
    """A bank with the given bucket and context of every row; the row index is stored as a joint position."""
    num = len(buckets)
    data = {
        "scene": {"joint_pos": torch.arange(num, dtype=torch.float).unsqueeze(1).repeat(1, 2)},
        "task": {"grasped": torch.zeros(num, 2, dtype=torch.bool)},
        "labels": {
            "bucket": torch.tensor(buckets, dtype=torch.long),
            "context_id": torch.tensor(contexts, dtype=torch.long),
        },
    }
    return ResetBank(data, {"schema_version": SCHEMA_VERSION, "fingerprint": dict(FINGERPRINT)})


def test_save_and_load_round_trip(tmp_path):
    bank = _bank([0, 1, 2, 3], [0, 0, 0, 0])
    path = tmp_path / "bank.pt"
    bank.save(path)
    loaded = ResetBank.load(path)
    assert loaded.meta == bank.meta
    torch.testing.assert_close(loaded.data["scene"]["joint_pos"], bank.data["scene"]["joint_pos"])
    assert loaded.bucket_counts() == {"approach": 1, "grasp": 1, "drag": 1, "goal": 1}
    with pytest.raises(ValueError):
        bank.save(tmp_path / "bank.npz")


def test_load_rejects_another_schema_version(tmp_path):
    bank = _bank([1], [0])
    bank.meta["schema_version"] = SCHEMA_VERSION + 1
    bank.save(tmp_path / "old.pt")
    with pytest.raises(ValueError, match="schema version"):
        ResetBank.load(tmp_path / "old.pt")


def test_fingerprint_mismatch():
    bank = _bank([1], [0])
    bank.check_fingerprint(dict(FINGERPRINT))
    with pytest.raises(ValueError, match="joint_names"):
        bank.check_fingerprint(FINGERPRINT | {"joint_names": ["b", "a"]})
    with pytest.raises(ValueError, match="mesh"):
        bank.check_fingerprint(FINGERPRINT | {"mesh_default_local": [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]})
    # other settings only load when allowed explicitly
    with pytest.raises(ValueError, match="physics_hash"):
        bank.check_fingerprint(FINGERPRINT | {"physics_hash": "other"})
    bank.check_fingerprint(FINGERPRINT | {"physics_hash": "other"}, allow_settings_mismatch=True)


def test_stable_hash_is_deterministic_and_sensitive():
    cfg = {"dt": 1.0 / 120, "solver": SimpleNamespace, "values": (1, 2)}
    assert stable_hash(cfg) == stable_hash(dict(cfg))
    assert stable_hash(cfg) != stable_hash(cfg | {"dt": 1.0 / 60})


def test_physics_hash_ignores_the_proxy_bodies_resolved_per_env():
    # the coupler replaces the proxy body patterns with the body ids of every env, so 4 and 16 envs differ
    def physics(bodies):
        return {
            "proxies": [{"bodies": bodies, "mass_scale": 5.0}],
            "entries": [{"bodies": [r"/World/envs/env_.*/Cube"]}],
        }

    assert stable_hash(_without_resolved_bodies(physics([8, 9]))) == stable_hash(
        _without_resolved_bodies(physics([8, 9, 20, 21]))
    )
    changed = physics([8, 9])
    changed["proxies"][0]["mass_scale"] = 1.0
    assert stable_hash(_without_resolved_bodies(changed)) != stable_hash(_without_resolved_bodies(physics([8, 9])))


def test_subsample_keeps_quotas_and_the_approach_rows_of_kept_rollouts():
    # rollouts 0-2 have all four levels, rollout 3 only an approach row
    buckets = [0, 1, 2, 3] * 3 + [0]
    contexts = [0] * 4 + [1] * 4 + [2] * 4 + [3]
    bank = _bank(buckets, contexts)
    rows = subsample_to_quotas(bank, {"grasp": 2, "drag": 2, "goal": 2}, torch.Generator().manual_seed(0))
    kept = bank.select(rows)
    counts = kept.bucket_counts()
    assert (counts["grasp"], counts["drag"], counts["goal"]) == (2, 2, 2)
    kept_contexts = set(kept.data["labels"]["context_id"].tolist())
    approach_contexts = set(kept.data["labels"]["context_id"][kept.bucket_rows("approach")].tolist())
    assert approach_contexts == kept_contexts
    assert 3 not in kept_contexts
    with pytest.raises(ValueError, match="fewer"):
        subsample_to_quotas(bank, {"grasp": 4})


def test_reset_modes_sample_their_buckets():
    bank = _bank([0, 1, 1, 2, 3, 3], [0] * 6)
    shares = check_mode_probabilities({"default": 1.0}, bank)
    assert (sample_reset_rows(bank, shares, 50) == -1).all()
    shares = check_mode_probabilities({"grasp": 1.0, "goal": 1.0}, bank)
    rows = sample_reset_rows(bank, shares, 200)
    assert set(rows.tolist()) == {1, 2, 4, 5}
    with pytest.raises(ValueError, match="Unknown"):
        check_mode_probabilities({"approach": 1.0}, bank)
    with pytest.raises(ValueError, match="no stored states"):
        check_mode_probabilities({"drag": 1.0}, _bank([1], [0]))
    with pytest.raises(ValueError, match="no stored states"):
        check_mode_probabilities({"default": 0.5, "grasp": 0.5}, None)


def _reset_term(monkeypatch, bank, success):
    """A ResetFromBank term on a fake environment; the restores are no-ops. The finished episodes' outcome is set in
    ``success``, the command's ``reached_*`` metrics and ``time_to_stage`` and the environment's episode length."""
    monkeypatch.setattr(reset_bank, "restore_physical", lambda *args, **kwargs: None)
    monkeypatch.setattr(reset_bank, "restore_task_state", lambda *args, **kwargs: None)
    num = len(success)
    metrics = {"success": success}
    metrics |= {f"reached_{stage.name.lower()}": torch.zeros(num) for stage in list(ClothDragStage)[1:]}
    command = SimpleNamespace(metrics=metrics, time_to_stage=torch.full((num, len(ClothDragStage)), -1.0))
    env = SimpleNamespace(
        num_envs=num,
        device="cpu",
        common_step_counter=0,
        extras={},
        episode_length_buf=torch.zeros(num, dtype=torch.long),
        step_dt=0.25,
        command_manager=SimpleNamespace(get_term=lambda name: command),
    )
    cfg = EventTermCfg(func=reset_bank.ResetFromBank, mode="reset", params={"bank_path": None})
    term = reset_bank.ResetFromBank(cfg, env)
    if bank is not None:
        term.set_bank(bank)
    return term, env


def _reset(term, env, env_ids, step):
    """One reset of ``env_ids`` as the environment does it: reset events, new log dict, reset hooks."""
    env.common_step_counter = step
    term(env, env_ids)
    env.extras["log"] = {}
    term.reset(env_ids)
    return {key: value for key, value in env.extras["log"].items() if key.startswith("Reset/")}


def test_reset_logs_success_per_source(monkeypatch):
    success = torch.zeros(4)
    term, env = _reset_term(monkeypatch, _bank([0, 1, 2, 3], [0] * 4), success)
    all_envs = torch.arange(4)
    term.force_rows(all_envs, torch.tensor([-1, 1, 2, 3]))  # default, grasp, drag, goal
    assert _reset(term, env, all_envs, step=0) == {}  # the initial reset ends no episode

    # envs 0-2 end (default success, grasp failure, drag success), env 3 (goal) runs on; all restart from defaults
    success[:] = torch.tensor([1.0, 0.0, 1.0, 1.0])
    command = env.command_manager.get_term("cloth_target")
    # env 0 reached GRASP at 2 s and DRAG at 3 s, env 1 never reached DRAG, env 2 reached GOAL at 1.5 s
    command.metrics["reached_grasp"][:] = torch.tensor([1.0, 1.0, 1.0, 1.0])
    command.metrics["reached_drag"][:] = torch.tensor([1.0, 0.0, 1.0, 1.0])
    command.metrics["reached_goal"][:] = torch.tensor([1.0, 0.0, 1.0, 1.0])
    command.time_to_stage[0, 1:] = torch.tensor([2.0, 3.0, 4.0])
    command.time_to_stage[1, 1] = 0.25
    command.time_to_stage[2, 1:] = torch.tensor([0.25, 0.25, 1.5])
    log = _reset(term, env, torch.tensor([0, 1, 2]), step=10)
    assert log == {
        "Reset/success_default": 1.0,
        "Reset/success_grasp": 0.0,
        "Reset/success_drag": 1.0,
        "Reset/next_stage_default": 1.0,
        "Reset/next_stage_grasp": 0.0,
        "Reset/next_stage_drag": 1.0,
        "Reset/time_to_next_default": 2.0,
        "Reset/time_to_next_drag": 1.5,
        "Reset/share_default": 1.0,
        "Reset/share_grasp": 0.0,
        "Reset/share_drag": 0.0,
        "Reset/share_goal": 0.0,
    }
    assert term.rows.tolist() == [-1, -1, -1, 3]

    # env 3 (goal) ends with the success after 8 steps; the time to success is its episode time
    env.episode_length_buf[3] = 8
    log = _reset(term, env, torch.tensor([3]), step=20)
    assert log == {
        "Reset/success_goal": 1.0,
        "Reset/time_to_next_goal": 2.0,
        "Reset/share_default": 1.0,
        "Reset/share_grasp": 0.0,
        "Reset/share_drag": 0.0,
        "Reset/share_goal": 0.0,
    }


def test_reset_logs_default_success_without_a_bank(monkeypatch):
    success = torch.tensor([1.0, 0.0, 0.0, 1.0])
    term, env = _reset_term(monkeypatch, None, success)
    _reset(term, env, torch.arange(4), step=0)
    command = env.command_manager.get_term("cloth_target")
    command.metrics["reached_grasp"][:] = torch.tensor([1.0, 1.0, 0.0, 1.0])
    command.time_to_stage[:, 1] = torch.tensor([1.0, 2.0, -1.0, 3.0])
    log = _reset(term, env, torch.arange(4), step=5)
    assert log == {
        "Reset/success_default": 0.5,
        "Reset/next_stage_default": 0.75,
        "Reset/time_to_next_default": 2.0,
        "Reset/share_default": 1.0,
        "Reset/share_grasp": 0.0,
        "Reset/share_drag": 0.0,
        "Reset/share_goal": 0.0,
    }


def test_bank_path_falls_back_to_the_repository_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="not found"):
        reset_bank._resolve_bank_path("logs/reset_banks/missing.pt")
    (tmp_path / "local.pt").touch()
    assert reset_bank._resolve_bank_path("local.pt") == reset_bank.Path("local.pt")
    # not in the working directory: found relative to the repository root (any file there serves the check)
    resolved = reset_bank._resolve_bank_path("isaaclab.sh")
    assert resolved.is_absolute() and resolved.exists() and resolved.parent != tmp_path


def test_quotas_split_evenly():
    assert CollectCfg(num_states=1000).quotas() == {"grasp": 334, "drag": 333, "goal": 333}
    assert CollectCfg(num_states=12).quotas() == {"grasp": 4, "drag": 4, "goal": 4}
    with pytest.raises(ValueError):
        CollectCfg(num_states=2)


def test_capture_windows():
    # env 0: pre-grasp pose held long enough, still, open; env 1: first step; env 2: pose held too briefly;
    # env 3: still moving; env 4: closed; env 5: fresh grasp, barely moved; env 6: grasp held too long;
    # env 7: moved too far; env 8: goal point inside the goal zone; env 9: inside the success disk; env 10: hold done;
    # envs 11-14: as envs 1, 0, 5 and 8, but in another task stage than their bucket's
    inf = math.inf
    collector = ResetCollector.__new__(ResetCollector)
    collector.cfg = CollectCfg()
    hold, other = 8, 4  # DragSmState.PREGRASP_HOLD, DRAG_CLOTH
    collector.sm = SimpleNamespace(
        sm_state=torch.tensor([hold, 0, hold, hold, hold] + [other] * 6 + [0, hold, other, other]),
        sm_wait_time=torch.tensor([0.3, 0.0, 0.1, 0.3, 0.3] + [0.0] * 6 + [0.0, 0.3, 0.0, 0.0]),
    )
    locked = torch.tensor([False] * 5 + [True] * 6 + [False, False, True, True])
    collector.gripper = SimpleNamespace(
        closed=torch.tensor([False, False, False, False, True] + [True] * 6 + [False, False, True, True]),
        constraint=SimpleNamespace(active=locked, contacts_disabled=locked),
    )
    collector.command = SimpleNamespace(
        stage=torch.tensor([1, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3] + [1, 0, 3, 2]),
        holding=torch.tensor([False] * 10 + [True] + [False] * 4),
    )
    ongoing = torch.ones(15, dtype=torch.bool)
    step = torch.tensor([5, 1] + [5] * 9 + [1, 5, 5, 5])
    progress = torch.tensor([0.0] * 5 + [0.05, 0.05, 0.2, 0.6, 0.9, 0.9] + [0.0, 0.0, 0.05, 0.6])
    distance = torch.tensor([0.3] * 5 + [0.25, 0.25, 0.2, 0.07, 0.03, 0.03] + [0.3, 0.3, 0.25, 0.07])
    ee_speed = torch.tensor([0.002, 0.0, 0.002, 0.05, 0.002] + [0.1] * 6 + [0.0, 0.002, 0.1, 0.1])
    since_squeeze = torch.tensor([inf] * 5 + [0.3, 0.8, 0.3, 2.0, 3.0, 3.0] + [inf, inf, 0.3, 2.0])
    approach, grasp, drag, goal = collector._windows(ongoing, step, progress, distance, ee_speed, since_squeeze)
    assert approach.nonzero().squeeze(-1).tolist() == [1]
    assert grasp.nonzero().squeeze(-1).tolist() == [0]
    assert drag.nonzero().squeeze(-1).tolist() == [5]
    assert goal.nonzero().squeeze(-1).tolist() == [8]
    # the window of a bucket requires the stage of the same index
    assert list(BUCKET_NAMES) == [stage.name.lower() for stage in ClothDragStage]
    # a step of a reset never captures
    windows = collector._windows(torch.zeros(15, dtype=torch.bool), step, progress, distance, ee_speed, since_squeeze)
    assert not any(window.any() for window in windows)


def test_pregrasp_offsets_fill_a_cone_biased_to_the_centre():
    collector = ResetCollector.__new__(ResetCollector)
    collector.cfg = CollectCfg()
    collector.device = "cpu"
    collector._generator = torch.Generator().manual_seed(0)
    offsets = collector._pregrasp_offsets(4000)
    (h_low, h_high), (r_low, r_high) = collector.cfg.pregrasp_height_range, collector.cfg.pregrasp_radius_range
    height, radius = offsets[:, 2], offsets[:, :2].norm(dim=1)
    assert ((height >= h_low) & (height <= h_high)).all()
    r_max = r_low + (r_high - r_low) * (height - h_low) / (h_high - h_low)
    assert (radius <= r_max + 1e-7).all()
    # uniform radius (mean share 0.5), not uniform area (mean share 2/3): more states close above the node
    assert (radius / r_max).mean().item() == pytest.approx(0.5, abs=0.03)
    # every pre-grasp pose lies within the GRASP stage's approach radius (0.04 m), with margin for the tracking error
    assert offsets.norm(dim=1).max().item() < 0.035


def test_grasp_node_choice():
    nodes = torch.zeros(2, 5, 3)
    candidates = torch.tensor([[1, 2, 3], [0, 2, 4]])
    choice = torch.tensor([2, 4])
    center = torch.zeros(2, 3)
    target = torch.ones(2, 3)
    assert grasp_node_ids(nodes, center, target, candidates, choice).tolist() == [3, 2]

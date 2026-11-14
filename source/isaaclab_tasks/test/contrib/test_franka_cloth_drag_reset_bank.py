# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the reset-state bank of the Franka cloth dragging task. No Kit/GPU required."""

import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab_tasks.contrib.franka_cloth_drag.config.franka.franka_cloth_drag_env_cfg import (
    FRONTIER_SAMPLER_CFG,
    AdaptiveResetsCfg,
    DefaultResetsCfg,
    ResetStrategiesCfg,
    UniformResetsCfg,
)
from isaaclab_tasks.contrib.franka_cloth_drag.mdp.commands import ClothTargetCommand, StartStageSuccess
from isaaclab_tasks.contrib.franka_cloth_drag.mdp.frames import raised_node_ids, raised_nodes_term
from isaaclab_tasks.contrib.franka_cloth_drag.mdp.stages import START_STAGE_NAMES, ClothDragStage
from isaaclab_tasks.contrib.franka_cloth_drag.reset_bank_generation import (
    ResetBankBuilder,
    ResetBankGenerationCfg,
    subsample_to_quotas,
)
from isaaclab_tasks.contrib.franka_cloth_drag.reset_strategies import ResetFromBankCfg, reset_from_bank
from isaaclab_tasks.contrib.franka_cloth_drag.reset_strategies.bank import (
    SCHEMA_VERSION,
    ResetBank,
    _without_resolved_bodies,
    stable_hash,
)
from isaaclab_tasks.contrib.franka_cloth_drag.reset_strategies.frontier import (
    FrontierSamplerCfg,
    ResetStateFrontierSampler,
)
from isaaclab_tasks.contrib.franka_cloth_drag.reset_strategies.stage_ratios import (
    cascade_shares,
    check_min_stage_ratios,
    check_stage_ratios,
    sample_reset_rows,
)

FINGERPRINT = {
    "num_nodes": 2,
    "joint_names": ["a", "b"],
    "scene_entities": ["cube", "deformable", "robot"],
    "mesh_default_local": [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
    "physics_hash": "p",
    "task_hash": "t",
}


def _bank(stages):
    """A bank with the given start stage of every row; the row index is stored as a joint position."""
    num = len(stages)
    data = {
        "scene": {"joint_pos": torch.arange(num, dtype=torch.float).unsqueeze(1).repeat(1, 2)},
        "task": {"grasped": torch.zeros(num, 2, dtype=torch.bool), "stage": torch.tensor(stages, dtype=torch.long)},
    }
    return ResetBank(data, {"schema_version": SCHEMA_VERSION, "fingerprint": dict(FINGERPRINT)})


def test_save_and_load_round_trip(tmp_path):
    bank = _bank([0, 1, 2, 3])
    path = tmp_path / "bank.pt"
    bank.save(path)
    loaded = ResetBank.load(path)
    assert loaded.meta == bank.meta
    torch.testing.assert_close(loaded.data["scene"]["joint_pos"], bank.data["scene"]["joint_pos"])
    assert loaded.stage_counts() == {"approach": 1, "grasp": 1, "drag": 1, "goal": 1}
    with pytest.raises(ValueError):
        bank.save(tmp_path / "bank.npz")


def test_load_rejects_another_schema_version(tmp_path):
    bank = _bank([1])
    bank.meta["schema_version"] = SCHEMA_VERSION + 1
    bank.save(tmp_path / "old.pt")
    with pytest.raises(ValueError, match="schema version"):
        ResetBank.load(tmp_path / "old.pt")


def test_fingerprint_mismatch():
    bank = _bank([1])
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


def test_subsample_keeps_exact_stage_quotas():
    bank = _bank([0, 1, 2, 3] * 3 + [0])
    quotas = {"approach": 2, "grasp": 2, "drag": 2, "goal": 2}
    rows = subsample_to_quotas(bank, quotas, torch.Generator().manual_seed(0))
    kept = bank.select(rows)
    assert len(kept) == 8
    assert kept.stage_counts() == quotas
    with pytest.raises(ValueError, match="fewer"):
        subsample_to_quotas(bank, {"grasp": 4})


def test_stage_ratios_sample_their_stages():
    bank = _bank([0, 1, 1, 2, 3, 3])
    # the beginning of the task is a stored state too
    shares = check_stage_ratios({"approach": 1.0}, bank)
    assert (sample_reset_rows(bank, shares, 50) == 0).all()
    shares = check_stage_ratios({"grasp": 1.0, "goal": 1.0}, bank)
    rows = sample_reset_rows(bank, shares, 200)
    assert set(rows.tolist()) == {1, 2, 4, 5}
    with pytest.raises(ValueError, match="Unknown"):
        check_stage_ratios({"unknown": 1.0}, bank)
    with pytest.raises(ValueError, match="no stored states"):
        check_stage_ratios({"drag": 1.0}, _bank([1]))
    with pytest.raises(ValueError, match="no stored states"):
        check_stage_ratios({"approach": 1.0}, _bank([1]))
    with pytest.raises(ValueError, match="no stored states"):
        check_stage_ratios({"approach": 0.5, "grasp": 0.5}, None)


def test_default_start_stage_name_is_a_compatibility_alias():
    bank = _bank([0, 1])
    assert bank.stage_rows("default").tolist() == bank.stage_rows("approach").tolist() == [0]
    torch.testing.assert_close(check_stage_ratios({"default": 1.0}, bank), torch.tensor([1.0, 0.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="provided more than once"):
        check_stage_ratios({"default": 0.5, "approach": 0.5}, bank)


def test_reset_rows_use_weights_inside_the_selected_stage():
    bank = _bank([1, 1, 2])
    shares = check_stage_ratios({"grasp": 1.0}, bank)
    rows = sample_reset_rows(bank, shares, 50, torch.tensor([0.0, 1.0, 100.0]))
    assert rows.tolist() == [1] * 50
    with pytest.raises(ValueError, match="shape"):
        sample_reset_rows(bank, shares, 1, torch.ones(2))


def test_frontier_uses_linearly_weighted_recent_outcomes():
    sampler = ResetStateFrontierSampler(2, "cpu", FrontierSamplerCfg(window_size=4, weight_increment=1.0))
    sampler.record(torch.tensor([0, 0, 0, 0]), torch.tensor([0, 0, 0, 1]))
    assert sampler.success_rate[0] == pytest.approx(0.4)

    sampler.record(torch.tensor([1, 1]), torch.tensor([0, 1]))
    assert sampler.success_rate[1] == pytest.approx(2.0 / 3.0)

    # The next outcome replaces the oldest one: chronological history is now [0, 0, 1, 0].
    sampler.record(torch.tensor([0]), torch.tensor([0]))
    assert sampler.success_rate[0] == pytest.approx(0.3)
    assert sampler.history_size.tolist() == [4, 2]


def test_frontier_beta_weights_peak_at_the_target_and_stay_positive():
    sampler = ResetStateFrontierSampler(3, "cpu", FrontierSamplerCfg(target_success_estimate=0.5))
    sampler.success_rate[:] = torch.tensor([0.0, 0.5, 1.0])
    weights = sampler.sampling_weights()
    assert weights[1] > weights[0]
    assert weights[1] > weights[2]
    assert weights[0] > 0.0 and weights[2] > 0.0
    assert weights[0] == pytest.approx(float(weights[2]))

    metrics = sampler.metrics(torch.arange(3))
    assert metrics["frontier_success_mean"] == pytest.approx(0.5)
    assert metrics["frontier_success_median"] == pytest.approx(0.5)
    assert metrics["frontier_hard_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["frontier_medium_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["frontier_solved_fraction"] == pytest.approx(1.0 / 3.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [("window_size", 0), ("weight_increment", -1.0), ("target_success_estimate", 1.0), ("temperature", 0.5)],
)
def test_frontier_settings_are_checked(field, value):
    with pytest.raises(ValueError):
        FrontierSamplerCfg(**{field: value})


def test_reset_strategy_presets():
    """Three self-contained strategies: the default resets use no bank, the bank strategies no default reset event."""
    cfg = ResetStrategiesCfg()
    # without an override, the default resets
    assert isinstance(cfg.default, DefaultResetsCfg) and isinstance(cfg.default_resets, DefaultResetsCfg)
    assert isinstance(cfg.uniform_resets, UniformResetsCfg) and isinstance(cfg.adaptive_resets, AdaptiveResetsCfg)
    # only the three randomized resets: the success per start stage is a metric of the cloth target command
    assert set(vars(cfg.default_resets)) == {"reset_robot_joints", "reset_deformable", "reset_cube"}
    for strategy in (cfg.uniform_resets, cfg.adaptive_resets):
        # only the bank term, with all four stages drawn from the bank
        assert set(vars(strategy)) == {"reset_from_bank"}
        term = strategy.reset_from_bank
        assert term.func.endswith(".reset_strategies.reset_from_bank:ResetFromBank") and term.mode == "reset"
        assert term.bank_path is not None
        assert set(term.start_stage_ratios) == {"approach", "grasp", "drag", "goal"}
        # the frontier is a switch of both strategies, with the settings shared in the env cfg
        assert isinstance(term.use_frontier_sampling, bool)
        assert term.frontier_sampling_parameters.to_dict() == FRONTIER_SAMPLER_CFG.to_dict()
    assert cfg.uniform_resets.reset_from_bank.min_stage_ratios is None
    assert cfg.adaptive_resets.reset_from_bank.min_stage_ratios is not None


@pytest.mark.parametrize(
    "success, expected",
    [
        ([1.0, 0.5, 0.5, 0.5], [0.30, 0.45, 0.15, 0.10]),  # nothing solved; default's success does not count
        ([0.0, 0.5, 0.5, 1.0], [0.30, 0.45, 0.20, 0.05]),  # goal solved: its share goes to drag
        ([0.0, 0.5, 1.0, 1.0], [0.30, 0.60, 0.05, 0.05]),  # drag too: all it holds goes to grasp
        ([0.0, 1.0, 1.0, 1.0], [0.85, 0.05, 0.05, 0.05]),  # all solved: only the minimums stay in the stages
        ([0.0, 1.0, 0.5, 0.5], [0.70, 0.05, 0.15, 0.10]),  # only grasp solved: drag and goal keep theirs
        ([0.0, 0.5, 0.5, 0.91], [0.30, 0.45, 0.175, 0.075]),  # goal halfway through the range: half of its surplus
    ],
)
def test_cascade_moves_solved_shares_to_the_stage_before(success, expected):
    start = torch.tensor([0.30, 0.45, 0.15, 0.10])
    minimum = torch.tensor([0.20, 0.05, 0.05, 0.05])
    shares = cascade_shares(start, minimum, torch.tensor(success), (0.85, 0.97))
    assert shares.tolist() == pytest.approx(expected, abs=1e-6)
    assert shares.sum().item() == pytest.approx(1.0)


def test_adaptive_share_settings_are_checked():
    start = torch.tensor([0.30, 0.45, 0.15, 0.10])
    with pytest.raises(ValueError, match="Unknown"):
        check_min_stage_ratios({"unknown": 0.1}, start, (0.85, 0.97))
    with pytest.raises(ValueError, match="minimum"):
        check_min_stage_ratios({"goal": 0.2}, start, (0.85, 0.97))
    with pytest.raises(ValueError, match="success_range"):
        check_min_stage_ratios({"goal": 0.05}, start, (0.97, 0.85))


APPROACH_ONLY_SHARES = {
    "Reset/share_approach_raw": 1.0,
    "Reset/share_grasp_raw": 0.0,
    "Reset/share_drag_raw": 0.0,
    "Reset/share_goal_raw": 0.0,
}


class _Command:
    """The parts of :class:`ClothTargetCommand` that the reset term and the success log use: the outcome and the start
    stage of the episodes, their real bookkeeping, and the same reset."""

    def __init__(self, success, window_steps):
        num = len(success)
        self.success = success
        self._time_to_stage = torch.full((num, len(ClothDragStage)), -1.0)
        self._start_stage = torch.zeros(num, dtype=torch.long)
        self.start_stage_success = StartStageSuccess(window_steps)

    time_to_stage = property(lambda self: self._time_to_stage)
    start_stage = property(lambda self: self._start_stage)
    episode_outcomes = ClothTargetCommand.episode_outcomes

    def reset(self, env_ids, step):
        """As :meth:`ClothTargetCommand.reset`: log the finished episodes, then start at the beginning of the task."""
        log = {}
        if step > 0:
            success, reached = self.episode_outcomes(env_ids)
            log = self.start_stage_success.record(step, self._start_stage[env_ids], success, reached)
        self._start_stage[env_ids] = int(ClothDragStage.APPROACH)
        return {f"Metrics/cloth_target/{key}": value for key, value in log.items()}


def _fake_env(success, log_dir=None, window_steps=480):
    """A fake environment whose finished episodes' outcome is set in ``success`` and the command's
    ``time_to_stage``."""
    command = _Command(success, window_steps)
    return SimpleNamespace(
        num_envs=len(success),
        device="cpu",
        common_step_counter=0,
        extras={},
        cfg=SimpleNamespace(log_dir=log_dir),
        command_manager=SimpleNamespace(get_term=lambda name: command),
    )


def _reset_term(monkeypatch, bank, success, log_dir=None, window_steps=480, **params):
    """A ResetFromBank term with the configuration fields ``params`` on a fake environment. The physical restore is a
    no-op; the task restore sets the stored stage as the command's start stage. Without ``start_stage_ratios`` the
    term draws nothing: only forced rows start from a stored state."""
    env = _fake_env(success, log_dir, window_steps)
    command = env.command_manager.get_term("cloth_target")

    def restore_task_state(env, bank, rows, env_ids):
        command.start_stage[env_ids] = bank.stage[rows]

    monkeypatch.setattr(reset_from_bank, "restore_physical", lambda *args, **kwargs: None)
    monkeypatch.setattr(reset_from_bank, "restore_task_state", restore_task_state)
    term = reset_from_bank.ResetFromBank(ResetFromBankCfg(**params), env)
    if bank is not None:
        term.set_bank(bank)
    return term, env


def _reset(term, env, env_ids, step):
    """One reset of ``env_ids`` as the environment does it: reset events, new log dict, command reset, reset hooks.
    Without a term, the default resets."""
    env.common_step_counter = step
    if term is not None:
        term(env, env_ids)
    env.extras["log"] = env.command_manager.get_term("cloth_target").reset(env_ids, step)
    if term is not None:
        term.reset(env_ids)
    return env.extras["log"]


def _rates(source, suffix, task, stage):
    """Expected task and stage success of a start stage."""
    prefix = "Metrics/cloth_target"
    return {f"{prefix}/task_success_{source}_{suffix}": task, f"{prefix}/stage_success_{source}_{suffix}": stage}


def _without_state_logs(log):
    """Remove per-state diagnostics when a test targets only the per-source log contract."""
    return {key: value for key, value in log.items() if "state_visit_fraction" not in key}


def test_reset_logs_success_per_start_stage(monkeypatch):
    success = torch.zeros(4)
    term, env = _reset_term(monkeypatch, _bank([0, 1, 2, 3]), success)
    command = env.command_manager.get_term("cloth_target")
    all_envs = torch.arange(4)
    term.force_rows(all_envs, torch.tensor([-1, 1, 2, 3]))  # default, grasp, drag, goal
    assert _reset(term, env, all_envs, step=0) == {}  # the initial reset ends no episode
    assert command.start_stage.tolist() == [0, 1, 2, 3]

    # envs 0-2 end (default success, grasp failure, drag success), env 3 (goal) runs on; all restart from defaults
    success[:] = torch.tensor([1.0, 0.0, 1.0, 1.0])
    # env 0 reached GRASP, env 1 never reached DRAG, env 2 reached GOAL
    command.time_to_stage[0, 1:] = torch.tensor([2.0, 3.0, 4.0])
    command.time_to_stage[1, 1] = 0.25
    command.time_to_stage[2, 1:] = torch.tensor([0.25, 0.25, 1.5])
    log = _without_state_logs(_reset(term, env, torch.tensor([0, 1, 2]), step=10))
    first_episodes = _rates("approach", "smoothed", 1.0, 1.0)
    first_episodes |= _rates("grasp", "smoothed", 0.0, 0.0) | _rates("drag", "smoothed", 1.0, 1.0)
    assert log == (
        APPROACH_ONLY_SHARES
        | _rates("approach", "raw", 1.0, 1.0)
        | _rates("grasp", "raw", 0.0, 0.0)
        | _rates("drag", "raw", 1.0, 1.0)
        | first_episodes
    )
    assert term.rows.tolist() == [-1, -1, -1, 3]
    assert command.start_stage.tolist() == [0, 0, 0, 3]

    # env 3 (goal) ends with the success, which is its next stage, though it reached no stage on the way
    log = _without_state_logs(_reset(term, env, torch.tensor([3]), step=20))
    assert log == (
        APPROACH_ONLY_SHARES | _rates("goal", "raw", 1.0, 1.0) | first_episodes | _rates("goal", "smoothed", 1.0, 1.0)
    )


@pytest.mark.parametrize("with_bank_term", [False, True])
def test_default_starts_log_the_same_success_keys(monkeypatch, with_bank_term):
    """The command logs the success of the default resets, with or without a bank term; the term adds its shares."""
    success = torch.tensor([1.0, 0.0, 0.0, 1.0])
    if with_bank_term:
        term, env = _reset_term(monkeypatch, None, success)
    else:
        term, env = None, _fake_env(success)
    _reset(term, env, torch.arange(4), step=0)
    command = env.command_manager.get_term("cloth_target")
    command.time_to_stage[:, 1] = torch.tensor([1.0, 2.0, -1.0, 3.0])
    log = _reset(term, env, torch.arange(4), step=5)
    rates = _rates("approach", "raw", 0.5, 0.75) | _rates("approach", "smoothed", 0.5, 0.75)
    assert log == (APPROACH_ONLY_SHARES | rates if with_bank_term else rates)


def test_start_stage_success_smooths_over_the_window():
    """The window grows from the first episodes, weighs every episode equally and forgets the steps that left it."""
    tracker = StartStageSuccess(window_steps=10)
    approach_key, goal_key = "task_success_approach_smoothed", "task_success_goal_smoothed"

    def record(step, start_stage, success):
        start_stage, success = torch.tensor(start_stage), torch.tensor(success)
        return tracker.record(step, start_stage, success, success)

    log = record(1, [0, 3], [1.0, 1.0])
    assert log[approach_key] == 1.0 and log[goal_key] == 1.0

    log = record(5, [0, 0, 3], [0.0, 0.0, 0.0])
    assert log["task_success_approach_raw"] == 0.0
    assert log[approach_key] == pytest.approx(1.0 / 3.0)  # 1 of 3 episodes, not the mean of the two steps
    assert log[goal_key] == pytest.approx(0.5)

    log = record(10, [0], [0.0])  # the window of 10 steps still holds step 1
    assert log[approach_key] == pytest.approx(0.25)

    log = record(11, [0], [0.0])
    assert log[approach_key] == 0.0 and log[goal_key] == 0.0

    log = record(15, [0], [1.0])  # step 5 left the window too: no goal episode is left
    assert log[approach_key] == pytest.approx(1.0 / 3.0) and goal_key not in log
    with pytest.raises(ValueError, match="success_window_steps"):
        StartStageSuccess(window_steps=0)


def test_reset_averaging_settings_are_checked(monkeypatch):
    with pytest.raises(ValueError, match="state_success_export_interval_steps"):
        _reset_term(monkeypatch, None, torch.zeros(1), state_success_export_interval_steps=-1)
    with pytest.raises(ValueError, match="ratio_hold_steps"):
        _reset_term(monkeypatch, None, torch.zeros(1), ratio_hold_steps=-1)
    # ratios need the stored states they draw, and adaptive ratios their start ratios
    with pytest.raises(ValueError, match="no stored states"):
        _reset_term(monkeypatch, None, torch.zeros(1), start_stage_ratios={"approach": 1.0})
    with pytest.raises(ValueError, match="start ratios"):
        _reset_term(monkeypatch, None, torch.zeros(1), min_stage_ratios={"goal": 0.05})


def test_reset_counts_visits_per_stored_state(monkeypatch):
    term, env = _reset_term(monkeypatch, _bank([1, 1, 2, 3]), torch.zeros(4))
    all_envs = torch.arange(4)
    term.force_rows(all_envs, torch.tensor([0, 0, 2, -1]))
    _reset(term, env, all_envs, step=0)  # the initial reset ends no episode
    assert term.state_visit_count.tolist() == [0, 0, 0, 0]

    _reset(term, env, all_envs, step=1)
    # row 0 started two finished episodes, row 2 one; env 3 started without a stored state
    assert term.state_visit_count.tolist() == [2, 0, 1, 0]

    term.set_bank(_bank([1]))
    assert term.state_visit_count.tolist() == [0]


def test_frontier_tracks_next_stage_outcomes_and_logs_visits(monkeypatch, tmp_path):
    bank = _bank([1, 1, 2, 3])
    bank.save(tmp_path / "bank.pt")
    success = torch.zeros(4)
    term, env = _reset_term(
        monkeypatch,
        bank,
        success,
        log_dir=str(tmp_path),
        bank_path=str(tmp_path / "bank.pt"),
        use_frontier_sampling=True,
        frontier_sampling_parameters=FrontierSamplerCfg(window_size=4, weight_increment=1.0),
        state_success_export_interval_steps=1,
    )
    all_envs = torch.arange(4)
    term.force_rows(all_envs, torch.arange(4))
    _reset(term, env, all_envs, step=0)

    command = env.command_manager.get_term("cloth_target")
    command.time_to_stage[0, 2] = 1.0  # grasp row reaches DRAG
    command.time_to_stage[2, 3] = 1.0  # drag row reaches GOAL
    success[:] = torch.tensor([0.0, 1.0, 0.0, 1.0])
    log = _reset(term, env, all_envs, step=1)

    # Full-task success on env 1 must not make its grasp row a frontier success.
    assert term._frontier.success_rate.tolist() == [1.0, 0.0, 1.0, 1.0]
    assert log["Reset/state_visit_fraction_1_grasp"] == 1.0
    assert log["Reset/state_visit_fraction_0_grasp"] == 0.0
    assert log["Reset/frontier_success_mean_grasp"] == pytest.approx(0.5)
    assert log["Reset/frontier_success_median_grasp"] == 0.0
    assert log["Reset/frontier_hard_fraction_grasp"] == pytest.approx(0.5)
    assert log["Reset/frontier_medium_fraction_grasp"] == 0.0
    assert log["Reset/frontier_solved_fraction_grasp"] == pytest.approx(0.5)
    assert log["Reset/frontier_effective_pool_fraction_grasp"] == pytest.approx(1.0)
    assert log["Reset/frontier_top_1_percent_mass_grasp"] == pytest.approx(0.5)

    snapshot = torch.load(tmp_path / "reset_state_success" / "step_00000001.pt", weights_only=True)
    assert snapshot["schema_version"] == 3
    assert snapshot["visit_count"].tolist() == [1, 1, 1, 1]
    assert snapshot["frontier"]["cfg"]["window_size"] == 4
    assert snapshot["frontier"]["history_size"].tolist() == [1, 1, 1, 1]
    assert snapshot["frontier"]["success_estimate"].tolist() == [1.0, 0.0, 1.0, 1.0]
    assert snapshot["frontier"]["sampling_probability"].tolist() == pytest.approx([0.5, 0.5, 1.0, 1.0])


def test_reset_logs_per_stage_visit_frequency_ratios(monkeypatch):
    term, _ = _reset_term(monkeypatch, _bank([1] * 5 + [2] * 5 + [3] * 5), torch.zeros(1))
    term.state_visit_count[:] = torch.tensor([0, 1, 2, 3, 4] * 3)

    log = term._state_log()
    for stage in ("grasp", "drag", "goal"):
        for visits in ("0", "1", "2", "3", "4_plus"):
            assert log[f"Reset/state_visit_fraction_{visits}_{stage}"] == pytest.approx(0.2)


def test_reset_exports_state_statistics_periodically(monkeypatch, tmp_path):
    success = torch.ones(1)
    term, env = _reset_term(
        monkeypatch,
        _bank([3]),
        success,
        log_dir=str(tmp_path),
        state_success_export_interval_steps=5,
    )
    term.force_rows(torch.tensor([0]), torch.tensor([0]))
    _reset(term, env, torch.tensor([0]), step=0)
    term.force_rows(torch.tensor([0]), torch.tensor([0]))
    _reset(term, env, torch.tensor([0]), step=4)
    assert not (tmp_path / "reset_state_success").exists()

    _reset(term, env, torch.tensor([0]), step=5)
    snapshot = torch.load(tmp_path / "reset_state_success" / "step_00000005.pt", weights_only=True)
    latest = torch.load(tmp_path / "reset_state_success" / "latest.pt", weights_only=True)
    assert snapshot.keys() == latest.keys()
    assert snapshot["schema_version"] == 3
    assert snapshot["env_step"] == 5
    assert snapshot["bank_fingerprint"] == FINGERPRINT
    assert snapshot["stage"].tolist() == [3]
    assert snapshot["visit_count"].tolist() == [2]
    assert snapshot["frontier"] is None


def test_adaptive_shares_follow_the_smoothed_success(monkeypatch, tmp_path):
    success = torch.ones(4)
    bank = _bank([0, 1, 2, 3])
    bank.save(tmp_path / "bank.pt")
    term, env = _reset_term(
        monkeypatch,
        bank,
        success,
        bank_path=str(tmp_path / "bank.pt"),
        start_stage_ratios={"approach": 0.30, "grasp": 0.45, "drag": 0.15, "goal": 0.10},
        min_stage_ratios={"approach": 0.20, "grasp": 0.05, "drag": 0.05, "goal": 0.05},
        ratio_shift_success_range=(0.85, 0.97),
        window_steps=3,
        ratio_hold_steps=3,
    )
    drawn = []

    def record_shares(_, shares, num):
        drawn.append(shares.tolist())
        return torch.full((num,), -1)

    monkeypatch.setattr(reset_from_bank, "sample_reset_rows", record_shares)
    all_envs, goal_envs = torch.arange(4), torch.arange(1, 4)

    def goal_batch(step):
        """Restart three environments from a goal state and let the first draw, which shows the current shares."""
        term.force_rows(goal_envs, torch.full((3,), 3))
        return _reset(term, env, all_envs, step=step), drawn[-1]

    start, handed_over = [0.30, 0.45, 0.15, 0.10], [0.30, 0.45, 0.20, 0.05]
    goal_key = "Metrics/cloth_target/task_success_goal_smoothed"
    assert goal_batch(step=0)[1] == pytest.approx(start)
    # the hold keeps the start shares, although the goal starts succeed from the first episodes
    log, shares = goal_batch(step=1)
    assert log[goal_key] == 1.0 and shares == pytest.approx(start)
    assert goal_batch(step=2)[1] == pytest.approx(start)
    # after the hold, the solved goal stage hands its share to drag; the draw reads the success up to the last reset
    assert goal_batch(step=3)[1] == pytest.approx(handed_over)
    # a stage without episodes in the window stays solved
    for step in range(4, 8):
        log = _reset(term, env, torch.tensor([0]), step=step)
    assert goal_key not in log
    assert drawn[-1] == pytest.approx(handed_over)
    # failures take it back
    success.zero_()
    goal_batch(step=8)
    assert goal_batch(step=9)[1] == pytest.approx(start)


def test_raised_nodes_come_from_the_bank_term_without_the_cloth_reset_event():
    """A bank strategy has no cloth reset event: the raised nodes are those the bank term restores."""
    raised = torch.tensor([[2, 4]])
    terms = {"reset_from_bank": SimpleNamespace(func=SimpleNamespace(raised_node_ids=raised))}

    def get_term_cfg(name):
        if name not in terms:
            raise ValueError(f"Event term '{name}' not found.")
        return terms[name]

    manager = SimpleNamespace(active_terms={"reset": list(terms)}, get_term_cfg=get_term_cfg)
    env = SimpleNamespace(event_manager=manager)
    assert raised_node_ids(env, "reset_deformable") is raised
    assert raised_node_ids(env, None) is None
    # the cloth reset event comes first where a strategy has both (bank generation)
    cloth_term = SimpleNamespace(raised_node_ids=torch.tensor([[1, 3]]))
    terms["reset_deformable"] = SimpleNamespace(func=cloth_term)
    assert raised_nodes_term(env, "reset_deformable") is cloth_term
    # a term that is not set up yet shows its class, which has no node ids
    terms.pop("reset_deformable")
    terms["reset_from_bank"] = SimpleNamespace(func=reset_from_bank.ResetFromBank)
    assert raised_node_ids(env, "reset_deformable") is None


def test_bank_path_falls_back_to_the_repository_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="not found"):
        reset_from_bank._resolve_bank_path("logs/reset_banks/missing.pt")
    (tmp_path / "local.pt").touch()
    assert reset_from_bank._resolve_bank_path("local.pt") == reset_from_bank.Path("local.pt")
    # not in the working directory: found relative to the repository root (any file there serves the check)
    resolved = reset_from_bank._resolve_bank_path("isaaclab.sh")
    assert resolved.is_absolute() and resolved.exists() and resolved.parent != tmp_path


def test_quotas_split_evenly_over_all_stages():
    assert ResetBankGenerationCfg().quotas() == {"approach": 1000, "grasp": 1000, "drag": 1000, "goal": 1000}
    assert ResetBankGenerationCfg(num_states=14).quotas() == {"approach": 4, "grasp": 4, "drag": 3, "goal": 3}
    with pytest.raises(ValueError):
        ResetBankGenerationCfg(num_states=3)


def test_capture_windows():
    # env 0: pre-grasp pose held long enough, still, open; env 1: first step; env 2: pose held too briefly;
    # env 3: still moving; env 4: closed; env 5: fresh grasp, barely moved; env 6: grasp held too long;
    # env 7: moved too far; env 8: goal point inside the goal zone; env 9: inside the success disk; env 10: hold done;
    # envs 11-14: as envs 1, 0, 5 and 8, but in another task stage than their window's
    inf = math.inf
    collector = ResetBankBuilder.__new__(ResetBankBuilder)
    collector.cfg = ResetBankGenerationCfg()
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
    # the window of a start stage requires the task stage of the same index
    assert len(START_STAGE_NAMES) == len(ClothDragStage)
    # a step of a reset never captures
    windows = collector._windows(torch.zeros(15, dtype=torch.bool), step, progress, distance, ee_speed, since_squeeze)
    assert not any(window.any() for window in windows)


def test_pregrasp_offsets_fill_a_cone_biased_to_the_centre():
    collector = ResetBankBuilder.__new__(ResetBankBuilder)
    collector.cfg = ResetBankGenerationCfg()
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

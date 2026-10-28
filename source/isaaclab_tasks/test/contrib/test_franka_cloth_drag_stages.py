# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the stage checks of the Franka cloth dragging task. No Kit/GPU required."""

from types import SimpleNamespace

import torch

from isaaclab_tasks.contrib.franka_cloth_drag.config.franka.franka_cloth_drag_env_cfg import RewardsCfg
from isaaclab_tasks.contrib.franka_cloth_drag.mdp.cloth_grasp import ClothGraspConstraint
from isaaclab_tasks.contrib.franka_cloth_drag.mdp.rewards import gripper_closed, gripper_closed_on_raised_node
from isaaclab_tasks.contrib.franka_cloth_drag.mdp.stages import (
    cloth_goal_xy_distance,
    gripper_holds_raised_node,
    pad_box_mask,
)

NUM_NODES = 6
CFG = SimpleNamespace(gripper_action_name="gripper_action", raised_nodes_event="reset_deformable")


class _EventManager:
    """Event manager with one reset term that raises the given nodes, or none."""

    def __init__(self, raised: torch.Tensor | None):
        self._raised = raised

    def get_term_cfg(self, name):
        if self._raised is None:
            raise ValueError(f"Event term '{name}' not found.")
        return SimpleNamespace(func=SimpleNamespace(raised_node_ids=self._raised))


def _env(active, grasped, raised, constraint=True):
    """A fake environment: per env, whether a lock is active, which nodes it locked, and which nodes are raised."""
    active = torch.tensor(active)
    locked = torch.zeros(len(active), NUM_NODES, dtype=torch.bool)
    for env_id, nodes in enumerate(grasped):
        locked[env_id, nodes] = True
    term = SimpleNamespace(
        is_grasping=active,
        constraint=SimpleNamespace(active=active, grasped=locked) if constraint else None,
    )
    return SimpleNamespace(
        action_manager=SimpleNamespace(get_term=lambda name: term),
        event_manager=_EventManager(None if raised is None else torch.tensor(raised)),
    )


def test_only_locks_with_a_raised_node_count():
    # env 0 locks raised node 4, env 1 locks only edge nodes, env 2 had locked node 4 but is no longer active
    env = _env(active=[True, True, False], grasped=[[3, 4], [0, 1], [4]], raised=[[2, 4], [2, 4], [2, 4]])
    assert gripper_holds_raised_node(env, CFG).tolist() == [True, False, False]


def test_without_raised_nodes_or_constraint_any_grasp_counts():
    env = _env(active=[True, False], grasped=[[0, 1], []], raised=None)
    assert gripper_holds_raised_node(env, CFG).tolist() == [True, False]
    env = _env(active=[True, False], grasped=[[0], []], raised=[[4], [4]], constraint=False)
    assert gripper_holds_raised_node(env, CFG).tolist() == [True, False]


def _grasp_reward_env(points, edges=(), triangles=(), raised=((0,),), opening=0.005):
    """Use the real grasp selector with a stationary pad frame and a synthetic cloth mesh."""

    points = torch.tensor([points], dtype=torch.float)
    cfg = SimpleNamespace(
        gripper_action_name="gripper_action",
        raised_nodes_event="reset_deformable",
        ee_frame_name="ee_frame",
        asset_name="robot",
        object_name="deformable",
        finger_joint_name="finger",
        finger_pad_half_width=0.0125,
        finger_pad_half_length=0.0085,
        finger_gap_margin=0.003,
        grasp_min_nodes=2,
    )
    frame = SimpleNamespace(
        data=SimpleNamespace(
            target_pos_w=SimpleNamespace(torch=torch.zeros(1, 1, 3)),
            target_quat_w=SimpleNamespace(torch=torch.tensor([[[0.0, 0.0, 0.0, 1.0]]])),
        )
    )
    robot = SimpleNamespace(
        find_joints=lambda name: ([0], ["finger"]),
        data=SimpleNamespace(joint_pos=SimpleNamespace(torch=torch.tensor([[opening]]))),
    )
    cloth = SimpleNamespace(data=SimpleNamespace(nodal_pos_w=SimpleNamespace(torch=points)))
    constraint = ClothGraspConstraint.__new__(ClothGraspConstraint)
    constraint._command_cfg = cfg
    constraint._cloth = cloth
    constraint._edges = torch.tensor(edges, dtype=torch.long).reshape(-1, 2)
    constraint._triangles = torch.tensor(triangles, dtype=torch.long).reshape(-1, 3)
    constraint.active = torch.tensor([False])
    constraint.grasped = torch.zeros(1, points.shape[1], dtype=torch.bool)
    term = SimpleNamespace(closed=torch.tensor([True]), constraint=constraint, is_grasping=constraint.active)
    command = SimpleNamespace(cfg=cfg, stage=torch.tensor([1]))
    env = SimpleNamespace(
        scene={"ee_frame": frame, "robot": robot, "deformable": cloth},
        action_manager=SimpleNamespace(get_term=lambda name: term),
        command_manager=SimpleNamespace(get_term=lambda name: command),
        event_manager=_EventManager(None if raised is None else torch.tensor(raised)),
    )
    constraint._env = env
    return env, command, term


def test_closing_reward_includes_raised_nodes_captured_through_mesh_cells():
    # The raised vertex misses the pad width, but its edge midpoint or triangle center is inside.
    cases = [
        ([[0.020, 0.0, 0.0], [-0.005, 0.0, 0.008]], [[0, 1]], []),
        ([[0.020, 0.0, 0.0], [-0.005, 0.0, 0.008], [0.0, 0.0, 0.008]], [], [[0, 1, 2]]),
    ]
    for points, edges, triangles in cases:
        env, command, term = _grasp_reward_env(points, edges, triangles)
        assert not pad_box_mask(env, command.cfg, env.scene["deformable"].data.nodal_pos_w.torch)[0, 0]
        assert gripper_closed_on_raised_node(env, ["grasp"]).item() == 1.0
        term.closed[:] = False
        assert gripper_closed_on_raised_node(env, ["grasp"]).item() == 0.0


def test_closing_reward_requires_minimum_count_and_a_selected_raised_node():
    # Two capturable flat nodes do not qualify while the raised node is outside every selected cell.
    env, command, term = _grasp_reward_env([[0.04, 0.0, 0.0], [0.0, 0.0, 0.0], [0.005, 0.0, 0.0]])
    assert gripper_closed_on_raised_node(env, ["grasp"]).item() == 0.0
    env.event_manager = _EventManager(torch.tensor([[1]]))
    assert gripper_closed_on_raised_node(env, ["grasp"]).item() == 1.0
    command.cfg.grasp_min_nodes = 3
    assert gripper_closed_on_raised_node(env, ["grasp"]).item() == 0.0


def test_closing_reward_uses_stored_attachment_and_does_not_pay_for_wrong_attachment():
    env, command, term = _grasp_reward_env([[0.04, 0.0, 0.0], [0.05, 0.0, 0.0], [0.06, 0.0, 0.0]])
    term.constraint.active[:] = True
    term.constraint.grasped[:, :2] = True
    assert gripper_closed_on_raised_node(env, ["grasp"]).item() == 1.0
    # A capturable raised node beside an existing flat-node attachment cannot earn a valid-grasp reward.
    env.scene["deformable"].data.nodal_pos_w.torch.zero_()
    term.constraint.grasped[:, 0] = False
    term.constraint.grasped[:, 2] = True
    assert gripper_closed_on_raised_node(env, ["grasp"]).item() == 0.0


def test_closing_rewards_are_limited_to_their_stages():
    env, command, term = _grasp_reward_env([[0.0, 0.0, 0.0], [0.005, 0.0, 0.0]])
    for stage in range(4):
        command.stage[:] = stage
        assert gripper_closed(env, ["approach"]).item() == float(stage == 0)
        assert gripper_closed_on_raised_node(env, ["grasp"]).item() == float(stage == 1)
    command.stage[:] = 0
    term.closed[:] = False
    assert gripper_closed(env, ["approach"]).item() == 0.0


def test_closing_reward_without_raised_nodes_or_attachment_constraint():
    env, command, term = _grasp_reward_env([[0.0, 0.0, 0.0], [0.005, 0.0, 0.0]], raised=None)
    assert gripper_closed_on_raised_node(env, ["grasp"]).item() == 1.0
    term.constraint = None
    assert gripper_closed_on_raised_node(env, ["grasp"]).item() == 0.0


def test_configured_closing_rewards_prefer_capture_over_waiting_open():
    cfg = RewardsCfg()
    closing_terms = [
        getattr(cfg, name)
        for name in ("grasp_between_fingers", "pen_close_without_bump", "grasp_close_on_bump", "pen_close_in_approach")
        if hasattr(cfg, name)
    ]
    env, command, term = _grasp_reward_env([[0.020, 0.0, 0.0], [-0.005, 0.0, 0.008]], edges=[[0, 1]])

    def closing_reward():
        return sum(reward.weight * reward.func(env, **reward.params) for reward in closing_terms).item()

    term.closed[:] = False
    assert closing_reward() == 0.0
    term.closed[:] = True
    assert closing_reward() > 0.0
    command.stage[:] = 0
    assert closing_reward() < 0.0
    command.stage[:] = 1
    env.scene["deformable"].data.nodal_pos_w.torch += 0.1
    assert closing_reward() == 0.0


class _Cloth:
    """Fake cloth asset; hashable like a real asset, since its landmark ids are cached per asset."""

    def __init__(self, nodes: torch.Tensor, shift: list[float]):
        self.data = SimpleNamespace(
            nodal_pos_w=SimpleNamespace(torch=nodes + torch.tensor(shift)),
            default_nodal_state_w=SimpleNamespace(torch=torch.cat((nodes, torch.zeros_like(nodes)), dim=-1)),
        )


def _goal_env(cloth_shift, cube_pos):
    """A fake environment with a 3 x 3 node cloth shifted in the plane, a cube, and a robot at the origin."""
    grid = torch.cartesian_prod(torch.tensor([-0.1, 0.0, 0.1]), torch.tensor([-0.1, 0.0, 0.1]))
    nodes = torch.cat((grid, torch.zeros(9, 1)), dim=1).unsqueeze(0)
    cloth = _Cloth(nodes, cloth_shift)
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=torch.zeros(1, 3)),
            root_quat_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.0, 1.0]])),
        )
    )
    cube = SimpleNamespace(data=SimpleNamespace(root_pos_w=SimpleNamespace(torch=torch.tensor([cube_pos]))))
    return SimpleNamespace(scene={"deformable": cloth, "robot": robot, "cube": cube})


def test_goal_distance_measures_the_goal_object_if_set():
    # the cube has slid 2 cm along x and 1 cm along y off the central node
    env = _goal_env(cloth_shift=[0.6, 0.0, 0.0], cube_pos=[0.62, 0.01, 0.022])
    target = torch.tensor([[0.4, 0.0, 0.0]])
    cfg = SimpleNamespace(asset_name="robot", object_name="deformable", goal_object_name=None)
    assert torch.allclose(cloth_goal_xy_distance(env, target, cfg), torch.tensor([0.2]))
    cfg.goal_object_name = "cube"
    expected = torch.tensor([0.22, 0.01]).norm().unsqueeze(0)
    assert torch.allclose(cloth_goal_xy_distance(env, target, cfg), expected)


def test_contact_latch_marks_only_touching_monitored_pairs():
    import warp as wp

    from isaaclab_tasks.contrib.franka_cloth_drag.coupler import _latch_touching_pairs

    # shapes: world 0 gripper 0, cube 1, table 2 (static); world 1 gripper 3, cube 4; bodies 0, 1, -, 2, 3
    shape_body = [0, 1, -1, 2, 3]
    shape_world = [0, 0, 0, 1, 1]
    shape_groups = [0b01, 0b10, 0, 0b01, 0b10]  # pair 0: gripper side (bit 0), cube side (bit 1)
    body_pos = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.004]]
    # touching in world 0 (0.5 mm gap), 4 mm apart in world 1 (body offset), cube on the unmonitored table
    contacts = [(0, 1, [0.0, 0.0, 0.0005]), (3, 4, [0.0, 0.0, 0.0]), (1, 2, [0.0, 0.0, 0.0])]
    device = "cpu"
    latch = wp.zeros((2, 1), dtype=wp.int32, device=device)
    wp.launch(
        _latch_touching_pairs,
        dim=len(contacts) + 1,  # one spare slot past the contact count
        inputs=[
            wp.array([len(contacts)], dtype=wp.int32, device=device),
            wp.array([c[0] for c in contacts] + [-1], dtype=wp.int32, device=device),
            wp.array([c[1] for c in contacts] + [-1], dtype=wp.int32, device=device),
            wp.zeros(len(contacts) + 1, dtype=wp.vec3, device=device),
            wp.array([c[2] for c in contacts] + [[0.0, 0.0, 0.0]], dtype=wp.vec3, device=device),
            wp.array([[0.0, 0.0, 1.0]] * (len(contacts) + 1), dtype=wp.vec3, device=device),
            wp.zeros(len(contacts) + 1, dtype=wp.float32, device=device),
            wp.zeros(len(contacts) + 1, dtype=wp.float32, device=device),
            wp.array(shape_body, dtype=wp.int32, device=device),
            wp.array(shape_world, dtype=wp.int32, device=device),
            wp.array(shape_groups, dtype=wp.int32, device=device),
            wp.array([wp.transform(p, wp.quat_identity()) for p in body_pos], dtype=wp.transform, device=device),
            1.0e-3,
            1,
        ],
        outputs=[latch],
        device=device,
    )
    assert latch.numpy()[:, 0].tolist() == [1, 0]


def test_action_term_penalties_count_only_the_named_term():
    from isaaclab_tasks.contrib.franka_cloth_drag.mdp.rewards import action_term_l2, action_term_rate_l2

    # arm term (4 dims) then gripper term (1 dim); the gripper changes a lot, the arm a little
    manager = SimpleNamespace(
        active_terms=["arm_action", "gripper_action"],
        action_term_dim=[4, 1],
        prev_action=torch.tensor([[0.0, 0.0, 0.0, 0.0, -1.0]]),
        action=torch.tensor([[0.1, 0.0, -0.2, 0.0, 1.0]]),
    )
    env = SimpleNamespace(action_manager=manager)
    assert torch.allclose(action_term_l2(env, "arm_action"), torch.tensor([0.05]))
    assert torch.allclose(action_term_l2(env, "gripper_action"), torch.tensor([1.0]))
    assert torch.allclose(action_term_rate_l2(env, "arm_action"), torch.tensor([0.05]))
    assert torch.allclose(action_term_rate_l2(env, "gripper_action"), torch.tensor([4.0]))

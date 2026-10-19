# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to drag a cloth over the table to a target with a robotic arm, to watch and debug the cloth grasp.

The state machine hovers above the crest of the central wrinkle of the ``IsaacContrib-Drag-Cloth-Franka`` cloth,
descends with an open gripper, closes it on the wrinkle, drags the cloth's central node to the task's target, holds
it there, releases the cloth and retreats. It is implemented in the kernel function `infer_state_machine`, which uses
the `warp` library to run all environments in parallel on the GPU.

The state machine commands the task's own action terms, the 4-DoF delta end-effector action and the cloth grasp
gripper, so the grasp behaves exactly as in training: the fingers close on the wrinkle and, once they rest, the cloth
nodes between the pads are locked to the end-effector frame (``--grasp_mode attach``). ``--grasp_mode friction``
holds the cloth by friction alone. Only the time-out terminates an episode, so a physics blow-up plays out on screen.

.. code-block:: bash

    # Viser web viewer on port 8750 (default); open http://localhost:8750.
    uv run python scripts/environments/state_machine/franka_cloth_drag_sm.py --num_envs 4

    # Friction-only grasp.
    uv run python scripts/environments/state_machine/franka_cloth_drag_sm.py --grasp_mode friction

    # Headless, a fixed number of steps.
    uv run python scripts/environments/state_machine/franka_cloth_drag_sm.py --viz none --num_steps 1200

"""

import argparse
import sys

import gymnasium as gym
import torch
import warp as wp

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.franka_cloth_drag.mdp import TopDownDeltaEEActionCfg
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

# add argparse arguments
parser = argparse.ArgumentParser(description="Drag a cloth to a target with a robotic arm.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--num_steps", type=int, default=None, help="Number of environment steps to run. Defaults to running until Ctrl+C."
)
parser.add_argument("--task", type=str, default="IsaacContrib-Drag-Cloth-Franka", help="The task to run.")
parser.add_argument(
    "--grasp_mode",
    type=str,
    default="attach",
    choices=["attach", "friction"],
    help="Hold the cloth by locking the grasped nodes to the end-effector frame, or by friction alone.",
)
parser.add_argument("--drag_speed", type=float, default=0.08, help="Speed of the drag [m/s].")
parser.add_argument("--episode_length", type=float, default=30.0, help="Episode length [s].")
parser.add_argument(
    "--grasp_height",
    type=float,
    default=0.0135,
    help="End-effector height for the grasp [m]. On the LOD0 mesh 13.5 mm puts 5 nodes between the pads.",
)
parser.add_argument("--drag_height", type=float, default=0.02, help="End-effector height while dragging [m].")
parser.add_argument(
    "--goal_tolerance",
    type=float,
    default=0.02,
    help="Planar distance of the central node to the target at which the drag stops [m], at most the success radius.",
)
add_launcher_args(parser)
# the task runs on Newton, so default to the Viser web viewer and let the task pick the device
parser.set_defaults(device=None, visualizer=["viser"])
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

# initialize warp
wp.init()

# height of the approach pose above the grasp pose [m]
HOVER_HEIGHT = 0.05
# nodes within this distance of the cloth's central node along y belong to the central wrinkle [m]
WRINKLE_HALF_WIDTH = 0.02
# farthest the drag target may lead the lagging end-effector [m]
DRAG_MAX_LEAD = wp.constant(0.03)
# the drag counts as stalled once the end-effector moved less than this per step for STALL_TIME [m], [s]
STALL_STEP = 0.0001
STALL_TIME = 0.5
# retreat pose: straight up is out of reach next to the robot base after the drag, so retreat outward and up [m]
RETREAT_X = wp.constant(0.45)
RETREAT_Z = wp.constant(0.12)


class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class DragSmState:
    """States for the drag state machine."""

    REST = wp.constant(0)
    APPROACH_ABOVE_CLOTH = wp.constant(1)
    APPROACH_CLOTH = wp.constant(2)
    GRASP_CLOTH = wp.constant(3)
    DRAG_CLOTH = wp.constant(4)
    HOLD_AT_TARGET = wp.constant(5)
    RELEASE_CLOTH = wp.constant(6)
    RETREAT = wp.constant(7)


STATE_NAMES = [
    "REST",
    "APPROACH_ABOVE_CLOTH",
    "APPROACH_CLOTH",
    "GRASP_CLOTH",
    "DRAG_CLOTH",
    "HOLD_AT_TARGET",
    "RELEASE_CLOTH",
    "RETREAT",
]


class DragSmWaitTime:
    """Additional wait times (in s) for states before switching."""

    REST = wp.constant(0.2)
    APPROACH_ABOVE_CLOTH = wp.constant(0.5)
    APPROACH_CLOTH = wp.constant(0.5)
    # time for the squeeze to ramp in after the grasp locked the cloth
    GRASP_SQUEEZE = wp.constant(0.3)
    # time after which a grasp that locked nothing (or a friction grasp) moves on
    GRASP_TIMEOUT = wp.constant(2.0)
    HOLD_TIMEOUT = wp.constant(2.0)
    RELEASE_CLOTH = wp.constant(0.5)


@wp.func
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    return wp.length(current_pos - desired_pos) < threshold


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pos: wp.array(dtype=wp.vec3),
    grasp_pos: wp.array(dtype=wp.vec3),
    center_pos: wp.array(dtype=wp.vec3),
    target_pos: wp.array(dtype=wp.vec3),
    is_grasping: wp.array(dtype=int),
    holding: wp.array(dtype=int),
    drag_stalled: wp.array(dtype=int),
    hold_pos: wp.array(dtype=wp.vec3),
    des_ee_pos: wp.array(dtype=wp.vec3),
    des_gripper_state: wp.array(dtype=float),
    hover_height: float,
    grasp_height: float,
    drag_height: float,
    drag_speed: float,
    goal_tolerance: float,
    approach_threshold: float,
    grasp_threshold: float,
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    grasp = wp.vec3(grasp_pos[tid][0], grasp_pos[tid][1], grasp_height)
    # planar vector from the cloth's central node to the target
    to_target = wp.vec3(target_pos[tid][0] - center_pos[tid][0], target_pos[tid][1] - center_pos[tid][1], 0.0)
    target_distance = wp.length(to_target)
    # decide next state
    if state == DragSmState.REST:
        des_ee_pos[tid] = ee_pos[tid]
        des_gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= DragSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = DragSmState.APPROACH_ABOVE_CLOTH
            sm_wait_time[tid] = 0.0
    elif state == DragSmState.APPROACH_ABOVE_CLOTH:
        des_ee_pos[tid] = grasp + wp.vec3(0.0, 0.0, hover_height)
        des_gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(ee_pos[tid], des_ee_pos[tid], approach_threshold):
            # wait for a while
            if sm_wait_time[tid] >= DragSmWaitTime.APPROACH_ABOVE_CLOTH:
                # move to next state and reset wait time
                sm_state[tid] = DragSmState.APPROACH_CLOTH
                sm_wait_time[tid] = 0.0
    elif state == DragSmState.APPROACH_CLOTH:
        des_ee_pos[tid] = grasp
        des_gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(ee_pos[tid], des_ee_pos[tid], grasp_threshold):
            # wait for a while
            if sm_wait_time[tid] >= DragSmWaitTime.APPROACH_CLOTH:
                # move to next state and reset wait time
                sm_state[tid] = DragSmState.GRASP_CLOTH
                sm_wait_time[tid] = 0.0
    elif state == DragSmState.GRASP_CLOTH:
        des_ee_pos[tid] = grasp
        des_gripper_state[tid] = GripperState.CLOSE
        # the gripper action locks the cloth once the closing fingers rest; without a lock, move on after a timeout
        locked = is_grasping[tid] == 1 and sm_wait_time[tid] >= DragSmWaitTime.GRASP_SQUEEZE
        if locked or sm_wait_time[tid] >= DragSmWaitTime.GRASP_TIMEOUT:
            # move to next state and reset wait time
            sm_state[tid] = DragSmState.DRAG_CLOTH
            sm_wait_time[tid] = 0.0
    elif state == DragSmState.DRAG_CLOTH:
        # advance the drag target the way the central node has to go, at the drag speed, leading the lagging arm by at
        # most DRAG_MAX_LEAD
        step = wp.min(target_distance, drag_speed * dt[tid])
        des = des_ee_pos[tid]
        if target_distance > 1.0e-6:
            des = des + to_target * (step / target_distance)
        des = wp.vec3(des[0], des[1], drag_height)
        lead = des - ee_pos[tid]
        if wp.length(lead) > DRAG_MAX_LEAD:
            des = ee_pos[tid] + lead * (DRAG_MAX_LEAD / wp.length(lead))
        des_ee_pos[tid] = des
        des_gripper_state[tid] = GripperState.CLOSE
        # a drag that stalls at the arm's reach limit holds where it is and lets the task judge the result
        if target_distance < goal_tolerance or drag_stalled[tid] == 1:
            # move to next state and reset wait time
            hold_pos[tid] = wp.vec3(ee_pos[tid][0], ee_pos[tid][1], drag_height)
            sm_state[tid] = DragSmState.HOLD_AT_TARGET
            sm_wait_time[tid] = 0.0
    elif state == DragSmState.HOLD_AT_TARGET:
        des_ee_pos[tid] = hold_pos[tid]
        des_gripper_state[tid] = GripperState.CLOSE
        # hold until the task counts the cloth as held at the target
        if holding[tid] == 1 or sm_wait_time[tid] >= DragSmWaitTime.HOLD_TIMEOUT:
            # move to next state and reset wait time
            sm_state[tid] = DragSmState.RELEASE_CLOTH
            sm_wait_time[tid] = 0.0
    elif state == DragSmState.RELEASE_CLOTH:
        des_ee_pos[tid] = hold_pos[tid]
        des_gripper_state[tid] = GripperState.OPEN
        # wait until the gripper action released the cloth
        if is_grasping[tid] == 0 and sm_wait_time[tid] >= DragSmWaitTime.RELEASE_CLOTH:
            # move to next state and reset wait time
            sm_state[tid] = DragSmState.RETREAT
            sm_wait_time[tid] = 0.0
    elif state == DragSmState.RETREAT:
        # final state: hold the gripper away from the cloth until the episode ends
        des_ee_pos[tid] = wp.vec3(RETREAT_X, hold_pos[tid][1], RETREAT_Z)
        des_gripper_state[tid] = GripperState.OPEN
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class DragClothSm:
    """A simple state machine in the robot's base frame to drag a cloth to a target.

    The state machine is implemented as a warp kernel. It takes in the end-effector position, the grasp point on the
    cloth, the cloth's central node, the target and the grasp state, and outputs the desired end-effector position and
    the binary gripper command. The state machine has the following states:

    1. REST: The robot is at rest.
    2. APPROACH_ABOVE_CLOTH: The robot hovers above the grasp point with an open gripper.
    3. APPROACH_CLOTH: The robot descends to the grasp height with an open gripper.
    4. GRASP_CLOTH: The robot closes the gripper until the gripper action locks the cloth, or a timeout.
    5. DRAG_CLOTH: The robot drags the cloth until its central node is within the goal tolerance of the target, or
       until the drag stalls at the arm's reach limit.
    6. HOLD_AT_TARGET: The robot holds still until the task counts the cloth as held at the target, or a timeout.
    7. RELEASE_CLOTH: The robot opens the gripper until the gripper action releases the cloth.
    8. RETREAT: The robot moves away from the cloth. This is the final state.
    """

    def __init__(
        self,
        dt: float,
        num_envs: int,
        device: torch.device | str,
        grasp_height: float,
        drag_height: float,
        drag_speed: float,
        goal_tolerance: float,
        approach_threshold: float = 0.01,
        grasp_threshold: float = 0.005,
    ):
        """Initialize the state machine.

        Args:
            dt: The environment time step [s].
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
            grasp_height: End-effector height for the grasp [m].
            drag_height: End-effector height while dragging [m].
            drag_speed: Speed of the drag [m/s].
            goal_tolerance: Planar distance of the central node to the target at which the drag stops [m].
            approach_threshold: Distance at which the end-effector counts as above the grasp point [m].
            grasp_threshold: Distance at which the end-effector counts as at the grasp point [m].
        """
        # save parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.grasp_height = grasp_height
        self.drag_height = drag_height
        self.drag_speed = drag_speed
        self.goal_tolerance = goal_tolerance
        self.approach_threshold = approach_threshold
        self.grasp_threshold = grasp_threshold
        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)
        # end-effector position held at the target, in the robot base frame [m]
        self.hold_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # desired state
        self.des_ee_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 1.0, device=self.device)

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.hold_pos_wp = wp.from_torch(self.hold_pos, wp.vec3)
        self.des_ee_pos_wp = wp.from_torch(self.des_ee_pos, wp.vec3)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)

    def reset_idx(self, env_ids: torch.Tensor | None = None):
        """Reset the state machine of the given environments."""
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(
        self,
        ee_pos: torch.Tensor,
        grasp_pos: torch.Tensor,
        center_pos: torch.Tensor,
        target_pos: torch.Tensor,
        is_grasping: torch.Tensor,
        holding: torch.Tensor,
        drag_stalled: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the desired end-effector position [m] and the binary gripper command (1 open, -1 close).

        Args:
            ee_pos: End-effector positions in the robot base frame [m], shape [N, 3].
            grasp_pos: Grasp points in the robot base frame [m], shape [N, 3].
            center_pos: Positions of the cloth's central node in the robot base frame [m], shape [N, 3].
            target_pos: Targets of the central node in the robot base frame [m], shape [N, 3].
            is_grasping: Whether the gripper action holds locked cloth nodes, shape [N].
            holding: Whether the task counts the cloth as held at the target, shape [N].
            drag_stalled: Whether the end-effector has stopped moving during the drag, shape [N].

        Returns:
            The desired end-effector positions [m], shape [N, 3], and gripper commands, shape [N].
        """
        # run state machine
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                wp.from_torch(ee_pos.contiguous(), wp.vec3),
                wp.from_torch(grasp_pos.contiguous(), wp.vec3),
                wp.from_torch(center_pos.contiguous(), wp.vec3),
                wp.from_torch(target_pos.contiguous(), wp.vec3),
                wp.from_torch(is_grasping.int().contiguous(), wp.int32),
                wp.from_torch(holding.int().contiguous(), wp.int32),
                wp.from_torch(drag_stalled.int().contiguous(), wp.int32),
                self.hold_pos_wp,
                self.des_ee_pos_wp,
                self.des_gripper_state_wp,
                HOVER_HEIGHT,
                self.grasp_height,
                self.drag_height,
                self.drag_speed,
                self.goal_tolerance,
                self.approach_threshold,
                self.grasp_threshold,
            ],
            device=self.device,
        )
        return self.des_ee_pos, self.des_gripper_state


def wrinkle_grasp_point(nodes_b: torch.Tensor, center_b: torch.Tensor) -> torch.Tensor:
    """Return the crest of the cloth's central wrinkle: its highest node within :data:`WRINKLE_HALF_WIDTH` of the
    central node along y.

    Args:
        nodes_b: Cloth node positions in the robot base frame [m], shape [N, P, 3].
        center_b: Position of the cloth's central node in the robot base frame [m], shape [N, 3].

    Returns:
        The grasp points in the robot base frame [m], shape [N, 3].
    """
    in_wrinkle = (nodes_b[..., 1] - center_b[:, 1:2]).abs() < WRINKLE_HALF_WIDTH
    height = torch.where(in_wrinkle, nodes_b[..., 2], -torch.inf)
    crest_ids = height.argmax(dim=1)
    return nodes_b[torch.arange(len(nodes_b), device=nodes_b.device), crest_ids]


def main():
    # parse configuration via Hydra, so presets can be selected on the CLI
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    args_cli.device = env_cfg.sim.device
    env_cfg.scene.num_envs = args_cli.num_envs
    # Scripted debugging: only the time-out ends an episode, so a physics blow-up plays out on screen
    env_cfg.episode_length_s = args_cli.episode_length
    for term_name in list(vars(env_cfg.terminations)):
        if term_name != "time_out":
            setattr(env_cfg.terminations, term_name, None)
    # the state machine never reads rewards, and reward terms may read the removed terminations (termination penalty,
    # success bonus), so drop them all
    for term_name in list(vars(env_cfg.rewards)):
        setattr(env_cfg.rewards, term_name, None)
    # grasp with the task's own gripper action: lock the grasped nodes, or hold the cloth by friction alone
    env_cfg.actions.gripper_action.attach = args_cli.grasp_mode == "attach"
    if not isinstance(env_cfg.actions.arm_action, TopDownDeltaEEActionCfg):
        raise SystemExit("The state machine commands the delta end-effector action: use the 'ee_delta' actions preset.")
    # Keep only the requested visualizers: the task also configures headless Kit and Newton viewers, and a Kit
    # visualizer in the config starts Kit even when --viz does not select it.
    aliases = {"newton": "newton_gl"}
    requested = {aliases.get(viz, viz) for viz in (args_cli.visualizer or []) if viz != "none"}
    env_cfg.sim.visualizer_cfgs = [
        cfg for cfg in env_cfg.sim.visualizer_cfgs if getattr(cfg, "visualizer_type", None) in requested
    ]

    with launch_simulation(env_cfg, args_cli):
        from isaaclab_tasks.contrib.franka_cloth_drag.mdp.frames import (
            cloth_center_pos_b,
            ee_pose_in_base,
            points_to_base,
            top_down_yaw,
        )

        env = gym.make(args_cli.task, cfg=env_cfg)
        unwrapped = env.unwrapped
        device = unwrapped.device
        num_envs = unwrapped.num_envs

        # reset environment at start
        env.reset()

        command = unwrapped.command_manager.get_term("cloth_target")
        gripper = unwrapped.action_manager.get_term("gripper_action")
        arm_cfg = unwrapped.action_manager.get_term("arm_action").cfg
        cloth, robot = unwrapped.scene["deformable"], unwrapped.scene["robot"]
        finger_joint_id = robot.find_joints("panda_finger_joint1")[0][0]

        # create state machine
        drag_sm = DragClothSm(
            unwrapped.step_dt,
            num_envs,
            device,
            grasp_height=args_cli.grasp_height,
            drag_height=args_cli.drag_height,
            drag_speed=args_cli.drag_speed,
            goal_tolerance=min(args_cli.goal_tolerance, command.cfg.success_radius),
        )
        grasp_pos = torch.zeros((num_envs, 3), device=device)
        # steps in a row the end-effector has barely moved while dragging
        prev_ee_pos = torch.zeros((num_envs, 3), device=device)
        stall_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        stall_limit = round(STALL_TIME / unwrapped.step_dt)
        actions = torch.zeros(unwrapped.action_space.shape, device=device)
        actions[:, 4] = 1.0  # open
        print(f"[INFO]: Dragging the cloth with a {args_cli.grasp_mode} grasp in {num_envs} env(s).")

        step = 0
        while args_cli.num_steps is None or step < args_cli.num_steps:
            # step environment
            _, _, terminated, time_outs, _ = env.step(actions)
            step += 1
            dones = terminated | time_outs
            # reset the state machine of reset environments
            if dones.any():
                done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                drag_sm.reset_idx(done_ids)
                print(f"[INFO]: Episode ended in env(s) {done_ids.tolist()}; restarting the state machine.")

            # observations in the robot base frame
            ee_pos, ee_quat = ee_pose_in_base(unwrapped)
            center_pos = cloth_center_pos_b(unwrapped)
            # -- grasp point on the central wrinkle, tracked while resting and held once the approach starts
            resting = drag_sm.sm_state == DragSmState.REST
            if resting.any():
                nodes_b = points_to_base(unwrapped, cloth.data.nodal_pos_w.torch)
                grasp_pos[resting] = wrinkle_grasp_point(nodes_b, center_pos)[resting]

            # -- drag stall: the arm stopped, e.g. at its reach limit next to the robot base
            still = (ee_pos - prev_ee_pos).norm(dim=-1) < STALL_STEP
            dragging = drag_sm.sm_state == DragSmState.DRAG_CLOTH
            stall_steps = torch.where(dragging & still, stall_steps + 1, 0)
            prev_ee_pos = ee_pos.clone()

            # advance state machine
            previous_state = drag_sm.sm_state.clone()
            des_ee_pos, des_gripper_state = drag_sm.compute(
                ee_pos,
                grasp_pos,
                center_pos,
                command.command,
                gripper.is_grasping,
                command.holding,
                stall_steps >= stall_limit,
            )

            # convert the desired end-effector position to the delta end-effector action, keeping zero yaw
            actions[:, :3] = ((des_ee_pos - ee_pos) / arm_cfg.scale_trans).clamp(-1.0, 1.0)
            yaw = top_down_yaw(ee_quat)
            actions[:, 3] = (torch.atan2(torch.sin(-yaw), torch.cos(-yaw)) / arm_cfg.scale_rot).clamp(-1.0, 1.0)
            actions[:, 4] = des_gripper_state

            # report the grasp and the release
            grasped = (previous_state == DragSmState.GRASP_CLOTH) & (drag_sm.sm_state == DragSmState.DRAG_CLOTH)
            for env_id in grasped.nonzero(as_tuple=False).squeeze(-1).tolist():
                opening_mm = robot.data.joint_pos.torch[env_id, finger_joint_id].item() * 1e3
                if gripper.is_grasping[env_id]:
                    grasp_info = f"locked {gripper.num_grasped[env_id].item()} cloth nodes"
                else:
                    grasp_info = "locked nothing (empty grasp)" if args_cli.grasp_mode == "attach" else "friction grasp"
                print(f"[INFO]: env {env_id}: closed at a finger opening of {opening_mm:.1f} mm, {grasp_info}.")
            held = (previous_state == DragSmState.DRAG_CLOTH) & (drag_sm.sm_state == DragSmState.HOLD_AT_TARGET)
            for env_id in held.nonzero(as_tuple=False).squeeze(-1).tolist():
                distance_cm = (command.command[env_id, :2] - center_pos[env_id, :2]).norm().item() * 1e2
                reason = "drag stalled (reach limit)" if stall_steps[env_id] >= stall_limit else "reached the target"
                print(f"[INFO]: env {env_id}: {reason}, central node {distance_cm:.1f} cm from the target; holding.")
            released = (previous_state == DragSmState.HOLD_AT_TARGET) & (drag_sm.sm_state == DragSmState.RELEASE_CLOTH)
            for env_id in released.nonzero(as_tuple=False).squeeze(-1).tolist():
                distance_cm = (command.command[env_id, :2] - center_pos[env_id, :2]).norm().item() * 1e2
                outcome = "held at the target (success)" if command.holding[env_id] else "hold timed out"
                print(f"[INFO]: env {env_id}: releasing the cloth, {outcome}, {distance_cm:.1f} cm from the target.")
            changed = previous_state != drag_sm.sm_state
            if num_envs == 1 and changed.any():
                print(f"[INFO]: env 0: {STATE_NAMES[int(drag_sm.sm_state[0])]}")

        # close the environment
        env.close()


if __name__ == "__main__":
    main()

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to drag a cloth over the table to a target with a robotic arm, to watch and debug the cloth grasp.

The state machine grasps the ``IsaacContrib-Drag-Cloth-Franka`` cloth at the raised node closest to the segment from
the cloth's central node to the target (without raised nodes, at the node on that segment a few centimeters from the
central node). It hovers above the node with a fully open gripper, the hand's long side tangent around the cloth
center so its narrow side faces the center, lowers the open fingers onto the cloth and squeezes, so the cloth creases
up between the jaws. It then drags the task's goal point (the cube on the cloth, or the central node without one) to
the target, stops until the task counts the cloth as held there (the success needs the cloth held by a raised node),
releases the cloth and rises. It is
implemented in the kernel function `infer_state_machine`, which uses the `warp` library to run all environments in
parallel on the GPU.

The state machine commands the task's own action terms, the 4-DoF delta end-effector action and the cloth grasp
gripper, so the grasp behaves exactly as in training: once the closing fingers rest, the cloth nodes between the pads
are locked to the end-effector frame (``--grasp_mode attach``). ``--grasp_mode friction`` holds the cloth by friction
alone. Its transitions follow the task's stages: the hover pose lies within the task's approach radius, and the drag
starts once the task counts the cloth as grasped by a raised node. Episodes end and reset as in training (task
episode length, terminations and reset events), except that success does not end the episode, so the release and the
rise play out; success is reported per episode from the task's success check. The state machine restarts in every
reset environment.

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
parser.add_argument(
    "--episode_length", type=float, default=None, help="Episode length [s]. Defaults to the task's episode length."
)
parser.add_argument(
    "--grasp_height",
    type=float,
    default=0.0095,
    help=(
        "End-effector height commanded for the squeeze [m], at least the action's floor. The hand comes to rest with"
        " the pads on the cloth at about 12.3 mm, so 9.5 mm presses the open pads onto the cloth."
    ),
)
parser.add_argument(
    "--drag_height",
    type=float,
    default=0.02,
    help="End-effector height while dragging [m], at most the task's success height.",
)
parser.add_argument(
    "--goal_tolerance",
    type=float,
    default=0.02,
    help="Planar distance of the goal point to the target at which the drag stops [m], at most the success radius.",
)
add_launcher_args(parser)
# the task runs on Newton, so default to the Viser web viewer and let the task pick the device
parser.set_defaults(device=None, visualizer=["viser"])
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

# initialize warp
wp.init()

# the hover pose lies this far inside the task's approach radius above the grasp node [m]
HOVER_MARGIN = 0.01
# without raised nodes, grasp the node this far from the central node towards the target [m]
FALLBACK_GRASP_OFFSET = 0.06
# the hand counts as aligned for the descent within this yaw error [rad]
YAW_TOLERANCE = wp.constant(0.1)
# yaw action scale of the state machine [rad]: the wrist turns at about kp/kd times the commanded yaw step, so the
# task's scale (0.2 rad, ~0.67 rad/s) makes the head alignment slow; the action still caps the joint speed
YAW_ACTION_SCALE = 0.5
# while dragging, the hand turns back towards the central node only beyond this misalignment [rad], at this rate
# [rad/s], so the held cloth is not twisted
YAW_DRAG_TOLERANCE = wp.constant(0.2)
YAW_DRAG_RATE = wp.constant(0.5)
# farthest the drag target may lead the lagging end-effector [m]
DRAG_MAX_LEAD = wp.constant(0.03)
# the drag counts as stalled once the end-effector moved less than this per step for STALL_TIME [m], [s]
STALL_STEP = 0.0001
STALL_TIME = 0.5
# rise of the end-effector after the release [m]
RISE_HEIGHT = wp.constant(0.05)
# grasp attempts before the state machine gives up
MAX_GRASP_TRIES = wp.constant(2)


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
    # time the hover and the touch pose must be held
    APPROACH_ABOVE_CLOTH = wp.constant(0.3)
    APPROACH_CLOTH = wp.constant(0.3)
    # time for the squeeze to ramp in after the grasp locked the cloth
    GRASP_SQUEEZE = wp.constant(0.3)
    # time after which a grasp that locked nothing is retried (or a friction grasp moves on)
    GRASP_TIMEOUT = wp.constant(2.0)
    # longest the drag stops at the target waiting for the task's success before the release; the success needs the
    # cloth held by a raised node for the hold time
    HOLD_TIMEOUT = wp.constant(2.0)
    RELEASE_CLOTH = wp.constant(0.5)


# task stages (isaaclab_tasks.contrib.franka_cloth_drag.mdp.stages.ClothDragStage)
STAGE_GRASP = wp.constant(1)
STAGE_DRAG = wp.constant(2)


@wp.func
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    return wp.length(current_pos - desired_pos) < threshold


@wp.func
def wrap_half_turn(angle: float) -> float:
    """Wrap an angle to [-pi/2, pi/2]: the gripper is symmetric under half a turn about the vertical axis."""
    return angle - wp.pi * wp.round(angle / wp.pi)


@wp.func
def heading(from_pos: wp.vec3, to_pos: wp.vec3) -> float:
    """Planar heading from one point to another [rad]."""
    return wp.atan2(to_pos[1] - from_pos[1], to_pos[0] - from_pos[0])


@wp.func
def reachable_yaw_step(des_yaw: float, ee_yaw: float, wrist_pos: float, wrist_lower: float, wrist_upper: float):
    """Yaw change [rad] to the nearer of ``des_yaw`` and ``des_yaw + pi`` that the wrist window allows.

    The delta end-effector action turns the hand by the opposite of the last arm joint's change and clamps that joint
    to the wrist window, so a turn the window would clip goes the other way round.
    """
    step = wrap_half_turn(des_yaw - ee_yaw)
    wrist = wrist_pos - step
    if wrist < wrist_lower:
        step = step - wp.pi
    elif wrist > wrist_upper:
        step = step + wp.pi
    return step


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    num_tries: wp.array(dtype=int),
    ee_pos: wp.array(dtype=wp.vec3),
    ee_yaw: wp.array(dtype=float),
    wrist_pos: wp.array(dtype=float),
    node_pos: wp.array(dtype=wp.vec3),
    center_pos: wp.array(dtype=wp.vec3),
    goal_pos: wp.array(dtype=wp.vec3),
    target_pos: wp.array(dtype=wp.vec3),
    stage: wp.array(dtype=int),
    is_grasping: wp.array(dtype=int),
    holding: wp.array(dtype=int),
    drag_stalled: wp.array(dtype=int),
    grasp_pos: wp.array(dtype=wp.vec3),
    hold_pos: wp.array(dtype=wp.vec3),
    des_ee_pos: wp.array(dtype=wp.vec3),
    des_gripper_state: wp.array(dtype=float),
    des_yaw: wp.array(dtype=float),
    yaw_step: wp.array(dtype=float),
    hover_height: float,
    grasp_height: float,
    drag_height: float,
    drag_speed: float,
    goal_tolerance: float,
    approach_threshold: float,
    grasp_threshold: float,
    wrist_lower: float,
    wrist_upper: float,
    attach: int,
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # planar vector from the goal point (cube or central node) to the target
    to_target = wp.vec3(target_pos[tid][0] - goal_pos[tid][0], target_pos[tid][1] - goal_pos[tid][1], 0.0)
    target_distance = wp.length(to_target)
    # yaw change towards the current heading, before this step's update
    step_now = reachable_yaw_step(des_yaw[tid], ee_yaw[tid], wrist_pos[tid], wrist_lower, wrist_upper)
    # decide next state
    if state == DragSmState.REST:
        des_ee_pos[tid] = ee_pos[tid]
        des_gripper_state[tid] = GripperState.OPEN
        # the hand's x-axis points at the cloth center, so its long side is tangent around it
        des_yaw[tid] = heading(node_pos[tid], center_pos[tid])
        # wait for a while
        if sm_wait_time[tid] >= DragSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = DragSmState.APPROACH_ABOVE_CLOTH
            sm_wait_time[tid] = 0.0
    elif state == DragSmState.APPROACH_ABOVE_CLOTH:
        # follow the grasp node until the descent starts
        des_ee_pos[tid] = node_pos[tid] + wp.vec3(0.0, 0.0, hover_height)
        des_gripper_state[tid] = GripperState.OPEN
        des_yaw[tid] = heading(node_pos[tid], center_pos[tid])
        aligned = wp.abs(step_now) < YAW_TOLERANCE
        if (
            stage[tid] >= STAGE_GRASP
            and aligned
            and distance_below_threshold(ee_pos[tid], des_ee_pos[tid], approach_threshold)
        ):
            # hold the hover pose for a while
            if sm_wait_time[tid] >= DragSmWaitTime.APPROACH_ABOVE_CLOTH:
                # move to next state and reset wait time; the descent keeps the grasp point and the yaw
                grasp_pos[tid] = wp.vec3(node_pos[tid][0], node_pos[tid][1], grasp_height)
                sm_state[tid] = DragSmState.APPROACH_CLOTH
                sm_wait_time[tid] = 0.0
        else:
            sm_wait_time[tid] = 0.0
    elif state == DragSmState.APPROACH_CLOTH:
        # lower the open fingers onto the cloth
        des_ee_pos[tid] = grasp_pos[tid]
        des_gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(ee_pos[tid], des_ee_pos[tid], grasp_threshold):
            # hold the touch pose for a while
            if sm_wait_time[tid] >= DragSmWaitTime.APPROACH_CLOTH:
                # move to next state and reset wait time
                sm_state[tid] = DragSmState.GRASP_CLOTH
                sm_wait_time[tid] = 0.0
        else:
            sm_wait_time[tid] = 0.0
    elif state == DragSmState.GRASP_CLOTH:
        # squeeze: the closing fingers push the cloth together into a crease between the jaws
        des_ee_pos[tid] = grasp_pos[tid]
        des_gripper_state[tid] = GripperState.CLOSE
        # the task counts the cloth as grasped once the gripper action locked it
        if stage[tid] == STAGE_DRAG and sm_wait_time[tid] >= DragSmWaitTime.GRASP_SQUEEZE:
            # move to next state and reset wait time
            sm_state[tid] = DragSmState.DRAG_CLOTH
            sm_wait_time[tid] = 0.0
        elif sm_wait_time[tid] >= DragSmWaitTime.GRASP_TIMEOUT:
            if attach == 0:
                # a friction grasp never locks the cloth: drag what the fingers hold
                sm_state[tid] = DragSmState.DRAG_CLOTH
            else:
                num_tries[tid] = num_tries[tid] + 1
                if num_tries[tid] < MAX_GRASP_TRIES:
                    # open and hover again for another try
                    sm_state[tid] = DragSmState.APPROACH_ABOVE_CLOTH
                else:
                    # give up: open and rise where the hand is
                    hold_pos[tid] = ee_pos[tid]
                    sm_state[tid] = DragSmState.RETREAT
            sm_wait_time[tid] = 0.0
    elif state == DragSmState.DRAG_CLOTH:
        # advance the drag target the way the goal point has to go, at the drag speed, leading the lagging arm by at
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
        # keep the grasp yaw, and turn back towards the central node only once the cloth turned too far under the hand
        misalignment = wrap_half_turn(heading(ee_pos[tid], center_pos[tid]) - des_yaw[tid])
        if wp.abs(misalignment) > YAW_DRAG_TOLERANCE:
            max_turn = YAW_DRAG_RATE * dt[tid]
            des_yaw[tid] = des_yaw[tid] + wp.clamp(misalignment, -max_turn, max_turn)
        # a drag that stalls at the arm's reach limit stops where it is and lets the task judge the result
        if target_distance < goal_tolerance or drag_stalled[tid] == 1:
            # move to next state and reset wait time
            hold_pos[tid] = wp.vec3(ee_pos[tid][0], ee_pos[tid][1], drag_height)
            sm_state[tid] = DragSmState.HOLD_AT_TARGET
            sm_wait_time[tid] = 0.0
    elif state == DragSmState.HOLD_AT_TARGET:
        # stop at the target until the task counts the cloth as held there (or a timeout)
        des_ee_pos[tid] = hold_pos[tid]
        des_gripper_state[tid] = GripperState.CLOSE
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
        # final state: rise above the release pose until the episode ends
        des_ee_pos[tid] = hold_pos[tid] + wp.vec3(0.0, 0.0, RISE_HEIGHT)
        des_gripper_state[tid] = GripperState.OPEN
    # turn the hand towards the desired yaw, the way the wrist window allows
    yaw_step[tid] = reachable_yaw_step(des_yaw[tid], ee_yaw[tid], wrist_pos[tid], wrist_lower, wrist_upper)
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class DragClothSm:
    """A simple state machine in the robot's base frame to drag a cloth to a target.

    The state machine is implemented as a warp kernel. It takes in the end-effector pose, the wrist joint position,
    the grasp node, the cloth's central node, the target, the task stage and the grasp state, and outputs the desired
    end-effector position, the yaw change and the binary gripper command. The hand's x-axis points at the cloth
    center from the grasp node, so the hand's long side, along which the fingers close, is tangent around the center.
    The state machine has the following states:

    1. REST: The robot is at rest.
    2. APPROACH_ABOVE_CLOTH: The robot hovers above the grasp node with an open gripper, inside the task's approach
       radius, and turns the hand, until the task counts the hand as near the cloth.
    3. APPROACH_CLOTH: The robot lowers the open fingers onto the cloth, keeping the yaw.
    4. GRASP_CLOTH: The robot squeezes until the task counts the cloth as grasped. A grasp that locked nothing is
       retried from the hover pose once, then the robot gives up and rises; a friction grasp moves on after a timeout.
    5. DRAG_CLOTH: The robot drags the cloth until its central node is within the goal tolerance of the target, or
       until the drag stalls at the arm's reach limit. The yaw changes only if the cloth turns too far under the hand.
    6. HOLD_AT_TARGET: The robot stops until the task counts the cloth as held at the target, or a timeout.
    7. RELEASE_CLOTH: The robot opens the gripper until the gripper action releases the cloth.
    8. RETREAT: The robot rises above the release pose. This is the final state.
    """

    def __init__(
        self,
        dt: float,
        num_envs: int,
        device: torch.device | str,
        hover_height: float,
        grasp_height: float,
        drag_height: float,
        drag_speed: float,
        goal_tolerance: float,
        wrist_limits: tuple[float, float],
        attach: bool,
        approach_threshold: float = 0.01,
        grasp_threshold: float = 0.005,
    ):
        """Initialize the state machine.

        Args:
            dt: The environment time step [s].
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
            hover_height: Height of the hover pose above the grasp node [m].
            grasp_height: End-effector height with the open fingertips resting on the cloth [m].
            drag_height: End-effector height while dragging [m].
            drag_speed: Speed of the drag [m/s].
            goal_tolerance: Planar distance of the central node to the target at which the drag stops [m].
            wrist_limits: Range of the last arm joint that the yaw action may command [rad].
            attach: Whether the gripper locks the grasped cloth nodes, so an empty grasp can be detected and retried.
            approach_threshold: Distance at which the end-effector counts as at the hover pose [m].
            grasp_threshold: Distance at which the end-effector counts as at the touch pose [m].
        """
        # save parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.hover_height = hover_height
        self.grasp_height = grasp_height
        self.drag_height = drag_height
        self.drag_speed = drag_speed
        self.goal_tolerance = goal_tolerance
        self.wrist_limits = wrist_limits
        self.attach = attach
        self.approach_threshold = approach_threshold
        self.grasp_threshold = grasp_threshold
        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)
        self.num_tries = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)
        # grasp point of the descent and end-effector position at the target, in the robot base frame [m]
        self.grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.hold_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # desired state
        self.des_ee_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 1.0, device=self.device)
        self.des_yaw = torch.zeros((self.num_envs,), device=self.device)
        self.yaw_step = torch.zeros((self.num_envs,), device=self.device)

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.num_tries_wp = wp.from_torch(self.num_tries, wp.int32)
        self.grasp_pos_wp = wp.from_torch(self.grasp_pos, wp.vec3)
        self.hold_pos_wp = wp.from_torch(self.hold_pos, wp.vec3)
        self.des_ee_pos_wp = wp.from_torch(self.des_ee_pos, wp.vec3)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.des_yaw_wp = wp.from_torch(self.des_yaw, wp.float32)
        self.yaw_step_wp = wp.from_torch(self.yaw_step, wp.float32)

    def reset_idx(self, env_ids: torch.Tensor | None = None):
        """Reset the state machine of the given environments."""
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0
        self.num_tries[env_ids] = 0

    def compute(
        self,
        ee_pos: torch.Tensor,
        ee_yaw: torch.Tensor,
        wrist_pos: torch.Tensor,
        node_pos: torch.Tensor,
        center_pos: torch.Tensor,
        goal_pos: torch.Tensor,
        target_pos: torch.Tensor,
        stage: torch.Tensor,
        is_grasping: torch.Tensor,
        holding: torch.Tensor,
        drag_stalled: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the desired end-effector position [m], the yaw change [rad] and the binary gripper command.

        Args:
            ee_pos: End-effector positions in the robot base frame [m], shape [N, 3].
            ee_yaw: Heading of the hand's x-axis in the robot base frame [rad], shape [N].
            wrist_pos: Positions of the last arm joint [rad], shape [N].
            node_pos: Positions of the grasp node in the robot base frame [m], shape [N, 3].
            center_pos: Positions of the cloth's central node in the robot base frame [m], shape [N, 3].
            goal_pos: Positions of the point that must reach the target (the task's goal point) in the robot base
                frame [m], shape [N, 3].
            target_pos: Targets of the goal point in the robot base frame [m], shape [N, 3].
            stage: Task stage index (:class:`ClothDragStage`), shape [N].
            is_grasping: Whether the gripper action holds locked cloth nodes, shape [N].
            holding: Whether the task counts the cloth as held at the target (its success), shape [N].
            drag_stalled: Whether the end-effector has stopped moving during the drag, shape [N].

        Returns:
            The desired end-effector positions [m], shape [N, 3], yaw changes [rad], shape [N], and gripper commands
            (1 open, -1 close), shape [N].
        """
        # run state machine
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                self.num_tries_wp,
                wp.from_torch(ee_pos.contiguous(), wp.vec3),
                wp.from_torch(ee_yaw.float().contiguous(), wp.float32),
                wp.from_torch(wrist_pos.float().contiguous(), wp.float32),
                wp.from_torch(node_pos.contiguous(), wp.vec3),
                wp.from_torch(center_pos.contiguous(), wp.vec3),
                wp.from_torch(goal_pos.contiguous(), wp.vec3),
                wp.from_torch(target_pos.contiguous(), wp.vec3),
                wp.from_torch(stage.int().contiguous(), wp.int32),
                wp.from_torch(is_grasping.int().contiguous(), wp.int32),
                wp.from_torch(holding.int().contiguous(), wp.int32),
                wp.from_torch(drag_stalled.int().contiguous(), wp.int32),
                self.grasp_pos_wp,
                self.hold_pos_wp,
                self.des_ee_pos_wp,
                self.des_gripper_state_wp,
                self.des_yaw_wp,
                self.yaw_step_wp,
                self.hover_height,
                self.grasp_height,
                self.drag_height,
                self.drag_speed,
                self.goal_tolerance,
                self.approach_threshold,
                self.grasp_threshold,
                self.wrist_limits[0],
                self.wrist_limits[1],
                int(self.attach),
            ],
            device=self.device,
        )
        return self.des_ee_pos, self.yaw_step, self.des_gripper_state


def grasp_node_ids(
    nodes_b: torch.Tensor, center_b: torch.Tensor, target_b: torch.Tensor, candidate_ids: torch.Tensor | None
) -> torch.Tensor:
    """Return the cloth node to grasp: the candidate closest to the segment from the central node to the target.

    Without candidates (no raised nodes), the node closest to the point :data:`FALLBACK_GRASP_OFFSET` from the central
    node towards the target; the squeeze creases the flat cloth there.

    Args:
        nodes_b: Cloth node positions in the robot base frame [m], shape [N, P, 3].
        center_b: Position of the cloth's central node in the robot base frame [m], shape [N, 3].
        target_b: Target of the central node in the robot base frame [m], shape [N, 3].
        candidate_ids: Ids of the candidate nodes, shape [N, K], or None.

    Returns:
        The node ids, shape [N].
    """
    rows = torch.arange(len(nodes_b), device=nodes_b.device)
    to_target = (target_b - center_b)[:, :2]
    if candidate_ids is None or candidate_ids.shape[1] == 0:
        point = center_b[:, :2] + FALLBACK_GRASP_OFFSET * torch.nn.functional.normalize(to_target, dim=-1)
        return (nodes_b[..., :2] - point.unsqueeze(1)).norm(dim=-1).argmin(dim=1)
    rel = nodes_b[rows.unsqueeze(1), candidate_ids, :2] - center_b[:, None, :2]
    t = (rel * to_target.unsqueeze(1)).sum(dim=-1) / to_target.square().sum(dim=-1, keepdim=True).clamp(min=1e-12)
    distance = (rel - t.clamp(0.0, 1.0).unsqueeze(-1) * to_target.unsqueeze(1)).norm(dim=-1)
    return candidate_ids[rows, distance.argmin(dim=1)]


def pad_cloth_gaps(
    ee_pos_w: torch.Tensor,
    ee_quat_w: torch.Tensor,
    nodes_w: torch.Tensor,
    opening: torch.Tensor,
    pad_half_width: float,
    pad_thickness: float,
    pad_half_length: float,
    layer_half_thickness: float,
) -> torch.Tensor:
    """Gap between the lower end of each finger pad and the cloth surface below it [m], shape [N, 2].

    The gap is measured along the EE frame's z-axis, which points from the hand to the finger ends, to the highest
    cloth surface (node minus the layer half thickness) in the pad's footprint. Negative gaps press into the cloth.
    A pad without a node in its footprint has a NaN gap.

    Args:
        ee_pos_w: EE frame positions in the world frame [m], shape [N, 3].
        ee_quat_w: EE frame orientations (x, y, z, w) in the world frame, shape [N, 4].
        nodes_w: Cloth node positions in the world frame [m], shape [N, P, 3].
        opening: Distance of each pad's inner face from the EE frame origin [m], shape [N].
        pad_half_width: Half width of the pads along the EE frame's x-axis [m].
        pad_thickness: Thickness of the pads along the EE frame's y-axis [m].
        pad_half_length: Half length of the pads along the EE frame's z-axis [m].
        layer_half_thickness: Half thickness of the cloth layer [m].
    """
    from isaaclab.utils.math import quat_apply_inverse

    quat = ee_quat_w.unsqueeze(1).expand(-1, nodes_w.shape[1], -1)
    rel = quat_apply_inverse(quat, nodes_w - ee_pos_w.unsqueeze(1))
    surface_gap = rel[..., 2] - layer_half_thickness - pad_half_length
    in_width = rel[..., 0].abs() <= pad_half_width
    inner = opening.unsqueeze(1)
    gaps = []
    for side in (1.0, -1.0):
        lateral = side * rel[..., 1]
        under = in_width & (lateral >= inner) & (lateral <= inner + pad_thickness)
        gaps.append(torch.where(under, surface_gap, torch.inf).amin(dim=1))
    gaps = torch.stack(gaps, dim=1)
    return torch.where(torch.isinf(gaps), torch.nan, gaps)


def main():
    # parse configuration via Hydra, so presets can be selected on the CLI
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    args_cli.device = env_cfg.sim.device
    env_cfg.scene.num_envs = args_cli.num_envs
    # episodes end and reset as in training (task episode length, terminations and reset events), except that success
    # does not end them, so the release and the rise play out; the success bonus reads the success termination
    if args_cli.episode_length is not None:
        env_cfg.episode_length_s = args_cli.episode_length
    env_cfg.terminations.success = None
    env_cfg.rewards.success = None
    # grasp with the task's own gripper action: lock the grasped nodes, or hold the cloth by friction alone
    env_cfg.actions.gripper_action.attach = args_cli.grasp_mode == "attach"
    if not isinstance(env_cfg.actions.arm_action, TopDownDeltaEEActionCfg):
        raise SystemExit("The state machine commands the delta end-effector action: use the 'ee_delta' actions preset.")
    # turn the hand faster than the task's yaw scale allows
    env_cfg.actions.arm_action.scale_rot = YAW_ACTION_SCALE
    # Keep only the requested visualizers: the task also configures headless Kit and Newton viewers, and the default
    # --viz of this script is set with set_defaults, not on the command line, so it does not count as an explicit
    # choice.
    aliases = {"newton": "newton_gl"}
    requested = {aliases.get(viz, viz) for viz in (args_cli.visualizer or []) if viz != "none"}
    env_cfg.sim.visualizer_cfgs = [
        cfg for cfg in env_cfg.sim.visualizer_cfgs if getattr(cfg, "visualizer_type", None) in requested
    ]

    with launch_simulation(env_cfg, args_cli):
        from isaaclab_tasks.contrib.franka_cloth_drag.mdp.actions import PAD_HALF_WIDTH, PAD_THICKNESS
        from isaaclab_tasks.contrib.franka_cloth_drag.mdp.frames import (
            cloth_center_pos_b,
            ee_pose_in_base,
            points_to_base,
            top_down_yaw,
        )
        from isaaclab_tasks.contrib.franka_cloth_drag.mdp.stages import ClothDragStage, goal_point_pos_b

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
        wrist_joint_id = robot.find_joints("panda_joint7")[0][0]
        # the raised nodes of the cloth reset, redrawn at every reset; a reset without them grasps the flat cloth
        try:
            reset_term = unwrapped.event_manager.get_term_cfg("reset_deformable").func
        except ValueError:
            reset_term = None
        raised_node_ids = getattr(reset_term, "raised_node_ids", None)
        grasp_info = "raised nodes" if raised_node_ids is not None and raised_node_ids.shape[1] > 0 else "flat cloth"

        # heights from the task: hover inside the approach radius, touch above the action's floor, drag below the
        # success height
        grasp_height = max(args_cli.grasp_height, arm_cfg.z_min)
        drag_height = args_cli.drag_height
        if command.cfg.max_ee_height is not None and drag_height > command.cfg.max_ee_height:
            print(
                f"[WARN]: Drag height {drag_height} m exceeds the success height; using {command.cfg.max_ee_height} m."
            )
            drag_height = command.cfg.max_ee_height

        # create state machine
        drag_sm = DragClothSm(
            unwrapped.step_dt,
            num_envs,
            device,
            hover_height=command.cfg.approach_radius - HOVER_MARGIN,
            grasp_height=grasp_height,
            drag_height=drag_height,
            drag_speed=args_cli.drag_speed,
            goal_tolerance=min(args_cli.goal_tolerance, command.cfg.success_radius),
            wrist_limits=arm_cfg.wrist_joint_limits,
            attach=args_cli.grasp_mode == "attach",
        )
        grasp_node = torch.zeros(num_envs, dtype=torch.long, device=device)
        rows = torch.arange(num_envs, device=device)
        # steps in a row the end-effector has barely moved while dragging
        prev_ee_pos = torch.zeros((num_envs, 3), device=device)
        stall_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        stall_limit = round(STALL_TIME / unwrapped.step_dt)
        # episode outcome: whether the task counted the cloth as held at the target, and the last distance [m]
        succeeded = torch.zeros(num_envs, dtype=torch.bool, device=device)
        last_distance = torch.zeros(num_envs, device=device)
        # peak cloth node speed of the episode [m/s], to spot a physics blow-up that stays finite
        peak_node_speed = torch.zeros(num_envs, device=device)
        layer_half_thickness = getattr(cloth.cfg.spawn.physics_material, "particle_radius", 0.002)
        ee_frame = unwrapped.scene["ee_frame"]
        # time in the hover state, and when its position, yaw and stage conditions first held in this visit [s]
        hover_time = torch.zeros(num_envs, device=device)
        hover_ready = torch.full((num_envs, 3), -1.0, device=device)
        actions = torch.zeros(unwrapped.action_space.shape, device=device)
        actions[:, 4] = 1.0  # open
        print(
            f"[INFO]: Dragging the cloth with a {args_cli.grasp_mode} grasp on the {grasp_info} in {num_envs} env(s)."
        )

        step = 0
        while args_cli.num_steps is None or step < args_cli.num_steps:
            # step environment
            _, _, terminated, time_outs, _ = env.step(actions)
            step += 1
            dones = terminated | time_outs
            # report the episode and reset the state machine of reset environments
            if dones.any():
                done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                for env_id in done_ids.tolist():
                    outcome = "success" if succeeded[env_id] else "no success"
                    fired = [
                        name
                        for name in unwrapped.termination_manager.active_terms
                        if unwrapped.termination_manager.get_term(name)[env_id]
                    ]
                    print(
                        f"[INFO]: env {env_id}: episode ended ({', '.join(fired)}), {outcome}, goal point"
                        f" {last_distance[env_id].item() * 1e2:.1f} cm from the target, peak cloth node speed"
                        f" {peak_node_speed[env_id].item():.2f} m/s; restarting the state machine."
                    )
                drag_sm.reset_idx(done_ids)
                succeeded[done_ids] = False
                peak_node_speed[done_ids] = 0.0
                hover_time[done_ids] = 0.0
                hover_ready[done_ids] = -1.0

            # observations in the robot base frame
            ee_pos, ee_quat = ee_pose_in_base(unwrapped)
            center_pos = cloth_center_pos_b(unwrapped)
            goal_pos = goal_point_pos_b(unwrapped, command.cfg)
            nodes_b = points_to_base(unwrapped, cloth.data.nodal_pos_w.torch)
            target_pos = command.command
            distance = (target_pos[:, :2] - goal_pos[:, :2]).norm(dim=-1)
            succeeded |= command.holding & ~dones
            last_distance = torch.where(dones, last_distance, distance)
            node_speed = torch.nan_to_num(cloth.data.nodal_vel_w.torch.norm(dim=-1), nan=torch.inf).amax(dim=1)
            peak_node_speed = torch.where(dones, peak_node_speed, torch.maximum(peak_node_speed, node_speed))
            # -- grasp node, chosen while resting
            resting = drag_sm.sm_state == DragSmState.REST
            if resting.any():
                grasp_node[resting] = grasp_node_ids(nodes_b, center_pos, target_pos, raised_node_ids)[resting]

            # -- drag stall: the arm stopped, e.g. at its reach limit next to the robot base
            still = (ee_pos - prev_ee_pos).norm(dim=-1) < STALL_STEP
            dragging = drag_sm.sm_state == DragSmState.DRAG_CLOTH
            stall_steps = torch.where(dragging & still, stall_steps + 1, 0)
            prev_ee_pos = ee_pos.clone()

            # advance state machine
            previous_state = drag_sm.sm_state.clone()
            des_ee_pos, yaw_step, des_gripper_state = drag_sm.compute(
                ee_pos,
                top_down_yaw(ee_quat),
                robot.data.joint_pos.torch[:, wrist_joint_id],
                nodes_b[rows, grasp_node],
                center_pos,
                goal_pos,
                target_pos,
                command.stage,
                gripper.is_grasping,
                command.holding,
                stall_steps >= stall_limit,
            )

            # convert the desired end-effector position and yaw change to the delta end-effector action
            actions[:, :3] = ((des_ee_pos - ee_pos) / arm_cfg.scale_trans).clamp(-1.0, 1.0)
            actions[:, 3] = (yaw_step / arm_cfg.scale_rot).clamp(-1.0, 1.0)
            actions[:, 4] = des_gripper_state

            # report the touch, the grasp, the retries, the stop and the release
            new_state = drag_sm.sm_state
            hovering = previous_state == DragSmState.APPROACH_ABOVE_CLOTH
            ready = torch.stack(
                [
                    (ee_pos - drag_sm.des_ee_pos).norm(dim=-1) < drag_sm.approach_threshold,
                    yaw_step.abs() < float(YAW_TOLERANCE),
                    command.stage >= int(ClothDragStage.GRASP),
                ],
                dim=1,
            )
            first = hovering.unsqueeze(1) & ready & (hover_ready < 0.0)
            hover_ready = torch.where(first, hover_time.unsqueeze(1), hover_ready)
            lowering = hovering & (new_state == DragSmState.APPROACH_CLOTH)
            for env_id in lowering.nonzero(as_tuple=False).squeeze(-1).tolist():
                pos_s, yaw_s, stage_s = hover_ready[env_id].tolist()
                print(
                    f"[INFO]: env {env_id}: lowering after {hover_time[env_id].item():.2f} s in the hover state"
                    f" (position {pos_s:.2f} s, yaw {yaw_s:.2f} s, stage {stage_s:.2f} s)."
                )
            hover_time = torch.where(
                hovering & (new_state == DragSmState.APPROACH_ABOVE_CLOTH), hover_time + unwrapped.step_dt, 0.0
            )
            hover_ready = torch.where((new_state == DragSmState.APPROACH_ABOVE_CLOTH).unsqueeze(1), hover_ready, -1.0)
            touched = (previous_state == DragSmState.APPROACH_CLOTH) & (new_state == DragSmState.GRASP_CLOTH)
            if touched.any():
                gaps = pad_cloth_gaps(
                    ee_frame.data.target_pos_w.torch[:, 0],
                    ee_frame.data.target_quat_w.torch[:, 0],
                    cloth.data.nodal_pos_w.torch,
                    robot.data.joint_pos.torch[:, finger_joint_id],
                    PAD_HALF_WIDTH,
                    PAD_THICKNESS,
                    command.cfg.finger_pad_half_length,
                    layer_half_thickness,
                )
                for env_id in touched.nonzero(as_tuple=False).squeeze(-1).tolist():
                    gap_mm = ", ".join(f"{gap * 1e3:.1f}" for gap in gaps[env_id].tolist())
                    print(
                        f"[INFO]: env {env_id}: squeezing at an EE height of {ee_pos[env_id, 2].item() * 1e3:.1f} mm,"
                        f" pad-cloth gaps {gap_mm} mm."
                    )
            grasped = (previous_state == DragSmState.GRASP_CLOTH) & (new_state == DragSmState.DRAG_CLOTH)
            for env_id in grasped.nonzero(as_tuple=False).squeeze(-1).tolist():
                opening_mm = robot.data.joint_pos.torch[env_id, finger_joint_id].item() * 1e3
                if gripper.is_grasping[env_id]:
                    info = f"locked {gripper.num_grasped[env_id].item()} cloth nodes"
                else:
                    info = "friction grasp"
                print(
                    f"[INFO]: env {env_id}: grasped node {grasp_node[env_id].item()} at a finger opening of"
                    f" {opening_mm:.1f} mm, {info}."
                )
            retried = (previous_state == DragSmState.GRASP_CLOTH) & (new_state == DragSmState.APPROACH_ABOVE_CLOTH)
            for env_id in retried.nonzero(as_tuple=False).squeeze(-1).tolist():
                print(f"[INFO]: env {env_id}: the grasp locked nothing; retrying from the hover pose.")
            gave_up = (previous_state == DragSmState.GRASP_CLOTH) & (new_state == DragSmState.RETREAT)
            for env_id in gave_up.nonzero(as_tuple=False).squeeze(-1).tolist():
                print(f"[INFO]: env {env_id}: the grasp locked nothing {MAX_GRASP_TRIES} times; giving up.")
            stopped = (previous_state == DragSmState.DRAG_CLOTH) & (new_state == DragSmState.HOLD_AT_TARGET)
            for env_id in stopped.nonzero(as_tuple=False).squeeze(-1).tolist():
                reason = "drag stalled (reach limit)" if stall_steps[env_id] >= stall_limit else "reached the target"
                print(f"[INFO]: env {env_id}: {reason}, central node {distance[env_id].item() * 1e2:.1f} cm away.")
            released = (previous_state == DragSmState.HOLD_AT_TARGET) & (new_state == DragSmState.RELEASE_CLOTH)
            for env_id in released.nonzero(as_tuple=False).squeeze(-1).tolist():
                print(
                    f"[INFO]: env {env_id}: releasing the cloth, central node"
                    f" {distance[env_id].item() * 1e2:.1f} cm from the target."
                )
            changed = previous_state != new_state
            if num_envs == 1 and changed.any():
                stage_name = ClothDragStage(int(command.stage[0])).name
                print(f"[INFO]: env 0: {STATE_NAMES[int(new_state[0])]} (task stage {stage_name})")

        # close the environment
        env.close()


if __name__ == "__main__":
    main()

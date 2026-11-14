# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted state machine that drags the cloth of ``IsaacContrib-Drag-Cloth-Franka`` to its target.

The state machine grasps the cloth at the task's grasp node (:attr:`~..mdp.commands.ClothTargetCommand.grasp_node_id`):
the raised node closest to the segment from the cloth's central node to the target. It hovers above the node with a
fully open gripper, the hand's long side tangent around the cloth center so its narrow side faces the center, lowers the
open fingers onto the cloth and squeezes, so the cloth creases up between the jaws. It then drags the task's goal point
(the cube on the cloth, or the central node without one) to the target, stops until the task counts the cloth as held
there, releases the cloth and rises. It is implemented in the kernel function :func:`infer_state_machine`, which uses
the ``warp`` library to run all environments in parallel on the GPU.

Drag speed, drag height and an offset of the grasp yaw are set per environment, so the state machine can be randomized
for the reset states (:mod:`.collect`). With ``pregrasp_hold``, it first holds a per-environment pre-grasp pose (an
offset from the grasp node and a yaw error, gripper open) before the approach; the grasp itself is unchanged. The
gripper is the task's locking grasp action, so an empty grasp is detected and retried.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

# the hover pose lies this far inside the task's approach radius above the grasp node [m]
HOVER_MARGIN = 0.01
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
    PREGRASP_HOLD = wp.constant(8)


STATE_NAMES = [
    "REST",
    "APPROACH_ABOVE_CLOTH",
    "APPROACH_CLOTH",
    "GRASP_CLOTH",
    "DRAG_CLOTH",
    "HOLD_AT_TARGET",
    "RELEASE_CLOTH",
    "RETREAT",
    "PREGRASP_HOLD",
]


class DragSmWaitTime:
    """Additional wait times (in s) for states before switching."""

    REST = wp.constant(0.2)
    # time the pre-grasp pose is held, arm still, before the approach (only with pregrasp_hold)
    PREGRASP = wp.constant(0.5)
    # time the hover and the touch pose must be held
    APPROACH_ABOVE_CLOTH = wp.constant(0.3)
    APPROACH_CLOTH = wp.constant(0.3)
    # time for the squeeze to ramp in after the grasp locked the cloth
    GRASP_SQUEEZE = wp.constant(0.3)
    # time after which a grasp that locked nothing is retried
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
    drag_height: wp.array(dtype=float),
    drag_speed: wp.array(dtype=float),
    yaw_offset: wp.array(dtype=float),
    pregrasp_offset: wp.array(dtype=wp.vec3),
    pregrasp_yaw_error: wp.array(dtype=float),
    grasp_pos: wp.array(dtype=wp.vec3),
    hold_pos: wp.array(dtype=wp.vec3),
    des_ee_pos: wp.array(dtype=wp.vec3),
    des_gripper_state: wp.array(dtype=float),
    des_yaw: wp.array(dtype=float),
    yaw_step: wp.array(dtype=float),
    hover_height: float,
    grasp_height: float,
    goal_tolerance: float,
    approach_threshold: float,
    grasp_threshold: float,
    wrist_lower: float,
    wrist_upper: float,
    pregrasp_hold: int,
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
        # the hand's x-axis points at the cloth center (plus the offset), so its long side is tangent around it
        des_yaw[tid] = heading(node_pos[tid], center_pos[tid]) + yaw_offset[tid]
        # wait for a while
        if sm_wait_time[tid] >= DragSmWaitTime.REST:
            # move to next state and reset wait time
            if pregrasp_hold == 1:
                sm_state[tid] = DragSmState.PREGRASP_HOLD
            else:
                sm_state[tid] = DragSmState.APPROACH_ABOVE_CLOTH
            sm_wait_time[tid] = 0.0
    elif state == DragSmState.PREGRASP_HOLD:
        # hold the pre-grasp pose: offset from the grasp node, yaw off the grasp yaw, gripper open
        des_ee_pos[tid] = node_pos[tid] + pregrasp_offset[tid]
        des_gripper_state[tid] = GripperState.OPEN
        des_yaw[tid] = heading(node_pos[tid], center_pos[tid]) + yaw_offset[tid] + pregrasp_yaw_error[tid]
        aligned = wp.abs(step_now) < YAW_TOLERANCE
        if aligned and distance_below_threshold(ee_pos[tid], des_ee_pos[tid], approach_threshold):
            if sm_wait_time[tid] >= DragSmWaitTime.PREGRASP:
                # approach and grasp as without the hold
                sm_state[tid] = DragSmState.APPROACH_ABOVE_CLOTH
                sm_wait_time[tid] = 0.0
        else:
            sm_wait_time[tid] = 0.0
    elif state == DragSmState.APPROACH_ABOVE_CLOTH:
        # follow the grasp node until the descent starts
        des_ee_pos[tid] = node_pos[tid] + wp.vec3(0.0, 0.0, hover_height)
        des_gripper_state[tid] = GripperState.OPEN
        des_yaw[tid] = heading(node_pos[tid], center_pos[tid]) + yaw_offset[tid]
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
        step = wp.min(target_distance, drag_speed[tid] * dt[tid])
        des = des_ee_pos[tid]
        if target_distance > 1.0e-6:
            des = des + to_target * (step / target_distance)
        des = wp.vec3(des[0], des[1], drag_height[tid])
        lead = des - ee_pos[tid]
        if wp.length(lead) > DRAG_MAX_LEAD:
            des = ee_pos[tid] + lead * (DRAG_MAX_LEAD / wp.length(lead))
        des_ee_pos[tid] = des
        des_gripper_state[tid] = GripperState.CLOSE
        # keep the grasp yaw, and turn back towards the central node only once the cloth turned too far under the hand
        misalignment = wrap_half_turn(heading(ee_pos[tid], center_pos[tid]) + yaw_offset[tid] - des_yaw[tid])
        if wp.abs(misalignment) > YAW_DRAG_TOLERANCE:
            max_turn = YAW_DRAG_RATE * dt[tid]
            des_yaw[tid] = des_yaw[tid] + wp.clamp(misalignment, -max_turn, max_turn)
        # a drag that stalls at the arm's reach limit stops where it is and lets the task judge the result
        if target_distance < goal_tolerance or drag_stalled[tid] == 1:
            # move to next state and reset wait time
            hold_pos[tid] = wp.vec3(ee_pos[tid][0], ee_pos[tid][1], drag_height[tid])
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
    center from the grasp node, turned by the per-environment :attr:`yaw_offset`, so the hand's long side, along which
    the fingers close, is tangent around the center. The state machine has the following states:

    1. REST: The robot is at rest.
    1a. PREGRASP_HOLD (only with ``pregrasp_hold``): The robot holds the per-environment pre-grasp pose with an open
       gripper: :attr:`pregrasp_offset` from the grasp node, the yaw :attr:`pregrasp_yaw_error` off the grasp yaw.
    2. APPROACH_ABOVE_CLOTH: The robot hovers above the grasp node with an open gripper, inside the task's approach
       radius, and turns the hand, until the task counts the hand as near the cloth.
    3. APPROACH_CLOTH: The robot lowers the open fingers onto the cloth, keeping the yaw.
    4. GRASP_CLOTH: The robot squeezes until the task counts the cloth as grasped. A grasp that locked nothing is
       retried from the hover pose once, then the robot gives up and rises.
    5. DRAG_CLOTH: The robot drags the cloth at the per-environment :attr:`drag_speed` and :attr:`drag_height` until
       the goal point is within the goal tolerance of the target, or until the drag stalls at the arm's reach limit.
       The yaw changes only if the cloth turns too far under the hand.
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
        approach_threshold: float = 0.01,
        grasp_threshold: float = 0.005,
        pregrasp_hold: bool = False,
    ):
        """Initialize the state machine.

        Args:
            dt: The environment time step [s].
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
            hover_height: Height of the hover pose above the grasp node [m].
            grasp_height: End-effector height with the open fingertips resting on the cloth [m].
            drag_height: End-effector height while dragging [m], the start value of every environment.
            drag_speed: Speed of the drag [m/s], the start value of every environment.
            goal_tolerance: Planar distance of the goal point to the target at which the drag stops [m].
            wrist_limits: Range of the last arm joint that the yaw action may command [rad].
            approach_threshold: Distance at which the end-effector counts as at the hover pose [m].
            grasp_threshold: Distance at which the end-effector counts as at the touch pose [m].
            pregrasp_hold: Whether to hold the pre-grasp pose (:attr:`pregrasp_offset`, :attr:`pregrasp_yaw_error`)
                before the approach.
        """
        # save parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.hover_height = hover_height
        self.grasp_height = grasp_height
        self.goal_tolerance = goal_tolerance
        self.wrist_limits = wrist_limits
        self.approach_threshold = approach_threshold
        self.grasp_threshold = grasp_threshold
        self.pregrasp_hold = pregrasp_hold
        # per-environment parameters
        self.drag_height = torch.full((self.num_envs,), float(drag_height), device=self.device)
        """End-effector height while dragging [m], shape [num_envs]."""
        self.drag_speed = torch.full((self.num_envs,), float(drag_speed), device=self.device)
        """Speed of the drag [m/s], shape [num_envs]."""
        self.yaw_offset = torch.zeros((self.num_envs,), device=self.device)
        """Offset added to the grasp yaw [rad], shape [num_envs]."""
        self.pregrasp_offset = torch.zeros((self.num_envs, 3), device=self.device)
        """Pre-grasp end-effector position relative to the grasp node, robot base frame [m], shape [num_envs, 3]."""
        self.pregrasp_yaw_error = torch.zeros((self.num_envs,), device=self.device)
        """Yaw of the pre-grasp pose relative to the grasp yaw [rad], shape [num_envs]."""
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
        self.drag_height_wp = wp.from_torch(self.drag_height, wp.float32)
        self.drag_speed_wp = wp.from_torch(self.drag_speed, wp.float32)
        self.yaw_offset_wp = wp.from_torch(self.yaw_offset, wp.float32)
        self.pregrasp_offset_wp = wp.from_torch(self.pregrasp_offset, wp.vec3)
        self.pregrasp_yaw_error_wp = wp.from_torch(self.pregrasp_yaw_error, wp.float32)
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

    def set_params(
        self,
        env_ids: torch.Tensor,
        drag_height: torch.Tensor | None = None,
        drag_speed: torch.Tensor | None = None,
        yaw_offset: torch.Tensor | None = None,
        pregrasp_offset: torch.Tensor | None = None,
        pregrasp_yaw_error: torch.Tensor | None = None,
    ):
        """Set the per-environment parameters of the given environments; None keeps a parameter.

        Args:
            env_ids: The environments, shape [N].
            drag_height: End-effector heights while dragging [m], shape [N].
            drag_speed: Speeds of the drag [m/s], shape [N].
            yaw_offset: Offsets added to the grasp yaw [rad], shape [N].
            pregrasp_offset: Pre-grasp positions relative to the grasp node [m], shape [N, 3].
            pregrasp_yaw_error: Yaw of the pre-grasp pose relative to the grasp yaw [rad], shape [N].
        """
        params = ((self.drag_height, drag_height), (self.drag_speed, drag_speed), (self.yaw_offset, yaw_offset))
        params += ((self.pregrasp_offset, pregrasp_offset), (self.pregrasp_yaw_error, pregrasp_yaw_error))
        for buffer, values in params:
            if values is not None:
                buffer[env_ids] = values

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
                self.drag_height_wp,
                self.drag_speed_wp,
                self.yaw_offset_wp,
                self.pregrasp_offset_wp,
                self.pregrasp_yaw_error_wp,
                self.grasp_pos_wp,
                self.hold_pos_wp,
                self.des_ee_pos_wp,
                self.des_gripper_state_wp,
                self.des_yaw_wp,
                self.yaw_step_wp,
                self.hover_height,
                self.grasp_height,
                self.goal_tolerance,
                self.approach_threshold,
                self.grasp_threshold,
                self.wrist_limits[0],
                self.wrist_limits[1],
                int(self.pregrasp_hold),
            ],
            device=self.device,
        )
        return self.des_ee_pos, self.yaw_step, self.des_gripper_state


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


class ClothDragSmRunner:
    """Feed the task's state to a :class:`DragClothSm` and turn its output into the task's actions.

    The runner reads the task state in the robot base frame, takes the task's grasp node while the state machine rests,
    detects a stalled drag, runs the state machine and converts its output to the 4-DoF delta end-effector action and
    the binary gripper command. The quantities of the last :meth:`compute` stay readable as attributes.
    """

    def __init__(self, env: ManagerBasedRLEnv, sm: DragClothSm, raised_nodes_event: str | None = "reset_deformable"):
        """Look up the task terms.

        Args:
            env: The unwrapped environment, with the ``ee_delta`` actions.
            sm: The state machine of all environments.
            raised_nodes_event: Name of the reset event that raises the grasp candidates, or None.
        """
        from ..mdp.frames import raised_node_ids

        self.env = env
        self.sm = sm
        self.command = env.command_manager.get_term("cloth_target")
        self.gripper = env.action_manager.get_term("gripper_action")
        self.arm_cfg = env.action_manager.get_term("arm_action").cfg
        self.cloth = env.scene[self.command.cfg.object_name]
        self.robot = env.scene[self.command.cfg.asset_name]
        self.wrist_joint_id = self.robot.find_joints("panda_joint7")[0][0]
        # the raised nodes of the cloth reset, redrawn at every reset (live tensor); None grasps the flat cloth
        self.raised_node_ids = raised_node_ids(env, raised_nodes_event)
        num_envs, device = env.num_envs, env.device
        self.grasp_node = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.stall_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.stall_limit = round(STALL_TIME / env.step_dt)
        self._prev_ee_pos = torch.zeros((num_envs, 3), device=device)
        self._rows = torch.arange(num_envs, device=device)
        self.previous_state = sm.sm_state.clone()
        self.actions = torch.zeros(env.action_space.shape, device=device)
        self.actions[:, 4] = 1.0  # open

    def reset_idx(self, env_ids: torch.Tensor):
        """Restart the state machine and the stall detection of the given environments."""
        self.sm.reset_idx(env_ids)
        self.stall_steps[env_ids] = 0

    def compute(self) -> torch.Tensor:
        """Run the state machine on the current task state.

        Returns:
            The actions of all environments: translation and yaw deltas, normalized to [-1, 1], and the gripper command
            (1 open, -1 close), shape [num_envs, 5].
        """
        from ..mdp.frames import cloth_center_pos_b, ee_pose_in_base, points_to_base, top_down_yaw
        from ..mdp.stages import goal_point_pos_b

        env = self.env
        # observations in the robot base frame
        self.ee_pos, self.ee_quat = ee_pose_in_base(env)
        self.center_pos = cloth_center_pos_b(env)
        self.goal_pos = goal_point_pos_b(env, self.command.cfg)
        self.nodes_b = points_to_base(env, self.cloth.data.nodal_pos_w.torch)
        target_pos = self.command.command
        # -- grasp node, chosen while resting
        resting = self.sm.sm_state == DragSmState.REST
        if resting.any():
            self.grasp_node[resting] = self.command.grasp_node_id[resting]
        # -- drag stall: the arm stopped, e.g. at its reach limit next to the robot base
        still = (self.ee_pos - self._prev_ee_pos).norm(dim=-1) < STALL_STEP
        dragging = self.sm.sm_state == DragSmState.DRAG_CLOTH
        self.stall_steps = torch.where(dragging & still, self.stall_steps + 1, 0)
        self._prev_ee_pos = self.ee_pos.clone()

        # advance state machine
        self.previous_state = self.sm.sm_state.clone()
        des_ee_pos, self.yaw_step, des_gripper_state = self.sm.compute(
            self.ee_pos,
            top_down_yaw(self.ee_quat),
            self.robot.data.joint_pos.torch[:, self.wrist_joint_id],
            self.nodes_b[self._rows, self.grasp_node],
            self.center_pos,
            self.goal_pos,
            target_pos,
            self.command.stage,
            self.gripper.is_grasping,
            self.command.holding,
            self.stall_steps >= self.stall_limit,
        )

        # convert the desired end-effector position and yaw change to the delta end-effector action
        self.actions[:, :3] = ((des_ee_pos - self.ee_pos) / self.arm_cfg.scale_trans).clamp(-1.0, 1.0)
        self.actions[:, 3] = (self.yaw_step / self.arm_cfg.scale_rot).clamp(-1.0, 1.0)
        self.actions[:, 4] = des_gripper_state
        return self.actions


def keep_requested_visualizers(env_cfg: ManagerBasedRLEnvCfg, visualizers: Sequence[str] | None):
    """Keep only the requested visualizers of the task config.

    The task configures headless Kit and Newton viewers too, and the scripts set their default ``--viz`` with
    ``set_defaults``, not on the command line, so it does not count as an explicit choice.

    Args:
        env_cfg: The task config, changed in place.
        visualizers: The requested visualizer names (``none`` for none), or None for none.
    """
    aliases = {"newton": "newton_gl"}
    requested = {aliases.get(viz, viz) for viz in (visualizers or []) if viz != "none"}
    env_cfg.sim.visualizer_cfgs = [
        cfg for cfg in env_cfg.sim.visualizer_cfgs if getattr(cfg, "visualizer_type", None) in requested
    ]

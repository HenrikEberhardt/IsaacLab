# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to drag a creased cloth, and the cube resting on it, with a robotic arm.

The state machine hovers above the crease of the ``IsaacContrib-Franka-nonp-cloth`` sheet, descends with an
open gripper, pinches the crease, drags it to a target point, and releases it. Each episode draws the target
at random on a line in front of the robot base. The state machine is implemented in the kernel function
`infer_state_machine`, which uses the `warp` library to run all environments in parallel on the GPU.

By default, the gripper holds the cloth with position-level bilateral constraints, as in the FLASH cloth
gripping model: once the fingers close, every cloth vertex within :data:`GRASP_PROXIMITY` of the middle or the
tip of a finger is locked to the gripper by the equality constraint ``x_i - p_grip,i(t) = 0``, a zero-length
rigid link to the gripper point where the vertex was grasped, and is released again when the gripper opens.
With grasped vertices spread along both fingers, the cloth cannot rotate about the grasp. While the constraint
holds the cloth, the gripper stops colliding with it, like the virtual gripper of FLASH.
``--grasp_mode friction`` holds the cloth with the closed gripper and friction alone.

.. code-block:: bash

    # Newton OpenGL viewer (default).
    uv run python scripts/environments/state_machine/franka_nonp_cloth_sm.py

    # Friction-only grasp, headless.
    uv run python scripts/environments/state_machine/franka_nonp_cloth_sm.py --grasp_mode friction --viz none

    # Record the first 20 s episode (600 frames at 30 fps) to ./videos with the kitless OVRTX path tracer.
    uv run python scripts/environments/state_machine/franka_nonp_cloth_sm.py --num_envs 2 --viz newton_rtx --video

"""

import argparse
import sys
from collections.abc import Sequence

import gymnasium as gym
import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import mdp
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

# add argparse arguments
parser = argparse.ArgumentParser(description="Drag a creased cloth by its crease with a robotic arm.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--num_steps", type=int, default=1000, help="Number of environment steps to run.")
parser.add_argument("--task", type=str, default="IsaacContrib-Franka-nonp-cloth", help="The task to run.")
parser.add_argument(
    "--grasp_mode",
    type=str,
    default="attach",
    choices=["attach", "friction"],
    help="Hold the cloth by locking the grasped vertices to the gripper, or by friction alone.",
)
parser.add_argument("--drag_speed", type=float, default=0.08, help="Speed of the drag target [m/s].")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record the viewport of the visualizer to ./videos."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=None,
    help="Number of env steps (frames) to record. Defaults to one episode.",
)
add_launcher_args(parser)
# the task runs on Newton, so default to the Newton viewer and let the task pick the device
parser.set_defaults(device=None, visualizer=["newton_gl"])
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

# initialize warp
wp.init()

# Drag targets are drawn uniformly on this line in front of the robot base, which the gripper reaches over its
# whole width at table height. The line lies closer to the robot than any crease, so every drag pulls the sheet
# toward the robot instead of folding it over itself [m].
TARGET_LINE_X = 0.35
TARGET_LINE_Y = (-0.30, 0.30)
# Damping of the differential IK. The task's IK preset (0.6) settles too slowly for the low-PD Franka to reach
# down onto the crease.
IK_DAMPING = 0.1
# height of the approach pose above the crease, and of the retreat pose above the release pose [m]
HOVER_HEIGHT = 0.10
# depth of the grasp point below the crest, so the finger pads straddle the fold [m]
GRASP_DEPTH = 0.012
# lift of the grasp at the start of the drag, which keeps the fingertips clear of the table [m]
DRAG_LIFT = 0.01
# cloth nodes within this height of the highest node form the crest of the crease [m]
CREST_BAND = 0.005
# Gripper anchors along each finger, as offsets from the fingertip point (0.1034 m below the hand) toward the
# fingertips [m]. A Franka finger runs from its joint 0.0584 m below the hand to its tip at about 0.112 m, so
# its middle lies 1.8 cm above the fingertip point and its tip 0.7 cm below it.
FINGER_ANCHOR_DEPTHS = {"middle": -0.018, "tip": 0.007}
# cloth vertices within this distance of a gripper anchor are grasped when the gripper closes [m]
GRASP_PROXIMITY = 0.015
# height above the grasp at which the retreating gripper collides with the cloth again [m]
CONTACT_CLEARANCE = 0.05


class DragSmState:
    """States for the drag state machine."""

    REST = wp.constant(0)
    APPROACH_ABOVE_CREASE = wp.constant(1)
    APPROACH_CREASE = wp.constant(2)
    GRASP_CREASE = wp.constant(3)
    DRAG_CLOTH = wp.constant(4)
    RELEASE_CLOTH = wp.constant(5)
    RETREAT = wp.constant(6)


class DragSmWaitTime:
    """Additional wait times (in s) for states before switching.

    Wait times are generous because the low-PD Franka takes a while to settle on each IK target.
    """

    REST = wp.constant(0.2)
    APPROACH_ABOVE_CREASE = wp.constant(1.0)
    APPROACH_CREASE = wp.constant(1.5)
    GRASP_CREASE = wp.constant(1.0)
    # settle time after the drag target has stopped
    DRAG_CLOTH = wp.constant(0.5)
    RELEASE_CLOTH = wp.constant(0.5)


@wp.func
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    return wp.length(current_pos - desired_pos) < threshold


@wp.func
def translated(pose: wp.transform, offset: wp.vec3) -> wp.transform:
    return wp.transform(wp.transform_get_translation(pose) + offset, wp.transform_get_rotation(pose))


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    grasp_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    des_gripper_opening: wp.array(dtype=float),
    gripper_close_opening: wp.array(dtype=float),
    gripper_open_opening: float,
    drag_target: wp.array(dtype=wp.vec3),
    hover_height: float,
    drag_speed: float,
    drag_lift: float,
    position_threshold: float,
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    hover = wp.vec3(0.0, 0.0, hover_height)
    rotation = wp.transform_get_rotation(grasp_pose[tid])
    # the drag runs straight from the lifted grasp to the target, at the height of the lifted grasp
    drag_start = wp.transform_get_translation(grasp_pose[tid]) + wp.vec3(0.0, 0.0, drag_lift)
    drag_end = wp.vec3(drag_target[tid][0], drag_target[tid][1], drag_start[2])
    drag_vector = drag_end - drag_start
    drag_distance = wp.length(drag_vector)
    # decide next state
    if state == DragSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        des_gripper_opening[tid] = gripper_open_opening
        # wait for a while
        if sm_wait_time[tid] >= DragSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = DragSmState.APPROACH_ABOVE_CREASE
            sm_wait_time[tid] = 0.0
    elif state == DragSmState.APPROACH_ABOVE_CREASE:
        des_ee_pose[tid] = translated(grasp_pose[tid], hover)
        des_gripper_opening[tid] = gripper_open_opening
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= DragSmWaitTime.APPROACH_ABOVE_CREASE:
                # move to next state and reset wait time
                sm_state[tid] = DragSmState.APPROACH_CREASE
                sm_wait_time[tid] = 0.0
    elif state == DragSmState.APPROACH_CREASE:
        des_ee_pose[tid] = grasp_pose[tid]
        des_gripper_opening[tid] = gripper_open_opening
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= DragSmWaitTime.APPROACH_CREASE:
                # move to next state and reset wait time
                sm_state[tid] = DragSmState.GRASP_CREASE
                sm_wait_time[tid] = 0.0
    elif state == DragSmState.GRASP_CREASE:
        des_ee_pose[tid] = grasp_pose[tid]
        des_gripper_opening[tid] = gripper_close_opening[tid]
        # wait for a while
        if sm_wait_time[tid] >= DragSmWaitTime.GRASP_CREASE:
            # move to next state and reset wait time
            sm_state[tid] = DragSmState.DRAG_CLOTH
            sm_wait_time[tid] = 0.0
    elif state == DragSmState.DRAG_CLOTH:
        # move the end-effector target toward the drag target at the drag speed
        progress = wp.min(sm_wait_time[tid] * drag_speed, drag_distance)
        drag_pos = drag_end
        if drag_distance > 1.0e-6:
            drag_pos = drag_start + drag_vector * (progress / drag_distance)
        des_ee_pose[tid] = wp.transform(drag_pos, rotation)
        des_gripper_opening[tid] = gripper_close_opening[tid]
        if progress >= drag_distance and distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= drag_distance / drag_speed + DragSmWaitTime.DRAG_CLOTH:
                # move to next state and reset wait time
                sm_state[tid] = DragSmState.RELEASE_CLOTH
                sm_wait_time[tid] = 0.0
    elif state == DragSmState.RELEASE_CLOTH:
        des_ee_pose[tid] = wp.transform(drag_end, rotation)
        des_gripper_opening[tid] = gripper_open_opening
        # wait for a while
        if sm_wait_time[tid] >= DragSmWaitTime.RELEASE_CLOTH:
            # move to next state and reset wait time
            sm_state[tid] = DragSmState.RETREAT
            sm_wait_time[tid] = 0.0
    elif state == DragSmState.RETREAT:
        # final state: hold the gripper above the release pose
        des_ee_pose[tid] = wp.transform(drag_end + hover, rotation)
        des_gripper_opening[tid] = gripper_open_opening
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class DragClothSm:
    """A simple state machine in a robot's task space to drag a cloth by its crease.

    The state machine is implemented as a warp kernel. It takes in the current pose of the robot's
    end-effector and the grasp pose on the crease, and outputs the desired pose of the end-effector and the
    desired opening of each finger. The state machine has the following states:

    1. REST: The robot is at rest.
    2. APPROACH_ABOVE_CREASE: The robot hovers above the crease with an open gripper.
    3. APPROACH_CREASE: The robot descends onto the crease with an open gripper.
    4. GRASP_CREASE: The robot closes the gripper on the crease.
    5. DRAG_CLOTH: The robot drags the crease to the episode's target point.
    6. RELEASE_CLOTH: The robot opens the gripper.
    7. RETREAT: The robot lifts the gripper away from the cloth. This is the final state.
    """

    def __init__(
        self,
        dt: float,
        num_envs: int,
        device: torch.device | str = "cpu",
        gripper_open_opening: float = 0.04,
        gripper_close_opening: float = 0.0,
        drag_speed: float = 0.08,
        position_threshold: float = 0.02,
    ):
        """Initialize the state machine.

        Args:
            dt: The environment time step [s].
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
            gripper_open_opening: Opening of each finger when the gripper is open [m].
            gripper_close_opening: Opening each finger closes to, unless :meth:`hold_gripper` changes it [m].
            drag_speed: Speed of the drag target [m/s].
            position_threshold: Distance at which the end-effector counts as on target [m].
        """
        # save parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.gripper_open_opening = gripper_open_opening
        self.default_gripper_close_opening = gripper_close_opening
        self.drag_speed = drag_speed
        self.position_threshold = position_threshold
        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_opening = torch.full((self.num_envs,), gripper_open_opening, device=self.device)
        # opening each finger closes to [m]
        self.gripper_close_opening = torch.full((self.num_envs,), gripper_close_opening, device=self.device)
        # drag target of each episode, in the robot base frame [m]
        self.drag_target = torch.zeros((self.num_envs, 3), device=self.device)
        self.sample_drag_targets()

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_opening_wp = wp.from_torch(self.des_gripper_opening, wp.float32)
        self.gripper_close_opening_wp = wp.from_torch(self.gripper_close_opening, wp.float32)
        self.drag_target_wp = wp.from_torch(self.drag_target, wp.vec3)

    def sample_drag_targets(self, env_ids: Sequence[int] | None = None):
        """Draw new drag targets uniformly on the target line."""
        if env_ids is None:
            env_ids = slice(None)
        num_targets = self.drag_target[env_ids].shape[0]
        self.drag_target[env_ids, 0] = TARGET_LINE_X
        self.drag_target[env_ids, 1] = math_utils.sample_uniform(*TARGET_LINE_Y, (num_targets,), device=self.device)

    def hold_gripper(self, env_ids: Sequence[int], opening: torch.Tensor):
        """Keep the closed gripper at the given finger opening [m] until the next reset."""
        self.gripper_close_opening[env_ids] = opening

    def reset_idx(self, env_ids: Sequence[int] = None):
        """Reset the state machine and draw new drag targets."""
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0
        self.gripper_close_opening[env_ids] = self.default_gripper_close_opening
        self.sample_drag_targets(env_ids)

    def compute(self, ee_pose: torch.Tensor, grasp_pose: torch.Tensor) -> torch.Tensor:
        """Compute the desired state of the robot's end-effector and the gripper."""

        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        grasp_pose_wp = wp.from_torch(grasp_pose.contiguous(), wp.transform)

        # run state machine
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                grasp_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_opening_wp,
                self.gripper_close_opening_wp,
                self.gripper_open_opening,
                self.drag_target_wp,
                HOVER_HEIGHT,
                self.drag_speed,
                DRAG_LIFT,
                self.position_threshold,
            ],
            device=self.device,
        )

        # convert to torch
        return torch.cat([self.des_ee_pose, self.des_gripper_opening.unsqueeze(-1)], dim=-1)


def crease_grasp_pose(nodes: torch.Tensor, top_down_quat: torch.Tensor) -> torch.Tensor:
    """Return a top-down grasp pose on the crest of the crease.

    The crest is the band of nodes within :data:`CREST_BAND` of the highest node. The grasp point lies at the
    mean position of that band, :data:`GRASP_DEPTH` below the crest, so the finger pads straddle the fold.
    The gripper yaws so that its fingers close across the crease, along the line from the sheet center to
    the crest.

    Args:
        nodes: Cloth node positions in the robot base frame [m], shape [N, P, 3].
        top_down_quat: Orientation (x, y, z, w) that points the gripper straight down, shape [4].

    Returns:
        The grasp pose (position [m] and quaternion (x, y, z, w)) in the robot base frame, shape [N, 7].
    """
    height = nodes[..., 2]
    crest_height = height.amax(dim=1)
    crest = (height >= crest_height.unsqueeze(1) - CREST_BAND).unsqueeze(-1)
    crest_xy = (nodes[..., :2] * crest).sum(dim=1) / crest.sum(dim=1)
    across = crest_xy - nodes[..., :2].mean(dim=1)
    # the top-down orientation closes the fingers along world -y; yaw that axis onto the crease normal
    yaw = torch.atan2(across[:, 0], -across[:, 1])
    # the gripper is symmetric, so keep the yaw within +-90 deg to limit the wrist turn
    yaw = torch.remainder(yaw + torch.pi / 2, torch.pi) - torch.pi / 2
    zeros = torch.zeros_like(yaw)
    quat = math_utils.quat_mul(math_utils.quat_from_euler_xyz(zeros, zeros, yaw), top_down_quat.expand(len(yaw), 4))
    position = torch.cat([crest_xy, (crest_height - GRASP_DEPTH).unsqueeze(-1)], dim=-1)
    return torch.cat([position, quat], dim=-1)


class GripperConstraint:
    """Lock grasped cloth vertices to the gripper with position-level bilateral constraints.

    This follows the cloth gripping model of FLASH. When the gripper closes, every cloth vertex within
    :attr:`proximity` of one of the gripper's anchors joins the grasped set. While the constraint is active, each
    grasped vertex is driven as a kinematic node at the gripper point where it was grasped, which enforces the
    equality constraint ``x_i - p_grip,i(t) = 0``, a zero-length rigid link. Opening the gripper deactivates the
    constraint and returns the vertices to the cloth's internal elastic forces and gravity.

    The gripper points move rigidly with the hand. Locking each vertex to its own grasp point, instead of pulling
    all grasped vertices onto the anchors themselves, keeps the grasped patch intact: collapsing up to 1.5 cm of
    cloth onto a few points degenerates its triangles, and VBD's elastic forces then blow up.

    While the constraint is active, the gripper stops colliding with the cloth in that environment, as the
    virtual gripper of FLASH does: the constraint is then its whole interaction with the grasped cloth. The
    fingers would otherwise press against the immovable locked vertices, and that contact destabilizes the
    coupling between the robot and the cloth solver. The state machine holds the fingers at the opening they closed
    to on the fold, so they keep resting on it instead of closing through it. The gripper collides with the cloth
    again only once :meth:`restore_contacts` is called after it has left the cloth; turning the contacts on with the
    fingers inside the cloth blows up the coupling the same way.

    Each finger carries one anchor per entry of :data:`FINGER_ANCHOR_DEPTHS`, at its middle and at its tip, on the
    inner pad surface, so the grasped vertices are the cloth pinched against the pads along both fingers and the
    grasped cloth cannot rotate about the grasp.

    When the constraint is disabled, the grasped set is still recorded, to report how far the cloth slips in a
    friction-only grasp.
    """

    def __init__(self, env, enabled: bool, particle_radius: float, proximity: float = GRASP_PROXIMITY):
        """Initialize the constraint.

        Args:
            env: The unwrapped environment.
            enabled: Whether to enforce the constraint on the grasped vertices.
            particle_radius: Collision radius of the cloth vertices [m].
            proximity: Distance from an anchor within which a vertex is grasped [m].
        """
        self.enabled = enabled
        self.proximity = proximity
        self.particle_radius = particle_radius
        # the Newton coupler manager of the task can switch the gripper's cloth contacts per environment
        self._set_gripper_contacts = getattr(env.sim.physics_manager, "set_proxy_particle_contacts", None)
        self._robot = env.scene["robot"]
        self._cloth = env.scene["deformable"]
        self._finger_ids, _ = self._robot.find_bodies(["panda_leftfinger", "panda_rightfinger"], preserve_order=True)
        self._finger_joint_ids, _ = self._robot.find_joints(
            ["panda_finger_joint1", "panda_finger_joint2"], preserve_order=True
        )
        self._hand_id = self._robot.find_bodies("panda_hand")[0][0]

        num_envs, device = env.num_envs, env.device
        # one anchor per finger and depth: left middle, left tip, right middle, right tip
        depths = torch.tensor(list(FINGER_ANCHOR_DEPTHS.values()), device=device)
        self.anchor_finger = torch.arange(2, device=device).repeat_interleave(len(depths))
        self.anchor_depth = depths.repeat(2)
        num_vertices = self._cloth.max_sim_vertices_per_body
        # grasp point of each vertex in the hand frame [m]
        self.grasp_offsets = torch.zeros((num_envs, num_vertices, 3), device=device)
        self.grasped = torch.zeros((num_envs, num_vertices), dtype=torch.bool, device=device)
        self.grasp_vertex_pos = torch.zeros((num_envs, num_vertices, 3), device=device)
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.contacts_disabled = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def activate(self, env_ids: torch.Tensor, tcp_pos_w: torch.Tensor):
        """Grasp the cloth vertices near the gripper anchors.

        Args:
            env_ids: Environments that just closed the gripper.
            tcp_pos_w: World position of the fingertip point between the fingers [m], shape [E, 3].
        """
        hand = self._robot.data.body_link_pose_w.torch[env_ids, self._hand_id]
        hand_pos, hand_quat = hand[:, :3], hand[:, 3:7]
        # the fingers close along the hand's y axis and point along its z axis
        hand_axes = torch.eye(3, device=hand.device)
        closing_axis = math_utils.quat_apply(hand_quat, hand_axes[1].expand_as(tcp_pos_w))
        approach_axis = math_utils.quat_apply(hand_quat, hand_axes[2].expand_as(tcp_pos_w))
        # each finger's inner pad surface lies its joint opening away from the fingertip point, on its side
        finger_pos = self._robot.data.body_link_pose_w.torch[env_ids][:, self._finger_ids, :3]
        side = torch.sign(((finger_pos - tcp_pos_w.unsqueeze(1)) * closing_axis.unsqueeze(1)).sum(dim=-1))
        opening = self._robot.data.joint_pos.torch[env_ids][:, self._finger_joint_ids]
        lateral = side * (opening - self.particle_radius).clamp_min(0.0)
        anchors = (
            tcp_pos_w.unsqueeze(1)
            + lateral[:, self.anchor_finger].unsqueeze(-1) * closing_axis.unsqueeze(1)
            + self.anchor_depth.view(1, -1, 1) * approach_axis.unsqueeze(1)
        )

        # grasp every vertex within the proximity threshold of an anchor, at its current position on the gripper
        vertices = self._cloth.data.nodal_pos_w.torch[env_ids]
        distance = (vertices.unsqueeze(1) - anchors.unsqueeze(2)).norm(dim=-1).amin(dim=1)
        num_vertices = vertices.shape[1]
        self.grasped[env_ids] = distance <= self.proximity
        self.grasp_offsets[env_ids] = math_utils.quat_apply_inverse(
            hand_quat.unsqueeze(1).expand(-1, num_vertices, -1), vertices - hand_pos.unsqueeze(1)
        )
        self.grasp_vertex_pos[env_ids] = vertices
        self.active[env_ids] = True
        if self.enabled and self._set_gripper_contacts is not None:
            self._set_gripper_contacts(env_ids.tolist(), enabled=False)
            self.contacts_disabled[env_ids] = True

    def deactivate(self, env_ids: torch.Tensor):
        """Release the grasped vertices of the given environments.

        The gripper keeps passing through the cloth until :meth:`restore_contacts` is called.
        """
        self.active[env_ids] = False

    def restore_contacts(self, env_ids: torch.Tensor):
        """Let the gripper collide with the cloth again, once it has left the cloth."""
        env_ids = env_ids[self.contacts_disabled[env_ids]]
        if len(env_ids) > 0 and self._set_gripper_contacts is not None:
            self._set_gripper_contacts(env_ids.tolist(), enabled=True)
            self.contacts_disabled[env_ids] = False

    def grasp_points(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Return the current world position of each vertex's grasp point on the gripper [m], shape [E, P, 3]."""
        hand = self._robot.data.body_link_pose_w.torch[env_ids, self._hand_id]
        num_vertices = self.grasp_offsets.shape[1]
        hand_quat = hand[:, 3:7].unsqueeze(1).expand(-1, num_vertices, -1)
        return hand[:, :3].unsqueeze(1) + math_utils.quat_apply(hand_quat, self.grasp_offsets[env_ids])

    def slip(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Return the mean distance of the grasped vertices from their grasp points [m], shape [E]."""
        grasped = self.grasped[env_ids]
        vertices = self._cloth.data.nodal_pos_w.torch[env_ids]
        distance = (vertices - self.grasp_points(env_ids)).norm(dim=-1)
        return (distance * grasped).sum(dim=1) / grasped.sum(dim=1).clamp_min(1)

    def kinematic_targets(self) -> torch.Tensor:
        """Return nodal kinematic targets that lock the grasped vertices to the gripper, shape [N, P, 4]."""
        vertices = self._cloth.data.nodal_pos_w.torch
        targets = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
        locked = self.grasped & self.active.unsqueeze(1)
        if locked.any():
            env_ids = torch.arange(vertices.shape[0], device=vertices.device)
            targets[..., :3] = torch.where(locked.unsqueeze(-1), self.grasp_points(env_ids), vertices)
            targets[..., 3] = torch.where(locked, 0.0, 1.0)
        return targets


def main():
    # parse configuration via Hydra, so presets can be selected on the CLI
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    args_cli.device = env_cfg.sim.device
    env_cfg.scene.num_envs = args_cli.num_envs
    # Scripted demo: keep only the time-out, extended so the slow low-PD Franka can finish the drag, and
    # drop the failure terminations so a transient bound or velocity spike does not cut a run short.
    env_cfg.episode_length_s = 20.0
    for term_name in list(vars(env_cfg.terminations)):
        if term_name != "time_out":
            setattr(env_cfg.terminations, term_name, None)
    # simulate full gravity from the start instead of the training curriculum's ramp from zero
    env_cfg.curriculum.gravity = None
    # the state machine emits absolute end-effector poses, so pick the task's own IK action preset
    env_cfg.actions = type(env_cfg)().actions.ik
    env_cfg.actions.arm_action.controller.ik_params = {"lambda_val": IK_DAMPING}
    # Command the finger opening directly instead of the preset's open/close switch, so the gripper can hold the
    # opening at which it closed on the cloth. The task's IK preset sets the open and closed openings; a friction
    # grasp closes fully instead, since it needs the fingers to squeeze the fold.
    preset_gripper = env_cfg.actions.gripper_action
    gripper_open_opening = preset_gripper.open_command_expr["panda_finger_joint1"]
    gripper_close_opening = preset_gripper.close_command_expr["panda_finger_joint1"]
    if args_cli.grasp_mode == "friction":
        gripper_close_opening = 0.0
    env_cfg.actions.gripper_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["panda_finger_joint1"], use_default_offset=False
    )
    # the visualizers keep the task's camera, which is the same for every visualizer type
    requested_visualizers = [viz for viz in (args_cli.visualizer or []) if viz != "none"]
    # A Kit visualizer in the config starts Kit even when --viz does not select it, and Kit cannot share a
    # process with OVRTX (--viz newton_rtx), so keep it only when requested.
    if "kit" not in requested_visualizers:
        env_cfg.sim.visualizer_cfgs = [
            cfg for cfg in env_cfg.sim.visualizer_cfgs if getattr(cfg, "visualizer_type", None) != "kit"
        ]
    if args_cli.video:
        # render cameras, which a headless Kit viewport needs for capture, as the RL entry points do for --video
        args_cli.enable_cameras = True
        # one frame per env step: a 20 s episode at the 1/30 s env step is 600 frames
        video_length = args_cli.video_length or round(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))
        if env_cfg.video_recorders:
            # e.g. a camera task's sensor recorders, selected with env.video_recorders=rgb
            for recorder_cfg in env_cfg.video_recorders:
                recorder_cfg.video_length = video_length
        else:
            capture_visualizers = [viz for viz in requested_visualizers if viz in ("kit", "newton_gl", "newton_rtx")]
            if not capture_visualizers:
                raise SystemExit(
                    "--video records the viewport of a visualizer: pass --viz newton_gl, --viz newton_rtx (kitless"
                    " OVRTX), or --viz kit."
                )
            env_cfg.video_recorders = [
                VideoRecorderCfg(
                    source=f"visualizer:{capture_visualizers[0]}",
                    video_length=video_length,
                    output_dir="videos",
                    output_filename_prefix="franka_nonp_cloth_sm",
                )
            ]

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        unwrapped = env.unwrapped
        device = unwrapped.device

        # reset environment at start
        env.reset()

        # create action buffers (position + quaternion + gripper)
        actions = torch.zeros(unwrapped.action_space.shape, device=device)
        actions[:, 3] = 1.0
        actions[:, 7] = gripper_open_opening
        # (x, y, z, w) half turn about x, which points the gripper straight down
        top_down_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        grasp_pose = torch.zeros((unwrapped.num_envs, 7), device=device)

        # create state machine and the gripper constraint
        drag_sm = DragClothSm(
            env_cfg.sim.dt * env_cfg.decimation,
            unwrapped.num_envs,
            device,
            gripper_open_opening=gripper_open_opening,
            gripper_close_opening=gripper_close_opening,
            drag_speed=args_cli.drag_speed,
        )
        cloth_material = env_cfg.scene.deformable.spawn.physics_material
        constraint = GripperConstraint(
            unwrapped,
            enabled=args_cli.grasp_mode == "attach",
            particle_radius=getattr(cloth_material, "particle_radius", 0.002),
        )
        cloth, cube, robot = unwrapped.scene["deformable"], unwrapped.scene["cube"], unwrapped.scene["robot"]
        finger_joint_id = robot.find_joints("panda_finger_joint1")[0][0]
        ee_frame_sensor = unwrapped.scene["ee_frame"]
        cube_grasp_pos = torch.zeros((unwrapped.num_envs, 3), device=device)

        for _ in range(args_cli.num_steps):
            # run everything in inference mode
            with torch.inference_mode():
                # step environment
                _, _, terminated, time_outs, _ = env.step(actions)
                dones = terminated | time_outs

                # reset state machine and release the cloth of reset environments
                if dones.any():
                    done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                    drag_sm.reset_idx(done_ids)
                    constraint.deactivate(done_ids)
                    # the reset moves the arm away from the cloth
                    constraint.restore_contacts(done_ids)

                # observations
                env_origins = unwrapped.scene.env_origins
                # -- end-effector frame
                tcp_pos_w = ee_frame_sensor.data.target_pos_w.torch[..., 0, :].clone()
                tcp_quat_w = ee_frame_sensor.data.target_quat_w.torch[..., 0, :].clone()
                # -- grasp pose on the crease, tracked while resting and held once the approach starts
                resting = drag_sm.sm_state == DragSmState.REST
                if resting.any():
                    nodes = cloth.data.nodal_pos_w.torch - env_origins.unsqueeze(1)
                    grasp_pose[resting] = crease_grasp_pose(nodes, top_down_quat)[resting]

                # advance state machine
                previous_state = drag_sm.sm_state.clone()
                actions = drag_sm.compute(torch.cat([tcp_pos_w - env_origins, tcp_quat_w], dim=-1), grasp_pose)

                # grasp the cloth vertices near the gripper once it has closed
                grasped = (previous_state == DragSmState.GRASP_CREASE) & (drag_sm.sm_state == DragSmState.DRAG_CLOTH)
                if grasped.any():
                    grasped_ids = grasped.nonzero(as_tuple=False).squeeze(-1)
                    constraint.activate(grasped_ids, tcp_pos_w[grasped_ids])
                    if constraint.enabled:
                        # the fingers stop colliding with the held cloth, so keep them where they closed on the fold
                        drag_sm.hold_gripper(grasped_ids, robot.data.joint_pos.torch[grasped_ids, finger_joint_id])
                    cube_grasp_pos[grasped_ids] = cube.data.root_pos_w.torch[grasped_ids]
                    num_grasped = constraint.grasped[grasped_ids].sum(dim=1).float().mean().item()
                    print(
                        f"[INFO]: Grasped the crease in {len(grasped_ids)} env(s): {num_grasped:.1f} cloth vertices"
                        f" within {constraint.proximity * 100:.1f} cm of the gripper anchors"
                        f" ({args_cli.grasp_mode} grasp)."
                    )

                # report the drag and release the cloth
                released = (previous_state == DragSmState.DRAG_CLOTH) & (drag_sm.sm_state == DragSmState.RELEASE_CLOTH)
                if released.any():
                    released_ids = released.nonzero(as_tuple=False).squeeze(-1)
                    grasped_mask = constraint.grasped[released_ids].unsqueeze(-1)
                    num_grasped = grasped_mask.sum(dim=1).clamp_min(1)
                    vertices = cloth.data.nodal_pos_w.torch[released_ids]
                    crease_pos = (vertices * grasped_mask).sum(dim=1) / num_grasped
                    grasp_pos = (constraint.grasp_vertex_pos[released_ids] * grasped_mask).sum(dim=1) / num_grasped
                    node_shift = (crease_pos - grasp_pos).mean(dim=0)
                    cube_shift = (cube.data.root_pos_w.torch[released_ids] - cube_grasp_pos[released_ids]).mean(dim=0)
                    slip = constraint.slip(released_ids).mean().item()
                    crease_xy = crease_pos[:, :2] - env_origins[released_ids, :2]
                    target_error = (crease_xy - drag_sm.drag_target[released_ids, :2]).norm(dim=-1).mean().item()
                    print(
                        f"[INFO]: Released the cloth in {len(released_ids)} env(s). Mean shift of the grasped vertices:"
                        f" ({node_shift[0]:+.3f}, {node_shift[1]:+.3f}) m, of the cube:"
                        f" ({cube_shift[0]:+.3f}, {cube_shift[1]:+.3f}) m in (x, y); grasped vertices end"
                        f" {target_error * 100:.1f} cm from the target, {slip * 1000:.1f} mm from their grasp points."
                    )
                    constraint.deactivate(released_ids)

                # let the gripper collide with the cloth again once it has retreated above it
                clear = (drag_sm.sm_state == DragSmState.RETREAT) & (
                    tcp_pos_w[:, 2] - env_origins[:, 2] > grasp_pose[:, 2] + CONTACT_CLEARANCE
                )
                if clear.any():
                    constraint.restore_contacts(clear.nonzero(as_tuple=False).squeeze(-1))

                # enforce the gripper constraint on the grasped vertices
                if constraint.enabled:
                    cloth.write_nodal_kinematic_target_to_sim_index(constraint.kinematic_targets())

        # close the environment
        env.close()


if __name__ == "__main__":
    main()

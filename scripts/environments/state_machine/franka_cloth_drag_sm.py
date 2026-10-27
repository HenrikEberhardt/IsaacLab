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
releases the cloth and rises. The state machine is implemented in
:mod:`isaaclab_tasks.contrib.franka_cloth_drag.state_machine` (warp kernel, all environments in parallel on the GPU).

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
from isaaclab_tasks.contrib.franka_cloth_drag.state_machine import (
    HOVER_MARGIN,
    MAX_GRASP_TRIES,
    STATE_NAMES,
    YAW_ACTION_SCALE,
    YAW_TOLERANCE,
    ClothDragSmRunner,
    DragClothSm,
    DragSmState,
    keep_requested_visualizers,
    pad_cloth_gaps,
)
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
    keep_requested_visualizers(env_cfg, args_cli.visualizer)

    with launch_simulation(env_cfg, args_cli):
        from isaaclab_tasks.contrib.franka_cloth_drag.mdp.actions import PAD_HALF_WIDTH, PAD_THICKNESS
        from isaaclab_tasks.contrib.franka_cloth_drag.mdp.stages import ClothDragStage

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
        runner = ClothDragSmRunner(unwrapped, drag_sm)
        # a reset without raised nodes grasps the flat cloth
        grasp_info = "raised nodes" if runner.raised_node_ids is not None else "flat cloth"
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
        actions = runner.actions
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
                runner.reset_idx(done_ids)
                succeeded[done_ids] = False
                peak_node_speed[done_ids] = 0.0
                hover_time[done_ids] = 0.0
                hover_ready[done_ids] = -1.0

            # advance the state machine on the task state and get the actions of the next step
            runner.compute()
            ee_pos, yaw_step, previous_state = runner.ee_pos, runner.yaw_step, runner.previous_state
            distance = (command.command[:, :2] - runner.goal_pos[:, :2]).norm(dim=-1)
            succeeded |= command.holding & ~dones
            last_distance = torch.where(dones, last_distance, distance)
            node_speed = torch.nan_to_num(cloth.data.nodal_vel_w.torch.norm(dim=-1), nan=torch.inf).amax(dim=1)
            peak_node_speed = torch.where(dones, peak_node_speed, torch.maximum(peak_node_speed, node_speed))

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
                    f"[INFO]: env {env_id}: grasped node {runner.grasp_node[env_id].item()} at a finger opening of"
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
                stalled = runner.stall_steps[env_id] >= runner.stall_limit
                reason = "drag stalled (reach limit)" if stalled else "reached the target"
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

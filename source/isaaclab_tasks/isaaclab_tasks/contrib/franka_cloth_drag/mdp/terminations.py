# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the Franka cloth dragging task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply

from ..constrained_gripper.gripper_state import gripper_cfg, pad_gap, pad_shift
from ..constrained_gripper.robotiq import ROBOTIQ_PAD_THICKNESS

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, DeformableObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import TerminationTermCfg


def external_force_w(robot: Articulation, joint_ids: torch.Tensor, body_id: int) -> torch.Tensor:
    """Estimate the external force on the robot from its joint drives, as the Franka's ``O_F_ext_hat_K`` [N], shape [N, 3].

    The external joint torque is ``tau_ext = (M + armature) qdd - tau_drive`` with the PD drive torque
    ``tau_drive = Kp (q_target - q) - Kd qd``; gravity is compensated in the simulation and Coriolis forces are
    ignored. The least-squares solution of ``J^T w = tau_ext`` at ``body_id`` gives the wrench, whose force part does
    not depend on the point it refers to. The world z component is positive when the environment pushes the robot up.

    Args:
        robot: The fixed-base robot.
        joint_ids: The joints the estimate uses (the arm joints), shape [J].
        body_id: The body the wrench acts on, e.g. the hand.
    """
    data = robot.data
    q = data.joint_pos.torch[:, joint_ids]
    qd = data.joint_vel.torch[:, joint_ids]
    qdd = data.joint_acc.torch[:, joint_ids]
    tau_drive = (
        data.joint_stiffness.torch[:, joint_ids] * (data.joint_pos_target.torch[:, joint_ids] - q)
        - data.joint_damping.torch[:, joint_ids] * qd
    )
    dof_ids = joint_ids + robot.num_base_dofs
    # Newton's mass matrix holds the links only; the joint armature adds on its diagonal
    mass = data.mass_matrix.torch[:, dof_ids][:, :, dof_ids]
    tau_ext = torch.einsum("nij,nj->ni", mass, qdd) + data.joint_armature.torch[:, joint_ids] * qdd - tau_drive
    # the fixed root has no Jacobian row
    jacobian_body = body_id - 1 if robot.is_fixed_base else body_id
    jacobian_t = data.body_link_jacobian_w.torch[:, jacobian_body][..., dof_ids].transpose(1, 2)
    return torch.linalg.lstsq(jacobian_t, tau_ext.unsqueeze(-1)).solution[:, :3, 0]


class GripperForceLimit(ManagerTermBase):
    """Terminate when the external force on the gripper exceeds ``max_force`` [N], as the real robot's safety stop.

    The force is :func:`external_force_w` at the hand, whatever the gripper touches: the real Franka cannot tell the
    table from the cloth lying on it (a policy checked only in gripper-table contact learned to press the cloth into
    the table at 66 N). With ``all_axes`` it is the largest axis ``max(|F_x|, |F_y|, |F_z|)``, as the Franka's per-axis
    collision thresholds; otherwise the upward ``F_z`` (the gripper pressing down). The term keeps it in :attr:`force`
    for :func:`~.rewards.gripper_force_penalty` and the critic; the termination manager computes before the reward and
    observation managers.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        """Resolve the arm joints and the hand body."""
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._robot: Articulation = env.scene[asset_cfg.name]
        self._joint_ids = torch.tensor(asset_cfg.joint_ids, device=env.device)
        self._body_id = asset_cfg.body_ids[0]
        self.force = torch.zeros(env.num_envs, device=env.device)
        """External force on the gripper in the last step [N], zero after a reset."""

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Zero the force of the reset environments."""
        self.force[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(
        self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, max_force: float, all_axes: bool = True
    ) -> torch.Tensor:
        """Update :attr:`force` and terminate where it exceeds ``max_force`` [N]."""
        force = torch.nan_to_num(external_force_w(self._robot, self._joint_ids, self._body_id), nan=0.0)
        self.force[:] = force.abs().amax(dim=1) if all_axes else force[:, 2].clamp(min=0.0)
        return self.force > max_force


def joint_speed_exceeded(
    env: ManagerBasedRLEnv, max_fraction: float, asset_cfg: SceneEntityCfg, skip_steps: int = 3
) -> torch.Tensor:
    """Terminate when a joint is faster than ``max_fraction`` of its velocity limit, as the real robot's safety stop.

    The first ``skip_steps`` policy steps of an episode are skipped: a reset (or a bank start) writes joint velocities
    the policy did not cause.

    Args:
        env: The environment.
        max_fraction: Allowed fraction of the joint velocity limits.
        asset_cfg: The robot and the checked joints (the arm, not the gripper).
        skip_steps: Policy steps after a reset without the check.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    joint_vel = robot.data.joint_vel.torch[:, asset_cfg.joint_ids].abs()
    limits = robot.data.joint_vel_limits.torch[:, asset_cfg.joint_ids]
    too_fast = (joint_vel > max_fraction * limits).any(dim=1)
    return too_fast & (env.episode_length_buf >= skip_steps)


def joint_near_range_limit(env: ManagerBasedRLEnv, margin: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when a joint comes within ``margin`` [rad] of its hardware range, as the real robot's safety stop.

    Args:
        env: The environment.
        margin: Distance to the joint limits that ends the episode [rad].
        asset_cfg: The robot and the checked joints (the arm, not the gripper).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos.torch[:, asset_cfg.joint_ids]
    limits = robot.data.joint_pos_limits.torch[:, asset_cfg.joint_ids]
    room = torch.minimum(joint_pos - limits[..., 0], limits[..., 1] - joint_pos)
    return (room < margin).any(dim=1)


def nonfinite_state(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cloth_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Terminate when a robot joint or a cloth node has a non-finite position or velocity."""

    robot: Articulation = env.scene[robot_cfg.name]
    cloth: DeformableObject = env.scene[cloth_cfg.name]
    finite = torch.isfinite(robot.data.joint_pos.torch).all(dim=1)
    finite &= torch.isfinite(robot.data.joint_vel.torch).all(dim=1)
    finite &= torch.isfinite(cloth.data.nodal_pos_w.torch).all(dim=(1, 2))
    finite &= torch.isfinite(cloth.data.nodal_vel_w.torch).all(dim=(1, 2))
    return ~finite


def cloth_drag_success(env: ManagerBasedRLEnv, command_name: str = "cloth_target") -> torch.Tensor:
    """Terminate after the goal point (central cloth node or object on the cloth) satisfies the command's target radius
    and consecutive hold time."""
    command = env.command_manager.get_term(command_name)
    command.update_goal_status()
    return command.holding


def gripper_below_table(
    env: ManagerBasedRLEnv, max_penetration: float, command_name: str = "cloth_target"
) -> torch.Tensor:
    """Terminate when a finger pad's lower end is more than ``max_penetration`` [m] below the table top (z = 0).

    Checks the corners of both pads' lower ends from the measured finger joint, so the limit holds in every gripper
    state (the pads move 13.5 mm along the EE frame's z-axis while closing) and for a tilted hand.
    """
    cfg = env.command_manager.get_term(command_name).cfg
    action_cfg = gripper_cfg(env, cfg)
    ee_frame = env.scene[cfg.ee_frame_name]
    half_gap = 0.5 * pad_gap(env, cfg)
    num_envs = half_gap.shape[0]
    # pad lower-end corners in the EE frame, shape [N, 8, 3]
    x = torch.tensor([-1.0, 1.0], device=env.device) * action_cfg.finger_pad_half_width
    y = torch.stack([half_gap, half_gap + ROBOTIQ_PAD_THICKNESS], dim=1)
    y = torch.cat([y, -y], dim=1)
    corners = torch.stack(
        [
            x.view(1, 2, 1).expand(num_envs, 2, 4),
            y.view(num_envs, 1, 4).expand(num_envs, 2, 4),
            (pad_shift(env, cfg) + action_cfg.finger_pad_half_length).view(num_envs, 1, 1).expand(num_envs, 2, 4),
        ],
        dim=-1,
    ).reshape(num_envs, 8, 3)
    ee_quat_w = ee_frame.data.target_quat_w.torch[:, 0].unsqueeze(1).expand(-1, 8, -1)
    corner_z = quat_apply(ee_quat_w, corners)[..., 2] + ee_frame.data.target_pos_w.torch[:, 0, 2].unsqueeze(1)
    return corner_z.amin(dim=1) - env.scene.env_origins[:, 2] < -max_penetration


def physics_contact(env: ManagerBasedRLEnv, contact_name: str) -> torch.Tensor:
    """Terminate when the monitored contact pair ``contact_name`` touched during the last step.

    The physics manager latches the pair after every physics substep (see
    :meth:`~..physics.coupler.NewtonClothDragCouplerManager.contact_latch`), so a touch within a step is
    not missed. This term reads the latch and clears it, so it must be the only reader of that pair.
    """
    physics = env.sim.physics_manager
    touched = physics.contact_latch(contact_name).clone()
    physics.clear_contact_latch(contact_name)
    return touched


def bank_start_advanced(
    env: ManagerBasedRLEnv,
    start_stages: tuple[str, ...] = ("grasp", "drag"),
    steps_after_entry: int = 0,
    command_name: str = "cloth_target",
) -> torch.Tensor:
    """End an episode that started in one of ``start_stages`` of a reset bank once it reaches the next stage.

    Use as a truncation (``time_out=True``) so the value of the reached stage is bootstrapped: bank starts become short,
    stage-local episodes and the critic chains the stages. Approach and goal starts are never ended by it.

    The time-out bootstrap uses the value of the state before the last action. With ``steps_after_entry`` zero that
    state is the one before the stage entry, whose value already anticipates the reward paid for the entry; one or more
    steps end the episode only that many steps after the entry, so the bootstrapped state lies past it.

    Args:
        env: The environment.
        start_stages: Bank start stages whose episodes are ended, out of ``"grasp"`` and ``"drag"``.
        steps_after_entry: Environment steps the episode keeps running in the reached stage before it ends.
        command_name: Name of the cloth target command.
    """
    from .stages import START_STAGE_NAMES, ClothDragStage

    command = env.command_manager.get_term(command_name)
    start = command.start_stage
    bank_start = torch.zeros_like(start, dtype=torch.bool)
    for name in start_stages:
        stage_id = START_STAGE_NAMES.index(name)
        if stage_id not in (int(ClothDragStage.GRASP), int(ClothDragStage.DRAG)):
            raise ValueError(f"start_stages must be out of 'grasp' and 'drag', got {start_stages}.")
        bank_start |= start == stage_id
    advanced = bank_start & (command.stage > start)
    if steps_after_entry > 0:
        advanced &= command.time_in_stage >= (steps_after_entry - 0.5) * env.step_dt
    return advanced

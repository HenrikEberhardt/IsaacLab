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
    """Terminate when the gripper presses down harder than ``max_force`` [N], as the real robot's safety stop.

    The force is the simulator's own contact force on the gripper: the coupler's feedback from everything the gripper
    proxies touch in the cloth solver (table, cloth, cube), see
    :meth:`~..physics.coupler.NewtonClothDragCouplerManager.proxy_body_force`. It is exactly zero in free air, unlike
    :func:`external_force_w`, which reads up to 35 N there. The term uses its vertical part, the push into the table
    and the cloth ``F_down = max(0, F_z)``, averaged over the policy step's physics substeps (33 ms), which filters
    single-substep impact spikes as the real robot's filtered estimate would.

    The term keeps the push in :attr:`force` for :func:`~.rewards.gripper_force_penalty`,
    :func:`~.rewards.gripper_force_kernel` and the critic; the termination manager computes before the reward and
    observation managers. At every reset it logs the finished episodes' force statistics under ``Force/``.
    """

    _LOG_THRESHOLDS = (10.0, 20.0)
    """Forces [N] whose share of exceeding episodes is logged."""

    _CONTACT_FORCE = 0.5
    """Push [N] above which a step counts as contact for the ``*_contact_mean`` logs."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        """Allocate the force and the episode statistics."""
        super().__init__(cfg, env)
        self.force = torch.zeros(env.num_envs, device=env.device)
        """Push of the gripper into the table and the cloth in the last policy step [N], zero after a reset."""
        self.force_peak = torch.zeros(env.num_envs, device=env.device)
        """Largest push of a single physics substep in the last policy step [N]."""
        self._episode_max = torch.zeros(env.num_envs, device=env.device)
        self._episode_peak = torch.zeros(env.num_envs, device=env.device)
        # per stage (GRASP, DRAG): force sum and step count, over all steps and over contact steps
        self._stage_sum = torch.zeros(env.num_envs, 2, device=env.device)
        self._stage_steps = torch.zeros(env.num_envs, 2, device=env.device)
        self._contact_steps = torch.zeros(env.num_envs, 2, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Log the finished episodes' force statistics and zero the force of the reset environments."""
        env_ids = slice(None) if env_ids is None else env_ids
        if getattr(self._env, "common_step_counter", 0) > 0 and self._episode_max[env_ids].numel() > 0:
            episode_max = self._episode_max[env_ids]
            log = {
                "Force/episode_max": episode_max.mean().item(),
                "Force/episode_substep_peak": self._episode_peak[env_ids].mean().item(),
            }
            for threshold in self._LOG_THRESHOLDS:
                log[f"Force/episodes_over_{threshold:.0f}N"] = (episode_max > threshold).float().mean().item()
            stage_sum, steps, contact = self._stage_sum[env_ids], self._stage_steps[env_ids], self._contact_steps[env_ids]
            for k, name in enumerate(("grasp", "drag")):
                if (steps[:, k] > 0).any():
                    log[f"Force/{name}_mean"] = (stage_sum[:, k].sum() / steps[:, k].sum()).item()
                if (contact[:, k] > 0).any():
                    # steps below the contact force add (almost) nothing to the sum
                    log[f"Force/{name}_contact_mean"] = (stage_sum[:, k].sum() / contact[:, k].sum()).item()
            # own top-level prefix: the environment recreates the log dict before the managers reset
            self._env.extras.setdefault("log", {}).update(log)
        for buffer in (self.force, self.force_peak, self._episode_max, self._episode_peak):
            buffer[env_ids] = 0.0
        for buffer in (self._stage_sum, self._stage_steps, self._contact_steps):
            buffer[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv, max_force: float, command_name: str = "cloth_target") -> torch.Tensor:
        """Read the push of this policy step, update the statistics, and terminate where it exceeds ``max_force``.

        Args:
            env: The environment.
            max_force: Push that ends the episode [N]; ``math.inf`` for none.
            command_name: Name of the cloth target command.
        """
        from .stages import ClothDragStage

        physics = env.sim.physics_manager
        force, up_peak = physics.proxy_body_force()
        physics.clear_proxy_body_force()
        self.force[:] = torch.nan_to_num(force[:, 2], nan=0.0).clamp(min=0.0)
        self.force_peak[:] = torch.nan_to_num(up_peak, nan=0.0).clamp(min=0.0)
        self._episode_max = torch.maximum(self._episode_max, self.force)
        self._episode_peak = torch.maximum(self._episode_peak, self.force_peak)
        stage = env.command_manager.get_term(command_name).stage
        touching = (self.force > self._CONTACT_FORCE).float()
        for k, stage_id in enumerate((ClothDragStage.GRASP, ClothDragStage.DRAG)):
            in_stage = (stage == int(stage_id)).float()
            self._stage_sum[:, k] += self.force * in_stage
            self._stage_steps[:, k] += in_stage
            self._contact_steps[:, k] += in_stage * touching
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


def pad_lowest_height(env: ManagerBasedRLEnv, command_name: str = "cloth_target") -> torch.Tensor:
    """Height of the lowest finger-pad corner above the table top (z = 0) [m], shape [num_envs].

    Uses the corners of both pads' lower ends from the measured finger joint, so it holds in every gripper state (the
    pads move 13.5 mm along the EE frame's z-axis while closing) and for a tilted hand.

    Args:
        env: The environment.
        command_name: Name of the cloth target command.
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
    return corner_z.amin(dim=1) - env.scene.env_origins[:, 2]


def gripper_below_table(
    env: ManagerBasedRLEnv, max_penetration: float, command_name: str = "cloth_target"
) -> torch.Tensor:
    """Terminate when a finger pad's lower end is more than ``max_penetration`` [m] below the table top (z = 0).

    See :func:`pad_lowest_height`.
    """
    return pad_lowest_height(env, command_name) < -max_penetration


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

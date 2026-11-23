# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IK teacher (dx, dy, dz, dyaw, gripper) to joint-delta student (7 joint deltas, gripper)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.franka_cloth_drag.mdp.frames import top_down_yaw

from ..teacher.ik_action import TeacherTopDownIKAction, TeacherTopDownIKActionCfg, top_down_quat
from .base import ActionMapping, ActionMappingCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

NUM_ARM_JOINTS = 7
IK_STATE_DIM = NUM_ARM_JOINTS + 3 + 4 + 6 * NUM_ARM_JOINTS + 3 * NUM_ARM_JOINTS
"""Joint pos, EE pos and quat, Jacobian, lower and upper joint bounds and PD error cap."""

##
# Mapping
##


@configclass
class IKToJointDeltaMappingCfg(ActionMappingCfg):
    """Teacher IK parameters and the student's joint-delta scale."""

    class_name: str = f"{__name__}:IKToJointDeltaMapping"
    teacher_num_actions: int | None = 5

    obs_group: str = "ik_state"
    """Observation group with the conversion inputs."""

    scale_trans: tuple[float, float, float] = (0.015, 0.015, 0.015)
    """Teacher (dx, dy, dz) scale [m]."""

    scale_rot: float = 0.2
    """Teacher dyaw scale [rad]."""

    wrist_joint_limits: tuple[float, float] = (-2.0 * math.pi / 3.0, 2.0 * math.pi / 3.0)
    """Teacher wrist joint window [rad]."""

    z_min: float | None = 0.0193
    """Teacher EE target floor [m]; None for no floor."""

    ik_lambda: float = 0.1
    """Damping of the teacher's DLS IK."""

    joint_delta_scale: float | tuple[float, ...] | None = None
    """Student joint-delta scale [rad], one value or one per arm joint; None reads it from the student's action."""

    student_action_name: str = "arm_action"
    """Student arm action term whose scale the labels use when joint_delta_scale is None."""

    preserve_direction: bool = False
    """Scale over-range labels uniformly instead of clipping per joint."""

    def apply_to_env(self, env_cfg, teacher_arm_cfg: TeacherTopDownIKActionCfg) -> None:
        """Add the ik_state group and give the teacher its last action in its own action space."""
        env_cfg.observations.ik_state = IKStateObsCfg()
        env_cfg.observations.ik_state.ik_state = ObsTerm(func=ik_state, params={"teacher_arm_cfg": teacher_arm_cfg})
        env_cfg.observations.teacher.last_action = ObsTerm(
            func=teacher_last_action, params={"teacher_arm_cfg": teacher_arm_cfg}
        )


def ik_to_joint_delta_mapping_cfg(
    teacher_arm_cfg: TeacherTopDownIKActionCfg, joint_delta_scale: float | tuple[float, ...] | None = None, **kwargs
) -> IKToJointDeltaMappingCfg:
    """Mapping configuration that matches the given teacher arm action."""
    return IKToJointDeltaMappingCfg(
        scale_trans=teacher_arm_cfg.scale_trans,
        scale_rot=teacher_arm_cfg.scale_rot,
        wrist_joint_limits=teacher_arm_cfg.wrist_joint_limits,
        z_min=teacher_arm_cfg.z_min,
        ik_lambda=teacher_arm_cfg.controller.ik_params["lambda_val"],
        joint_delta_scale=joint_delta_scale,
        **kwargs,
    )


class IKToJointDeltaMapping(ActionMapping):
    """Label = the teacher's first-substep joint step divided by the student's joint-delta scale."""

    cfg_type = IKToJointDeltaMappingCfg

    @staticmethod
    def resolve_cfg(cfg: dict, env) -> dict:
        """Take the joint-delta scale from the student's arm action, so labels and actions cannot disagree."""
        if cfg.get("joint_delta_scale") is None:
            scale = env.action_manager.get_term(cfg.get("student_action_name", "arm_action")).cfg.scale
            if not isinstance(scale, float):
                raise ValueError(f"Expected one float joint-delta scale on the student's arm action, got {scale}.")
            cfg = {**cfg, "joint_delta_scale": scale}
        print(f"[INFO] Label joint-delta scale: {cfg['joint_delta_scale']} rad")
        return cfg

    def __call__(self, teacher_actions: torch.Tensor, obs: TensorDict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        labels, clipped = teacher_actions_to_joint_deltas(teacher_actions, obs[self.cfg.obs_group], self.cfg)
        return labels, {"label_clip_fraction": clipped.float().mean()}


def unpack_ik_state(ik_state: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split the ik_state observation [N, IK_STATE_DIM] into named parts."""
    n = NUM_ARM_JOINTS
    sizes = (n, 3, 4, 6 * n, n, n, n)
    names = ("joint_pos", "ee_pos", "ee_quat", "jacobian", "pos_lower", "pos_upper", "max_offset")
    parts = dict(zip(names, torch.split(ik_state, sizes, dim=-1)))
    parts["jacobian"] = parts["jacobian"].reshape(-1, 6, n)
    return parts


def teacher_joint_step(
    teacher_arm: torch.Tensor, ik_state: torch.Tensor, cfg: IKToJointDeltaMappingCfg
) -> torch.Tensor:
    """The teacher's first-substep joint target offset for raw arm actions [N, 4] [rad], shape [N, 7]."""
    state = unpack_ik_state(ik_state)
    joint_pos, ee_pos, ee_quat = state["joint_pos"], state["ee_pos"], state["ee_quat"]
    scale = torch.tensor((*cfg.scale_trans, cfg.scale_rot), device=teacher_arm.device)
    delta = teacher_arm * scale
    # pose target, as TeacherTopDownIKAction.process_actions
    if cfg.z_min is not None:
        delta[:, 2] = (ee_pos[:, 2] + delta[:, 2]).clamp(min=cfg.z_min) - ee_pos[:, 2]
    wrist_pos = joint_pos[:, -1]
    wrist_lower, wrist_upper = cfg.wrist_joint_limits
    delta[:, 3] = torch.clamp(delta[:, 3], wrist_pos - wrist_upper, wrist_pos - wrist_lower)
    yaw = top_down_yaw(ee_quat) + delta[:, 3]
    _, rotation = math_utils.compute_pose_error(ee_pos, ee_quat, ee_pos, top_down_quat(yaw), rot_error_type="axis_angle")
    pos_des, quat_des = math_utils.apply_delta_pose(ee_pos, ee_quat, torch.cat((delta[:, :3], rotation), dim=-1))
    # DLS step, as DifferentialIKController.compute
    pos_error, rot_error = math_utils.compute_pose_error(ee_pos, ee_quat, pos_des, quat_des, rot_error_type="axis_angle")
    jacobian = state["jacobian"]
    jacobian_t = jacobian.transpose(1, 2)
    damping = (cfg.ik_lambda**2) * torch.eye(6, device=jacobian.device)
    error = torch.cat((pos_error, rot_error), dim=-1).unsqueeze(-1)
    step = (jacobian_t @ torch.inverse(jacobian @ jacobian_t + damping) @ error).squeeze(-1)
    # uniform scaling to the bounds, as TeacherTopDownIKAction.apply_actions
    ratio = torch.maximum(torch.ones_like(step), step.abs() / state["max_offset"])
    room = torch.where(step > 0, state["pos_upper"] - joint_pos, joint_pos - state["pos_lower"]).clamp(min=0.0)
    ratio = torch.maximum(ratio, step.abs() / room.clamp(min=1e-9))
    return step / ratio.amax(dim=1, keepdim=True)


def teacher_actions_to_joint_deltas(
    teacher_actions: torch.Tensor, ik_state: torch.Tensor, cfg: IKToJointDeltaMappingCfg
) -> tuple[torch.Tensor, torch.Tensor]:
    """Labels (7 joint deltas in [-1, 1], gripper) [N, 8] and the mask of clipped joint deltas [N, 7]."""
    step = teacher_joint_step(teacher_actions[:, :4], ik_state, cfg)
    scale = torch.as_tensor(cfg.joint_delta_scale, dtype=step.dtype, device=step.device)
    arm = torch.nan_to_num(step / scale, nan=0.0, posinf=1.0, neginf=-1.0)
    clipped = arm.abs() > 1.0
    if cfg.preserve_direction:
        arm = arm / arm.abs().amax(dim=1, keepdim=True).clamp(min=1.0)
    return torch.cat((arm.clamp(-1.0, 1.0), teacher_actions[:, 4:5]), dim=-1), clipped


##
# Observations the mapping needs
##


@configclass
class IKStateObsCfg(ObsGroup):
    """Noise-free conversion inputs (not a network input); the term is set by apply_to_env."""

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


class ik_state(ManagerTermBase):
    """Conversion inputs from a private copy of the teacher's arm action, shape [N, IK_STATE_DIM]."""

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._teacher_arm = TeacherTopDownIKAction(cfg.params["teacher_arm_cfg"], env)

    def __call__(self, env: ManagerBasedEnv, teacher_arm_cfg: TeacherTopDownIKActionCfg) -> torch.Tensor:
        return self._teacher_arm.ik_state()


class teacher_last_action(ManagerTermBase):
    """The executed student action in the teacher's action space (EE twist / teacher scale, gripper), shape [N, 5]."""

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        teacher_arm_cfg: TeacherTopDownIKActionCfg = cfg.params["teacher_arm_cfg"]
        self._teacher_arm = TeacherTopDownIKAction(teacher_arm_cfg, env)
        self._scale = torch.tensor((*teacher_arm_cfg.scale_trans, teacher_arm_cfg.scale_rot), device=env.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        teacher_arm_cfg: TeacherTopDownIKActionCfg,
        arm_action_name: str = "arm_action",
        gripper_action_name: str = "gripper_action",
    ) -> torch.Tensor:
        joint_delta = env.action_manager.get_term(arm_action_name).processed_actions
        twist = (self._teacher_arm.frame_jacobian() @ joint_delta.unsqueeze(-1)).squeeze(-1)
        arm = torch.cat((twist[:, :3], twist[:, 5:6]), dim=-1) / self._scale  # x, y, z, yaw
        gripper = env.action_manager.get_term(gripper_action_name).raw_actions
        return torch.cat((arm, gripper), dim=-1)

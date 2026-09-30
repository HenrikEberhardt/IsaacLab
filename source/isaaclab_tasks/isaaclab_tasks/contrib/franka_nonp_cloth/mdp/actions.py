# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cloth grasp action for the Franka non-prehensile cloth task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction

from .cloth_grasp import GripperConstraint
from .top_down import top_down_quat, top_down_yaw

if TYPE_CHECKING:
    from isaaclab.assets import DeformableObject
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.sensors import FrameTransformer

    from .actions_cfg import ClothGraspActionCfg, TopDownDifferentialIKActionCfg


class ClothGraspAction(BinaryJointPositionAction):
    """Binary gripper action that grasps the cloth like the task's state machine.

    Once the commanded-closed fingers have closed to :attr:`ClothGraspActionCfg.grasp_opening`, the cloth vertices
    pinched between them are locked to the gripper by a :class:`GripperConstraint`, and close commands hold the fingers
    at the opening where they grasped. The grasp is tried once per close, so a gripper that closed away from the cloth
    cannot pick it up by touching it; it can try again once its fingers have opened to
    :attr:`ClothGraspActionCfg.release_opening`, which also releases a held cloth. Since the fingers take several
    steps to move between the two openings, single noisy gripper actions neither grasp nor drop the cloth. The gripper
    collides with the cloth again once its fingertip point is :attr:`ClothGraspActionCfg.contact_clearance` away from
    every cloth vertex.
    """

    cfg: ClothGraspActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: ClothGraspActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._cloth: DeformableObject = env.scene["deformable"]
        self._ee_frame: FrameTransformer = env.scene[cfg.ee_frame_name]
        self._finger_joint_ids, _ = self._asset.find_joints(self.cfg.joint_names, preserve_order=True)
        self.constraint = GripperConstraint(
            env,
            enabled=cfg.attach,
            particle_radius=getattr(self._cloth.cfg.spawn.physics_material, "particle_radius", 0.002),
            proximity=cfg.grasp_proximity,
        )
        # whether each gripper has tried to grasp since its fingers last opened
        self._grasp_tried = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # finger opening at which each grasp closed on the fold [m]
        self._held_opening = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._any_active = False
        # the kinematic targets of released vertices must be written once to free them
        self._targets_dirty = True

    """
    Properties.
    """

    @property
    def is_grasping(self) -> torch.Tensor:
        """Whether each gripper holds cloth vertices. Shape is (num_envs,)."""
        return self.constraint.active & self.constraint.grasped.any(dim=1)

    @property
    def finger_opening(self) -> torch.Tensor:
        """Opening of each finger [m]. Shape is (num_envs,)."""
        return self._asset.data.joint_pos.torch[:, self._finger_joint_ids[0]]

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        close = self._raw_actions[:, 0] < 0.0
        opening = self.finger_opening

        # fingers that have opened wide release the cloth and may grasp again
        opened = opening >= self.cfg.release_opening
        released = self.constraint.active & opened
        if released.any():
            self.constraint.deactivate(released.nonzero(as_tuple=False).squeeze(-1))
            self._targets_dirty = True
        self._grasp_tried &= ~opened

        # grasp the cloth once the closing fingers have pinched it
        grasping = close & ~self._grasp_tried & (opening <= self.cfg.grasp_opening)
        if grasping.any():
            grasp_ids = grasping.nonzero(as_tuple=False).squeeze(-1)
            self._grasp_tried[grasp_ids] = True
            self.constraint.activate(grasp_ids, self._ee_frame.data.target_pos_w.torch[grasp_ids, 0, :])
            empty_ids = grasp_ids[~self.constraint.grasped[grasp_ids].any(dim=1)]
            self.constraint.deactivate(empty_ids)
            self._held_opening[grasp_ids] = self._asset.data.joint_pos.torch[grasp_ids][:, self._finger_joint_ids]

        # the fingers stop colliding with held cloth, so closing keeps them where they grasped the fold
        if self.cfg.attach:
            self._processed_actions = torch.where(
                (self.is_grasping & close).unsqueeze(-1), self._held_opening, self._processed_actions
            )

        # let released grippers collide with the cloth again once they have left it
        clearing = self.constraint.contacts_disabled & ~self.constraint.active
        if clearing.any():
            clearing_ids = clearing.nonzero(as_tuple=False).squeeze(-1)
            tcp_pos_w = self._ee_frame.data.target_pos_w.torch[clearing_ids, 0, :]
            vertices = self._cloth.data.nodal_pos_w.torch[clearing_ids]
            clearance = (vertices - tcp_pos_w.unsqueeze(1)).norm(dim=-1).amin(dim=1)
            self.constraint.restore_contacts(clearing_ids[clearance > self.cfg.contact_clearance])

        self._any_active = bool(self.constraint.active.any())

    def apply_actions(self):
        super().apply_actions()
        # lock the grasped vertices to the gripper at its current pose
        if self.cfg.attach and (self._any_active or self._targets_dirty):
            self._cloth.write_nodal_kinematic_target_to_sim_index(self.constraint.kinematic_targets())
            self._targets_dirty = False

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids or slice(None)]
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        # the reset moves the arm away from the cloth, so the gripper can collide with it again
        self.constraint.deactivate(env_ids)
        self.constraint.restore_contacts(env_ids)
        self.constraint.grasped[env_ids] = False
        self._grasp_tried[env_ids] = False
        self._any_active = bool(self.constraint.active.any())
        self._targets_dirty = True


class TopDownDifferentialIKAction(DifferentialInverseKinematicsAction):
    """Differential IK action that keeps the hand pointing straight down and turns it only about the base z-axis.

    The action is ``(dx, dy, dz, dyaw)``, scaled by :attr:`TopDownDifferentialIKActionCfg.scale`: 
    
    Each step it commands the relative IK controller to the moved position with the hand pointing straight down. The yaw is clamped to :attr:`TopDownDifferentialIKActionCfg yaw_limits`.
    """

    cfg: TopDownDifferentialIKActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: TopDownDifferentialIKActionCfg, env: ManagerBasedEnv):
        if cfg.controller.command_type != "pose" or not cfg.controller.use_relative_mode:
            raise ValueError("TopDownDifferentialIKAction needs an IK controller that takes relative poses.")
        super().__init__(cfg, env)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 4

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = self.raw_actions * self._scale
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        if self.cfg.min_height is not None:
            # never command the end-effector frame below the height floor
            self._processed_actions[:, 2] = torch.maximum(
                self._processed_actions[:, 2], self.cfg.min_height - ee_pos_curr[:, 2]
            )
        yaw = (top_down_yaw(ee_quat_curr) + self._processed_actions[:, 3]).clamp(*self.cfg.yaw_limits)
        # rotation from the current orientation to pointing straight down at the new yaw
        _, rotation = math_utils.compute_pose_error(
            ee_pos_curr, ee_quat_curr, ee_pos_curr, top_down_quat(yaw), rot_error_type="axis_angle"
        )
        pose_delta = torch.cat([self._processed_actions[:, :3], rotation], dim=-1)
        self._ik_controller.set_command(pose_delta, ee_pos_curr, ee_quat_curr)

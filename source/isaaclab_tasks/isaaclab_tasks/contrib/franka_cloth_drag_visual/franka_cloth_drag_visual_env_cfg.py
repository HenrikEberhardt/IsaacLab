# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cloth-drag distillation envs: the teacher task with adaptive resets, the student's actions, inputs and camera."""

from __future__ import annotations

from isaaclab_visualizers.viser import ViserVisualizerCfg

from isaaclab.cloner import expand_env_regex_ns
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.franka_cloth_drag.config.franka.franka_cloth_drag_env_cfg import (
    ActorObsCfg,
    AdaptiveResetsCfg,
    DefaultResetsCfg,
    FrankaClothDragEnvCfg,
)

from .action_mapping import ik_to_joint_delta_mapping_cfg
from .domain_randomization import (
    NOMINAL_CLOTH_COLOR,
    NOMINAL_IMAGE_LATENCY_STEPS,
    add_nominal_visuals,
    add_visual_randomization,
)
from .student.actions import JointDeltaActionsCfg
from .student.camera import WRIST_CAMERA_CFG
from .student.observations import StudentImageObsCfg, StudentProprioObsCfg
from .teacher import TEACHER_ARM_CFG, apply_final_curriculum_values

##
# Teacher -> student action mapping (swap point: same controller = IdentityMappingCfg + teacher's actions)
##

ACTION_MAPPING_CFG = ik_to_joint_delta_mapping_cfg(
    TEACHER_ARM_CFG,  # the label scale is read from env.actions.arm_action.scale
    preserve_direction=True,  # 96.7 % vs 84.9 % with per-joint clipping (check_label_conversion.py, 2026-11-22)
)

##
# MDP settings
##


@configclass
class JointObservationsCfg:
    """Teacher and student proprio groups; the mapping adds its own groups."""

    teacher: ActorObsCfg = ActorObsCfg()
    student_proprio: StudentProprioObsCfg = StudentProprioObsCfg()


@configclass
class VisualObservationsCfg(JointObservationsCfg):
    """Joint groups and the wrist image stack."""

    student_image: StudentImageObsCfg = StudentImageObsCfg()


##
# Environment configuration
##


@configclass
class FrankaClothDragJointEnvCfg(FrankaClothDragEnvCfg):
    """Student action space and teacher observations without the camera (label conversion checks)."""

    observations: JointObservationsCfg = JointObservationsCfg()
    actions: JointDeltaActionsCfg = JointDeltaActionsCfg()
    events: AdaptiveResetsCfg = AdaptiveResetsCfg()
    curriculum = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 256
        self.rewards.pen_action_magnitude = None  # penalize the teacher's action; rewards are only logged
        self.rewards.pen_action_rate = None
        self.observations.teacher.enable_corruption = False  # the teacher labels from the exact privileged state
        apply_final_curriculum_values(self)  # hard final force and joint speed stops, no curriculum
        ACTION_MAPPING_CFG.apply_to_env(self, TEACHER_ARM_CFG)

    def play_mode(self):
        """Few envs, no corruption, task-start resets at nominal physics and timing, and a Viser view."""
        ManagerBasedRLEnvCfg.play_mode(self)  # the task's play_mode edits its curriculum
        self.events = DefaultResetsCfg()
        # nominal physics: no randomization events (robotiq_linkage stiffens the gripper and stays)
        for term in ("arm_gains", "arm_friction", "arm_armature", "cloth_material", "cloth_friction", "cube_mass"):
            setattr(self.events, term, None)
        # nominal timing: no command latency or drops, the gripper's measured 45 ms delay
        self.actions.arm_action.command_delay_range = (0, 0)
        self.actions.arm_action.hold_prob = 0.0
        gripper_drive = self.scene.robot.actuators["gripper"]
        gripper_drive.min_delay = gripper_drive.max_delay = 5  # physics steps at 120 Hz
        self.sim.visualizer_cfgs = [
            ViserVisualizerCfg(eye=(0.5 + 3.18, -3.18, 7.59), lookat=(0.5, 0.0, -0.2), focal_length=23.5, port=8756)
        ]


@configclass
class FrankaClothDragVisualEnvCfg(FrankaClothDragJointEnvCfg):
    """The joint env with the OAK wrist camera, the image stack and the visual randomization."""

    observations: VisualObservationsCfg = VisualObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.wrist_camera = WRIST_CAMERA_CFG.copy()
        add_visual_randomization(self)

    def play_mode(self):
        """Joint play settings, nominal images and a second Viser server with the wrist camera."""
        super().play_mode()
        add_nominal_visuals(self)  # play mode replaced the events; the camera keeps its nominal mount
        image = self.observations.student_image.rgb.params
        image["augmentation"] = None
        image["cloth_colors"] = NOMINAL_CLOTH_COLOR
        image["latency_steps"] = NOMINAL_IMAGE_LATENCY_STEPS
        image["hold_prob"] = 0.0
        self.sim.visualizer_cfgs = [*self.sim.visualizer_cfgs, wrist_camera_stream_cfg()]


def wrist_camera_stream_cfg(port: int = 8757, num_envs: int = 16) -> ViserVisualizerCfg:
    """Viser stream of the raw wrist render of the first envs (no recolor, latency or augmentation)."""
    return ViserVisualizerCfg(
        port=port,
        streaming_view=True,
        streaming_sensor_prim_path=expand_env_regex_ns(WRIST_CAMERA_CFG.prim_path),
        streaming_gt_types=("rgb",),
        streaming_envs=num_envs,
        max_visible_envs=num_envs,
        randomly_sample_visible_envs=False,
    )

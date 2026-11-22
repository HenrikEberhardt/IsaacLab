# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Distillation environments of the Franka cloth-drag task.

The cloth-drag task with the deploy stack's action space: relative joint targets on the arm (scale 0.05 rad) and the
binary Robotiq gripper. Next to the student's groups, the env provides the teacher's observation group and the
``ik_state`` group from which the distillation algorithm converts the IK teacher's actions into joint-delta labels.
"""

from __future__ import annotations

import math

from isaaclab_visualizers.viser import ViserVisualizerCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg, UniformNoiseCfg

from isaaclab_tasks.contrib.franka_cloth_drag import mdp as drag_mdp
from isaaclab_tasks.contrib.franka_cloth_drag.config.franka.franka_cloth_drag_env_cfg import (
    GRIPPER_ACTION_CFG,
    ActorObsCfg,
    DefaultResetsCfg,
    FrankaClothDragEnvCfg,
)

from . import mdp
from .camera import WRIST_CAMERA_CFG
from .teacher_cfg import STUDENT_JOINT_DELTA_SCALE, TEACHER_ARM_CFG, apply_teacher_terminations

##
# MDP settings
##


@configclass
class JointActionsCfg:
    """Relative joint targets on the arm and the binary Robotiq gripper (8 actions)."""

    arm_action = mdp.DelayedRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=STUDENT_JOINT_DELTA_SCALE,  # [rad] per action unit
        command_delay_range=(0, 4),  # physics steps at 120 Hz, as the teacher's IK target delay
        hold_prob=0.1,  # a dropped command keeps the previous offset
    )
    gripper_action = GRIPPER_ACTION_CFG


@configclass
class IKStateObsCfg(ObsGroup):
    """Noise-free inputs of the teacher-label conversion (not a network input)."""

    ik_state = ObsTerm(func=mdp.ik_state, params={"teacher_arm_cfg": TEACHER_ARM_CFG})

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class StudentProprioObsCfg(ObsGroup):
    """What the real robot reports: arm joints, the Robotiq width, the previous action and the target point."""

    joint_pos = ObsTerm(
        func=drag_mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])}
    )
    joint_vel = ObsTerm(
        func=drag_mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])}
    )
    # the raw Robotiq reading converted to metres, noisy as for the teacher
    gripper_width = ObsTerm(
        func=drag_mdp.gripper_width,
        noise=NoiseModelWithAdditiveBiasCfg(
            noise_cfg=GaussianNoiseCfg(std=0.0004),  # [m] Robotiq position resolution
            bias_noise_cfg=UniformNoiseCfg(n_min=-0.005, n_max=0.005, operation="abs"),  # [m] per episode
        ),
    )
    last_action = ObsTerm(func=drag_mdp.last_action)
    target_pos = ObsTerm(func=drag_mdp.generated_commands, params={"command_name": "cloth_target"})

    def __post_init__(self) -> None:
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class JointObservationsCfg:
    """Teacher, label-conversion and student observation groups."""

    teacher: ActorObsCfg = ActorObsCfg()
    ik_state: IKStateObsCfg = IKStateObsCfg()
    student_proprio: StudentProprioObsCfg = StudentProprioObsCfg()


##
# Environment configuration
##


@configclass
class FrankaClothDragJointEnvCfg(FrankaClothDragEnvCfg):
    """Cloth drag with the joint-delta action space, the teacher's observations and the label-conversion state."""

    observations: JointObservationsCfg = JointObservationsCfg()
    actions: JointActionsCfg = JointActionsCfg()
    events: DefaultResetsCfg = DefaultResetsCfg()
    curriculum = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 256
        # the teacher sees the executed action in its own action space, in place of the student's
        self.observations.teacher.last_action = ObsTerm(
            func=mdp.teacher_last_action, params={"teacher_arm_cfg": TEACHER_ARM_CFG}
        )
        # rewards are only logged; these two penalize the teacher's action
        self.rewards.pen_action_magnitude = None
        self.rewards.pen_action_rate = None
        apply_teacher_terminations(self)

    def play_mode(self):
        """Settings for playing the policy: few environments, no corruption and a Viser view."""
        ManagerBasedRLEnvCfg.play_mode(self)
        self.sim.visualizer_cfgs = [
            ViserVisualizerCfg(eye=(0.5 + 3.18, -3.18, 7.59), lookat=(0.5, 0.0, -0.2), focal_length=23.5, port=8756)
        ]


##
# Visual student
##

CLOTH_COLORS = {
    "palette": [
        (0.21, 0.37, 0.56),  # the lab towel
        (0.33, 0.50, 0.70),  # lighter
        (0.14, 0.25, 0.42),  # darker
        (0.29, 0.37, 0.50),  # greyer
        (0.14, 0.34, 0.64),  # more saturated
        (0.19, 0.41, 0.55),  # towards teal
        (0.26, 0.33, 0.59),  # towards violet
    ],
    "brightness": (0.7, 1.3),
    "jitter": 0.1,
}
"""Per-episode cloth color (sRGB): variations of the lab's blue towel, from Joran's visual cloth tasks. Refit from a
wrist-camera frame of the real cloth."""

NEUTRAL_TINTS = [(1.0, 1.0, 1.0), (1.0, 0.97, 0.93), (0.95, 0.97, 1.0)]

SHAPE_COLORS = {
    # slight brightness and white-balance shifts of the authored colors
    "Robot": {"palette": NEUTRAL_TINTS, "brightness": (0.85, 1.05), "jitter": 0.02, "tint": True},
    # orange shades, a placeholder until the real cube's color is fixed
    "Cube": {
        "palette": [(0.95, 0.45, 0.10), (0.90, 0.38, 0.08), (1.00, 0.55, 0.15)],
        "brightness": (0.8, 1.2),
        "jitter": 0.05,
    },
    # beech-like solid colors around the real table's sRGB (0.68, 0.61, 0.51), until textured table tops are ported
    "Table": {
        "palette": [(0.68, 0.61, 0.51), (0.72, 0.62, 0.48), (0.64, 0.58, 0.50), (0.70, 0.64, 0.56)],
        "brightness": (0.7, 1.1),
        "jitter": 0.05,
    },
}
"""Per-episode shape colors (sRGB) of the rendered scene prims."""


@configclass
class StudentImageObsCfg(ObsGroup):
    """The wrist RGB image in [0, 1], channel first, delayed and occasionally repeated like the real camera."""

    rgb = ObsTerm(
        func=mdp.image_rgb_deploy,
        params={
            "sensor_cfg": SceneEntityCfg("wrist_camera"),
            "mean": 0.0,
            # 66-100 ms at 30 Hz: about 66 ms from capture to publish, plus up to one frame of waiting
            "latency_steps": (2, 3),
            "hold_prob": 0.1,  # a dropped frame repeats the previous one
            "cloth_colors": CLOTH_COLORS,
        },
        # photometric camera differences, drawn per episode (off in play mode)
        noise=mdp.RgbAugmentationCfg(),
    )

    def __post_init__(self) -> None:
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class VisualObservationsCfg(JointObservationsCfg):
    """The joint env's groups and the wrist image."""

    student_image: StudentImageObsCfg = StudentImageObsCfg()


@configclass
class FrankaClothDragVisualEnvCfg(FrankaClothDragJointEnvCfg):
    """The joint env with the OAK wrist camera and the student's image observation."""

    observations: VisualObservationsCfg = VisualObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.wrist_camera = WRIST_CAMERA_CFG.copy()
        # the mount is perturbed per episode, to tolerate calibration error
        self.events.randomize_wrist_camera = EventTerm(
            func=mdp.randomize_camera_offset,
            mode="reset",
            params={
                "sensor_cfg": SceneEntityCfg("wrist_camera"),
                "parent_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
                "pos_range": 0.01,  # [m] per axis
                "rot_range": math.radians(1.0),  # [rad] per axis
            },
        )
        # the light is shared by all environments, so it moves on one global timer
        self.events.randomize_light = EventTerm(
            func=mdp.randomize_light,
            mode="interval",
            interval_range_s=(2.0, 5.0),
            is_global_time=True,
            params={"sensor_cfg": SceneEntityCfg("wrist_camera"), "elevation": (math.radians(35.0), math.radians(88.0))},
        )
        self.events.randomize_colors = EventTerm(
            func=mdp.randomize_shape_colors,
            mode="reset",
            params={"sensor_cfg": SceneEntityCfg("wrist_camera"), "colors": SHAPE_COLORS},
        )

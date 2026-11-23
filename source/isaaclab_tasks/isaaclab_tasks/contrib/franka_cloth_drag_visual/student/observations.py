# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# image term follows srl-ethz/IsaacLab-Cloth (cloth-fold-environments), src/Cloth_Manipulation/visual/mdp.py

"""Student inputs: proprioception and wrist RGB frames, both over the last NUM_FRAMES policy steps.

Both groups use the observation manager's history, which a LEAPP export keeps as graph state: the robot feeds one
camera frame and one robot state per tick, and the exported graph stacks them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.buffers import DelayBuffer
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg, UniformNoiseCfg

from isaaclab_tasks.contrib.franka_cloth_drag import mdp as drag_mdp

from ..domain_randomization import (
    CLOTH_COLORS,
    IMAGE_AUGMENTATION_CFG,
    IMAGE_HOLD_PROB,
    IMAGE_LATENCY_STEPS,
    ClothRecolor,
    RgbAugmentationCfg,
)
from .export import leapp_exporting

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import ObservationTermCfg

NUM_FRAMES = 5
"""Policy steps of history in both student groups."""


class wrist_rgb(ManagerTermBase):
    """Wrist RGB frame as the real camera delivers it, uint8 [N, 3, H, W].

    In training the frame is recolored, delayed, sometimes dropped (the previous frame repeats) and augmented, once,
    before the group's history stacks it. A LEAPP export reads the raw frame instead, as the camera's float32
    0-255 input (``state/camera/image``), since on the robot all of this is physical.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        params = cfg.params
        self._latency = params.get("latency_steps", (0, 0))
        self._delay = DelayBuffer(self._latency[1], env.num_envs, env.device) if self._latency[1] > 0 else None
        colors = params.get("cloth_colors")
        self._recolor = ClothRecolor(colors, env.num_envs, env.device) if colors is not None else None
        augmentation: RgbAugmentationCfg | None = params.get("augmentation")
        self._augment = augmentation.class_type(augmentation, env.num_envs, env.device) if augmentation else None
        self._last: torch.Tensor | None = None  # previous output, repeated on a dropped frame
        self._fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.reset(None)

    def reset(self, env_ids=None):
        env_ids = torch.arange(self.num_envs, device=self.device)[slice(None) if env_ids is None else env_ids]
        self._fresh[env_ids] = True  # the first frame after a reset is never a dropped one
        if self._delay is not None:
            self._delay.reset(env_ids)
            lags = torch.randint(self._latency[0], self._latency[1] + 1, (len(env_ids),), device=self.device)
            self._delay.set_time_lag(lags.int(), env_ids)
        if self._recolor is not None:
            self._recolor.reset(env_ids)
        if self._augment is not None:
            self._augment.reset(env_ids)

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg,
        latency_steps: tuple[int, int] = (0, 0),
        hold_prob: float = 0.0,
        cloth_colors: dict | None = None,
        augmentation: RgbAugmentationCfg | None = None,
    ) -> torch.Tensor:
        if leapp_exporting():
            return self._export_input(env, sensor_cfg)
        raw = env.scene.sensors[sensor_cfg.name].data.output["rgb"].torch[..., :3]  # uint8 [N, H, W, 3]
        if self._recolor is not None:
            raw = self._recolor(env, sensor_cfg, raw)
        if self._delay is not None:
            raw = self._delay.compute(raw)
        frame = raw.permute(0, 3, 1, 2)
        if self._augment is not None:
            frame = (self._augment(frame.float() / 255.0) * 255.0).round().to(torch.uint8)
        if self._last is None:
            self._last = frame
        hold = (torch.rand(self.num_envs, device=self.device) < hold_prob) & ~self._fresh  # dropped frame
        frame = torch.where(hold[:, None, None, None], self._last, frame)
        self._last = frame
        self._fresh[:] = False
        return frame

    @staticmethod
    def _export_input(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
        """The raw frame as the graph's float32 0-255 [N, H, W, 3] camera input, returned as uint8 [N, 3, H, W].

        Isaac ROS Deploy feeds rgb8 images as float32, so the input is declared on the float frame. The real scene is
        read past the export proxy, which would declare the uint8 buffer instead. The graph casts the frame back to
        uint8, the dtype of the training frames and of the exported history state.
        """
        from leapp import TensorSemantics, annotate

        from isaaclab.utils.leapp import build_state_connection

        real_env = env.unwrapped
        raw = real_env.scene.sensors[sensor_cfg.name].data.output["rgb"].torch[..., :3]
        image = annotate.input_tensors(
            str(real_env.spec.id),
            TensorSemantics(
                name=f"{sensor_cfg.name}_rgb",
                ref=raw.float(),
                kind="state/camera/image",
                extra=build_state_connection(sensor_cfg.name, "output.rgb"),
            ),
        )
        return image.round().clamp(0.0, 255.0).to(torch.uint8).permute(0, 3, 1, 2)


@configclass
class StudentProprioObsCfg(ObsGroup):
    """What the real robot reports, over the last NUM_FRAMES policy steps."""

    joint_pos = ObsTerm(
        func=drag_mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])}
    )
    joint_vel = ObsTerm(
        func=drag_mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])}
    )
    gripper_width = ObsTerm(  # raw Robotiq reading in metres
        func=drag_mdp.gripper_width,
        noise=NoiseModelWithAdditiveBiasCfg(
            noise_cfg=GaussianNoiseCfg(std=0.0004),  # [m] position resolution
            bias_noise_cfg=UniformNoiseCfg(n_min=-0.005, n_max=0.005, operation="abs"),  # [m] per episode
        ),
    )
    last_action = ObsTerm(func=drag_mdp.last_action)
    target_pos = ObsTerm(func=drag_mdp.generated_commands, params={"command_name": "cloth_target"})

    def __post_init__(self) -> None:
        self.enable_corruption = True
        self.concatenate_terms = True
        self.history_length = NUM_FRAMES
        self.flatten_history_dim = True


@configclass
class StudentImageObsCfg(ObsGroup):
    """The last NUM_FRAMES wrist frames, uint8 [N, NUM_FRAMES, 3, H, W], oldest first; the model folds them into
    3 * NUM_FRAMES channels."""

    rgb = ObsTerm(
        func=wrist_rgb,
        params={
            "sensor_cfg": SceneEntityCfg("wrist_camera"),
            "latency_steps": IMAGE_LATENCY_STEPS,
            "hold_prob": IMAGE_HOLD_PROB,
            "cloth_colors": CLOTH_COLORS,
            "augmentation": IMAGE_AUGMENTATION_CFG,  # set None for clean images (play mode)
        },
    )

    def __post_init__(self) -> None:
        self.enable_corruption = False  # the term applies its own augmentation
        self.concatenate_terms = True
        self.history_length = NUM_FRAMES
        self.flatten_history_dim = False  # keep [N, T, 3, H, W] for the CNN

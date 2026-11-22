# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# oak_camera_cfg follows srl-ethz/IsaacLab-Cloth (branch cloth-fold-environments), src/Cloth_Manipulation/assets/robotiq.py,
# which follows mmichelis/IsaacLab-Soft (branch liftsoft-robotiq), src/isaaclab_soft/assets/robotiq_oak.py.

"""The OAK-1 Lite W wrist camera on the Robotiq gripper."""

from __future__ import annotations

from pxr import Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.sim import OpenCvPinholeDistortionCfg

from isaaclab_newton.renderers import NewtonWarpRendererCfg

from isaaclab_tasks.contrib.franka_cloth_drag.constrained_gripper.robotiq import (
    FRANKA_PANDA_ROBOTIQ_USD,
    ROBOTIQ_BASE_PATH,
)

OAK_CAMERA_PATH = ROBOTIQ_BASE_PATH + "/OAKHardware/Mount/CameraHousing/OpticalCamera"
"""Prim path of the OAK's optical camera below the robot's root prim."""


def oak_camera_cfg(base: CameraCfg) -> CameraCfg:
    """The OAK's optical camera of the robot asset as a camera, keeping ``base``'s acquisition and render settings.

    The pose relative to the Robotiq ``base_link`` and the calibrated OpenCV lens (ChArUco at 1352x1014) are read from
    the asset. The camera prim already exists there, so it is not spawned again, but Newton's ray tracer reads its lens
    model from ``spawn``.
    """
    stage = Usd.Stage.Open(FRANKA_PANDA_ROBOTIQ_USD)
    root = stage.GetDefaultPrim()
    camera = root.GetPrimAtPath(OAK_CAMERA_PATH)
    mount = root.GetPrimAtPath(ROBOTIQ_BASE_PATH)
    if not camera.IsA(UsdGeom.Camera) or not mount:
        raise ValueError(f"Missing OAK camera or mount body in {FRANKA_PANDA_ROBOTIQ_USD}")
    if camera.GetAttribute("omni:lensdistortion:model").Get() != "opencvPinhole":
        raise ValueError(f"Expected a calibrated OpenCV pinhole camera in {FRANKA_PANDA_ROBOTIQ_USD}")
    prefix = "omni:lensdistortion:opencvPinhole:"
    names = ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2", "s1", "s2", "s3", "s4")
    lens = OpenCvPinholeDistortionCfg(
        image_size=tuple(camera.GetAttribute(prefix + "imageSize").Get()),
        **{name: camera.GetAttribute(prefix + name).Get() for name in names},
    )
    cache = UsdGeom.XformCache()
    relative = cache.GetLocalToWorldTransform(camera) * cache.GetLocalToWorldTransform(mount).GetInverse()
    rotation = relative.ExtractRotationQuat()
    offset = CameraCfg.OffsetCfg(
        pos=tuple(relative.ExtractTranslation()),
        rot=(*rotation.GetImaginary(), rotation.GetReal()),  # (x, y, z, w)
        convention="opengl",
    )
    spawn = base.spawn if base.spawn is not None else sim_utils.PinholeCameraCfg()
    return base.replace(
        prim_path="{ENV_REGEX_NS}/Robot/" + OAK_CAMERA_PATH,
        offset=offset,
        spawn=spawn.replace(distortion=lens),
    )


WRIST_CAMERA_CFG = oak_camera_cfg(
    CameraCfg(
        prim_path="",
        # the camera rides on the hand, so the renderer needs its pose refreshed every capture
        update_latest_camera_pose=True,
        # the segmentation output is unused: requesting it makes the renderer fill the shape-index buffer that
        # image_rgb_deploy reads to find and recolor the cloth
        data_types=["rgb", "instance_segmentation"],
        # the raw, distorted OAK frame resized to this resolution
        width=160,
        height=120,
        spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.01, 3.0)),  # [m]
        renderer_cfg=NewtonWarpRendererCfg(enable_shadows=True, colorize_instance_segmentation=False),
    )
)
"""The OAK-1 Lite W on the Robotiq mount, 160x120 RGB at the policy rate."""

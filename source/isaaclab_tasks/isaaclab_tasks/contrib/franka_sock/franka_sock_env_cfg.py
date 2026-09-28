# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka sock lifting environment."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonSoftContactCfg,
    VBDSolverCfg,
)
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxCollisionCfg, PhysxDeformableBodyPropertiesCfg
from isaaclab_physx.sim.spawners.materials import PhysxSurfaceDeformableBodyMaterialCfg
from isaaclab_visualizers.kit import KitVisualizerCfg
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg
from isaaclab_visualizers.viser import ViserVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.cloner import expand_env_regex_ns
from isaaclab.cloner import random as random_clone_strategy
from isaaclab.envs import VideoRecorderCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

from isaaclab_tasks.core.lift import mdp
from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import (
    TABLE_SPAWN_CFG,
    FrankaSoftEnvCfg,
    _FrankaSoftSceneCfg,
)
from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import (
    RewardsCfg as FrankaSoftRewardsCfg,
)
from isaaclab_tasks.utils import PresetCfg
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

if TYPE_CHECKING:
    from pxr import Usd

##
# Sock geometry
##

# tube dimensions, spawn and resting heights, and lift clearance [m]
SOCK_RADIUS = 0.03
SOCK_LENGTH = 0.15
SOCK_NUM_SEGMENTS = 28
SOCK_SPAWN_HEIGHT = SOCK_RADIUS + 0.002
SOCK_RESTING_HEIGHT = 0.0113
SOCK_LIFT_CLEARANCE = 0.10
SOCK_VISUAL_MATERIAL = sim_utils.MdlFileCfg(
    mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Textiles/Cloth_Gray.mdl",
    project_uvw=True,
    texture_scale=(4.0, 4.0),  # tile the cloth texture 4x over the sock
)

##
# Scene definition
##


@configclass
class PhysicsCfg(PresetCfg):
    """Preset physics configurations for the sock.

    The parent task's presets with a larger VBD particle contact buffer (1024 vs 256) and a larger
    PhysX found/lost pair capacity (2**22). The Newton preset is tuned; the PhysX preset is provided
    for parity and is untuned.
    """

    newton_mjwarp_vbd_proxy: NewtonCfg = NewtonCfg(
        solver_cfg=CouplerProxyCfg(
            entries=[
                CouplerEntryCfg(
                    name="rigid",
                    solver_cfg=MJWarpSolverCfg(
                        cone="elliptic",
                        ls_iterations=30,
                        integrator="implicitfast",
                    ),
                    bodies=[r"/World/envs/env_[^/]+/Robot"],
                ),
                CouplerEntryCfg(
                    name="soft",
                    solver_cfg=VBDSolverCfg(iterations=10, rigid_body_particle_contact_buffer_size=1024),
                    all_particles=True,
                    include_static_shapes=True,
                ),
            ],
            proxies=[
                CouplerProxyMappingCfg(
                    source="rigid",
                    destination="soft",
                    bodies=[
                        r"/World/envs/env_[^/]+/Robot/Geometry/.*panda_hand",
                        r"/World/envs/env_[^/]+/Robot/Geometry/.*panda_(left|right)finger",
                    ],
                    collide_interval=1,
                    collision_pipeline=NewtonCollisionPipelineCfg(
                        enable_rigid_soft_full_surface_contact=True,
                    ),
                )
            ],
            iterations=1,
        ),
        soft_contact_cfg=NewtonSoftContactCfg(
            soft_contact_ke=8.0e3,
            soft_contact_kd=1.0e-2,
            soft_contact_mu=10.0,
        ),
        num_substeps=2,
    )

    isaacsim_physx: PhysxCfg = PhysxCfg(gpu_found_lost_pairs_capacity=2**22)

    physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx)

    default = newton_mjwarp_vbd_proxy


@configclass
class DeformableCfg(PresetCfg):
    """Preset configurations for the sock."""

    newton_mjwarp_vbd_proxy: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Deformable",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.45, 0.0, SOCK_SPAWN_HEIGHT)),
        spawn=sim_utils.MeshTubeCfg(
            radius=SOCK_RADIUS,
            height=SOCK_LENGTH,
            axis="X",
            num_segments=SOCK_NUM_SEGMENTS,
            cap_start=True,
            cap_end=False,
            edge_refinement=1.0,
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            visual_material=SOCK_VISUAL_MATERIAL,
            physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
                density=1.0,
                particle_radius=0.002,
                tri_ke=5e2,
                tri_ka=5e2,
                tri_kd=1e-3,
                # Stiffness of the edge to prevent the sock from collapsing.
                edge_ke=2.0,
                edge_kd=5e-3,
            ),
        ),
    )

    physx: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Deformable",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.45, 0.0, SOCK_SPAWN_HEIGHT)),
        spawn=sim_utils.MeshTubeCfg(
            radius=SOCK_RADIUS,
            height=SOCK_LENGTH,
            axis="X",
            num_segments=SOCK_NUM_SEGMENTS,
            cap_start=True,
            cap_end=False,
            edge_refinement=1.0,
            deformable_props=PhysxDeformableBodyPropertiesCfg(),
            collision_props=[PhysxCollisionCfg(rest_offset=0.002, contact_offset=0.01)],
            visual_material=SOCK_VISUAL_MATERIAL,
            physics_material=PhysxSurfaceDeformableBodyMaterialCfg(
                density=1000.0,
                surface_thickness=0.001,
                poissons_ratio=0.25,
                youngs_modulus=1e6,
                surface_bend_stiffness=1e6,
                elasticity_damping=1e-1,
                bend_damping=1e-1,
                static_friction=10.0,
                dynamic_friction=10.0,
            ),
        ),
    )
    isaacsim_physx = physx

    default = newton_mjwarp_vbd_proxy


@configclass
class _WorldMdlFileCfg(sim_utils.MdlFileCfg):
    """MDL material projected in world space, so the texture keeps its scale on stretched cuboids."""

    project_uvw: bool = True

    world_or_object: bool = True
    """Whether the OmniPBR UVW projection uses world instead of object coordinates."""


##
# Room
##

ROOM_SIZE = 3.0
"""Side length of the square room around each environment, used as the environment spacing [m]."""

ROOM_CENTER_X = 0.5
"""Room center along x in the environment frame, over the table rather than the robot base [m]."""

FLOOR_Z = -1.05
"""Floor height below the table top at z = 0 [m]."""

# each room draws one of these wall materials at random
WALL_MATERIAL_PATHS = [
    "{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wall_Board/Plaster.mdl",
    "{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wall_Board/Gypsum.mdl",
    "{NVIDIA_NUCLEUS_DIR}/Materials/Base/Masonry/Brick_Wall_Red.mdl",
    "{NVIDIA_NUCLEUS_DIR}/Materials/Base/Masonry/Brick_Wall_Brown.mdl",
    "{NVIDIA_NUCLEUS_DIR}/Materials/Base/Masonry/Concrete_Smooth.mdl",
]


@sim_utils.clone
def _spawn_room_walls(
    prim_path: str,
    cfg: _RoomWallsCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn the four visual-only walls of a square room around a root prim on the floor.

    Returns:
        The room root prim.
    """
    prim = sim_utils.create_prim(prim_path, "Xform", translation=translation, orientation=orientation)
    offset = 0.5 * (cfg.size - cfg.thickness)
    x_wall_size = (cfg.thickness, cfg.size - 2 * cfg.thickness, cfg.height)
    y_wall_size = (cfg.size, cfg.thickness, cfg.height)
    z = 0.5 * cfg.height
    walls = {
        "WallNegX": (x_wall_size, (-offset, 0.0, z)),
        "WallPosX": (x_wall_size, (offset, 0.0, z)),
        "WallNegY": (y_wall_size, (0.0, -offset, z)),
        "WallPosY": (y_wall_size, (0.0, offset, z)),
    }
    for name, (size, pos) in walls.items():
        sim_utils.spawn_cuboid(f"{prim_path}/{name}", sim_utils.CuboidCfg(size=size), translation=pos)
    if cfg.visual_material is not None:
        material_path = f"{prim_path}/material"
        cfg.visual_material.func(material_path, cfg.visual_material)
        sim_utils.bind_visual_material(prim_path, material_path)
    return prim


@configclass
class _RoomWallsCfg(sim_utils.SpawnerCfg):
    """Four visual-only walls of a square room sharing one material, see :func:`_spawn_room_walls`."""

    func: Callable = _spawn_room_walls

    size: float = ROOM_SIZE
    """Outer side length of the room [m]."""

    thickness: float = 0.02
    """Thickness of a wall panel [m]."""

    height: float = 1.0 - FLOOR_Z
    """Wall height, from the floor to 1 m above the table top [m]."""

    visual_material: sim_utils.VisualMaterialCfg | None = None
    """Material shared by all four walls. Defaults to None, in which case no material is bound."""


TABLE_VISUAL_MATERIAL = _WorldMdlFileCfg(mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Stone/Marble_Smooth.mdl")

# floor tile per room, replacing the default ground plane, which has no ``visual_material`` field
GROUND_THICKNESS = 0.05
GROUND_SPAWN_CFG = sim_utils.CuboidCfg(
    size=(ROOM_SIZE, ROOM_SIZE, GROUND_THICKNESS),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    visual_material=_WorldMdlFileCfg(mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Masonry/Concrete_Rough.mdl"),
)


@configclass
class FrankaSockSceneCfg(_FrankaSoftSceneCfg):
    """Scene for the Franka sock environment."""

    deformable: DeformableCfg = DeformableCfg()

    # The 4 walls of a room are one asset, so they share one material
    walls: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Walls",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(ROOM_CENTER_X, 0.0, FLOOR_Z)),
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[_RoomWallsCfg(visual_material=_WorldMdlFileCfg(mdl_path=path)) for path in WALL_MATERIAL_PATHS]
        ),
    )

    # ttiled ground plane. Each room has own ground plate
    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Floor",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(ROOM_CENTER_X, 0.0, FLOOR_Z - 0.5 * GROUND_THICKNESS)),
        spawn=GROUND_SPAWN_CFG,
    )

    # directional light added to the inherited dome light
    sun_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/sunLight",
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.3007, 0.0, 0.0, 0.9537)),
        spawn=sim_utils.DistantLightCfg(
            intensity=1500.0,
            angle=0.3,
            color_temperature=5500.0,
            enable_color_temperature=True,
        ),
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        # stiffer gripper so it squeezes the thin sock hard enough to hold it
        self.robot.actuators["panda_hand"].stiffness = 2000.0

        # random wall material per room; the default ``sequential`` would cycle through them in env order
        self.clone_cfg.clone_strategy = random_clone_strategy

        # show the table; the success marker draws a rim around it
        self.table.spawn = self.table.spawn.replace(visible=True, visual_material=TABLE_VISUAL_MATERIAL)

        # softer, warmer sky than the default preview-render sky
        self.sky_light.spawn.intensity = 250.0
        self.sky_light.spawn.texture_file = (
            f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/kloofendal_48d_partly_cloudy_4k.hdr"
        )


@configclass
class FrankaSockScenePresetCfg(PresetCfg):
    """Preset config for the Franka sock scene."""

    # half the cloth task's 2048 environments, since the sock mesh has more particles
    newton_mjwarp_vbd_proxy: FrankaSockSceneCfg = FrankaSockSceneCfg(
        num_envs=1024, env_spacing=ROOM_SIZE, replicate_physics=True
    )

    # Isaac Sim PhysX does not support replicating physics for deformable objects
    physx: FrankaSockSceneCfg = FrankaSockSceneCfg(num_envs=1024, env_spacing=ROOM_SIZE, replicate_physics=False)
    isaacsim_physx = physx

    default = newton_mjwarp_vbd_proxy


##
# MDP settings
##

# Width of the success rim that shows around the table's edges [m].
SUCCESS_RIM_WIDTH = 0.05

# Success marker: a green box slightly larger than the table shows as a rim once the goal is reached.
# The failure prototype is a hidden 1 mm cube, since Newton visualizers ignore ``visible=False``.
SUCCESS_VISUALIZER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/SuccessMarkers",
    markers={
        "failure": sim_utils.CuboidCfg(size=(0.001, 0.001, 0.001), visible=False),
        "success": TABLE_SPAWN_CFG.replace(
            size=(
                TABLE_SPAWN_CFG.size[0] + 2 * SUCCESS_RIM_WIDTH,
                TABLE_SPAWN_CFG.size[1] + 2 * SUCCESS_RIM_WIDTH,
                TABLE_SPAWN_CFG.size[2] - 0.01,
            ),
            collision_props=None,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.05, 0.75, 0.15), emissive_color=(0.0, 0.35, 0.05), roughness=0.5
            ),
            visible=True,
        ),
    },
)


@configclass
class FrankaSockRewardsCfg(FrankaSoftRewardsCfg):
    """Rewards for the Franka sock environment."""

    reaching_deformable = RewTerm(
        func=mdp.deformable_ee_distance,
        params={"std": 0.1, "asset_cfg": SceneEntityCfg("deformable")},
        weight=5.0,
    )

    lifting_deformable = RewTerm(
        func=mdp.deformable_lifting,
        params={
            "std": 0.1,
            "minimal_height": SOCK_RESTING_HEIGHT + SOCK_LIFT_CLEARANCE,
            "asset_cfg": SceneEntityCfg("deformable"),
        },
        weight=5.0,
    )


##
# Environment configuration
##


@configclass
class FrankaSockEnvCfg(FrankaSoftEnvCfg):
    """Manager-based RL environment: Franka Panda lifting a sock."""

    scene: FrankaSockScenePresetCfg = FrankaSockScenePresetCfg()
    rewards: FrankaSockRewardsCfg = FrankaSockRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # sock physics presets, see :class:`PhysicsCfg`
        self.sim.physics = PhysicsCfg()

        # Include the success marker that draws a green rim
        self.commands.deformable_pose.success_visualizer_cfg = SUCCESS_VISUALIZER_CFG

        # Set up 3 visualizer that we are using to have the right camera position for video creation.
        self.sim.visualizer_cfgs = [
            KitVisualizerCfg(
                # steep diagonal view, 9 m away at 60 deg, that keeps rooms in every image corner with 16 envs
                eye=(ROOM_CENTER_X + 3.18, -3.18, 7.59),
                lookat=(ROOM_CENTER_X, 0.0, -0.2),
                headless=True,  # no display attached to the workstation
                max_visible_envs=16,
                randomly_sample_visible_envs=False,  # render envs 0..15 instead of a random sample
            ),
            NewtonGLVisualizerCfg(
                eye=(ROOM_CENTER_X + 3.18, -3.18, 7.59),
                lookat=(ROOM_CENTER_X, 0.0, -0.2),
                focal_length=23.5,  # Kit ignores focal_length; 23.5 mm matches Kit's default lens
                headless=True,
                max_visible_envs=16,
                randomly_sample_visible_envs=False,
            ),
            ViserVisualizerCfg(
                eye=(ROOM_CENTER_X + 3.18, -3.18, 7.59),
                lookat=(ROOM_CENTER_X, 0.0, -0.2),
                focal_length=23.5,
                port=8750,  # off the default 8080 to avoid clashes on the shared workstation
            ),
        ]

    def play_mode(self):
        super().play_mode()
        # show the end-effector frame when inspecting a policy; too costly to leave on for training.
        # play mode runs after preset resolution, so the scene is the concrete cfg here.
        self.scene.ee_frame.debug_vis = True


##
# Camera variants
##

# Path to the Franka hand prim, use as the parent for the wrist camera
PANDA_HAND_PRIM_PATH = (
    "{ENV_REGEX_NS}/Robot/Geometry/panda_link0/panda_link1/panda_link2/panda_link3/"
    "panda_link4/panda_link5/panda_link6/panda_link7/panda_hand"
)

# Depth Span of the cameras in meters
WRIST_CAMERA_DEPTH_RANGE = (0.05, 1.0)
HEAD_CAMERA_DEPTH_RANGE = (0.3, 2.0)

# RealSense D405-like camera on the hand (7 cm off the gripper axis, 3 cm below the flange and pitched
# 25 deg toward the axis)
WRIST_CAMERA_CFG = CameraCfg(
    prim_path=f"{PANDA_HAND_PRIM_PATH}/WristCamera",
    # the Newton Warp renderer only moves camera prims when we update the attached camera
    update_latest_camera_pose=True,
    offset=CameraCfg.OffsetCfg(pos=(0.07, 0.0, 0.03), rot=(-0.15305, -0.15305, 0.69035, 0.69035), convention="ros"),
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=18.90, clipping_range=(0.01, 3.0)
    ),  # camera parameters of the RealSense
    width=128,  # default values for the image quality used in most papers
    height=128,
    renderer_cfg=MultiBackendRendererCfg(),
)

# Fixed RealSense D455-like camera 0.8 m above the robot base. It is pitched 54.5 deg down to look at the table center.
HEAD_CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/HeadCamera",
    offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.8), rot=(0.0, 0.45758, 0.0, 0.88917), convention="world"),
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=16.45, clipping_range=(0.05, 4.0)
    ),  # camera parameters of the RealSense
    width=128,  # default values for the image quality used in most papers
    height=128,
    renderer_cfg=MultiBackendRendererCfg(),
)


def _camera_video(
    sensor_name: str, data_type: str, depth_range: tuple[float, float] | None = None
) -> list[VideoRecorderCfg]:
    """Return a video recorder for one data type of the first environment's camera."""

    # Create a video recorder for the specified camera (wrist, head) and data type (depth, rgb).
    recorder = VideoRecorderCfg(
        source=f"sensor:{sensor_name}:{data_type}", output_filename_prefix=f"{sensor_name}_{data_type}"
    )

    # Only depth camera has a depth field
    if depth_range is not None:
        recorder.depth_colormap_min, recorder.depth_colormap_max = depth_range
    return [recorder]


def _camera_streaming_viser_cfg(camera_cfg: CameraCfg, depth_range: tuple[float, float]) -> ViserVisualizerCfg:
    """Return a Viser server that streams a camera's RGB and depth images for the first 16 environments.

    It serves on the 8751 port to avoid overlap with the default 8750 port.
    """
    return ViserVisualizerCfg(
        port=8751,
        streaming_view=True,
        streaming_sensor_prim_path=expand_env_regex_ns(camera_cfg.prim_path),  # stream wrist or head camera
        streaming_gt_types=("rgb", "depth"),
        streaming_envs=16,
        streaming_depth_min=depth_range[0],
        streaming_depth_max=depth_range[1],
        max_visible_envs=16,
        randomly_sample_visible_envs=False,  # stream the first 16 environments
    )


def _camera_visualizer_cfgs(visualizer_cfgs: list, camera_cfg: CameraCfg, depth_range: tuple[float, float]) -> list:
    """Return the task's visualizers without Kit plus the camera streaming Viser server.

    Kit cannot share a process with OVRTX, and a Kit visualizer in the config starts Kit even with
    ``--viz none``, so the camera tasks leave it out to run kitless.
    """
    kitless = [cfg for cfg in visualizer_cfgs if not isinstance(cfg, KitVisualizerCfg)]
    return [*kitless, _camera_streaming_viser_cfg(camera_cfg, depth_range)]


@configclass
class FrankaSockWristCameraSceneCfg(FrankaSockSceneCfg):
    """Franka sock scene with a wrist camera."""

    wrist_camera: CameraCfg = WRIST_CAMERA_CFG


@configclass
class FrankaSockWristCameraScenePresetCfg(PresetCfg):
    """Scene presets for the Franka sock task with a wrist camera."""

    # Default number of environments is 128 as in the franka soft task
    newton_mjwarp_vbd_proxy: FrankaSockWristCameraSceneCfg = FrankaSockWristCameraSceneCfg(
        num_envs=128, env_spacing=ROOM_SIZE, replicate_physics=True
    )
    physx: FrankaSockWristCameraSceneCfg = FrankaSockWristCameraSceneCfg(
        num_envs=128, env_spacing=ROOM_SIZE, replicate_physics=False
    )
    isaacsim_physx = physx
    default = newton_mjwarp_vbd_proxy


@configclass
class FrankaSockHeadCameraSceneCfg(FrankaSockSceneCfg):
    """Franka sock scene with a fixed head camera above the robot base."""

    head_camera: CameraCfg = HEAD_CAMERA_CFG


@configclass
class FrankaSockHeadCameraScenePresetCfg(PresetCfg):
    """Scene presets for the Franka sock task with a head camera."""

    # Default number of environments is 128 as in the franka soft task
    newton_mjwarp_vbd_proxy: FrankaSockHeadCameraSceneCfg = FrankaSockHeadCameraSceneCfg(
        num_envs=128, env_spacing=ROOM_SIZE, replicate_physics=True
    )
    physx: FrankaSockHeadCameraSceneCfg = FrankaSockHeadCameraSceneCfg(
        num_envs=128, env_spacing=ROOM_SIZE, replicate_physics=False
    )
    isaacsim_physx = physx
    default = newton_mjwarp_vbd_proxy


@configclass
class WristCameraVideoCfg(PresetCfg):
    """Preset for the wrist camera options``.

    ``default`` leaves the recorders unset
    ``--video`` records from the viewpoint of the cameraless setup
    ``rgb`` records rgb from the wrist
    ``depth`` record the depth from the wrist
    """

    default: list[VideoRecorderCfg] = []
    rgb: list[VideoRecorderCfg] = _camera_video("wrist_camera", "rgb")
    depth: list[VideoRecorderCfg] = _camera_video("wrist_camera", "depth", WRIST_CAMERA_DEPTH_RANGE)


@configclass
class HeadCameraVideoCfg(PresetCfg):
    """Preset for the head camera options``.

    ``default`` leaves the recorders unset
    ``--video`` records from the viewpoint of the cameraless setup
    ``rgb`` records rgb from the head
    ``depth`` record the depth from the head
    """

    default: list[VideoRecorderCfg] = []
    rgb: list[VideoRecorderCfg] = _camera_video("head_camera", "rgb")
    depth: list[VideoRecorderCfg] = _camera_video("head_camera", "depth", HEAD_CAMERA_DEPTH_RANGE)


@configclass
class FrankaSockWristCameraEnvCfg(FrankaSockEnvCfg):
    """Franka sock lifting with a RealSense D405-like wrist camera."""

    scene: FrankaSockWristCameraScenePresetCfg = FrankaSockWristCameraScenePresetCfg()
    video_recorders: WristCameraVideoCfg = WristCameraVideoCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Warm up the RTX render product/annotator (Newton skips the PhysX assets_loading render loop).
        self.num_rerenders_on_reset = 2
        self.sim.visualizer_cfgs = _camera_visualizer_cfgs(
            self.sim.visualizer_cfgs, WRIST_CAMERA_CFG, WRIST_CAMERA_DEPTH_RANGE
        )


@configclass
class FrankaSockHeadCameraEnvCfg(FrankaSockEnvCfg):
    """Franka sock lifting with a fixed RealSense D455-like head camera above the robot base."""

    scene: FrankaSockHeadCameraScenePresetCfg = FrankaSockHeadCameraScenePresetCfg()
    video_recorders: HeadCameraVideoCfg = HeadCameraVideoCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Warm up the RTX render product/annotator (Newton skips the PhysX assets_loading render loop).
        self.num_rerenders_on_reset = 2
        self.sim.visualizer_cfgs = _camera_visualizer_cfgs(
            self.sim.visualizer_cfgs, HEAD_CAMERA_CFG, HEAD_CAMERA_DEPTH_RANGE
        )

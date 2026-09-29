# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka non-prehensile cloth environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonSoftContactCfg,
    VBDSolverCfg,
)
from isaaclab_newton.sim.schemas import MujocoJointCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

from isaaclab_tasks.contrib.franka_sock.franka_sock_env_cfg import (
    HEAD_CAMERA_CFG,
    HEAD_CAMERA_DEPTH_RANGE,
    ROOM_SIZE,
    SOCK_VISUAL_MATERIAL,
    WRIST_CAMERA_CFG,
    WRIST_CAMERA_DEPTH_RANGE,
    FrankaSockEnvCfg,
    FrankaSockSceneCfg,
    HeadCameraVideoCfg,
    WristCameraVideoCfg,
    _camera_visualizer_cfgs,
)
from isaaclab_tasks.contrib.franka_sock.franka_sock_env_cfg import DeformableCfg as FrankaSockDeformableCfg
from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import EventCfg as FrankaSoftEventCfg
from isaaclab_tasks.utils import PresetCfg

from . import mdp
from .cloth_mesh import CreasedMeshRectangleCfg

if TYPE_CHECKING:
    from isaaclab_newton.physics import NewtonManager

##
# Cloth and cube geometry
##

# center of the cloth and the cube on the table [m]; 15 cm farther out than the sock, so the arm reaches the
# crease near the robot without folding up against its joint limits
CLOTH_CENTER_XY = (0.60, 0.0)
CLOTH_SIZE = (0.30, 0.30)

# Edge refinement is how many particles along the longest edge (diagonal) => (edge_refinement + 1)^2.
CLOTH_EDGE_REFINEMENT = 30.0
CLOTH_SPAWN_HEIGHT = 0.004
# crease across the near-right corner (-x toward the robot, -y to its right), centered 6.5 cm from the corner,
# so pulling the crease to the robot's right drags the sheet away from the corner
CLOTH_CREASE_CFG = {"ridge_corner": (-1.0, -1.0), "ridge_height": 0.03, "ridge_width": 0.04, "ridge_distance": 0.065}
# opening of each finger when the IK gripper closes [m]: a pinched fold of the crease is about 1.5 cm wide, so the
# fingers close onto the fold without pushing through it
GRIPPER_CLOSED_OPENING = 0.007

# edge length of the cube [m]
CUBE_SIZE = 0.05
# solid PLA [kg/m^3]
PLA_DENSITY = 1240.0
# mass of a solid PLA cube [kg]
CUBE_MASS = PLA_DENSITY * CUBE_SIZE**3

# the sock's cloth materials, reused unchanged for the sheet
_SOCK_DEFORMABLE_CFG = FrankaSockDeformableCfg()
_CLOTH_PARTICLE_RADIUS = _SOCK_DEFORMABLE_CFG.newton_mjwarp_vbd_proxy.spawn.physics_material.particle_radius

# the cube spawns 2 mm above the cloth's particle shell
CUBE_SPAWN_HEIGHT = CLOTH_SPAWN_HEIGHT + _CLOTH_PARTICLE_RADIUS + 0.5 * CUBE_SIZE + 0.002

##
# Scene definition
##


@configclass
class _ClothCubeVBDSolverCfg(VBDSolverCfg):
    """VBD solver configuration that also integrates the free cube as a rigid body."""

    rigid_compliant_alm: bool = True
    """Whether VBD uses the compliant augmented Lagrangian formulation for rigid bodies.

    Newton recommends it over the deprecated legacy AVBD path, which is used when the flag is omitted.
    """


@configclass
class _ClothCubeCouplerProxyCfg(CouplerProxyCfg):
    """Proxy coupler whose manager keeps the cloth's crease and the cube's joint coordinates, see :mod:`.coupler`."""

    class_type: type[NewtonManager] | str = "{DIR}.coupler:NewtonClothCubeCouplerManager"
    """Coupler implementation class."""


@configclass
class PhysicsCfg(PresetCfg):
    """Preset physics configurations for the cloth and the cube.

    The sock task's presets, except that the VBD entry also owns the cube.
    """

    newton_mjwarp_vbd_proxy: NewtonCfg = NewtonCfg(
        solver_cfg=_ClothCubeCouplerProxyCfg(
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
                    solver_cfg=_ClothCubeVBDSolverCfg(iterations=10, rigid_body_particle_contact_buffer_size=1024),
                    bodies=[r"/World/envs/env_[^/]+/Cube"],
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


def _cloth_cfg(sock_cfg: DeformableObjectCfg) -> DeformableObjectCfg:
    """Return a creased cloth sheet with the sock's deformable, collision, and material settings."""
    return DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Deformable",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(*CLOTH_CENTER_XY, CLOTH_SPAWN_HEIGHT)),
        spawn=CreasedMeshRectangleCfg(
            size=CLOTH_SIZE,
            edge_refinement=CLOTH_EDGE_REFINEMENT,
            deformable_props=sock_cfg.spawn.deformable_props,
            collision_props=sock_cfg.spawn.collision_props,
            visual_material=SOCK_VISUAL_MATERIAL,
            physics_material=sock_cfg.spawn.physics_material,
            **CLOTH_CREASE_CFG,
        ),
    )


@configclass
class DeformableCfg(PresetCfg):
    """Preset configurations for the cloth sheet."""

    newton_mjwarp_vbd_proxy: DeformableObjectCfg = _cloth_cfg(_SOCK_DEFORMABLE_CFG.newton_mjwarp_vbd_proxy)

    physx: DeformableObjectCfg = _cloth_cfg(_SOCK_DEFORMABLE_CFG.physx)
    isaacsim_physx = physx

    default = newton_mjwarp_vbd_proxy


# Solid PLA cube; friction is the middle of the reported PLA ranges (static 0.3-0.4, dynamic 0.2-0.3)
CUBE_SPAWN_CFG = sim_utils.CuboidCfg(
    size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    mass_props=sim_utils.MassPropertiesCfg(mass=CUBE_MASS),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.35, dynamic_friction=0.25, restitution=0.0),
    # matte orange print, to stand out against the gray cloth in the camera images
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.35, 0.05), roughness=0.4),
)


@configclass
class FrankaNonpClothSceneCfg(FrankaSockSceneCfg):
    """Scene for the Franka non-prehensile cloth environment: a cloth sheet with a cube on top."""

    deformable: DeformableCfg = DeformableCfg()

    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*CLOTH_CENTER_XY, CUBE_SPAWN_HEIGHT)),
        spawn=CUBE_SPAWN_CFG,
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        # MuJoCo ignores the inherited ``disable_gravity``, so the low-PD arm sagged about 4 cm below its
        # IK targets under full gravity; compensate gravity through the actuators instead, as a real Franka does
        self.robot.spawn.joint_drive_props = [MujocoJointCfg(actuatorgravcomp=True)]


@configclass
class FrankaNonpClothScenePresetCfg(PresetCfg):
    """Preset config for the Franka non-prehensile cloth scene."""

    newton_mjwarp_vbd_proxy: FrankaNonpClothSceneCfg = FrankaNonpClothSceneCfg(
        num_envs=1024, env_spacing=ROOM_SIZE, replicate_physics=True
    )

    # Isaac Sim PhysX does not support replicating physics for deformable objects
    physx: FrankaNonpClothSceneCfg = FrankaNonpClothSceneCfg(
        num_envs=1024, env_spacing=ROOM_SIZE, replicate_physics=False
    )
    isaacsim_physx = physx

    default = newton_mjwarp_vbd_proxy


##
# MDP settings
##


@configclass
class FrankaNonpClothEventCfg(FrankaSoftEventCfg):
    """Reset events for the Franka non-prehensile cloth environment."""

    # the cube moves with the cloth, so it stays on it; x starts closer than the sock's -0.15 m,
    # since the longer cloth would otherwise reach the robot base
    reset_deformable = EventTerm(
        func=mdp.reset_deformable_with_object,
        mode="reset",
        params={
            "position_range": {"x": (-0.1, 0.1), "y": (-0.2, 0.2)},
            "object_cfg": SceneEntityCfg("cube"),
            "asset_cfg": SceneEntityCfg("deformable"),
        },
    )


##
# Environment configuration
##


@configclass
class FrankaNonpClothEnvCfg(FrankaSockEnvCfg):
    """Manager-based RL environment: Franka Panda moving a cube by pulling the cloth it rests on.

    Only the scene differs from the sock task so far; the rewards, commands, and observations are the sock's.
    """

    scene: FrankaNonpClothScenePresetCfg = FrankaNonpClothScenePresetCfg()
    events: FrankaNonpClothEventCfg = FrankaNonpClothEventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # cloth and cube physics presets, see :class:`PhysicsCfg`
        self.sim.physics = PhysicsCfg()
        # Close the gripper onto the crease's fold: the shared soft-lift default (0.01 m per finger) stays clear of
        # it, and closing fully pushes the fingers through the cloth once they stop colliding with a held cloth.
        self.actions.ik.gripper_action.close_command_expr = {"panda_finger_joint1": GRIPPER_CLOSED_OPENING}

        # Visualizers requested with --viz that the sock's visualizer list does not configure, such as the OVRTX
        # newton_rtx, take their camera from the default visualizer config; give it the sock's view, so videos
        # show the same view whichever visualizer records them.
        sock_view = next(cfg for cfg in self.sim.visualizer_cfgs if isinstance(cfg, NewtonGLVisualizerCfg))
        for name in ("eye", "lookat", "focal_length", "max_visible_envs", "randomly_sample_visible_envs"):
            setattr(self.sim.default_visualizer_cfg, name, getattr(sock_view, name))


##
# Camera variants
##


@configclass
class FrankaNonpClothWristCameraSceneCfg(FrankaNonpClothSceneCfg):
    """Franka non-prehensile cloth scene with a wrist camera."""

    wrist_camera: CameraCfg = WRIST_CAMERA_CFG


@configclass
class FrankaNonpClothWristCameraScenePresetCfg(PresetCfg):
    """Scene presets for the Franka non-prehensile cloth task with a wrist camera."""

    # Default number of environments is 128 as in the franka soft task
    newton_mjwarp_vbd_proxy: FrankaNonpClothWristCameraSceneCfg = FrankaNonpClothWristCameraSceneCfg(
        num_envs=128, env_spacing=ROOM_SIZE, replicate_physics=True
    )
    physx: FrankaNonpClothWristCameraSceneCfg = FrankaNonpClothWristCameraSceneCfg(
        num_envs=128, env_spacing=ROOM_SIZE, replicate_physics=False
    )
    isaacsim_physx = physx
    default = newton_mjwarp_vbd_proxy


@configclass
class FrankaNonpClothHeadCameraSceneCfg(FrankaNonpClothSceneCfg):
    """Franka non-prehensile cloth scene with a fixed head camera above the robot base."""

    head_camera: CameraCfg = HEAD_CAMERA_CFG


@configclass
class FrankaNonpClothHeadCameraScenePresetCfg(PresetCfg):
    """Scene presets for the Franka non-prehensile cloth task with a head camera."""

    # Default number of environments is 128 as in the franka soft task
    newton_mjwarp_vbd_proxy: FrankaNonpClothHeadCameraSceneCfg = FrankaNonpClothHeadCameraSceneCfg(
        num_envs=128, env_spacing=ROOM_SIZE, replicate_physics=True
    )
    physx: FrankaNonpClothHeadCameraSceneCfg = FrankaNonpClothHeadCameraSceneCfg(
        num_envs=128, env_spacing=ROOM_SIZE, replicate_physics=False
    )
    isaacsim_physx = physx
    default = newton_mjwarp_vbd_proxy


@configclass
class FrankaNonpClothWristCameraEnvCfg(FrankaNonpClothEnvCfg):
    """Franka non-prehensile cloth task with a RealSense D405-like wrist camera."""

    scene: FrankaNonpClothWristCameraScenePresetCfg = FrankaNonpClothWristCameraScenePresetCfg()
    video_recorders: WristCameraVideoCfg = WristCameraVideoCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Warm up the RTX render product/annotator (Newton skips the PhysX assets_loading render loop).
        self.num_rerenders_on_reset = 2
        self.sim.visualizer_cfgs = _camera_visualizer_cfgs(
            self.sim.visualizer_cfgs, WRIST_CAMERA_CFG, WRIST_CAMERA_DEPTH_RANGE
        )


@configclass
class FrankaNonpClothHeadCameraEnvCfg(FrankaNonpClothEnvCfg):
    """Franka non-prehensile cloth task with a fixed RealSense D455-like head camera above the robot base."""

    scene: FrankaNonpClothHeadCameraScenePresetCfg = FrankaNonpClothHeadCameraScenePresetCfg()
    video_recorders: HeadCameraVideoCfg = HeadCameraVideoCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Warm up the RTX render product/annotator (Newton skips the PhysX assets_loading render loop).
        self.num_rerenders_on_reset = 2
        self.sim.visualizer_cfgs = _camera_visualizer_cfgs(
            self.sim.visualizer_cfgs, HEAD_CAMERA_CFG, HEAD_CAMERA_DEPTH_RANGE
        )

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka sock lifting environment.

The sock is a surface deformable shaped as a tube with a closed toe and an open cuff, which is the
only structural difference from ``Isaac-Lift-Cloth-Franka``. Because the tube holds a cross-section,
the gripper can pinch its wall directly and the cloth task's support rails are not needed, so the
reset event, observations, actions, commands, terminations, and curriculum are all inherited
unchanged.
"""

from __future__ import annotations

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

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.utils import configclass

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

##
# Sock geometry
##

# Radius of the sock tube [m]. Also its resting half-height, which is what sets the lift threshold.
SOCK_RADIUS = 0.03
# Length of the sock's cylindrical section, excluding the toe cap [m].
SOCK_LENGTH = 0.15
# Circumferential resolution. This, and not ``edge_refinement``, controls the mesh density: the
# tube is slender enough that its bounding-box diagonal stays far above the circumferential edge
# length, so the diagonal-based refinement never triggers.
SOCK_NUM_SEGMENTS = 20
# Spawn height [m]: the lowest node sits 2 mm above the table top at z = 0, so the first substep
# resolves contact rather than a penetration.
SOCK_SPAWN_HEIGHT = SOCK_RADIUS + 0.002
# Centroid height of the settled sock [m], measured by letting it fall under full gravity. It sits
# below SOCK_RADIUS because the resting tube sags to about 0.053 m across instead of 0.06 m.
SOCK_RESTING_HEIGHT = 0.025
# Clearance above the resting centroid that counts as lifted [m].
SOCK_LIFT_CLEARANCE = 0.05


##
# Scene definition
##


@configclass
class PhysicsCfg(PresetCfg):
    """Preset physics configurations for the sock.

    Mirrors the cloth task's coupled MJWarp/VBD setup, without the support rails it owns. Only the
    Newton preset is tuned; the PhysX preset is provided for parity and is untuned.
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
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.35, 0.45)),
            physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
                density=1.0,
                particle_radius=0.002,
                tri_ke=5e2,
                tri_ka=5e2,
                tri_kd=1e-3,
                # much stiffer in bending than the cloth's 0.5: a zero-pressure tube with cloth-like
                # bending collapses into a flat double layer, which is the shape the rails exist to
                # work around in the cloth task
                edge_ke=30.0,
                edge_kd=1e-3,
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
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.35, 0.45)),
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


WALL_THICKNESS = 0.02
"""Thickness of a divider panel [m]."""

WALL_HEIGHT = 0.6
"""Height of a divider panel above the table top [m]."""

# Divider panels between environments. Visual only: no collision, rigid, mass, or physics material,
# so they never enter the contact solve. Their length spans one environment, which is only known
# once ``env_spacing`` is set, so the scene resizes them in ``__post_init__``.
WALL_SPAWN_CFG = sim_utils.CuboidCfg(
    size=(1.0, WALL_THICKNESS, WALL_HEIGHT),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.26, 0.28, 0.33), roughness=0.9),
)


@configclass
class FrankaSockSceneCfg(_FrankaSoftSceneCfg):
    """Scene for the Franka sock environment."""

    deformable: DeformableCfg = DeformableCfg()

    wall_neg_y: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DividerNegY",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=WALL_SPAWN_CFG,
    )
    wall_pos_y: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DividerPosY",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=WALL_SPAWN_CFG,
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        # increase franka gripper stiffness
        self.robot.actuators["panda_hand"].joint_effort_limit = 500.0
        self.robot.actuators["panda_hand"].stiffness = 2000.0
        self.robot.actuators["panda_hand"].damping = 100.0

        # place the dividers on the environment boundary, resting on the table top at z = 0
        half_spacing = 0.5 * self.env_spacing
        for wall, sign in ((self.wall_neg_y, -1.0), (self.wall_pos_y, 1.0)):
            wall.spawn = WALL_SPAWN_CFG.replace(size=(self.env_spacing, WALL_THICKNESS, WALL_HEIGHT))
            wall.init_state.pos = (0.5, sign * half_spacing, 0.5 * WALL_HEIGHT)


@configclass
class FrankaSockScenePresetCfg(PresetCfg):
    """Preset config for the Franka sock scene."""

    # fewer environments than the cloth task: the sock mesh carries ~5x the particles of the sheet
    newton_mjwarp_vbd_proxy: FrankaSockSceneCfg = FrankaSockSceneCfg(
        num_envs=1024, env_spacing=2.0, replicate_physics=True
    )

    # Isaac Sim PhysX does not support replicating physics for deformable objects
    physx: FrankaSockSceneCfg = FrankaSockSceneCfg(num_envs=1024, env_spacing=2.0, replicate_physics=False)
    isaacsim_physx = physx

    default = newton_mjwarp_vbd_proxy


##
# MDP settings
##

# Success visualizer for the invisible table, re-skinned from the shared soft-task markers. The two
# prototypes must stay clearly distinguishable, since their color is the only success readout.
SUCCESS_VISUALIZER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/SuccessMarkers",
    markers={
        "failure": TABLE_SPAWN_CFG.replace(
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.62, 0.24, 0.26), roughness=0.5),
            visible=True,
        ),
        "success": TABLE_SPAWN_CFG.replace(
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.24, 0.62, 0.34), roughness=0.5),
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
        # override the soft-beam physics with the sock presets
        self.sim.physics = PhysicsCfg()
        # re-skin the table that doubles as the success readout
        self.commands.deformable_pose.success_visualizer_cfg = SUCCESS_VISUALIZER_CFG
        # the cloth closes the gripper fully for a thin sheet; the sock wall is thicker, so the
        # shared 0.01 m close command is kept

    def play_mode(self):
        super().play_mode()
        # show the end-effector frame when inspecting a policy; too costly to leave on for training.
        # play mode runs after preset resolution, so the scene is the concrete cfg here.
        self.scene.ee_frame.debug_vis = True

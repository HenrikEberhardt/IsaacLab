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
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.physics import PhysxAutoCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

from isaaclab_tasks.contrib.franka_sock.franka_sock_env_cfg import (
    HEAD_CAMERA_CFG,
    HEAD_CAMERA_DEPTH_RANGE,
    ROOM_SIZE,
    SOCK_VISUAL_MATERIAL,
    SUCCESS_VISUALIZER_CFG,
    WRIST_CAMERA_CFG,
    WRIST_CAMERA_DEPTH_RANGE,
    FrankaSockEnvCfg,
    FrankaSockSceneCfg,
    HeadCameraVideoCfg,
    WristCameraVideoCfg,
    _camera_visualizer_cfgs,
)
from isaaclab_tasks.contrib.franka_sock.franka_sock_env_cfg import DeformableCfg as FrankaSockDeformableCfg
from isaaclab_tasks.contrib.lift import mdp as lift_mdp
from isaaclab_tasks.core.lift import mdp as soft_lift_mdp
from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import CurriculumCfg as FrankaSoftCurriculumCfg
from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import EventCfg as FrankaSoftEventCfg
from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import TerminationsCfg as FrankaSoftTerminationsCfg
from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import _IkActionsCfg as FrankaSoftIkActionsCfg
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
# Task geometry
##

# Drag targets are drawn uniformly on this line in front of the robot base. The line lies closer to the robot than any crease, so every drag pulls the sheet towards the robot.
DRAG_TARGET_X = 0.35
DRAG_TARGET_Y = (-0.30, 0.30)

CREASE_HEIGHT = CLOTH_SPAWN_HEIGHT + CLOTH_CREASE_CFG["ridge_height"]
# Margin around the goal to drag to
CREASE_SUCCESS_THRESHOLD = 0.03
# Damping of the differential IK.
IK_DAMPING = 0.1
# 10s at 120Hz simulation with decimation of 4 are 300 frames
EPISODE_LENGTH_S = 10.0
# Arm pose with the hand pointing straight down at zero yaw
TOP_DOWN_ARM_JOINT_POS = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.569,
    "panda_joint3": 0.0,
    "panda_joint4": -2.810,
    "panda_joint5": 0.0,
    "panda_joint6": 2.2445,
    "panda_joint7": 0.7854,
}
# Joint randomization band in [rad] for the top-down reset. 0.18-0.35 m above the table and 0.19-0.52 m from the crease, with hand yaw up to 50 deg either way.
TOP_DOWN_RESET_OFFSETS = {
    "panda_joint1": (-0.25, 0.25),
    "panda_joint2": (-0.15, 0.15),
    "panda_joint4": (-0.15, 0.15),
    "panda_joint7": (-0.6, 0.6),
}
# share of the full drag at which the drag-distance curriculum starts
INITIAL_DRAG_FRACTION = 0.2

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
                    # Increased the collison buffer as flat cube on the cloth has more contacts then sock
                    solver_cfg=_ClothCubeVBDSolverCfg(iterations=10, rigid_body_particle_contact_buffer_size=2048),
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

        # MuJoCo ignores the inherited ``disable_gravity``, so we compensate gravity through the actuators instead, as a real Franka does
        self.robot.spawn.joint_drive_props = [MujocoJointCfg(actuatorgravcomp=True)]
        # start with the hand pointing straight down, as the top-down arm action and reset keep it
        self.robot.init_state = self.robot.init_state.replace(
            joint_pos={**self.robot.init_state.joint_pos, **TOP_DOWN_ARM_JOINT_POS}
        )


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
class CommandsCfg:
    """The target to drag the crease of the cloth to."""

    # named as the sock task's command, whose configuration sets its success visualizer
    deformable_pose = mdp.CreaseDragCommandCfg(
        asset_name="robot",
        object_name="deformable",
        # longer than an episode, so the target drawn at a reset holds for the whole episode
        resampling_time_range=(2 * EPISODE_LENGTH_S, 2 * EPISODE_LENGTH_S),
        debug_vis=True,
        ranges=mdp.CreaseDragCommandCfg.Ranges(
            pos_x=(DRAG_TARGET_X, DRAG_TARGET_X),
            pos_y=DRAG_TARGET_Y,
            pos_z=(CREASE_HEIGHT, CREASE_HEIGHT),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        success_threshold=CREASE_SUCCESS_THRESHOLD,
        # raised to the full drag by the drag-distance curriculum
        drag_fraction=INITIAL_DRAG_FRACTION,
        # the sock task's green rim around the table shows once the crease reaches its target
        success_vis_asset_name="table",
        success_visualizer_cfg=SUCCESS_VISUALIZER_CFG,
    )


@configclass
class _IkRelActionsCfg:
    """4-dim relative top-down end-effector motion (translation and yaw) via differential IK + 1-dim binary gripper
    that grasps the cloth."""

    arm_action = mdp.TopDownDifferentialIKActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
            ik_params={"lambda_val": IK_DAMPING},
        ),
        # motion per step at a unit action: 2 cm and 0.05 rad, i.e. 0.6 m/s and 1.5 rad/s
        scale=(0.02, 0.02, 0.02, 0.05),
        body_offset=mdp.TopDownDifferentialIKActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        # The robot does not collide with the table, which only the cloth solver knows about, so keep the fingertips,
        # about 5 mm below this frame, at least 7 mm above it. The crease grasp puts this frame about 18 mm high.
        min_height=0.012,
    )

    gripper_action = mdp.ClothGraspActionCfg(
        asset_name="robot",
        joint_names=["panda_finger_joint1"],
        open_command_expr={"panda_finger_joint1": 0.04},
        close_command_expr={"panda_finger_joint1": GRIPPER_CLOSED_OPENING},
    )


@configclass
class ActionsCfg(PresetCfg):
    """Action-space presets: relative top-down task-space IK for RL, absolute task-space IK for the state machine."""

    ik_rel: _IkRelActionsCfg = _IkRelActionsCfg()

    ik: FrankaSoftIkActionsCfg = FrankaSoftIkActionsCfg()

    default = ik_rel


@configclass
class ObservationsCfg:
    """Policy observations: arm state, crease, cube, drag target, their relative positions and yaw, grasp state,
    cloth points, and last action."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        ee_position = ObsTerm(func=mdp.ee_position_in_robot_root_frame)
        crease_position = ObsTerm(
            func=mdp.crease_position_in_robot_root_frame, params={"command_name": "deformable_pose"}
        )
        cube_position = ObsTerm(
            func=lift_mdp.object_position_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("cube")}
        )
        target_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "deformable_pose"})
        crease_from_ee = ObsTerm(func=mdp.crease_relative_to_ee, params={"command_name": "deformable_pose"})
        target_from_crease = ObsTerm(func=mdp.target_relative_to_crease, params={"command_name": "deformable_pose"})
        crease_yaw_error = ObsTerm(func=mdp.crease_yaw_error, params={"command_name": "deformable_pose"})
        grasp_state = ObsTerm(func=mdp.grasp_state)
        deformable_sampled_points = ObsTerm(
            func=soft_lift_mdp.DeformableSampledPointsInRobotRootFrame,
            params={"asset_cfg": SceneEntityCfg("deformable"), "num_points": 20},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Staged rewards to pinch the crease and drag it to its target.

    Before the grasp, the reach, yaw alignment, and closing terms guide the gripper onto the crease and close it there.
    While the gripper holds the crease they pay their maximum, so grasping never costs reward, and only then do the
    goal terms pay for moving the crease toward its target. Letting go of the crease drops the grasp and goal terms
    again.
    """

    # wide enough to pull the gripper in from its start 0.2-0.5 m away from the crease
    reaching_crease = RewTerm(
        func=mdp.crease_ee_distance, params={"std": 0.3, "command_name": "deformable_pose"}, weight=1.0
    )

    # wide enough to pay for descending from a hover onto the crease: 0.24 at 5 cm above its grasp point
    reaching_crease_fine = RewTerm(
        func=mdp.crease_ee_distance, params={"std": 0.05, "command_name": "deformable_pose"}, weight=1.0
    )

    yaw_alignment = RewTerm(func=mdp.crease_yaw_alignment, params={"command_name": "deformable_pose"}, weight=1.0)

    # closing the fingers pays only within about a finger's width of the grasp point
    closing_at_crease = RewTerm(
        func=mdp.crease_closing, params={"std": 0.015, "command_name": "deformable_pose"}, weight=1.0
    )

    grasping_crease = RewTerm(func=mdp.crease_grasped, params={"command_name": "deformable_pose"}, weight=2.0)

    crease_goal_tracking = RewTerm(
        func=mdp.crease_goal_distance, params={"std": 0.2, "command_name": "deformable_pose"}, weight=4.0
    )

    crease_goal_tracking_fine = RewTerm(
        func=mdp.crease_goal_distance, params={"std": 0.03, "command_name": "deformable_pose"}, weight=4.0
    )

    success = RewTerm(
        func=mdp.CreaseGoalReached,
        params={"command_name": "deformable_pose", "success_threshold": CREASE_SUCCESS_THRESHOLD},
        weight=10.0,
    )

    # Closing the gripper away from the crease pays nothing, but it pushes the fold instead of straddling it and spends
    # the close's one grasp try; keep the gripper open on the approach. Zero within 2.5 cm of the grasp point, about
    # half the weight at 5.3 cm and the full weight beyond 15 cm.
    closing_away_from_crease = RewTerm(
        func=mdp.crease_closing_away,
        params={"margin": 0.025, "std": 0.05, "command_name": "deformable_pose"},
        weight=-0.5,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2, params={"asset_cfg": SceneEntityCfg("robot", joint_names="panda_joint.*")}, weight=-1e-4
    )


@configclass
class FrankaNonpClothEventCfg(FrankaSoftEventCfg):
    """Reset events for the Franka non-prehensile cloth environment."""

    # random arm poses with the hand pointing straight down
    reset_robot_arm_joints = EventTerm(
        func=mdp.reset_franka_top_down,
        mode="reset",
        params={"joint_offset_ranges": TOP_DOWN_RESET_OFFSETS, "asset_cfg": SceneEntityCfg("robot")},
    )

    # Reset the cloth and cube to a random position on the table, with the cube on top of the cloth.
    reset_deformable = EventTerm(
        func=mdp.reset_deformable_with_object,
        mode="reset",
        params={
            "position_range": {"x": (-0.1, 0.1), "y": (-0.2, 0.2)},
            "object_cfg": SceneEntityCfg("cube"),
            "asset_cfg": SceneEntityCfg("deformable"),
        },
    )


@configclass
class TerminationsCfg(FrankaSoftTerminationsCfg):
    """The soft lift tasks' terminations, and an end to episodes whose simulation state is no longer finite."""

    nonfinite_state = DoneTerm(func=mdp.nonfinite_state)


@configclass
class CurriculumCfg(FrankaSoftCurriculumCfg):
    """A softer action-rate ramp and a drag-distance curriculum.

    The drag target starts :data:`INITIAL_DRAG_FRACTION` of the way to the full drag, and the curriculum raises it to 1.0 as the policy learns to drag the crease.
    """

    # After 15,000 env steps (about 625 iterations) the action-rate weight is increased with curricula.
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-2, "num_steps": 15000}
    )

    # The drag-difficulty is integer counter that starts at 0 and increases to 10 as the policy learns to drag the crease. Mean of successful drags larger then 0.5 increases the difficulty, mean smaller than 0.5 decreases it. The drag length therefore settles near the distance at which the policy succeeds about half the time.
    drag_difficulty = CurrTerm(
        func=soft_lift_mdp.DifficultyScheduler, params={"init_difficulty": 0, "min_difficulty": 0, "max_difficulty": 10}
    )

    # The drag fraction is increased with the difficulty.
    drag_fraction = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.deformable_pose.drag_fraction",
            "modify_fn": soft_lift_mdp.initial_final_interpolate_fn,
            "modify_params": {
                "initial_value": INITIAL_DRAG_FRACTION,
                "final_value": 1.0,
                "difficulty_term_str": "drag_difficulty",
            },
        },
    )


##
# Environment configuration
##


@configclass
class FrankaNonpClothEnvCfg(FrankaSockEnvCfg):
    """Manager-based RL environment: Franka Panda moving a cube by pulling the cloth it rests on.

    The policy pinches the crease of the cloth and drags it to a target on a line in front of the robot, as the
    ``franka_nonp_cloth_sm.py`` state machine does. 
    """

    scene: FrankaNonpClothScenePresetCfg = FrankaNonpClothScenePresetCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: FrankaNonpClothEventCfg = FrankaNonpClothEventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.episode_length_s = EPISODE_LENGTH_S
        # Need Full gravity from the start as the cube would not stick to the cloth without
        self.curriculum.gravity = None

        # cloth and cube physics presets, see :class:`PhysicsCfg`
        self.sim.physics = PhysicsCfg()
        # Close only to threshold (1.5cm) as otherwise the fingers push through the crease
        self.actions.ik.gripper_action.close_command_expr = {"panda_finger_joint1": GRIPPER_CLOSED_OPENING}

        # Use the same visuliazer configuration (viewpoint, focal lenght, environment limits,...) as the sock task
        sock_view = next(cfg for cfg in self.sim.visualizer_cfgs if isinstance(cfg, NewtonGLVisualizerCfg))
        for name in ("eye", "lookat", "focal_length", "max_visible_envs", "randomly_sample_visible_envs"):
            setattr(self.sim.default_visualizer_cfg, name, getattr(sock_view, name))

    def play_mode(self):
        super().play_mode()

        # When playing the policy use the full drag lenght (no curriculum).
        self.commands.deformable_pose.drag_fraction = 1.0
        if self.curriculum is not None:
            self.curriculum.drag_difficulty = None
            self.curriculum.drag_fraction = None


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

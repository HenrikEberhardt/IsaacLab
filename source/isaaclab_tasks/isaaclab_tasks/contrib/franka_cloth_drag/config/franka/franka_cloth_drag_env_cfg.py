# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka cloth dragging environment."""

from __future__ import annotations

import math
from dataclasses import MISSING

from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonSoftContactCfg,
    VBDSolverCfg,
)
from isaaclab_newton.sim.schemas import MujocoRigidBodyCfg, NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg
from isaaclab_visualizers.viser import ViserVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import TABLE_SPAWN_CFG
from isaaclab_tasks.utils import PresetCfg

from ... import mdp

##
#################### Pre-defined configs ####################
##

from isaaclab_assets.robots.franka import FRANKA_PANDA_MENAGERIE_CFG  # isort:skip

# Robot Home Position with head pointing down and gripper open
HOME_JOINT_POS = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.569,
    "panda_joint3": 0.0,
    "panda_joint4": -2.810,
    "panda_joint5": 0.0,
    "panda_joint6": 2.241,
    "panda_joint7": math.pi / 4,
    "panda_finger_joint.*": 0.04,
}

##
#################### Physics ####################
##

# Newton physics: MJWarp simulates the robot, VBD the cloth, coupled through proxy bodies for the hand and fingers.
PHYSICS_CFG = NewtonCfg(
    solver_cfg=CouplerProxyCfg(
        entries=[
            CouplerEntryCfg(
                # MJWarp solver for rigid robot
                name="rigid",
                solver_cfg=MJWarpSolverCfg(
                    cone="elliptic",  # shape of the friction cone
                    ls_iterations=30,  # line search iterations of MuJoCo constraint solver
                    integrator="implicitfast",  # solver type
                ),
                bodies=[r"/World/envs/env_[^/]+/Robot"],
            ),
            CouplerEntryCfg(
                # VBD solver for soft cloth
                name="soft",
                # per rigid body how many cloth contact nodes can be stored
                solver_cfg=VBDSolverCfg(iterations=10, rigid_body_particle_contact_buffer_size=1024),
                all_particles=True,
                include_static_shapes=True,  # VBD owns static shapes (e.g. table)
            ),
        ],
        proxies=[
            CouplerProxyMappingCfg(
                source="rigid",
                destination="soft",
                # Collision of cloth only with head and fingers of robot arm
                bodies=[
                    r"/World/envs/env_[^/]+/Robot/Geometry/.*panda_hand",
                    r"/World/envs/env_[^/]+/Robot/Geometry/.*panda_(left|right)finger",
                ],
                collide_interval=1,  # Compute collision with proxy at every step
                collision_pipeline=NewtonCollisionPipelineCfg(
                    enable_rigid_soft_full_surface_contact=True,
                ),
            )
        ],
        iterations=1,  # Solve coupling once per step, more iterations are more accurate but slower
    ),
    soft_contact_cfg=NewtonSoftContactCfg(
        soft_contact_ke=8.0e3,  # stiffness, higher less penetration but more unstable
        soft_contact_kd=1.0e-2,  # contact damping, higher more stable but slower response
        soft_contact_mu=100.0,  # Increase friction to prevent cloth slipping out of the gripper
    ),
    num_substeps=2,  # Solver substeps per physics step
)


##
#################### Scene definition ####################
##


@configclass
class FrankaClothDragSceneCfg(InteractiveSceneCfg):
    """Scene for the Franka cloth dragging environment."""

    # Franka Panda robot
    robot: ArticulationCfg = FRANKA_PANDA_MENAGERIE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot", init_state=ArticulationCfg.InitialStateCfg(joint_pos=HOME_JOINT_POS)
    )
    robot.spawn.variants = {"Physics": "mujoco"}

    # end-effector frame
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/panda_link0",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=(
                    "{ENV_REGEX_NS}/Robot/Geometry/panda_link0/panda_link1/panda_link2/panda_link3/"
                    "panda_link4/panda_link5/panda_link6/panda_link7/panda_hand"
                ),
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
            ),
        ],
    )

    # Cloth simulated by the Newton VBD solver
    deformable: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Deformable",
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=(0.8, 0.0, 0.005), rot=(0.0, 0.0, 0.0, 1.0)
        ),  # Flat at slightly above table height
        spawn=sim_utils.MeshRectangleCfg(
            size=(0.2, 0.2),
            edge_refinement=8,
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.1)),
            physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
                density=1.0,
                particle_radius=0.002,
                tri_ke=5e2,
                tri_ka=5e2,
                tri_kd=1e-3,
                edge_ke=0.5,
                edge_kd=1e-3,
            ),
        ),
    )

    # Static table collider with its top surface at z = 0
    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, -0.525]),
        spawn=TABLE_SPAWN_CFG,
    )

    # ground plane
    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    def __post_init__(self) -> None:
        # Franka actuators, inspired by libfranka's joint_impedance_control.cpp
        self.robot.actuators = {
            "panda_arm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-7]"],
                joint_effort_limit={"panda_joint[1-4]": 87.0, "panda_joint[5-7]": 12.0},
                joint_velocity_limit={"panda_joint[1-4]": 2.175, "panda_joint[5-7]": 2.61},
                stiffness={
                    "panda_joint[1-4]": 600.0,
                    "panda_joint5": 250.0,
                    "panda_joint6": 150.0,
                    "panda_joint7": 50.0,
                },
                damping={
                    "panda_joint[1-4]": 50.0,
                    "panda_joint5": 30.0,
                    "panda_joint6": 25.0,
                    "panda_joint7": 15.0,
                },
                armature={
                    "panda_joint[1-2]": 0.6057,
                    "panda_joint[3-4]": 0.4625,
                    "panda_joint[5-7]": 0.2055,
                },
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint1"],
                joint_effort_limit=500.0,
                actuator_velocity_limit=0.2,
                joint_velocity_limit=2.0,
                stiffness=2000.0,
                damping=100.0,
                armature=0.1,
            ),
            # Mimics the first finger actions
            "panda_finger2_passive": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint2"],
                joint_effort_limit=1.0,
                actuator_velocity_limit=0.2,
                joint_velocity_limit=2.0,
                stiffness=0.0,
                damping=0.0,
                armature=0.1,
            ),
        }

        # Use MJWarp solver gravity compensation for the robot
        self.robot.spawn.rigid_props = [MujocoRigidBodyCfg(gravcomp=1.0)]


##
#################### MDP settings ####################
##

# ------------------- Action Terms -------------------
GRIPPER_ACTION_CFG = mdp.BinaryJointPositionActionCfg(
    # Only finger 1 is actuated, binary action <0 => closes and >=0 => opens
    asset_name="robot",
    joint_names=["panda_finger_joint1"],
    # Fully opened and closed position of gripper finger
    open_command_expr={"panda_finger_joint1": 0.04},
    close_command_expr={"panda_finger_joint1": 0.0},
)


@configclass
class JointDeltaActionsCfg:
    """7-DoF delta joint position + 1-DoF binary gripper."""

    arm_action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=0.03,  # How many rad/step can be max commanded for each joint
    )

    gripper_action = GRIPPER_ACTION_CFG


@configclass
class EEDeltaActionsCfg:
    """4-DoF delta EE motion (dx, dy, dz, dyaw) => hand pointing down + 1-DoF binary gripper."""

    arm_action = mdp.TopDownDeltaEEActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale_trans=0.01,  # How many m/step can be max commanded for dx, dy, dz
        scale_rot=0.05,  # How many rad/step can be max commanded for dyaw
        # Panda hand frame is commanded with offset of 0.1034m to match EE frame of robot
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,  # Commands are relative pose
            ik_method="dls",  # Damped Least Squares to stay stable near singularities
            ik_params={"lambda_val": 0.3},  # damping, larger lambda_val => more stable but slower response
        ),
        body_offset=mdp.TopDownDeltaEEActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.1034]),
    )

    gripper_action = GRIPPER_ACTION_CFG


@configclass
class ActionsCfg(PresetCfg):
    """Action-space presets: delta joint space or delta EE task space."""

    # Use "env.actions=joint_delta" Flag
    joint_delta: JointDeltaActionsCfg = JointDeltaActionsCfg()

    # Use "env.actions=ee_delta" Flag
    ee_delta: EEDeltaActionsCfg = EEDeltaActionsCfg()

    default = ee_delta


# ------------------- Observation Terms -------------------
@configclass
class ActorObsCfg(ObsGroup):
    """Actor observations shared by both action spaces.

    The actor observes the task partially (POMDP), so every term carries a history of the last 5 steps.
    """

    # End-effector fingertip position and yaw expressed in base frame
    ee_pose = ObsTerm(func=mdp.ee_pose_b)

    # End-effector fingertip position in the cloth COM frame: p_EE - p_COM, along the base frame axes
    ee_pos_com = ObsTerm(func=mdp.ee_pos_com)

    # Target relative to the cloth's COM: the displacement the cloth still has to make
    target_pos_com = ObsTerm(func=mdp.target_pos_com, params={"command_name": "cloth_target"})

    # Measured finger opening distance in [m]
    gripper_opening = ObsTerm(
        func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_finger_joint1"])}
    )

    # Last action; with the group history the actor sees the last 5 actions
    actions = ObsTerm(func=mdp.last_action)

    # Current stage of the task: approach, grasping, dragging, near goal
    # privileged: computed from all cloth nodes and the true COM; revisit for deployment
    stage = ObsTerm(func=mdp.cloth_drag_stage, params={"command_name": "cloth_target"})

    # 20 random cloth nodes (sorted ids, fixed per episode) in the cloth COM frame
    cloth_points = ObsTerm(func=mdp.ClothSampledPointsInComFrame, params={"num_points": 20})

    def __post_init__(self) -> None:
        self.enable_corruption = False  # no observation noise
        self.concatenate_terms = True  # join all terms into one flat tensor for the MLP
        self.history_length = 5  # stack the last 5 steps of every term, including terms added in subclasses
        self.flatten_history_dim = True  # (num_envs, 5, dim) -> (num_envs, 5 * dim)


@configclass
class JointActorObsCfg(ActorObsCfg):
    """Actor observations for delta joint-space actions: adds the arm joint state."""

    # Add robot joint positions and velocities
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])}
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])}
    )


@configclass
class EEActorObsCfg(ActorObsCfg):
    """Actor observations for delta end-effector actions: adds the end-effector velocity and yaw rate."""

    # Add end-effector linear velocity and angular velocity in base frame
    ee_velocity = ObsTerm(func=mdp.ee_velocity_b, params={"hand_cfg": SceneEntityCfg("robot", body_names="panda_hand")})


@configclass
class CriticObsCfg(ObsGroup):
    """Privileged critic observations shared by both action spaces.

    The critic observes the complete current state (MDP), so it needs no history.
    """

    # Robot state
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    ee_pose = ObsTerm(func=mdp.ee_pose_b)
    ee_velocity = ObsTerm(func=mdp.ee_velocity_b, params={"hand_cfg": SceneEntityCfg("robot", body_names="panda_hand")})

    # The stage and the rewards depend on the last action, so it is part of the state
    actions = ObsTerm(func=mdp.last_action)
    stage = ObsTerm(func=mdp.cloth_drag_stage, params={"command_name": "cloth_target"})

    # Full cloth state
    cloth_nodal_pos = ObsTerm(func=mdp.cloth_nodal_pos_b)
    cloth_nodal_vel = ObsTerm(func=mdp.cloth_nodal_vel_b)
    cloth_com_pos = ObsTerm(func=mdp.cloth_com_pos_b)
    cloth_com_vel = ObsTerm(func=mdp.cloth_com_vel_b)

    # Target of the cloth's center of mass
    target_pos = ObsTerm(func=mdp.generated_commands, params={"command_name": "cloth_target"})
    target_pos_com = ObsTerm(func=mdp.target_pos_com, params={"command_name": "cloth_target"})

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class JointCriticObsCfg(CriticObsCfg):
    """Critic observations for delta joint-space actions: the shared state, no extras yet."""


@configclass
class EECriticObsCfg(CriticObsCfg):
    """Critic observations for delta end-effector actions: the shared state, no extras yet."""


@configclass
class ActorCriticObsCfg:
    """Actor (``policy``) and critic observation groups of one preset, kept as a configclass for Hydra overrides."""

    policy: ObsGroup = MISSING
    critic: ObsGroup = MISSING


@configclass
class ObservationsCfg(PresetCfg):
    """Observation presets for actor-critic training and both action spaces (joint and end-effector).

    joint_symm: actor and critic see the same joint action observations
    joint_asymm: actor sees joint observations, critic sees privileged observations
    ee_symm: actor and critic see the same ee action observations
    ee_asymm: actor sees ee observations, critic sees privileged observations
    """

    joint_symm = ActorCriticObsCfg(policy=JointActorObsCfg(), critic=JointActorObsCfg())
    joint_asymm = ActorCriticObsCfg(policy=JointActorObsCfg(), critic=JointCriticObsCfg())
    ee_symm = ActorCriticObsCfg(policy=EEActorObsCfg(), critic=EEActorObsCfg())
    ee_asymm = ActorCriticObsCfg(policy=EEActorObsCfg(), critic=EECriticObsCfg())

    default = ee_asymm


# ------------------- Command Terms -------------------
@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # Target for the cloth's center of mass on a line in front of the robot, resampled at every reset
    cloth_target = mdp.ClothTargetCommandCfg(
        asset_name="robot",
        object_name="deformable",
        ranges=mdp.ClothTargetCommandCfg.Ranges(pos_x=(0.35, 0.35), pos_y=(-0.2, 0.2), pos_z=(0.002, 0.002)),
        debug_vis=True,
    )


# ------------------- Event Terms -------------------


@configclass
class EventCfg:
    """Configuration for events."""

    # Always reset Robot to HOME_JOINT_POS
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Always reset the cloth to its flat spawn
    reset_deformable = EventTerm(
        func=mdp.reset_nodal_state_uniform,
        mode="reset",
        params={
            "position_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("deformable"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Bring the fingertips to the cloth
    reaching_cloth = RewTerm(
        func=mdp.deformable_ee_distance,
        params={"std": 0.1, "asset_cfg": SceneEntityCfg("deformable")},
        weight=5.0,
    )

    # Close the gripper at the cloth
    gripper_closed_near_cloth = RewTerm(
        func=mdp.gripper_closed_near_cloth,
        params={"grasp_radius": 0.03},
        weight=2.0,
    )

    # Close the gripper on the cloth: the cloth between the fingers keeps them from closing fully
    cloth_pinched = RewTerm(
        func=mdp.cloth_pinched,
        params={"grasp_radius": 0.03, "min_finger_opening": 0.001},
        weight=5.0,
    )

    # Do not close the gripper away from the cloth
    gripper_closed_away = RewTerm(
        func=mdp.gripper_closed_away_from_cloth,
        params={"grasp_radius": 0.03},
        weight=-0.5,
    )

    # Pull the cloth's center of mass towards the target in the table plane, only while it is grasped
    cloth_goal_tracking = RewTerm(
        func=mdp.cloth_goal_distance_xy,
        params={"std": 0.3, "command_name": "cloth_target", "grasp_radius": 0.03, "min_finger_opening": 0.001},
        weight=2.0,
    )

    # Slow the cloth down inside the target radius so it does not overshoot, only while it is grasped
    cloth_settled = RewTerm(
        func=mdp.cloth_settled_at_target,
        params={
            "std_speed": 0.05,
            "success_radius": 0.05,
            "command_name": "cloth_target",
            "grasp_radius": 0.03,
            "min_finger_opening": 0.001,
        },
        weight=2.0,
    )

    # Paid every step once the cloth has been held at the target for hold_time
    success_bonus = RewTerm(
        func=mdp.ClothHeldAtTarget,
        params={"success_radius": 0.05, "max_speed": 0.02, "hold_time": 0.5, "command_name": "cloth_target"},
        weight=20.0,
    )

    # The cloth should be dragged over the table, not lifted
    cloth_lifted = RewTerm(func=mdp.cloth_lifted, params={"max_height": 0.02}, weight=-5.0)

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Episode timeout
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Any cloth node leaves the table area or drops off the table
    cloth_out_of_bounds = DoneTerm(
        func=mdp.deformable_outside_bounds,
        params={
            "x_bounds": (0.0, 1.1),
            "y_bounds": (-0.5, 0.5),
            "z_bounds": (-0.02, 1.0),
            "asset_cfg": SceneEntityCfg("deformable"),
        },
    )

    # If robot reaches to far below the table for grabbing the cloth
    ee_below_table = DoneTerm(
        func=mdp.ee_below_minimum,
        params={"minimum_height": -0.1, "ee_frame_cfg": SceneEntityCfg("ee_frame")},
    )

    # Physics blow-up
    joint_vel_out_of_limit = DoneTerm(
        func=mdp.joint_vel_out_of_sim_limit,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""


##
# Environment configuration
##


@configclass
class FrankaClothDragEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based RL environment: Franka Panda dragging a cloth across the table."""

    # Scene settings
    scene: FrankaClothDragSceneCfg = FrankaClothDragSceneCfg(num_envs=1024, env_spacing=2.0, replicate_physics=True)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        # general settings
        self.decimation = 4
        self.episode_length_s = 5.0

        # simulation settings
        self.sim.dt = 1.0 / 120
        self.sim.render_interval = self.decimation
        self.sim.physics = PHYSICS_CFG

        # Default seed; training overwrites it with the agent seed (--seed). The cloth node sampling uses it too.
        self.seed = 42

        # Setup correct viewpoint for each visualizer. No Kit visualizer here: a Kit entry boots the Kit runtime even
        # with --viz none, and the workstation has no display. Run without any visualizer with --viz none.
        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                eye=(0.5 + 3.18, -3.18, 7.59),
                lookat=(0.5, 0.0, -0.2),
                focal_length=23.5,  # 23.5 mm to matche Kit's focal length
                headless=True,  # no display attached to the workstation
                max_visible_envs=16,
                randomly_sample_visible_envs=False,  # show environment 0 - 15
            ),
            ViserVisualizerCfg(
                eye=(0.5 + 3.18, -3.18, 7.59),
                lookat=(0.5, 0.0, -0.2),
                focal_length=23.5,
                port=8750,  # Change default port to avoid clash with other users on Comino
            ),
        ]

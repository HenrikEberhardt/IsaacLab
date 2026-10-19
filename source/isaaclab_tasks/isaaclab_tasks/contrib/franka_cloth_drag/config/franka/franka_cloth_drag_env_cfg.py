# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka cloth dragging environment."""

from __future__ import annotations

import math
import os

from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonSoftContactCfg,
    VBDSolverCfg,
)
from isaaclab_newton.sim.schemas import MujocoRigidBodyCfg, NewtonArticulationCfg, NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg
from isaaclab_visualizers.kit import KitVisualizerCfg
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

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyMappingCfg

from isaaclab_tasks.utils import PresetCfg

from ... import mdp
from ...coupler_cfg import ClothGraspCouplerProxyCfg

##
#################### Pre-defined configs ####################
##

from isaaclab_assets.robots.franka import FRANKA_PANDA_MENAGERIE_CFG  # isort:skip

# Blender cloth meshes
CLOTH_MESH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", "cloth20")

##
#################### Physics ####################
##


@configclass
class PhysicsCfg(NewtonCfg):
    """MJWarp robot and VBD cloth with the constraint-grasp proxy coupler."""

    solver_cfg = ClothGraspCouplerProxyCfg(
        entries=[
            CouplerEntryCfg(
                # MJWarp solver for rigid robot
                name="rigid",
                solver_cfg=MJWarpSolverCfg(
                    solver="newton",
                    integrator="implicitfast",
                    use_mujoco_contacts=True,
                    cone="elliptic",  # shape of the friction cone
                    iterations=100,
                    ls_iterations=30,  # line search iterations of MuJoCo constraint solver
                ),
                bodies=[r"/World/envs/env_[^/]+/Robot"],
            ),
            CouplerEntryCfg(
                # VBD solver for soft cloth
                name="soft",
                solver_cfg=VBDSolverCfg(
                    iterations=10,
                    particle_enable_self_contact=False,
                    rigid_body_particle_contact_buffer_size=1024,
                ),
                all_particles=True,
                include_static_shapes=True,  # VBD owns static shapes (e.g. table)
            ),
        ],
        proxies=[
            CouplerProxyMappingCfg(
                source="rigid",
                destination="soft",
                mode="lagged",
                # Collision of cloth only with hand and fingers of robot arm
                bodies=[
                    r"/World/envs/env_[^/]+/Robot/Geometry/.*panda_hand",
                    r"/World/envs/env_[^/]+/Robot/Geometry/.*panda_(left|right)finger",
                ],
                collide_interval=1,  # Compute collision with proxy at every step
                collision_pipeline=NewtonCollisionPipelineCfg(
                    enable_rigid_soft_full_surface_contact=True,
                ),
                mass_scale=5.0,  # heavier proxy fingers in VBD # default 1.0
            )
        ],
        iterations=1,  # Solve coupling once per step
        rest_angle_scale=0.5,  # keep half of the mesh's wrinkles as rest shape (default 0.0)
    )
    soft_contact_cfg = NewtonSoftContactCfg(
        soft_contact_ke=8.0e3,  # stiffness, higher less penetration but more unstable
        soft_contact_kd=1.0,  # contact damping, higher more stable but slower response
        soft_contact_mu=1.3,  # realistic cloth-cloth mu
    )
    num_substeps = 2

    solver_cfg.proxies[0].proxy_relaxation = 0.5


PHYSICS_CFG = PhysicsCfg()

##
#################### Scene definition ####################
##


@configclass
class FrankaClothDragSceneCfg(InteractiveSceneCfg):
    """Scene for the Franka Cloth Dragging environment."""

    # Franka Panda Robot Arm
    robot: ArticulationCfg = FRANKA_PANDA_MENAGERIE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={  # Head down home position
                "panda_joint1": 0.0,
                "panda_joint2": -0.110,
                "panda_joint3": 0.0,
                "panda_joint4": -2.272,
                "panda_joint5": 0.0,
                "panda_joint6": 2.162,
                "panda_joint7": math.pi / 4,
                "panda_finger_joint.*": 0.03,  # half the opening
            },
        ),
        spawn=FRANKA_PANDA_MENAGERIE_CFG.spawn.replace(
            # Physics Engine Setup
            variants={"Physics": "mujoco", "Colliders": "gripper_only"},
            articulation_props=NewtonArticulationCfg(self_collision_enabled=False),
            rigid_props=[MujocoRigidBodyCfg(gravcomp=1.0)],
        ),
    )

    # End-effector frame
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/panda_link0",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=(
                    "{ENV_REGEX_NS}/Robot/Geometry/panda_link0/panda_link1/panda_link2/panda_link3/"
                    "panda_link4/panda_link5/panda_link6/panda_link7/panda_hand"
                ),
                name="end_effector",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.10365)),  # centered between finger pads
                # 0.11185m to the end of the collision box, 0.1123m to the end of finger mesh
            ),
        ],
    )

    # Cloth imported from assets
    deformable: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Deformable",
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=(0.62, 0.0, 0.005),  # slightly above the table to avoid penetration at init
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(CLOTH_MESH_DIR, "cloth20_lod0_wrinkled.usd"),
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.1)),
            physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
                density=0.2,  # areal density [kg/m^2], real cotton 0.12-0.3
                particle_radius=0.002,
                tri_ke=5e2,
                tri_ka=5e2,
                tri_kd=1e-3,
                edge_ke=1.0,
                edge_kd=1e-3,
            ),
        ),
    )

    # Table
    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, -0.5]),
        spawn=sim_utils.CuboidCfg(
            size=(1.5, 1.0, 1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.12,
                dynamic_friction=0.12,
            ),
            # cloth-table friction: sqrt(soft_contact_mu * table mu) = sqrt(1.3 * 0.12) = 0.39 (cotton on wood 0.3-0.5)
        ),
    )

    # Ground plane
    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -1.0]),
        spawn=GroundPlaneCfg(),
    )

    # Lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )

    def __post_init__(self) -> None:
        # Real controller actuator settings (joint_impedance_control.cpp)
        self.robot.actuators = {
            "panda_arm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-7]"],
                joint_effort_limit={  # max torque of the joints [Nm]
                    "panda_joint[1-4]": 87.0,
                    "panda_joint[5-7]": 12.0,
                },
                joint_velocity_limit={  # max speed of the joints [rad/s]
                    "panda_joint[1-4]": 2.175,
                    "panda_joint[5-7]": 2.61,
                },
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
                armature={  # inertias of the joints
                    "panda_joint[1-2]": 0.6057,
                    "panda_joint[3-4]": 0.4625,
                    "panda_joint[5-7]": 0.2055,
                },
            ),
            # Real gripper: closes with grasp(width=0, speed=0.1, force=70) and opens with move(width=0.08, speed=0.1)
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint1"],
                joint_effort_limit=70.0,  # grasp force limit [N]
                joint_velocity_limit=0.05,
                stiffness=7.5e3,  # Tune to achieve stable simulation
                damping=100.0,  # Tune to achieve stable simulation
                armature=0.1,
            ),
            # Finger 2 mimics finger 1
            "panda_finger2_passive": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint2"],
                joint_effort_limit=70.0,
                joint_velocity_limit=0.05,
                stiffness=0.0,  # zero stiffness and damping => accept mimic constraint
                damping=0.0,
                armature=0.1,
            ),
        }


##
#################### MDP settings ####################
##

# ------------------- Action Terms -------------------
GRIPPER_ACTION_CFG = mdp.ClothGraspActionCfg(
    # Binary gripper action: open <0 or close >0
    asset_name="robot",
    joint_names=["panda_finger_joint1"],
    # Ensure controller respects joint range and velocity limits
    open_command_expr={"panda_finger_joint1": 0.03},
    close_command_expr={"panda_finger_joint1": 0.005},  # gap keeps the pads
    joint_vel_factor=0.75,  # closing speed as fraction of hardware max speed
    # Grasping parameters
    release_travel=0.002,  # reopen after failed grasp to retry
    squeeze_steps=5,  # Squeezing Model
    latch_threshold=0.5,  # action threshold to change from open to close
)


@configclass
class JointDeltaActionsCfg:
    """7-DoF delta joint position + 1-DoF binary gripper."""

    arm_action = mdp.LimitedRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        # Ensure controller respects joint range and velocity limits
        scale=0.03,  # action scale
        joint_range_margin=0.05,  # reduce range by [rad] on both sides
        joint_vel_factor=0.90,  # limit commanded veloctiy by [factor] of hardware max
    )

    gripper_action = GRIPPER_ACTION_CFG


@configclass
class EEDeltaActionsCfg:
    """4-DoF delta EE motion (dx, dy, dz, dyaw) => hand pointing down + 1-DoF binary gripper."""

    arm_action = mdp.TopDownDeltaEEActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",  # "Damped Least Squares" => stable near singularities
            ik_params={"lambda_val": 0.1},  # Damping factor
        ),
        # Ensure controller respects joint range and velocity limits
        scale_trans=0.03,  # action scale
        scale_rot=0.2,
        limit_operation_space=True,  # limit end-effector to reasonable task space
        joint_vel_factor=0.9,  # limit commanded veloctiy by [factor] of hardware max
        joint_range_margin=0.05,  # stop the whole arm 0.05 rad before one joint reaches limit
    )

    gripper_action = GRIPPER_ACTION_CFG


@configclass
class ActionsCfg(PresetCfg):
    """Action-space presets: delta joint space or delta EE task space."""

    # "env.actions=joint_delta" Flag to use joint actions
    joint_delta: JointDeltaActionsCfg = JointDeltaActionsCfg()

    # "env.actions=ee_delta" Flag to use end-effector actions
    ee_delta: EEDeltaActionsCfg = EEDeltaActionsCfg()

    default = ee_delta


# ------------------- Observation Terms -------------------
@configclass
class ActorObsCfg(ObsGroup):
    """Position-only cloth observations with five-frame history and robot proprioception."""

    # ---------- Proprioception ----------

    # Arm revolute joint positions and velocities as ratios of robot limits
    joint_pos = ObsTerm(
        func=mdp.joint_pos_ratio,
        params={"action_name": "arm_action", "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_ratio,
        params={"action_name": "arm_action", "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
    )

    # EE position and veloctiy for all 4 DoFs (x, y, z, yaw) in base frame
    ee_pose = ObsTerm(func=mdp.ee_pose_b)
    ee_velocity = ObsTerm(func=mdp.ee_velocity_b, params={"hand_cfg": SceneEntityCfg("robot", body_names="panda_hand")})

    # Gripper width (distance between the finger pads) in [m]
    gripper_width = ObsTerm(
        func=mdp.gripper_width, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"])}
    )

    # Whether the gripper holds the cloth (the real Franka reports is_grasped)
    cloth_grasped = ObsTerm(func=mdp.cloth_grasped)

    # Last action applied to the robot
    actions = ObsTerm(func=mdp.last_action)

    # ---------- Exteroception ----------

    # Nine permanent landmarks plus episode-sampled nodes, tracked consistently through position history.
    cloth_points = ObsTerm(func=mdp.ClothPointsInBaseFrame, params={"num_random_nodes": 11})

    # ---------- Task ----------

    # Explicit planar error in the same base frame as the cloth positions and EE action.
    cloth_target_displacement = ObsTerm(func=mdp.cloth_target_displacement_b)

    # Current task stage.
    stage = ObsTerm(func=mdp.cloth_drag_stage, params={"command_name": "cloth_target"})

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 5  # stack the last 5 steps of every term
        self.flatten_history_dim = True


@configclass
class CriticObsCfg(ObsGroup):
    """Privileged critic observations with full cloth positions and velocities, without history."""

    # ---------- Proprioception ----------

    joint_pos = ObsTerm(
        func=mdp.joint_pos_ratio,
        params={"action_name": "arm_action", "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_ratio,
        params={"action_name": "arm_action", "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
    )

    ee_pose = ObsTerm(func=mdp.ee_pose_b)

    ee_velocity = ObsTerm(func=mdp.ee_velocity_b, params={"hand_cfg": SceneEntityCfg("robot", body_names="panda_hand")})

    gripper_width = ObsTerm(
        func=mdp.gripper_width, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"])}
    )

    # Whether the gripper holds the cloth and the share of cloth nodes locked to the hand
    cloth_grasped = ObsTerm(func=mdp.cloth_grasped)
    grasped_node_ratio = ObsTerm(func=mdp.grasped_node_ratio)

    actions = ObsTerm(func=mdp.last_action)

    # ---------- Exterioceptive ----------

    # Privileged simulation information (position and velocity of all nodes)
    cloth_nodal_pos = ObsTerm(func=mdp.cloth_nodal_pos_b)
    cloth_nodal_vel = ObsTerm(func=mdp.cloth_nodal_vel_b)

    # ---------- Task ----------

    # Target and central cloth node positions in the base frame
    target_pos = ObsTerm(func=mdp.generated_commands, params={"command_name": "cloth_target"})
    cloth_center_pos = ObsTerm(func=mdp.cloth_center_pos_b)
    cloth_target_displacement = ObsTerm(func=mdp.cloth_target_displacement_b)
    stage = ObsTerm(func=mdp.cloth_drag_stage, params={"command_name": "cloth_target"})

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ObservationsCfg:
    """The actor (``policy``) and the privileged critic (``critic``)"""

    policy: ActorObsCfg = ActorObsCfg()
    critic: CriticObsCfg = CriticObsCfg()


# ------------------- Command Terms -------------------
@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # Specify the task and stage objectives that wants to be achieved by the agent
    cloth_target = mdp.ClothTargetCommandCfg(
        asset_name="robot",
        object_name="deformable",
        # Task goal definition
        ranges=mdp.ClothTargetCommandCfg.Ranges(
            # x >= 0.40: a grasp on the robot-facing edge trails the central node by ~10 cm, and the hand reaches
            # only ~0.28 m towards the base at grasp height # was (0.35, 0.35)
            pos_x=(0.40, 0.45),
            pos_y=(-0.2, 0.2),
            pos_z=(0.002, 0.002),  # Spawn line for target selection
        ),
        success_radius=0.05,  # Radius [m] around target for success (target disc)
        hold_time=0.5,  # Duration [s] the center node must stay inside the target disc to count as success
        debug_vis=True,  # Show marker at the target location
        # Stage transitions conditions
        approach_radius=0.07,  # APPROACH -> GRASP: EE frame origin closer than [m] to any cloth node
        grasp_min_nodes=3,  # GRASP -> DRAG: constraint locks (fingers stopped closing) with at least this many nodes
        # was 5. LOD0 mesh: an EE at 13.5 mm selects 5 nodes, at 15 mm mostly 1-3, so 4 forces low grasps
        goal_zone_margin=0.075,  # DRAG -> GOAL: center node closer than success_radius + this (0.125 m) to the target
    )


# ------------------- Event Terms -------------------


@configclass
class EventCfg:
    """Configuration for events."""

    # Reset robot with variation in ee position (x = 0.19-0.42, y = -0.13-0.13 and z = 0.19-0.42 m)
    reset_robot_joints = EventTerm(
        func=mdp.reset_franka_top_down,
        mode="reset",
        params={
            "joint_offset_ranges": {
                "panda_joint1": (-0.25, 0.25),
                "panda_joint2": (-0.15, 0.15),
                "panda_joint4": (-0.15, 0.15),
                "panda_joint7": (-0.6, 0.6),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Reset the cloth with randomization in position and orientation
    reset_deformable = EventTerm(
        func=mdp.reset_cloth_planar_pose,
        mode="reset",
        params={
            "position_range": {"x": (-0.03, 0.03), "y": (-0.05, 0.05)},
            "yaw_range": (-0.25 * math.pi, 0.25 * math.pi),
            "asset_cfg": SceneEntityCfg("deformable"),
        },
    )


@configclass
class RewardsCfg:
    """Progress shaping with a small grasp aid and a discrete task-success bonus."""

    # Integrated progress: 10 cm of approach pays 1; 10 cm of attached transport pays 10.
    approach_progress = RewTerm(
        func=mdp.ClothDistanceProgress, params={"objective": "approach", "distance_scale": 0.1}, weight=1.0
    )
    grasp_between_fingers = RewTerm(
        func=mdp.cloth_between_fingers,
        params={"num_nodes_full": 3, "stages": ["grasp"], "saturate": False},
        weight=0.05,
    )
    drag_progress = RewTerm(
        func=mdp.ClothDistanceProgress, params={"objective": "drag", "distance_scale": 0.1}, weight=10.0
    )
    success = RewTerm(func=mdp.cloth_drag_success_bonus, weight=100.0)

    pen_termination = RewTerm(
        func=mdp.is_terminated_term, params={"term_keys": ["cloth_out_of_bounds", "ee_below_table"]}, weight=-200.0
    )
    pen_action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)
    pen_joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
        weight=-1.0,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    success = DoneTerm(func=mdp.cloth_drag_success)

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

    # If robot reaches to far below the table (relevant with joint delta actions)
    ee_below_table = DoneTerm(
        func=mdp.ee_below_minimum,
        params={"minimum_height": -0.1, "ee_frame_cfg": SceneEntityCfg("ee_frame")},
    )

    # Physics blow-up that ended in NaN or Inf, which every threshold check above misses
    nonfinite_state = DoneTerm(func=mdp.nonfinite_state)

    # Termination if joints exceed their limits (gripper is excluded)
    joint_vel_out_of_limit = DoneTerm(
        func=mdp.joint_vel_out_of_sim_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
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
    scene: FrankaClothDragSceneCfg = FrankaClothDragSceneCfg(num_envs=1024, env_spacing=2.5, replicate_physics=True)

    # MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        # General settings
        self.decimation = 4
        self.episode_length_s = 10.0

        # simulation settings
        self.sim.dt = 1.0 / 120
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysicsCfg()

        # Setup Visulaizer settings
        self.sim.visualizer_cfgs = [
        
            KitVisualizerCfg(
                eye=(0.5 + 3.18, -3.18, 7.59),
                lookat=(0.5, 0.0, -0.2),
                headless=True,
                max_visible_envs=16,
                randomly_sample_visible_envs=False,  # show environment 0 - 15
            ),
            NewtonGLVisualizerCfg(
                eye=(0.5 + 3.18, -3.18, 7.59),
                lookat=(0.5, 0.0, -0.2),
                focal_length=23.5,  # 23.5 mm matches Kit's focal length
                headless=True,
                max_visible_envs=16,
                randomly_sample_visible_envs=False,
            ),
            ViserVisualizerCfg(
                eye=(0.5 + 3.18, -3.18, 7.59),
                lookat=(0.5, 0.0, -0.2),
                focal_length=23.5,
                port=8750,  # Change default port to avoid clash with other users on Comino
            )
        ]
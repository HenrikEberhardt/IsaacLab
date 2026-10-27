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
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
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
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
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
                # MJWarp solver for RIGID robot
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
                # VBD solver for Soft Cloth
                name="soft",
                solver_cfg=VBDSolverCfg(
                    iterations=10,
                    particle_enable_self_contact=False,
                    # compliant ALM for VBD rigid-rigid interaction (gripper-cube-table)
                    rigid_compliant_alm=True,
                    rigid_body_particle_contact_buffer_size=1024,  # rigid-cloth
                    rigid_body_contact_buffer_size=64,  # rigid-rigid
                    friction_epsilon=1.0e-2,  # dynamic-static friction threshold
                ),
                bodies=[r"/World/envs/env_[^/]+/Cube"],  # VBD integrates the cube with the cloth
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
        # No cube-table collision: a cube that leaves the cloth falls through the table (cube_fell termination)
        collision_filter_pairs=[(r"/World/envs/env_[^/]+/Cube", r"/World/envs/env_[^/]+/Table(/.*)?")],
        # Gripper touching the cube, latched after every substep (gripper_cube_contact termination)
        monitored_contacts={
            "gripper_cube": (
                r"/World/envs/env_[^/]+/Robot/Geometry/.*panda_(hand|leftfinger|rightfinger)",
                r"/World/envs/env_[^/]+/Cube",
            ),
        },
        contact_touch_tolerance=1.0e-3,  # largest surface gap [m] that counts as touching
    )
    soft_contact_cfg = NewtonSoftContactCfg(
        soft_contact_ke=8.0e3,  # stiffness, higher less penetration but more unstable
        soft_contact_kd=1.0,  # contact damping, higher more stable but slower response
        soft_contact_mu=1.5,  # realistic cloth-cloth mu
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
            # Material of all gripper colliders (finger pads and hand capsule), replaces the asset's material
            physics_material=RigidBodyMaterialBaseCfg(
                dynamic_friction=1.5,  # finger friction
            ),
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
                # z = 8.5 for ideal cloth interaction
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
            usd_path=os.path.join(CLOTH_MESH_DIR, "120_nodes_optimized_mesh.usd"),
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.1)),
            physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
                density=0.2,  # areal density [kg/m^2], real cotton 0.12-0.3
                particle_radius=0.002,
                tri_ke=1e2,  # 5e2
                tri_ka=2e3,  # 5e2
                tri_kd=1e-3,  # 1e-3
                edge_ke=0.7,  # 1.0
                edge_kd=1e-3,  # 1e-3
            ),
        ),
    )

    # Cube spawns on central cloth node
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        # z = cloth spawn height + clearance + half edge length [m]
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.62, 0.0, 0.005 + 0.004 + 0.5 * 0.04)),
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),  # edge length [m]
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.3, 0.9)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),  # [kg]
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=RigidBodyMaterialBaseCfg(
                dynamic_friction=1.5,  # total friction: sqrt(soft_contact_mu * cube mu)
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
            physics_material=RigidBodyMaterialBaseCfg(
                dynamic_friction=0.20,
            ),
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
                stiffness=7.5e3,  # 3e4
                damping=150.0,  # 150 for closer sim2real
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
    # Binary gripper action: open <0 or close >0 (e.g sign from distance)
    asset_name="robot",
    joint_names=["panda_finger_joint1"],
    # Ensure controller respects joint range and velocity limits
    open_command_expr={"panda_finger_joint1": 0.03},
    close_command_expr={"panda_finger_joint1": 0.005},  # gap keeps the pads
    joint_vel_factor=0.75,  # closing speed as fraction of hardware max speed
    latch_threshold=0.5,  # action threshold to change from open to close
    # Grasping parameters
    release_travel=0.002,  # reopen after failed grasp to retry
    squeeze_steps=5,  # Squeezing Model
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
        scale_trans=0.025,  # [m] per step at action 1 (0.75 m/s); 0.03 left the mean action saturated 75 % of the time
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

    # Gripper information
    gripper_width = ObsTerm(
        func=mdp.gripper_width, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"])}
    )
    # Privileged grasp flag (+1 / -1), hidden for whole episodes by the curriculum after success (zero)
    cloth_grasped = ObsTerm(func=mdp.cloth_grasped, params={"hide_curriculum": "grasp_hiding"})

    # Last action applied to the robot
    actions = ObsTerm(func=mdp.last_action)

    # ---------- Exteroception ----------

    # Nine permanent landmarks, the episode's raised nodes and 18 episode-sampled nodes, drawn at each episode beginning
    # and kept fixed, so every slot tracks the same node through the position history
    cloth_points = ObsTerm(func=mdp.ClothPointsInBaseFrame, params={"num_random_nodes": 18})

    # Cube position and orientation (quaternion, w >= 0) in base frame
    cube_pose = ObsTerm(func=mdp.object_pose_b, params={"asset_cfg": SceneEntityCfg("cube")})

    # ---------- Task ----------

    # Target relative to the EE along the base axes (forward kinematics and command, known on the real robot)
    target_pos_ee = ObsTerm(func=mdp.target_pos_ee, params={"command_name": "cloth_target"})

    # Current task stage, hidden for whole episodes by the curriculum after success (all zeros)
    stage = ObsTerm(
        func=mdp.cloth_drag_stage, params={"command_name": "cloth_target", "hide_curriculum": "stage_hiding"}
    )

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 5  # stack the last 5 steps of every term
        self.flatten_history_dim = True


@configclass
class CriticObsCfg(ObsGroup):
    """Privileged critic observations with full cloth positions and velocities, without history."""

    # ---------- Proprioception ----------

    # Joint information
    joint_pos = ObsTerm(
        func=mdp.joint_pos_ratio,
        params={"action_name": "arm_action", "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_ratio,
        params={"action_name": "arm_action", "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
    )

    # EE information
    ee_pose = ObsTerm(func=mdp.ee_pose_b)
    ee_velocity = ObsTerm(func=mdp.ee_velocity_b, params={"hand_cfg": SceneEntityCfg("robot", body_names="panda_hand")})

    # Privileged gripper information
    cloth_grasped = ObsTerm(func=mdp.cloth_grasped)
    grasped_node_ratio = ObsTerm(func=mdp.grasped_node_ratio)
    gripper_width = ObsTerm(
        func=mdp.gripper_width, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"])}
    )

    # Last actions
    actions = ObsTerm(func=mdp.last_action)

    # ---------- Exterioceptive ----------

    # Privileged simulation information (position and velocity of all nodes)
    cloth_nodal_pos = ObsTerm(func=mdp.cloth_nodal_pos_b)
    cloth_nodal_vel = ObsTerm(func=mdp.cloth_nodal_vel_b)

    # Cube pose, velocity, offset from the cloth's central node (sliding) and distance to the EE frame origin
    cube_pose = ObsTerm(func=mdp.object_pose_b, params={"asset_cfg": SceneEntityCfg("cube")})
    cube_velocity = ObsTerm(func=mdp.object_velocity_b, params={"asset_cfg": SceneEntityCfg("cube")})
    cube_center_offset = ObsTerm(func=mdp.object_center_offset_b, params={"asset_cfg": SceneEntityCfg("cube")})
    ee_cube_distance = ObsTerm(func=mdp.ee_object_distance, params={"asset_cfg": SceneEntityCfg("cube")})
    # Gripper touched the cube in the last step (solver contacts)
    gripper_cube_contact = ObsTerm(func=mdp.termination_flag, params={"term_name": "gripper_cube_contact"})

    # ---------- Task ----------

    # Target position in base frame
    target_pos = ObsTerm(func=mdp.generated_commands, params={"command_name": "cloth_target"})

    # Cloth center node position in base frame
    cloth_center_pos = ObsTerm(func=mdp.cloth_center_pos_b)

    # Displacement from center node to target position in base frame
    cloth_target_displacement = ObsTerm(func=mdp.cloth_target_displacement_b)

    # Displacement from cube to target position in base frame
    cube_target_displacement = ObsTerm(
        func=mdp.target_object_displacement_b,
        params={"command_name": "cloth_target", "asset_cfg": SceneEntityCfg("cube")},
    )

    # Current task stage
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
    """Task specifications for the MDP."""

    cloth_target = mdp.ClothTargetCommandCfg(
        asset_name="robot",
        object_name="deformable",
        goal_object_name="cube",  # the cube must reach the target (success, GOAL stage, drag progress)
        # Task goal definition
        ranges=mdp.ClothTargetCommandCfg.Ranges(  # Target sampled on line
            pos_x=(0.40, 0.45),
            pos_y=(-0.2, 0.2),
            pos_z=(0.002, 0.002),
        ),
        success_radius=0.04,  # Radius [m] around the target position for success
        hold_time=0.75,  # Time [s] to hold the cube in the target zone for success
        max_ee_height=0.05,  # success only with low EE (maybe delete this)
        # Success needs a raised node locked during the hold; pushing or dragging with different nodes is not enough.
        debug_vis=True,  # Target marker
        # Task stage transitions
        approach_radius=0.04,  # APPROACH -> GRASP: EE frame closer than [m] to a raised node (any node if none)
        grasp_min_nodes=2,  # GRASP -> DRAG: a grasp locking at least this many nodes to the EE, one of them raised
        goal_zone_margin=0.06,  # DRAG -> GOAL: center node closer than success_radius + this (0.10 m) to the target
    )


# ------------------- Event Terms -------------------


@configclass
class ResetFromBankCfg(PresetCfg):
    """Reset-state presets: default resets only, or the default reset and the bank's stored states, each drawn
    uniformly inside its bucket."""

    # "env.events.reset_from_bank=off" Default resets only (the term just logs the success per reset source)
    off = EventTerm(
        func=mdp.ResetFromBank,
        mode="reset",
        params={"bank_path": None, "mode_probabilities": {"default": 1.0}, "command_name": "cloth_target"},
    )

    # "env.events.reset_from_bank=uniform" Share of the resets per mode (default reset or a stored state of the bucket:
    # pre-grasp, fresh grasp before the drag, goal-zone edge); states drawn uniformly inside each bucket
    uniform = EventTerm(
        func=mdp.ResetFromBank,
        mode="reset",
        params={
            "bank_path": "logs/reset_banks/bank_5000.pt",  # relative to the working directory or the repository root
            "mode_probabilities": {"default": 0.4, "grasp": 0.35, "drag": 0.15, "goal": 0.10},
            "command_name": "cloth_target",
        },
    )

    default = uniform


@configclass
class EventCfg:
    """Configuration for events."""

    # Robot resets
    reset_robot_joints = EventTerm(
        func=mdp.reset_franka_top_down,
        mode="reset",
        params={
            "joint_offset_ranges": {  # linear ee pos offset through joint variations
                "panda_joint1": (-0.25, 0.25),
                "panda_joint2": (-0.15, 0.15),
                "panda_joint4": (-0.15, 0.15),
                "panda_joint7": (-0.6, 0.6),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Cloth resets
    reset_deformable = EventTerm(
        func=mdp.ResetClothWithRaisedNodes,
        mode="reset",
        params={
            # Cloth position and orientation
            "position_range": {"x": (-0.03, 0.03), "y": (-0.05, 0.05)},
            "yaw_range": (-0.25 * math.pi, 0.25 * math.pi),  # symmetrie covers all angles
            # Node perturbation
            "include_edge_nodes": False,
            "num_raised_nodes": 3,  # redrawn at every episode reset
            "node_raise_height": 0.01,  # height above the table [m]
            "min_spacing": 0.05,  # minimum distance between raised nodes [m]
            "asset_cfg": SceneEntityCfg("deformable"),
        },
    )

    # Cube reset (position offset from central cloth node)
    reset_cube = EventTerm(
        func=mdp.reset_object_on_cloth_center,
        mode="reset",
        params={
            # Cube position and orientation (position offset from cloth center node)
            "position_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (0.004, 0.004)},
            "yaw_range": (-0.25 * math.pi, 0.25 * math.pi),
            "object_cfg": SceneEntityCfg("cube"),
            "cloth_cfg": SceneEntityCfg("deformable"),
        },
    )

    # Reset states from the offline bank (implementation_log/reset_states.md), after the resets above: it overwrites
    # them. "env.events.reset_from_bank=uniform" selects the bank; also logs the success per reset source
    reset_from_bank: ResetFromBankCfg = ResetFromBankCfg()

    # EE frame sensor refresh after the resets (Newton updates it only in a physics step, so the first observation
    # and grasp check would see the previous episode's pose); runs after every reset event, whatever its position
    refresh_frame_sensors = EventTerm(
        func=mdp.RefreshFrameSensors,
        mode="reset",
        params={"sensor_names": ["ee_frame"]},
    )


@configclass
class RewardsCfg:
    """Progress shaping with a small grasp aid and a discrete task-success bonus."""

    # Integrated progress: 10 cm of approach to the nearest raised node pays 1; 10 cm of attached transport of the
    # central node pays 10.
    approach_progress = RewTerm(
        func=mdp.ClothDistanceProgress, params={"objective": "approach", "distance_scale": 0.1}, weight=1.0
    )

    # Closing earns a small aid only when the attachment selector can capture a raised node.
    grasp_close_on_bump = RewTerm(func=mdp.gripper_closed_on_raised_node, params={"stages": ["grasp"]}, weight=0.05)

    drag_progress = RewTerm(
        func=mdp.ClothDistanceProgress, params={"objective": "drag", "distance_scale": 0.1}, weight=10.0
    )

    success = RewTerm(func=mdp.cloth_drag_success_bonus, weight=100.0)

    # Penalties
    pen_termination = RewTerm(
        func=mdp.is_terminated_term,
        params={"term_keys": ["cloth_out_of_bounds", "ee_below_table", "cube_fell", "gripper_cube_contact"]},
        weight=-200.0,
    )

    # Arm actions only (dx, dy, dz, dyaw): the binary gripper action's raw value changes without moving anything.
    # The weight is the curriculum's start weight.
    pen_action_magnitude = RewTerm(func=mdp.action_term_l2, params={"action_name": "arm_action"}, weight=-1e-3)
    pen_action_rate = RewTerm(func=mdp.action_term_rate_l2, params={"action_name": "arm_action"}, weight=-1e-2)
    # Steep penalty only within 0.1 rad of the action's joint bounds (wrist: the yaw window), zero elsewhere; arm only
    pen_joint_limit_barrier = RewTerm(
        func=mdp.joint_limit_barrier,
        params={
            "action_name": "arm_action",
            "margin": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
        },
        weight=-1.0,
    )
    # Approach with open fingers; closing and alignment attempts in GRASP carry no penalty.
    pen_close_in_approach = RewTerm(func=mdp.gripper_closed, params={"stages": ["approach"]}, weight=-0.05)

    # Drag, don't lift: linear penalty if the EE is above 3cm
    pen_ee_height = RewTerm(
        func=mdp.ee_height_above, params={"max_height": 0.03, "stages": ["drag", "goal"]}, weight=-50.0
    )

    # Cube sliding on the cloth: planar cube speed relative to the central node above the friction creep
    # (prepared, off until a run shows sliding; creep analysis in implementation_log/rewards.md section 20)
    # Threshold 0.01 m/s = VBD friction_epsilon: smoothed-friction creep stays below it, faster is real sliding
    # pen_cube_slip = RewTerm(
    #     func=mdp.object_slip_speed,
    #     params={"threshold": 0.01, "stages": ["drag", "goal"], "asset_cfg": SceneEntityCfg("cube")},
    #     weight=-10.0,
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Task success
    success = DoneTerm(func=mdp.cloth_drag_success)

    # Episode timeout
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Any cloth node leaves the table area (e.g. cloth lost)
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

    # Cube left the cloth: it does not collide with the table, so it falls through (-10 cm below the table top)
    cube_fell = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.1, "asset_cfg": SceneEntityCfg("cube")}
    )

    # Gripper (hand or fingers) touched the cube, from the solver contacts after every physics substep
    gripper_cube_contact = DoneTerm(func=mdp.physics_contact, params={"contact_name": "gripper_cube"})

    # Physics blow-up that ended in NaN or Inf, which every threshold check above misses
    nonfinite_state = DoneTerm(func=mdp.nonfinite_state)

    # Termination if joints exceed their limits (gripper is excluded)
    joint_vel_out_of_limit = DoneTerm(
        func=mdp.joint_vel_out_of_sim_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP: stage hiding first, then grasp-flag hiding and the action rate in parallel."""

    # 1. Hide actor observation of current task stage (easy)
    stage_hiding = CurrTerm(
        func=mdp.ObservationHidingCurriculum,
        params={
            "command_name": "cloth_target",
            "success_threshold": 0.80,  # success ratio; the cube task levels off at about 0.87-0.89
            "dwell_steps": 2400,  # 100 iterations above the threshold before the ramp starts
            "resume_dwell_steps": 240,  # 10 iterations back above the threshold before a paused ramp continues
            "ramp_steps": 3600,  # 150 iterations at 24 steps/iteration
            "ema_episodes": 2048,
            "initial_progress": 0.1,  # stage hidden in 10 % of episodes from the start, never relied on
            "reset_term": "reset_from_bank",  # success gate on episodes from default resets only
        },
    )

    # 2. Hide actor grasp flag as this is unknown on real Franka (the DRAG stage would reveal the grasp)
    grasp_hiding = CurrTerm(
        func=mdp.ObservationHidingCurriculum,
        params={
            "command_name": "cloth_target",
            "success_threshold": 0.80,
            "dwell_steps": 600,  # 25 iterations
            "resume_dwell_steps": 240,  # 10 iterations
            "ramp_steps": 3000,  # 125 iterations
            "ema_episodes": 2048,
            "after": "stage_hiding",
            "initial_progress": 0.1,  # hidden in 10 % of episodes from the start (from 0 %, success fell 0.06-0.09)
            "reset_term": "reset_from_bank",  # success gate on episodes from default resets only
        },
    )

    # 3. Penalize sustained full-range arm actions once the stage is hidden. At full weight, saturated motion for a
    # typical 5-10 s successful episode costs about 2-6 points against the 100-point success bonus.
    action_magnitude = CurrTerm(
        func=mdp.ActionRateCurriculum,
        params={
            "command_name": "cloth_target",
            "success_threshold": 0.80,
            "dwell_steps": 600,  # 25 iterations
            "resume_dwell_steps": 240,  # 10 iterations
            "ramp_steps": 7200,  # 300 iterations
            "reward_term": "pen_action_magnitude",
            "final_weight": -0.2,
            "ema_episodes": 2048,
            "after": "stage_hiding",  # in parallel with grasp-flag hiding and action-rate regularization
            "initial_progress": 0.0,
            "reset_term": "reset_from_bank",  # success gate on episodes from default resets only
        },
    )

    # 4. Increase action rate penalty for smoother trajectories, alongside grasp-flag hiding once the stage is hidden
    action_rate = CurrTerm(
        func=mdp.ActionRateCurriculum,
        params={
            "command_name": "cloth_target",
            "success_threshold": 0.80,
            "dwell_steps": 600,  # 25 iterations
            "resume_dwell_steps": 240,  # 10 iterations
            "ramp_steps": 7200,  # 300 iterations
            "reward_term": "pen_action_rate",
            "final_weight": -0.6,  # -1.0 was ~12 % of the success reward and raised touch terminations
            "ema_episodes": 2048,
            "after": "stage_hiding",  # in parallel with grasp-flag hiding
            "initial_progress": 0.0,
            "reset_term": "reset_from_bank",  # success gate on episodes from default resets only
        },
    )


##
# Environment configuration
##


@configclass
class FrankaClothDragEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based RL environment: Franka Panda dragging a cloth across the table."""

    # Scene settings
    scene: FrankaClothDragSceneCfg = FrankaClothDragSceneCfg(num_envs=2048, env_spacing=2.2, replicate_physics=True)

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

        # Simulation settings
        self.sim.dt = 1.0 / 120
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysicsCfg()

        """
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
                port=8755,  # Change default port to avoid clash with other users on Comino
            ),
        ]
        """

    def play_mode(self):
        """Play with full curriculum progress"""
        super().play_mode()
        if self.curriculum is not None:
            # Play with full curriculum progress
            if self.curriculum.stage_hiding is not None:
                self.curriculum.stage_hiding.params["initial_progress"] = 1.0
            if self.curriculum.grasp_hiding is not None:
                self.curriculum.grasp_hiding.params["initial_progress"] = 1.0
            if self.curriculum.action_rate is not None:
                self.curriculum.action_rate.params["initial_progress"] = 1.0

        # Play and evaluation start from the task's default resets only, never from stored reset states
        if self.events.reset_from_bank is not None:
            self.events.reset_from_bank.params["bank_path"] = None
            self.events.reset_from_bank.params["mode_probabilities"] = {"default": 1.0}

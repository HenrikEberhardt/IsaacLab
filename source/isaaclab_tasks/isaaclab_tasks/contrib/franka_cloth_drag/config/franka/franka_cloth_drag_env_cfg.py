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
from isaaclab_visualizers.viser import ViserVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg, UniformNoiseCfg

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyMappingCfg

from isaaclab_tasks.utils import PresetCfg

from ... import mdp, reset_strategies
from ...constrained_gripper import ConstrainedGripperActionCfg
from ...constrained_gripper.robotiq import (
    FRANKA_PANDA_ROBOTIQ_CFG,
    ROBOTIQ_CLOSED,
    ROBOTIQ_DRIVE_JOINT,
    ROBOTIQ_GRIPPER_ACTUATOR_CFG,
    ROBOTIQ_OPEN,
    ROBOTIQ_PASSIVE_ACTUATOR_CFG,
    stiffen_robotiq_mimics,
)
from ...physics import ClothDragCouplerCfg

##
#################### Pre-defined configs ####################
##

# Blender cloth meshes
CLOTH_MESH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", "cloth20")

##
#################### Physics ####################
##

# Robotiq 2F-85 bodies with colliders, as the labels of their prims. The outer knuckles carry none: the outer-finger
# box encloses them (fixed joint).
ROBOTIQ_BODIES = (
    r"/World/envs/env_[^/]+/Robot/Geometry/.*/Robotiq/"
    r"(base_link|(left|right)_(inner_finger|outer_finger|inner_knuckle))"
)


@configclass
class PhysicsCfg(NewtonCfg):
    """MJWarp robot and VBD cloth with the cloth-drag proxy coupler."""

    solver_cfg = ClothDragCouplerCfg(
        entries=[
            CouplerEntryCfg(
                name="rigid",
                solver_cfg=MJWarpSolverCfg(
                    solver="newton",
                    integrator="implicitfast",
                    use_mujoco_contacts=True,
                    cone="elliptic",  # shape of the friction cone
                    iterations=100,
                    ls_iterations=30,  # LS iterations of constraint solver
                ),
                bodies=[r"/World/envs/env_[^/]+/Robot"],
            ),
            CouplerEntryCfg(
                name="soft",
                solver_cfg=VBDSolverCfg(
                    iterations=10,
                    particle_enable_self_contact=False,
                    rigid_compliant_alm=True,  # AVBD for rigid-rigid contacts
                    rigid_body_particle_contact_buffer_size=1024,  # rigid-soft
                    rigid_body_contact_buffer_size=64,  # rigid-rigid
                    friction_epsilon=1.0e-2,  # dynamic-static friction threshold
                ),
                bodies=[r"/World/envs/env_[^/]+/Cube"],  # Include Cube
                all_particles=True,
                include_static_shapes=True,  # Include Table
            ),
        ],
        proxies=[
            CouplerProxyMappingCfg(
                source="rigid",
                destination="soft",
                mode="lagged",
                bodies=[ROBOTIQ_BODIES],  # the gripper collides with the cloth, the arm does not
                collide_interval=1,  # Compute collision with proxy at every step
                collision_pipeline=NewtonCollisionPipelineCfg(
                    enable_rigid_soft_full_surface_contact=True,
                ),
                mass_scale=5.0,
            )
        ],
        iterations=1,  # Solve coupling once per step
        # No Cube-Tablecollision
        collision_filter_pairs=[(r"/World/envs/env_[^/]+/Cube", r"/World/envs/env_[^/]+/Table(/.*)?")],
        # Monitor Franka-Cube Contacts
        monitored_contacts={
            "gripper_cube": (
                ROBOTIQ_BODIES,
                r"/World/envs/env_[^/]+/Cube",
            ),
        },
        contact_touch_tolerance=1.0e-3,
    )
    soft_contact_cfg = NewtonSoftContactCfg(
        soft_contact_ke=8.0e3,  # stiffness, higher less penetration but more unstable
        soft_contact_kd=1.0,  # contact damping, higher more stable but slower response
        soft_contact_mu=1.5,
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
    robot: ArticulationCfg = FRANKA_PANDA_ROBOTIQ_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={  # Home position with head facing down
                "panda_joint1": 0.0,
                "panda_joint2": -0.110,
                "panda_joint3": 0.0,
                "panda_joint4": -2.272,
                "panda_joint5": 0.0,
                "panda_joint6": 2.162,
                "panda_joint7": math.pi / 4,
            },
        ),
        spawn=FRANKA_PANDA_ROBOTIQ_CFG.spawn.replace(
            articulation_props=NewtonArticulationCfg(self_collision_enabled=False),
            rigid_props=[MujocoRigidBodyCfg(gravcomp=1.0)],
            physics_material=RigidBodyMaterialBaseCfg(
                dynamic_friction=1.5,  # Friction of the Franka Colliders
            ),
        ),
    )

    # End Effector frame
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/panda_link0",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=(
                    "{ENV_REGEX_NS}/Robot/Geometry/panda_link0/panda_link1/panda_link2/panda_link3/"
                    "panda_link4/panda_link5/panda_link6/panda_link7/robotiq_mount/Robotiq/base_link"
                ),
                name="end_effector",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.1429)),  # centered between the closed finger pads
            ),
        ],
    )

    # Cloth
    deformable: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Deformable",
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=(0.62, 0.0, 0.005),  # above table to avoid penetration at init
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(CLOTH_MESH_DIR, "120_nodes_optimized_mesh.usd"),
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.1)),
            physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
                density=0.2,  # areal density [kg/m^2]
                particle_radius=0.002,
                tri_ke=1e2,  # 5e2
                tri_ka=2e3,  # 5e2
                tri_kd=1e-3,  # 1e-3
                edge_ke=0.7,  # 1.0
                edge_kd=1e-3,  # 1e-3
            ),
        ),
    )

    # Cube
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.62, 0.0, 0.005 + 0.004 + 0.5 * 0.04)),
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),  # [m]
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.3, 0.9)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),  # [kg]
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=RigidBodyMaterialBaseCfg(
                dynamic_friction=1.5,  # Total friction: sqrt(contact_proxy_mu * cube_mu)
            ),
        ),
    )

    # Table
    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.55, 0.0, -0.02]),
        spawn=sim_utils.CuboidCfg(
            size=(1.4, 0.8, 0.04),  # [m]
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=RigidBodyMaterialBaseCfg(
                dynamic_friction=0.20,  # Total friction: sqrt(contact_proxy_mu * table_mu)
            ),
        ),
    )

    # Ground
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
        # Real controller actuator settings (Franka joint_impedance_control.cpp)
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
                armature={  # inertias / delay of the joints
                    "panda_joint[1-2]": 0.6057,
                    "panda_joint[3-4]": 0.4625,
                    "panda_joint[5-7]": 0.2055,
                },
                viscous_friction=0.0,
            ),
            # Add the Robotiq 2F-85 Gripper Linkage
            "gripper": ROBOTIQ_GRIPPER_ACTUATOR_CFG.replace(min_delay=4, max_delay=6),
            "gripper_passive": ROBOTIQ_PASSIVE_ACTUATOR_CFG,
        }


##
#################### MDP settings ####################
##

# ------------------- Action Terms -------------------
GRIPPER_ACTION_CFG = ConstrainedGripperActionCfg(
    # Binary Robotiq Gripper
    asset_name="robot",
    joint_names=[ROBOTIQ_DRIVE_JOINT],
    finger_joint_name=ROBOTIQ_DRIVE_JOINT,
    open_command_expr={ROBOTIQ_DRIVE_JOINT: ROBOTIQ_OPEN},
    close_command_expr={ROBOTIQ_DRIVE_JOINT: ROBOTIQ_CLOSED},
    
    # Constrained Gripper Parameters
    settle_speed=0.05,  # [rad/s] lower finger speed counts as settled
    release_travel=0.004,  # [m] gap growth after a failed grasp to re-arm
    release_gap=0.002,  # [m] gap growth unlock the nodes
    squeeze_steps=5,
    grasp_min_nodes=2,
    contact_clearance=0.01,  # [m] distance to restore collisions
)


@configclass
class ActionsCfg:
    """5-DoF delta EE motion => head pointing down (dx, dy, dz, dyaw) + binary gripper."""

    arm_action = mdp.TopDownDeltaEEActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",  # dampled least squares => stable around singularities
            ik_params={"lambda_val": 0.1},
        ),
        # Ensure controller respects joint range and velocity limits
        scale_trans=(0.015, 0.015, 0.015),  # [m] per action step (dx, dy, dz)
        scale_rot=0.2,
        limit_operation_space=True,
        joint_vel_factor=0.9,  # reduce joint veloctiy by [factor] of hardware max
        joint_range_margin=0.05,  # reduce joint range by 0.05 [rad] of hardware max
        command_delay_range=(0, 4),  # IK target latency steps at 120 Hz
    )

    gripper_action = GRIPPER_ACTION_CFG


# ------------------- Observation Terms -------------------
@configclass
class ActorObsCfg(ObsGroup):
    """Proprioception, Exterioception and Task Observation with five-frame history."""
    # ---------- Proprioception ----------
    joint_angles = ObsTerm(
        func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])}
    )
    joint_velocities = ObsTerm(
        func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])}
    )
    ee_pose = ObsTerm(func=mdp.ee_pose_b)
    ee_velocity = ObsTerm(
        func=mdp.ee_velocity_b, params={"hand_cfg": SceneEntityCfg("robot", body_names="base_link")}
    )
    # Randomize gripper width to avoid detecting grasp based on finger pad width
    gripper_finger_width = ObsTerm(
        func=mdp.gripper_width,
        noise=NoiseModelWithAdditiveBiasCfg(
            noise_cfg=GaussianNoiseCfg(std=0.0004),  # [m] Robotiq position resolution
            bias_noise_cfg=UniformNoiseCfg(n_min=-0.005, n_max=0.005, operation="abs"),  # [m] offset per episode
        ),
    )
    last_action = ObsTerm(
        func=mdp.last_action
        )

    # ---------- Exteroception ----------
    # 3x3 node landmarks, the raised node and 18 random nodes, fixed per episode
    cloth_points = ObsTerm(
        func=mdp.ClothPointsInBaseFrame,
        params={"num_random_nodes": 18, "command_name": "cloth_target"},
        noise=mdp.ClothPointsNoiseModelCfg(
            num_raised=1,
            cloth_offset_range=(-0.003, 0.003),  # [m] per episode, all points
            raised_offset_range=(-0.004, 0.004),  # [m] per episode, raised nodes
            raised_noise_std=0.0025,  # [m] per frame, raised nodes
        ),
    )
    # Cube Postion
    cube_position = ObsTerm(
        func=mdp.object_position_b,
        params={"asset_cfg": SceneEntityCfg("cube")},
        noise=NoiseModelWithAdditiveBiasCfg(
            noise_cfg=GaussianNoiseCfg(std=0.005),  # [m]
            bias_noise_cfg=UniformNoiseCfg(n_min=-0.01, n_max=0.01, operation="abs"),  # [m]
        ),
    )
    cube_orientation = ObsTerm(
        func=mdp.object_quat_b, params={"asset_cfg": SceneEntityCfg("cube")}
    )

    # ---------- Task ----------
    target_pos = ObsTerm(
        func=mdp.generated_commands, params={"command_name": "cloth_target"}
    )

    def __post_init__(self) -> None:
        self.enable_corruption = True
        self.concatenate_terms = True
        self.history_length = 5  # observe history the estimate cloth movement
        self.flatten_history_dim = True


@configclass
class CriticObsCfg(ObsGroup):
    """Privileged critic observations with full cloth pos/vel, detailed grasp information, domain raondomization values and no history"""

    # ---------- Proprioception ----------
    joint_angles = ObsTerm(
        func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])}
    )
    joint_velocities = ObsTerm(
        func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])}
    )
    ee_pose = ObsTerm(func=mdp.ee_pose_b)
    ee_velocity = ObsTerm(
        func=mdp.ee_velocity_b, params={"hand_cfg": SceneEntityCfg("robot", body_names="base_link")}
    )
    gripper_finger_width = ObsTerm(func=mdp.gripper_width)
    last_action = ObsTerm(
        func=mdp.last_action
        )

    # ---------- Exteroception ----------
    cloth_nodal_pos = ObsTerm(func=mdp.cloth_nodal_pos_b)
    cloth_nodal_vel = ObsTerm(func=mdp.cloth_nodal_vel_b)
    grasp_node_pos = ObsTerm(
        func=mdp.grasp_node_pos_b, params={"command_name": "cloth_target"}
        )
    cloth_center_pos = ObsTerm(
        func=mdp.cloth_center_pos_b
    )
    
    cube_pose = ObsTerm(
        func=mdp.object_pose_b, params={"asset_cfg": SceneEntityCfg("cube")}
        )
    cube_velocity = ObsTerm(
        func=mdp.object_velocity_b, params={"asset_cfg": SceneEntityCfg("cube")}
    )
    cube_cloth_center_offset = ObsTerm(
        func=mdp.object_center_offset_b, params={"asset_cfg": SceneEntityCfg("cube")}
    )
    ee_cube_distance = ObsTerm(
        func=mdp.ee_object_distance, params={"asset_cfg": SceneEntityCfg("cube")}
        )

    # ---------- Task ----------
    target_pos = ObsTerm(
        func=mdp.generated_commands, params={"command_name": "cloth_target"}
    )
    cloth_center_to_target = ObsTerm( # planer distance
        func=mdp.cloth_center_to_target_b, params={"command_name": "cloth_target"}
    )
    cube_to_target = ObsTerm( # planer distance
        func=mdp.object_to_target_b, params={"command_name": "cloth_target", "asset_cfg": SceneEntityCfg("cube")}
    )
    stage = ObsTerm(
        func=mdp.cloth_drag_stage, params={"command_name": "cloth_target"}
    )

    # priviledged constrained-gripper grasping information
    cloth_grasped = ObsTerm(func=mdp.cloth_grasped)
    locked_nodes = ObsTerm(func=mdp.locked_node_count)
    nodes_between_pads = ObsTerm(func=mdp.nodes_between_pads)
    grasp_node_capturable = ObsTerm(func=mdp.grasp_node_capturable)
    grasp_tried = ObsTerm(func=mdp.grasp_tried)

    # Values of cube mass, cloth density and bending, cloth-table and cloth-cube friction
    domain_randomization = ObsTerm(func=mdp.domain_randomization_params)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ObservationsCfg:
    """The actor (``policy``) and the privileged critic (``critic``)."""

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
        max_ee_height=0.06,  # success only with low EE (maybe delete this)
        # Success needs a raised node locked during the hold; pushing or dragging with different nodes is not enough.
        debug_vis=True,  # Target marker
        # Task stage transitions
        approach_radius=0.04,  # APPROACH -> GRASP: EE frame closer than [m] to a raised node (any node if none)
        goal_zone_margin=0.06,  # DRAG -> GOAL: center node closer than success_radius + this (0.10 m) to the target
        # Success per start stage (Metrics/cloth_target/task_success_<stage>_*), also used by the adaptive resets
        success_window_steps=480,  # averaging window (24 x 20 iterations), longer than an episode (300 steps)
    )


# ------------------- Event Manager -------------------
FRONTIER_SAMPLER_CFG = reset_strategies.FrontierSamplerCfg(
    # Weighted Moving Average Success Estimate
    window_size=4,
    weight_increment=1.0,
    
    # Beta Sampling Variables
    target_success_estimate=0.5,
    kappa=1.0,  # small kappa = uniform, large = focus near target_success
    temperature=2.0,
    epsilon=1.0e-4,  # ensure 0 and 1 probability can also be drawn
    min_history=2,
    coverage_mix_max=0.25,
    coverage_mix_min=0.02,
)


@configclass
class DomainRandomizationCfg:
    """Add randomization to train a more robust policy"""

    # ---------- Gripper ----------
    robotiq_linkage = EventTerm(
        func=stiffen_robotiq_mimics, 
        mode="startup"
    )

    # ---------- Robot ----------
    arm_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    arm_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
            "friction_distribution_params": (0.0, 0.5),  # joint friction [Nm]
            "operation": "abs",
        },
    )
    arm_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
            "armature_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    # ---------- Cloth ----------
    cloth_material = EventTerm(
        func=mdp.randomize_cloth_material,
        mode="reset",
        params={
            "density_range": (0.13, 0.27),
            "tri_ka_range": (0.5, 2.0),
            "tri_kd_range": (0.5, 2.0),
            "edge_ke_range": (0.25, 2.0),
        },
    )
    cloth_friction = EventTerm(
        func=mdp.randomize_cloth_friction,
        mode="reset",
        params={
            "shape_patterns": {
                "table": r"/World/envs/env_[^/]+/Table(/.*)?",
                "cube": r"/World/envs/env_[^/]+/Cube",
            },
            "mu_scale_range": (0.71, 2.14),
        },
    )

    # ---------- Cube ----------
    cube_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )


@configclass
class DefaultResetsCfg(DomainRandomizationCfg):
    """Fresh randomized resets that start at the beginning of the task."""

    # Robot Reset
    reset_robot_joints = EventTerm(
        func=mdp.reset_franka_top_down,
        mode="reset",
        params={
            "joint_offset_ranges": {  # ensure head facing down
                "panda_joint1": (-0.25, 0.25),
                "panda_joint2": (-0.15, 0.15),
                "panda_joint4": (-0.15, 0.15),
                "panda_joint7": (-0.6, 0.6),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Cloth Reset
    reset_deformable = EventTerm(
        func=mdp.ResetClothWithRaisedNodes,
        mode="reset",
        params={
            # Cloth position and orientation
            "position_range": {"x": (-0.03, 0.03), "y": (-0.05, 0.05)},
            "yaw_range": (-0.25 * math.pi, 0.25 * math.pi),  # symmetrie covers all angles
            # One raised node between the cube and the target, where the real lower-and-squeeze grasp happens
            "include_edge_nodes": False,
            "node_raise_height": 0.01,  # height above the table [m]
            "grasp_distance_range": (0.08, 0.10),  # [m] from the cube towards the target, per episode
            "grasp_lateral_range": (-0.01, 0.01),  # [m] sideways from that line, per episode
            "asset_cfg": SceneEntityCfg("deformable"),
        },
    )

    # Cube Reset (offset from central cloth node)
    reset_cube = EventTerm(
        func=mdp.reset_object_on_cloth_center,
        mode="reset",
        params={
            "position_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (0.004, 0.004)},
            "yaw_range": (-0.25 * math.pi, 0.25 * math.pi),
            "object_cfg": SceneEntityCfg("cube"),
            "cloth_cfg": SceneEntityCfg("deformable"),
        },
    )


@configclass
class UniformResetsCfg(DomainRandomizationCfg):
    """Bank resets using fixed configured ratios between start stages."""

    reset_from_bank = reset_strategies.ResetFromBankCfg(
        bank_path="logs/reset_banks/bank_v13_one_raised_node.pt",  # approach 8000, grasp 1000, drag 100, goal 100
        # one raised node between cube and target; later settings changes that leave the stored states valid pass
        allow_settings_mismatch=True,
        start_stage_ratios={"approach": 0.30, "grasp": 0.45, "drag": 0.15, "goal": 0.10},
        use_frontier_sampling=False,
        frontier_sampling_parameters=FRONTIER_SAMPLER_CFG,
    )


@configclass
class AdaptiveResetsCfg(DomainRandomizationCfg):
    """Bank resets whose start-stage ratios adapt to training success."""

    reset_from_bank = reset_strategies.ResetFromBankCfg(
        bank_path="logs/reset_banks/bank_v13_one_raised_node.pt",  # approach 8000, grasp 1000, drag 100, goal 100
        # one raised node between cube and target; later settings changes that leave the stored states valid pass
        allow_settings_mismatch=True,
        start_stage_ratios={"approach": 0.35, "grasp": 0.45, "drag": 0.10, "goal": 0.10},
        min_stage_ratios={"grasp": 0.10, "drag": 0.01, "goal": 0.01},
        # Ratio Changing Variables
        ratio_shift_success_range=(0.775, 0.95),
        ratio_max_shift_per_step=0.0016,
        ratio_hold_steps=300,
        # Frontier Sampling
        use_frontier_sampling=False,
        frontier_sampling_parameters=FRONTIER_SAMPLER_CFG,
        frontier_stages=("grasp", "drag", "goal"),
    )


@configclass
class ResetStrategiesCfg(PresetCfg):
    """Reset-strategy presets: default, uniform or adaptive"""

    default_resets: DefaultResetsCfg = DefaultResetsCfg()
    uniform_resets: UniformResetsCfg = UniformResetsCfg()
    adaptive_resets: AdaptiveResetsCfg = AdaptiveResetsCfg()

    default = default_resets


@configclass
class RewardsCfg:
    """Heavily stage dependent rewards shaping with a discrete large success bonus"""

    # -------------- Approach --------------
    # Linear reward for approaching towards point above the raised node
    approach_progress = RewTerm(
        func=mdp.ClothDistanceProgress,
        params={"stages": ["approach", "grasp"], "distance_scale": 0.1, "target_height": 0.05},
        weight=1.0,
    )
    # Penalty per second for closing the gripper if the target node is not between the pads. Kept low: at -1 every
    # exploratory close cost more than a rarely completed on-node close paid, so the policy learned never to close
    pen_close_off_node = RewTerm(
        func=mdp.gripper_closed_off_grasp_node, params={"stages": ["approach", "grasp"]}, weight=-0.25
    )

    # -------------- Grasp --------------
    # Tanh reward for approaching heightened node at the same height as in the state machine
    approach_precision = RewTerm(
        func=mdp.ClothDistanceProgress,
        params={"stages": ["approach", "grasp"], "kernel_std": 0.02, "target_height": 0.0195},
        weight=2.0,
    )
    # Signed progress on closing the gripper with target node between the pads, a full close of 6+ nodes pays the weight
    grasp_close_on_node = RewTerm(
        func=mdp.GripperCloseProgress, params={"stages": ["grasp"], "num_nodes_full": 6}, weight=3.0
    )

    # -------------- Drag --------------
    # Linear reward on moving the cube closer to the target in planar distance
    drag_progress = RewTerm(
        func=mdp.ClothDistanceProgress, params={"stages": ["drag", "goal"], "distance_scale": 0.1}, weight=3.0
    )
    # Penalty for lifting the EE above 4cm during the drag stage
    pen_ee_height_drag = RewTerm(
        func=mdp.ee_height_above, params={"max_height": 0.05, "stages": ["drag"]}, weight=-15.0
    )
    # Reward per second of holding a raised node locked; at most ~3.5 per episode, below drag + goal progress (~8)
    drag_hold = RewTerm(func=mdp.cloth_held, params={"stages": ["drag", "goal"]}, weight=0.5)
    # Off: the telescoping progress terms already make drop/re-grasp laps pay nothing, and a lump drop cost made the
    # policy avoid the lock with default resets (run 2026-11-13_21-03-01); a drop now costs the lost drag_hold
    # pen_drop = RewTerm(func=mdp.cloth_drag_drop_penalty, params={"stages": ["drag", "goal"]}, weight=-3.0)

    # -------------- Goal --------------
    # Penalty for lifting the EE above 4cm during the goal stage
    pen_ee_height_goal = RewTerm(
        func=mdp.ee_height_above, params={"max_height": 0.05, "stages": ["goal"]}, weight=-15.0
    )
    # Tanh reward for placing the cube precisely at the target
    goal_precision = RewTerm(
        func=mdp.ClothDistanceProgress, params={"stages": ["drag", "goal"], "kernel_std": 0.02}, weight=2.0
    )
    # Peanilty on cube relative velocity to cloth above 0.01 m/s
    # pen_cube_slip = RewTerm(
    #     func=mdp.object_slip_speed,
    #     params={"threshold": 0.01, "stages": ["drag", "goal"], "asset_cfg": SceneEntityCfg("cube")},
    #     weight=-10.0,
    # )

    # -------------- Stage Potential --------------
    # Telescoping reward on the stage index: entering the next stage pays, falling back takes it back
    stage_progress = RewTerm(func=mdp.StageProgress, weight=10.0)

    # -------------- Task Rewards / Penalties --------------
    # Onetime reward after task success
    success = RewTerm(func=mdp.cloth_drag_success_bonus, weight=40.0)
    # Penalty for failing episode
    pen_termination = RewTerm(
        func=mdp.cloth_drag_termination_penalty,
        params={
            "term_keys": [
                "cloth_out_of_bounds",
                "ee_below_table",
                "cube_fell",
                "gripper_cube_contact",
                "nonfinite_state",
                "joint_vel_out_of_limit",
            ]
        },
        weight=-75.0,
    )

    # -------------- Action Penalties --------------
    # Smooth action magnitude to avoid action clipping
    pen_action_magnitude = RewTerm(func=mdp.action_term_l2, params={"action_name": "arm_action"}, weight=-5e-1)
    # Penalty on action rate to avoid jerky motion
    pen_action_rate = RewTerm(func=mdp.action_term_rate_l2, params={"action_name": "arm_action"}, weight=-1e-1)
    # Penalty for exceeding joint limits on position and veloctiy
    pen_joint_limit_barrier = RewTerm(
        func=mdp.joint_limit_barrier,
        params={
            "action_name": "arm_action",
            "margin": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
        },
        weight=-10.0,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # -------- Task Success --------
    success = DoneTerm(func=mdp.cloth_drag_success)

    # -------- Task Failure --------
    # Episode timeout
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Grasp bank starts end as a truncation (value bootstrapped) once they lock a raised node (reach DRAG)
    bank_start_advanced = DoneTerm(
        func=mdp.bank_start_advanced, params={"start_stages": ("grasp",), "steps_after_entry": 0}, time_out=True
    )

    # Cloth fallen of table
    cloth_out_of_bounds = DoneTerm(
        func=mdp.deformable_outside_bounds,
        params={
            "x_bounds": (-0.15, 1.25),  # [m] table edges
            "y_bounds": (-0.40, 0.40),  # [m] table edges
            "z_bounds": (-0.02, 1.0),
            "asset_cfg": SceneEntityCfg("deformable"),
        },
    )

    # If robot reaches to far below the table (relevant with joint delta actions)
    ee_below_table = DoneTerm(
        func=mdp.ee_below_minimum,
        params={"minimum_height": -0.1, "ee_frame_cfg": SceneEntityCfg("ee_frame")},
    )

    # Cube left the cloth (fallen through the table)
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


##
# Environment configuration
##


@configclass
class FrankaClothDragEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based RL environment"""

    # Scene settings
    scene: FrankaClothDragSceneCfg = FrankaClothDragSceneCfg(num_envs=2048, env_spacing=2.2, replicate_physics=True)

    # MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: ResetStrategiesCfg = ResetStrategiesCfg()

    def __post_init__(self) -> None:
        # Simulation settings
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 1.0 / 120
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysicsCfg()

    def play_mode(self):
        """Settings for playing the policy"""
        super().play_mode()

        # Visualization settings
        self.sim.visualizer_cfgs = [
            ViserVisualizerCfg(eye=(0.5 + 3.18, -3.18, 7.59), lookat=(0.5, 0.0, -0.2), focal_length=23.5, port=8755)
        ]

        # Reset Strategie settings
        self.events = DefaultResetsCfg()

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka cloth dragging environment."""

from __future__ import annotations

import math

from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonSoftContactCfg,
    VBDSolverCfg,
)
from isaaclab_newton.sim.schemas import MujocoRigidBodyCfg, NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.visualizers import VisualizerCfg

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
        iterations=1, # Solve coupling once per step, more iterations are more accurate but slower
    ),
    soft_contact_cfg=NewtonSoftContactCfg(
        soft_contact_ke=8.0e3,  # stiffness, higher less penetration but more unstable
        soft_contact_kd=1.0e-2, # contact damping, higher more stable but slower response
        soft_contact_mu=2.0,  # Increase friction to prevent cloth slipping out of the gripper
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
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.102), rot=(0.70710678, 0.0, 0.0, 0.70710678)),
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
            ik_method="dls", # Damped Least Squares to stay stable near singularities
            ik_params={"lambda_val": 0.3}, # damping, larger lambda_val => more stable but slower response
        ),
        body_offset=mdp.TopDownDeltaEEActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.1034]),
    )

    gripper_action = GRIPPER_ACTION_CFG


@configclass
class ActionsCfg(PresetCfg):
    """Action-space presets: delta joint space or delta EE task space."""

    # Use env.actions=joint_delta Flag
    joint_delta: JointDeltaActionsCfg = JointDeltaActionsCfg()

    # Use env.actions=ee_delta Flag
    ee_delta: EEDeltaActionsCfg = EEDeltaActionsCfg()

    default = joint_delta


# ------------------- Observation Terms -------------------
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

# ------------------- Command Terms -------------------
@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

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


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


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
    scene: FrankaClothDragSceneCfg = FrankaClothDragSceneCfg(num_envs=2048, env_spacing=2.0, replicate_physics=True)

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

        # visualizer settings
        self.viewer.eye = (1.2, -0.9, 0.9)
        self.viewer.lookat = (0.4, 0.0, 0.15)
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=self.viewer.eye, lookat=self.viewer.lookat)

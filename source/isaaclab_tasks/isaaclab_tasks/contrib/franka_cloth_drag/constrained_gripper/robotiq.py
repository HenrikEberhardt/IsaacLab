# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Ported from srl-ethz/IsaacLab-Cloth (branch cloth-fold-environments, commit c445370),
# src/Cloth_Manipulation/assets/robotiq.py, which follows mmichelis/IsaacLab-Soft (branch liftsoft-robotiq,
# commit 1ee1ab3); see assets/robotiq/README.md. The camera helpers were left out.

"""Franka Panda with a Robotiq 2F-85: the asset, its constants, the pad kinematics and the linkage stiffening."""

from __future__ import annotations

import functools
import itertools
import math
from pathlib import Path

import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from isaaclab.actuators import ImplicitActuatorCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

from .robotiq_actuator import RobotiqGripperActuatorCfg

FRANKA_PANDA_ROBOTIQ_USD = str(
    Path(__file__).resolve().parent.parent / "assets" / "robotiq" / "franka_panda_robotiq_simplified.usda"
)
"""Panda + Robotiq + coupling + visual-only camera assembly with simplified colliders.

The override layer keeps the pads exact, retains a grown oriented box for each inner knuckle, and replaces the other
gripper mechanism colliders by one enclosing box per rigid unit (two for the inner-finger backing), grown by 2 mm.
The 11 boxes enclose the original mechanism and coupling colliders; the camera and arm colliders are disabled. Every
Newton shape of an environment is a cloth contact candidate, so each collider costs throughput even where it cannot
touch the cloth; see ``assets/robotiq/README.md``.
"""

FRANKA_PANDA_ROBOTIQ_CFG = FRANKA_PANDA_CFG.replace(
    spawn=FRANKA_PANDA_CFG.spawn.replace(usd_path=FRANKA_PANDA_ROBOTIQ_USD, variants=None),
    init_state=FRANKA_PANDA_CFG.init_state.replace(
        joint_pos={
            key: value
            for key, value in FRANKA_PANDA_CFG.init_state.joint_pos.items()
            if not key.startswith("panda_finger_joint")
        },
    ),
    actuators={},
)
"""Panda arm plus analytical Robotiq, without actuators and finger joint positions; the task assigns the actuators."""

ROBOTIQ_BASE_PATH = (
    "Geometry/panda_link0/panda_link1/panda_link2/panda_link3/panda_link4"
    "/panda_link5/panda_link6/panda_link7/robotiq_mount/Robotiq/base_link"
)
"""Prim path of ``base_link`` below the robot's root prim."""

ROBOTIQ_DRIVE_JOINT = "finger_joint"
"""The single driven joint; mimic constraints make the other five gripper joints follow it, keeping the pads
parallel."""

ROBOTIQ_OPEN = 0.0
"""``finger_joint`` angle [rad] of the fully open gripper (about 87 mm between the pad faces)."""

ROBOTIQ_CLOSED = 0.8
"""``finger_joint`` angle [rad] of the closed gripper, just short of the 0.82 rad limit (about 2 mm between the pad
faces)."""

ROBOTIQ_PAD_WIDTH = 0.022
"""Width [m] of the physical finger pad, along the EE frame's x-axis."""

ROBOTIQ_PAD_LENGTH = 0.038
"""Length [m] of the physical finger pad, along the EE frame's z-axis."""

ROBOTIQ_GAP_CURVE = (0.0127769, 0.1143161, -0.7077544)
"""``(a, b, c)`` of the gap between the pads' inner faces, ``a - b * sin(q + c)`` [m], over ``finger_joint`` ``q``
[rad]: the parallel linkage moves the pads on a circle. Fitted to the pad hulls under the asset's forward kinematics
(0-0.82 rad) to within 0.001 mm."""


def pad_gap_from_joint(q: torch.Tensor, curve: tuple[float, float, float]) -> torch.Tensor:
    """Distance [m] between the pad faces at finger joint angle ``q`` [rad], ``a - b * sin(q + c)``."""
    a, b, c = curve
    return a - b * torch.sin(q + c)


def joint_from_pad_gap(gap: torch.Tensor, curve: tuple[float, float, float]) -> torch.Tensor:
    """Finger joint angle [rad] at which the pad faces are ``gap`` [m] apart (inverse of the gap curve)."""
    a, b, c = curve
    return torch.asin(((a - gap) / b).clamp(-1.0, 1.0)) - c


def pad_shift_from_joint(
    q: torch.Tensor, curve: tuple[float, float, float], closed_q: float, direction: float
) -> torch.Tensor:
    """Position [m] of the pads along the EE frame's z-axis at angle ``q`` [rad], zero when closed at ``closed_q``."""
    _, b, c = curve
    return direction * 0.5 * b * (torch.cos(q + c) - torch.cos(torch.as_tensor(closed_q + c, device=q.device)))


ROBOTIQ_MAX_SQUEEZE_FORCE = 25.0
"""Maximum pad squeeze force [N]: 25 N open, 19 N near closed (2F-85 range 20-235 N)."""

ROBOTIQ_DRIVE_EFFORT = ROBOTIQ_MAX_SQUEEZE_FORCE * ROBOTIQ_GAP_CURVE[1] * math.cos(ROBOTIQ_GAP_CURVE[2])
"""Drive effort limit [Nm]: the squeeze force cap at the open gripper's pad gap rate, the smallest, by virtual work."""

ROBOTIQ_SPEED = 0.95
"""Free ``finger_joint`` speed [rad/s] measured on the real 2F-85 at speed register 192 (0.70 at 128, 1.20 at 255)."""

ROBOTIQ_GRIPPER_ACTUATOR_CFG = RobotiqGripperActuatorCfg(
    joint_names_expr=[ROBOTIQ_DRIVE_JOINT],
    # stiffness and damping are weakly determined by free motion; this pair matches the real overshoot from rest
    stiffness=200.0,
    damping=5.0,
    armature=0.02,  # sets the rise time, fitted to real step responses
    # passive damping acts outside the effort limit, so the saturated drive cruises at effort / viscous friction
    # (ROBOTIQ_SPEED), as the real finger does
    viscous_friction=ROBOTIQ_DRIVE_EFFORT / ROBOTIQ_SPEED,
    joint_effort_limit=ROBOTIQ_DRIVE_EFFORT,  # about 25 N per pad
    actuator_velocity_limit=3.0,
    joint_velocity_limit=2.0,
    # command delay of the real gripper: 5 physics steps = 45 ms at the task's 120 Hz
    min_delay=5,
    max_delay=5,
)
"""Delayed position drive of the Robotiq 2F-85 on ``finger_joint``, fitted to the real gripper's free motion (as in
srl-ethz/IsaacLab-Cloth). It takes a step open/close target; the effort limit and viscous friction set the speed."""

ROBOTIQ_PASSIVE_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[".*_inner_finger_joint", ".*_inner_finger_knuckle_joint", "right_outer_knuckle_joint"],
    stiffness=0.0,
    damping=0.0,
    # the drive's reflected inertia on every coupled joint: with only the links' few grams, contact throws the pads
    # about against the mimic constraints
    armature=ROBOTIQ_GRIPPER_ACTUATOR_CFG.armature,
    viscous_friction=0.0,
)
"""The five gripper joints that follow ``finger_joint`` through the mimic constraints, with no drive of their own."""

_CYLINDER_RIM_POINTS = 32
"""Points per rim circle that stand for a collision cylinder in :func:`robotiq_collision_corners`."""


def _shape_points(prim: Usd.Prim) -> list[Gf.Vec3d]:
    """Corners of a collision cube, or rim points of a collision cylinder, in the prim's own frame [m]."""
    if prim.IsA(UsdGeom.Cube):
        half = UsdGeom.Cube(prim).GetSizeAttr().Get() / 2.0
        return [Gf.Vec3d(*(sign * half for sign in signs)) for signs in itertools.product((-1.0, 1.0), repeat=3)]
    if prim.IsA(UsdGeom.Cylinder):
        cylinder = UsdGeom.Cylinder(prim)
        radius, half = cylinder.GetRadiusAttr().Get(), cylinder.GetHeightAttr().Get() / 2.0
        axis = "XYZ".index(cylinder.GetAxisAttr().Get())
        points = []
        for index, sign in itertools.product(range(_CYLINDER_RIM_POINTS), (-1.0, 1.0)):
            angle = 2.0 * math.pi * index / _CYLINDER_RIM_POINTS
            radial = (radius * math.cos(angle), radius * math.sin(angle))
            point = [0.0, 0.0, 0.0]
            point[axis] = sign * half
            point[(axis + 1) % 3], point[(axis + 2) % 3] = radial
            points.append(Gf.Vec3d(*point))
        return points
    raise ValueError(f"Unexpected gripper collision shape {prim.GetTypeName()} at {prim.GetPath()}")


@functools.cache
def robotiq_collision_corners() -> dict[str, tuple[tuple[float, float, float], ...]]:
    """Corners [m] of every gripper collider, in the frame of the body that carries it, keyed by body name.

    These are the asset's enabled gripper colliders (boxes, or a cylinder as points on its two rims), read from the
    local asset, so this works offline. Bodies without colliders are left out.
    """
    stage = Usd.Stage.Open(FRANKA_PANDA_ROBOTIQ_USD)
    root = stage.GetDefaultPrim()
    gripper = root.GetPrimAtPath(ROBOTIQ_BASE_PATH).GetParent()
    cache = UsdGeom.XformCache()
    corners = {}
    for body in gripper.GetChildren():
        if not body.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        to_body = cache.GetLocalToWorldTransform(body).GetInverse()
        points = []
        for prim in Usd.PrimRange(body):
            if prim != body and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError(f"Unexpected nested rigid body {prim.GetPath()}")
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                continue
            transform = cache.GetLocalToWorldTransform(prim) * to_body
            points.extend(tuple(transform.Transform(point)) for point in _shape_points(prim))
        if points:
            corners[body.GetName()] = tuple(points)
    return corners


ROBOTIQ_MIMIC_SOLREF = (0.004, 1.0)
"""MuJoCo ``solref`` (time constant [s], damping ratio) of the mimic constraints: about twice the 2 ms MuJoCo step,
the stiffest MuJoCo allows. MuJoCo's default 20 ms let contact loads bend the parallel-mode linkage by up to 1 rad."""

ROBOTIQ_MIMIC_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2.0)
"""MuJoCo ``solimp`` of the mimic constraints, raised from the default (0.9, 0.95, ...) for the same reason."""


def stiffen_robotiq_mimics(env, env_ids) -> None:
    """Make the gripper linkage effectively rigid by stiffening the MuJoCo equality rows of its mimic constraints.

    Newton turns the asset's mimic joints into MuJoCo joint equalities with MuJoCo's default softness and offers no
    setting for them, so this writes the MuJoCo model directly. Use it as a ``startup`` event: the solver only exists
    once the simulation has started, after ``PhysicsEvent.PHYSICS_READY``.
    """
    solver = NewtonManager._solver
    # the coupler nests the robot's MuJoCo solver; a plain MuJoCo solver is used directly
    solver = solver.solver("rigid") if hasattr(solver, "entry_names") and "rigid" in solver.entry_names() else solver
    rows = getattr(solver, "mjc_eq_to_newton_joint_mimic", None)
    if rows is None:
        return
    mimic = wp.to_torch(rows) >= 0
    solref, solimp = wp.to_torch(solver.mjw_model.eq_solref), wp.to_torch(solver.mjw_model.eq_solimp)
    solref[mimic] = solref.new_tensor(ROBOTIQ_MIMIC_SOLREF)
    solimp[mimic] = solimp.new_tensor(ROBOTIQ_MIMIC_SOLIMP)

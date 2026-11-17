# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Ported from srl-ethz/IsaacLab-Cloth (branch cloth-fold-environments, commit c445370),
# src/Cloth_Manipulation/assets/robotiq.py and common/gripper_actuator.py, which follow mmichelis/IsaacLab-Soft
# (branch liftsoft-robotiq, commit 1ee1ab3); see assets/robotiq/README.md. The camera helpers were left out.

"""Franka Panda with a Robotiq 2F-85: the asset, its gap curve, and the delayed finger drive."""

from __future__ import annotations

import functools
import itertools
import math
from collections.abc import Sequence
from pathlib import Path

import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from isaaclab.actuators import ImplicitActuator, ImplicitActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

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

ROBOTIQ_BASE_BODY = "base_link"
"""The gripper's root body. It is fixed to ``panda_link7`` 125 mm along its z axis with no yaw, so the jaws open along
the flange's y axis. The coupling's and camera's masses are folded into it."""

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
"""``finger_joint`` angle [rad] of the closed gripper, just short of the 0.82 rad limit where the pads meet."""

ROBOTIQ_PAD_BODY_NAMES = ("left_inner_finger", "right_inner_finger")
"""Bodies that carry the gripping pads."""

ROBOTIQ_PAD_CENTERS = ((0.0, -0.04674, 0.12940), (0.0, 0.04674, 0.12940))
"""Centres [m] of the thin pad boxes in the frames of :data:`ROBOTIQ_PAD_BODY_NAMES`.

The pad bodies' frames coincide with ``base_link``'s while the gripper is open, so these are also the open pads'
centres in the ``base_link`` frame. Each pad is 6.5 mm thick, with its gripping face 3.2 mm inward of its centre.
"""

ROBOTIQ_PAD_HALF_WIDTH = 0.011
"""Half width of a gripping pad along the base_link x-axis [m]; the pad is 22 mm wide."""

ROBOTIQ_PAD_THICKNESS = 0.0065
"""Thickness of a gripping pad along the closing direction [m]."""

ROBOTIQ_PAD_HALF_LENGTH = 0.019
"""Half length of a gripping pad along the base_link z-axis [m]; the pad is 38 mm long."""

ROBOTIQ_GAP_CURVE = (0.0127769, 0.1143161, -0.7077544)
"""``(a, b, c)`` of the gap between the pads' inner faces, ``a - b * sin(q + c)`` [m], over ``finger_joint`` ``q``
[rad]: the parallel linkage moves the pads on a circle. Fitted to the pad hulls under the asset's forward kinematics
(0-0.82 rad) to within 0.001 mm."""

ROBOTIQ_MAX_SQUEEZE_FORCE = 25.0
"""Maximum pad squeeze force [N]: 25 N open, 19 N near closed (2F-85 range 20-235 N)."""

ROBOTIQ_DRIVE_EFFORT = ROBOTIQ_MAX_SQUEEZE_FORCE * ROBOTIQ_GAP_CURVE[1] * math.cos(ROBOTIQ_GAP_CURVE[2])
"""Drive effort limit [Nm]: the squeeze force cap at the open gripper's pad gap rate, the smallest, by virtual work."""

ROBOTIQ_SPEED = 0.95
"""Free ``finger_joint`` speed [rad/s] measured on the real 2F-85 at speed register 192 (0.70 at 128, 1.20 at 255)."""

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


class RobotiqGripperActuator(ImplicitActuator):
    """Robotiq 2F-85: the solver's PD drive tracks a delayed, optionally speed-limited copy of the commanded position.

    The policy commands the gripper fully open or fully closed. The real gripper starts moving some time after the
    command and then travels at a set speed. This model reproduces that in front of the implicit PD drive:

    * the commanded target is delayed by ``min_delay`` to ``max_delay`` physics steps, sampled per environment at
      reset. After a reset the delay line is filled with the measured position, so the gripper holds still until the
      first command comes through;
    * if ``stroke_time_range`` is set, the drive target moves from its current value towards the delayed command at a
      constant speed, so that it covers the open-to-closed stroke in ``stroke_time_range`` seconds, sampled per
      environment at reset. Otherwise the drive gets the delayed command directly, and its effort limit and viscous
      friction set the speed.

    The PD gains and effort limit stay the drive's own, so the squeeze force on a grasped object is unchanged: once
    the fingers stop on the object, the drive target still reaches the closed position and the drive saturates at its
    effort limit. The delay and speed act in the simulator only. The exported policy still outputs the open or
    closed position, and on the robot the gripper itself adds the delay and the speed limit.
    """

    cfg: RobotiqGripperActuatorCfg

    def __init__(self, cfg: RobotiqGripperActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if not 0 <= cfg.min_delay <= cfg.max_delay:
            raise ValueError(f"Expected 0 <= min_delay <= max_delay, got {cfg.min_delay}, {cfg.max_delay}")
        if cfg.stroke_time_range is not None:
            if not 0.0 < cfg.stroke_time_range[0] <= cfg.stroke_time_range[1]:
                raise ValueError(
                    f"Expected 0 < stroke_time_range[0] <= stroke_time_range[1], got {cfg.stroke_time_range}"
                )
            if cfg.stroke is None or cfg.physics_dt is None:
                raise ValueError("stroke_time_range needs stroke and physics_dt")
        shape = (self._num_envs, self.num_joints)
        self._history = torch.zeros(cfg.max_delay + 1, *shape, device=self._device)
        self._write_index = 0
        self._lag = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        self._fresh = torch.ones(self._num_envs, dtype=torch.bool, device=self._device)
        self._env_index = torch.arange(self._num_envs, device=self._device)
        # largest drive target change per physics step [rad]; unlimited without a speed limit
        self._max_step = torch.full(shape, torch.inf, device=self._device)
        self._drive_target = torch.zeros(shape, device=self._device)
        self.reset(None)

    def reset(self, env_ids: Sequence[int] | None = None):
        ids = slice(None) if env_ids is None else env_ids
        num = self._lag[ids].shape[0]
        self._lag[ids] = torch.randint(self.cfg.min_delay, self.cfg.max_delay + 1, (num,), device=self._device)
        self._fresh[ids] = True
        if self.cfg.stroke_time_range is None:
            return
        low, high = self.cfg.stroke_time_range
        stroke_time = low + (high - low) * torch.rand(num, self.num_joints, device=self._device)
        self._max_step[ids] = self.cfg.stroke * self.cfg.physics_dt / stroke_time

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # after a reset, the delay line and drive target start from the measured position
        fresh = self._fresh.view(-1, 1)
        self._history = torch.where(fresh.unsqueeze(0), joint_pos.unsqueeze(0), self._history)
        self._drive_target = torch.where(fresh, joint_pos, self._drive_target)
        self._fresh[:] = False

        self._history[self._write_index] = control_action.joint_positions
        read_index = (self._write_index - self._lag) % self._history.shape[0]
        delayed = self._history[read_index, self._env_index]
        self._write_index = (self._write_index + 1) % self._history.shape[0]

        step = (delayed - self._drive_target).clamp(-self._max_step, self._max_step)
        self._drive_target = self._drive_target + step
        control_action.joint_positions = self._drive_target.clone()
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class RobotiqGripperActuatorCfg(ImplicitActuatorCfg):
    """Configuration for :class:`RobotiqGripperActuator`."""

    class_type: type = RobotiqGripperActuator

    physics_dt: float | None = None
    """Physics step [s], at which the actuator is computed; needed with ``stroke_time_range``."""

    stroke: float | None = None
    """Distance [rad] between the open and closed positions of the driven joint; needed with ``stroke_time_range``."""

    stroke_time_range: tuple[float, float] | None = None
    """Time [s] in which the drive target covers the full stroke, sampled per environment at reset; None leaves the
    speed to the drive."""

    min_delay: int = 0
    """Minimum command delay [physics steps]."""

    max_delay: int = 0
    """Maximum command delay [physics steps]."""

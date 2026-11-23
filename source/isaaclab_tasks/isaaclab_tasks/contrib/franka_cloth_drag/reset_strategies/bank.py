# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bank of stored reset states of the Franka cloth dragging task.

A bank holds N reset states as a nested dictionary of tensors whose first dimension is N, with two groups:

* ``scene``: the physical state, positions relative to the environment origin: the robot's joint positions and
  velocities, the cloth's nodal positions [m] and velocities [m/s], and the cube's root pose and velocity.
* ``task``: the task state a reset would clear: raised nodes, target, ``stage`` (the
  :class:`~..mdp.stages.ClothDragStage` the state starts in) and the gripper and grasp state.

``meta`` holds JSON-serializable information on how the bank was made: the schema version, the fingerprint of the task
the states belong to (:func:`task_fingerprint`), ``node_raise_height`` (height of the stored raised nodes above the
rest of the cloth [m]), the task name, the creation time and git commit, the state machine settings it was generated
with (``generator``) and the generation statistics.

The bank is saved with :func:`torch.save` and loaded with ``weights_only=True``, so loading runs no pickled code. A
loaded bank sits on the environment's device as a whole; a reset indexes its rows.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from ..mdp.stages import START_STAGE_NAMES, canonical_start_stage_name

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

SCHEMA_VERSION = 7
"""Version of the bank layout; a bank of another version does not load. Version 5 (2026-11-13): the states of all four
stages, the task-start states included. Version 6 (2026-11-13): no ``labels`` group (the stage is ``task.stage``; the
per-state rollout labels are dropped), the generator settings in ``meta`` identify the bank. Version 7 (2026-11-16):
the Robotiq finger angle in ``task.held_finger_pos`` replaces the Panda finger opening."""

STRICT_FINGERPRINT_KEYS = ("num_nodes", "joint_names")
"""Fingerprint entries that must match exactly: without them the stored states do not fit the task's tensors."""

SOFT_FINGERPRINT_KEYS = ("physics_hash", "task_hash")
"""Fingerprint entries whose mismatch means the states were made with other physics or task settings."""


@dataclass
class ResetBank:
    """Stored reset states, see the module docstring for the layout."""

    data: dict[str, dict]
    """Nested dictionary of tensors with the states as first dimension: ``scene`` and ``task``."""

    meta: dict[str, Any] = field(default_factory=dict)
    """JSON-serializable information on how the bank was made, see the module docstring."""

    def __len__(self) -> int:
        return len(self.stage)

    @property
    def stage(self) -> torch.Tensor:
        """Index of the stage every state starts in, into :data:`~..mdp.stages.START_STAGE_NAMES`, shape [N]."""
        return self.data["task"]["stage"]

    @property
    def device(self) -> torch.device:
        """Device of the tensors."""
        return self.stage.device

    def select(self, rows: torch.Tensor) -> ResetBank:
        """Return the bank of the given rows, with the same meta information."""
        rows = rows.to(self.device)
        return ResetBank(_map_tree(lambda value: value[rows], self.data), dict(self.meta))

    def to(self, device: torch.device | str) -> ResetBank:
        """Return the bank with its tensors on ``device``."""
        return ResetBank(_map_tree(lambda value: value.to(device), self.data), dict(self.meta))

    @classmethod
    def concat(cls, banks: Sequence[ResetBank], meta: dict[str, Any] | None = None) -> ResetBank:
        """Concatenate banks of the same layout; the meta information is ``meta`` or that of the first bank."""
        if not banks:
            raise ValueError("At least one bank is needed.")
        return cls(_cat_trees([bank.data for bank in banks]), dict(banks[0].meta if meta is None else meta))

    def stage_rows(self, stage: str | int) -> torch.Tensor:
        """Rows of the states that start in the given stage, by name (:data:`~..mdp.stages.START_STAGE_NAMES`) or
        index."""
        index = START_STAGE_NAMES.index(canonical_start_stage_name(stage)) if isinstance(stage, str) else int(stage)
        return (self.stage == index).nonzero(as_tuple=False).squeeze(-1)

    def stage_counts(self) -> dict[str, int]:
        """Number of states per stage name."""
        counts = torch.bincount(self.stage.long().cpu(), minlength=len(START_STAGE_NAMES))
        return {name: int(count) for name, count in zip(START_STAGE_NAMES, counts)}

    def save(self, path: str | Path) -> None:
        """Save the bank to ``path`` (``.pt``), replacing an existing file only once the new one is complete."""
        path = Path(path)
        if path.suffix != ".pt":
            raise ValueError(f"Reset banks are saved as '.pt' files, got '{path}'.")
        path.parent.mkdir(parents=True, exist_ok=True)
        save_atomic(path, {"data": _map_tree(lambda value: value.detach().cpu(), self.data), "meta": self.meta})

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str = "cpu") -> ResetBank:
        """Load a bank saved by :meth:`save` onto ``device``.

        Raises:
            ValueError: If the bank has another schema version.
        """
        payload = torch.load(path, map_location=device, weights_only=True)
        bank = cls(payload["data"], payload["meta"])
        version = bank.meta.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"Reset bank '{path}' has schema version {version}, expected {SCHEMA_VERSION}.")
        return bank

    def check_fingerprint(self, fingerprint: dict[str, Any], allow_settings_mismatch: bool = False) -> None:
        """Check that the states fit the task with the given fingerprint (see :func:`task_fingerprint`).

        Args:
            fingerprint: Fingerprint of the task the states are restored into.
            allow_settings_mismatch: Whether states made with other physics or task settings may be used.

        Raises:
            ValueError: If the mesh, the joints or the scene differ (added sensors are allowed), or, unless allowed,
                the settings.
        """
        stored = self.meta.get("fingerprint", {})
        for key in STRICT_FINGERPRINT_KEYS:
            if stored.get(key) != fingerprint.get(key):
                raise ValueError(f"The reset bank does not fit the task: '{key}' differs. Generate a new bank.")
        # the task may add sensors (e.g. a camera), which hold no stored state
        stored_entities, entities = set(stored.get("scene_entities", [])), set(fingerprint.get("scene_entities", []))
        extra = entities - stored_entities - set(fingerprint.get("scene_sensors", []))
        if stored_entities - entities or extra:
            raise ValueError("The reset bank does not fit the task: 'scene_entities' differs. Generate a new bank.")
        stored_mesh = torch.tensor(stored.get("mesh_default_local", []))
        mesh = torch.tensor(fingerprint.get("mesh_default_local", []))
        if stored_mesh.shape != mesh.shape or not torch.allclose(stored_mesh, mesh, atol=1.0e-5):
            raise ValueError("The reset bank does not fit the task: the cloth mesh differs. Generate a new bank.")
        changed = [key for key in SOFT_FINGERPRINT_KEYS if stored.get(key) != fingerprint.get(key)]
        if changed and not allow_settings_mismatch:
            raise ValueError(
                f"The reset bank was made with other settings ({', '.join(changed)} differ). Generate a new bank, or"
                " allow the mismatch explicitly."
            )


def task_fingerprint(
    env: ManagerBasedRLEnv,
    robot_name: str = "robot",
    cloth_name: str = "deformable",
    cube_name: str | None = "cube",
    command_name: str = "cloth_target",
    gripper_action_name: str = "gripper_action",
) -> dict[str, Any]:
    """Return what the stored states depend on, to check a bank against the task it is loaded into.

    Strict entries: the number of cloth nodes, the robot's joint names, the scene's entity names (of which
    ``scene_sensors`` may be missing from the bank) and the cloth's default nodes relative to the environment origin. Soft entries: hashes of the physics settings (solver, time step,
    decimation, robot, cloth, cube) and of the task settings (gripper action, command). The default reset events are not
    part of it: a strategy that resets from the bank has none.

    Args:
        env: The environment.
        robot_name: Scene name of the robot.
        cloth_name: Scene name of the cloth.
        cube_name: Scene name of the cube standing on the cloth, or None without one.
        command_name: Name of the cloth target command.
        gripper_action_name: Name of the cloth grasp action term.

    Returns:
        The JSON-serializable fingerprint.
    """
    cfg = env.cfg
    robot, cloth = env.scene[robot_name], env.scene[cloth_name]
    mesh = cloth.data.default_nodal_state_w.torch[0, :, :3] - env.scene.env_origins[0]
    physics = {
        "sim_dt": cfg.sim.dt,
        "decimation": cfg.decimation,
        "physics": _without_resolved_bodies(cfg.sim.physics.to_dict()),
        "robot_spawn": getattr(cfg.scene, robot_name).spawn,
        "robot_actuators": getattr(cfg.scene, robot_name).actuators,
        "cloth_spawn": getattr(cfg.scene, cloth_name).spawn,
        "cube_spawn": getattr(cfg.scene, cube_name).spawn if cube_name is not None else None,
    }
    command_cfg = env.command_manager.get_term(command_name).cfg.to_dict()
    # settings the stored states do not depend on: visualization and logging
    for key in ("debug_vis", "goal_visualizer_cfg", "resampling_time_range", "success_window_steps"):
        command_cfg.pop(key, None)
    task = {
        "gripper_action": env.action_manager.get_term(gripper_action_name).cfg,
        "command": command_cfg,
    }
    return {
        "num_nodes": int(mesh.shape[0]),
        "joint_names": list(robot.joint_names),
        "scene_entities": sorted(env.scene.keys()),
        "scene_sensors": sorted(env.scene.sensors.keys()),
        "mesh_default_local": [[round(float(x), 6) for x in node] for node in mesh.tolist()],
        "physics_hash": stable_hash(physics),
        "task_hash": stable_hash(task),
    }


def stable_hash(obj: Any) -> str:
    """SHA-256 of a JSON rendering of ``obj`` that is the same in every process.

    Config classes are rendered through ``to_dict``, functions and classes by their qualified names, sequences as
    lists, everything else by ``repr``.
    """

    def _jsonable(value: Any) -> Any:
        if isinstance(value, type):
            return f"{value.__module__}.{value.__qualname__}"
        if hasattr(value, "to_dict"):
            return value.to_dict()
        if callable(value):
            return f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', type(value).__name__)}"
        if isinstance(value, (tuple, set, frozenset)):
            return list(value)
        if isinstance(value, torch.Tensor):
            return value.tolist()
        return repr(value)

    text = json.dumps(obj, sort_keys=True, default=_jsonable)
    return hashlib.sha256(text.encode()).hexdigest()


def save_atomic(path: Path, payload: Any) -> None:
    """Save ``payload`` with :func:`torch.save`, replacing an existing ``path`` only once the new file is complete."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _without_resolved_bodies(tree: Any) -> Any:
    """Drop the ``bodies`` entries that hold body indices instead of patterns from a physics config dictionary.

    The coupler replaces the body patterns of its proxies with the model body indices of every environment at runtime,
    so they depend on the number of environments, not on the physics.
    """
    if isinstance(tree, dict):
        return {
            key: _without_resolved_bodies(value)
            for key, value in tree.items()
            if not (key == "bodies" and isinstance(value, list) and all(isinstance(v, int) for v in value))
        }
    if isinstance(tree, list):
        return [_without_resolved_bodies(value) for value in tree]
    return tree


def _map_tree(fn: Callable[[torch.Tensor], torch.Tensor], tree: dict) -> dict:
    """Apply ``fn`` to every tensor of a nested dictionary and return the new dictionary."""
    return {key: _map_tree(fn, value) if isinstance(value, dict) else fn(value) for key, value in tree.items()}


def _cat_trees(trees: Sequence[dict]) -> dict:
    """Concatenate nested dictionaries of the same layout along the first dimension."""
    first = trees[0]
    return {
        key: _cat_trees([tree[key] for tree in trees])
        if isinstance(value, dict)
        else torch.cat([tree[key] for tree in trees])
        for key, value in first.items()
    }

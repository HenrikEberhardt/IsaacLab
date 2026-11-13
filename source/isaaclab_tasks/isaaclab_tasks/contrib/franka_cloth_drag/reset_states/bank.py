# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bank of stored reset states of the Franka cloth dragging task.

A bank holds N reset states as a nested dictionary of tensors whose first dimension is N, with three groups:

* ``scene``: the physical state, positions relative to the environment origin, see :mod:`.capture`.
* ``task``: the task state a reset would clear: raised nodes, target, stage and the gripper and grasp state.
* ``labels``: where the state comes from: ``bucket`` (index into :data:`BUCKET_NAMES`, the capture window), ``progress``
  (share of the goal distance covered), ``episode_time`` [s] and ``sm_params`` (grasp choice, drag speed [m/s], drag
  height [m], grasp yaw offset [rad], pre-grasp offset x, y, z [m], pre-grasp yaw error [rad]).

``meta`` holds JSON-serializable information: the schema version, the fingerprint of the task the states belong to,
the generator settings and statistics.

The bank is saved with :func:`torch.save` and loaded with ``weights_only=True``, so loading runs no pickled code.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

SCHEMA_VERSION = 4
"""Version of the bank layout; a bank of another version does not load. Version 2 (2026-10-27): the buckets hold
pre-grasp, early drag and goal-zone-edge states (see :meth:`~.collect.ResetCollector._windows`). Version 3
(2026-10-28): every state is stored in the task stage of its bucket. Version 4 (2026-11-04): the bank holds only
training reset states and drops the rollout ladder fields and approach rows."""

BUCKET_NAMES = ("approach", "grasp", "drag", "goal")
"""Names of the buckets, in :class:`~..mdp.stages.ClothDragStage` order; each state is in the stage of its bucket:
the episode start, a pre-grasp pose with an open gripper, a fresh grasp before the drag, and the goal point inside the
goal zone (:meth:`~.collect.ResetCollector._windows`)."""

RESET_MODES = ("default", "grasp", "drag", "goal")
"""Reset modes of training: the task's default reset events, or a random stored state of the named bucket."""

STRICT_FINGERPRINT_KEYS = ("schema_version", "num_nodes", "joint_names", "scene_entities")
"""Fingerprint entries that must match exactly: without them the stored states do not fit the task's tensors."""

SOFT_FINGERPRINT_KEYS = ("physics_hash", "task_hash")
"""Fingerprint entries whose mismatch means the states were made with other physics or task settings."""


def map_tree(fn: Callable[[torch.Tensor], torch.Tensor], tree: dict) -> dict:
    """Apply ``fn`` to every tensor of a nested dictionary and return the new dictionary."""
    return {key: map_tree(fn, value) if isinstance(value, dict) else fn(value) for key, value in tree.items()}


def cat_trees(trees: Sequence[dict]) -> dict:
    """Concatenate nested dictionaries of the same layout along the first dimension."""
    first = trees[0]
    return {
        key: cat_trees([tree[key] for tree in trees])
        if isinstance(value, dict)
        else torch.cat([tree[key] for tree in trees])
        for key, value in first.items()
    }


def assign_tree(dst: dict, ids: torch.Tensor, src: dict) -> None:
    """Write ``src`` into the rows ``ids`` of ``dst``, two nested dictionaries of the same layout."""
    for key, value in src.items():
        if isinstance(value, dict):
            assign_tree(dst[key], ids, value)
        else:
            dst[key][ids] = value


def empty_like_tree(tree: dict, num_rows: int) -> dict:
    """Zero tensors with ``num_rows`` rows and the trailing shapes and types of a nested dictionary of tensors."""
    return map_tree(lambda value: value.new_zeros((num_rows, *value.shape[1:])), tree)


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


@dataclass
class ResetBank:
    """Stored reset states, see the module docstring for the layout."""

    data: dict[str, dict]
    """Nested dictionary of tensors with the states as first dimension: ``scene``, ``task`` and ``labels``."""

    meta: dict[str, Any] = field(default_factory=dict)
    """JSON-serializable information: ``schema_version``, ``fingerprint``, generator settings and statistics."""

    def __len__(self) -> int:
        return len(self.data["labels"]["bucket"])

    @property
    def device(self) -> torch.device:
        """Device of the tensors."""
        return self.data["labels"]["bucket"].device

    def select(self, rows: torch.Tensor) -> ResetBank:
        """Return the bank of the given rows, with the same meta information."""
        rows = rows.to(self.device)
        return ResetBank(map_tree(lambda value: value[rows], self.data), dict(self.meta))

    def to(self, device: torch.device | str) -> ResetBank:
        """Return the bank with its tensors on ``device``."""
        return ResetBank(map_tree(lambda value: value.to(device), self.data), dict(self.meta))

    @classmethod
    def concat(cls, banks: Sequence[ResetBank], meta: dict[str, Any] | None = None) -> ResetBank:
        """Concatenate banks of the same layout; the meta information is ``meta`` or that of the first bank."""
        if not banks:
            raise ValueError("At least one bank is needed.")
        return cls(cat_trees([bank.data for bank in banks]), dict(banks[0].meta if meta is None else meta))

    def bucket_rows(self, bucket: str | int) -> torch.Tensor:
        """Rows of the given bucket, by name (:data:`BUCKET_NAMES`) or index."""
        index = BUCKET_NAMES.index(bucket) if isinstance(bucket, str) else int(bucket)
        return (self.data["labels"]["bucket"] == index).nonzero(as_tuple=False).squeeze(-1)

    def bucket_counts(self) -> dict[str, int]:
        """Number of states per bucket name."""
        counts = torch.bincount(self.data["labels"]["bucket"].long().cpu(), minlength=len(BUCKET_NAMES))
        return {name: int(count) for name, count in zip(BUCKET_NAMES, counts)}

    def save(self, path: str | Path) -> None:
        """Save the bank to ``path`` (``.pt``), replacing an existing file only once the new one is complete."""
        path = Path(path)
        if path.suffix != ".pt":
            raise ValueError(f"Reset banks are saved as '.pt' files, got '{path}'.")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"data": map_tree(lambda value: value.detach().cpu(), self.data), "meta": self.meta}
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
        """Check that the states fit the task with the given fingerprint (see :func:`~.capture.task_fingerprint`).

        Args:
            fingerprint: Fingerprint of the task the states are restored into.
            allow_settings_mismatch: Whether states made with other physics or task settings may be used.

        Raises:
            ValueError: If the mesh, the joints or the scene differ, or, unless allowed, the settings.
        """
        stored = self.meta.get("fingerprint", {})
        for key in STRICT_FINGERPRINT_KEYS:
            if stored.get(key) != fingerprint.get(key):
                raise ValueError(f"The reset bank does not fit the task: '{key}' differs. Generate a new bank.")
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


def check_mode_probabilities(mode_probabilities: dict[str, float], bank: ResetBank | None) -> torch.Tensor:
    """Check the shares of the reset modes and return them in the order of :data:`RESET_MODES`, normalized.

    Args:
        mode_probabilities: Share of the resets per mode name (:data:`RESET_MODES`); missing modes get none.
        bank: The bank the stored modes draw from, or None without one.

    Raises:
        ValueError: On an unknown mode, negative or all-zero shares, or a stored mode without a bank or states.
    """
    unknown = set(mode_probabilities) - set(RESET_MODES)
    if unknown:
        raise ValueError(f"Unknown reset modes {sorted(unknown)}, expected names from {RESET_MODES}.")
    shares = torch.tensor([float(mode_probabilities.get(mode, 0.0)) for mode in RESET_MODES])
    if not torch.isfinite(shares).all() or (shares < 0.0).any() or shares.sum() <= 0.0:
        raise ValueError(f"The reset mode shares must be finite, non-negative and not all zero: {mode_probabilities}.")
    for mode, share in zip(RESET_MODES[1:], shares[1:]):
        if share > 0.0 and (bank is None or len(bank.bucket_rows(mode)) == 0):
            raise ValueError(f"Reset mode '{mode}' has a share of {share.item()} but no stored states.")
    return shares / shares.sum()


def check_min_probabilities(
    min_probabilities: dict[str, float],
    start_shares: torch.Tensor,
    success_range: Sequence[float],
) -> torch.Tensor:
    """Check the settings of the adaptive reset shares and return the minimum shares in the order of
    :data:`RESET_MODES`.

    Args:
        min_probabilities: Minimum share of the resets per mode name; missing modes get none.
        start_shares: Start shares of the modes, from :func:`check_mode_probabilities`, shape [M].
        success_range: Success (low, high) between which a bucket gives up its share.

    Raises:
        ValueError: On an unknown mode, a minimum outside 0 and its start share, or a ``success_range`` that is not
            increasing inside [0, 1].
    """
    unknown = set(min_probabilities) - set(RESET_MODES)
    if unknown:
        raise ValueError(f"Unknown reset modes {sorted(unknown)}, expected names from {RESET_MODES}.")
    minimum = torch.tensor([float(min_probabilities.get(mode, 0.0)) for mode in RESET_MODES])
    if (minimum < 0.0).any() or (minimum > start_shares + 1e-6).any():
        raise ValueError(f"Every minimum must lie between 0 and its start share {start_shares.tolist()}: {minimum}.")
    low, high = success_range
    if not 0.0 <= low < high <= 1.0:
        raise ValueError(f"success_range must increase inside [0, 1], got {success_range}.")
    return minimum


def cascade_shares(
    start: torch.Tensor, minimum: torch.Tensor, success: torch.Tensor, success_range: Sequence[float]
) -> torch.Tensor:
    """Move the reset shares of solved buckets to the stage before them.

    Resets of an earlier stage run through the later ones, so a bucket gives up its share to the mode before it in
    :data:`RESET_MODES`: goal to drag, drag to grasp, grasp to default. A bucket keeps all it holds, its start share and
    what it received, up to the low end of ``success_range``, only its minimum from the high end, and linearly between.
    The default reset gives up nothing, so it never falls below its start share.

    Args:
        start: Start shares in the order of :data:`RESET_MODES`, shape [M].
        minimum: Minimum shares in the same order, shape [M].
        success: Success of each mode in [0, 1] in the same order, shape [M]. The default reset's is not used.
        success_range: Success (low, high) between which a bucket gives up its share.

    Returns:
        The shares in the same order, shape [M].
    """
    low, high = success_range
    given_up = ((success - low) / (high - low)).clamp(0.0, 1.0)
    shares = start.clone()
    for mode_id in range(len(shares) - 1, 0, -1):
        released = (shares[mode_id] - minimum[mode_id]) * given_up[mode_id]
        shares[mode_id] -= released
        shares[mode_id - 1] += released
    return shares


def sample_reset_rows(
    bank: ResetBank, shares: torch.Tensor, num: int, row_weights: torch.Tensor | None = None
) -> torch.Tensor:
    """Draw the reset of ``num`` environments: a mode by its share, then a state of that mode's bucket.

    Args:
        bank: The bank to draw from.
        shares: Shares of the modes in the order of :data:`RESET_MODES`, from :func:`check_mode_probabilities`.
        num: Number of environments.
        row_weights: Optional non-negative weight of every bank row, shape [N]. Weights are normalized independently
            inside each selected bucket. Without them, rows are drawn uniformly.

    Returns:
        Bank rows, -1 for the default reset, shape [num], on the bank's device.
    """
    device = bank.device
    if row_weights is not None:
        if row_weights.shape != (len(bank),):
            raise ValueError(f"row_weights must have shape ({len(bank)},), got {tuple(row_weights.shape)}.")
        if not torch.isfinite(row_weights).all() or (row_weights < 0.0).any():
            raise ValueError("row_weights must be finite and non-negative.")
        row_weights = row_weights.to(device)
    modes = torch.multinomial(shares.to(device), num, replacement=True)
    rows = torch.full((num,), -1, dtype=torch.long, device=device)
    for mode_id, mode in enumerate(RESET_MODES[1:], start=1):
        selected = modes == mode_id
        count = int(selected.sum())
        if count > 0:
            bucket = bank.bucket_rows(mode)
            if row_weights is None:
                positions = torch.randint(len(bucket), (count,), device=device)
            else:
                weights = row_weights[bucket]
                if weights.sum() <= 0.0:
                    raise ValueError(f"row_weights sum to zero inside the '{mode}' bucket.")
                positions = torch.multinomial(weights, count, replacement=True)
            rows[selected] = bucket[positions]
    return rows


def subsample_to_quotas(
    bank: ResetBank, quotas: dict[str, int], generator: torch.Generator | None = None
) -> torch.Tensor:
    """Pick the rows of a bank to keep: ``quotas[name]`` random rows of each named bucket.

    Args:
        bank: The bank to pick from.
        quotas: Number of rows to keep per bucket name, e.g. ``{"grasp": 333, "drag": 333, "goal": 333}``.
        generator: Random generator on the CPU, for a reproducible pick.

    Returns:
        The kept rows, sorted, on the CPU.

    Raises:
        ValueError: If a bucket has fewer rows than its quota.
    """
    kept = []
    for name, quota in quotas.items():
        rows = bank.bucket_rows(name).cpu()
        if len(rows) < quota:
            raise ValueError(f"Bucket '{name}' has {len(rows)} states, fewer than its quota of {quota}.")
        kept.append(rows[torch.randperm(len(rows), generator=generator)[:quota]])
    return (torch.cat(kept) if kept else torch.zeros(0, dtype=torch.long)).sort().values

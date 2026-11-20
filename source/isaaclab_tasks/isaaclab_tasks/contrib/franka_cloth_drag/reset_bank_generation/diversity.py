# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical-diversity analysis and deterministic selection for cloth-drag reset banks."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..mdp.stages import START_STAGE_NAMES
from ..reset_strategies.bank import ResetBank


@dataclass
class ResetBankDiversityCfg:
    """Settings for robust farthest-point selection in physically meaningful coordinates."""

    progress_bins: int = 3
    """Number of stage-progress strata; three preserves beginning, middle, and end states."""

    cloth_components: int = 8
    """Principal components retained from translation-normalized cloth shape."""

    clip: float = 4.0
    """Absolute robust-scaled feature limit used to prevent outliers dominating selection."""

    seed: int = 0
    """Seed used only to break exact distance ties."""


def physical_features(bank: ResetBank, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, slice]]:
    """Build relative physical features and a stage-progress coordinate for bank rows.

    The feature blocks are arm joint pose (the stored representation that determines hand pose), cloth extent and
    translation-normalized shape, cube and target relative to the cloth, raised-node geometry, and grasp state. The
    progress coordinate is the cloth goal point's planar distance to the target; within-stage quantiles therefore
    represent early, middle, and late captures without depending on absolute workspace position.

    Args:
        bank: Source reset bank.
        rows: Rows to describe, shape [N].

    Returns:
        Tuple of feature matrix [N, D], progress [N], and named feature slices.
    """
    rows = rows.to(bank.device)
    scene, task = bank.data["scene"], bank.data["task"]
    nodes = scene["nodal_pos"][rows].float()
    center = nodes.mean(dim=1)
    centered = nodes - center.unsqueeze(1)
    extent = nodes.amax(dim=1) - nodes.amin(dim=1)
    flat_shape = centered.flatten(1)
    components = min(8, len(rows), flat_shape.shape[1])
    if components > 0 and len(rows) > 1:
        centered_shape = flat_shape - flat_shape.mean(dim=0)
        _, _, vectors = torch.linalg.svd(centered_shape, full_matrices=False)
        shape = centered_shape @ vectors[:components].T
    else:
        shape = flat_shape[:, :components]

    raised_ids = task["raised_node_ids"][rows].long()
    raised = nodes.gather(1, raised_ids.unsqueeze(-1).expand(-1, -1, 3))
    raised_center = raised.mean(dim=1)
    raised_relative = (raised - raised_center.unsqueeze(1)).flatten(1)
    target_relative = task["target_pos_b"][rows].float() - center
    blocks = {
        "arm_joint_pose": scene["joint_pos"][rows, :7].float(),
        "cloth_extent": extent,
        "cloth_shape": shape,
        "target_relative_cloth": target_relative,
        "raised_relative_cloth": torch.cat((raised_center - center, raised_relative), dim=1),
    }
    if "cube_root_pose" in scene:
        blocks["cube_relative_cloth"] = scene["cube_root_pose"][rows, :3].float() - center
    if "ee_pos_b" in task:
        ee_pos = task["ee_pos_b"][rows].float()
        blocks["hand_relative_raised"] = ee_pos - raised_center
        if "cube_root_pose" in scene:
            blocks["hand_relative_cube"] = ee_pos - scene["cube_root_pose"][rows, :3].float()
    if "squeeze_progress" in task:
        blocks["grasp_progress"] = task["squeeze_progress"][rows].float().reshape(len(rows), -1)

    slices: dict[str, slice] = {}
    offset = 0
    for name, value in blocks.items():
        slices[name] = slice(offset, offset + value.shape[1])
        offset += value.shape[1]
    features = torch.cat(list(blocks.values()), dim=1)
    stage = START_STAGE_NAMES[int(bank.stage[rows[0]])]
    if stage == "grasp" and "ee_pos_b" in task:
        progress = (raised - task["ee_pos_b"][rows].float().unsqueeze(1)).norm(dim=2).min(dim=1).values
    else:
        progress = target_relative[:, :2].norm(dim=1)
    return features, progress, slices


def select_diverse_rows(
    bank: ResetBank, stage: str, count: int, cfg: ResetBankDiversityCfg | None = None
) -> torch.Tensor:
    """Select a deterministic, progress-stratified physical k-center subset of one stage."""
    cfg = ResetBankDiversityCfg() if cfg is None else cfg
    rows = bank.stage_rows(stage)
    if count < 1 or count > len(rows):
        raise ValueError(f"Requested {count} rows from stage '{stage}' with {len(rows)} states.")
    features, progress, _ = physical_features(bank, rows)
    features = _robust_scale(features, cfg.clip)
    order = torch.argsort(progress, stable=True)
    bins = torch.tensor_split(order, min(cfg.progress_bins, count, len(rows)))
    base, remainder = divmod(count, len(bins))
    selected = []
    for index, members in enumerate(bins):
        quota = base + (index < remainder)
        selected.append(members[_farthest_points(features[members], quota, cfg.seed + index)])
    return rows[torch.cat(selected)].sort().values.cpu()


def diversity_report(bank: ResetBank) -> dict:
    """Return JSON-serializable relative-physics diversity statistics for every non-empty stage."""
    report = {"stage_counts": bank.stage_counts(), "stages": {}}
    for stage in START_STAGE_NAMES:
        rows = bank.stage_rows(stage)
        if rows.numel() == 0:
            continue
        features, progress, blocks = physical_features(bank, rows)
        scaled = _robust_scale(features, 4.0)
        nearest = _nearest_neighbor_distance(scaled)
        report["stages"][stage] = {
            "count": len(rows),
            "feature_dimensions": {name: part.stop - part.start for name, part in blocks.items()},
            "effective_rank": float(torch.linalg.matrix_rank(scaled).cpu()),
            "nearest_neighbor_median": float(nearest.median().cpu()),
            "nearest_neighbor_p10": float(nearest.quantile(0.1).cpu()),
            "progress_min": float(progress.min().cpu()),
            "progress_median": float(progress.median().cpu()),
            "progress_max": float(progress.max().cpu()),
        }
    return report


def _robust_scale(features: torch.Tensor, clip: float) -> torch.Tensor:
    median = features.median(dim=0).values
    deviation = (features - median).abs().median(dim=0).values
    scale = (1.4826 * deviation).clamp_min(1.0e-6)
    usable = deviation > 1.0e-8
    return ((features[:, usable] - median[usable]) / scale[usable]).clamp(-clip, clip)


def _farthest_points(features: torch.Tensor, count: int, seed: int) -> torch.Tensor:
    if count >= len(features):
        return torch.arange(len(features), device=features.device)
    generator = torch.Generator(device=features.device).manual_seed(seed)
    tie_noise = torch.rand(len(features), generator=generator, device=features.device) * 1.0e-7
    center = features.median(dim=0).values
    first = ((features - center).square().sum(dim=1) + tie_noise).argmin()
    chosen = [first]
    distance = torch.cdist(features, features[first].unsqueeze(0)).squeeze(1)
    for _ in range(1, count):
        next_row = (distance + tie_noise).argmax()
        chosen.append(next_row)
        distance.minimum_(torch.cdist(features, features[next_row].unsqueeze(0)).squeeze(1))
    return torch.stack(chosen)


def _nearest_neighbor_distance(features: torch.Tensor) -> torch.Tensor:
    if len(features) < 2:
        return torch.zeros(len(features), device=features.device)
    distances = torch.cdist(features, features)
    distances.fill_diagonal_(torch.inf)
    return distances.min(dim=1).values

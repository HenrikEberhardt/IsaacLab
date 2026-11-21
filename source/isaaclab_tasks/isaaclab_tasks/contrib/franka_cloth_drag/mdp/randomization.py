# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Domain randomization of the cloth: its material, the friction of its contact partners, and the perception error of
its observed points.

Both events write through the task's physics manager (see :mod:`..physics.coupler`), which keeps the model and the
solver views in sync, and keep their per-environment draws in ``scales`` for the critic.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils.noise import NoiseModel

if TYPE_CHECKING:
    from isaaclab.assets import DeformableObject
    from isaaclab.envs import ManagerBasedEnv

    from .randomization_cfg import ClothPointsNoiseModelCfg


class ClothPointsNoiseModel(NoiseModel):
    """Perception error of the observed cloth points, see :class:`~.randomization_cfg.ClothPointsNoiseModelCfg`.

    The offsets are drawn per episode, so an episode's history stays coherent; only the raised nodes jitter per frame.
    """

    def __init__(self, noise_model_cfg: ClothPointsNoiseModelCfg, num_envs: int, device: str):
        """Draw the first offsets of every environment."""
        super().__init__(noise_model_cfg, num_envs, device)
        self._cfg = noise_model_cfg
        self._cloth_offset = torch.zeros(num_envs, 2, device=device)
        self._raised_offset = torch.zeros(num_envs, 2, device=device)
        self.reset()

    def reset(self, env_ids: Sequence[int] | None = None):
        """Draw new planar offsets [m] for the given environments, or all of them."""
        env_ids = slice(None) if env_ids is None else env_ids
        num = self._cloth_offset[env_ids].shape[0]
        for offset, bounds in (
            (self._cloth_offset, self._cfg.cloth_offset_range),
            (self._raised_offset, self._cfg.raised_offset_range),
        ):
            low, high = bounds
            offset[env_ids] = low + (high - low) * torch.rand(num, 2, device=self._device)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Shift all points by the cloth offset and the raised nodes by their offset and per-frame noise.

        Args:
            data: Flattened points [m], shape [num_envs, num_points * 3].

        Returns:
            The noisy points [m], same shape.
        """
        points = data.reshape(data.shape[0], -1, 3).clone()
        points[..., :2] += self._cloth_offset.unsqueeze(1)
        raised = slice(self._cfg.num_landmarks, self._cfg.num_landmarks + self._cfg.num_raised)
        points[:, raised, :2] += self._raised_offset.unsqueeze(1)
        points[:, raised] += self._cfg.raised_noise_std * torch.randn_like(points[:, raised])
        return points.reshape(data.shape)


def _sample(bounds: tuple[float, float], num: int, device: str, log: bool = False) -> torch.Tensor:
    """Draw ``num`` values from ``bounds``, uniformly or log-uniformly."""
    low, high = (math.log(bounds[0]), math.log(bounds[1])) if log else bounds
    values = low + (high - low) * torch.rand(num, device=device)
    return values.exp() if log else values


def _env_ids(env: ManagerBasedEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
    """Return the given environment ids, or all of them."""
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device)
    return env_ids.to(env.device).long()


def _physics_manager(env: ManagerBasedEnv, method: str):
    """Return the physics manager, which must provide ``method`` (the task's coupler)."""
    physics = env.sim.physics_manager
    if not hasattr(physics, method):
        raise TypeError(f"The physics manager {physics.__name__} does not provide '{method}'.")
    return physics


class randomize_cloth_material(ManagerTermBase):
    """Draw the cloth's areal density, area stiffness, area damping and bending stiffness per environment.

    ``density_range`` is the areal density [kg/m^2] and rescales the authored particle masses. The other ranges
    multiply the authored ``tri_ka``, ``tri_kd`` and ``edge_ke`` and are sampled log-uniformly, so (0.5, 2.0) is
    centred on the authored value. Adapted from ``randomize_cloth_material`` of srl-ethz/IsaacLab-Cloth.

    The cloth asset restores the inverse masses of its free particles every step from its own defaults (they are
    zeroed while the gripper locks nodes), so those defaults are rescaled as well.
    """

    SCALE_NAMES = ("density", "tri_ka", "tri_kd", "edge_ke")
    """Names of the columns of :attr:`scales`."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Map the cloth particles, triangles and edges to their environments."""
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("deformable"))
        self._cloth: DeformableObject = env.scene[asset_cfg.name]
        self._authored_density = self._cloth.cfg.spawn.physics_material.density
        self._physics = _physics_manager(env, "set_cloth_material")
        model = NewtonManager.get_model()
        self._particle_env = wp.to_torch(model.particle_world).long()
        self._tri_env = self._particle_env[wp.to_torch(model.tri_indices)[:, 0].long()]
        # the third node of a bending edge always exists
        self._edge_env = self._particle_env[wp.to_torch(model.edge_indices)[:, 2].long()]
        self.scales = torch.ones(env.num_envs, len(self.SCALE_NAMES), device=env.device)
        """Drawn values relative to the authored ones, columns :attr:`SCALE_NAMES`, shape [num_envs, 4]."""
        # authored values, read at the first call once every model setup has run
        self._authored: dict[str, torch.Tensor] | None = None

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        density_range: tuple[float, float],
        tri_ka_range: tuple[float, float],
        tri_kd_range: tuple[float, float],
        edge_ke_range: tuple[float, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    ):
        """Draw new materials for the given environments and write them into the simulation."""
        env_ids = _env_ids(env, env_ids)
        if self._authored is None:
            self._authored = {
                "particle_mass": self._physics.get_cloth_material(
                    "particle_mass", torch.arange(len(self._particle_env), device=env.device)
                ),
                "particle_inv_mass": wp.to_torch(self._cloth._default_particle_inv_mass).clone(),
                "tri_materials": self._physics.get_cloth_material(
                    "tri_materials", torch.arange(len(self._tri_env), device=env.device)
                ),
                "edge_bending_properties": self._physics.get_cloth_material(
                    "edge_bending_properties", torch.arange(len(self._edge_env), device=env.device)
                ),
            }
        num = len(env_ids)
        self.scales[env_ids, 0] = _sample(density_range, num, env.device) / self._authored_density
        self.scales[env_ids, 1] = _sample(tri_ka_range, num, env.device, log=True)
        self.scales[env_ids, 2] = _sample(tri_kd_range, num, env.device, log=True)
        self.scales[env_ids, 3] = _sample(edge_ke_range, num, env.device, log=True)
        density, ka, kd, ke = self.scales.unbind(dim=1)

        # masses: the model and the views, and the cloth asset's defaults of the inverse masses
        ids = torch.isin(self._particle_env, env_ids).nonzero().squeeze(-1)
        scale = density[self._particle_env[ids]]
        inv_mass = self._authored["particle_inv_mass"][ids] / scale
        self._physics.set_cloth_material("particle_mass", ids, self._authored["particle_mass"][ids] * scale)
        self._physics.set_cloth_material("particle_inv_mass", ids, inv_mass)
        wp.to_torch(self._cloth._default_particle_inv_mass)[ids] = inv_mass

        # triangles: (ke, ka, kd, drag, lift); the stretch stiffness ke stays authored
        ids = torch.isin(self._tri_env, env_ids).nonzero().squeeze(-1)
        materials = self._authored["tri_materials"][ids].clone()
        materials[:, 1] *= ka[self._tri_env[ids]]
        materials[:, 2] *= kd[self._tri_env[ids]]
        self._physics.set_cloth_material("tri_materials", ids, materials)

        # bending edges: (ke, kd)
        ids = torch.isin(self._edge_env, env_ids).nonzero().squeeze(-1)
        bending = self._authored["edge_bending_properties"][ids].clone()
        bending[:, 0] *= ke[self._edge_env[ids]]
        self._physics.set_cloth_material("edge_bending_properties", ids, bending)


class randomize_cloth_friction(ManagerTermBase):
    """Scale the friction of the cloth's contact partners per environment, one draw per shape group.

    VBD's body-particle friction is ``sqrt(soft_contact_mu * shape_mu)``, so scaling the friction of a shape by ``s``
    scales its friction with the cloth by ``sqrt(s)``. ``shape_patterns`` maps group names to regular expressions of
    shape or body labels (e.g. the table and the cube); each group draws its own factor uniformly from
    ``mu_scale_range``. Adapted from ``randomize_cloth_friction`` of srl-ethz/IsaacLab-Cloth (cloth-fold-environments),
    which sets one value for every shape of an environment.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Find the shapes of every group."""
        super().__init__(cfg, env)
        self._physics = _physics_manager(env, "set_shape_friction")
        patterns: dict[str, str] = cfg.params["shape_patterns"]
        self.group_names = list(patterns)
        """Names of the shape groups, the columns of :attr:`scales`."""
        self._groups = [self._physics.find_shapes(pattern) for pattern in patterns.values()]
        self.scales = torch.ones(env.num_envs, len(self.group_names), device=env.device)
        """Drawn friction factors of the shape groups, shape [num_envs, num_groups]."""
        # authored friction of every group, read at the first call once every model setup has run
        self._authored: list[torch.Tensor] | None = None

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        shape_patterns: dict[str, str],
        mu_scale_range: tuple[float, float],
    ):
        """Draw new friction factors for the given environments and write them into the simulation."""
        env_ids = _env_ids(env, env_ids)
        if self._authored is None:
            self._authored = [self._physics.get_shape_friction(ids) for ids, _ in self._groups]
        for group, ((ids, worlds), authored) in enumerate(zip(self._groups, self._authored)):
            self.scales[env_ids, group] = _sample(mu_scale_range, len(env_ids), env.device)
            selected = torch.isin(worlds, env_ids)
            values = authored[selected] * self.scales[worlds[selected], group]
            self._physics.set_shape_friction(ids[selected], values)

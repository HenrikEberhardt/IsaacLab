# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import contextlib
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import gymnasium as gym
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import (
        DirectRLEnv,
        ManagerBasedRLEnv,
    )

    with contextlib.suppress(ImportError):
        from isaaclab_experimental.envs import DirectRLEnvWarp, ManagerBasedRLEnvWarp


class RslRlVecEnvWrapper(VecEnv):
    """Wraps around Isaac Lab environment for the RSL-RL library

    .. caution::
        This class must be the last wrapper in the wrapper chain. This is because the wrapper does not follow
        the :class:`gym.Wrapper` interface. Any subsequent wrappers will need to be modified to work with this
        wrapper.

    Reference:
        https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/env/vec_env.py
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv | DirectRLEnv,
        clip_actions: float | None = None,
        action_clip_diagnostics: float | None = None,
        action_dim_names: Sequence[str] | None = None,
    ):
        """Initializes the wrapper.

        Note:
            The wrapper calls :meth:`reset` at the start since the RSL-RL runner does not call reset.

        Args:
            env: The environment to wrap around.
            clip_actions: The clipping value for actions. If ``None``, then no clipping is done.
            action_clip_diagnostics: The nominal action magnitude threshold for clipping diagnostics. If ``None``,
                nominal-range diagnostics are not logged. Safety-clip diagnostics use ``clip_actions`` when configured.
            action_dim_names: Names of the action dimensions in the per-dimension diagnostics. If ``None``, or if
                their number differs from the action dimension, the dimensions are named ``dim_<index>``.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.
        """
        # check that input is valid
        # NOTE: import here (not at module level) to avoid loading heavy env classes before Isaac Sim is initialized.
        from isaaclab.envs import DirectRLEnv, ManagerBasedEnv, ManagerBasedRLEnv

        try:
            from isaaclab_experimental.envs import DirectRLEnvWarp, ManagerBasedEnvWarp, ManagerBasedRLEnvWarp
        except ImportError:
            DirectRLEnvWarp = None
            ManagerBasedEnvWarp = None
            ManagerBasedRLEnvWarp = None

        allowed_types = (ManagerBasedRLEnv, ManagerBasedEnv, DirectRLEnv)
        if DirectRLEnvWarp is not None:
            allowed_types += (DirectRLEnvWarp,)
        if ManagerBasedEnvWarp is not None:
            allowed_types += (ManagerBasedEnvWarp,)
        if ManagerBasedRLEnvWarp is not None:
            allowed_types += (ManagerBasedRLEnvWarp,)

        if not isinstance(env.unwrapped, allowed_types):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv / DirectRLEnv / DirectRLEnvWarp /"
                " ManagerBasedRLEnvWarp. Environment type:"
                f" {type(env.unwrapped)}"
            )

        # initialize the wrapper
        self.env = env
        self.clip_actions = clip_actions
        self.action_clip_diagnostics = action_clip_diagnostics

        # store information required by wrapper
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        # obtain dimensions of the environment
        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)

        # names of the action dimensions in the per-dimension diagnostics
        self.action_dim_names = [f"dim_{index}" for index in range(self.num_actions)]
        if action_dim_names is not None:
            if len(action_dim_names) == self.num_actions:
                self.action_dim_names = list(action_dim_names)
            else:
                warnings.warn(
                    f"Got {len(action_dim_names)} action dimension names for {self.num_actions} action dimensions;"
                    " naming them by index instead."
                )
        # policy whose action distribution is logged with the diagnostics, see :meth:`set_diagnostics_policy`
        self._diagnostics_policy = None

        # modify the action space to the clip range
        self._modify_action_space()

        # reset at the start since the RSL-RL runner does not call reset
        self.env.reset()

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def cfg(self) -> object:
        """Returns the configuration class instance of the environment."""
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`observation_space`."""
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv | DirectRLEnvWarp | ManagerBasedRLEnvWarp:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    """
    Properties
    """

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.

        Note:
            This is needed to perform random initialization of episode lengths in RSL-RL.
        """
        self.unwrapped.episode_length_buf = value

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def reset(self) -> tuple[TensorDict, dict]:  # noqa: D102
        # reset the environment
        obs_dict, extras = self.env.reset()
        return TensorDict(obs_dict, batch_size=[self.num_envs]), extras

    def get_observations(self) -> TensorDict:
        """Returns the current observations of the environment."""
        return TensorDict(self.unwrapped.obs_buf, batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        raw_actions = actions
        # clip actions
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        self._log_action_clip_diagnostics(raw_actions, extras)
        # return the step information
        return TensorDict(obs_dict, batch_size=[self.num_envs]), rew, dones, extras

    def close(self):  # noqa: D102
        return self.env.close()

    def set_diagnostics_policy(self, policy: object | None):
        """Set the policy whose action distribution the action diagnostics log.

        At every :meth:`step`, the diagnostics then log the mean over the environments of the policy's distribution
        mean and standard deviation per action dimension (``Action/policy_mean_<name>`` and
        ``Action/policy_std_<name>``). The policy must expose ``output_mean`` and ``output_std`` of the distribution
        it sampled the stepped actions from, as the RSL-RL models do after acting.

        Args:
            policy: The policy, or None to stop logging its distribution.
        """
        self._diagnostics_policy = policy

    """
    Helper functions
    """

    def _modify_action_space(self):
        """Modifies the action space to the clip range."""
        if self.clip_actions is None:
            return

        # modify the action space to the clip range
        # note: this is only possible for the box action space. we need to change it in the future for other
        #   action spaces.
        self.env.unwrapped.single_action_space = gym.spaces.Box(
            low=-self.clip_actions, high=self.clip_actions, shape=(self.num_actions,)
        )
        self.env.unwrapped.action_space = gym.vector.utils.batch_space(
            self.env.unwrapped.single_action_space, self.num_envs
        )

    def _log_action_clip_diagnostics(self, actions: torch.Tensor, extras: dict):
        """Log nominal-range violations and actual safety clipping of raw policy actions, and the distribution of the
        diagnostics policy per action dimension."""
        if self.action_clip_diagnostics is None and self.clip_actions is None and self._diagnostics_policy is None:
            return

        log = extras.setdefault("log", {})
        if self.action_clip_diagnostics is not None:
            self._log_action_threshold(actions, log, self.action_clip_diagnostics, "Action/outside_nominal")
        if self.clip_actions is not None:
            self._log_action_threshold(actions, log, self.clip_actions, "Action/safety_clip")
        if self._diagnostics_policy is not None:
            mean = self._diagnostics_policy.output_mean.reshape(-1, self.num_actions).mean(dim=0)
            std = self._diagnostics_policy.output_std.reshape(-1, self.num_actions).mean(dim=0)
            for index, name in enumerate(self.action_dim_names):
                log[f"Action/policy_mean_{name}"] = mean[index]
                log[f"Action/policy_std_{name}"] = std[index]

    def _log_action_threshold(self, actions: torch.Tensor, log: dict, threshold: float, prefix: str):
        """Log raw-action exceedance statistics for one symmetric magnitude threshold."""
        excess = torch.clamp(actions.abs() - threshold, min=0.0)
        exceeded = excess > 0.0
        log[f"{prefix}_fraction"] = exceeded.float().mean()
        log[f"{prefix}_env_fraction"] = exceeded.any(dim=-1).float().mean()
        log[f"{prefix}_correction_l1_mean"] = excess.mean()
        log[f"{prefix}_excess_max"] = excess.max()
        log[f"{prefix}_excess_mean"] = excess.sum() / exceeded.sum().clamp(min=1)
        for index, name in enumerate(self.action_dim_names):
            log[f"{prefix}_fraction_{name}"] = exceeded[..., index].float().mean()

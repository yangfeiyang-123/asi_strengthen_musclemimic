"""Observation filtering for distilled student policies.

The default student policy keeps all non-goal observations and only the final
goal element, which is the motion phase when GoalTrajMimic has
enable_motion_phase=True.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from loco_mujoco.core.utils import Box
from musclemimic.core.wrappers.mjx import BaseWrapper


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


@dataclass(frozen=True)
class StudentObsSpec:
    """Index mapping from teacher/full observations to student observations."""

    raw_obs_dim: int
    goal_indices: np.ndarray
    state_indices: np.ndarray
    student_indices: np.ndarray
    phase_index: int | None
    keep_motion_phase: bool = True

    @property
    def student_obs_dim(self) -> int:
        return int(self.student_indices.size)

    @property
    def phase_student_index(self) -> int | None:
        if self.phase_index is None:
            return None
        matches = np.where(self.student_indices == self.phase_index)[0]
        return int(matches[-1]) if matches.size else None


class StudentObsContainer:
    """Minimal observation container for filtered student observations."""

    def __init__(self, group_indices: dict[str, np.ndarray]):
        self._group_indices = {
            name: np.asarray(indices, dtype=int)
            for name, indices in group_indices.items()
        }

    def get_obs_ind_by_group(self, group_name: str | None = None) -> np.ndarray:
        if group_name is None:
            if not self._group_indices:
                return np.array([], dtype=int)
            return np.concatenate(list(self._group_indices.values())).astype(int)
        return self._group_indices.get(group_name, np.array([], dtype=int))

    def get_all_group_names(self) -> list[str]:
        return list(self._group_indices)

    def filter_by_group(self, obs, group_name: str | None = None):
        return obs[..., self.get_obs_ind_by_group(group_name)]

    def entries(self):
        return []

    def keys(self):
        return self._group_indices.keys()

    def items(self):
        return self._group_indices.items()

    def __contains__(self, key: str) -> bool:
        return key in self._group_indices

    def __getitem__(self, key: str) -> np.ndarray:
        return self._group_indices[key]

    def get(self, key: str, default=None):
        return self._group_indices.get(key, default)


def build_student_obs_indices(env: Any, config: Any = None) -> StudentObsSpec:
    """Build the student observation index spec from an environment."""
    require_goal_group = bool(_cfg_get(config, "require_goal_group", True))
    keep_motion_phase = bool(_cfg_get(config, "keep_motion_phase", True))
    require_motion_phase = bool(_cfg_get(config, "require_motion_phase", keep_motion_phase))
    drop_goal_lookahead = bool(_cfg_get(config, "drop_goal_lookahead", True))

    if keep_motion_phase and not drop_goal_lookahead:
        raise ValueError(
            "student_obs_filter keep_motion_phase=True requires drop_goal_lookahead=True "
            "for the v1 state+phase student observation layout"
        )

    if not hasattr(env, "obs_container"):
        if require_goal_group:
            raise ValueError("student_obs_filter requires env.obs_container with a goal group")
        goal_indices = np.array([], dtype=int)
    else:
        goal_indices = np.asarray(env.obs_container.get_obs_ind_by_group("goal"), dtype=int)

    if goal_indices.size == 0:
        if require_goal_group:
            raise ValueError("student_obs_filter requires non-empty goal observations")
        phase_index = None
    else:
        phase_index = int(goal_indices[-1]) if keep_motion_phase else None

    if keep_motion_phase and phase_index is None and require_motion_phase:
        raise ValueError("student_obs_filter keep_motion_phase=True requires a goal phase element")

    raw_obs_dim = int(env.info.observation_space.shape[0])
    all_indices = np.arange(raw_obs_dim, dtype=int)
    state_mask = np.ones(raw_obs_dim, dtype=bool)
    if goal_indices.size:
        state_mask[goal_indices] = False
    state_indices = all_indices[state_mask]

    if not drop_goal_lookahead:
        student_indices = all_indices.copy()
    elif keep_motion_phase and phase_index is not None:
        student_indices = np.concatenate([state_indices, np.array([phase_index], dtype=int)])
    else:
        student_indices = state_indices.copy()

    return StudentObsSpec(
        raw_obs_dim=raw_obs_dim,
        goal_indices=goal_indices,
        state_indices=state_indices,
        student_indices=student_indices.astype(int),
        phase_index=phase_index,
        keep_motion_phase=keep_motion_phase,
    )


def filter_student_obs(obs, spec: StudentObsSpec):
    """Filter raw observations using a StudentObsSpec."""
    return obs[..., jnp.asarray(spec.student_indices)]


class StudentObservationFilterWrapper(BaseWrapper):
    """Filter policy observations for student policies.

    The wrapper keeps the underlying environment state and reward computation
    unchanged. Only observations returned to the policy are filtered.
    """

    def __init__(self, env: Any, config: Any = None, **kwargs: Any):
        super().__init__(env)
        cfg = dict(config or {})
        cfg.update(kwargs)
        self.config = cfg
        self.spec = build_student_obs_indices(self.env, cfg)
        self.info = self._update_info(self.env.info)
        self.mdp_info = self.info
        self.obs_container = self._build_obs_container()

    def _update_info(self, info):
        new_info = deepcopy(info)
        high = np.asarray(info.observation_space.high)[self.spec.student_indices]
        low = np.asarray(info.observation_space.low)[self.spec.student_indices]
        new_info.observation_space = Box(low, high)
        return new_info

    def _build_obs_container(self) -> StudentObsContainer:
        state_dim = int(self.spec.state_indices.size)
        groups = {"state": np.arange(state_dim, dtype=int)}
        if self.spec.keep_motion_phase and self.spec.phase_index is not None:
            groups["goal"] = np.array([state_dim], dtype=int)
            groups["phase"] = np.array([state_dim], dtype=int)
        return StudentObsContainer(groups)

    def reset(self, rng_key):
        obs, env_state = self.env.reset(rng_key)
        return filter_student_obs(obs, self.spec), env_state

    def reset_to(self, rng_key, traj_idx):
        obs, env_state = self.env.reset_to(rng_key, traj_idx)
        return filter_student_obs(obs, self.spec), env_state

    def step(self, state, action):
        obs, reward, absorbing, done, info, env_state = self.env.step(state, action)
        return filter_student_obs(obs, self.spec), reward, absorbing, done, info, env_state

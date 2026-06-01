from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mujoco
import numpy as np

from environment.overall_environment.src.body_obs_adapter import _legacy_body_env_from_params


class TrajectoryGoalProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrajectoryGoalProviderReport:
    source: str
    motion_path: str
    traj_step: int
    traj_len: int
    goal_size: int


class TrajectoryGoalProvider:
    """Build checkpoint-compatible GoalTrajMimic observations from cached trajectories."""

    def __init__(
        self,
        *,
        checkpoint: Path,
        motion_path: str,
        cache_path: Path,
        env_params: dict[str, Any],
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.motion_path = str(motion_path)
        self.cache_path = Path(cache_path)
        self.source = "trajectory_cache"
        if not self.cache_path.is_file():
            raise TrajectoryGoalProviderError(f"trajectory cache not found: {self.cache_path}")

        self._legacy_env = _legacy_body_env_from_params(env_params)
        self._legacy_model = getattr(self._legacy_env, "_model", None) or getattr(self._legacy_env, "model", None)
        if self._legacy_model is None:
            raise TrajectoryGoalProviderError("could not access legacy body MuJoCo model")
        self._legacy_data = mujoco.MjData(self._legacy_model)
        mujoco.mj_resetData(self._legacy_model, self._legacy_data)
        mujoco.mj_forward(self._legacy_model, self._legacy_data)

        self._goal_params = dict(env_params["goal_params"])
        self._trajectory_handler = self._load_trajectory_handler(self.cache_path)
        self._goal = self._init_goal()
        self._traj_step = 0
        self._last_built_step = 0
        self._copy_maps_by_source_signature: dict[tuple[int, int, int, int], _StateCopyMap] = {}

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        motion_index: int = 0,
        cache_root: str | Path | None = None,
    ) -> "TrajectoryGoalProvider":
        checkpoint_path = Path(checkpoint)
        metadata = _checkpoint_metadata(checkpoint_path)
        env_params = dict(metadata["experiment"]["env_params"])
        dataset_conf = (
            metadata.get("experiment", {})
            .get("task_factory", {})
            .get("params", {})
            .get("amass_dataset_conf", {})
        )
        rel_paths = dataset_conf.get("rel_dataset_path", [])
        if not rel_paths:
            raise TrajectoryGoalProviderError("checkpoint metadata has no amass rel_dataset_path")
        if motion_index < 0 or motion_index >= len(rel_paths):
            raise TrajectoryGoalProviderError(f"motion_index {motion_index} out of range for {len(rel_paths)} motions")
        motion_path = str(rel_paths[motion_index])

        retargeting_method = str(dataset_conf.get("retargeting_method", "gmr"))
        env_name = str(env_params.get("env_name", "MjxMyoFullBody")).replace("Mjx", "")
        root = Path(cache_root) if cache_root is not None else _repo_root() / "caches" / "AMASS"
        cache_parts = [root, env_name]
        if retargeting_method == "gmr":
            cache_parts.append("gmr")
        cache_path = Path(*cache_parts) / f"{motion_path}.npz"
        return cls(
            checkpoint=checkpoint_path,
            motion_path=motion_path,
            cache_path=cache_path,
            env_params=env_params,
        )

    def reset(self, *, traj_step: int = 0) -> None:
        self._traj_step = self._clamp_step(traj_step)

    def advance(self, steps: int = 1) -> None:
        self._traj_step = self._clamp_step(self._traj_step + int(steps))

    def build(self, source_model: mujoco.MjModel, source_data: mujoco.MjData) -> np.ndarray:
        self._copy_source_state_to_legacy(source_model, source_data)
        self._last_built_step = int(self._traj_step)
        carry = SimpleNamespace(
            traj_state=SimpleNamespace(
                traj_no=0,
                subtraj_step_no=int(self._last_built_step),
                subtraj_step_no_init=0,
            )
        )
        goal_env = SimpleNamespace(
            root_free_joint_xml_name=self._legacy_env.root_free_joint_xml_name,
            sites_for_mimic=list(self._goal_params["sites_for_mimic"]),
            th=self._trajectory_handler,
        )
        goal_obs, _ = self._goal.get_obs_and_update_state(
            goal_env,
            self._legacy_model,
            self._legacy_data,
            carry,
            np,
        )
        goal = np.asarray(goal_obs, dtype=float)
        if goal.shape != (self.goal_size,):
            raise TrajectoryGoalProviderError(f"goal observation has shape {goal.shape}, expected ({self.goal_size},)")
        if not np.isfinite(goal).all():
            raise TrajectoryGoalProviderError("goal observation contains non-finite values")
        return goal

    @property
    def goal_size(self) -> int:
        return int(self._goal.dim)

    @property
    def traj_step(self) -> int:
        return int(self._traj_step)

    @property
    def last_built_step(self) -> int:
        return int(self._last_built_step)

    @property
    def traj_len(self) -> int:
        return int(self._trajectory_handler.len_trajectory(0))

    def report(self) -> TrajectoryGoalProviderReport:
        return TrajectoryGoalProviderReport(
            source=self.source,
            motion_path=self.motion_path,
            traj_step=self.last_built_step,
            traj_len=self.traj_len,
            goal_size=self.goal_size,
        )

    def _load_trajectory_handler(self, cache_path: Path):
        import musclemimic.core  # noqa: F401
        from loco_mujoco.trajectory import Trajectory, TrajectoryHandler

        trajectory = Trajectory.load(cache_path, backend=np)
        trajectory = replace(
            trajectory,
            data=trajectory.data.to_numpy(),
            info=replace(trajectory.info, model=trajectory.info.model.to_numpy()),
        )
        return TrajectoryHandler(
            self._legacy_model,
            control_dt=self._legacy_env.dt,
            traj=trajectory,
            random_start=False,
            fixed_start_conf=(0, 0),
            start_from_random_step=False,
        )

    def _init_goal(self):
        from musclemimic.core.goals.trajectory import GoalTrajMimic

        goal = GoalTrajMimic(self._goal_params, **self._goal_params)
        goal._init_from_mj(self._legacy_env, self._legacy_model, self._legacy_data, 0)
        goal.init_from_traj(self._trajectory_handler)
        return goal

    def _copy_source_state_to_legacy(self, source_model: mujoco.MjModel, source_data: mujoco.MjData) -> None:
        signature = (int(source_model.njnt), int(source_model.nq), int(source_model.nv), id(source_model))
        copy_map = self._copy_maps_by_source_signature.get(signature)
        if copy_map is None:
            copy_map = _build_state_copy_map(source_model, self._legacy_model)
            self._copy_maps_by_source_signature[signature] = copy_map
        for src_start, dst_start, width in copy_map.qpos:
            self._legacy_data.qpos[dst_start : dst_start + width] = source_data.qpos[src_start : src_start + width]
        for src_start, dst_start, width in copy_map.qvel:
            self._legacy_data.qvel[dst_start : dst_start + width] = source_data.qvel[src_start : src_start + width]
        mujoco.mj_forward(self._legacy_model, self._legacy_data)

    def _clamp_step(self, step: int) -> int:
        return max(0, min(int(step), self.traj_len - 1))


@dataclass(frozen=True)
class _StateCopyMap:
    qpos: tuple[tuple[int, int, int], ...]
    qvel: tuple[tuple[int, int, int], ...]


def _build_state_copy_map(source_model: mujoco.MjModel, target_model: mujoco.MjModel) -> _StateCopyMap:
    qpos: list[tuple[int, int, int]] = []
    qvel: list[tuple[int, int, int]] = []
    for target_joint_id in range(target_model.njnt):
        joint_name = mujoco.mj_id2name(target_model, mujoco.mjtObj.mjOBJ_JOINT, target_joint_id)
        if not joint_name:
            continue
        source_joint_id = mujoco.mj_name2id(source_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if source_joint_id < 0:
            raise TrajectoryGoalProviderError(f"source model is missing joint {joint_name!r}")
        qpos_width = _joint_qpos_width(target_model, target_joint_id)
        qvel_width = _joint_dof_width(target_model, target_joint_id)
        if qpos_width != _joint_qpos_width(source_model, source_joint_id):
            raise TrajectoryGoalProviderError(f"qpos width mismatch for joint {joint_name!r}")
        if qvel_width != _joint_dof_width(source_model, source_joint_id):
            raise TrajectoryGoalProviderError(f"qvel width mismatch for joint {joint_name!r}")
        qpos.append((int(source_model.jnt_qposadr[source_joint_id]), int(target_model.jnt_qposadr[target_joint_id]), qpos_width))
        qvel.append((int(source_model.jnt_dofadr[source_joint_id]), int(target_model.jnt_dofadr[target_joint_id]), qvel_width))
    return _StateCopyMap(qpos=tuple(qpos), qvel=tuple(qvel))


def _joint_qpos_width(model: mujoco.MjModel, joint_id: int) -> int:
    joint_type = int(model.jnt_type[joint_id])
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return 4
    return 1


def _joint_dof_width(model: mujoco.MjModel, joint_id: int) -> int:
    joint_type = int(model.jnt_type[joint_id])
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return 6
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return 3
    return 1


def _checkpoint_metadata(checkpoint: Path) -> dict[str, Any]:
    metadata_path = checkpoint / "config" / "metadata"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"checkpoint config metadata not found: {metadata_path}")
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"checkpoint config metadata root must be an object: {metadata_path}")
    return data


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]

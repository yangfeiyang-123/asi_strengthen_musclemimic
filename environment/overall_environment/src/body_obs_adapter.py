from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


class BodyObsCompatibilityError(ValueError):
    pass


@dataclass(frozen=True)
class BodyObsCompatibilityReport:
    checkpoint_obs_size: int
    overall_obs_size: int
    compatible: bool
    reason: str


@dataclass(frozen=True)
class BodyObsSchema:
    total_size: int
    kinematic_size: int
    muscle_size: int
    touch_size: int
    goal_size: int
    action_size: int
    root_joint_name: str
    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    touch_sensor_names: tuple[str, ...]
    observation_names: tuple[str, ...]
    student_filtered: bool = False


@dataclass(frozen=True)
class BodyObsAdapter:
    expected_obs_size: int
    schema: BodyObsSchema | None = None

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path) -> "BodyObsAdapter":
        checkpoint_path = Path(checkpoint)
        obs_size = checkpoint_obs_size(checkpoint_path)
        if obs_size <= 0:
            raise BodyObsCompatibilityError(f"checkpoint has no valid RunningMeanStd obs size: {checkpoint}")
        schema = reconstruct_body_obs_schema(checkpoint_path)
        if schema.total_size != obs_size:
            raise BodyObsCompatibilityError(
                f"checkpoint normalizer obs size {obs_size} does not match reconstructed schema {schema.total_size}"
            )
        return cls(expected_obs_size=obs_size, schema=schema)

    def check_compatibility(self, *, overall_obs_size: int) -> BodyObsCompatibilityReport:
        overall_size = int(overall_obs_size)
        if overall_size == self.expected_obs_size:
            return BodyObsCompatibilityReport(
                checkpoint_obs_size=self.expected_obs_size,
                overall_obs_size=overall_size,
                compatible=True,
                reason="overall observation size matches checkpoint observation size",
            )
        return BodyObsCompatibilityReport(
            checkpoint_obs_size=self.expected_obs_size,
            overall_obs_size=overall_size,
            compatible=False,
            reason=(
                f"overall observation size {overall_size} does not match checkpoint observation size "
                f"{self.expected_obs_size}; an explicit body observation schema mapping is required"
            ),
        )

    def adapt(self, overall_obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(overall_obs, dtype=float)
        if obs.shape != (self.expected_obs_size,):
            raise BodyObsCompatibilityError(
                f"body observation adapter expected {self.expected_obs_size} values, got {obs.shape}"
            )
        if not np.isfinite(obs).all():
            raise BodyObsCompatibilityError("body observation contains non-finite values")
        return obs.copy()

    def build_from_mujoco(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        goal_obs: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.schema is None:
            raise BodyObsCompatibilityError("body observation schema is required to build from MuJoCo state")
        goal = _validate_goal_obs(goal_obs, self.schema.goal_size)
        mujoco.mj_forward(model, data)

        obs_parts = [
            _free_joint_pos_no_xy(model, data, self.schema.root_joint_name),
            _joint_pos_array(model, data, self.schema.joint_names),
            _free_joint_vel(model, data, self.schema.root_joint_name),
            _joint_vel_array(model, data, self.schema.joint_names),
            _muscle_observations(model, data, self.schema.actuator_names),
            _touch_observations(model, data, self.schema.touch_sensor_names),
            goal,
        ]
        obs = np.concatenate(obs_parts).astype(float, copy=False)
        if obs.shape != (self.schema.total_size,):
            raise BodyObsCompatibilityError(
                f"built body observation has shape {obs.shape}, expected ({self.schema.total_size},)"
            )
        if not np.isfinite(obs).all():
            raise BodyObsCompatibilityError("built body observation contains non-finite values")
        return obs


def checkpoint_obs_size(checkpoint: str | Path) -> int:
    return _array_shape_size(checkpoint, "run_stats.RunningMeanStd_0.mean")


def checkpoint_normalizer_shapes(checkpoint: str | Path) -> dict[str, tuple[int, ...]]:
    return {
        "mean": _array_shape(checkpoint, "run_stats.RunningMeanStd_0.mean"),
        "var": _array_shape(checkpoint, "run_stats.RunningMeanStd_0.var"),
        "count": _array_shape(checkpoint, "run_stats.RunningMeanStd_0.count"),
    }


def _array_shape_size(checkpoint: str | Path, param_name: str) -> int:
    shape = _array_shape(checkpoint, param_name)
    if len(shape) != 1:
        return 0
    return int(shape[0])


def _array_shape(checkpoint: str | Path, param_name: str) -> tuple[int, ...]:
    metadata = _train_state_metadata(checkpoint)
    tree_metadata = metadata.get("tree_metadata", {})
    legacy_key = tuple(param_name.split("."))
    for key, value in tree_metadata.items():
        normalized = str(key).replace("'", "").replace("(", "").replace(")", "").replace(", ", ".")
        if normalized == ".".join(legacy_key):
            shape = value.get("value_metadata", {}).get("write_shape", [])
            return tuple(int(dim) for dim in shape)
    for item in metadata.get("array_metadatas", []):
        array_metadata = item.get("array_metadata", {})
        if array_metadata.get("param_name") == param_name:
            shape = array_metadata.get("write_shape", [])
            return tuple(int(dim) for dim in shape)
    return ()


def _train_state_metadata(checkpoint: str | Path) -> dict:
    metadata_path = Path(checkpoint) / "train_state" / "_METADATA"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"checkpoint train_state metadata not found: {metadata_path}")
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"checkpoint metadata root must be an object: {metadata_path}")
    return data


def reconstruct_body_obs_schema(checkpoint: str | Path) -> BodyObsSchema:
    checkpoint_path = Path(checkpoint)
    experiment = _checkpoint_experiment(checkpoint_path)
    env_params = dict(experiment["env_params"])
    env = _legacy_body_env_from_params(env_params)
    model = getattr(env, "_model", None) or getattr(env, "model", None)
    if model is None:
        raise BodyObsCompatibilityError("could not access legacy body MuJoCo model")

    root_joint_name = str(getattr(env, "root_free_joint_xml_name", "root"))
    joint_names = tuple(
        name
        for name in (_mj_name(model, mujoco.mjtObj.mjOBJ_JOINT, index) for index in range(model.njnt))
        if name and name != root_joint_name
    )
    actuator_names = tuple(
        name
        for name in (_mj_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) for index in range(model.nu))
        if name
    )
    observation_items = list(env.obs_container.items())
    observation_names = tuple(str(name) for name, _ in observation_items)
    dims_by_name = {str(name): int(obs.dim) for name, obs in observation_items}

    kinematic_names = ("q_free_joint", "q_all_pos", "dq_free_joint", "dq_all_vel")
    kinematic_size = sum(dims_by_name[name] for name in kinematic_names)
    muscle_size = sum(dim for name, dim in dims_by_name.items() if name.startswith("muscle_"))
    touch_names = tuple(name.removeprefix("touch_") for name in observation_names if name.startswith("touch_"))
    touch_size = sum(dims_by_name[f"touch_{name}"] for name in touch_names)
    goal_size = dims_by_name.get("GoalTrajMimic", 0)
    total_size = sum(int(obs.dim) for _, obs in observation_items)
    student_filtered = False

    filter_cfg = dict(experiment.get("student_obs_filter", {}) or {})
    if bool(filter_cfg.get("enabled", False)):
        # Frozen direct students use the same v1 state+phase projection as the
        # rollout collector.  Reconstructing the teacher's complete lookahead
        # here would silently build a 2418-D input for a 1950-D checkpoint.
        from musclemimic.distill.obs_filter import build_student_obs_indices

        spec = build_student_obs_indices(env, filter_cfg)
        condition_groups = tuple(filter_cfg.get("condition_groups", ()) or ())
        if condition_groups:
            raise BodyObsCompatibilityError(
                "BodyObsAdapter cannot synthesize student condition_groups from a MuJoCo state; "
                "provide a dedicated observation builder"
            )
        base_size = kinematic_size + muscle_size + touch_size
        if int(spec.state_indices.size) != base_size:
            raise BodyObsCompatibilityError(
                "student observation state projection contains unsupported non-body fields: "
                f"projected_state={int(spec.state_indices.size)} reconstructable_body={base_size}"
            )
        goal_size = 1 if spec.phase_student_index is not None else 0
        total_size = int(spec.student_obs_dim)
        observation_names = tuple(name for name in observation_names if name != "GoalTrajMimic")
        if goal_size:
            observation_names = observation_names + ("motion_phase",)
        student_filtered = True

    return BodyObsSchema(
        total_size=total_size,
        kinematic_size=kinematic_size,
        muscle_size=muscle_size,
        touch_size=touch_size,
        goal_size=goal_size,
        action_size=len(actuator_names),
        root_joint_name=root_joint_name,
        joint_names=joint_names,
        actuator_names=actuator_names,
        touch_sensor_names=touch_names,
        observation_names=observation_names,
        student_filtered=student_filtered,
    )


def _checkpoint_env_params(checkpoint: Path) -> dict:
    return dict(_checkpoint_experiment(checkpoint)["env_params"])


def _checkpoint_experiment(checkpoint: Path) -> dict:
    metadata_path = checkpoint / "config" / "metadata"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"checkpoint config metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return dict(metadata["experiment"])


def _legacy_body_env_from_params(env_params: dict) -> object:
    env_name = str(env_params.get("env_name", "MyoFullBody"))
    if env_name in {"MyoFullBodyRacket", "MjxMyoFullBodyRacket"}:
        from musclemimic.environments.humanoids.myofullbody_racket import MyoFullBodyRacket as EnvClass
    elif env_name in {"MyoFullBody", "MjxMyoFullBody"}:
        from musclemimic.environments.humanoids.myofullbody import MyoFullBody as EnvClass
    else:
        raise BodyObsCompatibilityError(f"unsupported checkpoint env_name for body observation schema: {env_name}")

    goal_params = dict(env_params.get("goal_params", {}))
    return EnvClass(
        headless=True,
        disable_fingers=bool(env_params.get("disable_fingers", True)),
        enable_muscle_length_observations=bool(env_params.get("enable_muscle_length_observations", False)),
        enable_muscle_velocity_observations=bool(env_params.get("enable_muscle_velocity_observations", False)),
        enable_muscle_force_observations=bool(env_params.get("enable_muscle_force_observations", False)),
        enable_muscle_excitation_observations=bool(env_params.get("enable_muscle_excitation_observations", False)),
        enable_muscle_activation_observations=bool(env_params.get("enable_muscle_activation_observations", False)),
        enable_touch_sensor_observations=bool(env_params.get("enable_touch_sensor_observations", True)),
        goal_type=str(env_params.get("goal_type", "GoalTrajMimic")),
        goal_params=goal_params,
        mjx_backend=str(env_params.get("mjx_backend", "jax")),
    )


def _validate_goal_obs(goal_obs: np.ndarray | None, goal_size: int) -> np.ndarray:
    if int(goal_size) == 0 and goal_obs is None:
        return np.zeros((0,), dtype=float)
    if goal_obs is None:
        raise BodyObsCompatibilityError("goal_obs is required to build checkpoint-compatible body observations")
    goal = np.asarray(goal_obs, dtype=float)
    if goal.shape != (goal_size,):
        raise BodyObsCompatibilityError(f"goal_obs must have shape ({goal_size},), got {goal.shape}")
    if not np.isfinite(goal).all():
        raise BodyObsCompatibilityError("goal_obs contains non-finite values")
    return goal


def _free_joint_pos_no_xy(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str) -> np.ndarray:
    qadr = _joint_qpos_adr(model, joint_name)
    return np.asarray(data.qpos[qadr + 2 : qadr + 7], dtype=float)


def _free_joint_vel(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str) -> np.ndarray:
    dadr = _joint_dof_adr(model, joint_name)
    return np.asarray(data.qvel[dadr : dadr + 6], dtype=float)


def _joint_pos_array(model: mujoco.MjModel, data: mujoco.MjData, joint_names: tuple[str, ...]) -> np.ndarray:
    values = []
    for joint_name in joint_names:
        joint_id = _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qadr = int(model.jnt_qposadr[joint_id])
        width = _joint_qpos_width(model, joint_id)
        values.append(np.asarray(data.qpos[qadr : qadr + width], dtype=float))
    return np.concatenate(values) if values else np.zeros(0, dtype=float)


def _joint_vel_array(model: mujoco.MjModel, data: mujoco.MjData, joint_names: tuple[str, ...]) -> np.ndarray:
    values = []
    for joint_name in joint_names:
        joint_id = _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        dadr = int(model.jnt_dofadr[joint_id])
        width = _joint_dof_width(model, joint_id)
        values.append(np.asarray(data.qvel[dadr : dadr + width], dtype=float))
    return np.concatenate(values) if values else np.zeros(0, dtype=float)


def _muscle_observations(model: mujoco.MjModel, data: mujoco.MjData, actuator_names: tuple[str, ...]) -> np.ndarray:
    values = []
    for actuator_name in actuator_names:
        actuator_id = _named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        values.extend(
            [
                float(data.actuator_length[actuator_id]),
                float(data.actuator_velocity[actuator_id]),
                float(data.actuator_force[actuator_id]),
                float(data.ctrl[actuator_id]),
                float(data.act[actuator_id]),
            ]
        )
    return np.asarray(values, dtype=float)


def _touch_observations(model: mujoco.MjModel, data: mujoco.MjData, sensor_names: tuple[str, ...]) -> np.ndarray:
    values = []
    for sensor_name in sensor_names:
        sensor_id = _named_id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        sensor_adr = int(model.sensor_adr[sensor_id])
        values.append(float(data.sensordata[sensor_adr]))
    return np.asarray(values, dtype=float)


def _joint_qpos_adr(model: mujoco.MjModel, joint_name: str) -> int:
    joint_id = _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return int(model.jnt_qposadr[joint_id])


def _joint_dof_adr(model: mujoco.MjModel, joint_name: str) -> int:
    joint_id = _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return int(model.jnt_dofadr[joint_id])


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


def _named_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise BodyObsCompatibilityError(f"overall model is missing {obj_type.name} {name!r}")
    return int(obj_id)


def _mj_name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, index: int) -> str | None:
    return mujoco.mj_id2name(model, obj_type, index)

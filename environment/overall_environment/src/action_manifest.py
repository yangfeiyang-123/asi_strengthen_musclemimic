from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ActionManifest:
    schema_version: int
    env_name: str
    disable_fingers: bool
    action_size: int
    actuator_names: list[str]
    obs_size: int
    obs_fields: list[str]
    control_min: float
    control_max: float

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported action manifest schema_version: {self.schema_version}")
        if self.action_size != len(self.actuator_names):
            raise ValueError("action_size must match actuator_names length")
        if len(set(self.actuator_names)) != len(self.actuator_names):
            raise ValueError("actuator_names contains duplicates")
        if self.obs_size < 0:
            raise ValueError("obs_size must be non-negative")

    @classmethod
    def from_env_params(
        cls,
        env_params: dict[str, Any],
        *,
        actuator_names: list[str],
        obs_size: int,
        obs_fields: list[str],
    ) -> "ActionManifest":
        return cls(
            schema_version=1,
            env_name=str(env_params["env_name"]),
            disable_fingers=bool(env_params.get("disable_fingers", True)),
            action_size=len(actuator_names),
            actuator_names=list(actuator_names),
            obs_size=int(obs_size),
            obs_fields=list(obs_fields),
            control_min=-1.0,
            control_max=1.0,
        )


def _coerce(data: dict[str, Any]) -> ActionManifest:
    return ActionManifest(
        schema_version=int(data["schema_version"]),
        env_name=str(data["env_name"]),
        disable_fingers=bool(data["disable_fingers"]),
        action_size=int(data["action_size"]),
        actuator_names=[str(name) for name in data["actuator_names"]],
        obs_size=int(data.get("obs_size", 0)),
        obs_fields=[str(name) for name in data.get("obs_fields", [])],
        control_min=float(data.get("control_min", -1.0)),
        control_max=float(data.get("control_max", 1.0)),
    )


def load_action_manifest(path: str | Path) -> ActionManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("action manifest root must be an object")
    return _coerce(data)


def write_action_manifest(path: str | Path, manifest: ActionManifest) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def find_action_manifest(checkpoint: str | Path) -> Path:
    root = Path(checkpoint)
    candidates = [
        root / "action_manifest.json",
        root.parent / "action_manifest.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no action_manifest.json found for checkpoint: {checkpoint}")


def reconstruct_action_manifest(checkpoint: str | Path) -> ActionManifest:
    checkpoint_path = Path(checkpoint)
    metadata_path = checkpoint_path / "config" / "metadata"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"checkpoint config metadata not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    env_params = metadata["experiment"]["env_params"]
    actuator_names = _actuator_names_from_env_params(env_params)
    obs_size = _obs_size_from_orbax_metadata(checkpoint_path)
    return ActionManifest.from_env_params(
        env_params,
        actuator_names=actuator_names,
        obs_size=obs_size,
        obs_fields=[],
    )


def _actuator_names_from_env_params(env_params: dict[str, Any]) -> list[str]:
    env_name = str(env_params.get("env_name", ""))
    if env_name not in {"MyoFullBody", "MjxMyoFullBody"}:
        raise ValueError(f"unsupported env_name for action manifest reconstruction: {env_name}")

    import mujoco
    from musclemimic.environments.humanoids.myofullbody import MyoFullBody

    env = MyoFullBody(
        disable_fingers=bool(env_params.get("disable_fingers", True)),
        enable_muscle_length_observations=bool(env_params.get("enable_muscle_length_observations", False)),
        enable_muscle_velocity_observations=bool(env_params.get("enable_muscle_velocity_observations", False)),
        enable_muscle_force_observations=bool(env_params.get("enable_muscle_force_observations", False)),
        enable_muscle_excitation_observations=bool(env_params.get("enable_muscle_excitation_observations", False)),
        enable_muscle_activation_observations=bool(env_params.get("enable_muscle_activation_observations", False)),
        enable_touch_sensor_observations=bool(env_params.get("enable_touch_sensor_observations", True)),
        mjx_backend=str(env_params.get("mjx_backend", "jax")),
        num_envs=int(env_params.get("num_envs", 1)),
    )
    model = getattr(env, "_model", None) or getattr(env, "model", None)
    if model is None:
        raise ValueError("could not access MuJoCo model from reconstructed environment")
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) for index in range(model.nu)]


def _obs_size_from_orbax_metadata(checkpoint_path: Path) -> int:
    metadata_path = checkpoint_path / "train_state" / "_METADATA"
    if not metadata_path.is_file():
        return 0
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    tree_metadata = data.get("tree_metadata", {})
    value = tree_metadata.get("('run_stats', 'RunningMeanStd_0', 'mean')")
    if not value:
        return 0
    shape = value.get("value_metadata", {}).get("write_shape", [])
    if len(shape) != 1:
        return 0
    return int(shape[0])


def main() -> int:
    parser = argparse.ArgumentParser(description="Print checkpoint action manifest.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--print", dest="do_print", action="store_true")
    parser.add_argument("--reconstruct", action="store_true")
    args = parser.parse_args()

    if args.reconstruct:
        manifest = reconstruct_action_manifest(args.checkpoint)
    else:
        manifest = load_action_manifest(find_action_manifest(args.checkpoint))
    if args.do_print:
        print(json.dumps(asdict(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

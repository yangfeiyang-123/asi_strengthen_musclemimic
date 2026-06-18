from __future__ import annotations

import json
import ast
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from environment.overall_environment.src.action_manifest import reconstruct_action_manifest, write_action_manifest
from environment.overall_environment.src.body_obs_adapter import (
    BodyObsAdapter,
    BodyObsCompatibilityReport,
    reconstruct_body_obs_schema,
    checkpoint_normalizer_shapes,
)


class FrozenBodyPolicyLoadError(RuntimeError):
    pass


class FrozenBodyPolicyArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class ActorCheckpointSpec:
    obs_size: int
    action_size: int
    actor_hidden_layers: tuple[int, ...]
    critic_hidden_layers: tuple[int, ...]
    activation: str
    init_std: float
    learnable_std: bool
    use_layernorm: bool
    layernorm_eps: float


@dataclass(frozen=True)
class ActorCheckpointShapeReport:
    valid: bool
    actor_output_kernel_shape: tuple[int, ...]
    log_std_shape: tuple[int, ...]
    run_stats_mean_shape: tuple[int, ...]
    reason: str


@dataclass(frozen=True)
class FrozenBodyPolicyManifest:
    schema_version: int
    source_checkpoint: str
    tensor_format: str
    has_tensors: bool
    actor_spec: ActorCheckpointSpec
    shape_report: ActorCheckpointShapeReport
    params_file: str
    run_stats_file: str
    body_obs_schema_file: str
    action_manifest_file: str


@dataclass(frozen=True)
class FrozenBodyPolicy:
    checkpoint: Path
    obs_adapter: BodyObsAdapter
    action_size: int
    obs_report: BodyObsCompatibilityReport
    actor_spec: ActorCheckpointSpec | None = None
    params: dict[str, Any] | None = None
    run_stats: dict[str, Any] | None = None

    @classmethod
    def load(cls, checkpoint: str | Path, *, overall_obs_size: int) -> "FrozenBodyPolicy":
        checkpoint_path = Path(checkpoint)
        obs_adapter = BodyObsAdapter.from_checkpoint(checkpoint_path)
        obs_report = obs_adapter.check_compatibility(overall_obs_size=overall_obs_size)
        if not obs_report.compatible and obs_adapter.schema is None:
            raise FrozenBodyPolicyLoadError(f"observation compatibility failed: {obs_report.reason}")

        normalizer_shapes = checkpoint_normalizer_shapes(checkpoint_path)
        expected = (obs_adapter.expected_obs_size,)
        if normalizer_shapes["mean"] != expected or normalizer_shapes["var"] != expected:
            raise FrozenBodyPolicyLoadError(
                f"normalizer shape mismatch: {normalizer_shapes}, expected mean/var {expected}"
            )
        if normalizer_shapes["count"] != ():
            raise FrozenBodyPolicyLoadError(f"normalizer count must be scalar, got {normalizer_shapes['count']}")

        manifest = reconstruct_action_manifest(checkpoint_path)
        shape_report = validate_actor_checkpoint_shapes(checkpoint_path)
        if not shape_report.valid:
            raise FrozenBodyPolicyLoadError(shape_report.reason)
        raise FrozenBodyPolicyLoadError(
            "checkpoint actor restore is not implemented yet; refusing to create a random body policy "
            f"for action_size={manifest.action_size}"
        )

    @classmethod
    def load_from_export(cls, artifact_dir: str | Path) -> "FrozenBodyPolicy":
        root = Path(artifact_dir)
        manifest = load_frozen_body_policy_manifest(root)
        if not manifest.has_tensors:
            raise FrozenBodyPolicyArtifactError(f"artifact does not contain exported tensors: {root}")
        params_path = root / manifest.params_file
        run_stats_path = root / manifest.run_stats_file
        if not params_path.is_file() or not run_stats_path.is_file():
            raise FrozenBodyPolicyArtifactError(f"artifact tensor files are missing: {root}")
        params = _load_npz_tree(params_path)
        run_stats = _load_npz_tree(run_stats_path)
        obs_adapter = BodyObsAdapter(expected_obs_size=manifest.actor_spec.obs_size)
        obs_report = obs_adapter.check_compatibility(overall_obs_size=manifest.actor_spec.obs_size)
        return cls(
            checkpoint=root,
            obs_adapter=obs_adapter,
            action_size=manifest.actor_spec.action_size,
            obs_report=obs_report,
            actor_spec=manifest.actor_spec,
            params=params,
            run_stats=run_stats,
        )

    @classmethod
    def write_synthetic_export(cls, artifact_dir: str | Path, spec: ActorCheckpointSpec) -> FrozenBodyPolicyManifest:
        import jax
        import jax.numpy as jnp

        root = Path(artifact_dir)
        root.mkdir(parents=True, exist_ok=True)
        network = _network_from_spec(spec)
        variables = network.init(jax.random.key(0), jnp.zeros((spec.obs_size,), dtype=jnp.float32))
        shape_report = ActorCheckpointShapeReport(
            valid=True,
            actor_output_kernel_shape=(spec.actor_hidden_layers[-1], spec.action_size),
            log_std_shape=(spec.action_size,),
            run_stats_mean_shape=(spec.obs_size,),
            reason="synthetic actor checkpoint shapes match spec",
        )
        manifest = FrozenBodyPolicyManifest(
            schema_version=1,
            source_checkpoint="synthetic",
            tensor_format="npz",
            has_tensors=True,
            actor_spec=spec,
            shape_report=shape_report,
            params_file="params.npz",
            run_stats_file="run_stats.npz",
            body_obs_schema_file="body_obs_schema.json",
            action_manifest_file="action_manifest.json",
        )
        _save_npz_tree(root / manifest.params_file, _actor_only_params(variables["params"]))
        _save_npz_tree(root / manifest.run_stats_file, variables["run_stats"])
        _write_json(root / "manifest.json", _manifest_to_dict(manifest))
        _write_json(root / manifest.body_obs_schema_file, {"synthetic": True, "obs_size": spec.obs_size})
        _write_json(root / manifest.action_manifest_file, {"synthetic": True, "action_size": spec.action_size})
        return manifest

    def act(self, body_obs: np.ndarray) -> np.ndarray:
        if self.actor_spec is None or self.params is None or self.run_stats is None:
            raise FrozenBodyPolicyArtifactError("frozen body policy has no exported actor tensors")
        obs = np.asarray(body_obs, dtype=np.float32)
        if obs.shape != (self.actor_spec.obs_size,):
            raise FrozenBodyPolicyArtifactError(f"body_obs must have shape ({self.actor_spec.obs_size},), got {obs.shape}")
        if not np.isfinite(obs).all():
            raise FrozenBodyPolicyArtifactError("body_obs contains non-finite values")

        action = _actor_mean_numpy(self.actor_spec, self.params, self.run_stats, obs)
        if action.shape != (self.actor_spec.action_size,):
            raise FrozenBodyPolicyArtifactError(
                f"policy action has shape {action.shape}, expected ({self.actor_spec.action_size},)"
            )
        if not np.isfinite(action).all():
            raise FrozenBodyPolicyArtifactError("policy action contains non-finite values")
        return action


def reconstruct_actor_checkpoint_spec(checkpoint: str | Path) -> ActorCheckpointSpec:
    checkpoint_path = Path(checkpoint)
    metadata = _checkpoint_metadata(checkpoint_path)
    experiment = metadata["experiment"]
    ppo_config = experiment.get("ppo_config", {})
    obs_size = checkpoint_normalizer_shapes(checkpoint_path)["mean"][0]
    action_size = reconstruct_action_manifest(checkpoint_path).action_size
    init_std = float(experiment.get("init_std", ppo_config.get("init_std", 1.0)))
    learnable_std = bool(experiment.get("learnable_std", ppo_config.get("learnable_std", True)))
    return ActorCheckpointSpec(
        obs_size=int(obs_size),
        action_size=int(action_size),
        actor_hidden_layers=tuple(int(v) for v in experiment["actor_hidden_layers"]),
        critic_hidden_layers=tuple(int(v) for v in experiment["critic_hidden_layers"]),
        activation=str(experiment["activation"]),
        init_std=init_std,
        learnable_std=learnable_std,
        use_layernorm=bool(experiment.get("use_layernorm", False)),
        layernorm_eps=float(experiment.get("layernorm_eps", 1e-5)),
    )


def validate_actor_checkpoint_shapes(checkpoint: str | Path) -> ActorCheckpointShapeReport:
    checkpoint_path = Path(checkpoint)
    spec = reconstruct_actor_checkpoint_spec(checkpoint_path)
    tree_metadata = _train_state_tree_metadata(checkpoint_path)
    last_actor_layer = len(spec.actor_hidden_layers)
    previous_width = spec.actor_hidden_layers[-1]
    actor_output_kernel_shape = _metadata_shape(
        tree_metadata,
        ("params", "actor", f"Dense_{last_actor_layer}", "kernel"),
    )
    log_std_shape = _metadata_shape(tree_metadata, ("params", "log_std"))
    run_stats_mean_shape = _metadata_shape(tree_metadata, ("run_stats", "RunningMeanStd_0", "mean"))
    expected_kernel = (previous_width, spec.action_size)
    expected_log_std = (spec.action_size,)
    expected_obs = (spec.obs_size,)
    failures = []
    if actor_output_kernel_shape != expected_kernel:
        failures.append(f"actor output kernel {actor_output_kernel_shape} != {expected_kernel}")
    if log_std_shape != expected_log_std:
        failures.append(f"log_std {log_std_shape} != {expected_log_std}")
    if run_stats_mean_shape != expected_obs:
        failures.append(f"run_stats mean {run_stats_mean_shape} != {expected_obs}")
    if failures:
        return ActorCheckpointShapeReport(
            valid=False,
            actor_output_kernel_shape=actor_output_kernel_shape,
            log_std_shape=log_std_shape,
            run_stats_mean_shape=run_stats_mean_shape,
            reason="; ".join(failures),
        )
    return ActorCheckpointShapeReport(
        valid=True,
        actor_output_kernel_shape=actor_output_kernel_shape,
        log_std_shape=log_std_shape,
        run_stats_mean_shape=run_stats_mean_shape,
        reason="actor checkpoint shapes match reconstructed spec",
    )


def export_frozen_body_policy(
    checkpoint: str | Path,
    artifact_dir: str | Path,
    *,
    restore_tensors: bool = True,
) -> FrozenBodyPolicyManifest:
    checkpoint_path = Path(checkpoint).resolve()
    root = Path(artifact_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    actor_spec = reconstruct_actor_checkpoint_spec(checkpoint_path)
    shape_report = validate_actor_checkpoint_shapes(checkpoint_path)
    if not shape_report.valid:
        raise FrozenBodyPolicyArtifactError(shape_report.reason)

    action_manifest = reconstruct_action_manifest(checkpoint_path)
    body_schema = reconstruct_body_obs_schema(checkpoint_path)
    write_action_manifest(root / "action_manifest.json", action_manifest)
    _write_json(root / "body_obs_schema.json", asdict(body_schema))

    has_tensors = False
    if restore_tensors:
        params, run_stats = _restore_checkpoint_policy_tensors(checkpoint_path)
        _save_npz_tree(root / "params.npz", params)
        _save_npz_tree(root / "run_stats.npz", run_stats)
        has_tensors = True

    manifest = FrozenBodyPolicyManifest(
        schema_version=1,
        source_checkpoint=str(checkpoint_path),
        tensor_format="npz",
        has_tensors=has_tensors,
        actor_spec=actor_spec,
        shape_report=shape_report,
        params_file="params.npz",
        run_stats_file="run_stats.npz",
        body_obs_schema_file="body_obs_schema.json",
        action_manifest_file="action_manifest.json",
    )
    _write_json(root / "manifest.json", _manifest_to_dict(manifest))
    return manifest


def load_frozen_body_policy_manifest(artifact_dir: str | Path) -> FrozenBodyPolicyManifest:
    manifest_path = Path(artifact_dir) / "manifest.json"
    if not manifest_path.is_file():
        raise FrozenBodyPolicyArtifactError(f"frozen body policy manifest not found: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return _manifest_from_dict(data)


def _checkpoint_metadata(checkpoint: Path) -> dict:
    metadata_path = checkpoint / "config" / "metadata"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"checkpoint config metadata not found: {metadata_path}")
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"checkpoint config metadata root must be an object: {metadata_path}")
    return data


def _train_state_tree_metadata(checkpoint: Path) -> dict:
    metadata_path = checkpoint / "train_state" / "_METADATA"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"checkpoint train_state metadata not found: {metadata_path}")
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    tree_metadata = data.get("tree_metadata", {})
    if not isinstance(tree_metadata, dict):
        raise ValueError(f"checkpoint tree_metadata must be an object: {metadata_path}")
    return tree_metadata


def _metadata_shape(tree_metadata: dict, key: tuple[str, ...]) -> tuple[int, ...]:
    value = tree_metadata.get(str(key))
    if not value:
        return ()
    shape = value.get("value_metadata", {}).get("write_shape", [])
    return tuple(int(dim) for dim in shape)


def _network_from_spec(spec: ActorCheckpointSpec):
    import jax.numpy as jnp
    from musclemimic.algorithms.common.networks import ActorCritic

    return ActorCritic(
        spec.action_size,
        activation=spec.activation,
        init_std=spec.init_std,
        learnable_std=spec.learnable_std,
        hidden_layer_dims=spec.actor_hidden_layers,
        critic_hidden_layer_dims=spec.critic_hidden_layers
        if spec.critic_hidden_layers != spec.actor_hidden_layers
        else None,
        actor_obs_ind=jnp.arange(spec.obs_size),
        critic_obs_ind=jnp.arange(spec.obs_size),
        use_layernorm=spec.use_layernorm,
        layernorm_eps=spec.layernorm_eps,
    )


def _restore_checkpoint_policy_tensors(checkpoint: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from jax.sharding import SingleDeviceSharding

    checkpoint = checkpoint.resolve()
    target = _checkpoint_policy_tensor_target(checkpoint)
    restore_sharding = SingleDeviceSharding(jax.devices()[0])
    restore_args = jax.tree.map(
        lambda _: ocp.type_handlers.ArrayRestoreArgs(sharding=restore_sharding),
        target,
    )
    manager = ocp.CheckpointManager(
        str(checkpoint.parent),
        options=ocp.CheckpointManagerOptions(step_prefix="checkpoint"),
        item_names=("train_state", "config", "metadata"),
    )
    try:
        result = manager.restore(
            _checkpoint_step(checkpoint),
            args=ocp.args.Composite(
                train_state=ocp.args.PyTreeRestore(
                    item=target,
                    restore_args=restore_args,
                    partial_restore=True,
                )
            ),
        )
    finally:
        manager.close()
    train_state_data = result.train_state
    return train_state_data["params"], train_state_data["run_stats"]


def _checkpoint_policy_tensor_target(checkpoint: Path) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    tree_metadata = _train_state_tree_metadata(checkpoint)
    target: dict[str, Any] = {"params": {}, "run_stats": {}}
    for raw_key, value in tree_metadata.items():
        key = ast.literal_eval(raw_key)
        if not isinstance(key, tuple):
            continue
        if _is_actor_policy_key(key):
            shape = tuple(int(dim) for dim in value.get("value_metadata", {}).get("write_shape", []))
            _set_nested(target, key, jax.ShapeDtypeStruct(shape, jnp.float32))
    return target


def _is_actor_policy_key(key: tuple[str, ...]) -> bool:
    return (
        len(key) >= 2
        and (
            key[:2] == ("params", "actor")
            or key == ("params", "log_std")
            or key[0] == "run_stats"
        )
    )


def _set_nested(root: dict[str, Any], key: tuple[str, ...], value: Any) -> None:
    cursor = root
    for part in key[:-1]:
        cursor = cursor.setdefault(str(part), {})
    cursor[str(key[-1])] = value


def _checkpoint_step(checkpoint: Path) -> int:
    name = checkpoint.name
    if name.startswith("checkpoint_"):
        return int(name.split("_")[-1])
    if name.isdigit():
        return int(name)
    raise FrozenBodyPolicyArtifactError(f"cannot infer checkpoint step from path: {checkpoint}")


def _actor_only_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "actor": params["actor"],
        "log_std": params["log_std"],
    }


def _actor_mean_numpy(
    spec: ActorCheckpointSpec,
    params: dict[str, Any],
    run_stats: dict[str, Any],
    obs: np.ndarray,
) -> np.ndarray:
    actor = params["actor"]
    stats = run_stats["RunningMeanStd_0"]
    x = (obs - np.asarray(stats["mean"], dtype=np.float32)) / np.sqrt(
        np.asarray(stats["var"], dtype=np.float32) + 1e-8
    )
    activation = _activation_fn(spec.activation)
    for index, _width in enumerate(spec.actor_hidden_layers):
        dense = actor[f"Dense_{index}"]
        x = x @ np.asarray(dense["kernel"], dtype=np.float32) + np.asarray(dense["bias"], dtype=np.float32)
        if spec.use_layernorm:
            layernorm = actor[f"LayerNorm_{index}"]
            x = _layer_norm(
                x,
                np.asarray(layernorm["scale"], dtype=np.float32),
                np.asarray(layernorm["bias"], dtype=np.float32),
                spec.layernorm_eps,
            )
        x = activation(x)
    output = actor[f"Dense_{len(spec.actor_hidden_layers)}"]
    return x @ np.asarray(output["kernel"], dtype=np.float32) + np.asarray(output["bias"], dtype=np.float32)


def restore_flax_actor_mean_for_verification(policy: FrozenBodyPolicy, obs: np.ndarray) -> np.ndarray:
    if policy.actor_spec is None or policy.params is None or policy.run_stats is None:
        raise FrozenBodyPolicyArtifactError("frozen body policy has no exported actor tensors")
    obs_array = np.asarray(obs, dtype=np.float32)
    if obs_array.shape != (policy.actor_spec.obs_size,):
        raise FrozenBodyPolicyArtifactError(f"obs must have shape ({policy.actor_spec.obs_size},), got {obs_array.shape}")

    import jax.numpy as jnp
    from musclemimic.algorithms.common.networks import FullyConnectedNet

    stats = policy.run_stats["RunningMeanStd_0"]
    normalized_obs = (obs_array - np.asarray(stats["mean"], dtype=np.float32)) / np.sqrt(
        np.asarray(stats["var"], dtype=np.float32) + 1e-8
    )
    actor = FullyConnectedNet(
        hidden_layer_dims=policy.actor_spec.actor_hidden_layers,
        output_dim=policy.actor_spec.action_size,
        activation=policy.actor_spec.activation,
        output_activation=None,
        use_running_mean_stand=False,
        squeeze_output=False,
        use_layernorm=policy.actor_spec.use_layernorm,
        layernorm_eps=policy.actor_spec.layernorm_eps,
    )
    action = actor.apply({"params": policy.params["actor"]}, jnp.asarray(normalized_obs, dtype=jnp.float32))
    return np.asarray(action, dtype=np.float32).reshape(policy.actor_spec.action_size)


def _layer_norm(x: np.ndarray, scale: np.ndarray, bias: np.ndarray, eps: float) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean(np.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * scale + bias


def _activation_fn(name: str):
    if name == "tanh":
        return np.tanh
    if name == "relu":
        return lambda x: np.maximum(x, 0.0)
    if name == "silu":
        return lambda x: x / (1.0 + np.exp(-x))
    if name == "elu":
        return lambda x: np.where(x > 0.0, x, np.expm1(x))
    raise FrozenBodyPolicyArtifactError(f"unsupported actor activation: {name}")


def _save_npz_tree(path: Path, tree: dict[str, Any]) -> None:
    flat = {}
    for key, value in _flatten_tree(tree).items():
        flat[key] = np.asarray(value)
    np.savez_compressed(path, **flat)


def _load_npz_tree(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        flat = {key: np.asarray(data[key]) for key in data.files}
    return _unflatten_tree(flat)


def _flatten_tree(tree: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in tree.items():
        path = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_tree(value, path))
        else:
            flat[path] = value
    return flat


def _unflatten_tree(flat: dict[str, Any]) -> dict[str, Any]:
    root: dict[str, Any] = {}
    for key, value in flat.items():
        cursor = root
        parts = key.split("/")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return root


def _manifest_to_dict(manifest: FrozenBodyPolicyManifest) -> dict[str, Any]:
    return {
        "schema_version": manifest.schema_version,
        "source_checkpoint": manifest.source_checkpoint,
        "tensor_format": manifest.tensor_format,
        "has_tensors": manifest.has_tensors,
        "actor_spec": asdict(manifest.actor_spec),
        "shape_report": asdict(manifest.shape_report),
        "params_file": manifest.params_file,
        "run_stats_file": manifest.run_stats_file,
        "body_obs_schema_file": manifest.body_obs_schema_file,
        "action_manifest_file": manifest.action_manifest_file,
    }


def _manifest_from_dict(data: dict[str, Any]) -> FrozenBodyPolicyManifest:
    actor_spec_data = data["actor_spec"]
    shape_report_data = data["shape_report"]
    return FrozenBodyPolicyManifest(
        schema_version=int(data["schema_version"]),
        source_checkpoint=str(data["source_checkpoint"]),
        tensor_format=str(data["tensor_format"]),
        has_tensors=bool(data["has_tensors"]),
        actor_spec=ActorCheckpointSpec(
            obs_size=int(actor_spec_data["obs_size"]),
            action_size=int(actor_spec_data["action_size"]),
            actor_hidden_layers=tuple(int(v) for v in actor_spec_data["actor_hidden_layers"]),
            critic_hidden_layers=tuple(int(v) for v in actor_spec_data["critic_hidden_layers"]),
            activation=str(actor_spec_data["activation"]),
            init_std=float(actor_spec_data["init_std"]),
            learnable_std=bool(actor_spec_data["learnable_std"]),
            use_layernorm=bool(actor_spec_data["use_layernorm"]),
            layernorm_eps=float(actor_spec_data["layernorm_eps"]),
        ),
        shape_report=ActorCheckpointShapeReport(
            valid=bool(shape_report_data["valid"]),
            actor_output_kernel_shape=tuple(int(v) for v in shape_report_data["actor_output_kernel_shape"]),
            log_std_shape=tuple(int(v) for v in shape_report_data["log_std_shape"]),
            run_stats_mean_shape=tuple(int(v) for v in shape_report_data["run_stats_mean_shape"]),
            reason=str(shape_report_data["reason"]),
        ),
        params_file=str(data["params_file"]),
        run_stats_file=str(data["run_stats_file"]),
        body_obs_schema_file=str(data["body_obs_schema_file"]),
        action_manifest_file=str(data["action_manifest_file"]),
    )


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value

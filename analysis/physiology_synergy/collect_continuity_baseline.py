"""Build source-bound identities for a real held-out continuity rollout.

The numerical samples are captured by :mod:`musclemimic.runner.eval_utils`.
This module seals the checkpoint, promoted-policy evidence, resolved runtime
model, exact interpolated held-out trajectories, and rollout protocol before
the samples are accepted by the coefficient calibrator.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jax
import numpy as np
from omegaconf import OmegaConf

from musclemimic.distill.motion_identity import resolve_config_motion_paths
from musclemimic.physiology.anatomical_groups import AnatomicalTaxonomy
from musclemimic.physiology.continuity_groups import FascicleContinuityGraph
from musclemimic.physiology.effective_excitation import resolve_muscle_channel_layout

CONTINUITY_BASELINE_ENVIRONMENT_SCHEMA_VERSION = "continuity_baseline_environment_v1"
CONTINUITY_BASELINE_HELDOUT_SCHEMA_VERSION = "continuity_baseline_heldout_split_v1"
CONTINUITY_BASELINE_ROLLOUT_MANIFEST_SCHEMA_VERSION = "continuity_baseline_rollout_manifest_v1"
_TRAJECTORY_ARRAY_FIELDS = (
    "qpos",
    "qvel",
    "xpos",
    "xquat",
    "cvel",
    "subtree_com",
    "site_xpos",
    "site_xmat",
    "cvel_parent",
    "subtree_com_root",
    "xpos_parent",
    "xquat_parent",
    "split_points",
)


def validate_diagnostics_collection_contract(
    config: Any,
    *,
    taxonomy: AnatomicalTaxonomy,
    graph: FascicleContinuityGraph,
) -> dict[str, Any]:
    """Require a diagnostics-only, coefficient-zero baseline configuration."""

    experiment = _node(config, "experiment", config)
    env_params = _mapping(_node(experiment, "env_params"), "experiment.env_params")
    reward_params = _mapping(env_params.get("reward_params"), "reward_params")
    continuity = _mapping(
        reward_params.get("intra_muscle_consistency"),
        "reward_params.intra_muscle_consistency",
    )
    if continuity.get("mode") != "diagnostics":
        raise ValueError("baseline coefficient collection requires continuity mode=diagnostics")
    if continuity.get("signal") != "activation":
        raise ValueError("baseline coefficient collection requires activation continuity")
    coefficient = float(continuity.get("coefficient", float("nan")))
    if not np.isfinite(coefficient) or coefficient != 0.0:
        raise ValueError("baseline coefficient collection requires continuity coefficient=0")
    if any(bool(chain["training_enabled"]) for chain in graph.chains):
        raise ValueError("baseline calibration must use the pre-promotion diagnostics graph")
    if continuity.get("expected_taxonomy_fingerprint") != taxonomy.fingerprint:
        raise ValueError("resolved baseline config taxonomy fingerprint differs from the loaded taxonomy")
    if continuity.get("expected_continuity_fingerprint") != graph.graph_fingerprint:
        raise ValueError("resolved baseline config continuity fingerprint differs from the loaded graph")
    return continuity


def infer_action_mode(config: Any) -> str:
    experiment = _node(config, "experiment", config)
    action = _node(experiment, "action_representation", {})
    if not bool(_node(action, "enabled", False)):
        return "full_354"
    mode = str(_node(action, "mode", ""))
    if mode not in {"fixed_synergy", "fixed_synergy_residual"}:
        raise ValueError(
            "continuity calibration supports only direct-354, fixed-synergy, or fixed-synergy-residual policies"
        )
    residual = _node(action, "residual", {})
    residual_enabled = bool(_node(residual, "enabled", False))
    if (mode == "fixed_synergy_residual") != residual_enabled:
        raise ValueError("continuity calibration action mode and residual.enabled disagree")
    if residual_enabled:
        alpha = float(_node(residual, "alpha", 0.0))
        if not np.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("enabled structured residual requires a positive finite alpha")
        return "fixed_synergy_residual"
    return "fixed_synergy"


def build_heldout_split_manifest(config: Any, env: Any) -> dict[str, Any]:
    """Hash the exact loaded/interpolated trajectory arrays used by evaluation."""

    handler = getattr(env, "th", None)
    if handler is None or getattr(handler, "traj", None) is None:
        raise ValueError("continuity baseline environment has no trajectory handler")
    motion_paths = resolve_config_motion_paths(config)
    trajectory_count = int(handler.n_trajectories)
    if trajectory_count <= 0:
        raise ValueError("continuity baseline held-out split is empty")
    lengths = [int(handler.len_trajectory(index)) for index in range(trajectory_count)]
    if any(length <= 0 for length in lengths):
        raise ValueError("continuity baseline held-out trajectory length is invalid")
    frequency = float(handler.traj.info.frequency)
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("continuity baseline trajectory frequency is invalid")
    array_manifest, content_sha256 = _trajectory_content_manifest(handler.traj.data)
    payload = {
        "schema_version": CONTINUITY_BASELINE_HELDOUT_SCHEMA_VERSION,
        "ordered_motion_paths": motion_paths,
        "trajectory_count": trajectory_count,
        "trajectory_lengths": lengths,
        "frequency_hz": frequency,
        "array_manifest": array_manifest,
        "trajectory_content_sha256": content_sha256,
    }
    payload["heldout_split_fingerprint"] = _json_sha256(payload)
    return validate_heldout_split_manifest(payload)


def build_environment_manifest(
    config: Any,
    env: Any,
    *,
    taxonomy: AnatomicalTaxonomy,
    resolved_env_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fingerprint the resolved evaluation environment and physical muscle ABI."""

    model = env.get_model() if hasattr(env, "get_model") else getattr(env, "_model", None)
    if model is None:
        raise ValueError("continuity baseline environment has no MuJoCo model")
    layout = resolve_muscle_channel_layout(
        model,
        taxonomy.actuator_names,
        require_unit_ctrlrange=True,
        require_scalar_activation=True,
    )
    experiment = _node(config, "experiment", config)
    payload = {
        "schema_version": CONTINUITY_BASELINE_ENVIRONMENT_SCHEMA_VERSION,
        "resolved_env_params": _canonical_value(
            _node(experiment, "env_params", {}) if resolved_env_params is None else resolved_env_params
        ),
        "resolved_task_factory": _canonical_value(_node(experiment, "task_factory", {})),
        "resolved_action_representation": _canonical_value(_node(experiment, "action_representation", {})),
        "runtime_model_hash": layout.runtime_model_hash,
        "muscle_channel_core_fingerprint": layout.muscle_channel_core_fingerprint,
        "ordered_actuator_schema_hash": layout.actuator_schema_hash,
        "ordered_action_dim": layout.width,
        "control_dt": float(env.dt),
        "horizon": int(env.info.horizon),
    }
    payload["environment_fingerprint"] = _json_sha256(payload)
    return validate_environment_manifest(payload)


def validate_environment_manifest(value: Any) -> dict[str, Any]:
    payload = _mapping(value, "continuity baseline environment manifest")
    expected = {
        "schema_version",
        "resolved_env_params",
        "resolved_task_factory",
        "resolved_action_representation",
        "runtime_model_hash",
        "muscle_channel_core_fingerprint",
        "ordered_actuator_schema_hash",
        "ordered_action_dim",
        "control_dt",
        "horizon",
        "environment_fingerprint",
    }
    if set(payload) != expected:
        raise ValueError("continuity baseline environment manifest fields differ from contract")
    if payload["schema_version"] != CONTINUITY_BASELINE_ENVIRONMENT_SCHEMA_VERSION:
        raise ValueError("unsupported continuity baseline environment manifest schema")
    for field in (
        "runtime_model_hash",
        "muscle_channel_core_fingerprint",
        "ordered_actuator_schema_hash",
    ):
        _sha256(payload[field], field)
    if int(payload["ordered_action_dim"]) != 354:
        raise ValueError("continuity baseline environment must expose exactly 354 ordered muscles")
    if float(payload["control_dt"]) <= 0.0 or not np.isfinite(float(payload["control_dt"])):
        raise ValueError("continuity baseline environment control_dt must be positive and finite")
    if int(payload["horizon"]) <= 0:
        raise ValueError("continuity baseline environment horizon must be positive")
    supplied = _sha256(payload["environment_fingerprint"], "environment_fingerprint")
    unsigned = dict(payload)
    unsigned.pop("environment_fingerprint")
    if _json_sha256(unsigned) != supplied:
        raise ValueError("continuity baseline environment fingerprint is stale")
    return _canonical_value(payload)


def validate_heldout_split_manifest(value: Any) -> dict[str, Any]:
    payload = _mapping(value, "continuity baseline held-out manifest")
    expected = {
        "schema_version",
        "ordered_motion_paths",
        "trajectory_count",
        "trajectory_lengths",
        "frequency_hz",
        "array_manifest",
        "trajectory_content_sha256",
        "heldout_split_fingerprint",
    }
    if set(payload) != expected:
        raise ValueError("continuity baseline held-out manifest fields differ from contract")
    if payload["schema_version"] != CONTINUITY_BASELINE_HELDOUT_SCHEMA_VERSION:
        raise ValueError("unsupported continuity baseline held-out manifest schema")
    paths = payload["ordered_motion_paths"]
    if not isinstance(paths, list) or not paths or any(not isinstance(path, str) or not path for path in paths):
        raise ValueError("continuity baseline held-out motion paths are invalid")
    if len(set(paths)) != len(paths):
        raise ValueError("continuity baseline held-out motion paths contain duplicates")
    trajectory_count = int(payload["trajectory_count"])
    lengths = payload["trajectory_lengths"]
    if trajectory_count <= 0 or not isinstance(lengths, list) or len(lengths) != trajectory_count:
        raise ValueError("continuity baseline held-out trajectory coverage is invalid")
    if any(int(length) <= 0 or int(length) != float(length) for length in lengths):
        raise ValueError("continuity baseline held-out trajectory lengths are invalid")
    frequency = float(payload["frequency_hz"])
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("continuity baseline held-out frequency is invalid")
    array_manifest = _mapping(payload["array_manifest"], "held-out array manifest")
    if set(array_manifest) != set(_TRAJECTORY_ARRAY_FIELDS):
        raise ValueError("continuity baseline held-out array inventory is incomplete")
    for field, descriptor_value in array_manifest.items():
        descriptor = _mapping(descriptor_value, f"held-out array {field}")
        if set(descriptor) != {"dtype", "shape", "sha256"}:
            raise ValueError(f"held-out array {field} descriptor fields differ from contract")
        if not isinstance(descriptor["dtype"], str) or not descriptor["dtype"]:
            raise ValueError(f"held-out array {field} dtype is invalid")
        shape = descriptor["shape"]
        if not isinstance(shape, list) or any(int(size) < 0 or int(size) != float(size) for size in shape):
            raise ValueError(f"held-out array {field} shape is invalid")
        _sha256(descriptor["sha256"], f"held-out array {field} sha256")
    content = _sha256(payload["trajectory_content_sha256"], "trajectory_content_sha256")
    if _trajectory_manifest_digest(array_manifest) != content:
        raise ValueError("continuity baseline trajectory content fingerprint is stale")
    supplied = _sha256(payload["heldout_split_fingerprint"], "heldout_split_fingerprint")
    unsigned = dict(payload)
    unsigned.pop("heldout_split_fingerprint")
    if _json_sha256(unsigned) != supplied:
        raise ValueError("continuity baseline held-out split fingerprint is stale")
    return _canonical_value(payload)


def build_rollout_identity(
    *,
    config: Any,
    checkpoint_identity: Mapping[str, Any],
    promoted_artifact: Mapping[str, Any],
    taxonomy: AnatomicalTaxonomy,
    graph: FascicleContinuityGraph,
    environment_manifest: Mapping[str, Any],
    heldout_split_manifest: Mapping[str, Any],
    backend: str,
    deterministic: bool,
    eval_seed: int,
    num_envs: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create the calibrator identity and its auditable rollout manifest."""

    if deterministic is not True:
        raise ValueError("continuity baseline calibration requires deterministic policy evaluation")
    if backend not in {"mjx", "mujoco"}:
        raise ValueError("continuity baseline backend must be mjx or mujoco")
    if int(num_envs) <= 0:
        raise ValueError("continuity baseline num_envs must be positive")
    checkpoint = _mapping(checkpoint_identity, "checkpoint identity")
    promotion = _mapping(promoted_artifact, "promoted artifact")
    promotion_checkpoint = _mapping(promotion.get("checkpoint"), "promoted checkpoint identity")
    if promotion_checkpoint.get("checkpoint_content_sha256") != checkpoint.get("checkpoint_content_sha256"):
        raise ValueError("promoted artifact does not bind the evaluated checkpoint")
    promotion_fingerprint = _sha256(promotion.get("binding_sha256"), "promotion binding")

    experiment = _node(config, "experiment", config)
    configured_run_id = str(_node(experiment, "run_id", ""))
    if not configured_run_id or configured_run_id != str(checkpoint.get("run_id", "")):
        raise ValueError("evaluated run_id differs from checkpoint run identity")
    config_hash = str(checkpoint.get("config_hash", ""))
    if not config_hash:
        raise ValueError("evaluated checkpoint has no config hash")
    checkpoint_fingerprint = _sha256(
        checkpoint.get("checkpoint_content_sha256"),
        "checkpoint content fingerprint",
    )
    environment_manifest = validate_environment_manifest(environment_manifest)
    heldout_split_manifest = validate_heldout_split_manifest(heldout_split_manifest)
    environment_fingerprint = _sha256(
        environment_manifest["environment_fingerprint"],
        "environment fingerprint",
    )
    heldout_fingerprint = _sha256(
        heldout_split_manifest["heldout_split_fingerprint"],
        "heldout split fingerprint",
    )
    rollout_manifest: dict[str, Any] = {
        "schema_version": CONTINUITY_BASELINE_ROLLOUT_MANIFEST_SCHEMA_VERSION,
        "semantics": "evaluate_all_once_per_heldout_from_frame_zero_v1",
        "policy": {
            "run_id": configured_run_id,
            "action_mode": infer_action_mode(config),
            "config_hash": config_hash,
            "checkpoint_fingerprint": checkpoint_fingerprint,
            "promotion_fingerprint": promotion_fingerprint,
        },
        "physiology": {
            "taxonomy_fingerprint": taxonomy.fingerprint,
            "continuity_graph_fingerprint": graph.graph_fingerprint,
            "measured_chain_count": len(graph.chains),
            "measured_edge_count": graph.edge_count,
        },
        "environment_manifest": _canonical_value(environment_manifest),
        "heldout_split_manifest": _canonical_value(heldout_split_manifest),
        "protocol": {
            "backend": backend,
            "deterministic": True,
            "eval_seed": int(eval_seed),
            "num_envs": int(num_envs),
            "evaluate_all": True,
            "start_from_beginning": True,
            "padding_steps_included": False,
            "post_completion_steps_included": False,
            "primary_reward_sample": "reward_imitation_total_before_penalties",
            "primary_continuity_sample": "post_transition_data.act_fascicle_continuity_loss",
        },
    }
    rollout_manifest["rollout_manifest_fingerprint"] = rollout_manifest_fingerprint(rollout_manifest)
    identity = {
        "run_id": configured_run_id,
        "action_mode": infer_action_mode(config),
        "config_hash": config_hash,
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "promotion_fingerprint": promotion_fingerprint,
        "rollout_manifest_fingerprint": rollout_manifest["rollout_manifest_fingerprint"],
        "environment_fingerprint": environment_fingerprint,
        "heldout_split_fingerprint": heldout_fingerprint,
        "taxonomy_fingerprint": taxonomy.fingerprint,
        "continuity_graph_fingerprint": graph.graph_fingerprint,
    }
    return identity, validate_rollout_manifest(rollout_manifest)


def rollout_manifest_fingerprint(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("rollout_manifest_fingerprint", None)
    return _json_sha256(unsigned)


def validate_rollout_manifest(value: Any) -> dict[str, Any]:
    payload = _mapping(value, "continuity baseline rollout manifest")
    expected = {
        "schema_version",
        "semantics",
        "policy",
        "physiology",
        "environment_manifest",
        "heldout_split_manifest",
        "protocol",
        "rollout_manifest_fingerprint",
    }
    if set(payload) != expected:
        raise ValueError("continuity baseline rollout manifest fields differ from contract")
    if payload["schema_version"] != CONTINUITY_BASELINE_ROLLOUT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported continuity baseline rollout manifest schema")
    if payload["semantics"] != "evaluate_all_once_per_heldout_from_frame_zero_v1":
        raise ValueError("continuity baseline rollout semantics differ from contract")

    policy = _mapping(payload["policy"], "continuity baseline rollout policy")
    expected_policy = {
        "run_id",
        "action_mode",
        "config_hash",
        "checkpoint_fingerprint",
        "promotion_fingerprint",
    }
    if set(policy) != expected_policy:
        raise ValueError("continuity baseline rollout policy fields differ from contract")
    for field in ("run_id", "config_hash"):
        if not isinstance(policy[field], str) or not policy[field]:
            raise ValueError(f"continuity baseline rollout policy {field} is invalid")
    if policy["action_mode"] not in {
        "full_354",
        "fixed_synergy",
        "fixed_synergy_residual",
    }:
        raise ValueError("continuity baseline rollout policy action_mode is invalid")
    _sha256(policy["checkpoint_fingerprint"], "checkpoint_fingerprint")
    _sha256(policy["promotion_fingerprint"], "promotion_fingerprint")

    physiology = _mapping(payload["physiology"], "continuity baseline rollout physiology")
    expected_physiology = {
        "taxonomy_fingerprint",
        "continuity_graph_fingerprint",
        "measured_chain_count",
        "measured_edge_count",
    }
    if set(physiology) != expected_physiology:
        raise ValueError("continuity baseline rollout physiology fields differ from contract")
    _sha256(physiology["taxonomy_fingerprint"], "taxonomy_fingerprint")
    _sha256(
        physiology["continuity_graph_fingerprint"],
        "continuity_graph_fingerprint",
    )
    for field in ("measured_chain_count", "measured_edge_count"):
        if (
            isinstance(physiology[field], bool)
            or not isinstance(physiology[field], int | np.integer)
            or int(physiology[field]) <= 0
        ):
            raise ValueError(f"continuity baseline rollout {field} must be a positive integer")

    environment = validate_environment_manifest(payload["environment_manifest"])
    heldout = validate_heldout_split_manifest(payload["heldout_split_manifest"])
    declared_action_mode = infer_action_mode(
        {"experiment": {"action_representation": environment["resolved_action_representation"]}}
    )
    if declared_action_mode != policy["action_mode"]:
        raise ValueError("rollout policy action mode differs from the environment manifest")

    protocol = _mapping(payload["protocol"], "continuity baseline rollout protocol")
    expected_protocol = {
        "backend",
        "deterministic",
        "eval_seed",
        "num_envs",
        "evaluate_all",
        "start_from_beginning",
        "padding_steps_included",
        "post_completion_steps_included",
        "primary_reward_sample",
        "primary_continuity_sample",
    }
    if set(protocol) != expected_protocol:
        raise ValueError("continuity baseline rollout protocol fields differ from contract")
    if (
        protocol.get("deterministic") is not True
        or protocol.get("evaluate_all") is not True
        or protocol.get("start_from_beginning") is not True
        or protocol.get("padding_steps_included") is not False
        or protocol.get("post_completion_steps_included") is not False
    ):
        raise ValueError("continuity baseline rollout protocol is not deterministic evaluate-all")
    if protocol["backend"] not in {"mjx", "mujoco"}:
        raise ValueError("continuity baseline rollout backend is invalid")
    for field in ("eval_seed", "num_envs"):
        if isinstance(protocol[field], bool) or not isinstance(protocol[field], int | np.integer):
            raise ValueError(f"continuity baseline rollout {field} must be an integer")
    if int(protocol["eval_seed"]) < 0:
        raise ValueError("continuity baseline rollout eval_seed must be non-negative")
    if int(protocol["num_envs"]) <= 0:
        raise ValueError("continuity baseline rollout num_envs must be positive")
    if protocol["backend"] == "mujoco" and int(protocol["num_envs"]) != 1:
        raise ValueError("MuJoCo continuity baseline rollout must use exactly one environment")
    if protocol["primary_reward_sample"] != "reward_imitation_total_before_penalties":
        raise ValueError("continuity baseline rollout reward sample semantics differ from contract")
    if protocol["primary_continuity_sample"] != "post_transition_data.act_fascicle_continuity_loss":
        raise ValueError("continuity baseline rollout continuity sample semantics differ from contract")
    if int(heldout["trajectory_count"]) <= 0:
        raise ValueError("continuity baseline rollout held-out split is empty")
    supplied = _sha256(
        payload["rollout_manifest_fingerprint"],
        "rollout_manifest_fingerprint",
    )
    if rollout_manifest_fingerprint(payload) != supplied:
        raise ValueError("continuity baseline rollout manifest fingerprint is stale")
    return _canonical_value(payload)


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return target.resolve()


def _trajectory_content_manifest(data: Any) -> tuple[dict[str, Any], str]:
    manifest: dict[str, Any] = {}
    for field in _TRAJECTORY_ARRAY_FIELDS:
        if not hasattr(data, field):
            raise ValueError(f"held-out trajectory data lacks {field}")
        array = np.asarray(jax.device_get(getattr(data, field)))
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError(f"held-out trajectory array {field} must be finite numeric data")
        contiguous = np.ascontiguousarray(array)
        descriptor = {
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
        }
        manifest[field] = descriptor
    return manifest, _trajectory_manifest_digest(manifest)


def _trajectory_manifest_digest(manifest: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(b"continuity-heldout-trajectory-arrays-v1\0")
    for field in _TRAJECTORY_ARRAY_FIELDS:
        descriptor = manifest[field]
        encoded_field = field.encode("utf-8")
        digest.update(len(encoded_field).to_bytes(8, "big"))
        digest.update(encoded_field)
        digest.update(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("ascii"))
    return digest.hexdigest()


def _canonical_value(value: Any) -> Any:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_canonical_value(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("continuity rollout manifest cannot contain non-finite values")
        return value
    raise TypeError(f"continuity rollout manifest cannot canonicalize {type(value).__name__}")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return dict(value)


def _node(value: Any, key: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(key, default)
    try:
        return value.get(key, default)
    except (AttributeError, TypeError):
        return getattr(value, key, default)


def _sha256(value: Any, field: str) -> str:
    result = str(value)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{field} must be lowercase SHA-256")
    return result


__all__ = [
    "atomic_write_json",
    "build_environment_manifest",
    "build_heldout_split_manifest",
    "build_rollout_identity",
    "infer_action_mode",
    "validate_diagnostics_collection_contract",
    "validate_environment_manifest",
    "validate_heldout_split_manifest",
    "validate_rollout_manifest",
]

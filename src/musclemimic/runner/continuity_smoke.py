"""Evidence contract for continuity-reward GPU training smoke runs.

The artifact emitted here is deliberately narrower than a training manifest:
it proves that one exact resolved formal configuration was exercised through
three real PPO updates on a GPU, including the released reward, action
decoder, MuJoCo/MJX ABI, and an Orbax save/restore round trip.  Formal reward
runs validate this artifact before creating their checkpoint directory.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

CONTINUITY_TRAINING_SMOKE_SCHEMA_VERSION = "forehand_continuity_training_smoke_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_EXPECTATION_UNSET = object()

REQUIRED_SMOKE_CHECKS = (
    "environment_is_myofullbody",
    "ordered_muscle_channels_354",
    "disable_fingers",
    "unit_ctrlrange",
    "unique_activation_addresses",
    "runtime_core_fingerprint_matches",
    "global_coverage_matches",
    "target_coverage_matches",
    "target_loss_nonconstant",
    "target_loss_nonzero",
    "continuity_penalty_nonzero",
    "continuity_penalty_not_always_total_clipped",
    "reward_finite",
    "reset_jit",
    "step_jit",
    "rollout_scan",
    "ppo_loss_finite",
    "gradient_finite",
    "optimizer_update",
    "checkpoint_saved",
    "checkpoint_restored",
    "restored_release_fingerprint_matches",
    "action_interface_mode_matches",
    "graph_nmf_contract_matches",
    "bare_racket_portable_core_matches",
    "gpu_backend",
    "three_updates_eight_envs",
)


def _native(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True, throw_on_missing=True)
    return copy.deepcopy(value)


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


def resolved_training_config_sha256(config: Any) -> str:
    """Hash the resolved formal config without a self-referential smoke hash."""

    payload = _native(config)
    experiment = payload.get("experiment", {}) if isinstance(payload, dict) else {}
    gate = experiment.get("continuity_smoke_gate", {}) if isinstance(experiment, dict) else {}
    if isinstance(gate, dict):
        # The optional expected artifact fingerprint is known only after the
        # smoke completes. It is independently checked by the gate and cannot
        # participate in the pre-smoke resolved-config identity.
        gate["expected_artifact_fingerprint"] = ""
    return _json_sha256(payload)


def continuity_training_smoke_fingerprint(payload: dict[str, Any]) -> str:
    unsigned = copy.deepcopy(payload)
    unsigned.pop("artifact_fingerprint", None)
    return _json_sha256(unsigned)


def repository_git_commit(repo_root: str | Path, *, require_clean: bool) -> str:
    root = Path(repo_root).resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not _GIT_SHA.fullmatch(commit):
        raise ValueError("repository HEAD is not a full Git commit SHA")
    if require_clean:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if status:
            raise ValueError(
                "continuity smoke evidence requires a clean worktree so commit SHA binds the executed code"
            )
    return commit


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty UTC timestamp")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must carry an explicit UTC offset")
    return parsed.astimezone(UTC)


def _required_sha256(value: Any, label: str) -> str:
    text = str(value or "")
    if not _SHA256.fullmatch(text):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def _optional_sha256(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _required_sha256(value, label)


def _strict_object(payload: Any, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _payload_series(payload: dict[str, Any], key: str) -> np.ndarray:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"measurements.{key} must be a non-empty numeric list")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"measurements.{key} must be numeric") from exc
    if result.ndim != 1:
        raise ValueError(f"measurements.{key} must be one-dimensional")
    return result


def load_continuity_training_smoke(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid continuity smoke JSON: {source}") from exc
    return validate_continuity_training_smoke(payload)


def validate_continuity_training_smoke(
    payload: dict[str, Any],
    *,
    expected_commit_sha: str | None = None,
    expected_resolved_config_sha256: str | None = None,
    expected_release_fingerprint: str | None = None,
    expected_basis_fingerprint: str | None | object = _EXPECTATION_UNSET,
    expected_action_mode: str | None = None,
    expected_condition: str | None = None,
    expected_artifact_fingerprint: str | None = None,
    max_age_hours: float | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate intrinsic integrity and optional formal-launch expectations."""

    payload = _strict_object(payload, "continuity training smoke")
    required_top_level = {
        "schema_version",
        "created_at_utc",
        "completed_at_utc",
        "git_commit_sha",
        "formal_config",
        "runtime",
        "contracts",
        "execution",
        "checks",
        "measurements",
        "errors",
        "passed",
        "artifact_fingerprint",
    }
    if set(payload) != required_top_level:
        missing = sorted(required_top_level - set(payload))
        extra = sorted(set(payload) - required_top_level)
        raise ValueError(f"continuity smoke keys differ: missing={missing} extra={extra}")
    if payload["schema_version"] != CONTINUITY_TRAINING_SMOKE_SCHEMA_VERSION:
        raise ValueError("unsupported continuity training smoke schema")

    supplied_fingerprint = _required_sha256(payload["artifact_fingerprint"], "artifact_fingerprint")
    if continuity_training_smoke_fingerprint(payload) != supplied_fingerprint:
        raise ValueError("continuity training smoke fingerprint is stale")
    if expected_artifact_fingerprint and supplied_fingerprint != _required_sha256(
        expected_artifact_fingerprint,
        "expected artifact fingerprint",
    ):
        raise ValueError("continuity smoke artifact fingerprint differs from launch expectation")

    commit = str(payload["git_commit_sha"])
    if not _GIT_SHA.fullmatch(commit):
        raise ValueError("continuity smoke git_commit_sha must be a full lowercase Git SHA")
    if expected_commit_sha is not None and commit != expected_commit_sha:
        raise ValueError("continuity smoke was produced by a different Git commit")

    formal = _strict_object(payload["formal_config"], "formal_config")
    resolved_hash = _required_sha256(
        formal.get("resolved_config_sha256"),
        "formal_config.resolved_config_sha256",
    )
    condition = str(formal.get("condition", ""))
    if not re.fullmatch(r"[ABCG]1", condition):
        raise ValueError("continuity smoke condition must be one of A1/B1/C1/G1")
    if expected_resolved_config_sha256 is not None and resolved_hash != _required_sha256(
        expected_resolved_config_sha256,
        "expected resolved config hash",
    ):
        raise ValueError("continuity smoke resolved config hash differs from formal launch")
    if expected_condition is not None and condition != expected_condition:
        raise ValueError("continuity smoke condition differs from formal launch")

    runtime = _strict_object(payload["runtime"], "runtime")
    if runtime.get("jax_backend") != "gpu":
        raise ValueError("continuity smoke must run on the JAX GPU backend")
    if int(runtime.get("ordered_muscle_channels", 0)) != 354:
        raise ValueError("continuity smoke did not bind 354 ordered muscle channels")
    _required_sha256(
        runtime.get("muscle_channel_core_fingerprint"),
        "runtime muscle_channel_core_fingerprint",
    )
    if int(runtime.get("activation_address_count", 0)) != 354:
        raise ValueError("continuity smoke activation addresses are not one-to-one")
    if float(runtime.get("ctrlrange_min", math.nan)) != 0.0 or float(runtime.get("ctrlrange_max", math.nan)) != 1.0:
        raise ValueError("continuity smoke muscle ctrlrange is not [0,1]")
    if runtime.get("racket_muscle_channel_core_fingerprint") != runtime.get("muscle_channel_core_fingerprint"):
        raise ValueError("continuity smoke bare/racket portable core differs")

    contracts = _strict_object(payload["contracts"], "contracts")
    release_fingerprint = _required_sha256(
        contracts.get("release_fingerprint"),
        "contracts.release_fingerprint",
    )
    basis_fingerprint = _optional_sha256(
        contracts.get("basis_fingerprint"),
        "contracts.basis_fingerprint",
    )
    action_mode = str(contracts.get("action_mode", ""))
    if action_mode not in {"full_354", "fixed_synergy", "fixed_synergy_residual"}:
        raise ValueError("continuity smoke action_mode is invalid")
    if expected_release_fingerprint is not None and release_fingerprint != _required_sha256(
        expected_release_fingerprint,
        "expected release fingerprint",
    ):
        raise ValueError("continuity smoke release fingerprint differs from formal launch")
    if expected_basis_fingerprint is not _EXPECTATION_UNSET:
        if basis_fingerprint != _optional_sha256(
            expected_basis_fingerprint,
            "expected basis fingerprint",
        ):
            raise ValueError("continuity smoke basis fingerprint differs from formal launch")
    if expected_action_mode is not None and action_mode != expected_action_mode:
        raise ValueError("continuity smoke action mode differs from formal launch")
    for name in (
        "taxonomy_fingerprint",
        "diagnostic_graph_fingerprint",
        "candidate_graph_fingerprint",
        "loss_spec_fingerprint",
        "calibration_fingerprint",
    ):
        _required_sha256(contracts.get(name), f"contracts.{name}")
    basis_factor_fingerprint = _optional_sha256(
        contracts.get("basis_factor_contract_fingerprint"),
        "contracts.basis_factor_contract_fingerprint",
    )
    graph_lineage_fingerprint = _optional_sha256(
        contracts.get("graph_regularization_lineage_fingerprint"),
        "contracts.graph_regularization_lineage_fingerprint",
    )
    residual_basis_fingerprint = _optional_sha256(
        contracts.get("residual_basis_fingerprint"),
        "contracts.residual_basis_fingerprint",
    )
    global_chain_count = int(contracts.get("global_chain_count", 0))
    global_edge_count = int(contracts.get("global_edge_count", 0))
    target_chain_count = int(contracts.get("target_chain_count", 0))
    target_edge_count = int(contracts.get("target_edge_count", 0))
    if min(global_chain_count, global_edge_count, target_chain_count, target_edge_count) <= 0:
        raise ValueError("continuity smoke global/target coverage must be non-empty")
    expected_action_contract = {
        "A1": ("full_354", "direct_354"),
        "B1": ("fixed_synergy", "standard_nmf"),
        "C1": ("fixed_synergy_residual", "standard_nmf_structured_residual"),
        "G1": ("fixed_synergy", "graph_nmf"),
    }[condition]
    if (action_mode, str(contracts.get("basis_family", ""))) != expected_action_contract:
        raise ValueError("continuity smoke condition/action/basis family contract differs")
    if condition == "A1":
        if any(
            value is not None
            for value in (
                basis_fingerprint,
                basis_factor_fingerprint,
                graph_lineage_fingerprint,
                residual_basis_fingerprint,
            )
        ):
            raise ValueError("A1 continuity smoke must not carry synergy artifacts")
    else:
        if basis_fingerprint is None or basis_factor_fingerprint is None:
            raise ValueError("synergy continuity smoke requires W and basis-factor fingerprints")
        if condition == "C1" and residual_basis_fingerprint is None:
            raise ValueError("C1 continuity smoke requires a residual basis fingerprint")
        if condition == "G1" and graph_lineage_fingerprint is None:
            raise ValueError("G1 continuity smoke requires graph regularization lineage")
        if condition != "G1" and graph_lineage_fingerprint is not None:
            raise ValueError("non-G continuity smoke cannot carry graph regularization lineage")

    execution = _strict_object(payload["execution"], "execution")
    if int(execution.get("num_updates", 0)) != 3 or int(execution.get("num_envs", 0)) != 8:
        raise ValueError("continuity smoke must contain exactly three updates with eight environments")
    num_steps = int(execution.get("num_steps", 0))
    if num_steps <= 0 or int(execution.get("total_timesteps", 0)) != 3 * 8 * num_steps:
        raise ValueError("continuity smoke timestep budget differs from its rollout dimensions")
    if not str(execution.get("checkpoint_path", "") or ""):
        raise ValueError("continuity smoke has no finalized checkpoint path")
    if int(execution.get("restored_update_number", -1)) != 3:
        raise ValueError("continuity smoke did not restore checkpoint update 3")
    if execution.get("restored_body_contract_matches") is not True:
        raise ValueError("continuity smoke restored body action contract differs")
    if execution.get("restored_release_fingerprint") != release_fingerprint:
        raise ValueError("continuity smoke restored release fingerprint differs")

    checks = _strict_object(payload["checks"], "checks")
    if set(checks) != set(REQUIRED_SMOKE_CHECKS):
        missing = sorted(set(REQUIRED_SMOKE_CHECKS) - set(checks))
        extra = sorted(set(checks) - set(REQUIRED_SMOKE_CHECKS))
        raise ValueError(f"continuity smoke checks differ: missing={missing} extra={extra}")
    measurements = _strict_object(payload["measurements"], "measurements")
    measurement_names = {
        "reward_total",
        "continuity_global_loss",
        "continuity_global_chain_count",
        "continuity_global_edge_count",
        "continuity_target_loss",
        "continuity_target_chain_count",
        "continuity_target_edge_count",
        "penalty_continuity_raw",
        "penalty_continuity_after_local_clip",
        "penalty_continuity_effective_after_total_clip",
        "continuity_penalty_masked_fraction",
        "ppo_total_loss",
        "gradient_l2_norm",
        "gradients_all_finite",
        "parameters_all_finite",
        "parameter_update_l2_norm",
        "optimizer_step",
    }
    if set(measurements) != measurement_names:
        missing = sorted(measurement_names - set(measurements))
        extra = sorted(set(measurements) - measurement_names)
        raise ValueError(f"continuity smoke measurements differ: missing={missing} extra={extra}")
    series = {name: _payload_series(measurements, name) for name in measurement_names}
    if any(values.size != 3 for values in series.values()):
        raise ValueError("every continuity smoke measurement must contain exactly three updates")
    derived_checks = {
        "environment_is_myofullbody": runtime.get("environment_class") == "MyoFullBody",
        "ordered_muscle_channels_354": int(runtime["ordered_muscle_channels"]) == 354,
        "disable_fingers": runtime.get("disable_fingers") is True,
        "unit_ctrlrange": float(runtime["ctrlrange_min"]) == 0.0 and float(runtime["ctrlrange_max"]) == 1.0,
        "unique_activation_addresses": int(runtime["activation_address_count"]) == 354,
        "runtime_core_fingerprint_matches": runtime["muscle_channel_core_fingerprint"]
        == contracts.get("muscle_channel_core_fingerprint"),
        "global_coverage_matches": bool(
            np.all(series["continuity_global_chain_count"] == global_chain_count)
            and np.all(series["continuity_global_edge_count"] == global_edge_count)
        ),
        "target_coverage_matches": bool(
            np.all(series["continuity_target_chain_count"] == target_chain_count)
            and np.all(series["continuity_target_edge_count"] == target_edge_count)
        ),
        "target_loss_nonconstant": bool(np.ptp(series["continuity_target_loss"]) > 1e-12),
        "target_loss_nonzero": bool(np.any(np.abs(series["continuity_target_loss"]) > 1e-12)),
        "continuity_penalty_nonzero": bool(
            np.any(np.abs(series["penalty_continuity_raw"]) > 1e-12)
            and np.any(np.abs(series["penalty_continuity_after_local_clip"]) > 1e-12)
        ),
        "continuity_penalty_not_always_total_clipped": bool(
            np.any(np.abs(series["penalty_continuity_effective_after_total_clip"]) > 1e-12)
            and np.any(series["continuity_penalty_masked_fraction"] < 1.0 - 1e-7)
        ),
        "reward_finite": bool(np.isfinite(series["reward_total"]).all()),
        "reset_jit": series["reward_total"].size == 3,
        "step_jit": series["reward_total"].size == 3,
        "rollout_scan": series["reward_total"].size == 3,
        "ppo_loss_finite": bool(np.isfinite(series["ppo_total_loss"]).all()),
        "gradient_finite": bool(
            np.isfinite(series["gradient_l2_norm"]).all()
            and np.all(series["gradients_all_finite"] > 0.5)
            and np.all(series["parameters_all_finite"] > 0.5)
        ),
        "optimizer_update": bool(
            np.all(series["parameter_update_l2_norm"] > 0.0) and np.all(np.diff(series["optimizer_step"]) > 0.0)
        ),
        "checkpoint_saved": bool(execution.get("checkpoint_path")),
        "checkpoint_restored": bool(
            int(execution["restored_update_number"]) == 3 and execution["restored_body_contract_matches"]
        ),
        "restored_release_fingerprint_matches": execution["restored_release_fingerprint"] == release_fingerprint,
        "action_interface_mode_matches": (action_mode, contracts["basis_family"]) == expected_action_contract,
        "graph_nmf_contract_matches": (graph_lineage_fingerprint is not None) == (condition == "G1"),
        "bare_racket_portable_core_matches": runtime["racket_muscle_channel_core_fingerprint"]
        == runtime["muscle_channel_core_fingerprint"]
        == contracts.get("muscle_channel_core_fingerprint"),
        "gpu_backend": runtime["jax_backend"] == "gpu",
        "three_updates_eight_envs": int(execution["num_updates"]) == 3 and int(execution["num_envs"]) == 8,
    }
    inconsistent = sorted(name for name, value in derived_checks.items() if checks.get(name) is not value)
    if inconsistent:
        raise ValueError(f"continuity smoke check values disagree with evidence: {inconsistent}")
    failed = sorted(name for name, passed in checks.items() if passed is not True)
    if failed:
        raise ValueError(f"continuity smoke has failed checks: {failed}")
    if payload["passed"] is not True or payload["errors"] != []:
        raise ValueError("continuity smoke is not a clean passing artifact")

    created = _parse_utc(payload["created_at_utc"], "created_at_utc")
    completed = _parse_utc(payload["completed_at_utc"], "completed_at_utc")
    if completed < created:
        raise ValueError("continuity smoke completed before it started")
    if max_age_hours is not None:
        age_limit = float(max_age_hours)
        if not math.isfinite(age_limit) or age_limit <= 0.0:
            raise ValueError("continuity smoke max_age_hours must be finite and positive")
        reference = datetime.now(UTC) if now is None else now.astimezone(UTC)
        if completed > reference + timedelta(minutes=5):
            raise ValueError("continuity smoke completion time is in the future")
        if reference - completed > timedelta(hours=age_limit):
            raise ValueError("continuity smoke artifact is too old for a formal launch")
    return copy.deepcopy(payload)


def _base_env(env: Any) -> Any:
    current = env
    visited: set[int] = set()
    while hasattr(current, "env") and id(current) not in visited:
        visited.add(id(current))
        next_env = current.env
        if next_env is current:
            break
        current = next_env
    return current


def _series(container: Any, name: str) -> np.ndarray:
    import jax

    value = container[name] if isinstance(container, dict) else getattr(container, name)
    result = np.asarray(jax.device_get(value))
    if result.ndim == 0:
        result = result.reshape(1)
    return result


def _all_tree_finite(value: Any) -> bool:
    import jax

    return all(np.isfinite(np.asarray(jax.device_get(leaf))).all() for leaf in jax.tree_util.tree_leaves(value))


def _contract_fingerprints(config: Any) -> tuple[dict[str, Any], dict[str, Any] | None]:
    experiment = config.experiment
    release_contract = _native(experiment.get("continuity_training_contract", None))
    if not isinstance(release_contract, dict):
        raise ValueError("training smoke requires a bound continuity training contract")
    body_contract = _native(experiment.get("body_synergy_contract", None))
    if body_contract is not None and not isinstance(body_contract, dict):
        raise ValueError("body synergy contract must resolve to an object")
    return release_contract, body_contract


def build_continuity_training_smoke_artifact(
    *,
    config: Any,
    env: Any,
    training_result: dict[str, Any],
    checkpoint_dir: str | Path,
    agent_conf: Any,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Build evidence after a real short PPO run and restore its checkpoint."""

    import jax

    from musclemimic.algorithms.ppo.checkpoint import load_checkpoint_for_resume
    from musclemimic.environments.humanoids.myofullbody_racket import MyoFullBodyRacket
    from musclemimic.physiology.anatomical_groups import validate_taxonomy_against_model
    from musclemimic.physiology.continuity_groups import validate_continuity_graph_against_model
    from musclemimic.physiology.release import (
        load_continuity_training_release,
        resolve_continuity_training_release,
        validate_release_against_runtime,
    )
    from musclemimic.physiology.runtime_binding import resolve_ordered_policy_muscle_layout
    from musclemimic.runner.checkpointing import (
        find_latest_checkpoint,
        validate_checkpoint_continuity_training_contract,
    )

    experiment = config.experiment
    smoke = experiment.get("training_smoke", {})
    ablation = experiment.get("continuity_ablation", {})
    condition = str(ablation.get("condition", ""))
    action_mode = str(ablation.get("action_mode", ""))
    basis_family = str(ablation.get("basis_family", ""))
    release_contract, body_contract = _contract_fingerprints(config)
    release = load_continuity_training_release(release_contract["release_path"])
    release_artifacts = resolve_continuity_training_release(release)

    base = _base_env(env)
    layout = resolve_ordered_policy_muscle_layout(base)
    validate_taxonomy_against_model(
        release_artifacts.taxonomy,
        base._model,
        compatibility="portable_muscle_channel_abi",
    )
    validate_continuity_graph_against_model(
        release_artifacts.diagnostic_graph,
        release_artifacts.taxonomy,
        base._model,
    )
    validate_continuity_graph_against_model(
        release_artifacts.candidate_graph,
        release_artifacts.taxonomy,
        base._model,
    )
    validate_release_against_runtime(
        release,
        taxonomy=release_artifacts.taxonomy,
        graph=release_artifacts.candidate_graph,
        runtime_loss_identity=release_artifacts.loss_identity,
        action_mode=action_mode,
    )

    errors: list[str] = []
    bare_core = layout.muscle_channel_core_fingerprint
    racket_core: str | None = None
    try:
        racket = MyoFullBodyRacket(disable_fingers=True)
        validate_taxonomy_against_model(
            release_artifacts.taxonomy,
            racket._model,
            compatibility="portable_muscle_channel_abi",
        )
        racket_layout = resolve_ordered_policy_muscle_layout(racket)
        racket_core = racket_layout.muscle_channel_core_fingerprint
    except Exception as exc:  # Keep a signed failed artifact for diagnosis.
        errors.append(f"bare/racket portability: {type(exc).__name__}: {exc}")

    metrics = training_result["training_metrics"]
    optimizer = training_result["optimizer_diagnostics"]
    reward_total = _series(metrics, "reward_total")
    global_loss = _series(metrics, "continuity_global_loss")
    target_loss = _series(metrics, "continuity_target_loss")
    global_chains = _series(metrics, "continuity_global_chain_count")
    global_edges = _series(metrics, "continuity_global_edge_count")
    target_chains = _series(metrics, "continuity_target_chain_count")
    target_edges = _series(metrics, "continuity_target_edge_count")
    penalty_raw = _series(metrics, "penalty_continuity_raw")
    penalty_local = _series(metrics, "penalty_continuity_after_local_clip")
    penalty_effective = _series(metrics, "penalty_continuity_effective_after_total_clip")
    masked_fraction = _series(metrics, "continuity_penalty_masked_fraction")
    total_loss = _series(optimizer, "ppo_total_loss")
    gradient_norm = _series(optimizer, "gradient_l2_norm")
    gradients_finite = _series(optimizer, "gradients_all_finite")
    parameters_finite = _series(optimizer, "parameters_all_finite")
    parameter_delta = _series(optimizer, "parameter_update_l2_norm")
    optimizer_step = _series(optimizer, "optimizer_step")

    num_updates = int(experiment.num_updates)
    num_envs = int(experiment.num_envs)
    num_steps = int(experiment.num_steps)
    expected_global_chains = int(release.diagnostic_graph["global_chain_count"])
    expected_global_edges = int(release.diagnostic_graph["global_edge_count"])
    expected_target_chains = int(release.loss_spec["target_chain_count"])
    expected_target_edges = int(release.loss_spec["target_edge_count"])

    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    restored = False
    restored_finite = False
    restored_body_contract_matches = False
    restored_release: str | None = None
    restored_update_number: int | None = None
    if latest_checkpoint is None:
        errors.append("checkpoint save: no finalized checkpoint was found")
    else:
        try:
            validate_checkpoint_continuity_training_contract(
                latest_checkpoint,
                release_contract,
            )
            restored_state, restored_metadata = load_checkpoint_for_resume(
                latest_checkpoint,
                agent_conf,
            )
            restored = True
            restored_finite = _all_tree_finite(restored_state.train_state.params)
            restored_update_number = int(restored_metadata["update_number"])
            metadata_contract = restored_metadata.get("continuity_training_contract")
            if isinstance(metadata_contract, dict):
                restored_release = metadata_contract.get("release_fingerprint")
            restored_body_contract_matches = restored_metadata.get("body_synergy_contract") == body_contract
        except Exception as exc:  # Keep a failed artifact instead of losing diagnostics.
            errors.append(f"checkpoint restore: {type(exc).__name__}: {exc}")

    expected_core = release.taxonomy["muscle_channel_core_fingerprint"]
    expected_action_mode = "full_354" if body_contract is None else str(body_contract.get("mode", ""))
    basis_fingerprint = None if body_contract is None else body_contract.get("basis_fingerprint")
    graph_expected = condition == "G1"
    graph_actual = bool(experiment.get("action_representation", {}).get("require_graph_regularization", False))
    checks = {
        "environment_is_myofullbody": type(base).__name__ == "MyoFullBody",
        "ordered_muscle_channels_354": len(layout.actuator_names) == 354,
        "disable_fingers": bool(experiment.env_params.get("disable_fingers", False)),
        "unit_ctrlrange": bool(np.array_equal(layout.ctrlrange, np.tile([0.0, 1.0], (354, 1)))),
        "unique_activation_addresses": bool(np.unique(layout.activation_addresses).size == 354),
        "runtime_core_fingerprint_matches": bare_core == expected_core,
        "global_coverage_matches": bool(
            np.all(global_chains == expected_global_chains) and np.all(global_edges == expected_global_edges)
        ),
        "target_coverage_matches": bool(
            np.all(target_chains == expected_target_chains) and np.all(target_edges == expected_target_edges)
        ),
        "target_loss_nonconstant": bool(np.ptp(target_loss.astype(np.float64)) > 1e-12),
        "target_loss_nonzero": bool(np.any(np.abs(target_loss) > 1e-12)),
        "continuity_penalty_nonzero": bool(
            np.any(np.abs(penalty_raw) > 1e-12) and np.any(np.abs(penalty_local) > 1e-12)
        ),
        "continuity_penalty_not_always_total_clipped": bool(
            np.any(np.abs(penalty_effective) > 1e-12) and np.any(masked_fraction < 1.0 - 1e-7)
        ),
        "reward_finite": bool(np.isfinite(reward_total).all()),
        # These three operations are enclosed by the real outer-jitted train
        # function. Reaching exactly three returned scan entries is their
        # direct execution evidence, not a configuration-only assertion.
        "reset_jit": len(reward_total) == num_updates,
        "step_jit": len(reward_total) == num_updates,
        "rollout_scan": len(reward_total) == num_updates,
        "ppo_loss_finite": bool(np.isfinite(total_loss).all()),
        "gradient_finite": bool(
            np.isfinite(gradient_norm).all() and np.all(gradients_finite > 0.5) and np.all(parameters_finite > 0.5)
        ),
        "optimizer_update": bool(np.all(parameter_delta > 0.0) and optimizer_step[-1] > optimizer_step[0] - 1),
        "checkpoint_saved": latest_checkpoint is not None,
        "checkpoint_restored": (
            restored and restored_finite and restored_update_number == num_updates and restored_body_contract_matches
        ),
        "restored_release_fingerprint_matches": restored_release == release.release_fingerprint,
        "action_interface_mode_matches": action_mode == expected_action_mode,
        "graph_nmf_contract_matches": graph_actual is graph_expected,
        "bare_racket_portable_core_matches": racket_core == bare_core == expected_core,
        "gpu_backend": jax.default_backend() == "gpu",
        "three_updates_eight_envs": num_updates == 3 and num_envs == 8,
    }
    for name, passed in checks.items():
        if not passed:
            errors.append(f"failed check: {name}")

    contract_payload = {
        "release_fingerprint": release.release_fingerprint,
        "taxonomy_fingerprint": release.taxonomy["taxonomy_fingerprint"],
        "diagnostic_graph_fingerprint": release.diagnostic_graph["graph_fingerprint"],
        "candidate_graph_fingerprint": release.candidate_graph["graph_fingerprint"],
        "loss_spec_fingerprint": release.loss_spec["loss_spec_fingerprint"],
        "calibration_fingerprint": release.calibration["calibration_fingerprint"],
        "muscle_channel_core_fingerprint": expected_core,
        "global_chain_count": expected_global_chains,
        "global_edge_count": expected_global_edges,
        "target_chain_count": expected_target_chains,
        "target_edge_count": expected_target_edges,
        "action_mode": action_mode,
        "basis_family": basis_family,
        "basis_fingerprint": basis_fingerprint,
        "residual_basis_fingerprint": (
            None if body_contract is None else body_contract.get("residual_basis_fingerprint")
        ),
        "basis_factor_contract_fingerprint": experiment.get("action_representation", {}).get(
            "expected_basis_factor_contract_fingerprint",
            None,
        ),
        "graph_regularization_lineage_fingerprint": experiment.get("action_representation", {}).get(
            "expected_graph_regularization_lineage_fingerprint",
            None,
        ),
    }
    created_at = str(smoke.get("started_at_utc", "") or _utc_now())
    payload: dict[str, Any] = {
        "schema_version": CONTINUITY_TRAINING_SMOKE_SCHEMA_VERSION,
        "created_at_utc": created_at,
        "completed_at_utc": _utc_now(),
        "git_commit_sha": repository_git_commit(repo_root, require_clean=False),
        "formal_config": {
            "config_name": str(smoke.get("formal_config_name", "")),
            "resolved_config_sha256": str(smoke.get("formal_resolved_config_sha256", "")),
            "condition": condition,
            "seed": int(ablation.get("seed", -1)),
            "run_id": str(smoke.get("formal_run_id", "")),
        },
        "runtime": {
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "cuda_visible_devices": str(__import__("os").environ.get("CUDA_VISIBLE_DEVICES", "")),
            "environment_class": type(base).__name__,
            "disable_fingers": bool(experiment.env_params.get("disable_fingers", False)),
            "ordered_muscle_channels": len(layout.actuator_names),
            "activation_address_count": int(np.unique(layout.activation_addresses).size),
            "ctrlrange_min": float(np.min(layout.ctrlrange)),
            "ctrlrange_max": float(np.max(layout.ctrlrange)),
            "muscle_channel_core_fingerprint": bare_core,
            "racket_muscle_channel_core_fingerprint": racket_core,
        },
        "contracts": contract_payload,
        "execution": {
            "num_updates": num_updates,
            "num_envs": num_envs,
            "num_steps": num_steps,
            "total_timesteps": int(experiment.total_timesteps),
            "checkpoint_dir": str(Path(checkpoint_dir).resolve()),
            "checkpoint_path": latest_checkpoint,
            "restored_update_number": restored_update_number,
            "restored_release_fingerprint": restored_release,
            "restored_body_contract_matches": restored_body_contract_matches,
            "final_optimizer_step": int(optimizer_step[-1]),
        },
        "checks": checks,
        "measurements": {
            "reward_total": reward_total.tolist(),
            "continuity_global_loss": global_loss.tolist(),
            "continuity_global_chain_count": global_chains.tolist(),
            "continuity_global_edge_count": global_edges.tolist(),
            "continuity_target_loss": target_loss.tolist(),
            "continuity_target_chain_count": target_chains.tolist(),
            "continuity_target_edge_count": target_edges.tolist(),
            "penalty_continuity_raw": penalty_raw.tolist(),
            "penalty_continuity_after_local_clip": penalty_local.tolist(),
            "penalty_continuity_effective_after_total_clip": penalty_effective.tolist(),
            "continuity_penalty_masked_fraction": masked_fraction.tolist(),
            "ppo_total_loss": total_loss.tolist(),
            "gradient_l2_norm": gradient_norm.tolist(),
            "gradients_all_finite": gradients_finite.tolist(),
            "parameters_all_finite": parameters_finite.tolist(),
            "parameter_update_l2_norm": parameter_delta.tolist(),
            "optimizer_step": optimizer_step.tolist(),
        },
        "errors": errors,
        "passed": not errors,
    }
    payload["artifact_fingerprint"] = continuity_training_smoke_fingerprint(payload)
    return payload


def write_continuity_training_smoke(
    path: str | Path,
    payload: dict[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    destination = Path(path).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite continuity smoke artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def validate_continuity_smoke_launch_gate(
    config: Any,
    *,
    formal_resolved_config_sha256: str,
    repo_root: str | Path,
) -> dict[str, Any] | None:
    """Fail closed on stale or mismatched smoke evidence before PPO starts."""

    experiment = config.experiment
    gate = experiment.get("continuity_smoke_gate", None)
    if gate is None or not bool(gate.get("required", False)):
        return None
    if bool(experiment.get("training_smoke", {}).get("enabled", False)):
        # Generation is safe only under the exact short-run budget enforced by
        # the artifact builder; it cannot be used to bypass a long formal run.
        if int(experiment.num_updates) != 3 or int(experiment.num_envs) != 8:
            raise ValueError("continuity smoke generation must use exactly three updates and eight environments")
        if bool(experiment.get("auto_resume", True)) or experiment.get("resume_from", None) is not None:
            raise ValueError("continuity smoke generation requires a fresh optimizer and no checkpoint resume")
        if bool(experiment.get("promotion", {}).get("auto_stop", False)):
            raise ValueError("continuity smoke generation must disable host promotion early stopping")
        if bool(experiment.get("validation", {}).get("active", True)):
            raise ValueError("continuity smoke generation must disable long validation rollouts")
        _required_sha256(
            experiment.training_smoke.get("formal_resolved_config_sha256", None),
            "training_smoke.formal_resolved_config_sha256",
        )
        return None

    artifact_path = str(gate.get("artifact_path", "") or "").strip()
    if not artifact_path:
        raise ValueError("formal continuity reward training requires MUSCLEMIMIC_CONTINUITY_SMOKE_ARTIFACT")
    resolved_artifact_path = Path(artifact_path).expanduser()
    if not resolved_artifact_path.is_absolute():
        resolved_artifact_path = Path(repo_root).resolve() / resolved_artifact_path
    release_contract, body_contract = _contract_fingerprints(config)
    action_mode = "full_354" if body_contract is None else str(body_contract.get("mode", ""))
    basis_fingerprint = None if body_contract is None else body_contract.get("basis_fingerprint")
    ablation = experiment.get("continuity_ablation", {})
    commit = repository_git_commit(
        repo_root,
        require_clean=bool(gate.get("require_clean_git", True)),
    )
    artifact = load_continuity_training_smoke(resolved_artifact_path)
    return validate_continuity_training_smoke(
        artifact,
        expected_commit_sha=commit,
        expected_resolved_config_sha256=formal_resolved_config_sha256,
        expected_release_fingerprint=release_contract["release_fingerprint"],
        expected_basis_fingerprint=basis_fingerprint,
        expected_action_mode=action_mode,
        expected_condition=str(ablation.get("condition", "")),
        expected_artifact_fingerprint=str(gate.get("expected_artifact_fingerprint", "") or "") or None,
        max_age_hours=float(gate.get("max_age_hours", 24.0)),
    )

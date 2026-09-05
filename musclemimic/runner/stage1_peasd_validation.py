"""Sealed validation evidence for the Stage-1 PEASD-Lite comparison.

The paired promotion gate must never consume hand-entered scalar metrics.  One
evidence document therefore binds the exact held-out evaluation to an immutable
checkpoint identity, its run manifest, the declared train/validation split and
the runtime EMG contract.  The document is deterministic and self-fingerprinted
so it can be copied without weakening its content identity.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from musclemimic.badminton.promotion_artifact import checkpoint_identity, sha256_path
from musclemimic.runner.checkpointing import (
    STAGE1_SOURCE_TREE_INCLUDED_SUFFIXES,
    STAGE1_SOURCE_TREE_SCOPES,
    STAGE1_SOURCE_TREE_SNAPSHOT_SCHEMA_VERSION,
    stage1_source_tree_snapshot,
)

STAGE1_PEASD_VALIDATION_SCHEMA_VERSION = "stage1_peasd_validation_metrics_v1"
STAGE1_PEASD_HELDOUT_SCHEMA_VERSION = "stage1_peasd_heldout_validation_v1"
STAGE1_PEASD_ARM_SCHEMA_VERSION = "stage1_peasd_lite_matched_arm_v1"
STAGE1_PEASD_FIXED_BUDGET_SCHEMA_VERSION = "stage1_peasd_fixed_budget_contract_v1"
STAGE1_PEASD_VALIDATION_HISTORY_SCHEMA_VERSION = "stage1_peasd_validation_history_v1"

STAGE1_PEASD_VALIDATION_METRIC_KEYS = (
    "val_mean_episode_return",
    "val_early_termination_rate",
    "val_frame_coverage",
    "val_err_root_xyz",
    "val_err_root_yaw",
    "val_err_joint_pos",
    "val_err_joint_vel",
    "val_err_rpos",
    "val_activation_energy",
    "val_activation_saturation_fraction",
    "val_action_saturation_fraction",
    "val_action_rate_mean_square",
    "val_activation_rate_mean_square",
    "val_emg_anchor_loss",
    "val_emg_anchor_violation_fraction",
    "val_emg_anchor_mean_abs_deviation",
    "val_emg_anchor_max_abs_deviation",
    "val_emg_anchor_correlation",
    "val_emg_anchor_valid_channel_fraction",
    # This is the arm's reward lookup (shifted only for T4).
    "val_emg_synergy_loss",
    "val_emg_synergy_shape_loss",
    "val_emg_synergy_intensity_loss",
    # These three always use the common, unshifted real-human reference.
    "val_emg_synergy_real_reference_loss",
    "val_emg_synergy_real_reference_shape_loss",
    "val_emg_synergy_real_reference_intensity_loss",
    "val_emg_synergy_real_reference_shape_cosine",
    "val_emg_synergy_real_reference_intensity",
    # Delivered-treatment audit.  These are measured inside the reward graph,
    # not reconstructed from configuration after training.
    "val_emg_anchor_weight",
    "val_emg_synergy_weight",
    "val_emg_curriculum_factor_anchor",
    "val_emg_curriculum_factor_synergy",
    "val_penalty_emg_anchor_raw",
    "val_penalty_emg_anchor_after_local_clip",
    "val_penalty_emg_synergy_raw",
    "val_penalty_emg_synergy_after_local_clip",
    "val_penalty_emg_consistency_after_local_clip",
    "val_penalty_emg_consistency_effective_after_total_clip",
    "val_emg_consistency_penalty_masked_fraction",
    "val_penalty_emg_consistency_effective_after_reward_floor",
    "val_emg_consistency_final_reward_masked_fraction",
)

STAGE1_PEASD_TRAINING_TREATMENT_METRIC_KEYS = (
    "emg_anchor_weight",
    "emg_synergy_weight",
    "emg_curriculum_factor_anchor",
    "emg_curriculum_factor_synergy",
    "penalty_emg_anchor_raw",
    "penalty_emg_anchor_after_local_clip",
    "penalty_emg_synergy_raw",
    "penalty_emg_synergy_after_local_clip",
    "penalty_emg_consistency_after_local_clip",
    "penalty_emg_consistency_effective_after_total_clip",
    "emg_consistency_penalty_masked_fraction",
    "penalty_emg_consistency_effective_after_reward_floor",
    "emg_consistency_final_reward_masked_fraction",
    "emg_anchor_valid_channel_fraction",
    "emg_synergy_real_reference_intensity",
)

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "arm",
        "action_id",
        "tube_action_id",
        "seed",
        "config_hash",
        "run_manifest_path",
        "run_manifest_content_sha256",
        "source_tree_snapshot",
        "checkpoint_identity",
        "training_budget_contract",
        "action_release_contract",
        "numeric_data_qc_contract",
        "training_treatment_samples",
        "training_emg_consistency_runtime_binding_sha256",
        "evaluation_mode",
        "evaluation_emg_consistency_runtime_contract",
        "validation_provenance",
        "metric_semantics",
        "validation_history_path",
        "validation_history_content_sha256",
        "metrics",
        "evidence_fingerprint",
    }
)

_HISTORY_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "arm",
        "action_id",
        "seed",
        "config_hash",
        "run_manifest_path",
        "run_manifest_content_sha256",
        "source_tree_snapshot",
        "training_budget_contract",
        "action_release_contract",
        "numeric_data_qc_contract",
        "training_treatment_samples",
        "validation_provenance",
        "metric_semantics",
        "entries",
        "history_fingerprint",
    }
)
_HELDOUT_KEYS = frozenset(
    {
        "schema_version",
        "semantics",
        "deterministic_policy",
        "run_stats_frozen",
        "start_frame",
        "auto_reset_samples_included",
        "heldout_trajectory_count",
        "eval_seed",
        "train_motion_paths",
        "validation_motion_paths",
        "split_sha256",
    }
)
_ACTION_RELEASE_VALIDATION_CACHE: set[tuple[str, str]] = set()
_SOURCE_TREE_SNAPSHOT_KEYS = frozenset(
    {
        "schema_version",
        "git_sha",
        "scopes",
        "included_suffixes",
        "file_count",
        "worktree_dirty",
        "worktree_status_sha256",
        "source_tree_fingerprint",
        "binding_sha256",
    }
)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _validated_source_tree_snapshot(
    manifest: Mapping[str, Any], *, require_current: bool
) -> dict[str, Any]:
    """Validate the manifest's scoped source snapshot and optionally remeasure it.

    Requiring the current snapshot while writing history/evidence prevents a
    post-training evaluator or reward implementation from silently changing
    underneath the checkpoint.  Read-only validators compare the sealed copy
    to the run manifest so archived evidence remains portable.
    """

    raw = manifest.get("source_tree_snapshot")
    if not isinstance(raw, Mapping):
        raise ValueError("Stage1 run manifest lacks a source-tree snapshot")
    snapshot = dict(raw)
    _require_exact_keys(
        snapshot,
        _SOURCE_TREE_SNAPSHOT_KEYS,
        label="Stage1 source-tree snapshot",
    )
    if snapshot.get("schema_version") != STAGE1_SOURCE_TREE_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("unsupported Stage1 source-tree snapshot schema")
    git_sha = str(snapshot.get("git_sha") or "").strip()
    scopes = snapshot.get("scopes")
    suffixes = snapshot.get("included_suffixes")
    if not git_sha or scopes != list(STAGE1_SOURCE_TREE_SCOPES):
        raise ValueError("Stage1 source-tree snapshot has no Git/scoped source identity")
    if suffixes != sorted(STAGE1_SOURCE_TREE_INCLUDED_SUFFIXES):
        raise ValueError("Stage1 source-tree snapshot included suffixes are invalid")
    if isinstance(snapshot.get("worktree_dirty"), bool) is False:
        raise ValueError("Stage1 source-tree snapshot worktree_dirty must be boolean")
    if isinstance(snapshot.get("file_count"), bool) or int(snapshot.get("file_count", 0)) <= 0:
        raise ValueError("Stage1 source-tree snapshot must cover at least one source file")
    for field in (
        "worktree_status_sha256",
        "source_tree_fingerprint",
        "binding_sha256",
    ):
        _require_sha256(snapshot.get(field), field=f"source_tree_snapshot.{field}")
    unsigned = {key: value for key, value in snapshot.items() if key != "binding_sha256"}
    if snapshot["binding_sha256"] != _canonical_sha256(unsigned):
        raise ValueError("Stage1 source-tree snapshot binding mismatch")
    if require_current:
        current = stage1_source_tree_snapshot()
        if current is None:
            raise ValueError("cannot remeasure the current Stage1 source-tree snapshot")
        if snapshot != current:
            raise ValueError(
                "current Stage1 evaluator/source tree differs from the training run manifest"
            )
    return snapshot


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable: {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return dict(payload)


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], *, label: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise ValueError(f"{label} key mismatch: missing={missing}, unknown={unknown}")


def _require_sha256(value: Any, *, field: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _finite_metric(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite numeric")
    return number


def _experiment_from_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    experiment = manifest.get("experiment_config")
    if not isinstance(experiment, Mapping):
        raise ValueError("Stage1 validation run manifest has no experiment_config")
    return dict(experiment)


def _motion_split(experiment: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    try:
        train = experiment["task_factory"]["params"]["amass_dataset_conf"]["rel_dataset_path"]
        validation = experiment["validation"]["amass_dataset_conf"]["rel_dataset_path"]
    except (KeyError, TypeError) as error:
        raise ValueError("Stage1 validation manifest lacks an explicit train/validation motion split") from error
    if not isinstance(train, list) or not isinstance(validation, list) or not train or not validation:
        raise ValueError("Stage1 train/validation motion paths must be non-empty JSON lists")
    train_paths = [str(item) for item in train]
    validation_paths = [str(item) for item in validation]
    if len(set(train_paths)) != len(train_paths) or len(set(validation_paths)) != len(validation_paths):
        raise ValueError("Stage1 train/validation motion paths must not contain duplicates")
    if set(train_paths) & set(validation_paths):
        raise ValueError("Stage1 train and validation motion paths must be disjoint")
    return train_paths, validation_paths


def _validated_runtime_contract(value: Any) -> dict[str, Any]:
    """Validate one self-bound training or post-hoc EMG runtime contract."""

    if not isinstance(value, Mapping):
        raise ValueError("emg_consistency runtime contract must be a JSON object")
    contract = dict(value)
    if contract.get("schema_version") != "stage1_peasd_lite_runtime_contract_v1":
        raise ValueError("unsupported emg_consistency runtime contract schema")
    supplied = _require_sha256(
        contract.get("binding_sha256"),
        field="emg_consistency_runtime_contract.binding_sha256",
    )
    unsigned = {key: item for key, item in contract.items() if key != "binding_sha256"}
    if supplied != _canonical_sha256(unsigned):
        raise ValueError("emg_consistency_runtime_contract binding is stale")
    return contract


def _runtime_contract(experiment: Mapping[str, Any]) -> dict[str, Any] | None:
    value = experiment.get("emg_consistency_runtime_contract")
    if value is None:
        return None
    return _validated_runtime_contract(value)


def _budget_contract(experiment: Mapping[str, Any]) -> dict[str, Any]:
    value = experiment.get("stage1_peasd_fixed_budget_contract")
    if not isinstance(value, Mapping):
        raise ValueError("run manifest has no Stage1 fixed-budget contract")
    contract = dict(value)
    if contract.get("schema_version") != STAGE1_PEASD_FIXED_BUDGET_SCHEMA_VERSION:
        raise ValueError("unsupported Stage1 fixed-budget contract")
    supplied = _require_sha256(contract.get("binding_sha256"), field="fixed budget binding_sha256")
    unsigned = {key: item for key, item in contract.items() if key != "binding_sha256"}
    if supplied != _canonical_sha256(unsigned):
        raise ValueError("Stage1 fixed-budget contract binding is stale")
    return contract


def _release_file_stat_fingerprint(contract: Mapping[str, Any]) -> str:
    """Cheaply invalidate per-process hash verification when bound files move."""

    repo_root = Path(__file__).resolve().parents[2]
    paths: set[Path] = set()

    def _visit(value: Any, parent_key: str = "") -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                _visit(item, str(key))
        elif isinstance(value, list):
            for item in value:
                _visit(item, parent_key)
        elif isinstance(value, str) and (
            parent_key == "path" or parent_key.endswith("_path")
        ):
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = repo_root / path
            paths.add(path)

    _visit(contract)
    rows = []
    for path in sorted(paths, key=str):
        try:
            stat = path.stat()
            rows.append([str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ino])
        except OSError:
            rows.append([str(path), None, None, None])
    return _canonical_sha256(rows)


def _action_release_contract(
    experiment: Mapping[str, Any],
    *,
    revalidate_external: bool = True,
) -> dict[str, Any]:
    value = experiment.get("stage1_peasd_action_release_contract")
    if not isinstance(value, Mapping):
        raise ValueError("run manifest has no Stage1 action release/QC contract")
    contract = dict(value)
    if contract.get("schema_version") != "musclemimic_action_release_validation_v1":
        raise ValueError("unsupported Stage1 action release/QC contract")
    supplied = _require_sha256(
        contract.get("release_binding_sha256"), field="action release binding_sha256"
    )
    unsigned = {key: item for key, item in contract.items() if key != "release_binding_sha256"}
    if supplied != _canonical_sha256(unsigned) or contract.get("passed") is not True:
        raise ValueError("Stage1 action release/QC contract is stale or failed")
    if revalidate_external:
        cache_key = (supplied, _release_file_stat_fingerprint(contract))
        if cache_key not in _ACTION_RELEASE_VALIDATION_CACHE:
            if contract.get("data_variant") == "raw_smooth_v1_aug100":
                from musclemimic.badminton.aug100_release import (
                    validate_forehand_clear_aug100_release,
                )

                rebuilt = validate_forehand_clear_aug100_release(
                    contract.get("train_motions", ()),
                    contract.get("validation_motions", ()),
                )
            else:
                from musclemimic.badminton.action_release import validate_action_release

                rebuilt = validate_action_release(str(contract.get("action_id", "")))
            if rebuilt != contract:
                raise ValueError("Stage1 action release/QC bytes changed after training preflight")
            _ACTION_RELEASE_VALIDATION_CACHE.add(cache_key)
    return contract


def _numeric_data_qc_contract(experiment: Mapping[str, Any]) -> dict[str, Any]:
    """Require the exact warning-free numeric report produced at preflight.

    The sibling action-release contract binds and re-hashes every source/cache
    byte.  Re-running the expensive array audit for every one of 33 history
    entries adds no drift coverage once those same bytes are unchanged.
    """

    value = experiment.get("stage1_peasd_numeric_data_qc_contract")
    if not isinstance(value, Mapping):
        raise ValueError("run manifest has no Stage1 numeric data-QC contract")
    contract = dict(value)
    if contract.get("schema_version") != "stage1_peasd_numeric_data_qc_contract_v1":
        raise ValueError("unsupported Stage1 numeric data-QC contract")
    supplied = _require_sha256(
        contract.get("binding_sha256"), field="numeric data-QC binding_sha256"
    )
    unsigned = {key: item for key, item in contract.items() if key != "binding_sha256"}
    if supplied != _canonical_sha256(unsigned):
        raise ValueError("Stage1 numeric data-QC contract binding is stale")
    report = contract.get("report")
    if not isinstance(report, Mapping) or report.get("clean_passed") is not True:
        raise ValueError("Stage1 numeric data-QC report is not a warning-free clean pass")
    if contract.get("report_sha256") != _canonical_sha256(report):
        raise ValueError("Stage1 numeric data-QC report hash is stale")

    if contract.get("cache_variant") == "raw_smooth_v1_aug100":
        from musclemimic.badminton.aug100_release import (
            ACTION_ID,
            ACTION_SLUG,
            CACHE_NAMESPACE,
            CACHE_VARIANT,
            EXPECTED_MOTION_COUNT,
            SOURCE_NAMESPACE,
            SOURCE_VARIANT,
        )
        from musclemimic.badminton.aug100_release import (
            REPO_ROOT as AUG100_REPO_ROOT,
        )

        release = _action_release_contract(experiment, revalidate_external=False)
        if (
            contract.get("action_id") != ACTION_ID
            or contract.get("action_slug") != ACTION_SLUG
            or contract.get("source_namespace") != SOURCE_NAMESPACE
            or contract.get("source_variant") != SOURCE_VARIANT
            or contract.get("cache_variant") != CACHE_VARIANT
            or report.get("action") != ACTION_ID
            or report.get("action_slug") != ACTION_SLUG
            or report.get("source_variant") != SOURCE_VARIANT
            or report.get("cache_variant") != CACHE_VARIANT
            or report.get("train_motions") != release.get("train_motions")
            or report.get("validation_motions") != release.get("validation_motions")
            or int(report.get("expected_motion_count", -1)) != EXPECTED_MOTION_COUNT
            or report.get("release_binding_sha256")
            != release.get("release_binding_sha256")
        ):
            raise ValueError("Stage1 Aug100 numeric data-QC contract differs from its release")
        expected_source_dir = (
            AUG100_REPO_ROOT / "datasets" / ACTION_ID / SOURCE_NAMESPACE
        ).resolve()
        expected_cache_dir = (
            AUG100_REPO_ROOT / "datasets" / ACTION_ID / CACHE_NAMESPACE
        ).resolve()
        if (
            Path(str(report.get("resolved_source_dir", ""))).resolve()
            != expected_source_dir
            or Path(str(report.get("resolved_cache_dir", ""))).resolve()
            != expected_cache_dir
        ):
            raise ValueError("Stage1 Aug100 numeric data-QC points to a foreign namespace")
        return contract

    from musclemimic.badminton.action_registry import resolve

    spec = resolve(str(contract.get("action_id", "")))
    source_namespace = str(
        contract.get("source_namespace", f"temp/{contract.get('source_variant', '')}")
    )
    if (
        contract.get("action_slug") != spec.slug
        or source_namespace != spec.source_namespace
        or contract.get("source_variant") != spec.source_variant
        or contract.get("cache_variant") != spec.cache_variant
    ):
        raise ValueError("Stage1 numeric data-QC contract differs from the action registry")
    if (
        report.get("action") != spec.action_id
        or report.get("action_slug") != spec.slug
        or report.get("source_variant")
        not in {spec.source_variant, spec.source_namespace}
        or report.get("cache_variant") != spec.cache_variant
        or report.get("train_motions") != list(spec.train_motions)
        or report.get("validation_motions") != list(spec.val_motions)
        or int(report.get("expected_motion_count", -1)) != len(spec.all_motions)
    ):
        raise ValueError("Stage1 numeric data-QC report differs from the action registry")
    expected_source_dir = (spec.dataset_root / spec.source_namespace).resolve()
    expected_cache_dir = (spec.dataset_root / spec.cache_namespace).resolve()
    if (
        Path(str(report.get("resolved_source_dir", ""))).resolve()
        != expected_source_dir
        or Path(str(report.get("resolved_cache_dir", ""))).resolve()
        != expected_cache_dir
    ):
        raise ValueError("Stage1 numeric data-QC report points to a foreign source/cache namespace")
    return contract


def _stage1_arm_contract(experiment: Mapping[str, Any]) -> dict[str, Any]:
    value = experiment.get("stage1_peasd")
    if not isinstance(value, Mapping):
        raise ValueError("run manifest has no Stage1 PEASD arm contract")
    contract = dict(value)
    if contract.get("schema_version") != STAGE1_PEASD_ARM_SCHEMA_VERSION:
        raise ValueError("run manifest has an unsupported Stage1 PEASD arm contract")
    return contract


def _heldout_provenance(
    experiment: Mapping[str, Any],
    raw_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    train_paths, validation_paths = _motion_split(experiment)
    raw = dict(raw_provenance)
    expected = {
        "schema_version": STAGE1_PEASD_HELDOUT_SCHEMA_VERSION,
        "semantics": "evaluate_all_once_per_heldout_v1",
        "deterministic_policy": True,
        "run_stats_frozen": True,
        "start_frame": 0,
        "auto_reset_samples_included": False,
        "heldout_trajectory_count": len(validation_paths),
        "eval_seed": int(raw.get("eval_seed", 0)),
        "train_motion_paths": train_paths,
        "validation_motion_paths": validation_paths,
    }
    for key in (
        "semantics",
        "deterministic_policy",
        "run_stats_frozen",
        "start_frame",
        "auto_reset_samples_included",
        "heldout_trajectory_count",
    ):
        if raw.get(key) != expected[key]:
            raise ValueError(f"Stage1 validation provenance has invalid {key}")
    split = {
        "train_motion_paths": train_paths,
        "validation_motion_paths": validation_paths,
    }
    return {**expected, "split_sha256": _canonical_sha256(split)}


def _metric_semantics(experiment: Mapping[str, Any]) -> dict[str, Any]:
    reward = experiment.get("env_params", {}).get("reward_params", {})
    margin = float(reward.get("activation_saturation_margin", 0.02))
    if not math.isfinite(margin) or not 0.0 < margin < 1.0:
        raise ValueError("Stage1 activation saturation margin must be in (0,1)")
    return {
        "schema_version": "stage1_peasd_metric_semantics_v1",
        "activation_saturation_fraction": {
            "signal": "ordered_native_mujoco_scalar_muscle_activation_354",
            "range": [0.0, 1.0],
            "upper_threshold_inclusive": 1.0 - margin,
            "zero_is_valid_relaxed_state_not_saturation": True,
        },
        "phase_coordinate": "normalized_trajectory_progress",
        "impact_or_named_phase_inferred": False,
    }


def stage1_peasd_training_treatment_targets(
    experiment: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Return the pre-registered rollout indices sampled from training tensors."""

    raw_budget = experiment.get("stage1_peasd_fixed_budget_contract", {})
    budget = dict(raw_budget) if hasattr(raw_budget, "items") else {}
    curriculum = budget.get("emg_curriculum", {})
    curriculum = dict(curriculum) if hasattr(curriculum, "items") else {}
    num_updates = int(budget.get("num_updates", 0))
    if num_updates <= 0:
        raise ValueError("Stage1 treatment sampling requires a positive fixed budget")
    roles_by_index: dict[int, list[str]] = {}

    def _add(index: int, role: str) -> None:
        if 0 <= index < num_updates:
            roles_by_index.setdefault(index, []).append(role)

    _add(0, "first_rollout")
    _add(int(curriculum.get("start_update", 0)), "curriculum_start")
    _add(int(curriculum.get("full_weight_update", 0)), "full_weight")
    _add(num_updates - 1, "endpoint_rollout")
    return tuple(
        {
            "rollout_update_index": index,
            "roles": roles_by_index[index],
        }
        for index in sorted(roles_by_index)
    )


def build_stage1_peasd_training_treatment_sample(
    *,
    experiment: Mapping[str, Any],
    rollout_update_index: int,
    training_metrics: Any,
) -> dict[str, Any]:
    """Extract one pre-registered sample from a runner training metric tensor."""

    target_by_index = {
        int(item["rollout_update_index"]): item
        for item in stage1_peasd_training_treatment_targets(experiment)
    }
    index = int(rollout_update_index)
    if index not in target_by_index:
        raise ValueError("training-treatment sample is not a pre-registered rollout")
    budget = dict(experiment.get("stage1_peasd_fixed_budget_contract", {}))
    rollout_batch = int(budget.get("rollout_batch_size", 0))

    def _read(key: str) -> float:
        value = (
            training_metrics.get(key)
            if isinstance(training_metrics, Mapping)
            else getattr(training_metrics, key)
        )
        return _finite_metric(value, field=f"training_treatment.{key}")

    return {
        "schema_version": "stage1_peasd_training_treatment_sample_v1",
        "rollout_update_index": index,
        "completed_update": index + 1,
        "global_timestep_after_rollout": (index + 1) * rollout_batch,
        "roles": list(target_by_index[index]["roles"]),
        "metrics": {
            key: _read(key) for key in STAGE1_PEASD_TRAINING_TREATMENT_METRIC_KEYS
        },
    }


def _expected_curriculum_weight(
    update_index: int,
    *,
    maximum: float,
    start_update: int,
    ramp_updates: int,
) -> float:
    if ramp_updates == 0:
        factor = 1.0 if update_index >= start_update else 0.0
    else:
        factor = min(1.0, max(0.0, (update_index - start_update) / ramp_updates))
    return maximum * factor


def _validate_training_treatment_samples(
    samples: Any,
    *,
    experiment: Mapping[str, Any],
    completed_updates: int,
) -> list[dict[str, Any]]:
    if not isinstance(samples, list):
        raise ValueError("Stage1 training_treatment_samples must be a JSON list")
    targets = stage1_peasd_training_treatment_targets(experiment)
    expected_targets = [
        item for item in targets if int(item["rollout_update_index"]) < int(completed_updates)
    ]
    if len(samples) != len(expected_targets):
        raise ValueError(
            "Stage1 training-treatment samples do not cover the pre-registered observed rollouts"
        )
    budget = dict(experiment.get("stage1_peasd_fixed_budget_contract", {}))
    curriculum = dict(budget.get("emg_curriculum", {}))
    rollout_batch = int(budget.get("rollout_batch_size", 0))
    arm = str(budget.get("arm", "") or "").strip().upper()
    anchor_max = float(curriculum.get("anchor_weight_max", 0.0))
    synergy_max = float(curriculum.get("synergy_weight_max", 0.0))
    start_update = int(curriculum.get("start_update", 0))
    ramp_updates = int(curriculum.get("ramp_updates", 0))
    validated: list[dict[str, Any]] = []
    for position, (raw, target) in enumerate(zip(samples, expected_targets, strict=True)):
        if not isinstance(raw, Mapping):
            raise ValueError(f"training_treatment_samples[{position}] must be an object")
        sample = dict(raw)
        expected_keys = {
            "schema_version",
            "rollout_update_index",
            "completed_update",
            "global_timestep_after_rollout",
            "roles",
            "metrics",
        }
        if set(sample) != expected_keys:
            raise ValueError("Stage1 training-treatment sample schema mismatch")
        index = int(target["rollout_update_index"])
        if (
            sample.get("schema_version") != "stage1_peasd_training_treatment_sample_v1"
            or int(sample.get("rollout_update_index", -1)) != index
            or int(sample.get("completed_update", -1)) != index + 1
            or int(sample.get("global_timestep_after_rollout", -1))
            != (index + 1) * rollout_batch
            or sample.get("roles") != target["roles"]
        ):
            raise ValueError("Stage1 training-treatment sample identity/schedule mismatch")
        raw_metrics = sample.get("metrics")
        if not isinstance(raw_metrics, Mapping) or set(raw_metrics) != set(
            STAGE1_PEASD_TRAINING_TREATMENT_METRIC_KEYS
        ):
            raise ValueError("Stage1 training-treatment metric set differs from schema v1")
        metrics = {
            key: _finite_metric(raw_metrics[key], field=f"training_treatment[{position}].{key}")
            for key in STAGE1_PEASD_TRAINING_TREATMENT_METRIC_KEYS
        }
        expected_anchor = _expected_curriculum_weight(
            index,
            maximum=anchor_max,
            start_update=start_update,
            ramp_updates=ramp_updates,
        )
        expected_synergy = _expected_curriculum_weight(
            index,
            maximum=synergy_max,
            start_update=start_update,
            ramp_updates=ramp_updates,
        )
        expected_values = {
            "emg_anchor_weight": expected_anchor,
            "emg_synergy_weight": expected_synergy,
            "emg_curriculum_factor_anchor": (
                expected_anchor / anchor_max if anchor_max > 0.0 else 0.0
            ),
            "emg_curriculum_factor_synergy": (
                expected_synergy / synergy_max if synergy_max > 0.0 else 0.0
            ),
        }
        for key, expected in expected_values.items():
            if not math.isclose(metrics[key], expected, rel_tol=1e-5, abs_tol=1e-7):
                raise ValueError(
                    f"Stage1 training rollout {index} delivered {key}={metrics[key]}, expected {expected}"
                )
        penalty_keys = (
            "penalty_emg_anchor_raw",
            "penalty_emg_anchor_after_local_clip",
            "penalty_emg_synergy_raw",
            "penalty_emg_synergy_after_local_clip",
            "penalty_emg_consistency_after_local_clip",
            "penalty_emg_consistency_effective_after_total_clip",
            "penalty_emg_consistency_effective_after_reward_floor",
        )
        if any(metrics[key] > 1e-8 for key in penalty_keys):
            raise ValueError("Stage1 training EMG penalties must use non-positive reward sign")
        masked = metrics["emg_consistency_penalty_masked_fraction"]
        if not 0.0 <= masked <= 1.0:
            raise ValueError("Stage1 training EMG masking fraction must be in [0,1]")
        final_masked = metrics["emg_consistency_final_reward_masked_fraction"]
        if not 0.0 <= final_masked <= 1.0:
            raise ValueError("Stage1 final reward-floor EMG masking fraction must be in [0,1]")
        fully_exposed = index >= int(curriculum.get("full_weight_update", 0))
        active = anchor_max > 0.0 or synergy_max > 0.0
        if active and fully_exposed and masked >= 1.0:
            raise ValueError("Stage1 fully exposed training EMG treatment was completely masked")
        if active and fully_exposed and final_masked >= 1.0:
            raise ValueError(
                "Stage1 fully exposed training EMG treatment was erased by the final reward floor"
            )
        if arm == "T0":
            zero_keys = (
                *expected_values,
                *penalty_keys,
                "emg_consistency_penalty_masked_fraction",
                "emg_consistency_final_reward_masked_fraction",
            )
            if any(not math.isclose(metrics[key], 0.0, abs_tol=1e-8) for key in zero_keys):
                raise ValueError("T0 training-treatment audit must remain strictly reward-neutral")
        if fully_exposed and anchor_max > 0.0 and metrics["emg_anchor_valid_channel_fraction"] <= 0.0:
            raise ValueError("fully exposed anchor training sample has no valid measured channels")
        if (
            fully_exposed
            and synergy_max > 0.0
            and metrics["emg_synergy_real_reference_intensity"] <= 0.0
        ):
            raise ValueError("fully exposed synergy training sample has no real-reference signal")
        validated.append({**sample, "metrics": metrics})
    return validated


def _validate_endpoint_delivered_treatment(
    metrics: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> None:
    """Prove that the endpoint graph delivered the configured EMG treatment.

    Configuration alone is insufficient: a stale carry, unfinished curriculum,
    or the global penalty floor could silently turn an enabled arm into an
    ineffective control.  The values checked here are emitted by the reward
    graph during the same strict held-out rollout as the scientific metrics.
    """

    anchor_max = float(runtime.get("anchor_weight_max", -1.0))
    synergy_max = float(runtime.get("synergy_weight_max", -1.0))
    if anchor_max < 0.0 or synergy_max < 0.0:
        raise ValueError("Stage1 evaluation runtime has invalid EMG reward maxima")

    observed_anchor = float(metrics["val_emg_anchor_weight"])
    observed_synergy = float(metrics["val_emg_synergy_weight"])
    anchor_factor = float(metrics["val_emg_curriculum_factor_anchor"])
    synergy_factor = float(metrics["val_emg_curriculum_factor_synergy"])
    for label, observed, expected in (
        ("anchor weight", observed_anchor, anchor_max),
        ("synergy weight", observed_synergy, synergy_max),
        ("anchor curriculum factor", anchor_factor, 1.0 if anchor_max > 0.0 else 0.0),
        ("synergy curriculum factor", synergy_factor, 1.0 if synergy_max > 0.0 else 0.0),
    ):
        if not math.isclose(observed, expected, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError(
                f"Stage1 endpoint delivered {label}={observed}, expected {expected}"
            )

    penalty_keys = (
        "val_penalty_emg_anchor_raw",
        "val_penalty_emg_anchor_after_local_clip",
        "val_penalty_emg_synergy_raw",
        "val_penalty_emg_synergy_after_local_clip",
        "val_penalty_emg_consistency_after_local_clip",
        "val_penalty_emg_consistency_effective_after_total_clip",
        "val_penalty_emg_consistency_effective_after_reward_floor",
    )
    if any(float(metrics[key]) > 1e-8 for key in penalty_keys):
        raise ValueError("Stage1 EMG penalty diagnostics must use non-positive reward sign")
    masked = float(metrics["val_emg_consistency_penalty_masked_fraction"])
    if not 0.0 <= masked <= 1.0:
        raise ValueError("Stage1 EMG global-clip masking fraction must be in [0,1]")
    final_masked = float(metrics["val_emg_consistency_final_reward_masked_fraction"])
    if not 0.0 <= final_masked <= 1.0:
        raise ValueError("Stage1 EMG final reward-floor masking fraction must be in [0,1]")

    treatment_active = anchor_max > 0.0 or synergy_max > 0.0
    if treatment_active and masked >= 1.0:
        raise ValueError("Stage1 endpoint EMG treatment was fully masked by the global penalty clip")
    if treatment_active and final_masked >= 1.0:
        raise ValueError("Stage1 endpoint EMG treatment was erased by the final reward floor")
    if not treatment_active:
        for key in penalty_keys:
            if not math.isclose(float(metrics[key]), 0.0, rel_tol=0.0, abs_tol=1e-8):
                raise ValueError("reward-neutral Stage1 evaluation emitted an EMG penalty")
        if not math.isclose(masked, 0.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("reward-neutral Stage1 evaluation emitted EMG clip masking")
        if not math.isclose(final_masked, 0.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("reward-neutral Stage1 evaluation emitted final reward-floor masking")

    if anchor_max > 0.0 and float(metrics["val_emg_anchor_valid_channel_fraction"]) <= 0.0:
        raise ValueError("Stage1 activation-anchor treatment had no valid measured channels")
    if (
        synergy_max > 0.0
        and float(metrics["val_emg_synergy_real_reference_intensity"]) <= 0.0
    ):
        raise ValueError("Stage1 synergy treatment had no non-zero real-reference signal")


def build_stage1_peasd_validation_evidence(
    *,
    checkpoint_identity_payload: Mapping[str, Any],
    validation_provenance: Mapping[str, Any],
    metrics: Mapping[str, Any],
    validation_history: str | Path,
    evaluation_runtime_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build final checkpoint evidence from one strict evaluate-all result.

    T1--T4 evaluate with the runtime frozen in their training manifest.  T0
    must supply a separately compiled ``diagnostics_only`` runtime after the
    verified tube gate; the training manifest remains correctly tube-free.
    """

    identity = dict(checkpoint_identity_payload)
    checkpoint_path = Path(str(identity.get("checkpoint_path", ""))).expanduser().resolve(strict=True)
    rebuilt_identity = checkpoint_identity(checkpoint_path)
    if identity != rebuilt_identity:
        raise ValueError("supplied Stage1 validation checkpoint identity is stale")

    manifest_path = (checkpoint_path.parent / "manifest.json").resolve(strict=True)
    manifest = _load_json_object(manifest_path, label="Stage1 validation run manifest")
    source_snapshot = _validated_source_tree_snapshot(manifest, require_current=True)
    experiment = _experiment_from_manifest(manifest)
    stage1 = _stage1_arm_contract(experiment)
    arm = str(stage1.get("arm", "")).strip().upper()
    if arm not in {"T0", "T1", "T2", "T3", "T4"}:
        raise ValueError("Stage1 validation arm must be T0--T4")

    run_id = str(experiment.get("run_id", "") or "").strip()
    action_id = str(stage1.get("action_id", "") or "").strip()
    if not run_id or not action_id:
        raise ValueError("Stage1 validation requires explicit run_id and dataset action_id")
    if run_id != str(identity.get("run_id", "")):
        raise ValueError("Stage1 validation checkpoint run_id differs from the manifest")
    config_hash = str(manifest.get("config_hash", "") or "")
    if config_hash != str(identity.get("config_hash", "")):
        raise ValueError("Stage1 validation checkpoint config hash differs from the manifest")

    seeds = experiment.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 1:
        raise ValueError("sealed Stage1 validation requires exactly one explicit seed")
    seed = int(seeds[0])

    budget = _budget_contract(experiment)
    release = _action_release_contract(experiment)
    numeric_data_qc = _numeric_data_qc_contract(experiment)
    if (
        budget.get("arm") != arm
        or budget.get("action_id") != action_id
        or int(budget.get("seed", -1)) != seed
        or int(identity.get("update_number", -1))
        != int(budget.get("expected_endpoint_update_number", -2))
        or int(identity.get("global_timestep", -1))
        != int(budget.get("expected_endpoint_global_timestep", -2))
    ):
        raise ValueError("Stage1 validation checkpoint is not the fixed-budget endpoint")

    training_runtime = _runtime_contract(experiment)
    if arm == "T0":
        if training_runtime is not None:
            raise ValueError("T0 training manifest must not bind an EMG reward runtime")
        if evaluation_runtime_contract is None:
            raise ValueError("T0 physiology evidence requires a post-hoc diagnostics-only EMG runtime")
        evaluation_runtime = _validated_runtime_contract(evaluation_runtime_contract)
        if (
            evaluation_runtime.get("arm") != "T0"
            or evaluation_runtime.get("mode") != "diagnostics_only"
            or evaluation_runtime.get("training_signal_enabled") is not False
            or float(evaluation_runtime.get("anchor_weight_max", -1.0)) != 0.0
            or float(evaluation_runtime.get("synergy_weight_max", -1.0)) != 0.0
        ):
            raise ValueError("T0 post-hoc EMG runtime is not diagnostics-only and reward-neutral")
        evaluation_mode = "posthoc_diagnostics_only"
        training_runtime_binding = None
    else:
        if training_runtime is None or training_runtime.get("arm") != arm:
            raise ValueError("active Stage1 validation has no matching EMG reward runtime")
        evaluation_runtime = (
            training_runtime
            if evaluation_runtime_contract is None
            else _validated_runtime_contract(evaluation_runtime_contract)
        )
        if evaluation_runtime != training_runtime:
            raise ValueError("active Stage1 evidence must evaluate its exact training EMG runtime")
        evaluation_mode = "training_runtime"
        training_runtime_binding = _require_sha256(
            training_runtime.get("binding_sha256"),
            field="emg_consistency_runtime_contract.binding_sha256",
        )
    tube_action_id = str(evaluation_runtime.get("action_id", "") or "").strip()
    if tube_action_id != budget.get("tube_action_id"):
        raise ValueError("Stage1 evaluation tube action differs from the fixed action contract")
    if (
        evaluation_runtime.get("reference_review_status") != "verified"
        or evaluation_runtime.get("reference_training_enabled") is not True
        or evaluation_runtime.get("mapping_review_status") != "verified"
        or not evaluation_runtime.get("trial_qc_review_schema_version")
    ):
        raise ValueError("Stage1 physiology evaluation lacks verified reference/mapping/trial-QC identity")
    _require_sha256(
        evaluation_runtime.get("trial_qc_review_sha256"),
        field="evaluation runtime trial_qc_review_sha256",
    )

    heldout = _heldout_provenance(experiment, validation_provenance)
    history_path = Path(validation_history).expanduser().resolve(strict=True)
    history = validate_stage1_peasd_validation_history(history_path, require_complete=True)
    if (
        history.get("run_id") != run_id
        or history.get("config_hash") != config_hash
        or history.get("arm") != arm
        or history.get("validation_provenance") != heldout
        or history.get("source_tree_snapshot") != source_snapshot
        or history["entries"][-1]["checkpoint_identity"] != identity
    ):
        raise ValueError("Stage1 final evidence is bound to a foreign validation history")

    evidence_metrics = {
        key: _finite_metric(metrics.get(key), field=f"metrics.{key}")
        for key in STAGE1_PEASD_VALIDATION_METRIC_KEYS
    }
    _validate_endpoint_delivered_treatment(evidence_metrics, evaluation_runtime)
    training_treatment_samples = _validate_training_treatment_samples(
        history.get("training_treatment_samples"),
        experiment=experiment,
        completed_updates=int(budget["num_updates"]),
    )
    history_metrics = history["entries"][-1]["metrics"]
    shared_metric_keys = (
        STAGE1_PEASD_VALIDATION_METRIC_KEYS
        if arm != "T0"
        else tuple(
            key for key in STAGE1_PEASD_VALIDATION_METRIC_KEYS if not key.startswith("val_emg_")
        )
    )
    if any(evidence_metrics[key] != float(history_metrics[key]) for key in shared_metric_keys):
        raise ValueError(
            "Stage1 final evidence metrics differ from the endpoint history evaluation"
        )
    unsigned = {
        "schema_version": STAGE1_PEASD_VALIDATION_SCHEMA_VERSION,
        "run_id": run_id,
        "arm": arm,
        "action_id": action_id,
        "tube_action_id": tube_action_id,
        "seed": seed,
        "config_hash": config_hash,
        "run_manifest_path": str(manifest_path),
        "run_manifest_content_sha256": sha256_path(manifest_path),
        "source_tree_snapshot": source_snapshot,
        "checkpoint_identity": identity,
        "training_budget_contract": budget,
        "action_release_contract": release,
        "numeric_data_qc_contract": numeric_data_qc,
        "training_treatment_samples": training_treatment_samples,
        "training_emg_consistency_runtime_binding_sha256": training_runtime_binding,
        "evaluation_mode": evaluation_mode,
        "evaluation_emg_consistency_runtime_contract": evaluation_runtime,
        "validation_provenance": heldout,
        "metric_semantics": _metric_semantics(experiment),
        "validation_history_path": str(history_path),
        "validation_history_content_sha256": sha256_path(history_path),
        "metrics": evidence_metrics,
    }
    return {**unsigned, "evidence_fingerprint": _canonical_sha256(unsigned)}


def validate_stage1_peasd_validation_evidence(
    source: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute every content/split/runtime binding in sealed evidence."""

    payload = (
        dict(source)
        if isinstance(source, Mapping)
        else _load_json_object(
            Path(source).expanduser().resolve(strict=True),
            label="Stage1 PEASD validation evidence",
        )
    )
    _require_exact_keys(payload, _TOP_LEVEL_KEYS, label="Stage1 validation evidence")
    if payload.get("schema_version") != STAGE1_PEASD_VALIDATION_SCHEMA_VERSION:
        raise ValueError("unsupported Stage1 validation evidence schema")
    supplied_fingerprint = _require_sha256(
        payload.get("evidence_fingerprint"), field="evidence_fingerprint"
    )
    unsigned = {key: value for key, value in payload.items() if key != "evidence_fingerprint"}
    if supplied_fingerprint != _canonical_sha256(unsigned):
        raise ValueError("Stage1 validation evidence fingerprint mismatch")

    identity = payload.get("checkpoint_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("Stage1 validation evidence lacks checkpoint_identity")
    rebuilt_identity = checkpoint_identity(str(identity.get("checkpoint_path", "")))
    if dict(identity) != rebuilt_identity:
        raise ValueError("Stage1 validation checkpoint identity changed")
    manifest_path = Path(str(payload.get("run_manifest_path", ""))).expanduser().resolve(strict=True)
    expected_manifest_path = (Path(rebuilt_identity["checkpoint_dir"]) / "manifest.json").resolve(strict=True)
    if manifest_path != expected_manifest_path:
        raise ValueError("Stage1 validation evidence points to a foreign run manifest")
    actual_manifest_sha = sha256_path(manifest_path)
    if payload.get("run_manifest_content_sha256") != actual_manifest_sha:
        raise ValueError("Stage1 validation run manifest content changed")
    if rebuilt_identity.get("run_manifest_content_sha256") != actual_manifest_sha:
        raise ValueError("checkpoint identity and Stage1 evidence disagree on the run manifest")

    manifest = _load_json_object(manifest_path, label="Stage1 validation run manifest")
    source_snapshot = _validated_source_tree_snapshot(manifest, require_current=False)
    if payload.get("source_tree_snapshot") != source_snapshot:
        raise ValueError("Stage1 validation evidence source snapshot differs from its manifest")
    experiment = _experiment_from_manifest(manifest)
    stage1 = _stage1_arm_contract(experiment)
    arm = str(stage1.get("arm", "")).strip().upper()
    if payload.get("arm") != arm or payload.get("action_id") != stage1.get("action_id"):
        raise ValueError("Stage1 validation evidence arm/action differs from the run manifest")
    if payload.get("run_id") != rebuilt_identity.get("run_id"):
        raise ValueError("Stage1 validation evidence run_id differs from the checkpoint")
    if payload.get("config_hash") != rebuilt_identity.get("config_hash"):
        raise ValueError("Stage1 validation evidence config_hash differs from the checkpoint")
    seeds = experiment.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 1 or int(seeds[0]) != int(payload.get("seed", -1)):
        raise ValueError("Stage1 validation evidence seed differs from the run manifest")

    budget = _budget_contract(experiment)
    release = _action_release_contract(experiment)
    numeric_data_qc = _numeric_data_qc_contract(experiment)
    if payload.get("training_budget_contract") != budget:
        raise ValueError("Stage1 evidence fixed-budget contract differs from the manifest")
    if payload.get("action_release_contract") != release:
        raise ValueError("Stage1 evidence action release/QC contract differs from the manifest")
    if payload.get("numeric_data_qc_contract") != numeric_data_qc:
        raise ValueError("Stage1 evidence numeric data-QC contract differs from the manifest")
    if (
        int(rebuilt_identity.get("update_number", -1))
        != int(budget.get("expected_endpoint_update_number", -2))
        or int(rebuilt_identity.get("global_timestep", -1))
        != int(budget.get("expected_endpoint_global_timestep", -2))
    ):
        raise ValueError("Stage1 validation evidence is not the fixed-budget endpoint")

    training_runtime = _runtime_contract(experiment)
    evaluation_runtime = _validated_runtime_contract(
        payload.get("evaluation_emg_consistency_runtime_contract")
    )
    if arm == "T0":
        if training_runtime is not None:
            raise ValueError("T0 training manifest unexpectedly binds an EMG runtime")
        if payload.get("training_emg_consistency_runtime_binding_sha256") is not None:
            raise ValueError("T0 evidence unexpectedly binds a training EMG runtime hash")
        if (
            payload.get("evaluation_mode") != "posthoc_diagnostics_only"
            or evaluation_runtime.get("arm") != "T0"
            or evaluation_runtime.get("mode") != "diagnostics_only"
            or evaluation_runtime.get("training_signal_enabled") is not False
            or float(evaluation_runtime.get("anchor_weight_max", -1.0)) != 0.0
            or float(evaluation_runtime.get("synergy_weight_max", -1.0)) != 0.0
        ):
            raise ValueError("T0 physiology evidence did not use a reward-neutral diagnostic runtime")
    else:
        if training_runtime is None or training_runtime.get("arm") != arm:
            raise ValueError("active Stage1 validation runtime differs from the arm")
        if payload.get("evaluation_mode") != "training_runtime" or evaluation_runtime != training_runtime:
            raise ValueError("active Stage1 evidence did not use its training EMG runtime")
        if payload.get("training_emg_consistency_runtime_binding_sha256") != training_runtime.get(
            "binding_sha256"
        ):
            raise ValueError("Stage1 validation runtime binding differs from the manifest")
    if payload.get("tube_action_id") != evaluation_runtime.get("action_id"):
        raise ValueError("Stage1 validation tube action differs from the evaluation runtime")
    if (
        payload.get("tube_action_id") != budget.get("tube_action_id")
        or evaluation_runtime.get("reference_review_status") != "verified"
        or evaluation_runtime.get("reference_training_enabled") is not True
        or evaluation_runtime.get("mapping_review_status") != "verified"
        or not evaluation_runtime.get("trial_qc_review_schema_version")
    ):
        raise ValueError("Stage1 physiology evidence lacks its verified reference/QC identity")
    _require_sha256(
        evaluation_runtime.get("trial_qc_review_sha256"),
        field="evaluation runtime trial_qc_review_sha256",
    )

    provenance = payload.get("validation_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Stage1 validation evidence lacks validation_provenance")
    _require_exact_keys(provenance, _HELDOUT_KEYS, label="Stage1 validation provenance")
    train_paths, validation_paths = _motion_split(experiment)
    expected_split = {
        "train_motion_paths": train_paths,
        "validation_motion_paths": validation_paths,
    }
    if provenance.get("schema_version") != STAGE1_PEASD_HELDOUT_SCHEMA_VERSION:
        raise ValueError("unsupported Stage1 held-out validation schema")
    if (
        provenance.get("semantics") != "evaluate_all_once_per_heldout_v1"
        or provenance.get("deterministic_policy") is not True
        or provenance.get("run_stats_frozen") is not True
        or int(provenance.get("start_frame", -1)) != 0
        or provenance.get("auto_reset_samples_included") is not False
        or int(provenance.get("heldout_trajectory_count", -1)) != len(validation_paths)
        or provenance.get("train_motion_paths") != train_paths
        or provenance.get("validation_motion_paths") != validation_paths
        or provenance.get("split_sha256") != _canonical_sha256(expected_split)
    ):
        raise ValueError("Stage1 validation provenance is not the manifest's strict held-out split")
    if payload.get("metric_semantics") != _metric_semantics(experiment):
        raise ValueError("Stage1 validation metric semantics differ from the run manifest")

    history_path = Path(str(payload.get("validation_history_path", ""))).expanduser().resolve(strict=True)
    if payload.get("validation_history_content_sha256") != sha256_path(history_path):
        raise ValueError("Stage1 validation history content changed")
    history = validate_stage1_peasd_validation_history(history_path, require_complete=True)
    if (
        history.get("run_id") != payload.get("run_id")
        or history.get("config_hash") != payload.get("config_hash")
        or history.get("validation_provenance") != dict(provenance)
        or history.get("source_tree_snapshot") != source_snapshot
        or history["entries"][-1]["checkpoint_identity"] != rebuilt_identity
    ):
        raise ValueError("Stage1 evidence validation history belongs to another run/split")
    if payload.get("training_treatment_samples") != history.get(
        "training_treatment_samples"
    ):
        raise ValueError("Stage1 evidence training-treatment audit differs from its history")
    _validate_training_treatment_samples(
        payload.get("training_treatment_samples"),
        experiment=experiment,
        completed_updates=int(budget["num_updates"]),
    )

    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("Stage1 validation evidence metrics must be a JSON object")
    if set(metrics) != set(STAGE1_PEASD_VALIDATION_METRIC_KEYS):
        raise ValueError("Stage1 validation evidence metric set differs from schema v1")
    for key in STAGE1_PEASD_VALIDATION_METRIC_KEYS:
        _finite_metric(metrics[key], field=f"metrics.{key}")
    _validate_endpoint_delivered_treatment(metrics, evaluation_runtime)
    history_metrics = history["entries"][-1]["metrics"]
    shared_metric_keys = (
        STAGE1_PEASD_VALIDATION_METRIC_KEYS
        if arm != "T0"
        else tuple(
            key for key in STAGE1_PEASD_VALIDATION_METRIC_KEYS if not key.startswith("val_emg_")
        )
    )
    if any(float(metrics[key]) != float(history_metrics[key]) for key in shared_metric_keys):
        raise ValueError(
            "Stage1 final evidence metrics differ from the endpoint history evaluation"
        )
    return payload


def validate_stage1_peasd_validation_history(
    source: str | Path | Mapping[str, Any],
    *,
    require_complete: bool = False,
    revalidate_external: bool = True,
) -> dict[str, Any]:
    """Validate a cumulative, checkpoint-aligned strict-evaluation history."""

    payload = (
        dict(source)
        if isinstance(source, Mapping)
        else _load_json_object(
            Path(source).expanduser().resolve(strict=True),
            label="Stage1 PEASD validation history",
        )
    )
    _require_exact_keys(payload, _HISTORY_TOP_LEVEL_KEYS, label="Stage1 validation history")
    if payload.get("schema_version") != STAGE1_PEASD_VALIDATION_HISTORY_SCHEMA_VERSION:
        raise ValueError("unsupported Stage1 validation history schema")
    supplied = _require_sha256(payload.get("history_fingerprint"), field="history_fingerprint")
    unsigned = {key: value for key, value in payload.items() if key != "history_fingerprint"}
    if supplied != _canonical_sha256(unsigned):
        raise ValueError("Stage1 validation history fingerprint mismatch")

    manifest_path = Path(str(payload.get("run_manifest_path", ""))).expanduser().resolve(strict=True)
    if payload.get("run_manifest_content_sha256") != sha256_path(manifest_path):
        raise ValueError("Stage1 validation history run manifest content changed")
    manifest = _load_json_object(manifest_path, label="Stage1 validation run manifest")
    source_snapshot = _validated_source_tree_snapshot(manifest, require_current=False)
    if payload.get("source_tree_snapshot") != source_snapshot:
        raise ValueError("Stage1 validation history source snapshot differs from its manifest")
    experiment = _experiment_from_manifest(manifest)
    stage1 = _stage1_arm_contract(experiment)
    budget = _budget_contract(experiment)
    release = _action_release_contract(
        experiment,
        revalidate_external=revalidate_external,
    )
    numeric_data_qc = _numeric_data_qc_contract(experiment)
    if payload.get("training_budget_contract") != budget:
        raise ValueError("Stage1 validation history fixed budget differs from the manifest")
    if payload.get("action_release_contract") != release:
        raise ValueError("Stage1 validation history release/QC differs from the manifest")
    if payload.get("numeric_data_qc_contract") != numeric_data_qc:
        raise ValueError("Stage1 validation history numeric data-QC differs from the manifest")
    if (
        payload.get("run_id") != experiment.get("run_id")
        or payload.get("arm") != stage1.get("arm")
        or payload.get("action_id") != stage1.get("action_id")
        or payload.get("config_hash") != manifest.get("config_hash")
        or int(payload.get("seed", -1)) != int(budget.get("seed", -2))
    ):
        raise ValueError("Stage1 validation history identity differs from its manifest")
    provenance = payload.get("validation_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Stage1 validation history lacks held-out provenance")
    _require_exact_keys(provenance, _HELDOUT_KEYS, label="Stage1 history provenance")
    if dict(provenance) != _heldout_provenance(experiment, provenance):
        raise ValueError("Stage1 validation history uses a foreign held-out split")
    if payload.get("metric_semantics") != _metric_semantics(experiment):
        raise ValueError("Stage1 validation history metric semantics differ from the run manifest")

    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Stage1 validation history must contain at least one entry")
    interval = int(budget.get("validation_interval_updates", 0))
    rollout_batch = int(budget.get("rollout_batch_size", 0))
    scheduled_count = int(budget.get("scheduled_validation_count", 0))
    expected_count = int(budget.get("expected_history_count", 0))
    if interval <= 0 or rollout_batch <= 0 or scheduled_count <= 0 or expected_count <= 0:
        raise ValueError("Stage1 validation history has an invalid fixed schedule")
    expected_updates = [interval * index for index in range(1, scheduled_count + 1)]
    endpoint_update = int(budget.get("expected_endpoint_update_number", -1))
    if not expected_updates or expected_updates[-1] != endpoint_update:
        expected_updates.append(endpoint_update)
    if len(expected_updates) != expected_count:
        raise ValueError("Stage1 validation history count differs from its fixed-budget schedule")
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, Mapping) or set(raw_entry) != {"checkpoint_identity", "metrics"}:
            raise ValueError("Stage1 validation history entry schema mismatch")
        identity = raw_entry.get("checkpoint_identity")
        metrics = raw_entry.get("metrics")
        if not isinstance(identity, Mapping) or not isinstance(metrics, Mapping):
            raise ValueError("Stage1 validation history entry must bind identity and metrics")
        checkpoint_path = Path(str(identity.get("checkpoint_path", ""))).expanduser()
        if checkpoint_path.exists():
            rebuilt = checkpoint_identity(checkpoint_path)
            if dict(identity) != rebuilt:
                raise ValueError("Stage1 validation history checkpoint identity changed")
        else:
            # Retention may prune early Orbax leaves after their identity was
            # sealed.  Preserve their hashes in the cumulative document while
            # requiring the final endpoint leaf below to remain recomputable.
            rebuilt = dict(identity)
            for field in (
                "checkpoint_content_sha256",
                "metadata_content_sha256",
                "run_manifest_content_sha256",
            ):
                _require_sha256(rebuilt.get(field), field=f"history[{index}].{field}")
        if index >= len(expected_updates):
            raise ValueError("Stage1 validation history exceeds the fixed schedule")
        expected_update = expected_updates[index]
        if (
            int(rebuilt.get("update_number", -1)) != expected_update
            or int(rebuilt.get("global_timestep", -1)) != expected_update * rollout_batch
            or rebuilt.get("config_hash") != payload.get("config_hash")
            or rebuilt.get("run_id") != payload.get("run_id")
        ):
            raise ValueError("Stage1 validation history entry is off the fixed schedule")
        if set(metrics) != set(STAGE1_PEASD_VALIDATION_METRIC_KEYS):
            raise ValueError("Stage1 validation history metric set differs from schema v1")
        for key in STAGE1_PEASD_VALIDATION_METRIC_KEYS:
            _finite_metric(metrics[key], field=f"history[{index}].metrics.{key}")
    if len(entries) > expected_count:
        raise ValueError("Stage1 validation history exceeds the fixed schedule")
    completed_for_treatment = int(entries[-1]["checkpoint_identity"]["update_number"])
    _validate_training_treatment_samples(
        payload.get("training_treatment_samples"),
        experiment=experiment,
        completed_updates=completed_for_treatment,
    )
    if require_complete:
        final_identity = entries[-1]["checkpoint_identity"]
        if (
            len(entries) != expected_count
            or int(final_identity["update_number"])
            != int(budget.get("expected_endpoint_update_number", -1))
            or int(final_identity["global_timestep"])
            != int(budget.get("expected_endpoint_global_timestep", -1))
        ):
            raise ValueError("Stage1 validation history is incomplete at the fixed-budget endpoint")
        if checkpoint_identity(str(final_identity.get("checkpoint_path", ""))) != dict(final_identity):
            raise ValueError("Stage1 final validation checkpoint identity changed")
    return payload


def append_stage1_peasd_validation_history(
    *,
    checkpoint_identity_payload: Mapping[str, Any],
    validation_provenance: Mapping[str, Any],
    metrics: Mapping[str, Any],
    training_treatment_samples: list[Mapping[str, Any]],
) -> tuple[dict[str, Any], Path, Path]:
    """Append one strict evaluate-all boundary and seal a cumulative snapshot."""

    identity = dict(checkpoint_identity_payload)
    rebuilt = checkpoint_identity(str(identity.get("checkpoint_path", "")))
    if identity != rebuilt:
        raise ValueError("supplied Stage1 history checkpoint identity is stale")
    checkpoint_dir = Path(str(identity["checkpoint_dir"])).resolve(strict=True)
    manifest_path = (checkpoint_dir / "manifest.json").resolve(strict=True)
    manifest = _load_json_object(manifest_path, label="Stage1 validation run manifest")
    source_snapshot = _validated_source_tree_snapshot(manifest, require_current=True)
    experiment = _experiment_from_manifest(manifest)
    stage1 = _stage1_arm_contract(experiment)
    budget = _budget_contract(experiment)
    release = _action_release_contract(experiment, revalidate_external=False)
    numeric_data_qc = _numeric_data_qc_contract(experiment)
    heldout = _heldout_provenance(experiment, validation_provenance)
    entry_metrics = {
        key: _finite_metric(metrics.get(key), field=f"metrics.{key}")
        for key in STAGE1_PEASD_VALIDATION_METRIC_KEYS
    }
    latest = checkpoint_dir / "stage1_peasd_validation_history.json"
    entries: list[dict[str, Any]] = []
    normalized_treatment_samples = [dict(item) for item in training_treatment_samples]
    if latest.exists():
        previous = validate_stage1_peasd_validation_history(
            latest,
            revalidate_external=False,
        )
        for key, expected in (
            ("run_id", experiment.get("run_id")),
            ("arm", stage1.get("arm")),
            ("action_id", stage1.get("action_id")),
            ("config_hash", manifest.get("config_hash")),
            ("validation_provenance", heldout),
            ("source_tree_snapshot", source_snapshot),
        ):
            if previous.get(key) != expected:
                raise ValueError(f"existing Stage1 validation history differs in {key}")
        entries = [dict(item) for item in previous["entries"]]
        previous_samples = previous.get("training_treatment_samples")
        if normalized_treatment_samples[: len(previous_samples)] != previous_samples:
            raise ValueError("Stage1 training-treatment audit does not extend its sealed prefix")
    new_entry = {"checkpoint_identity": identity, "metrics": entry_metrics}
    if entries and int(entries[-1]["checkpoint_identity"]["update_number"]) == int(
        identity["update_number"]
    ):
        if entries[-1] != new_entry:
            raise ValueError("refusing to replace a different Stage1 validation boundary")
    else:
        entries.append(new_entry)

    unsigned = {
        "schema_version": STAGE1_PEASD_VALIDATION_HISTORY_SCHEMA_VERSION,
        "run_id": str(experiment["run_id"]),
        "arm": str(stage1["arm"]),
        "action_id": str(stage1["action_id"]),
        "seed": int(budget["seed"]),
        "config_hash": str(manifest["config_hash"]),
        "run_manifest_path": str(manifest_path),
        "run_manifest_content_sha256": sha256_path(manifest_path),
        "source_tree_snapshot": source_snapshot,
        "training_budget_contract": budget,
        "action_release_contract": release,
        "numeric_data_qc_contract": numeric_data_qc,
        "training_treatment_samples": normalized_treatment_samples,
        "validation_provenance": heldout,
        "metric_semantics": _metric_semantics(experiment),
        "entries": entries,
    }
    payload = {**unsigned, "history_fingerprint": _canonical_sha256(unsigned)}
    validate_stage1_peasd_validation_history(payload, revalidate_external=False)
    update_number = int(identity["update_number"])
    versioned = checkpoint_dir / "stage1_peasd_validation_history" / f"checkpoint_{update_number}.json"
    if versioned.exists():
        if validate_stage1_peasd_validation_history(
            versioned,
            revalidate_external=False,
        ) != payload:
            raise ValueError("refusing to overwrite a different sealed Stage1 validation history")
    else:
        _atomic_write(versioned, payload)
    _atomic_write(latest, payload)
    return payload, versioned, latest


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_stage1_peasd_validation_evidence(
    payload: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Seal a versioned document and atomically update the run-local latest copy."""

    validated = validate_stage1_peasd_validation_evidence(payload)
    identity = validated["checkpoint_identity"]
    checkpoint_dir = Path(str(identity["checkpoint_dir"])).resolve(strict=True)
    update_number = int(identity["update_number"])
    versioned = (
        checkpoint_dir
        / "stage1_peasd_validation_metrics"
        / f"checkpoint_{update_number}.json"
    )
    latest = checkpoint_dir / "stage1_peasd_validation_metrics.json"
    if versioned.exists():
        existing = validate_stage1_peasd_validation_evidence(versioned)
        if existing != validated:
            raise ValueError("refusing to overwrite different sealed Stage1 validation evidence")
    else:
        _atomic_write(versioned, validated)
    _atomic_write(latest, validated)
    return versioned, latest

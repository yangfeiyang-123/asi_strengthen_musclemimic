"""Fail-closed gates for the Stage-1 PEASD-Lite matched ablation.

T0 is a tube-independent training baseline.  T1--T4 may start only after the
action-owned EMG tube is independently reviewed.  Promotion consumes sealed
validation evidence produced by the runner, never scalar metrics copied into a
comparison JSON.  T4 is a deterministic half-cycle shift of synergy phase
bins; it never fabricates or shuffles task events.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from musclemimic.badminton.action_registry import ActionSpec, resolve
from musclemimic.badminton.json_contract import load_json_strict
from musclemimic.badminton.promotion_artifact import checkpoint_identity
from musclemimic.badminton.training_gates import (
    CANONICAL_PROMOTION_THRESHOLDS,
    validate_promotion_threshold_config,
)
from musclemimic.physiology.emg_reference import (
    load_emg_phase_reference_tube,
    resolve_emg_reference_reward_gate,
)
from musclemimic.runner.stage1_peasd_validation import (
    STAGE1_PEASD_VALIDATION_SCHEMA_VERSION,
    validate_stage1_peasd_validation_evidence,
    validate_stage1_peasd_validation_history,
)

TUBE_GATE_SCHEMA_VERSION = "stage1_peasd_verified_tube_gate_v1"
PAIRWISE_METRICS_SCHEMA_VERSION = "stage1_peasd_pairwise_evidence_index_v2"
PAIRWISE_GATE_SCHEMA_VERSION = "stage1_peasd_pairwise_promotion_gate_v2"
PAIRWISE_COMPARISON_SCHEMA_VERSION = "stage1_peasd_pairwise_comparison_v2"
PEASD_TEACHER_PROMOTION_SCHEMA_VERSION = "stage1_peasd_teacher_promotion_v1"
PEASD_BLIND_VISUAL_EVIDENCE_SCHEMA_VERSION = (
    "stage1_peasd_blind_visual_evidence_v1"
)
RUNTIME_CONTRACT_SCHEMA_VERSION = "stage1_peasd_lite_runtime_contract_v1"
STAGE1_ARM_SCHEMA_VERSION = "stage1_peasd_lite_matched_arm_v1"

CANONICAL_ARMS = ("T0", "T1", "T2", "T3", "T4")
CANONICAL_SEEDS = (0, 1, 2)
PRIMARY_TEACHER_ARM = "T3"
PRIMARY_TEACHER_SEED = 0
PRIMARY_TEACHER_SELECTION_RULE = "pre_registered_t3_seed_0_v1"
MIN_RELATIVE_REAL_OVER_SHIFTED_IMPROVEMENT = 0.05
MAX_RELATIVE_KINEMATIC_DEGRADATION = 0.10
MAX_RELATIVE_CONTROL_EFFORT_DEGRADATION = 0.10
MAX_ABSOLUTE_SAFETY_RATE_INCREASE = 0.02
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONFIG_HASH = re.compile(r"^[0-9a-f]{12,64}$")
REPO_ROOT = Path(__file__).resolve().parents[2]

# These fields are invariant across every active treatment arm.  Treatment
# identity, active weights and the T4 offset are checked separately.
_MATCHED_ACTIVE_RUNTIME_FIELDS = (
    "action_id",
    "action_index",
    "reference_id",
    "reference_fingerprint",
    "array_bundle_sha256",
    "reference_review_status",
    "reference_training_enabled",
    "trial_qc_review_schema_version",
    "trial_qc_review_sha256",
    "mapping_id",
    "mapping_sha256",
    "mapping_review_status",
    "phase_coordinate",
    "signal",
    "phase_bin_count",
    "channel_count",
    "synergy_count",
    "ordered_actuator_count",
    "actuator_schema_hash",
    "runtime_model_hash",
    "muscle_channel_core_fingerprint",
    "anchor_loss_spec_fingerprint",
    "anchor_max_penalty_each",
    "synergy_max_penalty_each",
    "start_update",
    "ramp_updates",
    "tube_kappa",
    "huber_delta",
    "synergy_shape_weight",
    "synergy_intensity_weight",
)

_EMG_CONFIG_TREATMENT_FIELDS = frozenset(
    {
        "enabled",
        "arm",
        "mode",
        "reference_cache",
        "mapping_path",
        "expected_reference_fingerprint",
        "expected_mapping_sha256",
        "anchor_weight_max",
        "synergy_weight_max",
        "synergy_phase_shuffle_offset_bins",
    }
)
_DYNAMIC_EXPERIMENT_FIELDS = frozenset(
    {
        "emg_consistency_preflight_contract",
        "emg_consistency_runtime_contract",
        "stage1_peasd_action_release_contract",
        "stage1_peasd_numeric_data_qc_contract",
        "stage1_peasd_fixed_budget_contract",
    }
)
# Keep this explicit projection aligned with runner.checkpointing's training
# config-hash contract.  These paths/control flags are injected or rewritten
# per launch and do not alter the scientific treatment.
_TRAINING_HASH_VOLATILE_FIELDS = frozenset(
    {
        "resume_from",
        "reset_logging_timestep",
        "checkpoint_dir",
        "checkpoint_root",
        "training_root",
        "validation_video_dir",
        "run_id",
        "auto_resume",
        "resume_lr_override",
        "extend_completed_run",
    }
)

_LOWER_IS_BETTER_TRACKING_METRICS = (
    "val_err_root_xyz",
    "val_err_root_yaw",
    "val_err_joint_pos",
    "val_err_joint_vel",
    "val_err_rpos",
    "val_early_termination_rate",
    "val_activation_energy",
    "val_activation_saturation_fraction",
    "val_action_saturation_fraction",
    "val_action_rate_mean_square",
    "val_activation_rate_mean_square",
)
_HIGHER_IS_BETTER_TRACKING_METRICS = (
    "val_frame_coverage",
)
_DESCRIPTIVE_CROSS_ARM_METRICS = ("val_mean_episode_return",)
_MEASURED_ACTIVATION_METRICS = (
    "val_emg_anchor_loss",
    "val_emg_anchor_violation_fraction",
    "val_emg_anchor_mean_abs_deviation",
    "val_emg_anchor_max_abs_deviation",
    "val_emg_anchor_correlation",
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return dict(value)


def _require_sha256(value: Any, *, field: str) -> str:
    text = str(value or "")
    if not _SHA256.fullmatch(text):
        raise ValueError(f"{field} must be a lowercase sha256 digest")
    return text


def _finite_nonnegative(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite non-negative numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{field} must be finite non-negative numeric")
    return number


def _finite_numeric(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite numeric")
    return number


def _resolve_source_path(value: str | Path, *, relative_to: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and relative_to is not None:
        path = relative_to / path
    return path.resolve(strict=True)


def _tube_manifest_path(value: str | Path) -> Path:
    source = _resolve_source_path(value)
    return source if source.is_file() else (source / "emg_reference_manifest.json").resolve(strict=True)


def _atomic_write(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, output)
    return output


def _validate_tube_action(spec: ActionSpec, action_ids: tuple[str, ...]) -> None:
    resolved = {resolve(action_id).slug for action_id in action_ids}
    if resolved != {spec.slug}:
        raise ValueError(
            f"EMG reference actions {list(action_ids)} resolve to {sorted(resolved)}, "
            f"not requested action {spec.slug!r}"
        )
    primary = spec.emg_trial_actions[0]
    if primary not in action_ids:
        raise ValueError(f"EMG reference tube lacks pre-registered primary trial {primary!r}")


def _portable_repo_path(path: Path) -> str:
    """Use a repository-relative identity when an artifact lives in the repo."""

    resolved = path.resolve(strict=True)
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def build_verified_tube_gate(tube_source: str | Path, *, action: str) -> dict[str, Any]:
    """Validate and content-bind one action-specific production tube."""

    spec = resolve(action)
    manifest_path = _tube_manifest_path(tube_source)
    tube = load_emg_phase_reference_tube(manifest_path)
    _validate_tube_action(spec, tube.action_ids)
    resolve_emg_reference_reward_gate(tube, enabled=True)
    source = {
        "manifest_path": _portable_repo_path(manifest_path),
        "manifest_content_sha256": _file_sha256(manifest_path),
        "reference_id": tube.reference_id,
        "reference_fingerprint": tube.reference_fingerprint,
        "array_bundle_sha256": _require_sha256(
            tube.array_bundle_sha256, field="tube.array_bundle_sha256"
        ),
        "mapping_sha256": _require_sha256(
            tube.mapping_binding.get("mapping_sha256"),
            field="tube.mapping_binding.mapping_sha256",
        ),
        "action_ids": list(tube.action_ids),
        "primary_tube_action_id": spec.emg_trial_actions[0],
        "review_status": tube.review_status,
        "training_enabled": tube.training_enabled,
        "mapping_review_status": tube.mapping_binding.get("mapping_review_status"),
        "trial_qc_review_schema_version": tube.provenance["trial_qc_review"][
            "schema_version"
        ],
        "trial_qc_review_sha256": _require_sha256(
            tube.provenance["trial_qc_review"].get("review_sha256"),
            field="tube.provenance.trial_qc_review.review_sha256",
        ),
        "phase_bin_count": int(tube.phase_bin_count),
    }
    unsigned = {
        "schema_version": TUBE_GATE_SCHEMA_VERSION,
        "action": spec.slug,
        "action_id": spec.action_id,
        "passed": True,
        "source": source,
    }
    return {**unsigned, "binding_sha256": _canonical_sha256(unsigned)}


def validate_verified_tube_gate(
    report_source: str | Path | Mapping[str, Any],
    *,
    expected_action: str,
    expected_tube: str | Path,
) -> dict[str, Any]:
    """Rebuild the tube gate and reject stale or hand-edited reports."""

    payload = (
        dict(report_source)
        if isinstance(report_source, Mapping)
        else _require_mapping(load_json_strict(report_source), field="tube gate")
    )
    if payload.get("schema_version") != TUBE_GATE_SCHEMA_VERSION or payload.get("passed") is not True:
        raise ValueError("Stage-1 PEASD verified-tube gate did not pass")
    supplied = _require_sha256(payload.get("binding_sha256"), field="tube gate binding_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "binding_sha256"}
    if supplied != _canonical_sha256(unsigned):
        raise ValueError("Stage-1 PEASD verified-tube gate binding mismatch")
    rebuilt = build_verified_tube_gate(expected_tube, action=expected_action)
    if payload != rebuilt:
        raise ValueError("Stage-1 PEASD verified-tube gate is stale or belongs to another tube/action")
    return payload


def _expected_comparison_contract(spec: ActionSpec) -> dict[str, Any]:
    split = {
        "train_motion_paths": list(spec.train_motion_paths),
        "validation_motion_paths": list(spec.val_motion_paths),
    }
    return {
        "schema_version": PAIRWISE_COMPARISON_SCHEMA_VERSION,
        "arms": list(CANONICAL_ARMS),
        "seeds": list(CANONICAL_SEEDS),
        **split,
        "split_sha256": _canonical_sha256(split),
        "real_arm": "T3",
        "shifted_arm": "T4",
        "shifted_control": "deterministic_half_cycle_circular_phase_shift",
        "comparison_metric": "val_emg_synergy_real_reference_loss",
        "primary_teacher_arm": PRIMARY_TEACHER_ARM,
        "primary_teacher_seed": PRIMARY_TEACHER_SEED,
        "primary_teacher_selection_rule": PRIMARY_TEACHER_SELECTION_RULE,
        "min_relative_real_over_shifted_improvement": (
            MIN_RELATIVE_REAL_OVER_SHIFTED_IMPROVEMENT
        ),
        "max_relative_kinematic_degradation_vs_t0": (
            MAX_RELATIVE_KINEMATIC_DEGRADATION
        ),
    }


def build_pairwise_metrics_document(
    *, action: str, runs: Mapping[str, Any], notes: str = ""
) -> dict[str, Any]:
    """Build the index of immutable runner-produced evidence artifacts.

    Run rows contain only a seed plus an evidence path/content hash.  Supplying
    a ``metrics`` object here is rejected later rather than trusted.
    """

    spec = resolve(action)
    unsigned = {
        "schema_version": PAIRWISE_METRICS_SCHEMA_VERSION,
        "action": spec.slug,
        "action_id": spec.action_id,
        "comparison_contract": _expected_comparison_contract(spec),
        "runs": dict(runs),
        "notes": str(notes),
    }
    return {**unsigned, "report_fingerprint": _canonical_sha256(unsigned)}


def _parse_evidence_selector(value: str) -> tuple[str, int, Path]:
    """Parse one ``ARM:SEED:PATH`` selector without trusting its identity."""

    arm, separator, remainder = str(value).partition(":")
    seed_text, second_separator, path_text = remainder.partition(":")
    arm = arm.strip().upper()
    if not separator or not second_separator or arm not in CANONICAL_ARMS:
        raise ValueError(
            "--evidence must use ARM:SEED:PATH with ARM in T0,T1,T2,T3,T4"
        )
    try:
        seed = int(seed_text)
    except ValueError as exc:
        raise ValueError("--evidence seed must be one of 0,1,2") from exc
    if seed not in CANONICAL_SEEDS or not path_text.strip():
        raise ValueError("--evidence seed/path must name one canonical sealed artifact")
    return arm, seed, _resolve_source_path(path_text.strip())


def build_pairwise_evidence_index(
    *, action: str, evidence_selectors: list[str] | tuple[str, ...], notes: str = ""
) -> dict[str, Any]:
    """Build and fully validate an exact T0--T4 x three-seed evidence index.

    Each selector names a runner-sealed validation document.  Arm, seed,
    action, tube-trial identity, run manifest, checkpoint, split, QC and
    history are rebuilt by the production validator; CLI labels are never
    accepted as identity evidence.
    """

    spec = resolve(action)
    expected = {(arm, seed) for arm in CANONICAL_ARMS for seed in CANONICAL_SEEDS}
    supplied: dict[tuple[str, int], Path] = {}
    for selector in evidence_selectors:
        arm, seed, path = _parse_evidence_selector(selector)
        key = (arm, seed)
        if key in supplied:
            raise ValueError(f"duplicate sealed evidence selector for {arm}/seed{seed}")
        evidence = validate_stage1_peasd_validation_evidence(path)
        if (
            evidence.get("arm") != arm
            or int(evidence.get("seed", -1)) != seed
            or evidence.get("action_id") != spec.action_id
            or evidence.get("tube_action_id") != spec.emg_trial_actions[0]
        ):
            raise ValueError(
                f"sealed evidence identity differs from selector {arm}/seed{seed}/{spec.slug}"
            )
        supplied[key] = path
    missing = sorted(expected - set(supplied))
    extra = sorted(set(supplied) - expected)
    if missing or extra or len(evidence_selectors) != len(expected):
        raise ValueError(
            "evidence index requires exactly T0--T4 x seeds 0,1,2; "
            f"missing={missing}, extra={extra}"
        )
    runs = {
        arm: [
            {
                "seed": seed,
                "validation_metrics_path": str(supplied[(arm, seed)]),
                "validation_metrics_content_sha256": _file_sha256(supplied[(arm, seed)]),
            }
            for seed in CANONICAL_SEEDS
        ]
        for arm in CANONICAL_ARMS
    }
    payload = build_pairwise_metrics_document(action=spec.slug, runs=runs, notes=notes)
    # Reuse the promotion reader to prove that the just-built index is not
    # merely complete-looking: all 15 manifests/checkpoints/histories and the
    # matched-family contracts must verify before the index can be written.
    evaluate_pairwise_promotion(payload, action=spec.slug)
    return payload


def _nested_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        result = {str(key) for key in value}
        for child in value.values():
            result.update(_nested_keys(child))
        return result
    if isinstance(value, list | tuple):
        result: set[str] = set()
        for child in value:
            result.update(_nested_keys(child))
        return result
    return set()


def _validate_runtime_binding(
    contract: Mapping[str, Any], *, arm: str, spec: ActionSpec
) -> dict[str, Any]:
    value = dict(contract)
    if value.get("schema_version") != RUNTIME_CONTRACT_SCHEMA_VERSION:
        raise ValueError(f"{arm} run has no Stage-1 PEASD-Lite runtime contract")
    supplied = _require_sha256(value.get("binding_sha256"), field=f"{arm} runtime binding_sha256")
    unsigned = {key: item for key, item in value.items() if key != "binding_sha256"}
    if supplied != _canonical_sha256(unsigned):
        raise ValueError(f"{arm} runtime contract binding mismatch")
    if value.get("enabled") is not True or value.get("arm") != arm:
        raise ValueError(f"{arm} runtime contract has the wrong enabled/arm identity")
    tube_action_id = spec.emg_trial_actions[0]
    if value.get("action_id") != tube_action_id:
        raise ValueError(f"{arm} runtime must select primary tube trial {tube_action_id!r}")
    if resolve(str(value.get("action_id"))).slug != spec.slug:
        raise ValueError(f"{arm} runtime tube action belongs to another registry action")
    if (
        value.get("reference_review_status") != "verified"
        or value.get("reference_training_enabled") is not True
        or value.get("mapping_review_status") != "verified"
    ):
        raise ValueError(f"{arm} runtime uses a non-production tube or mapping")
    if int(value.get("ordered_actuator_count", -1)) != 354:
        raise ValueError(f"{arm} runtime contract is not bound to ordered Full-354")
    if value.get("phase_coordinate") != "normalized_trajectory_progress":
        raise ValueError(f"{arm} runtime uses a non-canonical phase coordinate")
    if value.get("signal") != "mujoco_scalar_activation_state":
        raise ValueError(f"{arm} runtime uses a non-canonical activation signal")
    for field in (
        "reference_fingerprint",
        "array_bundle_sha256",
        "trial_qc_review_sha256",
        "mapping_sha256",
        "actuator_schema_hash",
        "runtime_model_hash",
        "muscle_channel_core_fingerprint",
        "anchor_loss_spec_fingerprint",
        "matched_reward_core_fingerprint",
    ):
        _require_sha256(value.get(field), field=f"{arm} runtime {field}")

    anchor = _finite_nonnegative(value.get("anchor_weight_max"), field=f"{arm}.anchor_weight_max")
    synergy = _finite_nonnegative(value.get("synergy_weight_max"), field=f"{arm}.synergy_weight_max")
    expected_activity = {
        "T0": (False, False),
        "T1": (True, False),
        "T2": (False, True),
        "T3": (True, True),
        "T4": (True, True),
    }[arm]
    if (anchor > 0.0, synergy > 0.0) != expected_activity:
        raise ValueError(f"{arm} runtime anchor/synergy treatment matrix is invalid")
    if arm == "T0":
        if (
            value.get("mode") != "diagnostics_only"
            or value.get("training_signal_enabled") is not False
        ):
            raise ValueError("T0 evaluation runtime must be reward-neutral diagnostics_only")
    elif (
        value.get("mode") != "reward"
        or value.get("training_signal_enabled") is not True
    ):
        raise ValueError(f"{arm} training runtime must enable reward mode")
    if _finite_nonnegative(
        value.get("synergy_intensity_weight"), field=f"{arm}.synergy_intensity_weight"
    ) > _finite_nonnegative(
        value.get("synergy_shape_weight"), field=f"{arm}.synergy_shape_weight"
    ):
        raise ValueError(f"{arm} synergy intensity weight must not exceed shape weight")

    phase_bins = int(value.get("phase_bin_count", -1))
    offset = int(value.get("synergy_phase_shuffle_offset_bins", -1))
    if arm != "T4":
        if (
            value.get("synergy_phase_shuffled") is not False
            or value.get("synergy_phase_shift_strategy") != "none"
            or offset != 0
        ):
            raise ValueError(f"{arm} must use real, unshifted synergy phase bins")
    else:
        if (
            phase_bins < 2
            or phase_bins % 2 != 0
            or value.get("synergy_phase_shuffled") is not True
            or value.get("synergy_phase_shift_strategy") != "half_cycle_circular"
            or offset != phase_bins // 2
        ):
            raise ValueError("T4 must use the deterministic half-cycle circular phase shift")
        forbidden = sorted(key for key in _nested_keys(value) if "event" in key.lower())
        if forbidden:
            raise ValueError(f"T4 must not fabricate/shuffle events; found fields {forbidden}")
    return value


def _validate_delivered_treatment(
    metrics: Mapping[str, Any], *, runtime: Mapping[str, Any], arm: str, seed: int
) -> None:
    """Independently enforce the endpoint treatment matrix from rollout metrics."""

    anchor = float(runtime["anchor_weight_max"])
    synergy = float(runtime["synergy_weight_max"])
    expected = {
        "val_emg_anchor_weight": anchor,
        "val_emg_synergy_weight": synergy,
        "val_emg_curriculum_factor_anchor": 1.0 if anchor > 0.0 else 0.0,
        "val_emg_curriculum_factor_synergy": 1.0 if synergy > 0.0 else 0.0,
    }
    for field, target in expected.items():
        observed = _finite_numeric(metrics.get(field), field=f"{arm}/seed{seed}.{field}")
        if not math.isclose(observed, target, rel_tol=1.0e-6, abs_tol=1.0e-8):
            raise ValueError(
                f"{arm}/seed{seed} delivered treatment {field}={observed}, expected {target}"
            )
    masked = _finite_nonnegative(
        metrics.get("val_emg_consistency_penalty_masked_fraction"),
        field=f"{arm}/seed{seed}.val_emg_consistency_penalty_masked_fraction",
    )
    if masked > 1.0 or ((anchor > 0.0 or synergy > 0.0) and masked >= 1.0):
        raise ValueError(f"{arm}/seed{seed} EMG treatment was fully masked")
    total_clip_effective = _finite_numeric(
        metrics.get("val_penalty_emg_consistency_effective_after_total_clip"),
        field=(
            f"{arm}/seed{seed}."
            "val_penalty_emg_consistency_effective_after_total_clip"
        ),
    )
    reward_floor_effective = _finite_numeric(
        metrics.get("val_penalty_emg_consistency_effective_after_reward_floor"),
        field=(
            f"{arm}/seed{seed}."
            "val_penalty_emg_consistency_effective_after_reward_floor"
        ),
    )
    final_masked = _finite_nonnegative(
        metrics.get("val_emg_consistency_final_reward_masked_fraction"),
        field=(
            f"{arm}/seed{seed}."
            "val_emg_consistency_final_reward_masked_fraction"
        ),
    )
    active = anchor > 0.0 or synergy > 0.0
    if final_masked > 1.0 or (active and final_masked >= 1.0):
        raise ValueError(
            f"{arm}/seed{seed} EMG treatment was erased by the final reward floor"
        )
    if total_clip_effective > 1.0e-8 or reward_floor_effective > 1.0e-8:
        raise ValueError(f"{arm}/seed{seed} EMG penalty diagnostics use the wrong sign")
    if not active and any(
        not math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1.0e-8)
        for value in (
            masked,
            final_masked,
            total_clip_effective,
            reward_floor_effective,
        )
    ):
        raise ValueError(f"{arm}/seed{seed} reward-neutral evaluation emitted EMG treatment")


def _extract_split(experiment: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    try:
        train = experiment["task_factory"]["params"]["amass_dataset_conf"]["rel_dataset_path"]
        validation = experiment["validation"]["amass_dataset_conf"]["rel_dataset_path"]
    except (KeyError, TypeError) as exc:
        raise ValueError("run manifest lacks the Stage-1 train/validation motion split") from exc
    if not isinstance(train, list) or not isinstance(validation, list):
        raise ValueError("run manifest train/validation motion paths must be JSON lists")
    return [str(item) for item in train], [str(item) for item in validation]


def _matched_config_core(experiment: Mapping[str, Any]) -> dict[str, Any]:
    """Redact only pre-registered treatment/volatile fields from a run config."""

    core = copy.deepcopy(dict(experiment))
    core.pop("seeds", None)
    for field in _TRAINING_HASH_VOLATILE_FIELDS:
        core.pop(field, None)
    for field in _DYNAMIC_EXPERIMENT_FIELDS:
        core.pop(field, None)
    stage1 = _require_mapping(core.get("stage1_peasd"), field="stage1_peasd")
    stage1.pop("arm", None)
    stage1.pop("control_kind", None)
    core["stage1_peasd"] = stage1
    try:
        reward_params = core["env_params"]["reward_params"]
        emg = _require_mapping(reward_params.get("emg_consistency"), field="emg_consistency")
    except (KeyError, TypeError) as exc:
        raise ValueError("matched run config has no emg_consistency block") from exc
    for field in _EMG_CONFIG_TREATMENT_FIELDS:
        emg.pop(field, None)
    reward_params["emg_consistency"] = emg
    return core


def _matched_budget_core(contract: Mapping[str, Any]) -> dict[str, Any]:
    core = copy.deepcopy(dict(contract))
    for field in ("arm", "seed", "binding_sha256"):
        core.pop(field, None)
    curriculum = _require_mapping(core.get("emg_curriculum"), field="budget emg_curriculum")
    for field in ("mode", "anchor_weight_max", "synergy_weight_max"):
        curriculum.pop(field, None)
    core["emg_curriculum"] = curriculum
    return core


def _validate_arm_contract(
    experiment: Mapping[str, Any], *, arm: str, seed: int, spec: ActionSpec
) -> None:
    contract = _require_mapping(experiment.get("stage1_peasd"), field=f"{arm}/seed{seed} stage1_peasd")
    if (
        contract.get("schema_version") != STAGE1_ARM_SCHEMA_VERSION
        or contract.get("arm") != arm
        or contract.get("action_id") != spec.action_id
        or contract.get("canonical_seeds") != list(CANONICAL_SEEDS)
        or contract.get("fresh_optimizer_required") is not True
        or contract.get("parent_initialization_checkpoint") is not None
    ):
        raise ValueError(f"{arm}/seed{seed} has an invalid matched-arm config contract")
    if (
        experiment.get("auto_resume") is not False
        or experiment.get("resume_from") is not None
        or experiment.get("reset_optimizer_on_resume") is not True
        or experiment.get("resume_lr_override") is not None
        or bool(experiment.get("extend_completed_run", False))
        or int(experiment.get("n_seeds", -1)) != 1
        or experiment.get("seeds") != [seed]
    ):
        raise ValueError(f"{arm}/seed{seed} is not a fresh single-seed optimizer run")
    promotion = _require_mapping(experiment.get("promotion"), field=f"{arm}/seed{seed} promotion")
    if promotion.get("auto_stop") is not False:
        raise ValueError(f"{arm}/seed{seed} enables arm-dependent early stopping")


def _validate_run_record(
    record: Mapping[str, Any],
    *,
    index_path: Path,
    arm: str,
    spec: ActionSpec,
) -> dict[str, Any]:
    row = dict(record)
    allowed = {"seed", "validation_metrics_path", "validation_metrics_content_sha256"}
    if set(row) != allowed:
        raise ValueError(
            f"{arm} run record must contain only sealed evidence path/hash/seed; "
            f"found {sorted(row)}"
        )
    seed = int(row.get("seed", -1))
    if seed not in CANONICAL_SEEDS:
        raise ValueError(f"{arm} record has unsupported seed {seed}")
    evidence_path = _resolve_source_path(
        str(row.get("validation_metrics_path", "")), relative_to=index_path.parent
    )
    evidence_sha = _require_sha256(
        row.get("validation_metrics_content_sha256"),
        field=f"{arm}/seed{seed} validation_metrics_content_sha256",
    )
    if _file_sha256(evidence_path) != evidence_sha:
        raise ValueError(f"{arm}/seed{seed} sealed validation evidence content changed")
    evidence = validate_stage1_peasd_validation_evidence(evidence_path)
    if evidence.get("schema_version") != STAGE1_PEASD_VALIDATION_SCHEMA_VERSION:
        raise ValueError(f"{arm}/seed{seed} validation evidence schema is incompatible")
    if (
        evidence.get("arm") != arm
        or int(evidence.get("seed", -1)) != seed
        or evidence.get("action_id") != spec.action_id
    ):
        raise ValueError(f"{arm}/seed{seed} validation evidence identity mismatch")
    if evidence.get("tube_action_id") != spec.emg_trial_actions[0]:
        raise ValueError(f"{arm}/seed{seed} selected a non-primary tube trial")

    manifest_path = Path(str(evidence["run_manifest_path"])).resolve(strict=True)
    manifest = _require_mapping(load_json_strict(manifest_path), field=f"{arm}/seed{seed} run manifest")
    git_sha = str(manifest.get("git_sha") or "").strip()
    if not git_sha:
        raise ValueError(f"{arm}/seed{seed} run manifest has no git_sha")
    source_snapshot = _require_mapping(
        evidence.get("source_tree_snapshot"),
        field=f"{arm}/seed{seed} source_tree_snapshot",
    )
    source_git_sha = str(source_snapshot.get("git_sha") or "").strip()
    if not source_git_sha or not source_git_sha.startswith(git_sha):
        raise ValueError(f"{arm}/seed{seed} Git SHA differs from its source-tree snapshot")
    config_hash = str(manifest.get("config_hash", ""))
    if not _CONFIG_HASH.fullmatch(config_hash) or evidence.get("config_hash") != config_hash:
        raise ValueError(f"{arm}/seed{seed} config hash binding mismatch")
    experiment = _require_mapping(manifest.get("experiment_config"), field=f"{arm}/seed{seed} experiment_config")
    _validate_arm_contract(experiment, arm=arm, seed=seed, spec=spec)
    run_id = str(experiment.get("run_id", "")).strip()
    if not run_id or evidence.get("run_id") != run_id:
        raise ValueError(f"{arm}/seed{seed} run_id binding mismatch")
    train, validation = _extract_split(experiment)
    if train != list(spec.train_motion_paths) or validation != list(spec.val_motion_paths):
        raise ValueError(f"{arm}/seed{seed} does not use the canonical action split")

    identity = _require_mapping(evidence.get("checkpoint_identity"), field=f"{arm}/seed{seed} checkpoint_identity")
    total_timesteps = int(experiment.get("total_timesteps", -1))
    global_timestep = int(identity.get("global_timestep", -1))
    target_timestep = int(identity.get("target_global_timestep", -1))
    if total_timesteps <= 0 or global_timestep != total_timesteps or target_timestep != total_timesteps:
        raise ValueError(f"{arm}/seed{seed} checkpoint is not the pre-registered hard-budget endpoint")
    update_number = int(identity.get("update_number", -1))
    if update_number < 0:
        raise ValueError(f"{arm}/seed{seed} checkpoint update is invalid")

    runtime = _validate_runtime_binding(
        _require_mapping(
            evidence.get("evaluation_emg_consistency_runtime_contract"),
            field=f"{arm}/seed{seed} evaluation_emg_consistency_runtime_contract",
        ),
        arm=arm,
        spec=spec,
    )
    metrics = _require_mapping(evidence.get("metrics"), field=f"{arm}/seed{seed} metrics")
    validated_metrics = {
        key: _finite_numeric(value, field=f"{arm}/seed{seed}.{key}")
        for key, value in metrics.items()
    }
    _validate_delivered_treatment(
        validated_metrics,
        runtime=runtime,
        arm=arm,
        seed=seed,
    )
    history_path = Path(str(evidence["validation_history_path"])).resolve(strict=True)
    history = validate_stage1_peasd_validation_history(history_path, require_complete=True)
    history_tail_metrics = _require_mapping(
        history["entries"][-1].get("metrics"), field="validation history tail metrics"
    )
    if arm == "T0":
        # The post-hoc tube may change only physiology diagnostics; tracking
        # and safety must remain the exact final training-checkpoint result.
        stable_keys = {
            *_LOWER_IS_BETTER_TRACKING_METRICS,
            *_HIGHER_IS_BETTER_TRACKING_METRICS,
            *_DESCRIPTIVE_CROSS_ARM_METRICS,
        }
        mismatched = sorted(
            key
            for key in stable_keys
            if float(history_tail_metrics[key]) != float(validated_metrics[key])
        )
        if mismatched:
            raise ValueError(f"T0 post-hoc evaluation changed tracking metrics: {mismatched}")
    elif history_tail_metrics != metrics:
        raise ValueError(f"{arm}/seed{seed} final evidence differs from its history endpoint")
    return {
        "seed": seed,
        "run_id": run_id,
        "config_hash": config_hash,
        "git_sha": git_sha,
        "source_tree_snapshot": source_snapshot,
        "run_manifest": str(manifest_path),
        "run_manifest_content_sha256": evidence["run_manifest_content_sha256"],
        "validation_metrics_path": str(evidence_path),
        "validation_metrics_content_sha256": evidence_sha,
        "validation_evidence_fingerprint": evidence["evidence_fingerprint"],
        "checkpoint_identity": identity,
        "metrics": validated_metrics,
        "validation_history_path": str(history_path),
        "validation_history_content_sha256": evidence[
            "validation_history_content_sha256"
        ],
        "validation_history": history,
        "action_release_contract": evidence["action_release_contract"],
        "numeric_data_qc_contract": evidence["numeric_data_qc_contract"],
        "training_budget_contract": evidence["training_budget_contract"],
        "runtime_contract": runtime,
        "matched_config_core": _matched_config_core(experiment),
    }


def _paired_statistics(values: list[float], *, favorable_direction: str) -> dict[str, Any]:
    """Return transparent seed-level small-sample descriptive statistics."""

    if len(values) != len(CANONICAL_SEEDS):
        raise ValueError("paired statistics require exactly the three canonical seeds")
    mean = sum(values) / len(values)
    sample_variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    sample_std = math.sqrt(max(0.0, sample_variance))
    # Two-sided 95% Student-t critical value for df=2.  This interval describes
    # three seed-paired runs only; it is explicitly not a subject/population CI.
    half_width = 4.302652729911275 * sample_std / math.sqrt(len(values))
    effect_size = None if sample_std <= 1.0e-12 else mean / sample_std
    return {
        "unit": "paired_seed",
        "n": len(values),
        "values_by_seed": {
            str(seed): float(values[index]) for index, seed in enumerate(CANONICAL_SEEDS)
        },
        "mean": mean,
        "sample_std": sample_std,
        "paired_cohens_dz": effect_size,
        "confidence_interval": {
            "method": "student_t_interval_df2",
            "level": 0.95,
            "lower": mean - half_width,
            "upper": mean + half_width,
            "scope": "seed_level_small_sample_not_population",
        },
        "favorable_direction": favorable_direction,
    }


def _relative_lower_is_better_degradation(baseline: float, treatment: float) -> float:
    if baseline < 0.0 or treatment < 0.0:
        raise ValueError("lower-is-better tracking metrics must be non-negative")
    if baseline <= 1.0e-12:
        return 0.0 if treatment <= 1.0e-12 else 1.0e30
    return (treatment - baseline) / abs(baseline)


def _relative_higher_is_better_degradation(baseline: float, treatment: float) -> float:
    denominator = max(abs(baseline), 1.0e-12)
    return (baseline - treatment) / denominator


def _convergence_diagnostics(record: Mapping[str, Any]) -> dict[str, Any]:
    history = _require_mapping(record.get("validation_history"), field="validation history")
    entries = history.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("validation history has no entries")
    timesteps = [int(entry["checkpoint_identity"]["global_timestep"]) for entry in entries]
    metrics = [_require_mapping(entry.get("metrics"), field="history metrics") for entry in entries]
    final_timestep = timesteps[-1]

    def normalized_auc(metric: str) -> float:
        values = [_finite_numeric(row.get(metric), field=f"history.{metric}") for row in metrics]
        area = values[0] * timesteps[0]
        for index in range(1, len(values)):
            area += 0.5 * (values[index - 1] + values[index]) * (
                timesteps[index] - timesteps[index - 1]
            )
        return area / max(final_timestep, 1)

    tracking_threshold = float(
        record["matched_config_core"]["promotion"]["max_relative_site_position_error_m"]
    )
    time_to_tracking_threshold = next(
        (
            timesteps[index]
            for index, row in enumerate(metrics)
            if float(row["val_err_rpos"]) <= tracking_threshold
        ),
        None,
    )
    result: dict[str, Any] = {
        "validation_count": len(entries),
        "final_global_timestep": final_timestep,
        "val_err_rpos_normalized_auc": normalized_auc("val_err_rpos"),
        "tracking_threshold": tracking_threshold,
        "time_to_tracking_threshold_global_timestep": time_to_tracking_threshold,
    }
    if record["runtime_contract"].get("mode") == "reward":
        result["val_emg_synergy_real_reference_loss_normalized_auc"] = normalized_auc(
            "val_emg_synergy_real_reference_loss"
        )
    return result


def evaluate_pairwise_promotion(
    metrics_source: str | Path | Mapping[str, Any], *, action: str
) -> dict[str, Any]:
    """Evaluate the complete T0--T4 family from sealed, seed-matched runs."""

    spec = resolve(action)
    if isinstance(metrics_source, Mapping):
        source_path = None
        payload = dict(metrics_source)
        index_path = Path.cwd() / "stage1_peasd_pairwise_evidence_index.json"
        index_content_sha = _canonical_sha256(payload)
    else:
        source_path = _resolve_source_path(metrics_source)
        index_path = source_path
        payload = _require_mapping(load_json_strict(source_path), field="pairwise evidence index")
        index_content_sha = _file_sha256(source_path)
    if payload.get("schema_version") != PAIRWISE_METRICS_SCHEMA_VERSION:
        raise ValueError("Stage-1 PEASD pairwise evidence index schema is incompatible")
    if payload.get("action") != spec.slug or payload.get("action_id") != spec.action_id:
        raise ValueError("Stage-1 PEASD pairwise evidence belongs to another action")
    supplied_fingerprint = _require_sha256(payload.get("report_fingerprint"), field="pairwise report_fingerprint")
    unsigned = {key: value for key, value in payload.items() if key != "report_fingerprint"}
    if supplied_fingerprint != _canonical_sha256(unsigned):
        raise ValueError("Stage-1 PEASD pairwise evidence index fingerprint mismatch")
    if payload.get("comparison_contract") != _expected_comparison_contract(spec):
        raise ValueError("Stage-1 PEASD split/seeds/comparison contract drifted")
    runs = _require_mapping(payload.get("runs"), field="pairwise evidence runs")
    if set(runs) != set(CANONICAL_ARMS):
        raise ValueError("pairwise evidence must contain exactly T0--T4")

    validated: dict[str, dict[int, dict[str, Any]]] = {}
    all_run_ids: list[str] = []
    all_config_hashes: list[str] = []
    all_git_shas: set[str] = set()
    all_source_tree_snapshots: set[str] = set()
    all_config_cores: set[str] = set()
    all_budget_cores: set[str] = set()
    all_release_contracts: set[str] = set()
    all_numeric_data_qc_contracts: set[str] = set()
    endpoint_updates: set[int] = set()
    endpoint_timesteps: set[int] = set()
    for arm in CANONICAL_ARMS:
        records = runs[arm]
        if not isinstance(records, list) or len(records) != len(CANONICAL_SEEDS):
            raise ValueError(f"{arm} must contain exactly the three canonical seeds")
        by_seed: dict[int, dict[str, Any]] = {}
        for raw in records:
            record = _validate_run_record(
                _require_mapping(raw, field=f"{arm} run record"),
                index_path=index_path,
                arm=arm,
                spec=spec,
            )
            seed = int(record["seed"])
            if seed in by_seed:
                raise ValueError(f"{arm} repeats seed {seed}")
            by_seed[seed] = record
            all_run_ids.append(record["run_id"])
            all_config_hashes.append(record["config_hash"])
            all_git_shas.add(record["git_sha"])
            all_source_tree_snapshots.add(
                _canonical_sha256(record["source_tree_snapshot"])
            )
            all_config_cores.add(_canonical_sha256(record["matched_config_core"]))
            all_budget_cores.add(
                _canonical_sha256(_matched_budget_core(record["training_budget_contract"]))
            )
            all_release_contracts.add(
                _canonical_sha256(record["action_release_contract"])
            )
            all_numeric_data_qc_contracts.add(
                _canonical_sha256(record["numeric_data_qc_contract"])
            )
            endpoint_updates.add(int(record["checkpoint_identity"]["update_number"]))
            endpoint_timesteps.add(int(record["checkpoint_identity"]["global_timestep"]))
        if set(by_seed) != set(CANONICAL_SEEDS):
            raise ValueError(f"{arm} seed set differs from {CANONICAL_SEEDS}")
        validated[arm] = by_seed
    if len(set(all_run_ids)) != len(all_run_ids):
        raise ValueError("every Stage-1 PEASD arm/seed must use a unique run_id")
    if len(set(all_config_hashes)) != len(all_config_hashes):
        raise ValueError("every Stage-1 PEASD arm/seed must have a distinct resolved config hash")
    if len(all_git_shas) != 1:
        raise ValueError("all Stage-1 PEASD arm/seed runs must use one non-empty git_sha")
    if len(all_source_tree_snapshots) != 1:
        raise ValueError(
            "all Stage-1 PEASD arm/seed runs must use one source-tree snapshot"
        )
    if len(all_budget_cores) != 1:
        raise ValueError("T0--T4 runs do not share one fixed-budget/validation schedule")
    if len(all_config_cores) != 1:
        raise ValueError("T0--T4 runs do not share one matched config core")
    if len(all_release_contracts) != 1:
        raise ValueError("T0--T4 runs do not share one action release/QC contract")
    if len(all_numeric_data_qc_contracts) != 1:
        raise ValueError("T0--T4 runs do not share one numeric data-QC contract")
    if len(endpoint_updates) != 1 or len(endpoint_timesteps) != 1:
        raise ValueError("T0--T4 checkpoints do not share one fixed-budget endpoint")
    matched_config_core_sha = next(iter(all_config_cores))
    promotion_threshold_config = _require_mapping(
        validated["T0"][0]["matched_config_core"].get("promotion"),
        field="matched Stage1 promotion thresholds",
    )
    validate_promotion_threshold_config("stage1", promotion_threshold_config)
    absolute_stage1_thresholds = dict(CANONICAL_PROMOTION_THRESHOLDS["stage1"])

    for field in _MATCHED_ACTIVE_RUNTIME_FIELDS:
        values = {
            validated[arm][seed]["runtime_contract"].get(field)
            for arm in CANONICAL_ARMS
            for seed in CANONICAL_SEEDS
        }
        if len(values) != 1:
            raise ValueError(
                "T1--T4 seeds do not share one tube/loss/model contract; "
                f"field {field!r} drifted"
            )
    for field in (
        "anchor_weight_max",
        "synergy_weight_max",
        "matched_reward_core_fingerprint",
    ):
        values = {
            validated[arm][seed]["runtime_contract"].get(field)
            for arm in ("T3", "T4")
            for seed in CANONICAL_SEEDS
        }
        if len(values) != 1:
            raise ValueError(f"T3/T4 matched treatment field {field!r} drifted")
    for field, arms in (
        ("anchor_weight_max", ("T1", "T3", "T4")),
        ("synergy_weight_max", ("T2", "T3", "T4")),
    ):
        values = {
            validated[arm][seed]["runtime_contract"].get(field)
            for arm in arms
            for seed in CANONICAL_SEEDS
        }
        if len(values) != 1:
            raise ValueError(
                f"single-leg decomposition field {field!r} differs across {arms}"
            )

    checks: list[dict[str, Any]] = []
    synergy_improvements: list[float] = []
    measured_activation_improvements: dict[str, list[float]] = {
        metric: [] for metric in _MEASURED_ACTIVATION_METRICS
    }
    tracking_deltas: dict[str, list[float]] = {
        key: []
        for key in (*_LOWER_IS_BETTER_TRACKING_METRICS, *_HIGHER_IS_BETTER_TRACKING_METRICS)
    }
    descriptive_deltas: dict[str, list[float]] = {
        key: [] for key in _DESCRIPTIVE_CROSS_ARM_METRICS
    }
    absolute_rules = (
        (
            "val_early_termination_rate",
            "<=",
            absolute_stage1_thresholds["max_early_termination_rate"],
        ),
        (
            "val_frame_coverage",
            ">=",
            absolute_stage1_thresholds["min_frame_coverage"],
        ),
        (
            "val_err_rpos",
            "<=",
            absolute_stage1_thresholds["max_relative_site_position_error_m"],
        ),
        (
            "val_action_saturation_fraction",
            "<=",
            absolute_stage1_thresholds["max_action_saturation_fraction"],
        ),
        (
            "val_activation_energy",
            "<=",
            absolute_stage1_thresholds["max_activation_energy"],
        ),
    )
    for arm in CANONICAL_ARMS:
        for seed in CANONICAL_SEEDS:
            run_metrics = validated[arm][seed]["metrics"]
            for metric, operator, threshold in absolute_rules:
                value = float(run_metrics[metric])
                passed = value <= threshold if operator == "<=" else value >= threshold
                checks.append(
                    {
                        "name": (
                            f"seed_{seed}_{arm.lower()}_absolute_stage1_"
                            f"{metric.removeprefix('val_')}"
                        ),
                        "metric": metric,
                        "value": value,
                        "operator": operator,
                        "threshold": threshold,
                        "passed": passed,
                    }
                )
    for seed in CANONICAL_SEEDS:
        baseline = validated["T0"][seed]
        real = validated["T3"][seed]
        shifted = validated["T4"][seed]
        shifted_loss = float(shifted["metrics"]["val_emg_synergy_real_reference_loss"])
        real_loss = float(real["metrics"]["val_emg_synergy_real_reference_loss"])
        if shifted_loss < 0.0 or real_loss < 0.0:
            raise ValueError("synergy real-reference loss must be non-negative")
        relative_improvement = (shifted_loss - real_loss) / max(abs(shifted_loss), 1.0e-12)
        synergy_improvements.append(shifted_loss - real_loss)
        checks.append(
            {
                "name": f"seed_{seed}_real_synergy_beats_half_cycle_shift",
                "metric": "val_emg_synergy_real_reference_loss",
                "value": relative_improvement,
                "operator": ">=",
                "threshold": MIN_RELATIVE_REAL_OVER_SHIFTED_IMPROVEMENT,
                "passed": relative_improvement >= MIN_RELATIVE_REAL_OVER_SHIFTED_IMPROVEMENT,
            }
        )
        for metric in _MEASURED_ACTIVATION_METRICS:
            baseline_value = float(baseline["metrics"][metric])
            real_value = float(real["metrics"][metric])
            if metric == "val_emg_anchor_correlation":
                if not (-1.0 <= baseline_value <= 1.0 and -1.0 <= real_value <= 1.0):
                    raise ValueError("measured activation correlation must be in [-1,1]")
                delta = real_value - baseline_value
            elif baseline_value < 0.0 or real_value < 0.0:
                raise ValueError(f"measured activation metric {metric!r} must be non-negative")
            else:
                delta = baseline_value - real_value
            measured_activation_improvements[metric].append(delta)
        anchor_improvement = measured_activation_improvements[
            "val_emg_anchor_loss"
        ][-1]
        checks.append(
            {
                "name": f"seed_{seed}_measured_activation_anchor_improves_vs_t0",
                "metric": "val_emg_anchor_loss",
                "value": anchor_improvement,
                "operator": ">",
                "threshold": 0.0,
                "passed": anchor_improvement > 0.0,
            }
        )
        for metric in _MEASURED_ACTIVATION_METRICS:
            improvement = measured_activation_improvements[metric][-1]
            checks.append(
                {
                    "name": (
                        f"seed_{seed}_{metric.removeprefix('val_emg_')}"
                        "_non_degraded_vs_t0"
                    ),
                    "metric": metric,
                    "value": improvement,
                    "operator": ">=",
                    "threshold": 0.0,
                    "passed": improvement >= 0.0,
                }
            )
        for metric in _LOWER_IS_BETTER_TRACKING_METRICS:
            baseline_value = float(baseline["metrics"][metric])
            real_value = float(real["metrics"][metric])
            tracking_deltas[metric].append(real_value - baseline_value)
            if metric in {
                "val_err_root_xyz",
                "val_err_root_yaw",
                "val_err_joint_pos",
                "val_err_joint_vel",
                "val_err_rpos",
            }:
                degradation = _relative_lower_is_better_degradation(
                    baseline_value, real_value
                )
                threshold = MAX_RELATIVE_KINEMATIC_DEGRADATION
                operator = "relative_degradation_<="
            elif metric in {
                "val_early_termination_rate",
                "val_activation_saturation_fraction",
                "val_action_saturation_fraction",
            }:
                degradation = real_value - baseline_value
                threshold = MAX_ABSOLUTE_SAFETY_RATE_INCREASE
                operator = "absolute_increase_<="
            else:
                degradation = _relative_lower_is_better_degradation(
                    baseline_value, real_value
                )
                threshold = MAX_RELATIVE_CONTROL_EFFORT_DEGRADATION
                operator = "relative_degradation_<="
            checks.append(
                {
                    "name": f"seed_{seed}_{metric.removeprefix('val_')}_non_degraded_vs_t0",
                    "metric": metric,
                    "value": degradation,
                    "operator": operator,
                    "threshold": threshold,
                    "passed": degradation <= threshold,
                }
            )
        for metric in _DESCRIPTIVE_CROSS_ARM_METRICS:
            descriptive_deltas[metric].append(
                float(real["metrics"][metric]) - float(baseline["metrics"][metric])
            )
        for metric in _HIGHER_IS_BETTER_TRACKING_METRICS:
            baseline_value = float(baseline["metrics"][metric])
            real_value = float(real["metrics"][metric])
            tracking_deltas[metric].append(real_value - baseline_value)
            degradation = _relative_higher_is_better_degradation(
                baseline_value, real_value
            )
            threshold = 0.02 if metric == "val_frame_coverage" else 0.10
            checks.append(
                {
                    "name": f"seed_{seed}_{metric.removeprefix('val_')}_non_degraded_vs_t0",
                    "metric": metric,
                    "value": degradation,
                    "operator": "relative_degradation_<=",
                    "threshold": threshold,
                    "passed": degradation <= threshold,
                }
            )

    aggregate_synergy = _paired_statistics(
        synergy_improvements,
        favorable_direction="positive_means_T3_lower_loss_than_T4",
    )
    checks.append(
        {
            "name": "aggregate_real_synergy_direction",
            "metric": "paired_T4_minus_T3_real_reference_loss",
            "value": aggregate_synergy["mean"],
            "operator": ">",
            "threshold": 0.0,
            "passed": float(aggregate_synergy["mean"]) > 0.0,
        }
    )
    measured_activation_statistics = {
        metric: _paired_statistics(
            values,
            favorable_direction=(
                "positive_means_T3_higher_than_T0"
                if metric == "val_emg_anchor_correlation"
                else "positive_means_T3_lower_than_T0"
            ),
        )
        for metric, values in measured_activation_improvements.items()
    }
    aggregate_anchor_improvement = measured_activation_statistics[
        "val_emg_anchor_loss"
    ]
    checks.append(
        {
            "name": "aggregate_measured_activation_anchor_direction",
            "metric": "paired_T0_minus_T3_val_emg_anchor_loss",
            "value": aggregate_anchor_improvement["mean"],
            "operator": ">",
            "threshold": 0.0,
            "passed": float(aggregate_anchor_improvement["mean"]) > 0.0,
        }
    )
    for metric, statistics in measured_activation_statistics.items():
        mean_improvement = float(statistics["mean"])
        checks.append(
            {
                "name": (
                    "aggregate_"
                    f"{metric.removeprefix('val_emg_')}"
                    "_non_degraded_vs_t0"
                ),
                "metric": f"paired_T3_vs_T0_{metric}",
                "value": mean_improvement,
                "operator": ">=",
                "threshold": 0.0,
                "passed": mean_improvement >= 0.0,
            }
        )
    tracking_statistics = {
        metric: _paired_statistics(
            values,
            favorable_direction=(
                "negative_means_T3_lower_than_T0"
                if metric in _LOWER_IS_BETTER_TRACKING_METRICS
                else "positive_means_T3_higher_than_T0"
            ),
        )
        for metric, values in tracking_deltas.items()
    }
    descriptive_statistics = {
        metric: _paired_statistics(
            values,
            favorable_direction=(
                "descriptive_only_T3_minus_T0_reward_definitions_differ"
            ),
        )
        for metric, values in descriptive_deltas.items()
    }
    decomposition_statistics = {
        arm: {
            metric: _paired_statistics(
                [
                    float(validated[arm][seed]["metrics"][metric])
                    - float(validated["T0"][seed]["metrics"][metric])
                    for seed in CANONICAL_SEEDS
                ],
                favorable_direction="descriptive_Treatment_minus_T0",
            )
            for metric in (
                "val_err_rpos",
                "val_emg_anchor_loss",
                "val_emg_synergy_real_reference_loss",
            )
        }
        for arm in ("T1", "T2")
    }
    convergence = {
        arm: {
            str(seed): _convergence_diagnostics(validated[arm][seed])
            for seed in CANONICAL_SEEDS
        }
        for arm in CANONICAL_ARMS
    }
    failed_seed_checks = [check for check in checks if not check["passed"] and check["name"].startswith("seed_")]
    seed_check_count = sum(check["name"].startswith("seed_") for check in checks)
    statistics = {
        "schema_version": "stage1_peasd_seed_statistics_v1",
        "statistical_unit": "paired_seed",
        "n_seeds": len(CANONICAL_SEEDS),
        "small_sample_caveat": (
            "n=3 seed-level intervals are descriptive and are not subject/population confidence intervals"
        ),
        "primary_synergy": aggregate_synergy,
        "measured_activation_vs_t0": measured_activation_statistics,
        "tracking_vs_t0": tracking_statistics,
        "noncomparable_reward_diagnostics_vs_t0": descriptive_statistics,
        "t1_t2_decomposition_vs_t0": decomposition_statistics,
        "convergence_diagnostics": convergence,
        "seed_check_failure_rate": (
            len(failed_seed_checks) / seed_check_count if seed_check_count else 1.0
        ),
    }

    evidence_by_arm_seed = {
        arm: {
            str(seed): {
                "path": validated[arm][seed]["validation_metrics_path"],
                "content_sha256": validated[arm][seed]["validation_metrics_content_sha256"],
                "evidence_fingerprint": validated[arm][seed]["validation_evidence_fingerprint"],
                "config_hash": validated[arm][seed]["config_hash"],
                "checkpoint_identity": validated[arm][seed]["checkpoint_identity"],
                "validation_history_path": validated[arm][seed]["validation_history_path"],
                "validation_history_content_sha256": validated[arm][seed][
                    "validation_history_content_sha256"
                ],
            }
            for seed in CANONICAL_SEEDS
        }
        for arm in CANONICAL_ARMS
    }
    source_binding = {
        "evidence_index_path": None if source_path is None else str(source_path),
        "evidence_index_content_sha256": index_content_sha,
        "evidence_index_fingerprint": supplied_fingerprint,
        "validation_evidence_by_arm_seed": evidence_by_arm_seed,
        "matched_config_core_sha256": matched_config_core_sha,
        "matched_budget_core_sha256": next(iter(all_budget_cores)),
        "action_release_contract_sha256": next(iter(all_release_contracts)),
        "numeric_data_qc_contract_sha256": next(iter(all_numeric_data_qc_contracts)),
        "action_release_formal_manifest": validated["T0"][0]["action_release_contract"].get(
            "formal_release_manifest"
        ),
        "action_release_evidence_limitations": validated["T0"][0][
            "action_release_contract"
        ].get("evidence_limitations", []),
        "fixed_endpoint_update_number": next(iter(endpoint_updates)),
        "fixed_endpoint_global_timestep": next(iter(endpoint_timesteps)),
        "git_sha": next(iter(all_git_shas)),
        "source_tree_snapshot": validated["T0"][0]["source_tree_snapshot"],
        "source_tree_snapshot_binding_sha256": validated["T0"][0][
            "source_tree_snapshot"
        ]["binding_sha256"],
        "absolute_stage1_promotion_thresholds": absolute_stage1_thresholds,
        "tube_reference_fingerprint": validated["T3"][0]["runtime_contract"]["reference_fingerprint"],
        "tube_reference_id": validated["T3"][0]["runtime_contract"]["reference_id"],
        "tube_array_bundle_sha256": validated["T3"][0]["runtime_contract"][
            "array_bundle_sha256"
        ],
        "tube_mapping_sha256": validated["T3"][0]["runtime_contract"]["mapping_sha256"],
        "tube_trial_qc_review_schema_version": validated["T3"][0]["runtime_contract"][
            "trial_qc_review_schema_version"
        ],
        "tube_trial_qc_review_sha256": validated["T3"][0]["runtime_contract"][
            "trial_qc_review_sha256"
        ],
        "tube_phase_bin_count": validated["T3"][0]["runtime_contract"]["phase_bin_count"],
        "training_action_id": spec.action_id,
        "tube_action_id": spec.emg_trial_actions[0],
        "anchor_loss_spec_fingerprint": validated["T3"][0]["runtime_contract"]["anchor_loss_spec_fingerprint"],
        "matched_reward_core_fingerprint": validated["T3"][0]["runtime_contract"]["matched_reward_core_fingerprint"],
        "split_sha256": _expected_comparison_contract(spec)["split_sha256"],
        "arms": list(CANONICAL_ARMS),
        "seeds": list(CANONICAL_SEEDS),
    }
    unsigned_gate = {
        "schema_version": PAIRWISE_GATE_SCHEMA_VERSION,
        "action": spec.slug,
        "action_id": spec.action_id,
        "passed": all(bool(check["passed"]) for check in checks),
        "checks": checks,
        "statistics": statistics,
        "source_binding": source_binding,
    }
    return {**unsigned_gate, "binding_sha256": _canonical_sha256(unsigned_gate)}


def _validate_pairwise_gate_payload(payload: Mapping[str, Any], *, spec: ActionSpec) -> dict[str, Any]:
    gate = dict(payload)
    if (
        gate.get("schema_version") != PAIRWISE_GATE_SCHEMA_VERSION
        or gate.get("action") != spec.slug
        or gate.get("action_id") != spec.action_id
        or gate.get("passed") is not True
    ):
        raise ValueError("Stage-1 PEASD pairwise gate is not a passed artifact for this action")
    supplied = _require_sha256(gate.get("binding_sha256"), field="pairwise gate binding_sha256")
    unsigned = {key: value for key, value in gate.items() if key != "binding_sha256"}
    if supplied != _canonical_sha256(unsigned):
        raise ValueError("Stage-1 PEASD pairwise gate binding mismatch")
    return gate


def build_stage1_peasd_teacher_promotion(
    *,
    pairwise_gate: str | Path,
    blind_review: str | Path,
    blind_mapping: str | Path,
    action: str,
) -> dict[str, Any]:
    """Promote T3/seed-0 using an opaque review and private endpoint mapping."""

    # Keep ordinary tube/train-gate imports lightweight.  The recorder module
    # pulls the rendering stack and is needed only at the visual-promotion
    # boundary.
    from musclemimic.runner.validation_video_recorder import (
        STAGE1_BLIND_REVIEW_SCHEMA_VERSION,
        validate_stage1_peasd_blind_review,
    )

    spec = resolve(action)
    gate_path = _resolve_source_path(pairwise_gate)
    gate = _validate_pairwise_gate_payload(
        _require_mapping(load_json_strict(gate_path), field="pairwise gate"), spec=spec
    )
    selected = gate["source_binding"]["validation_evidence_by_arm_seed"][PRIMARY_TEACHER_ARM][
        str(PRIMARY_TEACHER_SEED)
    ]
    evidence_path = _resolve_source_path(str(selected["path"]))
    if _file_sha256(evidence_path) != selected["content_sha256"]:
        raise ValueError("pre-registered teacher validation evidence content changed")
    evidence = validate_stage1_peasd_validation_evidence(evidence_path)
    checkpoint = _require_mapping(evidence.get("checkpoint_identity"), field="teacher checkpoint identity")
    if checkpoint_identity(checkpoint["checkpoint_path"]) != checkpoint:
        raise ValueError("pre-registered teacher checkpoint identity changed")

    review_path = _resolve_source_path(blind_review)
    mapping_path = _resolve_source_path(blind_mapping)
    blind_validation = validate_stage1_peasd_blind_review(
        review_path,
        private_mapping=mapping_path,
        expected_candidate=checkpoint,
        expected_motions=list(spec.val_motion_paths),
    )
    review = _require_mapping(
        blind_validation.get("review"), field="opaque T3 visual review"
    )
    private_mapping = _require_mapping(
        blind_validation.get("private_mapping"), field="T3 private blind mapping"
    )
    package = _require_mapping(
        blind_validation.get("package"), field="T3 blind review package"
    )
    visual_report = _require_mapping(
        blind_validation.get("validation_report"),
        field="T3 reconstructed visual review report",
    )
    if visual_report.get("passed") is not True:
        raise ValueError("pre-registered T3 opaque visual review did not pass")
    package_manifest_path = _resolve_source_path(
        str(private_mapping.get("package_manifest_path", ""))
    )
    review_set_path = _resolve_source_path(
        str(private_mapping.get("endpoint_review_set_path", ""))
    )
    source = gate["source_binding"]
    emg_reference_binding = {
        "training_action_id": spec.action_id,
        "tube_action_id": spec.emg_trial_actions[0],
        "reference_id": source["tube_reference_id"],
        "reference_fingerprint": source["tube_reference_fingerprint"],
        "array_bundle_sha256": source["tube_array_bundle_sha256"],
        "mapping_sha256": source["tube_mapping_sha256"],
        "trial_qc_review_schema_version": source[
            "tube_trial_qc_review_schema_version"
        ],
        "trial_qc_review_sha256": source["tube_trial_qc_review_sha256"],
        "phase_bin_count": source["tube_phase_bin_count"],
    }
    unsigned = {
        "schema_version": PEASD_TEACHER_PROMOTION_SCHEMA_VERSION,
        "action": spec.slug,
        "action_id": spec.action_id,
        "teacher_arm": PRIMARY_TEACHER_ARM,
        "teacher_seed": PRIMARY_TEACHER_SEED,
        "selection_rule": PRIMARY_TEACHER_SELECTION_RULE,
        "emg_reference_binding": emg_reference_binding,
        "checkpoint": checkpoint,
        "validation_evidence": {
            "path": str(evidence_path),
            "content_sha256": _file_sha256(evidence_path),
            "evidence_fingerprint": evidence["evidence_fingerprint"],
        },
        "pairwise_gate": {
            "path": str(gate_path),
            "content_sha256": _file_sha256(gate_path),
            "binding_sha256": gate["binding_sha256"],
            "matched_config_core_sha256": gate["source_binding"]["matched_config_core_sha256"],
        },
        "visual_review": {
            "schema_version": PEASD_BLIND_VISUAL_EVIDENCE_SCHEMA_VERSION,
            "review": {
                "schema_version": STAGE1_BLIND_REVIEW_SCHEMA_VERSION,
                "path": str(review_path),
                "content_sha256": _file_sha256(review_path),
                "reviewer_id": str(review["reviewer_id"]),
            },
            "private_mapping": {
                "path": str(mapping_path),
                "content_sha256": _file_sha256(mapping_path),
                "binding_sha256": private_mapping["binding_sha256"],
            },
            "review_package": {
                "directory": str(package_manifest_path.parent),
                "manifest_path": str(package_manifest_path),
                "manifest_content_sha256": _file_sha256(package_manifest_path),
                "package_id": package["package_id"],
                "package_content_sha256": package["package_content_sha256"],
            },
            "endpoint_review_set": {
                "path": str(review_set_path),
                "content_sha256": _file_sha256(review_set_path),
                "binding_sha256": private_mapping[
                    "endpoint_review_set_binding_sha256"
                ],
            },
            "validation_report": visual_report,
        },
        "passed": True,
    }
    return {**unsigned, "binding_sha256": _canonical_sha256(unsigned)}


def validate_stage1_peasd_teacher_promotion(
    source: str | Path | Mapping[str, Any],
    *,
    expected_action: str | None = None,
    expected_checkpoint: str | Path | None = None,
    expected_tube: str | Path | None = None,
) -> dict[str, Any]:
    """Rebuild a PEASD teacher promotion from all bound source artifacts."""

    payload = (
        dict(source)
        if isinstance(source, Mapping)
        else _require_mapping(load_json_strict(source), field="PEASD teacher promotion")
    )
    if payload.get("schema_version") != PEASD_TEACHER_PROMOTION_SCHEMA_VERSION:
        raise ValueError("unsupported Stage-1 PEASD teacher promotion schema")
    spec = resolve(expected_action or str(payload.get("action", "")))
    if payload.get("action") != spec.slug or payload.get("action_id") != spec.action_id:
        raise ValueError("PEASD teacher promotion belongs to another action")
    supplied = _require_sha256(payload.get("binding_sha256"), field="PEASD promotion binding_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "binding_sha256"}
    if supplied != _canonical_sha256(unsigned):
        raise ValueError("PEASD teacher promotion binding mismatch")
    pairwise = _require_mapping(payload.get("pairwise_gate"), field="promotion pairwise_gate")
    gate_path = _resolve_source_path(str(pairwise.get("path", "")))
    if _file_sha256(gate_path) != pairwise.get("content_sha256"):
        raise ValueError("PEASD promotion pairwise gate content changed")
    rebuilt = build_stage1_peasd_teacher_promotion(
        pairwise_gate=gate_path,
        blind_review=str(
            payload.get("visual_review", {}).get("review", {}).get("path", "")
        ),
        blind_mapping=str(
            payload.get("visual_review", {})
            .get("private_mapping", {})
            .get("path", "")
        ),
        action=spec.slug,
    )
    if rebuilt != payload:
        raise ValueError("PEASD teacher promotion or one of its sources changed")
    if expected_checkpoint is not None:
        expected = checkpoint_identity(expected_checkpoint)
        if payload.get("checkpoint") != expected:
            raise ValueError("PEASD teacher promotion points to another checkpoint")
    if expected_tube is not None:
        tube = build_verified_tube_gate(expected_tube, action=spec.slug)["source"]
        expected_reference = {
            "training_action_id": spec.action_id,
            "tube_action_id": spec.emg_trial_actions[0],
            "reference_id": tube["reference_id"],
            "reference_fingerprint": tube["reference_fingerprint"],
            "array_bundle_sha256": tube["array_bundle_sha256"],
            "mapping_sha256": tube["mapping_sha256"],
            "trial_qc_review_schema_version": tube[
                "trial_qc_review_schema_version"
            ],
            "trial_qc_review_sha256": tube["trial_qc_review_sha256"],
            "phase_bin_count": tube["phase_bin_count"],
        }
        if payload.get("emg_reference_binding") != expected_reference:
            raise ValueError(
                "PEASD teacher promotion tube differs from downstream EMG reference"
            )
    return payload


def _write_and_print(path: str | Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(path, payload)
    print(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    tube = subparsers.add_parser("tube", help="validate a production EMG reference")
    tube.add_argument("--action", required=True)
    tube.add_argument("--tube", required=True)
    tube.add_argument("--output", required=True)
    tube.add_argument("--require-pass", action="store_true")
    index = subparsers.add_parser(
        "index", help="assemble and validate the exact 15 sealed validation artifacts"
    )
    index.add_argument("--action", required=True)
    index.add_argument(
        "--evidence",
        action="append",
        required=True,
        metavar="ARM:SEED:PATH",
        help="repeat exactly once for every T0--T4 / seed 0--2 pair",
    )
    index.add_argument("--notes", default="")
    index.add_argument("--output", required=True)
    pairwise = subparsers.add_parser("pairwise", help="evaluate sealed T0--T4 evidence")
    pairwise.add_argument("--action", required=True)
    pairwise.add_argument("--metrics", required=True, help="sealed evidence index JSON")
    pairwise.add_argument(
        "--blind-review",
        required=True,
        help="completed reviewer-visible stage1_blind_review_package/review.json",
    )
    pairwise.add_argument(
        "--blind-mapping",
        required=True,
        help="private stage1_blind_private_mapping.json; never share with reviewer",
    )
    pairwise.add_argument("--output", required=True, help="pairwise gate JSON")
    pairwise.add_argument("--promotion-output", required=True)
    pairwise.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()

    if args.command == "tube":
        try:
            report = build_verified_tube_gate(args.tube, action=args.action)
        except Exception as exc:
            unsigned = {
                "schema_version": TUBE_GATE_SCHEMA_VERSION,
                "action": str(args.action),
                "passed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            report = {**unsigned, "binding_sha256": _canonical_sha256(unsigned)}
        _write_and_print(args.output, report)
        return 2 if args.require_pass and report.get("passed") is not True else 0

    if args.command == "index":
        report = build_pairwise_evidence_index(
            action=args.action,
            evidence_selectors=args.evidence,
            notes=args.notes,
        )
        _write_and_print(args.output, report)
        return 0

    report = evaluate_pairwise_promotion(args.metrics, action=args.action)
    gate_path = _atomic_write(args.output, report)
    if report.get("passed") is True:
        promotion = build_stage1_peasd_teacher_promotion(
            pairwise_gate=gate_path,
            blind_review=args.blind_review,
            blind_mapping=args.blind_mapping,
            action=args.action,
        )
        _atomic_write(args.promotion_output, promotion)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 2 if args.require_pass and report.get("passed") is not True else 0


if __name__ == "__main__":
    raise SystemExit(main())

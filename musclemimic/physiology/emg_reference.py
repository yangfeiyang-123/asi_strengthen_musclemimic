"""Phase-conditioned surface-EMG reference tubes for partial physiological anchoring.

This module owns the *human side* of the PEASD contract.  It stores, validates
and fingerprints a phase-conditioned uncertainty tube built from repeated real
sEMG trials, so that training code can ask "was this simulated activation inside
the range a human actually produced at this movement phase?" instead of matching
a single averaged curve.

Three deliberate limits are encoded here and must not be relaxed silently:

* the tube describes only the ``M`` electrode channels that have a verified
  model homolog.  It is never a ground-truth label for all 354 actuators;
* a tube whose mapping or event evidence is unreviewed is loadable for
  diagnostics but :func:`resolve_emg_reference_reward_gate` refuses to arm it
  for training;
* the dispersion leg is a robust median/MAD scale, not a standard deviation,
  because trial counts are small and single trials are heavy-tailed.

The numeric convention matches ``continuity_groups``: build a semantic payload,
fingerprint it, and re-validate the payload through the same validator that
reads it back from disk.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

EMG_REFERENCE_TUBE_SCHEMA_VERSION = "emg_phase_reference_tube_v2"
"""Schema string stamped into every tube artifact."""

EMG_DUAL_TRACK_NORMALIZATION_SCHEMA_VERSION = "emg_dual_track_normalization_v1"
"""Unclipped percent-MVC audit plus train-P99 model-normalization contract."""

EMG_AUDIT_NORMALIZATION = "percent_mvc_unclipped"
EMG_MODEL_NORMALIZATION = "train_p99_per_channel"
EMG_NORMALIZATION_PERCENTILE = 99.0
EMG_NORMALIZATION_EPS = 1e-8
EMG_MVC_QUALITY_LEVELS = (
    "good",
    "questionable",
    "unreliable",
    "invalid_for_absolute_amplitude",
)

EMG_TUBE_STATISTIC = "median_mad_1p4826_v1"
"""Robust centre/scale estimator: median and 1.4826 * MAD plus a scale floor."""

EMG_TUBE_MAD_SCALE = 1.4826
"""Consistency constant converting MAD to a Gaussian-equivalent sigma."""

EMG_TUBE_SCALE_FLOOR = 0.02
"""Default additive scale floor; keeps a zero-dispersion bin from being a hard label."""

EMG_TUBE_MIN_TRIALS = 3
"""A phase bin summarised from fewer trials than this is marked invalid."""

DEFAULT_EMG_REFERENCE_BEHAVIOR = "diagnostics_only_no_reward"
"""Tubes are inert until an explicit review promotes them."""

EMG_REFERENCE_REVIEW_STATES = ("provisional", "verified")
"""Only ``verified`` may drive a training signal."""

EMG_TRIAL_QC_REVIEW_SCHEMA_VERSION = "emg_trial_channel_qc_review_v1"
"""Human trial/channel QC contract required by every training-enabled tube."""

EMG_TRIAL_QC_REVIEW_FILENAME = "emg_trial_qc_review.json"
"""Exact reviewed evidence bundle stored beside a training-enabled tube."""

_REQUIRED_TRIAL_QC_RISKS = frozenset({"s9_progressive_near_flatline"})

EMG_SYNERGY_PROJECTION_METHOD = "ridge_pseudoinverse_relu_v1"
"""In-graph simulated-side synergy projection ``h = relu(Q y)``.

The offline evaluators in :mod:`musclemimic.evaluation` use non-negative least
squares, which is the better estimator but is not differentiable or jittable.
The training path therefore uses a ridge pseudo-inverse followed by a ReLU, and
records that choice in the fingerprint so a runtime spec is never compared
against a tube fitted under the other convention.
"""

EMG_SYNERGY_RIDGE = 1e-3
"""Ridge term lambda in ``Q = (W^T W + lambda I)^-1 W^T``."""


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r} in EMG reference artifact")
        result[key] = value
    return result


def _require_keys(
    value: Mapping[str, Any],
    *,
    required: Sequence[str],
    optional: Sequence[str] = (),
    context: str,
) -> None:
    present = set(value)
    missing = sorted(set(required) - present)
    if missing:
        raise ValueError(f"{context} is missing required keys {missing}")
    unknown = sorted(present - set(required) - set(optional))
    if unknown:
        raise ValueError(f"{context} has unsupported keys {unknown}")


def _nonempty_text(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be a non-empty string")
    return text


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return result


def _finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{field} must be finite numeric")
    return result


def _require_sha256(value: Any, *, field: str) -> str:
    text = _nonempty_text(value, field=field)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{field} must be a lowercase sha256 hex digest")
    return text


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    import hashlib

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def classify_mvc_reference_quality(ratio: Any) -> tuple[str, float]:
    """Map ``P99(task) / MVC`` to the project QC grade and loss confidence.

    Values above one are deliberately accepted.  The confidence only controls
    absolute-amplitude supervision; it never deletes a trial or clips the
    percent-MVC audit signal.
    """

    value = _finite_float(ratio, field="task_p99_over_mvc")
    if value < 0.0:
        raise ValueError("task_p99_over_mvc must be non-negative")
    if value <= 1.20:
        return "good", 1.0
    if value <= 1.50:
        return "questionable", 0.7
    if value <= 2.00:
        return "unreliable", 0.4
    return "invalid_for_absolute_amplitude", 0.2


def build_emg_dual_track_normalization(
    *,
    action_samples: Mapping[str, Sequence[Any]],
    channel_names: Sequence[str],
    training_cohorts: Mapping[str, Sequence[Mapping[str, Any]]],
    mvc_final_reference_mv: Any,
    mvc_reference_binding: Mapping[str, Any],
    mvc_original_mv: Any | None = None,
    epsilon: float = EMG_NORMALIZATION_EPS,
) -> dict[str, Any]:
    """Build the leakage-safe dual-track normalization identity.

    ``action_samples`` is the exact normalization-training cohort.  Its arrays
    may have different durations, but each must be ``[sample, channel]`` and
    remain in unclipped percent-MVC units.  A formal caller must first restrict
    it to human-reviewed clean training trials; a provisional audit may bind an
    explicitly unreviewed candidate cohort but cannot arm training.  P99 is
    estimated across those exact samples before phase binning and is then frozen
    for every downstream split.
    """

    names = tuple(_nonempty_text(name, field="channel_names") for name in channel_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("channel_names must be non-empty and unique")
    eps = _finite_float(epsilon, field="normalization epsilon")
    if eps <= 0.0:
        raise ValueError("normalization epsilon must be positive")
    final_mvc = np.asarray(mvc_final_reference_mv, dtype=np.float64)
    original_mvc = final_mvc.copy() if mvc_original_mv is None else np.asarray(mvc_original_mv, dtype=np.float64)
    expected_shape = (len(names),)
    for field, values in (
        ("mvc_final_reference_mV", final_mvc),
        ("mvc_original_mV", original_mvc),
    ):
        if values.shape != expected_shape:
            raise ValueError(f"{field} must have shape {expected_shape}, found {values.shape}")
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError(f"{field} must be finite and positive")

    mvc_source = dict(mvc_reference_binding)
    _require_keys(
        mvc_source,
        required=("path", "sha256", "scope", "algorithm"),
        context="MVC reference binding",
    )
    _nonempty_text(mvc_source["path"], field="mvc_reference.path")
    _require_sha256(mvc_source["sha256"], field="mvc_reference.sha256")
    _nonempty_text(mvc_source["scope"], field="mvc_reference.scope")
    _nonempty_text(mvc_source["algorithm"], field="mvc_reference.algorithm")

    action_statistics: list[dict[str, Any]] = []
    for action_id, raw_samples in action_samples.items():
        action = _nonempty_text(action_id, field="action_id")
        samples = [np.asarray(item, dtype=np.float64) for item in raw_samples]
        if not samples:
            raise ValueError(f"action {action!r} has no normalization-training samples")
        for index, values in enumerate(samples):
            if values.ndim != 2 or values.shape[1] != len(names):
                raise ValueError(
                    f"action {action!r} normalization sample {index} must be "
                    f"[sample, {len(names)}], found {values.shape}"
                )
            if not np.all(np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(f"action {action!r} normalization sample {index} must be finite and non-negative")
        concatenated = np.concatenate(samples, axis=0)
        p95 = np.percentile(concatenated, 95.0, axis=0)
        p99 = np.percentile(concatenated, EMG_NORMALIZATION_PERCENTILE, axis=0)
        maximum = np.max(concatenated, axis=0)
        if np.any(p99 <= eps):
            starved = [names[index] for index in np.flatnonzero(p99 <= eps)]
            raise ValueError(
                f"action {action!r} has zero/near-zero train-P99 scale for channels {starved}; "
                "this is a signal-quality failure, not an MVC exceedance"
            )

        raw_cohort = training_cohorts.get(action)
        if not isinstance(raw_cohort, Sequence) or not raw_cohort:
            raise ValueError(f"action {action!r} requires a non-empty training cohort binding")
        cohort: list[dict[str, str]] = []
        seen_trials: set[str] = set()
        for index, record in enumerate(raw_cohort):
            if not isinstance(record, Mapping):
                raise ValueError(f"training cohort {action}[{index}] must be an object")
            trial_id = _nonempty_text(record.get("trial_id"), field="training cohort trial_id")
            if trial_id in seen_trials:
                raise ValueError(f"training cohort repeats trial_id {trial_id!r}")
            seen_trials.add(trial_id)
            source_sha = _require_sha256(
                record.get("mvc_normalized_emg_sha256"),
                field="training cohort mvc_normalized_emg_sha256",
            )
            cohort.append({"trial_id": trial_id, "mvc_normalized_emg_sha256": source_sha})
        if len(cohort) != len(samples):
            raise ValueError(
                f"action {action!r} has {len(samples)} normalization samples but {len(cohort)} training cohort records"
            )

        channels: list[dict[str, Any]] = []
        for index, name in enumerate(names):
            quality, confidence = classify_mvc_reference_quality(p99[index])
            channels.append(
                {
                    "channel_name": name,
                    "mvc_original_mV": float(original_mvc[index]),
                    "mvc_final_reference_mV": float(final_mvc[index]),
                    "mvc_quality": quality,
                    "task_p95_mV": float(p95[index] * final_mvc[index]),
                    "task_p99_mV": float(p99[index] * final_mvc[index]),
                    "task_max_mV": float(maximum[index] * final_mvc[index]),
                    "task_p95_percent_mvc": float(p95[index]),
                    "task_p99_over_mvc": float(p99[index]),
                    "task_max_over_mvc": float(maximum[index]),
                    "normalization_report": EMG_AUDIT_NORMALIZATION,
                    "normalization_synergy": EMG_MODEL_NORMALIZATION,
                    "robust_scale_mV": float(p99[index] * final_mvc[index]),
                    "robust_scale_percent_mvc": float(p99[index]),
                    "amplitude_confidence": confidence,
                }
            )
        action_statistics.append(
            {
                "action_id": action,
                "training_cohort_scope": "tube_training_trials_only",
                "training_trial_count": len(cohort),
                "training_trials": cohort,
                "training_cohort_sha256": _canonical_json_sha256({"action_id": action, "training_trials": cohort}),
                "channels": channels,
            }
        )

    return {
        "schema_version": EMG_DUAL_TRACK_NORMALIZATION_SCHEMA_VERSION,
        "audit_normalization": EMG_AUDIT_NORMALIZATION,
        "model_normalization": EMG_MODEL_NORMALIZATION,
        "percentile": EMG_NORMALIZATION_PERCENTILE,
        "epsilon": eps,
        "mvc_reference": mvc_source,
        "actions": action_statistics,
    }


def _validate_dual_track_normalization(
    binding: Mapping[str, Any],
    *,
    action_ids: Sequence[str],
    channel_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the normalization report and return scale/confidence arrays."""

    payload = dict(binding)
    _require_keys(
        payload,
        required=(
            "schema_version",
            "audit_normalization",
            "model_normalization",
            "percentile",
            "epsilon",
            "mvc_reference",
            "actions",
        ),
        context="EMG dual-track normalization",
    )
    if payload["schema_version"] != EMG_DUAL_TRACK_NORMALIZATION_SCHEMA_VERSION:
        raise ValueError("unsupported EMG dual-track normalization schema")
    if payload["audit_normalization"] != EMG_AUDIT_NORMALIZATION:
        raise ValueError("EMG audit normalization must preserve unclipped percent-MVC")
    if payload["model_normalization"] != EMG_MODEL_NORMALIZATION:
        raise ValueError("EMG model normalization must use train-P99 per channel")
    if _finite_float(payload["percentile"], field="normalization percentile") != 99.0:
        raise ValueError("EMG model normalization percentile must be exactly P99")
    epsilon = _finite_float(payload["epsilon"], field="normalization epsilon")
    if epsilon <= 0.0:
        raise ValueError("normalization epsilon must be positive")
    mvc_source = payload["mvc_reference"]
    if not isinstance(mvc_source, Mapping):
        raise ValueError("normalization mvc_reference must be an object")
    _require_keys(
        mvc_source,
        required=("path", "sha256", "scope", "algorithm"),
        context="normalization mvc_reference",
    )
    _nonempty_text(mvc_source["path"], field="mvc_reference.path")
    _require_sha256(mvc_source["sha256"], field="mvc_reference.sha256")
    _nonempty_text(mvc_source["scope"], field="mvc_reference.scope")
    _nonempty_text(mvc_source["algorithm"], field="mvc_reference.algorithm")

    actions = payload["actions"]
    if not isinstance(actions, list) or len(actions) != len(action_ids):
        raise ValueError("normalization actions must match the tube action count")
    scales = np.empty((len(action_ids), len(channel_names)), dtype=np.float64)
    confidences = np.empty_like(scales)
    for action_index, (entry, expected_action) in enumerate(zip(actions, action_ids, strict=True)):
        if not isinstance(entry, Mapping):
            raise ValueError("normalization action entry must be an object")
        _require_keys(
            entry,
            required=(
                "action_id",
                "training_cohort_scope",
                "training_trial_count",
                "training_trials",
                "training_cohort_sha256",
                "channels",
            ),
            context=f"normalization action {action_index}",
        )
        if entry["action_id"] != expected_action:
            raise ValueError("normalization action order/identity differs from the tube")
        if entry["training_cohort_scope"] != "tube_training_trials_only":
            raise ValueError("normalization scale must be estimated from training trials only")
        trials = entry["training_trials"]
        if not isinstance(trials, list) or not trials:
            raise ValueError("normalization training_trials must be non-empty")
        if _positive_int(entry["training_trial_count"], field="training_trial_count") != len(trials):
            raise ValueError("normalization training_trial_count is inconsistent")
        seen_trials: set[str] = set()
        canonical_trials: list[dict[str, str]] = []
        for record in trials:
            if not isinstance(record, Mapping):
                raise ValueError("normalization training trial must be an object")
            _require_keys(
                record,
                required=("trial_id", "mvc_normalized_emg_sha256"),
                context="normalization training trial",
            )
            trial_id = _nonempty_text(record["trial_id"], field="training trial_id")
            if trial_id in seen_trials:
                raise ValueError(f"normalization repeats trial {trial_id!r}")
            seen_trials.add(trial_id)
            canonical_trials.append(
                {
                    "trial_id": trial_id,
                    "mvc_normalized_emg_sha256": _require_sha256(
                        record["mvc_normalized_emg_sha256"],
                        field="training trial mvc_normalized_emg_sha256",
                    ),
                }
            )
        expected_cohort_sha = _canonical_json_sha256(
            {"action_id": expected_action, "training_trials": canonical_trials}
        )
        if entry["training_cohort_sha256"] != expected_cohort_sha:
            raise ValueError("normalization training cohort fingerprint mismatch")

        channels = entry["channels"]
        if not isinstance(channels, list) or len(channels) != len(channel_names):
            raise ValueError("normalization channel statistics differ from tube channels")
        for channel_index, (stats, expected_channel) in enumerate(zip(channels, channel_names, strict=True)):
            if not isinstance(stats, Mapping):
                raise ValueError("normalization channel statistics must be objects")
            required = (
                "channel_name",
                "mvc_original_mV",
                "mvc_final_reference_mV",
                "mvc_quality",
                "task_p95_mV",
                "task_p99_mV",
                "task_max_mV",
                "task_p95_percent_mvc",
                "task_p99_over_mvc",
                "task_max_over_mvc",
                "normalization_report",
                "normalization_synergy",
                "robust_scale_mV",
                "robust_scale_percent_mvc",
                "amplitude_confidence",
            )
            _require_keys(stats, required=required, context="normalization channel")
            if stats["channel_name"] != expected_channel:
                raise ValueError("normalization channel order/identity differs from the tube")
            mvc_original = _finite_float(stats["mvc_original_mV"], field="mvc_original_mV")
            mvc_final = _finite_float(stats["mvc_final_reference_mV"], field="mvc_final_reference_mV")
            if mvc_original <= 0.0 or mvc_final <= 0.0:
                raise ValueError("MVC references must be positive")
            numeric = {
                key: _finite_float(stats[key], field=key)
                for key in (
                    "task_p95_mV",
                    "task_p99_mV",
                    "task_max_mV",
                    "task_p95_percent_mvc",
                    "task_p99_over_mvc",
                    "task_max_over_mvc",
                    "robust_scale_mV",
                    "robust_scale_percent_mvc",
                    "amplitude_confidence",
                )
            }
            if any(value < 0.0 for value in numeric.values()):
                raise ValueError("normalization channel statistics must be non-negative")
            if not (numeric["task_p95_percent_mvc"] <= numeric["task_p99_over_mvc"] <= numeric["task_max_over_mvc"]):
                raise ValueError("normalization percent-MVC quantiles are not monotonic")
            for statistic in ("p95", "p99", "max"):
                mv_key = f"task_{statistic}_mV"
                ratio_key = (
                    "task_p99_over_mvc"
                    if statistic == "p99"
                    else f"task_{statistic}_percent_mvc"
                    if statistic == "p95"
                    else "task_max_over_mvc"
                )
                if not np.isclose(
                    numeric[mv_key],
                    numeric[ratio_key] * mvc_final,
                    rtol=1e-9,
                    atol=1e-12,
                ):
                    raise ValueError(f"{mv_key} is inconsistent with the final MVC reference")
            if not np.isclose(
                numeric["task_p99_over_mvc"],
                numeric["robust_scale_percent_mvc"],
                rtol=1e-9,
                atol=1e-12,
            ):
                raise ValueError("robust percent-MVC scale must equal training P99")
            if not np.isclose(
                numeric["task_p99_mV"],
                numeric["robust_scale_mV"],
                rtol=1e-9,
                atol=1e-12,
            ):
                raise ValueError("robust mV scale is inconsistent with P99/MVC")
            quality, confidence = classify_mvc_reference_quality(numeric["task_p99_over_mvc"])
            if stats["mvc_quality"] != quality:
                raise ValueError("mvc_quality is inconsistent with task P99/MVC")
            if not np.isclose(numeric["amplitude_confidence"], confidence):
                raise ValueError("amplitude_confidence is inconsistent with mvc_quality")
            if stats["normalization_report"] != EMG_AUDIT_NORMALIZATION:
                raise ValueError("percent-MVC audit normalization must remain unclipped")
            if stats["normalization_synergy"] != EMG_MODEL_NORMALIZATION:
                raise ValueError("synergy normalization must use train-P99")
            if numeric["robust_scale_percent_mvc"] <= epsilon:
                raise ValueError("train-P99 robust scale must be positive")
            scales[action_index, channel_index] = numeric["robust_scale_percent_mvc"]
            confidences[action_index, channel_index] = confidence
    return scales.astype(np.float32), confidences.astype(np.float32)


def _validate_trial_qc_review_binding(
    provenance: Mapping[str, Any],
    *,
    action_ids: Sequence[str],
    channel_names: Sequence[str],
    mapping_sha256: str,
    source_path: Path | None = None,
) -> dict[str, Any]:
    """Validate the immutable human QC evidence behind a formal tube."""

    raw = provenance.get("trial_qc_review")
    if not isinstance(raw, Mapping):
        raise ValueError("training-enabled EMG reference requires provenance.trial_qc_review")
    binding = dict(raw)
    if binding.get("schema_version") != EMG_TRIAL_QC_REVIEW_SCHEMA_VERSION:
        raise ValueError("provenance.trial_qc_review has unsupported schema_version")
    if str(binding.get("review_status", "")).strip().lower() != "verified":
        raise ValueError("provenance.trial_qc_review must be verified")
    if binding.get("training_enabled") is not True:
        raise ValueError("provenance.trial_qc_review must set training_enabled=true")
    action = _nonempty_text(binding.get("action"), field="provenance.trial_qc_review.action")
    if len(action_ids) != 1 or action != str(action_ids[0]):
        raise ValueError("trial-QC action must exactly match the single-action tube identity")
    review_sha256 = _require_sha256(
        binding.get("review_sha256"),
        field="provenance.trial_qc_review.review_sha256",
    )
    bound_mapping_sha256 = _require_sha256(
        binding.get("mapping_sha256"),
        field="provenance.trial_qc_review.mapping_sha256",
    )
    if bound_mapping_sha256 != mapping_sha256:
        raise ValueError("trial-QC review is bound to a different mapping")
    _nonempty_text(
        binding.get("reviewer_id"),
        field="provenance.trial_qc_review.reviewer_id",
    )
    _nonempty_text(
        binding.get("reviewed_at"),
        field="provenance.trial_qc_review.reviewed_at",
    )
    evidence = binding.get("review_evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("trial-QC review_evidence must be a non-empty list")
    for index, item in enumerate(evidence):
        _nonempty_text(item, field=f"provenance.trial_qc_review.review_evidence[{index}]")

    trial_decisions = binding.get("trial_decisions")
    if not isinstance(trial_decisions, list) or not trial_decisions:
        raise ValueError("trial-QC binding must contain trial_decisions")
    included = 0
    seen_trials: set[str] = set()
    for index, entry in enumerate(trial_decisions):
        if not isinstance(entry, Mapping):
            raise ValueError(f"trial-QC trial_decisions[{index}] must be an object")
        trial_id = _nonempty_text(entry.get("trial_id"), field=f"trial-QC trial_decisions[{index}].trial_id")
        if trial_id in seen_trials:
            raise ValueError(f"duplicate trial-QC trial_id {trial_id!r}")
        seen_trials.add(trial_id)
        decision = str(entry.get("decision", "")).strip().lower()
        if decision not in {"include", "exclude"}:
            raise ValueError("trial-QC trial decision must be include or exclude")
        included += int(decision == "include")
        _nonempty_text(entry.get("reason"), field=f"trial-QC trial_decisions[{index}].reason")
        _require_sha256(
            entry.get("mvc_normalized_emg_sha256"),
            field=f"trial-QC trial_decisions[{index}].mvc_normalized_emg_sha256",
        )
        _require_sha256(
            entry.get("preprocessing_qc_sha256"),
            field=f"trial-QC trial_decisions[{index}].preprocessing_qc_sha256",
        )
    if included <= 0:
        raise ValueError("trial-QC binding includes no training trial")

    channel_decisions = binding.get("channel_decisions")
    if not isinstance(channel_decisions, list):
        raise ValueError("trial-QC binding must contain channel_decisions")
    decided_channels: set[str] = set()
    for index, entry in enumerate(channel_decisions):
        if not isinstance(entry, Mapping):
            raise ValueError(f"trial-QC channel_decisions[{index}] must be an object")
        name = _nonempty_text(
            entry.get("emg_channel"),
            field=f"trial-QC channel_decisions[{index}].emg_channel",
        )
        if name in decided_channels:
            raise ValueError(f"duplicate trial-QC channel {name!r}")
        decided_channels.add(name)
        if str(entry.get("decision", "")).strip().lower() != "include_after_review":
            raise ValueError("formal tube cannot silently exclude a channel; create a new mapping/profile ABI")
        _nonempty_text(
            entry.get("reason"),
            field=f"trial-QC channel_decisions[{index}].reason",
        )
    if decided_channels != set(channel_names):
        raise ValueError("trial-QC channel decisions do not match the tube channel identity")

    risk_decisions = binding.get("risk_decisions")
    if not isinstance(risk_decisions, list):
        raise ValueError("trial-QC binding must contain risk_decisions")
    resolved_risks: set[str] = set()
    for index, entry in enumerate(risk_decisions):
        if not isinstance(entry, Mapping):
            raise ValueError(f"trial-QC risk_decisions[{index}] must be an object")
        risk_id = _nonempty_text(entry.get("risk_id"), field=f"trial-QC risk_decisions[{index}].risk_id")
        decision = str(entry.get("decision", "")).strip().lower()
        if decision not in {"accepted_after_review", "mitigated"}:
            raise ValueError(f"trial-QC risk {risk_id!r} remains unresolved")
        _nonempty_text(entry.get("reason"), field=f"trial-QC risk_decisions[{index}].reason")
        risk_evidence = entry.get("evidence")
        if not isinstance(risk_evidence, list) or not risk_evidence:
            raise ValueError(f"trial-QC risk {risk_id!r} requires evidence")
        for evidence_index, item in enumerate(risk_evidence):
            _nonempty_text(
                item,
                field=(f"trial-QC risk_decisions[{index}].evidence[{evidence_index}]"),
            )
        resolved_risks.add(risk_id)
    missing_risks = sorted(_REQUIRED_TRIAL_QC_RISKS - resolved_risks)
    if missing_risks:
        raise ValueError(f"trial-QC binding leaves known risks unresolved: {missing_risks}")

    if source_path is not None:
        review_path = source_path.with_name(EMG_TRIAL_QC_REVIEW_FILENAME)
        if not review_path.is_file():
            raise FileNotFoundError(f"training-enabled EMG trial-QC review bundle is absent: {review_path}")
        actual_review_sha256 = hashlib.sha256(review_path.read_bytes()).hexdigest()
        if actual_review_sha256 != review_sha256:
            raise ValueError("EMG trial-QC review bundle content hash mismatch")
    return binding


@dataclass(frozen=True)
class EmgPhaseReferenceTube:
    """A validated, fingerprinted phase-conditioned EMG reference tube.

    Array layout is ``[action, phase_bin, channel]`` for the activation legs and
    ``[action, phase_bin, synergy]`` for the synergy legs.  ``phase_bin`` indexes
    a uniform partition of the normalised movement phase ``[0, 1]``.
    """

    schema_version: str
    reference_id: str
    review_status: str
    training_enabled: bool
    default_behavior: str
    statistic: str
    scale_floor: float
    min_trials: int
    mapping_binding: dict[str, Any]
    synergy_binding: dict[str, Any]
    normalization_binding: dict[str, Any]
    action_ids: tuple[str, ...]
    channel_names: tuple[str, ...]
    phase_bin_count: int
    anchor_mean: np.ndarray
    anchor_scale: np.ndarray
    mvc_anchor_mean: np.ndarray
    mvc_anchor_scale: np.ndarray
    robust_scale: np.ndarray
    amplitude_confidence: np.ndarray
    anchor_valid: np.ndarray
    anchor_trial_count: np.ndarray
    synergy_mean: np.ndarray
    synergy_scale: np.ndarray
    synergy_valid: np.ndarray
    synergy_basis: np.ndarray
    provenance: dict[str, Any]
    reference_fingerprint: str
    array_bundle_sha256: str | None = None
    notes: str = ""
    source_path: Path | None = None

    @property
    def channel_count(self) -> int:
        """``M``, the number of model-comparable electrode channels."""

        return len(self.channel_names)

    @property
    def synergy_count(self) -> int:
        """``K``, the number of extracted coordination components."""

        return int(self.synergy_basis.shape[1])

    @property
    def action_count(self) -> int:
        return len(self.action_ids)

    def action_index(self, action_id: str) -> int:
        """Resolve an action identifier to its row, failing closed on unknowns."""

        try:
            return self.action_ids.index(str(action_id))
        except ValueError as exc:
            raise KeyError(
                f"action {action_id!r} is absent from EMG reference {self.reference_id!r}; "
                f"known actions are {list(self.action_ids)}"
            ) from exc

    def to_manifest(self) -> dict[str, Any]:
        """Serialisable identity block, without the bulk arrays."""

        return {
            "schema_version": self.schema_version,
            "reference_id": self.reference_id,
            "review_status": self.review_status,
            "training_enabled": self.training_enabled,
            "default_behavior": self.default_behavior,
            "statistic": self.statistic,
            "scale_floor": self.scale_floor,
            "min_trials": self.min_trials,
            "mapping_binding": dict(self.mapping_binding),
            "synergy_binding": dict(self.synergy_binding),
            "normalization_binding": dict(self.normalization_binding),
            "action_ids": list(self.action_ids),
            "channel_names": list(self.channel_names),
            "phase_bin_count": self.phase_bin_count,
            "channel_count": self.channel_count,
            "synergy_count": self.synergy_count,
            "valid_anchor_bin_fraction": float(np.mean(self.anchor_valid)),
            "valid_synergy_bin_fraction": float(np.mean(self.synergy_valid)),
            "provenance": dict(self.provenance),
            "reference_fingerprint": self.reference_fingerprint,
            "array_bundle_sha256": self.array_bundle_sha256,
            "notes": self.notes,
        }


def emg_reference_fingerprint(payload: Mapping[str, Any]) -> str:
    """Fingerprint tube semantics; the array bundle has its own byte digest."""

    material = {str(key): value for key, value in payload.items()}
    material.pop("reference_fingerprint", None)
    material.pop("array_bundle_sha256", None)
    return _canonical_json_sha256(material)


_IDENTITY_REQUIRED = (
    "schema_version",
    "reference_id",
    "review_status",
    "training_enabled",
    "default_behavior",
    "statistic",
    "scale_floor",
    "min_trials",
    "mapping_binding",
    "synergy_binding",
    "normalization_binding",
    "action_ids",
    "channel_names",
    "phase_bin_count",
    "provenance",
)
_IDENTITY_OPTIONAL = (
    "notes",
    "reference_fingerprint",
    "review_evidence",
    "array_bundle_sha256",
)

_MAPPING_BINDING_REQUIRED = (
    "mapping_id",
    "mapping_sha256",
    "mapping_review_status",
    "acquired_channel_count",
    "comparable_channel_count",
    "actuator_schema_hash",
)

_SYNERGY_BINDING_REQUIRED = (
    "basis_id",
    "basis_sha256",
    "synergy_count",
    "channel_normalization",
    "projection_method",
    "projection_ridge",
)

_ARRAY_NAMES = (
    "anchor_mean",
    "anchor_scale",
    "mvc_anchor_mean",
    "mvc_anchor_scale",
    "robust_scale",
    "amplitude_confidence",
    "anchor_valid",
    "anchor_trial_count",
    "synergy_mean",
    "synergy_scale",
    "synergy_valid",
    "synergy_basis",
)


def _validate_unit_interval(array: np.ndarray, *, field: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{field} must be finite")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"{field} must lie inside the unit interval")
    return values.astype(np.float32)


def _validate_nonnegative(array: np.ndarray, *, field: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{field} must be finite")
    if np.any(values < 0.0):
        raise ValueError(f"{field} must be non-negative")
    return values.astype(np.float32)


def validate_emg_phase_reference_tube(
    payload: Mapping[str, Any],
    arrays: Mapping[str, Any],
    *,
    source_path: Path | None = None,
) -> EmgPhaseReferenceTube:
    """Validate an identity payload plus its arrays, failing closed on any gap."""

    _require_keys(
        payload,
        required=_IDENTITY_REQUIRED,
        optional=_IDENTITY_OPTIONAL,
        context="EMG reference identity",
    )
    schema_version = _nonempty_text(payload["schema_version"], field="schema_version")
    if schema_version != EMG_REFERENCE_TUBE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported EMG reference schema {schema_version!r}; expected {EMG_REFERENCE_TUBE_SCHEMA_VERSION!r}"
        )
    statistic = _nonempty_text(payload["statistic"], field="statistic")
    if statistic != EMG_TUBE_STATISTIC:
        raise ValueError(f"unsupported EMG tube statistic {statistic!r}")
    review_status = _nonempty_text(payload["review_status"], field="review_status")
    if review_status not in EMG_REFERENCE_REVIEW_STATES:
        raise ValueError(f"review_status must be one of {list(EMG_REFERENCE_REVIEW_STATES)}")
    training_enabled = payload["training_enabled"]
    if not isinstance(training_enabled, bool):
        raise ValueError("training_enabled must be boolean")
    if training_enabled and review_status != "verified":
        raise ValueError("training_enabled requires review_status=verified")
    review_evidence = list(payload.get("review_evidence", ()))
    if review_status == "verified" and not review_evidence:
        raise ValueError("review_status=verified requires non-empty review_evidence")
    for index, item in enumerate(review_evidence):
        _nonempty_text(item, field=f"review_evidence[{index}]")

    mapping_binding = dict(payload["mapping_binding"])
    _require_keys(
        mapping_binding,
        required=_MAPPING_BINDING_REQUIRED,
        context="EMG reference mapping_binding",
    )
    _require_sha256(mapping_binding["mapping_sha256"], field="mapping_binding.mapping_sha256")
    _require_sha256(
        mapping_binding["actuator_schema_hash"],
        field="mapping_binding.actuator_schema_hash",
    )
    acquired = _positive_int(
        mapping_binding["acquired_channel_count"],
        field="mapping_binding.acquired_channel_count",
    )
    comparable = _positive_int(
        mapping_binding["comparable_channel_count"],
        field="mapping_binding.comparable_channel_count",
    )
    if comparable > acquired:
        raise ValueError("comparable_channel_count cannot exceed acquired_channel_count")

    synergy_binding = dict(payload["synergy_binding"])
    _require_keys(
        synergy_binding,
        required=_SYNERGY_BINDING_REQUIRED,
        context="EMG reference synergy_binding",
    )
    _require_sha256(synergy_binding["basis_sha256"], field="synergy_binding.basis_sha256")
    projection_method = _nonempty_text(
        synergy_binding["projection_method"],
        field="synergy_binding.projection_method",
    )
    if projection_method != EMG_SYNERGY_PROJECTION_METHOD:
        raise ValueError(f"unsupported synergy projection method {projection_method!r}")
    projection_ridge = _finite_float(
        synergy_binding["projection_ridge"],
        field="synergy_binding.projection_ridge",
    )
    if projection_ridge <= 0.0:
        raise ValueError("synergy_binding.projection_ridge must be positive")

    action_ids = tuple(_nonempty_text(item, field="action_ids") for item in payload["action_ids"])
    if not action_ids:
        raise ValueError("action_ids must be non-empty")
    if len(set(action_ids)) != len(action_ids):
        raise ValueError("action_ids must be unique")
    channel_names = tuple(_nonempty_text(item, field="channel_names") for item in payload["channel_names"])
    if not channel_names:
        raise ValueError("channel_names must be non-empty")
    if len(set(channel_names)) != len(channel_names):
        raise ValueError("channel_names must be unique")
    if len(channel_names) != comparable:
        raise ValueError(f"channel_names has {len(channel_names)} entries but comparable_channel_count is {comparable}")
    raw_normalization_binding = payload["normalization_binding"]
    if not isinstance(raw_normalization_binding, Mapping):
        raise ValueError("normalization_binding must be an object")
    normalization_binding = dict(raw_normalization_binding)
    expected_robust_scale, expected_amplitude_confidence = _validate_dual_track_normalization(
        normalization_binding,
        action_ids=action_ids,
        channel_names=channel_names,
    )
    raw_provenance = payload["provenance"]
    if not isinstance(raw_provenance, Mapping):
        raise ValueError("provenance must be an object")
    provenance = dict(raw_provenance)
    trial_qc_review_binding: dict[str, Any] | None = None
    if training_enabled:
        trial_qc_review_binding = _validate_trial_qc_review_binding(
            provenance,
            action_ids=action_ids,
            channel_names=channel_names,
            mapping_sha256=str(mapping_binding["mapping_sha256"]),
            source_path=source_path,
        )
    phase_bin_count = _positive_int(payload["phase_bin_count"], field="phase_bin_count")
    scale_floor = _finite_float(payload["scale_floor"], field="scale_floor")
    if scale_floor <= 0.0:
        raise ValueError("scale_floor must be positive so a bin never becomes a hard label")
    min_trials = _positive_int(payload["min_trials"], field="min_trials")
    default_behavior = _nonempty_text(payload["default_behavior"], field="default_behavior")

    missing_arrays = sorted(set(_ARRAY_NAMES) - set(arrays))
    if missing_arrays:
        raise ValueError(f"EMG reference arrays are missing {missing_arrays}")
    synergy_count = _positive_int(
        synergy_binding["synergy_count"],
        field="synergy_binding.synergy_count",
    )

    action_count = len(action_ids)
    channel_count = len(channel_names)
    anchor_shape = (action_count, phase_bin_count, channel_count)
    synergy_shape = (action_count, phase_bin_count, synergy_count)

    # Human references are never clipped.  Even train-P99-normalized values may
    # exceed one in the upper one percent; that is valid evidence, not a range
    # violation.
    anchor_mean = _validate_nonnegative(arrays["anchor_mean"], field="anchor_mean")
    anchor_scale = np.asarray(arrays["anchor_scale"], dtype=np.float64)
    mvc_anchor_mean = _validate_nonnegative(arrays["mvc_anchor_mean"], field="mvc_anchor_mean")
    mvc_anchor_scale = np.asarray(arrays["mvc_anchor_scale"], dtype=np.float64)
    robust_scale = np.asarray(arrays["robust_scale"], dtype=np.float64)
    amplitude_confidence = _validate_unit_interval(arrays["amplitude_confidence"], field="amplitude_confidence")
    synergy_mean = np.asarray(arrays["synergy_mean"], dtype=np.float64)
    synergy_scale = np.asarray(arrays["synergy_scale"], dtype=np.float64)
    synergy_basis = np.asarray(arrays["synergy_basis"], dtype=np.float64)
    anchor_valid = np.asarray(arrays["anchor_valid"])
    synergy_valid = np.asarray(arrays["synergy_valid"])
    anchor_trial_count = np.asarray(arrays["anchor_trial_count"])

    if anchor_mean.shape != anchor_shape:
        raise ValueError(f"anchor_mean must have shape {anchor_shape}, found {anchor_mean.shape}")
    for name, array in (
        ("anchor_scale", anchor_scale),
        ("mvc_anchor_mean", mvc_anchor_mean),
        ("mvc_anchor_scale", mvc_anchor_scale),
        ("anchor_valid", anchor_valid),
    ):
        if array.shape != anchor_shape:
            raise ValueError(f"{name} must have shape {anchor_shape}, found {array.shape}")
    normalization_shape = (action_count, channel_count)
    for name, array in (
        ("robust_scale", robust_scale),
        ("amplitude_confidence", amplitude_confidence),
    ):
        if array.shape != normalization_shape:
            raise ValueError(f"{name} must have shape {normalization_shape}, found {array.shape}")
    if not np.allclose(robust_scale, expected_robust_scale, rtol=1e-7, atol=1e-9):
        raise ValueError("robust_scale array differs from normalization_binding")
    if not np.allclose(
        amplitude_confidence,
        expected_amplitude_confidence,
        rtol=1e-7,
        atol=1e-9,
    ):
        raise ValueError("amplitude_confidence array differs from normalization_binding")
    if anchor_trial_count.shape != (action_count, phase_bin_count):
        raise ValueError(
            f"anchor_trial_count must have shape {(action_count, phase_bin_count)}, found {anchor_trial_count.shape}"
        )
    for name, array in (
        ("synergy_mean", synergy_mean),
        ("synergy_scale", synergy_scale),
        ("synergy_valid", synergy_valid),
    ):
        if array.shape != synergy_shape:
            raise ValueError(f"{name} must have shape {synergy_shape}, found {array.shape}")
    if synergy_basis.shape != (channel_count, synergy_count):
        raise ValueError(f"synergy_basis must have shape {(channel_count, synergy_count)}, found {synergy_basis.shape}")

    for name, array in (
        ("anchor_scale", anchor_scale),
        ("mvc_anchor_scale", mvc_anchor_scale),
        ("synergy_scale", synergy_scale),
    ):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
        if np.any(array < scale_floor - 1e-9):
            raise ValueError(f"{name} must never fall below the configured scale_floor")
    if not np.all(np.isfinite(synergy_mean)) or np.any(synergy_mean < 0.0):
        raise ValueError("synergy_mean must be finite and non-negative")
    if not np.all(np.isfinite(synergy_basis)) or np.any(synergy_basis < 0.0):
        raise ValueError("synergy_basis must be finite and non-negative")
    if np.any(np.all(synergy_basis <= 0.0, axis=0)):
        raise ValueError("every synergy_basis column must carry at least one positive weight")
    stored_basis_sha256 = hashlib.sha256(np.ascontiguousarray(synergy_basis, dtype=np.float32).tobytes()).hexdigest()
    # This normalization label marks the content-bound v1 builder contract.
    # Historical diagnostic artifacts used a descriptive basis id, so retain
    # read compatibility while requiring newly built production tubes to bind
    # the exact persisted float32 matrix.
    if (
        synergy_binding.get("channel_normalization")
        in {"mvc_normalized_no_additional_scaling", EMG_MODEL_NORMALIZATION}
        and synergy_binding["basis_sha256"] != stored_basis_sha256
    ):
        raise ValueError("synergy_binding.basis_sha256 does not identify the persisted float32 synergy_basis")
    for name, array in (("anchor_valid", anchor_valid), ("synergy_valid", synergy_valid)):
        values = np.asarray(array)
        if values.dtype != np.bool_ and not np.all(np.isin(values, (0, 1))):
            raise ValueError(f"{name} must be boolean or a 0/1 mask")
    trial_counts = np.asarray(anchor_trial_count, dtype=np.int64)
    if np.any(trial_counts < 0):
        raise ValueError("anchor_trial_count must be non-negative")
    if trial_qc_review_binding is not None:
        included_trial_count = sum(
            str(entry.get("decision", "")).strip().lower() == "include"
            for entry in trial_qc_review_binding["trial_decisions"]
        )
        if not np.all(trial_counts == included_trial_count):
            raise ValueError("anchor_trial_count does not match the included trial-QC cohort")
    starved = trial_counts < min_trials
    if np.any(np.asarray(anchor_valid, dtype=bool)[starved]):
        raise ValueError("anchor_valid must be false wherever the trial count is below min_trials")

    identity = {key: payload[key] for key in _IDENTITY_REQUIRED}
    identity["notes"] = str(payload.get("notes", ""))
    identity["review_evidence"] = review_evidence
    array_bundle_sha256 = payload.get("array_bundle_sha256")
    if array_bundle_sha256 is not None:
        array_bundle_sha256 = _require_sha256(
            array_bundle_sha256,
            field="array_bundle_sha256",
        )
        identity["array_bundle_sha256"] = array_bundle_sha256
        if source_path is not None:
            array_path = source_path.with_name(EMG_REFERENCE_ARRAY_FILENAME)
            if not array_path.is_file():
                raise FileNotFoundError(f"EMG reference array bundle is absent: {array_path}")
            actual_array_sha256 = hashlib.sha256(array_path.read_bytes()).hexdigest()
            if actual_array_sha256 != array_bundle_sha256:
                raise ValueError("EMG reference array bundle content hash mismatch")
    expected_fingerprint = emg_reference_fingerprint(identity)
    supplied = payload.get("reference_fingerprint")
    if supplied is not None and str(supplied) != expected_fingerprint:
        raise ValueError(
            "EMG reference fingerprint mismatch: artifact declares "
            f"{str(supplied)!r} but its identity hashes to {expected_fingerprint!r}"
        )

    return EmgPhaseReferenceTube(
        schema_version=schema_version,
        reference_id=_nonempty_text(payload["reference_id"], field="reference_id"),
        review_status=review_status,
        training_enabled=training_enabled,
        default_behavior=default_behavior,
        statistic=statistic,
        scale_floor=scale_floor,
        min_trials=min_trials,
        mapping_binding=mapping_binding,
        synergy_binding=synergy_binding,
        normalization_binding=normalization_binding,
        action_ids=action_ids,
        channel_names=channel_names,
        phase_bin_count=phase_bin_count,
        anchor_mean=anchor_mean,
        anchor_scale=anchor_scale.astype(np.float32),
        mvc_anchor_mean=mvc_anchor_mean,
        mvc_anchor_scale=mvc_anchor_scale.astype(np.float32),
        robust_scale=robust_scale.astype(np.float32),
        amplitude_confidence=amplitude_confidence,
        anchor_valid=np.asarray(anchor_valid, dtype=bool),
        anchor_trial_count=trial_counts.astype(np.int32),
        synergy_mean=synergy_mean.astype(np.float32),
        synergy_scale=synergy_scale.astype(np.float32),
        synergy_valid=np.asarray(synergy_valid, dtype=bool),
        synergy_basis=synergy_basis.astype(np.float32),
        provenance=provenance,
        reference_fingerprint=expected_fingerprint,
        array_bundle_sha256=array_bundle_sha256,
        notes=str(payload.get("notes", "")),
        source_path=source_path,
    )


EMG_REFERENCE_MANIFEST_FILENAME = "emg_reference_manifest.json"
EMG_REFERENCE_ARRAY_FILENAME = "emg_reference_tube.npz"


def load_emg_phase_reference_tube(directory: str | Path) -> EmgPhaseReferenceTube:
    """Load a tube from ``directory`` holding the manifest and the array bundle."""

    root = Path(directory).expanduser()
    if root.is_file():
        root = root.parent
    manifest_path = root / EMG_REFERENCE_MANIFEST_FILENAME
    array_path = root / EMG_REFERENCE_ARRAY_FILENAME
    for path in (manifest_path, array_path):
        if not path.is_file():
            raise FileNotFoundError(f"EMG reference artifact is incomplete: {path} is absent")
    payload = json.loads(
        manifest_path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_pairs,
    )
    with np.load(array_path, allow_pickle=False) as bundle:
        arrays = {name: bundle[name] for name in bundle.files}
    return validate_emg_phase_reference_tube(payload, arrays, source_path=manifest_path)


def save_emg_phase_reference_tube(
    tube: EmgPhaseReferenceTube,
    directory: str | Path,
) -> tuple[Path, Path]:
    """Write ``tube`` to disk and return the manifest and array paths."""

    root = Path(directory).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / EMG_REFERENCE_MANIFEST_FILENAME
    array_path = root / EMG_REFERENCE_ARRAY_FILENAME

    identity = {key: getattr(tube, key) for key in _IDENTITY_REQUIRED if hasattr(tube, key)}
    identity["mapping_binding"] = dict(tube.mapping_binding)
    identity["synergy_binding"] = dict(tube.synergy_binding)
    identity["action_ids"] = list(tube.action_ids)
    identity["channel_names"] = list(tube.channel_names)
    identity["provenance"] = dict(tube.provenance)
    identity["notes"] = tube.notes
    identity["review_evidence"] = list(tube.provenance.get("review_evidence", ()))
    identity["reference_fingerprint"] = emg_reference_fingerprint(identity)

    np.savez_compressed(
        array_path,
        **{name: getattr(tube, name) for name in _ARRAY_NAMES},
    )
    identity["array_bundle_sha256"] = hashlib.sha256(array_path.read_bytes()).hexdigest()
    identity["reference_fingerprint"] = emg_reference_fingerprint(identity)
    manifest_path.write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path, array_path


def resolve_emg_reference_reward_gate(
    tube: EmgPhaseReferenceTube,
    *,
    enabled: bool,
) -> tuple[bool, str]:
    """Decide whether ``tube`` may drive a training signal.

    Returns ``(active, reason)`` and raises when the caller asked for a reward
    the artifact is not cleared to provide.  Mirrors
    :func:`musclemimic.physiology.resolve_fascicle_continuity_reward_gate`.
    """

    if not enabled:
        return False, "reward disabled by configuration"
    if tube.review_status != "verified":
        raise ValueError(
            f"EMG reference {tube.reference_id!r} is {tube.review_status!r}; "
            "anatomical mapping review must complete before it can shape training"
        )
    if not tube.training_enabled:
        raise ValueError(
            f"EMG reference {tube.reference_id!r} sets training_enabled=false; it is available for diagnostics only"
        )
    _validate_trial_qc_review_binding(
        getattr(tube, "provenance", {}),
        action_ids=tuple(tube.action_ids),
        channel_names=tuple(tube.channel_names),
        mapping_sha256=str(tube.mapping_binding.get("mapping_sha256", "")),
        source_path=getattr(tube, "source_path", None),
    )
    mapping_review = str(tube.mapping_binding.get("mapping_review_status", "")).strip().lower()
    if mapping_review != "verified":
        raise ValueError(
            f"the bound 354<-M observation mapping is {mapping_review!r}; a provisional mapping cannot anchor a reward"
        )
    if not bool(np.any(tube.anchor_valid)) and not bool(np.any(tube.synergy_valid)):
        raise ValueError("EMG reference carries no valid phase bin on either leg")
    return True, "verified EMG reference armed for training"


def build_phase_reference_tube(
    *,
    reference_id: str,
    action_envelopes: Mapping[str, Any],
    channel_names: Sequence[str],
    synergy_basis: Any,
    mapping_binding: Mapping[str, Any],
    synergy_binding: Mapping[str, Any],
    normalization_binding: Mapping[str, Any],
    provenance: Mapping[str, Any],
    phase_bin_count: int = 20,
    scale_floor: float = EMG_TUBE_SCALE_FLOOR,
    min_trials: int = EMG_TUBE_MIN_TRIALS,
    review_status: str = "provisional",
    training_enabled: bool = False,
    notes: str = "",
) -> EmgPhaseReferenceTube:
    """Summarise repeated per-trial envelopes into a robust phase-binned tube.

    ``action_envelopes`` maps an action id to an array shaped
    ``[trial, sample, channel]`` of unclipped percent-MVC, non-negative envelopes
    already resampled onto a common movement-normalised time axis.  The audit
    track retains these values exactly.  The model track divides each channel by
    the train-only P99 frozen in ``normalization_binding`` before fitting phase
    centres.  Neither track clips values above one.
    """

    action_ids = tuple(str(key) for key in action_envelopes)
    if not action_ids:
        raise ValueError("action_envelopes must describe at least one action")
    names = tuple(str(name) for name in channel_names)
    basis = np.asarray(synergy_basis, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[0] != len(names):
        raise ValueError(f"synergy_basis must be [channel, synergy] with {len(names)} rows, found {basis.shape}")
    synergy_count = int(basis.shape[1])
    bins = _positive_int(phase_bin_count, field="phase_bin_count")
    robust_scales, amplitude_confidence = _validate_dual_track_normalization(
        normalization_binding,
        action_ids=action_ids,
        channel_names=names,
    )
    normalization_epsilon = float(normalization_binding["epsilon"])

    anchor_mean = np.zeros((len(action_ids), bins, len(names)), dtype=np.float64)
    anchor_scale = np.zeros_like(anchor_mean)
    mvc_anchor_mean = np.zeros_like(anchor_mean)
    mvc_anchor_scale = np.zeros_like(anchor_mean)
    anchor_valid = np.zeros(anchor_mean.shape, dtype=bool)
    trial_counts = np.zeros((len(action_ids), bins), dtype=np.int64)
    synergy_mean = np.zeros((len(action_ids), bins, synergy_count), dtype=np.float64)
    synergy_scale = np.zeros_like(synergy_mean)
    synergy_valid = np.zeros(synergy_mean.shape, dtype=bool)

    projector = synergy_projection_matrix(basis)
    for action_index, action_id in enumerate(action_ids):
        envelopes = np.asarray(action_envelopes[action_id], dtype=np.float64)
        if envelopes.ndim != 3 or envelopes.shape[2] != len(names):
            raise ValueError(
                f"action {action_id!r} envelopes must be [trial, sample, channel] "
                f"with {len(names)} channels, found {envelopes.shape}"
            )
        if not np.all(np.isfinite(envelopes)) or np.any(envelopes < 0.0):
            raise ValueError(f"action {action_id!r} percent-MVC envelopes must be finite and non-negative")
        model_envelopes = envelopes / (robust_scales[action_index][None, None, :] + normalization_epsilon)
        trials, samples, _ = envelopes.shape
        if samples < bins:
            raise ValueError(f"action {action_id!r} has {samples} samples, fewer than {bins} phase bins")
        edges = np.linspace(0, samples, bins + 1).astype(int)
        for bin_index in range(bins):
            mvc_window = envelopes[:, edges[bin_index] : edges[bin_index + 1], :]
            mvc_per_trial = np.mean(mvc_window, axis=1)
            mvc_centre, mvc_scale = _robust_centre_scale(mvc_per_trial, scale_floor=scale_floor)
            window = model_envelopes[:, edges[bin_index] : edges[bin_index + 1], :]
            per_trial = np.mean(window, axis=1)
            centre, scale = _robust_centre_scale(per_trial, scale_floor=scale_floor)
            if not np.all(np.isfinite(centre)):
                raise ValueError(f"action {action_id!r} phase bin {bin_index} has a non-finite robust anchor centre")
            anchor_mean[action_index, bin_index] = centre
            anchor_scale[action_index, bin_index] = scale
            mvc_anchor_mean[action_index, bin_index] = mvc_centre
            mvc_anchor_scale[action_index, bin_index] = mvc_scale
            trial_counts[action_index, bin_index] = trials
            anchor_valid[action_index, bin_index] = trials >= min_trials

            coefficients = np.maximum(per_trial @ projector.T, 0.0)
            syn_centre, syn_scale = _robust_centre_scale(coefficients, scale_floor=scale_floor)
            synergy_mean[action_index, bin_index] = np.maximum(syn_centre, 0.0)
            synergy_scale[action_index, bin_index] = syn_scale
            synergy_valid[action_index, bin_index] = trials >= min_trials

    identity = {
        "schema_version": EMG_REFERENCE_TUBE_SCHEMA_VERSION,
        "reference_id": str(reference_id),
        "review_status": str(review_status),
        "training_enabled": bool(training_enabled),
        "default_behavior": DEFAULT_EMG_REFERENCE_BEHAVIOR,
        "statistic": EMG_TUBE_STATISTIC,
        "scale_floor": float(scale_floor),
        "min_trials": int(min_trials),
        "mapping_binding": dict(mapping_binding),
        "synergy_binding": dict(synergy_binding),
        "normalization_binding": dict(normalization_binding),
        "action_ids": list(action_ids),
        "channel_names": list(names),
        "phase_bin_count": bins,
        "provenance": dict(provenance),
        "notes": str(notes),
        "review_evidence": list(dict(provenance).get("review_evidence", ())),
    }
    arrays = {
        "anchor_mean": anchor_mean,
        "anchor_scale": anchor_scale,
        "mvc_anchor_mean": mvc_anchor_mean,
        "mvc_anchor_scale": mvc_anchor_scale,
        "robust_scale": robust_scales,
        "amplitude_confidence": amplitude_confidence,
        "anchor_valid": anchor_valid,
        "anchor_trial_count": trial_counts,
        "synergy_mean": synergy_mean,
        "synergy_scale": synergy_scale,
        "synergy_valid": synergy_valid,
        "synergy_basis": basis,
    }
    return validate_emg_phase_reference_tube(identity, arrays)


def _robust_centre_scale(
    values: np.ndarray,
    *,
    scale_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Median centre and MAD-derived scale across the leading (trial) axis."""

    centre = np.median(values, axis=0)
    deviation = np.abs(values - centre[None, :])
    scale = EMG_TUBE_MAD_SCALE * np.median(deviation, axis=0) + float(scale_floor)
    return centre, scale


def synergy_projection_matrix(
    basis: Any,
    *,
    ridge: float = EMG_SYNERGY_RIDGE,
) -> np.ndarray:
    """Build ``Q = (W^T W + lambda I)^-1 W^T`` for the fixed human basis ``W``."""

    weights = np.asarray(basis, dtype=np.float64)
    if weights.ndim != 2:
        raise ValueError("synergy basis must be two dimensional")
    lam = _finite_float(ridge, field="ridge")
    if lam <= 0.0:
        raise ValueError("synergy projection ridge must be positive")
    gram = weights.T @ weights + lam * np.eye(weights.shape[1])
    return np.linalg.solve(gram, weights.T)

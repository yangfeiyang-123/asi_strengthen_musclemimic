"""Impact-aligned, explicitly local sEMG validation for muscle activations.

This module is deliberately an evaluation-only boundary.  It does not expose
an EMG training loss: surface EMG, MVC normalization, electrode placement and
model-to-electrode mappings are too uncertain to silently become supervision.

Input NPZ contracts
-------------------
Both inputs carry ``trial_uid``, ``subject_uid`` and ``session_uid`` vectors,
the scalar ``dataset_split='heldout'``, and the same non-empty
``training_session_uid`` inventory.  Trials are paired by UID (never row
position), subject/session provenance must match exactly, and held-out sessions
must be disjoint from the training inventory.  Simulation additionally stores
``muscle_activation [trial,time,muscle]``, ``actuator_names``,
``physical_signal_schema_version``, ``muscle_activation_source``,
``muscle_activation_semantics``, ``muscle_activation_roundoff_policy``,
the name-aligned boolean ``activation_valid_mask``, ``sampling_rate_hz`` and
``impact_frame [trial]``.  It also carries the selected policy decoder type,
checkpoint/promotion fingerprints, and the formal/analysis synergy-basis
fingerprints.  Synergy decoders additionally bind their embedded runtime basis
and its formal source.  EMG stores
``emg [trial,time,channel]``, ``channel_names``, ``sampling_rate_hz`` and
``impact_frame [trial]``; ``mvc_values [channel]`` is required for MVC mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

from musclemimic.badminton.json_contract import load_json_strict
from musclemimic.distill.physical import (
    MUSCLE_ACTIVATION_SEMANTICS,
    MUSCLE_ACTIVATION_SOURCE,
    PHYSICAL_SIGNAL_SCHEMA_VERSION,
    UNIT_INTERVAL_ROUNDOFF_POLICY,
    validate_unit_muscle_activation,
)
from musclemimic.synergy.nmf import fit_nmf

EMG_MAPPING_SCHEMA_VERSION = "emg_local_mapping_v1"
EMG_REPORT_SCHEMA_VERSION = "emg_local_validation_v2"
EMG_POLICY_EVIDENCE_SCHEMA_VERSION = "emg_simulation_policy_evidence_v1"
LOCAL_VALIDATION_SCOPE = "right_upper_limb_local"
CLAIM_LIMITATIONS = (
    "This validates only mapped surface-accessible right-upper-limb channels.",
    "It does not validate full-body or all-354-muscle neural control.",
    "Deep muscles and unmapped model actuators are outside the evidence scope.",
    "Mapping, electrode placement, MVC and trial-normalization uncertainty must be reported.",
)


@dataclass(frozen=True)
class EmgFilterConfig:
    bandpass_low_hz: float = 20.0
    bandpass_high_hz: float = 450.0
    notch_hz: float = 50.0
    notch_quality: float = 30.0
    envelope_lowpass_hz: float = 6.0
    filter_order: int = 4

    def validated(self, sampling_rate_hz: float) -> EmgFilterConfig:
        fs = float(sampling_rate_hz)
        nyquist = 0.5 * fs
        if not np.isfinite(fs) or fs <= 0.0:
            raise ValueError("sampling_rate_hz must be finite and positive")
        if not 0.0 < self.bandpass_low_hz < self.bandpass_high_hz < nyquist:
            raise ValueError("EMG bandpass must satisfy 0 < low < high < Nyquist")
        if not 0.0 < self.notch_hz < nyquist or self.notch_quality <= 0.0:
            raise ValueError("EMG notch frequency/quality are invalid for this sample rate")
        if not 0.0 < self.envelope_lowpass_hz < nyquist:
            raise ValueError("EMG envelope cutoff must lie below Nyquist")
        if int(self.filter_order) <= 0:
            raise ValueError("EMG filter_order must be positive")
        return self

    def to_manifest(self) -> dict[str, Any]:
        return {
            "bandpass_low_hz": float(self.bandpass_low_hz),
            "bandpass_high_hz": float(self.bandpass_high_hz),
            "notch_hz": float(self.notch_hz),
            "notch_quality": float(self.notch_quality),
            "envelope_lowpass_hz": float(self.envelope_lowpass_hz),
            "filter_order": int(self.filter_order),
        }


def preprocess_emg(
    raw_emg: np.ndarray,
    *,
    sampling_rate_hz: float,
    config: EmgFilterConfig | None = None,
) -> np.ndarray:
    """Band-pass, notch, rectify and low-pass without changing trial length."""

    cfg = (config or EmgFilterConfig()).validated(sampling_rate_hz)
    values = _as_trial_time_channel(raw_emg, field_name="emg")
    fs = float(sampling_rate_hz)
    bandpass = butter(
        int(cfg.filter_order),
        [cfg.bandpass_low_hz, cfg.bandpass_high_hz],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    envelope_filter = butter(
        int(cfg.filter_order),
        cfg.envelope_lowpass_hz,
        btype="lowpass",
        fs=fs,
        output="sos",
    )
    notch_b, notch_a = iirnotch(cfg.notch_hz, cfg.notch_quality, fs=fs)
    try:
        filtered = sosfiltfilt(bandpass, values, axis=1)
        filtered = filtfilt(notch_b, notch_a, filtered, axis=1)
        envelope = sosfiltfilt(envelope_filter, np.abs(filtered), axis=1)
    except ValueError as exc:
        raise ValueError("EMG trials are too short for zero-phase filtering or have an invalid time axis") from exc
    envelope = np.maximum(envelope, 0.0)
    if not np.all(np.isfinite(envelope)):
        raise ValueError("EMG preprocessing produced NaN/Inf")
    return envelope.astype(np.float64)


def impact_aligned_resample(
    signals: np.ndarray,
    impact_frames: np.ndarray,
    *,
    sampling_rate_hz: float,
    pre_impact_s: float,
    post_impact_s: float,
    output_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a fixed physical-time window around each measured impact."""

    values = _as_trial_time_channel(signals, field_name="signals")
    impacts = np.asarray(impact_frames)
    if impacts.shape != (values.shape[0],) or not np.issubdtype(impacts.dtype, np.integer):
        raise ValueError("impact_frames must be one integer index per trial")
    fs = float(sampling_rate_hz)
    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError("sampling_rate_hz must be finite and positive")
    if pre_impact_s <= 0.0 or post_impact_s <= 0.0 or int(output_samples) < 3:
        raise ValueError("impact window and output_samples must be positive")
    relative_time = np.linspace(
        -float(pre_impact_s),
        float(post_impact_s),
        int(output_samples),
        dtype=np.float64,
    )
    aligned = np.empty(
        (values.shape[0], int(output_samples), values.shape[2]),
        dtype=np.float64,
    )
    source_offsets = relative_time * fs
    source_grid = np.arange(values.shape[1], dtype=np.float64)
    for trial, impact in enumerate(impacts.astype(np.int64).tolist()):
        sample_positions = float(impact) + source_offsets
        if sample_positions[0] < 0.0 or sample_positions[-1] > values.shape[1] - 1:
            raise ValueError(f"trial {trial} lacks the requested complete pre/post-impact window")
        for channel in range(values.shape[2]):
            aligned[trial, :, channel] = np.interp(
                sample_positions,
                source_grid,
                values[trial, :, channel],
            )
    return aligned, relative_time


def impact_aligned_labels(
    labels: np.ndarray,
    impact_frames: np.ndarray,
    *,
    sampling_rate_hz: float,
    pre_impact_s: float,
    post_impact_s: float,
    output_samples: int,
) -> np.ndarray:
    """Nearest-neighbour resampling for integer event-phase labels."""

    values = np.asarray(labels)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("event labels must be integer [trial,time]")
    impacts = np.asarray(impact_frames)
    if impacts.shape != (values.shape[0],) or not np.issubdtype(impacts.dtype, np.integer):
        raise ValueError("impact_frames must be one integer index per label trial")
    fs = float(sampling_rate_hz)
    relative_time = np.linspace(
        -float(pre_impact_s),
        float(post_impact_s),
        int(output_samples),
        dtype=np.float64,
    )
    positions = np.rint(impacts[:, None] + relative_time[None, :] * fs).astype(np.int64)
    if np.any(positions < 0) or np.any(positions >= values.shape[1]):
        raise ValueError("event labels lack the requested complete impact window")
    return np.take_along_axis(values, positions, axis=1).astype(np.int32)


def validate_emg_mapping(
    payload: Mapping[str, Any],
    *,
    emg_channel_names: Sequence[str] | None = None,
    actuator_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate channel ownership, uncertainty and exact actuator names."""

    if payload.get("schema_version") != EMG_MAPPING_SCHEMA_VERSION:
        raise ValueError(f"EMG mapping schema_version must be {EMG_MAPPING_SCHEMA_VERSION!r}")
    if payload.get("validation_scope") != LOCAL_VALIDATION_SCOPE:
        raise ValueError(f"EMG validation_scope must be {LOCAL_VALIDATION_SCOPE!r}")
    channels = payload.get("channels")
    if not isinstance(channels, list) or len(channels) < 2:
        raise ValueError("EMG mapping requires at least two local surface channels")
    known_emg = None if emg_channel_names is None else _unique_names(emg_channel_names, "emg_channel_names")
    known_actuators = None if actuator_names is None else _unique_names(actuator_names, "actuator_names")
    normalized: list[dict[str, Any]] = []
    mapped_emg: list[str] = []
    for index, entry in enumerate(channels):
        if not isinstance(entry, Mapping):
            raise ValueError(f"EMG mapping channel {index} must be an object")
        emg_channel = str(entry.get("emg_channel", ""))
        muscles = [str(value) for value in entry.get("simulation_actuators", ())]
        uncertainty = str(entry.get("mapping_uncertainty", "")).strip()
        if not emg_channel or not muscles or len(set(muscles)) != len(muscles):
            raise ValueError(f"EMG mapping channel {index} has empty/duplicate names")
        if not uncertainty:
            raise ValueError(f"EMG mapping channel {emg_channel!r} must state mapping_uncertainty")
        if known_emg is not None and emg_channel not in known_emg:
            raise ValueError(f"mapped EMG channel {emg_channel!r} is absent from EMG data")
        missing = [] if known_actuators is None else [name for name in muscles if name not in known_actuators]
        if missing:
            raise ValueError(f"mapped simulation actuators are absent: {missing}")
        weights = np.asarray(entry.get("weights", np.ones(len(muscles))), dtype=np.float64)
        if weights.shape != (len(muscles),) or not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError(f"invalid mapping weights for EMG channel {emg_channel!r}")
        if float(np.sum(weights)) <= 0.0:
            raise ValueError(f"mapping weights for {emg_channel!r} sum to zero")
        weights /= np.sum(weights)
        mapped_emg.append(emg_channel)
        normalized.append(
            {
                "emg_channel": emg_channel,
                "simulation_actuators": muscles,
                "weights": weights.tolist(),
                "mapping_uncertainty": uncertainty,
            }
        )
    if len(set(mapped_emg)) != len(mapped_emg):
        raise ValueError("an EMG channel may appear only once in the mapping")
    pairs = []
    for pair in payload.get("cocontraction_pairs", ()):
        names = [str(value) for value in pair]
        if len(names) != 2 or names[0] == names[1] or any(name not in mapped_emg for name in names):
            raise ValueError(f"invalid co-contraction pair: {pair!r}")
        pairs.append(names)
    return {
        "schema_version": EMG_MAPPING_SCHEMA_VERSION,
        "validation_scope": LOCAL_VALIDATION_SCOPE,
        "subject_or_cohort": str(payload.get("subject_or_cohort", "unspecified")),
        "normalization": str(payload.get("normalization", "per_trial_peak")),
        "channels": normalized,
        "cocontraction_pairs": pairs,
        "notes": str(payload.get("notes", "")),
    }


def map_simulation_activation(
    muscle_activation: np.ndarray,
    *,
    actuator_names: Sequence[str],
    mapping: Mapping[str, Any],
) -> tuple[np.ndarray, list[str]]:
    values = _as_trial_time_channel(
        validate_unit_muscle_activation(muscle_activation),
        field_name="muscle_activation",
    )
    names = _unique_names(actuator_names, "actuator_names")
    if values.shape[2] != len(names):
        raise ValueError("muscle_activation width does not match actuator_names")
    contract = validate_emg_mapping(mapping, actuator_names=names)
    mapped = []
    channel_names = []
    for entry in contract["channels"]:
        indices = [names.index(name) for name in entry["simulation_actuators"]]
        weights = np.asarray(entry["weights"], dtype=np.float64)
        mapped.append(np.sum(values[:, :, indices] * weights[None, None, :], axis=2))
        channel_names.append(entry["emg_channel"])
    return np.stack(mapped, axis=2), channel_names


def validate_simulation_activation_contract(
    simulation: Mapping[str, np.ndarray],
    *,
    actuator_names: Sequence[str],
    mapping: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind EMG inputs to actual unit MuJoCo activation state channels."""

    names = _unique_names(actuator_names, "actuator_names")
    schema = _identity_scalar(
        simulation["physical_signal_schema_version"],
        "physical_signal_schema_version",
    )
    source = _identity_scalar(
        simulation["muscle_activation_source"],
        "muscle_activation_source",
    )
    semantics = _identity_scalar(
        simulation["muscle_activation_semantics"],
        "muscle_activation_semantics",
    )
    roundoff = _identity_scalar(
        simulation["muscle_activation_roundoff_policy"],
        "muscle_activation_roundoff_policy",
    )
    if schema != PHYSICAL_SIGNAL_SCHEMA_VERSION:
        raise ValueError("simulation activation physical signal schema is unsupported")
    if source != MUSCLE_ACTIVATION_SOURCE:
        raise ValueError(
            "simulation muscle_activation must come from transition_state.data.act via model.actuator_actadr"
        )
    if semantics != MUSCLE_ACTIVATION_SEMANTICS:
        raise ValueError("simulation muscle_activation semantics are not unit MuJoCo activation")
    if roundoff != UNIT_INTERVAL_ROUNDOFF_POLICY:
        raise ValueError("simulation muscle_activation roundoff policy is unsupported")
    mask = np.asarray(simulation["activation_valid_mask"])
    if mask.shape != (len(names),) or mask.dtype.kind != "b":
        raise ValueError("simulation activation_valid_mask must be boolean and name-aligned with actuator_names")
    validate_unit_muscle_activation(simulation["muscle_activation"])
    mapped_actuators = {
        str(actuator) for channel in mapping["channels"] for actuator in channel["simulation_actuators"]
    }
    invalid = [name for index, name in enumerate(names) if name in mapped_actuators and not bool(mask[index])]
    if invalid:
        raise ValueError(f"EMG mapping includes actuators without a scalar MuJoCo activation state: {invalid}")
    return {
        "schema_version": PHYSICAL_SIGNAL_SCHEMA_VERSION,
        "source": MUSCLE_ACTIVATION_SOURCE,
        "semantics": MUSCLE_ACTIVATION_SEMANTICS,
        "roundoff_policy": UNIT_INTERVAL_ROUNDOFF_POLICY,
        "activation_valid_mask": mask.astype(bool).tolist(),
        "mapped_activation_valid": 1.0,
    }


def validate_simulation_policy_evidence(
    simulation: Mapping[str, np.ndarray],
    *,
    expected_policy_checkpoint_fingerprint: str,
    expected_policy_promotion_fingerprint: str,
    expected_formal_synergy_basis_fingerprint: str,
) -> dict[str, Any]:
    """Bind simulated activations to one externally selected promoted policy."""

    expected_checkpoint = _require_sha256_text(
        expected_policy_checkpoint_fingerprint,
        "expected_policy_checkpoint_fingerprint",
    )
    expected_promotion = _require_sha256_text(
        expected_policy_promotion_fingerprint,
        "expected_policy_promotion_fingerprint",
    )
    expected_formal = _require_sha256_text(
        expected_formal_synergy_basis_fingerprint,
        "expected_formal_synergy_basis_fingerprint",
    )
    decoder_type = _identity_scalar(simulation["policy_decoder_type"], "policy_decoder_type")
    if decoder_type not in {"direct", "fixed_synergy", "synergy_residual"}:
        raise ValueError("policy_decoder_type must be direct, fixed_synergy, or synergy_residual")
    checkpoint = _sha256_scalar(
        simulation["policy_checkpoint_fingerprint"],
        "policy_checkpoint_fingerprint",
    )
    promotion = _sha256_scalar(
        simulation["policy_promotion_fingerprint"],
        "policy_promotion_fingerprint",
    )
    formal = _sha256_scalar(
        simulation["formal_synergy_basis_fingerprint"],
        "formal_synergy_basis_fingerprint",
    )
    analysis = _sha256_scalar(
        simulation["analysis_synergy_basis_fingerprint"],
        "analysis_synergy_basis_fingerprint",
    )
    if checkpoint != expected_checkpoint:
        raise ValueError("simulation policy checkpoint differs from selected policy evidence")
    if promotion != expected_promotion:
        raise ValueError("simulation policy promotion differs from selected policy evidence")
    if formal != expected_formal or analysis != formal:
        raise ValueError("simulation formal/analysis synergy basis differs from selected formal basis")

    runtime = None
    runtime_source = None
    if decoder_type == "direct":
        forbidden = {
            "runtime_synergy_basis_fingerprint",
            "runtime_synergy_basis_source_fingerprint",
        } & set(simulation)
        if forbidden:
            raise ValueError("direct policy evidence must not claim an embedded runtime synergy basis")
        basis_role = "formal_analysis_basis_only"
    else:
        required = {
            "runtime_synergy_basis_fingerprint",
            "runtime_synergy_basis_source_fingerprint",
        }
        if missing := sorted(required - set(simulation)):
            raise ValueError(f"synergy policy evidence is missing runtime basis fields: {missing}")
        runtime = _sha256_scalar(
            simulation["runtime_synergy_basis_fingerprint"],
            "runtime_synergy_basis_fingerprint",
        )
        runtime_source = _sha256_scalar(
            simulation["runtime_synergy_basis_source_fingerprint"],
            "runtime_synergy_basis_source_fingerprint",
        )
        if runtime_source != formal:
            raise ValueError("runtime synergy basis source fingerprint differs from formal basis")
        basis_role = "formal_runtime_source_and_analysis_basis"
    return {
        "schema_version": EMG_POLICY_EVIDENCE_SCHEMA_VERSION,
        "binding_verified": 1.0,
        "policy_decoder_type": decoder_type,
        "policy_checkpoint_fingerprint": checkpoint,
        "policy_promotion_fingerprint": promotion,
        "formal_synergy_basis_fingerprint": formal,
        "analysis_synergy_basis_fingerprint": analysis,
        "runtime_synergy_basis_fingerprint": runtime,
        "runtime_synergy_basis_source_fingerprint": runtime_source,
        "formal_basis_role": basis_role,
    }


def normalize_envelopes(
    values: np.ndarray,
    *,
    method: str,
    mvc_values: np.ndarray | None = None,
) -> np.ndarray:
    array = _as_trial_time_channel(values, field_name="envelopes")
    if np.min(array) < -1e-10:
        raise ValueError("envelopes must be non-negative")
    if method == "per_trial_peak":
        denominator = np.max(array, axis=1, keepdims=True)
    elif method == "mvc":
        mvc = np.asarray(mvc_values, dtype=np.float64)
        if mvc.shape != (array.shape[2],):
            raise ValueError("mvc normalization requires mvc_values[channel]")
        denominator = mvc[None, None, :]
    else:
        raise ValueError("normalization must be 'per_trial_peak' or 'mvc'")
    if not np.all(np.isfinite(denominator)) or np.any(denominator <= 0.0):
        raise ValueError("normalization denominator must be finite and positive for every channel")
    result = array / denominator
    if not np.all(np.isfinite(result)):
        raise ValueError("envelope normalization produced NaN/Inf")
    return result


def match_synergy_bases(
    first_basis: np.ndarray,
    second_basis: np.ndarray,
    *,
    first_coefficients: np.ndarray | None = None,
    second_coefficients: np.ndarray | None = None,
) -> dict[str, Any]:
    """Hungarian-match [channel,synergy] bases and optional coefficients."""

    first = _finite_nonnegative_matrix(first_basis, "first_basis")
    second = _finite_nonnegative_matrix(second_basis, "second_basis")
    if first.shape != second.shape:
        raise ValueError("synergy bases must have the same [channel,rank] shape")
    first_norm = first / _positive_column_norm(first, "first_basis")
    second_norm = second / _positive_column_norm(second, "second_basis")
    cosine = first_norm.T @ second_norm
    rows, columns = linear_sum_assignment(-cosine)
    order = columns[np.argsort(rows)]
    matched = cosine[np.arange(first.shape[1]), order]
    result: dict[str, Any] = {
        "first_to_second_assignment": order.astype(int).tolist(),
        "weight_cosine_similarity": matched.tolist(),
        "mean_weight_cosine_similarity": float(np.mean(matched)),
    }
    if first_coefficients is not None or second_coefficients is not None:
        if first_coefficients is None or second_coefficients is None:
            raise ValueError("both coefficient matrices are required for coefficient comparison")
        first_c = _finite_nonnegative_matrix(first_coefficients, "first_coefficients")
        second_c = _finite_nonnegative_matrix(second_coefficients, "second_coefficients")
        if first_c.shape != second_c.shape or first_c.shape[1] != first.shape[1]:
            raise ValueError("coefficient matrices must share [sample,rank] shape")
        correlations = [
            _safe_correlation(first_c[:, index], second_c[:, order[index]]) for index in range(first.shape[1])
        ]
        finite = np.asarray([value for value in correlations if np.isfinite(value)])
        result["coefficient_correlation"] = correlations
        result["mean_coefficient_correlation"] = None if finite.size == 0 else float(np.mean(finite))
    return result


def evaluate_aligned_envelopes(
    simulation: np.ndarray,
    emg: np.ndarray,
    *,
    relative_time_s: np.ndarray,
    channel_names: Sequence[str],
    subject_ids: Sequence[str] | None = None,
    onset_fraction: float = 0.2,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    sim = _as_trial_time_channel(simulation, field_name="simulation")
    measured = _as_trial_time_channel(emg, field_name="emg")
    if sim.shape != measured.shape:
        raise ValueError("aligned simulation and EMG envelopes must have identical shape")
    names = _unique_names(channel_names, "channel_names")
    time = np.asarray(relative_time_s, dtype=np.float64)
    if time.shape != (sim.shape[1],) or len(names) != sim.shape[2] or np.any(np.diff(time) <= 0.0):
        raise ValueError("relative_time_s/channel_names do not match aligned data")
    if not 0.0 < float(onset_fraction) < 1.0:
        raise ValueError("onset_fraction must lie in (0,1)")
    subjects = _bootstrap_subject_ids(subject_ids, trial_count=sim.shape[0])
    correlation = np.empty((sim.shape[0], sim.shape[2]), dtype=np.float64)
    dtw = np.empty_like(correlation)
    onset_error = np.empty_like(correlation)
    peak_error = np.empty_like(correlation)
    for trial in range(sim.shape[0]):
        for channel in range(sim.shape[2]):
            first = sim[trial, :, channel]
            second = measured[trial, :, channel]
            correlation[trial, channel] = _safe_correlation(first, second)
            dtw[trial, channel] = _dtw_distance(first, second)
            onset_error[trial, channel] = abs(
                time[_onset_index(first, onset_fraction)] - time[_onset_index(second, onset_fraction)]
            )
            peak_error[trial, channel] = abs(time[int(np.argmax(first))] - time[int(np.argmax(second))])
    metrics = {
        "envelope_correlation": correlation,
        "normalized_dtw": dtw,
        "onset_error_s": onset_error,
        "peak_timing_error_s": peak_error,
    }
    return {
        "per_channel": {
            name: {
                metric_name: _summary_with_clustered_bootstrap(
                    values[:, channel],
                    subject_ids=subjects,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + channel,
                )
                for metric_name, values in metrics.items()
            }
            for channel, name in enumerate(names)
        },
        "aggregate": {
            metric_name: _summary_with_clustered_bootstrap(
                values,
                subject_ids=subjects,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
            for metric_name, values in metrics.items()
        },
        "bootstrap_design": {
            "method": "hierarchical_subject_trial_cluster_bootstrap_v1",
            "subject_count": len(set(subjects)),
            "trial_count": len(subjects),
            "bootstrap_samples": int(bootstrap_samples),
            "channels_within_trial_resampled_together": True,
        },
    }


def cocontraction_index(
    values: np.ndarray,
    *,
    channel_names: Sequence[str],
    pairs: Sequence[Sequence[str]],
    epsilon: float = 1e-12,
) -> dict[str, float]:
    array = _as_trial_time_channel(values, field_name="values")
    names = _unique_names(channel_names, "channel_names")
    result: dict[str, float] = {}
    for pair in pairs:
        pair_names = [str(value) for value in pair]
        if len(pair_names) != 2 or any(name not in names for name in pair_names):
            raise ValueError(f"invalid co-contraction pair: {pair!r}")
        first = array[:, :, names.index(pair_names[0])]
        second = array[:, :, names.index(pair_names[1])]
        ratio = 2.0 * np.minimum(first, second) / (first + second + float(epsilon))
        result[f"{pair_names[0]}__{pair_names[1]}"] = float(np.mean(ratio))
    return result


def phase_envelope_comparison(
    simulation: np.ndarray,
    emg: np.ndarray,
    phase_id: np.ndarray,
    *,
    channel_names: Sequence[str],
) -> dict[str, Any]:
    sim = _as_trial_time_channel(simulation, field_name="simulation")
    measured = _as_trial_time_channel(emg, field_name="emg")
    phase = np.asarray(phase_id)
    names = _unique_names(channel_names, "channel_names")
    if sim.shape != measured.shape or phase.shape != sim.shape[:2] or len(names) != sim.shape[2]:
        raise ValueError("phase envelope inputs have inconsistent shapes")
    if not np.issubdtype(phase.dtype, np.integer):
        raise ValueError("phase_id must be integer event labels")
    result: dict[str, Any] = {}
    for value in sorted(np.unique(phase).astype(int).tolist()):
        mask = phase == value
        sim_mean = np.mean(sim[mask], axis=0)
        emg_mean = np.mean(measured[mask], axis=0)
        result[str(value)] = {
            name: {
                "simulation_mean": float(sim_mean[channel]),
                "emg_mean": float(emg_mean[channel]),
                "absolute_error": float(abs(sim_mean[channel] - emg_mean[channel])),
            }
            for channel, name in enumerate(names)
        }
    return result


def evaluate_emg_validation(
    *,
    simulation_npz: str | Path,
    emg_npz: str | Path,
    mapping_json: str | Path,
    expected_policy_checkpoint_fingerprint: str,
    expected_policy_promotion_fingerprint: str,
    expected_formal_synergy_basis_fingerprint: str,
    pre_impact_s: float = 0.5,
    post_impact_s: float = 0.8,
    output_samples: int = 261,
    synergy_rank: int = 2,
    normalization: str | None = None,
    filter_config: EmgFilterConfig | None = None,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Run the complete local-validation report once user data are available."""

    simulation_path = Path(simulation_npz)
    emg_path = Path(emg_npz)
    mapping_path = Path(mapping_json)
    mapping_raw = load_json_strict(mapping_path)
    with np.load(simulation_path, allow_pickle=False) as source:
        simulation = {name: np.asarray(source[name]) for name in source.files}
    with np.load(emg_path, allow_pickle=False) as source:
        emg_data = {name: np.asarray(source[name]) for name in source.files}
    identity_required = {
        "trial_uid",
        "subject_uid",
        "session_uid",
        "dataset_split",
        "training_session_uid",
    }
    simulation_required = {
        "muscle_activation",
        "actuator_names",
        "physical_signal_schema_version",
        "muscle_activation_source",
        "muscle_activation_semantics",
        "muscle_activation_roundoff_policy",
        "activation_valid_mask",
        "sampling_rate_hz",
        "impact_frame",
        "policy_decoder_type",
        "policy_checkpoint_fingerprint",
        "policy_promotion_fingerprint",
        "formal_synergy_basis_fingerprint",
        "analysis_synergy_basis_fingerprint",
        *identity_required,
    }
    emg_required = {
        "emg",
        "channel_names",
        "sampling_rate_hz",
        "impact_frame",
        *identity_required,
    }
    if missing := sorted(simulation_required - set(simulation)):
        raise ValueError(f"simulation NPZ is missing fields: {missing}")
    if missing := sorted(emg_required - set(emg_data)):
        raise ValueError(f"EMG NPZ is missing fields: {missing}")
    actuator_names = _string_vector(simulation["actuator_names"], "actuator_names")
    emg_names = _string_vector(emg_data["channel_names"], "channel_names")
    mapping = validate_emg_mapping(
        mapping_raw,
        emg_channel_names=emg_names,
        actuator_names=actuator_names,
    )
    activation_contract = validate_simulation_activation_contract(
        simulation,
        actuator_names=actuator_names,
        mapping=mapping,
    )
    policy_evidence = validate_simulation_policy_evidence(
        simulation,
        expected_policy_checkpoint_fingerprint=expected_policy_checkpoint_fingerprint,
        expected_policy_promotion_fingerprint=expected_policy_promotion_fingerprint,
        expected_formal_synergy_basis_fingerprint=expected_formal_synergy_basis_fingerprint,
    )
    emg_order, trial_binding = _bind_paired_trials(simulation, emg_data)
    mapped_sim, channel_names = map_simulation_activation(
        simulation["muscle_activation"],
        actuator_names=actuator_names,
        mapping=mapping,
    )
    emg_indices = [emg_names.index(name) for name in channel_names]
    raw_emg = _as_trial_time_channel(emg_data["emg"], field_name="emg")[emg_order, :, :][:, :, emg_indices]
    if mapped_sim.shape[0] != raw_emg.shape[0]:
        raise ValueError("simulation and EMG paired trial arrays have inconsistent lengths")
    emg_fs = _scalar(emg_data["sampling_rate_hz"], "EMG sampling_rate_hz")
    sim_fs = _scalar(simulation["sampling_rate_hz"], "simulation sampling_rate_hz")
    cfg = filter_config or EmgFilterConfig()
    emg_envelope = preprocess_emg(raw_emg, sampling_rate_hz=emg_fs, config=cfg)
    aligned_emg, relative_time = impact_aligned_resample(
        emg_envelope,
        np.asarray(emg_data["impact_frame"])[emg_order],
        sampling_rate_hz=emg_fs,
        pre_impact_s=pre_impact_s,
        post_impact_s=post_impact_s,
        output_samples=output_samples,
    )
    aligned_sim, sim_time = impact_aligned_resample(
        mapped_sim,
        simulation["impact_frame"],
        sampling_rate_hz=sim_fs,
        pre_impact_s=pre_impact_s,
        post_impact_s=post_impact_s,
        output_samples=output_samples,
    )
    np.testing.assert_allclose(relative_time, sim_time, rtol=0.0, atol=1e-12)
    selected_normalization = normalization or mapping["normalization"]
    mvc_values = emg_data.get("mvc_values")
    if mvc_values is not None:
        mvc_values = np.asarray(mvc_values, dtype=np.float64)[emg_indices]
    aligned_emg = normalize_envelopes(
        aligned_emg,
        method=selected_normalization,
        mvc_values=mvc_values,
    )
    # Simulation activation has no EMG MVC scale; compare normalized timing/envelopes.
    aligned_sim = normalize_envelopes(aligned_sim, method="per_trial_peak")
    envelope_metrics = evaluate_aligned_envelopes(
        aligned_sim,
        aligned_emg,
        relative_time_s=relative_time,
        channel_names=channel_names,
        subject_ids=trial_binding["subject_uids"],
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    sim_matrix = aligned_sim.reshape(-1, aligned_sim.shape[2])
    emg_matrix = aligned_emg.reshape(-1, aligned_emg.shape[2])
    sim_nmf = fit_nmf(sim_matrix, rank=int(synergy_rank), seed=int(seed))
    emg_nmf = fit_nmf(emg_matrix, rank=int(synergy_rank), seed=int(seed) + 1)
    synergy_similarity = match_synergy_bases(
        sim_nmf.basis,
        emg_nmf.basis,
        first_coefficients=sim_nmf.coefficients,
        second_coefficients=emg_nmf.coefficients,
    )
    sim_cocontraction = cocontraction_index(
        aligned_sim,
        channel_names=channel_names,
        pairs=mapping["cocontraction_pairs"],
    )
    emg_cocontraction = cocontraction_index(
        aligned_emg,
        channel_names=channel_names,
        pairs=mapping["cocontraction_pairs"],
    )
    report = {
        "schema_version": EMG_REPORT_SCHEMA_VERSION,
        "claim_scope": LOCAL_VALIDATION_SCOPE,
        "claim_limitations": list(CLAIM_LIMITATIONS),
        "paired_trials": int(aligned_sim.shape[0]),
        "trial_binding": trial_binding,
        "activation_contract": activation_contract,
        "policy_evidence": policy_evidence,
        "mapped_channels": channel_names,
        "impact_alignment": {
            "source": "measured_impact_frame_per_trial",
            "pre_impact_s": float(pre_impact_s),
            "post_impact_s": float(post_impact_s),
            "output_samples": int(output_samples),
        },
        "preprocessing": {
            "emg_filter": cfg.to_manifest(),
            "emg_normalization": selected_normalization,
            "simulation_normalization": "per_trial_peak",
        },
        "mapping": mapping,
        "input_fingerprints": {
            "simulation_npz_sha256": _file_sha256(simulation_path),
            "emg_npz_sha256": _file_sha256(emg_path),
            "mapping_json_sha256": _file_sha256(mapping_path),
        },
        "envelope_metrics": envelope_metrics,
        "synergy": {
            "rank": int(synergy_rank),
            "simulation_basis": sim_nmf.basis.tolist(),
            "emg_basis": emg_nmf.basis.tolist(),
            "similarity": synergy_similarity,
        },
        "co_contraction": {
            "simulation": sim_cocontraction,
            "emg": emg_cocontraction,
            "absolute_error": {key: abs(sim_cocontraction[key] - emg_cocontraction[key]) for key in sim_cocontraction},
        },
    }
    if "phase_id" in simulation:
        aligned_phase = impact_aligned_labels(
            simulation["phase_id"],
            simulation["impact_frame"],
            sampling_rate_hz=sim_fs,
            pre_impact_s=pre_impact_s,
            post_impact_s=post_impact_s,
            output_samples=output_samples,
        )
        report["phase_activation"] = phase_envelope_comparison(
            aligned_sim,
            aligned_emg,
            aligned_phase,
            channel_names=channel_names,
        )
    return report


def _resolve_expected_policy_bindings(
    *,
    policy_evidence_json: str | Path | None,
    expected_policy_checkpoint_fingerprint: str | None,
    expected_policy_promotion_fingerprint: str | None,
    expected_formal_synergy_basis_fingerprint: str | None,
) -> dict[str, str]:
    explicit = {
        "policy_checkpoint_fingerprint": expected_policy_checkpoint_fingerprint,
        "policy_promotion_fingerprint": expected_policy_promotion_fingerprint,
        "formal_synergy_basis_fingerprint": expected_formal_synergy_basis_fingerprint,
    }
    if policy_evidence_json:
        from musclemimic.evaluation.stage3_signal_export import (
            load_paired_policy_evidence,
        )

        evidence = load_paired_policy_evidence(policy_evidence_json)
        sealed = {
            "policy_checkpoint_fingerprint": evidence.policy_checkpoint_fingerprint,
            "policy_promotion_fingerprint": evidence.policy_promotion_fingerprint,
            "formal_synergy_basis_fingerprint": evidence.formal_synergy_basis_fingerprint,
        }
        for key, value in explicit.items():
            if value is not None and str(value) != sealed[key]:
                raise ValueError(f"explicit EMG policy binding differs from paired evidence: {key}")
        return sealed
    missing = [key for key, value in explicit.items() if not value]
    if missing:
        raise SystemExit(
            "missing policy evidence: supply --policy-evidence-json or all explicit expected fingerprints "
            f"({', '.join(missing)})"
        )
    return {key: str(value) for key, value in explicit.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation-npz")
    parser.add_argument("--emg-npz")
    parser.add_argument("--mapping-json")
    parser.add_argument("--output-json")
    parser.add_argument("--expected-policy-checkpoint-fingerprint")
    parser.add_argument("--expected-policy-promotion-fingerprint")
    parser.add_argument("--expected-formal-synergy-basis-fingerprint")
    parser.add_argument(
        "--policy-evidence-json",
        help="sealed Stage-3 paired comparison; derives and verifies all expected policy fingerprints",
    )
    parser.add_argument("--pre-impact-s", type=float, default=0.5)
    parser.add_argument("--post-impact-s", type=float, default=0.8)
    parser.add_argument("--output-samples", type=int, default=261)
    parser.add_argument("--synergy-rank", type=int, default=2)
    parser.add_argument("--normalization", choices=["per_trial_peak", "mvc"], default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bandpass-low-hz", type=float, default=20.0)
    parser.add_argument("--bandpass-high-hz", type=float, default=450.0)
    parser.add_argument("--notch-hz", type=float, default=50.0)
    parser.add_argument("--envelope-lowpass-hz", type=float, default=6.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the data contract without loading data or writing a report.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema_version": EMG_REPORT_SCHEMA_VERSION,
                    "claim_scope": LOCAL_VALIDATION_SCOPE,
                    "claim_limitations": list(CLAIM_LIMITATIONS),
                    "required_simulation_npz_fields": [
                        "muscle_activation",
                        "actuator_names",
                        "physical_signal_schema_version",
                        "muscle_activation_source",
                        "muscle_activation_semantics",
                        "muscle_activation_roundoff_policy",
                        "activation_valid_mask",
                        "sampling_rate_hz",
                        "impact_frame",
                        "policy_decoder_type",
                        "policy_checkpoint_fingerprint",
                        "policy_promotion_fingerprint",
                        "formal_synergy_basis_fingerprint",
                        "analysis_synergy_basis_fingerprint",
                        "trial_uid",
                        "subject_uid",
                        "session_uid",
                        "dataset_split",
                        "training_session_uid",
                    ],
                    "required_emg_npz_fields": [
                        "emg",
                        "channel_names",
                        "sampling_rate_hz",
                        "impact_frame",
                        "trial_uid",
                        "subject_uid",
                        "session_uid",
                        "dataset_split",
                        "training_session_uid",
                    ],
                    "mapping_schema_version": EMG_MAPPING_SCHEMA_VERSION,
                    "required_external_bindings": [
                        "policy_evidence_json or all three explicit expected policy fingerprints",
                    ],
                    "conditional_synergy_policy_fields": [
                        "runtime_synergy_basis_fingerprint",
                        "runtime_synergy_basis_source_fingerprint",
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    expected_policy = _resolve_expected_policy_bindings(
        policy_evidence_json=args.policy_evidence_json,
        expected_policy_checkpoint_fingerprint=args.expected_policy_checkpoint_fingerprint,
        expected_policy_promotion_fingerprint=args.expected_policy_promotion_fingerprint,
        expected_formal_synergy_basis_fingerprint=args.expected_formal_synergy_basis_fingerprint,
    )
    required = {
        "--simulation-npz": args.simulation_npz,
        "--emg-npz": args.emg_npz,
        "--mapping-json": args.mapping_json,
        "--output-json": args.output_json,
    }
    missing = [flag for flag, value in required.items() if not value]
    if missing:
        raise SystemExit(f"missing required arguments outside --dry-run: {', '.join(missing)}")
    report = evaluate_emg_validation(
        simulation_npz=args.simulation_npz,
        emg_npz=args.emg_npz,
        mapping_json=args.mapping_json,
        expected_policy_checkpoint_fingerprint=expected_policy["policy_checkpoint_fingerprint"],
        expected_policy_promotion_fingerprint=expected_policy["policy_promotion_fingerprint"],
        expected_formal_synergy_basis_fingerprint=expected_policy["formal_synergy_basis_fingerprint"],
        pre_impact_s=args.pre_impact_s,
        post_impact_s=args.post_impact_s,
        output_samples=args.output_samples,
        synergy_rank=args.synergy_rank,
        normalization=args.normalization,
        filter_config=EmgFilterConfig(
            bandpass_low_hz=args.bandpass_low_hz,
            bandpass_high_hz=args.bandpass_high_hz,
            notch_hz=args.notch_hz,
            envelope_lowpass_hz=args.envelope_lowpass_hz,
        ),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(output)
    return 0


def _as_trial_time_channel(values: np.ndarray, *, field_name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3 or min(array.shape) <= 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must be finite [trial,time,channel]")
    return array


def _unique_names(values: Sequence[str], field_name: str) -> list[str]:
    names = [str(value) for value in values]
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError(f"{field_name} must be non-empty and unique")
    return names


def _string_vector(values: np.ndarray, field_name: str) -> list[str]:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{field_name} must be a one-dimensional string array")
    return _unique_names(array.astype(str).tolist(), field_name)


def _bind_paired_trials(
    simulation: Mapping[str, np.ndarray],
    emg: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the EMG row order after strict identity/provenance validation."""

    sim_trials = _identity_vector(simulation["trial_uid"], "simulation trial_uid")
    emg_trials = _identity_vector(emg["trial_uid"], "EMG trial_uid")
    if set(sim_trials) != set(emg_trials):
        missing_emg = sorted(set(sim_trials) - set(emg_trials))
        missing_sim = sorted(set(emg_trials) - set(sim_trials))
        raise ValueError(
            "simulation/EMG trial_uid sets differ: "
            f"missing_from_emg={missing_emg}, missing_from_simulation={missing_sim}"
        )
    emg_index = {uid: index for index, uid in enumerate(emg_trials)}
    order = np.asarray([emg_index[uid] for uid in sim_trials], dtype=np.int64)

    sim_subjects = _identity_vector(simulation["subject_uid"], "simulation subject_uid", expected=len(sim_trials))
    sim_sessions = _identity_vector(simulation["session_uid"], "simulation session_uid", expected=len(sim_trials))
    emg_subjects = _identity_vector(emg["subject_uid"], "EMG subject_uid", expected=len(emg_trials))
    emg_sessions = _identity_vector(emg["session_uid"], "EMG session_uid", expected=len(emg_trials))
    reordered_subjects = [emg_subjects[index] for index in order.tolist()]
    reordered_sessions = [emg_sessions[index] for index in order.tolist()]
    if reordered_subjects != sim_subjects or reordered_sessions != sim_sessions:
        raise ValueError("simulation/EMG subject_uid and session_uid must match for every trial_uid")

    sim_split = _identity_scalar(simulation["dataset_split"], "simulation dataset_split")
    emg_split = _identity_scalar(emg["dataset_split"], "EMG dataset_split")
    heldout_labels = {"heldout", "validation", "test"}
    if sim_split != emg_split or sim_split not in heldout_labels:
        raise ValueError("simulation and EMG dataset_split must match and be heldout/validation/test")
    sim_training_sessions = set(_identity_vector(simulation["training_session_uid"], "simulation training_session_uid"))
    emg_training_sessions = set(_identity_vector(emg["training_session_uid"], "EMG training_session_uid"))
    if sim_training_sessions != emg_training_sessions:
        raise ValueError("simulation/EMG training_session_uid inventories differ")
    heldout_sessions = set(sim_sessions)
    leakage = sorted(heldout_sessions & sim_training_sessions)
    if leakage:
        raise ValueError(f"EMG held-out session leakage with policy training sessions: {leakage}")

    binding_payload = {
        "schema_version": "emg_paired_trial_binding_v1",
        "binding_verified": 1.0,
        "pairing_key": "trial_uid",
        "dataset_split": sim_split,
        "trial_uids": sim_trials,
        "subject_uids": sim_subjects,
        "session_uids": sim_sessions,
        "training_session_uids": sorted(sim_training_sessions),
        "emg_reordered_to_simulation": bool(not np.array_equal(order, np.arange(len(order), dtype=np.int64))),
    }
    binding_payload["binding_fingerprint"] = hashlib.sha256(
        json.dumps(
            binding_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return order, binding_payload


def _identity_vector(
    values: np.ndarray,
    field_name: str,
    *,
    expected: int | None = None,
) -> list[str]:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in {"U", "S", "i", "u"}:
        raise ValueError(f"{field_name} must be a one-dimensional string/integer UID array")
    identifiers = [str(value).strip() for value in array.tolist()]
    if not identifiers or any(not value for value in identifiers):
        raise ValueError(f"{field_name} must contain non-empty stable UIDs")
    if expected is not None and len(identifiers) != int(expected):
        raise ValueError(f"{field_name} length does not match trial_uid")
    if "trial_uid" in field_name and len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{field_name} must be unique")
    return identifiers


def _identity_scalar(values: np.ndarray, field_name: str) -> str:
    array = np.asarray(values)
    if array.size != 1 or array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{field_name} must be one string scalar")
    value = str(array.reshape(-1)[0]).strip().lower()
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _scalar(value: np.ndarray, field_name: str) -> float:
    array = np.asarray(value, dtype=np.float64)
    if array.size != 1 or not np.isfinite(array.reshape(-1)[0]):
        raise ValueError(f"{field_name} must be one finite scalar")
    return float(array.reshape(-1)[0])


def _finite_nonnegative_matrix(values: np.ndarray, field_name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) <= 0 or not np.all(np.isfinite(array)) or np.min(array) < -1e-10:
        raise ValueError(f"{field_name} must be a finite non-negative matrix")
    return np.maximum(array, 0.0)


def _positive_column_norm(values: np.ndarray, field_name: str) -> np.ndarray:
    norm = np.linalg.norm(values, axis=0, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError(f"{field_name} contains an empty synergy")
    return norm


def _safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("correlation inputs must be same-shape vectors")
    if np.std(first) <= 1e-12 or np.std(second) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def _dtw_distance(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    previous = np.full(second.size + 1, np.inf, dtype=np.float64)
    previous[0] = 0.0
    for first_value in first:
        current = np.full(second.size + 1, np.inf, dtype=np.float64)
        for index, second_value in enumerate(second, start=1):
            current[index] = abs(first_value - second_value) + min(
                current[index - 1],
                previous[index],
                previous[index - 1],
            )
        previous = current
    return float(previous[-1] / max(first.size, second.size))


def _onset_index(values: np.ndarray, fraction: float) -> int:
    threshold = float(fraction) * float(np.max(values))
    indices = np.flatnonzero(values >= threshold)
    return int(indices[0]) if indices.size else int(len(values) - 1)


def _bootstrap_subject_ids(
    subject_ids: Sequence[str] | None,
    *,
    trial_count: int,
) -> list[str]:
    if subject_ids is None:
        return [f"trial-cluster-{index}" for index in range(int(trial_count))]
    subjects = [str(value).strip() for value in subject_ids]
    if len(subjects) != int(trial_count) or any(not value for value in subjects):
        raise ValueError("subject_ids must provide one non-empty identity per trial")
    return subjects


def _summary_with_clustered_bootstrap(
    values: np.ndarray,
    *,
    subject_ids: Sequence[str],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or array.shape[0] != len(subject_ids):
        raise ValueError("cluster bootstrap values must have one leading row per subject-bound trial")
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "mean": None,
            "std": None,
            "ci95": [None, None],
            "n": 0,
            "n_trials": 0,
            "n_subjects": 0,
            "bootstrap_method": "hierarchical_subject_trial_cluster_bootstrap_v1",
        }
    if int(bootstrap_samples) <= 0:
        raise ValueError("bootstrap_samples must be positive")
    subjects = [str(value) for value in subject_ids]
    unique_subjects = list(dict.fromkeys(subjects))
    by_subject = {subject: np.flatnonzero(np.asarray(subjects, dtype=object) == subject) for subject in unique_subjects}
    rng = np.random.default_rng(int(seed))
    means: list[float] = []
    for _ in range(int(bootstrap_samples)):
        sampled_subjects = rng.integers(0, len(unique_subjects), size=len(unique_subjects))
        sampled_trials: list[np.ndarray] = []
        for subject_index in sampled_subjects.tolist():
            candidates = by_subject[unique_subjects[subject_index]]
            sampled_trials.append(candidates[rng.integers(0, len(candidates), size=len(candidates))])
        replicate = array[np.concatenate(sampled_trials)]
        replicate_finite = replicate[np.isfinite(replicate)]
        if replicate_finite.size:
            means.append(float(np.mean(replicate_finite)))
    trial_means = []
    for row in array:
        row_finite = row[np.isfinite(row)] if np.asarray(row).ndim else np.asarray([row])
        if row_finite.size:
            trial_means.append(float(np.mean(row_finite)))
    if not means:
        ci95: list[float | None] = [None, None]
    else:
        ci95 = np.quantile(np.asarray(means), [0.025, 0.975]).tolist()
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(trial_means, ddof=1)) if len(trial_means) > 1 else 0.0,
        "ci95": ci95,
        "n": int(finite.size),
        "n_trials": len(trial_means),
        "n_subjects": len(unique_subjects),
        "bootstrap_method": "hierarchical_subject_trial_cluster_bootstrap_v1",
    }


def _sha256_scalar(value: np.ndarray, field_name: str) -> str:
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{field_name} must be one lowercase SHA-256 string scalar")
    return _require_sha256_text(str(array.reshape(-1)[0]).strip(), field_name)


def _require_sha256_text(value: Any, field_name: str) -> str:
    digest = str(value or "").strip()
    if (
        len(digest) != 64
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{field_name} must be a lowercase 64-hex SHA-256")
    return digest


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Build the phase-indexed EMG reference tube for the PEASD pipeline.

Inputs (read-only):

* per-trial ``mvc_normalized_emg.npz`` from
  ``jidian_measurement/data/P002/S20260721_A/trials/<action>/trial_XXX/``
  (16 channels, 2000 Hz, MVC-normalised);
* the reviewed channel mapping JSON (16 acquired / 15 comparable).

Output: a validated ``EmgPhaseReferenceTube`` and its exact bound evidence,
written as ``emg_reference_manifest.json`` + ``emg_reference_tube.npz`` +
``emg_observation_mapping.json`` under ``output_dir``.  A formal tube also
bundles the exact ``emg_trial_qc_review.json`` bytes.

Phase axis
----------
``forehand_high_clear`` uses the time-normalised ``build_synergy_dataset``
product (101 samples, software-cue cropped, exploratory).  Other registered
actions use duration-normalised raw trials.  Every action therefore lands on
the same uniform ``[0, 1]`` phase axis.

Review state
------------
The tube defaults to ``provisional`` (``training_enabled=false``).  ``--verified``
is an explicit, fail-closed opt-in that succeeds only after the mapping review
and action-specific trial/channel review complete with traceable evidence.
The downstream gate is additionally enforced by
``resolve_emg_reference_reward_gate``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from musclemimic.badminton.action_registry import emg_trial_action_choices, resolve
from musclemimic.physiology.emg_anchor import load_json_mapping
from musclemimic.physiology.emg_reference import (
    EMG_MODEL_NORMALIZATION,
    EMG_REFERENCE_TUBE_SCHEMA_VERSION,
    EMG_SYNERGY_PROJECTION_METHOD,
    EMG_SYNERGY_RIDGE,
    EMG_TUBE_MIN_TRIALS,
    build_emg_dual_track_normalization,
    build_phase_reference_tube,
    save_emg_phase_reference_tube,
)

DEFAULT_SESSION = "P002/S20260721_A"
DEFAULT_MAPPING = (
    "configs/physiology/emg_badminton_synergy_16_v2_myofullbody_observation_v1.json"
)
# Historical v1 location is retained as a named constant but never overwritten
# by the v2 dual-track builder.
LEGACY_FOREHAND_OUTPUT = "artifacts/forehand_clear_peasd_v1/data/emg_reference"
DEFAULT_FOREHAND_OUTPUT = "artifacts/forehand_clear_peasd_v1/data/emg_reference_v2"
EMG_OBSERVATION_MAPPING_FILENAME = "emg_observation_mapping.json"
EMG_TRIAL_QC_REVIEW_FILENAME = "emg_trial_qc_review.json"
EMG_TRIAL_QC_REVIEW_SCHEMA_VERSION = "emg_trial_channel_qc_review_v1"
_REQUIRED_QC_RISKS = frozenset({"s9_progressive_near_flatline"})


def _trial_dir(session_root: Path, action: str, trial_index: int) -> Path:
    trial = session_root / "trials" / action / f"trial_{trial_index:03d}"
    if not (trial / "mvc_normalized_emg.npz").is_file():
        raise FileNotFoundError(f"trial {trial} has no mvc_normalized_emg.npz")
    return trial


def _normalized_trials(session_root: Path, action: str) -> list[tuple[Path, np.ndarray]]:
    """Load per-trial 16-channel MVC-normalised envelopes as [sample, channel]."""

    out: list[tuple[Path, np.ndarray]] = []
    for trial_dir in sorted((session_root / "trials" / action).glob("trial_*")):
        npz = trial_dir / "mvc_normalized_emg.npz"
        if not npz.is_file():
            continue
        arr = np.asarray(np.load(npz, allow_pickle=False)["normalized_envelope"], dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 16:
            raise ValueError(f"{npz}: expected [sample, 16] envelope, found {arr.shape}")
        out.append((trial_dir, arr))
    return out


def _phase_binned(trials: list[np.ndarray], *, bins: int, rng) -> np.ndarray:
    """Duration-normalise each trial to ``bins`` phase bins."""

    if not trials:
        raise ValueError("no trials to phase-bin")
    channel_count = int(trials[0].shape[1])
    if any(signal.ndim != 2 or signal.shape[1] != channel_count for signal in trials):
        raise ValueError("all phase-binning trials must share one channel width")
    result = np.empty((len(trials), bins, channel_count), dtype=np.float64)
    for index, signal in enumerate(trials):
        samples = signal.shape[0]
        if samples < bins:
            raise ValueError(
                f"trial {index} has {samples} samples, fewer than {bins} phase bins"
            )
        edges = np.linspace(0, samples, bins + 1).astype(int)
        for bin_index in range(bins):
            result[index, bin_index] = np.mean(
                signal[edges[bin_index] : edges[bin_index + 1]], axis=0
            )
    return result


def _software_cue_time_normalized_trial(
    trial_dir: Path,
    signal: np.ndarray,
    *,
    samples: int = 101,
) -> np.ndarray:
    """Reproduce the exploratory clear crop without inventing impact timing."""

    events_path = trial_dir / "events.csv"
    if not events_path.is_file():
        raise ValueError(f"software-cue crop requires events.csv: {trial_dir}")
    with events_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    def exact_event(name: str) -> dict[str, str]:
        matches = [row for row in rows if str(row.get("event_name", "")).strip() == name]
        if len(matches) != 1:
            raise ValueError(
                f"software-cue crop requires exactly one {name!r}, found "
                f"{len(matches)}: {trial_dir}"
            )
        return matches[0]

    cue = exact_event("movement_cue")
    stop_event = exact_event("recording_stop")
    if not str(cue.get("source", "")).strip().lower().startswith("software"):
        raise ValueError(
            f"movement_cue must be an explicitly labelled software cue: {trial_dir}"
        )
    try:
        start = int(cue["sample_index"])
        stop = int(stop_event["sample_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"software-cue crop has invalid sample indices: {trial_dir}") from exc
    if not 0 <= start < stop <= signal.shape[0]:
        raise ValueError(
            f"software-cue crop bounds start={start} stop={stop} are outside "
            f"{signal.shape[0]} samples: {trial_dir}"
        )
    if int(samples) < 2:
        raise ValueError("time-normalized trial must contain at least two samples")
    from scipy.signal import resample

    return np.maximum(resample(signal[start:stop], int(samples), axis=0), 0.0)


def _phase_input_trials(
    action: str,
    trials: list[tuple[Path, np.ndarray]],
) -> list[np.ndarray]:
    if action == "forehand_high_clear":
        return [
            _software_cue_time_normalized_trial(path, signal, samples=101)
            for path, signal in trials
        ]
    return [signal for _path, signal in trials]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256_text(value: object, *, field: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return text


def _require_nonempty_text(value: object, *, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _require_evidence_list(value: object, *, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list of traceable strings")
    return [
        _require_nonempty_text(item, field=f"{field}[{index}]")
        for index, item in enumerate(value)
    ]


def _load_verified_trial_qc_review(
    review_path: Path,
    *,
    action: str,
    mapping_sha256: str,
    trials: list[tuple[Path, np.ndarray]],
    channel_names: list[str],
) -> tuple[list[tuple[Path, np.ndarray]], dict]:
    """Validate and bind the human trial/channel QC used by a formal tube.

    The preprocessing flags are deliberately not auto-promoted here.  A
    versioned review must account for every source file, every comparable
    channel and the known S9 decay.  Super-MVC values are retained and graded by
    the dual-track normalization contract rather than treated as a rejection
    risk.  Trial exclusion
    is supported and changes the actual fitted cohort.  Channel exclusion is
    rejected because it would change the observation ABI and therefore needs
    a new mapping/profile version rather than an implicit dimension drop.
    """

    path = review_path.expanduser().resolve(strict=True)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"trial-QC review is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("trial-QC review must be a JSON object")
    if payload.get("schema_version") != EMG_TRIAL_QC_REVIEW_SCHEMA_VERSION:
        raise ValueError(
            "trial-QC review schema_version must be "
            f"{EMG_TRIAL_QC_REVIEW_SCHEMA_VERSION!r}"
        )
    if str(payload.get("action", "")).strip() != action:
        raise ValueError(
            f"trial-QC review action {payload.get('action')!r} does not match {action!r}"
        )
    if str(payload.get("review_status", "")).strip().lower() != "verified":
        raise ValueError("trial-QC review must set review_status=verified")
    if payload.get("training_enabled") is not True:
        raise ValueError("trial-QC review must set training_enabled=true")
    reviewer_id = _require_nonempty_text(
        payload.get("reviewer_id"), field="trial-QC reviewer_id"
    )
    reviewed_at = _require_nonempty_text(
        payload.get("reviewed_at"), field="trial-QC reviewed_at"
    )
    review_evidence = _require_evidence_list(
        payload.get("review_evidence"), field="trial-QC review_evidence"
    )
    declared_mapping_sha = _require_sha256_text(
        payload.get("mapping_sha256"), field="trial-QC mapping_sha256"
    )
    if declared_mapping_sha != mapping_sha256:
        raise ValueError(
            "trial-QC review mapping_sha256 does not match the exact mapping bytes"
        )

    trial_decisions = payload.get("trial_decisions")
    if not isinstance(trial_decisions, list) or not trial_decisions:
        raise ValueError("trial-QC review requires non-empty trial_decisions")
    decisions_by_id: dict[str, dict] = {}
    for index, entry in enumerate(trial_decisions):
        if not isinstance(entry, dict):
            raise ValueError(f"trial_decisions[{index}] must be an object")
        trial_id = _require_nonempty_text(
            entry.get("trial_id"), field=f"trial_decisions[{index}].trial_id"
        )
        if trial_id in decisions_by_id:
            raise ValueError(f"duplicate trial-QC decision for {trial_id!r}")
        decision = str(entry.get("decision", "")).strip().lower()
        if decision not in {"include", "exclude"}:
            raise ValueError(
                f"trial_decisions[{index}].decision must be include or exclude"
            )
        _require_nonempty_text(
            entry.get("reason"), field=f"trial_decisions[{index}].reason"
        )
        decisions_by_id[trial_id] = entry

    actual_by_id = {path.name: (path, values) for path, values in trials}
    if set(decisions_by_id) != set(actual_by_id):
        missing = sorted(set(actual_by_id) - set(decisions_by_id))
        extra = sorted(set(decisions_by_id) - set(actual_by_id))
        raise ValueError(
            "trial-QC review must account for the exact discovered trial set; "
            f"missing={missing}, extra={extra}"
        )

    included: list[tuple[Path, np.ndarray]] = []
    bound_trials: list[dict] = []
    for trial_id in sorted(actual_by_id):
        trial_path, values = actual_by_id[trial_id]
        entry = decisions_by_id[trial_id]
        normalized_path = trial_path / "mvc_normalized_emg.npz"
        preprocessing_qc_path = trial_path / "preprocessing_qc.json"
        if not preprocessing_qc_path.is_file():
            raise ValueError(
                f"formal trial-QC requires preprocessing_qc.json for {trial_id}"
            )
        actual_normalized_sha = _sha256(normalized_path)
        actual_qc_sha = _sha256(preprocessing_qc_path)
        declared_normalized_sha = _require_sha256_text(
            entry.get("mvc_normalized_emg_sha256"),
            field=f"trial_decisions[{trial_id}].mvc_normalized_emg_sha256",
        )
        declared_qc_sha = _require_sha256_text(
            entry.get("preprocessing_qc_sha256"),
            field=f"trial_decisions[{trial_id}].preprocessing_qc_sha256",
        )
        if declared_normalized_sha != actual_normalized_sha:
            raise ValueError(
                f"trial-QC normalized EMG hash mismatch for {trial_id}"
            )
        if declared_qc_sha != actual_qc_sha:
            raise ValueError(
                f"trial-QC preprocessing QC hash mismatch for {trial_id}"
            )
        decision = str(entry["decision"]).strip().lower()
        if decision == "include":
            included.append((trial_path, values))
        bound_trials.append(
            {
                "trial_id": trial_id,
                "decision": decision,
                "reason": str(entry["reason"]).strip(),
                "mvc_normalized_emg_sha256": actual_normalized_sha,
                "preprocessing_qc_sha256": actual_qc_sha,
            }
        )
    if not included:
        raise ValueError("trial-QC review excludes every trial")

    channel_decisions = payload.get("channel_decisions")
    if not isinstance(channel_decisions, list) or not channel_decisions:
        raise ValueError("trial-QC review requires non-empty channel_decisions")
    channel_by_name: dict[str, dict] = {}
    for index, entry in enumerate(channel_decisions):
        if not isinstance(entry, dict):
            raise ValueError(f"channel_decisions[{index}] must be an object")
        name = _require_nonempty_text(
            entry.get("emg_channel"),
            field=f"channel_decisions[{index}].emg_channel",
        )
        if name in channel_by_name:
            raise ValueError(f"duplicate trial-QC channel decision for {name!r}")
        decision = str(entry.get("decision", "")).strip().lower()
        if decision not in {"include_after_review", "exclude"}:
            raise ValueError(
                "channel decision must be include_after_review or exclude"
            )
        _require_nonempty_text(
            entry.get("reason"), field=f"channel_decisions[{index}].reason"
        )
        if decision == "exclude":
            raise ValueError(
                f"trial-QC excludes channel {name!r}, which changes the EMG observation ABI; "
                "create and review a new mapping/profile version instead"
            )
        channel_by_name[name] = entry
    if set(channel_by_name) != set(channel_names):
        missing = sorted(set(channel_names) - set(channel_by_name))
        extra = sorted(set(channel_by_name) - set(channel_names))
        raise ValueError(
            "trial-QC review must account for the exact comparable channel set; "
            f"missing={missing}, extra={extra}"
        )

    risk_decisions = payload.get("risk_decisions")
    if not isinstance(risk_decisions, list):
        raise ValueError("trial-QC review requires risk_decisions")
    risk_by_id: dict[str, dict] = {}
    for index, entry in enumerate(risk_decisions):
        if not isinstance(entry, dict):
            raise ValueError(f"risk_decisions[{index}] must be an object")
        risk_id = _require_nonempty_text(
            entry.get("risk_id"), field=f"risk_decisions[{index}].risk_id"
        )
        if risk_id in risk_by_id:
            raise ValueError(f"duplicate risk decision for {risk_id!r}")
        decision = str(entry.get("decision", "")).strip().lower()
        if decision not in {"accepted_after_review", "mitigated"}:
            raise ValueError(
                f"risk {risk_id!r} is unresolved; expected accepted_after_review or mitigated"
            )
        _require_nonempty_text(
            entry.get("reason"), field=f"risk_decisions[{index}].reason"
        )
        _require_evidence_list(
            entry.get("evidence"), field=f"risk_decisions[{index}].evidence"
        )
        risk_by_id[risk_id] = entry
    missing_risks = sorted(_REQUIRED_QC_RISKS - set(risk_by_id))
    if missing_risks:
        raise ValueError(
            f"trial-QC review does not resolve known risks: {missing_risks}"
        )

    review_sha256 = _sha256(path)
    binding = {
        "schema_version": EMG_TRIAL_QC_REVIEW_SCHEMA_VERSION,
        "action": action,
        "review_status": "verified",
        "training_enabled": True,
        "source_path": str(path),
        "review_sha256": review_sha256,
        "mapping_sha256": mapping_sha256,
        "reviewer_id": reviewer_id,
        "reviewed_at": reviewed_at,
        "review_evidence": review_evidence,
        "trial_decisions": bound_trials,
        "channel_decisions": [
            {
                "emg_channel": name,
                "decision": "include_after_review",
                "reason": str(channel_by_name[name]["reason"]).strip(),
            }
            for name in channel_names
        ],
        "risk_decisions": [risk_by_id[risk] for risk in sorted(risk_by_id)],
    }
    return included, binding


def _tube_id(action: str, session: str) -> str:
    return f"P002_{action}_16ch_v2"


def _mapping_binding(mapping: dict, *, mapping_sha256: str) -> dict:
    return {
        "mapping_id": mapping["mapping_id"],
        # Bind the exact JSON bytes shipped next to the tube.  The acquisition
        # profile hash is a different identity and must not stand in for this.
        "mapping_sha256": str(mapping_sha256),
        "mapping_review_status": str(mapping.get("review_status", "provisional")),
        "acquired_channel_count": int(mapping["profile_binding"]["acquired_channel_count"]),
        "comparable_channel_count": int(mapping["profile_binding"]["comparable_channel_count"]),
        "actuator_schema_hash": mapping["model_binding"]["actuator_schema_hash"],
    }


def _synergy_binding(basis: np.ndarray, mapping: dict) -> dict:
    # The persisted tube stores float32.  Hash those exact bytes rather than
    # the transient float64 fitter output, otherwise provenance and payload
    # describe different bases.
    basis_sha = hashlib.sha256(np.ascontiguousarray(basis, dtype=np.float32).tobytes()).hexdigest()
    return {
        "basis_id": f"{mapping['mapping_id']}_nmf",
        "basis_sha256": basis_sha,
        "synergy_count": int(basis.shape[1]),
        "channel_normalization": EMG_MODEL_NORMALIZATION,
        "projection_method": EMG_SYNERGY_PROJECTION_METHOD,
        "projection_ridge": EMG_SYNERGY_RIDGE,
    }


def _provenance(
    session_root: Path,
    action: str,
    trials: list[Path],
    mapping: dict,
    *,
    trial_qc_review: dict | None = None,
) -> dict:
    clear = action == "forehand_high_clear"
    provenance = {
        "subject": "P002",
        "session": "S20260721_A",
        "action": action,
        "normalization": "dual_track_percent_mvc_audit_and_train_p99_model",
        "source": "jidian_measurement/data/P002/S20260721_A",
        "phase_axis": (
            "software_cue_exploratory_time_normalized_101"
            if clear
            else "full_trial_duration_normalized"
        ),
        "trial_paths": [str(path) for path in trials],
        "trial_files": [
            {
                "path": str(path / "mvc_normalized_emg.npz"),
                "sha256": _sha256(path / "mvc_normalized_emg.npz"),
            }
            for path in trials
        ],
        "trial_selection": (
            "verified_action_specific_trial_qc_review"
            if trial_qc_review is not None
            else "all_available_mvc_normalized_npz_no_qc_filter"
        ),
        "mapping_review_status": str(mapping.get("review_status", "provisional")),
        "review_evidence": [
            *list(mapping.get("review_evidence", ())),
            *(
                list(trial_qc_review.get("review_evidence", ()))
                if trial_qc_review is not None
                else []
            ),
        ],
        "crop": (
            "movement_cue_to_recording_stop_software_cue_exploratory"
            if clear
            else "full_trial_duration_normalized"
        ),
        "evidence_limitations": [
            "software_cue_or_duration_normalized_not_impact_aligned",
            *(
                []
                if trial_qc_review is not None
                else ["trial_level_manual_qc_not_applied_by_builder"]
            ),
        ],
    }
    if trial_qc_review is not None:
        provenance["trial_qc_review"] = dict(trial_qc_review)
    return provenance


def _load_mvc_reference(
    session_root: Path,
    *,
    kept_sensor_indices: list[int],
    channel_names: list[str],
) -> tuple[np.ndarray, dict]:
    """Load the exact MVC denominator source used by the processed trials."""

    path = (session_root / "preprocessing_mvc_reference.json").resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    channels = payload.get("channels")
    if not isinstance(channels, list) or len(channels) != 16:
        raise ValueError("MVC reference manifest must contain exactly 16 channels")
    values: list[float] = []
    for output_index, sensor_index in enumerate(kept_sensor_indices):
        entry = channels[sensor_index]
        sensor_id = int(entry.get("sensor_id", -1))
        expected_sensor = sensor_index + 1
        if sensor_id != expected_sensor:
            raise ValueError("MVC reference channel order differs from sensor order")
        expected_prefix = f"S{expected_sensor} "
        if not channel_names[output_index].startswith(expected_prefix):
            raise ValueError("MVC reference sensor order differs from mapping channel order")
        peak = float(entry.get("selected_peak_mV", float("nan")))
        if not np.isfinite(peak) or peak <= 0.0:
            raise ValueError(f"MVC reference sensor {sensor_id} must be finite and positive")
        values.append(peak)
    return np.asarray(values, dtype=np.float64), {
        "path": str(path),
        "sha256": _sha256(path),
        "scope": str(payload.get("scope", "")),
        "algorithm": str(payload.get("algorithm", "")),
    }


def _require_verified_mapping(mapping: dict) -> list[str]:
    """Return review evidence only when the whole mapping is training-ready.

    ``review_status=verified`` is not sufficient by itself.  The explicit CLI
    opt-in may arm training only when every comparable channel has left the
    provisional state, the mapping enables training, and the review leaves a
    non-empty evidence trail.
    """

    if str(mapping.get("review_status", "")).strip().lower() != "verified":
        raise ValueError("--verified requires mapping review_status=verified")
    if mapping.get("training_enabled") is not True:
        raise ValueError("--verified requires mapping training_enabled=true")

    evidence = mapping.get("review_evidence")
    if not isinstance(evidence, list) or not evidence or any(
        not isinstance(item, str) or not item.strip() for item in evidence
    ):
        raise ValueError(
            "--verified requires non-empty, traceable mapping review_evidence strings"
        )

    channels = mapping.get("channels")
    if not isinstance(channels, list) or not channels:
        raise ValueError("--verified requires a non-empty reviewed channel mapping")
    provisional: list[str] = []
    invalid: list[str] = []
    for index, entry in enumerate(channels):
        if not isinstance(entry, dict):
            invalid.append(f"channel[{index}]")
            continue
        status = str(entry.get("mapping_status", "")).strip().lower()
        name = str(entry.get("emg_channel") or f"channel[{index}]")
        if status == "excluded_no_verified_model_homolog":
            if not str(entry.get("exclusion_reason", "")).strip():
                invalid.append(name)
            continue
        confidence = str(entry.get("mapping_confidence", "")).strip().lower()
        if status != "mapped" or confidence not in {"high", "medium", "low"}:
            provisional.append(name)
    if invalid:
        raise ValueError(
            "--verified requires traceable exclusion evidence for every excluded channel; "
            f"invalid={invalid}"
        )
    if provisional:
        raise ValueError(
            "--verified requires every comparable channel mapping to be reviewed "
            f"(no provisional confidence); unresolved={provisional}"
        )
    return [item.strip() for item in evidence]


def _default_output_root(action: str) -> Path:
    spec = resolve(action)
    if spec.slug == "forehand_clear":
        return Path(DEFAULT_FOREHAND_OUTPUT)
    return Path(f"artifacts/{spec.slug}_peasd_v1/data/emg_reference_v2")


def _bundle_mapping(mapping_path: Path, output_dir: Path, *, expected_sha256: str) -> Path:
    """Copy the exact mapping bytes beside the tube and verify the copy."""

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / EMG_OBSERVATION_MAPPING_FILENAME
    if mapping_path.resolve() != destination.resolve():
        shutil.copyfile(mapping_path, destination)
    actual_sha256 = _sha256(destination)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "bundled EMG mapping hash mismatch: "
            f"expected {expected_sha256}, copied {actual_sha256}"
        )
    return destination


def _bundle_trial_qc_review(
    review_path: Path,
    output_dir: Path,
    *,
    expected_sha256: str,
) -> Path:
    """Copy the exact human review bytes next to a verified tube."""

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / EMG_TRIAL_QC_REVIEW_FILENAME
    if review_path.resolve() != destination.resolve():
        shutil.copyfile(review_path, destination)
    actual_sha256 = _sha256(destination)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "bundled EMG trial-QC review hash mismatch: "
            f"expected {expected_sha256}, copied {actual_sha256}"
        )
    return destination


def _fit_basis(envelopes: np.ndarray, *, rank: int, seeds: tuple[int, ...]) -> np.ndarray:
    """Fit a non-negative basis ``W`` [16, rank] on the phase-binned data."""

    from musclemimic.synergy.nmf import fit_best_initialization

    flat = envelopes.reshape(-1, envelopes.shape[-1])
    flat = np.clip(flat, 0.0, None)
    result, _ = fit_best_initialization(
        flat,
        rank=rank,
        seeds=seeds,
        max_iter=1000,
        tol=1e-6,
    )
    return result.basis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-root", type=Path, default=Path("jidian_measurement/data") / DEFAULT_SESSION)
    parser.add_argument("--mapping", type=Path, default=Path(DEFAULT_MAPPING))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            f"default: {DEFAULT_FOREHAND_OUTPUT} for forehand clear, otherwise "
            "artifacts/<slug>_peasd_v1/data/emg_reference_v2; legacy v1 tubes "
            "are never overwritten"
        ),
    )
    parser.add_argument("--action", choices=emg_trial_action_choices(), required=True)
    parser.add_argument("--phase-bins", type=int, default=20)
    parser.add_argument("--synergy-rank", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--min-trials", type=int, default=EMG_TUBE_MIN_TRIALS)
    parser.add_argument(
        "--verified",
        action="store_true",
        help=(
            "arm the emitted tube for training; fails closed unless the mapping, "
            "all comparable channels, training flag, and review evidence are verified"
        ),
    )
    parser.add_argument(
        "--trial-qc-review",
        type=Path,
        default=None,
        help=(
            "required with --verified: action-specific human trial/channel QC "
            f"using schema {EMG_TRIAL_QC_REVIEW_SCHEMA_VERSION}"
        ),
    )
    args = parser.parse_args()

    session_root = args.session_root.resolve()
    mapping_path = args.mapping.resolve()
    mapping = load_json_mapping(mapping_path)
    mapping_sha256 = _sha256(mapping_path)
    if args.verified:
        _require_verified_mapping(mapping)
        if args.trial_qc_review is None:
            raise ValueError("--verified requires --trial-qc-review")
    elif args.trial_qc_review is not None:
        raise ValueError(
            "--trial-qc-review is formal evidence and may only be used with --verified"
        )

    trials = _normalized_trials(session_root, args.action)

    # The tube's channel order must match the mapping's comparable order, which
    # is the order the anchor spec compiles against.  Both are 15 channels.
    channel_names = [
        entry["emg_channel"]
        for entry in mapping["channels"]
        if entry.get("mapping_status") != "excluded_no_verified_model_homolog"
    ]
    if len(channel_names) != 15:
        raise ValueError(
            f"mapping declares {len(channel_names)} comparable channels, expected 15"
        )

    trial_qc_review: dict | None = None
    if args.verified:
        trials, trial_qc_review = _load_verified_trial_qc_review(
            args.trial_qc_review,
            action=args.action,
            mapping_sha256=mapping_sha256,
            trials=trials,
            channel_names=channel_names,
        )
    if len(trials) < args.min_trials:
        raise ValueError(
            f"action {args.action!r} has {len(trials)} included trials, "
            f"below min_trials={args.min_trials}"
        )
    # Select the 15 comparable channels from the 16-channel recordings.  The
    # excluded sensor is fixed as sensor 1 (no verified model homolog); the
    # mapping lists channels in sensor order, so excluding it by index is exact.
    excluded = int(mapping["profile_binding"].get("excluded_sensor_ids", [1])[0])
    kept = [index for index in range(16) if index != excluded - 1]
    if len(kept) != 15:
        raise ValueError(f"expected 15 kept channels after excluding sensor {excluded}, got {len(kept)}")
    rng = np.random.default_rng(0)
    phase_inputs = _phase_input_trials(args.action, trials)
    comparable_inputs = [values[:, kept] for values in phase_inputs]
    comparable = _phase_binned(
        comparable_inputs,
        bins=args.phase_bins,
        rng=rng,
    )

    mvc_values_mv, mvc_reference_binding = _load_mvc_reference(
        session_root,
        kept_sensor_indices=kept,
        channel_names=channel_names,
    )
    training_cohort = [
        {
            "trial_id": path.name,
            "mvc_normalized_emg_sha256": _sha256(
                path / "mvc_normalized_emg.npz"
            ),
        }
        for path, _values in trials
    ]
    normalization_binding = build_emg_dual_track_normalization(
        action_samples={args.action: comparable_inputs},
        channel_names=channel_names,
        training_cohorts={args.action: training_cohort},
        mvc_final_reference_mv=mvc_values_mv,
        mvc_original_mv=mvc_values_mv,
        mvc_reference_binding=mvc_reference_binding,
    )
    robust_scale = np.asarray(
        [
            stats["robust_scale_percent_mvc"]
            for stats in normalization_binding["actions"][0]["channels"]
        ],
        dtype=np.float64,
    )
    robust_comparable = comparable / (
        robust_scale[None, None, :] + float(normalization_binding["epsilon"])
    )

    # The basis must live in the same 15-channel space as the tube.
    basis = _fit_basis(
        robust_comparable,
        rank=args.synergy_rank,
        seeds=tuple(args.seeds),
    )
    if basis.shape != (15, args.synergy_rank):
        raise ValueError(f"basis shape {basis.shape}, expected (15, {args.synergy_rank})")

    tube = build_phase_reference_tube(
        reference_id=_tube_id(args.action, DEFAULT_SESSION),
        action_envelopes={args.action: comparable},
        channel_names=channel_names,
        synergy_basis=basis,
        mapping_binding=_mapping_binding(mapping, mapping_sha256=mapping_sha256),
        synergy_binding=_synergy_binding(basis, mapping),
        normalization_binding=normalization_binding,
        provenance=_provenance(
            session_root,
            args.action,
            [path for path, _ in trials],
            mapping,
            trial_qc_review=trial_qc_review,
        ),
        phase_bin_count=args.phase_bins,
        min_trials=args.min_trials,
        review_status="verified" if args.verified else "provisional",
        training_enabled=bool(args.verified),
        notes=(
            f"{args.action} EMG tube built with unclipped percent-MVC audit and "
            f"train-P99 model normalization ({len(trials)} trials, 16ch->15ch). "
            + (
                "Verified mapping review explicitly armed this tube for training."
                if args.verified
                else "Provisional: not cleared for training until the mapping review "
                "in doc 03 §8 completes."
            )
        ),
    )

    base_output = args.output_dir
    if base_output is None:
        base_output = _default_output_root(args.action)
    output_dir = base_output / args.action
    manifest_path, array_path = save_emg_phase_reference_tube(tube, output_dir)
    bundled_mapping_path = _bundle_mapping(
        mapping_path,
        output_dir,
        expected_sha256=mapping_sha256,
    )
    bundled_trial_qc_path = None
    if trial_qc_review is not None:
        bundled_trial_qc_path = _bundle_trial_qc_review(
            args.trial_qc_review.resolve(strict=True),
            output_dir,
            expected_sha256=str(trial_qc_review["review_sha256"]),
        )
    print(f"tube_reference_id: {tube.reference_id}")
    print(f"tube_action: {args.action}")
    print(f"tube_trials: {len(trials)}")
    print(f"tube_phase_bins: {tube.phase_bin_count}")
    print(f"tube_channels: {tube.channel_count}")
    print(f"tube_synergy_rank: {tube.synergy_count}")
    quality_counts: dict[str, int] = {}
    for stats in normalization_binding["actions"][0]["channels"]:
        quality = str(stats["mvc_quality"])
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
    print("tube_normalization: percent_mvc_unclipped + train_p99_per_channel")
    print(
        "tube_mvc_quality_counts: "
        + json.dumps(quality_counts, sort_keys=True)
    )
    print(
        "tube_max_task_p99_over_mvc: "
        f"{max(float(stats['task_p99_over_mvc']) for stats in normalization_binding['actions'][0]['channels']):.6g}"
    )
    print(f"tube_review_status: {tube.review_status} (training_enabled={tube.training_enabled})")
    print(f"tube_manifest: {manifest_path}")
    print(f"tube_arrays: {array_path}")
    print(f"tube_mapping: {bundled_mapping_path}")
    if bundled_trial_qc_path is not None:
        print(f"tube_trial_qc_review: {bundled_trial_qc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

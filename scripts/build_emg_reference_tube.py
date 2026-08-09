#!/usr/bin/env python
"""Build the phase-indexed EMG reference tube for the PEASD pipeline.

Inputs (read-only):

* per-trial ``mvc_normalized_emg.npz`` from
  ``jidian_measurement/data/P002/S20260721_A/trials/<action>/trial_XXX/``
  (16 channels, 2000 Hz, MVC-normalised);
* the reviewed channel mapping JSON (16 acquired / 15 comparable).

Output: a validated ``EmgPhaseReferenceTube`` written as
``emg_reference_manifest.json`` + ``emg_reference_tube.npz`` under
``output_dir``.

Phase axis
----------
``forehand_high_clear`` uses the time-normalised ``build_synergy_dataset``
product (101 samples, software-cue cropped, exploratory).  ``china_jump_high_clear``
has no such product, so its raw trials are duration-normalised to a matching
axis.  Both actions therefore land on the same uniform ``[0, 1]`` phase axis.

Review state
------------
The tube is written ``provisional`` (``training_enabled=false``) until the
mapping review in doc 03 §8 completes.  Nothing downstream can consume it for
training before then; that gate is enforced by
``resolve_emg_reference_reward_gate``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from musclemimic.badminton.action_registry import emg_trial_action_choices, resolve
from musclemimic.physiology.emg_anchor import load_json_mapping
from musclemimic.physiology.emg_reference import (
    EMG_REFERENCE_TUBE_SCHEMA_VERSION,
    EMG_SYNERGY_PROJECTION_METHOD,
    EMG_SYNERGY_RIDGE,
    EMG_TUBE_MIN_TRIALS,
    build_phase_reference_tube,
    save_emg_phase_reference_tube,
)

DEFAULT_SESSION = "P002/S20260721_A"
DEFAULT_MAPPING = (
    "configs/physiology/emg_badminton_synergy_16_v2_myofullbody_observation_v1.json"
)
DEFAULT_OUTPUT = "artifacts/peasd_v1/data/emg_reference"
# Kept so pre-generalization forehand-clear tubes stay discoverable in place.
LEGACY_FOREHAND_OUTPUT = "artifacts/forehand_clear_peasd_v1/data/emg_reference"


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
    """Duration-normalise each trial to ``bins`` phase bins: [trial, bin, 16]."""

    if not trials:
        raise ValueError("no trials to phase-bin")
    result = np.empty((len(trials), bins, 16), dtype=np.float64)
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tube_id(action: str, session: str) -> str:
    return f"P002_{action}_16ch_v1"


def _mapping_binding(mapping: dict) -> dict:
    return {
        "mapping_id": mapping["mapping_id"],
        "mapping_sha256": mapping["profile_binding"]["profile_sha256"],
        "mapping_review_status": str(mapping.get("review_status", "provisional")),
        "acquired_channel_count": int(mapping["profile_binding"]["acquired_channel_count"]),
        "comparable_channel_count": int(mapping["profile_binding"]["comparable_channel_count"]),
        "actuator_schema_hash": mapping["model_binding"]["actuator_schema_hash"],
    }


def _synergy_binding(basis: np.ndarray, mapping: dict) -> dict:
    basis_sha = hashlib.sha256(np.ascontiguousarray(basis, dtype=np.float64)).hexdigest()
    return {
        "basis_id": f"{mapping['mapping_id']}_nmf",
        "basis_sha256": basis_sha,
        "synergy_count": int(basis.shape[1]),
        "channel_normalization": "unit_variance_per_channel",
        "projection_method": EMG_SYNERGY_PROJECTION_METHOD,
        "projection_ridge": EMG_SYNERGY_RIDGE,
    }


def _provenance(session_root: Path, action: str, trials: list[Path], mapping: dict) -> dict:
    return {
        "subject": "P002",
        "session": "S20260721_A",
        "action": action,
        "normalization": "mvc",
        "source": "jidian_measurement/data/P002/S20260721_A",
        "phase_axis": "duration_normalized_exploratory",
        "trial_paths": [str(path) for path in trials],
        "mapping_review_status": str(mapping.get("review_status", "provisional")),
        "review_evidence": [],
        "crop": "software_cue_exploratory",
    }


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
        help=f"default: {LEGACY_FOREHAND_OUTPUT} for forehand clear, else {DEFAULT_OUTPUT}",
    )
    parser.add_argument("--action", choices=emg_trial_action_choices(), required=True)
    parser.add_argument("--phase-bins", type=int, default=20)
    parser.add_argument("--synergy-rank", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--min-trials", type=int, default=EMG_TUBE_MIN_TRIALS)
    args = parser.parse_args()

    session_root = args.session_root.resolve()
    mapping_path = args.mapping.resolve()
    mapping = load_json_mapping(mapping_path)

    trials = _normalized_trials(session_root, args.action)
    if len(trials) < args.min_trials:
        raise ValueError(
            f"action {args.action!r} has {len(trials)} trials, below min_trials={args.min_trials}"
        )
    rng = np.random.default_rng(0)
    phase_binned = _phase_binned([signal for _, signal in trials], bins=args.phase_bins, rng=rng)

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

    # Select the 15 comparable channels from the 16-channel recordings.  The
    # excluded sensor is fixed as sensor 1 (no verified model homolog); the
    # mapping lists channels in sensor order, so excluding it by index is exact.
    excluded = int(mapping["profile_binding"].get("excluded_sensor_ids", [1])[0])
    kept = [index for index in range(16) if index != excluded - 1]
    if len(kept) != 15:
        raise ValueError(f"expected 15 kept channels after excluding sensor {excluded}, got {len(kept)}")
    comparable = phase_binned[:, :, kept]

    # The basis must live in the same 15-channel space as the tube.
    basis = _fit_basis(comparable, rank=args.synergy_rank, seeds=tuple(args.seeds))
    if basis.shape != (15, args.synergy_rank):
        raise ValueError(f"basis shape {basis.shape}, expected (15, {args.synergy_rank})")

    tube = build_phase_reference_tube(
        reference_id=_tube_id(args.action, DEFAULT_SESSION),
        action_envelopes={args.action: comparable},
        channel_names=channel_names,
        synergy_basis=basis,
        mapping_binding=_mapping_binding(mapping),
        synergy_binding=_synergy_binding(basis, mapping),
        provenance=_provenance(session_root, args.action, [path for path, _ in trials], mapping),
        phase_bin_count=args.phase_bins,
        min_trials=args.min_trials,
        review_status="provisional",
        training_enabled=False,
        notes=(
            f"{args.action} EMG tube built from MVC-normalised envelopes "
            f"({len(trials)} trials, 16ch->15ch). Provisional: not cleared for "
            "training until the mapping review in doc 03 §8 completes."
        ),
    )

    base_output = args.output_dir
    if base_output is None:
        spec = resolve(args.action)
        base_output = Path(
            LEGACY_FOREHAND_OUTPUT if spec.slug == "forehand_clear" else DEFAULT_OUTPUT
        )
    output_dir = base_output / args.action
    manifest_path, array_path = save_emg_phase_reference_tube(tube, output_dir)
    print(f"tube_reference_id: {tube.reference_id}")
    print(f"tube_action: {args.action}")
    print(f"tube_trials: {len(trials)}")
    print(f"tube_phase_bins: {tube.phase_bin_count}")
    print(f"tube_channels: {tube.channel_count}")
    print(f"tube_synergy_rank: {tube.synergy_count}")
    print(f"tube_review_status: {tube.review_status} (training_enabled={tube.training_enabled})")
    print(f"tube_manifest: {manifest_path}")
    print(f"tube_arrays: {array_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

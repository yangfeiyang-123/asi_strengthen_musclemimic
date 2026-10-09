from __future__ import annotations

import hashlib
from pathlib import Path

import emg.primitive_library as primitive_library
import numpy as np
import pytest
from emg.cli import main as cli_main
from emg.primitive_library import (
    QC_REVIEW_SCHEMA_VERSION,
    PrimitiveLibraryConfig,
    action_balance_weights,
    build_primitive_synergy_library,
    verify_primitive_synergy_library,
)
from emg.profiles import BADMINTON_SYNERGY_16_V2
from emg.storage import atomic_save_npz, atomic_write_json, dataset_sha256, read_json


def _activation(samples: int, *, shift: float, first_gain: float, second_gain: float) -> np.ndarray:
    phase = np.linspace(0.0, 1.0, samples)
    first = first_gain * np.exp(-0.5 * ((phase - (0.28 + shift)) / 0.10) ** 2)
    second = second_gain * np.exp(-0.5 * ((phase - (0.70 - shift)) / 0.12) ** 2)
    return np.stack([first, second])


def _write_dataset(
    root: Path,
    action_counts: dict[str, int],
    *,
    exploratory: bool,
    scope: str = "primitive",
    samples: int = 41,
) -> Path:
    profile = BADMINTON_SYNERGY_16_V2
    true_basis = np.zeros((16, 2), dtype=np.float64)
    true_basis[:8, 0] = np.linspace(0.4, 1.0, 8)
    true_basis[8:, 1] = np.linspace(1.0, 0.4, 8)
    segments: list[np.ndarray] = []
    trials: list[dict[str, object]] = []
    boundaries = [0]
    for action_index, (action, count) in enumerate(action_counts.items()):
        for trial_index in range(count):
            coefficients = _activation(
                samples,
                shift=(trial_index - count / 2) * 0.003,
                first_gain=1.0 + 0.25 * action_index,
                second_gain=1.2 - 0.15 * action_index,
            )
            # Small strictly-positive variation prevents a degenerate exact-zero
            # objective while retaining the same two directions in every trial.
            rng = np.random.default_rng(action_index * 100 + trial_index)
            segment = np.maximum(
                true_basis @ coefficients + 1e-4 * rng.random((16, samples)),
                0.0,
            )
            segments.append(segment)
            boundaries.append(boundaries[-1] + samples)
            trial_id = f"{action}_trial_{trial_index + 1:03d}"
            trials.append(
                {
                    "participant_id": "PTEST",
                    "session_id": "S1",
                    "trial_id": trial_id,
                    "action_id": action,
                    "category": "primitive" if action != "china_jump_high_clear" else "complete",
                    "valid_for_analysis": True,
                    "preprocessing_analysis_ready": True,
                    "normalization_reference_key": "participant:PTEST",
                    "source": f"PTEST/S1/trials/{action}/{trial_id}/processed_emg.npz",
                    "crop_method": (
                        "software_cue_to_recording_stop_exploratory_resampled_41"
                        if exploratory
                        else "annotated_movement_start_to_recording_stop_resampled_41"
                    ),
                    "crop_is_exploratory": exploratory,
                    "movement_start_annotation": None if exploratory else {"source": "video_frame"},
                    "source_start_sample": 10,
                    "source_stop_sample": 90,
                    "output_samples": samples,
                }
            )
    values = np.concatenate(segments, axis=1).astype(np.float32)
    metadata: dict[str, object] = {
        "dataset_format_version": 1,
        "profile_id": profile.profile_id,
        "profile_version": profile.version,
        "channel_profile_snapshot": profile.to_dict(),
        "protocol_id": "badminton_primitive_protocol_v1",
        "processing": {
            "normalization_method": "mvc",
            "normalization_reference_scope": "participant_raw_mvc_across_sessions",
        },
        "normalization_references": {"participant:PTEST": {"normalization_method": "mvc"}},
        "only_valid": True,
        "only_valid_includes_preprocessing_qc": True,
        "scope": scope,
        "crop_mode": "software_cue_exploratory" if exploratory else "annotated_movement_events",
        "exploratory": exploratory,
        "event_semantics": (
            "software_cue_is_not_physical_movement_start" if exploratory else "evidence_backed_manual_movement_start"
        ),
        "time_normalize_samples": samples,
        "participants": ["PTEST"],
        "sessions": ["PTEST/S1"],
        "actions": sorted(action_counts),
        "trials": trials,
        "matrix_shape": list(values.shape),
        "trial_boundaries": boundaries,
        "fs_hz": 2000.0,
    }
    metadata["dataset_sha256"] = dataset_sha256(values, metadata)
    path = root / "primitive_dataset.npz"
    atomic_save_npz(
        path,
        V=values,
        channel_ids=np.asarray(profile.channel_ids, dtype=np.int16),
        muscle_slugs=np.asarray([channel.muscle_slug for channel in profile.channels]),
        sides=np.asarray([channel.side for channel in profile.channels]),
        trial_boundaries=np.asarray(boundaries, dtype=np.int64),
        fs_hz=np.asarray(2000.0),
    )
    atomic_write_json(path.with_suffix(".json"), metadata)
    return path


def _fast_config(**overrides: object) -> PrimitiveLibraryConfig:
    values: dict[str, object] = {
        "k_min": 1,
        "k_max": 2,
        "n_init": 2,
        "seed": 17,
        "split_half_repeats": 3,
        "bootstrap_repeats": 3,
        "initialization_restarts": 2,
        "stability_cosine_threshold": 0.75,
        "split_half_fraction_required": 0.50,
        "bootstrap_median_threshold": 0.75,
        "initialization_minimum_threshold": 0.75,
        "minimum_effective_rank_fraction": 0.70,
        "minimum_fit_global_vaf": 0.85,
        "minimum_fit_local_vaf": 0.70,
        "minimum_fit_local_fraction": 0.70,
        "minimum_heldout_global_vaf": 0.75,
        "minimum_heldout_action_fraction": 0.75,
        "minimum_trials_per_action": 4,
    }
    values.update(overrides)
    return PrimitiveLibraryConfig(**values)  # type: ignore[arg-type]


def _clean_source_provenance() -> dict[str, object]:
    return {
        "repo_root": "/test/repo",
        "git_available": True,
        "git_commit_hash": "a" * 40,
        "dirty": False,
        "git_status_entries": [],
        "source_files": {"primitive_library.py": {"sha256": "b" * 64, "num_bytes": 1}},
        "source_bundle_sha256": "c" * 64,
        "formal_reproducible": True,
    }


def _write_qc_review(dataset: Path, root: Path, config: PrimitiveLibraryConfig) -> Path:
    arrays, _metadata, integrity = primitive_library._load_dataset(dataset)
    boundaries = np.asarray(arrays["trial_boundaries"], dtype=np.int64)
    metadata = read_json(dataset.with_suffix(".json"))
    _actions, columns, _indices = primitive_library._action_columns(metadata["trials"], boundaries)
    values = np.asarray(arrays["V"], dtype=np.float64)
    scale = primitive_library.action_balanced_channel_scale(
        values,
        columns,
        config.channel_normalization,
    )
    diagnostics = primitive_library.channel_scale_diagnostics(values, scale)
    diagnostics_sha256 = primitive_library._channel_diagnostics_sha256(diagnostics)
    evidence = root / "qc_review_evidence.txt"
    evidence.write_text("independent channel plots and reviewer disposition\n", encoding="utf-8")
    review = root / "qc_review.json"
    atomic_write_json(
        review,
        {
            "schema_version": QC_REVIEW_SCHEMA_VERSION,
            "review_id": "review_v1",
            "reviewer_id": "reviewer_02",
            "reviewed_time": "2026-08-06T12:00:00+00:00",
            "status": "approved",
            "source_content_sha256": integrity["content_sha256"],
            "channel_diagnostics_sha256": diagnostics_sha256,
            "evidence": {
                "path": evidence.name,
                "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
            },
        },
    )
    return review


def test_action_balance_gives_every_action_equal_total_squared_weight() -> None:
    weights, multipliers = action_balance_weights(
        40,
        {"short": np.arange(0, 10), "long": np.arange(10, 40)},
    )
    assert multipliers["short"] > multipliers["long"]
    assert np.sum(weights[:10] ** 2) == pytest.approx(20.0)
    assert np.sum(weights[10:] ** 2) == pytest.approx(20.0)


def test_cli_builds_formal_measurement_library_with_joint_rank_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _write_dataset(
        tmp_path,
        {"split_step": 4, "trunk_rotation": 4},
        exploratory=False,
    )
    config = _fast_config()
    review = _write_qc_review(dataset, tmp_path, config)
    monkeypatch.setattr(
        primitive_library,
        "_source_code_provenance",
        _clean_source_provenance,
    )
    output = tmp_path / "primitive_v1"
    result = cli_main(
        [
            "build-primitive-library",
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--required-action",
            "split_step",
            "--required-action",
            "trunk_rotation",
            "--k-min",
            "1",
            "--k-max",
            "2",
            "--n-init",
            "2",
            "--split-half-repeats",
            "3",
            "--bootstrap-repeats",
            "3",
            "--initialization-restarts",
            "2",
            "--stability-cosine",
            "0.75",
            "--bootstrap-median",
            "0.75",
            "--initialization-minimum",
            "0.75",
            "--minimum-effective-rank-fraction",
            "0.70",
            "--minimum-fit-global-vaf",
            "0.85",
            "--minimum-fit-local-vaf",
            "0.70",
            "--minimum-fit-local-fraction",
            "0.70",
            "--minimum-heldout-global-vaf",
            "0.75",
            "--minimum-heldout-action-fraction",
            "0.75",
            "--qc-review-manifest",
            str(review),
        ]
    )
    assert result == 0
    manifest = read_json(output / "primitive_synergy_library_manifest.json")
    assert manifest["basis"]["selected_k"] == 2
    assert manifest["basis"]["selection_method"] == "smallest_k_passing_all_adequacy_and_stability_gates"
    assert manifest["basis"]["selected_k_stability_pass"] is True
    assert manifest["basis"]["selected_k_adequacy_pass"] is True
    assert manifest["basis"]["selected_k_selection_pass"] is True
    assert manifest["release"]["formal_ready"] is True
    assert manifest["observation_space_contract"]["observation_space_only"] is True
    assert manifest["observation_space_contract"]["training_enabled"] is False
    assert manifest["observation_space_contract"]["reverse_mapping_15_or_16_to_354_allowed"] is False
    squared_masses = [entry["total_squared_fit_weight"] for entry in manifest["action_balancing"]["actions"].values()]
    assert squared_masses[0] == pytest.approx(squared_masses[1])
    assert len(manifest["basis"]["basis_sha256"]) == 64
    assert len(manifest["source_dataset"]["content_sha256"]) == 64
    assert manifest["source_dataset"]["all_semantic_arrays_and_metadata_hash_verified"] is True
    assert len(manifest["basis"]["ordered_channel_schema_sha256"]) == 64
    assert len(manifest["manifest_sha256"]) == 64
    assert verify_primitive_synergy_library(output)["manifest_sha256"] == manifest["manifest_sha256"]
    with np.load(output / "primitive_synergy_library.npz", allow_pickle=False) as artifact:
        assert artifact["W"].shape == (16, 2)
        assert artifact["W_recorded_units"].shape == (16, 2)
        assert artifact["H_projected"].shape == (2, 8 * 41)
        assert str(artifact["basis_sha256"]) == manifest["basis"]["basis_sha256"]
        assert str(artifact["source_content_sha256"]) == manifest["source_dataset"]["content_sha256"]
        assert "action_000_H_mean" in artifact.files
        assert "action_001_H_std" in artifact.files

    scan = read_json(output / "primitive_synergy_k_scan.json")
    selected = next(row for row in scan["by_k"] if row["k"] == 2)
    assert selected["split_half"]["cosine_threshold"] == pytest.approx(0.75)
    for run in selected["split_half"]["runs"]:
        for group in ("a", "b"):
            action_entries = run["groups"][group]["actions"]
            assert set(action_entries) == {"split_step", "trunk_rotation"}
            masses = [entry["total_squared_fit_weight"] for entry in action_entries.values()]
            assert masses[0] == pytest.approx(masses[1])
    for run in selected["bootstrap"]["runs"]:
        assert all(entry["draw_count"] == 4 for entry in run["draw"]["actions"].values())


def test_incomplete_exploratory_dataset_publishes_fail_closed_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dirty_provenance = _clean_source_provenance()
    dirty_provenance.update(
        {
            "dirty": True,
            "git_status_entries": [" M jidian_measurement/emg/primitive_library.py"],
            "formal_reproducible": False,
        }
    )
    monkeypatch.setattr(
        primitive_library,
        "_source_code_provenance",
        lambda: dirty_provenance,
    )
    dataset = _write_dataset(tmp_path, {"split_step": 3}, exploratory=True)
    output = tmp_path / "candidate_v1"
    manifest_path = build_primitive_synergy_library(
        dataset,
        output,
        config=_fast_config(),
    )
    manifest = read_json(manifest_path)
    assert manifest["release"]["formal_ready"] is False
    assert manifest["release"]["release_tier"] == "exploratory_candidate"
    codes = {blocker["code"] for blocker in manifest["release"]["blockers"]}
    assert "missing_primitive_actions" in codes
    assert "insufficient_analysis_ready_trials" in codes
    assert "formal_event_crop_not_proven" in codes
    assert "independent_channel_qc_review_required" in codes
    assert "source_tree_not_clean_and_reproducible" in codes
    assert "no_k_passed_all_adequacy_and_stability_gates" in codes
    review_blocker = next(
        blocker
        for blocker in manifest["release"]["blockers"]
        if blocker["code"] == "independent_channel_qc_review_required"
    )
    diagnostics = manifest["quality"]["channel_scale_diagnostics"]
    expected_review_channels = set(diagnostics["channels_far_below_median_scale"]) | set(
        diagnostics["channels_with_outlier_driven_peak"]
    )
    assert set(review_blocker["zero_based_channel_indices"]) == expected_review_channels
    assert manifest["observation_space_contract"]["training_enabled"] is False
    assert (output / "primitive_synergy_library.npz").exists()
    assert (output / "primitive_synergy_k_scan.json").exists()
    scan = read_json(output / "primitive_synergy_k_scan.json")
    for row in scan["by_k"]:
        assert row["split_half"]["available"] is False
        assert "split_requires_minimum_trials" in row["split_half"]["reason"]
        assert row["bootstrap"]["available"] is False


def test_library_rejects_pinned_k_and_complete_actions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires a scan"):
        PrimitiveLibraryConfig(k_min=2, k_max=2)

    complete = _write_dataset(
        tmp_path,
        {"china_jump_high_clear": 4},
        exploratory=False,
        scope="all",
    )
    with pytest.raises(ValueError, match="complete/unknown"):
        build_primitive_synergy_library(
            complete,
            tmp_path / "invalid",
            required_action_ids=["china_jump_high_clear"],
            config=_fast_config(),
        )

    with pytest.raises(ValueError, match="cannot be an empty"):
        build_primitive_synergy_library(
            _write_dataset(tmp_path / "empty_contract", {"split_step": 4}, exploratory=False),
            tmp_path / "empty_contract_output",
            required_action_ids=[],
            config=_fast_config(),
        )


def test_stable_underfit_rank_fails_adequacy_and_threshold_is_not_hardcoded(
    tmp_path: Path,
) -> None:
    dataset = _write_dataset(
        tmp_path,
        {"split_step": 4, "trunk_rotation": 4},
        exploratory=False,
    )
    output = tmp_path / "rank_contract_v2"
    config = _fast_config(stability_cosine_threshold=0.99999)
    build_primitive_synergy_library(dataset, output, config=config)
    scan = read_json(output / "primitive_synergy_k_scan.json")
    rank_one = next(row for row in scan["by_k"] if row["k"] == 1)
    assert rank_one["adequacy_gates"]["action_balanced_fit_global_vaf"] is False
    assert rank_one["adequacy_pass"] is False
    for row in scan["by_k"]:
        split = row["split_half"]
        minima = np.asarray(
            [run["minimum_cosine_similarity"] for run in split["runs"]],
            dtype=np.float64,
        )
        assert split["cosine_threshold"] == pytest.approx(0.99999)
        assert split["fraction_of_repeats_all_synergies_ge_threshold"] == pytest.approx(np.mean(minima >= 0.99999))


def test_all_included_actions_are_subject_to_minimum_trial_gate(tmp_path: Path) -> None:
    dataset = _write_dataset(
        tmp_path,
        {"split_step": 4, "trunk_rotation": 4, "quiet_stance": 1},
        exploratory=False,
    )
    output = tmp_path / "included_gate_v2"
    manifest_path = build_primitive_synergy_library(
        dataset,
        output,
        required_action_ids=["split_step", "trunk_rotation"],
        config=_fast_config(),
    )
    manifest = read_json(manifest_path)
    blocker = next(
        item for item in manifest["release"]["blockers"] if item["code"] == "insufficient_analysis_ready_trials"
    )
    assert blocker["counts"] == {"quiet_stance": 1}
    assert manifest["release"]["analysis_ready_trial_counts"]["quiet_stance"] == 1


@pytest.mark.parametrize("mutated_array", ["trial_boundaries", "muscle_slugs", "sides", "fs_hz"])
def test_source_semantic_arrays_cannot_change_under_legacy_dataset_hash(
    tmp_path: Path,
    mutated_array: str,
) -> None:
    case_root = tmp_path / mutated_array
    dataset = _write_dataset(case_root, {"split_step": 4}, exploratory=False)
    with np.load(dataset, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]).copy() for name in payload.files}
    if mutated_array == "trial_boundaries":
        arrays[mutated_array][1] += 1
    elif mutated_array == "muscle_slugs":
        arrays[mutated_array][0] = "wrong_muscle"
    elif mutated_array == "sides":
        arrays[mutated_array][0] = "left"
    else:
        arrays[mutated_array] = np.asarray(1999.0)
    atomic_save_npz(dataset, **arrays)
    with pytest.raises(ValueError, match=r"trial_boundaries|muscle_slugs|sides|fs_hz"):
        build_primitive_synergy_library(
            dataset,
            case_root / "invalid_output",
            required_action_ids=["split_step"],
            config=_fast_config(),
        )


def test_qc_review_must_bind_current_diagnostics_and_hashed_evidence(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, {"split_step": 4, "trunk_rotation": 4}, exploratory=False)
    review = _write_qc_review(dataset, tmp_path, _fast_config())
    payload = read_json(review)
    payload["channel_diagnostics_sha256"] = "f" * 64
    atomic_write_json(review, payload)
    with pytest.raises(ValueError, match="channel_diagnostics_sha256"):
        build_primitive_synergy_library(
            dataset,
            tmp_path / "invalid_review_v2",
            required_action_ids=["split_step", "trunk_rotation"],
            config=_fast_config(),
            qc_review_manifest=review,
        )


def test_published_directory_and_library_id_are_immutable(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, {"split_step": 4, "trunk_rotation": 4}, exploratory=True)
    output = tmp_path / "immutable_v2"
    build_primitive_synergy_library(
        dataset,
        output,
        library_id="immutable_library_v2",
        config=_fast_config(),
    )
    original_hash = hashlib.sha256((output / "primitive_synergy_library_manifest.json").read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="overwrite is forbidden"):
        build_primitive_synergy_library(dataset, output, config=_fast_config(), overwrite=True)
    with pytest.raises(FileExistsError, match="library_id"):
        build_primitive_synergy_library(
            dataset,
            tmp_path / "other_directory_v2",
            library_id="immutable_library_v2",
            config=_fast_config(),
        )
    assert (
        hashlib.sha256((output / "primitive_synergy_library_manifest.json").read_bytes()).hexdigest() == original_hash
    )


def test_failed_staging_publish_leaves_no_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _write_dataset(tmp_path, {"split_step": 4, "trunk_rotation": 4}, exploratory=True)
    output = tmp_path / "transaction_v2"
    real_atomic_write_json = primitive_library.atomic_write_json

    def fail_manifest(path: Path, payload: object) -> Path:
        path = Path(path)
        if path.name == "primitive_synergy_library_manifest.json":
            raise OSError("injected manifest publish failure")
        return real_atomic_write_json(path, payload)

    monkeypatch.setattr(primitive_library, "atomic_write_json", fail_manifest)
    with pytest.raises(OSError, match="injected"):
        build_primitive_synergy_library(dataset, output, config=_fast_config())
    assert not output.exists()
    assert not list(tmp_path.glob(".transaction_v2.staging-*"))


def test_verifier_recomputes_manifest_and_artifact_hashes(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, {"split_step": 4, "trunk_rotation": 4}, exploratory=True)
    output = tmp_path / "verified_v2"
    build_primitive_synergy_library(dataset, output, config=_fast_config())
    verified = verify_primitive_synergy_library(output)
    assert len(verified["manifest_sha256"]) == 64
    scan_path = output / "primitive_synergy_k_scan.json"
    scan_path.write_text(scan_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="K scan"):
        verify_primitive_synergy_library(output)

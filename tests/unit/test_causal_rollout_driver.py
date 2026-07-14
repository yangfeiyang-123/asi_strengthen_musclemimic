from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from musclemimic.distill.physical import physical_signal_metadata
from musclemimic.distill.provenance import canonical_json_sha256, file_sha256
from musclemimic.latent_muscle.analysis_export import ANALYSIS_INPUT_SCHEMA_VERSION
from musclemimic.latent_muscle.causal_rollout_artifact import (
    REQUIRED_OUTCOMES,
    build_causal_rollout_artifact,
)
from musclemimic.latent_muscle.causal_rollout_driver import (
    ADAPTER_SCHEMA_VERSION,
    REPLAY_SOURCE_SCHEMA_VERSION,
    ReplayRecordAdapter,
    _sample_seed,
    produce_paired_rollouts,
    validate_job,
)


def _outcome_schemas():
    widths = {
        "muscle_excitation": 2,
        "muscle_activation": 2,
        "joint_position": 3,
        "joint_velocity": 2,
        "trunk_state": 2,
        "racket_state": 3,
        "impact_outcome": 2,
        "landing_outcome": 2,
    }
    semantics = {
        "muscle_excitation": "unit_interval_excitation",
        "muscle_activation": "mujoco_unit_interval_activation_state",
        "joint_position": "ordered_joint_qpos",
        "joint_velocity": "ordered_joint_qvel",
        "trunk_state": "ordered_trunk_state",
        "racket_state": "ordered_racket_state",
        "impact_outcome": "ordered_impact_outcome",
        "landing_outcome": "ordered_landing_outcome",
    }
    result = {}
    for name, width in widths.items():
        names = ["muscle_a", "muscle_b"] if name.startswith("muscle_") else [f"{name}_{i}" for i in range(width)]
        units = ["unit_interval"] * width if name.startswith("muscle_") else ["test_unit"] * width
        result[name] = {
            "feature_names": names,
            "units": units,
            "coordinate_frame": "explicit_test_frame",
            "semantics": semantics[name],
        }
    return result


def _descriptor():
    return {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "checkpoint_fingerprint": "c" * 64,
        "synergy_basis_fingerprint": "b" * 64,
        "environment_fingerprint": "e" * 64,
        "policy_abi_hash": "a" * 64,
        "rollout_engine": "cpu_mock_exact_state_v1",
        "physical_signal_semantics": physical_signal_metadata(),
        "activation_valid_mask": [True, True],
        "outcome_schemas": _outcome_schemas(),
    }


def _write_analysis(tmp_path):
    path = tmp_path / "analysis_inputs.npz"
    np.savez_compressed(
        path,
        sample_uids=np.asarray(["sample-a", "sample-b"]),
        latents=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        intervention_directions=np.eye(2, dtype=np.float32),
        intervention_epsilons=np.asarray([-0.5, 0.5], dtype=np.float32),
    )
    sidecar_path = tmp_path / "analysis_inputs.json"
    sidecar = {
        "schema_version": ANALYSIS_INPUT_SCHEMA_VERSION,
        "npz_sha256": file_sha256(path),
        "checkpoint_fingerprint": "c" * 64,
        "formal_synergy_basis_fingerprint": "b" * 64,
    }
    sidecar["manifest_fingerprint"] = canonical_json_sha256(sidecar)
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    return path, sidecar_path


class _MockAdapter:
    def __init__(self, *, corrupt_restore=False):
        self.corrupt_restore = corrupt_restore
        self.snapshot = b""
        self.random_fingerprint = ""
        self.prepare_calls = 0
        self.evaluate_calls = 0

    def descriptor(self):
        return _descriptor()

    def prepare_analysis_sample(self, sample_uid):
        self.prepare_calls += 1
        self.snapshot = f"full-state:{sample_uid}".encode()
        return self.snapshot

    def snapshot_to_bytes(self, snapshot):
        return bytes(snapshot)

    def restore_snapshot(self, snapshot):
        self.snapshot = bytes(snapshot) + (b"-changed" if self.corrupt_restore else b"")

    def capture_snapshot(self):
        return self.snapshot

    def set_common_random_seed(self, seed):
        self.random_fingerprint = hashlib.sha256(f"all-rng:{seed}".encode()).hexdigest()

    def random_state_fingerprint(self):
        return self.random_fingerprint

    def evaluate_rollout(self, request):
        self.evaluate_calls += 1
        delta = 0.0 if request.is_baseline else 0.02 * request.intervention_epsilon
        widths = {name: len(schema["feature_names"]) for name, schema in _outcome_schemas().items()}
        return {
            name: np.full(width, 0.4 + delta if name.startswith("muscle_") else 1.0 + delta, dtype=np.float32)
            for name, width in widths.items()
        }


def test_driver_evaluates_exact_pairs_and_publishes_builder_compatible_triplet(tmp_path):
    analysis, sidecar = _write_analysis(tmp_path)
    adapter = _MockAdapter()
    output = tmp_path / "paired"
    manifest = produce_paired_rollouts(
        analysis_inputs=analysis,
        analysis_manifest=sidecar,
        adapter=adapter,
        output_dir=output,
        base_seed=7,
        adapter_import="test:mock",
    )
    assert adapter.prepare_calls == 2
    assert adapter.evaluate_calls == 2 * (1 + 2 * 2)
    assert manifest["fixed_state_initialization"] == "exact_snapshot_restore"
    assert manifest["common_random_numbers"] is True
    assert sorted(path.name for path in output.iterdir()) == [
        "baseline_records.npz",
        "paired_rollout_manifest.json",
        "perturbed_records.npz",
    ]
    with np.load(output / "baseline_records.npz", allow_pickle=False) as baseline:
        baseline_seeds = np.asarray(baseline["rollout_seeds"])
    with np.load(output / "perturbed_records.npz", allow_pickle=False) as perturbed:
        assert np.array_equal(
            perturbed["rollout_seeds"],
            np.broadcast_to(baseline_seeds[:, None, None], (2, 2, 2)),
        )
        assert perturbed["muscle_activation"].shape == (2, 2, 2, 2)

    sealed = build_causal_rollout_artifact(
        analysis_inputs=analysis,
        analysis_manifest=sidecar,
        baseline_records=output / "baseline_records.npz",
        perturbed_records=output / "perturbed_records.npz",
        rollout_manifest=output / "paired_rollout_manifest.json",
        output_npz=tmp_path / "causal_effects.npz",
    )
    assert sealed["evidence_kind"] == "environment_rollout"
    assert sealed["policy_abi_hash"] == "a" * 64


def test_driver_fails_closed_before_publication_on_non_exact_restore(tmp_path):
    analysis, sidecar = _write_analysis(tmp_path)
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="exact snapshot restore"):
        produce_paired_rollouts(
            analysis_inputs=analysis,
            analysis_manifest=sidecar,
            adapter=_MockAdapter(corrupt_restore=True),
            output_dir=output,
        )
    assert not output.exists()


def test_dry_run_validates_descriptor_without_state_capture_or_evaluation(tmp_path, monkeypatch):
    analysis, sidecar = _write_analysis(tmp_path)
    adapter = _MockAdapter()
    monkeypatch.setattr(
        "musclemimic.latent_muscle.causal_rollout_driver.load_adapter",
        lambda *_args, **_kwargs: adapter,
    )
    report = validate_job(
        job={
            "schema_version": "latent_causal_rollout_job_v1",
            "analysis_inputs": str(analysis),
            "analysis_manifest": str(sidecar),
            "output_dir": str(tmp_path / "unused"),
            "base_seed": 0,
            "adapter_import": "test:mock",
            "adapter_config": {},
        }
    )
    assert report["rollouts_executed"] is False
    assert report["output_published"] is False
    assert adapter.prepare_calls == 0
    assert adapter.evaluate_calls == 0
    assert not (tmp_path / "unused").exists()


def test_replay_record_adapter_replays_only_bound_external_records(tmp_path):
    analysis, sidecar = _write_analysis(tmp_path)
    first_output = tmp_path / "first"
    produce_paired_rollouts(
        analysis_inputs=analysis,
        analysis_manifest=sidecar,
        adapter=_MockAdapter(),
        output_dir=first_output,
        base_seed=11,
    )
    with np.load(analysis, allow_pickle=False) as values:
        analysis_data = {name: np.asarray(values[name]) for name in values.files}
    with np.load(first_output / "baseline_records.npz", allow_pickle=False) as values:
        baseline = {name: np.asarray(values[name]) for name in values.files}
    with np.load(first_output / "perturbed_records.npz", allow_pickle=False) as values:
        perturbed = {name: np.asarray(values[name]) for name in values.files}
    snapshots = [f"full-state:{uid}".encode() for uid in analysis_data["sample_uids"].astype(str)]
    width = max(map(len, snapshots))
    snapshot_array = np.zeros((len(snapshots), width), dtype=np.uint8)
    for index, value in enumerate(snapshots):
        snapshot_array[index, : len(value)] = np.frombuffer(value, dtype=np.uint8)
    records = {
        "sample_uids": analysis_data["sample_uids"],
        "baseline_latents": analysis_data["latents"],
        "intervention_directions": analysis_data["intervention_directions"],
        "intervention_epsilons": analysis_data["intervention_epsilons"],
        "snapshot_bytes": snapshot_array,
        "snapshot_lengths": np.asarray([len(value) for value in snapshots], dtype=np.int64),
        "snapshot_fingerprints": np.asarray([hashlib.sha256(value).hexdigest() for value in snapshots]),
        "rollout_seeds": np.asarray(
            [_sample_seed(11, uid) for uid in analysis_data["sample_uids"].astype(str)],
            dtype=np.int64,
        ),
        "random_state_fingerprints": np.asarray(
            [
                hashlib.sha256(f"all-rng:{_sample_seed(11, uid)}".encode()).hexdigest()
                for uid in analysis_data["sample_uids"].astype(str)
            ]
        ),
    }
    for name in REQUIRED_OUTCOMES:
        records[f"baseline__{name}"] = baseline[name]
        records[f"perturbed__{name}"] = perturbed[name]
    records_path = tmp_path / "replay_records.npz"
    np.savez_compressed(records_path, **records)
    replay_manifest_path = tmp_path / "replay_records.json"
    replay_manifest = {
        "schema_version": REPLAY_SOURCE_SCHEMA_VERSION,
        "records_sha256": file_sha256(records_path),
        "adapter_descriptor": _descriptor(),
    }
    replay_manifest["manifest_fingerprint"] = canonical_json_sha256(replay_manifest)
    replay_manifest_path.write_text(json.dumps(replay_manifest), encoding="utf-8")

    replay = ReplayRecordAdapter({"records_npz": str(records_path), "records_manifest": str(replay_manifest_path)})
    second_output = tmp_path / "second"
    produce_paired_rollouts(
        analysis_inputs=analysis,
        analysis_manifest=sidecar,
        adapter=replay,
        output_dir=second_output,
        base_seed=11,
        adapter_import="replay-record",
    )
    with np.load(second_output / "perturbed_records.npz", allow_pickle=False) as replayed:
        assert np.array_equal(replayed["landing_outcome"], perturbed["landing_outcome"])

    duplicate_manifest = tmp_path / "duplicate_replay_records.json"
    duplicate_manifest.write_text(
        replay_manifest_path.read_text(encoding="utf-8").replace(
            '"schema_version":',
            '"schema_version": "silently-shadowed", "schema_version":',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        ReplayRecordAdapter({"records_npz": str(records_path), "records_manifest": str(duplicate_manifest)})

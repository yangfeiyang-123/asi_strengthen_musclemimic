"""Immutable continuity training release construction and runtime gates."""

from __future__ import annotations

import copy
import json

import pytest
from omegaconf import OmegaConf

from analysis.physiology_synergy.build_continuity_training_release import (
    build_continuity_training_release,
)
from analysis.physiology_synergy.calibrate_continuity_reward import (
    build_continuity_reward_calibration,
)
from musclemimic.physiology.release import (
    CONTINUITY_TRAINING_RELEASE_SCHEMA_VERSION,
    continuity_training_release_fingerprint,
    load_continuity_training_release,
    resolve_continuity_training_release,
    validate_continuity_training_release,
    validate_release_against_runtime,
)
from musclemimic.runner.checkpointing import validate_checkpoint_continuity_training_contract
from musclemimic.runner.engine import bind_continuity_training_release
from tests.unit.continuity_v3_fixtures import (
    TAXONOMY_PATH,
    baseline_evidence,
    candidate_assets,
)


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _release_fixture(tmp_path):
    taxonomy, _diagnostic, review, candidate, loss_identity = candidate_assets()
    baseline, rollout_manifest, baseline_loss_identity = baseline_evidence()
    assert baseline_loss_identity.loss_spec_fingerprint == loss_identity.loss_spec_fingerprint
    calibration = build_continuity_reward_calibration(
        baseline,
        rollout_manifest=rollout_manifest,
        expected_loss_spec=loss_identity,
        selected_budget_fraction=0.01,
    )
    paths = {
        "review": _write(tmp_path / "review.json", review),
        "candidate": _write(tmp_path / "candidate.json", candidate.to_manifest()),
        "loss": _write(tmp_path / "loss.json", loss_identity.to_manifest()),
        "baseline": _write(tmp_path / "baseline.json", baseline),
        "manifest": _write(tmp_path / "manifest.json", rollout_manifest),
        "calibration": _write(tmp_path / "calibration.json", calibration),
    }
    payload = build_continuity_training_release(
        taxonomy_path=TAXONOMY_PATH,
        topology_review_path=paths["review"],
        candidate_graph_path=paths["candidate"],
        loss_spec_path=paths["loss"],
        baseline_rollout_path=paths["baseline"],
        rollout_manifest_path=paths["manifest"],
        calibration_path=paths["calibration"],
        release_id="fixture_continuity_release_v1",
        created_at_utc="2026-07-30T00:00:00Z",
    )
    release_path = _write(tmp_path / "release.json", payload)
    return taxonomy, candidate, loss_identity, calibration, release_path, paths


def test_release_binds_review_graph_loss_baseline_calibration_and_coefficient(tmp_path):
    taxonomy, candidate, loss_identity, calibration, release_path, _paths = _release_fixture(tmp_path)
    release = load_continuity_training_release(release_path)
    artifacts = resolve_continuity_training_release(release)

    assert release.schema_version == CONTINUITY_TRAINING_RELEASE_SCHEMA_VERSION
    assert release.release_fingerprint == continuity_training_release_fingerprint(release.to_manifest())
    assert release.reward["coefficient"] == calibration["selection"]["coefficient"]
    assert artifacts.loss_identity.loss_spec_fingerprint == loss_identity.loss_spec_fingerprint
    validate_release_against_runtime(
        release,
        taxonomy=taxonomy,
        graph=candidate,
        runtime_loss_identity=loss_identity,
        action_mode="fixed_synergy",
    )


def test_release_rejects_cross_experiment_splicing_and_tampering(tmp_path):
    _taxonomy, _candidate, _loss, _calibration, release_path, paths = _release_fixture(tmp_path)
    release_payload = json.loads(release_path.read_text(encoding="utf-8"))
    release_payload["calibration"]["candidate_loss_spec_fingerprint"] = "9" * 64
    release_payload["release_fingerprint"] = continuity_training_release_fingerprint(release_payload)
    with pytest.raises(ValueError, match="loss spec differs from calibration"):
        validate_continuity_training_release(release_payload)

    calibration = json.loads(paths["calibration"].read_text(encoding="utf-8"))
    calibration["selection"]["coefficient"] *= 2.0
    _write(paths["calibration"], calibration)
    with pytest.raises(ValueError, match=r"selected coefficient is stale|calibration"):
        resolve_continuity_training_release(load_continuity_training_release(release_path))


def test_release_runtime_gate_rejects_action_mode_and_loss_drift(tmp_path):
    taxonomy, candidate, loss_identity, _calibration, release_path, _paths = _release_fixture(tmp_path)
    release = load_continuity_training_release(release_path)
    restricted = copy.deepcopy(release.to_manifest())
    restricted["allowed_action_modes"] = ["full_354"]
    restricted["release_fingerprint"] = continuity_training_release_fingerprint(restricted)
    restricted_release = validate_continuity_training_release(restricted)
    with pytest.raises(ValueError, match="does not allow action mode"):
        validate_release_against_runtime(
            restricted_release,
            taxonomy=taxonomy,
            graph=candidate,
            runtime_loss_identity=loss_identity,
            action_mode="fixed_synergy",
        )

    drifted_identity = copy.copy(loss_identity)
    object.__setattr__(drifted_identity, "loss_spec_fingerprint", "8" * 64)
    with pytest.raises(ValueError, match="loss spec differs from runtime"):
        validate_release_against_runtime(
            release,
            taxonomy=taxonomy,
            graph=candidate,
            runtime_loss_identity=drifted_identity,
            action_mode="fixed_synergy",
        )


def test_training_preflight_injects_release_contract_before_manifest(tmp_path):
    _taxonomy, _candidate, _loss, _calibration, release_path, _paths = _release_fixture(tmp_path)
    release = load_continuity_training_release(release_path)
    config = OmegaConf.create(
        {
            "experiment": {
                "action_representation": {"enabled": False},
                "env_params": {
                    "reward_params": {
                        "intra_muscle_consistency": {
                            "mode": "reward",
                            "release_path": str(release_path),
                            "expected_release_fingerprint": release.release_fingerprint,
                            "action_mode": None,
                        }
                    }
                },
            }
        }
    )
    result_dir = tmp_path / "result"
    contract = bind_continuity_training_release(
        config,
        launch_dir=tmp_path,
        result_dir=result_dir,
    )

    assert contract["release_fingerprint"] == release.release_fingerprint
    assert contract["action_mode"] == "full_354"
    assert (
        config.experiment.continuity_training_contract.loss_spec_fingerprint
        == release.loss_spec["loss_spec_fingerprint"]
    )
    assert config.experiment.env_params.reward_params.intra_muscle_consistency.action_mode == "full_354"
    assert (result_dir / "continuity_training_preflight.json").is_file()


def test_checkpoint_restore_requires_same_release_contract_in_manifest_config_and_metadata(tmp_path):
    contract = {
        "schema_version": "continuity_training_runtime_contract_v1",
        "release_fingerprint": "a" * 64,
        "loss_spec_fingerprint": "b" * 64,
        "binding_sha256": "c" * 64,
    }
    run_dir = tmp_path / "run"
    leaf = run_dir / "checkpoint_3"
    (leaf / "train_state").mkdir(parents=True)
    (leaf / "config").mkdir()
    (leaf / "metadata").mkdir()
    (leaf / "_CHECKPOINT_METADATA").write_text("{}", encoding="utf-8")
    _write(
        leaf / "config" / "metadata",
        {"experiment": {"continuity_training_contract": contract}},
    )
    _write(
        leaf / "metadata" / "metadata",
        {"update_number": 3, "continuity_training_contract": contract},
    )
    _write(
        run_dir / "manifest.json",
        {
            "continuity_training_contract": contract,
            "experiment_config": {"continuity_training_contract": contract},
        },
    )

    validate_checkpoint_continuity_training_contract(leaf, contract)

    drifted = {**contract, "release_fingerprint": "9" * 64}
    _write(
        leaf / "metadata" / "metadata",
        {"update_number": 3, "continuity_training_contract": drifted},
    )
    with pytest.raises(ValueError, match="differs between config and metadata"):
        validate_checkpoint_continuity_training_contract(leaf, contract)

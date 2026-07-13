from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from fullbody import eval_finger_robustness
from fullbody.eval_finger_robustness import (
    ROLLOUT_METRIC_MAP,
    SPIKE_METRIC_MAP,
    _rows_to_payload,
    compare_finger_robustness,
)
from musclemimic.badminton import training_gates
from musclemimic.badminton.promotion_artifact import checkpoint_identity
from musclemimic.badminton.stage1r_artifact import (
    CANONICAL_METRICS_ENVS,
    CANONICAL_METRICS_STEPS,
    CANONICAL_SEEDS,
    LEGACY_EVIDENCE_KIND,
    build_evaluation_contract,
    build_verified_report,
    canonical_heldout_motion_paths,
    canonical_mapping_sha256,
    validate_stage1r_report,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _make_checkpoint(tmp_path: Path, name: str, *, payload: bytes) -> Path:
    run_dir = tmp_path / name
    checkpoint = run_dir / "checkpoint_7"
    metadata = checkpoint / "metadata" / "metadata"
    metadata.parent.mkdir(parents=True)
    _write_json(
        run_dir / "manifest.json",
        {"config_hash": f"config-{name}"},
    )
    _write_json(
        metadata,
        {
            "update_number": 7,
            "global_timestep": 700,
            "target_global_timestep": 1000,
        },
    )
    (checkpoint / "policy.bin").write_bytes(payload)
    return checkpoint


def _rollout_row(value: float = 0.1) -> dict[str, float]:
    keys = set(ROLLOUT_METRIC_MAP.values()) | set(SPIKE_METRIC_MAP.values())
    return {
        key: (0.0 if key == "val_early_termination_rate" else value)
        for key in keys
    }


def _make_verified_artifact(
    tmp_path: Path,
    checkpoint: Path,
    *,
    scale: float = 0.03,
    name: str = "evidence",
) -> tuple[Path, Path, Path]:
    output_dir = tmp_path / name
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = build_evaluation_contract(
        motion_paths=canonical_heldout_motion_paths(),
        seeds=CANONICAL_SEEDS,
        perturb_qpos_scale=scale,
        perturb_qvel_scale=0.0,
        metrics_envs=CANONICAL_METRICS_ENVS,
        metrics_steps=CANONICAL_METRICS_STEPS,
    )
    identity = checkpoint_identity(checkpoint)
    base_provenance = {
        "checkpoint": identity["checkpoint_path"],
        "checkpoint_identity": identity,
        "motion_paths": contract["heldout_motion_identity"]["motion_paths"],
        "heldout_motion_identity": contract["heldout_motion_identity"],
        "deterministic_policy": True,
        "evaluate_all": True,
        "finger_perturb_rng_mode": "fold_in",
        "finger_perturb_side": "right",
        "metrics_envs": CANONICAL_METRICS_ENVS,
        "metrics_steps": CANONICAL_METRICS_STEPS,
    }
    rows = [_rollout_row() for _ in CANONICAL_SEEDS]
    clean = _rows_to_payload(
        rows,
        list(CANONICAL_SEEDS),
        base_provenance
        | {
            "finger_qpos_perturb_scale": 0.0,
            "finger_qvel_perturb_scale": 0.0,
        },
        condition="clean",
        checkpoint_identity_payload=identity,
        evaluation_contract=contract,
    )
    perturbed = _rows_to_payload(
        rows,
        list(CANONICAL_SEEDS),
        base_provenance
        | {
            "finger_qpos_perturb_scale": scale,
            "finger_qvel_perturb_scale": 0.0,
        },
        condition="perturbed",
        checkpoint_identity_payload=identity,
        evaluation_contract=contract,
    )
    clean_path = output_dir / "clean_rollouts.json"
    perturbed_path = output_dir / "perturbed_rollouts.json"
    report_path = output_dir / "paired_robustness.json"
    _write_json(clean_path, clean)
    _write_json(perturbed_path, perturbed)
    report = build_verified_report(
        compare_finger_robustness(clean, perturbed),
        checkpoint=checkpoint,
        evaluation_contract=contract,
        clean_source_path=clean_path,
        perturbed_source_path=perturbed_path,
    )
    _write_json(report_path, report)
    return report_path, clean_path, perturbed_path


def test_verified_stage1r_report_round_trips(tmp_path):
    checkpoint = _make_checkpoint(tmp_path, "run-a", payload=b"policy-a")
    report_path, _, _ = _make_verified_artifact(tmp_path, checkpoint)

    report = validate_stage1r_report(
        report_path,
        expected_checkpoint=checkpoint,
        expected_perturb_qpos_scale=0.03,
    )

    assert report["schema_version"] == "stage1r_paired_robustness_v3"
    assert report["production_eligible"] is True
    assert report["pair_count"] == 5
    assert report["checkpoint_identity"]["checkpoint_content_sha256"]
    assert set(report["source_payloads"]) == {"clean", "perturbed"}


def test_checkpoint_mode_cli_writes_verified_v3_report(
    tmp_path,
    monkeypatch,
):
    checkpoint = _make_checkpoint(tmp_path, "run-a", payload=b"policy-a")
    report_path = tmp_path / "cli-evidence" / "paired_robustness.json"
    monkeypatch.setattr(
        eval_finger_robustness,
        "_run_eval_once",
        lambda **_kwargs: _rollout_row(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_finger_robustness",
            "--checkpoint",
            str(checkpoint),
            "--motion_path",
            *canonical_heldout_motion_paths(),
            "--output",
            str(report_path),
            "--require_pass",
        ],
    )

    assert eval_finger_robustness.main() == 0
    report = validate_stage1r_report(
        report_path,
        expected_checkpoint=checkpoint,
        expected_perturb_qpos_scale=0.03,
    )
    assert report["schema_version"] == "stage1r_paired_robustness_v3"
    assert report["evaluation_contract"]["seeds"] == list(CANONICAL_SEEDS)


def test_production_training_gate_accepts_only_bound_report(tmp_path, monkeypatch):
    checkpoint = _make_checkpoint(tmp_path, "run-a", payload=b"policy-a")
    report_path, _, _ = _make_verified_artifact(tmp_path, checkpoint)
    gate_path = tmp_path / "stage1r_gate.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "training_gates",
            "--stage",
            "stage1r",
            "--metrics",
            str(report_path),
            "--checkpoint",
            str(checkpoint),
            "--finger-perturb-qpos-scale",
            "0.03",
            "--output",
            str(gate_path),
            "--require_pass",
        ],
    )

    assert training_gates.main() == 0
    assert json.loads(gate_path.read_text(encoding="utf-8"))["passed"] is True


def test_production_validator_rejects_legacy_offline_report(tmp_path):
    checkpoint = _make_checkpoint(tmp_path, "run-a", payload=b"policy-a")
    metrics = {
        "body_site_error": [0.1],
        "right_hand_site_error": [0.1],
        "racket_head_position_error": [0.1],
        "racket_head_rotation_error": [0.1],
        "early_termination": [0.0],
        "val_max_err_root_xyz": [0.1],
        "val_max_err_right_hand_pos": [0.1],
        "val_max_err_racket_pos": [0.1],
        "val_max_err_racket_rot": [0.1],
    }
    legacy = compare_finger_robustness(
        {"seeds": [9], "metrics": metrics},
        {
            "seeds": [9],
            "metrics": metrics,
            "provenance": {"finger_qpos_perturb_scale": 0.03},
        },
    )
    report_path = tmp_path / "legacy.json"
    _write_json(report_path, legacy)

    assert legacy["evidence_kind"] == LEGACY_EVIDENCE_KIND
    with pytest.raises(ValueError, match="legacy/offline"):
        validate_stage1r_report(
            report_path,
            expected_checkpoint=checkpoint,
            expected_perturb_qpos_scale=0.03,
        )


def test_stage1r_report_fails_after_checkpoint_byte_change(tmp_path):
    checkpoint = _make_checkpoint(tmp_path, "run-a", payload=b"policy-a")
    report_path, _, _ = _make_verified_artifact(tmp_path, checkpoint)
    (checkpoint / "policy.bin").write_bytes(b"mutated-policy")

    with pytest.raises(ValueError, match="checkpoint identity"):
        validate_stage1r_report(
            report_path,
            expected_checkpoint=checkpoint,
            expected_perturb_qpos_scale=0.03,
        )


def test_stage1r_report_fails_after_source_payload_byte_change(tmp_path):
    checkpoint = _make_checkpoint(tmp_path, "run-a", payload=b"policy-a")
    report_path, clean_path, _ = _make_verified_artifact(tmp_path, checkpoint)
    clean_path.write_text(
        clean_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="clean source payload hash is stale"):
        validate_stage1r_report(
            report_path,
            expected_checkpoint=checkpoint,
            expected_perturb_qpos_scale=0.03,
        )


def test_stage1r_report_fails_for_a_different_perturbation_rung(tmp_path):
    checkpoint = _make_checkpoint(tmp_path, "run-a", payload=b"policy-a")
    report_path, _, _ = _make_verified_artifact(
        tmp_path,
        checkpoint,
        scale=0.03,
    )

    with pytest.raises(ValueError, match="evaluation contract"):
        validate_stage1r_report(
            report_path,
            expected_checkpoint=checkpoint,
            expected_perturb_qpos_scale=0.05,
        )


def test_stage1r_report_fails_when_checkpoint_symlink_is_retargeted(tmp_path):
    checkpoint_a = _make_checkpoint(tmp_path, "run-a", payload=b"policy-a")
    checkpoint_b = _make_checkpoint(tmp_path, "run-b", payload=b"policy-b")
    latest = tmp_path / "latest"
    latest.symlink_to(checkpoint_a, target_is_directory=True)
    report_path, _, _ = _make_verified_artifact(
        tmp_path,
        latest,
        name="symlink-evidence",
    )
    latest.unlink()
    latest.symlink_to(checkpoint_b, target_is_directory=True)

    with pytest.raises(ValueError, match="checkpoint identity"):
        validate_stage1r_report(
            report_path,
            expected_checkpoint=latest,
            expected_perturb_qpos_scale=0.03,
        )


def test_stage1r_report_rejects_clean_perturbed_checkpoint_mix(tmp_path):
    checkpoint_a = _make_checkpoint(tmp_path, "run-a", payload=b"policy-a")
    checkpoint_b = _make_checkpoint(tmp_path, "run-b", payload=b"policy-b")
    report_a, _, _ = _make_verified_artifact(
        tmp_path,
        checkpoint_a,
        name="evidence-a",
    )
    report_b, _, _ = _make_verified_artifact(
        tmp_path,
        checkpoint_b,
        name="evidence-b",
    )
    mixed = json.loads(report_a.read_text(encoding="utf-8"))
    other = json.loads(report_b.read_text(encoding="utf-8"))
    mixed["source_payloads"]["perturbed"] = other["source_payloads"]["perturbed"]
    mixed.pop("binding_sha256")
    mixed["binding_sha256"] = canonical_mapping_sha256(mixed)
    _write_json(report_a, mixed)

    with pytest.raises(ValueError, match="perturbed checkpoint identity"):
        validate_stage1r_report(
            report_a,
            expected_checkpoint=checkpoint_a,
            expected_perturb_qpos_scale=0.03,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"seeds": (0, 1, 2, 3)}, "exact distinct seeds"),
        ({"motion_paths": ("not-canonical",) * 5}, "canonical ordered five"),
        ({"metrics_envs": 4}, "metrics_envs=5"),
        ({"metrics_steps": 499}, "metrics_steps=500"),
        ({"perturb_qvel_scale": 0.01}, "zero qvel"),
    ],
)
def test_stage1r_production_contract_rejects_noncanonical_settings(kwargs, message):
    values = {
        "motion_paths": canonical_heldout_motion_paths(),
        "seeds": CANONICAL_SEEDS,
        "perturb_qpos_scale": 0.03,
        "perturb_qvel_scale": 0.0,
        "metrics_envs": CANONICAL_METRICS_ENVS,
        "metrics_steps": CANONICAL_METRICS_STEPS,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        build_evaluation_contract(**values)

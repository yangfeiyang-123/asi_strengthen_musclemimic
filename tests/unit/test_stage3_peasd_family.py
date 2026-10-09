from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from musclemimic.badminton import stage3_peasd_family as family


def _canonical(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _environment() -> dict:
    return {
        "scene_sha256": "1" * 64,
        "full_action_size": 354,
        "control_substeps": 10,
        "max_episode_steps": 420,
        "reward_weights": {"hit_bonus": 2.0},
        "player_half_sign": -1,
        "singles": True,
        "terminate_on_body_fall": True,
        "swing_duration_s": 1.2,
        "contact_phase": 0.55,
        "task_profile": "impact_recovery_v2",
        "v2_observation_size": 19,
        "recovery_horizon_steps": 60,
        "task_curriculum_stage": "C7_recovery",
    }


def _selection(path: Path, fingerprint: str) -> dict:
    value = {
        "selection_manifest_fingerprint": "8" * 64,
        "checkpoints": {
            "best_synergy": {
                "checkpoint_fingerprint": fingerprint,
                "decoder_type": "synergy_residual",
                "latent_dim": 4,
            }
        },
    }
    _write(path, value)
    return value


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    s2b_fingerprint = "b" * 64
    s2c_fingerprint = "c" * 64
    s2b_path = tmp_path / "s2b_selection.json"
    s2c_path = tmp_path / "s2c_selection.json"
    selections = {
        str(s2b_path.resolve()): _selection(s2b_path, s2b_fingerprint),
        str(s2c_path.resolve()): _selection(s2c_path, s2c_fingerprint),
    }
    stage2_index = {
        "binding_sha256": "d" * 64,
        "action": {
            "slug": "forehand_clear",
            "action_id": "forehandClear_standard",
            "tube_action_id": "Forehand Clear",
        },
        "shared_inputs": {"path": "/sealed/stage2_shared.json"},
        "arms": {
            "S2-B": {
                "selection_manifest": {
                    "path": str(s2b_path.resolve()),
                    "content_sha256": _sha(s2b_path),
                }
            },
            "S2-C": {
                "selection_manifest": {
                    "path": str(s2c_path.resolve()),
                    "content_sha256": _sha(s2c_path),
                }
            },
        },
    }
    index_path = _write(tmp_path / "stage2_index.json", stage2_index)
    stage2_gate = {
        "binding_sha256": "e" * 64,
        "passed": True,
        "family_index": {
            "path": str(index_path.resolve()),
            "content_sha256": _sha(index_path),
            "artifact_fingerprint": stage2_index["binding_sha256"],
        },
    }
    gate_path = _write(tmp_path / "stage2_gate.json", stage2_gate)

    monkeypatch.setattr(
        family,
        "validate_stage2_context_family_gate",
        lambda path, require_pass=True: copy.deepcopy(stage2_gate),
    )
    monkeypatch.setattr(
        family,
        "validate_stage2_context_family_index",
        lambda path: copy.deepcopy(stage2_index),
    )
    monkeypatch.setattr(
        family,
        "validate_selected_artifact",
        lambda path: copy.deepcopy(selections[str(Path(path).resolve())]),
    )

    feed = {
        "sample_fingerprints": ["a" * 64]
        + [f"feed-{index:03d}" for index in range(1, 128)]
    }
    reports: dict[str, dict[int, Path]] = {arm: {} for arm in family.EXACT_ARMS}
    releases: dict[str, dict[int, Path]] = {arm: {} for arm in family.EXACT_ARMS}
    release_values: dict[str, dict] = {}
    correction_values: dict[str, dict] = {}
    expected_entries: dict[str, dict] = {}
    report_offsets = {
        "H1": (0.50, 0.12),
        "H2": (0.60, 0.10),
        "H3": (0.61, 0.08),
    }
    for arm in family.EXACT_ARMS:
        latent = s2b_fingerprint if arm == "H1" else s2c_fingerprint
        for seed in family.EXACT_SEEDS:
            root = tmp_path / arm / f"seed-{seed}"
            root.mkdir(parents=True)
            release_path = _write(root / "reachability_release.json", {"arm": arm, "seed": seed})
            correction_path = _write(root / "correction_manifest.json", {"passed": True})
            dataset_path = root / "correction.npz"
            dataset_path.write_bytes(b"correction")
            source_checkpoint = root / "source" / "policy.npz"
            source_checkpoint.parent.mkdir(parents=True)
            source_checkpoint.write_bytes(f"source-{arm}-{seed}".encode())
            cem_report = _write(root / "cem" / "cem_report.json", {"passed": True})
            cem_candidate = _write(
                root / "cem" / "best_teacher.json", {"passed": True}
            )
            cpu_audit_trace = root / "cpu_audit_trace.npz"
            cpu_audit_trace.write_bytes(f"cpu-{arm}-{seed}".encode())
            correction_values[str(correction_path.resolve())] = {
                "source_checkpoint": {"payload_path": str(source_checkpoint)},
                "cem": {
                    "report": {"path": str(cem_report)},
                    "candidate": {"path": str(cem_candidate)},
                },
                "cpu_audit": {"trace": {"path": str(cpu_audit_trace)}},
                "correction_dataset": {"path": str(dataset_path)},
            }
            short_payload = root / "short_bc" / "checkpoints" / "checkpoint_0" / "policy.npz"
            short_payload.parent.mkdir(parents=True)
            short_payload.write_bytes(b"policy")
            release = {
                "release_binding_sha256": hashlib.sha256(
                    f"release-{arm}-{seed}".encode()
                ).hexdigest(),
                "latent_identity": {
                    "kind": "latent_checkpoint",
                    "fingerprint": latent,
                },
                "spec_identity": {
                    "spec_sha256": "2" * 64,
                    "scene_sha256": "1" * 64,
                },
                "target_identity": {
                    "action": "forehand_clear",
                    "dataset_action_id": "forehandClear_standard",
                    "single_feed_fingerprint": "a" * 64,
                },
                "correction_dataset_manifest": {"path": str(correction_path)},
                "short_bc": {
                    "checkpoint": {"payload_path": str(short_payload)},
                    "runtime_control_manifest": {"control": arm},
                    "runtime_training_feed_manifest": feed,
                },
            }
            release_values[str(release_path.resolve())] = release
            entry_unsigned = {
                "schema_version": "stage3_static_ppo_reachability_entry_v1",
                "verified": True,
                "release_path": str(release_path.resolve()),
            }
            entry = {**entry_unsigned, "binding_sha256": _canonical(entry_unsigned)}
            expected_entries[str(release_path.resolve())] = entry
            prerequisite_unsigned = {
                "schema_version": "stage3_training_prerequisite_binding_v1",
                "verified": True,
                "stage3_reachability_release": entry,
            }
            prerequisite = {
                **prerequisite_unsigned,
                "binding_sha256": _canonical(prerequisite_unsigned),
            }
            metadata_path = _write(
                root / "policy.json",
                {"config": {"seed": seed}, "training_prerequisite_binding": prerequisite},
            )
            residual = (
                {
                    "bounded_residual_dim": 1,
                    "bounded_residual_schema_hash": "7" * 64,
                    "bounded_residual_groups": [
                        {
                            "name": "wrist_forearm",
                            "actuator_names": ["wrist"],
                            "alpha": 0.05,
                        }
                    ],
                }
                if arm == "H3"
                else {
                    "bounded_residual_dim": 0,
                    "bounded_residual_schema_hash": None,
                    "bounded_residual_groups": None,
                }
            )
            landing, impact = report_offsets[arm]
            binding = {
                "binding_sha256": hashlib.sha256(
                    f"binding-{arm}-{seed}".encode()
                ).hexdigest(),
                "checkpoint_payload_sha256": hashlib.sha256(
                    f"checkpoint-{arm}-{seed}".encode()
                ).hexdigest(),
                "checkpoint_metadata_path": str(metadata_path.resolve()),
                "checkpoint_metadata_sha256": _sha(metadata_path),
                "training_prerequisite_binding_sha256": prerequisite[
                    "binding_sha256"
                ],
                "action_family": "fixed_synergy",
                "latent_checkpoint_fingerprint": latent,
                "spec_sha256": "2" * 64,
                "scene_sha256": "1" * 64,
                "training_feed_manifest_sha256": _canonical(feed),
                "evaluation_feed_manifest_sha256": _canonical(feed),
                "training_target_bank_sha256": "3" * 64,
                "training_target_source_fingerprint": "4" * 64,
                "training_target_file_sha256": "5" * 64,
                "evaluation_target_bank_sha256": "6" * 64,
                "evaluation_target_source_fingerprint": "7" * 64,
                "evaluation_target_file_sha256": "8" * 64,
                "training_seed": seed,
                "evaluation_seed": 123,
                "checkpoint_env_steps": 30_000_000,
                "checkpoint_task_curriculum_max_stage": "C7_recovery",
                "checkpoint_task_curriculum_complete": True,
            }
            report = {
                "schema_version": "incoming_shuttle_hit_evaluate_v3",
                "action_family": "fixed_synergy",
                "evaluation_seed": 123,
                "evaluation_feed_source": "heldout_evaluation_bank",
                "evaluated_feed_count": 128,
                "required_heldout_feed_count": 128,
                "episodes": [{"episode": index} for index in range(128)],
                "opponent_back_landing_rate": landing + seed * 0.001,
                "impact_position_error_m": impact + seed * 0.001,
                "hit_rate": 0.90,
                "no_fall_rate": 1.0,
                "training_feed_manifest": copy.deepcopy(feed),
                "evaluation_feed_manifest": copy.deepcopy(feed),
                "control_manifest": {
                    "schema_version": "stage3_lab_control_v1",
                    "latent_checkpoint_fingerprint": latent,
                    "environment_abi": _environment(),
                    "racket_attachment": {"attachment_hash": "9" * 64},
                    **residual,
                },
                "artifact_binding": binding,
            }
            report_path = _write(root / "evaluate_report.json", report)
            reports[arm][seed] = report_path
            releases[arm][seed] = release_path

    def fake_evaluation(
        report: dict,
        *,
        report_path: Path,
        family: str,
        expected_action_family: str,
        expected_latent_fingerprint: str,
    ) -> dict:
        if report.get("schema_version") != "incoming_shuttle_hit_evaluate_v3":
            raise ValueError("evaluation report was tampered")
        binding = report["artifact_binding"]
        if (
            binding.get("action_family") != expected_action_family
            or binding.get("latent_checkpoint_fingerprint")
            != expected_latent_fingerprint
            or report["control_manifest"].get("latent_checkpoint_fingerprint")
            != expected_latent_fingerprint
        ):
            raise ValueError("evaluation uses the wrong latent identity")
        return {
            "binding": binding,
            "episode_indices": [item["episode"] for item in report["episodes"]],
        }

    monkeypatch.setattr(family, "_validate_evaluation_binding", fake_evaluation)
    monkeypatch.setattr(
        family,
        "validate_stage3_reachability_release",
        lambda path: copy.deepcopy(release_values[str(Path(path).resolve())]),
    )
    monkeypatch.setattr(
        family,
        "validate_successful_correction_dataset_manifest",
        lambda path: copy.deepcopy(
            correction_values[str(Path(path).resolve())]
        ),
    )
    monkeypatch.setattr(
        family,
        "validate_static_ppo_entry",
        lambda **kwargs: copy.deepcopy(
            expected_entries[str(Path(kwargs["release_path"]).resolve())]
        ),
    )
    return {
        "stage2_gate": gate_path,
        "reports": reports,
        "releases": releases,
        "release_values": release_values,
        "correction_values": correction_values,
    }


def _build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict]:
    fixture = _fixture(tmp_path, monkeypatch)
    index = family.build_stage3_peasd_family_index(
        stage2_family_gate=fixture["stage2_gate"],
        reports=fixture["reports"],
        reachability_releases=fixture["releases"],
    )
    return fixture, index


def test_family_seals_exact_stage2_selections_reachability_and_seed_statistics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, index = _build(tmp_path, monkeypatch)
    assert index["arms"]["H1"]["stage2_source_arm"] == "S2-B"
    assert index["arms"]["H2"]["stage2_source_arm"] == "S2-C"
    assert index["arms"]["H3"]["stage2_source_arm"] == "S2-C"
    assert index["arms"]["H2"]["latent_checkpoint_fingerprint"] == index["arms"]["H3"][
        "latent_checkpoint_fingerprint"
    ]
    assert index["arms"]["H1"]["latent_checkpoint_fingerprint"] != index["arms"]["H2"][
        "latent_checkpoint_fingerprint"
    ]
    index_path = _write(tmp_path / "stage3_family_index.json", index)
    gate = family.build_stage3_peasd_family_gate(family_index=index_path)
    assert gate["passed"] is True
    assert gate["h2_vs_h1"]["primary"]["statistics"]["n"] == 3
    assert gate["h2_vs_h1"]["primary"]["statistics"]["degrees_of_freedom"] == 2
    assert gate["h2_vs_h1"]["primary"]["statistics"]["significance_claimed"] is False


def test_family_rejects_missing_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    reports = dict(fixture["reports"])
    reports.pop("H3")
    with pytest.raises(ValueError, match="exact H1/H2/H3"):
        family.build_stage3_peasd_family_index(
            stage2_family_gate=fixture["stage2_gate"],
            reports=reports,
            reachability_releases=fixture["releases"],
        )


def test_family_rejects_wrong_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    report_path = fixture["reports"]["H2"][1]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["artifact_binding"]["training_seed"] = 0
    _write(report_path, report)
    with pytest.raises(ValueError, match="wrong training seed"):
        family.build_stage3_peasd_family_index(
            stage2_family_gate=fixture["stage2_gate"],
            reports=fixture["reports"],
            reachability_releases=fixture["releases"],
        )


def test_family_rejects_wrong_latent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    report_path = fixture["reports"]["H1"][0]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["control_manifest"]["latent_checkpoint_fingerprint"] = "c" * 64
    _write(report_path, report)
    with pytest.raises(ValueError, match="wrong latent"):
        family.build_stage3_peasd_family_index(
            stage2_family_gate=fixture["stage2_gate"],
            reports=fixture["reports"],
            reachability_releases=fixture["releases"],
        )


def test_family_rejects_wrong_feed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    report_path = fixture["reports"]["H3"][2]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["evaluation_feed_manifest"]["sample_fingerprints"][5] = "wrong-feed"
    _write(report_path, report)
    with pytest.raises(ValueError, match="protocols differ"):
        family.build_stage3_peasd_family_index(
            stage2_family_gate=fixture["stage2_gate"],
            reports=fixture["reports"],
            reachability_releases=fixture["releases"],
        )


def test_family_rejects_wrong_reachability_checkpoint_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    report_path = fixture["reports"]["H2"][2]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metadata_path = Path(report["artifact_binding"]["checkpoint_metadata_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    prerequisite = metadata["training_prerequisite_binding"]
    prerequisite["stage3_reachability_release"] = copy.deepcopy(
        prerequisite["stage3_reachability_release"]
    )
    prerequisite["stage3_reachability_release"]["release_path"] = "/wrong/release.json"
    entry = prerequisite["stage3_reachability_release"]
    unsigned_entry = dict(entry)
    unsigned_entry.pop("binding_sha256")
    entry["binding_sha256"] = _canonical(unsigned_entry)
    unsigned = dict(prerequisite)
    unsigned.pop("binding_sha256")
    prerequisite["binding_sha256"] = _canonical(unsigned)
    _write(metadata_path, metadata)
    report["artifact_binding"]["training_prerequisite_binding_sha256"] = prerequisite[
        "binding_sha256"
    ]
    _write(report_path, report)
    with pytest.raises(ValueError, match="wrong reachability release"):
        family.build_stage3_peasd_family_index(
            stage2_family_gate=fixture["stage2_gate"],
            reports=fixture["reports"],
            reachability_releases=fixture["releases"],
        )


def test_family_rejects_cross_arm_training_hyperparameter_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    for report_path in fixture["reports"]["H2"].values():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        metadata_path = Path(report["artifact_binding"]["checkpoint_metadata_path"])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["config"]["learning_rate"] = 1.0e-2
        _write(metadata_path, metadata)
    with pytest.raises(ValueError, match="matched PPO or network hyperparameters"):
        family.build_stage3_peasd_family_index(
            stage2_family_gate=fixture["stage2_gate"],
            reports=fixture["reports"],
            reachability_releases=fixture["releases"],
        )


def test_family_rejects_reused_internal_reachability_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    h1_manifest = str(
        (fixture["releases"]["H1"][0].parent / "correction_manifest.json").resolve()
    )
    h2_manifest = str(
        (fixture["releases"]["H2"][1].parent / "correction_manifest.json").resolve()
    )
    fixture["correction_values"][h2_manifest]["source_checkpoint"] = copy.deepcopy(
        fixture["correction_values"][h1_manifest]["source_checkpoint"]
    )
    with pytest.raises(ValueError, match="reuse internal reachability lineage"):
        family.build_stage3_peasd_family_index(
            stage2_family_gate=fixture["stage2_gate"],
            reports=fixture["reports"],
            reachability_releases=fixture["releases"],
        )


def test_family_rejects_reachability_release_from_another_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    release_path = fixture["releases"]["H3"][2]
    release = fixture["release_values"][str(release_path.resolve())]
    release["target_identity"]["action"] = "forehand_lift"
    release["target_identity"]["dataset_action_id"] = "forehandLift_standard"
    with pytest.raises(ValueError, match="different action"):
        family.build_stage3_peasd_family_index(
            stage2_family_gate=fixture["stage2_gate"],
            reports=fixture["reports"],
            reachability_releases=fixture["releases"],
        )


def test_family_index_revalidation_rejects_report_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, index = _build(tmp_path, monkeypatch)
    index_path = _write(tmp_path / "stage3_family_index.json", index)
    report_path = fixture["reports"]["H1"][0]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["opponent_back_landing_rate"] = 0.99
    _write(report_path, report)
    with pytest.raises(ValueError, match="bound source changed"):
        family.validate_stage3_peasd_family_index(index_path)


def test_family_rejects_posthoc_threshold_contract(tmp_path: Path) -> None:
    contract = copy.deepcopy(family.PRE_REGISTERED_COMPARISON_CONTRACT)
    contract["h2_vs_h1"]["primary"][
        "per_seed_improvement_strictly_greater_than"
    ] = -0.01
    path = _write(tmp_path / "posthoc_contract.json", contract)
    with pytest.raises(ValueError, match="pre-registered source contract"):
        family.validate_comparison_contract(path)


def test_family_gate_fails_one_seed_primary_without_significance_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    report_path = fixture["reports"]["H2"][1]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["opponent_back_landing_rate"] = 0.40
    _write(report_path, report)
    index = family.build_stage3_peasd_family_index(
        stage2_family_gate=fixture["stage2_gate"],
        reports=fixture["reports"],
        reachability_releases=fixture["releases"],
    )
    index_path = _write(tmp_path / "stage3_family_index.json", index)
    gate = family.build_stage3_peasd_family_gate(family_index=index_path)
    assert gate["passed"] is False
    assert gate["h2_vs_h1"]["primary"]["statistics"]["seed_failure_count"] == 1
    assert gate["statistical_scope"]["significance_claimed"] is False

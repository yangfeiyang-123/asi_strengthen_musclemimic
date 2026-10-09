from __future__ import annotations

import hashlib
import json
from pathlib import Path

from musclemimic.badminton.scripts.latent_synergy_sweep import (
    _canonical_json_sha256,
    _materialize_selected_artifact,
)
from musclemimic.badminton.scripts.run_incoming_shuttle_hit import (
    _stage3_evaluation_content_sha256,
)
from musclemimic.badminton.stage3_paired_comparison import (
    build_paired_comparison,
)
from musclemimic.evaluation.stage3_signal_export import load_paired_policy_evidence
from musclemimic.latent_muscle.checkpoint import latent_checkpoint_fingerprint


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _selection(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    models = {}
    fingerprints = {}
    for family, decoder in (
        ("best_direct", "direct"),
        ("best_synergy", "synergy_residual"),
    ):
        checkpoint = tmp_path / family / "checkpoint"
        checkpoint.mkdir(parents=True)
        (checkpoint / "prior.msgpack").write_bytes(f"prior-{family}".encode())
        (checkpoint / "decoder.msgpack").write_bytes(f"decoder-{family}".encode())
        fingerprint = latent_checkpoint_fingerprint(checkpoint)
        fingerprints[family] = fingerprint
        models[family] = {
            "run_name": family,
            "checkpoint_dir": str(checkpoint),
            "checkpoint_fingerprint": fingerprint,
            "formal_synergy_basis_fingerprint": "b" * 64,
            "runtime_synergy_basis_fingerprint": (None if decoder == "direct" else "c" * 64),
            "runtime_synergy_basis_source_fingerprint": (None if decoder == "direct" else "b" * 64),
            "dataset_fingerprint": "d" * 64,
            "validation_dataset_fingerprint": "e" * 64,
            "motion_split_fingerprint": "f" * 64,
            "latent_dim": 4,
            "decoder_type": decoder,
            "seed": 0,
        }
    promotion = {
        "selection_rule": {"deployment_seed": "smallest"},
        "selected_models": models,
        "selected_model": models["best_synergy"],
    }
    promotion["promotion_metrics_fingerprint"] = _canonical_json_sha256(promotion)
    _materialize_selected_artifact(
        tmp_path / "latent",
        promotion_metrics=promotion,
        plan={
            "plan_fingerprint": "a" * 64,
            "jobs": [
                {"decoder_type": "direct"},
                {"decoder_type": "synergy_residual"},
            ],
        },
    )
    # Formal Stage-3 selection binds only the deployed synergy checkpoint.
    # The latent-direct sweep artifact is an optional ablation and must not be
    # a prerequisite for the independent full354 policy.
    manifest_path = tmp_path / "latent" / "selected" / "selection_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoints"].pop("best_direct")
    manifest.pop("selection_manifest_fingerprint")
    manifest["selection_manifest_fingerprint"] = _canonical_json_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "latent" / "selected" / "best_direct").unlink()
    return (
        manifest_path,
        fingerprints,
    )


def _episode(index: int, offset: float, *, is_lab: bool) -> dict:
    episode = {
        "episode": index,
        "return": 10.0 + offset,
        "hit": True,
        "crossed_net": True,
        "body_fall": False,
        "landing_region": "opponent_back",
        "contact_racket_head_speed_m_s": 10.0 + offset,
        "net_clearance_m": 0.5 + offset,
        "recovery_complete": True,
        "lab_diagnostics": {
            "normalized_control_energy": 0.2 - offset,
            "body_action_saturation_fraction": 0.0,
            "full_action_saturation_fraction": 0.0,
        },
        "stage3_v2_metrics": {
            "impact_position_error_m": 0.05 - offset,
            "impact_rho2": 0.1,
            "impact_timing_error_s": 0.03 - offset,
            "stringbed_normal_error_rad": 0.1 - offset,
            "racket_linear_velocity_error_m_s": 0.5 - offset,
            "racket_angular_velocity_error_rad_s": 2.0 - offset,
            "landing_error_m": 0.3 - offset,
            "apex_error_m": 0.1 - offset,
            "ready_pose_error": 0.05,
        },
    }
    if is_lab:
        episode["lab_diagnostics"].update(
            {
                "raw_latent_saturation": 0.0,
                "lab_state_ood_fraction": 0.0,
            }
        )
        episode["naturalness"] = {
            "body_relative_deviation_to_prior": 0.1 - offset,
            "right_hand_site_rmse_to_prior_m": 0.1 - offset,
            "racket_position_rmse_to_prior_m": 0.1 - offset,
            "racket_rotation_rmse_to_prior_rad": 0.1 - offset,
        }
    return episode


def _report(
    tmp_path: Path,
    *,
    family: str,
    latent_fingerprint: str | None,
    offset: float,
    shared: dict[str, Path],
) -> Path:
    branch = tmp_path / f"stage3_{family}"
    branch.mkdir()
    payload = branch / "policy.npz"
    metadata = branch / "policy.json"
    train_report = branch / "train_report.json"
    payload.write_bytes(f"policy-{family}".encode())
    metadata.write_text(json.dumps({"config": {"seed": 0}}), encoding="utf-8")
    train_report.write_text(json.dumps({"passed": True}), encoding="utf-8")
    feed = {"sample_fingerprints": ["feed-0", "feed-1"]}
    action_family = "full_354" if family == "best_direct" else "fixed_synergy"
    is_lab = action_family == "fixed_synergy"
    environment_abi = {
        "scene_sha256": _sha(shared["scene"]),
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
    report = {
        "schema_version": "incoming_shuttle_hit_evaluate_v3",
        "runner_stage": "evaluate",
        "checkpoint": str(payload),
        "evaluation_seed": 123,
        "episodes": [_episode(0, offset, is_lab=is_lab), _episode(1, offset, is_lab=is_lab)],
        "action_family": action_family,
        "lab_metrics_applicable": is_lab,
        "evaluated_feed_count": 2,
        "required_heldout_feed_count": 2,
        "mean_return": 10.0 + offset,
        "no_fall_rate": 1.0,
        "hit_rate": 1.0,
        "crossed_net_rate": 1.0,
        "opponent_back_landing_rate": 1.0,
        "mean_contact_racket_head_speed_m_s": 10.0 + offset,
        "mean_net_clearance_m": 0.5 + offset,
        "impact_position_error_m": 0.05 - offset,
        "center_hit_rate": 1.0,
        "impact_timing_mae_s": 0.03 - offset,
        "stringbed_normal_error_rad": 0.1 - offset,
        "racket_linear_velocity_rmse_m_s": 0.5 - offset,
        "racket_angular_velocity_rmse_rad_s": 2.0 - offset,
        "landing_rmse_m": 0.3 - offset,
        "apex_mae_m": 0.1 - offset,
        "recovery_ready_rate": 1.0,
        "normalized_control_energy": 0.2 - offset,
        "body_action_saturation_fraction": 0.0,
        "full_action_saturation_fraction": 0.0,
        "control_manifest": {
            "schema_version": (
                "stage3_lab_control_v1"
                if is_lab
                else "incoming_hit_direct_action_impact_recovery_v2"
            ),
            "latent_checkpoint_fingerprint": latent_fingerprint,
            "control_hash": f"control-{family}",
            "environment_abi": environment_abi,
            "racket_attachment": {"attachment_hash": "9" * 64},
        },
        "training_feed_manifest": feed,
        "evaluation_feed_manifest": feed,
        "promotion_gates": {"all_metrics": True, "artifact_binding_verified": True},
        "promotion_thresholds": {"artifact_binding_verified": 1.0},
        "artifact_binding_verified": 1.0,
        "passed": True,
    }
    binding = {
        "schema_version": "incoming_hit_evaluation_artifact_binding_v3",
        "checkpoint_payload_path": str(payload.resolve()),
        "checkpoint_payload_sha256": _sha(payload),
        "checkpoint_metadata_path": str(metadata.resolve()),
        "checkpoint_metadata_sha256": _sha(metadata),
        "action_family": action_family,
        "latent_checkpoint_fingerprint": latent_fingerprint,
        "spec_path": str(shared["spec"].resolve()),
        "spec_sha256": _sha(shared["spec"]),
        "scene_path": str(shared["scene"].resolve()),
        "scene_sha256": _sha(shared["scene"]),
        "train_report_path": str(train_report.resolve()),
        "train_report_sha256": _sha(train_report),
        "training_control_hash": f"control-{family}",
        "evaluation_control_hash": f"control-{family}",
        "policy_abi_hash": f"abi-{family}",
        "training_feed_manifest_sha256": _canonical(feed),
        "evaluation_feed_manifest_sha256": _canonical(feed),
        "training_target_path": str(shared["train_target"].resolve()),
        "training_target_file_sha256": _sha(shared["train_target"]),
        "training_target_bank_sha256": "1" * 64,
        "training_target_source_fingerprint": "2" * 64,
        "evaluation_target_path": str(shared["eval_target"].resolve()),
        "evaluation_target_file_sha256": _sha(shared["eval_target"]),
        "evaluation_target_bank_sha256": "3" * 64,
        "evaluation_target_source_fingerprint": "4" * 64,
        "training_seed": 0,
        "evaluation_seed": 123,
        "checkpoint_env_steps": 30_000_000,
        "checkpoint_task_curriculum_max_stage": "C7_recovery",
        "checkpoint_task_curriculum_complete": True,
        "evaluation_content_sha256": _stage3_evaluation_content_sha256(report),
        "verified": True,
    }
    binding["binding_sha256"] = _canonical(binding)
    report["artifact_binding"] = binding
    path = branch / "evaluate_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_paired_comparison_seals_shared_feed_seed_and_selection(tmp_path):
    manifest, fingerprints = _selection(tmp_path)
    spec = tmp_path / "spec.yaml"
    scene = tmp_path / "scene.xml"
    train_target = tmp_path / "train_target.json"
    eval_target = tmp_path / "eval_target.json"
    spec.write_text("runner: paired\n", encoding="utf-8")
    scene.write_text("<mujoco/>\n", encoding="utf-8")
    train_target.write_text(
        json.dumps({"bank_sha256": "1" * 64, "source_fingerprint": "2" * 64}),
        encoding="utf-8",
    )
    eval_target.write_text(
        json.dumps({"bank_sha256": "3" * 64, "source_fingerprint": "4" * 64}),
        encoding="utf-8",
    )
    shared = {
        "spec": spec,
        "scene": scene,
        "train_target": train_target,
        "eval_target": eval_target,
    }
    direct = _report(
        tmp_path,
        family="best_direct",
        latent_fingerprint=None,
        offset=0.0,
        shared=shared,
    )
    synergy = _report(
        tmp_path,
        family="best_synergy",
        latent_fingerprint=fingerprints["best_synergy"],
        offset=0.01,
        shared=shared,
    )
    result = build_paired_comparison(
        direct_report_path=direct,
        synergy_report_path=synergy,
        selection_manifest_path=manifest,
        bootstrap_samples=100,
    )
    assert result["passed"] is True
    assert result["shared_protocol"]["training_seed"] == 0
    assert result["shared_protocol"]["evaluation_seed"] == 123
    assert result["paired_metrics"]["mean_return"]["synergy_improvement"] > 0.0
    assert result["branch_identities"]["best_direct"]["action_family"] == "full_354"
    assert result["branch_identities"]["best_direct"]["latent_checkpoint_fingerprint"] is None
    assert result["branch_identities"]["best_synergy"]["action_family"] == "fixed_synergy"
    assert result["latent_selection"]["scope"] == "fixed_synergy_branch_only"
    assert result["selected_policy_for_emg"]["family"] == "best_synergy"
    paired_path = tmp_path / "stage3_paired_comparison.json"
    paired_path.write_text(json.dumps(result), encoding="utf-8")
    evidence = load_paired_policy_evidence(
        paired_path,
        stage3_checkpoint_payload_sha256=result["selected_policy_for_emg"]["stage3_checkpoint_payload_sha256"],
    )
    assert evidence.decoder_type == "synergy_residual"
    assert evidence.policy_checkpoint_fingerprint == fingerprints["best_synergy"]
    assert evidence.event_reference_fingerprint == "4" * 64

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from environment.overall_environment.src.stage3_target_bank_v2 import (
    DesiredImpactTarget,
    build_target_bank,
    load_target_bank,
    load_targets_jsonl,
    save_target_bank,
    source_fingerprint_from_event_metrics,
    target_arrays,
)
from environment.overall_environment.src.stage3_task_curriculum_v2 import (
    canonical_stage3_v2_curriculum,
    stage_for_steps,
)
from musclemimic.badminton.racket_mass_curriculum import (
    _fingerprint,
    canonical_racket_mass_stages,
    curriculum_plan,
    validate_curriculum_plan,
    validate_racket_physics_manifest,
)
from musclemimic.badminton.scripts.run_incoming_shuttle_hit import (
    _stage3_evaluation_summary,
    _validate_checkpoint_latent_dim,
)


def _control_manifest_env(tmp_path, *, task_profile: str, target_sha256: str):
    from environment.overall_environment.src.incoming_shuttle_hit_env import (
        IncomingShuttleHitEnv,
    )

    scene = tmp_path / f"{task_profile}-{target_sha256[:4]}.xml"
    scene.write_text("<mujoco/>\n", encoding="utf-8")
    env = IncomingShuttleHitEnv.__new__(IncomingShuttleHitEnv)
    env.lab_controller = SimpleNamespace(
        control_manifest={
            "schema_version": "stage3_lab_control_v1",
            "latent_checkpoint_fingerprint": "a" * 64,
        }
    )
    env.lab_state_builder = SimpleNamespace(schema_hash="state-schema")
    env.model = SimpleNamespace(nu=416)
    env.xml_path = scene
    env.filter_finger_observation = True
    env._effective_ctrlrange_hash = "ctrlrange"
    env.control_substeps = 10
    env.max_episode_steps = 420
    env.reward_weights = {"effort": 0.01}
    env.player_half_sign = 1
    env.singles = True
    env.terminate_on_body_fall = True
    env.swing_duration_s = 1.2
    env.contact_phase = 0.55
    env.curriculum = None
    env.task_profile = task_profile
    env.impact_target_bank = SimpleNamespace(bank_sha256=target_sha256)
    env.recovery_horizon_steps = 60
    env.task_curriculum_stage = "C7_recovery"
    env._control_manifest_cache = None
    return env


def _target(feed: str = "a" * 64) -> DesiredImpactTarget:
    return DesiredImpactTarget(
        feed_fingerprint=feed,
        impact_position_world=(-2.7, 0.2, 1.9),
        impact_time_s=0.75,
        stringbed_normal_world=(1.0, 0.0, 0.0),
        racket_linear_velocity_world=(8.0, 0.0, 2.0),
        racket_angular_velocity_world=(0.0, 5.0, 0.0),
        landing_target_xy=(5.8, 0.0),
        apex_height_m=3.2,
        recovery_horizon_steps=60,
    )


def test_target_bank_roundtrip_and_exact_feed_alignment(tmp_path):
    bank = build_target_bank([_target()], source_fingerprint="b" * 64)
    path = save_target_bank(tmp_path / "bank.json", bank)
    loaded = load_target_bank(path, expected_feed_fingerprints=["a" * 64])
    assert loaded.bank_sha256 == bank.bank_sha256
    assert target_arrays(loaded)["impact_position_world"].shape == (1, 3)
    with pytest.raises(ValueError, match="exact feed-bank order"):
        load_target_bank(path, expected_feed_fingerprints=["c" * 64])


def test_target_jsonl_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(f'{{"feed_fingerprint":"{"a" * 64}","feed_fingerprint":"{"b" * 64}"}}\n')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_targets_jsonl(path)


def test_target_source_is_derived_from_bound_event_metrics(tmp_path):
    payload = {
        "schema_version": "event_reference_promotion_metrics_v1",
        "artifact_binding_verified": 1.0,
        "event_bank_binding_verified": 1.0,
        "train_reference_set_fingerprint": "a" * 64,
        "validation_reference_set_fingerprint": "b" * 64,
    }
    payload["metrics_fingerprint"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    path = tmp_path / "event_metrics.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    fingerprint, binding = source_fingerprint_from_event_metrics(path, split="train")
    assert fingerprint == "a" * 64
    assert binding["reference_split"] == "train"
    payload["train_reference_set_fingerprint"] = "c" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint is stale"):
        source_fingerprint_from_event_metrics(path, split="train")


def test_curriculum_and_mass_plans_are_versioned_and_ordered():
    stages = canonical_stage3_v2_curriculum()
    assert [stage.name for stage in stages] == [
        f"C{i}_{suffix}"
        for i, suffix in enumerate(
            (
                "ready_pose",
                "static_center_time",
                "static_normal",
                "static_velocity",
                "deterministic_feed",
                "feed_jitter",
                "full_flight",
                "recovery",
            )
        )
    ]
    assert stage_for_steps(10**9) == stages[-1]
    masses = canonical_racket_mass_stages()
    assert [stage.mass_scale for stage in masses] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert validate_curriculum_plan(curriculum_plan())["legacy_stage2_unchanged"] is True


def test_stage3_v2_derives_selected_latent_dimension_from_checkpoint():
    for latent_dim in (2, 4, 8, 16, 32):
        assert _validate_checkpoint_latent_dim({"expected_latent_dim": None}, latent_dim) == latent_dim
    assert _validate_checkpoint_latent_dim({"latent_dim": 16}, 16) == 16
    with pytest.raises(ValueError, match="expected_latent_dim=16 != checkpoint=4"):
        _validate_checkpoint_latent_dim({"expected_latent_dim": 16}, 4)


def test_legacy_control_manifest_hash_is_unchanged_and_v2_separates_policy_abi(
    tmp_path,
    monkeypatch,
):
    import environment.overall_environment.src.stage3_lab as stage3_lab
    from environment.overall_environment.src.incoming_shuttle_hit_env import (
        IMPACT_RECOVERY_PROFILE,
        LEGACY_PROFILE,
    )

    attachment = {"schema_version": "stage3_attachment_v1", "attachment_hash": "attachment"}
    monkeypatch.setattr(stage3_lab, "stage3_attachment_report", lambda *_args, **_kwargs: attachment)
    legacy = _control_manifest_env(tmp_path, task_profile=LEGACY_PROFILE, target_sha256="b" * 64)
    manifest = legacy.control_manifest
    assert "policy_abi_hash" not in manifest
    expected = {
        "schema_version": "stage3_lab_control_v1",
        "latent_checkpoint_fingerprint": "a" * 64,
        "lab_state_schema_hash": "state-schema",
        "racket_attachment": attachment,
        "filter_finger_observation": True,
        "environment_abi": {
            "schema_version": "incoming_hit_environment_v1",
            "scene_sha256": hashlib.sha256(legacy.xml_path.read_bytes()).hexdigest(),
            "effective_ctrlrange_hash": "ctrlrange",
            "full_action_size": 416,
            "control_substeps": 10,
            "max_episode_steps": 420,
            "reward_weights": {"effort": 0.01},
            "player_half_sign": 1,
            "singles": True,
            "terminate_on_body_fall": True,
            "swing_duration_s": 1.2,
            "contact_phase": 0.55,
        },
        "curriculum": None,
    }
    expected["control_hash"] = hashlib.sha256(
        json.dumps(expected, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert manifest == expected

    first = _control_manifest_env(
        tmp_path,
        task_profile=IMPACT_RECOVERY_PROFILE,
        target_sha256="c" * 64,
    ).control_manifest
    second = _control_manifest_env(
        tmp_path,
        task_profile=IMPACT_RECOVERY_PROFILE,
        target_sha256="d" * 64,
    ).control_manifest
    assert first["control_hash"] != second["control_hash"]
    assert first["policy_abi_hash"] == second["policy_abi_hash"]


def test_racket_physics_manifest_contract_rejects_tampering(tmp_path):
    asset = tmp_path / "racket.xml"
    asset.write_text("<mujoco/>", encoding="utf-8")
    payload = {
        "schema_version": "racket_mass_physics_manifest_v1",
        "stage": "mass_025",
        "mass_scale": 0.25,
        "racket_body_name": "racket_racket",
        "racket_mass_kg": 0.0225,
        "racket_center_of_mass_m": [0.0, 0.0, 0.1],
        "racket_inertia_tensor_kg_m2": [
            [0.001, 0.0, 0.0],
            [0.0, 0.002, 0.0],
            [0.0, 0.0, 0.003],
        ],
        "attachment_transform": {
            "parent_body": "thirdmc_r",
            "translation_m": [0.0, 0.0, 0.0],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "joint_count": 0,
        },
        "weld_parameters": {
            "kind": "jointless_spec_attach",
            "count": 0,
            "solref": [],
            "solimp": [],
        },
        "racket_asset": {
            "path": str(asset),
            "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
        },
        "compiled_model_sha256": "c" * 64,
    }
    payload["manifest_sha256"] = _fingerprint(payload)
    assert validate_racket_physics_manifest(
        payload,
        expected_stage="mass_025",
        verify_compiled_model=False,
    )["racket_mass_kg"] == pytest.approx(0.0225)
    payload["racket_mass_kg"] = 0.03
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_racket_physics_manifest(
            payload,
            expected_stage="mass_025",
            verify_compiled_model=False,
        )


def test_stage3_v2_evaluation_summary_exposes_gate_metrics():
    diagnostics = {
        "control_finite": 1.0,
        "body_action_saturation_fraction": 0.0,
        "full_action_saturation_fraction": 0.0,
        "normalized_control_energy": 0.1,
        "raw_latent_saturation": 0.0,
        "lab_state_ood_fraction": 0.0,
        "lab_state_unclipped_z_rms": 0.1,
    }
    naturalness = {
        "body_relative_deviation_to_prior": 0.0,
        "right_hand_site_rmse_to_prior_m": 0.0,
        "right_hand_site_relative_deviation_to_prior": 0.0,
        "racket_position_rmse_to_prior_m": 0.0,
        "racket_position_relative_deviation_to_prior": 0.0,
        "racket_rotation_rmse_to_prior_rad": 0.0,
        "racket_rotation_relative_deviation_to_prior": 0.0,
    }
    v2 = {
        "impact_position_error_m": 0.05,
        "impact_rho2": 0.1,
        "impact_timing_error_s": 0.03,
        "stringbed_normal_error_rad": 0.1,
        "racket_linear_velocity_error_m_s": 0.5,
        "racket_angular_velocity_error_rad_s": 2.0,
        "landing_error_m": 0.3,
        "apex_error_m": 0.1,
        "ready_pose_error": 0.05,
    }
    results = [
        {
            "hit": True,
            "crossed_net": True,
            "body_fall": False,
            "landing_region": "opponent_back",
            "contact_racket_head_speed_m_s": 10.0,
            "net_clearance_m": 0.5,
            "min_root_height_m": 1.0,
            "max_attachment_translation_drift_m": 0.0,
            "max_attachment_rotation_drift_rad": 0.0,
            "lab_diagnostics": diagnostics,
            "naturalness": naturalness,
            "stage3_v2_metrics": v2,
            "recovery_complete": True,
        }
        for _ in range(2)
    ]
    summary = _stage3_evaluation_summary(
        results,
        gate_config={"min_racket_head_speed_m_s": 8.0},
        required_feed_count=2,
        lab_state_ood_values=[0.0, 0.0],
        prior_direct_baseline={"prior_vs_direct_body_racket_relative_degradation": 0.0},
        task_profile="impact_recovery_v2",
    )
    assert summary["impact_position_error_m"] == pytest.approx(0.05)
    assert summary["center_hit_rate"] == 1.0
    assert summary["recovery_ready_rate"] == 1.0

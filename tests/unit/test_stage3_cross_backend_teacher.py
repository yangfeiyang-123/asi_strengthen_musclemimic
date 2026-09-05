from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from environment.overall_environment.src.train_incoming_hit_mjx import (
    load_quality_teacher_dataset,
)
from scripts.audit_cem_candidate_cpu import (
    _parameter_sha256,
    _validated_trace_path,
)
from scripts.seal_cross_backend_hit_teacher import seal_cross_backend_teacher


def _rewrite_source_backend(
    source: dict,
    *,
    backend: str,
    verification_context: str,
) -> dict:
    rewritten = json.loads(json.dumps(source))
    rewritten["contract"]["mjx_impl"] = backend
    rewritten["contract"]["verification_context_semantics"] = (
        verification_context
    )
    rewritten["teacher_trace"]["verification_context_semantics"] = (
        verification_context
    )
    return rewritten


def test_cpu_audit_uses_cem_float32_parameter_hash_and_npz_path(
    tmp_path: Path,
) -> None:
    parameters = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
    expected = hashlib.sha256(parameters.tobytes(order="C")).hexdigest()

    assert _parameter_sha256(parameters.tolist()) == expected
    assert _validated_trace_path(tmp_path / "trace.npz").suffix == ".npz"
    with pytest.raises(ValueError, match=r"end in \.npz"):
        _validated_trace_path(tmp_path / "trace")


def test_cpu_quality_sealer_preserves_cross_backend_provenance(tmp_path: Path) -> None:
    n = 32
    event_index = 20
    event = np.zeros((n,), dtype=bool)
    event[event_index] = True
    velocity = np.zeros((n, 3), dtype=np.float32)
    velocity[event_index] = [3.0, 0.0, 1.0]
    stringbed = np.zeros((n, 3), dtype=np.float32)
    stringbed[:, 2] = 2.80
    stringbed[event_index, 2] = 2.75
    right_arm = np.zeros((n, 7, 3), dtype=np.float32)
    right_arm[:, -1, 2] = 2.70
    right_arm[event_index, -1, 2] = 2.65
    cpu_trace = tmp_path / "teacher_trajectory_cpu_audit.npz"
    np.savez_compressed(
        cpu_trace,
        observation_normalized=np.zeros((n, 3), dtype=np.float32),
        correction_raw=np.full((n, 2), 0.25, dtype=np.float32),
        correction_window=np.ones((n,), dtype=np.float32),
        time_to_intercept_s=np.linspace(0.31, 0.0, n, dtype=np.float32),
        event_rebound=event,
        shuttle_velocity=velocity,
        stringbed_position=stringbed,
        right_arm_body_position_xyz_m=right_arm,
        body_fall=np.zeros((n,), dtype=bool),
        selected_action_indices=np.asarray([0, 1], dtype=np.int32),
        physical_scales=np.asarray([0.1, 0.2], dtype=np.float32),
        feed_fingerprint=np.asarray("feed"),
        swing_phase_advance_s=np.asarray(0.18, dtype=np.float32),
        outgoing_velocity_semantics=np.asarray(
            "post_control_step_after_all_physics_substeps"
        ),
        event_rebound_contact_semantics=np.asarray(
            "single_event_impulse_with_stringbed_force_suppressed_during_cooldown_v2"
        ),
    )
    source_report = {
        "schema_version": "stage3_single_feed_mjx_cem_report_v3",
        "passed": True,
        "mjx_teacher_passed": True,
        "contract": {
            "contract_sha256": "b" * 64,
            "source_checkpoint_sha256": "a" * 64,
            "feed_fingerprint": "feed",
            "swing_phase_advance_s": 0.18,
            "swing_phase_timing_semantics": (
                "frozen_base_swing_phase_advance_applied_identically_to_search_"
                "backend_and_cpu_replays"
            ),
            "mjx_impl": "warp",
            "outgoing_velocity_semantics": (
                "post_control_step_after_all_physics_substeps"
            ),
            "event_rebound_contact_semantics": (
                "single_event_impulse_with_stringbed_force_suppressed_during_cooldown_v2"
            ),
            "min_replica_fraction": 0.6666667,
            "replicas_per_candidate": 3,
            "required_replica_count": 2,
            "population": 8,
            "verification_repeats": 2,
            "verification_context_semantics": (
                "same_candidate_relocated_across_deterministic_warp_batch_lanes"
            ),
            "return_quality_search_margin": {
                "semantics": "same_replica_training_backend_margin_gate",
                "min_outgoing_z_m_s": 0.8,
                "min_forward_m_s": 2.5,
            },
            "high_region_contact": {
                "semantics": "soft_window_teacher_gate_not_exact_apex",
                "max_stringbed_height_deficit_m": 0.10,
                "max_hand_height_deficit_m": 0.10,
            },
        },
        "best_search_metrics": {
            "teacher_success": True,
            "teacher_success_rate": 2.0 / 3.0,
            "high_region_contact": True,
            "no_fall": True,
        },
        "verified_metrics": {
            "teacher_success": True,
            "teacher_success_rate": 2.0 / 3.0,
            "event_rebound": True,
            "event_rebound_rate": 2.0 / 3.0,
            "high_region_contact": True,
            "no_fall": True,
            "return_quality": True,
            "return_quality_rate": 2.0 / 3.0,
            "positive_outgoing_z_rate": 2.0 / 3.0,
            "positive_outgoing_forward_rate": 2.0 / 3.0,
            "outgoing_z_m_s": 0.8,
            "outgoing_forward_m_s": 3.0,
        },
        "teacher_trace": {
            "verification_context_semantics": (
                "same_candidate_relocated_across_deterministic_warp_batch_lanes"
            ),
            "verification_group_indices": [1, 5],
            "selected_batch_group": 1,
        },
        "cpu_replay_event_equivalent": True,
        "cpu_replay_audit": {
            "hit": True,
            "event_rebound": True,
            "body_fall": False,
            "feed_fingerprint": "feed",
            "swing_phase_advance_s": 0.18,
            "trace_path": str(cpu_trace.resolve()),
            "trace_sha256": hashlib.sha256(cpu_trace.read_bytes()).hexdigest(),
        },
    }
    source_path = tmp_path / "source_cem_report.json"
    source_path.write_text(json.dumps(source_report), encoding="utf-8")
    output_dir = tmp_path / "sealed"

    report = seal_cross_backend_teacher(
        source_path,
        output_dir,
        source_base_phase_advance_s=0.58,
        runtime_base_phase_advance_s=0.18,
    )

    assert report["schema_version"] == "stage3_cross_backend_quality_teacher_report_v3"
    assert report["passed"] is True
    assert report["verification_source"] == (
        "warp_training_backend_plus_independent_cpu_mujoco_quality_replay"
    )
    assert report["cross_backend_evidence"]["independent_mjx_replay_metrics"][
        "teacher_success"
    ] is True
    assert report["base_timing_transfer"]["source_phase_advance_s"] == 0.58
    assert report["base_timing_transfer"]["runtime_phase_advance_s"] == 0.18
    dataset = load_quality_teacher_dataset(
        output_dir / "teacher_trajectory_cpu_quality.npz",
        selected_action_indices=(0, 1),
        correction_physical_scales=(0.1, 0.2),
        source_checkpoint_sha256="a" * 64,
    )
    assert dataset.observation_normalized.shape == (32, 3)
    assert dataset.binding["verification_source"] == (
        "warp_training_backend_plus_independent_cpu_mujoco_quality_replay"
    )
    assert dataset.binding["training_backend_quality_verified"] is True
    assert dataset.binding["training_backend_outgoing_z_m_s"] == 0.8
    assert dataset.binding["training_backend_verification_group_indices"] == [1, 5]
    assert dataset.binding["base_timing_transfer"]["cpu_quality_verified"] is True

    # Version 1 was allowed to rewrite CPU quality into verified_metrics even
    # when the training backend returned the shuttle downward.  It must remain
    # permanently incompatible with the production loader.
    sealed_report_path = output_dir / "cem_report.json"
    sealed_report = json.loads(sealed_report_path.read_text(encoding="utf-8"))
    sealed_report["schema_version"] = "stage3_cross_backend_quality_teacher_report_v1"
    sealed_report_path.write_text(json.dumps(sealed_report), encoding="utf-8")
    with pytest.raises(ValueError, match="schema is incompatible"):
        load_quality_teacher_dataset(
            output_dir / "teacher_trajectory_cpu_quality.npz",
            selected_action_indices=(0, 1),
            correction_physical_scales=(0.1, 0.2),
            source_checkpoint_sha256="a" * 64,
        )
    sealed_report_path.write_text(json.dumps(report), encoding="utf-8")

    tampered_lanes = json.loads(json.dumps(report))
    tampered_lanes["cross_backend_evidence"]["verification_group_indices"] = [1, 1]
    sealed_report_path.write_text(json.dumps(tampered_lanes), encoding="utf-8")
    with pytest.raises(ValueError, match="actual training backend"):
        load_quality_teacher_dataset(
            output_dir / "teacher_trajectory_cpu_quality.npz",
            selected_action_indices=(0, 1),
            correction_physical_scales=(0.1, 0.2),
            source_checkpoint_sha256="a" * 64,
        )
    sealed_report_path.write_text(json.dumps(report), encoding="utf-8")

    # The loader independently checks the embedded training-backend evidence;
    # changing only the accepted CPU-facing summary cannot hide a downward
    # Warp replay behind a syntactically valid v3 label.
    tampered_report = json.loads(json.dumps(report))
    tampered_training_metrics = tampered_report["cross_backend_evidence"][
        "independent_mjx_replay_metrics"
    ]
    tampered_training_metrics.update(
        {
            "teacher_success": False,
            "teacher_success_rate": 0.0,
            "return_quality": False,
            "return_quality_rate": 0.0,
            "positive_outgoing_z_rate": 0.0,
            "outgoing_z_m_s": -2.5,
        }
    )
    sealed_report_path.write_text(json.dumps(tampered_report), encoding="utf-8")
    with pytest.raises(ValueError, match="actual training backend"):
        load_quality_teacher_dataset(
            output_dir / "teacher_trajectory_cpu_quality.npz",
            selected_action_indices=(0, 1),
            correction_physical_scales=(0.1, 0.2),
            source_checkpoint_sha256="a" * 64,
        )
    sealed_report_path.write_text(json.dumps(report), encoding="utf-8")

    # A CPU-only upward return must never be sealed when the actual vectorized
    # training backend replays the same correction as a downward return.
    source_report["passed"] = False
    source_report["mjx_teacher_passed"] = False
    source_report["verified_metrics"].update(
        {
            "teacher_success": False,
            "teacher_success_rate": 0.0,
            "return_quality": False,
            "return_quality_rate": 0.0,
            "positive_outgoing_z_rate": 0.0,
            "outgoing_z_m_s": -2.5,
        }
    )
    failed_source = tmp_path / "source_cem_report_cpu_only.json"
    failed_source.write_text(json.dumps(source_report), encoding="utf-8")
    with pytest.raises(ValueError, match="training-backend"):
        seal_cross_backend_teacher(failed_source, tmp_path / "rejected")

    same_lane_source = json.loads(json.dumps(source_report))
    same_lane_source["passed"] = True
    same_lane_source["mjx_teacher_passed"] = True
    same_lane_source["verified_metrics"].update(
        {
            "teacher_success": True,
            "teacher_success_rate": 2.0 / 3.0,
            "return_quality": True,
            "return_quality_rate": 2.0 / 3.0,
            "positive_outgoing_z_rate": 2.0 / 3.0,
            "outgoing_z_m_s": 0.8,
        }
    )
    same_lane_source["teacher_trace"]["verification_group_indices"] = [1, 1]
    same_lane_path = tmp_path / "source_cem_report_same_lane.json"
    same_lane_path.write_text(json.dumps(same_lane_source), encoding="utf-8")
    with pytest.raises(ValueError, match="distinct Warp batch lanes"):
        seal_cross_backend_teacher(same_lane_path, tmp_path / "same_lane_rejected")

    jax_context = (
        "same_candidate_relocated_across_deterministic_"
        "standard_mjx_jax_batch_lanes"
    )
    jax_source = _rewrite_source_backend(
        source_report,
        backend="jax",
        verification_context=jax_context,
    )
    jax_source["passed"] = True
    jax_source["mjx_teacher_passed"] = True
    jax_source["verified_metrics"].update(
        {
            "teacher_success": True,
            "teacher_success_rate": 2.0 / 3.0,
            "return_quality": True,
            "return_quality_rate": 2.0 / 3.0,
            "positive_outgoing_z_rate": 2.0 / 3.0,
            "outgoing_z_m_s": 0.8,
        }
    )
    jax_source_path = tmp_path / "source_cem_report_jax.json"
    jax_source_path.write_text(json.dumps(jax_source), encoding="utf-8")
    jax_report = seal_cross_backend_teacher(
        jax_source_path,
        tmp_path / "sealed_jax",
    )
    assert jax_report["verification_source"] == (
        "standard_mjx_jax_training_backend_plus_independent_"
        "cpu_mujoco_quality_replay"
    )
    assert jax_report["cross_backend_evidence"]["training_backend"] == "jax"
    jax_dataset = load_quality_teacher_dataset(
        tmp_path / "sealed_jax" / "teacher_trajectory_cpu_quality.npz",
        selected_action_indices=(0, 1),
        correction_physical_scales=(0.1, 0.2),
        source_checkpoint_sha256="a" * 64,
    )
    assert jax_dataset.binding["training_backend"] == "jax"
    assert (
        jax_dataset.binding[
            "training_backend_verification_context_semantics"
        ]
        == jax_context
    )


def test_cpu_certified_exploration_prior_is_explicit_and_non_promotable(
    tmp_path: Path,
) -> None:
    n = 32
    event_index = 20
    event = np.zeros((n,), dtype=bool)
    event[event_index] = True
    outgoing = np.zeros((n, 3), dtype=np.float32)
    outgoing[event_index] = [4.0, 0.0, 1.1]
    trajectory = tmp_path / "teacher_trajectory_mjx.npz"
    np.savez_compressed(
        trajectory,
        observation_normalized=np.zeros((n, 3), dtype=np.float32),
        correction_raw=np.full((n, 2), 0.25, dtype=np.float32),
        correction_window=np.ones((n,), dtype=np.float32),
        time_to_intercept_s=np.linspace(0.31, 0.0, n, dtype=np.float32),
        event_rebound=event,
        outgoing_shuttle_velocity_xyz_m_s=outgoing,
        selected_action_indices=np.asarray([0, 1], dtype=np.int32),
        physical_scales=np.asarray([0.1, 0.2], dtype=np.float32),
        feed_fingerprint=np.asarray("feed"),
        swing_phase_advance_s=np.asarray(0.28, dtype=np.float32),
        source_checkpoint_sha256=np.asarray("a" * 64),
        search_contract_sha256=np.asarray("b" * 64),
        outgoing_velocity_semantics=np.asarray(
            "post_control_step_after_all_physics_substeps"
        ),
        event_rebound_contact_semantics=np.asarray(
            "single_event_impulse_with_stringbed_force_suppressed_during_cooldown_v2"
        ),
        trace_schema_version=np.asarray("stage3_cem_teacher_trajectory_v3"),
    )
    report = {
        "schema_version": "stage3_single_feed_mjx_cem_report_v3",
        "passed": False,
        "mjx_teacher_passed": False,
        "cpu_replay_passed": True,
        "cpu_replay_event_equivalent": True,
        "contract": {
            "contract_sha256": "b" * 64,
            "source_checkpoint_sha256": "a" * 64,
            "feed_fingerprint": "feed",
            "configured_swing_phase_advance_s": 0.18,
            "swing_phase_advance_s": 0.28,
            "swing_phase_timing_semantics": (
                "frozen_base_swing_phase_advance_applied_identically_to_search_"
                "backend_and_cpu_replays"
            ),
            "mjx_impl": "warp",
            "outgoing_velocity_semantics": (
                "post_control_step_after_all_physics_substeps"
            ),
            "event_rebound_contact_semantics": (
                "single_event_impulse_with_stringbed_force_suppressed_during_cooldown_v2"
            ),
            "high_region_contact": {
                "semantics": "soft_window_teacher_gate_not_exact_apex",
                "max_stringbed_height_deficit_m": 0.10,
                "max_hand_height_deficit_m": 0.10,
            },
        },
        "best_search_metrics": {
            "teacher_success": True,
            "high_region_contact": True,
            "no_fall": True,
        },
        "verified_metrics": {
            "teacher_success": False,
            "teacher_success_rate": 0.05,
            "return_quality": False,
            "return_quality_rate": 0.05,
            "positive_outgoing_z_rate": 0.05,
            "positive_outgoing_forward_rate": 0.05,
            "high_region_contact_rate": 0.15,
            "no_fall_rate": 1.0,
        },
        "teacher_trace": {
            "trace_path": str(trajectory.resolve()),
            "trace_sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest(),
            "selected_replica_metrics": {
                "teacher_success": True,
                "high_region_contact": True,
            },
        },
        "cpu_replay_audit": {
            "hit": True,
            "event_rebound": True,
            "body_fall": False,
            "feed_fingerprint": "feed",
            "swing_phase_advance_s": 0.28,
        },
        "cpu_gated_best_audit": {
            "cpu_quality_passed": True,
            "hit": True,
            "event_rebound": True,
            "high_region_contact": True,
            "body_fall": False,
            "feed_fingerprint": "feed",
            "swing_phase_advance_s": 0.28,
            "outgoing_z_m_s": 1.1,
            "outgoing_forward_m_s": 4.0,
        },
    }
    report_path = tmp_path / "cem_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="robust return-success"):
        load_quality_teacher_dataset(
            trajectory,
            selected_action_indices=(0, 1),
            correction_physical_scales=(0.1, 0.2),
            source_checkpoint_sha256="a" * 64,
        )

    dataset = load_quality_teacher_dataset(
        trajectory,
        selected_action_indices=(0, 1),
        correction_physical_scales=(0.1, 0.2),
        source_checkpoint_sha256="a" * 64,
        source_base_phase_advance_s=0.58,
        allow_cpu_certified_exploration_prior=True,
    )
    assert dataset.binding["schema_version"] == (
        "stage3_cpu_certified_exploration_prior_binding_v1"
    )
    assert dataset.binding["quality_teacher"] is False
    assert dataset.binding["training_backend_quality_verified"] is False
    assert dataset.binding["training_backend"] == "warp"
    assert dataset.binding["training_backend_observed_teacher_success_rate"] == 0.05
    # Production binds the actual inherited checkpoint timing, not the search
    # spec's stale configured value (0.18 in this fixture).
    assert dataset.binding["base_timing_transfer"]["source_phase_advance_s"] == 0.58
    assert dataset.binding["base_timing_transfer"]["runtime_phase_advance_s"] == 0.28

    tampered = json.loads(json.dumps(report))
    tampered["verified_metrics"]["teacher_success_rate"] = 0.0
    report_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="non-robust"):
        load_quality_teacher_dataset(
            trajectory,
            selected_action_indices=(0, 1),
            correction_physical_scales=(0.1, 0.2),
            source_checkpoint_sha256="a" * 64,
            allow_cpu_certified_exploration_prior=True,
        )

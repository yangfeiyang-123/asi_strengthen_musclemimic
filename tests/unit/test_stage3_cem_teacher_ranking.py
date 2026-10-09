from __future__ import annotations

import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.derive_cem_search_seed import derive_search_seed
from scripts.export_cem_search_seed import (
    export_search_seed,
    export_snapshot_candidate_seed,
)
from scripts.optimize_single_feed_hit_mjx import (
    CPU_ACTOR_INFERENCE_SEMANTICS,
    CPU_AUDIT_TRACE_SCHEMA,
    _aggregate_replica_metrics,
    _build_explicit_cpu_actor_fn,
    _build_search_frontier_challenge_batch,
    _candidate_metrics_improve_frontier,
    _cem_failure_reason,
    _cpu_unqualified_search_improves,
    _cpu_unqualified_search_progress_key,
    _cross_backend_promotion_passes,
    _expand_candidates_across_stratified_lanes,
    _gather_candidate_major_lane_values,
    _inject_coordinate_probe_candidates,
    _inject_search_frontier_copies,
    _json_hash,
    _load_initial_candidate,
    _rank_order,
    _required_replica_count,
    _retain_cpu_search_frontier_mean,
    _stratified_candidate_lane_indices,
    _summarize_cpu_quality_trace,
    _trainable_parameter_mask,
    _verification_group_indices,
    _wandb_iteration_best_payload,
)


def test_unpromoted_iteration_metrics_remain_visible_in_wandb() -> None:
    payload = _wandb_iteration_best_payload(
        {
            "event_rebound": True,
            "outgoing_forward_m_s": np.float32(4.25),
            "teacher_success": False,
            "future_nested_diagnostic": {"ignored": True},
        }
    )

    assert payload == {
        "cem/iteration_best_event_rebound": 1.0,
        "cem/iteration_best_outgoing_forward_m_s": 4.25,
        "cem/iteration_best_teacher_success": 0.0,
    }


def test_cem_failure_reason_distinguishes_backend_and_cpu_rejection() -> None:
    assert _cem_failure_reason(0) == ("no_backend_candidate_passed_strict_teacher_gate")
    assert _cem_failure_reason(3) == ("no_candidate_passed_independent_cpu_quality_gate")


def test_unqualified_search_frontier_keeps_the_best_iteration() -> None:
    metrics = _base_metrics()
    for name in (
        "event_rebound",
        "stringbed_contact",
        "high_region_contact",
    ):
        metrics[name][:] = True
    for name in (
        "event_rebound_rate",
        "stringbed_contact_rate",
        "high_region_contact_rate",
        "return_quality_rate",
    ):
        metrics[name][:] = 1.0
    metrics["outgoing_forward_m_s"][:] = (5.0, 5.5)
    metrics["outgoing_z_m_s"][:] = (2.0, 2.5)
    metrics["predicted_clearance_m"][:] = (-2.0, -1.5)

    reference = {name: values[0].item() for name, values in metrics.items()}
    improved = {name: values[1].item() for name, values in metrics.items()}

    assert _candidate_metrics_improve_frontier(
        None,
        reference,
        metric_names=tuple(metrics),
    )
    assert _candidate_metrics_improve_frontier(
        reference,
        improved,
        metric_names=tuple(metrics),
    )
    assert not _candidate_metrics_improve_frontier(
        improved,
        reference,
        metric_names=tuple(metrics),
    )


def _base_metrics(count: int = 2) -> dict[str, np.ndarray]:
    zeros = np.zeros((count,), dtype=np.float64)
    return {
        "event_rebound": np.zeros((count,), dtype=bool),
        "event_rebound_rate": zeros.copy(),
        "stringbed_contact": np.zeros((count,), dtype=bool),
        "stringbed_contact_rate": zeros.copy(),
        "teacher_success": np.zeros((count,), dtype=bool),
        "teacher_success_rate": zeros.copy(),
        "positive_outgoing_z_rate": zeros.copy(),
        "positive_outgoing_forward_rate": zeros.copy(),
        "return_quality": np.zeros((count,), dtype=bool),
        "return_quality_rate": zeros.copy(),
        "high_region_contact": np.zeros((count,), dtype=bool),
        "high_region_contact_rate": zeros.copy(),
        "soft_high_region_excess_m": zeros.copy(),
        "contact_acquisition_cost_m": np.full((count,), 0.20),
        "min_ball_racket_distance_m": np.full((count,), 0.20),
        "closest_inverse_impact_decomposed_score": zeros.copy(),
        "closest_inverse_impact_normal_alignment": zeros.copy(),
        "closest_inverse_impact_racket_velocity_error_m_s": np.full((count,), 10.0),
        "outgoing_z_m_s": zeros.copy(),
        "outgoing_forward_m_s": zeros.copy(),
        "hit_contact_speed_m_s": zeros.copy(),
        "stringbed_contact_speed_m_s": zeros.copy(),
        "stringbed_contact_closing_speed_m_s": zeros.copy(),
        "predicted_clearance_m": np.full((count,), -1000.0),
        "crossed_net": np.zeros((count,), dtype=bool),
        "opponent_back_landing": np.zeros((count,), dtype=bool),
        "no_fall": np.ones((count,), dtype=bool),
        "no_fall_rate": np.ones((count,), dtype=np.float64),
        "correction_rate_cost": zeros.copy(),
        "correction_rms": zeros.copy(),
    }


def test_cross_backend_promotion_requires_both_quality_gates() -> None:
    assert _cross_backend_promotion_passes(
        {"teacher_success": True},
        {"cpu_quality_passed": True},
    )
    assert not _cross_backend_promotion_passes(
        {"teacher_success": False},
        {"cpu_quality_passed": True},
    )
    assert not _cross_backend_promotion_passes(
        {"teacher_success": True},
        {"cpu_quality_passed": False},
    )
    assert not _cross_backend_promotion_passes(
        {"teacher_success": True},
        None,
    )


def _cpu_progress_audit(**overrides: object) -> dict[str, object]:
    audit: dict[str, object] = {
        "body_fall": False,
        "hit": True,
        "event_rebound": True,
        "pre_event_velocity_consistent": True,
        "event_settled_velocity_consistent": True,
        "high_region_contact": True,
        "cpu_quality_passed": False,
        "crossed_net": False,
        "legal_return": False,
        "outgoing_z_m_s": 0.0,
        "outgoing_forward_m_s": 2.0,
        "predicted_net_clearance_m": -4.0,
        "return_direction_signed_score": 0.25,
        "event_settled_velocity_delta_m_s": 0.05,
    }
    audit.update(overrides)
    return audit


def _cpu_progress_thresholds() -> dict[str, float | bool]:
    return {
        "min_outgoing_z_m_s": 1.0,
        "min_forward_m_s": 4.0,
        "min_predicted_clearance_m": 0.2,
        "min_return_direction_signed_score": 0.65,
    }


def test_cpu_unqualified_guidance_rejects_attractive_duplicate_collision() -> None:
    physical = _cpu_progress_audit()
    duplicate = _cpu_progress_audit(
        event_settled_velocity_consistent=False,
        outgoing_z_m_s=3.0,
        outgoing_forward_m_s=6.0,
        predicted_net_clearance_m=1.0,
        return_direction_signed_score=0.9,
        event_settled_velocity_delta_m_s=6.0,
    )

    assert _cpu_unqualified_search_improves(
        duplicate,
        physical,
        **_cpu_progress_thresholds(),
    )
    assert not _cpu_unqualified_search_improves(
        physical,
        duplicate,
        **_cpu_progress_thresholds(),
    )


def test_cpu_unqualified_guidance_rejects_a_native_collision_before_event() -> None:
    physical = _cpu_progress_audit()
    frame_then_stringbed = _cpu_progress_audit(
        pre_event_velocity_consistent=False,
        cpu_quality_passed=True,
        crossed_net=True,
        legal_return=True,
        outgoing_z_m_s=2.0,
        outgoing_forward_m_s=10.0,
        predicted_net_clearance_m=0.5,
        return_direction_signed_score=0.8,
    )

    assert not _cpu_unqualified_search_improves(
        physical,
        frame_then_stringbed,
        **_cpu_progress_thresholds(),
    )


def test_cpu_unqualified_guidance_requires_a_valid_high_contact_anchor() -> None:
    miss = _cpu_progress_audit(hit=False, event_rebound=False)
    low_contact = _cpu_progress_audit(high_region_contact=False)

    assert not _cpu_unqualified_search_improves(
        None,
        miss,
        **_cpu_progress_thresholds(),
    )
    assert not _cpu_unqualified_search_improves(
        None,
        low_contact,
        **_cpu_progress_thresholds(),
    )


def test_cpu_unqualified_guidance_uses_constraint_progress_after_validity() -> None:
    upward = _cpu_progress_audit(
        outgoing_z_m_s=0.12,
        outgoing_forward_m_s=3.03,
        predicted_net_clearance_m=-3.66,
        return_direction_signed_score=0.294,
    )
    better_clearance = _cpu_progress_audit(
        outgoing_z_m_s=-0.12,
        outgoing_forward_m_s=3.41,
        predicted_net_clearance_m=-3.52,
        return_direction_signed_score=0.319,
    )

    assert _cpu_unqualified_search_progress_key(
        better_clearance,
        **_cpu_progress_thresholds(),
    ) > _cpu_unqualified_search_progress_key(
        upward,
        **_cpu_progress_thresholds(),
    )
    assert _cpu_unqualified_search_improves(
        upward,
        better_clearance,
        **_cpu_progress_thresholds(),
    )


def test_authoritative_real_cross_stops_projected_clearance_from_hiding_direction_progress() -> None:
    better_direction = _cpu_progress_audit(
        crossed_net=True,
        outgoing_z_m_s=2.0,
        outgoing_forward_m_s=8.0,
        predicted_net_clearance_m=-2.0,
        return_direction_signed_score=0.64,
    )
    better_projection = _cpu_progress_audit(
        crossed_net=True,
        outgoing_z_m_s=2.0,
        outgoing_forward_m_s=8.0,
        predicted_net_clearance_m=0.5,
        return_direction_signed_score=0.50,
    )
    thresholds = {
        **_cpu_progress_thresholds(),
        "real_net_cross_authoritative": True,
    }

    assert _cpu_unqualified_search_improves(
        better_projection,
        better_direction,
        **thresholds,
    )
    assert not _cpu_unqualified_search_improves(
        better_direction,
        better_projection,
        **thresholds,
    )


def test_projected_clearance_still_guides_before_an_authoritative_cross() -> None:
    poor_projection = _cpu_progress_audit(
        crossed_net=False,
        outgoing_z_m_s=2.0,
        outgoing_forward_m_s=8.0,
        predicted_net_clearance_m=-2.0,
        return_direction_signed_score=0.64,
    )
    better_projection = _cpu_progress_audit(
        crossed_net=False,
        outgoing_z_m_s=2.0,
        outgoing_forward_m_s=8.0,
        predicted_net_clearance_m=0.5,
        return_direction_signed_score=0.50,
    )

    assert _cpu_unqualified_search_improves(
        poor_projection,
        better_projection,
        **{
            **_cpu_progress_thresholds(),
            "real_net_cross_authoritative": True,
        },
    )


def test_cpu_frontier_remains_mean_after_a_non_improving_iteration() -> None:
    warp_mean = np.asarray([0.7, -0.8, 0.9, -1.0], dtype=np.float32)
    warp_std = np.asarray([0.02, 0.03, 0.04, 0.05], dtype=np.float32)
    cpu_frontier = np.asarray([-0.2, 0.3, -0.4, 0.5], dtype=np.float32)
    initial = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    mask = np.asarray([False, True, True, False])

    retained_mean, retained_std = _retain_cpu_search_frontier_mean(
        warp_mean,
        warp_std,
        frontier_parameters=cpu_frontier,
        initial_parameters=initial,
        trainable_parameter_mask=mask,
        initial_std=0.2,
    )

    np.testing.assert_array_equal(
        retained_mean,
        np.asarray([0.1, 0.3, -0.4, 0.4], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        retained_std,
        np.asarray([0.0, 0.1, 0.1, 0.0], dtype=np.float32),
    )


def test_cpu_frontier_retention_rejects_an_invalid_search_state() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        _retain_cpu_search_frontier_mean(
            np.zeros(2, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            frontier_parameters=np.zeros(2, dtype=np.float32),
            initial_parameters=np.zeros(2, dtype=np.float32),
            trainable_parameter_mask=np.ones(2, dtype=bool),
            initial_std=0.2,
        )


def test_nonrobust_one_off_contact_cannot_beat_reproducible_proximity() -> None:
    metrics = _base_metrics()
    metrics["event_rebound_rate"][0] = 1.0 / 3.0
    metrics["contact_acquisition_cost_m"][:] = (0.14, 0.08)
    metrics["min_ball_racket_distance_m"][:] = (0.09, 0.08)

    assert int(_rank_order(metrics)[-1]) == 1


def test_rounded_two_thirds_fraction_requires_two_of_three() -> None:
    assert _required_replica_count(3, 0.6666667) == 2
    assert _required_replica_count(3, 2.0 / 3.0) == 2
    assert _required_replica_count(3, 0.67) == 3
    assert _required_replica_count(3, 1.0) == 3


def test_distal_synergy_search_freezes_proximal_parameters_at_every_knot() -> None:
    synergies = (
        "shoulder_elevation",
        "shoulder_retraction",
        "elbow_extension",
        "forearm_pronation",
        "wrist_extension",
        "wrist_flexion",
    )
    requested = ("forearm_pronation", "wrist_extension", "wrist_flexion")

    mask, selected, knots = _trainable_parameter_mask(
        parameterization="anatomical_synergies",
        synergy_names=synergies,
        time_knots=3,
        requested_synergies=requested,
    )

    assert selected == requested
    assert knots == (0, 1, 2)
    np.testing.assert_array_equal(
        mask.reshape(3, len(synergies)),
        np.asarray([[False, False, False, True, True, True]] * 3),
    )


def test_trainable_synergy_selection_rejects_unknown_and_duplicate_names() -> None:
    synergies = ("shoulder_elevation", "wrist_flexion")
    for requested in (("wrist_flexion", "wrist_flexion"), ("missing",)):
        try:
            _trainable_parameter_mask(
                parameterization="anatomical_synergies",
                synergy_names=synergies,
                time_knots=2,
                requested_synergies=requested,
            )
        except ValueError:
            pass
        else:  # pragma: no cover - assertion clarity
            raise AssertionError("invalid trainable synergy selection was accepted")


def test_impact_window_knot_selection_freezes_early_parameters() -> None:
    synergies = ("shoulder_rotation", "wrist_flexion", "wrist_deviation")

    mask, selected, knots = _trainable_parameter_mask(
        parameterization="anatomical_synergies",
        synergy_names=synergies,
        time_knots=6,
        requested_synergies=("wrist_flexion", "wrist_deviation"),
        requested_knot_indices=(2, 3, 4, 5),
    )

    assert selected == ("wrist_flexion", "wrist_deviation")
    assert knots == (2, 3, 4, 5)
    np.testing.assert_array_equal(
        mask.reshape(6, 3),
        np.asarray([[False, False, False], [False, False, False]] + [[False, True, True]] * 4),
    )


def test_trainable_knot_selection_rejects_duplicates_and_out_of_range() -> None:
    for requested_knots in ((), (2, 2), (-1,), (4,)):
        try:
            _trainable_parameter_mask(
                parameterization="anatomical_synergies",
                synergy_names=("wrist_flexion",),
                time_knots=4,
                requested_synergies=None,
                requested_knot_indices=requested_knots,
            )
        except ValueError:
            pass
        else:  # pragma: no cover - assertion clarity
            raise AssertionError("invalid trainable knot selection was accepted")


def test_coordinate_probe_covers_both_directions_and_preserves_frozen_values() -> None:
    center = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    candidates = np.zeros((6, 4), dtype=np.float32)
    candidates[0] = center
    candidates[1] = center

    probed, indices = _inject_coordinate_probe_candidates(
        candidates,
        center=center,
        trainable_parameter_mask=np.asarray([False, True, False, True]),
        radius=0.25,
    )

    assert indices == (2, 3, 4, 5)
    np.testing.assert_allclose(
        probed,
        np.asarray(
            [
                center,
                center,
                [0.1, 0.45, 0.3, 0.4],
                [0.1, -0.05, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.65],
                [0.1, 0.2, 0.3, 0.15],
            ],
            dtype=np.float32,
        ),
    )


def test_coordinate_probe_rejects_an_undersized_population() -> None:
    try:
        _inject_coordinate_probe_candidates(
            np.zeros((5, 2), dtype=np.float32),
            center=np.zeros((2,), dtype=np.float32),
            trainable_parameter_mask=np.ones((2,), dtype=bool),
            radius=0.1,
        )
    except ValueError:
        pass
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("undersized coordinate-probe population was accepted")


def test_search_frontier_copies_occupy_independent_candidate_groups() -> None:
    frontier = np.asarray([0.25, -0.5, 0.75], dtype=np.float32)
    candidates = np.arange(24, dtype=np.float32).reshape(8, 3)

    replayed, indices = _inject_search_frontier_copies(
        candidates,
        frontier=frontier,
        copies=3,
    )

    assert indices == (1, 2, 3)
    np.testing.assert_allclose(
        replayed[list(indices)],
        np.repeat(frontier[None, :], len(indices), axis=0),
    )
    np.testing.assert_array_equal(replayed[0], candidates[0])
    np.testing.assert_array_equal(replayed[4:], candidates[4:])


def test_search_frontier_copies_reject_an_undersized_population() -> None:
    try:
        _inject_search_frontier_copies(
            np.zeros((3, 2), dtype=np.float32),
            frontier=np.zeros((2,), dtype=np.float32),
            copies=3,
        )
    except ValueError:
        pass
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("oversized search-frontier replay was accepted")


def test_search_frontier_challenge_uses_equal_shared_batch_halves() -> None:
    template = np.arange(24, dtype=np.float32).reshape(8, 3)
    incumbent = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    challenger = np.asarray([-0.4, -0.5, -0.6], dtype=np.float32)

    batch, slices = _build_search_frontier_challenge_batch(
        template,
        incumbent=incumbent,
        challenger=challenger,
    )

    assert slices == (slice(0, 4), slice(4, 8))
    np.testing.assert_allclose(batch[slices[0]], np.repeat(incumbent[None, :], 4, axis=0))
    np.testing.assert_allclose(batch[slices[1]], np.repeat(challenger[None, :], 4, axis=0))
    np.testing.assert_array_equal(template, np.arange(24).reshape(8, 3))


def test_final_verification_relocates_candidate_across_distinct_batch_lanes() -> None:
    groups = _verification_group_indices(
        population=256,
        repeats=5,
        anchor_group=37,
    )

    assert groups[0] == 37
    assert len(groups) == len(set(groups)) == 5
    assert groups == (37, 88, 139, 190, 241)


def test_search_replicas_are_stratified_across_the_full_warp_batch() -> None:
    lanes = _stratified_candidate_lane_indices(population=8, replicas=4)

    assert lanes.shape == (8, 4)
    np.testing.assert_array_equal(lanes[0], np.asarray([0, 10, 20, 30]))
    np.testing.assert_array_equal(lanes[1], np.asarray([1, 11, 21, 31]))
    np.testing.assert_array_equal(np.sort(lanes.reshape(-1)), np.arange(32))


def test_stratified_warp_lane_metrics_restore_candidate_major_replica_order() -> None:
    candidates = np.asarray([[10.0], [20.0], [30.0], [40.0]], dtype=np.float32)
    expanded, lanes = _expand_candidates_across_stratified_lanes(
        candidates,
        replicas=2,
    )

    gathered = _gather_candidate_major_lane_values(
        expanded[:, 0],
        lane_indices=lanes,
    )
    np.testing.assert_array_equal(
        gathered.reshape(4, 2),
        np.repeat(candidates, 2, axis=1),
    )


def test_cpu_promotion_gate_requires_upward_forward_high_region_and_no_fall(
    tmp_path,
) -> None:
    path = tmp_path / "cpu_trace.npz"
    np.savez(
        path,
        trace_schema_version=np.asarray(CPU_AUDIT_TRACE_SCHEMA),
        actor_inference_semantics=np.asarray(CPU_ACTOR_INFERENCE_SEMANTICS),
        actor_inference_platform=np.asarray("cpu"),
        event_rebound=np.asarray([False, True, False]),
        body_fall=np.asarray([False, False, False]),
        outgoing_velocity_semantics=np.asarray("post_control_step_after_all_physics_substeps"),
        event_rebound_contact_semantics=np.asarray(
            "single_event_impulse_with_stringbed_force_suppressed_during_cooldown_v2"
        ),
        # Only aero and gravity remain after the event; the settled velocity
        # must therefore remain close to the instantaneous rebound.
        event_impulse_velocity_after_world_m_s=np.asarray([[0.0, 0.0, 0.0], [4.1, 1.0, 1.25], [0.0, 0.0, 0.0]]),
        event_shuttle_velocity_before_world_m_s=np.asarray(
            [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.0, 0.0]]
        ),
        shuttle_velocity=np.asarray([[0.0, 0.0, 0.0], [4.0, 1.0, 1.2], [0.0, 0.0, 0.0]]),
        correction_window=np.asarray([1.0, 1.0, 0.0]),
        time_to_intercept_s=np.asarray([0.01, -0.01, -0.03]),
        stringbed_position=np.asarray([[0.0, 0.0, 2.0], [0.0, 0.0, 1.95], [0.0, 0.0, 1.0]]),
        right_arm_body_position_xyz_m=np.asarray(
            [
                [[0.0, 0.0, 1.9]],
                [[0.0, 0.0, 1.86]],
                [[0.0, 0.0, 1.0]],
            ]
        ),
    )

    passed = _summarize_cpu_quality_trace(
        path,
        player_half_sign=-1,
        min_outgoing_z_m_s=0.5,
        min_forward_m_s=2.0,
        max_stringbed_height_deficit_m=0.10,
        max_hand_height_deficit_m=0.10,
    )
    downward = _summarize_cpu_quality_trace(
        path,
        player_half_sign=-1,
        min_outgoing_z_m_s=1.5,
        min_forward_m_s=2.0,
        max_stringbed_height_deficit_m=0.10,
        max_hand_height_deficit_m=0.10,
    )

    assert passed["cpu_quality_passed"] is True
    assert passed["outgoing_forward_m_s"] == 4.0
    assert passed["outgoing_z_m_s"] == 1.2
    assert passed["event_impulse_velocity_after_xyz_m_s"] == [4.1, 1.0, 1.25]
    assert passed["pre_event_velocity_consistent"] is True
    assert passed["event_settled_velocity_consistent"] is True
    assert downward["cpu_quality_passed"] is False

    contaminated_path = tmp_path / "cpu_trace_native_second_collision.npz"
    with np.load(path, allow_pickle=False) as payload:
        contaminated = {name: np.asarray(payload[name]) for name in payload.files}
    contaminated["event_impulse_velocity_after_world_m_s"] = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 7.0, -2.5], [0.0, 0.0, 0.0]]
    )
    np.savez(contaminated_path, **contaminated)
    rejected = _summarize_cpu_quality_trace(
        contaminated_path,
        player_half_sign=-1,
        min_outgoing_z_m_s=0.5,
        min_forward_m_s=2.0,
        max_stringbed_height_deficit_m=0.10,
        max_hand_height_deficit_m=0.10,
    )
    assert rejected["cpu_quality_passed"] is False
    assert rejected["event_settled_velocity_consistent"] is False

    frame_first_path = tmp_path / "cpu_trace_native_first_collision.npz"
    frame_first = dict(contaminated)
    frame_first["event_impulse_velocity_after_world_m_s"] = np.asarray(
        [[0.0, 0.0, 0.0], [4.1, 1.0, 1.25], [0.0, 0.0, 0.0]]
    )
    frame_first["event_shuttle_velocity_before_world_m_s"] = np.asarray(
        [[0.0, 0.0, 0.0], [8.0, 0.5, -3.0], [0.0, 0.0, 0.0]]
    )
    np.savez(frame_first_path, **frame_first)
    rejected_before_event = _summarize_cpu_quality_trace(
        frame_first_path,
        player_half_sign=-1,
        min_outgoing_z_m_s=0.5,
        min_forward_m_s=2.0,
        max_stringbed_height_deficit_m=0.10,
        max_hand_height_deficit_m=0.10,
    )
    assert rejected_before_event["cpu_quality_passed"] is False
    assert rejected_before_event["pre_event_velocity_consistent"] is False


def test_cpu_audit_actor_mean_is_explicitly_compiled_on_cpu() -> None:
    actor = {
        "policy": [
            {
                "w": jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
                "b": jnp.asarray([0.0, 0.0]),
            }
        ]
    }

    actor_fn, cpu_device = _build_explicit_cpu_actor_fn(actor)
    output = actor_fn(jax.device_put(jnp.asarray([0.25, -0.5]), cpu_device))

    assert cpu_device.platform == "cpu"
    assert {device.platform for device in output.devices()} == {"cpu"}
    np.testing.assert_array_equal(np.asarray(output), np.asarray([0.25, -0.5]))


def test_cpu_quality_gate_rejects_trace_without_cpu_actor_provenance(
    tmp_path,
) -> None:
    path = tmp_path / "legacy_gpu_actor_cpu_physics_trace.npz"
    np.savez(path, event_rebound=np.asarray([False]))

    with pytest.raises(ValueError, match="explicit CPU actor provenance"):
        _summarize_cpu_quality_trace(
            path,
            player_half_sign=-1,
            min_outgoing_z_m_s=0.5,
            min_forward_m_s=2.0,
            max_stringbed_height_deficit_m=0.10,
            max_hand_height_deficit_m=0.10,
        )


def test_sealed_compatible_candidate_can_seed_a_fresh_local_search(tmp_path) -> None:
    source_contract = {
        "schema_version": "stage3_single_feed_mjx_cem_v2",
        "feed_fingerprint": "a" * 64,
        "parameter_count": 4,
        "authority_multiplier": 2.0,
    }
    source_contract["contract_sha256"] = _json_hash(source_contract)
    (tmp_path / "cem_contract.json").write_text(
        json.dumps(source_contract),
        encoding="utf-8",
    )
    candidate = {
        "schema_version": "stage3_cem_teacher_candidate_v1",
        "contract_sha256": source_contract["contract_sha256"],
        "parameters": [0.1, -0.2, 0.3, -0.4],
        "metrics": {},
    }
    candidate_path = tmp_path / "best_teacher.json"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    parameters, binding = _load_initial_candidate(
        candidate_path,
        dimension=4,
        expected_source_contract={
            "feed_fingerprint": "a" * 64,
            "parameter_count": 4,
            "authority_multiplier": 2.0,
        },
    )

    np.testing.assert_allclose(parameters, candidate["parameters"])
    assert binding["source_contract_sha256"] == source_contract["contract_sha256"]


def test_unqualified_optimizer_mean_can_seed_a_different_mjx_backend(tmp_path) -> None:
    source_contract = {
        "schema_version": "stage3_single_feed_mjx_cem_v3",
        "feed_fingerprint": "b" * 64,
        "parameter_count": 4,
        "authority_multiplier": 2.0,
        "mjx_impl": "warp",
    }
    source_contract["contract_sha256"] = _json_hash(source_contract)
    contract_path = tmp_path / "cem_contract.json"
    contract_path.write_text(json.dumps(source_contract), encoding="utf-8")
    state_path = tmp_path / "cem_state.npz"
    parameters = np.asarray([0.4, -0.3, 0.2, -0.1], dtype=np.float32)
    np.savez(
        state_path,
        contract_sha256=np.asarray(source_contract["contract_sha256"]),
        iteration=np.asarray(5, dtype=np.int32),
        mean=parameters,
    )
    seed_path = tmp_path / "search_seed_iter0005.json"

    exported = export_search_seed(
        state_path=state_path,
        contract_path=contract_path,
        output_path=seed_path,
    )
    loaded, binding = _load_initial_candidate(
        seed_path,
        dimension=4,
        expected_source_contract={
            "feed_fingerprint": "b" * 64,
            "parameter_count": 4,
            "authority_multiplier": 2.0,
            "mjx_impl": "jax",
        },
    )

    assert exported["qualified_teacher"] is False
    assert exported["source_iteration"] == 5
    np.testing.assert_array_equal(loaded, parameters)
    assert binding["candidate_role"] == "unqualified_optimizer_mean"
    assert binding["source_mjx_impl"] == "warp"


def test_unqualified_search_seed_can_intervene_on_feed_and_swing_timing(
    tmp_path,
) -> None:
    source_contract = {
        "schema_version": "stage3_single_feed_mjx_cem_v3",
        "feed_fingerprint": "d" * 64,
        "swing_phase_advance_s": 0.18,
        "parameter_count": 2,
        "authority_multiplier": 2.0,
        "mjx_impl": "warp",
    }
    source_contract["contract_sha256"] = _json_hash(source_contract)
    (tmp_path / "cem_contract.json").write_text(
        json.dumps(source_contract),
        encoding="utf-8",
    )
    parameters = np.asarray([0.2, -0.3], dtype=np.float32)
    seed = {
        "schema_version": "stage3_cem_search_seed_v1",
        "qualified_teacher": False,
        "seed_role": "unqualified_optimizer_mean",
        "contract_sha256": source_contract["contract_sha256"],
        "parameter_f32_sha256": hashlib.sha256(parameters.tobytes(order="C")).hexdigest(),
        "parameters": parameters.tolist(),
    }
    seed_path = tmp_path / "search_seed.json"
    seed_path.write_text(json.dumps(seed), encoding="utf-8")

    loaded, binding = _load_initial_candidate(
        seed_path,
        dimension=2,
        expected_source_contract={
            "feed_fingerprint": "e" * 64,
            "swing_phase_advance_s": 0.28,
            "parameter_count": 2,
            "authority_multiplier": 2.0,
            "mjx_impl": "jax",
        },
    )

    np.testing.assert_array_equal(loaded, parameters)
    assert binding["source_feed_fingerprint"] == "d" * 64
    assert binding["target_feed_fingerprint"] == "e" * 64
    assert binding["source_swing_phase_advance_s"] == 0.18
    assert binding["target_swing_phase_advance_s"] == 0.28


def test_physical_scale_rebind_requires_explicit_unqualified_intervention(
    tmp_path,
) -> None:
    source_contract = {
        "schema_version": "stage3_single_feed_mjx_cem_v4",
        "parameter_count": 2,
        "physical_scales": [0.25, 0.50],
    }
    source_contract["contract_sha256"] = _json_hash(source_contract)
    (tmp_path / "cem_contract.json").write_text(json.dumps(source_contract), encoding="utf-8")
    parameters = np.asarray([0.2, -0.3], dtype=np.float32)
    seed_path = tmp_path / "search_seed.json"
    seed_path.write_text(
        json.dumps(
            {
                "schema_version": "stage3_cem_search_seed_v1",
                "qualified_teacher": False,
                "seed_role": "unqualified_snapshot_candidate",
                "contract_sha256": source_contract["contract_sha256"],
                "parameter_f32_sha256": hashlib.sha256(parameters.tobytes(order="C")).hexdigest(),
                "parameters": parameters.tolist(),
            }
        ),
        encoding="utf-8",
    )
    target_contract = {
        "parameter_count": 2,
        "physical_scales": [0.35, 0.70],
    }

    try:
        _load_initial_candidate(
            seed_path,
            dimension=2,
            expected_source_contract=target_contract,
        )
    except ValueError:
        pass
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("physical scales changed without explicit rebind")

    loaded, binding = _load_initial_candidate(
        seed_path,
        dimension=2,
        expected_source_contract=target_contract,
        allow_unqualified_physical_scale_rebind=True,
    )

    np.testing.assert_array_equal(loaded, parameters)
    assert binding["unqualified_physical_scale_rebind"] is True
    assert binding["source_physical_scales"] == [0.25, 0.50]
    assert binding["target_physical_scales"] == [0.35, 0.70]


def test_physical_scale_rebind_cannot_reinterpret_synergy_basis(tmp_path) -> None:
    source_contract = {
        "schema_version": "stage3_single_feed_mjx_cem_v4",
        "parameter_count": 2,
        "physical_scales": [0.25, 0.50],
        "synergy_basis_sha256": "a" * 64,
    }
    source_contract["contract_sha256"] = _json_hash(source_contract)
    (tmp_path / "cem_contract.json").write_text(json.dumps(source_contract), encoding="utf-8")
    parameters = np.asarray([0.2, -0.3], dtype=np.float32)
    seed_path = tmp_path / "search_seed.json"
    seed_path.write_text(
        json.dumps(
            {
                "schema_version": "stage3_cem_search_seed_v1",
                "qualified_teacher": False,
                "seed_role": "unqualified_snapshot_candidate",
                "contract_sha256": source_contract["contract_sha256"],
                "parameter_f32_sha256": hashlib.sha256(parameters.tobytes(order="C")).hexdigest(),
                "parameters": parameters.tolist(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="synergy_basis_sha256"):
        _load_initial_candidate(
            seed_path,
            dimension=2,
            expected_source_contract={
                "parameter_count": 2,
                "physical_scales": [0.35, 0.70],
                "synergy_basis_sha256": "b" * 64,
            },
            allow_unqualified_physical_scale_rebind=True,
        )


def test_unqualified_nested_time_knot_rebind_preserves_physical_trajectory(
    tmp_path,
) -> None:
    basis = np.eye(2, dtype=np.float32)
    basis_sha = hashlib.sha256(basis.tobytes(order="C")).hexdigest()
    source_contract = {
        "schema_version": "stage3_single_feed_mjx_cem_v4",
        "parameterization": "anatomical_synergies",
        "time_knots": 2,
        "latent_size": 2,
        "parameter_count": 4,
        "physical_scales": [0.5, 0.8],
        "synergy_names": ["proximal_a", "proximal_b"],
        "synergy_basis_sha256": basis_sha,
    }
    source_contract["contract_sha256"] = _json_hash(source_contract)
    (tmp_path / "cem_contract.json").write_text(json.dumps(source_contract), encoding="utf-8")
    source_parameters = np.asarray([0.8, -0.6, -1.1, 0.9], dtype=np.float32)
    seed_path = tmp_path / "search_seed.json"
    seed_path.write_text(
        json.dumps(
            {
                "schema_version": "stage3_cem_search_seed_v1",
                "qualified_teacher": False,
                "seed_role": "unqualified_snapshot_candidate",
                "contract_sha256": source_contract["contract_sha256"],
                "parameter_f32_sha256": hashlib.sha256(source_parameters.tobytes(order="C")).hexdigest(),
                "parameters": source_parameters.tolist(),
            }
        ),
        encoding="utf-8",
    )
    target_contract = {
        "parameterization": "anatomical_synergies",
        "time_knots": 6,
        "latent_size": 2,
        "parameter_count": 12,
        "physical_scales": [1.0, 1.2],
        "synergy_names": ["proximal_a", "proximal_b"],
        "synergy_basis_sha256": basis_sha,
    }

    loaded, binding = _load_initial_candidate(
        seed_path,
        dimension=12,
        expected_source_contract=target_contract,
        allow_unqualified_physical_scale_rebind=True,
        allow_unqualified_time_knot_rebind=True,
        synergy_basis=basis,
    )

    assert loaded.shape == (12,)
    assert binding["unqualified_time_knot_rebind"] is True
    assert binding["source_time_knots"] == 2
    assert binding["target_time_knots"] == 6
    report = binding["time_knot_rebind"]
    assert report["schema_version"] == ("stage3_unqualified_time_knot_physical_rebind_v1")
    assert report["nested_grid_factor"] == 5
    assert report["physical_rms_error"] <= 0.002
    assert report["physical_max_abs_error"] <= 0.010
    assert binding["bound_parameter_f32_sha256"] == hashlib.sha256(loaded.tobytes(order="C")).hexdigest()
    assert binding["candidate_parameter_f32_sha256"] != binding["bound_parameter_f32_sha256"]


def test_time_knot_rebind_rejects_a_target_grid_that_drops_source_knots(
    tmp_path,
) -> None:
    basis = np.eye(1, dtype=np.float32)
    basis_sha = hashlib.sha256(basis.tobytes(order="C")).hexdigest()
    source_contract = {
        "schema_version": "stage3_single_feed_mjx_cem_v4",
        "parameterization": "anatomical_synergies",
        "time_knots": 3,
        "latent_size": 1,
        "parameter_count": 3,
        "physical_scales": [0.5],
        "synergy_names": ["proximal"],
        "synergy_basis_sha256": basis_sha,
    }
    source_contract["contract_sha256"] = _json_hash(source_contract)
    (tmp_path / "cem_contract.json").write_text(json.dumps(source_contract), encoding="utf-8")
    parameters = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    seed_path = tmp_path / "search_seed.json"
    seed_path.write_text(
        json.dumps(
            {
                "schema_version": "stage3_cem_search_seed_v1",
                "qualified_teacher": False,
                "seed_role": "unqualified_snapshot_candidate",
                "contract_sha256": source_contract["contract_sha256"],
                "parameter_f32_sha256": hashlib.sha256(parameters.tobytes(order="C")).hexdigest(),
                "parameters": parameters.tolist(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contain every source knot"):
        _load_initial_candidate(
            seed_path,
            dimension=4,
            expected_source_contract={
                "parameterization": "anatomical_synergies",
                "time_knots": 4,
                "latent_size": 1,
                "parameter_count": 4,
                "physical_scales": [0.5],
                "synergy_names": ["proximal"],
                "synergy_basis_sha256": basis_sha,
            },
            allow_unqualified_time_knot_rebind=True,
            synergy_basis=basis,
        )


def test_snapshot_candidate_exports_with_exact_index_and_hash(tmp_path) -> None:
    contract = {
        "schema_version": "stage3_single_feed_mjx_cem_v3",
        "parameter_count": 3,
    }
    contract["contract_sha256"] = _json_hash(contract)
    contract_path = tmp_path / "cem_contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    candidates = np.asarray(
        [[0.1, 0.2, 0.3], [-0.4, 0.5, -0.6]],
        dtype=np.float32,
    )
    snapshot_path = tmp_path / "iteration_0001.npz"
    np.savez(
        snapshot_path,
        contract_sha256=np.asarray(contract["contract_sha256"]),
        iteration=np.asarray(1, dtype=np.int32),
        candidates=candidates,
    )
    seed_path = tmp_path / "search_seed_candidate0001.json"

    seed = export_snapshot_candidate_seed(
        snapshot_path=snapshot_path,
        contract_path=contract_path,
        output_path=seed_path,
        candidate_index=1,
    )

    assert seed["qualified_teacher"] is False
    assert seed["seed_role"] == "unqualified_snapshot_candidate"
    assert seed["source_candidate_index"] == 1
    np.testing.assert_array_equal(
        np.asarray(seed["parameters"], dtype=np.float32),
        candidates[1],
    )
    assert seed["parameter_f32_sha256"] == hashlib.sha256(candidates[1].tobytes(order="C")).hexdigest()


def test_qualified_teacher_cannot_cross_feed_or_swing_timing(tmp_path) -> None:
    source_contract = {
        "schema_version": "stage3_single_feed_mjx_cem_v3",
        "feed_fingerprint": "f" * 64,
        "swing_phase_advance_s": 0.18,
        "parameter_count": 2,
    }
    source_contract["contract_sha256"] = _json_hash(source_contract)
    (tmp_path / "cem_contract.json").write_text(
        json.dumps(source_contract),
        encoding="utf-8",
    )
    candidate_path = tmp_path / "best_teacher.json"
    candidate_path.write_text(
        json.dumps(
            {
                "schema_version": "stage3_cem_teacher_candidate_v1",
                "contract_sha256": source_contract["contract_sha256"],
                "parameters": [0.1, -0.2],
            }
        ),
        encoding="utf-8",
    )

    for changed_contract in (
        {
            "feed_fingerprint": "0" * 64,
            "swing_phase_advance_s": 0.18,
            "parameter_count": 2,
        },
        {
            "feed_fingerprint": "f" * 64,
            "swing_phase_advance_s": 0.28,
            "parameter_count": 2,
        },
    ):
        try:
            _load_initial_candidate(
                candidate_path,
                dimension=2,
                expected_source_contract=changed_contract,
            )
        except ValueError:
            pass
        else:  # pragma: no cover - assertion clarity
            raise AssertionError("qualified teacher crossed a physical condition")


def test_named_parameter_intervention_remains_an_unqualified_search_seed(
    tmp_path,
) -> None:
    source_contract = {
        "schema_version": "stage3_single_feed_mjx_cem_v3",
        "feed_fingerprint": "c" * 64,
        "parameter_count": 4,
        "parameterization": "anatomical_synergies",
        "time_knots": 2,
        "synergy_names": ["forearm_pronation", "forearm_supination"],
        "authority_multiplier": 2.0,
        "mjx_impl": "warp",
    }
    source_contract["contract_sha256"] = _json_hash(source_contract)
    (tmp_path / "cem_contract.json").write_text(
        json.dumps(source_contract),
        encoding="utf-8",
    )
    parameters = np.asarray([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    source_seed = {
        "schema_version": "stage3_cem_search_seed_v1",
        "qualified_teacher": False,
        "seed_role": "unqualified_optimizer_mean",
        "contract_sha256": source_contract["contract_sha256"],
        "parameter_f32_sha256": hashlib.sha256(parameters.tobytes(order="C")).hexdigest(),
        "parameters": parameters.tolist(),
    }
    source_path = tmp_path / "search_seed.json"
    source_path.write_text(json.dumps(source_seed), encoding="utf-8")
    derived_path = tmp_path / "search_seed_face_opening.json"

    derived = derive_search_seed(
        source_path=source_path,
        output_path=derived_path,
        deltas=(
            "1:forearm_pronation:-0.5",
            "1:forearm_supination:0.75",
        ),
    )
    loaded, binding = _load_initial_candidate(
        derived_path,
        dimension=4,
        expected_source_contract={
            "feed_fingerprint": "c" * 64,
            "parameter_count": 4,
            "authority_multiplier": 2.0,
            "mjx_impl": "jax",
        },
    )

    assert derived["qualified_teacher"] is False
    assert derived["seed_role"] == "unqualified_parameter_intervention"
    np.testing.assert_allclose(loaded, [0.1, -0.2, -0.2, 0.35])
    assert binding["candidate_role"] == "unqualified_parameter_intervention"


def test_replica_robust_contact_starts_the_return_quality_stage() -> None:
    metrics = _base_metrics()
    metrics["event_rebound"][0] = True
    metrics["event_rebound_rate"][0] = 2.0 / 3.0
    metrics["contact_acquisition_cost_m"][:] = (0.20, 0.01)

    assert int(_rank_order(metrics)[-1]) == 0


def test_partial_strict_returns_robustify_before_more_downward_rebounds() -> None:
    metrics = _base_metrics()
    # Candidate 0 clears the 6/8 contact gate but only returns one replica
    # upward and forward.  Candidate 1 has five contacts and three useful
    # returns.  The latter must seed robustification; otherwise CEM is pulled
    # back toward the stable side/downward local optimum observed in v43k.
    metrics["event_rebound"][:] = (True, False)
    metrics["event_rebound_rate"][:] = (6.0 / 8.0, 5.0 / 8.0)
    metrics["stringbed_contact"][:] = True
    metrics["stringbed_contact_rate"][:] = (6.0 / 8.0, 5.0 / 8.0)
    metrics["return_quality_rate"][:] = (1.0 / 8.0, 3.0 / 8.0)
    metrics["positive_outgoing_z_rate"][:] = metrics["return_quality_rate"]
    metrics["positive_outgoing_forward_rate"][:] = metrics["return_quality_rate"]

    assert int(_rank_order(metrics)[-1]) == 1


def test_equal_quality_rate_prefers_drag_progress_over_extra_bad_rebounds() -> None:
    metrics = _base_metrics()
    metrics["return_quality_rate"][:] = 2.0 / 8.0
    metrics["event_rebound_rate"][:] = (7.0 / 8.0, 3.0 / 8.0)
    metrics["ballistic_return_progress_mean_score"] = np.asarray([0.0025, 0.0080], dtype=np.float64)

    assert int(_rank_order(metrics)[-1]) == 1


def test_drag_progress_blocks_more_sideways_upward_forward_events() -> None:
    metrics = _base_metrics()
    # Candidate 1 has more replicas above the coarse forward/z thresholds,
    # but their drag-aware direction/clearance progress is worse.
    metrics["return_quality_rate"][:] = (1.0 / 8.0, 3.0 / 8.0)
    metrics["ballistic_return_progress_mean_score"] = np.asarray([0.0080, 0.0030], dtype=np.float64)

    assert int(_rank_order(metrics)[-1]) == 0


def test_return_search_rejects_quality_that_falls_below_safety_floor() -> None:
    metrics = _base_metrics()
    metrics["return_quality_rate"][:] = (1.0 / 8.0, 5.0 / 8.0)
    metrics["ballistic_return_progress_mean_score"] = np.asarray([0.002, 0.020], dtype=np.float64)
    metrics["no_fall_rate"][:] = (0.98, 0.97)

    assert int(_rank_order(metrics)[-1]) == 0


def test_replica_robust_stringbed_touch_beats_more_proximity() -> None:
    metrics = _base_metrics()
    metrics["stringbed_contact"][0] = True
    metrics["stringbed_contact_rate"][0] = 2.0 / 3.0
    metrics["stringbed_contact_speed_m_s"][0] = 1.0
    metrics["contact_acquisition_cost_m"][:] = (0.20, 0.01)

    assert int(_rank_order(metrics)[-1]) == 0


def test_one_off_stringbed_touch_does_not_beat_robust_proximity() -> None:
    metrics = _base_metrics()
    metrics["stringbed_contact_rate"][0] = 1.0 / 3.0
    metrics["contact_acquisition_cost_m"][:] = (0.14, 0.08)

    assert int(_rank_order(metrics)[-1]) == 1


def test_three_of_three_stringbed_contact_beats_two_of_three() -> None:
    metrics = _base_metrics()
    metrics["stringbed_contact"][:] = True
    metrics["stringbed_contact_rate"][:] = (2.0 / 3.0, 1.0)
    # The less stable candidate has the better inverse-impact score, but
    # contact reproducibility is the earlier search stage.
    metrics["closest_inverse_impact_decomposed_score"][:] = (1.0, 0.0)

    assert int(_rank_order(metrics)[-1]) == 1


def test_contact_stage_uses_combined_impact_score_before_tiny_normal_difference() -> None:
    metrics = _base_metrics()
    metrics["stringbed_contact"][:] = True
    metrics["stringbed_contact_rate"][:] = 1.0
    metrics["closest_inverse_impact_decomposed_score"][:] = (0.54, 0.62)
    metrics["closest_inverse_impact_normal_alignment"][:] = (0.6691, 0.6687)
    metrics["closest_inverse_impact_racket_velocity_error_m_s"][:] = (7.94, 6.37)

    assert int(_rank_order(metrics)[-1]) == 1


def test_contact_stage_prefers_impact_quality_before_raw_closing_speed() -> None:
    metrics = _base_metrics()
    metrics["stringbed_contact"][:] = True
    metrics["stringbed_contact_rate"][:] = 1.0
    metrics["closest_inverse_impact_decomposed_score"][:] = (0.8, 0.7)
    metrics["stringbed_contact_closing_speed_m_s"][:] = (1.0, 10.0)

    assert int(_rank_order(metrics)[-1]) == 0


def test_rebound_stage_lifts_downward_return_before_optimizing_forward_speed() -> None:
    metrics = _base_metrics()
    metrics["event_rebound"][:] = True
    metrics["event_rebound_rate"][:] = 1.0
    metrics["stringbed_contact"][:] = True
    metrics["stringbed_contact_rate"][:] = 1.0
    # Candidate 0 is the observed failure mode: reliably forward but sharply
    # downward.  Candidate 1 preserves the rebound and moves vertically toward
    # a clear even though its forward component has not been recovered yet.
    metrics["outgoing_z_m_s"][:] = (-3.0, -1.0)
    metrics["outgoing_forward_m_s"][:] = (4.0, 0.0)
    metrics["positive_outgoing_forward_rate"][:] = (1.0, 0.0)

    assert int(_rank_order(metrics)[-1]) == 1


def test_rebound_stage_requires_upward_and_forward_on_the_same_replica() -> None:
    metrics = _base_metrics()
    metrics["event_rebound"][:] = True
    metrics["event_rebound_rate"][:] = 1.0
    metrics["positive_outgoing_z_rate"][:] = 1.0 / 3.0
    metrics["positive_outgoing_forward_rate"][:] = 1.0 / 3.0
    # Only candidate 1 has a replica that is simultaneously upward and
    # forward; equal marginal rates alone cannot prove that conjunction.
    metrics["return_quality_rate"][:] = (0.0, 1.0 / 3.0)

    assert int(_rank_order(metrics)[-1]) == 1


def test_partial_teacher_success_stabilizes_remaining_replicas_before_expansion() -> None:
    metrics = _base_metrics()
    metrics["event_rebound"][:] = True
    metrics["event_rebound_rate"][:] = 1.0
    metrics["teacher_success_rate"][:] = 1.0 / 3.0
    metrics["no_fall_rate"][:] = (2.0 / 3.0, 1.0)
    metrics["return_quality_rate"][:] = (1.0, 1.0 / 3.0)

    assert int(_rank_order(metrics)[-1]) == 1


def test_partial_success_prefers_all_upright_before_more_unsafe_replicas() -> None:
    metrics = _base_metrics()
    metrics["event_rebound"][:] = True
    metrics["event_rebound_rate"][:] = 1.0
    metrics["teacher_success_rate"][:] = (1.0 / 8.0, 3.0 / 8.0)
    metrics["return_quality_rate"][:] = metrics["teacher_success_rate"]
    metrics["no_fall_rate"][:] = (1.0, 7.0 / 8.0)

    assert int(_rank_order(metrics)[-1]) == 0


def test_robust_teacher_boolean_requires_every_replica_to_remain_upright() -> None:
    replicas = 4
    values = {
        "event_rebound": np.ones((replicas,), dtype=bool),
        "event_settled_velocity_delta_m_s": np.zeros((replicas,)),
        "stringbed_contact": np.ones((replicas,), dtype=bool),
        "no_fall": np.asarray([True, True, True, False]),
        "high_region_contact": np.ones((replicas,), dtype=bool),
        "outgoing_z_m_s": np.ones((replicas,), dtype=np.float64),
        "outgoing_forward_m_s": np.full((replicas,), 3.0),
        "predicted_clearance_m": np.ones((replicas,), dtype=np.float64),
        "crossed_net": np.ones((replicas,), dtype=bool),
        "opponent_back_landing": np.zeros((replicas,), dtype=bool),
        "min_ball_racket_distance_m": np.full((replicas,), 0.05),
        "soft_high_region_excess_m": np.zeros((replicas,)),
        "correction_rms": np.zeros((replicas,)),
        "correction_rate_cost": np.zeros((replicas,)),
        "stringbed_height_deficit_at_hit_m": np.zeros((replicas,)),
        "hand_height_deficit_at_hit_m": np.zeros((replicas,)),
        "hit_racket_vertical_velocity_m_s": np.ones((replicas,)),
        "hit_contact_speed_m_s": np.ones((replicas,)),
        "stringbed_contact_speed_m_s": np.ones((replicas,)),
        "stringbed_contact_closing_speed_m_s": np.ones((replicas,)),
        "closest_inverse_impact_decomposed_score": np.ones((replicas,)),
        "closest_inverse_impact_normal_alignment": np.ones((replicas,)),
        "closest_inverse_impact_racket_velocity_error_m_s": np.zeros((replicas,)),
    }
    aggregated = _aggregate_replica_metrics(
        values,
        population=1,
        replicas=replicas,
        min_replica_fraction=0.5,
        min_outgoing_z_m_s=0.8,
        min_forward_m_s=2.5,
    )

    assert aggregated["teacher_success_rate"].item() == 0.75
    assert not aggregated["teacher_success"].item()
    assert not aggregated["no_fall"].item()


def test_robust_teacher_rejects_event_followed_by_duplicate_native_collision() -> None:
    replicas = 4
    values = {
        "event_rebound": np.ones((replicas,), dtype=bool),
        "event_settled_velocity_delta_m_s": np.asarray([0.10, 0.20, 4.0, 5.0]),
        "stringbed_contact": np.ones((replicas,), dtype=bool),
        "no_fall": np.ones((replicas,), dtype=bool),
        "high_region_contact": np.ones((replicas,), dtype=bool),
        "outgoing_z_m_s": np.ones((replicas,), dtype=np.float64),
        "outgoing_forward_m_s": np.full((replicas,), 3.0),
        "predicted_clearance_m": np.ones((replicas,), dtype=np.float64),
        "crossed_net": np.ones((replicas,), dtype=bool),
        "opponent_back_landing": np.zeros((replicas,), dtype=bool),
        "min_ball_racket_distance_m": np.full((replicas,), 0.05),
        "soft_high_region_excess_m": np.zeros((replicas,)),
        "correction_rms": np.zeros((replicas,)),
        "correction_rate_cost": np.zeros((replicas,)),
        "stringbed_height_deficit_at_hit_m": np.zeros((replicas,)),
        "hand_height_deficit_at_hit_m": np.zeros((replicas,)),
        "hit_racket_vertical_velocity_m_s": np.ones((replicas,)),
        "hit_contact_speed_m_s": np.ones((replicas,)),
        "stringbed_contact_speed_m_s": np.ones((replicas,)),
        "stringbed_contact_closing_speed_m_s": np.ones((replicas,)),
        "closest_inverse_impact_decomposed_score": np.ones((replicas,)),
        "closest_inverse_impact_normal_alignment": np.ones((replicas,)),
        "closest_inverse_impact_racket_velocity_error_m_s": np.zeros((replicas,)),
    }
    aggregated = _aggregate_replica_metrics(
        values,
        population=1,
        replicas=replicas,
        min_replica_fraction=0.75,
    )

    assert aggregated["raw_event_rebound_rate"].item() == 1.0
    assert aggregated["event_settled_velocity_consistency_rate"].item() == 0.5
    assert aggregated["event_rebound_rate"].item() == 0.5
    assert not aggregated["teacher_success"].item()


def test_drag_aware_teacher_requires_a_real_cross_and_ranks_balanced_progress() -> None:
    population = 2
    replicas = 4
    count = population * replicas
    values = {
        "event_rebound": np.ones((count,), dtype=bool),
        "event_settled_velocity_delta_m_s": np.zeros((count,)),
        "stringbed_contact": np.ones((count,), dtype=bool),
        "no_fall": np.ones((count,), dtype=bool),
        "high_region_contact": np.ones((count,), dtype=bool),
        "outgoing_z_m_s": np.asarray([4.0] * replicas + [4.0] * replicas),
        "outgoing_forward_m_s": np.asarray([7.5] * replicas + [7.5] * replicas),
        "outgoing_lateral_m_s": np.asarray([4.0] * replicas + [0.5] * replicas),
        "predicted_clearance_m": np.asarray([0.5] * replicas + [0.5] * replicas),
        "return_clearance_score": np.asarray([0.70] * replicas + [0.70] * replicas),
        "return_direction_signed_score": np.asarray([0.55] * replicas + [0.85] * replicas),
        # A plausible projection is not a teacher certificate.  Only the
        # second candidate actually crosses in every simulated replica.
        "crossed_net": np.asarray([False] * replicas + [True] * replicas),
        "opponent_back_landing": np.zeros((count,), dtype=bool),
        "min_ball_racket_distance_m": np.full((count,), 0.05),
        "soft_high_region_excess_m": np.zeros((count,)),
        "correction_rms": np.zeros((count,)),
        "correction_rate_cost": np.zeros((count,)),
        "stringbed_height_deficit_at_hit_m": np.zeros((count,)),
        "hand_height_deficit_at_hit_m": np.zeros((count,)),
        "hit_racket_vertical_velocity_m_s": np.ones((count,)),
        "hit_contact_speed_m_s": np.ones((count,)),
        "stringbed_contact_speed_m_s": np.ones((count,)),
        "stringbed_contact_closing_speed_m_s": np.ones((count,)),
        "closest_inverse_impact_decomposed_score": np.ones((count,)),
        "closest_inverse_impact_normal_alignment": np.ones((count,)),
        "closest_inverse_impact_racket_velocity_error_m_s": np.zeros((count,)),
    }
    aggregated = _aggregate_replica_metrics(
        values,
        population=population,
        replicas=replicas,
        min_replica_fraction=0.75,
        min_outgoing_z_m_s=3.0,
        min_forward_m_s=7.0,
        require_legal_return_for_teacher=True,
        require_real_net_cross_for_teacher=True,
        min_predicted_clearance_m=0.2,
        min_return_direction_signed_score=0.65,
    )

    assert not aggregated["teacher_success"][0]
    assert aggregated["teacher_success"][1]
    assert aggregated["legal_return_rate"].tolist() == [0.0, 1.0]
    assert aggregated["ballistic_return_progress_score"][1] > aggregated["ballistic_return_progress_score"][0]
    assert int(_rank_order(aggregated)[-1]) == 1

    # A real valid crossing already includes the environment's configured
    # 20 cm net clearance. The maximum-drag projection remains useful for
    # ranking, but must not veto an observed legal trajectory when the
    # explicit authoritative mode is selected.
    values["predicted_clearance_m"][replicas:] = -2.0
    conservative = _aggregate_replica_metrics(
        values,
        population=population,
        replicas=replicas,
        min_replica_fraction=0.75,
        min_outgoing_z_m_s=3.0,
        min_forward_m_s=7.0,
        require_legal_return_for_teacher=True,
        require_real_net_cross_for_teacher=True,
        min_predicted_clearance_m=0.2,
        min_return_direction_signed_score=0.65,
    )
    authoritative = _aggregate_replica_metrics(
        values,
        population=population,
        replicas=replicas,
        min_replica_fraction=0.75,
        min_outgoing_z_m_s=3.0,
        min_forward_m_s=7.0,
        require_legal_return_for_teacher=True,
        require_real_net_cross_for_teacher=True,
        real_net_cross_authoritative_for_teacher=True,
        min_predicted_clearance_m=0.2,
        min_return_direction_signed_score=0.65,
    )
    assert not conservative["teacher_success"][1]
    assert authoritative["teacher_success"][1]
    assert authoritative["legal_return_rate"][1] == 1.0


def test_cpu_quality_gate_can_require_drag_clearance_direction_and_real_cross(
    tmp_path,
) -> None:
    path = tmp_path / "cpu_legal_return_trace.npz"
    np.savez(
        path,
        trace_schema_version=np.asarray(CPU_AUDIT_TRACE_SCHEMA),
        actor_inference_semantics=np.asarray(CPU_ACTOR_INFERENCE_SEMANTICS),
        actor_inference_platform=np.asarray("cpu"),
        event_rebound=np.asarray([False, True, False]),
        body_fall=np.asarray([False, False, False]),
        outgoing_velocity_semantics=np.asarray("post_control_step_after_all_physics_substeps"),
        event_rebound_contact_semantics=np.asarray(
            "single_event_impulse_with_stringbed_force_suppressed_during_cooldown_v2"
        ),
        event_impulse_velocity_after_world_m_s=np.asarray([[0.0, 0.0, 0.0], [8.0, 0.0, 4.0], [0.0, 0.0, 0.0]]),
        event_shuttle_velocity_before_world_m_s=np.asarray(
            [[0.0, 0.0, 0.0], [-0.95, 0.0, -6.0], [0.0, 0.0, 0.0]]
        ),
        shuttle_velocity=np.asarray([[-1.0, 0.0, -6.0], [8.0, 0.0, 4.0], [6.0, 0.0, 2.0]]),
        correction_window=np.asarray([1.0, 1.0, 0.0]),
        time_to_intercept_s=np.asarray([-0.01, -0.02, -0.03]),
        stringbed_position=np.asarray([[0.0, 0.0, 2.0], [0.0, 0.0, 1.95], [0.0, 0.0, 1.0]]),
        right_arm_body_position_xyz_m=np.asarray([[[0.0, 0.0, 1.9]], [[0.0, 0.0, 1.86]], [[0.0, 0.0, 1.0]]]),
        predicted_net_clearance_m=np.asarray([0.45, 0.45, 0.45]),
        return_direction_signed_score=np.asarray([0.82, 0.82, 0.82]),
        valid_net_cross_event=np.asarray([False, False, True]),
    )

    result = _summarize_cpu_quality_trace(
        path,
        player_half_sign=-1,
        min_outgoing_z_m_s=3.0,
        min_forward_m_s=7.0,
        max_stringbed_height_deficit_m=0.10,
        max_hand_height_deficit_m=0.10,
        min_predicted_clearance_m=0.20,
        min_return_direction_signed_score=0.65,
        require_real_net_cross=True,
    )

    assert result["cpu_quality_passed"] is True
    assert result["legal_return"] is True
    assert result["crossed_net"] is True
    assert result["predicted_net_clearance_m"] == 0.45
    assert result["return_direction_signed_score"] == 0.82

    observed_cross_path = tmp_path / "cpu_observed_cross_trace.npz"
    with np.load(path, allow_pickle=False) as payload:
        observed_cross_payload = {name: np.asarray(payload[name]) for name in payload.files}
    observed_cross_payload["predicted_net_clearance_m"] = np.asarray([-2.0, -2.0, -2.0])
    np.savez(observed_cross_path, **observed_cross_payload)

    conservative = _summarize_cpu_quality_trace(
        observed_cross_path,
        player_half_sign=-1,
        min_outgoing_z_m_s=3.0,
        min_forward_m_s=7.0,
        max_stringbed_height_deficit_m=0.10,
        max_hand_height_deficit_m=0.10,
        min_predicted_clearance_m=0.20,
        min_return_direction_signed_score=0.65,
        require_real_net_cross=True,
    )
    authoritative = _summarize_cpu_quality_trace(
        observed_cross_path,
        player_half_sign=-1,
        min_outgoing_z_m_s=3.0,
        min_forward_m_s=7.0,
        max_stringbed_height_deficit_m=0.10,
        max_hand_height_deficit_m=0.10,
        min_predicted_clearance_m=0.20,
        min_return_direction_signed_score=0.65,
        require_real_net_cross=True,
        real_net_cross_authoritative=True,
    )
    assert conservative["cpu_quality_passed"] is False
    assert authoritative["cpu_quality_passed"] is True
    assert authoritative["real_net_cross_authoritative"] is True


def test_outcome_quality_not_exact_apex_selects_teacher_timing() -> None:
    metrics = _base_metrics()
    for name in (
        "event_rebound",
        "teacher_success",
        "high_region_contact",
    ):
        metrics[name][:] = True
    for name in (
        "event_rebound_rate",
        "teacher_success_rate",
        "positive_outgoing_z_rate",
        "positive_outgoing_forward_rate",
        "high_region_contact_rate",
    ):
        metrics[name][:] = 1.0
    # Candidate 0 represents exact apex; candidate 1 is still inside the broad
    # high window but produces the clearly better physical return.
    metrics["outgoing_z_m_s"][:] = (1.0, 4.0)
    metrics["outgoing_forward_m_s"][:] = (3.0, 8.0)

    assert int(_rank_order(metrics)[-1]) == 1

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from scripts.optimize_single_feed_hit_mjx import (
    _aggregate_replica_metrics,
    _anatomical_synergy_basis,
    _trainable_parameter_mask,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = (
    REPO_ROOT
    / "experiments/posttrain/incoming_shuttle_hit_forehand_clear_cem_ppo_demo_v6.yaml"
)
WRAPPER = REPO_ROOT / "scripts/run_forehand_clear_stage3_cem_ppo_demo.sh"


def _payload() -> dict:
    return yaml.safe_load(SPEC.read_text(encoding="utf-8"))


def test_v6_preserves_reference_stance_and_uses_selected_right_chain() -> None:
    payload = _payload()
    ready = payload["scene"]["reference_ready_pose"]
    direct = payload["stage3_direct"]

    assert ready["path"].endswith("raw_smooth_v1/6月2日(1)-1.npz")
    assert ready["min_left_foot_forward_lead_m"] == 0.10
    assert direct["policy_update_mode"] == "selected_physical_correction"
    assert direct["teacher_action_prior_mode"] == "time_interpolated_frozen_plus_delta"
    assert len(direct["policy_trainable_actuator_names"]) == 32
    basis, names = _anatomical_synergy_basis(
        tuple(direct["policy_trainable_actuator_names"])
    )
    assert basis.shape == (12, 32)
    assert set(names) >= {
        "shoulder_elevation",
        "shoulder_retraction",
        "shoulder_internal_rotation",
        "elbow_extension",
        "forearm_pronation",
        "wrist_extension",
        "wrist_radial_deviation",
    }


def test_v6_wrapper_searches_exactly_35_active_parameters() -> None:
    payload = _payload()
    basis, names = _anatomical_synergy_basis(
        tuple(payload["stage3_direct"]["policy_trainable_actuator_names"])
    )
    requested = (
        "shoulder_elevation",
        "shoulder_retraction",
        "shoulder_internal_rotation",
        "elbow_extension",
        "forearm_pronation",
        "wrist_extension",
        "wrist_radial_deviation",
    )
    mask, selected_names, selected_knots = _trainable_parameter_mask(
        parameterization="anatomical_synergies",
        synergy_names=names,
        time_knots=6,
        requested_synergies=requested,
        requested_knot_indices=(1, 2, 3, 4, 5),
    )

    assert basis.shape == (12, 32)
    assert selected_names == requested
    assert selected_knots == (1, 2, 3, 4, 5)
    assert int(mask.sum()) == 35
    wrapper = WRAPPER.read_text(encoding="utf-8")
    assert "--min-racket-face-forward-alignment 0.5" in wrapper
    assert "--require-cpu-quality-for-best" in wrapper


def test_forward_racket_face_is_a_teacher_promotion_gate() -> None:
    def metrics(face_alignment: float) -> dict[str, np.ndarray]:
        return {
            "event_rebound": np.array([True, True, True]),
            "event_settled_velocity_delta_m_s": np.zeros(3),
            "stringbed_contact": np.ones(3, dtype=bool),
            "no_fall": np.ones(3, dtype=bool),
            "high_region_contact": np.ones(3, dtype=bool),
            "outgoing_z_m_s": np.ones(3),
            "outgoing_forward_m_s": np.full(3, 4.0),
            "hit_racket_face_forward_alignment": np.full(3, face_alignment),
            "predicted_clearance_m": np.full(3, 0.3),
            "return_clearance_score": np.ones(3),
            "return_direction_signed_score": np.ones(3),
            "crossed_net": np.ones(3, dtype=bool),
            "opponent_back_landing": np.zeros(3, dtype=bool),
            "correction_rms": np.zeros(3),
            "correction_rate_cost": np.zeros(3),
            "soft_high_region_excess_m": np.zeros(3),
            "stringbed_height_deficit_at_hit_m": np.zeros(3),
            "hand_height_deficit_at_hit_m": np.zeros(3),
            "hit_racket_vertical_velocity_m_s": np.ones(3),
            "hit_contact_speed_m_s": np.ones(3),
            "stringbed_contact_speed_m_s": np.ones(3),
            "stringbed_contact_closing_speed_m_s": np.ones(3),
            "closest_inverse_impact_decomposed_score": np.ones(3),
            "closest_inverse_impact_normal_alignment": np.ones(3),
            "closest_inverse_impact_racket_velocity_error_m_s": np.zeros(3),
            "min_ball_racket_distance_m": np.zeros(3),
        }

    accepted = _aggregate_replica_metrics(
        metrics(0.6),
        population=1,
        replicas=3,
        min_replica_fraction=2.0 / 3.0,
        min_racket_face_forward_alignment=0.5,
    )
    rejected = _aggregate_replica_metrics(
        metrics(0.4),
        population=1,
        replicas=3,
        min_replica_fraction=2.0 / 3.0,
        min_racket_face_forward_alignment=0.5,
    )

    assert bool(accepted["teacher_success"][0]) is True
    assert bool(rejected["teacher_success"][0]) is False


def test_v6_ppo_uses_one_complete_on_policy_batch() -> None:
    payload = _payload()
    ppo = payload["ppo"]

    assert ppo["rollout_steps"] * 256 == ppo["minibatch_size"]
    assert ppo["update_epochs"] == 1
    assert ppo["total_steps"] == 12_000_000

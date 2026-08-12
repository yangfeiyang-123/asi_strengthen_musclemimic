from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from environment.overall_environment.src.shuttle_feeder import (
    build_feed_bank,
    feed_sample_fingerprint,
)
from environment.overall_environment.src.stage3_lab import Stage3ActionRouter
from environment.overall_environment.src.stage3_target_bank_v2 import (
    DesiredImpactTarget,
    build_target_bank,
)
from musclemimic.badminton.scripts.run_incoming_shuttle_hit import (
    _build_stage3_direct_curriculum,
    _build_stage3_lab_components,
    _feed_config,
    _hit_window,
    _make_env,
    load_incoming_hit_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = REPO_ROOT / "experiments/posttrain/incoming_shuttle_hit_full354_v1.yaml"
LAB_SPEC = REPO_ROOT / "experiments/posttrain/incoming_shuttle_hit_impact_recovery_v2.yaml"
V37_SPEC = (
    REPO_ROOT
    / "experiments/posttrain/incoming_shuttle_hit_high_point_selected_physical_v37.yaml"
)


def test_v37_teacher_curriculum_starts_from_audited_feed_and_phase() -> None:
    paths = load_incoming_hit_spec(V37_SPEC)
    direct = paths.stage3_direct
    fingerprints = tuple(direct["seed_feed_fingerprints"])

    assert direct["teacher_action_prior_mode"] == (
        "time_interpolated_frozen_plus_delta"
    )
    assert direct["swing_phase_advance_s"] == 0.28
    assert len(fingerprints) == len(set(fingerprints)) == 16
    assert fingerprints[0] == (
        "9f5ba6edf006232ce444dfb82d5c28ff2e234a987eee09f6e57ee04b88165dfa"
    )
    assert direct["curriculum"]["jitter_feed_count"] == 16
    assert paths.ppo_overrides["total_steps"] == 12_000_000


def test_selected_physical_correction_honors_feed_curriculum_without_external_base() -> None:
    paths = SimpleNamespace(
        stage3_direct={
            "control_mode": "frozen_base_residual",
            "policy_update_mode": "selected_physical_correction",
            "curriculum": {
                "enabled": True,
                "fixed_feed_steps": 123,
                "jitter_feed_count": 7,
                "jitter_expand_steps": 456,
                "full_bank_expand_steps": 789,
                "fixed_min_positive_outgoing_z_rate_on_hit": 0.6,
                "jitter_min_positive_outgoing_z_rate_on_hit": 0.5,
                "full_bank_min_positive_outgoing_z_rate_on_hit": 0.4,
            },
        }
    )

    curriculum = _build_stage3_direct_curriculum(
        paths,
        base_policy_artifact=None,
    )

    assert curriculum is not None
    assert curriculum.fixed_feed_steps == 123
    assert curriculum.jitter_feed_count == 7
    assert curriculum.jitter_expand_steps == 456
    assert curriculum.full_bank_expand_steps == 789


def test_full_network_without_external_base_does_not_gain_adapter_curriculum() -> None:
    paths = SimpleNamespace(
        stage3_direct={
            "control_mode": "frozen_base_residual",
            "policy_update_mode": "full_network",
            "curriculum": {"enabled": True},
        }
    )

    assert (
        _build_stage3_direct_curriculum(paths, base_policy_artifact=None) is None
    )


def _one_target_bank(paths, feed):
    target = DesiredImpactTarget(
        feed_fingerprint=feed_sample_fingerprint(feed),
        impact_position_world=tuple(float(v) for v in feed.intercept_point),
        impact_time_s=float(feed.intercept_time_s),
        stringbed_normal_world=(1.0, 0.0, 0.0),
        racket_linear_velocity_world=(8.0, 0.0, 2.0),
        racket_angular_velocity_world=(0.0, 5.0, 0.0),
        landing_target_xy=(5.8, 0.0),
        apex_height_m=3.2,
        recovery_horizon_steps=paths.recovery_horizon_steps,
        provenance="unit_test",
    )
    return build_target_bank([target], source_fingerprint="b" * 64)


def test_full354_spec_builds_real_direct_action_environment() -> None:
    paths = load_incoming_hit_spec(SPEC)
    lab_paths = load_incoming_hit_spec(LAB_SPEC)

    assert paths.task_profile == "impact_recovery_v2"
    assert paths.stage3_lab["enabled"] is False
    assert paths.ppo_overrides["action_std_init"] == 0.16
    assert "latent_checkpoint_dir" not in paths.stage3_lab
    assert _build_stage3_lab_components(paths) is None
    assert paths.output_dir != lab_paths.output_dir
    assert paths.output_dir.parts[-3:] == (
        "posttrain_full354_v1",
        "IncomingShuttleHitImpactRecoveryFull354",
        "full354_v1_rigid_tool_v4_overhead_feed_v2",
    )
    assert paths.feed_bank_path == lab_paths.feed_bank_path
    assert paths.eval_feed_bank_path == lab_paths.eval_feed_bank_path
    assert paths.stage3_lab["contact_phase"] == 0.76
    assert paths.feed_kwargs["intercept_time_range_s"] == (1.05, 1.45)
    assert paths.hit_window_kwargs["z_range"] == (1.85, 2.25)

    hand = paths.stage3_lab["hand_fixture"]
    attachment = paths.stage3_lab["racket_attachment"]
    assert hand == {
        "mode": "removed",
        "policy_enabled": False,
        "observations_enabled": False,
    }
    assert attachment == {
        "mode": "exact_child",
        "contract_path": "configs/racket_attachment/forehand_clear_rigid_v4_custom.json",
        "hand_racket_contact_enabled": False,
    }

    feeds = build_feed_bank(
        1,
        seed=paths.feed_seed,
        cfg=_feed_config(paths),
        window=_hit_window(paths),
    )
    env = _make_env(
        paths,
        feed_bank=feeds,
        impact_target_bank=_one_target_bank(paths, feeds[0]),
        seed=0,
    )

    router = Stage3ActionRouter.from_model(env.model)
    assert env.model.nu == 354
    assert env.action_size == 354
    assert env.full_action_size == 354
    assert env.expects_raw_latent is False
    assert env.lab_controller is None
    assert env.contact_phase == 0.76
    assert router.expected_sizes == (354, 0, 0)

    manifest = env.control_manifest
    assert manifest["schema_version"] == "incoming_hit_direct_action_impact_recovery_v2"
    assert manifest["environment_abi"]["full_action_size"] == 354
    assert manifest["filter_finger_observation"] is False
    assert manifest["racket_attachment"]["attachment_mode"] == "exact_child"
    assert manifest["racket_attachment"]["contract_passed"] is True
    assert manifest["racket_attachment"]["hand_racket_contact_enabled"] is False
    assert manifest["racket_attachment"]["racket_joint_count"] == 0
    assert manifest["racket_attachment"]["racket_equality_constraint_count"] == 0

    observation, _ = env.reset(feed_index=0)
    next_observation, reward, terminated, truncated, _ = env.step(np.zeros(354, dtype=np.float32))
    assert observation.shape == next_observation.shape == (env.observation_size,)
    assert np.isfinite(next_observation).all()
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

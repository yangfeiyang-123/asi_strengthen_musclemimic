from __future__ import annotations

from pathlib import Path

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
    _build_stage3_lab_components,
    _feed_config,
    _hit_window,
    _make_env,
    load_incoming_hit_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = REPO_ROOT / "experiments/posttrain/incoming_shuttle_hit_full354_v1.yaml"
LAB_SPEC = REPO_ROOT / "experiments/posttrain/incoming_shuttle_hit_impact_recovery_v2.yaml"


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
        "full354_v1_rigid_tool_v3",
    )

    hand = paths.stage3_lab["hand_fixture"]
    attachment = paths.stage3_lab["racket_attachment"]
    assert hand == {
        "mode": "removed",
        "policy_enabled": False,
        "observations_enabled": False,
    }
    assert attachment == {
        "mode": "exact_child",
        "contract_path": "configs/racket_attachment/forehand_clear_rigid_v2.json",
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

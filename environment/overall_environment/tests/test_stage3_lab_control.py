from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment.overall_environment.src.incoming_shuttle_hit_env import (  # noqa: E402
    IncomingShuttleHitEnv,
)
from environment.overall_environment.src.incoming_shuttle_hit_mjx_env import (  # noqa: E402
    IncomingHitMjxEnv,
)
from environment.overall_environment.src.paths import default_incoming_scene_path  # noqa: E402
from environment.overall_environment.src.shuttle_feeder import (  # noqa: E402
    FeedSample,
    build_feed_bank,
)
from environment.overall_environment.src.stage3_lab import (  # noqa: E402
    DEFAULT_RIGHT_WRIST_FOREARM_RESIDUAL_NAMES,
    BoundedResidualGroup,
    BoundedResidualMask,
    ConstantGripProvider,
    Stage3ActionRouter,
    Stage3Curriculum,
    Stage3LABController,
    Stage3LabStateBuilder,
    apply_teacher_body_ctrlrange,
    bounded_residual_mask_from_config,
    stage3_attachment_report,
)
from musclemimic.badminton.scripts.run_incoming_shuttle_hit import (  # noqa: E402
    _base_only_summary,
    _compare_naturalness_to_prior,
    _ensure_feed_bank_artifact,
    _feed_bank_identity_qc,
    _policy_update_contract,
    _return_net_clearance,
    _run_ppo,
    _seal_stage3_training_run_manifest,
    _stage3_evaluation_summary,
    load_incoming_hit_spec,
    preflight,
)


def test_stage3_naturalness_is_paired_against_prior_body_site_and_racket():
    def row(offset: float):
        angle = 0.1 + offset
        c, s = np.cos(angle), np.sin(angle)
        return {
            "body": np.asarray([0.0, 1.0 + offset]),
            "right_hand_site": np.asarray([offset, 0.0, 1.0]),
            "racket_position": np.asarray([offset, 0.2, 1.5]),
            "racket_rotation": np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]),
        }

    prior = [row(0.0), row(0.1), row(0.2)]
    identical = _compare_naturalness_to_prior(prior, prior)
    assert identical["body_relative_deviation_to_prior"] == pytest.approx(0.0)
    assert identical["racket_rotation_rmse_to_prior_rad"] == pytest.approx(0.0)

    degraded = _compare_naturalness_to_prior([row(0.0), row(0.2), row(0.4)], prior)
    assert degraded["right_hand_site_rmse_to_prior_m"] > 0.0
    assert degraded["racket_position_relative_deviation_to_prior"] > 0.0
    assert degraded["racket_rotation_rmse_to_prior_rad"] > 0.0


def _synthetic_feed(identity: float) -> FeedSample:
    launch_pos = np.asarray([3.0, 0.0, 1.0], dtype=float)
    launch_vel = np.asarray([-10.0, 0.0, 2.0], dtype=float)
    trajectory = np.asarray(
        [
            [0.0, *launch_pos, *launch_vel],
            [0.5 + identity, -2.7, identity, 1.8, -8.0, 0.0, -1.0],
        ],
        dtype=float,
    )
    return FeedSample(
        launch_pos=launch_pos,
        launch_vel=launch_vel,
        trajectory=trajectory,
        intercept_index=1,
        intercept_point=trajectory[1, 1:4].copy(),
        intercept_velocity=trajectory[1, 4:7].copy(),
        intercept_time_s=float(trajectory[1, 0]),
    )


def test_feed_bank_cache_rebuilds_legacy_drifted_and_tampered_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import environment.overall_environment.src.shuttle_feeder as feeder

    calls: list[tuple[int, int, float]] = []

    def fake_build(count, seed, cfg, window, aero_cfg=None):
        calls.append((int(count), int(seed), float(cfg.azimuth_jitter_deg)))
        offset = 1e-4 * int(seed) + 1e-3 * float(cfg.azimuth_jitter_deg)
        return [_synthetic_feed(offset + index * 1e-3) for index in range(int(count))]

    monkeypatch.setattr(feeder, "build_feed_bank", fake_build)
    base = load_incoming_hit_spec("experiments/posttrain/incoming_shuttle_hit_v1.yaml")
    bank_path = tmp_path / "train_bank.npz"
    bank_path.write_bytes(b"legacy-npz-without-manifest")
    paths = replace(
        base,
        feed_bank_path=bank_path,
        feed_bank_size=2,
        feed_seed=17,
    )

    first = _ensure_feed_bank_artifact(paths)
    assert len(first.bank) == 2
    assert first.manifest["seed"] == 17
    assert calls == [(2, 17, 6.0)]

    drifted = replace(
        paths,
        feed_kwargs={**paths.feed_kwargs, "azimuth_jitter_deg": 3.0},
    )
    second = _ensure_feed_bank_artifact(drifted)
    assert second.manifest["feed_config"]["azimuth_jitter_deg"] == 3.0
    assert calls[-1] == (2, 17, 3.0)

    with bank_path.open("ab") as handle:
        handle.write(b"tamper")
    third = _ensure_feed_bank_artifact(drifted)
    assert third.manifest == second.manifest
    assert calls.count((2, 17, 3.0)) == 2
    assert not list(tmp_path.glob("*.tmp*"))


def test_feed_bank_identity_qc_detects_duplicates_overlap_and_shared_path() -> None:
    report = _feed_bank_identity_qc(
        {"sample_fingerprints": ["train-a", "train-a"]},
        {"sample_fingerprints": ["train-a", "eval-b"]},
        paths_distinct=False,
    )
    assert report["bank_paths_distinct"] is False
    assert report["train_duplicate_count"] == 1
    assert report["eval_duplicate_count"] == 0
    assert report["train_eval_fingerprint_overlap_count"] == 1


def test_stage3_base_only_summary_is_fail_closed() -> None:
    thresholds = {
        "min_rollout_count": 5.0,
        "min_completion_rate": 1.0,
        "min_finite_rate": 1.0,
        "min_no_fall_rate": 0.95,
        "min_root_height_m": 0.55,
        "max_body_action_saturation_fraction": 0.01,
        "max_full_action_saturation_fraction": 0.01,
        "max_normalized_control_energy": 0.35,
        "max_lab_state_ood_fraction": 0.01,
        "min_control_finite": 1.0,
        "max_attachment_translation_drift_m": 0.005,
        "max_attachment_rotation_drift_rad": 0.05,
    }
    safe = {
        "completed_steps": 120,
        "finite": True,
        "body_fall": False,
        "min_root_height_m": 0.80,
        "max_body_action_saturation_fraction": 0.0,
        "max_full_action_saturation_fraction": 0.0,
        "max_normalized_control_energy": 0.10,
        "max_lab_state_ood_fraction": 0.0,
        "min_control_finite": 1.0,
        "max_attachment_translation_drift_m": 0.001,
        "max_attachment_rotation_drift_rad": 0.01,
    }
    passed = _base_only_summary(
        [safe.copy() for _ in range(5)],
        thresholds=thresholds,
        required_steps=120,
    )
    assert passed["passed"] is True

    missing = safe.copy()
    missing.pop("max_lab_state_ood_fraction")
    failed = _base_only_summary(
        [missing, *[safe.copy() for _ in range(4)]],
        thresholds=thresholds,
        required_steps=120,
    )
    assert failed["gates"]["lab_state_ood_fraction"] is False
    assert failed["passed"] is False


def test_stage3_evaluate_report_schema_matches_pipeline_and_requires_128_feeds() -> None:
    successful_episode = {
        "hit": True,
        "crossed_net": True,
        "body_fall": False,
        "landing_region": "opponent_back",
        "max_racket_head_speed_m_s": 8.5,
        "contact_racket_head_speed_m_s": 8.5,
        "hit_outgoing_velocity_xyz_m_s": [3.0, 0.0, 2.0],
        "net_clearance_m": 0.30,
        "min_root_height_m": 0.80,
        "max_attachment_translation_drift_m": 0.001,
        "max_attachment_rotation_drift_rad": 0.01,
        "lab_diagnostics": {
            "control_finite": 1.0,
            "body_action_saturation_fraction": 0.0,
            "full_action_saturation_fraction": 0.0,
            "normalized_control_energy": 0.10,
            "raw_latent_saturation": 0.0,
            "lab_state_ood_fraction": 0.0,
            "lab_state_unclipped_z_rms": 1.0,
        },
        "naturalness": {
            "body_relative_deviation_to_prior": 0.10,
            "right_hand_site_rmse_to_prior_m": 0.05,
            "right_hand_site_relative_deviation_to_prior": 0.10,
            "racket_position_rmse_to_prior_m": 0.05,
            "racket_position_relative_deviation_to_prior": 0.10,
            "racket_rotation_rmse_to_prior_rad": 0.10,
            "racket_rotation_relative_deviation_to_prior": 0.10,
        },
    }
    summary = _stage3_evaluation_summary(
        [successful_episode.copy() for _ in range(128)],
        gate_config={"min_positive_outgoing_z_rate_on_hit": 0.5},
        required_feed_count=128,
        prior_direct_baseline={"prior_vs_direct_body_racket_relative_degradation": 0.05},
    )

    assert summary["passed"] is True
    assert summary["evaluated_feed_count"] == 128
    assert summary["racket_head_speed_m_s"] == summary["mean_racket_head_speed_m_s"]
    assert summary["racket_head_speed_m_s"] == summary["mean_contact_racket_head_speed_m_s"]
    assert summary["net_clearance_m"] == summary["mean_net_clearance_m"]
    assert summary["positive_outgoing_z_rate_on_hit"] == 1.0
    assert summary["mean_hit_outgoing_velocity_z_m_s"] == 2.0
    assert summary["promotion_gates"]["positive_outgoing_z_rate_on_hit"] is True

    short_summary = _stage3_evaluation_summary(
        [successful_episode.copy() for _ in range(127)],
        gate_config={},
        required_feed_count=128,
    )
    assert short_summary["promotion_gates"]["evaluated_feed_count"] is False
    assert short_summary["passed"] is False

    missing_guardrails = successful_episode.copy()
    missing_guardrails.pop("lab_diagnostics")
    unsafe_summary = _stage3_evaluation_summary(
        [missing_guardrails.copy() for _ in range(128)],
        gate_config={},
        required_feed_count=128,
    )
    assert unsafe_summary["promotion_gates"]["control_finite"] is False
    assert unsafe_summary["passed"] is False

    no_contact_speed = successful_episode.copy()
    no_contact_speed.pop("contact_racket_head_speed_m_s")
    no_contact_speed_summary = _stage3_evaluation_summary(
        [no_contact_speed.copy() for _ in range(128)],
        gate_config={},
        required_feed_count=128,
    )
    assert no_contact_speed_summary["promotion_gates"]["racket_head_speed_m_s"] is False
    assert no_contact_speed_summary["passed"] is False

    ood_tail = _stage3_evaluation_summary(
        [successful_episode.copy() for _ in range(128)],
        gate_config={"max_lab_state_ood_fraction_p95": 0.01},
        required_feed_count=128,
        lab_state_ood_values=[0.0] * 114 + [0.20] * 14,
    )
    assert ood_tail["lab_state_ood_fraction"] == 0.0
    assert ood_tail["lab_state_ood_fraction_p95"] == pytest.approx(0.20)
    assert ood_tail["lab_state_ood_fraction_max"] == pytest.approx(0.20)
    assert ood_tail["promotion_gates"]["lab_state_ood_fraction_p95"] is False
    assert ood_tail["passed"] is False


class _FakeMask:
    all_actuator_names = ["body_a", "right_finger", "body_b", "left_finger"]
    body_actuator_names = ["body_a", "body_b"]
    correction_actuator_names = ["right_finger"]
    neutral_actuator_names = ["left_finger"]


class _FakeRuntime:
    state_dim = 3
    latent_dim = 2
    action_dim = 2
    action_mask = _FakeMask()
    schema_hash = "fake-schema"

    def __init__(self) -> None:
        self.numpy_decoder_latent = None

    def prior_raw_numpy(self, state):
        np.testing.assert_allclose(state, np.array([1.0, 2.0, 3.0]))
        return np.array([0.25, -0.5]), np.array([0.0, 0.0])

    def decoder_numpy(self, state, latent):
        self.numpy_decoder_latent = np.asarray(latent)
        return np.array([latent[0], latent[1]])

    def prior_raw_jax(self, state):
        return jnp.broadcast_to(jnp.array([0.25, -0.5]), (*state.shape[:-1], 2)), jnp.zeros((*state.shape[:-1], 2))

    def decoder_jax(self, state, latent):
        return latent


def _router() -> Stage3ActionRouter:
    return Stage3ActionRouter(
        all_actuator_names=["body_a", "right_finger", "body_b", "left_finger"],
        body_actuator_names=["body_a", "body_b"],
        right_grip_actuator_names=["right_finger"],
        left_neutral_actuator_names=["left_finger"],
        expected_sizes=(2, 1, 1),
    )


def test_lab_controller_uses_latent_only_and_routes_all_actuators() -> None:
    runtime = _FakeRuntime()
    controller = Stage3LABController(
        runtime=runtime,
        router=_router(),
        right_grip_provider=ConstantGripProvider([0.7]),
        lambda_lab=0.5,
    )

    assert controller.task_action_size == 2
    output = controller.decode_numpy(
        lab_state=np.array([1.0, 2.0, 3.0]),
        raw_latent=np.array([2.0, -2.0]),
    )

    sigma = np.log1p(np.exp(0.0))
    expected_latent = np.array([0.25, -0.5]) + 0.5 * sigma * np.tanh([2.0, -2.0])
    np.testing.assert_allclose(runtime.numpy_decoder_latent, expected_latent, atol=1e-7)
    np.testing.assert_allclose(
        output.full_action,
        np.array([expected_latent[0], 0.7, expected_latent[1], 0.0]),
        atol=1e-7,
    )
    assert output.full_action.shape == (4,)

    with pytest.raises(ValueError, match="raw_latent"):
        controller.decode_numpy(
            lab_state=np.array([1.0, 2.0, 3.0]),
            raw_latent=np.zeros(4),
        )


def test_lab_controller_jax_matches_numpy_without_double_tanh() -> None:
    runtime = _FakeRuntime()
    controller = Stage3LABController(
        runtime=runtime,
        router=_router(),
        right_grip_provider=ConstantGripProvider([0.7]),
        lambda_lab=0.5,
    )
    state = jnp.array([[1.0, 2.0, 3.0]])
    raw = jnp.array([[2.0, -2.0]])

    output = controller.decode_jax(lab_state=state, raw_latent=raw)
    sigma = np.log1p(np.exp(0.0))
    once = np.array([0.25, -0.5]) + 0.5 * sigma * np.tanh([2.0, -2.0])
    twice = np.array([0.25, -0.5]) + 0.5 * sigma * np.tanh(np.tanh([2.0, -2.0]))
    np.testing.assert_allclose(np.asarray(output.latent[0]), once, atol=1e-7)
    assert not np.allclose(np.asarray(output.latent[0]), twice)
    np.testing.assert_allclose(np.asarray(output.full_action[0]), [once[0], 0.7, once[1], 0.0])


def test_router_rejects_unowned_or_416_bypass_actions() -> None:
    router = _router()
    np.testing.assert_allclose(
        router.merge_numpy(
            body_action=np.array([0.1, 0.2]),
            right_grip_action=np.array([0.3]),
            left_neutral_action=np.array([0.0]),
        ),
        np.array([0.1, 0.3, 0.2, 0.0]),
    )
    with pytest.raises(ValueError, match="body_action"):
        router.merge_numpy(
            body_action=np.zeros(4),
            right_grip_action=np.zeros(1),
            left_neutral_action=np.zeros(1),
        )


def test_stage3_curriculum_orders_fixed_jitter_full_bank_then_lambda() -> None:
    curriculum = Stage3Curriculum(
        lambda_start=0.25,
        lambda_end=0.5,
        fixed_feed_steps=10,
        jitter_feed_count=4,
        jitter_expand_steps=10,
        full_bank_expand_steps=20,
        lambda_expand_steps=10,
    )

    fixed = curriculum.values(env_steps=5, feed_bank_size=10)
    assert fixed.lambda_lab == pytest.approx(0.25)
    assert fixed.active_feed_count == 1

    jitter = curriculum.values(env_steps=15, feed_bank_size=10)
    assert jitter.lambda_lab == pytest.approx(0.25)
    assert jitter.active_feed_count == 3

    full = curriculum.values(env_steps=30, feed_bank_size=10)
    assert full.lambda_lab == pytest.approx(0.25)
    assert full.active_feed_count == 7

    radius = curriculum.values(env_steps=45, feed_bank_size=10)
    assert radius.lambda_lab == pytest.approx(0.375)
    assert radius.feed_fraction == pytest.approx(1.0)
    assert radius.active_feed_count == 10


def test_stage3_curriculum_holds_phase_boundary_until_training_gate_passes() -> None:
    curriculum = Stage3Curriculum(
        fixed_feed_steps=100,
        jitter_feed_count=4,
        jitter_expand_steps=100,
        full_bank_expand_steps=100,
        lambda_expand_steps=100,
    )
    failed_steps, failed = curriculum.advance(
        effective_steps=90,
        delta_steps=20,
        metrics={
            "episodes_finished": 10,
            "fall_rate": 0.01,
            "hit_rate": 0.49,
            "crossed_net_rate": 0.9,
        },
    )
    assert failed_steps == 99
    assert failed["checked"] is True
    assert failed["passed"] is False
    assert curriculum.phase(failed_steps) == "fixed_feed"

    passed_steps, passed = curriculum.advance(
        effective_steps=failed_steps,
        delta_steps=20,
        metrics={
            "episodes_finished": 10,
            "fall_rate": 0.01,
            "hit_rate": 0.51,
            "crossed_net_rate": 0.0,
        },
    )
    assert passed_steps == 119
    assert passed["passed"] is True
    assert curriculum.phase(passed_steps) == "intercept_jitter"


def test_stage3_curriculum_gate_fails_closed_without_completed_episodes() -> None:
    curriculum = Stage3Curriculum(fixed_feed_steps=10)
    steps, report = curriculum.advance(
        effective_steps=9,
        delta_steps=10,
        metrics={
            "episodes_finished": 0,
            "fall_rate": 0.0,
            "hit_rate": 1.0,
            "crossed_net_rate": 1.0,
        },
    )
    assert steps == 9
    assert report["passed"] is False


def test_optional_bounded_residual_is_name_masked_and_never_reaches_fingers() -> None:
    runtime = _FakeRuntime()
    mask = BoundedResidualMask(
        body_actuator_names=["body_a", "body_b"],
        residual_actuator_names=["body_b"],
        alpha=0.1,
    )
    controller = Stage3LABController(
        runtime=runtime,
        router=_router(),
        right_grip_provider=ConstantGripProvider([0.7]),
        lambda_lab=0.5,
        bounded_residual_mask=mask,
    )

    assert controller.latent_action_size == 2
    assert controller.residual_action_size == 1
    assert controller.task_action_size == 3
    output = controller.decode_task_numpy(
        lab_state=np.array([1.0, 2.0, 3.0]),
        task_action=np.array([0.0, 0.0, 20.0]),
    )
    assert output.body_action[1] == pytest.approx(-0.4, abs=1e-6)
    assert output.full_action[1] == pytest.approx(0.7)
    assert output.full_action[3] == pytest.approx(0.0)

    with pytest.raises(ValueError, match="finger"):
        BoundedResidualMask(
            body_actuator_names=["body", "FDS5"],
            residual_actuator_names=["FDS5"],
        )
    with pytest.raises(ValueError, match=r"\[0, 0.10\]"):
        BoundedResidualMask(
            body_actuator_names=["body"],
            residual_actuator_names=["body"],
            alpha=0.11,
        )


_GROUPED_BODY_NAMES: tuple[str, ...] = (
    "glut_max1_r",
    "SUP",
    "BRD",
    "TRIlong",
    "BIClong",
    "DELT1",
    "DELT2",
    "soleus_r",
)
_NON_RIGHT_ARM_INDICES: tuple[int, ...] = (0, 7)


def _grouped_mask(
    *,
    wrist_alpha: float = 0.05,
    elbow_alpha: float = 0.03,
    shoulder_alpha: float = 0.02,
) -> BoundedResidualMask:
    return BoundedResidualMask(
        body_actuator_names=_GROUPED_BODY_NAMES,
        groups=(
            BoundedResidualGroup("wrist_forearm", ("SUP", "BRD"), wrist_alpha),
            BoundedResidualGroup("elbow_forearm", ("TRIlong", "BIClong"), elbow_alpha),
            BoundedResidualGroup("shoulder", ("DELT1", "DELT2"), shoulder_alpha),
        ),
    )


def test_grouped_bounded_residual_leaves_zero_correction_bit_identical() -> None:
    mask = _grouped_mask()
    assert mask.residual_size == 6
    decoder = np.array([0.3, -0.7, 0.11, 0.42, -0.9, 0.05, -0.25, 0.6], dtype=np.float32)

    corrected = mask.apply_numpy(decoder, np.zeros(6, dtype=np.float32))

    np.testing.assert_array_equal(corrected, decoder)
    np.testing.assert_array_equal(
        np.asarray(mask.apply_jax(jnp.asarray(decoder), jnp.zeros(6, dtype=jnp.float32))),
        decoder,
    )


def test_grouped_bounded_residual_alphas_and_channels_stay_independent() -> None:
    decoder = np.zeros(8)
    raw = np.array([0.8, -0.4, 0.6, -0.2, 0.5, -0.9])
    baseline = _grouped_mask().apply_numpy(decoder, raw)
    wrist_only = _grouped_mask(wrist_alpha=0.10).apply_numpy(decoder, raw)
    shoulder_only = _grouped_mask(shoulder_alpha=0.01).apply_numpy(decoder, raw)

    wrist_channels = [1, 2]
    elbow_channels = [3, 4]
    shoulder_channels = [5, 6]

    # Raising the wrist alpha moves only wrist channels.
    assert not np.allclose(wrist_only[wrist_channels], baseline[wrist_channels])
    np.testing.assert_array_equal(
        wrist_only[elbow_channels + shoulder_channels],
        baseline[elbow_channels + shoulder_channels],
    )
    # Lowering the shoulder alpha moves only shoulder channels.
    assert not np.allclose(shoulder_only[shoulder_channels], baseline[shoulder_channels])
    np.testing.assert_array_equal(
        shoulder_only[wrist_channels + elbow_channels],
        baseline[wrist_channels + elbow_channels],
    )
    # Non-right-arm channels never move, whatever the correction is.
    for corrected in (baseline, wrist_only, shoulder_only):
        np.testing.assert_array_equal(corrected[list(_NON_RIGHT_ARM_INDICES)], decoder[list(_NON_RIGHT_ARM_INDICES)])


def test_grouped_bounded_residual_applies_each_group_alpha_with_a_single_tanh() -> None:
    mask = _grouped_mask()
    raw = np.array([2.5, -2.5, 3.0, -3.0, 4.0, -4.0])
    corrected = mask.apply_numpy(np.zeros(8), raw)

    expected = np.concatenate(
        [
            0.05 * np.tanh(raw[:2]),
            0.03 * np.tanh(raw[2:4]),
            0.02 * np.tanh(raw[4:]),
        ]
    )
    np.testing.assert_allclose(corrected[1:7], expected, atol=1e-12)
    # tanh applied twice would shrink every channel further.
    assert not np.allclose(corrected[1], 0.05 * np.tanh(np.tanh(raw[0])))
    np.testing.assert_allclose(
        np.asarray(mask.apply_jax(jnp.zeros(8), jnp.asarray(raw))),
        corrected,
        atol=1e-6,
    )


def test_grouped_bounded_residual_rejects_overlapping_correction_actuators() -> None:
    with pytest.raises(ValueError, match=r"disjoint actuators: 'BRD'"):
        BoundedResidualMask(
            body_actuator_names=_GROUPED_BODY_NAMES,
            groups=(
                BoundedResidualGroup("wrist_forearm", ("SUP", "BRD"), 0.05),
                BoundedResidualGroup("elbow_forearm", ("BRD", "TRIlong"), 0.03),
            ),
        )
    with pytest.raises(ValueError, match="finger"):
        BoundedResidualMask(
            body_actuator_names=("SUP", "FDS5"),
            groups=(BoundedResidualGroup("shoulder", ("FDS5",), 0.02),),
        )
    with pytest.raises(ValueError, match=r"'shoulder' alpha must lie in \[0, 0.10\]"):
        BoundedResidualGroup("shoulder", ("DELT1",), 0.11)
    with pytest.raises(ValueError, match="absent from the body decoder"):
        BoundedResidualMask(
            body_actuator_names=("SUP",),
            groups=(BoundedResidualGroup("shoulder", ("DELT1",), 0.02),),
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        BoundedResidualMask(
            body_actuator_names=_GROUPED_BODY_NAMES,
            residual_actuator_names=("SUP",),
            groups=(BoundedResidualGroup("shoulder", ("DELT1",), 0.02),),
        )


def test_grouped_bounded_residual_schema_hash_is_group_aware_and_v1_stays_reachable() -> None:
    legacy = BoundedResidualMask(
        body_actuator_names=_GROUPED_BODY_NAMES[:1] + DEFAULT_RIGHT_WRIST_FOREARM_RESIDUAL_NAMES
    )
    assert legacy.schema_version == "bounded_body_residual_v1"
    assert legacy.residual_actuator_names == DEFAULT_RIGHT_WRIST_FOREARM_RESIDUAL_NAMES
    assert legacy.alpha == pytest.approx(0.05)

    grouped = _grouped_mask()
    assert grouped.schema_version == "bounded_body_residual_v2"
    assert grouped.schema_hash != legacy.schema_hash
    assert grouped.schema_hash != _grouped_mask(shoulder_alpha=0.01).schema_hash
    assert grouped.group_manifest == (
        {"name": "wrist_forearm", "actuator_names": ["SUP", "BRD"], "alpha": 0.05, "dim": 2},
        {"name": "elbow_forearm", "actuator_names": ["TRIlong", "BIClong"], "alpha": 0.03, "dim": 2},
        {"name": "shoulder", "actuator_names": ["DELT1", "DELT2"], "alpha": 0.02, "dim": 2},
    )
    assert grouped.group_slice("elbow_forearm") == slice(2, 4)


def test_bounded_residual_config_accepts_groups_and_legacy_flat_form() -> None:
    body = _GROUPED_BODY_NAMES + tuple(
        name for name in DEFAULT_RIGHT_WRIST_FOREARM_RESIDUAL_NAMES if name not in _GROUPED_BODY_NAMES
    )

    assert bounded_residual_mask_from_config(None, body_actuator_names=body) is None
    assert bounded_residual_mask_from_config({"enabled": False}, body_actuator_names=body) is None

    flat = bounded_residual_mask_from_config({"enabled": True, "alpha": 0.05}, body_actuator_names=body)
    assert flat is not None
    assert flat.schema_version == "bounded_body_residual_v1"
    assert flat.residual_actuator_names == DEFAULT_RIGHT_WRIST_FOREARM_RESIDUAL_NAMES
    assert flat.schema_hash == BoundedResidualMask(body_actuator_names=body).schema_hash

    grouped = bounded_residual_mask_from_config(
        {
            "enabled": True,
            "groups": {
                "wrist_forearm": {"actuator_names": ["SUP", "BRD"], "alpha": 0.05},
                "elbow_forearm": {"actuator_names": ["TRIlong"], "alpha": 0.03},
                "shoulder": {"actuator_names": ["DELT1"], "alpha": 0.02},
            },
        },
        body_actuator_names=body,
    )
    assert grouped is not None
    assert grouped.residual_actuator_names == ("SUP", "BRD", "TRIlong", "DELT1")
    np.testing.assert_allclose(grouped.channel_alphas, [0.05, 0.05, 0.03, 0.02])

    # Empty-by-default elbow/shoulder groups never widen the production set.
    default_groups = bounded_residual_mask_from_config(
        {"enabled": True, "groups": {"wrist_forearm": {}, "elbow_forearm": {}, "shoulder": {}}},
        body_actuator_names=body,
    )
    assert default_groups is not None
    assert default_groups.residual_actuator_names == DEFAULT_RIGHT_WRIST_FOREARM_RESIDUAL_NAMES

    for payload, message in (
        ({"enabled": True, "actuator_names": ["TRIlong"]}, "subset of the verified wrist_forearm"),
        ({"enabled": True, "groups": {"bogus": {}}}, "unknown bounded_residual correction groups"),
        (
            {"enabled": True, "groups": {"shoulder": {"actuator_names": ["DELT1"]}}, "alpha": 0.02},
            "cannot be combined with the flat",
        ),
        ({"enabled": True, "groups": {"shoulder": {"gain": 1.0}}}, "unknown bounded_residual.groups"),
        ({"enabled": True, "groups": {"shoulder": {}}}, "at least one actuator"),
        ({"enabled": True, "scale": 1.0}, "unknown bounded_residual keys"),
        (
            {"enabled": True, "groups": {"shoulder": {"actuator_names": ["DELT1"], "alpha": 0.5}}},
            r"'shoulder' alpha must lie in \[0, 0.10\]",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            bounded_residual_mask_from_config(payload, body_actuator_names=body)


def test_grouped_bounded_residual_reaches_the_controller_manifest() -> None:
    router = Stage3ActionRouter(
        all_actuator_names=["body_a", "SUP", "DELT1", "right_finger", "left_finger"],
        body_actuator_names=["body_a", "SUP", "DELT1"],
        right_grip_actuator_names=["right_finger"],
        left_neutral_actuator_names=["left_finger"],
        expected_sizes=(3, 1, 1),
    )

    class _Mask:
        all_actuator_names = tuple(router.all_actuator_names)
        body_actuator_names = tuple(router.body_actuator_names)
        correction_actuator_names = tuple(router.right_grip_actuator_names)
        neutral_actuator_names = tuple(router.left_neutral_actuator_names)

    class _Runtime(_FakeRuntime):
        action_dim = 3
        action_mask = _Mask()

        def decoder_numpy(self, state, latent):
            return np.array([0.0, latent[0], latent[1]])

    mask = BoundedResidualMask(
        body_actuator_names=router.body_actuator_names,
        groups=(
            BoundedResidualGroup("wrist_forearm", ("SUP",), 0.05),
            BoundedResidualGroup("shoulder", ("DELT1",), 0.02),
        ),
    )
    controller = Stage3LABController(
        runtime=_Runtime(),
        router=router,
        right_grip_provider=ConstantGripProvider([0.7]),
        lambda_lab=0.5,
        bounded_residual_mask=mask,
    )

    assert controller.residual_action_size == 2
    assert controller.task_action_size == controller.latent_action_size + 2
    manifest = controller.control_manifest
    assert manifest["bounded_residual_dim"] == 2
    assert manifest["bounded_residual_schema_hash"] == mask.schema_hash
    assert manifest["bounded_residual_groups"] == [
        {"name": "wrist_forearm", "actuator_names": ["SUP"], "alpha": 0.05, "dim": 1},
        {"name": "shoulder", "actuator_names": ["DELT1"], "alpha": 0.02, "dim": 1},
    ]

    output = controller.decode_task_numpy(
        lab_state=np.array([1.0, 2.0, 3.0]),
        task_action=np.array([0.0, 0.0, 1.5, -1.5]),
    )
    assert output.body_action[1] == pytest.approx(0.25 + 0.05 * np.tanh(1.5), abs=1e-9)
    assert output.body_action[2] == pytest.approx(-0.5 + 0.02 * np.tanh(-1.5), abs=1e-9)
    assert output.body_action[0] == pytest.approx(0.0)


def test_teacher_ctrlrange_requires_v2_unit_muscles_and_updates_only_body() -> None:
    class Model:
        actuator_ctrlrange = np.asarray([[-1.0, 1.0], [-2.0, 2.0], [-3.0, 3.0], [-4.0, 4.0]])
        actuator_ctrllimited = np.zeros(4, dtype=bool)

    runtime = _FakeRuntime()
    runtime.body_ctrlrange = np.asarray([[-0.2, 0.8], [-0.4, 0.6]])
    with pytest.raises(ValueError, match=r"exactly \[0,1\]"):
        Stage3LABController(
            runtime=runtime,
            router=_router(),
            right_grip_provider=ConstantGripProvider([0.7]),
        )
    runtime.body_ctrlrange = np.asarray([[0.0, 1.0], [0.0, 1.0]])
    controller = Stage3LABController(
        runtime=runtime,
        router=_router(),
        right_grip_provider=ConstantGripProvider([0.7]),
    )
    digest = apply_teacher_body_ctrlrange(Model, controller)

    np.testing.assert_allclose(Model.actuator_ctrlrange[[0, 2]], runtime.body_ctrlrange)
    np.testing.assert_allclose(Model.actuator_ctrlrange[[1, 3]], [[-2.0, 2.0], [-4.0, 4.0]])
    np.testing.assert_array_equal(Model.actuator_ctrllimited, [True, False, True, False])
    assert isinstance(digest, str) and len(digest) == 64


class _FullRuntime:
    state_dim = 3
    latent_dim = 2
    action_dim = 354
    schema_hash = "full-runtime"
    body_ctrlrange = np.tile(np.asarray([[0.0, 1.0]]), (354, 1))
    ctrlrange_schema_hash = "full-runtime-ctrlrange"

    def prior_raw_numpy(self, state):
        return np.zeros((*np.asarray(state).shape[:-1], 2)), np.zeros((*np.asarray(state).shape[:-1], 2))

    def decoder_numpy(self, state, latent):
        return np.zeros((*np.asarray(state).shape[:-1], 354))

    def prior_raw_jax(self, state):
        return jnp.zeros((*state.shape[:-1], 2)), jnp.zeros((*state.shape[:-1], 2))

    def decoder_jax(self, state, latent):
        return jnp.zeros((*state.shape[:-1], 354))


class _FullStateBuilder:
    expected_state_dim = 3
    schema_hash = "full-state"

    def build_numpy(self, *, model, data, phase):
        return np.asarray([data.qpos[2], data.qvel[0], phase], dtype=float)

    def build_jax(self, *, data, phase):
        return jnp.stack([data.qpos[:, 2], data.qvel[:, 0], phase], axis=-1)


def test_cpu_incoming_env_exposes_only_latent_and_never_a_354_bypass() -> None:
    scene = default_incoming_scene_path()
    if not scene.is_file():
        pytest.skip("incoming scene has not been built")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene))
    router = Stage3ActionRouter.from_model(model)
    controller = Stage3LABController(
        runtime=_FullRuntime(),
        router=router,
        lambda_lab=0.25,
    )
    curriculum = Stage3Curriculum()
    lab_env = IncomingShuttleHitEnv(
        scene,
        feed_bank=build_feed_bank(1, seed=2),
        control_substeps=1,
        max_episode_steps=2,
        terminate_on_body_fall=True,
        lab_controller=controller,
        lab_state_builder=_FullStateBuilder(),
        curriculum=curriculum,
        filter_finger_observation=False,
    )
    direct_env = IncomingShuttleHitEnv(
        scene,
        feed_bank=build_feed_bank(1, seed=2),
        control_substeps=1,
        max_episode_steps=2,
        terminate_on_body_fall=False,
    )

    assert lab_env.action_size == 2
    assert lab_env.full_action_size == 354
    assert lab_env.observation_size == direct_env.observation_size
    lab_env.reset(feed_index=0)
    _obs, _reward, _terminated, _truncated, info = lab_env.step(np.zeros(2))
    output = lab_env._last_lab_output
    assert output.full_action.shape == (354,)
    assert output.right_grip_action.shape == (0,)
    assert output.left_neutral_action.shape == (0,)
    assert info["lab_state_schema_hash"] == "full-state"
    assert "body_action_saturation_fraction" in info
    assert "full_action_saturation_fraction" in info
    with pytest.raises(ValueError, match="action must have shape"):
        lab_env.step(np.zeros(416))

    mjx_env = IncomingHitMjxEnv(
        scene,
        build_feed_bank(1, seed=2),
        impl="jax",
        control_substeps=1,
        max_episode_steps=2,
        lab_controller=controller,
        lab_state_builder=_FullStateBuilder(),
        curriculum=curriculum,
        filter_finger_observation=False,
    )
    assert mjx_env.action_size == 2
    assert mjx_env.control_hash == lab_env.control_hash
    different_curriculum_env = IncomingHitMjxEnv(
        scene,
        build_feed_bank(1, seed=2),
        impl="jax",
        control_substeps=1,
        max_episode_steps=2,
        lab_controller=controller,
        lab_state_builder=_FullStateBuilder(),
        curriculum=replace(curriculum, lambda_end=0.75),
        filter_finger_observation=False,
    )
    assert different_curriculum_env.control_hash != mjx_env.control_hash
    stored_cpu_env = IncomingShuttleHitEnv(
        scene,
        feed_bank=build_feed_bank(1, seed=2),
        control_substeps=1,
        max_episode_steps=2,
        reward_weights={"return_clearance": 1.0},
        min_return_net_clearance_m=0.20,
        direction_reward_mode="signed_projection",
        lab_controller=controller,
        lab_state_builder=_FullStateBuilder(),
        curriculum=curriculum,
        curriculum_feed_order="stored",
        filter_finger_observation=False,
    )
    stored_mjx_env = IncomingHitMjxEnv(
        scene,
        build_feed_bank(1, seed=2),
        impl="jax",
        control_substeps=1,
        max_episode_steps=2,
        reward_weights={"return_clearance": 1.0},
        min_return_net_clearance_m=0.20,
        direction_reward_mode="signed_projection",
        lab_controller=controller,
        lab_state_builder=_FullStateBuilder(),
        curriculum=curriculum,
        curriculum_feed_order="stored",
        filter_finger_observation=False,
    )
    assert stored_cpu_env.control_hash == stored_mjx_env.control_hash
    assert stored_cpu_env.control_hash != lab_env.control_hash
    assert (
        stored_cpu_env.control_manifest["environment_abi"]["reward_semantics"]
        == "incoming_hit_signed_task_direction_v8"
    )
    rebound_cpu_env = IncomingShuttleHitEnv(
        scene,
        feed_bank=build_feed_bank(1, seed=2),
        control_substeps=1,
        max_episode_steps=2,
        reward_weights={"return_clearance": 1.0},
        min_return_net_clearance_m=0.20,
        direction_reward_mode="signed_projection",
        hit_event_mode="event_rebound",
        racket_guidance_mode="counterfactual_rebound",
        lab_controller=controller,
        lab_state_builder=_FullStateBuilder(),
        curriculum=curriculum,
        curriculum_feed_order="stored",
        filter_finger_observation=False,
    )
    rebound_mjx_env = IncomingHitMjxEnv(
        scene,
        build_feed_bank(1, seed=2),
        impl="jax",
        control_substeps=1,
        max_episode_steps=2,
        reward_weights={"return_clearance": 1.0},
        min_return_net_clearance_m=0.20,
        direction_reward_mode="signed_projection",
        hit_event_mode="event_rebound",
        racket_guidance_mode="counterfactual_rebound",
        lab_controller=controller,
        lab_state_builder=_FullStateBuilder(),
        curriculum=curriculum,
        curriculum_feed_order="stored",
        filter_finger_observation=False,
    )
    assert rebound_cpu_env.control_hash == rebound_mjx_env.control_hash
    assert rebound_cpu_env.control_hash != stored_cpu_env.control_hash
    assert (
        rebound_cpu_env.control_manifest["environment_abi"]["reward_semantics"]
        == "incoming_hit_counterfactual_rebound_guidance_v10"
    )
    clearance_cpu_env = IncomingShuttleHitEnv(
        scene,
        feed_bank=build_feed_bank(1, seed=2),
        control_substeps=1,
        max_episode_steps=2,
        reward_weights={"return_clearance": 1.0},
        min_return_net_clearance_m=0.20,
        direction_reward_mode="signed_projection",
        hit_event_mode="event_rebound",
        racket_guidance_mode="counterfactual_clearance_priority",
        lab_controller=controller,
        lab_state_builder=_FullStateBuilder(),
        curriculum=curriculum,
        curriculum_feed_order="stored",
        filter_finger_observation=False,
    )
    clearance_mjx_env = IncomingHitMjxEnv(
        scene,
        build_feed_bank(1, seed=2),
        impl="jax",
        control_substeps=1,
        max_episode_steps=2,
        reward_weights={"return_clearance": 1.0},
        min_return_net_clearance_m=0.20,
        direction_reward_mode="signed_projection",
        hit_event_mode="event_rebound",
        racket_guidance_mode="counterfactual_clearance_priority",
        lab_controller=controller,
        lab_state_builder=_FullStateBuilder(),
        curriculum=curriculum,
        curriculum_feed_order="stored",
        filter_finger_observation=False,
    )
    assert clearance_cpu_env.control_hash == clearance_mjx_env.control_hash
    assert clearance_cpu_env.control_hash != rebound_cpu_env.control_hash
    assert (
        clearance_cpu_env.control_manifest["environment_abi"]["reward_semantics"]
        == "incoming_hit_counterfactual_clearance_priority_v11"
    )
    inverse_cpu_env = IncomingShuttleHitEnv(
        scene,
        feed_bank=build_feed_bank(1, seed=2),
        control_substeps=1,
        max_episode_steps=2,
        reward_weights={"return_clearance": 1.0},
        min_return_net_clearance_m=0.20,
        direction_reward_mode="signed_projection",
        hit_event_mode="event_rebound",
        racket_guidance_mode="inverse_impact_target",
        inverse_target_speed_m_s=12.0,
        inverse_velocity_softness_m_s=6.0,
        lab_controller=controller,
        lab_state_builder=_FullStateBuilder(),
        curriculum=curriculum,
        curriculum_feed_order="stored",
        filter_finger_observation=False,
    )
    inverse_mjx_env = IncomingHitMjxEnv(
        scene,
        build_feed_bank(1, seed=2),
        impl="jax",
        control_substeps=1,
        max_episode_steps=2,
        reward_weights={"return_clearance": 1.0},
        min_return_net_clearance_m=0.20,
        direction_reward_mode="signed_projection",
        hit_event_mode="event_rebound",
        racket_guidance_mode="inverse_impact_target",
        inverse_target_speed_m_s=12.0,
        inverse_velocity_softness_m_s=6.0,
        lab_controller=controller,
        lab_state_builder=_FullStateBuilder(),
        curriculum=curriculum,
        curriculum_feed_order="stored",
        filter_finger_observation=False,
    )
    assert inverse_cpu_env.control_hash == inverse_mjx_env.control_hash
    assert inverse_cpu_env.control_hash != clearance_cpu_env.control_hash
    inverse_constraints = inverse_cpu_env.control_manifest["environment_abi"]["return_constraints"]
    assert inverse_constraints["inverse_target_speed_m_s"] == 12.0
    assert inverse_constraints["inverse_velocity_softness_m_s"] == 6.0
    assert (
        inverse_cpu_env.control_manifest["environment_abi"]["reward_semantics"]
        == "incoming_hit_inverse_impact_target_guidance_v12"
    )
    decomposed_cpu_env = IncomingShuttleHitEnv(
        scene,
        feed_bank=build_feed_bank(1, seed=2),
        control_substeps=1,
        max_episode_steps=2,
        reward_weights={"return_clearance": 1.0},
        min_return_net_clearance_m=0.20,
        direction_reward_mode="signed_projection",
        hit_event_mode="event_rebound",
        racket_guidance_mode="inverse_impact_decomposed",
        inverse_target_speed_m_s=12.0,
        inverse_velocity_softness_m_s=6.0,
        lab_controller=controller,
        lab_state_builder=_FullStateBuilder(),
        curriculum=curriculum,
        curriculum_feed_order="stored",
        filter_finger_observation=False,
    )
    decomposed_mjx_env = IncomingHitMjxEnv(
        scene,
        build_feed_bank(1, seed=2),
        impl="jax",
        control_substeps=1,
        max_episode_steps=2,
        reward_weights={"return_clearance": 1.0},
        min_return_net_clearance_m=0.20,
        direction_reward_mode="signed_projection",
        hit_event_mode="event_rebound",
        racket_guidance_mode="inverse_impact_decomposed",
        inverse_target_speed_m_s=12.0,
        inverse_velocity_softness_m_s=6.0,
        lab_controller=controller,
        lab_state_builder=_FullStateBuilder(),
        curriculum=curriculum,
        curriculum_feed_order="stored",
        filter_finger_observation=False,
    )
    assert decomposed_cpu_env.control_hash == decomposed_mjx_env.control_hash
    assert decomposed_cpu_env.control_hash != inverse_cpu_env.control_hash
    assert (
        decomposed_cpu_env.control_manifest["environment_abi"]["reward_semantics"]
        == "incoming_hit_inverse_impact_decomposed_quality_v16"
    )
    full_action, _output = mjx_env._compose_action(
        None,
        SimpleNamespace(lab_state=jnp.zeros((1, 3)), lambda_lab=jnp.asarray(0.25)),
        jnp.zeros((1, 2)),
    )
    assert full_action.shape == (1, 354)


def test_production_stage3_spec_is_latent_only_and_blocks_legacy_cpu_ppo(
    tmp_path: Path,
) -> None:
    paths = load_incoming_hit_spec(REPO_ROOT / "experiments/posttrain/incoming_shuttle_hit_v1.yaml")
    config = paths.stage3_lab
    assert config["enabled"] is True
    assert "latent_stage2_racket_raw_smooth_v1" in str(config["latent_checkpoint_dir"])
    assert config["filter_finger_observation"] is False
    assert config["hand_fixture"] == {
        "mode": "removed",
        "policy_enabled": False,
        "observations_enabled": False,
    }
    assert config["racket_attachment"]["mode"] == "exact_child"
    assert config["bounded_residual"]["enabled"] is False
    residual_groups = config["bounded_residual"]["groups"]
    assert set(residual_groups) == {"wrist_forearm", "elbow_forearm", "shoulder"}
    assert all(group["alpha"] <= 0.10 for group in residual_groups.values())
    assert tuple(residual_groups["wrist_forearm"]["actuator_names"]) == (
        "SUP",
        "BRA",
        "BRD",
        "ECRL",
        "ECRB",
        "ECU",
        "FCR",
        "FCU",
        "PT",
        "PQ",
    )
    # Elbow/shoulder correction stays empty until a config opts in explicitly.
    assert residual_groups["elbow_forearm"]["actuator_names"] == []
    assert residual_groups["shoulder"]["actuator_names"] == []
    assert config["curriculum"]["lambda_start"] == pytest.approx(0.25)
    assert config["curriculum"]["fixed_feed_steps"] == 2_000_000
    assert config["curriculum"]["jitter_feed_count"] == 16
    assert config["curriculum"]["gate_min_no_fall_rate"] == pytest.approx(0.95)
    assert paths.ppo_overrides["minibatch_size"] == 256
    assert paths.evaluation["heldout_feed_count"] == 128
    preflight_report = preflight(paths, out_dir=tmp_path / "preflight")
    assert preflight_report["passed"] is True
    assert preflight_report["attachment_contract_passed"] is True
    assert preflight_report["configuration_contract_passed"] is True
    assert preflight_report["finger_joint_count"] == 0
    assert preflight_report["finger_actuator_count"] == 0
    assert preflight_report["action_router"]["partition_sizes"] == [354, 0, 0]

    with pytest.raises(ValueError, match="latent-only"):
        _run_ppo(
            paths,
            out_dir=tmp_path,
            total_steps=1,
            rollout_steps=1,
        )


def test_v16_spec_preserves_signed_clearance_and_ppo_controls() -> None:
    paths = load_incoming_hit_spec(REPO_ROOT / "experiments/posttrain/incoming_shuttle_hit_decomposed_quality_v16.yaml")

    assert paths.return_constraints["clearance_reward_mode"] == "signed_centered"
    assert paths.return_constraints["racket_guidance_mode"] == "inverse_impact_decomposed"
    assert paths.ppo_overrides["action_std_init"] == pytest.approx(0.10)
    assert paths.ppo_overrides["learning_rate"] == pytest.approx(0.00002)
    assert paths.ppo_overrides["entropy_coef"] == pytest.approx(0.00010)


def test_v17_spec_rejects_rescaling_the_inherited_wrist_actor_mean() -> None:
    spec = REPO_ROOT / "experiments/posttrain/incoming_shuttle_hit_wrist_focus_v17.yaml"
    paths = load_incoming_hit_spec(spec)
    model_path = default_incoming_scene_path()
    if not model_path.is_file():
        pytest.skip("incoming scene has not been built")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    with pytest.raises(ValueError, match="inherited actor mean"):
        _policy_update_contract(paths, model)


def test_v18_spec_limits_learning_to_constant_authority_wrist_outputs() -> None:
    spec = REPO_ROOT / "experiments/posttrain/incoming_shuttle_hit_wrist_delta_v18.yaml"
    paths = load_incoming_hit_spec(spec)
    model_path = default_incoming_scene_path()
    if not model_path.is_file():
        pytest.skip("incoming scene has not been built")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    contract = _policy_update_contract(paths, model)

    expected_names = ["SUP", "ECRL", "ECRB", "ECU", "FCR", "FCU", "PL", "PT", "PQ"]
    assert contract["mode"] == "distal_output_head_only"
    assert contract["freeze_observation_normalizer"] is True
    assert contract["trainable_actuator_names"] == expected_names
    assert contract["trainable_action_indices"] == [229, 234, 235, 236, 237, 238, 239, 240, 241]
    assert contract["trainable_action_count"] == 9
    assert contract["full_action_count"] == 354
    assert contract["constant_residual_scale"] == pytest.approx(0.25)
    assert len(contract["contract_sha256"]) == 64
    assert "residual_scale_overrides" not in paths.stage3_direct
    assert "residual_scale_schedule" not in paths.stage3_direct
    assert paths.return_constraints["racket_guidance_mode"] == "inverse_impact_decomposed"


def test_v19_spec_freezes_body_exploration_and_preserves_reward_hierarchy() -> None:
    spec = REPO_ROOT / "experiments/posttrain/incoming_shuttle_hit_wrist_hierarchical_v19.yaml"
    paths = load_incoming_hit_spec(spec)
    model_path = default_incoming_scene_path()
    if not model_path.is_file():
        pytest.skip("incoming scene has not been built")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    contract = _policy_update_contract(paths, model)

    assert contract["mode"] == "distal_output_head_only"
    assert contract["frozen_action_std"] == pytest.approx(0.001)
    assert contract["trainable_action_count"] == 9
    assert paths.return_constraints["direction_reward_mode"] == "positive_projection"
    assert paths.return_constraints["clearance_reward_mode"] == "positive_score"
    assert paths.return_constraints["hit_event_mode"] == "event_rebound"
    assert paths.return_constraints["racket_velocity_direction_fraction"] == pytest.approx(0.15)
    # Even a minimum-speed real event followed by an invalid crossing remains
    # strictly better than missing, while a legal return earns a large margin.
    minimum_bad_hit = paths.reward_weights["hit_bonus"] - paths.reward_weights["invalid_net_crossing"]
    assert minimum_bad_hit > -paths.reward_weights["miss"]
    assert paths.reward_weights["crossed_net"] > paths.reward_weights["invalid_net_crossing"]
    assert paths.reward_weights["body_fall"] > (
        paths.reward_weights["hit_bonus"]
        + paths.reward_weights["hit_speed"]
        + paths.reward_weights["return_direction"]
        + paths.reward_weights["return_clearance"]
        + paths.reward_weights["crossed_net"]
        + paths.reward_weights["landing_region"]
    )

    env = IncomingHitMjxEnv(
        model_path,
        feed_bank=[_synthetic_feed(0.0)],
        reward_weights=paths.reward_weights,
        direction_reward_mode=paths.return_constraints["direction_reward_mode"],
        clearance_reward_mode=paths.return_constraints["clearance_reward_mode"],
        hit_event_mode=paths.return_constraints["hit_event_mode"],
        racket_guidance_mode=paths.return_constraints["racket_guidance_mode"],
        inverse_target_speed_m_s=paths.return_constraints["inverse_target_speed_m_s"],
        inverse_velocity_softness_m_s=paths.return_constraints["inverse_velocity_softness_m_s"],
        racket_velocity_direction_fraction=paths.return_constraints["racket_velocity_direction_fraction"],
    )
    assert env.control_manifest["environment_abi"]["reward_semantics"] == (
        "incoming_hit_wrist_hierarchical_quality_v19"
    )


def test_v22_contact_guidance_softness_is_bound_into_cpu_and_mjx_abi() -> None:
    spec = REPO_ROOT / "experiments/posttrain/incoming_shuttle_hit_right_arm_contact_shaped_v22.yaml"
    paths = load_incoming_hit_spec(spec)
    model_path = default_incoming_scene_path()
    if not model_path.is_file():
        pytest.skip("incoming scene has not been built")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    contract = _policy_update_contract(paths, model)
    assert contract["trainable_action_indices"] == list(range(210, 242))
    assert contract["trainable_action_count"] == 32
    assert contract["frozen_action_std"] == pytest.approx(0.001)
    assert paths.reward_weights["approach"] == pytest.approx(0.0)
    assert paths.return_constraints["shuttle_proximity_softness_m"] == pytest.approx(0.85)
    assert paths.return_constraints["timed_intercept_softness_m"] == pytest.approx(0.80)
    assert paths.return_constraints["direction_distance_softness_m"] == pytest.approx(0.75)

    common = {
        "control_substeps": 1,
        "max_episode_steps": 2,
        "reward_weights": paths.reward_weights,
        "min_return_net_clearance_m": paths.return_constraints["min_clearance_m"],
        "desired_return_up_component": paths.return_constraints["desired_up_component"],
        "ballistic_return_score_softness_m": paths.return_constraints["ballistic_score_softness_m"],
        "shuttle_proximity_softness_m": paths.return_constraints["shuttle_proximity_softness_m"],
        "timed_intercept_softness_m": paths.return_constraints["timed_intercept_softness_m"],
        "direction_distance_softness_m": paths.return_constraints["direction_distance_softness_m"],
        "racket_velocity_direction_fraction": paths.return_constraints["racket_velocity_direction_fraction"],
        "direction_reward_mode": paths.return_constraints["direction_reward_mode"],
        "clearance_reward_mode": paths.return_constraints["clearance_reward_mode"],
        "hit_event_mode": paths.return_constraints["hit_event_mode"],
        "racket_guidance_mode": paths.return_constraints["racket_guidance_mode"],
        "inverse_target_speed_m_s": paths.return_constraints["inverse_target_speed_m_s"],
        "inverse_velocity_softness_m_s": paths.return_constraints["inverse_velocity_softness_m_s"],
        "curriculum_feed_order": "stored",
        "filter_finger_observation": False,
    }
    feed = [_synthetic_feed(0.0)]
    cpu_env = IncomingShuttleHitEnv(model_path, feed_bank=feed, **common)
    mjx_env = IncomingHitMjxEnv(model_path, feed, impl="jax", **common)
    assert cpu_env.control_hash == mjx_env.control_hash
    constraints = cpu_env.control_manifest["environment_abi"]["return_constraints"]
    assert constraints["shuttle_proximity_softness_m"] == pytest.approx(0.85)
    assert constraints["timed_intercept_softness_m"] == pytest.approx(0.80)
    assert constraints["direction_distance_softness_m"] == pytest.approx(0.75)

    with pytest.raises(ValueError, match="shuttle_proximity_softness_m"):
        IncomingShuttleHitEnv(
            model_path,
            feed_bank=feed,
            shuttle_proximity_softness_m=0.0,
        )


def test_v23_bounded_contact_spec_has_a_fail_closed_reward_hierarchy() -> None:
    spec = REPO_ROOT / "experiments/posttrain/incoming_shuttle_hit_right_arm_bounded_contact_v23.yaml"
    paths = load_incoming_hit_spec(spec)
    model_path = default_incoming_scene_path()
    if not model_path.is_file():
        pytest.skip("incoming scene has not been built")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    contract = _policy_update_contract(paths, model)
    assert contract["trainable_action_indices"] == list(range(210, 242))
    assert paths.return_constraints["contact_guidance_reward_mode"] == "best_progress"
    guidance_cap = sum(
        paths.reward_weights[name] for name in ("shuttle_proximity", "timed_intercept", "racket_direction")
    )
    assert guidance_cap == pytest.approx(25.0)
    assert paths.reward_weights["miss"] > guidance_cap
    assert paths.reward_weights["hit_bonus"] - paths.reward_weights["invalid_net_crossing"] > (
        guidance_cap - paths.reward_weights["miss"]
    )

    common = {
        "control_substeps": 1,
        "max_episode_steps": 2,
        "reward_weights": paths.reward_weights,
        "min_return_net_clearance_m": paths.return_constraints["min_clearance_m"],
        "desired_return_up_component": paths.return_constraints["desired_up_component"],
        "ballistic_return_score_softness_m": paths.return_constraints["ballistic_score_softness_m"],
        "shuttle_proximity_softness_m": paths.return_constraints["shuttle_proximity_softness_m"],
        "timed_intercept_softness_m": paths.return_constraints["timed_intercept_softness_m"],
        "direction_distance_softness_m": paths.return_constraints["direction_distance_softness_m"],
        "contact_guidance_reward_mode": paths.return_constraints["contact_guidance_reward_mode"],
        "racket_velocity_direction_fraction": paths.return_constraints["racket_velocity_direction_fraction"],
        "direction_reward_mode": paths.return_constraints["direction_reward_mode"],
        "clearance_reward_mode": paths.return_constraints["clearance_reward_mode"],
        "hit_event_mode": paths.return_constraints["hit_event_mode"],
        "racket_guidance_mode": paths.return_constraints["racket_guidance_mode"],
        "inverse_target_speed_m_s": paths.return_constraints["inverse_target_speed_m_s"],
        "inverse_velocity_softness_m_s": paths.return_constraints["inverse_velocity_softness_m_s"],
        "curriculum_feed_order": "stored",
        "filter_finger_observation": False,
    }
    feed = [_synthetic_feed(0.0)]
    cpu_env = IncomingShuttleHitEnv(model_path, feed_bank=feed, **common)
    mjx_env = IncomingHitMjxEnv(model_path, feed, impl="jax", **common)
    assert cpu_env.control_hash == mjx_env.control_hash
    assert cpu_env.control_manifest["environment_abi"]["reward_semantics"] == (
        "incoming_hit_bounded_contact_progress_v23"
    )


def test_v24_selected_delta_adapter_contract_is_identity_and_right_arm_only() -> None:
    spec = REPO_ROOT / "experiments/posttrain/incoming_shuttle_hit_right_arm_delta_adapter_v24.yaml"
    paths = load_incoming_hit_spec(spec)
    model_path = default_incoming_scene_path()
    if not model_path.is_file():
        pytest.skip("incoming scene has not been built")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    contract = _policy_update_contract(paths, model)

    assert contract["schema_version"] == "stage3_policy_update_contract_v2"
    assert contract["mode"] == "selected_delta_adapter"
    assert contract["adapter_initialization"] == "zero_output_identity"
    assert contract["policy_delta_hidden_sizes"] == [128, 128]
    assert contract["trainable_action_indices"] == list(range(210, 242))
    assert contract["trainable_action_count"] == 32
    assert contract["full_action_count"] == 354
    assert contract["freeze_observation_normalizer"] is True
    assert contract["frozen_action_std"] == pytest.approx(0.001)
    assert contract["constant_residual_scale"] == pytest.approx(0.25)
    assert len(contract["contract_sha256"]) == 64
    assert paths.return_constraints["contact_guidance_reward_mode"] == "best_progress"
    guidance_cap = sum(
        paths.reward_weights[name] for name in ("shuttle_proximity", "timed_intercept", "racket_direction")
    )
    assert paths.reward_weights["miss"] > guidance_cap


def test_v31_selected_physical_correction_contract_is_32d_and_independent() -> None:
    spec = REPO_ROOT / "experiments/posttrain/incoming_shuttle_hit_selected_physical_v31.yaml"
    paths = load_incoming_hit_spec(spec)
    model_path = default_incoming_scene_path()
    if not model_path.is_file():
        pytest.skip("incoming scene has not been built")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    contract = _policy_update_contract(paths, model)
    assert contract["schema_version"] == "stage3_policy_update_contract_v5"
    assert contract["mode"] == "selected_physical_correction"
    assert contract["correction_action_space"] == "selected_only"
    assert contract["correction_composition"] == "independent_tanh_physical_addition_v1"
    assert contract["trainable_action_indices"] == list(range(210, 242))
    assert contract["trainable_action_count"] == 32
    assert len(contract["correction_physical_scales"]) == 32
    assert len(contract["correction_std_init"]) == 32
    assert contract["correction_window"] == {
        "time_to_intercept_open_s": 0.70,
        "time_to_intercept_close_s": -0.10,
        "smoothing_s": 0.05,
    }
    assert paths.stage3_direct["seed_feed_fingerprints"][0].startswith("c20b2d9c")
    assert paths.reward_weights["outgoing_vertical"] == pytest.approx(220.0)
    assert paths.ppo_overrides["rollout_steps"] == 128


def test_reference_graded_demo_seals_stance_full_body_authority_and_face_gate() -> None:
    spec = (
        REPO_ROOT
        / "experiments/posttrain/incoming_shuttle_hit_forehand_clear_reference_graded_demo_v5.yaml"
    )
    paths = load_incoming_hit_spec(spec)
    if not paths.scene_xml.is_file():
        pytest.skip("reference-graded incoming scene has not been built")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(paths.scene_xml))
    contract = _policy_update_contract(paths, model)

    assert paths.reference_ready_pose is not None
    assert paths.reference_ready_pose.path.name == "6月2日(1)-1.npz"
    assert paths.reference_ready_pose.frame_index == 0
    assert paths.reference_ready_pose.sha256 == (
        "67b92dfaff1f1a282d6adfb2d941a0a5f3ed98452fbba271b5b079e26bf0f80c"
    )
    assert paths.reference_ready_pose.min_left_foot_forward_lead_m == pytest.approx(0.10)
    assert paths.reference_ready_pose.min_lateral_stance_width_m == pytest.approx(0.20)
    assert paths.reference_ready_pose.max_lateral_stance_width_m == pytest.approx(0.40)

    base_artifact = Path(str(paths.stage3_direct["base_policy_artifact"]))
    assert base_artifact.name == "frozen_base_ckpt156"
    if (base_artifact / "manifest.json").is_file():
        base_manifest = json.loads((base_artifact / "manifest.json").read_text(encoding="utf-8"))
        assert base_manifest["source_checkpoint"].endswith(
            "/direct_student_ppo_repair_v3/checkpoint_156"
        )
        assert base_manifest["actor_spec"]["obs_size"] == 1950
        assert base_manifest["actor_spec"]["action_size"] == 354

    assert contract["schema_version"] == "stage3_graded_full_body_policy_update_contract_v2"
    assert contract["mode"] == "graded_full_body_correction"
    assert contract["correction_action_space"] == "all_model_actuators_graded"
    assert contract["inherited_residual_semantics"] == "exact_zero_standard_action_baseline"
    assert contract["frozen_actor_components"] == ["policy", "log_std"]
    assert contract["trainable_action_count"] == contract["full_action_count"] == 354
    assert len(contract["trainable_actuator_names"]) == 354
    assert {
        group: len(names)
        for group, names in contract["correction_group_actuator_names"].items()
    } == {
        "standard_body": 290,
        "left_arm": 32,
        "right_shoulder": 15,
        "right_elbow": 8,
        "right_forearm_rotation": 3,
        "right_wrist": 6,
    }
    assert {
        group: values["alpha"] for group, values in contract["correction_groups"].items()
    } == pytest.approx(
        {
            "standard_body": 0.005,
            "left_arm": 0.010,
            "right_shoulder": 0.050,
            "right_elbow": 0.120,
            "right_forearm_rotation": 0.200,
            "right_wrist": 0.300,
        }
    )
    assert {
        group: values["std_init"]
        for group, values in contract["correction_groups"].items()
    } == pytest.approx(
        {
            "standard_body": 0.0010,
            "left_arm": 0.0015,
            "right_shoulder": 0.0030,
            "right_elbow": 0.0040,
            "right_forearm_rotation": 0.0060,
            "right_wrist": 0.0080,
        }
    )
    assert contract["correction_groups"]["standard_body"]["std_max"] == pytest.approx(
        0.0010
    )
    assert contract["correction_groups"]["right_wrist"]["std_max"] == pytest.approx(
        0.0080
    )
    physical_exploration = {
        group: values["alpha"] * values["std_init"]
        for group, values in contract["correction_groups"].items()
    }
    assert physical_exploration == pytest.approx(
        {
            "standard_body": 0.000005,
            "left_arm": 0.000015,
            "right_shoulder": 0.000150,
            "right_elbow": 0.000480,
            "right_forearm_rotation": 0.001200,
            "right_wrist": 0.002400,
        }
    )
    assert contract["quality_success"]["min_racket_face_forward_alignment"] == pytest.approx(0.50)
    promotion_gates = paths.evaluation["promotion_gates"]
    assert promotion_gates["min_racket_face_forward_alignment"] == pytest.approx(0.50)
    assert promotion_gates["max_standard_body_state_rmse_m"] == pytest.approx(0.08)
    assert paths.ppo_overrides["update_epochs"] == 1
    assert paths.ppo_overrides["minibatch_size"] == 65_536
    assert paths.ppo_overrides["actor_learning_rate"] == pytest.approx(1.0e-7)
    assert paths.ppo_overrides["critic_learning_rate"] == pytest.approx(3.0e-4)
    assert paths.ppo_overrides["max_post_update_ratio_guard_fraction"] == pytest.approx(0.01)
    assert paths.ppo_overrides["max_post_update_kl_estimate"] == pytest.approx(0.02)


def test_reference_graded_demo_rejects_excessive_standard_body_authority() -> None:
    spec = (
        REPO_ROOT
        / "experiments/posttrain/incoming_shuttle_hit_forehand_clear_reference_graded_demo_v5.yaml"
    )
    paths = load_incoming_hit_spec(spec)
    if not paths.scene_xml.is_file():
        pytest.skip("reference-graded incoming scene has not been built")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(paths.scene_xml))
    groups = {
        group: dict(values)
        for group, values in paths.stage3_direct["correction_groups"].items()
    }
    groups["standard_body"]["alpha"] = 0.11
    bad_paths = replace(
        paths,
        stage3_direct={**paths.stage3_direct, "correction_groups": groups},
    )

    with pytest.raises(ValueError, match=r"caps standard_body\.alpha at 0\.10"):
        _policy_update_contract(bad_paths, model)


def test_stage3_training_run_manifest_is_immutable_and_config_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = load_incoming_hit_spec(
        REPO_ROOT
        / "experiments/posttrain/incoming_shuttle_hit_forehand_clear_reference_graded_demo_v5.yaml"
    )
    scene = tmp_path / "scene.xml"
    scene.write_text("<mujoco/>", encoding="utf-8")
    paths = replace(paths, scene_xml=scene)
    (tmp_path / "preflight_report.json").write_text(
        json.dumps(
            {
                "passed": True,
                "reference_ready_pose": {
                    "passed": True,
                    "left_foot_forward_lead_m": 0.125,
                },
            }
        ),
        encoding="utf-8",
    )
    prerequisites = {
        "verified": True,
        "binding_sha256": "a" * 64,
    }
    env = SimpleNamespace(
        control_manifest={"control_hash": "b" * 64},
        training_prerequisite_binding=prerequisites,
    )
    config_payload = {
        "total_env_steps": 12_000_000,
        "num_envs": 512,
        "rollout_steps": 128,
        "seed": 0,
        "policy_update_mode": "graded_full_body_correction",
    }
    cfg = SimpleNamespace(**config_payload, _asdict=lambda: dict(config_payload))
    monkeypatch.setenv("MUSCLEMIMIC_STAGE3_WANDB_PROJECT", "stage3-test")
    monkeypatch.setenv("MUSCLEMIMIC_STAGE3_WANDB_RUN_ID", "reference-graded-test-s0")
    monkeypatch.setenv("MUSCLEMIMIC_STAGE3_WANDB_MODE", "online")

    first = _seal_stage3_training_run_manifest(
        paths=paths,
        out_dir=tmp_path,
        env=env,
        cfg=cfg,
        policy_update_contract={"mode": "graded_full_body_correction"},
        impl="warp",
        resume_from=None,
        initialize_policy_from=None,
    )
    manifest_path = tmp_path / "training_run_manifest.json"
    original_bytes = manifest_path.read_bytes()
    manifest = json.loads(original_bytes)
    assert manifest["run_id"] == "reference-graded-test-s0"
    assert manifest["total_env_steps_requested"] == 12_000_000
    assert manifest["resolved_config"]["reward_weights"] == paths.reward_weights
    assert manifest["resolved_config"]["termination"]["body_fall_root_height_m"] == pytest.approx(0.55)
    assert manifest["reference_ready_pose"]["passed"] is True
    assert first["binding_sha256"] == manifest["binding_sha256"]

    repeated = _seal_stage3_training_run_manifest(
        paths=paths,
        out_dir=tmp_path,
        env=env,
        cfg=cfg,
        policy_update_contract={"mode": "graded_full_body_correction"},
        impl="warp",
        resume_from=None,
        initialize_policy_from=None,
    )
    assert repeated == first
    assert manifest_path.read_bytes() == original_bytes

    resumed = _seal_stage3_training_run_manifest(
        paths=paths,
        out_dir=tmp_path,
        env=env,
        cfg=cfg,
        policy_update_contract={"mode": "graded_full_body_correction"},
        impl="warp",
        resume_from=tmp_path / "checkpoints/checkpoint_000001/policy.npz",
        initialize_policy_from=None,
    )
    assert resumed == first
    assert manifest_path.read_bytes() == original_bytes

    changed_payload = {**config_payload, "total_env_steps": 13_000_000}
    changed_cfg = SimpleNamespace(
        **changed_payload,
        _asdict=lambda: dict(changed_payload),
    )
    with pytest.raises(ValueError, match="already sealed to a different training run"):
        _seal_stage3_training_run_manifest(
            paths=paths,
            out_dir=tmp_path,
            env=env,
            cfg=changed_cfg,
            policy_update_contract={"mode": "graded_full_body_correction"},
            impl="warp",
            resume_from=None,
            initialize_policy_from=None,
        )


def test_v40_ballistic_direction_repair_seals_strict_quality_success() -> None:
    spec = REPO_ROOT / "experiments/posttrain/incoming_shuttle_hit_high_point_selected_physical_v40.yaml"
    paths = load_incoming_hit_spec(spec)
    model_path = default_incoming_scene_path()
    if not model_path.is_file():
        pytest.skip("incoming scene has not been built")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    contract = _policy_update_contract(paths, model)

    assert contract["schema_version"] == "stage3_policy_update_contract_v6"
    assert contract["successful_action_imitation_coef"] == pytest.approx(500.0)
    assert contract["quality_success"] == {
        "min_outgoing_z_m_s": 1.0,
        "min_forward_m_s": 4.0,
        "min_predicted_net_clearance_m": 0.0,
        "min_return_direction_signed_score": 0.65,
        "require_episode_no_fall": True,
    }
    assert paths.reward_weights["hit_bonus"] == pytest.approx(160.0)
    assert paths.reward_weights["racket_direction"] == pytest.approx(240.0)
    assert paths.reward_weights["return_clearance"] == pytest.approx(320.0)
    assert paths.reward_weights["crossed_net"] == pytest.approx(600.0)
    guidance_cap = sum(
        paths.reward_weights[name] for name in ("shuttle_proximity", "timed_intercept", "racket_direction")
    )
    assert paths.reward_weights["miss"] > guidance_cap
    curriculum = paths.stage3_direct["curriculum"]
    assert (
        curriculum["fixed_feed_steps"] + curriculum["jitter_expand_steps"] + curriculum["full_bank_expand_steps"]
        <= paths.ppo_overrides["total_steps"]
    )


def test_v41_progressive_imitation_is_sealed_without_relaxing_success() -> None:
    spec = REPO_ROOT / "experiments/posttrain/incoming_shuttle_hit_high_point_selected_physical_v41.yaml"
    paths = load_incoming_hit_spec(spec)
    model_path = default_incoming_scene_path()
    if not model_path.is_file():
        pytest.skip("incoming scene has not been built")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    contract = _policy_update_contract(paths, model)

    assert contract["schema_version"] == "stage3_policy_update_contract_v7"
    assert contract["quality_success"] == {
        "min_outgoing_z_m_s": 1.0,
        "min_forward_m_s": 4.0,
        "min_predicted_net_clearance_m": 0.0,
        "min_return_direction_signed_score": 0.65,
        "require_episode_no_fall": True,
    }
    assert contract["quality_imitation"] == {
        "mode": "progressive_ballistic",
        "min_weight": pytest.approx(0.02),
        "forward_softness_m_s": pytest.approx(1.0),
        "vertical_softness_m_s": pytest.approx(0.75),
        "clearance_softness_m": pytest.approx(0.75),
        "direction_softness": pytest.approx(0.10),
        "require_episode_no_fall": False,
    }
    assert contract["successful_action_imitation_coef"] == pytest.approx(500.0)
    assert paths.evaluation["promotion_gates"]["min_crossed_net_rate"] == pytest.approx(0.50)


def test_drag_aware_clearance_mode_is_explicit_and_cpu_mjx_abi_identical() -> None:
    spec = REPO_ROOT / "experiments/posttrain/incoming_shuttle_hit_high_point_selected_physical_v41.yaml"
    paths = load_incoming_hit_spec(spec)
    model_path = default_incoming_scene_path()
    if not model_path.is_file():
        pytest.skip("incoming scene has not been built")

    common = {
        "control_substeps": 1,
        "max_episode_steps": 2,
        "reward_weights": paths.reward_weights,
        "return_net_x_m": paths.return_constraints["net_x_m"],
        "return_net_height_m": paths.return_constraints["net_height_m"],
        "min_return_net_clearance_m": paths.return_constraints["min_clearance_m"],
        "desired_return_up_component": paths.return_constraints["desired_up_component"],
        "ballistic_return_score_softness_m": paths.return_constraints["ballistic_score_softness_m"],
        "clearance_prediction_mode": "quadratic_drag_conservative_v1",
        "shuttle_proximity_softness_m": paths.return_constraints["shuttle_proximity_softness_m"],
        "timed_intercept_softness_m": paths.return_constraints["timed_intercept_softness_m"],
        "direction_distance_softness_m": paths.return_constraints["direction_distance_softness_m"],
        "contact_guidance_reward_mode": paths.return_constraints["contact_guidance_reward_mode"],
        "racket_velocity_direction_fraction": paths.return_constraints["racket_velocity_direction_fraction"],
        "direction_reward_mode": paths.return_constraints["direction_reward_mode"],
        "clearance_reward_mode": paths.return_constraints["clearance_reward_mode"],
        "hit_event_mode": paths.return_constraints["hit_event_mode"],
        "racket_guidance_mode": paths.return_constraints["racket_guidance_mode"],
        "inverse_target_speed_m_s": paths.return_constraints["inverse_target_speed_m_s"],
        "inverse_velocity_softness_m_s": paths.return_constraints["inverse_velocity_softness_m_s"],
        "curriculum_feed_order": "stored",
        "filter_finger_observation": False,
    }
    feed = [_synthetic_feed(0.0)]
    cpu_env = IncomingShuttleHitEnv(model_path, feed_bank=feed, **common)
    mjx_env = IncomingHitMjxEnv(model_path, feed, impl="jax", **common)

    assert cpu_env.control_hash == mjx_env.control_hash
    environment_abi = cpu_env.control_manifest["environment_abi"]
    assert environment_abi["return_constraints"]["clearance_prediction_mode"] == ("quadratic_drag_conservative_v1")
    assert environment_abi["reward_semantics"] == ("incoming_hit_drag_aware_closest_approach_event_direction_v30")

    invalid = {**common, "clearance_prediction_mode": "unsealed_projection"}
    with pytest.raises(ValueError, match="clearance_prediction_mode"):
        IncomingShuttleHitEnv(model_path, feed_bank=feed, **invalid)
    with pytest.raises(ValueError, match="clearance_prediction_mode"):
        IncomingHitMjxEnv(model_path, feed, impl="jax", **invalid)


def test_v42_drag_aware_repair_activates_intended_softness_and_strict_gates() -> None:
    spec = REPO_ROOT / "experiments/posttrain/incoming_shuttle_hit_high_point_selected_physical_v42.yaml"
    paths = load_incoming_hit_spec(spec)
    model_path = default_incoming_scene_path()
    if not model_path.is_file():
        pytest.skip("incoming scene has not been built")
    import mujoco

    raw_spec = spec.read_text(encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    contract = _policy_update_contract(paths, model)

    assert "ballistic_return_score_softness_m" not in raw_spec
    assert paths.return_constraints["ballistic_score_softness_m"] == pytest.approx(0.75)
    assert paths.return_constraints["clearance_prediction_mode"] == ("quadratic_drag_conservative_v1")
    assert contract["schema_version"] == "stage3_policy_update_contract_v7"
    assert contract["quality_success"]["min_predicted_net_clearance_m"] == (pytest.approx(0.0))
    assert contract["quality_imitation"]["mode"] == "progressive_ballistic"
    assert paths.stage3_direct["curriculum"]["fixed_min_crossed_net_rate"] == (pytest.approx(0.05))
    assert paths.evaluation["promotion_gates"]["min_crossed_net_rate"] == (pytest.approx(0.50))


@pytest.mark.parametrize(
    ("version", "expected_scales"),
    (
        ("v44", [0.60, 0.88, 1.26, 1.96]),
        ("v45", [1.00, 1.26, 1.40, 1.96]),
    ),
)
def test_v44_v45_preserve_strict_hit_contract_with_bounded_physical_authority(
    version: str,
    expected_scales: list[float],
) -> None:
    spec = REPO_ROOT / (f"experiments/posttrain/incoming_shuttle_hit_high_point_selected_physical_{version}.yaml")
    paths = load_incoming_hit_spec(spec)
    model_path = default_incoming_scene_path()
    if not model_path.is_file():
        pytest.skip("incoming scene has not been built")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    contract = _policy_update_contract(paths, model)
    scales = np.asarray(contract["correction_physical_scales"])

    assert contract["schema_version"] == "stage3_policy_update_contract_v7"
    assert contract["mode"] == "selected_physical_correction"
    assert contract["trainable_action_count"] == 32
    assert np.unique(scales).tolist() == pytest.approx(expected_scales)
    assert float(scales.max()) <= 2.0
    assert contract["correction_window"] == {
        "time_to_intercept_open_s": 0.60,
        "time_to_intercept_close_s": -0.08,
        "smoothing_s": 0.05,
    }
    assert contract["quality_success"] == {
        "min_outgoing_z_m_s": 1.0,
        "min_forward_m_s": 4.0,
        "min_predicted_net_clearance_m": 0.0,
        "min_return_direction_signed_score": 0.65,
        "require_episode_no_fall": True,
    }
    assert paths.return_constraints["clearance_prediction_mode"] == ("quadratic_drag_conservative_v1")


def test_return_constraint_config_rejects_unknown_and_ambiguous_keys(
    tmp_path: Path,
) -> None:
    source = (REPO_ROOT / "experiments/posttrain/incoming_shuttle_hit_high_point_selected_physical_v42.yaml").read_text(
        encoding="utf-8"
    )
    unknown = tmp_path / "unknown_return_constraint.yaml"
    unknown.write_text(
        source.replace(
            "  clearance_prediction_mode: quadratic_drag_conservative_v1",
            "  clearance_prediction_mode: quadratic_drag_conservative_v1\n  unsealed_clearance_knob: 1.0",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown keys: unsealed_clearance_knob"):
        load_incoming_hit_spec(unknown)

    ambiguous = tmp_path / "ambiguous_softness.yaml"
    ambiguous.write_text(
        source.replace(
            "  ballistic_score_softness_m: 0.75",
            "  ballistic_score_softness_m: 0.75\n  ballistic_return_score_softness_m: 0.75",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cannot mix canonical"):
        load_incoming_hit_spec(ambiguous)


def test_v33_teacher_physical_scales_are_replayable_and_low_noise() -> None:
    spec = REPO_ROOT / "experiments/posttrain/incoming_shuttle_hit_high_point_selected_physical_v33.yaml"
    paths = load_incoming_hit_spec(spec)
    model_path = default_incoming_scene_path()
    if not model_path.is_file():
        pytest.skip("incoming scene has not been built")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    contract = _policy_update_contract(paths, model)
    scales = np.asarray(contract["correction_physical_scales"])
    std = np.asarray(contract["correction_std_init"])

    assert np.unique(scales).tolist() == pytest.approx([0.24, 0.44, 0.90, 1.40])
    assert float(scales.max()) == pytest.approx(1.40)
    assert float(std.max()) == pytest.approx(0.025)
    assert contract["successful_action_imitation_coef"] == pytest.approx(1.0)
    assert paths.ppo_overrides["entropy_coef"] == pytest.approx(0.0)
    assert paths.stage3_direct["behavior_cloning"]["initial_coef"] == pytest.approx(20.0)


def test_v34_uses_precise_teacher_lock_and_high_clear_evaluation_gate() -> None:
    spec = REPO_ROOT / "experiments/posttrain/incoming_shuttle_hit_high_point_selected_physical_v34.yaml"
    paths = load_incoming_hit_spec(spec)
    model_path = default_incoming_scene_path()
    if not model_path.is_file():
        pytest.skip("incoming scene has not been built")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    contract = _policy_update_contract(paths, model)
    scales = np.asarray(contract["correction_physical_scales"])
    std = np.asarray(contract["correction_std_init"])
    bc = paths.stage3_direct["behavior_cloning"]

    assert np.unique(scales).tolist() == pytest.approx([0.24, 0.44, 0.90, 1.40])
    assert float(std.max()) == pytest.approx(0.005)
    assert bc["pretrain_steps"] == 20_000
    assert bc["batch_size"] >= 67
    assert bc["initial_coef"] == pytest.approx(2_000.0)
    assert paths.reward_weights["body_fall"] == pytest.approx(1_200.0)
    assert paths.evaluation["promotion_gates"]["min_positive_outgoing_z_rate_on_hit"] == pytest.approx(0.50)


@pytest.mark.parametrize("version", ("v35", "v36"))
def test_v35_v36_use_frozen_teacher_prior_with_zero_initialized_feedback_delta(
    version: str,
) -> None:
    spec = REPO_ROOT / (f"experiments/posttrain/incoming_shuttle_hit_high_point_selected_physical_{version}.yaml")
    paths = load_incoming_hit_spec(spec)
    model_path = default_incoming_scene_path()
    if not model_path.is_file():
        pytest.skip("incoming scene has not been built")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    contract = _policy_update_contract(paths, model)
    std = np.asarray(contract["correction_std_init"])
    bc = paths.stage3_direct["behavior_cloning"]

    assert contract["teacher_action_prior_mode"] == ("time_interpolated_frozen_plus_delta")
    assert float(std.max()) == pytest.approx(0.00125)
    assert bc["pretrain_steps"] == 0
    assert bc["initial_coef"] == pytest.approx(500.0)
    assert paths.ppo_overrides["actor_learning_rate"] == pytest.approx(1.0e-5)


def test_v25_wrist_refinement_freezes_phase_a_and_has_fail_closed_reward_hierarchy() -> None:
    spec = REPO_ROOT / "experiments/posttrain/incoming_shuttle_hit_wrist_refinement_v25.yaml"
    paths = load_incoming_hit_spec(spec)
    model_path = default_incoming_scene_path()
    if not model_path.is_file():
        pytest.skip("incoming scene has not been built")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    contract = _policy_update_contract(paths, model)

    expected_names = ["SUP", "ECRL", "ECRB", "ECU", "FCR", "FCU", "PL", "PT", "PQ"]
    assert contract["schema_version"] == "stage3_policy_update_contract_v3"
    assert contract["mode"] == "selected_refinement_delta_adapter"
    assert contract["adapter_initialization"] == "zero_output_refinement_identity"
    assert contract["frozen_actor_components"] == ["policy", "policy_delta"]
    assert contract["policy_delta_hidden_sizes"] == [128, 128]
    assert contract["policy_refinement_delta_hidden_sizes"] == [96, 96]
    assert contract["trainable_actuator_names"] == expected_names
    assert contract["trainable_action_indices"] == [229, 234, 235, 236, 237, 238, 239, 240, 241]
    assert contract["trainable_action_count"] == 9
    assert contract["full_action_count"] == 354
    assert contract["freeze_observation_normalizer"] is True
    assert contract["frozen_action_std"] == pytest.approx(0.001)
    assert contract["constant_residual_scale"] == pytest.approx(0.25)
    assert len(contract["contract_sha256"]) == 64
    assert paths.return_constraints["contact_guidance_reward_mode"] == "best_progress"
    assert paths.return_constraints["direction_reward_mode"] == "signed_projection"
    assert paths.return_constraints["clearance_reward_mode"] == "signed_centered"
    assert paths.return_constraints["racket_velocity_direction_fraction"] == pytest.approx(0.05)

    guidance_cap = sum(
        paths.reward_weights[name] for name in ("shuttle_proximity", "timed_intercept", "racket_direction")
    )
    best_miss = guidance_cap - paths.reward_weights["miss"]
    worst_real_hit = (
        paths.reward_weights["hit_bonus"]
        - paths.reward_weights["return_direction"]
        - paths.reward_weights["return_clearance"]
        - paths.reward_weights["invalid_net_crossing"]
        - paths.reward_weights["landing_region"]
    )
    maximum_success = guidance_cap + sum(
        paths.reward_weights[name]
        for name in (
            "hit_bonus",
            "hit_speed",
            "return_direction",
            "return_clearance",
            "crossed_net",
            "landing_region",
        )
    )
    assert guidance_cap == pytest.approx(41.0)
    assert worst_real_hit > best_miss
    assert paths.reward_weights["body_fall"] > maximum_success


def test_v29_closest_event_wrist_contract_and_cpu_mjx_abi_match() -> None:
    spec = REPO_ROOT / "experiments/posttrain/incoming_shuttle_hit_wrist_closest_event_v29.yaml"
    paths = load_incoming_hit_spec(spec)
    model_path = default_incoming_scene_path()
    if not model_path.is_file():
        pytest.skip("incoming scene has not been built")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    contract = _policy_update_contract(paths, model)
    assert contract["mode"] == "selected_refinement_delta_adapter"
    assert contract["trainable_action_indices"] == [
        229,
        234,
        235,
        236,
        237,
        238,
        239,
        240,
        241,
    ]
    assert contract["freeze_observation_normalizer"] is True
    assert contract["freeze_trainable_action_std"] is True
    assert paths.return_constraints["contact_guidance_reward_mode"] == ("closest_approach_event_direction")
    assert paths.return_constraints["racket_velocity_direction_fraction"] == pytest.approx(0.75)

    guidance_cap = sum(
        paths.reward_weights[name] for name in ("shuttle_proximity", "timed_intercept", "racket_direction")
    )
    best_miss = guidance_cap - paths.reward_weights["miss"]
    worst_real_hit = (
        paths.reward_weights["hit_bonus"]
        - paths.reward_weights["racket_direction"]
        - paths.reward_weights["return_direction"]
        - paths.reward_weights["return_clearance"]
        - paths.reward_weights["invalid_net_crossing"]
        - paths.reward_weights["landing_region"]
    )
    assert guidance_cap == pytest.approx(136.0)
    assert best_miss == pytest.approx(-24.0)
    assert worst_real_hit == pytest.approx(30.0)
    assert worst_real_hit > best_miss

    common = {
        "control_substeps": 1,
        "max_episode_steps": 2,
        "reward_weights": paths.reward_weights,
        "min_return_net_clearance_m": paths.return_constraints["min_clearance_m"],
        "desired_return_up_component": paths.return_constraints["desired_up_component"],
        "ballistic_return_score_softness_m": paths.return_constraints["ballistic_score_softness_m"],
        "shuttle_proximity_softness_m": paths.return_constraints["shuttle_proximity_softness_m"],
        "timed_intercept_softness_m": paths.return_constraints["timed_intercept_softness_m"],
        "direction_distance_softness_m": paths.return_constraints["direction_distance_softness_m"],
        "contact_guidance_reward_mode": paths.return_constraints["contact_guidance_reward_mode"],
        "racket_velocity_direction_fraction": paths.return_constraints["racket_velocity_direction_fraction"],
        "direction_reward_mode": paths.return_constraints["direction_reward_mode"],
        "clearance_reward_mode": paths.return_constraints["clearance_reward_mode"],
        "hit_event_mode": paths.return_constraints["hit_event_mode"],
        "racket_guidance_mode": paths.return_constraints["racket_guidance_mode"],
        "inverse_target_speed_m_s": paths.return_constraints["inverse_target_speed_m_s"],
        "inverse_velocity_softness_m_s": paths.return_constraints["inverse_velocity_softness_m_s"],
        "curriculum_feed_order": "stored",
        "filter_finger_observation": False,
    }
    feed = [_synthetic_feed(0.0)]
    cpu_env = IncomingShuttleHitEnv(model_path, feed_bank=feed, **common)
    mjx_env = IncomingHitMjxEnv(model_path, feed, impl="jax", **common)
    assert cpu_env.control_hash == mjx_env.control_hash
    assert cpu_env.control_manifest["environment_abi"]["reward_semantics"] == (
        "incoming_hit_closest_approach_event_direction_v29"
    )


def test_v30_closest_event_right_arm_refinement_contract() -> None:
    spec = REPO_ROOT / "experiments/posttrain/incoming_shuttle_hit_right_arm_closest_event_v30.yaml"
    paths = load_incoming_hit_spec(spec)
    model_path = default_incoming_scene_path()
    if not model_path.is_file():
        pytest.skip("incoming scene has not been built")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    contract = _policy_update_contract(paths, model)
    assert contract["mode"] == "selected_refinement_delta_adapter"
    assert contract["policy_delta_hidden_sizes"] == [128, 128]
    assert contract["policy_refinement_delta_hidden_sizes"] == [96, 96]
    assert contract["trainable_action_indices"] == list(range(210, 242))
    assert contract["freeze_observation_normalizer"] is True
    assert contract["freeze_trainable_action_std"] is True
    assert contract["frozen_action_std"] == pytest.approx(0.001)
    assert paths.ppo_overrides["action_std_init"] == pytest.approx(0.05)
    assert paths.return_constraints["contact_guidance_reward_mode"] == ("closest_approach_event_direction")
    assert paths.return_constraints["racket_velocity_direction_fraction"] == pytest.approx(0.75)

    guidance_cap = sum(
        paths.reward_weights[name] for name in ("shuttle_proximity", "timed_intercept", "racket_direction")
    )
    best_miss = guidance_cap - paths.reward_weights["miss"]
    worst_real_hit = (
        paths.reward_weights["hit_bonus"]
        - paths.reward_weights["racket_direction"]
        - paths.reward_weights["return_direction"]
        - paths.reward_weights["return_clearance"]
        - paths.reward_weights["invalid_net_crossing"]
        - paths.reward_weights["landing_region"]
    )
    assert worst_real_hit > best_miss


def test_v24b_contact_curriculum_reaches_all_feeds_without_direction_gate() -> None:
    spec = REPO_ROOT / "experiments/posttrain/incoming_shuttle_hit_right_arm_contact_generalization_v24b.yaml"
    paths = load_incoming_hit_spec(spec)
    model_path = default_incoming_scene_path()
    if not model_path.is_file():
        pytest.skip("incoming scene has not been built")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    contract = _policy_update_contract(paths, model)
    assert contract["mode"] == "selected_delta_adapter"
    assert contract["policy_delta_hidden_sizes"] == [128, 128]
    assert contract["trainable_action_indices"] == list(range(210, 242))
    assert contract["frozen_action_std"] == pytest.approx(0.001)

    configured = dict(paths.stage3_direct["curriculum"])
    configured.pop("enabled")
    curriculum = Stage3Curriculum(
        lambda_start=0.0,
        lambda_end=0.0,
        lambda_expand_steps=0,
        **configured,
    )
    assert curriculum.phase(0) == "intercept_jitter"
    assert curriculum.values(env_steps=0, feed_bank_size=128).active_feed_count == 1
    assert curriculum.values(env_steps=500_000, feed_bank_size=128).active_feed_count == 16
    assert curriculum.values(env_steps=3_500_000, feed_bank_size=128).active_feed_count == 128
    assert configured["jitter_min_crossed_net_rate"] == pytest.approx(0.0)
    assert configured["full_bank_min_crossed_net_rate"] == pytest.approx(0.0)


def test_v24c_mean_consolidation_uses_all_feeds_and_fixed_low_exploration() -> None:
    spec = REPO_ROOT / "experiments/posttrain/incoming_shuttle_hit_right_arm_mean_consolidation_v24c.yaml"
    paths = load_incoming_hit_spec(spec)
    model_path = default_incoming_scene_path()
    if not model_path.is_file():
        pytest.skip("incoming scene has not been built")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    contract = _policy_update_contract(paths, model)
    assert contract["mode"] == "selected_delta_adapter"
    assert contract["policy_delta_hidden_sizes"] == [128, 128]
    assert contract["trainable_action_indices"] == list(range(210, 242))
    assert contract["frozen_action_std"] == pytest.approx(0.001)
    assert contract["freeze_trainable_action_std"] is True
    assert paths.ppo_overrides["action_std_init"] == pytest.approx(0.05)
    assert paths.ppo_overrides["entropy_coef"] == pytest.approx(0.0)

    configured = dict(paths.stage3_direct["curriculum"])
    configured.pop("enabled")
    curriculum = Stage3Curriculum(
        lambda_start=0.0,
        lambda_end=0.0,
        lambda_expand_steps=0,
        **configured,
    )
    assert curriculum.phase(0) == "full_bank_expansion"
    assert curriculum.values(env_steps=0, feed_bank_size=128).active_feed_count == 128
    assert configured["full_bank_min_hit_rate"] == pytest.approx(0.05)
    assert configured["full_bank_min_crossed_net_rate"] == pytest.approx(0.0)


def test_v24d_success_imitation_is_sealed_without_changing_physics_or_feeds() -> None:
    v24c = load_incoming_hit_spec(
        REPO_ROOT / "experiments/posttrain/incoming_shuttle_hit_right_arm_mean_consolidation_v24c.yaml"
    )
    v24d = load_incoming_hit_spec(
        REPO_ROOT / "experiments/posttrain/incoming_shuttle_hit_right_arm_success_imitation_v24d.yaml"
    )
    model_path = default_incoming_scene_path()
    if not model_path.is_file():
        pytest.skip("incoming scene has not been built")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    old_contract = _policy_update_contract(v24c, model)
    contract = _policy_update_contract(v24d, model)

    assert old_contract["schema_version"] == "stage3_policy_update_contract_v2"
    assert "successful_action_imitation_coef" not in old_contract
    assert contract["schema_version"] == "stage3_policy_update_contract_v4"
    assert contract["successful_action_imitation_coef"] == pytest.approx(1.0)
    assert contract["mode"] == "selected_delta_adapter"
    assert contract["trainable_action_indices"] == list(range(210, 242))
    assert contract["freeze_observation_normalizer"] is True
    assert contract["freeze_trainable_action_std"] is True
    assert contract["frozen_action_std"] == pytest.approx(0.001)
    assert v24d.ppo_overrides == v24c.ppo_overrides
    assert v24d.feed_bank_path == v24c.feed_bank_path
    assert v24d.feed_bank_size == v24c.feed_bank_size
    assert v24d.feed_seed == v24c.feed_seed
    assert v24d.eval_feed_bank_path == v24c.eval_feed_bank_path
    assert v24d.eval_feed_bank_size == v24c.eval_feed_bank_size
    assert v24d.eval_feed_seed == v24c.eval_feed_seed
    assert v24d.feed_kwargs == v24c.feed_kwargs
    assert v24d.reward_weights == v24c.reward_weights
    assert v24d.return_constraints == v24c.return_constraints
    assert contract["contract_sha256"] != old_contract["contract_sha256"]


def test_production_scene_reports_exact_child_and_zero_human_racket_contact() -> None:
    scene = default_incoming_scene_path()
    if not scene.is_file():
        pytest.skip("incoming scene has not been built")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene))
    report = stage3_attachment_report(model, scene)

    assert report["attachment_mode"] == "exact_child"
    assert report["contract_passed"] is True
    assert report["parent_body_matches"] is True
    assert report["racket_joint_count"] == 0
    assert report["racket_equality_constraint_count"] == 0
    assert report["human_racket_mask_compatible_geom_pairs"] == 0
    assert report["human_racket_explicit_contact_pairs"] == 0
    assert report["hand_racket_contact_enabled"] is False
    assert report["racket_shuttle_contact_enabled"] is True
    assert len(report["attachment_hash"]) == 64


def test_return_net_clearance_ignores_incoming_feed_before_hit() -> None:
    # Feeds launch on +x (opponent side), cross to -x toward the player, then
    # only a successful return crosses from -x to +x.
    assert (
        _return_net_clearance(
            previous_shuttle_x=3.0,
            shuttle_xyz=np.asarray([2.9, 0.0, 2.0]),
            hit_registered=False,
        )
        is None
    )
    assert (
        _return_net_clearance(
            previous_shuttle_x=0.1,
            shuttle_xyz=np.asarray([-0.1, 0.0, 1.8]),
            hit_registered=False,
        )
        is None
    )
    assert _return_net_clearance(
        previous_shuttle_x=-0.1,
        shuttle_xyz=np.asarray([0.1, 0.0, 1.9]),
        hit_registered=True,
    ) == pytest.approx(0.35)


def test_exact_lab_state_builder_matches_cpu_and_batched_jax_order() -> None:
    scene = default_incoming_scene_path()
    if not scene.is_file():
        pytest.skip("incoming scene has not been built")
    import mujoco

    from environment.overall_environment.src.body_obs_adapter import BodyObsSchema

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model,
        data,
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "overall_ready"),
    )
    data.qvel[:] = np.linspace(-0.2, 0.2, model.nv)
    data.ctrl[:] = np.linspace(-0.4, 0.4, model.nu)
    data.act[:] = np.linspace(0.1, 0.9, model.na)
    mujoco.mj_forward(model, data)

    root_name = "root"
    joint_id = next(
        index
        for index in range(model.njnt)
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) not in {None, root_name}
    )
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    joint_type = int(model.jnt_type[joint_id])
    qwidth = 4 if joint_type == int(mujoco.mjtJoint.mjJNT_BALL) else 1
    dwidth = 3 if joint_type == int(mujoco.mjtJoint.mjJNT_BALL) else 1
    actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 0)
    kinematic_size = 5 + qwidth + 6 + dwidth
    schema = BodyObsSchema(
        total_size=kinematic_size + 5 + 1,
        kinematic_size=kinematic_size,
        muscle_size=5,
        touch_size=0,
        goal_size=1,
        action_size=1,
        root_joint_name=root_name,
        joint_names=(joint_name,),
        actuator_names=(actuator_name,),
        touch_sensor_names=(),
        observation_names=("state", "motion_phase"),
        student_filtered=True,
    )
    builder = Stage3LabStateBuilder(
        model=model,
        body_schema=schema,
        expected_state_dim=schema.total_size,
    )
    np.testing.assert_array_equal(
        np.asarray(builder.activation_addresses),
        model.actuator_actadr[[0]],
    )
    cpu = builder.build_numpy(model=model, data=data, phase=0.37)
    batched_data = SimpleNamespace(
        qpos=jnp.asarray(data.qpos)[None, :],
        qvel=jnp.asarray(data.qvel)[None, :],
        actuator_length=jnp.asarray(data.actuator_length)[None, :],
        actuator_velocity=jnp.asarray(data.actuator_velocity)[None, :],
        actuator_force=jnp.asarray(data.actuator_force)[None, :],
        ctrl=jnp.asarray(data.ctrl)[None, :],
        act=jnp.asarray(data.act)[None, :],
        sensordata=jnp.asarray(data.sensordata)[None, :],
    )
    jax_state = builder.build_jax(
        data=batched_data,
        phase=jnp.asarray([0.37]),
    )

    np.testing.assert_allclose(np.asarray(jax_state[0]), cpu, atol=1e-6)

from __future__ import annotations

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
    BoundedResidualMask,
    ConstantGripProvider,
    Stage3ActionRouter,
    Stage3Curriculum,
    Stage3LABController,
    Stage3LabStateBuilder,
    apply_teacher_body_ctrlrange,
    stage3_attachment_report,
)
from musclemimic.badminton.scripts.run_incoming_shuttle_hit import (  # noqa: E402
    _base_only_summary,
    _compare_naturalness_to_prior,
    _ensure_feed_bank_artifact,
    _feed_bank_identity_qc,
    _return_net_clearance,
    _run_ppo,
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
            "racket_rotation": np.asarray(
                [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
            ),
        }

    prior = [row(0.0), row(0.1), row(0.2)]
    identical = _compare_naturalness_to_prior(prior, prior)
    assert identical["body_relative_deviation_to_prior"] == pytest.approx(0.0)
    assert identical["racket_rotation_rmse_to_prior_rad"] == pytest.approx(0.0)

    degraded = _compare_naturalness_to_prior(
        [row(0.0), row(0.2), row(0.4)], prior
    )
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
    assert calls == [(2, 17, 8.0)]

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
            gate_config={},
            required_feed_count=128,
            prior_direct_baseline={
                "prior_vs_direct_body_racket_relative_degradation": 0.05
            },
        )

    assert summary["passed"] is True
    assert summary["evaluated_feed_count"] == 128
    assert summary["racket_head_speed_m_s"] == summary["mean_racket_head_speed_m_s"]
    assert summary["racket_head_speed_m_s"] == summary["mean_contact_racket_head_speed_m_s"]
    assert summary["net_clearance_m"] == summary["mean_net_clearance_m"]

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
        return jnp.broadcast_to(jnp.array([0.25, -0.5]), (*state.shape[:-1], 2)), jnp.zeros(
            (*state.shape[:-1], 2)
        )

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


def test_teacher_ctrlrange_requires_v2_unit_muscles_and_updates_only_body() -> None:
    class Model:
        actuator_ctrlrange = np.asarray(
            [[-1.0, 1.0], [-2.0, 2.0], [-3.0, 3.0], [-4.0, 4.0]]
        )
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
        return np.zeros((*np.asarray(state).shape[:-1], 2)), np.zeros(
            (*np.asarray(state).shape[:-1], 2)
        )

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
    full_action, _output = mjx_env._compose_action(
        None,
        SimpleNamespace(lab_state=jnp.zeros((1, 3)), lambda_lab=jnp.asarray(0.25)),
        jnp.zeros((1, 2)),
    )
    assert full_action.shape == (1, 354)


def test_production_stage3_spec_is_latent_only_and_blocks_legacy_cpu_ppo(
    tmp_path: Path,
) -> None:
    paths = load_incoming_hit_spec(
        REPO_ROOT / "experiments/posttrain/incoming_shuttle_hit_v1.yaml"
    )
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
    assert config["bounded_residual"]["alpha"] <= 0.10
    assert tuple(config["bounded_residual"]["actuator_names"]) == (
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
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        not in {None, root_name}
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

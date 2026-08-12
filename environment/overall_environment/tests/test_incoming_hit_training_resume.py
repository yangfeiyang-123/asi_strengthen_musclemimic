from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax

import environment.overall_environment.src.train_incoming_hit_mjx as training_module
import musclemimic.badminton.scripts.run_incoming_shuttle_hit as runner_module
from fullbody.run_forehand_clear_pipeline import _require_stage3_artifact_binding
from musclemimic.badminton.scripts.run_incoming_shuttle_hit import (
    _build_stage3_artifact_binding,
    _stage3_evaluation_content_sha256,
    _stage3_evaluation_summary,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment.overall_environment.src.train_incoming_hit_mjx import (  # noqa: E402
    ObsRms,
    TrainConfig,
    _CompletedEpisodeGateWindow,
    _dist,
    _inherited_policy_mean,
    _resume_action_prior_source,
    _selected_correction_dist,
    _tanh_normal_logprob,
    _full_batch_budget,
    apply_policy_exploration_contract,
    backfill_pre_hit_event_weight,
    backfill_pre_hit_success_mask,
    compute_rollout_gae,
    freeze_action_std_gradients,
    future_episode_outcome,
    init_agent,
    load_actor_initialization,
    load_quality_teacher_dataset,
    load_training_checkpoint,
    load_training_checkpoint_metadata,
    make_train_iteration,
    mask_policy_update_gradients,
    mask_selected_delta_adapter_gradients,
    mask_selected_refinement_delta_adapter_gradients,
    mask_selected_physical_correction_gradients,
    reconcile_metrics_history,
    pretrain_selected_correction_bc,
    progressive_quality_imitation_event_weight,
    quality_success_event_mask,
    resolve_training_checkpoint,
    sample_action,
    save_training_checkpoint,
    save_versioned_training_checkpoint,
    successful_action_imitation_loss,
    window_action_imitation_weight,
    window_successful_action_imitation_weight,
    validate_stage3_direct_training_prerequisites,
    validate_stage3_residual_training_prerequisites,
    validate_stage3_training_prerequisites,
    validate_training_feed_manifest,
)


def test_completed_episode_gate_window_is_weighted_and_fails_closed() -> None:
    window = _CompletedEpisodeGateWindow(min_completed_episodes=512, max_iterations=16)
    window.update(
        {
            "episodes_finished": 5.0,
            "hit_rate": 1.0,
            "crossed_net_rate": 0.8,
            "fall_rate": 0.0,
        }
    )
    window.update(
        {
            "episodes_finished": 100.0,
            "hit_rate": 0.2,
            "crossed_net_rate": 0.1,
            "fall_rate": 0.1,
        }
    )

    summary = window.summary()
    assert summary["episodes_finished"] == 105.0
    assert summary["hit_rate"] == 25.0 / 105.0
    assert summary["crossed_net_rate"] == 14.0 / 105.0
    assert summary["fall_rate"] == 10.0 / 105.0
    assert summary["ready"] is False
    assert window.metrics_for_gate()["episodes_finished"] == 0.0

    window.update(
        {
            "episodes_finished": 500.0,
            "hit_rate": 0.7,
            "crossed_net_rate": 0.5,
            "fall_rate": 0.02,
        }
    )
    assert window.summary()["ready"] is True
    restored = _CompletedEpisodeGateWindow.from_state_dict(window.state_dict())
    assert restored.state_dict() == window.state_dict()


def test_gate_window_keeps_contact_quality_across_rollout_boundaries() -> None:
    window = _CompletedEpisodeGateWindow(
        min_completed_episodes=1,
        max_iterations=16,
    )
    # Contact happens before the rollout boundary; no episode has ended yet.
    window.update(
        {
            "episodes_finished": 0.0,
            "hit_rate": 0.0,
            "hit_events": 10.0,
            "positive_outgoing_z_hit_events": 8.0,
        }
    )
    # Those episodes end in the next rollout, which contains no new contact.
    window.update(
        {
            "episodes_finished": 10.0,
            "hit_rate": 1.0,
            "hit_events": 0.0,
            "positive_outgoing_z_hit_events": 0.0,
        }
    )

    summary = window.summary()
    assert summary["hit_rate"] == 1.0
    assert summary["positive_outgoing_z_rate_on_hit"] == 0.8

    legacy_state = {
        **window.state_dict(),
        "schema_version": "stage3_completed_episode_gate_window_v1",
    }
    migrated = _CompletedEpisodeGateWindow.from_state_dict(legacy_state)
    assert migrated.state_dict()["rows"] == []


def test_training_feed_diagnostic_restores_checkpoint_consumer_order() -> None:
    bank = ["producer-a", "producer-b", "producer-c"]
    producer_manifest = {
        "schema_version": "incoming_shuttle_feed_bank_manifest_v1",
        "content_sha256": "producer-content",
        "sample_fingerprints": ["a", "b", "c"],
    }
    checkpoint_manifest = {
        **producer_manifest,
        "consumer_order": {
            "schema_version": "incoming_hit_curriculum_feed_order_v1",
            "mode": "explicit_fingerprint_order",
            "sample_fingerprints": ["c", "a", "b"],
            "passed": True,
        },
    }

    ordered = runner_module._ordered_training_diagnostic_bank(
        bank,
        producer_manifest=producer_manifest,
        checkpoint_manifest=checkpoint_manifest,
    )

    assert ordered == ["producer-c", "producer-a", "producer-b"]
    changed_checkpoint = {
        **checkpoint_manifest,
        "content_sha256": "different-content",
    }
    with np.testing.assert_raises_regex(ValueError, "producer artifact differs"):
        runner_module._ordered_training_diagnostic_bank(
            bank,
            producer_manifest=producer_manifest,
            checkpoint_manifest=changed_checkpoint,
        )


def _write_quality_teacher(tmp_path: Path, *, success: bool = True) -> Path:
    trajectory = tmp_path / "teacher_trajectory_mjx.npz"
    n = 32
    rebound = np.zeros(n, dtype=bool)
    rebound[20] = True
    outgoing = np.zeros((n, 3), dtype=np.float32)
    outgoing[20] = [3.0, 0.0, 1.0 if success else -1.0]
    np.savez_compressed(
        trajectory,
        observation_normalized=np.linspace(-1.0, 1.0, n * 3, dtype=np.float32).reshape(n, 3),
        correction_raw=np.full((n, 2), 0.25, dtype=np.float32),
        correction_window=np.ones(n, dtype=np.float32),
        time_to_intercept_s=np.linspace(0.31, 0.0, n, dtype=np.float32),
        event_rebound=rebound,
        outgoing_shuttle_velocity_xyz_m_s=outgoing,
        selected_action_indices=np.asarray([0, 1], dtype=np.int32),
        physical_scales=np.asarray([0.1, 0.2], dtype=np.float32),
        feed_fingerprint=np.asarray("feed"),
        swing_phase_advance_s=np.asarray(0.18, dtype=np.float32),
        source_checkpoint_sha256=np.asarray("a" * 64),
        search_contract_sha256=np.asarray("b" * 64),
        outgoing_velocity_semantics=np.asarray(
            "post_control_step_after_all_physics_substeps"
        ),
        event_rebound_contact_semantics=np.asarray(
            "single_event_impulse_with_stringbed_force_suppressed_during_cooldown_v2"
        ),
    )
    metrics = {
        "teacher_success": success,
        "teacher_success_rate": 1.0 if success else 0.0,
        "high_region_contact": success,
        "high_region_contact_rate": 1.0 if success else 0.0,
        "outgoing_z_m_s": 1.0 if success else -1.0,
        "outgoing_forward_m_s": 3.0,
    }
    report = {
        "schema_version": "stage3_single_feed_mjx_cem_report_v3",
        "passed": success,
        "verified_metrics": metrics,
        "cpu_replay_audit": {
            "hit": success,
            "event_rebound": success,
            "body_fall": False,
            "feed_fingerprint": "feed",
            "swing_phase_advance_s": 0.18,
        },
        "cpu_replay_event_equivalent": success,
        "contract": {
            "feed_fingerprint": "feed",
            "swing_phase_advance_s": 0.18,
            "swing_phase_timing_semantics": (
                "frozen_base_swing_phase_advance_applied_identically_to_search_"
                "backend_and_cpu_replays"
            ),
            "outgoing_velocity_semantics": (
                "post_control_step_after_all_physics_substeps"
            ),
            "event_rebound_contact_semantics": (
                "single_event_impulse_with_stringbed_force_suppressed_during_cooldown_v2"
            ),
            "high_region_contact": {
                "max_stringbed_height_deficit_m": 0.10,
                "max_hand_height_deficit_m": 0.10,
                "semantics": "soft_window_teacher_gate_not_exact_apex",
            }
        },
        "teacher_trace": {
            "trace_path": str(trajectory.resolve()),
            "trace_sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest(),
            "selected_replica_metrics": metrics,
        },
    }
    (tmp_path / "cem_report.json").write_text(json.dumps(report), encoding="utf-8")
    return trajectory


def test_resume_teacher_prior_requires_local_pretrain_binding(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset = training_module.QualityTeacherDataset(
        observation_normalized=np.zeros((32, 3), dtype=np.float32),
        correction_raw=np.zeros((32, 2), dtype=np.float32),
        sample_weight=np.ones((32,), dtype=np.float32),
        time_to_intercept_s=np.linspace(0.0, 0.31, 32, dtype=np.float32),
        binding={"binding_sha256": "teacher"},
    )
    monkeypatch.setattr(
        training_module,
        "load_quality_teacher_dataset",
        lambda *_args, **_kwargs: dataset,
    )
    cfg = TrainConfig(
        policy_update_mode="selected_physical_correction",
        policy_trainable_action_indices=(0, 1),
        correction_physical_scales=(0.1, 0.2),
        teacher_action_prior_mode="time_interpolated_frozen_plus_delta",
    )
    env = SimpleNamespace(
        expects_raw_latent=False,
        base_policy_artifact=None,
        task_profile="legacy_v1",
    )

    with np.testing.assert_raises_regex(
        ValueError,
        "missing its local pretrain binding",
    ):
        training_module.train(
            env,
            cfg,
            tmp_path / "run",
            resume_from=tmp_path / "unused_checkpoint.npz",
            teacher_dataset_path=tmp_path / "teacher.npz",
        )


def test_quality_teacher_loader_rejects_downward_contact(tmp_path: Path) -> None:
    path = _write_quality_teacher(tmp_path, success=False)
    with np.testing.assert_raises_regex(ValueError, "robust return-success"):
        load_quality_teacher_dataset(
            path,
            selected_action_indices=(0, 1),
            correction_physical_scales=(0.1, 0.2),
            source_checkpoint_sha256="a" * 64,
        )


def test_quality_teacher_loader_rejects_cpu_replay_divergence(tmp_path: Path) -> None:
    path = _write_quality_teacher(tmp_path, success=True)
    report_path = tmp_path / "cem_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["cpu_replay_audit"]["event_rebound"] = False
    report["cpu_replay_event_equivalent"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with np.testing.assert_raises_regex(ValueError, "independent CPU replay"):
        load_quality_teacher_dataset(
            path,
            selected_action_indices=(0, 1),
            correction_physical_scales=(0.1, 0.2),
            source_checkpoint_sha256="a" * 64,
        )


def test_quality_teacher_loader_rejects_physical_scale_mismatch(
    tmp_path: Path,
) -> None:
    path = _write_quality_teacher(tmp_path, success=True)

    with np.testing.assert_raises_regex(ValueError, "physical scales differ"):
        load_quality_teacher_dataset(
            path,
            selected_action_indices=(0, 1),
            correction_physical_scales=(0.2, 0.4),
            source_checkpoint_sha256="a" * 64,
        )


def test_quality_teacher_bc_updates_only_correction_and_reduces_loss(tmp_path: Path) -> None:
    path = _write_quality_teacher(tmp_path, success=True)
    dataset = load_quality_teacher_dataset(
        path,
        selected_action_indices=(0, 1),
        correction_physical_scales=(0.1, 0.2),
        source_checkpoint_sha256="a" * 64,
    )
    assert np.all(np.diff(dataset.time_to_intercept_s) > 0.0)
    agent = init_agent(
        jax.random.PRNGKey(91),
        obs_size=3,
        action_size=4,
        hidden=(8,),
        action_std_init=0.2,
        policy_correction_hidden=(8,),
        correction_action_size=2,
        correction_std_init=(0.1, 0.1),
    )
    inherited_before = jax.tree_util.tree_map(np.asarray, agent["policy"])
    updated, report = pretrain_selected_correction_bc(
        agent,
        dataset,
        steps=50,
        batch_size=16,
        learning_rate=1.0e-3,
        seed=7,
    )
    assert report["passed"] is True
    assert report["final_weighted_mse"] < report["initial_weighted_mse"]
    for before, after in zip(
        jax.tree_util.tree_leaves(inherited_before),
        jax.tree_util.tree_leaves(updated["policy"]),
        strict=True,
    ):
        np.testing.assert_array_equal(before, after)


def test_gae_bootstraps_time_limit_terminal_observation_but_not_true_termination() -> None:
    base = {
        "reward": jnp.asarray([[1.0]]),
        "value": jnp.asarray([[0.25]]),
        "next_value": jnp.asarray([[2.0]]),
        "done": jnp.asarray([[True]]),
    }
    truncated_advantage, _ = compute_rollout_gae(
        {**base, "terminated": jnp.asarray([[False]])},
        gamma=0.9,
        gae_lambda=0.95,
    )
    terminated_advantage, _ = compute_rollout_gae(
        {**base, "terminated": jnp.asarray([[True]])},
        gamma=0.9,
        gae_lambda=0.95,
    )

    np.testing.assert_allclose(truncated_advantage, [[1.0 + 0.9 * 2.0 - 0.25]])
    np.testing.assert_allclose(terminated_advantage, [[1.0 - 0.25]])


def test_gae_stops_recursive_chain_at_auto_reset_boundary() -> None:
    records = {
        "reward": jnp.asarray([[1.0], [10.0]]),
        "value": jnp.zeros((2, 1)),
        "next_value": jnp.zeros((2, 1)),
        "done": jnp.asarray([[True], [False]]),
        "terminated": jnp.asarray([[False], [False]]),
    }
    advantages, _ = compute_rollout_gae(records, gamma=0.9, gae_lambda=1.0)

    # The second transition belongs to the reset episode and must not leak
    # backward into the first transition's advantage.
    np.testing.assert_allclose(advantages[:, 0], [1.0, 10.0])


def test_pre_hit_success_mask_stops_at_done_and_excludes_post_hit_actions() -> None:
    hit_event = jnp.asarray(
        [
            [False, False],
            [False, False],
            [True, False],
            [False, False],
            [False, True],
            [False, False],
        ]
    )
    done = jnp.asarray(
        [
            [False, False],
            [False, True],
            [False, False],
            [False, False],
            [True, False],
            [False, True],
        ]
    )

    actual = backfill_pre_hit_success_mask(hit_event, done)

    np.testing.assert_array_equal(
        actual,
        np.asarray(
            [
                [True, False],
                [True, False],
                [True, True],
                [False, True],
                [False, True],
                [False, False],
            ]
        ),
    )


def test_successful_action_imitation_uses_only_selected_successful_values() -> None:
    mean = jnp.zeros((2, 4), dtype=jnp.float32)
    sampled = jnp.asarray(
        [
            [10.0, 2.0, 30.0, 4.0],
            [50.0, 6.0, 70.0, 8.0],
        ],
        dtype=jnp.float32,
    )
    success_weight = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    action_mask = jnp.asarray([0.0, 1.0, 0.0, 1.0], dtype=jnp.float32)

    loss = successful_action_imitation_loss(mean, sampled, success_weight, action_mask)
    gradient = jax.grad(
        lambda candidate: successful_action_imitation_loss(
            candidate,
            sampled,
            success_weight,
            action_mask,
        )
    )(mean)

    np.testing.assert_allclose(loss, 10.0)
    np.testing.assert_allclose(
        gradient,
        np.asarray(
            [
                [0.0, -2.0, 0.0, -4.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_successful_action_imitation_weights_only_active_correction_window() -> None:
    success = jnp.asarray(
        [[True, True], [True, False], [True, True]],
        dtype=jnp.bool_,
    )
    window = jnp.asarray(
        [[0.0, 0.25], [0.5, 0.75], [1.0, 1.5]],
        dtype=jnp.float32,
    )

    weights = window_successful_action_imitation_weight(success, window)

    np.testing.assert_allclose(
        weights,
        np.asarray(
            [[0.0, 0.0625], [0.25, 0.0], [1.0, 1.0]],
            dtype=np.float32,
        ),
    )


def test_progressive_event_weights_backfill_without_crossing_episode_end() -> None:
    event_weight = jnp.asarray(
        [[0.0, 0.0], [0.4, 0.0], [0.0, 0.8], [0.0, 0.0]],
        dtype=jnp.float32,
    )
    done = jnp.asarray(
        [[False, False], [False, True], [True, False], [False, False]],
        dtype=jnp.bool_,
    )

    backfilled = backfill_pre_hit_event_weight(event_weight, done)
    windowed = window_action_imitation_weight(
        backfilled,
        jnp.asarray(
            [[1.0, 1.0], [0.5, 1.0], [1.0, 0.25], [1.0, 1.0]],
            dtype=jnp.float32,
        ),
    )

    np.testing.assert_allclose(
        backfilled,
        np.asarray(
            [[0.4, 0.0], [0.4, 0.0], [0.0, 0.8], [0.0, 0.0]],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        windowed,
        np.asarray(
            [[0.4, 0.0], [0.1, 0.0], [0.0, 0.05], [0.0, 0.0]],
            dtype=np.float32,
        ),
    )


def test_progressive_quality_imitation_keeps_near_feasible_rebound_only() -> None:
    # Event 0 matches the recurrent bad lateral/sub-net mode from v40.  Event
    # 1 is the observed near-feasible outlier: it is not a strict success, but
    # contains useful action evidence worth consolidating.
    weights = progressive_quality_imitation_event_weight(
        hit_event=jnp.asarray([True, True, True]),
        rewarded_hit_was_event_rebound=jnp.asarray([True, True, False]),
        outgoing_z_m_s=jnp.asarray([1.04, 3.46, 3.46]),
        outgoing_forward_m_s=jnp.asarray([3.05, 3.71, 3.71]),
        predicted_net_clearance_m=jnp.asarray([-5.86, -0.43, -0.43]),
        return_direction_signed_score=jnp.asarray([0.60, 0.80, 0.80]),
        body_fall=jnp.asarray([False, False, False]),
        episode_completed_in_rollout=jnp.asarray([False, False, False]),
        episode_fell_in_rollout=jnp.asarray([False, False, False]),
        target_outgoing_z_m_s=1.0,
        target_forward_m_s=4.0,
        target_predicted_net_clearance_m=0.0,
        target_return_direction_signed_score=0.65,
        forward_softness_m_s=1.0,
        vertical_softness_m_s=0.75,
        clearance_softness_m=0.75,
        direction_softness=0.10,
        min_weight=0.02,
        require_episode_no_fall=False,
    )

    assert float(weights[0]) == 0.0
    assert float(weights[1]) > 0.10
    assert float(weights[2]) == 0.0

    known_future_fall_weight = progressive_quality_imitation_event_weight(
        hit_event=jnp.asarray([True]),
        rewarded_hit_was_event_rebound=jnp.asarray([True]),
        outgoing_z_m_s=jnp.asarray([3.46]),
        outgoing_forward_m_s=jnp.asarray([3.71]),
        predicted_net_clearance_m=jnp.asarray([-0.43]),
        return_direction_signed_score=jnp.asarray([0.80]),
        body_fall=jnp.asarray([False]),
        episode_completed_in_rollout=jnp.asarray([True]),
        episode_fell_in_rollout=jnp.asarray([True]),
        target_outgoing_z_m_s=1.0,
        target_forward_m_s=4.0,
        target_predicted_net_clearance_m=0.0,
        target_return_direction_signed_score=0.65,
        forward_softness_m_s=1.0,
        vertical_softness_m_s=0.75,
        clearance_softness_m=0.75,
        direction_softness=0.10,
        min_weight=0.02,
        require_episode_no_fall=False,
    )
    assert float(known_future_fall_weight[0]) == 0.0

    strict = quality_success_event_mask(
        hit_event=jnp.asarray([True]),
        rewarded_hit_was_event_rebound=jnp.asarray([True]),
        outgoing_z_m_s=jnp.asarray([3.46]),
        outgoing_forward_m_s=jnp.asarray([3.71]),
        predicted_net_clearance_m=jnp.asarray([-0.43]),
        return_direction_signed_score=jnp.asarray([0.80]),
        body_fall=jnp.asarray([False]),
        episode_completed_in_rollout=jnp.asarray([False]),
        episode_fell_in_rollout=jnp.asarray([False]),
        min_outgoing_z_m_s=1.0,
        min_forward_m_s=4.0,
        min_predicted_net_clearance_m=0.0,
        min_return_direction_signed_score=0.65,
        require_episode_no_fall=True,
    )
    assert not bool(strict[0])


def test_quality_success_requires_ballistics_direction_and_completed_no_fall() -> None:
    done = jnp.asarray(
        [[False, False, False], [False, False, False], [True, True, False]]
    )
    body_fall = jnp.asarray(
        [[False, False, False], [False, False, False], [False, True, False]]
    )
    completed, fell = future_episode_outcome(done, body_fall)

    actual = quality_success_event_mask(
        hit_event=jnp.asarray(
            [[False, False, False], [True, True, True], [False, False, False]]
        ),
        rewarded_hit_was_event_rebound=jnp.ones((3, 3), dtype=jnp.bool_),
        outgoing_z_m_s=jnp.full((3, 3), 1.5),
        outgoing_forward_m_s=jnp.full((3, 3), 5.0),
        predicted_net_clearance_m=jnp.asarray(
            [[0.0, 0.0, 0.0], [0.3, 0.3, -0.1], [0.0, 0.0, 0.0]]
        ),
        return_direction_signed_score=jnp.asarray(
            [[0.0, 0.0, 0.0], [0.8, 0.8, 0.8], [0.0, 0.0, 0.0]]
        ),
        body_fall=body_fall,
        episode_completed_in_rollout=completed,
        episode_fell_in_rollout=fell,
        min_outgoing_z_m_s=1.0,
        min_forward_m_s=4.0,
        min_predicted_net_clearance_m=0.2,
        min_return_direction_signed_score=0.65,
        require_episode_no_fall=True,
    )

    # Lane 0 has a good completed return.  Lane 1 later falls, while lane 2
    # neither clears the net nor completes inside this rollout.
    np.testing.assert_array_equal(
        actual,
        np.asarray(
            [[False, False, False], [True, False, False], [False, False, False]]
        ),
    )


def test_stage3_static_rollout_budget_never_exceeds_hard_cap() -> None:
    iterations, executed, unused = _full_batch_budget(
        total_env_steps=20_000_000,
        steps_per_iteration=512 * 64,
    )

    assert iterations == 610
    assert executed == 19_988_480
    assert unused == 11_520
    assert executed + unused == 20_000_000


def test_stage3_static_rollout_budget_rejects_cap_smaller_than_one_batch() -> None:
    with np.testing.assert_raises_regex(ValueError, "smaller than one static JIT rollout"):
        _full_batch_budget(total_env_steps=1_000, steps_per_iteration=32_768)


def test_resume_metrics_history_drops_future_and_duplicate_rows(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        "\n".join(
            [
                '{"iteration": 1, "reward": 1.0}',
                '{"iteration": 2, "reward": 2.0}',
                '{"iteration": 2, "reward": 2.5}',
                '{"iteration": 3, "reward": 3.0}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = reconcile_metrics_history(metrics_path, checkpoint_iteration=2)

    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert result == {"input_rows": 4, "retained_rows": 2, "removed_rows": 2}
    assert [row["iteration"] for row in rows] == [1, 2]
    assert rows[-1]["reward"] == 2.5


def test_resume_metrics_history_rejects_malformed_iteration(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text('{"iteration": "bad"}\n', encoding="utf-8")

    with np.testing.assert_raises_regex(ValueError, "integer iteration"):
        reconcile_metrics_history(metrics_path, checkpoint_iteration=0)


def test_latent_policy_sampling_returns_raw_gaussian_without_pre_tanh() -> None:
    agent = init_agent(jax.random.PRNGKey(0), obs_size=3, action_size=2, hidden=(4,), action_std_init=0.1)
    agent["policy"][-1]["b"] = jnp.array([4.0, -4.0])
    obs = jnp.zeros((1, 3))

    latent_action, raw, _ = sample_action(
        agent,
        obs,
        jax.random.PRNGKey(1),
        squash_action=False,
    )
    np.testing.assert_allclose(latent_action, raw)
    assert float(latent_action[0, 0]) > 1.0

    muscle_action, raw2, _ = sample_action(
        agent,
        obs,
        jax.random.PRNGKey(1),
        squash_action=True,
    )
    np.testing.assert_allclose(muscle_action, np.tanh(raw2), atol=1e-7)
    assert np.max(np.abs(muscle_action)) <= 1.0


def test_distal_policy_gradient_mask_freezes_trunk_and_unlisted_outputs() -> None:
    agent = init_agent(
        jax.random.PRNGKey(0),
        obs_size=3,
        action_size=4,
        hidden=(5, 6),
        action_std_init=0.1,
    )
    grads = jax.tree_util.tree_map(jnp.ones_like, agent)

    masked = mask_policy_update_gradients(
        grads,
        jnp.asarray([0.0, 1.0, 0.0, 1.0]),
    )

    for layer in masked["policy"][:-1]:
        np.testing.assert_array_equal(layer["w"], 0.0)
        np.testing.assert_array_equal(layer["b"], 0.0)
    np.testing.assert_array_equal(
        masked["policy"][-1]["w"],
        np.tile(np.asarray([0.0, 1.0, 0.0, 1.0]), (6, 1)),
    )
    np.testing.assert_array_equal(
        masked["policy"][-1]["b"],
        np.asarray([0.0, 1.0, 0.0, 1.0]),
    )
    np.testing.assert_array_equal(
        masked["log_std"],
        np.asarray([0.0, 1.0, 0.0, 1.0]),
    )
    for layer in masked["value"]:
        np.testing.assert_array_equal(layer["w"], 1.0)
        np.testing.assert_array_equal(layer["b"], 1.0)


def test_selected_delta_adapter_is_identity_then_changes_only_selected_means() -> None:
    key = jax.random.PRNGKey(17)
    base = init_agent(
        key,
        obs_size=3,
        action_size=4,
        hidden=(5, 6),
        action_std_init=0.1,
    )
    adapted = init_agent(
        key,
        obs_size=3,
        action_size=4,
        hidden=(5, 6),
        action_std_init=0.1,
        policy_delta_hidden=(7, 5),
    )
    obs = jnp.asarray([[0.2, -0.4, 0.7], [-0.1, 0.3, 0.9]])
    base_mean, _ = _dist(base, obs)
    initial_mean, _ = _dist(adapted, obs)
    np.testing.assert_array_equal(initial_mean, base_mean)
    for before, after in zip(
        jax.tree_util.tree_leaves(base["policy"]),
        jax.tree_util.tree_leaves(adapted["policy"]),
        strict=True,
    ):
        np.testing.assert_array_equal(after, before)

    grads = jax.tree_util.tree_map(jnp.ones_like, adapted)
    mask = jnp.asarray([0.0, 1.0, 0.0, 1.0])
    masked = mask_selected_delta_adapter_gradients(grads, mask)
    for leaf in jax.tree_util.tree_leaves(masked["policy"]):
        np.testing.assert_array_equal(leaf, 0.0)
    for layer in masked["policy_delta"][:-1]:
        np.testing.assert_array_equal(layer["w"], 1.0)
        np.testing.assert_array_equal(layer["b"], 1.0)
    np.testing.assert_array_equal(
        masked["policy_delta"][-1]["w"],
        np.tile(np.asarray([0.0, 1.0, 0.0, 1.0]), (5, 1)),
    )
    np.testing.assert_array_equal(masked["policy_delta"][-1]["b"], mask)
    np.testing.assert_array_equal(masked["log_std"], mask)

    updated = optax.apply_updates(adapted, jax.tree_util.tree_map(lambda value: -0.01 * value, masked))
    updated_mean, _ = _dist(updated, obs)
    np.testing.assert_array_equal(updated_mean[:, [0, 2]], base_mean[:, [0, 2]])
    assert not np.array_equal(np.asarray(updated_mean[:, [1, 3]]), np.asarray(base_mean[:, [1, 3]]))
    for before, after in zip(
        jax.tree_util.tree_leaves(base["policy"]),
        jax.tree_util.tree_leaves(updated["policy"]),
        strict=True,
    ):
        np.testing.assert_array_equal(after, before)


def test_selected_delta_adapter_checkpoint_roundtrip_preserves_composed_policy(tmp_path: Path) -> None:
    agent = init_agent(
        jax.random.PRNGKey(21),
        obs_size=3,
        action_size=4,
        hidden=(5,),
        action_std_init=0.2,
        policy_delta_hidden=(6,),
    )
    agent["policy_delta"][-1]["b"] = jnp.asarray([0.0, 0.3, 0.0, -0.2])
    optimizer = optax.adam(1e-4)
    opt_state = optimizer.init(agent)
    checkpoint = tmp_path / "adapter.npz"
    save_training_checkpoint(
        checkpoint,
        agent=agent,
        optimizer_state=opt_state,
        obs_rms=ObsRms.create(3),
        rng_key=jax.random.PRNGKey(22),
        metadata={"iteration": 1, "env_steps": 8, "action_size": 4, "obs_size": 3},
    )
    restored = load_training_checkpoint(
        checkpoint,
        agent_template=agent,
        optimizer_state_template=opt_state,
    )
    obs = jnp.asarray([[0.5, -0.2, 0.1]])
    np.testing.assert_array_equal(_dist(restored.agent, obs)[0], _dist(agent, obs)[0])


def test_selected_refinement_delta_freezes_learned_phase_a_and_changes_only_wrist_rows() -> None:
    phase_a = init_agent(
        jax.random.PRNGKey(41),
        obs_size=3,
        action_size=4,
        hidden=(5, 6),
        action_std_init=0.12,
        policy_delta_hidden=(7, 5),
    )
    phase_a["policy_delta"][-1]["w"] = jnp.asarray(
        np.arange(20, dtype=np.float32).reshape(5, 4) * 0.001
    )
    phase_a["policy_delta"][-1]["b"] = jnp.asarray([0.01, -0.02, 0.03, -0.04])
    refined = init_agent(
        jax.random.PRNGKey(42),
        obs_size=3,
        action_size=4,
        hidden=(5, 6),
        action_std_init=0.12,
        policy_delta_hidden=(7, 5),
        policy_refinement_delta_hidden=(6, 5),
    )
    refined = {
        **refined,
        "policy": phase_a["policy"],
        "policy_delta": phase_a["policy_delta"],
    }
    obs = jnp.asarray([[0.2, -0.4, 0.7], [-0.1, 0.3, 0.9]])
    phase_a_mean, _ = _dist(phase_a, obs)
    initial_mean, _ = _dist(refined, obs)
    np.testing.assert_array_equal(initial_mean, phase_a_mean)

    grads = jax.tree_util.tree_map(jnp.ones_like, refined)
    mask = jnp.asarray([0.0, 1.0, 0.0, 1.0])
    masked = mask_selected_refinement_delta_adapter_gradients(grads, mask)
    for frozen_name in ("policy", "policy_delta"):
        for leaf in jax.tree_util.tree_leaves(masked[frozen_name]):
            np.testing.assert_array_equal(leaf, 0.0)
    for layer in masked["policy_refinement_delta"][:-1]:
        np.testing.assert_array_equal(layer["w"], 1.0)
        np.testing.assert_array_equal(layer["b"], 1.0)
    np.testing.assert_array_equal(
        masked["policy_refinement_delta"][-1]["w"],
        np.tile(np.asarray([0.0, 1.0, 0.0, 1.0]), (5, 1)),
    )
    np.testing.assert_array_equal(masked["policy_refinement_delta"][-1]["b"], mask)
    np.testing.assert_array_equal(masked["log_std"], mask)

    updated = optax.apply_updates(refined, jax.tree_util.tree_map(lambda value: -0.01 * value, masked))
    updated_mean, _ = _dist(updated, obs)
    np.testing.assert_array_equal(updated_mean[:, [0, 2]], phase_a_mean[:, [0, 2]])
    assert not np.array_equal(np.asarray(updated_mean[:, [1, 3]]), np.asarray(phase_a_mean[:, [1, 3]]))
    for frozen_name in ("policy", "policy_delta"):
        for before, after in zip(
            jax.tree_util.tree_leaves(phase_a[frozen_name]),
            jax.tree_util.tree_leaves(updated[frozen_name]),
            strict=True,
        ):
            np.testing.assert_array_equal(after, before)


def test_selected_physical_correction_has_selected_only_distribution_and_frozen_inheritance() -> None:
    inherited = init_agent(
        jax.random.PRNGKey(51),
        obs_size=3,
        action_size=4,
        hidden=(5,),
        action_std_init=0.12,
        policy_delta_hidden=(6,),
    )
    corrected = init_agent(
        jax.random.PRNGKey(52),
        obs_size=3,
        action_size=4,
        hidden=(5,),
        action_std_init=0.12,
        policy_delta_hidden=(6,),
        policy_correction_hidden=(7,),
        correction_action_size=2,
        correction_std_init=(0.15, 0.20),
    )
    corrected = {
        **corrected,
        "policy": inherited["policy"],
        "policy_delta": inherited["policy_delta"],
    }
    obs = jnp.asarray([[0.2, -0.1, 0.6], [-0.3, 0.4, 0.1]])
    np.testing.assert_array_equal(
        _inherited_policy_mean(corrected, obs),
        _inherited_policy_mean(inherited, obs),
    )
    correction_mean, correction_std = _selected_correction_dist(
        corrected,
        obs,
        std_min=jnp.asarray([0.03, 0.04]),
        std_max=jnp.asarray([0.30, 0.40]),
    )
    assert correction_mean.shape == (2, 2)
    assert correction_std.shape == (2,)
    raw = jnp.asarray([[0.1, -0.2], [0.3, 0.4]])
    logp = _tanh_normal_logprob(correction_mean, correction_std, raw, jnp.tanh(raw))
    assert logp.shape == (2,)

    grads = jax.tree_util.tree_map(jnp.ones_like, corrected)
    masked = mask_selected_physical_correction_gradients(grads)
    for frozen_name in ("policy", "policy_delta"):
        for leaf in jax.tree_util.tree_leaves(masked[frozen_name]):
            np.testing.assert_array_equal(leaf, 0.0)
    np.testing.assert_array_equal(masked["log_std"], 0.0)
    for trainable_name in ("policy_correction", "correction_log_std", "value"):
        for leaf in jax.tree_util.tree_leaves(masked[trainable_name]):
            np.testing.assert_array_equal(leaf, 1.0)


def test_distal_exploration_contract_suppresses_only_frozen_actions() -> None:
    agent = init_agent(
        jax.random.PRNGKey(0),
        obs_size=3,
        action_size=4,
        hidden=(5,),
        action_std_init=0.16,
    )
    original_policy = jax.tree_util.tree_map(np.asarray, agent["policy"])

    contracted = apply_policy_exploration_contract(
        agent,
        action_size=4,
        trainable_action_indices=(1, 3),
        frozen_action_std=0.001,
    )

    np.testing.assert_allclose(
        np.exp(np.asarray(contracted["log_std"])),
        np.asarray([0.001, 0.16, 0.001, 0.16]),
        rtol=1e-6,
    )
    for before, after in zip(
        jax.tree_util.tree_leaves(original_policy),
        jax.tree_util.tree_leaves(contracted["policy"]),
        strict=True,
    ):
        np.testing.assert_array_equal(before, after)


def test_fixed_trainable_action_std_zeroes_only_log_std_gradients() -> None:
    agent = init_agent(
        jax.random.PRNGKey(43),
        obs_size=3,
        action_size=4,
        hidden=(5,),
        action_std_init=0.05,
        policy_delta_hidden=(6,),
    )
    grads = jax.tree_util.tree_map(jnp.ones_like, agent)
    frozen = freeze_action_std_gradients(grads)
    np.testing.assert_array_equal(frozen["log_std"], 0.0)
    for name in ("policy", "policy_delta", "value"):
        for leaf in jax.tree_util.tree_leaves(frozen[name]):
            np.testing.assert_array_equal(leaf, 1.0)


def test_training_checkpoint_restores_optimizer_rng_and_progress(tmp_path: Path) -> None:
    key = jax.random.PRNGKey(3)
    agent = init_agent(key, obs_size=3, action_size=2, hidden=(4,), action_std_init=0.2)
    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(agent)
    obs_rms = ObsRms(jnp.arange(3.0), jnp.ones(3) * 2.0, jnp.asarray(17.0))
    rng_key = jax.random.PRNGKey(9)
    path = tmp_path / "policy_latest.npz"

    save_training_checkpoint(
        path,
        agent=agent,
        optimizer_state=opt_state,
        obs_rms=obs_rms,
        rng_key=rng_key,
        metadata={"iteration": 7, "env_steps": 896, "action_size": 2, "obs_size": 3},
    )
    restored = load_training_checkpoint(
        path,
        agent_template=agent,
        optimizer_state_template=opt_state,
    )

    np.testing.assert_allclose(restored.obs_rms.mean, obs_rms.mean)
    np.testing.assert_array_equal(restored.rng_key, rng_key)
    assert restored.metadata["iteration"] == 7
    assert restored.metadata["env_steps"] == 896
    assert len(restored.metadata["training_payload_sha256"]) == 64
    assert not list(tmp_path.glob("*.tmp*"))
    for actual, expected in zip(
        jax.tree_util.tree_leaves(restored.agent),
        jax.tree_util.tree_leaves(agent),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected)
    for actual, expected in zip(
        jax.tree_util.tree_leaves(restored.optimizer_state),
        jax.tree_util.tree_leaves(opt_state),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected)


def test_actor_only_reward_repair_initialization_accepts_reward_change(
    tmp_path: Path,
) -> None:
    agent = init_agent(jax.random.PRNGKey(3), 3, 2, hidden=(4,), action_std_init=0.2)
    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(agent)
    common_control = {
        "schema_version": "incoming_hit_direct_action_v1",
        "filter_finger_observation": False,
        "frozen_base_residual": {"binding_sha256": "base"},
        "racket_attachment": {"attachment_hash": "grip"},
    }
    physical_environment = {
        "scene_sha256": "scene",
        "full_action_size": 2,
        "control_substeps": 10,
    }
    source_control = {
        **common_control,
        "control_hash": "source",
        "environment_abi": {
            **physical_environment,
            "reward_weights": {"hit_bonus": 12.0},
            "reward_semantics": "v3",
        },
    }
    runtime_control = {
        **common_control,
        "control_hash": "runtime",
        "environment_abi": {
            **physical_environment,
            "reward_weights": {"hit_bonus": 8.0, "body_fall": 50.0},
            "reward_semantics": "v4",
            "return_constraints": {"min_clearance_m": 0.2},
        },
    }
    feed_manifest = {"content_sha256": "feed"}
    checkpoint = tmp_path / "source.npz"
    save_training_checkpoint(
        checkpoint,
        agent=agent,
        optimizer_state=opt_state,
        obs_rms=ObsRms.create(3),
        rng_key=jax.random.PRNGKey(9),
        metadata={
            "iteration": 130,
            "env_steps": 4_259_840,
            "action_size": 2,
            "obs_size": 3,
            "control_manifest": source_control,
            "training_feed_manifest": feed_manifest,
        },
    )
    fake_env = SimpleNamespace(
        observation_size=3,
        action_size=2,
        control_manifest=runtime_control,
        feed_bank_manifest=feed_manifest,
    )

    restored, binding = load_actor_initialization(
        checkpoint,
        agent_template=agent,
        optimizer_state_template=opt_state,
        env=fake_env,
    )

    assert restored.metadata["env_steps"] == 4_259_840
    assert binding["transferred"] == ["policy_actor", "observation_normalizer"]
    assert "optimizer_state" in binding["reset"]
    assert len(binding["binding_sha256"]) == 64


def test_actor_initialization_resets_only_selected_correction_exploration_std(
    tmp_path: Path,
) -> None:
    source = init_agent(
        jax.random.PRNGKey(71),
        3,
        2,
        hidden=(4,),
        action_std_init=0.2,
        policy_delta_hidden=(5,),
        policy_correction_hidden=(6,),
        correction_action_size=2,
        correction_std_init=(0.001, 0.002),
    )
    source = {
        **source,
        "policy_correction": jax.tree_util.tree_map(
            lambda value: value + jnp.asarray(0.125, dtype=value.dtype),
            source["policy_correction"],
        ),
    }
    runtime = init_agent(
        jax.random.PRNGKey(72),
        3,
        2,
        hidden=(4,),
        action_std_init=0.2,
        policy_delta_hidden=(5,),
        policy_correction_hidden=(6,),
        correction_action_size=2,
        correction_std_init=(0.004, 0.008),
    )
    optimizer = optax.adam(3e-4)
    control = {
        "schema_version": "incoming_hit_direct_action_v1",
        "control_hash": "same",
        "filter_finger_observation": False,
        "frozen_base_residual": {"binding_sha256": "base"},
        "racket_attachment": {"attachment_hash": "grip"},
        "environment_abi": {"scene_sha256": "scene", "full_action_size": 2},
    }
    feed_manifest = {"content_sha256": "feed"}
    checkpoint = tmp_path / "source_correction.npz"
    save_training_checkpoint(
        checkpoint,
        agent=source,
        optimizer_state=optimizer.init(source),
        obs_rms=ObsRms.create(3),
        rng_key=jax.random.PRNGKey(73),
        metadata={
            "iteration": 60,
            "env_steps": 1_966_080,
            "action_size": 2,
            "obs_size": 3,
            "hidden": [4],
            "config": {
                "hidden": [4],
                "action_std_init": 0.2,
                "policy_delta_hidden": [5],
                "policy_refinement_delta_hidden": [],
                "policy_correction_hidden": [6],
                "policy_trainable_action_indices": [0, 1],
                "correction_std_init": [0.001, 0.002],
                "teacher_action_prior_mode": "none",
            },
            "control_manifest": control,
            "training_feed_manifest": feed_manifest,
        },
    )

    restored, binding = load_actor_initialization(
        checkpoint,
        agent_template=runtime,
        optimizer_state_template=optimizer.init(runtime),
        env=SimpleNamespace(
            observation_size=3,
            action_size=2,
            control_manifest=control,
            feed_bank_manifest=feed_manifest,
        ),
        reset_correction_std=True,
    )

    for actual, expected in zip(
        jax.tree_util.tree_leaves(restored.agent["policy_correction"]),
        jax.tree_util.tree_leaves(source["policy_correction"]),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        restored.agent["correction_log_std"], runtime["correction_log_std"]
    )
    assert binding["schema_version"] == (
        "stage3_actor_only_correction_exploration_repair_initialization_v1"
    )
    assert "selected_physical_correction_mean" in binding["transferred"]
    assert "selected_physical_correction_exploration_std" in binding["reset"]
    assert binding["correction_std_reset"]["runtime_std_max"] > (
        binding["correction_std_reset"]["source_std_max"]
    )


def test_actor_initialization_requires_exact_inherited_action_prior_lineage(
    tmp_path: Path,
) -> None:
    agent = init_agent(
        jax.random.PRNGKey(81),
        3,
        2,
        hidden=(4,),
        action_std_init=0.2,
    )
    optimizer = optax.adam(3e-4)
    control = {
        "schema_version": "incoming_hit_direct_action_v1",
        "control_hash": "same",
        "filter_finger_observation": False,
        "frozen_base_residual": {"binding_sha256": "base"},
        "racket_attachment": {"attachment_hash": "grip"},
        "environment_abi": {"scene_sha256": "scene", "full_action_size": 2},
    }
    feed_manifest = {"content_sha256": "feed"}
    teacher_binding = {
        "schema_version": "stage3_cpu_certified_exploration_prior_binding_v1",
        "source_checkpoint_sha256": "a" * 64,
        "trajectory_sha256": "b" * 64,
    }
    teacher_binding["binding_sha256"] = training_module._stable_json_hash(
        teacher_binding
    )
    teacher_report = {
        "schema_version": "stage3_selected_correction_bc_pretrain_v1",
        "passed": True,
        "steps": 0,
        "teacher_binding": teacher_binding,
    }
    teacher_report["report_sha256"] = training_module._stable_json_hash(
        teacher_report
    )
    checkpoint = tmp_path / "source_prior.npz"
    save_training_checkpoint(
        checkpoint,
        agent=agent,
        optimizer_state=optimizer.init(agent),
        obs_rms=ObsRms.create(3),
        rng_key=jax.random.PRNGKey(82),
        metadata={
            "iteration": 5,
            "env_steps": 160,
            "action_size": 2,
            "obs_size": 3,
            "hidden": [4],
            "config": {
                "hidden": [4],
                "action_std_init": 0.2,
                "teacher_action_prior_mode": "time_interpolated_frozen_plus_delta",
                "teacher_prior_time_to_intercept_s": [0.0, 0.1],
                "teacher_prior_correction_raw": [[0.0], [0.1]],
            },
            "teacher_bc_pretrain_report": teacher_report,
            "control_manifest": control,
            "training_feed_manifest": feed_manifest,
        },
    )
    env = SimpleNamespace(
        observation_size=3,
        action_size=2,
        control_manifest=control,
        feed_bank_manifest=feed_manifest,
    )

    with np.testing.assert_raises_regex(ValueError, "drop the inherited frozen action prior"):
        load_actor_initialization(
            checkpoint,
            agent_template=agent,
            optimizer_state_template=optimizer.init(agent),
            env=env,
        )

    _restored, binding = load_actor_initialization(
        checkpoint,
        agent_template=agent,
        optimizer_state_template=optimizer.init(agent),
        env=env,
        quality_teacher_binding=teacher_binding,
    )
    assert binding["action_prior_lineage"]["teacher_binding_sha256"] == (
        teacher_binding["binding_sha256"]
    )


def test_resume_action_prior_reuses_certification_source_and_timing() -> None:
    teacher_binding = {
        "schema_version": "stage3_cpu_certified_exploration_prior_binding_v1",
        "source_checkpoint_sha256": "a" * 64,
        "trajectory_sha256": "b" * 64,
        "base_timing_transfer": {
            "source_phase_advance_s": 0.58,
            "runtime_phase_advance_s": 0.28,
        },
    }
    teacher_binding["binding_sha256"] = training_module._stable_json_hash(
        teacher_binding
    )
    report = {
        "schema_version": "stage3_selected_correction_bc_pretrain_v1",
        "passed": True,
        "steps": 0,
        "teacher_binding": teacher_binding,
    }
    report["report_sha256"] = training_module._stable_json_hash(report)
    metadata = {
        "config": {
            "teacher_action_prior_mode": "time_interpolated_frozen_plus_delta",
            "teacher_prior_time_to_intercept_s": [0.0, 0.1],
            "teacher_prior_correction_raw": [[0.0], [0.1]],
        },
        "teacher_bc_pretrain_report": report,
    }

    binding, source_sha256, source_phase = _resume_action_prior_source(metadata)

    assert binding == teacher_binding
    assert source_sha256 == "a" * 64
    np.testing.assert_allclose(source_phase, 0.58)


def test_actor_initialization_adds_zero_adapter_without_changing_legacy_actor(tmp_path: Path) -> None:
    source = init_agent(jax.random.PRNGKey(31), 3, 2, hidden=(4,), action_std_init=0.2)
    runtime = init_agent(
        jax.random.PRNGKey(32),
        3,
        2,
        hidden=(4,),
        action_std_init=0.2,
        policy_delta_hidden=(5,),
    )
    source_optimizer = optax.adam(3e-4)
    runtime_optimizer = optax.adam(3e-4)
    common_control = {
        "schema_version": "incoming_hit_direct_action_v1",
        "control_hash": "same",
        "filter_finger_observation": False,
        "frozen_base_residual": {"binding_sha256": "base"},
        "racket_attachment": {"attachment_hash": "grip"},
        "environment_abi": {"scene_sha256": "scene", "full_action_size": 2},
    }
    feed_manifest = {"content_sha256": "feed"}
    checkpoint = tmp_path / "legacy_source.npz"
    save_training_checkpoint(
        checkpoint,
        agent=source,
        optimizer_state=source_optimizer.init(source),
        obs_rms=ObsRms.create(3),
        rng_key=jax.random.PRNGKey(33),
        metadata={
            "iteration": 2,
            "env_steps": 16,
            "action_size": 2,
            "obs_size": 3,
            "control_manifest": common_control,
            "training_feed_manifest": feed_manifest,
        },
    )
    restored, binding = load_actor_initialization(
        checkpoint,
        agent_template=runtime,
        optimizer_state_template=runtime_optimizer.init(runtime),
        env=SimpleNamespace(
            observation_size=3,
            action_size=2,
            control_manifest=common_control,
            feed_bank_manifest=feed_manifest,
        ),
    )

    assert "policy_delta" not in restored.agent
    assert binding["transferred"] == ["policy_actor", "observation_normalizer"]
    assert binding["initialized_zero"] == ["policy_delta_adapter"]
    for layer in runtime["policy_delta"][-1:]:
        np.testing.assert_array_equal(layer["w"], 0.0)
        np.testing.assert_array_equal(layer["b"], 0.0)


def test_actor_initialization_accepts_exact_initial_authority_schedule(tmp_path: Path) -> None:
    agent = init_agent(jax.random.PRNGKey(3), 3, 2, hidden=(4,), action_std_init=0.2)
    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(agent)
    initial = np.asarray([0.25, 0.25], dtype="<f8")
    target = np.asarray([0.25, 1.0], dtype="<f8")
    source_frozen = {
        "schema_version": "incoming_hit_frozen_base_residual_v1",
        "artifact_content_sha256": "base-content",
        "source_checkpoint": "base-checkpoint",
        "selected_skill": None,
        "residual_scale": 0.25,
        "actor_obs_size": 3,
        "actor_action_size": 2,
        "files": [],
    }
    runtime_frozen = {
        **source_frozen,
        "residual_scale_overrides": [{"actuator_name": "arm", "actuator_id": 1, "scale": 1.0}],
        "residual_scale_vector_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
        "residual_scale_schedule": {
            "schema_version": "incoming_hit_residual_authority_schedule_v1",
            "interpolation": "linear_env_steps",
            "initial_scale": 0.25,
            "ramp_steps": 1_000_000,
            "scheduled_actuators": [
                {
                    "actuator_name": "arm",
                    "actuator_id": 1,
                    "initial_scale": 0.25,
                    "target_scale": 1.0,
                }
            ],
            "initial_scale_vector_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
            "target_scale_vector_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
        },
    }
    common = {
        "schema_version": "incoming_hit_direct_action_v1",
        "filter_finger_observation": False,
        "racket_attachment": {"attachment_hash": "grip"},
        "environment_abi": {"scene_sha256": "scene", "full_action_size": 2},
    }
    source_control = {**common, "control_hash": "source", "frozen_base_residual": source_frozen}
    runtime_control = {**common, "control_hash": "runtime", "frozen_base_residual": runtime_frozen}
    feed_manifest = {"content_sha256": "feed"}
    checkpoint = tmp_path / "source.npz"
    save_training_checkpoint(
        checkpoint,
        agent=agent,
        optimizer_state=opt_state,
        obs_rms=ObsRms.create(3),
        rng_key=jax.random.PRNGKey(9),
        metadata={
            "iteration": 10,
            "env_steps": 100,
            "action_size": 2,
            "obs_size": 3,
            "control_manifest": source_control,
            "training_feed_manifest": feed_manifest,
        },
    )
    fake_env = SimpleNamespace(
        observation_size=3,
        action_size=2,
        control_manifest=runtime_control,
        feed_bank_manifest=feed_manifest,
    )

    _restored, binding = load_actor_initialization(
        checkpoint,
        agent_template=agent,
        optimizer_state_template=opt_state,
        env=fake_env,
    )

    assert binding["schema_version"] == "stage3_actor_only_scheduled_authority_initialization_v1"
    transfer = binding["residual_authority_transfer"]
    assert transfer["changed_actuator_count"] == 1
    assert transfer["source_effective_scale_vector_sha256"] == transfer["runtime_initial_scale_vector_sha256"]


def test_actor_initialization_requires_quality_teacher_for_base_timing_change(
    tmp_path: Path,
) -> None:
    agent = init_agent(jax.random.PRNGKey(13), 3, 2, hidden=(4,), action_std_init=0.2)
    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(agent)
    frozen_common = {
        "schema_version": "incoming_hit_frozen_base_residual_v1",
        "artifact_content_sha256": "base-content",
        "source_checkpoint": "base-checkpoint",
        "selected_skill": None,
        "residual_scale": 0.25,
        "actor_obs_size": 3,
        "actor_action_size": 2,
        "files": [],
    }
    source_frozen = {
        **frozen_common,
        "binding_sha256": "source-binding",
        "phase_advance_s": 0.58,
    }
    runtime_frozen = {
        **frozen_common,
        "binding_sha256": "runtime-binding",
        "phase_advance_s": 0.18,
    }
    common = {
        "schema_version": "incoming_hit_direct_action_v1",
        "filter_finger_observation": False,
        "racket_attachment": {"attachment_hash": "grip"},
    }
    source_control = {
        **common,
        "control_hash": "source",
        "frozen_base_residual": source_frozen,
        "environment_abi": {
            "scene_sha256": "scene",
            "full_action_size": 2,
            "swing_phase_advance_s": 0.58,
        },
    }
    runtime_control = {
        **common,
        "control_hash": "runtime",
        "frozen_base_residual": runtime_frozen,
        "environment_abi": {
            "scene_sha256": "scene",
            "full_action_size": 2,
            "swing_phase_advance_s": 0.18,
            "event_rebound_contact_semantics": (
                "single_event_impulse_with_stringbed_force_suppressed_"
                "during_cooldown_v2"
            ),
        },
    }
    checkpoint = tmp_path / "source_timing.npz"
    source_feed_manifest = {
        "schema_version": "incoming_shuttle_feed_bank_manifest_v1",
        "content_sha256": "source-feed",
        "sample_fingerprints": ["old-feed"],
    }
    runtime_feed_manifest = {
        "schema_version": "incoming_shuttle_feed_bank_manifest_v1",
        "content_sha256": "runtime-feed",
        "sample_fingerprints": ["teacher-feed", "later-feed"],
        "consumer_order": {
            "schema_version": "incoming_hit_curriculum_feed_order_v1",
            "mode": "explicit_fingerprint_order",
            "sample_fingerprints": ["teacher-feed", "later-feed"],
        },
    }
    save_training_checkpoint(
        checkpoint,
        agent=agent,
        optimizer_state=opt_state,
        obs_rms=ObsRms.create(3),
        rng_key=jax.random.PRNGKey(19),
        metadata={
            "iteration": 10,
            "env_steps": 100,
            "action_size": 2,
            "obs_size": 3,
            "control_manifest": source_control,
            "training_feed_manifest": source_feed_manifest,
        },
    )
    fake_env = SimpleNamespace(
        observation_size=3,
        action_size=2,
        control_manifest=runtime_control,
        feed_bank_manifest=runtime_feed_manifest,
    )

    with np.testing.assert_raises_regex(ValueError, "matching quality-teacher evidence"):
        load_actor_initialization(
            checkpoint,
            agent_template=agent,
            optimizer_state_template=opt_state,
            env=fake_env,
        )

    evidence = {
        "schema_version": "stage3_teacher_verified_base_timing_transfer_v1",
        "source_phase_advance_s": 0.58,
        "runtime_phase_advance_s": 0.18,
        "verification_source": "independent_cpu_mujoco_quality_replay_at_runtime_timing",
        "cpu_quality_verified": True,
        "source_cem_report_sha256": "a" * 64,
        "source_cpu_trace_sha256": "b" * 64,
    }
    evidence["evidence_sha256"] = training_module._stable_json_hash(evidence)
    evidence["teacher_cem_report_sha256"] = "c" * 64
    with np.testing.assert_raises_regex(
        ValueError,
        "impact semantics without matching",
    ):
        load_actor_initialization(
            checkpoint,
            agent_template=agent,
            optimizer_state_template=opt_state,
            env=fake_env,
            base_timing_transfer_evidence=evidence,
        )
    teacher_binding = {
        "schema_version": "stage3_quality_teacher_dataset_binding_v1",
        "verification_source": (
            "warp_training_backend_plus_independent_cpu_mujoco_quality_replay"
        ),
        "training_backend_quality_verified": True,
        "training_backend": "warp",
        "cpu_quality_verified": True,
        "outgoing_velocity_semantics": (
            "post_control_step_after_all_physics_substeps"
        ),
        "event_rebound_contact_semantics": (
            "single_event_impulse_with_stringbed_force_suppressed_"
            "during_cooldown_v2"
        ),
        "source_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "feed_fingerprint": "teacher-feed",
    }
    teacher_binding["binding_sha256"] = training_module._stable_json_hash(teacher_binding)
    _restored, binding = load_actor_initialization(
        checkpoint,
        agent_template=agent,
        optimizer_state_template=opt_state,
        env=fake_env,
        base_timing_transfer_evidence=evidence,
        quality_teacher_binding=teacher_binding,
    )

    assert binding["schema_version"] == "stage3_actor_only_teacher_timing_initialization_v1"
    assert binding["base_timing_transfer"]["source_phase_advance_s"] == 0.58
    assert binding["base_timing_transfer"]["runtime_phase_advance_s"] == 0.18
    assert binding["impact_semantics_transfer"]["mode"] == (
        "legacy_unmarked_to_single_event_cooldown_v2"
    )
    assert binding["feed_bank_transfer"]["teacher_feed_fingerprint"] == "teacher-feed"


def test_versioned_checkpoint_commits_directory_then_moves_latest_pointer(
    tmp_path: Path,
) -> None:
    agent = init_agent(jax.random.PRNGKey(1), 3, 2, hidden=(4,), action_std_init=0.2)
    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(agent)
    latest = save_versioned_training_checkpoint(
        tmp_path,
        agent=agent,
        optimizer_state=opt_state,
        obs_rms=ObsRms.create(3),
        rng_key=jax.random.PRNGKey(2),
        metadata={
            "checkpoint_version": "incoming_hit_training_v3",
            "iteration": 7,
            "env_steps": 896,
            "action_size": 2,
            "obs_size": 3,
        },
    )

    assert latest == tmp_path / "policy_latest.npz"
    assert latest.is_symlink()
    assert (tmp_path / "policy_latest.json").is_file()
    version = tmp_path / "checkpoints" / "checkpoint_000000000896"
    assert (version / "_COMPLETE.json").is_file()
    payload, metadata = resolve_training_checkpoint(tmp_path / "policy_latest.json")
    assert payload == (version / "policy.npz").resolve()
    assert metadata == (version / "policy.json").resolve()
    assert load_training_checkpoint_metadata(latest)["env_steps"] == 896

    restored = load_training_checkpoint(
        tmp_path / "policy_latest.json",
        agent_template=agent,
        optimizer_state_template=opt_state,
    )
    assert restored.metadata["versioned_checkpoint_schema"].endswith("_v1")


def test_versioned_checkpoint_rejects_pointer_or_payload_corruption(tmp_path: Path) -> None:
    agent = init_agent(jax.random.PRNGKey(1), 3, 2, hidden=(4,), action_std_init=0.2)
    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(agent)
    save_versioned_training_checkpoint(
        tmp_path,
        agent=agent,
        optimizer_state=opt_state,
        obs_rms=ObsRms.create(3),
        rng_key=jax.random.PRNGKey(2),
        metadata={"iteration": 1, "env_steps": 8, "action_size": 2, "obs_size": 3},
    )
    payload, _ = resolve_training_checkpoint(tmp_path / "policy_latest.json")
    with payload.open("ab") as handle:
        handle.write(b"corrupt")

    with np.testing.assert_raises_regex(ValueError, "fingerprint mismatch"):
        resolve_training_checkpoint(tmp_path / "policy_latest.json")


def test_versioned_checkpoint_accepts_exact_retry_but_rejects_same_step_drift(
    tmp_path: Path,
) -> None:
    agent = init_agent(jax.random.PRNGKey(1), 3, 2, hidden=(4,), action_std_init=0.2)
    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(agent)
    kwargs = {
        "agent": agent,
        "optimizer_state": opt_state,
        "obs_rms": ObsRms.create(3),
        "rng_key": jax.random.PRNGKey(2),
        "metadata": {
            "iteration": 1,
            "env_steps": 8,
            "action_size": 2,
            "obs_size": 3,
        },
    }
    save_versioned_training_checkpoint(tmp_path, **kwargs)
    save_versioned_training_checkpoint(tmp_path, **kwargs)
    assert load_training_checkpoint_metadata(tmp_path / "policy_latest.json")["env_steps"] == 8

    with np.testing.assert_raises_regex(ValueError, "version collision"):
        save_versioned_training_checkpoint(
            tmp_path,
            **{**kwargs, "rng_key": jax.random.PRNGKey(999)},
        )


def test_resume_feed_manifest_is_required_and_compared_strictly() -> None:
    manifest = {
        "schema_version": "incoming_shuttle_feed_bank_manifest_v1",
        "content_sha256": "a" * 64,
        "sample_fingerprints": ["b" * 64],
    }
    validate_training_feed_manifest(
        manifest,
        checkpoint_manifest=manifest.copy(),
        required=True,
    )
    with np.testing.assert_raises_regex(ValueError, "missing"):
        validate_training_feed_manifest(
            manifest,
            checkpoint_manifest=None,
            required=True,
        )
    changed = {**manifest, "content_sha256": "c" * 64}
    with np.testing.assert_raises_regex(ValueError, "contract changed"):
        validate_training_feed_manifest(
            manifest,
            checkpoint_manifest=changed,
            required=True,
        )


def test_production_main_passes_verified_feed_manifest_to_environment(tmp_path: Path, monkeypatch) -> None:
    manifest = {
        "schema_version": "incoming_shuttle_feed_bank_manifest_v1",
        "sample_fingerprints": ["feed-a"],
    }
    runtime_manifest = {
        **manifest,
        "consumer_order": {
            "schema_version": "incoming_hit_curriculum_feed_order_v1",
            "mode": "difficulty_sorted",
            "sample_fingerprints": ["feed-a"],
        },
    }
    artifact = SimpleNamespace(bank=[object()], manifest=manifest)
    paths = SimpleNamespace(
        scene_xml=tmp_path / "scene.xml",
        control_substeps=1,
        max_episode_steps=2,
        reward_weights={},
        stage3_lab={},
        return_constraints={"clearance_reward_mode": "signed_centered"},
        ppo_overrides={
            "num_minibatches": 3,
            "gamma": 0.97,
            "gae_lambda": 0.91,
            "clip_coef": 0.15,
            "value_coef": 0.4,
            "entropy_coef": 0.0001,
            "max_grad_norm": 0.3,
        },
        output_dir=tmp_path,
    )
    lab = SimpleNamespace(
        controller=object(),
        state_builder=object(),
        curriculum=object(),
        latent_checkpoint_dir=tmp_path / "latent",
    )
    runner = SimpleNamespace(
        _build_stage3_lab_components=lambda *args, **kwargs: lab,
        _ensure_feed_bank_artifact=lambda _paths: artifact,
        _ensure_scene=lambda _paths: None,
        load_incoming_hit_spec=lambda _spec: paths,
    )
    captured: dict[str, object] = {}

    class FakeEnv:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.model = SimpleNamespace(nu=2)
            self.feed_bank_manifest = runtime_manifest
            self.control_manifest = {
                "control_hash": "control-hash",
                "latent_checkpoint_fingerprint": "f" * 64,
            }

    monkeypatch.setitem(sys.modules, "run_incoming_shuttle_hit", runner)
    monkeypatch.setattr(training_module, "IncomingHitMjxEnv", FakeEnv)
    monkeypatch.setattr(training_module.jax, "devices", lambda: ["cpu"])
    validated: dict[str, object] = {}

    def fake_validate(*_args, **kwargs):
        validated.update(kwargs)
        return {"verified": True}

    monkeypatch.setattr(training_module, "validate_stage3_training_prerequisites", fake_validate)
    monkeypatch.setattr(
        training_module,
        "train",
        lambda *args, **kwargs: {"final": {}},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_incoming_hit_mjx",
            "--num-envs",
            "1",
            "--rollout-steps",
            "1",
            "--total-env-steps",
            "1",
            "--out-dir",
            str(tmp_path),
        ],
    )

    assert training_module.main() == 0
    assert captured["feed_bank"] is artifact.bank
    assert captured["feed_bank_manifest"] is manifest
    assert validated["training_feed_manifest"] is runtime_manifest


def test_unified_train_gpu_binds_runtime_feed_manifest(tmp_path: Path, monkeypatch) -> None:
    producer_manifest = {
        "schema_version": "incoming_shuttle_feed_bank_manifest_v1",
        "sample_fingerprints": ["feed-a"],
    }
    runtime_manifest = {
        **producer_manifest,
        "consumer_order": {
            "schema_version": "incoming_hit_curriculum_feed_order_v1",
            "mode": "difficulty_sorted",
            "sample_fingerprints": ["feed-a"],
        },
    }
    artifact = SimpleNamespace(bank=[object()], manifest=producer_manifest)
    paths = SimpleNamespace(
        scene_xml=tmp_path / "scene.xml",
        control_substeps=1,
        max_episode_steps=2,
        reward_weights={},
        stage3_lab={},
        return_constraints={"clearance_reward_mode": "signed_centered"},
        ppo_overrides={
            "num_minibatches": 3,
            "gamma": 0.97,
            "gae_lambda": 0.91,
            "clip_coef": 0.15,
            "value_coef": 0.4,
            "entropy_coef": 0.0001,
            "max_grad_norm": 0.3,
        },
        output_dir=tmp_path,
    )
    lab = SimpleNamespace(
        controller=object(),
        state_builder=object(),
        curriculum=object(),
        latent_checkpoint_dir=tmp_path / "latent",
    )

    captured_env: dict[str, object] = {}

    class FakeEnv:
        def __init__(self, **kwargs):
            captured_env.update(kwargs)
            self.model = SimpleNamespace(nu=2)
            self.feed_bank_manifest = runtime_manifest
            self.control_manifest = {
                "control_hash": "control-hash",
                "latent_checkpoint_fingerprint": "f" * 64,
            }

    validated: dict[str, object] = {}

    def fake_validate(*_args, **kwargs):
        validated.update(kwargs)
        return {"verified": True}

    monkeypatch.setattr(runner_module, "_ensure_scene", lambda _paths: None)
    monkeypatch.setattr(runner_module, "_ensure_feed_bank_artifact", lambda _paths: artifact)
    monkeypatch.setattr(
        runner_module,
        "_build_stage3_lab_components",
        lambda *_args, **_kwargs: lab,
    )
    monkeypatch.setattr(
        "environment.overall_environment.src.incoming_shuttle_hit_mjx_env.IncomingHitMjxEnv",
        FakeEnv,
    )
    monkeypatch.setattr(training_module, "validate_stage3_training_prerequisites", fake_validate)
    captured_train: dict[str, object] = {}

    def fake_train(_env, config, *_args, **_kwargs):
        captured_train["config"] = config
        return {"final": {}, "passed": True}

    monkeypatch.setattr(training_module, "train", fake_train)

    runner_module.train_gpu(
        paths,
        out_dir=tmp_path,
        num_envs=1,
        rollout_steps=1,
        total_env_steps=1,
    )

    assert validated["training_feed_manifest"] is runtime_manifest
    assert captured_env["clearance_reward_mode"] == "signed_centered"
    config = captured_train["config"]
    assert config.num_minibatches == 3
    assert config.gamma == 0.97
    assert config.gae_lambda == 0.91
    assert config.clip_coef == 0.15
    assert config.value_coef == 0.4
    assert config.entropy_coef == 0.0001
    assert config.max_grad_norm == 0.3


def _stable_hash(value: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write_valid_stage3_prerequisites(
    root: Path,
    *,
    spec: Path,
    scene: Path,
    latent: Path,
    control: dict[str, object],
    producer_feed: dict[str, object],
    consumer_order: dict[str, object],
) -> None:
    body_names = [f"body-{index}" for index in range(354)]
    right_names: list[str] = []
    left_names: list[str] = []
    all_names = [*body_names, *right_names, *left_names]
    router_identity = {
        "schema_version": "stage3_action_router_v2",
        "fixture_mode": "rigid_tool_fingerless",
        "all": all_names,
        "body": body_names,
        "right_grip": right_names,
        "left_neutral": left_names,
    }
    router = {
        "schema_version": "stage3_action_router_v2",
        "fixture_mode": "rigid_tool_fingerless",
        "all_actuator_names": all_names,
        "body_actuator_names": body_names,
        "right_grip_actuator_names": right_names,
        "left_neutral_actuator_names": left_names,
        "partition_sizes": [354, 0, 0],
        "schema_hash": _stable_hash(router_identity),
    }
    tolerances = {
        "relative_position_error_m": 1.0e-8,
        "relative_rotation_error_rad": 1.0e-6,
        "racket_mass_error_kg": 1.0e-10,
        "racket_center_of_mass_error_m": 1.0e-8,
        "racket_inertia_max_abs_error_kg_m2": 1.0e-10,
        "stringbed_position_error_m": 1.0e-8,
        "stringbed_rotation_error_rad": 1.0e-6,
    }
    checks = {
        "direct_parent": True,
        "jointless_racket_subtree": True,
        "no_racket_equality_constraint": True,
        "relative_position": True,
        "relative_rotation": True,
        "racket_mass": True,
        "racket_center_of_mass": True,
        "racket_inertia": True,
        "stringbed_position": True,
        "stringbed_rotation": True,
        "no_human_racket_contact": True,
        "racket_shuttle_contact_preserved": True,
    }
    attachment = {
        "schema_version": "stage3_exact_child_attachment_v2",
        "attachment_mode": "exact_child",
        "contract_id": "forehand_clear_rigid_v2",
        "contract_fingerprint": "sha256:" + "a" * 64,
        "parent_body_matches": True,
        "racket_joint_count": 0,
        "racket_equality_constraint_count": 0,
        "human_root_body_name": "Full Body",
        "racket_body_name": "overall_racket",
        "relative_position_error_m": 0.0,
        "relative_rotation_error_rad": 0.0,
        "racket_mass_error_kg": 0.0,
        "racket_center_of_mass_error_m": 0.0,
        "racket_inertia_max_abs_error_kg_m2": 0.0,
        "stringbed_position_error_m": 0.0,
        "stringbed_rotation_error_rad": 0.0,
        "human_racket_mask_compatible_geom_pairs": 0,
        "human_racket_explicit_contact_pairs": 0,
        "hand_racket_contact_enabled": False,
        "racket_shuttle_contact_enabled": True,
        "contract_tolerances": tolerances,
        "contract_checks": checks,
        "contract_passed": True,
    }
    attachment["attachment_hash"] = _stable_hash(attachment)
    (root / "preflight_report.json").write_text(
        json.dumps(
            {
                "passed": True,
                "runner_type": "incoming_shuttle_hit",
                "spec_path": str(spec),
                "scene_xml": str(scene),
                "scene_exists": True,
                "keyframe_found": True,
                "attachment_contract_passed": True,
                "configuration_contract_passed": True,
                "missing_sites": [],
                "actuator_count": 354,
                "finger_joint_count": 0,
                "finger_actuator_count": 0,
                "finger_joint_names": [],
                "finger_actuator_names": [],
                "root_pos": [-3.35, 0.0, 1.0],
                "expected_root_xy": [-3.35, 0.0],
                "action_router": router,
                "racket_attachment": attachment,
            }
        ),
        encoding="utf-8",
    )
    episode = {
        "completed_steps": 120,
        "finite": True,
        "body_fall": False,
        "min_root_height_m": 0.9,
        "max_body_action_saturation_fraction": 0.0,
        "max_full_action_saturation_fraction": 0.0,
        "max_normalized_control_energy": 0.1,
        "max_lab_state_ood_fraction": 0.0,
        "min_control_finite": 1.0,
        "max_attachment_translation_drift_m": 0.001,
        "max_attachment_rotation_drift_rad": 0.01,
    }
    thresholds = {
        "min_rollout_count": 1.0,
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
    gates = {
        "rollout_count": True,
        "completion_rate": True,
        "finite_rate": True,
        "no_fall_rate": True,
        "min_root_height_m": True,
        "body_action_saturation": True,
        "full_action_saturation": True,
        "normalized_control_energy": True,
        "lab_state_ood_fraction": True,
        "control_finite": True,
        "attachment_translation_drift": True,
        "attachment_rotation_drift": True,
    }
    (root / "base_only_report.json").write_text(
        json.dumps(
            {
                "schema_version": "stage3_base_only_v1",
                "runner_stage": "base-only-check",
                "passed": True,
                "latent_checkpoint": str(latent),
                "lambda_lab": 0.0,
                "task_action": "all_zero_raw_latent",
                "shuttle_mode": "parked_out_of_scene",
                "episodes": [episode],
                "rollout_count": 1.0,
                "completion_rate": 1.0,
                "finite_rate": 1.0,
                "no_fall_rate": 1.0,
                "min_root_height_m": 0.9,
                "max_body_action_saturation_fraction": 0.0,
                "max_full_action_saturation_fraction": 0.0,
                "max_normalized_control_energy": 0.1,
                "max_lab_state_ood_fraction": 0.0,
                "min_control_finite": 1.0,
                "max_attachment_translation_drift_m": 0.001,
                "max_attachment_rotation_drift_rad": 0.01,
                "required_steps": 120,
                "thresholds": thresholds,
                "gates": gates,
                "control_manifest": control,
            }
        ),
        encoding="utf-8",
    )
    train_fingerprints = list(producer_feed["sample_fingerprints"])
    eval_manifest = {
        "schema_version": "incoming_shuttle_feed_bank_manifest_v1",
        "sample_fingerprints": ["eval-a"],
    }
    feed_entries = {
        "train": {
            "bank_size": len(train_fingerprints),
            "expected_bank_size": len(train_fingerprints),
            "exact_count": True,
            "unique_sample_count": len(set(train_fingerprints)),
            "all_samples_unique": True,
            "all_in_window": True,
            "quality_passed": True,
            "quality": {
                "schema_version": "incoming_shuttle_feed_quality_v2",
                "passed": True,
            },
            "manifest": producer_feed,
            "consumer_order": {
                **consumer_order,
                "seed_feed_fingerprints": [],
                "seed_producer_indices": [],
                "seed_prefix_matches": True,
                "first_fingerprint": consumer_order["sample_fingerprints"][0],
                "content_sha256": _stable_hash(
                    {"sample_fingerprints": consumer_order["sample_fingerprints"]}
                ),
                "passed": True,
            },
        },
        "eval": {
            "bank_size": 1,
            "expected_bank_size": 1,
            "exact_count": True,
            "unique_sample_count": 1,
            "all_samples_unique": True,
            "all_in_window": True,
            "quality_passed": True,
            "quality": {
                "schema_version": "incoming_shuttle_feed_quality_v2",
                "passed": True,
            },
            "manifest": eval_manifest,
            "consumer_order": None,
        },
    }
    (root / "feed_check_report.json").write_text(
        json.dumps(
            {
                "runner_stage": "feed-check",
                "passed": True,
                **feed_entries,
                "bank_paths_distinct": True,
                "train_unique_sample_count": len(set(train_fingerprints)),
                "eval_unique_sample_count": 1,
                "train_duplicate_count": 0,
                "eval_duplicate_count": 0,
                "train_eval_fingerprint_overlap_count": 0,
                "train_eval_fingerprint_overlap": [],
            }
        ),
        encoding="utf-8",
    )


def test_stage3_training_prerequisites_bind_scene_latent_control_and_feed(
    tmp_path: Path,
) -> None:
    spec = tmp_path / "spec.yaml"
    scene = tmp_path / "scene.xml"
    spec.write_text("runner_type: incoming_shuttle_hit\n", encoding="utf-8")
    scene.write_text("<mujoco/>\n", encoding="utf-8")
    latent = tmp_path / "latent"
    control = {
        "control_hash": "control-hash",
        "latent_checkpoint_fingerprint": "f" * 64,
    }
    producer_feed = {
        "schema_version": "incoming_shuttle_feed_bank_manifest_v1",
        "content_sha256": "a" * 64,
        "sample_fingerprints": ["train-a"],
    }
    runtime_feed = {
        **producer_feed,
        "consumer_order": {
            "schema_version": "incoming_hit_curriculum_feed_order_v1",
            "mode": "difficulty_sorted",
            "sample_fingerprints": ["train-a"],
        },
    }
    _write_valid_stage3_prerequisites(
        tmp_path,
        spec=spec,
        scene=scene,
        latent=latent,
        control=control,
        producer_feed=producer_feed,
        consumer_order=runtime_feed["consumer_order"],
    )
    base_path = tmp_path / "base_only_report.json"
    paths = SimpleNamespace(
        spec_path=spec,
        scene_xml=scene,
        human_root_xy=(-3.35, 0.0),
        feed_bank_size=1,
        eval_feed_bank_size=1,
        feed_bank_path=tmp_path / "train-feed.npz",
        eval_feed_bank_path=tmp_path / "eval-feed.npz",
    )

    binding = validate_stage3_training_prerequisites(
        tmp_path,
        paths=paths,
        latent_checkpoint_dir=latent,
        control_manifest=control,
        training_feed_manifest=runtime_feed,
    )
    assert binding["verified"] is True
    assert len(binding["binding_sha256"]) == 64
    assert binding["training_feed_manifest_sha256"] == _stable_hash(runtime_feed)
    assert binding["training_feed_producer_manifest_sha256"] == _stable_hash(producer_feed)

    checkpoint = tmp_path / "policy_latest.npz"
    np.savez(checkpoint, placeholder=np.asarray([1.0]))
    metadata = {
        "control_hash": control["control_hash"],
        "config": {"seed": 0},
        "training_feed_manifest": runtime_feed,
        "iteration": 10,
        "env_steps": 20_000_000,
        "curriculum_complete": True,
        "promotion_eligible": True,
        "training_prerequisite_binding": binding,
        "curriculum_state": {
            "effective_steps": 20_000_000,
            "phase": "lambda_full",
        },
    }
    checkpoint.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    (tmp_path / "train_report.json").write_text(
        json.dumps(
            {
                "iterations": 10,
                "env_steps": 20_000_000,
                "curriculum_complete": True,
                "promotion_eligible": True,
                "curriculum_effective_steps": 20_000_000,
                "curriculum_phase": "lambda_full",
                "checkpoint": str(checkpoint),
                "training_prerequisite_binding": binding,
            }
        ),
        encoding="utf-8",
    )
    evaluation_feed = {
        "schema_version": "incoming_shuttle_feed_bank_manifest_v1",
        "sample_fingerprints": ["eval-a"],
    }
    evaluation_report = {
        "checkpoint": str(checkpoint),
        "control_manifest": control,
        "training_feed_manifest": runtime_feed,
        "evaluation_feed_manifest": evaluation_feed,
        "episodes": [{"hit": True}],
        "hit_rate": 1.0,
    }
    evaluation_binding = _build_stage3_artifact_binding(
        paths=paths,
        checkpoint_path=checkpoint,
        checkpoint_metadata=metadata,
        control_manifest=control,
        training_feed_manifest=runtime_feed,
        evaluation_feed_manifest=evaluation_feed,
        evaluation_content_sha256=_stage3_evaluation_content_sha256(evaluation_report),
    )
    evaluation_report["artifact_binding"] = evaluation_binding
    evaluation_report_path = tmp_path / "evaluate_report.json"
    evaluation_report_path.write_text(json.dumps(evaluation_report), encoding="utf-8")
    _require_stage3_artifact_binding(evaluation_report_path)

    tampered = json.loads(base_path.read_text(encoding="utf-8"))
    tampered["control_manifest"]["control_hash"] = "other"
    base_path.write_text(json.dumps(tampered), encoding="utf-8")
    with np.testing.assert_raises_regex(ValueError, "control contract changed"):
        validate_stage3_training_prerequisites(
            tmp_path,
            paths=paths,
            latent_checkpoint_dir=latent,
            control_manifest=control,
            training_feed_manifest=runtime_feed,
        )


def test_stage3_frozen_base_residual_prerequisites_bind_base_and_qc(
    tmp_path: Path,
) -> None:
    spec = tmp_path / "residual.yaml"
    scene = tmp_path / "scene.xml"
    base_artifact = tmp_path / "frozen-base"
    spec.write_text("runner_type: incoming_shuttle_hit\n", encoding="utf-8")
    scene.write_text("<mujoco/>\n", encoding="utf-8")
    base_artifact.mkdir()
    frozen_binding = {
        "schema_version": "incoming_hit_frozen_base_residual_v1",
        "artifact_content_sha256": "a" * 64,
        "binding_sha256": "b" * 64,
        "residual_scale": 0.25,
        "actor_action_size": 354,
    }
    control = {
        "schema_version": "incoming_hit_direct_action_v1",
        "control_hash": "residual-control-hash",
        "frozen_base_residual": frozen_binding,
        "environment_abi": {"full_action_size": 354},
        "curriculum": {"fixed_feed_steps": 500_000},
        "curriculum_feed_order": "stored",
        "seed_feed_fingerprints": ["train-a"],
    }
    producer_feed = {
        "schema_version": "incoming_shuttle_feed_bank_manifest_v1",
        "content_sha256": "c" * 64,
        "sample_fingerprints": ["train-a"],
    }
    runtime_feed = {
        **producer_feed,
        "consumer_order": {
            "schema_version": "incoming_hit_curriculum_feed_order_v1",
            "mode": "stored",
            "sample_fingerprints": ["train-a"],
        },
    }
    _write_valid_stage3_prerequisites(
        tmp_path,
        spec=spec,
        scene=scene,
        latent=tmp_path / "unused-latent",
        control=control,
        producer_feed=producer_feed,
        consumer_order=runtime_feed["consumer_order"],
    )
    base_report_path = tmp_path / "base_only_report.json"
    base_report = json.loads(base_report_path.read_text(encoding="utf-8"))
    base_control = dict(control)
    # A parked single-feed base-only rollout cannot consume production seed
    # fingerprints; the training feed manifest binds them separately.
    base_control.pop("seed_feed_fingerprints")
    base_control.pop("control_hash")
    base_control["control_hash"] = _stable_hash(base_control)
    base_report.update(
        {
            "control_mode": "frozen_base_residual",
            "latent_checkpoint": None,
            "base_policy_artifact": str(base_artifact),
            "lambda_lab": None,
            "residual_scale": 0.25,
            "task_action": "all_zero_full_action_residual",
            "control_manifest": base_control,
        }
    )
    base_report["episodes"][0].pop("max_lab_state_ood_fraction")
    base_report.pop("max_lab_state_ood_fraction")
    base_report["gates"].pop("lab_state_ood_fraction")
    base_report_path.write_text(json.dumps(base_report), encoding="utf-8")
    paths = SimpleNamespace(
        spec_path=spec,
        scene_xml=scene,
        human_root_xy=(-3.35, 0.0),
        feed_bank_size=1,
        eval_feed_bank_size=1,
        feed_bank_path=tmp_path / "train-feed.npz",
        eval_feed_bank_path=tmp_path / "eval-feed.npz",
    )

    binding = validate_stage3_residual_training_prerequisites(
        tmp_path,
        paths=paths,
        base_policy_artifact=base_artifact,
        control_manifest=control,
        training_feed_manifest=runtime_feed,
    )

    assert binding["verified"] is True
    assert binding["action_family"] == "frozen_base_residual"
    assert binding["base_policy_artifact_content_sha256"] == "a" * 64
    assert runner_module._stage3_action_family(control) == "frozen_base_residual"

    legacy_base_control = dict(base_control)
    legacy_base_control.pop("curriculum_feed_order")
    legacy_base_control.pop("control_hash")
    legacy_base_control["control_hash"] = _stable_hash(legacy_base_control)
    base_report["control_manifest"] = legacy_base_control
    base_report_path.write_text(json.dumps(base_report), encoding="utf-8")
    legacy_binding = validate_stage3_residual_training_prerequisites(
        tmp_path,
        paths=paths,
        base_policy_artifact=base_artifact,
        control_manifest=control,
        training_feed_manifest=runtime_feed,
    )
    assert legacy_binding["verified"] is True

    base_report["residual_scale"] = 0.3
    base_report_path.write_text(json.dumps(base_report), encoding="utf-8")
    with np.testing.assert_raises_regex(ValueError, "residual scale changed"):
        validate_stage3_residual_training_prerequisites(
            tmp_path,
            paths=paths,
            base_policy_artifact=base_artifact,
            control_manifest=control,
            training_feed_manifest=runtime_feed,
        )


def test_full354_checkpoint_metadata_evaluate_binding_and_promotion_contract(
    tmp_path: Path,
) -> None:
    spec = tmp_path / "full354.yaml"
    scene = tmp_path / "scene.xml"
    train_target = tmp_path / "targets_train.json"
    eval_target = tmp_path / "targets_eval.json"
    spec.write_text("runner_type: incoming_shuttle_hit\n", encoding="utf-8")
    scene.write_text("<mujoco/>\n", encoding="utf-8")
    train_target.write_text(
        json.dumps({"bank_sha256": "1" * 64, "source_fingerprint": "2" * 64}),
        encoding="utf-8",
    )
    eval_target.write_text(
        json.dumps({"bank_sha256": "3" * 64, "source_fingerprint": "4" * 64}),
        encoding="utf-8",
    )
    control = {
        "schema_version": "incoming_hit_direct_action_impact_recovery_v2",
        "control_hash": "control-full354",
        "policy_abi_hash": "policy-abi-full354",
        "latent_checkpoint_fingerprint": None,
        "environment_abi": {
            "task_profile": "impact_recovery_v2",
            "full_action_size": 354,
            "scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
            "target_bank_sha256": "1" * 64,
        },
    }
    producer_feed = {
        "schema_version": "incoming_shuttle_feed_bank_manifest_v1",
        "content_sha256": "a" * 64,
        "sample_fingerprints": ["train-a"],
    }
    runtime_feed = {
        **producer_feed,
        "consumer_order": {
            "schema_version": "incoming_hit_curriculum_feed_order_v1",
            "mode": "stored",
            "sample_fingerprints": ["train-a"],
        },
    }
    _write_valid_stage3_prerequisites(
        tmp_path,
        spec=spec,
        scene=scene,
        latent=tmp_path / "unused-latent-ablation",
        control=control,
        producer_feed=producer_feed,
        consumer_order=runtime_feed["consumer_order"],
    )
    paths = SimpleNamespace(
        spec_path=spec,
        scene_xml=scene,
        human_root_xy=(-3.35, 0.0),
        feed_bank_size=1,
        eval_feed_bank_size=1,
        feed_bank_path=tmp_path / "train-feed.npz",
        eval_feed_bank_path=tmp_path / "eval-feed.npz",
        target_bank_path=train_target,
        eval_target_bank_path=eval_target,
        task_profile="impact_recovery_v2",
    )
    prerequisite = validate_stage3_direct_training_prerequisites(
        tmp_path,
        paths=paths,
        control_manifest=control,
        training_feed_manifest=runtime_feed,
    )
    assert prerequisite["action_family"] == "full_354"
    assert prerequisite["policy_action_size"] == 354
    assert prerequisite["latent_checkpoint_fingerprint"] is None
    assert "base_only_report_path" not in prerequisite

    checkpoint = tmp_path / "policy_latest.npz"
    np.savez(checkpoint, placeholder=np.asarray([1.0]))
    metadata = {
        "control_hash": control["control_hash"],
        "control_manifest": control,
        "config": {"seed": 0},
        "training_feed_manifest": runtime_feed,
        "iteration": 10,
        "env_steps": 30_000_000,
        "curriculum_complete": True,
        "promotion_eligible": True,
        "training_prerequisite_binding": prerequisite,
        "curriculum_state": {"effective_steps": 0, "phase": "disabled"},
        "task_curriculum_state": {
            "max_stage": "C7_recovery",
            "stage": "C7_recovery",
            "complete": True,
        },
    }
    checkpoint.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    (tmp_path / "train_report.json").write_text(
        json.dumps(
            {
                "iterations": 10,
                "env_steps": 30_000_000,
                "curriculum_effective_steps": 0,
                "curriculum_phase": "disabled",
                "curriculum_complete": True,
                "promotion_eligible": True,
                "checkpoint": str(checkpoint),
                "training_prerequisite_binding": prerequisite,
            }
        ),
        encoding="utf-8",
    )
    evaluation_feed = {
        "schema_version": "incoming_shuttle_feed_bank_manifest_v1",
        "sample_fingerprints": ["eval-a"],
    }
    episode = {
        "episode": 0,
        "return": 10.0,
        "hit": True,
        "crossed_net": True,
        "body_fall": False,
        "landing_region": "opponent_back",
        "contact_racket_head_speed_m_s": 10.0,
        "net_clearance_m": 0.5,
        "min_root_height_m": 0.9,
        "max_attachment_translation_drift_m": 0.0,
        "max_attachment_rotation_drift_rad": 0.0,
        "recovery_complete": True,
        "lab_diagnostics": {
            "control_finite": 1.0,
            "normalized_control_energy": 0.2,
            "body_action_saturation_fraction": 0.0,
            "full_action_saturation_fraction": 0.0,
        },
        "stage3_v2_metrics": {
            "impact_position_error_m": 0.05,
            "impact_rho2": 0.1,
            "impact_timing_error_s": 0.03,
            "stringbed_normal_error_rad": 0.1,
            "racket_linear_velocity_error_m_s": 0.5,
            "racket_angular_velocity_error_rad_s": 2.0,
            "landing_error_m": 0.3,
            "apex_error_m": 0.1,
            "ready_pose_error": 0.05,
        },
    }
    summary = _stage3_evaluation_summary(
        [episode],
        gate_config={},
        required_feed_count=1,
        task_profile="impact_recovery_v2",
        action_family="full_354",
    )
    assert summary["passed"] is True
    assert summary["lab_metrics_applicable"] is False
    assert summary["raw_latent_saturation"] is None
    assert "raw_latent_saturation" not in summary["promotion_gates"]
    assert "lab_state_ood_fraction_p95" not in summary["promotion_gates"]
    assert summary["promotion_gates"]["normalized_control_energy"] is True

    report = {
        "schema_version": "incoming_shuttle_hit_evaluate_v3",
        "runner_stage": "evaluate",
        "checkpoint": str(checkpoint),
        "evaluation_seed": 123,
        "episodes": [episode],
        "mean_return": 10.0,
        **summary,
        "control_manifest": control,
        "training_feed_manifest": runtime_feed,
        "evaluation_feed_manifest": evaluation_feed,
    }
    binding = _build_stage3_artifact_binding(
        paths=paths,
        checkpoint_path=checkpoint,
        checkpoint_metadata=metadata,
        control_manifest=control,
        training_feed_manifest=runtime_feed,
        evaluation_feed_manifest=evaluation_feed,
        evaluation_seed=123,
        evaluation_content_sha256=_stage3_evaluation_content_sha256(report),
    )
    assert binding["action_family"] == "full_354"
    assert binding["latent_checkpoint_fingerprint"] is None
    report["artifact_binding"] = binding
    report_path = tmp_path / "evaluate_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    _require_stage3_artifact_binding(report_path)


def test_stage3_training_prerequisites_reject_failed_internal_predicates(
    tmp_path: Path,
) -> None:
    spec = tmp_path / "spec.yaml"
    scene = tmp_path / "scene.xml"
    spec.write_text("runner_type: incoming_shuttle_hit\n", encoding="utf-8")
    scene.write_text("<mujoco/>\n", encoding="utf-8")
    latent = tmp_path / "latent"
    control = {
        "control_hash": "control-hash",
        "latent_checkpoint_fingerprint": "f" * 64,
    }
    producer_feed = {
        "schema_version": "incoming_shuttle_feed_bank_manifest_v1",
        "sample_fingerprints": ["train-a"],
    }
    runtime_feed = {
        **producer_feed,
        "consumer_order": {
            "schema_version": "incoming_hit_curriculum_feed_order_v1",
            "mode": "difficulty_sorted",
            "sample_fingerprints": ["train-a"],
        },
    }
    paths = SimpleNamespace(
        spec_path=spec,
        scene_xml=scene,
        human_root_xy=(-3.35, 0.0),
        feed_bank_size=1,
        eval_feed_bank_size=1,
        feed_bank_path=tmp_path / "train-feed.npz",
        eval_feed_bank_path=tmp_path / "eval-feed.npz",
    )

    cases = (
        (
            "preflight_report.json",
            ("attachment_contract_passed",),
            False,
            "preflight",
        ),
        ("base_only_report.json", ("gates", "no_fall_rate"), False, "base-only"),
        (
            "feed_check_report.json",
            ("eval", "all_samples_unique"),
            False,
            "feed-check",
        ),
        (
            "feed_check_report.json",
            ("train", "consumer_order", "first_fingerprint"),
            "wrong-feed",
            "consumer-order",
        ),
    )
    for filename, key_path, replacement, expected_error in cases:
        _write_valid_stage3_prerequisites(
            tmp_path,
            spec=spec,
            scene=scene,
            latent=latent,
            control=control,
            producer_feed=producer_feed,
            consumer_order=runtime_feed["consumer_order"],
        )
        report_path = tmp_path / filename
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        target = payload
        for key in key_path[:-1]:
            target = target[key]
        target[key_path[-1]] = replacement
        payload["passed"] = True
        report_path.write_text(json.dumps(payload), encoding="utf-8")
        with np.testing.assert_raises_regex(ValueError, expected_error):
            validate_stage3_training_prerequisites(
                tmp_path,
                paths=paths,
                latent_checkpoint_dir=latent,
                control_manifest=control,
                training_feed_manifest=runtime_feed,
            )


def test_explicit_minibatch_size_must_divide_parallel_rollout() -> None:
    class Env:
        expects_raw_latent = True

        @staticmethod
        def make_step_fn(_mx, _num_envs):
            return lambda state, action: (state, action)

    config = TrainConfig(num_envs=3, rollout_steps=2, minibatch_size=4)
    with np.testing.assert_raises_regex(ValueError, "minibatch_size"):
        make_train_iteration(Env(), None, config, optax.adam(3e-4))

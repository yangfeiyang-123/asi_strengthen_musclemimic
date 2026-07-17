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
    _full_batch_budget,
    compute_rollout_gae,
    init_agent,
    load_training_checkpoint,
    load_training_checkpoint_metadata,
    make_train_iteration,
    reconcile_metrics_history,
    resolve_training_checkpoint,
    sample_action,
    save_training_checkpoint,
    save_versioned_training_checkpoint,
    validate_stage3_direct_training_prerequisites,
    validate_stage3_training_prerequisites,
    validate_training_feed_manifest,
)


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


def test_versioned_checkpoint_commits_directory_then_moves_latest_pointer(
    tmp_path: Path,
) -> None:
    agent = init_agent(
        jax.random.PRNGKey(1), 3, 2, hidden=(4,), action_std_init=0.2
    )
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
    agent = init_agent(
        jax.random.PRNGKey(1), 3, 2, hidden=(4,), action_std_init=0.2
    )
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
    agent = init_agent(
        jax.random.PRNGKey(1), 3, 2, hidden=(4,), action_std_init=0.2
    )
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
    assert load_training_checkpoint_metadata(tmp_path / "policy_latest.json")[
        "env_steps"
    ] == 8

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


def test_production_main_passes_verified_feed_manifest_to_environment(
    tmp_path: Path, monkeypatch
) -> None:
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
        ppo_overrides={},
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

    monkeypatch.setattr(
        training_module, "validate_stage3_training_prerequisites", fake_validate
    )
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


def test_unified_train_gpu_binds_runtime_feed_manifest(
    tmp_path: Path, monkeypatch
) -> None:
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
        ppo_overrides={},
        output_dir=tmp_path,
    )
    lab = SimpleNamespace(
        controller=object(),
        state_builder=object(),
        curriculum=object(),
        latent_checkpoint_dir=tmp_path / "latent",
    )

    class FakeEnv:
        def __init__(self, **_kwargs):
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
    monkeypatch.setattr(
        runner_module, "_ensure_feed_bank_artifact", lambda _paths: artifact
    )
    monkeypatch.setattr(
        runner_module,
        "_build_stage3_lab_components",
        lambda *_args, **_kwargs: lab,
    )
    monkeypatch.setattr(
        "environment.overall_environment.src.incoming_shuttle_hit_mjx_env.IncomingHitMjxEnv",
        FakeEnv,
    )
    monkeypatch.setattr(
        training_module, "validate_stage3_training_prerequisites", fake_validate
    )
    monkeypatch.setattr(
        training_module,
        "train",
        lambda *_args, **_kwargs: {"final": {}, "passed": True},
    )

    runner_module.train_gpu(
        paths,
        out_dir=tmp_path,
        num_envs=1,
        rollout_steps=1,
        total_env_steps=1,
    )

    assert validated["training_feed_manifest"] is runtime_manifest


def _stable_hash(value: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_valid_stage3_prerequisites(
    root: Path,
    *,
    spec: Path,
    scene: Path,
    latent: Path,
    control: dict[str, object],
    producer_feed: dict[str, object],
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
            "manifest": producer_feed,
        },
        "eval": {
            "bank_size": 1,
            "expected_bank_size": 1,
            "exact_count": True,
            "unique_sample_count": 1,
            "all_samples_unique": True,
            "all_in_window": True,
            "manifest": eval_manifest,
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
    assert binding["training_feed_producer_manifest_sha256"] == _stable_hash(
        producer_feed
    )

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
    checkpoint.with_suffix(".json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
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
        evaluation_content_sha256=_stage3_evaluation_content_sha256(
            evaluation_report
        ),
    )
    evaluation_report["artifact_binding"] = evaluation_binding
    evaluation_report_path = tmp_path / "evaluate_report.json"
    evaluation_report_path.write_text(
        json.dumps(evaluation_report), encoding="utf-8"
    )
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
    )
    for filename, key_path, replacement, expected_error in cases:
        _write_valid_stage3_prerequisites(
            tmp_path,
            spec=spec,
            scene=scene,
            latent=latent,
            control=control,
            producer_feed=producer_feed,
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

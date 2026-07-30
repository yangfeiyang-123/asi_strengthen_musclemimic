"""Raw rollout capture and identity gates for continuity calibration."""

from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from analysis.physiology_synergy.collect_continuity_baseline import (
    _TRAJECTORY_ARRAY_FIELDS,
    CONTINUITY_BASELINE_ENVIRONMENT_SCHEMA_VERSION,
    CONTINUITY_BASELINE_HELDOUT_SCHEMA_VERSION,
    _trajectory_manifest_digest,
    build_rollout_identity,
    infer_action_mode,
    validate_rollout_manifest,
)
from musclemimic.runner.eval_utils import (
    CONTINUITY_BASELINE_RAW_KEYS,
    _append_cpu_continuity_sample,
    _append_mjx_continuity_samples,
)


def _scan_out():
    values = jnp.arange(9, dtype=jnp.float32).reshape(3, 3)
    return {
        "valid_mask": jnp.asarray(
            [
                [True, True, True],
                [True, True, True],
                [True, False, True],
            ]
        ),
        "info": {
            "reward_imitation_total": values + 1.0,
            "fascicle_continuity_loss": values / 100.0,
            "fascicle_continuity_measured_chain_count": jnp.full((3, 3), 28.0),
            "fascicle_continuity_measured_edge_count": jnp.full((3, 3), 140.0),
        },
    }


def _json_sha(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _environment_manifest() -> dict:
    payload = {
        "schema_version": CONTINUITY_BASELINE_ENVIRONMENT_SCHEMA_VERSION,
        "resolved_env_params": {"disable_fingers": True},
        "resolved_task_factory": {"name": "fixture"},
        "resolved_action_representation": {"enabled": False},
        "runtime_model_hash": "1" * 64,
        "muscle_channel_core_fingerprint": "2" * 64,
        "ordered_actuator_schema_hash": "3" * 64,
        "ordered_action_dim": 354,
        "control_dt": 0.01,
        "horizon": 32,
    }
    payload["environment_fingerprint"] = _json_sha(payload)
    return payload


def _heldout_manifest() -> dict:
    arrays = {
        field: {
            "dtype": "<f4",
            "shape": [1],
            "sha256": hashlib.sha256(field.encode("utf-8")).hexdigest(),
        }
        for field in _TRAJECTORY_ARRAY_FIELDS
    }
    payload = {
        "schema_version": CONTINUITY_BASELINE_HELDOUT_SCHEMA_VERSION,
        "ordered_motion_paths": ["heldout/motion"],
        "trajectory_count": 1,
        "trajectory_lengths": [32],
        "frequency_hz": 100.0,
        "array_manifest": arrays,
        "trajectory_content_sha256": _trajectory_manifest_digest(arrays),
    }
    payload["heldout_split_fingerprint"] = _json_sha(payload)
    return payload


def test_mjx_raw_capture_excludes_padding_lane_and_post_completion_steps():
    sink = {}
    _append_mjx_continuity_samples(sink, _scan_out(), (0, 1))

    assert sink["trajectory_index"] == [0, 0, 0, 1, 1]
    assert sink["trajectory_step"] == [0, 1, 2, 0, 1]
    assert sink["reward_imitation_total"] == pytest.approx([1.0, 4.0, 7.0, 2.0, 5.0])
    assert all(len(sink[key]) == 5 for key in CONTINUITY_BASELINE_RAW_KEYS)


def test_raw_capture_fails_closed_when_runtime_metric_is_missing_or_vector_valued():
    missing = _scan_out()
    del missing["info"]["reward_imitation_total"]
    with pytest.raises(ValueError, match="missing raw metrics"):
        _append_mjx_continuity_samples({}, missing, (0, 1))

    vector = _scan_out()
    vector["info"]["reward_imitation_total"] = jnp.ones((3, 3, 2))
    with pytest.raises(ValueError, match="scalar per environment step"):
        _append_mjx_continuity_samples({}, vector, (0, 1))


def test_cpu_raw_capture_preserves_explicit_trajectory_coordinates():
    sink = {}
    info = {
        "reward_imitation_total": 0.8,
        "fascicle_continuity_loss": 0.12,
        "fascicle_continuity_measured_chain_count": 28,
        "fascicle_continuity_measured_edge_count": 140,
    }
    _append_cpu_continuity_sample(
        sink,
        info,
        trajectory_index=3,
        trajectory_step=7,
    )

    assert sink["trajectory_index"] == [3]
    assert sink["trajectory_step"] == [7]
    assert sink["fascicle_continuity_loss"] == pytest.approx([0.12])


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ({"enabled": False}, "full_354"),
        (
            {"enabled": True, "mode": "fixed_synergy", "residual": {"enabled": False}},
            "fixed_synergy",
        ),
        (
            {
                "enabled": True,
                "mode": "fixed_synergy_residual",
                "residual": {"enabled": True, "alpha": 0.1},
            },
            "fixed_synergy_residual",
        ),
    ],
)
def test_rollout_action_mode_is_derived_from_the_resolved_policy_contract(action, expected):
    config = {"experiment": {"action_representation": action}}
    assert infer_action_mode(config) == expected


def test_rollout_action_mode_rejects_inconsistent_residual_declaration():
    config = {
        "experiment": {
            "action_representation": {
                "enabled": True,
                "mode": "fixed_synergy",
                "residual": {"enabled": True, "alpha": 0.1},
            }
        }
    }
    with pytest.raises(ValueError, match=r"mode and residual.enabled disagree"):
        infer_action_mode(config)


def test_rollout_identity_binds_checkpoint_promotion_environment_and_split():
    checkpoint = {
        "run_id": "baseline-run",
        "config_hash": "config-hash",
        "checkpoint_content_sha256": "a" * 64,
    }
    promoted = {
        "binding_sha256": "b" * 64,
        "checkpoint": {"checkpoint_content_sha256": "a" * 64},
    }
    taxonomy = SimpleNamespace(fingerprint="c" * 64)
    graph = SimpleNamespace(
        graph_fingerprint="d" * 64,
        chains=tuple({"training_enabled": False} for _ in range(28)),
        edge_count=140,
    )
    config = {
        "experiment": {
            "run_id": "baseline-run",
            "action_representation": {"enabled": False},
        }
    }
    environment = _environment_manifest()
    heldout = _heldout_manifest()

    identity, manifest = build_rollout_identity(
        config=config,
        checkpoint_identity=checkpoint,
        promoted_artifact=promoted,
        taxonomy=taxonomy,
        graph=graph,
        environment_manifest=environment,
        heldout_split_manifest=heldout,
        backend="mjx",
        deterministic=True,
        eval_seed=11,
        num_envs=8,
    )

    assert identity["checkpoint_fingerprint"] == "a" * 64
    assert identity["promotion_fingerprint"] == "b" * 64
    assert identity["environment_fingerprint"] == environment["environment_fingerprint"]
    assert identity["heldout_split_fingerprint"] == heldout["heldout_split_fingerprint"]
    assert identity["rollout_manifest_fingerprint"] == manifest["rollout_manifest_fingerprint"]
    assert manifest["protocol"]["padding_steps_included"] is False

    tampered = {**manifest, "semantics": "aggregate_only"}
    with pytest.raises(ValueError, match="semantics differ"):
        validate_rollout_manifest(tampered)

    tampered = copy.deepcopy(manifest)
    tampered["policy"]["action_mode"] = "fixed_synergy"
    tampered["rollout_manifest_fingerprint"] = _json_sha(
        {key: value for key, value in tampered.items() if key != "rollout_manifest_fingerprint"}
    )
    with pytest.raises(ValueError, match="action mode differs"):
        validate_rollout_manifest(tampered)

    tampered = copy.deepcopy(manifest)
    tampered["protocol"]["primary_reward_sample"] = "clipped_final_reward"
    tampered["rollout_manifest_fingerprint"] = _json_sha(
        {key: value for key, value in tampered.items() if key != "rollout_manifest_fingerprint"}
    )
    with pytest.raises(ValueError, match="reward sample semantics differ"):
        validate_rollout_manifest(tampered)

    tampered = copy.deepcopy(manifest)
    tampered["environment_manifest"]["horizon"] = 31
    with pytest.raises(ValueError, match="environment fingerprint is stale"):
        validate_rollout_manifest(tampered)

    promoted["checkpoint"]["checkpoint_content_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="does not bind"):
        build_rollout_identity(
            config=config,
            checkpoint_identity=checkpoint,
            promoted_artifact=promoted,
            taxonomy=taxonomy,
            graph=graph,
            environment_manifest=environment,
            heldout_split_manifest=heldout,
            backend="mjx",
            deterministic=True,
            eval_seed=11,
            num_envs=8,
        )

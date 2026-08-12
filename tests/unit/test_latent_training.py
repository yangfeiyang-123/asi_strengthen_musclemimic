from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from musclemimic.distill.action_schema import actuator_schema_hash, ordered_schema_hash
from musclemimic.distill.dataset import write_split_shard
from musclemimic.distill.physical import (
    MUSCLE_CHANNEL_CONTRACT_SCHEMA_VERSION,
    PHYSICAL_CAPTURE_SCHEMA_VERSION,
    physical_signal_metadata,
)
from musclemimic.latent_muscle.action_mask import ActionMask


def _require_latent_deps():
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    pytest.importorskip("optax")


def _latent_action_schema(names):
    from musclemimic.latent_muscle.checkpoint import (
        build_latent_muscle_action_schema,
    )

    ordered = [str(name) for name in names]
    return build_latent_muscle_action_schema(
        ordered,
        muscle_channel_contract={
            "schema_version": MUSCLE_CHANNEL_CONTRACT_SCHEMA_VERSION,
            "actuator_names": ordered,
            "actuator_ids": list(range(len(ordered))),
            "actuator_dyntype": ["muscle"] * len(ordered),
            "actuator_actnum": [1] * len(ordered),
            "actuator_actadr": list(range(len(ordered))),
            "model_na": len(ordered),
        },
    )


def test_direct_latent_cli_clears_all_synergy_yaml_bindings():
    from fullbody.latent_train import _apply_latent_cli_overrides, build_parser

    args = build_parser().parse_args(["--decoder_type", "direct"])
    payload = {
        "decoder_type": "synergy_residual",
        "frozen_body_decoder_path": "/stage1/frozen",
        "frozen_body_decoder_expected_fingerprint": "a" * 64,
        "body_synergy_contract_expected_fingerprint": "b" * 64,
        "body_synergy_portable_core_expected_fingerprint": "c" * 64,
        "legacy_synergy_decoder_ablation": True,
        "synergy_basis_path": "/legacy/W",
        "synergy_basis_expected_fingerprint": "d" * 64,
        "synergy_include_baseline": True,
        "synergy_residual_actuator_names": ["finger"],
        "synergy_residual_alpha": 0.25,
        "synergy_residual_l1_weight": 1.0,
        "synergy_residual_l2_weight": 1.0,
        "synergy_residual_smooth_weight": 1.0,
        "synergy_baseline_l1_weight": 1.0,
        "synergy_baseline_l2_weight": 1.0,
    }

    _apply_latent_cli_overrides(payload, args, phase_balance_weights=None)

    assert payload["decoder_type"] == "direct"
    for field in (
        "frozen_body_decoder_path",
        "frozen_body_decoder_expected_fingerprint",
        "body_synergy_contract_expected_fingerprint",
        "body_synergy_portable_core_expected_fingerprint",
        "synergy_basis_path",
        "synergy_basis_expected_fingerprint",
    ):
        assert payload[field] is None
    assert payload["legacy_synergy_decoder_ablation"] is False
    assert payload["synergy_include_baseline"] is False
    assert payload["synergy_residual_actuator_names"] == []
    for field in (
        "synergy_residual_alpha",
        "synergy_residual_l1_weight",
        "synergy_residual_l2_weight",
        "synergy_residual_smooth_weight",
        "synergy_baseline_l1_weight",
        "synergy_baseline_l2_weight",
    ):
        assert payload[field] == 0.0


def test_portable_latent_cli_clears_legacy_learned_decoder_fields():
    from fullbody.latent_train import _apply_latent_cli_overrides, build_parser

    args = build_parser().parse_args(
        [
            "--decoder_type",
            "synergy_residual",
            "--frozen-body-decoder-path",
            "/stage1/frozen",
            "--frozen-body-decoder-expected-fingerprint",
            "a" * 64,
        ]
    )
    payload = {
        "legacy_synergy_decoder_ablation": True,
        "synergy_basis_path": "/legacy/W",
        "synergy_basis_expected_fingerprint": "b" * 64,
        "synergy_include_baseline": True,
        "synergy_baseline_l1_weight": 1.0,
        "synergy_baseline_l2_weight": 1.0,
        "synergy_residual_actuator_names": ["finger"],
        "synergy_residual_alpha": 0.25,
        # Structured-rho regularizers remain meaningful with portable R.
        "synergy_residual_l1_weight": 0.1,
    }

    _apply_latent_cli_overrides(payload, args, phase_balance_weights=None)

    assert payload["decoder_type"] == "synergy_residual"
    assert payload["frozen_body_decoder_path"] == "/stage1/frozen"
    assert payload["legacy_synergy_decoder_ablation"] is False
    assert payload["synergy_basis_path"] is None
    assert payload["synergy_basis_expected_fingerprint"] is None
    assert payload["synergy_include_baseline"] is False
    assert payload["synergy_baseline_l1_weight"] == 0.0
    assert payload["synergy_baseline_l2_weight"] == 0.0
    assert payload["synergy_residual_actuator_names"] == []
    assert payload["synergy_residual_alpha"] == 0.0
    assert payload["synergy_residual_l1_weight"] == 0.1


def _write_latent_dataset(
    path,
    *,
    ctrlrange=None,
    physical_capture_schema=PHYSICAL_CAPTURE_SCHEMA_VERSION,
    excitation_delta=0.0,
    emg_synergy_dim=0,
    emg_valid_scale=1.0,
):
    teacher_action = np.array(
        [
            [0.2, -0.2],
            [0.25, -0.15],
            [-0.2, 0.2],
            [-0.25, 0.15],
        ],
        dtype=np.float32,
    )
    teacher_ctrl_physical = 0.5 * (teacher_action + 1.0)
    muscle_excitation = teacher_ctrl_physical + float(excitation_delta)
    data = {
        "student_obs": np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.1, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "reference_features": np.array(
            [
                [1.0, 0.0],
                [1.0, 0.1],
                [0.0, 1.0],
                [0.1, 1.0],
            ],
            dtype=np.float32,
        ),
        "teacher_action": teacher_action,
        "teacher_mu": teacher_action,
        "teacher_ctrl_physical": teacher_ctrl_physical,
        "muscle_excitation": muscle_excitation,
        "muscle_activation": 0.8 * teacher_ctrl_physical,
        "muscle_force": 10.0 * teacher_ctrl_physical,
        "muscle_tendon_length": 0.2 + 0.01 * teacher_ctrl_physical,
        "muscle_tendon_velocity": np.zeros_like(teacher_ctrl_physical),
        "actuator_power": np.zeros_like(teacher_ctrl_physical),
        "qfrc_actuator": np.zeros_like(teacher_ctrl_physical),
        "traj_no": np.array([0, 0, 1, 1], dtype=np.int32),
        "subtraj_step_no": np.array([0, 1, 0, 1], dtype=np.int32),
    }
    if int(emg_synergy_dim) > 0:
        dim = int(emg_synergy_dim)
        # Correlate the reference with the teacher action so a posterior that
        # reads the context has something real to learn from.
        base = np.abs(teacher_action[:, :1])
        mean = np.repeat(base, dim, axis=1).astype(np.float32)
        mean += 0.05 * np.arange(dim, dtype=np.float32)[None, :]
        data["emg_synergy_mean"] = mean
        data["emg_synergy_scale"] = np.full((4, dim), 0.1, dtype=np.float32)
        data["emg_synergy_valid"] = np.full((4, dim), float(emg_valid_scale), dtype=np.float32)
    ctrlrange = (
        [[0.0, 1.0], [0.0, 1.0]]
        if ctrlrange is None
        else np.asarray(ctrlrange, dtype=float).tolist()
    )
    names = ["hip", "shoulder"]
    body_semantic = {
        "schema_version": "body_obs_v1",
        "total_size": 3,
        "kinematic_size": 3,
        "muscle_size": 0,
        "touch_size": 0,
        "goal_size": 0,
        "other_size": 0,
        "action_size": 2,
        "root_joint_name": "root",
        "joint_names": [],
        "actuator_names": ["hip", "shoulder"],
        "touch_sensor_names": [],
        "observation_names": ["state"],
        "student_filtered": True,
        "channels": [
            {
                "name": f"state[{index}]",
                "entry": "state",
                "entry_type": "test",
                "entry_offset": index,
                "student_index": index,
                "category": "kinematic",
            }
            for index in range(3)
        ],
    }
    body_schema = body_semantic | {
        "semantic_hash": ordered_schema_hash(kind="body_observation", payload=body_semantic),
        "provenance": {"teacher_ckpt": "/tmp/teacher"},
    }
    write_split_shard(
        path,
        data,
        split="train",
        shard_idx=0,
        metadata={
            "actuator_names": names,
            "actuator_ctrlrange": ctrlrange,
            "ctrlrange_schema_hash": ordered_schema_hash(
                kind="actuator_ctrlrange",
                payload={"actuator_names": names, "ctrlrange": ctrlrange},
            ),
            "physical_signal_semantics": physical_signal_metadata(),
            "physical_capture": {
                "schema_version": physical_capture_schema,
                "actuator_names": names,
                "activation_valid_mask": [True] * len(names),
                "muscle_channel_contract": {
                    "schema_version": MUSCLE_CHANNEL_CONTRACT_SCHEMA_VERSION,
                    "actuator_names": names,
                    "actuator_ids": list(range(len(names))),
                    "actuator_dyntype": ["muscle"] * len(names),
                    "actuator_actnum": [1] * len(names),
                    "actuator_actadr": list(range(len(names))),
                    "model_na": len(names),
                },
            },
            "body_obs_schema": body_schema,
            "body_obs_schema_hash": body_schema["semantic_hash"],
        },
    )


def test_kl_warmup_schedule_reaches_target_after_warmup():
    from musclemimic.latent_muscle.train_latent import kl_warmup_weight

    assert kl_warmup_weight(step=0, target=0.2, warmup_steps=10) == 0.0
    assert kl_warmup_weight(step=5, target=0.2, warmup_steps=10) == 0.1
    assert kl_warmup_weight(step=15, target=0.2, warmup_steps=10) == 0.2
    assert kl_warmup_weight(step=15, target=0.2, warmup_steps=0) == 0.2


def test_temporal_smoothness_tracks_teacher_delta_instead_of_flattening_motion():
    from musclemimic.latent_muscle.train_latent import teacher_delta_smooth_mse

    teacher = np.array([[[0.0], [0.5], [1.0], [0.25]]], dtype=np.float32)
    matching_student = teacher.copy()
    flat_student = np.zeros_like(teacher)

    assert float(teacher_delta_smooth_mse(matching_student, teacher)) == 0.0
    assert float(teacher_delta_smooth_mse(flat_student, teacher)) > 0.0


def test_latent_checkpoint_roundtrip_preserves_action_mask_manifest(tmp_path):
    _require_latent_deps()
    from musclemimic.latent_muscle.checkpoint import load_latent_checkpoint, save_latent_checkpoint

    mask = ActionMask.from_correction_actuators(
        all_actuator_names=["hip", "finger", "shoulder"],
        correction_actuator_names=["finger"],
    )
    variables = {"params": {"w": np.array([1.0, 2.0], dtype=np.float32)}}

    save_latent_checkpoint(
        tmp_path,
        encoder_variables=variables,
        prior_variables=variables,
        decoder_variables=variables,
        optimizer_state={"step": np.array(3, dtype=np.int32)},
        action_mask=mask,
        config={"latent_dim": 2, "action_min": -1.0, "action_max": 1.0},
        train_metrics=[{"step": 1, "total_loss": 0.5}],
        eval_metrics={"action_mse": 0.25},
        obs_norm={"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
        action_norm={"min": -1.0, "max": 1.0},
        action_schema=_latent_action_schema(mask.body_actuator_names),
    )

    loaded = load_latent_checkpoint(tmp_path)

    assert (tmp_path / "encoder.msgpack").is_file()
    assert (tmp_path / "prior.msgpack").is_file()
    assert (tmp_path / "decoder.msgpack").is_file()
    assert (tmp_path / "optimizer_state.msgpack").is_file()
    assert (tmp_path / "latent_config.yaml").is_file()
    assert (tmp_path / "train_metrics.csv").is_file()
    assert (tmp_path / "eval_metrics.json").is_file()
    assert loaded["action_mask"]["decoder_action_dim"] == 2
    assert loaded["action_mask"]["full_action_dim"] == 3
    np.testing.assert_array_equal(loaded["encoder_variables"]["params"]["w"], variables["params"]["w"])

    action_norm_path = tmp_path / "action_norm.json"
    original_action_norm = action_norm_path.read_bytes()
    action_norm_path.write_bytes(original_action_norm + b"\n")
    with pytest.raises(ValueError, match="content fingerprint mismatch"):
        load_latent_checkpoint(tmp_path, runtime_only=True)
    action_norm_path.write_bytes(original_action_norm)


def test_passed_production_checkpoint_cannot_omit_strict_closed_loop_report(tmp_path):
    _require_latent_deps()
    from musclemimic.latent_muscle.checkpoint import load_latent_checkpoint, save_latent_checkpoint

    mask = ActionMask.from_correction_actuators(
        all_actuator_names=["hip", "shoulder"],
        correction_actuator_names=[],
    )
    variables = {"params": {"w": np.array([1.0], dtype=np.float32)}}
    save_latent_checkpoint(
        tmp_path,
        encoder_variables=variables,
        prior_variables=variables,
        decoder_variables=variables,
        optimizer_state={"step": np.array(0, dtype=np.int32)},
        action_mask=mask,
        config={"latent_dim": 1, "require_closed_loop_metrics": True},
        train_metrics=[],
        eval_metrics={"promotion": {"passed": True}},
        action_schema=_latent_action_schema(mask.body_actuator_names),
    )

    with pytest.raises(ValueError, match="missing closed_loop_metrics.json"):
        load_latent_checkpoint(tmp_path)


def test_latent_train_one_batch_writes_checkpoint_and_metrics(tmp_path):
    _require_latent_deps()
    from musclemimic.latent_muscle.train_latent import LatentTrainConfig, train_latent

    dataset_dir = tmp_path / "dataset"
    _write_latent_dataset(dataset_dir)
    output_dir = tmp_path / "latent_run"

    result = train_latent(
        LatentTrainConfig(
            dataset_dir=str(dataset_dir),
            output_dir=str(output_dir),
            latent_dim=2,
            hidden_layer_dims=(8,),
            batch_size=2,
            horizon=2,
            num_steps=3,
            learning_rate=1e-2,
            seed=0,
            kl_weight=1e-3,
            kl_warmup_steps=2,
            smooth_weight=0.1,
            log_interval=0,
            action_mask={
                "all_actuator_names": ["hip", "shoulder"],
                "correction_actuator_names": [],
            },
            closed_loop_evaluator=lambda _context: {
                "fall_or_early_termination_rate": 0.0,
                "lambda_025_050_no_fall_rate": 1.0,
            },
            promotion_gates={
                "closed_loop_max_fall_or_early_termination_rate": 0.05,
                "closed_loop_min_lambda_025_050_no_fall_rate": 0.95,
            },
        )
    )

    checkpoint_dir = output_dir / "latent_checkpoint"
    metrics = json.loads((checkpoint_dir / "eval_metrics.json").read_text(encoding="utf-8"))
    rows = (checkpoint_dir / "train_metrics.csv").read_text(encoding="utf-8").strip().splitlines()

    assert result.checkpoint_dir == str(checkpoint_dir)
    assert checkpoint_dir.is_dir()
    assert (checkpoint_dir / "encoder.msgpack").is_file()
    assert (checkpoint_dir / "prior.msgpack").is_file()
    assert (checkpoint_dir / "decoder.msgpack").is_file()
    assert len(rows) >= 2
    assert np.isfinite(metrics["action_mse"])
    assert metrics["action_min"] == -1.0
    assert metrics["action_max"] == 1.0
    assert np.isfinite(metrics["posterior_action_mse"])
    assert np.isfinite(metrics["prior_mean_action_mse"])
    assert metrics["closed_loop_fall_or_early_termination_rate"] == 0.0
    assert (checkpoint_dir / "state_schema.json").is_file()
    assert (checkpoint_dir / "action_schema.json").is_file()
    assert (checkpoint_dir / "motion_split.json").is_file()
    assert (checkpoint_dir / "body_obs_schema.json").is_file()
    assert (checkpoint_dir / "checkpoint_fingerprint.txt").is_file()
    action_schema = json.loads(
        (checkpoint_dir / "action_schema.json").read_text(encoding="utf-8")
    )
    assert action_schema["schema_version"] == "latent_muscle_action_v2"
    assert action_schema["physical_signal_schema_version"] == (
        physical_signal_metadata()["schema_version"]
    )
    assert action_schema["target_ctrlrange"] == [[0.0, 1.0], [0.0, 1.0]]
    obs_norm = json.loads((checkpoint_dir / "obs_norm.json").read_text(encoding="utf-8"))
    assert obs_norm["count"] == 4
    assert obs_norm["source_split"] == "train"

    from musclemimic.latent_muscle.runtime import (
        LatentCheckpointCompatibilityError,
        LatentMuscleRuntime,
    )

    runtime = LatentMuscleRuntime.from_checkpoint(
        checkpoint_dir,
        runtime_body_actuator_names=["hip", "shoulder"],
    )
    assert runtime.body_obs_schema["actuator_names"] == ["hip", "shoulder"]
    assert runtime.body_obs_schema["action_size"] == 2
    state = np.array([[0.5, 0.0, 0.0]], dtype=np.float32)
    prior_mu, prior_sigma = runtime.prior_numpy(state)
    prior_mu_raw, prior_raw_sigma = runtime.prior_raw_numpy(state)
    action = runtime.decode_numpy(state, prior_mu)
    assert prior_mu.shape == (1, 2)
    assert prior_sigma.shape == (1, 2)
    assert action.shape == (1, 2)
    np.testing.assert_allclose(prior_mu_raw, prior_mu)
    expected_sigma = np.clip(
        np.log1p(np.exp(-np.abs(prior_raw_sigma))) + np.maximum(prior_raw_sigma, 0.0),
        runtime.sigma_min,
        runtime.sigma_max,
    )
    np.testing.assert_allclose(prior_sigma, expected_sigma, rtol=1e-6)

    import jax
    import jax.numpy as jnp

    jitted = jax.jit(runtime.prior_mean_action_jax)(jnp.asarray(state))
    np.testing.assert_allclose(np.asarray(jitted), runtime.prior_mean_action_numpy(state), rtol=1e-5)
    with pytest.raises(LatentCheckpointCompatibilityError, match="actuator names"):
        LatentMuscleRuntime.from_checkpoint(
            checkpoint_dir,
            runtime_body_actuator_names=["shoulder", "hip"],
        )

    # Deployment bundles intentionally omit posterior/optimizer training state.
    (checkpoint_dir / "encoder.msgpack").unlink()
    (checkpoint_dir / "optimizer_state.msgpack").unlink()
    runtime_only = LatentMuscleRuntime.from_checkpoint(checkpoint_dir)
    assert runtime_only.checkpoint_dir == str(checkpoint_dir)
    assert runtime_only.prior_mean_action_numpy(state).shape == (1, 2)
    assert runtime.body_obs_schema["root_joint_name"] == "root"
    np.testing.assert_allclose(runtime.body_ctrlrange, [[0.0, 1.0], [0.0, 1.0]])
    np.testing.assert_allclose(runtime.body_action_to_ctrl_numpy([[-1.0, 1.0]]), [[0.0, 1.0]])
    assert len(runtime.checkpoint_fingerprint) == 64
    assert runtime.control_manifest["checkpoint_fingerprint"] == runtime.checkpoint_fingerprint


def test_latent_checkpoint_action_schema_rejects_pre_v2_and_signed_ranges():
    from musclemimic.latent_muscle.checkpoint import (
        validate_latent_muscle_action_schema,
    )

    names = ["hip", "shoulder"]
    contract = {
        "schema_version": MUSCLE_CHANNEL_CONTRACT_SCHEMA_VERSION,
        "actuator_names": names,
        "actuator_ids": [0, 1],
        "actuator_dyntype": ["muscle", "muscle"],
        "actuator_actnum": [1, 1],
        "actuator_actadr": [0, 1],
        "model_na": 2,
    }
    schema = {
        "schema_version": "latent_muscle_action_v2",
        "selection_schema_version": "named_action_v1",
        "target_actuator_names": names,
        "target_schema_hash": actuator_schema_hash(names),
        "target_ctrlrange": [[0.0, 1.0], [0.0, 1.0]],
        "ctrlrange_schema_hash": ordered_schema_hash(
            kind="actuator_ctrlrange",
            payload={
                "actuator_names": names,
                "ctrlrange": [[0.0, 1.0], [0.0, 1.0]],
            },
        ),
        "physical_signal_schema_version": physical_signal_metadata()["schema_version"],
        "physical_capture_schema_version": PHYSICAL_CAPTURE_SCHEMA_VERSION,
        "muscle_excitation_transform": physical_signal_metadata()["muscle_excitation"][
            "transform"
        ],
        "muscle_excitation_formula": physical_signal_metadata()["muscle_excitation"][
            "formula"
        ],
        "muscle_excitation_roundoff_policy": physical_signal_metadata()[
            "muscle_excitation"
        ]["roundoff_policy"],
        "muscle_channel_contract": contract,
    }
    np.testing.assert_array_equal(
        validate_latent_muscle_action_schema(schema, expected_names=names),
        [[0.0, 1.0], [0.0, 1.0]],
    )

    legacy = {**schema, "schema_version": "named_action_v1"}
    with pytest.raises(ValueError, match="pre-v2"):
        validate_latent_muscle_action_schema(legacy, expected_names=names)

    signed = {
        **schema,
        "target_ctrlrange": [[-1.0, 1.0], [-1.0, 1.0]],
        "ctrlrange_schema_hash": ordered_schema_hash(
            kind="actuator_ctrlrange",
            payload={
                "actuator_names": names,
                "ctrlrange": [[-1.0, 1.0], [-1.0, 1.0]],
            },
        ),
    }
    with pytest.raises(ValueError, match=r"exactly \[0,1\]"):
        validate_latent_muscle_action_schema(signed, expected_names=names)


@pytest.mark.parametrize(
    ("ctrlrange", "capture_schema", "message"),
    [
        (
            [[-1.0, 1.0], [-1.0, 1.0]],
            PHYSICAL_CAPTURE_SCHEMA_VERSION,
            r"exactly \[0,1\]",
        ),
        (
            [[0.0, 1.0], [0.0, 1.0]],
            "physical_capture_spec_v1",
            "physical_capture_spec_v2",
        ),
    ],
)
def test_latent_train_rejects_pre_v2_or_signed_physical_dataset(
    tmp_path,
    ctrlrange,
    capture_schema,
    message,
):
    _require_latent_deps()
    from musclemimic.latent_muscle.train_latent import LatentTrainConfig, train_latent

    dataset_dir = tmp_path / "dataset"
    _write_latent_dataset(
        dataset_dir,
        ctrlrange=ctrlrange,
        physical_capture_schema=capture_schema,
    )

    with pytest.raises(ValueError, match=message):
        train_latent(
            LatentTrainConfig(
                dataset_dir=str(dataset_dir),
                output_dir=str(tmp_path / "latent_run"),
                latent_dim=2,
                hidden_layer_dims=(8,),
                batch_size=2,
                horizon=2,
                num_steps=1,
                action_mask={
                    "all_actuator_names": ["hip", "shoulder"],
                    "correction_actuator_names": [],
                },
            )
        )


def test_latent_train_rejects_forged_v2_excitation_values(tmp_path):
    _require_latent_deps()
    from musclemimic.latent_muscle.train_latent import (
        LatentTrainConfig,
        train_latent,
    )

    dataset_dir = tmp_path / "dataset"
    _write_latent_dataset(dataset_dir, excitation_delta=0.1)

    with pytest.raises(ValueError, match=r"differs from clip\(raw data.ctrl"):
        train_latent(
            LatentTrainConfig(
                dataset_dir=str(dataset_dir),
                output_dir=str(tmp_path / "latent_run"),
                latent_dim=2,
                hidden_layer_dims=(8,),
                batch_size=2,
                horizon=2,
                num_steps=1,
                action_mask={
                    "all_actuator_names": ["hip", "shoulder"],
                    "correction_actuator_names": [],
                },
            )
        )


def test_latent_train_uses_absolute_roundoff_tolerance_and_reports_location(
    tmp_path,
):
    _require_latent_deps()
    from musclemimic.latent_muscle.train_latent import (
        LatentTrainConfig,
        train_latent,
    )

    dataset_dir = tmp_path / "dataset"
    # This would pass the former relative tolerance for values near 0.5.
    _write_latent_dataset(dataset_dir, excitation_delta=4e-6)

    with pytest.raises(
        ValueError,
        match=r"sample=0 channel=0",
    ):
        train_latent(
            LatentTrainConfig(
                dataset_dir=str(dataset_dir),
                output_dir=str(tmp_path / "latent_run"),
                latent_dim=2,
                hidden_layer_dims=(8,),
                batch_size=2,
                horizon=2,
                num_steps=1,
                action_mask={
                    "all_actuator_names": ["hip", "shoulder"],
                    "correction_actuator_names": [],
                },
            )
        )


def _write_emg_reference_manifest(path, *, synergy_dim=3):
    """Minimal on-disk EMG reference stand-in for provenance hashing."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "emg_reference_v1",
                "synergy_dim": int(synergy_dim),
                "subject": "P002",
                "task": "forehand_clear",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _emg_latent_config(dataset_dir, output_dir, **overrides):
    from musclemimic.latent_muscle.train_latent import LatentTrainConfig

    manifest = _write_emg_reference_manifest(
        dataset_dir.parent / "emg_reference" / "manifest.json"
    )
    payload = {
        "emg_reference_manifest": str(manifest),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "latent_dim": 2,
        "hidden_layer_dims": (8,),
        "batch_size": 2,
        "horizon": 2,
        "num_steps": 2,
        "action_mask": {
            "all_actuator_names": ["hip", "shoulder"],
            "correction_actuator_names": [],
        },
        "emg_privileged_enabled": True,
        "emg_synergy_dim": 3,
        "emg_synergy_loss_weight": 0.5,
        "emg_context_dropout": 0.25,
        # Small optimizer tests use a minimal manifest stand-in.  Production
        # configs keep this true and are covered by the strict review-gate test
        # below.
        "emg_require_reference_hash": False,
    }
    payload.update(overrides)
    return LatentTrainConfig(**payload)


def _formal_trial_qc_provenance(
    *, action: str, channels: tuple[str, ...], mapping_sha256: str
):
    return {
        "trial_qc_review": {
            "schema_version": "emg_trial_channel_qc_review_v1",
            "action": action,
            "review_status": "verified",
            "training_enabled": True,
            "source_path": "/controlled/evidence/emg_trial_qc_review.json",
            "review_sha256": "d" * 64,
            "mapping_sha256": mapping_sha256,
            "reviewer_id": "domain_expert",
            "reviewed_at": "2025-07-20T00:00:00Z",
            "review_evidence": ["controlled-review-record"],
            "trial_decisions": [
                {
                    "trial_id": "trial_001",
                    "decision": "include",
                    "reason": "reviewed synthetic fixture",
                    "mvc_normalized_emg_sha256": "1" * 64,
                    "preprocessing_qc_sha256": "2" * 64,
                }
            ],
            "channel_decisions": [
                {
                    "emg_channel": name,
                    "decision": "include_after_review",
                    "reason": "reviewed synthetic fixture",
                }
                for name in channels
            ],
            "risk_decisions": [
                {
                    "risk_id": risk_id,
                    "decision": "mitigated",
                    "reason": "reviewed synthetic fixture",
                    "evidence": ["controlled-review-record"],
                }
                for risk_id in ("s9_progressive_near_flatline", "super_mvc")
            ],
        }
    }


def test_formal_privileged_latent_rejects_provisional_emg_reference(
    tmp_path,
    monkeypatch,
):
    from musclemimic.latent_muscle.train_latent import (
        _validate_emg_privileged_config,
    )

    provisional = SimpleNamespace(
        review_status="provisional",
        training_enabled=False,
        reference_id="diagnostics-only",
        mapping_binding={"mapping_review_status": "provisional"},
        anchor_valid=np.ones((1, 1, 1), dtype=bool),
        synergy_valid=np.ones((1, 1, 1), dtype=bool),
        synergy_count=3,
    )
    monkeypatch.setattr(
        "musclemimic.physiology.load_emg_phase_reference_tube",
        lambda _path: provisional,
    )
    config = _emg_latent_config(
        tmp_path / "dataset",
        tmp_path / "run",
        emg_require_reference_hash=True,
    )

    with pytest.raises(ValueError, match="mapping review must complete"):
        _validate_emg_privileged_config(config)


def test_formal_privileged_latent_binds_dataset_rows_to_exact_tube(
    tmp_path,
    monkeypatch,
):
    from musclemimic.latent_muscle.train_latent import (
        _validate_optional_training_fields,
    )

    verified = SimpleNamespace(
        review_status="verified",
        training_enabled=True,
        reference_id="verified-reference",
        reference_fingerprint="a" * 64,
        mapping_binding={
            "mapping_review_status": "verified",
            "mapping_sha256": "b" * 64,
        },
        action_ids=("forehand_clear",),
        channel_names=("S2",),
        provenance=_formal_trial_qc_provenance(
            action="forehand_clear",
            channels=("S2",),
            mapping_sha256="b" * 64,
        ),
        anchor_valid=np.ones((1, 1, 1), dtype=bool),
        synergy_valid=np.ones((1, 1, 3), dtype=bool),
        synergy_count=3,
    )
    monkeypatch.setattr(
        "musclemimic.physiology.load_emg_phase_reference_tube",
        lambda _path: verified,
    )
    config = _emg_latent_config(
        tmp_path / "dataset",
        tmp_path / "run",
        emg_require_reference_hash=True,
    )
    arrays = {
        "emg_synergy_mean": np.zeros((2, 3), dtype=np.float32),
        "emg_synergy_scale": np.ones((2, 3), dtype=np.float32),
        "emg_synergy_valid": np.ones((2, 3), dtype=np.float32),
    }
    dataset = SimpleNamespace(
        num_samples=2,
        arrays=arrays,
        metadata={
            "emg_reference_semantics": {
                "schema_version": "emg_reference_capture_v1",
                "reference_id": "verified-reference",
                "reference_fingerprint": "c" * 64,
                "mapping_sha256": "b" * 64,
                "synergy_count": 3,
            }
        },
    )

    with pytest.raises(ValueError, match="different reference tube"):
        _validate_optional_training_fields(
            dataset,
            config,
            SimpleNamespace(),
            split="train",
        )


def test_privileged_emg_latent_training_records_provenance_and_emg_metrics(tmp_path):
    _require_latent_deps()
    from musclemimic.latent_muscle.train_latent import train_latent

    dataset_dir = tmp_path / "dataset"
    _write_latent_dataset(dataset_dir, emg_synergy_dim=3)
    output_dir = tmp_path / "latent_run"

    result = train_latent(_emg_latent_config(dataset_dir, output_dir))
    checkpoint_dir = Path(result.checkpoint_dir)

    provenance = json.loads(
        (checkpoint_dir / "training_provenance.json").read_text(encoding="utf-8")
    )["emg_privileged"]
    assert provenance is not None
    assert provenance["synergy_dim"] == 3
    assert provenance["context_dropout"] == 0.25
    # 3 means + 3 log-scales + 3 confidences
    assert provenance["context_width"] == 9
    assert provenance["shuffle_context_ablation"] is False
    assert "EMG-free at runtime" in provenance["deployment_note"]

    metrics = json.loads(
        (checkpoint_dir / "eval_metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["emg_context_width"] == 9
    assert metrics["emg_valid_fraction"] == pytest.approx(1.0)
    assert np.isfinite(metrics["emg_synergy_head_loss"])
    assert metrics["emg_synergy_head_loss"] >= 0.0
    assert -1.0 <= metrics["emg_synergy_head_correlation"] <= 1.0
    # Zeroing the context must be a defined, finite operation: that is the
    # deployment condition, and it has to stay evaluable.
    assert np.isfinite(metrics["emg_blank_context_action_mse"])
    assert metrics["emg_blank_context_posterior_mu_l2"] >= 0.0


def test_deployed_prior_and_decoder_never_receive_emg_context(tmp_path):
    _require_latent_deps()
    import jax.numpy as jnp

    from musclemimic.latent_muscle.checkpoint import load_latent_checkpoint
    from musclemimic.latent_muscle.networks import ConditionalPrior
    from musclemimic.latent_muscle.train_latent import train_latent

    dataset_dir = tmp_path / "dataset"
    _write_latent_dataset(dataset_dir, emg_synergy_dim=3)
    output_dir = tmp_path / "latent_run"
    result = train_latent(_emg_latent_config(dataset_dir, output_dir))

    checkpoint = load_latent_checkpoint(result.checkpoint_dir)
    # The prior is the deployed encoder-side module.  It must take state only,
    # which is what makes the EMG dependency train-time-only.
    prior = ConditionalPrior(latent_dim=2, hidden_layer_dims=(8,))
    mu, raw_sigma = prior.apply(
        checkpoint["prior_variables"], jnp.zeros((1, 3), dtype=jnp.float32)
    )
    assert mu.shape == (1, 2)
    assert raw_sigma.shape == (1, 2)
    assert "synergy_head" not in checkpoint["prior_variables"]["params"]
    assert "synergy_head" not in checkpoint["decoder_variables"]["params"]


def test_shuffled_emg_context_ablation_is_recorded_as_such(tmp_path):
    _require_latent_deps()
    from musclemimic.latent_muscle.train_latent import train_latent

    dataset_dir = tmp_path / "dataset"
    _write_latent_dataset(dataset_dir, emg_synergy_dim=3)

    result = train_latent(
        _emg_latent_config(
            dataset_dir,
            tmp_path / "shuffled_run",
            emg_shuffle_context_ablation=True,
        )
    )

    provenance = json.loads(
        (Path(result.checkpoint_dir) / "training_provenance.json").read_text(
            encoding="utf-8"
        )
    )["emg_privileged"]
    assert provenance["shuffle_context_ablation"] is True


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"emg_privileged_enabled": False},
            r"requires emg_privileged_enabled",
        ),
        (
            {"emg_synergy_dim": 0},
            r"positive emg_synergy_dim",
        ),
        (
            {"emg_context_dropout": 1.5},
            r"emg_context_dropout must lie in \[0, 1\]",
        ),
        (
            {"emg_tube_kappa": 0.0},
            r"emg_tube_kappa must be positive",
        ),
    ],
)
def test_incoherent_privileged_emg_config_fails_closed(tmp_path, overrides, message):
    _require_latent_deps()
    from musclemimic.latent_muscle.train_latent import train_latent

    dataset_dir = tmp_path / "dataset"
    _write_latent_dataset(dataset_dir, emg_synergy_dim=3)

    with pytest.raises(ValueError, match=message):
        train_latent(
            _emg_latent_config(dataset_dir, tmp_path / "bad_run", **overrides)
        )


def test_privileged_emg_requires_reference_fields_in_dataset(tmp_path):
    _require_latent_deps()
    from musclemimic.latent_muscle.train_latent import train_latent

    dataset_dir = tmp_path / "dataset"
    _write_latent_dataset(dataset_dir, emg_synergy_dim=0)

    with pytest.raises(ValueError, match=r"requires 'emg_synergy_mean'"):
        train_latent(_emg_latent_config(dataset_dir, tmp_path / "missing_run"))


def test_all_invalid_emg_reference_is_rejected_as_pure_noise(tmp_path):
    _require_latent_deps()
    from musclemimic.latent_muscle.train_latent import train_latent

    dataset_dir = tmp_path / "dataset"
    _write_latent_dataset(dataset_dir, emg_synergy_dim=3, emg_valid_scale=0.0)

    with pytest.raises(ValueError, match=r"no valid synergy component"):
        train_latent(_emg_latent_config(dataset_dir, tmp_path / "noise_run"))

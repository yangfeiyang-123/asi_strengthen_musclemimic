from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from musclemimic.distill.collect_teacher import (
    SIMULATOR_PRE_STATE_FIELDS,
    _capture_physical_transition,
    _capture_simulator_pre_state,
)
from musclemimic.latent_muscle.causal_rollout_artifact import REQUIRED_OUTCOMES
from musclemimic.latent_muscle.stage2_causal_adapter import (
    SNAPSHOT_SCHEMA_VERSION,
    _build_outcome_schemas,
    _decode_snapshot,
    _encode_snapshot,
    _inject_dataset_state,
    _model_fingerprint,
    _time_major_activation_valid_mask,
    _trunk_state_spec,
    _validate_simulator_state_dataset,
    _validated_config,
)


def _config(**updates):
    result = {
        "latent_checkpoint": "latent",
        "teacher_ckpt": "teacher",
        "dataset_dir": "train-data",
        "val_dataset_dir": "val-data",
        "analysis_inputs": "analysis.npz",
        "analysis_manifest": "analysis.json",
    }
    result.update(updates)
    return result


def test_stage2_adapter_config_accepts_pipeline_keys_and_rejects_expansion():
    config = _validated_config(_config())
    assert config["rollout_horizon_steps"] == 120
    assert config["state_match_atol"] == pytest.approx(1e-5)

    custom = _validated_config(_config(val_dataset_dir=None, rollout_horizon_steps=7, state_match_atol=0.0))
    assert custom["val_dataset_dir"] is None
    assert custom["rollout_horizon_steps"] == 7
    assert custom["state_match_atol"] == 0.0

    with pytest.raises(ValueError, match="unsupported keys"):
        _validated_config(_config(approximate_frame_fallback=True))
    with pytest.raises(ValueError, match="missing"):
        incomplete = _config()
        incomplete.pop("analysis_manifest")
        _validated_config(incomplete)
    with pytest.raises(ValueError, match="positive"):
        _validated_config(_config(rollout_horizon_steps=0))
    with pytest.raises(ValueError, match="strict"):
        _validated_config(_config(state_match_atol=1e-3))


def test_stage2_snapshot_encoding_is_canonical_and_rejects_tampering():
    header = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "source_index": 0,
        "sample_uid": "stable-sample",
        "model_fingerprint": "f" * 64,
        "state_spec": "mjSTATE_INTEGRATION",
    }
    parts = {
        "observation": b"obs",
        "mujoco_state": b"state",
        "info": b"info",
        "carry": b"carry",
    }
    encoded = _encode_snapshot(header, parts)
    assert encoded == _encode_snapshot(header, dict(reversed(list(parts.items()))))
    decoded_header, decoded_parts = _decode_snapshot(encoded)
    assert decoded_header["sample_uid"] == "stable-sample"
    assert decoded_parts == parts

    with pytest.raises(ValueError, match="trailing bytes"):
        _decode_snapshot(encoded + b"tamper")
    with pytest.raises(ValueError, match="magic"):
        _decode_snapshot(b"not-a-snapshot")
    invalid_header = dict(header, model_fingerprint="not-canonical")
    with pytest.raises(ValueError, match="unsupported"):
        _decode_snapshot(_encode_snapshot(invalid_header, parts))


def test_stage2_outcome_contract_exposes_six_diagnostics_but_no_fake_task_values():
    schemas = _build_outcome_schemas(
        horizon=2,
        muscle_names=["muscle_a", "muscle_b"],
        qpos_names=["joint:q"],
        qvel_names=["joint:v"],
        trunk_names=["pelvis_root:position_x", "lumbar_joint:spine:angular_velocity"],
        trunk_units=["meter", "radian_per_second"],
    )
    assert set(schemas) == set(REQUIRED_OUTCOMES)
    diagnostics = REQUIRED_OUTCOMES[:6]
    assert all(schemas[name]["available"] for name in diagnostics)
    assert all(schemas[name]["feature_names"] for name in diagnostics)
    activation_mask = _time_major_activation_valid_mask(
        [True, True],
        horizon=2,
    )
    assert len(activation_mask) == len(schemas["muscle_activation"]["feature_names"])
    assert activation_mask == [True, True, True, True]
    for name in ("impact_outcome", "landing_outcome"):
        assert schemas[name]["available"] is False
        assert schemas[name]["feature_names"] == []
        assert schemas[name]["units"] == []


def test_trunk_contract_includes_pelvis_root_and_explicit_rotational_torso_chain():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="pelvis">
              <freejoint name="root"/>
              <geom type="sphere" size="0.1" mass="1"/>
              <body name="lumbar">
                <joint name="lumbar_slide" type="slide" axis="1 0 0"/>
                <joint name="lumbar_bend" type="hinge" axis="0 1 0"/>
                <geom type="sphere" size="0.05" mass="1"/>
                <body name="torso"/>
              </body>
              <body name="arm"><joint name="arm_hinge" type="hinge"/>
                <geom type="sphere" size="0.05" mass="1"/></body>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    base = SimpleNamespace(
        _model=model,
        root_free_joint_xml_name="root",
        upper_body_xml_name="torso",
    )
    qpos, qvel, names, units = _trunk_state_spec(base)
    assert qpos.shape == (7,)
    assert qvel.shape == (7,)
    assert names[:2] == ["pelvis_root:position_x", "pelvis_root:position_y"]
    assert names[-1] == "lumbar_joint:lumbar_bend:angular_velocity"
    assert "arm_hinge" not in " ".join(names)
    assert "lumbar_slide" not in " ".join(names)
    assert units[-1] == "radian_per_second"
    fingerprint = _model_fingerprint(model)
    assert fingerprint == _model_fingerprint(model)
    model.geom_friction[0, 0] += 0.25
    assert fingerprint != _model_fingerprint(model)

    no_spine = SimpleNamespace(
        _model=mujoco.MjModel.from_xml_string(
            """
            <mujoco><worldbody><body name="pelvis"><freejoint name="root"/>
            <geom type="sphere" size="0.1" mass="1"/>
            <body name="torso"/></body></worldbody></mujoco>
            """
        ),
        root_free_joint_xml_name="root",
        upper_body_xml_name="torso",
    )
    with pytest.raises(ValueError, match="pelvis-only trunk relabeling is forbidden"):
        _trunk_state_spec(no_spine)


def test_physical_collection_captures_every_pre_transition_integration_field():
    data = SimpleNamespace(
        **{
            name: np.arange(6, dtype=np.float32).reshape(2, 3) + index
            for index, name in enumerate(SIMULATOR_PRE_STATE_FIELDS)
        }
    )
    captured = _capture_simulator_pre_state(data)
    assert set(captured) == {f"sim_pre_{name}" for name in SIMULATOR_PRE_STATE_FIELDS}
    for name in SIMULATOR_PRE_STATE_FIELDS:
        assert np.array_equal(np.asarray(captured[f"sim_pre_{name}"]), getattr(data, name))

    del data.qpos
    with pytest.raises(ValueError, match="lacks required causal snapshot fields"):
        _capture_simulator_pre_state(data)


def test_physical_capture_accepts_cpu_and_mjx_site_rotation_layouts():
    spec = {
        "actuator_ids": np.asarray([0], dtype=np.int32),
        "act_addresses": np.asarray([0], dtype=np.int32),
        "activation_valid": np.asarray([True]),
        "has_activation": True,
        "ctrl_lower": np.asarray([0.0], dtype=np.float32),
        "ctrl_upper": np.asarray([1.0], dtype=np.float32),
        "racket_site_id": 0,
        "racket_body_id": 1,
        "racket_root_id": 1,
    }
    identity = np.eye(3, dtype=np.float32)
    data = SimpleNamespace(
        ctrl=np.asarray([0.4], dtype=np.float32),
        act=np.asarray([0.3], dtype=np.float32),
        actuator_velocity=np.asarray([0.2], dtype=np.float32),
        actuator_force=np.asarray([2.0], dtype=np.float32),
        actuator_length=np.asarray([0.8], dtype=np.float32),
        qfrc_actuator=np.asarray([1.0], dtype=np.float32),
        site_xpos=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
        site_xmat=identity.reshape(1, 9),
        cvel=np.zeros((2, 6), dtype=np.float32),
        subtree_com=np.zeros((2, 3), dtype=np.float32),
    )
    cpu = _capture_physical_transition(data, spec)
    assert np.array_equal(np.asarray(cpu["racket_rotation_matrix"]), identity)

    data.site_xmat = identity.reshape(1, 3, 3)
    mjx = _capture_physical_transition(data, spec)
    assert np.array_equal(np.asarray(mjx["racket_rotation_matrix"]), identity)


def test_state_dataset_and_unique_coordinate_restore_fail_closed(monkeypatch):
    legacy = SimpleNamespace(metadata={}, arrays={}, num_samples=1)
    with pytest.raises(ValueError, match="re-collect"):
        _validate_simulator_state_dataset(legacy, name="train")

    source = SimpleNamespace(
        name="train",
        dataset=SimpleNamespace(
            arrays={
                "traj_no": np.asarray([3]),
                "subtraj_step_no": np.asarray([5]),
                "student_obs": np.asarray([[1.0, 2.0]], dtype=np.float32),
            }
        ),
    )
    calls = []

    def fake_inject(_source, *, row, traj, step):
        calls.append((row, traj, step))
        return np.asarray([1.0, 2.0] if step == 4 else [9.0, 9.0], dtype=np.float32)

    monkeypatch.setattr(
        "musclemimic.latent_muscle.stage2_causal_adapter._inject_candidate",
        fake_inject,
    )
    _inject_dataset_state(source, row=0, state_match_atol=0.0)
    assert calls == [(0, 3, 4), (0, 3, 5), (0, 3, 4)]

    monkeypatch.setattr(
        "musclemimic.latent_muscle.stage2_causal_adapter._inject_candidate",
        lambda *_args, **_kwargs: np.asarray([9.0, 9.0], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="did not uniquely reproduce"):
        _inject_dataset_state(source, row=0, state_match_atol=0.0)

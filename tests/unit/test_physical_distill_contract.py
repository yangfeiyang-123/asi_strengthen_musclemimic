"""CPU-only tests for the physical muscle distillation ABI."""

from __future__ import annotations

import numpy as np
import pytest

from musclemimic.distill.action_schema import ordered_schema_hash
from musclemimic.distill.collect_teacher import _validate_physical_batch
from musclemimic.distill.dataset import PhysicalDistillDataset, write_split_shard
from musclemimic.distill.physical import (
    MUSCLE_CHANNEL_CONTRACT_SCHEMA_VERSION,
    PHYSICAL_CAPTURE_SCHEMA_VERSION,
    effective_muscle_excitation_to_physical_ctrl,
    normalized_action_to_physical_ctrl,
    physical_ctrl_to_effective_muscle_excitation,
    physical_ctrl_to_normalized_action,
    physical_signal_metadata,
    unit_excitation_to_physical_ctrl,
    validate_ordered_ctrlrange,
    validate_unit_muscle_activation,
    validate_unit_muscle_ctrlrange,
)


def _muscle_contract(names):
    width = len(names)
    return {
        "schema_version": MUSCLE_CHANNEL_CONTRACT_SCHEMA_VERSION,
        "actuator_names": list(names),
        "actuator_ids": list(range(width)),
        "actuator_dyntype": ["muscle"] * width,
        "actuator_actnum": [1] * width,
        "actuator_actadr": list(range(width)),
        "model_na": width,
    }


def _physical_capture(names):
    return {
        "schema_version": PHYSICAL_CAPTURE_SCHEMA_VERSION,
        "actuator_names": list(names),
        "activation_valid_mask": [True] * len(names),
        "muscle_channel_contract": _muscle_contract(names),
    }


def test_generic_signed_ctrl_coordinates_are_not_production_excitation():
    names = ["signed", "offset", "unit"]
    ctrlrange = validate_ordered_ctrlrange(
        names,
        [[-1.0, 1.0], [-2.0, 4.0], [0.0, 1.0]],
    )
    normalized = np.asarray(
        [[-1.0, 0.0, 1.0], [0.25, -0.5, 0.4]],
        dtype=np.float64,
    )

    physical_ctrl = normalized_action_to_physical_ctrl(normalized, ctrlrange)

    np.testing.assert_allclose(
        physical_ctrl_to_normalized_action(physical_ctrl, ctrlrange),
        normalized,
    )
    assert physical_ctrl[0, 0] == -1.0
    with pytest.raises(ValueError, match="legacy signed/mixed"):
        validate_unit_muscle_ctrlrange(names, ctrlrange)


def test_verified_muscle_excitation_is_exact_raw_ctrl_clip_and_keeps_raw():
    names = ("m0", "m1")
    contract = _muscle_contract(names)
    raw = np.asarray([[-0.2, 0.25], [0.75, 1.2]], dtype=np.float64)
    excitation = physical_ctrl_to_effective_muscle_excitation(
        raw,
        channel_contract=contract,
    )

    np.testing.assert_array_equal(excitation, [[0.0, 0.25], [0.75, 1.0]])
    np.testing.assert_array_equal(raw, [[-0.2, 0.25], [0.75, 1.2]])
    np.testing.assert_array_equal(
        effective_muscle_excitation_to_physical_ctrl(
            excitation,
            np.asarray([[0.0, 1.0], [0.0, 1.0]]),
            actuator_names=names,
            channel_contract=contract,
        ),
        excitation,
    )
    with pytest.warns(DeprecationWarning, match="effective_muscle_excitation_to_physical_ctrl"):
        legacy = unit_excitation_to_physical_ctrl(
            excitation,
            np.asarray([[0.0, 1.0], [0.0, 1.0]]),
            actuator_names=names,
            channel_contract=contract,
        )
    np.testing.assert_array_equal(legacy, excitation)


def test_physical_transform_fails_closed_without_verified_muscle_contract():
    contract = _muscle_contract(("m0", "m1"))
    contract["actuator_dyntype"][1] = "filter"
    with pytest.raises(ValueError, match="dyntype"):
        physical_ctrl_to_effective_muscle_excitation(
            [[0.2, 0.5]],
            channel_contract=contract,
        )
    with pytest.raises(ValueError, match="actuator_actnum=1"):
        physical_ctrl_to_effective_muscle_excitation(
            [[0.2, 0.5]],
            channel_contract={
                **_muscle_contract(("m0", "m1")),
                "actuator_actnum": [1, 0],
            },
        )


def test_activation_clips_roundoff_only_and_rejects_semantic_range_errors():
    activation = validate_unit_muscle_activation(
        np.asarray([[-5e-7, 1.0 + 5e-7]], dtype=np.float64)
    )
    np.testing.assert_array_equal(activation, [[0.0, 1.0]])
    with pytest.raises(ValueError, match=r"outside \[0,1\]"):
        validate_unit_muscle_activation([[0.2, 1.001]])


def test_physical_dataset_selects_every_actuator_channel_by_name(tmp_path):
    names = ["m0", "m1", "m2", "m3"]
    target_names = ["m3", "m1"]
    ctrlrange = np.tile(np.asarray([[0.0, 1.0]], dtype=np.float64), (4, 1))
    rows = np.arange(12, dtype=np.float32).reshape(3, 4)
    raw_ctrl = np.asarray(
        [[-0.1, 0.0, 0.0, 0.2], [0.0, 0.25, 0.5, 1.0], [1.1, 0.75, 1.0, 0.8]],
        dtype=np.float32,
    )
    data = {
        "student_obs": np.zeros((3, 2), dtype=np.float32),
        "teacher_action": np.zeros((3, 4), dtype=np.float32),
        "teacher_ctrl_physical": raw_ctrl,
        "muscle_excitation": physical_ctrl_to_effective_muscle_excitation(
            raw_ctrl,
            channel_contract=_muscle_contract(names),
        ),
        "muscle_activation": np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(3, 4),
        "muscle_force": rows + 20.0,
        "muscle_tendon_length": rows + 30.0,
        "muscle_tendon_velocity": rows + 40.0,
        "actuator_power": rows + 50.0,
        "qfrc_actuator": np.arange(18, dtype=np.float32).reshape(3, 6),
        "phase_id": np.asarray([0, 1, 2], dtype=np.int32),
        "phase_local": np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        "time_to_impact_s": np.asarray([0.2, 0.0, -0.1], dtype=np.float32),
        "time_from_impact_s": np.asarray([-0.2, 0.0, 0.1], dtype=np.float32),
        "impact_flag": np.asarray([False, True, False]),
    }
    metadata = {
        "actuator_names": names,
        "actuator_ctrlrange": ctrlrange.tolist(),
        "ctrlrange_schema_hash": ordered_schema_hash(
            kind="actuator_ctrlrange",
            payload={"actuator_names": names, "ctrlrange": ctrlrange.tolist()},
        ),
        "physical_signal_semantics": physical_signal_metadata(),
        "physical_capture": _physical_capture(names),
        "event_features_required": True,
    }
    write_split_shard(tmp_path, data, split="train", metadata=metadata)

    dataset = PhysicalDistillDataset(
        tmp_path,
        split="train",
        target_actuator_names=target_names,
        require_event_fields=True,
    )

    np.testing.assert_array_equal(dataset.arrays["muscle_force"], (rows + 20.0)[:, [3, 1]])
    np.testing.assert_array_equal(dataset.arrays["teacher_ctrl_physical"], raw_ctrl[:, [3, 1]])
    np.testing.assert_array_equal(
        dataset.arrays["qfrc_actuator"],
        data["qfrc_actuator"],
    )
    np.testing.assert_array_equal(dataset.actuator_ctrlrange, ctrlrange[[3, 1]])
    assert dataset.arrays["phase_id"].dtype == np.int32
    assert dataset.arrays["impact_flag"].dtype == np.bool_


def test_physical_batch_validator_recomputes_declared_excitation():
    names = ("m0", "m1")
    ctrlrange = np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    raw_ctrl = np.asarray([[0.0, 1.0]], dtype=np.float32)
    data = {
        "teacher_ctrl_physical": raw_ctrl,
        "muscle_excitation": physical_ctrl_to_effective_muscle_excitation(
            raw_ctrl,
            channel_contract=_muscle_contract(names),
        ),
        "muscle_activation": np.ones((1, 2), dtype=np.float32),
        "muscle_force": np.ones((1, 2), dtype=np.float32),
        "muscle_tendon_length": np.ones((1, 2), dtype=np.float32),
        "muscle_tendon_velocity": np.ones((1, 2), dtype=np.float32),
        "actuator_power": np.ones((1, 2), dtype=np.float32),
        "qfrc_actuator": np.ones((1, 3), dtype=np.float32),
    }

    _validate_physical_batch(
        data,
        actuator_ctrlrange=ctrlrange,
        channel_contract=_muscle_contract(names),
    )
    data["muscle_excitation"] = np.zeros((1, 2), dtype=np.float32)
    with pytest.raises(AssertionError, match=r"clip\(raw data.ctrl,0,1\)"):
        _validate_physical_batch(
            data,
            actuator_ctrlrange=ctrlrange,
            channel_contract=_muscle_contract(names),
        )


def test_physical_dataset_rejects_nonunit_or_unbound_activation(tmp_path):
    names = ["muscle"]
    ctrlrange = np.asarray([[0.0, 1.0]], dtype=np.float64)
    metadata = {
        "actuator_names": names,
        "actuator_ctrlrange": ctrlrange.tolist(),
        "ctrlrange_schema_hash": ordered_schema_hash(
            kind="actuator_ctrlrange",
            payload={"actuator_names": names, "ctrlrange": ctrlrange.tolist()},
        ),
        "physical_signal_semantics": physical_signal_metadata(),
        "physical_capture": _physical_capture(names),
    }
    data = {
        "student_obs": np.zeros((2, 1), dtype=np.float32),
        "teacher_action": np.zeros((2, 1), dtype=np.float32),
        "teacher_ctrl_physical": np.full((2, 1), 0.5, dtype=np.float32),
        "muscle_excitation": np.full((2, 1), 0.5, dtype=np.float32),
        "muscle_activation": np.asarray([[0.2], [1.01]], dtype=np.float32),
        "muscle_force": np.ones((2, 1), dtype=np.float32),
        "muscle_tendon_length": np.ones((2, 1), dtype=np.float32),
        "muscle_tendon_velocity": np.zeros((2, 1), dtype=np.float32),
        "actuator_power": np.zeros((2, 1), dtype=np.float32),
        "qfrc_actuator": np.zeros((2, 1), dtype=np.float32),
    }
    write_split_shard(tmp_path / "outside", data, split="train", metadata=metadata)
    with pytest.raises(ValueError, match=r"outside \[0,1\]"):
        PhysicalDistillDataset(tmp_path / "outside", split="train")

    data["muscle_activation"] = np.asarray([[0.2], [0.3]], dtype=np.float32)
    invalid_metadata = {
        **metadata,
        "physical_capture": {
            **metadata["physical_capture"],
            "activation_valid_mask": [False],
        },
    }
    write_split_shard(
        tmp_path / "invalid-mask", data, split="train", metadata=invalid_metadata
    )
    with pytest.raises(ValueError, match="without a scalar MuJoCo activation"):
        PhysicalDistillDataset(tmp_path / "invalid-mask", split="train")


def test_physical_dataset_rejects_legacy_v1_capture_even_with_unit_arrays(tmp_path):
    names = ["muscle"]
    data = {
        "student_obs": np.zeros((2, 1), dtype=np.float32),
        "teacher_action": np.zeros((2, 1), dtype=np.float32),
        "teacher_ctrl_physical": np.full((2, 1), 0.5, dtype=np.float32),
        "muscle_excitation": np.full((2, 1), 0.5, dtype=np.float32),
        "muscle_activation": np.full((2, 1), 0.5, dtype=np.float32),
        "muscle_force": np.ones((2, 1), dtype=np.float32),
        "muscle_tendon_length": np.ones((2, 1), dtype=np.float32),
        "muscle_tendon_velocity": np.zeros((2, 1), dtype=np.float32),
        "actuator_power": np.zeros((2, 1), dtype=np.float32),
        "qfrc_actuator": np.zeros((2, 1), dtype=np.float32),
    }
    metadata = {
        "actuator_names": names,
        "actuator_ctrlrange": [[0.0, 1.0]],
        "physical_signal_semantics": physical_signal_metadata(),
        "physical_capture": {
            "schema_version": "physical_capture_spec_v1",
            "actuator_names": names,
            "activation_valid_mask": [True],
        },
    }
    write_split_shard(tmp_path, data, split="train", metadata=metadata)
    with pytest.raises(ValueError, match="legacy datasets are rejected"):
        PhysicalDistillDataset(tmp_path, split="train")

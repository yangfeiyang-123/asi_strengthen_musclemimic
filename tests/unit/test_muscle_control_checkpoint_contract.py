from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from omegaconf import OmegaConf

from musclemimic.algorithms.common.env_utils import (
    MUSCLE_CONTROL_CONTRACT_SCHEMA_VERSION,
    _bind_muscle_control_contract,
)
from musclemimic.distill.action_schema import actuator_schema_hash
from musclemimic.distill.physical import PHYSICAL_SIGNAL_SCHEMA_VERSION
from musclemimic.runner.checkpointing import (
    validate_checkpoint_muscle_control_contract,
    write_manifest,
)


def _contract(names: tuple[str, ...] = ("m0", "m1")) -> dict:
    channel_layout = {
        "policy_action_dim": len(names),
        "ordered_policy_actuator_ids": list(range(len(names))),
        "ordered_policy_actuator_names": list(names),
        "ordered_muscle_policy_positions": list(range(len(names))),
        "policy_control_type": "DefaultControl",
        "policy_control_apply_mode": "direct",
        "actuator_names": list(names),
        "actuator_ids": list(range(len(names))),
        "actuator_actnum": [1] * len(names),
        "actuator_actadr": list(range(len(names))),
        "actuator_ctrlrange": [[0.0, 1.0]] * len(names),
        "actuator_ctrllimited": [True] * len(names),
        "model_na": len(names),
    }
    return {
        "schema_version": MUSCLE_CONTROL_CONTRACT_SCHEMA_VERSION,
        "physical_signal_schema_version": PHYSICAL_SIGNAL_SCHEMA_VERSION,
        "policy_action_semantics": "normalized_symmetric_action_-1_1",
        "policy_action_dim": len(names),
        "ordered_policy_actuator_ids": list(range(len(names))),
        "ordered_policy_actuator_names": list(names),
        "ordered_muscle_policy_positions": list(range(len(names))),
        "policy_control_type": "DefaultControl",
        "policy_control_apply_mode": "direct",
        "runtime_muscle_ctrlrange": [0.0, 1.0],
        "ordered_muscle_ctrllimited": [True] * len(names),
        "effective_excitation_semantics": "clip(raw_data_ctrl,0,1)",
        "activation_index_semantics": "model.actuator_actadr",
        "muscle_actuator_count": len(names),
        "ordered_muscle_actuator_names": list(names),
        "ordered_muscle_actuator_schema_hash": actuator_schema_hash(names),
        "ordered_activation_addresses": list(range(len(names))),
        "muscle_channel_layout_sha256": hashlib.sha256(
            json.dumps(
                channel_layout,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
    }


def _model(
    ctrlrange: list[list[float]],
    *,
    ctrllimited: list[bool] | None = None,
) -> SimpleNamespace:
    if ctrllimited is None:
        ctrllimited = [True] * len(ctrlrange)
    return SimpleNamespace(
        actuator_dyntype=np.asarray(
            [int(mujoco.mjtDyn.mjDYN_MUSCLE)] * len(ctrlrange)
        ),
        actuator_ctrlrange=np.asarray(ctrlrange, dtype=np.float64),
        actuator_ctrllimited=np.asarray(ctrllimited, dtype=np.bool_),
        actuator_actnum=np.ones((len(ctrlrange),), dtype=np.int32),
        actuator_actadr=np.arange(len(ctrlrange), dtype=np.int32),
        na=len(ctrlrange),
        nu=len(ctrlrange),
    )


class DefaultControl:
    def __init__(self, apply_mode: str = "direct") -> None:
        self._apply_mode = apply_mode


def _env(
    model: SimpleNamespace,
    *,
    action_indices: list[int] | None = None,
    control_apply_mode: str = "direct",
) -> SimpleNamespace:
    if action_indices is None:
        action_indices = list(range(int(model.nu)))
    return SimpleNamespace(
        _model=model,
        _action_indices=np.asarray(action_indices, dtype=np.int32),
        _control_func=DefaultControl(control_apply_mode),
        info=SimpleNamespace(
            action_space=SimpleNamespace(shape=(len(action_indices),))
        ),
    )


def _write_orbax_leaf(
    checkpoint: Path,
    contract: dict | None,
    *,
    update_number: int,
) -> None:
    checkpoint.mkdir(parents=True)
    (checkpoint / "_CHECKPOINT_METADATA").write_text("{}", encoding="utf-8")
    (checkpoint / "train_state").mkdir()
    (checkpoint / "config").mkdir()
    experiment = {}
    if contract is not None:
        experiment["muscle_control_contract"] = contract
    (checkpoint / "config" / "metadata").write_text(
        json.dumps({"experiment": experiment}),
        encoding="utf-8",
    )
    (checkpoint / "metadata").mkdir()
    (checkpoint / "metadata" / "metadata").write_text(
        json.dumps({"update_number": update_number}),
        encoding="utf-8",
    )


def _write_run_manifest(run_dir: Path, contract: dict) -> None:
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "muscle_control_contract": contract,
                "experiment_config": {"muscle_control_contract": contract},
            }
        ),
        encoding="utf-8",
    )


def test_runtime_binding_persists_v2_and_rejects_signed_muscles(monkeypatch):
    monkeypatch.setattr(
        mujoco,
        "mj_id2name",
        lambda _model, _object_type, actuator_id: f"m{int(actuator_id)}",
    )
    monkeypatch.setattr(
        mujoco,
        "mj_name2id",
        lambda _model, _object_type, name: int(str(name)[1:]),
    )
    config = OmegaConf.create({})
    _bind_muscle_control_contract(
        config,
        _env(_model([[0.0, 1.0], [0.0, 1.0]])),
    )
    assert OmegaConf.to_container(
        config.muscle_control_contract,
        resolve=True,
    ) == _contract()

    with pytest.raises(ValueError, match=r"exactly \[0,1\]"):
        _bind_muscle_control_contract(
            OmegaConf.create({}),
            _env(_model([[-1.0, 1.0]])),
        )

    with pytest.raises(ValueError, match="direct mode"):
        _bind_muscle_control_contract(
            OmegaConf.create({}),
            _env(
                _model([[0.0, 1.0]]),
                control_apply_mode="incremental",
            ),
        )

    with pytest.raises(ValueError, match="enforce"):
        _bind_muscle_control_contract(
            OmegaConf.create({}),
            _env(
                _model(
                    [[0.0, 1.0]],
                    ctrllimited=[False],
                )
            ),
        )


def test_runtime_binding_tracks_actual_policy_action_order(monkeypatch):
    monkeypatch.setattr(
        mujoco,
        "mj_id2name",
        lambda _model, _object_type, actuator_id: f"m{int(actuator_id)}",
    )
    monkeypatch.setattr(
        mujoco,
        "mj_name2id",
        lambda _model, _object_type, name: int(str(name)[1:]),
    )
    model = _model([[0.0, 1.0], [0.0, 1.0]])
    canonical = OmegaConf.create({})
    _bind_muscle_control_contract(
        canonical,
        _env(model, action_indices=[0, 1]),
    )
    swapped = OmegaConf.create({})
    _bind_muscle_control_contract(
        swapped,
        _env(model, action_indices=[1, 0]),
    )
    canonical_payload = OmegaConf.to_container(
        canonical.muscle_control_contract,
        resolve=True,
    )
    swapped_payload = OmegaConf.to_container(
        swapped.muscle_control_contract,
        resolve=True,
    )
    assert canonical_payload != swapped_payload
    assert swapped_payload["ordered_muscle_actuator_names"] == ["m1", "m0"]

    with pytest.raises(ValueError, match="differs"):
        _bind_muscle_control_contract(
            canonical,
            _env(model, action_indices=[1, 0]),
        )


def test_checkpoint_requires_matching_top_level_and_nested_contract(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoint_10"
    contract = _contract()
    _write_orbax_leaf(checkpoint, contract, update_number=10)
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "muscle_control_contract": contract,
                "experiment_config": {
                    "muscle_control_contract": contract,
                },
            }
        ),
        encoding="utf-8",
    )

    validate_checkpoint_muscle_control_contract(checkpoint, contract)

    manifest_path.write_text(
        json.dumps({"experiment_config": {"muscle_control_contract": contract}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pre-v2"):
        validate_checkpoint_muscle_control_contract(checkpoint, contract)


def test_checkpoint_rejects_semantic_drift(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoint_2"
    checkpoint.mkdir(parents=True)
    legacy = {
        **_contract(),
        "effective_excitation_semantics": "ctrlrange_affine",
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "muscle_control_contract": legacy,
                "experiment_config": {
                    "muscle_control_contract": legacy,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="differs"):
        validate_checkpoint_muscle_control_contract(checkpoint, _contract())


def test_checkpoint_rejects_ordered_muscle_channel_drift(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoint_3"
    checkpoint.mkdir(parents=True)
    saved = _contract(("m0", "m1"))
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "muscle_control_contract": saved,
                "experiment_config": {"muscle_control_contract": saved},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="differs"):
        validate_checkpoint_muscle_control_contract(
            checkpoint,
            _contract(("m1", "m0")),
        )


def test_checkpoint_alias_keeps_run_manifest_and_binds_external_leaf(tmp_path):
    contract = _contract()
    external_leaf = tmp_path / "exports" / "orbax_leaf"
    _write_orbax_leaf(external_leaf, contract, update_number=11)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_run_manifest(run_dir, contract)
    alias = run_dir / "checkpoint_11"
    alias.symlink_to(external_leaf, target_is_directory=True)

    validate_checkpoint_muscle_control_contract(alias, contract)


def test_checkpoint_run_dir_resolves_and_validates_actual_latest_leaf(tmp_path):
    contract = _contract()
    run_dir = tmp_path / "run"
    _write_orbax_leaf(run_dir / "checkpoint_1", contract, update_number=1)
    latest = run_dir / "checkpoint_2"
    _write_orbax_leaf(latest, contract, update_number=2)
    _write_run_manifest(run_dir, contract)

    validate_checkpoint_muscle_control_contract(run_dir, contract)

    (latest / "config" / "metadata").write_text(
        json.dumps({"experiment": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="leaf config has no"):
        validate_checkpoint_muscle_control_contract(run_dir, contract)


def test_checkpoint_rejects_old_leaf_moved_under_v2_manifest(tmp_path):
    contract = _contract()
    run_dir = tmp_path / "run"
    legacy_leaf = run_dir / "checkpoint_4"
    _write_orbax_leaf(legacy_leaf, None, update_number=4)
    _write_run_manifest(run_dir, contract)

    with pytest.raises(ValueError, match="leaf config has no"):
        validate_checkpoint_muscle_control_contract(legacy_leaf, contract)


def test_checkpoint_rejects_leaf_contract_and_metadata_alias_drift(tmp_path):
    contract = _contract()
    run_dir = tmp_path / "run"
    drifted = {
        **contract,
        "effective_excitation_semantics": "ctrlrange_affine",
    }
    leaf = run_dir / "checkpoint_7"
    _write_orbax_leaf(leaf, drifted, update_number=7)
    _write_run_manifest(run_dir, contract)

    with pytest.raises(ValueError, match="leaf muscle control contract differs"):
        validate_checkpoint_muscle_control_contract(leaf, contract)

    (leaf / "config" / "metadata").write_text(
        json.dumps({"experiment": {"muscle_control_contract": contract}}),
        encoding="utf-8",
    )
    (leaf / "metadata" / "metadata").write_text(
        json.dumps({"update_number": 6}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="alias update does not match"):
        validate_checkpoint_muscle_control_contract(leaf, contract)


def test_checkpoint_leaf_json_rejects_duplicate_contract_key(tmp_path):
    contract = _contract()
    run_dir = tmp_path / "run"
    leaf = run_dir / "checkpoint_5"
    _write_orbax_leaf(leaf, contract, update_number=5)
    _write_run_manifest(run_dir, contract)
    duplicate = json.dumps(contract)
    (leaf / "config" / "metadata").write_text(
        '{"experiment":{"muscle_control_contract":'
        f"{duplicate},"
        '"muscle_control_contract":'
        f"{duplicate}"
        "}}",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        validate_checkpoint_muscle_control_contract(leaf, contract)


def test_write_manifest_revalidates_existing_muscle_contract(tmp_path):
    contract = _contract()
    config = OmegaConf.create({"muscle_control_contract": contract})
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_run_manifest(run_dir, contract)
    manifest_path = run_dir / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["config_hash"] = "same-hash"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    write_manifest(run_dir, config, "same-hash")

    payload.pop("muscle_control_contract")
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="consistent muscle control contract"):
        write_manifest(run_dir, config, "same-hash")

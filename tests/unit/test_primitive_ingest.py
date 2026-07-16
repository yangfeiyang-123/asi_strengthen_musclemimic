from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from musclemimic.synergy.fit import (
    SynergyFitConfig,
    fit_synergy_dataset,
    load_synergy_split,
)
from musclemimic.synergy.primitive_ingest import (
    ingest_primitive_catalog,
    save_compiled_model_artifact,
)
from musclemimic.synergy.primitive_manifest import (
    save_primitive_source_manifest_from_splits,
)


def _write_model(path: Path) -> Path:
    path.write_text(
        """
<mujoco model="primitive-ingest-fixture">
  <option timestep="0.01"/>
  <worldbody>
    <body name="body_a" pos="0 0 0">
      <joint name="joint_a" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.02 0.10" mass="1"/>
    </body>
    <body name="body_b" pos="0.2 0 0">
      <joint name="joint_b" type="hinge" axis="1 0 0"/>
      <geom type="capsule" size="0.02 0.10" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="muscle_a" joint="joint_a" ctrlrange="-1 1"/>
    <motor name="muscle_b" joint="joint_b" ctrlrange="0 2"/>
  </actuator>
</mujoco>
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_phase(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "primitive_phase_schema_v1",
                "task_id": "P01_fixture",
                "phases": [
                    {"id": 0, "name": "prepare", "definition": "Prepare."},
                    {"id": 1, "name": "execute", "definition": "Execute."},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_raw(path: Path, *, offset: float = 0.0, kind: str = "physical") -> Path:
    phase_id = np.asarray([0, 0, 1, 1], dtype=np.int32)
    names = np.asarray(["muscle_a", "muscle_b"])
    if kind == "kinematic":
        np.savez(path, qpos=np.zeros((4, 2)), qvel=np.zeros((4, 2)), phase_id=phase_id, actuator_names=names)
        return path
    ctrl = np.asarray(
        [
            [-0.8 + offset, 0.2],
            [-0.3 + offset, 0.6],
            [0.2 + offset, 1.2],
            [0.7 + offset, 1.8],
        ],
        dtype=np.float32,
    )
    if kind == "normalized":
        np.savez(path, teacher_action=ctrl, phase_id=phase_id, actuator_names=names)
    else:
        np.savez(
            path,
            teacher_ctrl_physical=ctrl,
            phase_id=phase_id,
            actuator_names=names,
            success=np.asarray(True),
        )
    return path


def _write_fixture(root: Path, *, raw_kind: str = "physical") -> Path:
    root.mkdir(parents=True)
    _write_model(root / "model.xml")
    _write_phase(root / "phase.json")
    (root / "controller").mkdir()
    (root / "controller" / "params").write_bytes(b"immutable-controller")
    (root / "groups.json").write_text(
        json.dumps({"regions": {"all": ["muscle_a", "muscle_b"]}}),
        encoding="utf-8",
    )
    trial_rows = []
    for index, (split, offset) in enumerate((("train", 0.00), ("train", 0.05), ("val", 0.10))):
        raw_path = _write_raw(
            root / f"raw-{index}.npz",
            offset=offset,
            kind=raw_kind if index == 0 else "physical",
        )
        trial_rows.append(
            {
                "trial_id": f"fixture-{split}-{index}",
                "split": split,
                "motion_path": f"primitive/P01/independent-{index}.npz",
                "raw_npz_path": raw_path.name,
                "success": True,
                "quality_weight": 1.0 - 0.1 * index,
            }
        )
    catalog = {
        "schema_version": "primitive_synergy_catalog_v1",
        "catalog_id": "fixture",
        "target_skill_id": "ChinaJump",
        "model_xml_path": "model.xml",
        "expected_action_dim": 2,
        "regional_grouping_path": "groups.json",
        "tasks": [
            {
                "task_id": "P01_fixture",
                "display_name": "Fixture primitive",
                "enabled": True,
                "controller_artifact": "controller",
                "phase_schema_path": "phase.json",
                "trials": trial_rows,
            }
        ],
    }
    catalog_path = root / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    return catalog_path


def test_ingest_builds_manifest_and_fit_ready_physical_dataset(tmp_path):
    catalog = _write_fixture(tmp_path / "source")
    output = tmp_path / "dataset"
    result = ingest_primitive_catalog(catalog, output)

    assert result.idempotent is False
    assert result.metadata_path.is_file()
    assert result.source_checkpoints_path.is_file()
    assert result.dataset_qc_path.is_file()
    assert result.regional_grouping_path.is_file()
    assert len(list(output.glob("train_*.npz"))) == 2
    assert len(list(output.glob("val_*.npz"))) == 1

    train = load_synergy_split(output, split="train")
    validation = load_synergy_split(output, split="val")
    excitation = train.signal("physical_excitation")
    assert excitation.values.shape == (8, 2)
    assert np.all((excitation.values >= 0.0) & (excitation.values <= 1.0))
    assert not (set(train.motion_ids.tolist()) & set(validation.motion_ids.tolist()))

    checkpoints = json.loads(result.source_checkpoints_path.read_text(encoding="utf-8"))
    fit_config = SynergyFitConfig(
        ranks=(1,),
        seeds=(0, 1),
        max_iter=100,
        split_half_repeats=1,
        bootstrap_repeats=1,
        cross_trial_max_trials=2,
        min_val_global_vaf=0.0,
        min_val_local_vaf_quantile=0.0,
        min_initialization_similarity=0.0,
        min_split_half_similarity=0.0,
        min_bootstrap_similarity=0.0,
        min_cross_trial_similarity=0.0,
    )
    source = save_primitive_source_manifest_from_splits(
        tmp_path / "source-manifest",
        train_source=output,
        validation_source=output,
        target_skill_id="ChinaJump",
        excluded_target_motion_paths=["ChinaJump/target-motion"],
        source_checkpoint_fingerprints=checkpoints,
        fit_config=fit_config,
    )
    report = fit_synergy_dataset(
        output,
        output,
        output_dir=tmp_path / "fit",
        signal_kinds=("physical_excitation",),
        mode="both",
        grouping_json=result.regional_grouping_path,
        primitive_source_manifest=source.path,
        config=fit_config,
    )
    assert "physical_excitation_unit" in report["preferred_decoder_artifacts"]

    reused = ingest_primitive_catalog(catalog, output)
    assert reused.idempotent is True
    assert reused.build_fingerprint == result.build_fingerprint


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("kinematic", "kinematics-only"),
        ("normalized", "only normalized action"),
    ],
)
def test_ingest_rejects_nonphysical_action_evidence(tmp_path, kind, message):
    catalog = _write_fixture(tmp_path / "source", raw_kind=kind)

    with pytest.raises(ValueError, match=message):
        ingest_primitive_catalog(catalog, tmp_path / "dataset")


def test_ingest_recomputes_and_rejects_tampered_excitation(tmp_path):
    root = tmp_path / "source"
    catalog = _write_fixture(root)
    raw_path = root / "raw-0.npz"
    with np.load(raw_path, allow_pickle=False) as source:
        payload = {name: np.asarray(source[name]) for name in source.files}
    payload["muscle_excitation"] = np.zeros((4, 2), dtype=np.float32)
    np.savez(raw_path, **payload)

    with pytest.raises(ValueError, match="differs from the model-derived"):
        ingest_primitive_catalog(catalog, tmp_path / "dataset")


def test_existing_output_is_reused_only_for_identical_current_inputs(tmp_path):
    root = tmp_path / "source"
    catalog = _write_fixture(root)
    output = tmp_path / "dataset"
    ingest_primitive_catalog(catalog, output)

    raw_path = root / "raw-0.npz"
    with np.load(raw_path, allow_pickle=False) as source:
        payload = {name: np.asarray(source[name]) for name in source.files}
    payload["teacher_ctrl_physical"] = payload["teacher_ctrl_physical"].copy()
    payload["teacher_ctrl_physical"][0, 0] += 0.01
    np.savez(raw_path, **payload)

    with pytest.raises(FileExistsError, match="different current inputs"):
        ingest_primitive_catalog(catalog, output)


def test_compiled_mjb_round_trip_preserves_runtime_model_and_ingests(tmp_path):
    root = tmp_path / "source"
    catalog_path = _write_fixture(root)
    runtime_model = mujoco.MjModel.from_xml_path(str(root / "model.xml"))
    expected_hash = hashlib.sha256(runtime_model.__getstate__()).hexdigest()
    mjb_path = save_compiled_model_artifact(runtime_model, root / "runtime_354.mjb")
    restored = mujoco.MjModel.from_binary_path(str(mjb_path))

    assert restored.nu == runtime_model.nu == 2
    assert hashlib.sha256(restored.__getstate__()).hexdigest() == expected_hash
    assert save_compiled_model_artifact(runtime_model, mjb_path) == mjb_path

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    # The legacy field name remains stable, but its production value is the
    # exact compiled runtime model artifact.
    catalog["model_xml_path"] = mjb_path.name
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    result = ingest_primitive_catalog(catalog_path, tmp_path / "dataset")
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert metadata["model_artifact_format"] == ".mjb"
    assert metadata["model_hash"] == expected_hash

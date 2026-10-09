from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from musclemimic.distill.motion_identity import stable_motion_uid
from musclemimic.synergy import primitive_ingest
from musclemimic.synergy.fit import (
    SynergyFitConfig,
    fit_synergy_dataset,
    load_synergy_split,
)
from musclemimic.synergy.primitive_catalog import canonical_json_sha256
from musclemimic.synergy.primitive_ingest import (
    ingest_primitive_catalog,
    save_compiled_model_artifact,
)
from musclemimic.synergy.primitive_manifest import (
    save_primitive_source_manifest_from_splits,
)
from musclemimic.synergy.primitive_producer import (
    PhysicalOptimizerConfig,
    ensure_optimizer_artifact,
    resolve_axial_rotation_signal_contract,
    resolve_foot_floor_contact_contract,
)

_ROLLOUT_SCHEMA = "primitive_physical_rollout_manifest_v7"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_model(path: Path) -> Path:
    path.write_text(
        """
<mujoco model="primitive-ingest-fixture">
  <option timestep="0.01"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="body_a" pos="0 0 0">
      <joint name="joint_a" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.02 0.10" mass="1"/>
      <geom name="l_talus" type="sphere" size="0.005" mass="0.001"/>
      <geom name="l_foot" type="sphere" size="0.005" mass="0.001"/>
      <geom name="l_foot_col1" type="sphere" size="0.005" mass="0.001"/>
      <geom name="l_foot_col3" type="sphere" size="0.005" mass="0.001"/>
      <geom name="l_foot_col4" type="sphere" size="0.005" mass="0.001"/>
      <geom name="l_bofoot" type="sphere" size="0.005" mass="0.001"/>
      <geom name="l_bofoot_col1" type="sphere" size="0.005" mass="0.001"/>
      <geom name="l_bofoot_col2" type="sphere" size="0.005" mass="0.001"/>
    </body>
    <body name="body_b" pos="0.2 0 0">
      <joint name="joint_b" type="hinge" axis="1 0 0"/>
      <geom type="capsule" size="0.02 0.10" mass="1"/>
      <geom name="r_talus" type="sphere" size="0.005" mass="0.001"/>
      <geom name="r_foot" type="sphere" size="0.005" mass="0.001"/>
      <geom name="r_foot_col1" type="sphere" size="0.005" mass="0.001"/>
      <geom name="r_foot_col3" type="sphere" size="0.005" mass="0.001"/>
      <geom name="r_foot_col4" type="sphere" size="0.005" mass="0.001"/>
      <geom name="r_bofoot" type="sphere" size="0.005" mass="0.001"/>
      <geom name="r_bofoot_col1" type="sphere" size="0.005" mass="0.001"/>
      <geom name="r_bofoot_col2" type="sphere" size="0.005" mass="0.001"/>
    </body>
  </worldbody>
  <actuator>
    <general name="muscle_a" joint="joint_a" ctrllimited="true" ctrlrange="0 1"
      dyntype="muscle" gaintype="muscle" biastype="muscle"
      dynprm="0.01 0.04 0 0 0 0 0 0 0 0"
      gainprm="0.75 1.05 -1 400 0.5 1.6 1.5 1.3 1.2 0"
      biasprm="0.75 1.05 -1 400 0.5 1.6 1.5 1.3 1.2 0"
      lengthrange="0.1 1.0"/>
    <general name="muscle_b" joint="joint_b" ctrllimited="true" ctrlrange="0 1"
      dyntype="muscle" gaintype="muscle" biastype="muscle"
      dynprm="0.01 0.04 0 0 0 0 0 0 0 0"
      gainprm="0.75 1.05 -1 400 0.5 1.6 1.5 1.3 1.2 0"
      biasprm="0.75 1.05 -1 400 0.5 1.6 1.5 1.3 1.2 0"
      lengthrange="0.1 1.0"/>
  </actuator>
</mujoco>
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_phase(path: Path) -> Path:
    _write_json(
        path,
        {
            "schema_version": "primitive_phase_schema_v1",
            "task_id": "P01_fixture",
            "phases": [
                {
                    "id": 0,
                    "name": "natural_stance",
                    "definition": "Maintain bilateral support for the complete trial.",
                }
            ],
        },
    )
    return path


def _write_raw(path: Path, *, offset: float = 0.0, kind: str = "physical") -> Path:
    phase_id = np.asarray([0, 0, 0, 0], dtype=np.int32)
    names = np.asarray(["muscle_a", "muscle_b"])
    if kind == "kinematic":
        np.savez(path, qpos=np.zeros((4, 2)), qvel=np.zeros((4, 2)), phase_id=phase_id, actuator_names=names)
        return path
    ctrl = np.asarray(
        [
            [0.10 + offset, 0.20],
            [0.25 + offset, 0.35],
            [0.50 + offset, 0.60],
            [0.75 + offset, 0.85],
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


def _runtime_model_binding(model: mujoco.MjModel) -> dict[str, object]:
    state = model.__getstate__()
    assert isinstance(state, bytes)
    model_hash = hashlib.sha256(state).hexdigest()
    resolved_config = {"fixture": "primitive-ingest", "num_envs": 1}
    resolved_model_params = {
        "env_params": {"num_envs": 1, "env_name": "FixtureMyoModel"},
        "task_factory": {"name": "FixtureFactory", "params": {}},
    }
    return {
        "schema_version": "chinajump_taskfactory_runtime_model_binding_v1",
        "production_eligible": True,
        "config_name": "fixture/primitive_ingest",
        "hydra_overrides": [],
        "resolved_config_sha256": canonical_json_sha256(resolved_config),
        "resolved_model_params_sha256": canonical_json_sha256(resolved_model_params),
        "resolved_model_params": resolved_model_params,
        "declared_production_num_envs": 1,
        "construction_num_envs": 1,
        "num_env_model_hash_invariant": True,
        "construction_model_hash": model_hash,
        "declared_num_env_model_hash": model_hash,
    }


def _phase_runs(phase_id: np.ndarray) -> list[dict[str, int]]:
    runs: list[dict[str, int]] = []
    start = 0
    for index in range(1, int(phase_id.size) + 1):
        if index == int(phase_id.size) or int(phase_id[index]) != int(phase_id[start]):
            runs.append(
                {
                    "phase_id": int(phase_id[start]),
                    "start": start,
                    "end": index,
                    "length": index - start,
                }
            )
            start = index
    return runs


def _p12_recovery_report(position: np.ndarray, *, evidence_kind: str) -> dict[str, object]:
    phase_id = np.asarray([0, 0, 1, 1] + [2] * 10, dtype=np.int32)
    contact = np.ones(phase_id.shape, dtype=np.bool_)
    velocity = np.zeros(phase_id.shape, dtype=np.float64)
    landing_terminal = float(position[phase_id == 0][-1])
    restore_terminal = float(position[phase_id == 1][-1])
    ready_min = float(np.min(position[phase_id == 2]))
    restore_rise = restore_terminal - landing_terminal
    ready_margin = ready_min - landing_terminal
    return {
        "passed": True,
        "production_eligible": True,
        "supported": True,
        "semantic_gate": "required_fail_closed",
        "task_id": "P12_post_landing_recovery",
        "task_family": "P12",
        "phase_runs": _phase_runs(phase_id),
        "gates": {
            "bilateral_contact_entire_primitive": True,
            "post_impact_initial_vertical_speed": True,
            "posture_restore_com_rise": True,
            "ready_hold_min_frames": True,
            "ready_hold_bilateral": True,
            "ready_hold_vertical_speed": True,
            "ready_hold_recovered_height": True,
        },
        "metrics": {
            "landing_stabilization_terminal_com_height": landing_terminal,
            "posture_restore_terminal_com_height": restore_terminal,
            "posture_restore_com_rise": restore_rise,
            "ready_hold_min_com_height": ready_min,
            "ready_hold_min_height_above_landing_baseline": ready_margin,
        },
        "evidence": {
            "vertical_position": position.tolist(),
            "vertical_velocity": velocity.tolist(),
            "left_foot_floor_contact": contact.tolist(),
            "right_foot_floor_contact": contact.tolist(),
        },
        "evidence_kind": evidence_kind,
        "thresholds": {
            "min_com_vertical_excursion": 0.03,
            "min_ready_hold_frames": 10,
            "max_post_impact_com_vertical_speed": 0.2,
            "max_ready_hold_com_vertical_speed": 0.15,
        },
        "vertical_signal": "root_subtree_com_z/delta_over_transition_duration",
        "axial_signal_contract": None,
    }


def _p12_qc_config() -> dict[str, object]:
    return {
        "min_com_vertical_excursion": 0.03,
        "min_ready_hold_frames": 10,
        "max_post_impact_com_vertical_speed": 0.2,
        "max_ready_hold_com_vertical_speed": 0.15,
    }


def _write_rollout_manifest(
    trial_dir: Path,
    *,
    model: mujoco.MjModel,
    controller_dir: Path,
    controller_manifest: dict[str, object],
    phase_schema: dict[str, object],
    trial_id: str,
    task_id: str,
    target_skill_id: str,
    motion_path: str,
    seed: int,
) -> Path:
    raw_path = trial_dir / "primitive_trial.npz"
    with np.load(raw_path, allow_pickle=False) as raw:
        phase_id = np.asarray(raw["phase_id"], dtype=np.int32)
        if "teacher_ctrl_physical" in raw:
            applied_ctrl = np.asarray(raw["teacher_ctrl_physical"], dtype=np.float32)
        elif "applied_ctrl" in raw:
            applied_ctrl = np.asarray(raw["applied_ctrl"], dtype=np.float32)
        elif "teacher_action" in raw:
            applied_ctrl = np.asarray(raw["teacher_action"], dtype=np.float32)
        else:
            applied_ctrl = np.zeros((phase_id.size, int(model.nu)), dtype=np.float32)
    transition_count = int(phase_id.size)
    phase_runs = _phase_runs(phase_id)
    qc_path = trial_dir / "rollout_qc.npz"
    np.savez(
        qc_path,
        applied_ctrl=applied_ctrl,
        phase_id=phase_id,
        transition_substeps=np.ones((transition_count,), dtype=np.int32),
        fixture_evidence=np.ones((transition_count,), dtype=np.float64),
    )

    contact_contract = resolve_foot_floor_contact_contract(
        model,
        allow_unavailable=True,
    ).as_dict()
    axial_contract = resolve_axial_rotation_signal_contract(
        model,
        allow_unavailable=True,
    ).as_dict()
    target_semantic_report = {
        "passed": True,
        "production_eligible": True,
        "supported": True,
        "semantic_gate": "required_fail_closed",
        "task_id": task_id,
        "task_family": "P01",
        "phase_runs": phase_runs,
        "gates": {
            "aligned_contact_phase_arrays": True,
            "bilateral_contact_continuous": True,
            "each_phase_min_transitions": True,
            "phase_order_exact": True,
            "supported_task_semantics": True,
        },
        "metrics": {
            "airborne_count": 0,
            "bilateral_contact_count": transition_count,
            "transition_count": transition_count,
        },
        "evidence": {
            "left_foot_floor_contact": [True] * transition_count,
            "right_foot_floor_contact": [True] * transition_count,
            "left_foot_floor_normal_force": [1.0] * transition_count,
            "right_foot_floor_normal_force": [1.0] * transition_count,
            "vertical_position": [0.0] * transition_count,
            "vertical_velocity": [0.0] * transition_count,
            "axial_position": [],
            "axial_velocity": [],
            "axial_root_yaw": [],
            "axial_root_xy": [],
            "axial_initial_position": None,
            "axial_initial_root_yaw": None,
            "axial_initial_root_xy": [],
        },
        "evidence_kind": "target_mj_forward_exact_contact",
        "thresholds": {"min_contact_normal_force": 1.0e-6},
        "vertical_signal": "unscored_explicit_toy_fixture",
        "axial_signal_contract": None,
    }
    site_proxy_semantic_report = {
        **target_semantic_report,
        "evidence_kind": "target_site_xpos_hysteresis_proxy",
    }
    actual_semantic_report = {
        **target_semantic_report,
        "evidence_kind": "actual_rollout_exact_contact",
    }
    target_contact_semantics = {
        "passed": True,
        "task_id": task_id,
        "task_family": "P01",
        "gate_basis": "exact_mj_forward_contact",
        "proxy_fallback_allowed": False,
        "target_exact_contact_incomplete": False,
        "exact": target_semantic_report,
        "site_proxy": site_proxy_semantic_report,
        "contact_contract": contact_contract,
        "initial_state_contact": {
            "exact_left": True,
            "exact_right": True,
            "exact_left_normal_force": 1.0,
            "exact_right_normal_force": 1.0,
            "site_proxy_left": True,
            "site_proxy_right": True,
        },
        "vertical_signal": "unscored_explicit_toy_fixture",
    }
    qc = {
        "passed": True,
        "expected_transition_count": transition_count,
        "recorded_transition_count": transition_count,
        "gates": {
            "complete_trajectory": True,
            "forward_replay_activation": True,
            "forward_replay_position": True,
            "forward_replay_velocity": True,
            "target_contact_phase_semantics": True,
            "actual_contact_phase_semantics": True,
        },
        "metrics": {
            "forward_replay_activation_abs_max": 0.0,
            "forward_replay_position_abs_max": 0.0,
            "forward_replay_velocity_abs_max": 0.0,
            "position_abs_max": 0.0,
            "position_rmse": 0.0,
            "velocity_abs_max": 0.0,
            "velocity_rmse": 0.0,
            "saturation_fraction": 0.0,
        },
        "target_contact_semantics": target_contact_semantics,
        "actual_contact_semantics": actual_semantic_report,
        "contact_contract": contact_contract,
        "axial_rotation_signal_contract": axial_contract,
        "axial_rotation_direction_evidence": {
            "required": False,
            "target": None,
            "actual": None,
        },
        "replay_axial_rotation_evidence": {
            "position": [],
            "velocity": [],
            "root_yaw": [],
            "root_xy": [],
            "initial_position": 0.0,
            "initial_root_yaw": 0.0,
            "initial_root_xy": [],
        },
        "replay_contact_evidence": {
            "left_foot_floor_contact": [True] * transition_count,
            "right_foot_floor_contact": [True] * transition_count,
            "left_foot_floor_normal_force": [1.0] * transition_count,
            "right_foot_floor_normal_force": [1.0] * transition_count,
        },
        "initialization_evidence": {
            "contract": controller_manifest["initial_state_contract"],
            "gate_required": True,
            "fixture": True,
        },
        "success_does_not_depend_on_shooting_proposal_residual": True,
    }
    optimizer_manifest_path = controller_dir / "optimizer_manifest.json"
    model_artifact_path = controller_dir / str(controller_manifest["model_artifact_filename"])
    model_state = model.__getstate__()
    assert isinstance(model_state, bytes)
    model_hash = hashlib.sha256(model_state).hexdigest()
    manifest: dict[str, object] = {
        "schema_version": _ROLLOUT_SCHEMA,
        "trial_id": trial_id,
        "task_id": task_id,
        "target_skill_id": target_skill_id,
        "contains_target_skill_rollout": False,
        "source_motion_path": motion_path,
        "source_motion_uid": stable_motion_uid(motion_path),
        "source_artifact_sha256": hashlib.sha256(
            f"fixture-source:{motion_path}".encode()
        ).hexdigest(),
        "source_frequency_hz": 100.0,
        "source_frame_interval": {
            "start_frame": 0,
            "end_frame_exclusive": transition_count + 1,
            "source_total_frames": transition_count + 1,
        },
        "phase_schema_fingerprint": canonical_json_sha256(phase_schema),
        "contact_contract": contact_contract,
        "axial_rotation_signal_contract": axial_contract,
        "target_contact_semantics": target_contact_semantics,
        "seed": seed,
        "optimizer_fingerprint": controller_manifest["optimizer_fingerprint"],
        "controller_source_kind": controller_manifest["source_kind"],
        "runtime_model_binding": controller_manifest["runtime_model_binding"],
        "runtime_model_provenance": {
            "schema_version": "primitive_runtime_model_provenance_v1",
            "source_kind": "taskfactory_constructed",
            "verified_runtime_artifact": None,
            "model_hash": model_hash,
            "config_name": "fixture/primitive_ingest",
            "hydra_overrides": [],
        },
        "policy_rollout_binding": controller_manifest["policy_controller_binding"],
        "optimizer_artifact": {
            "path": str(controller_dir),
            "manifest_sha256": _file_sha256(optimizer_manifest_path),
            "model_artifact_sha256": _file_sha256(model_artifact_path),
        },
        "initialization_contract": {
            "contract": controller_manifest["initial_state_contract"],
            "solver_or_seed_alone_can_mark_success": False,
        },
        "transition_contract": {
            "control": "data.ctrl immediately before s_t_to_s_t_plus_1",
            "activation": "data.act[model.actuator_actadr] after s_t_to_s_t_plus_1",
            "foot_floor_contact": "post-transition exact contact fixture",
            "mujoco_state_spec": "mjSTATE_INTEGRATION",
            "physics_substeps": [1] * transition_count,
        },
        "requested_transition_count": transition_count,
        "recorded_transition_count": transition_count,
        "qc_config": {
            "max_position_abs": 1.0,
            "max_position_rmse": 1.0,
            "max_velocity_abs": 1.0,
            "max_velocity_rmse": 1.0,
            "max_saturation_fraction": 1.0,
        },
        "qc": qc,
        "artifacts": {
            "primitive_trial": {
                "filename": raw_path.name,
                "sha256": _file_sha256(raw_path),
            },
            "rollout_qc": {
                "filename": qc_path.name,
                "sha256": _file_sha256(qc_path),
            },
        },
        "status": "success",
        "success": True,
        "production_eligible": True,
    }
    manifest["rollout_fingerprint"] = canonical_json_sha256(manifest)
    manifest_path = trial_dir / "rollout_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def _write_fixture(root: Path, *, raw_kind: str = "physical") -> Path:
    root.mkdir(parents=True)
    model_xml = _write_model(root / "model.xml")
    phase_path = _write_phase(root / "phase.json")
    phase_schema = json.loads(phase_path.read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    controller_dir, controller_manifest = ensure_optimizer_artifact(
        model,
        controller_store=root / "controllers",
        config=PhysicalOptimizerConfig(
            horizon=1,
            population=2,
            elite_count=1,
            iterations=1,
        ),
        runtime_model_binding=_runtime_model_binding(model),
    )
    (root / "groups.json").write_text(
        json.dumps({"regions": {"all": ["muscle_a", "muscle_b"]}}),
        encoding="utf-8",
    )
    trial_rows = []
    for index, (split, offset) in enumerate((("train", 0.00), ("train", 0.05), ("val", 0.10))):
        trial_dir = root / f"trial-{index}"
        trial_dir.mkdir()
        raw_path = _write_raw(
            trial_dir / "primitive_trial.npz",
            offset=offset,
            kind=raw_kind if index == 0 else "physical",
        )
        trial_id = f"fixture-{split}-{index}"
        motion_path = f"primitive/P01/independent-{index}.npz"
        _write_rollout_manifest(
            trial_dir,
            model=model,
            controller_dir=controller_dir,
            controller_manifest=controller_manifest,
            phase_schema=phase_schema,
            trial_id=trial_id,
            task_id="P01_fixture",
            target_skill_id="ChinaJump",
            motion_path=motion_path,
            seed=index,
        )
        trial_rows.append(
            {
                "trial_id": trial_id,
                "split": split,
                "motion_path": motion_path,
                "raw_npz_path": raw_path.relative_to(root).as_posix(),
                "success": True,
                "quality_weight": 1.0 - 0.1 * index,
            }
        )
    catalog = {
        "schema_version": "primitive_synergy_catalog_v1",
        "catalog_id": "fixture",
        "target_skill_id": "ChinaJump",
        "model_xml_path": (controller_dir / "runtime_model.mjb").relative_to(root).as_posix(),
        "expected_action_dim": 2,
        "regional_grouping_path": "groups.json",
        "tasks": [
            {
                "task_id": "P01_fixture",
                "display_name": "Fixture primitive",
                "enabled": True,
                "controller_artifact": controller_dir.relative_to(root).as_posix(),
                "phase_schema_path": "phase.json",
                "trials": trial_rows,
            }
        ],
    }
    catalog_path = root / "catalog.json"
    _write_json(catalog_path, catalog)
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
    attestation = primitive_ingest.validate_ingested_primitive_dataset(output)
    assert attestation["primitive_semantic_contracts"] == {}
    assert len(attestation["attestation_fingerprint"]) == 64

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


def test_p12_ingest_recomputes_recovery_height_from_hash_bound_qc(tmp_path):
    phase_id = np.asarray([0, 0, 1, 1] + [2] * 10, dtype=np.int32)
    recovered = np.asarray([1.0, 1.0, 1.02, 1.04] + [1.04] * 10, dtype=np.float64)
    contact = np.ones(phase_id.shape, dtype=np.bool_)
    velocity = np.zeros(phase_id.shape, dtype=np.float64)
    evidence_arrays = {
        "target_com_vertical_velocity": velocity,
        "target_left_foot_floor_contact": contact,
        "target_right_foot_floor_contact": contact,
        "target_site_proxy_left_foot_contact": contact,
        "target_site_proxy_right_foot_contact": contact,
        "actual_com_vertical_velocity": velocity,
        "actual_left_foot_floor_contact": contact,
        "actual_right_foot_floor_contact": contact,
    }
    qc_path = tmp_path / "p12_qc.npz"
    np.savez(
        qc_path,
        phase_id=phase_id,
        target_com_vertical_position=recovered,
        actual_com_vertical_position=recovered,
        **evidence_arrays,
    )
    target = _p12_recovery_report(recovered, evidence_kind="target_site_xpos_hysteresis_proxy")
    actual = _p12_recovery_report(recovered, evidence_kind="actual_rollout_exact_contact")

    primitive_ingest._validate_p12_recovery_semantics(
        qc_path,
        phase_id=phase_id,
        qc_config=_p12_qc_config(),
        target_report=target,
        actual_report=actual,
    )
    exact_target = _p12_recovery_report(recovered, evidence_kind="target_mj_forward_exact_contact")
    primitive_ingest._validate_p12_recovery_semantics(
        qc_path,
        phase_id=phase_id,
        qc_config=_p12_qc_config(),
        target_report=exact_target,
        actual_report=actual,
    )

    legacy_target = json.loads(json.dumps(target))
    legacy_target["gates"].pop("posture_restore_com_rise")
    with pytest.raises(ValueError, match="required recovery-height contract"):
        primitive_ingest._validate_p12_recovery_semantics(
            qc_path,
            phase_id=phase_id,
            qc_config=_p12_qc_config(),
            target_report=legacy_target,
            actual_report=actual,
        )

    flat_qc_path = tmp_path / "p12_flat_qc.npz"
    np.savez(
        flat_qc_path,
        phase_id=phase_id,
        target_com_vertical_position=np.ones_like(recovered),
        actual_com_vertical_position=recovered,
        **evidence_arrays,
    )
    with pytest.raises(ValueError, match="differs from the QC NPZ"):
        primitive_ingest._validate_p12_recovery_semantics(
            flat_qc_path,
            phase_id=phase_id,
            qc_config=_p12_qc_config(),
            target_report=target,
            actual_report=actual,
        )


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


def test_ingest_rejects_missing_producer_rollout_manifest(tmp_path):
    root = tmp_path / "source"
    catalog = _write_fixture(root)
    (root / "trial-0" / "rollout_manifest.json").unlink()

    with pytest.raises(FileNotFoundError, match="producer-owned sibling rollout manifest"):
        ingest_primitive_catalog(catalog, tmp_path / "dataset")


def test_ingest_rejects_rollout_manifest_with_tampered_self_fingerprint(tmp_path):
    root = tmp_path / "source"
    catalog = _write_fixture(root)
    manifest_path = root / "trial-0" / "rollout_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["seed"] = 999
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="rollout manifest self-fingerprint mismatch"):
        ingest_primitive_catalog(catalog, tmp_path / "dataset")


def test_ingest_rejects_refingerprinted_manifest_with_wrong_raw_artifact_hash(tmp_path):
    root = tmp_path / "source"
    catalog = _write_fixture(root)
    manifest_path = root / "trial-0" / "rollout_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("rollout_fingerprint")
    manifest["artifacts"]["primitive_trial"]["sha256"] = "0" * 64
    manifest["rollout_fingerprint"] = canonical_json_sha256(manifest)
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="manifest raw artifact differs"):
        ingest_primitive_catalog(catalog, tmp_path / "dataset")


def test_rollout_manifest_identity_participates_in_idempotent_reuse(tmp_path):
    root = tmp_path / "source"
    catalog = _write_fixture(root)
    output = tmp_path / "dataset"
    first = ingest_primitive_catalog(catalog, output)

    manifest_path = root / "trial-0" / "rollout_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("rollout_fingerprint")
    manifest["seed"] = 1001
    manifest["rollout_fingerprint"] = canonical_json_sha256(manifest)
    _write_json(manifest_path, manifest)

    with pytest.raises(FileExistsError, match="different current inputs"):
        ingest_primitive_catalog(catalog, output)
    assert first.output_dir == output.resolve()


def test_ingest_recomputes_and_rejects_tampered_excitation(tmp_path):
    root = tmp_path / "source"
    catalog = _write_fixture(root)
    raw_path = root / "trial-0" / "primitive_trial.npz"
    with np.load(raw_path, allow_pickle=False) as source:
        payload = {name: np.asarray(source[name]) for name in source.files}
    payload["muscle_excitation"] = np.zeros((4, 2), dtype=np.float32)
    np.savez(raw_path, **payload)

    with pytest.raises(ValueError, match=r"differs from clip\(raw data.ctrl,0,1\)"):
        ingest_primitive_catalog(catalog, tmp_path / "dataset")


def test_existing_output_is_reused_only_for_identical_current_inputs(tmp_path):
    root = tmp_path / "source"
    catalog = _write_fixture(root)
    output = tmp_path / "dataset"
    ingest_primitive_catalog(catalog, output)

    raw_path = root / "trial-0" / "primitive_trial.npz"
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

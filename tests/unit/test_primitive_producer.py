from __future__ import annotations

import hashlib
import json
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

import musclemimic.synergy.primitive_producer as primitive_producer
from musclemimic.synergy.canonical_control import _array_hash, load_canonical_control_artifact
from musclemimic.synergy.primitive_catalog import canonical_json_sha256, load_primitive_phase_schema
from musclemimic.synergy.primitive_producer import (
    CanonicalTonicControlPlanner,
    ComputedMuscleCEMPlanner,
    PhysicalOptimizerConfig,
    RolloutQCConfig,
    ScriptedPhysicalControlPlanner,
    audit_target_contact_semantics,
    build_synthetic_motion_target,
    dense_actuator_moment,
    ensure_optimizer_artifact,
    evaluate_task_contact_semantics,
    execute_physical_rollout,
    load_full_action_policy_controls,
    load_retargeted_motion_target,
    load_transition_phase_plan,
    load_verified_runtime_artifact,
    produce_primitive_trial,
    resolve_chinajump_runtime_model,
    resolve_foot_floor_contact_contract,
)


def test_canonical_tonic_planner_is_target_independent_and_constant():
    model = _muscle_model()
    reference = np.asarray([0.37, 0.61], dtype=np.float64)
    planner = CanonicalTonicControlPlanner(model, reference)
    target = build_synthetic_motion_target(
        model,
        applied_ctrl=np.tile(reference, (3, 1)),
        phase_id=np.zeros(3, dtype=np.int32),
        transition_substeps=np.full(3, 5, dtype=np.int32),
    )
    planner.reset(7)
    rollout = execute_physical_rollout(model, target, planner)
    assert rollout.initialization.contract == "canonical_tonic_activation_control_v1"
    np.testing.assert_array_equal(rollout.initialization.initial_ctrl, reference)
    np.testing.assert_array_equal(rollout.initialization.initial_activation, reference)
    np.testing.assert_array_equal(rollout.applied_ctrl, np.tile(reference, (3, 1)))


def test_producer_rejects_canonical_binding_with_explicit_planner(tmp_path):
    model = _muscle_model()
    reference = np.asarray([0.37, 0.61], dtype=np.float64)
    target = build_synthetic_motion_target(
        model,
        applied_ctrl=np.tile(reference, (2, 1)),
        phase_id=np.zeros(2, dtype=np.int32),
        transition_substeps=np.full(2, 5, dtype=np.int32),
    )
    with pytest.raises(ValueError, match="explicit planner"):
        produce_primitive_trial(
            model,
            target,
            output_dir=tmp_path / "trial",
            controller_store=tmp_path / "controllers",
            trial_id="bad-binding",
            optimizer_config=_optimizer_config(),
            qc_config=_qc_config(),
            seed=0,
            planner=CanonicalTonicControlPlanner(model, reference),
            canonical_control_binding={},
        )


def test_canonical_optimizer_reuse_rejects_missing_embedded_copy(tmp_path):
    model = _muscle_model()
    names = primitive_producer.complete_actuator_names(model)
    ctrlrange = np.asarray(model.actuator_ctrlrange, dtype=np.float64)
    control = np.asarray([0.37, 0.61], dtype=np.float64)
    payload = {
        "schema_version": "primitive_canonical_tonic_control_v1",
        "task_id": "P01_natural_stance",
        "catalog_fingerprint": "1" * 64,
        "controller_fingerprint": "2" * 64,
        "model_hash": hashlib.sha256(model.__getstate__()).hexdigest(),
        "actuator_schema_hash": primitive_producer.actuator_schema_hash(names),
        "ctrlrange_schema_hash": primitive_producer.ordered_schema_hash(
            kind="actuator_ctrlrange", payload={"actuator_names": list(names), "ctrlrange": ctrlrange.tolist()}
        ),
        "action_dim": int(model.nu),
        "aggregation": "coordinate_mean_float64_train_only_v1",
        "train_trials": [
            {
                "trial_id": "train",
                "split": "train",
                "motion_uid": 7,
                "source_motion_path": "primitive/train",
                "source_sha256": "3" * 64,
                "source_frame_interval": {"start_frame": 0, "end_frame_exclusive": 3, "source_total_frames": 3},
                "rollout_manifest_sha256": "4" * 64,
                "rollout_qc_sha256": "5" * 64,
                "initial_ctrl_sha256": "6" * 64,
            }
        ],
        "control": control.tolist(),
        "control_sha256": _array_hash(control),
    }
    fingerprint = canonical_json_sha256(payload)
    payload["artifact_fingerprint"] = fingerprint
    artifact = tmp_path / "prior" / fingerprint
    artifact.mkdir(parents=True)
    (artifact / "canonical_control.json").write_text(json.dumps(payload), encoding="utf-8")
    binding = load_canonical_control_artifact(artifact, expected_width=int(model.nu))
    controller, _ = ensure_optimizer_artifact(
        model,
        controller_store=tmp_path / "controllers",
        config=_optimizer_config(),
        canonical_control_binding=binding,
    )
    (controller / "canonical_control.json").unlink()
    with pytest.raises(FileNotFoundError):
        ensure_optimizer_artifact(
            model,
            controller_store=tmp_path / "controllers",
            config=_optimizer_config(),
            canonical_control_binding=binding,
        )


def _muscle_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(
        """
<mujoco model="primitive-producer-fixture">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="body_a">
      <joint name="joint_a" type="hinge" damping="0.1"/>
      <geom type="capsule" size="0.02 0.1" mass="1"/>
    </body>
    <body name="body_b" pos="0.2 0 0">
      <joint name="joint_b" type="hinge" axis="0 1 0" damping="0.1"/>
      <geom type="capsule" size="0.02 0.1" mass="1"/>
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
"""
    )


def _named_contact_model(*, left_height: float = 0.0) -> mujoco.MjModel:
    left_z = float(left_height)
    return mujoco.MjModel.from_xml_string(
        f"""
<mujoco model="primitive-contact-fixture">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 .1"/>
    <body name="Full Body" pos="0 0 .02">
      <freejoint name="root"/>
      <geom name="l_talus" type="sphere" size=".001" pos="-.1 0 {left_z}" mass="0" contype="0" conaffinity="0"/>
      <geom name="l_foot" type="sphere" size=".001" pos="-.1 0 {left_z}" mass="0" contype="0" conaffinity="0"/>
      <geom name="l_foot_col1" type="sphere" size=".001" pos="-.1 0 {left_z}" mass="0" contype="0" conaffinity="0"/>
      <geom name="l_foot_col3" type="sphere" size=".001" pos="-.1 0 {left_z}" mass="0" contype="0" conaffinity="0"/>
      <geom name="l_foot_col4" type="sphere" size=".025" pos="-.1 0 {left_z}" mass="1"/>
      <geom name="l_bofoot" type="sphere" size=".001" pos="-.1 0 {left_z}" mass="0" contype="0" conaffinity="0"/>
      <geom name="l_bofoot_col1" type="sphere" size=".001" pos="-.1 0 {left_z}" mass="0" contype="0" conaffinity="0"/>
      <geom name="l_bofoot_col2" type="sphere" size=".001" pos="-.1 0 {left_z}" mass="0" contype="0" conaffinity="0"/>
      <geom name="r_talus" type="sphere" size=".001" pos=".1 0 0" mass="0" contype="0" conaffinity="0"/>
      <geom name="r_foot" type="sphere" size=".001" pos=".1 0 0" mass="0" contype="0" conaffinity="0"/>
      <geom name="r_foot_col1" type="sphere" size=".001" pos=".1 0 0" mass="0" contype="0" conaffinity="0"/>
      <geom name="r_foot_col3" type="sphere" size=".001" pos=".1 0 0" mass="0" contype="0" conaffinity="0"/>
      <geom name="r_foot_col4" type="sphere" size=".025" pos=".1 0 0" mass="1"/>
      <geom name="r_bofoot" type="sphere" size=".001" pos=".1 0 0" mass="0" contype="0" conaffinity="0"/>
      <geom name="r_bofoot_col1" type="sphere" size=".001" pos=".1 0 0" mass="0" contype="0" conaffinity="0"/>
      <geom name="r_bofoot_col2" type="sphere" size=".001" pos=".1 0 0" mass="0" contype="0" conaffinity="0"/>
      <site name="left_ankle_mimic" pos="-.1 0 {left_z}"/>
      <site name="left_toes_mimic" pos="-.1 0 {left_z}"/>
      <site name="right_ankle_mimic" pos=".1 0 0"/>
      <site name="right_toes_mimic" pos=".1 0 0"/>
      <body pos="0 0 .1">
        <joint name="joint_a" type="hinge" damping="0.1"/>
        <geom type="capsule" size=".02 .1" mass=".1" contype="0" conaffinity="0"/>
      </body>
      <body pos="0 0 .1">
        <joint name="joint_b" type="hinge" axis="0 1 0" damping="0.1"/>
        <geom type="capsule" size=".02 .1" mass=".1" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <general name="muscle_a" joint="joint_a" ctrllimited="true" ctrlrange="0 1"
      dyntype="muscle" gaintype="muscle" biastype="muscle"
      dynprm="0.01 0.04 0 0 0 0 0 0 0 0"
      gainprm="0.75 1.05 -1 400 0.5 1.6 1.5 1.3 1.2 0"
      biasprm="0.75 1.05 -1 400 0.5 1.6 1.5 1.3 1.2 0" lengthrange="0.1 1.0"/>
    <general name="muscle_b" joint="joint_b" ctrllimited="true" ctrlrange="0 1"
      dyntype="muscle" gaintype="muscle" biastype="muscle"
      dynprm="0.01 0.04 0 0 0 0 0 0 0 0"
      gainprm="0.75 1.05 -1 400 0.5 1.6 1.5 1.3 1.2 0"
      biasprm="0.75 1.05 -1 400 0.5 1.6 1.5 1.3 1.2 0" lengthrange="0.1 1.0"/>
  </actuator>
</mujoco>
"""
    )


def _p08_signal_model() -> mujoco.MjModel:
    axial_bodies = "\n".join(
        f"""
      <body name="axial_body_{index}" pos="0 0 {0.05 + 0.03 * index}">
        <joint name="{name}" type="hinge" axis="0 0 1"/>
        <geom type="sphere" size=".01" mass=".05" contype="0" conaffinity="0"/>
      </body>"""
        for index, name in enumerate(("axial_rotation", "Abs_r3", "L4_L5_AR", "L3_L4_AR", "L2_L3_AR", "L1_L2_AR"))
    )
    return mujoco.MjModel.from_xml_string(
        f"""
<mujoco model="p08-signal-fixture">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="Full Body" pos="0 0 .2">
      <freejoint name="root"/>
      <geom type="sphere" size=".02" mass=".1" contype="0" conaffinity="0"/>
      {axial_bodies}
    </body>
  </worldbody>
</mujoco>
"""
    )


def _optimizer_config() -> PhysicalOptimizerConfig:
    return PhysicalOptimizerConfig(horizon=1, population=2, elite_count=1, iterations=1)


def _runtime_354_model() -> mujoco.MjModel:
    actuators = "\n".join(
        f"""
    <general name="muscle_{index:03d}" joint="joint" ctrllimited="true" ctrlrange="0 1"
      dyntype="muscle" gaintype="muscle" biastype="muscle"
      dynprm="0.01 0.04 0 0 0 0 0 0 0 0"
      gainprm="0.75 1.05 -1 400 0.5 1.6 1.5 1.3 1.2 0"
      biasprm="0.75 1.05 -1 400 0.5 1.6 1.5 1.3 1.2 0"
      lengthrange="0.1 1.0"/>"""
        for index in range(354)
    )
    return mujoco.MjModel.from_xml_string(
        f"""
<mujoco model="verified-runtime-fixture">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="body">
      <joint name="joint" type="hinge" damping="0.1"/>
      <geom type="capsule" size="0.02 0.1" mass="1"/>
    </body>
  </worldbody>
  <actuator>{actuators}
  </actuator>
</mujoco>
"""
    )


def _verified_runtime_fixture(tmp_path: Path):
    model = _runtime_354_model()
    model_hash = hashlib.sha256(model.__getstate__()).hexdigest()
    config_name = "config_specific_task/stage1_body/fixture_chinajump"
    hydra_overrides = ("fixture.value=1",)
    resolved_config = {
        "experiment": {
            "training_action": "ChinaJump",
            "env_params": {"disable_fingers": True, "num_envs": 8},
        },
        "fixture": {"value": 1},
    }
    resolved_model_params = {
        "env_params": {"disable_fingers": True, "num_envs": 1},
        "task_factory": {"name": "FixtureFactory", "params": {"task": "ChinaJump"}},
    }
    binding = {
        "schema_version": "chinajump_taskfactory_runtime_model_binding_v1",
        "production_eligible": True,
        "config_name": config_name,
        "hydra_overrides": list(hydra_overrides),
        "resolved_config_sha256": canonical_json_sha256(resolved_config),
        "resolved_model_params_sha256": canonical_json_sha256(resolved_model_params),
        "resolved_model_params": resolved_model_params,
        "declared_production_num_envs": 8,
        "construction_num_envs": 1,
        "num_env_model_hash_invariant": True,
        "construction_model_hash": model_hash,
        "declared_num_env_model_hash": model_hash,
    }
    artifact, _ = ensure_optimizer_artifact(
        model,
        controller_store=tmp_path / "controllers",
        config=_optimizer_config(),
        runtime_model_binding=binding,
    )
    composed = SimpleNamespace(
        name=config_name,
        hydra_overrides=hydra_overrides,
        resolved_config=resolved_config,
        declared_production_num_envs=8,
    )
    return artifact, model_hash, config_name, hydra_overrides, composed, resolved_model_params


def _patch_runtime_composition(monkeypatch, *, composed, resolved_model_params) -> None:
    monkeypatch.setattr(
        primitive_producer,
        "_compose_chinajump_runtime_config",
        lambda *, config_name, hydra_overrides: composed,
    )
    monkeypatch.setattr(
        primitive_producer,
        "_resolved_runtime_model_params",
        lambda composed, *, construction_num_envs: resolved_model_params,
    )


def _qc_config(*, strict: bool = False) -> RolloutQCConfig:
    limit = 1.0e-14 if strict else 1.0e-8
    return RolloutQCConfig(
        max_position_rmse=limit,
        max_velocity_rmse=limit,
        max_position_abs=limit,
        max_velocity_abs=limit,
        max_saturation_fraction=0.99,
        replay_position_atol=1.0e-12,
        replay_velocity_atol=1.0e-12,
        replay_activation_atol=1.0e-12,
    )


def _semantic_report(
    task_id: str,
    phase_id: list[int],
    left: list[bool],
    right: list[bool],
    position: list[float] | None = None,
    velocity: list[float] | None = None,
) -> dict[str, object]:
    config = _qc_config()
    count = len(phase_id)
    left_array = np.asarray(left, dtype=np.bool_)
    right_array = np.asarray(right, dtype=np.bool_)
    return evaluate_task_contact_semantics(
        task_id=task_id,
        phase_id=np.asarray(phase_id, dtype=np.int32),
        left_contact=left_array,
        right_contact=right_array,
        left_normal_force=left_array.astype(np.float64) * config.min_contact_normal_force,
        right_normal_force=right_array.astype(np.float64) * config.min_contact_normal_force,
        vertical_position=np.asarray(np.zeros(count) if position is None else position),
        vertical_velocity=np.asarray(np.zeros(count) if velocity is None else velocity),
        config=config,
        evidence_kind="unit_test",
    )


def test_target_mj_forward_p01_rejects_one_foot_even_when_site_proxy_is_bilateral():
    model = _named_contact_model(left_height=0.12)
    contract = resolve_foot_floor_contact_contract(model)
    controls = np.zeros((12, int(model.nu)), dtype=np.float64)
    target = build_synthetic_motion_target(
        model,
        applied_ctrl=controls,
        phase_id=np.zeros((12,), dtype=np.int32),
        transition_substeps=np.ones((12,), dtype=np.int32),
        task_id="P01_natural_stance",
    )
    audit = audit_target_contact_semantics(
        model,
        target,
        config=_qc_config(),
        contact_contract=contract,
    )

    assert not np.any(audit.left_foot_floor_contact)
    assert np.any(audit.right_foot_floor_contact)
    assert np.all(audit.site_proxy_left_foot_contact)
    assert np.all(audit.site_proxy_right_foot_contact)
    assert audit.semantics["proxy_fallback_allowed"] is False
    assert audit.semantics["passed"] is False
    assert audit.semantics["gate_basis"] == "none_failed_closed"


def test_p06_contact_state_machine_requires_monotonic_two_foot_takeoff():
    phases = [0, 0, 1, 1, 2, 2, 2, 3, 3]
    left = [True, True, True, True, True, True, False, False, False]
    right = [True, True, True, True, True, False, False, False, False]
    passed = _semantic_report("P06_low_two_foot_jump", phases, left, right)
    assert passed["passed"] is True

    left[7] = True
    recontact = _semantic_report("P06_low_two_foot_jump", phases, left, right)
    assert recontact["passed"] is False
    assert recontact["gates"]["both_single_air_monotonic_no_recontact"] is False


def test_p05_p07_p11_and_p12_task_state_machines_accept_only_semantic_sequences():
    p05_phases = [0] * 5 + [1] * 3 + [2] * 3 + [3] * 3
    p05_position = [1.0] * 5 + [0.98, 0.95, 0.92] + [0.90, 0.88, 0.91] + [0.93, 0.97, 1.01]
    p05_velocity = [0.0] * 5 + [-0.2] * 3 + [-0.1, -0.05, 0.1] + [0.2] * 3
    assert (
        _semantic_report(
            "P05_countermovement",
            p05_phases,
            [True] * len(p05_phases),
            [True] * len(p05_phases),
            p05_position,
            p05_velocity,
        )["passed"]
        is True
    )

    p07_phases = [0] * 2 + [1] * 2 + [2] * 2 + [3] * 10
    p07_left = [False, False, True, True] + [True] * 12
    p07_right = [False, False, False, False] + [True] * 12
    assert _semantic_report("P07_two_foot_landing", p07_phases, p07_left, p07_right)["passed"] is True

    p11_phases = [0] * 5 + [1] * 3 + [2] * 4 + [3] * 2 + [4] * 12
    p11_left = [True] * 10 + [True, False] + [False] * 2 + [True] * 12
    p11_right = [True] * 10 + [False, False] + [False] * 2 + [False] + [True] * 11
    p11_position = [1.0] * 5 + [0.95, 0.90, 0.92] + [0.96, 1.0, 1.05, 1.1] + [1.1] * 2 + [1.0] * 12
    p11_velocity = [0.0] * 5 + [-0.1, -0.05, 0.1] + [0.2] * 4 + [0.0] * 2 + [-0.1] + [0.0] * 11
    assert (
        _semantic_report(
            "P11_decomposed_jump",
            p11_phases,
            p11_left,
            p11_right,
            p11_position,
            p11_velocity,
        )["passed"]
        is True
    )

    p12_phases = [0] * 2 + [1] * 2 + [2] * 10
    p12_position = [1.0] * 2 + [1.02, 1.04] + [1.04] * 10
    p12 = _semantic_report(
        "P12_post_landing_recovery",
        p12_phases,
        [True] * len(p12_phases),
        [True] * len(p12_phases),
        p12_position,
    )
    assert p12["passed"] is True
    assert p12["gates"]["posture_restore_com_rise"] is True
    assert p12["gates"]["ready_hold_recovered_height"] is True
    assert p12["metrics"]["posture_restore_com_rise"] == pytest.approx(0.04)
    assert p12["metrics"]["ready_hold_min_height_above_landing_baseline"] == pytest.approx(0.04)

    unsupported = _semantic_report("P10_unknown_production_task", [0, 0], [True, True], [True, True])
    assert unsupported["passed"] is False
    assert unsupported["gates"]["supported_task_semantics"] is False


def test_p12_recovery_height_contract_rejects_flat_short_and_collapsing_tails():
    phases = [0] * 2 + [1] * 2 + [2] * 10
    contact = [True] * len(phases)

    flat = _semantic_report(
        "P12_post_landing_recovery",
        phases,
        contact,
        contact,
        [1.0] * len(phases),
    )
    assert flat["passed"] is False
    assert flat["gates"]["posture_restore_com_rise"] is False
    assert flat["gates"]["ready_hold_recovered_height"] is False

    short_restore = _semantic_report(
        "P12_post_landing_recovery",
        phases,
        contact,
        contact,
        [1.0] * 2 + [1.015, 1.029] + [1.04] * 10,
    )
    assert short_restore["gates"]["posture_restore_com_rise"] is False
    assert short_restore["gates"]["ready_hold_recovered_height"] is True

    collapsing_hold = _semantic_report(
        "P12_post_landing_recovery",
        phases,
        contact,
        contact,
        [1.0] * 2 + [1.02, 1.04] + [1.04] * 5 + [1.029] + [1.04] * 4,
    )
    assert collapsing_hold["gates"]["posture_restore_com_rise"] is True
    assert collapsing_hold["gates"]["ready_hold_recovered_height"] is False

    missing_restore = _semantic_report(
        "P12_post_landing_recovery",
        [0] * 2 + [2] * 10,
        [True] * 12,
        [True] * 12,
        [1.0] * 2 + [1.04] * 10,
    )
    assert missing_restore["passed"] is False
    assert missing_restore["gates"]["posture_restore_com_rise"] is False
    assert missing_restore["gates"]["ready_hold_recovered_height"] is False


def test_p08_named_axial_signal_contract_and_state_machine_are_fail_closed():
    model = _p08_signal_model()
    initial = np.asarray(model.qpos0, dtype=np.float64).copy()
    post = np.tile(initial, (4, 1))
    contract = primitive_producer.resolve_axial_rotation_signal_contract(model)
    for index, value in enumerate((0.01, 0.02, 0.03, 0.04)):
        post[index, np.asarray(contract.qpos_addresses)] = value
    evidence = primitive_producer.reconstruct_axial_rotation_evidence(
        model,
        initial_qpos=initial,
        post_transition_qpos=post,
        transition_substeps=np.full((4,), 5, dtype=np.int32),
    )
    np.testing.assert_allclose(evidence.position, [0.06, 0.12, 0.18, 0.24], rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(evidence.velocity, [6.0, 6.0, 6.0, 6.0], rtol=0.0, atol=1.0e-13)
    assert evidence.contract.as_dict()["proxy_fallback_allowed"] is False

    phases = np.asarray([0] * 5 + [1] * 5 + [2] * 2 + [3] * 5, dtype=np.int32)
    axial_position = np.asarray(
        [0.0] * 5 + [0.03, 0.06, 0.09, 0.12, 0.15] + [0.16, 0.155] + [0.12, 0.09, 0.06, 0.03, 0.0]
    )
    axial_velocity = np.asarray([0.0] * 5 + [0.3] * 5 + [0.05, -0.05] + [-0.3] * 5)
    right = np.ones(phases.shape, dtype=np.bool_)
    left = np.ones(phases.shape, dtype=np.bool_)
    config = _qc_config()
    common = {
        "task_id": "P08_axial_rotation",
        "phase_id": phases,
        "left_contact": left,
        "right_contact": right,
        "left_normal_force": left.astype(np.float64) * config.min_contact_normal_force,
        "right_normal_force": right.astype(np.float64) * config.min_contact_normal_force,
        "vertical_position": np.zeros(phases.shape),
        "vertical_velocity": np.zeros(phases.shape),
        "config": config,
        "evidence_kind": "unit_test_exact_contact",
        "vertical_signal": "auxiliary_root_subtree_com_z_not_scored_for_p08",
        "axial_position": axial_position,
        "axial_velocity": axial_velocity,
        "axial_root_yaw": np.zeros(phases.shape),
        "axial_root_xy": np.zeros((phases.size, 2)),
        "axial_initial_position": 0.0,
        "axial_initial_root_yaw": 0.0,
        "axial_initial_root_xy": np.zeros((2,)),
        "axial_signal_contract": contract,
    }
    passed = evaluate_task_contact_semantics(**common)
    assert passed["passed"] is True
    assert passed["gates"]["phases_0_to_2_fixed_support"] is True

    missing_contract = evaluate_task_contact_semantics(**{**common, "axial_signal_contract": None})
    assert missing_contract["passed"] is False
    assert missing_contract["gates"]["named_axial_signal_contract"] is False

    changed_support = right.copy()
    changed_support[6] = False
    bad_support = evaluate_task_contact_semantics(
        **{
            **common,
            "right_contact": changed_support,
            "right_normal_force": changed_support.astype(np.float64) * config.min_contact_normal_force,
        }
    )
    assert bad_support["passed"] is False
    assert bad_support["gates"]["exact_any_foot_contact_entire_primitive"] is True
    assert bad_support["gates"]["phases_0_to_2_fixed_support"] is False


def test_scripted_physical_rollout_records_actual_ctrl_activation_and_exact_replay(tmp_path):
    model = _muscle_model()
    controls = np.asarray([[0.2, 0.3], [0.4, 0.25], [0.6, 0.2], [0.3, 0.55]])
    target = build_synthetic_motion_target(
        model,
        applied_ctrl=controls,
        phase_id=np.asarray([0, 0, 1, 1], dtype=np.int32),
        transition_substeps=np.full(4, 5, dtype=np.int32),
    )
    result = produce_primitive_trial(
        model,
        target,
        output_dir=tmp_path / "trial",
        controller_store=tmp_path / "controllers",
        trial_id="synthetic-pass",
        optimizer_config=_optimizer_config(),
        qc_config=_qc_config(),
        seed=7,
        planner=ScriptedPhysicalControlPlanner(model, controls),
    )

    assert result.success is True
    assert result.trial_path is not None
    with np.load(result.trial_path, allow_pickle=False) as trial:
        np.testing.assert_array_equal(trial["teacher_ctrl_physical"], controls.astype(np.float32))
        assert bool(trial["success"].item()) is True
        assert np.all(trial["muscle_activation"] >= 0.0)
    with np.load(result.output_dir / "rollout_qc.npz", allow_pickle=False) as qc:
        assert qc["initialization_contract"].item() == "explicit_zero_activation_control_v1"
        np.testing.assert_array_equal(qc["initial_muscle_activation"], np.zeros(int(model.nu)))
        np.testing.assert_array_equal(qc["initial_ctrl"], np.zeros(int(model.nu)))
        np.testing.assert_array_equal(qc["applied_ctrl"], controls)
        np.testing.assert_allclose(qc["actual_qpos"], qc["replay_qpos"], rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(
            qc["muscle_activation"],
            qc["replay_muscle_activation"],
            rtol=0.0,
            atol=1.0e-12,
        )
        for prefix in ("target", "actual", "replay"):
            assert qc[f"{prefix}_left_foot_floor_contact"].shape == (4,)
            assert qc[f"{prefix}_right_foot_floor_contact"].shape == (4,)
            assert qc[f"{prefix}_left_foot_floor_normal_force"].shape == (4,)
            assert qc[f"{prefix}_right_foot_floor_normal_force"].shape == (4,)
            assert qc[f"{prefix}_axial_rotation_position"].shape == (4,)
            assert qc[f"{prefix}_axial_rotation_velocity"].shape == (4,)
            assert qc[f"{prefix}_axial_rotation_root_yaw"].shape == (4,)
            assert qc[f"{prefix}_axial_rotation_root_xy"].shape == (4, 2)
    manifest = json.loads((result.output_dir / "rollout_manifest.json").read_text(encoding="utf-8"))
    assert manifest["success"] is True
    assert manifest["runtime_model_binding"] is None
    assert manifest["production_eligible"] is False
    assert manifest["target_contact_semantics"]["gate_basis"] == "exact_mj_forward_contact"
    assert manifest["qc"]["actual_contact_semantics"]["semantic_gate"] == ("skipped_explicit_toy_fixture")
    assert manifest["initialization_contract"]["contract"] == "explicit_zero_activation_control_v1"
    assert manifest["initialization_contract"]["warmup_transition_count"] == 0
    assert manifest["transition_contract"]["physics_substeps"] == [5, 5, 5, 5]


def test_computed_planner_uses_no_step_forward_seed_with_exact_shadow_evidence():
    model = _muscle_model()
    controls = np.asarray([[0.6, 0.4], [0.5, 0.3], [0.4, 0.2]])
    target = build_synthetic_motion_target(
        model,
        applied_ctrl=controls,
        phase_id=np.zeros((3,), dtype=np.int32),
        transition_substeps=np.full((3,), 5, dtype=np.int32),
    )
    planner = ComputedMuscleCEMPlanner(model, _optimizer_config())
    planner.reset(0)

    rollout = execute_physical_rollout(model, target, planner)

    assert rollout.initialization.contract == "inferred_hidden_muscle_state_no_step_shadow_gated_v1"
    assert rollout.initialization.target_acceleration_method == (
        "(target_qvel[1]-target_qvel[0])/first_transition_duration"
    )
    assert np.all(rollout.initialization.initial_activation >= 0.01)
    assert np.all(rollout.initialization.initial_activation <= 0.99)
    assert np.any(rollout.initialization.initial_activation > 0.0)
    np.testing.assert_array_equal(
        rollout.initialization.initial_activation,
        rollout.initialization.initial_ctrl,
    )
    assert rollout.initialization.solver_status > 0
    assert np.isfinite(rollout.initialization.linearized_acceleration_residual_norm)
    assert rollout.initialization.shadow_qpos.shape == (5, int(model.nq))
    assert rollout.initialization.shadow_qvel.shape == (5, int(model.nv))
    assert rollout.initialization.shadow_left_foot_floor_contact.shape == (5,)
    assert rollout.initialization.shadow_final_integration_state.ndim == 1
    data = mujoco.MjData(model)
    mujoco.mj_setState(
        model,
        data,
        rollout.initial_integration_state,
        mujoco.mjtState.mjSTATE_INTEGRATION,
    )
    np.testing.assert_array_equal(data.act, rollout.initialization.initial_activation)
    np.testing.assert_array_equal(data.ctrl, rollout.initialization.initial_ctrl)


def test_producer_rejects_planner_that_mutates_initial_kinematics():
    model = _muscle_model()
    controls = np.asarray([[0.2, 0.3], [0.4, 0.25]])
    target = build_synthetic_motion_target(
        model,
        applied_ctrl=controls,
        phase_id=np.zeros((2,), dtype=np.int32),
        transition_substeps=np.full((2,), 5, dtype=np.int32),
    )

    class MutatingPlanner(ScriptedPhysicalControlPlanner):
        def initialize(self, data, target, *, contact_contract, min_contact_normal_force):
            initialization = super().initialize(
                data,
                target,
                contact_contract=contact_contract,
                min_contact_normal_force=min_contact_normal_force,
            )
            data.qpos[0] += 0.01
            return initialization

    planner = MutatingPlanner(model, controls)
    planner.reset(0)
    with pytest.raises(ValueError, match="must equal the first target state"):
        execute_physical_rollout(model, target, planner)


def test_rollout_and_replay_record_exact_post_transition_contact_force(tmp_path):
    model = _named_contact_model()
    controls = np.zeros((4, int(model.nu)), dtype=np.float64)
    target = build_synthetic_motion_target(
        model,
        applied_ctrl=controls,
        phase_id=np.zeros((4,), dtype=np.int32),
        transition_substeps=np.ones((4,), dtype=np.int32),
    )
    result = produce_primitive_trial(
        model,
        target,
        output_dir=tmp_path / "contact-trial",
        controller_store=tmp_path / "controllers",
        trial_id="synthetic-contact",
        optimizer_config=_optimizer_config(),
        qc_config=_qc_config(),
        seed=0,
        planner=ScriptedPhysicalControlPlanner(model, controls),
    )

    with np.load(result.output_dir / "rollout_qc.npz", allow_pickle=False) as qc:
        assert np.any(qc["actual_left_foot_floor_contact"])
        assert np.any(qc["actual_right_foot_floor_contact"])
        assert np.any(qc["actual_left_foot_floor_normal_force"] > 0.0)
        assert np.any(qc["actual_right_foot_floor_normal_force"] > 0.0)
        np.testing.assert_array_equal(
            qc["actual_left_foot_floor_contact"],
            qc["replay_left_foot_floor_contact"],
        )
        np.testing.assert_allclose(
            qc["actual_left_foot_floor_normal_force"],
            qc["replay_left_foot_floor_normal_force"],
            rtol=0.0,
            atol=_qc_config().replay_contact_force_atol,
        )
    assert result.qc["gates"]["forward_replay_left_contact_bool"] is True
    assert result.qc["gates"]["forward_replay_left_contact_force"] is True


@pytest.mark.parametrize(
    ("task_id", "phase_id", "vertical_signal", "selected_prefix"),
    [
        (
            "P01_natural_stance",
            np.zeros((8,), dtype=np.int32),
            "root_z/root_freejoint_vz",
            "root",
        ),
        (
            "P05_countermovement",
            np.repeat(np.arange(4, dtype=np.int32), 2),
            "root_subtree_com_z/delta_over_transition_duration",
            "com",
        ),
    ],
)
def test_target_and_actual_semantics_use_the_same_task_vertical_signal(
    tmp_path,
    task_id,
    phase_id,
    vertical_signal,
    selected_prefix,
):
    model = _named_contact_model()
    controls = np.zeros((8, int(model.nu)), dtype=np.float64)
    substeps = np.full((8,), 3, dtype=np.int32)
    target = build_synthetic_motion_target(
        model,
        applied_ctrl=controls,
        phase_id=phase_id,
        transition_substeps=substeps,
        task_id=task_id,
    )
    result = produce_primitive_trial(
        model,
        target,
        output_dir=tmp_path / task_id,
        controller_store=tmp_path / "controllers",
        trial_id=f"{task_id}-vertical-signal",
        optimizer_config=_optimizer_config(),
        qc_config=_qc_config(),
        seed=0,
        planner=ScriptedPhysicalControlPlanner(model, controls),
    )

    actual_semantics = result.qc["actual_contact_semantics"]
    target_semantics = result.qc["target_contact_semantics"]
    assert actual_semantics["vertical_signal"] == vertical_signal
    assert target_semantics["vertical_signal"] == vertical_signal
    assert target_semantics["exact"]["vertical_signal"] == vertical_signal
    assert result.qc["gates"]["target_actual_vertical_signal_consistent"] is True
    with np.load(result.output_dir / "rollout_qc.npz", allow_pickle=False) as qc:
        for prefix in ("root", "com"):
            assert qc[f"actual_{prefix}_vertical_position"].shape == (8,)
            assert qc[f"actual_{prefix}_vertical_velocity"].shape == (8,)
        np.testing.assert_allclose(
            actual_semantics["evidence"]["vertical_position"],
            qc[f"actual_{selected_prefix}_vertical_position"],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            actual_semantics["evidence"]["vertical_velocity"],
            qc[f"actual_{selected_prefix}_vertical_velocity"],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            target_semantics["exact"]["evidence"]["vertical_position"],
            qc[f"target_{selected_prefix}_vertical_position"],
            rtol=0.0,
            atol=0.0,
        )

        root_qpos_z, root_qvel_z, root_body = primitive_producer._root_kinematic_binding(model)
        np.testing.assert_array_equal(
            qc["actual_root_vertical_position"],
            qc["actual_qpos"][:, root_qpos_z],
        )
        np.testing.assert_array_equal(
            qc["actual_root_vertical_velocity"],
            qc["actual_qvel"][:, root_qvel_z],
        )
        data = mujoco.MjData(model)
        mujoco.mj_setState(
            model,
            data,
            qc["initial_integration_state"],
            mujoco.mjtState.mjSTATE_INTEGRATION,
        )
        mujoco.mj_forward(model, data)
        initial_com_z = float(data.subtree_com[root_body, 2])
        expected_com_velocity = np.diff(np.concatenate(([initial_com_z], qc["actual_com_vertical_position"]))) / (
            qc["transition_substeps"] * float(model.opt.timestep)
        )
        np.testing.assert_allclose(
            qc["actual_com_vertical_velocity"],
            expected_com_velocity,
            rtol=0.0,
            atol=1.0e-14,
        )


def test_failed_tracking_is_written_with_success_false(tmp_path):
    model = _muscle_model()
    target_controls = np.full((6, 2), 0.8)
    applied_controls = np.full((6, 2), 0.2)
    target = build_synthetic_motion_target(
        model,
        applied_ctrl=target_controls,
        phase_id=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32),
        transition_substeps=np.full(6, 5, dtype=np.int32),
    )
    result = produce_primitive_trial(
        model,
        target,
        output_dir=tmp_path / "failed",
        controller_store=tmp_path / "controllers",
        trial_id="synthetic-fail",
        optimizer_config=_optimizer_config(),
        qc_config=_qc_config(strict=True),
        seed=0,
        planner=ScriptedPhysicalControlPlanner(model, applied_controls),
    )

    assert result.success is False
    assert result.qc["gates"]["forward_replay_position"] is True
    assert result.qc["gates"]["position_rmse"] is False
    with np.load(result.trial_path, allow_pickle=False) as trial:
        assert bool(trial["success"].item()) is False
    manifest = json.loads((result.output_dir / "rollout_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed_qc"
    assert manifest["success"] is False


def test_optimizer_artifact_is_content_addressed_and_contains_exact_mjb(tmp_path):
    model = _muscle_model()
    first_path, first = ensure_optimizer_artifact(
        model,
        controller_store=tmp_path / "controllers",
        config=_optimizer_config(),
    )
    second_path, second = ensure_optimizer_artifact(
        model,
        controller_store=tmp_path / "controllers",
        config=_optimizer_config(),
    )

    assert first_path == second_path
    assert first["optimizer_fingerprint"] == second["optimizer_fingerprint"] == first_path.name
    assert first["production_eligible"] is False
    restored = mujoco.MjModel.from_binary_path(str(first_path / "runtime_model.mjb"))
    assert restored.__getstate__() == model.__getstate__()


def test_verified_runtime_artifact_reuse_is_cpu_only_and_preserves_identity(tmp_path, monkeypatch):
    artifact, model_hash, config_name, overrides, composed, model_params = _verified_runtime_fixture(tmp_path)
    _patch_runtime_composition(
        monkeypatch,
        composed=composed,
        resolved_model_params=model_params,
    )
    monkeypatch.setattr(
        primitive_producer,
        "build_chinajump_taskfactory_runtime_model",
        lambda **kwargs: pytest.fail(f"TaskFactory construction was attempted: {kwargs}"),
    )

    runtime = resolve_chinajump_runtime_model(
        config_name=config_name,
        hydra_overrides=overrides,
        verified_runtime_artifact=artifact,
    )

    assert hashlib.sha256(runtime.model.__getstate__()).hexdigest() == model_hash
    assert runtime.binding["construction_model_hash"] == model_hash
    assert runtime.provenance["source_kind"] == "verified_runtime_artifact_reuse"
    assert runtime.provenance["verified_runtime_artifact"]["path"] == str(artifact.resolve())
    assert runtime.provenance["verified_runtime_artifact"]["optimizer_fingerprint"] == artifact.name


def test_verified_runtime_artifact_rejects_mjb_tampering(tmp_path, monkeypatch):
    artifact, _, config_name, overrides, composed, model_params = _verified_runtime_fixture(tmp_path)
    _patch_runtime_composition(
        monkeypatch,
        composed=composed,
        resolved_model_params=model_params,
    )
    model_path = artifact / "runtime_model.mjb"
    model_path.write_bytes(model_path.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="runtime MJB hash mismatch"):
        load_verified_runtime_artifact(
            artifact,
            config_name=config_name,
            hydra_overrides=overrides,
        )


def test_verified_runtime_artifact_rejects_current_config_drift(tmp_path, monkeypatch):
    artifact, _, config_name, overrides, composed, model_params = _verified_runtime_fixture(tmp_path)
    drifted = SimpleNamespace(
        name=composed.name,
        hydra_overrides=composed.hydra_overrides,
        resolved_config={**composed.resolved_config, "drift": True},
        declared_production_num_envs=composed.declared_production_num_envs,
    )
    _patch_runtime_composition(
        monkeypatch,
        composed=drifted,
        resolved_model_params=model_params,
    )

    with pytest.raises(ValueError, match="current resolved Hydra config differs"):
        load_verified_runtime_artifact(
            artifact,
            config_name=config_name,
            hydra_overrides=overrides,
        )


@pytest.mark.parametrize(
    ("config_name", "hydra_overrides", "message"),
    [
        ("config_specific_task/stage1_body/other", ("fixture.value=1",), "config_name differs"),
        (
            "config_specific_task/stage1_body/fixture_chinajump",
            ("fixture.value=2",),
            "hydra_overrides differ",
        ),
    ],
)
def test_verified_runtime_artifact_rejects_cli_identity_mismatch(
    tmp_path,
    config_name,
    hydra_overrides,
    message,
):
    artifact, *_ = _verified_runtime_fixture(tmp_path)

    with pytest.raises(ValueError, match=message):
        load_verified_runtime_artifact(
            artifact,
            config_name=config_name,
            hydra_overrides=hydra_overrides,
        )


def test_computed_muscle_cem_smoke_emits_bounded_actual_controls():
    model = _muscle_model()
    target = build_synthetic_motion_target(
        model,
        applied_ctrl=np.asarray([[0.3, 0.4], [0.35, 0.45]]),
        phase_id=np.asarray([0, 1], dtype=np.int32),
        transition_substeps=np.full(2, 5, dtype=np.int32),
    )
    planner = ComputedMuscleCEMPlanner(model, _optimizer_config())
    planner.reset(11)
    rollout = execute_physical_rollout(model, target, planner)

    assert rollout.transition_count == 2
    assert rollout.applied_ctrl.shape == (2, 2)
    assert np.all(np.isfinite(rollout.applied_ctrl))
    assert np.all((rollout.applied_ctrl >= 0.0) & (rollout.applied_ctrl <= 1.0))
    assert np.all(np.isfinite(rollout.proposal_tracking_residual_norm))


def test_p08_axial_tracking_weights_are_validated_and_fingerprinted(tmp_path):
    with pytest.raises(ValueError, match="finite and non-negative"):
        PhysicalOptimizerConfig(p08_axial_position_weight=-1.0).validated()

    model = _muscle_model()
    path, manifest = ensure_optimizer_artifact(
        model,
        controller_store=tmp_path / "controllers",
        config=PhysicalOptimizerConfig(
            horizon=1,
            population=2,
            elite_count=1,
            iterations=1,
            p08_axial_position_weight=5.0,
            p08_axial_velocity_weight=1.0,
            p08_position_abs_weight=3.0,
            p08_root_orientation_weight=2.0,
        ),
    )

    assert manifest["optimizer_config"]["p08_axial_position_weight"] == 5.0
    assert manifest["optimizer_config"]["p08_axial_velocity_weight"] == 1.0
    assert manifest["optimizer_config"]["p08_position_abs_weight"] == 3.0
    assert manifest["optimizer_config"]["p08_root_orientation_weight"] == 2.0
    assert path.name == manifest["optimizer_fingerprint"]


def test_p08_axial_tracking_rows_enter_transition_shooting_only_for_p08(monkeypatch):
    model = _muscle_model()
    config = PhysicalOptimizerConfig(
        horizon=1,
        population=2,
        elite_count=1,
        iterations=1,
        position_weight=1.0,
        velocity_weight=1.0,
        p08_axial_position_weight=4.0,
        p08_axial_velocity_weight=9.0,
        p08_position_abs_weight=16.0,
        p08_root_orientation_weight=25.0,
        effort_weight=0.0,
        rate_weight=0.0,
    )
    planner = ComputedMuscleCEMPlanner(model, config)
    monkeypatch.setattr(
        planner,
        "_p08_axial_tracking_dofs",
        lambda task_id: np.asarray([0, 1], dtype=np.int32) if task_id.split("_", 1)[0] == "P08" else None,
    )
    monkeypatch.setattr(
        planner,
        "_p08_root_orientation_tracking_dofs",
        lambda task_id: np.asarray([0, 1], dtype=np.int32) if task_id.split("_", 1)[0] == "P08" else None,
    )

    def fake_transition_error(state, *, ctrl, target_qpos, target_qvel, substeps):
        del state, target_qpos, target_qvel, substeps
        values = np.asarray(ctrl, dtype=np.float64)
        return (
            np.asarray([1.0 + 2.0 * values[0] + 3.0 * values[1], -2.0 + 4.0 * values[0] - values[1]]),
            np.asarray([0.5 - values[0] + 2.0 * values[1], 3.0 + 0.5 * values[0] + 5.0 * values[1]]),
        )

    captured: list[tuple[np.ndarray, np.ndarray]] = []

    def fake_lsq_linear(matrix, rhs, **kwargs):
        del kwargs
        captured.append((np.asarray(matrix).copy(), np.asarray(rhs).copy()))
        return SimpleNamespace(status=1, x=np.zeros((int(model.nu),), dtype=np.float64))

    monkeypatch.setattr(planner, "_simulate_transition_error", fake_transition_error)
    monkeypatch.setattr(primitive_producer, "lsq_linear", fake_lsq_linear)
    previous = np.asarray([0.2, 0.3], dtype=np.float64)
    proposal, residual = planner._transition_shooting_proposal(
        np.zeros((1,), dtype=np.float64),
        target_qpos=np.zeros((int(model.nq),), dtype=np.float64),
        target_qvel=np.zeros((int(model.nv),), dtype=np.float64),
        task_id="P08_axial_rotation",
        previous_ctrl=previous,
        substeps=5,
    )

    matrix, rhs = captured[-1]
    assert matrix.shape == (2 * int(model.nv) + 2, int(model.nu))
    np.testing.assert_allclose(matrix[-2], [12.0, 4.0], rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(matrix[-1], [-1.5, 21.0], rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(rhs[-2:], [-1.6, -16.5], rtol=0.0, atol=1.0e-12)
    np.testing.assert_array_equal(proposal, previous)
    base_position, base_velocity = fake_transition_error(
        None,
        ctrl=previous,
        target_qpos=None,
        target_qvel=None,
        substeps=5,
    )
    expected_squared = (
        np.mean(np.square(base_position))
        + np.mean(np.square(base_velocity))
        + 4.0 * np.square(np.sum(base_position))
        + 9.0 * np.square(np.sum(base_velocity))
        + 16.0 * np.max(np.square(base_position))
        + 25.0 * np.mean(np.square(base_position))
    )
    assert residual == pytest.approx(np.sqrt(expected_squared), rel=0.0, abs=1.0e-12)

    planner._transition_shooting_proposal(
        np.zeros((1,), dtype=np.float64),
        target_qpos=np.zeros((int(model.nq),), dtype=np.float64),
        target_qvel=np.zeros((int(model.nv),), dtype=np.float64),
        task_id="P01_natural_stance",
        previous_ctrl=previous,
        substeps=5,
    )
    assert captured[-1][0].shape == (2 * int(model.nv), int(model.nu))


def test_p08_axial_tracking_cost_matches_named_dof_aggregate(monkeypatch):
    model = _muscle_model()
    target = build_synthetic_motion_target(
        model,
        applied_ctrl=np.asarray([[0.8, 0.2]], dtype=np.float64),
        phase_id=np.asarray([0], dtype=np.int32),
        transition_substeps=np.asarray([5], dtype=np.int32),
        task_id="P08_axial_rotation",
    )
    data = mujoco.MjData(model)
    data.qpos[:] = target.qpos[0]
    data.qvel[:] = target.qvel[0]
    mujoco.mj_forward(model, data)
    state = primitive_producer._capture_integration_state(model, data)
    plan = np.asarray([[0.2, 0.8]], dtype=np.float64)
    previous = np.zeros((int(model.nu),), dtype=np.float64)
    common = {
        "horizon": 1,
        "population": 2,
        "elite_count": 1,
        "iterations": 1,
        "position_weight": 1.0,
        "velocity_weight": 1.0,
        "effort_weight": 0.0,
        "rate_weight": 0.0,
        "terminal_weight": 2.0,
    }
    baseline = ComputedMuscleCEMPlanner(model, PhysicalOptimizerConfig(**common))
    weighted = ComputedMuscleCEMPlanner(
        model,
        PhysicalOptimizerConfig(
            **common,
            p08_axial_position_weight=4.0,
            p08_axial_velocity_weight=9.0,
            p08_position_abs_weight=16.0,
            p08_root_orientation_weight=25.0,
        ),
    )
    monkeypatch.setattr(
        weighted,
        "_p08_axial_tracking_dofs",
        lambda task_id: np.asarray([0, 1], dtype=np.int32) if task_id.split("_", 1)[0] == "P08" else None,
    )
    monkeypatch.setattr(
        weighted,
        "_p08_root_orientation_tracking_dofs",
        lambda task_id: np.asarray([0, 1], dtype=np.int32) if task_id.split("_", 1)[0] == "P08" else None,
    )

    baseline_cost = baseline._plan_cost(state, plan, target, 0, previous)
    weighted_cost = weighted._plan_cost(state, plan, target, 0, previous)
    candidate = mujoco.MjData(model)
    primitive_producer._restore_integration_state(model, candidate, state)
    candidate.ctrl[:] = plan[0]
    for _ in range(5):
        mujoco.mj_step(model, candidate)
    position_error = primitive_producer._position_difference(model, candidate.qpos, target.qpos[1])
    velocity_error = np.asarray(candidate.qvel - target.qvel[1], dtype=np.float64)
    expected_extra = 2.0 * (
        4.0 * np.square(np.sum(position_error))
        + 9.0 * np.square(np.sum(velocity_error))
        + 16.0 * np.max(np.square(position_error))
        + 25.0 * np.mean(np.square(position_error))
    )
    assert weighted_cost - baseline_cost == pytest.approx(expected_extra, rel=0.0, abs=1.0e-12)

    non_p08_target = SimpleNamespace(
        qpos=target.qpos,
        qvel=target.qvel,
        transition_substeps=target.transition_substeps,
        task_id="P01_natural_stance",
    )
    assert weighted._plan_cost(state, plan, non_p08_target, 0, previous) == pytest.approx(
        baseline_cost,
        rel=0.0,
        abs=1.0e-12,
    )


def test_sparse_actuator_moment_decoder_matches_mujoco_generalized_force():
    model = mujoco.MjModel.from_xml_string(
        """
<mujoco model="sparse-moment-fixture">
  <option gravity="0 0 0"/>
  <worldbody>
    <body><joint name="j0" type="slide"/><geom type="sphere" size=".02" mass="1"/></body>
    <body pos=".1 0 0"><joint name="j1" type="slide" axis="0 1 0"/><geom type="sphere" size=".02" mass="1"/></body>
  </worldbody>
  <tendon><fixed name="multi"><joint joint="j0" coef="1"/><joint joint="j1" coef="-2"/></fixed></tendon>
  <actuator>
    <motor name="single" joint="j0" gear="3"/>
    <motor name="coupled" tendon="multi" gear="2"/>
  </actuator>
</mujoco>
"""
    )
    data = mujoco.MjData(model)
    data.ctrl[:] = [0.4, -0.3]
    mujoco.mj_forward(model, data)
    moment = dense_actuator_moment(data, nu=int(model.nu), nv=int(model.nv))

    assert np.count_nonzero(moment[1]) == 2
    np.testing.assert_allclose(
        moment.T @ np.asarray(data.actuator_force),
        np.asarray(data.qfrc_actuator),
        rtol=0.0,
        atol=1.0e-12,
    )


def _write_phase_files(root: Path, *, overlap: bool = False) -> tuple[Path, Path]:
    phase = root / "phase.json"
    phase.write_text(
        json.dumps(
            {
                "schema_version": "primitive_phase_schema_v1",
                "task_id": "P01_stable_stance",
                "phases": [
                    {"id": 0, "name": "settle", "definition": "Stable initial stance."},
                    {"id": 1, "name": "hold", "definition": "Maintain the stance."},
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = root / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": "primitive_transition_phase_plan_v1",
                "task_id": "P01_stable_stance",
                "source_motion_path": "forehandClear_standard/muscle_trajectory/raw_smooth_v1/clip",
                "start_frame": 1,
                "end_frame_exclusive": 5,
                "source_total_frames": 6,
                "transition_count": 3,
                "segments": [
                    {"phase_id": 0, "start_transition": 0, "end_transition": 2},
                    {"phase_id": 1, "start_transition": 1 if overlap else 2, "end_transition": 3},
                ],
            }
        ),
        encoding="utf-8",
    )
    return phase, plan


@pytest.mark.parametrize(
    ("filename", "source_motion_path", "start_frame", "end_frame", "source_total_frames", "boundaries"),
    [
        (
            "P08_kit3_turn_left05_frames_457_529_v1.json",
            "_global/muscle_trajectory/gmr_cache/MyoFullBody/gmr/KIT/3/turn_left05_poses",
            457,
            529,
            544,
            (0, 5, 28, 32, 71),
        ),
        (
            "P08_kit167_turn_right02_frames_385_446_v1.json",
            "_global/muscle_trajectory/gmr_cache/MyoFullBody/gmr/KIT/167/turn_right02_poses",
            385,
            446,
            708,
            (0, 5, 24, 28, 60),
        ),
        (
            "P08_kit10_left_turn09_frames_384_441_v1.json",
            "_global/muscle_trajectory/gmr_cache/MyoFullBody/gmr/KIT/10/LeftTurn09_poses",
            384,
            441,
            450,
            (0, 5, 24, 28, 56),
        ),
    ],
)
def test_checked_in_p08_retry_phase_plans_are_exact(
    filename,
    source_motion_path,
    start_frame,
    end_frame,
    source_total_frames,
    boundaries,
):
    repository = Path(__file__).resolve().parents[2]
    catalog_root = repository / "fullbody/config_specific_task/stage1_body/primitive_catalog"
    schema = load_primitive_phase_schema(catalog_root / "phase_schemas/P08_axial_rotation_v1.json")
    transition_count = end_frame - start_frame - 1
    phase_id = load_transition_phase_plan(
        catalog_root / "phase_plans" / filename,
        phase_schema=schema,
        source_motion_path=source_motion_path,
        transition_count=transition_count,
        start_frame=start_frame,
        end_frame_exclusive=end_frame,
        source_total_frames=source_total_frames,
    )

    expected = np.empty((transition_count,), dtype=np.int32)
    for phase, (start, end) in enumerate(pairwise(boundaries)):
        expected[start:end] = phase
    np.testing.assert_array_equal(phase_id, expected)


def test_crop_and_phase_plan_are_exact_and_reject_overlap_or_chinajump(tmp_path):
    model = _muscle_model()
    qpos = np.zeros((6, int(model.nq)), dtype=np.float64)
    qvel = np.zeros((6, int(model.nv)), dtype=np.float64)
    source = tmp_path / "clip.npz"
    np.savez(source, qpos=qpos, qvel=qvel, frequency=np.asarray(100.0))
    phase, plan = _write_phase_files(tmp_path)
    target = load_retargeted_motion_target(
        source,
        model=model,
        source_motion_path="forehandClear_standard/muscle_trajectory/raw_smooth_v1/clip",
        phase_schema_path=phase,
        phase_plan_path=plan,
        start_frame=1,
        end_frame_exclusive=5,
    )
    assert target.qpos.shape[0] == 4
    assert target.transition_count == 3
    assert (target.source_start_frame, target.source_end_frame_exclusive, target.source_total_frames) == (1, 5, 6)

    _, overlap_plan = _write_phase_files(tmp_path, overlap=True)
    with pytest.raises(ValueError, match="ordered, contiguous"):
        load_transition_phase_plan(
            overlap_plan,
            phase_schema=load_primitive_phase_schema(phase),
            source_motion_path=target.source_motion_path,
            transition_count=3,
            start_frame=1,
            end_frame_exclusive=5,
            source_total_frames=6,
        )
    with pytest.raises(ValueError, match="at least two frames"):
        load_retargeted_motion_target(
            source,
            model=model,
            source_motion_path=target.source_motion_path,
            phase_schema_path=phase,
            phase_plan_path=plan,
            start_frame=2,
            end_frame_exclusive=3,
        )
    with pytest.raises(ValueError, match="rejects target-skill source"):
        load_retargeted_motion_target(
            source,
            model=model,
            source_motion_path="ChinaJump/muscle_trajectory/optimized/forehandJump-1",
            phase_schema_path=phase,
            phase_plan_path=plan,
            start_frame=1,
            end_frame_exclusive=5,
        )


def test_full_action_policy_import_uses_only_physical_ctrl_and_rejects_signed_old_actions(
    tmp_path,
    monkeypatch,
):
    model = _muscle_model()
    controls = np.asarray([[0.2, 0.3], [0.4, 0.25], [0.6, 0.2]])
    target = build_synthetic_motion_target(
        model,
        applied_ctrl=controls,
        phase_id=np.asarray([0, 1, 1], dtype=np.int32),
        transition_substeps=np.full(3, 5, dtype=np.int32),
        source_motion_path="forehandClear_standard/muscle_trajectory/raw_smooth_v1/clip",
    )
    checkpoint_sha = "a" * 64
    checkpoint_record = {
        "schema_version": "checkpoint_content_fingerprint_v1",
        "supplied_path": "checkpoint",
        "resolved_path": str(tmp_path / "checkpoint"),
        "sha256": checkpoint_sha,
        "num_files": 1,
        "num_bytes": 4,
        "files": [{"path": "params", "sha256": "b" * 64, "num_bytes": 4}],
    }
    monkeypatch.setattr(
        "musclemimic.synergy.primitive_producer.checkpoint_content_fingerprint",
        lambda _: checkpoint_record,
    )
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) for index in range(int(model.nu))]
    from musclemimic.distill.action_schema import actuator_schema_hash

    metadata = {
        "collector": "teacher_lookahead_rollout",
        "physical_capture": {
            "schema_version": "physical_capture_spec_v2",
            "actuator_names": names,
        },
        "actuator_names": names,
        "actuator_ctrlrange": np.asarray(model.actuator_ctrlrange).tolist(),
        "action_schema_hash": actuator_schema_hash(names),
        "teacher_checkpoint_content": checkpoint_record,
        "teacher_checkpoint_fingerprint": checkpoint_sha,
        "teacher_action_target": "mean",
    }
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    shard_path = tmp_path / "shard.npz"
    motion_uid = target.source_motion_path
    from musclemimic.distill.motion_identity import stable_motion_uid

    arrays = {
        "teacher_ctrl_physical": controls,
        # This normalized field is present to prove it is not selected.
        "teacher_action": np.full_like(controls, -0.9),
        "rollout_uid": np.full(3, 17, dtype=np.int64),
        "motion_uid": np.full(3, stable_motion_uid(motion_uid), dtype=np.int64),
        "subtraj_step_no": np.arange(3, dtype=np.int32),
        "phase_id": target.phase_id,
        "done": np.asarray([False, False, True]),
        "sim_pre_qpos": target.qpos[:-1],
        "sim_pre_qvel": target.qvel[:-1],
    }
    np.savez(shard_path, **arrays)
    imported = load_full_action_policy_controls(
        shard_path,
        metadata_path=metadata_path,
        teacher_checkpoint=tmp_path / "checkpoint",
        model=model,
        target=target,
        rollout_uid=17,
    )
    np.testing.assert_array_equal(imported.planner.controls, controls)
    assert imported.controller_binding["checkpoint_sha256"] == checkpoint_sha
    assert imported.rollout_binding["transition_count"] == 3

    arrays["teacher_ctrl_physical"] = controls.copy()
    arrays["teacher_ctrl_physical"][0, 0] = -0.1
    np.savez(shard_path, **arrays)
    with pytest.raises(ValueError, match="signed/normalized Paper_Need actions are forbidden"):
        load_full_action_policy_controls(
            shard_path,
            metadata_path=metadata_path,
            teacher_checkpoint=tmp_path / "checkpoint",
            model=model,
            target=target,
            rollout_uid=17,
        )

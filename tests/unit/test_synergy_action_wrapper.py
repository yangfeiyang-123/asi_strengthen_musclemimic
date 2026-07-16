from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct
from omegaconf import OmegaConf

from loco_mujoco.core.utils import Box
from musclemimic.algorithms.common.env_utils import apply_policy_interface_wrappers
from musclemimic.core.wrappers.synergy_action import SynergyActionWrapper
from musclemimic.distill.action_schema import actuator_schema_hash
from musclemimic.synergy.action_interface import (
    build_early_synergy_action_interface,
    save_coefficient_statistics,
    save_structured_residual_basis,
)
from musclemimic.synergy.basis_artifact import save_synergy_basis
from musclemimic.synergy.oracle_coverage import (
    StaticProxyCoverageThresholds,
    evaluate_static_proxy_coverage,
    write_static_proxy_coverage_gate,
)
from musclemimic.synergy.primitive_manifest import save_primitive_source_manifest
from musclemimic.synergy.schema import (
    EXCITATION_SIGNAL_KIND,
    ctrlrange_schema_hash,
)

NAMES = ("muscle_a", "muscle_b", "muscle_c")


def _coverage_phase_schema(*phase_ids: int) -> dict:
    ids = phase_ids or (0, 1)
    return {
        "schema_version": "chinajump_coverage_phase_schema_v1",
        "target_skill_id": "test_skill",
        "phase_field": "phase_id",
        "producer_contract": "unit_test_explicit_labels_v1",
        "phases": [
            {
                "id": phase_id,
                "name": f"phase_{phase_id}",
                "definition": f"Unit-test semantic definition for phase {phase_id}.",
            }
            for phase_id in sorted(ids)
        ],
    }


@struct.dataclass
class _MockState:
    step: jnp.ndarray
    info: dict


class _MockBodyEnv:
    def __init__(self):
        self.mjx_env = False
        self.mjx_enabled = False
        self.policy_actuator_names = NAMES
        self.info = SimpleNamespace(
            observation_space=Box(-np.ones(4), np.ones(4)),
            action_space=Box(-np.ones(3), np.ones(3)),
            gamma=0.99,
            horizon=10,
            dt=0.01,
        )
        self.mdp_info = self.info
        self.obs_container = SimpleNamespace(get_obs_ind_by_group=lambda _group: np.arange(4))
        self.last_body_action = None

    def reset(self, _key):
        return jnp.zeros(4), SimpleNamespace(step=0)

    def reset_to(self, key, _traj_idx):
        return self.reset(key)

    def step(self, state, action):
        self.last_body_action = action
        return jnp.ones(4), 1.0, False, False, {"existing": jnp.asarray(1.0)}, state


def _basis_manifest(*, rank: int, selection_reason: str) -> dict:
    ctrlrange = np.asarray([[0.0, 1.0]] * len(NAMES), dtype=np.float64)
    eligible = selection_reason == "smallest_rank_meeting_all_vaf_and_stability_gates"
    selected_metrics = {
        "rank": rank,
        "validation": {"global_vaf": 0.95 if eligible else 0.50},
        "validation_local_vaf_quantile": 0.80 if eligible else 0.40,
        "validation_local_vaf_quantile_level": 0.10,
        "initialization_stability": {"mean_similarity": 0.90},
        "split_half_stability": {"mean_similarity": 0.90},
        "bootstrap_stability": {"mean_similarity": 0.90},
        "cross_trial_stability": {"available": True, "mean_similarity": 0.90},
        "eligible": eligible,
        "rejection_reasons": [] if eligible else ["validation.global_vaf below threshold"],
    }
    return {
        "signal_kind": EXCITATION_SIGNAL_KIND,
        "region": "whole_body",
        "rank": rank,
        "normalization": {"kind": "none"},
        "source_dataset_fingerprint": "d" * 64,
        "teacher_checkpoint_fingerprint": "c" * 64,
        "fit_seed": 0,
        "transform": {
            "kind": "ctrlrange_affine_to_unit",
            "raw_signal_kind": "applied_ctrl",
            "formula": "(ctrl-low)/(high-low)",
            "ctrlrange": ctrlrange.tolist(),
            "actuator_names": list(NAMES),
            "ctrlrange_schema_hash": ctrlrange_schema_hash(NAMES, ctrlrange),
            "roundoff_policy": "fail_outside_ctrlrange_then_clamp_within_tolerance_only",
        },
        "split_provenance": {"train": {}, "validation": {}},
        "train_motion_uids": [1, 2],
        "selection": {
            "selected_rank": rank,
            "reason": selection_reason,
            "eligible_ranks": [rank] if eligible else [],
            "rejected_too_large_ranks": [],
            "thresholds": {
                "min_val_global_vaf": 0.90,
                "min_val_local_vaf_quantile": 0.70,
                "local_vaf_quantile": 0.10,
                "min_initialization_similarity": 0.80,
                "min_split_half_similarity": 0.80,
                "min_bootstrap_similarity": 0.80,
                "min_cross_trial_similarity": 0.75,
            },
        },
        "selected_metrics": selected_metrics,
        "rank_scan": {str(rank): selected_metrics},
    }


def _artifacts(tmp_path, *, residual: bool = False, selection_reason=None):
    reason = selection_reason or "smallest_rank_meeting_all_vaf_and_stability_gates"
    basis = save_synergy_basis(
        tmp_path / "basis",
        basis=np.asarray(
            [
                [0.45, 0.05],
                [0.15, 0.40],
                [0.20, 0.25],
            ],
            dtype=np.float64,
        ),
        muscle_names=NAMES,
        manifest=_basis_manifest(rank=2, selection_reason=reason),
    )
    coefficients = np.asarray(
        [
            [0.05, 0.08],
            [0.10, 0.12],
            [0.15, 0.18],
            [0.20, 0.22],
            [0.25, 0.28],
        ]
    )
    stats = save_coefficient_statistics(
        basis.path / "coefficient_stats.npz",
        coefficients,
        basis_fingerprint=basis.fingerprint,
    )
    residual_basis = None
    if residual:
        residual_basis = save_structured_residual_basis(
            tmp_path / "residual",
            basis=np.asarray([[0.2], [-0.1], [0.05]], dtype=np.float64),
            actuator_names=NAMES,
            source_basis_fingerprint=basis.fingerprint,
            source_description="test-only low-rank takeoff correction",
        )
    return basis, stats, residual_basis


def _config(basis, stats, residual_basis=None):
    residual = residual_basis is not None
    return OmegaConf.create(
        {
            "enabled": True,
            "schema_version": "early_synergy_action_v1",
            "mode": "fixed_synergy_residual" if residual else "fixed_synergy",
            "basis_path": str(basis.path),
            "expected_basis_fingerprint": basis.fingerprint,
            "expected_underlying_action_dim": 3,
            "expected_actuator_schema_hash": actuator_schema_hash(NAMES),
            "require_all_basis_gates": True,
            "forbid_fallback_selected_basis": True,
            "require_coverage_gate": False,
            "max_basis_condition_number": 100.0,
            "min_effective_rank_fraction": 0.8,
            "coefficient_transform": {
                "kind": "bounded_sigmoid",
                "stats_path": stats["path"],
                "expected_stats_fingerprint": stats["stats_fingerprint"],
                "max_source": "train_q99_times_1p2",
                "center_source": "train_q50",
                "temperature": 1.0,
            },
            "tonic_baseline": {
                "kind": "zero",
                "learned_full_dimensional": False,
            },
            "residual": {
                "enabled": residual,
                "basis_path": None if residual_basis is None else residual_basis.source_path,
                "expected_fingerprint": (None if residual_basis is None else residual_basis.fingerprint),
                "alpha": 0.03 if residual else 0.0,
                "max_row_l1_norm": 2.0,
            },
            "exploration": {"calibrate_in_physical_space": False},
        }
    )


def test_fixed_synergy_wrapper_reduces_action_dimension_and_preserves_body_abi(tmp_path):
    basis, stats, _ = _artifacts(tmp_path)
    base = _MockBodyEnv()
    wrapper = SynergyActionWrapper(base, _config(basis, stats))

    assert wrapper.info.action_space.shape == (2,)
    assert wrapper.info.observation_space.shape == (4,)
    assert wrapper.body_actuator_names == NAMES
    assert len(wrapper.policy_action_names) == 2
    assert wrapper.action_manifest["body_action_dim"] == 3
    assert wrapper.action_manifest["basis_fingerprint"] == basis.fingerprint

    raw = jnp.zeros(2, dtype=jnp.float32)
    output = wrapper.decode_action(raw)
    np.testing.assert_allclose(
        np.asarray(output.synergy_coefficients),
        wrapper.action_interface.coefficient_transform.center,
        rtol=1e-6,
        atol=1e-7,
    )
    result = wrapper.step(
        _MockState(step=jnp.asarray(0), info={"state_existing": jnp.asarray(2.0)}),
        raw,
    )
    assert result[4]["existing"] == 1.0
    assert "synergy_decoded_excitation_rms" in result[4]
    assert result[5].info["state_existing"] == 2.0
    assert "synergy_decoded_excitation_rms" in result[5].info
    np.testing.assert_allclose(np.asarray(base.last_body_action), output.body_action)
    assert np.all(np.asarray(output.physical_excitation) >= 0.0)
    assert np.all(np.asarray(output.physical_excitation) <= 1.0)


def test_primitive_bootstrap_records_missing_target_coverage_without_weakening_basis_gates(
    tmp_path,
):
    basis, stats, _ = _artifacts(tmp_path)
    config = _config(basis, stats)
    config.bootstrap_without_target_coverage = True

    wrapper = SynergyActionWrapper(_MockBodyEnv(), config)
    assert wrapper.action_manifest["coverage_gate"] is None
    assert wrapper.action_manifest["target_coverage_evidence"] == {
        "status": "not_evaluated_primitive_bootstrap",
        "bootstrap_without_target_coverage": True,
    }

    config.require_coverage_gate = True
    with pytest.raises(ValueError, match="primitive bootstrap must not bind"):
        SynergyActionWrapper(_MockBodyEnv(), config)


def test_extreme_raw_actions_remain_bounded_and_jit_vmap_agree(tmp_path):
    basis, stats, _ = _artifacts(tmp_path)
    wrapper = SynergyActionWrapper(_MockBodyEnv(), _config(basis, stats))
    raw = jnp.asarray([[-1.0e4, 1.0e4], [1.0e4, -1.0e4]], dtype=jnp.float32)

    eager = wrapper.decode_action(raw)
    compiled = jax.jit(wrapper.decode_action)(raw)
    mapped = jax.vmap(wrapper.decode_action)(raw)
    np.testing.assert_allclose(compiled.body_action, eager.body_action, atol=1e-7)
    np.testing.assert_allclose(mapped.body_action, eager.body_action, atol=1e-7)
    assert np.all(np.isfinite(np.asarray(eager.body_action)))
    assert np.all(np.asarray(eager.physical_excitation) >= 0.0)
    assert np.all(np.asarray(eager.physical_excitation) <= 1.0)


def test_structured_residual_adds_only_low_rank_bounded_direction(tmp_path):
    basis, stats, residual = _artifacts(tmp_path, residual=True)
    wrapper = SynergyActionWrapper(_MockBodyEnv(), _config(basis, stats, residual))
    assert wrapper.info.action_space.shape == (3,)
    assert wrapper.action_interface.residual_dim == 1

    zero = wrapper.decode_action(jnp.zeros(3))
    changed = wrapper.decode_action(jnp.asarray([0.0, 0.0, 1.0e4]))
    np.testing.assert_array_equal(np.asarray(zero.residual_excitation), np.zeros(3))
    expected = 0.03 * np.asarray(residual.basis[:, 0])
    np.testing.assert_allclose(changed.residual_excitation, expected, rtol=1e-6, atol=1e-7)
    assert wrapper.action_manifest["residual_basis_fingerprint"] == residual.fingerprint
    assert residual.normalization == "unit_l2_columns"
    np.testing.assert_allclose(np.linalg.norm(residual.basis, axis=0), 1.0, atol=1e-6)
    assert wrapper.action_manifest["residual_max_per_muscle_correction"] <= 0.03


def test_structured_residual_fit_contract_requirement_rejects_handmade_matrix(tmp_path):
    basis, stats, residual = _artifacts(tmp_path, residual=True)
    config = _config(basis, stats, residual)
    config.residual.require_fit_contract = True

    with pytest.raises(ValueError, match="requires a passed train-only fit contract"):
        SynergyActionWrapper(_MockBodyEnv(), config)


def test_interface_wrapper_is_opt_in_and_binds_runtime_manifest(tmp_path):
    basis, stats, _ = _artifacts(tmp_path)
    base = _MockBodyEnv()
    disabled = OmegaConf.create({"action_representation": {"enabled": False}})
    assert apply_policy_interface_wrappers(base, disabled) is base

    exp = OmegaConf.create(
        {
            "action_representation": OmegaConf.to_container(_config(basis, stats), resolve=True),
            "init_std": 0.35,
        }
    )
    wrapped = apply_policy_interface_wrappers(base, exp)
    assert isinstance(wrapped, SynergyActionWrapper)
    assert wrapped.info.action_space.shape == (2,)
    assert exp.action_manifest.physical_action_interface_hash == wrapped.physical_action_interface_hash
    assert exp.action_manifest.exploration.kind == "configured_policy_std_v1"
    assert apply_policy_interface_wrappers(wrapped, exp) is wrapped


def test_fallback_basis_and_fingerprint_drift_fail_closed(tmp_path):
    fallback = "fallback_best_heldout_global_vaf_no_rank_met_all_gates"
    basis, stats, _ = _artifacts(tmp_path, selection_reason=fallback)
    with pytest.raises(ValueError, match="fallback-selected"):
        SynergyActionWrapper(_MockBodyEnv(), _config(basis, stats))

    basis2, stats2, _ = _artifacts(tmp_path / "other")
    config = _config(basis2, stats2)
    config.expected_basis_fingerprint = "0" * 64
    with pytest.raises(ValueError, match="expected fingerprint"):
        SynergyActionWrapper(_MockBodyEnv(), config)


def test_selection_reason_cannot_forge_failed_numeric_gates(tmp_path):
    manifest = _basis_manifest(
        rank=2,
        selection_reason="smallest_rank_meeting_all_vaf_and_stability_gates",
    )
    manifest["selected_metrics"]["validation"]["global_vaf"] = 0.10
    basis = save_synergy_basis(
        tmp_path / "forged_basis",
        basis=np.asarray(
            [[0.45, 0.05], [0.15, 0.40], [0.20, 0.25]],
            dtype=np.float64,
        ),
        muscle_names=NAMES,
        manifest=manifest,
    )
    stats = save_coefficient_statistics(
        basis.path / "coefficient_stats.npz",
        np.asarray([[0.10, 0.12], [0.20, 0.22]], dtype=np.float64),
        basis_fingerprint=basis.fingerprint,
    )
    with pytest.raises(ValueError, match="eligibility differs from recomputed gates"):
        SynergyActionWrapper(_MockBodyEnv(), _config(basis, stats))


def test_learned_full_dimensional_baseline_is_rejected(tmp_path):
    basis, stats, _ = _artifacts(tmp_path)
    config = _config(basis, stats)
    config.tonic_baseline.learned_full_dimensional = True
    with pytest.raises(ValueError, match="never learned full-dimensional"):
        SynergyActionWrapper(_MockBodyEnv(), config)


def test_primitive_source_contract_binds_basis_runtime_model_and_target_exclusion(
    tmp_path,
):
    ctrlrange = np.asarray([[0.0, 1.0]] * len(NAMES), dtype=np.float64)
    source = save_primitive_source_manifest(
        tmp_path / "primitive_source",
        target_skill_id="ChinaJump",
        excluded_target_motion_paths=["ChinaJump/train-1", "ChinaJump/val-1"],
        primitive_task_ids=["squat", "jump"],
        primitive_source_kinds={"squat": "primitive", "jump": "primitive"},
        primitive_trial_ids=["squat-01", "jump-01"],
        train_motion_uids=[10, 11],
        validation_motion_uids=[20],
        source_checkpoint_fingerprints={"squat": "1" * 64, "jump": "2" * 64},
        source_checkpoint_contents={
            task: {
                "schema_version": "checkpoint_content_fingerprint_v1",
                "supplied_path": f"fixtures/{task}",
                "resolved_path": f"/fixtures/{task}",
                "sha256": fingerprint,
                "num_files": 1,
                "num_bytes": 1,
                "files": [{"path": "params", "sha256": "9" * 64, "num_bytes": 1}],
            }
            for task, fingerprint in {"squat": "1" * 64, "jump": "2" * 64}.items()
        },
        primitive_required_phase_ids={"squat": [1], "jump": [1]},
        primitive_phase_schema_fingerprints={
            "squat": "a" * 64,
            "jump": "b" * 64,
        },
        source_dataset_fingerprint="d" * 64,
        model_hash="4" * 64,
        actuator_schema_hash=actuator_schema_hash(NAMES),
        control_range_hash="6" * 64,
        transform_ctrlrange_schema_hash=ctrlrange_schema_hash(NAMES, ctrlrange),
        preprocessing_fingerprint="7" * 64,
        phase_weight_fingerprint="8" * 64,
        nmf_seeds=[0, 1],
    )
    binding = {
        "schema_version": source.manifest["schema_version"],
        "manifest_fingerprint": source.fingerprint,
        "source_dataset_fingerprint": "d" * 64,
        "primitive_only": True,
        "contains_target_skill_rollouts": False,
        "target_skill_id": "ChinaJump",
        "excluded_target_motions": source.manifest["excluded_target_motions"],
        "primitive_task_ids": ["squat", "jump"],
        "model_hash": "4" * 64,
        "transform_ctrlrange_schema_hash": ctrlrange_schema_hash(NAMES, ctrlrange),
    }
    manifest = _basis_manifest(
        rank=2,
        selection_reason="smallest_rank_meeting_all_vaf_and_stability_gates",
    )
    selected = manifest["selected_metrics"]
    selected["validation_phase_balanced"] = {"global_vaf": 0.95}
    primitive_group_metrics = {"global_vaf": 0.95}
    selected["primitive_group_validation"] = {
        "per_task": {
            "jump": dict(primitive_group_metrics),
            "squat": dict(primitive_group_metrics),
        },
        "per_task_phase": {
            "jump/1": dict(primitive_group_metrics),
            "squat/1": dict(primitive_group_metrics),
        },
        "per_trial": {
            "jump-01": dict(primitive_group_metrics),
            "squat-01": dict(primitive_group_metrics),
        },
        "per_task_phase_trial": {
            "jump/1/jump-01": dict(primitive_group_metrics),
            "squat/1/squat-01": dict(primitive_group_metrics),
        },
        "minimum_global_vaf": 0.95,
    }
    manifest["primitive_source_binding"] = binding
    basis = save_synergy_basis(
        tmp_path / "primitive_basis",
        basis=np.asarray(
            [[0.45, 0.05], [0.15, 0.40], [0.20, 0.25]],
            dtype=np.float64,
        ),
        muscle_names=NAMES,
        manifest=manifest,
    )
    stats = save_coefficient_statistics(
        basis.path / "coefficient_stats.npz",
        np.asarray([[0.10, 0.12], [0.20, 0.22]], dtype=np.float64),
        basis_fingerprint=basis.fingerprint,
    )
    config = _config(basis, stats)
    config.require_runtime_ctrlrange_binding = True
    config.require_primitive_source_contract = True
    config.primitive_source_manifest_path = str(source.path)
    config.expected_primitive_source_manifest_fingerprint = source.fingerprint
    config.expected_target_skill_id = "ChinaJump"
    config.expected_excluded_target_motion_paths = [
        "ChinaJump/train-1",
        "ChinaJump/val-1",
    ]
    interface = build_early_synergy_action_interface(
        config,
        expected_actuator_names=NAMES,
        runtime_ctrlrange=ctrlrange,
        runtime_model_hash="4" * 64,
    )
    assert interface.action_manifest["primitive_source_binding"] == binding
    assert interface.action_manifest["runtime_model_hash"] == "4" * 64

    with pytest.raises(ValueError, match="model hash differs"):
        build_early_synergy_action_interface(
            config,
            expected_actuator_names=NAMES,
            runtime_ctrlrange=ctrlrange,
            runtime_model_hash="5" * 64,
        )


def test_coverage_gate_must_bind_runtime_coefficient_bounds(tmp_path):
    basis, stats, _ = _artifacts(tmp_path)
    runtime_upper = 1.2 * np.asarray(stats["coefficient_q99"])
    proxy_coefficients = np.asarray([[0.08, 0.10], [0.16, 0.18]])
    proxy = proxy_coefficients @ np.asarray(basis.basis).T
    report = evaluate_static_proxy_coverage(
        basis,
        proxy,
        phase_id=np.asarray([0, 1], dtype=np.int32),
        phase_schema=_coverage_phase_schema(),
        coefficient_upper_bounds=runtime_upper,
        thresholds=StaticProxyCoverageThresholds(required_phase_ids=(0, 1)),
        proxy_muscle_names=NAMES,
    )
    assert report["passed"] is True
    gate_path = tmp_path / "coverage_gate.json"
    write_static_proxy_coverage_gate(gate_path, report)

    config = _config(basis, stats)
    config.require_coverage_gate = True
    config.coverage_gate_path = str(gate_path)
    config.expected_coverage_gate_fingerprint = report["artifact_fingerprint"]
    config.expected_coverage_proxy_fingerprint = report["proxy_fingerprint"]
    config.required_coverage_thresholds = report["thresholds"]
    config.require_phase_conditioned_coverage = True
    config.required_coverage_phase_schema_fingerprint = report["proxy_binding"]["phase_schema_fingerprint"]
    config.min_required_coverage_phases = 2
    wrapper = SynergyActionWrapper(_MockBodyEnv(), config)
    np.testing.assert_allclose(
        wrapper.action_manifest["coverage_gate"]["coefficient_upper_bounds"],
        runtime_upper,
    )

    config.coefficient_transform.max_source = "train_q99"
    with pytest.raises(ValueError, match="coefficient bounds differ"):
        SynergyActionWrapper(_MockBodyEnv(), config)

    config.coefficient_transform.max_source = "train_q99_times_1p2"
    config.required_coverage_phase_schema_fingerprint = "f" * 64
    with pytest.raises(ValueError, match="semantic phase schema differs"):
        SynergyActionWrapper(_MockBodyEnv(), config)


def test_producer_bound_coverage_rejects_v3_and_preserves_v4_provenance(tmp_path):
    from musclemimic.synergy.coverage_proxy import (
        COVERAGE_PROXY_MANIFEST_SCHEMA_VERSION,
    )
    from musclemimic.synergy.oracle_coverage import (
        FORMAL_STATIC_PROXY_COVERAGE_SCHEMA_VERSION,
        STATIC_PROXY_COVERAGE_SCHEMA_VERSION,
    )

    basis, stats, _ = _artifacts(tmp_path)
    runtime_upper = 1.2 * np.asarray(stats["coefficient_q99"])
    proxy_coefficients = np.asarray([[0.08, 0.10], [0.16, 0.18]])
    proxy = proxy_coefficients @ np.asarray(basis.basis).T
    phase_schema = _coverage_phase_schema()
    thresholds = StaticProxyCoverageThresholds(required_phase_ids=(0, 1))
    legacy_report = evaluate_static_proxy_coverage(
        basis,
        proxy,
        phase_id=np.asarray([0, 1], dtype=np.int32),
        phase_schema=phase_schema,
        coefficient_upper_bounds=runtime_upper,
        thresholds=thresholds,
        proxy_muscle_names=NAMES,
    )
    assert legacy_report["schema_version"] == STATIC_PROXY_COVERAGE_SCHEMA_VERSION
    legacy_gate_path = tmp_path / "legacy_v3_coverage_gate.json"
    write_static_proxy_coverage_gate(legacy_gate_path, legacy_report)

    config = _config(basis, stats)
    config.require_coverage_gate = True
    config.require_producer_bound_coverage = True
    config.coverage_gate_path = str(legacy_gate_path)
    config.expected_coverage_gate_fingerprint = legacy_report["artifact_fingerprint"]
    config.expected_coverage_proxy_fingerprint = legacy_report["proxy_fingerprint"]
    config.required_coverage_thresholds = legacy_report["thresholds"]
    config.require_phase_conditioned_coverage = True
    config.required_coverage_phase_schema_fingerprint = legacy_report["proxy_binding"]["phase_schema_fingerprint"]
    config.min_required_coverage_phases = 2
    with pytest.raises(ValueError, match="formal v4 static proxy gate"):
        SynergyActionWrapper(_MockBodyEnv(), config)

    producer_binding = {
        "producer_manifest_schema_version": COVERAGE_PROXY_MANIFEST_SCHEMA_VERSION,
        "producer_manifest_fingerprint": "1" * 64,
        "producer_artifact_kind": "chinajump_target_physical_excitation_proxy",
        "source_kind": "full_action_teacher",
        "source_manifest_fingerprint": "2" * 64,
        "source_qc_fingerprint": "3" * 64,
        "proxy_content_fingerprint": legacy_report["proxy_fingerprint"],
        "phase_schema_fingerprint": legacy_report["proxy_binding"]["phase_schema_fingerprint"],
        "required_phase_ids": [0, 1],
        "min_phase_samples": 1,
        "per_phase_sample_counts": {"0": 1, "1": 1},
    }
    formal_report = evaluate_static_proxy_coverage(
        basis,
        proxy,
        phase_id=np.asarray([0, 1], dtype=np.int32),
        phase_schema=phase_schema,
        coefficient_upper_bounds=runtime_upper,
        thresholds=thresholds,
        proxy_muscle_names=NAMES,
        proxy_producer_binding=producer_binding,
    )
    assert formal_report["schema_version"] == FORMAL_STATIC_PROXY_COVERAGE_SCHEMA_VERSION
    formal_gate_path = tmp_path / "formal_v4_coverage_gate.json"
    write_static_proxy_coverage_gate(formal_gate_path, formal_report)
    config.coverage_gate_path = str(formal_gate_path)
    config.expected_coverage_gate_fingerprint = formal_report["artifact_fingerprint"]
    config.expected_coverage_proxy_fingerprint = formal_report["proxy_fingerprint"]

    wrapper = SynergyActionWrapper(_MockBodyEnv(), config)
    bound_gate = wrapper.action_manifest["coverage_gate"]
    assert bound_gate["schema_version"] == FORMAL_STATIC_PROXY_COVERAGE_SCHEMA_VERSION
    assert bound_gate["producer_binding"] == producer_binding
    assert bound_gate["producer_manifest_fingerprint"] == "1" * 64


def test_producer_bound_coverage_cannot_disable_the_coverage_gate(tmp_path):
    basis, stats, _ = _artifacts(tmp_path)
    config = _config(basis, stats)
    config.require_producer_bound_coverage = True

    with pytest.raises(
        ValueError,
        match="require_producer_bound_coverage requires require_coverage_gate=true",
    ):
        SynergyActionWrapper(_MockBodyEnv(), config)


def test_failed_zero_required_phase_gate_is_rejected_by_runtime_wrapper(tmp_path):
    basis, stats, _ = _artifacts(tmp_path)
    runtime_upper = 1.2 * np.asarray(stats["coefficient_q99"])
    active = np.asarray([0.12, 0.10]) @ np.asarray(basis.basis).T
    proxy = np.stack([active, np.zeros_like(active)])
    report = evaluate_static_proxy_coverage(
        basis,
        proxy,
        phase_id=np.asarray([1, 4], dtype=np.int32),
        phase_schema=_coverage_phase_schema(1, 4),
        coefficient_upper_bounds=runtime_upper,
        thresholds=StaticProxyCoverageThresholds(
            max_decoded_saturation_fraction=1.0,
            required_phase_ids=(1, 4),
        ),
        proxy_muscle_names=NAMES,
    )
    assert report["passed"] is False
    gate_path = tmp_path / "failed_zero_phase_gate.json"
    write_static_proxy_coverage_gate(gate_path, report)

    config = _config(basis, stats)
    config.require_coverage_gate = True
    config.coverage_gate_path = str(gate_path)
    config.expected_coverage_gate_fingerprint = report["artifact_fingerprint"]
    config.expected_coverage_proxy_fingerprint = report["proxy_fingerprint"]
    config.required_coverage_thresholds = report["thresholds"]
    config.require_phase_conditioned_coverage = True
    config.required_coverage_phase_schema_fingerprint = report["proxy_binding"]["phase_schema_fingerprint"]
    config.min_required_coverage_phases = 2
    with pytest.raises(ValueError, match="did not pass"):
        SynergyActionWrapper(_MockBodyEnv(), config)

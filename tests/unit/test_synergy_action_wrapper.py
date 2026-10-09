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
from musclemimic.distill.physical import (
    MUSCLE_CHANNEL_CONTRACT_SCHEMA_VERSION,
    MUSCLE_EXCITATION_FORMULA,
    MUSCLE_EXCITATION_ROUNDOFF_POLICY,
    PHYSICAL_SIGNAL_SCHEMA_VERSION,
    UNIT_EXCITATION_TRANSFORM,
)
from musclemimic.synergy.action_interface import (
    ACTION_SCHEMA_VERSION,
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
from musclemimic.synergy.rank_selection import (
    DYNAMIC_COVERAGE_EVIDENCE_KIND,
    DYNAMIC_COVERAGE_SCHEMA_VERSION,
    candidate_basis_fingerprint,
    dynamic_coverage_artifact_fingerprint,
    dynamic_coverage_requirement,
)
from musclemimic.synergy.schema import (
    EXCITATION_SIGNAL_KIND,
    ctrlrange_schema_hash,
)

NAMES = ("muscle_a", "muscle_b", "muscle_c")
BASIS_MATRIX = np.asarray(
    [
        [0.45, 0.05],
        [0.15, 0.40],
        [0.20, 0.25],
    ],
    dtype=np.float64,
)


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
        "physical_signal_schema_version": PHYSICAL_SIGNAL_SCHEMA_VERSION,
        "signal_kind": EXCITATION_SIGNAL_KIND,
        "region": "whole_body",
        "rank": rank,
        "normalization": {"kind": "none"},
        "source_dataset_fingerprint": "d" * 64,
        "teacher_checkpoint_fingerprint": "c" * 64,
        "fit_seed": 0,
        "transform": {
            "kind": UNIT_EXCITATION_TRANSFORM,
            "raw_signal_kind": "applied_ctrl",
            "formula": MUSCLE_EXCITATION_FORMULA,
            "ctrlrange": ctrlrange.tolist(),
            "actuator_names": list(NAMES),
            "ctrlrange_schema_hash": ctrlrange_schema_hash(NAMES, ctrlrange),
            "roundoff_policy": MUSCLE_EXCITATION_ROUNDOFF_POLICY,
            "physical_signal_schema_version": PHYSICAL_SIGNAL_SCHEMA_VERSION,
            "muscle_channel_contract": {
                "schema_version": MUSCLE_CHANNEL_CONTRACT_SCHEMA_VERSION,
                "actuator_names": list(NAMES),
                "actuator_ids": list(range(len(NAMES))),
                "actuator_dyntype": ["muscle"] * len(NAMES),
                "actuator_actnum": [1] * len(NAMES),
                "actuator_actadr": list(range(len(NAMES))),
                "model_na": len(NAMES),
            },
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


def _dynamic_coverage_report(
    *,
    candidate_fingerprint: str,
    passed: bool,
) -> dict:
    mean_gap = 0.10 if passed else 0.20
    phase_gap = 0.20 if passed else 0.30
    checks = {
        "mean_dynamic_gap": mean_gap <= 0.15,
        "key_phase_dynamic_gap": phase_gap <= 0.25,
        "nonempty_rollout_evidence": True,
    }
    report = {
        "schema_version": DYNAMIC_COVERAGE_SCHEMA_VERSION,
        "evidence_kind": DYNAMIC_COVERAGE_EVIDENCE_KIND,
        "signal_kind": EXCITATION_SIGNAL_KIND,
        "region": "whole_body",
        "rank": 2,
        "candidate_basis_fingerprint": candidate_fingerprint,
        "rollout_manifest_fingerprint": "1" * 64,
        "environment_fingerprint": "2" * 64,
        "metrics": {
            "mean_dynamic_gap": mean_gap,
            "max_key_phase_dynamic_gap": phase_gap,
            "rollout_count": 8,
            "key_phase_count": 3,
            "horizon_steps": 32,
        },
        "thresholds": {
            "max_mean_dynamic_gap": 0.15,
            "max_key_phase_dynamic_gap": 0.25,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    report["artifact_fingerprint"] = dynamic_coverage_artifact_fingerprint(report)
    return report


def _bind_dynamic_coverage_contract(manifest: dict, *, state: str) -> None:
    required = state != "optional_missing"
    requirement = dynamic_coverage_requirement(
        required=required,
        max_mean_dynamic_gap=0.15,
        max_key_phase_dynamic_gap=0.25,
        expected_environment_fingerprint=("2" * 64 if required else None),
        expected_rollout_manifest_fingerprint=("1" * 64 if required else None),
    )
    if state == "invalid_requirement":
        requirement.pop("schema_version")
    manifest["selection"]["dynamic_coverage_gate"] = requirement

    actual_candidate = candidate_basis_fingerprint(
        BASIS_MATRIX,
        muscle_names=NAMES,
        signal_kind=EXCITATION_SIGNAL_KIND,
        region="whole_body",
    )
    report_candidate = "f" * 64 if state == "forged_candidate" else actual_candidate
    dynamic_report = None
    validation_error = None
    if state not in {"missing", "optional_missing"}:
        dynamic_report = _dynamic_coverage_report(
            candidate_fingerprint=report_candidate,
            passed=state != "failed",
        )
    if state == "wrong_environment":
        dynamic_report["environment_fingerprint"] = "3" * 64
        dynamic_report["artifact_fingerprint"] = dynamic_coverage_artifact_fingerprint(dynamic_report)
    if state == "wrong_rollout":
        dynamic_report["rollout_manifest_fingerprint"] = "4" * 64
        dynamic_report["artifact_fingerprint"] = dynamic_coverage_artifact_fingerprint(dynamic_report)
    if state == "invalid_evidence":
        dynamic_report = None
        validation_error = "dynamic coverage evidence kind is invalid"

    manifest["selected_metrics"].update(
        {
            "candidate_basis_fingerprint": report_candidate,
            "dynamic_coverage_required": required,
            "dynamic_coverage": dynamic_report,
            "dynamic_coverage_validation_error": validation_error,
        }
    )


def _artifacts(
    tmp_path,
    *,
    residual: bool = False,
    selection_reason=None,
    dynamic_coverage_state: str | None = None,
):
    reason = selection_reason or "smallest_rank_meeting_all_vaf_and_stability_gates"
    manifest = _basis_manifest(rank=2, selection_reason=reason)
    if dynamic_coverage_state is not None:
        _bind_dynamic_coverage_contract(manifest, state=dynamic_coverage_state)
    basis = save_synergy_basis(
        tmp_path / "basis",
        basis=BASIS_MATRIX,
        muscle_names=NAMES,
        manifest=manifest,
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
            "schema_version": ACTION_SCHEMA_VERSION,
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
    assert "synergy_preclip_out_of_bounds_fraction" in result[4]
    assert "synergy_clip_correction_rms" in result[4]
    assert result[5].info["state_existing"] == 2.0
    assert "synergy_decoded_excitation_rms" in result[5].info
    np.testing.assert_allclose(np.asarray(base.last_body_action), output.body_action)
    assert np.all(np.asarray(output.physical_excitation) >= 0.0)
    assert np.all(np.asarray(output.physical_excitation) <= 1.0)
    np.testing.assert_allclose(output.preclip_excitation, output.physical_excitation)


def test_reset_and_step_keep_synergy_info_pytree_structure_for_jax_scan(tmp_path):
    basis, stats, _ = _artifacts(tmp_path)
    base = _MockBodyEnv()

    def reset_with_info(_key):
        return jnp.zeros(4), _MockState(
            step=jnp.asarray(0),
            info={"state_existing": jnp.asarray(2.0)},
        )

    base.reset = reset_with_info
    base.reset_to = lambda key, _traj_idx: reset_with_info(key)
    wrapper = SynergyActionWrapper(base, _config(basis, stats))
    _, reset_state = wrapper.reset(jax.random.PRNGKey(0))
    raw = jnp.zeros(wrapper.action_dim, dtype=jnp.float32)
    *_, next_state = wrapper.step(reset_state, raw)

    assert set(reset_state.info) == set(next_state.info)
    assert set(wrapper._reset_metrics) <= set(reset_state.info)
    assert all(float(reset_state.info[name]) == 0.0 for name in wrapper._reset_metrics)

    def scan_body(state, _):
        *_, scanned_state = wrapper.step(state, raw)
        return scanned_state, None

    scanned_state, _ = jax.jit(lambda state: jax.lax.scan(scan_body, state, None, length=2))(reset_state)
    assert set(scanned_state.info) == set(reset_state.info)


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


def test_preclip_diagnostics_report_clipping_without_losing_preclip_signal(tmp_path):
    basis, stats, _ = _artifacts(tmp_path)
    interface = SynergyActionWrapper(
        _MockBodyEnv(),
        _config(basis, stats),
    ).action_interface
    decoded = interface.decode(jnp.zeros(2, dtype=jnp.float32))
    preclip = jnp.asarray([-0.2, 0.5, 1.3], dtype=jnp.float32)
    clipped = jnp.clip(preclip, 0.0, 1.0)
    diagnostic_output = decoded._replace(
        preclip_excitation=preclip,
        physical_excitation=clipped,
    )

    metrics = interface.metrics(diagnostic_output)

    assert float(metrics["synergy_preclip_excitation_rms"]) == pytest.approx(np.sqrt((0.2**2 + 0.5**2 + 1.3**2) / 3.0))
    assert float(metrics["synergy_preclip_out_of_bounds_fraction"]) == pytest.approx(2.0 / 3.0)
    assert float(metrics["synergy_clip_correction_rms"]) == pytest.approx(np.sqrt((0.2**2 + 0.3**2) / 3.0))
    np.testing.assert_array_equal(diagnostic_output.preclip_excitation, preclip)


def test_structured_residual_adds_only_low_rank_bounded_direction(tmp_path):
    basis, stats, residual = _artifacts(tmp_path, residual=True)
    action_config = _config(basis, stats, residual)
    wrapper = SynergyActionWrapper(_MockBodyEnv(), action_config)
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

    exp = OmegaConf.create(
        {
            "action_representation": OmegaConf.to_container(action_config, resolve=True),
            "init_std": 0.35,
        }
    )
    assert apply_policy_interface_wrappers(wrapper, exp) is wrapper
    assert exp.body_synergy_contract.mode == "fixed_synergy_residual"
    assert exp.body_synergy_contract.residual_dim == 1
    assert exp.body_synergy_contract.residual_basis_fingerprint == residual.fingerprint


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
    with pytest.raises(ValueError, match="full_354 requires exactly 354"):
        apply_policy_interface_wrappers(base, disabled)
    legacy_native = OmegaConf.create({})
    assert apply_policy_interface_wrappers(base, legacy_native) is base

    exp = OmegaConf.create(
        {
            "action_representation": OmegaConf.to_container(_config(basis, stats), resolve=True),
            "init_std": 0.35,
        }
    )
    wrapped = apply_policy_interface_wrappers(base, exp)
    assert isinstance(wrapped, SynergyActionWrapper)
    assert wrapped.info.action_space.shape == (2,)
    assert exp.action_representation.mode == "fixed_synergy"
    assert exp.action_manifest.physical_action_interface_hash == wrapped.physical_action_interface_hash
    assert exp.action_manifest.exploration.kind == "configured_policy_std_v1"
    assert exp.body_synergy_contract.mode == "fixed_synergy"
    assert exp.body_synergy_contract.basis_fingerprint == basis.fingerprint
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


def test_numerical_basis_gates_are_revalidated_from_the_saved_decoder(tmp_path):
    stored_basis = BASIS_MATRIX.astype(np.float32).astype(np.float64)
    condition_number = float(np.linalg.cond(stored_basis))
    effective_rank_fraction = float(np.linalg.matrix_rank(stored_basis) / stored_basis.shape[1])
    manifest = _basis_manifest(
        rank=2,
        selection_reason="smallest_rank_meeting_all_vaf_and_stability_gates",
    )
    manifest["selection"]["thresholds"].update(
        {
            "max_basis_condition_number": condition_number + 1.0,
            "min_effective_rank_fraction": 1.0,
        }
    )
    manifest["selected_metrics"]["numerical_conditioning"] = {
        "basis_condition_number": condition_number,
        "effective_rank_fraction": effective_rank_fraction,
    }
    basis = save_synergy_basis(
        tmp_path / "numerically_gated_basis",
        basis=BASIS_MATRIX,
        muscle_names=NAMES,
        manifest=manifest,
    )
    stats = save_coefficient_statistics(
        basis.path / "coefficient_stats.npz",
        np.asarray([[0.10, 0.12], [0.20, 0.22]], dtype=np.float64),
        basis_fingerprint=basis.fingerprint,
    )
    assert SynergyActionWrapper(_MockBodyEnv(), _config(basis, stats)).info.action_space.shape == (2,)

    rejected = _basis_manifest(
        rank=2,
        selection_reason="smallest_rank_meeting_all_vaf_and_stability_gates",
    )
    rejected["selection"]["thresholds"].update(
        {
            "max_basis_condition_number": condition_number / 2.0,
            "min_effective_rank_fraction": 1.0,
        }
    )
    rejected["selected_metrics"]["numerical_conditioning"] = {
        "basis_condition_number": condition_number,
        "effective_rank_fraction": effective_rank_fraction,
    }
    rejected_basis = save_synergy_basis(
        tmp_path / "numerically_rejected_basis",
        basis=BASIS_MATRIX,
        muscle_names=NAMES,
        manifest=rejected,
    )
    rejected_stats = save_coefficient_statistics(
        rejected_basis.path / "coefficient_stats.npz",
        np.asarray([[0.10, 0.12], [0.20, 0.22]], dtype=np.float64),
        basis_fingerprint=rejected_basis.fingerprint,
    )
    with pytest.raises(ValueError, match="eligibility differs from recomputed gates"):
        SynergyActionWrapper(
            _MockBodyEnv(),
            _config(rejected_basis, rejected_stats),
        )


def test_required_dynamic_coverage_is_revalidated_against_saved_basis(tmp_path):
    basis, stats, _ = _artifacts(tmp_path, dynamic_coverage_state="passed")

    wrapper = SynergyActionWrapper(_MockBodyEnv(), _config(basis, stats))

    assert wrapper.action_manifest["basis_fingerprint"] == basis.fingerprint


@pytest.mark.parametrize(
    "state",
    [
        "missing",
        "invalid_evidence",
        "failed",
        "forged_candidate",
        "wrong_environment",
        "wrong_rollout",
    ],
)
def test_required_dynamic_coverage_failure_cannot_forge_rank_eligibility(
    tmp_path,
    state,
):
    basis, stats, _ = _artifacts(tmp_path, dynamic_coverage_state=state)

    with pytest.raises(ValueError, match="eligibility differs from recomputed gates"):
        SynergyActionWrapper(_MockBodyEnv(), _config(basis, stats))


def test_dynamic_coverage_requirement_schema_is_validated_exactly(tmp_path):
    basis, stats, _ = _artifacts(
        tmp_path,
        dynamic_coverage_state="invalid_requirement",
    )

    with pytest.raises(ValueError, match="dynamic-coverage requirement is invalid"):
        SynergyActionWrapper(_MockBodyEnv(), _config(basis, stats))


def test_optional_or_absent_dynamic_coverage_preserves_offline_gate_behavior(tmp_path):
    optional_basis, optional_stats, _ = _artifacts(
        tmp_path / "optional",
        dynamic_coverage_state="optional_missing",
    )
    legacy_basis, legacy_stats, _ = _artifacts(tmp_path / "legacy")

    assert SynergyActionWrapper(
        _MockBodyEnv(),
        _config(optional_basis, optional_stats),
    ).info.action_space.shape == (2,)
    assert SynergyActionWrapper(
        _MockBodyEnv(),
        _config(legacy_basis, legacy_stats),
    ).info.action_space.shape == (2,)


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
    assert interface.action_manifest["primitive_source_binding"] == {
        **binding,
        "runtime_model_compatibility": "exact_runtime_model",
    }
    assert interface.action_manifest["runtime_model_hash"] == "4" * 64

    with pytest.raises(ValueError, match="model hash differs"):
        build_early_synergy_action_interface(
            config,
            expected_actuator_names=NAMES,
            runtime_ctrlrange=ctrlrange,
            runtime_model_hash="5" * 64,
        )

    portable_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    portable_config.primitive_runtime_model_compatibility = "portable_body_action_abi"
    stage1 = build_early_synergy_action_interface(
        portable_config,
        expected_actuator_names=NAMES,
        runtime_ctrlrange=ctrlrange,
        runtime_model_hash="4" * 64,
    )
    stage2 = build_early_synergy_action_interface(
        portable_config,
        expected_actuator_names=NAMES,
        runtime_ctrlrange=ctrlrange,
        runtime_model_hash="5" * 64,
    )
    assert stage1.action_manifest["primitive_source_binding"] == {
        **binding,
        "runtime_model_compatibility": "portable_body_action_abi",
    }
    assert (
        stage1.body_synergy_contract.portable_decoder_core_fingerprint
        == stage2.body_synergy_contract.portable_decoder_core_fingerprint
    )
    assert (
        stage1.body_synergy_contract.stage_runtime_binding_fingerprint
        != stage2.body_synergy_contract.stage_runtime_binding_fingerprint
    )
    stage1.body_synergy_contract.assert_portable_compatible(stage2.body_synergy_contract)
    with pytest.raises(ValueError, match="exact runtime bindings"):
        stage1.body_synergy_contract.assert_exact_runtime_compatible(
            stage2.body_synergy_contract,
            require_complete=True,
        )
    with pytest.raises(ValueError, match="runtime MuJoCo model hash"):
        stage1.body_synergy_contract.validate_runtime(
            actuator_names=NAMES,
            ctrlrange=ctrlrange,
            runtime_model_hash="5" * 64,
            require_model_hash=True,
        )

    changed_ctrlrange = ctrlrange.copy()
    changed_ctrlrange[0, 1] = 0.9
    with pytest.raises(ValueError, match="exactly \\[0,1\\]"):
        build_early_synergy_action_interface(
            portable_config,
            expected_actuator_names=NAMES,
            runtime_ctrlrange=changed_ctrlrange,
            runtime_model_hash="5" * 64,
        )
    with pytest.raises(ValueError, match="actuator schema hash mismatch"):
        build_early_synergy_action_interface(
            portable_config,
            expected_actuator_names=tuple(reversed(NAMES)),
            runtime_ctrlrange=ctrlrange[::-1],
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
        COVERAGE_PROXY_ARTIFACT_KIND,
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
        "producer_artifact_kind": COVERAGE_PROXY_ARTIFACT_KIND,
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

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]


def _compose(name: str):
    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "fullbody")):
        return compose(config_name=f"config_specific_task/stage1_body/{name}")


def _plain(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def test_bootstrap_configs_are_isolated_from_formal_coverage_gated_arms():
    cases = (
        (
            "conf_fullbody_chinajump_early_synergy_bootstrap",
            "conf_fullbody_chinajump_early_synergy",
            "chinajump_root_control_v2_b0_early_synergy_bootstrap_excitation_v2",
            False,
        ),
        (
            "conf_fullbody_chinajump_early_synergy_bootstrap_asi",
            "conf_fullbody_chinajump_early_synergy_asi",
            "chinajump_root_control_v2_b1_early_synergy_bootstrap_asi_excitation_v2",
            True,
        ),
    )

    bootstrap_run_ids: list[str] = []
    for bootstrap_name, formal_name, expected_run_id, asi_enabled in cases:
        bootstrap = _compose(bootstrap_name)
        formal = _compose(formal_name)
        bootstrap_run_ids.append(str(bootstrap.experiment.run_id))

        assert bootstrap.experiment.run_id == expected_run_id
        assert bootstrap.experiment.run_id != formal.experiment.run_id
        assert bootstrap.experiment.resume_from is None
        assert bootstrap.experiment.total_timesteps == formal.experiment.total_timesteps
        assert bootstrap.experiment.asi.enabled is asi_enabled
        assert bootstrap.config_status.status == "experimental"
        assert bootstrap.config_status.canonical is False
        assert bootstrap.config_status.allow_nonproduction_runtime is False
        assert bootstrap.config_status.readiness == "bootstrap_only"
        assert list(bootstrap.config_status.evidence_limitations) == [
            "no_independent_chinajump_target_control_coverage"
        ]

        action = bootstrap.experiment.action_representation
        formal_action = formal.experiment.action_representation
        assert action.enabled is True
        assert action.mode == "fixed_synergy"
        assert action.bootstrap_without_target_coverage is True
        assert action.require_coverage_gate is False
        assert action.require_producer_bound_coverage is False
        assert action.coverage_gate_path == ""
        assert action.expected_coverage_gate_fingerprint == ""
        assert action.expected_coverage_proxy_fingerprint == ""
        assert action.require_phase_conditioned_coverage is False

        # Bootstrap still uses the same strict primitive W/source/runtime ABI
        # contract as its formal counterpart.  Only target-coverage evidence is
        # intentionally absent.
        for key in (
            "basis_path",
            "expected_basis_fingerprint",
            "expected_underlying_action_dim",
            "expected_actuator_schema_hash",
            "require_runtime_ctrlrange_binding",
            "max_policy_action_dim",
            "require_primitive_source_contract",
            "primitive_source_manifest_path",
            "expected_primitive_source_manifest_fingerprint",
            "expected_target_skill_id",
            "expected_excluded_target_motion_paths",
            "required_selection_thresholds",
            "require_all_basis_gates",
            "forbid_fallback_selected_basis",
            "coefficient_transform",
            "tonic_baseline",
            "residual",
            "exploration",
        ):
            assert _plain(action[key]) == _plain(formal_action[key])

    assert len(set(bootstrap_run_ids)) == len(bootstrap_run_ids)

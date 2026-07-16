from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from musclemimic.synergy.action_interface import (
    build_early_synergy_action_interface,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_chinajump_qc10_config_binds_existing_accepted_caches():
    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "fullbody")):
        cfg = compose(
            config_name=(
                "config_specific_task/stage1_body/"
                "conf_fullbody_chinajump_optimized_qc10"
            )
        )

    train = list(cfg.experiment.task_factory.params.amass_dataset_conf.rel_dataset_path)
    val = list(cfg.experiment.validation.amass_dataset_conf.rel_dataset_path)
    assert len(train) == 8
    assert len(val) == 2
    assert not set(train) & set(val)
    assert cfg.experiment.run_id == "chinajump_optimized_qc10_stage1_body_v1"
    assert cfg.experiment.training_action == "ChinaJump"
    assert cfg.experiment.training_source.source_fps == 60
    assert cfg.experiment.training_source.cache_fps == 100
    assert cfg.experiment.total_timesteps == 320_000_000
    assert cfg.experiment.env_params.disable_fingers is True
    assert cfg.experiment.validation.cover_all_trajectories is True
    assert cfg.experiment.promotion.auto_stop is True
    assert cfg.experiment.promotion.require_visual_validation_clips == 2

    for motion in train + val:
        cache = REPO_ROOT / "datasets" / f"{motion}.npz"
        assert cache.is_file(), cache
        with np.load(cache, allow_pickle=True) as data:
            assert float(data["frequency"]) == 100.0
            assert np.isfinite(data["qpos"]).all()
            assert np.isfinite(data["qvel"]).all()


def test_chinajump_root_control_v2_has_explicit_train_and_validation_guards():
    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "fullbody")):
        cfg = compose(
            config_name=(
                "config_specific_task/stage1_body/"
                "conf_fullbody_chinajump_root_control_v2"
            )
        )

    assert cfg.experiment.run_id == "chinajump_root_control_v2_stage1_body"
    assert cfg.experiment.total_timesteps == 640_000_000
    assert cfg.experiment.promotion.auto_stop is False
    reward = cfg.experiment.env_params.reward_params
    assert reward.root_orientation_w_sum == 0.15
    assert reward.root_orientation_w_exp == 8.0
    assert reward.root_ang_vel_w_sum == 0.15
    assert reward.root_ang_vel_w_exp == 0.5
    max_dense_reward = (
        reward.qpos_w_sum
        + reward.qvel_w_sum
        + reward.root_pos_w_sum
        + reward.root_vel_w_sum
        + reward.rpos_w_sum
        + reward.rquat_w_sum
        + 2 * reward.rvel_w_sum
        + reward.root_orientation_w_sum
        + reward.root_ang_vel_w_sum
    )
    assert max_dense_reward == 1.21

    train_terminal = cfg.experiment.env_params.terminal_state_params
    val_terminal = cfg.experiment.validation.terminal_state_params
    for terminal in (train_terminal, val_terminal):
        assert terminal.root_deviation_threshold == 0.5
        assert terminal.root_orientation_threshold == 0.5
        assert terminal.root_angular_velocity_error_threshold == 3.0
    assert (
        cfg.experiment.validation.terminal_state_type
        == "MeanRelativeSiteDeviationWithRootTerminalStateHandler"
    )


CHINAJUMP_STAGE1_ABLATIONS = {
    "conf_fullbody_chinajump_full_asi": {
        "id": "F1",
        "run_id": "chinajump_root_control_v2_f1_full_asi",
        "asi": True,
        "mode": None,
    },
    "conf_fullbody_chinajump_early_synergy": {
        "id": "S0",
        "run_id": "chinajump_root_control_v2_s0_early_synergy",
        "asi": False,
        "mode": "fixed_synergy",
    },
    "conf_fullbody_chinajump_early_synergy_asi": {
        "id": "S1",
        "run_id": "chinajump_root_control_v2_s1_early_synergy_asi",
        "asi": True,
        "mode": "fixed_synergy",
    },
    "conf_fullbody_chinajump_early_synergy_residual": {
        "id": "SR0",
        "run_id": "chinajump_root_control_v2_sr0_early_synergy_residual",
        "asi": False,
        "mode": "fixed_synergy_residual",
    },
    "conf_fullbody_chinajump_early_synergy_residual_asi": {
        "id": "SR1",
        "run_id": "chinajump_root_control_v2_sr1_early_synergy_residual_asi",
        "asi": True,
        "mode": "fixed_synergy_residual",
    },
}


def _compose_chinajump(name: str):
    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "fullbody")):
        return compose(config_name=f"config_specific_task/stage1_body/{name}")


def _plain(value):
    return OmegaConf.to_container(value, resolve=True)


def test_chinajump_stage1_ablation_configs_are_fair_and_use_fresh_run_ids():
    baseline = _compose_chinajump("conf_fullbody_chinajump_root_control_v2")
    configs = {
        name: _compose_chinajump(name)
        for name in CHINAJUMP_STAGE1_ABLATIONS
    }

    run_ids = [cfg.experiment.run_id for cfg in configs.values()]
    assert len(run_ids) == len(set(run_ids))
    assert baseline.experiment.run_id not in run_ids

    for name, cfg in configs.items():
        expected = CHINAJUMP_STAGE1_ABLATIONS[name]
        assert cfg.experiment.run_id == expected["run_id"]
        assert cfg.experiment.resume_from is None
        assert cfg.experiment.total_timesteps == 640_000_000
        assert cfg.config_status.status == "experimental"
        assert cfg.config_status.canonical is False
        assert cfg.config_status.allow_nonproduction_runtime is False

        # These are the controlled quantities for the F1/S0/S1/SR0/SR1
        # comparison.  ASI and the action interface are the only differences.
        assert _plain(cfg.experiment.task_factory.params.amass_dataset_conf) == _plain(
            baseline.experiment.task_factory.params.amass_dataset_conf
        )
        assert _plain(cfg.experiment.validation.amass_dataset_conf) == _plain(
            baseline.experiment.validation.amass_dataset_conf
        )
        assert _plain(cfg.experiment.env_params.reward_params) == _plain(
            baseline.experiment.env_params.reward_params
        )
        assert cfg.experiment.env_params.terminal_state_type == (
            baseline.experiment.env_params.terminal_state_type
        )
        assert _plain(cfg.experiment.env_params.terminal_state_params) == _plain(
            baseline.experiment.env_params.terminal_state_params
        )
        assert cfg.experiment.validation.terminal_state_type == (
            baseline.experiment.validation.terminal_state_type
        )
        assert _plain(cfg.experiment.validation.terminal_state_params) == _plain(
            baseline.experiment.validation.terminal_state_params
        )
        assert _plain(cfg.experiment.promotion) == _plain(baseline.experiment.promotion)

        assert cfg.experiment.asi.enabled is expected["asi"]
        assert cfg.experiment.adaptive_sampling.enabled is False
        assert cfg.experiment.adaptive_termination.enabled is False
        assert cfg.experiment.reward_curriculum.enabled is False
        assert cfg.experiment.get("hard_state_mining") is None

        mode = expected["mode"]
        action = cfg.experiment.get("action_representation")
        if mode is None:
            assert action is None or action.get("enabled", False) is False
            continue
        assert action.enabled is True
        assert action.schema_version == "early_synergy_action_v1"
        assert action.mode == mode


def test_chinajump_early_synergy_configs_bind_fail_closed_artifact_contract(
    monkeypatch,
):
    artifact_env = (
        "MUSCLEMIMIC_CHINAJUMP_SYNERGY_BASIS",
        "MUSCLEMIMIC_CHINAJUMP_SYNERGY_BASIS_FINGERPRINT",
        "MUSCLEMIMIC_CHINAJUMP_SYNERGY_COVERAGE_GATE",
        "MUSCLEMIMIC_CHINAJUMP_SYNERGY_COVERAGE_GATE_FINGERPRINT",
        "MUSCLEMIMIC_CHINAJUMP_SYNERGY_PROXY_FINGERPRINT",
        "MUSCLEMIMIC_CHINAJUMP_PRIMITIVE_SOURCE_MANIFEST",
        "MUSCLEMIMIC_CHINAJUMP_PRIMITIVE_SOURCE_FINGERPRINT",
        "MUSCLEMIMIC_CHINAJUMP_SYNERGY_COEFFICIENT_STATS_FINGERPRINT",
        "MUSCLEMIMIC_CHINAJUMP_SYNERGY_RESIDUAL_BASIS",
        "MUSCLEMIMIC_CHINAJUMP_SYNERGY_RESIDUAL_FINGERPRINT",
    )
    for variable in artifact_env:
        monkeypatch.delenv(variable, raising=False)

    expected_actuator_hash = (
        "a3db62371f17eaad6332f5f9076cada0d86cad0053af8aa7748479630054c68e"
    )
    for name, expected in CHINAJUMP_STAGE1_ABLATIONS.items():
        if expected["mode"] is None:
            continue
        action = _compose_chinajump(name).experiment.action_representation
        assert action.basis_path == ""
        assert action.expected_basis_fingerprint == ""
        assert action.coverage_gate_path == ""
        assert action.expected_coverage_gate_fingerprint == ""
        assert action.expected_coverage_proxy_fingerprint == ""
        assert action.expected_underlying_action_dim == 354
        assert action.max_policy_action_dim == 64
        assert action.expected_actuator_schema_hash == expected_actuator_hash
        assert action.require_all_basis_gates is True
        assert action.forbid_fallback_selected_basis is True
        assert action.require_coverage_gate is True
        assert action.require_producer_bound_coverage is True
        assert action.require_phase_conditioned_coverage is True
        assert action.required_coverage_phase_schema_fingerprint == (
            "b184d8791731cf33ea96012bdadce244c3b120c3945b35c7983a69d93adc1fab"
        )
        assert action.min_required_coverage_phases == 4
        assert list(action.required_coverage_thresholds.required_phase_ids) == [1, 2, 3, 4]
        assert action.require_primitive_source_contract is True
        assert action.primitive_source_manifest_path == ""
        assert action.expected_primitive_source_manifest_fingerprint == ""
        assert action.coefficient_transform.expected_stats_fingerprint == ""
        assert action.tonic_baseline.kind == "zero"
        assert action.tonic_baseline.learned_full_dimensional is False
        assert action.exploration.calibrate_in_physical_space is True
        assert action.exploration.target_initial_excitation_rms == 0.08
        assert action.exploration.std_mode == "per_dimension"
        assert action.exploration.residual_std_scale == 0.25

        residual = action.residual
        expected_residual = expected["mode"] == "fixed_synergy_residual"
        assert residual.enabled is expected_residual
        if expected_residual:
            assert residual.require_fit_contract is True
            assert _plain(residual.required_fit_thresholds) == {
                "min_validation_residual_energy_reduction": 0.01,
                "min_group_validation_residual_energy_reduction": 0.01,
                "max_validation_coordinate_saturation_fraction": 0.75,
            }
            assert residual.min_dimension == 4
            assert residual.max_dimension == 12
            assert residual.basis_path == ""
            assert residual.expected_fingerprint == ""
            assert residual.alpha == 0.03
            assert residual.alpha_schedule.enabled is False
            assert residual.alpha_schedule.kind == "constant_phase_a"

    # Hydra composition is intentionally artifact-independent, but a runtime
    # wrapper may never silently fall back when production artifacts are unset.
    action = _compose_chinajump(
        "conf_fullbody_chinajump_early_synergy"
    ).experiment.action_representation
    action.expected_underlying_action_dim = 1
    action.expected_actuator_schema_hash = ""
    with pytest.raises(ValueError, match="requires basis_path"):
        build_early_synergy_action_interface(
            action,
            expected_actuator_names=("test_muscle",),
        )

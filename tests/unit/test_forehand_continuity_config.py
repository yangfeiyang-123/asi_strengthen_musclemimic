"""Hydra contracts for opt-in forehand fascicle-continuity modes."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError

ROOT = Path(__file__).resolve().parents[2]
FULLBODY = ROOT / "fullbody"
CURATED_FINGERPRINT = "c044f7d4b1d037c314cc04ef209f3dbb89e652935cf3063a30b38881fb255d27"
GRAPH_FINGERPRINT = "fed541d4bbf0cf5a63e1db82bb988219f412c7614ecda9f2d7ac301fb7ca90e5"


def _compose(name: str):
    with initialize_config_dir(version_base=None, config_dir=str(FULLBODY)):
        return compose(config_name=f"config_specific_task/stage1_body/{name}")


def test_default_forehand_reward_keeps_continuity_off():
    config = _compose("conf_fullbody_forehand_clear_body_local")
    consistency = config.experiment.env_params.reward_params.intra_muscle_consistency

    assert consistency.mode == "off"
    assert consistency.coefficient == 0.0
    assert consistency.taxonomy_path is None
    assert consistency.diagnostic_graph_path is None
    assert consistency.candidate_graph_path is None
    assert consistency.continuity_path is None


def test_diagnostics_config_is_fresh_pinned_and_reward_neutral():
    baseline = _compose("conf_fullbody_forehand_clear_early_unified_synergy_v4")
    config = _compose("conf_fullbody_forehand_clear_early_unified_synergy_v4_continuity_diag")
    experiment = config.experiment
    consistency = experiment.env_params.reward_params.intra_muscle_consistency

    assert experiment.run_id == "forehand_clear_stage1_early_unified_synergy_v4_contdiag_s0"
    assert experiment.auto_resume is False
    assert experiment.resume_from is None
    assert experiment.total_timesteps == baseline.experiment.total_timesteps
    assert consistency.mode == "diagnostics"
    assert consistency.signal == "activation"
    assert consistency.coefficient == 0.0
    assert consistency.expected_taxonomy_fingerprint == CURATED_FINGERPRINT
    assert consistency.expected_diagnostic_graph_fingerprint == GRAPH_FINGERPRINT
    assert consistency.candidate_graph_path is None
    assert consistency.candidate_reward_enabled is False
    assert consistency.runtime_compatibility == "portable_muscle_channel_abi"
    assert consistency.require_verified_training_chains is True


@pytest.mark.parametrize(
    "name",
    [
        "conf_fullbody_forehand_clear_early_unified_synergy_v4_continuity_reward",
        "conf_fullbody_forehand_clear_full354_continuity_reward",
    ],
)
def test_reward_ablation_configs_fail_resolution_without_verified_evidence(
    monkeypatch,
    name,
):
    for variable in (
        "MUSCLEMIMIC_CONTINUITY_RELEASE",
        "MUSCLEMIMIC_CONTINUITY_RELEASE_FINGERPRINT",
    ):
        monkeypatch.delenv(variable, raising=False)
    config = _compose(name)
    consistency = config.experiment.env_params.reward_params.intra_muscle_consistency

    assert config.experiment.auto_resume is False
    assert config.experiment.resume_from is None
    assert consistency.mode == "reward"
    assert consistency.expected_taxonomy_fingerprint is None
    assert consistency.coefficient == 0.0
    with pytest.raises(InterpolationResolutionError):
        OmegaConf.to_container(consistency, resolve=True)


def test_reward_preset_never_points_at_the_provisional_graph():
    preset = OmegaConf.load(FULLBODY / "config_specific_task/presets/forehand_fascicle_continuity_reward_v1.yaml")
    consistency = preset.experiment.env_params.reward_params.intra_muscle_consistency
    smoke_gate = preset.experiment.continuity_smoke_gate
    smoke_execution = preset.experiment.training_smoke

    assert "MUSCLEMIMIC_CONTINUITY_RELEASE" in str(consistency._get_node("release_path"))
    assert "MUSCLEMIMIC_CONTINUITY_RELEASE_FINGERPRINT" in str(consistency._get_node("expected_release_fingerprint"))
    assert consistency.continuity_path is None
    assert consistency.candidate_graph_path is None
    assert consistency.coefficient == 0.0
    assert smoke_gate.required is True
    assert smoke_gate.max_age_hours == 24.0
    assert smoke_gate.require_clean_git is True
    assert "MUSCLEMIMIC_CONTINUITY_SMOKE_ARTIFACT" in str(smoke_gate._get_node("artifact_path"))
    assert smoke_execution.enabled is False


def test_complete_matched_ablation_matrix_has_three_fresh_seeds_per_condition():
    directory = FULLBODY / "config_specific_task/stage1_body/continuity_ablation_v1"
    files = sorted(directory.glob("conf_forehand_continuity_??_s?.yaml"))
    assert len(files) == 24
    expected = {
        f"{condition}_s{seed}" for condition in ("a0", "a1", "b0", "b1", "c0", "c1", "g0", "g1") for seed in range(3)
    }
    assert {path.stem.removeprefix("conf_forehand_continuity_") for path in files} == expected

    for path in files:
        suffix = path.stem.removeprefix("conf_forehand_continuity_")
        condition, raw_seed = suffix.split("_s")
        seed = int(raw_seed)
        config = _compose(f"continuity_ablation_v1/{path.stem}")
        experiment = config.experiment
        contract = experiment.continuity_ablation
        consistency = experiment.env_params.reward_params.intra_muscle_consistency

        assert experiment.run_id == f"forehand_clear_continuity_ablation_v1_{suffix}"
        assert experiment.auto_resume is False
        assert experiment.resume_from is None
        assert experiment.n_seeds == 1
        assert list(experiment.seeds) == [seed]
        assert experiment.validation.eval_seed == seed
        assert experiment.total_timesteps == 320_000_000
        assert contract.condition == condition.upper()
        assert contract.seed == seed
        assert contract.fresh_optimizer_required is True
        assert contract.parent_initialization_checkpoint is None
        reward_enabled = condition.endswith("1")
        assert contract.continuity_reward_enabled is reward_enabled
        assert consistency.mode == ("reward" if reward_enabled else "off")
        if condition.startswith("g"):
            assert experiment.action_representation.require_graph_regularization is True
            assert experiment.action_representation.forbid_graph_regularization is False
            assert experiment.action_representation.require_basis_factor_contract is True
            assert experiment.action_representation.required_basis_family == "graph_nmf"
            assert experiment.action_representation.require_raw_unit_basis_factor is True
        elif condition[0] in {"b", "c"}:
            assert experiment.action_representation.require_graph_regularization is False
            assert experiment.action_representation.forbid_graph_regularization is True
            assert experiment.action_representation.require_basis_factor_contract is True
            assert experiment.action_representation.required_basis_family == "standard_nmf"
            assert experiment.action_representation.require_raw_unit_basis_factor is True
        if condition.startswith("c"):
            assert experiment.action_representation.mode == "fixed_synergy_residual"
            assert experiment.action_representation.residual.enabled is True


@pytest.mark.parametrize("pair", ["a", "b", "c", "g"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ablation_pairs_keep_data_budget_promotion_and_seed_matched(pair, seed):
    baseline = _compose(f"continuity_ablation_v1/conf_forehand_continuity_{pair}0_s{seed}")
    rewarded = _compose(f"continuity_ablation_v1/conf_forehand_continuity_{pair}1_s{seed}")
    left = baseline.experiment
    right = rewarded.experiment

    assert OmegaConf.to_container(left.training_source, resolve=False) == OmegaConf.to_container(
        right.training_source,
        resolve=False,
    )
    assert OmegaConf.to_container(left.task_factory, resolve=False) == OmegaConf.to_container(
        right.task_factory,
        resolve=False,
    )
    assert OmegaConf.to_container(left.promotion, resolve=False) == OmegaConf.to_container(
        right.promotion,
        resolve=False,
    )
    assert left.total_timesteps == right.total_timesteps
    assert list(left.seeds) == list(right.seeds) == [seed]


def test_graph_nmf_ablation_preset_resolves_to_fail_closed_empty_artifacts(monkeypatch):
    for variable in (
        "MUSCLEMIMIC_FOREHAND_GRAPH_SYNERGY_BASIS",
        "MUSCLEMIMIC_FOREHAND_GRAPH_SYNERGY_BASIS_FINGERPRINT",
        "MUSCLEMIMIC_FOREHAND_GRAPH_COEFFICIENT_STATS",
        "MUSCLEMIMIC_FOREHAND_GRAPH_COEFFICIENT_STATS_FINGERPRINT",
        "MUSCLEMIMIC_FOREHAND_GRAPH_NMF_LINEAGE_FINGERPRINT",
        "MUSCLEMIMIC_FOREHAND_GRAPH_NMF_LAMBDA",
        "MUSCLEMIMIC_FOREHAND_GRAPH_NMF_LAMBDA_SELECTION_FINGERPRINT",
        "MUSCLEMIMIC_FOREHAND_GRAPH_CANDIDATE_FINGERPRINT",
        "MUSCLEMIMIC_FOREHAND_BG_BASIS_FACTOR_FINGERPRINT",
        "MUSCLEMIMIC_CONTINUITY_RELEASE_FINGERPRINT",
    ):
        monkeypatch.delenv(variable, raising=False)
    config = _compose("continuity_ablation_v1/conf_forehand_continuity_g0_s0")
    resolved = OmegaConf.to_container(config.experiment.action_representation, resolve=True)
    assert resolved["require_graph_regularization"] is True
    assert resolved["basis_path"] == ""
    assert resolved["expected_basis_fingerprint"] == ""
    assert resolved["expected_graph_regularization_lineage_fingerprint"] == ""
    assert resolved["expected_graph_regularization_lambda"] == ""
    assert resolved["expected_graph_lambda_selection_fingerprint"] == ""
    assert resolved["expected_graph_continuity_release_fingerprint"] == ""
    assert resolved["expected_basis_factor_contract_fingerprint"] == ""


def test_standard_nmf_ablation_uses_dedicated_raw_unit_artifacts(monkeypatch):
    for variable in (
        "MUSCLEMIMIC_FOREHAND_RAW_STANDARD_SYNERGY_BASIS",
        "MUSCLEMIMIC_FOREHAND_RAW_STANDARD_SYNERGY_BASIS_FINGERPRINT",
        "MUSCLEMIMIC_FOREHAND_RAW_STANDARD_COEFFICIENT_STATS",
        "MUSCLEMIMIC_FOREHAND_RAW_STANDARD_COEFFICIENT_STATS_FINGERPRINT",
        "MUSCLEMIMIC_FOREHAND_BG_BASIS_FACTOR_FINGERPRINT",
    ):
        monkeypatch.delenv(variable, raising=False)
    config = _compose("continuity_ablation_v1/conf_forehand_continuity_b0_s0")
    resolved = OmegaConf.to_container(config.experiment.action_representation, resolve=True)
    assert resolved["basis_path"] == ""
    assert resolved["expected_basis_fingerprint"] == ""
    assert resolved["coefficient_transform"]["stats_path"] == ""
    assert resolved["required_basis_family"] == "standard_nmf"
    assert resolved["require_raw_unit_basis_factor"] is True

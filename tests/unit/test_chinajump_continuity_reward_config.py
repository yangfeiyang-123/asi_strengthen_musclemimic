"""Hydra contract for release-gated ChinaJump continuity reward training."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
FULLBODY = REPO_ROOT / "fullbody"
BASELINE_CONFIG = "conf_fullbody_chinajump_early_synergy_bootstrap_continuity_diag_retry_v3"
REWARD_CONFIG = "conf_fullbody_chinajump_early_synergy_bootstrap_continuity_reward"
BASELINE_RUN_ID = "chinajump_root_control_v2_b0cd_early_synergy_bootstrap_contdiag_excitation_v3"
REWARD_RUN_ID = "chinajump_root_control_v2_b1_early_synergy_bootstrap_continuity_reward_v1"


def _compose(name: str, *, overrides: list[str] | None = None):
    with initialize_config_dir(version_base=None, config_dir=str(FULLBODY)):
        return compose(
            config_name=f"config_specific_task/stage1_body/{name}",
            overrides=overrides or [],
        )


def _plain(value):
    return OmegaConf.to_container(value, resolve=True)


def _reward_config(monkeypatch):
    monkeypatch.setenv("MUSCLEMIMIC_CONTINUITY_RELEASE", "/verified/continuity/release.json")
    monkeypatch.setenv("MUSCLEMIMIC_CONTINUITY_RELEASE_FINGERPRINT", "a" * 64)
    monkeypatch.setenv("MUSCLEMIMIC_CONTINUITY_SMOKE_ARTIFACT", "/verified/continuity/smoke.json")
    monkeypatch.setenv("MUSCLEMIMIC_CONTINUITY_SMOKE_ARTIFACT_FINGERPRINT", "b" * 64)
    return _compose(REWARD_CONFIG)


def test_chinajump_continuity_reward_is_release_backed_and_fresh(monkeypatch):
    config = _reward_config(monkeypatch)
    experiment = config.experiment
    consistency = experiment.env_params.reward_params.intra_muscle_consistency
    smoke = experiment.continuity_smoke_gate
    contract = experiment.continuity_reward_experiment_contract

    assert experiment.run_id == REWARD_RUN_ID
    assert config.wandb.name == REWARD_RUN_ID
    assert experiment.auto_resume is False
    assert experiment.resume_from is None
    assert experiment.asi.enabled is False

    assert consistency.mode == "reward"
    assert consistency.signal == "activation"
    assert consistency.release_path == "/verified/continuity/release.json"
    assert consistency.expected_release_fingerprint == "a" * 64
    assert consistency.coefficient == 0.0
    assert consistency.raw_penalty_clip is None
    assert consistency.taxonomy_path is None
    assert consistency.diagnostic_graph_path is None
    assert consistency.candidate_graph_path is None
    assert consistency.require_verified_training_chains is True

    assert smoke.required is True
    assert smoke.artifact_path == "/verified/continuity/smoke.json"
    assert smoke.expected_artifact_fingerprint == "b" * 64
    assert smoke.require_clean_git is True

    assert contract.condition == "B1"
    assert contract.matched_baseline_run_id == BASELINE_RUN_ID
    assert contract.continuity_reward_enabled is True
    assert contract.coefficient_source == "immutable_continuity_training_release"
    assert contract.verified_training_chains_required is True
    assert contract.fresh_optimizer_required is True
    assert contract.parent_initialization_checkpoint is None


def test_chinajump_continuity_reward_is_matched_to_b0cd_except_reward(monkeypatch):
    baseline = _compose(BASELINE_CONFIG).experiment
    reward = _reward_config(monkeypatch).experiment

    assert reward.total_timesteps == baseline.total_timesteps == 640_000_000
    assert reward.training_action == baseline.training_action == "ChinaJump"
    assert _plain(reward.training_source) == _plain(baseline.training_source)
    assert _plain(reward.task_factory.params.amass_dataset_conf) == _plain(
        baseline.task_factory.params.amass_dataset_conf
    )
    assert _plain(reward.validation.amass_dataset_conf) == _plain(baseline.validation.amass_dataset_conf)
    assert _plain(reward.env_params.terminal_state_params) == _plain(baseline.env_params.terminal_state_params)
    assert _plain(reward.validation.terminal_state_params) == _plain(baseline.validation.terminal_state_params)
    assert _plain(reward.promotion) == _plain(baseline.promotion)
    assert _plain(reward.action_representation) == _plain(baseline.action_representation)

    baseline_reward = _plain(baseline.env_params.reward_params)
    reward_reward = _plain(reward.env_params.reward_params)
    baseline_reward.pop("intra_muscle_consistency")
    reward_reward.pop("intra_muscle_consistency")
    assert reward_reward == baseline_reward


def test_chinajump_continuity_reward_has_unique_run_identity(monkeypatch):
    reward = _reward_config(monkeypatch)
    config_names = sorted(
        path.stem for path in (FULLBODY / "config_specific_task/stage1_body").glob("conf_fullbody_chinajump*.yaml")
    )
    run_ids = []
    for name in config_names:
        if name == REWARD_CONFIG:
            run_ids.append(reward.experiment.run_id)
        else:
            run_ids.append(_compose(name).experiment.run_id)
    assert len(run_ids) == len(set(run_ids))

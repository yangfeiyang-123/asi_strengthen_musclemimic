"""Hydra contract for reward-neutral ChinaJump continuity diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from musclemimic.runner.checkpointing import config_hash, write_manifest
from musclemimic.synergy.stage1_pipeline import _pipeline_config_contract

REPO_ROOT = Path(__file__).resolve().parents[2]
FULLBODY = REPO_ROOT / "fullbody"
CONFIG_NAME = "conf_fullbody_chinajump_early_synergy_bootstrap_continuity_diag"
RUN_ID = "chinajump_root_control_v2_b0cd_early_synergy_bootstrap_contdiag_excitation_v2"
RETRY_CONFIG_NAME = "conf_fullbody_chinajump_early_synergy_bootstrap_continuity_diag_retry_v3"
RETRY_RUN_ID = "chinajump_root_control_v2_b0cd_early_synergy_bootstrap_contdiag_excitation_v3"
CURATED_FINGERPRINT = "c044f7d4b1d037c314cc04ef209f3dbb89e652935cf3063a30b38881fb255d27"
GRAPH_FINGERPRINT = "fed541d4bbf0cf5a63e1db82bb988219f412c7614ecda9f2d7ac301fb7ca90e5"


def _compose(name: str):
    with initialize_config_dir(version_base=None, config_dir=str(FULLBODY)):
        return compose(config_name=f"config_specific_task/stage1_body/{name}")


def _plain(value):
    return OmegaConf.to_container(value, resolve=True)


def test_chinajump_fixed_w_bootstrap_continuity_diagnostics_contract():
    config = _compose(CONFIG_NAME)
    experiment = config.experiment
    consistency = experiment.env_params.reward_params.intra_muscle_consistency
    action = experiment.action_representation
    contract = experiment.continuity_diagnostics_contract

    assert experiment.run_id == RUN_ID
    assert experiment.auto_resume is False
    assert experiment.resume_from is None
    assert experiment.total_timesteps == 640_000_000
    assert experiment.asi.enabled is False
    assert experiment.adaptive_sampling.enabled is False
    assert experiment.adaptive_termination.enabled is False
    assert experiment.reward_curriculum.enabled is False

    assert config.config_status.status == "experimental"
    assert config.config_status.canonical is False
    assert config.config_status.readiness == "bootstrap_only"
    assert list(config.config_status.evidence_limitations) == [
        "no_independent_chinajump_target_control_coverage",
        "provisional_continuity_graph_diagnostics_only",
        "no_verified_intra_muscle_reward_claim",
    ]

    assert action.enabled is True
    assert action.mode == "fixed_synergy"
    assert action.bootstrap_without_target_coverage is True
    assert action.require_primitive_source_contract is True
    assert action.require_coverage_gate is False
    assert action.require_producer_bound_coverage is False
    assert action.require_phase_conditioned_coverage is False
    assert action.residual.enabled is False

    assert consistency.mode == "diagnostics"
    assert consistency.signal == "activation"
    assert consistency.coefficient == 0.0
    assert consistency.raw_penalty_clip is None
    assert consistency.taxonomy_path == ("configs/physiology/myofullbody_354_muscle_taxonomy_curated_v2.json")
    assert consistency.diagnostic_graph_path == ("configs/physiology/myofullbody_354_fascicle_continuity_v2.json")
    assert consistency.expected_taxonomy_fingerprint == CURATED_FINGERPRINT
    assert consistency.expected_diagnostic_graph_fingerprint == GRAPH_FINGERPRINT
    assert consistency.runtime_compatibility == "portable_muscle_channel_abi"
    assert consistency.method == "robust_fascicle_continuity_v1"
    assert consistency.candidate_graph_path is None
    assert consistency.candidate_reward_enabled is False
    assert consistency.continuity_path is None
    assert consistency.require_verified_training_chains is True

    assert contract.schema_version == "chinajump_fascicle_continuity_diagnostics_v1"
    assert contract.condition == "B0-CD"
    assert contract.action_mode == "fixed_synergy"
    assert contract.primitive_readiness == "bootstrap_only"
    assert contract.signal == "actual_mujoco_activation_via_actuator_actadr"
    assert contract.diagnostic_chain_count == 28
    assert contract.graph_review_status == "provisional"
    assert contract.continuity_reward_enabled is False
    assert contract.verified_imr_reward_claim is False
    assert contract.asi_enabled is False
    assert contract.fresh_optimizer_required is True
    assert contract.parent_initialization_checkpoint is None

    china_jump_configs = sorted(
        path.stem for path in (FULLBODY / "config_specific_task/stage1_body").glob("conf_fullbody_chinajump*.yaml")
    )
    resolved_run_ids = [_compose(name).experiment.run_id for name in china_jump_configs]
    assert len(resolved_run_ids) == len(set(resolved_run_ids))


def test_chinajump_continuity_diagnostics_preserves_bootstrap_experiment_contract():
    baseline = _compose("conf_fullbody_chinajump_early_synergy_bootstrap")
    diagnostic = _compose(CONFIG_NAME)
    left = baseline.experiment
    right = diagnostic.experiment

    assert right.run_id != left.run_id
    assert right.run_id != _compose("conf_fullbody_chinajump_early_synergy_bootstrap_asi").experiment.run_id
    assert right.total_timesteps == left.total_timesteps == 640_000_000
    assert right.training_action == left.training_action == "ChinaJump"
    assert _plain(right.training_source) == _plain(left.training_source)
    assert _plain(right.task_factory.params.amass_dataset_conf) == _plain(left.task_factory.params.amass_dataset_conf)
    assert _plain(right.validation.amass_dataset_conf) == _plain(left.validation.amass_dataset_conf)
    assert right.env_params.terminal_state_type == left.env_params.terminal_state_type
    assert _plain(right.env_params.terminal_state_params) == _plain(left.env_params.terminal_state_params)
    assert right.validation.terminal_state_type == left.validation.terminal_state_type
    assert _plain(right.validation.terminal_state_params) == _plain(left.validation.terminal_state_params)
    assert _plain(right.promotion) == _plain(left.promotion)
    assert _plain(right.action_representation) == _plain(left.action_representation)

    baseline_reward = _plain(left.env_params.reward_params)
    diagnostic_reward = _plain(right.env_params.reward_params)
    baseline_consistency = baseline_reward.pop("intra_muscle_consistency")
    diagnostic_consistency = diagnostic_reward.pop("intra_muscle_consistency")
    assert baseline_reward == diagnostic_reward
    assert baseline_consistency["mode"] == "off"
    assert diagnostic_consistency["mode"] == "diagnostics"
    assert baseline_consistency["coefficient"] == diagnostic_consistency["coefficient"] == 0.0


def test_chinajump_continuity_diagnostics_retry_has_fresh_identity_and_exact_contract():
    predecessor = _compose(CONFIG_NAME)
    retry = _compose(RETRY_CONFIG_NAME)

    assert retry.experiment.run_id == RETRY_RUN_ID
    assert retry.wandb.name == RETRY_RUN_ID
    assert retry.experiment.auto_resume is False
    assert retry.experiment.resume_from is None

    lineage = retry.experiment.launch_retry_contract
    assert lineage.schema_version == "chinajump_launch_retry_contract_v1"
    assert lineage.predecessor_run_id == RUN_ID
    assert lineage.predecessor_wandb_run_id == "e73krmmm"
    assert lineage.failure_class == "fixed_synergy_reset_step_info_pytree_mismatch"
    assert lineage.failed_before_first_ppo_update is True
    assert lineage.runtime_fix_source_sha256 == ("f3d972c87bcd9fa4913795a92335a4c27173d8c235f85bb7e65457b228f1424c")
    assert lineage.fresh_optimizer_required is True
    assert lineage.parent_initialization_checkpoint is None

    predecessor_experiment = _plain(predecessor.experiment)
    retry_experiment = _plain(retry.experiment)
    predecessor_experiment.pop("run_id")
    retry_experiment.pop("run_id")
    retry_experiment.pop("launch_retry_contract")
    assert retry_experiment == predecessor_experiment


def test_chinajump_continuity_diagnostics_uses_checked_in_provisional_assets():
    taxonomy_path = REPO_ROOT / "configs/physiology/myofullbody_354_muscle_taxonomy_curated_v2.json"
    graph_path = REPO_ROOT / "configs/physiology/myofullbody_354_fascicle_continuity_v2.json"
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    graph = json.loads(graph_path.read_text(encoding="utf-8"))

    assert taxonomy["taxonomy_fingerprint"] == CURATED_FINGERPRINT
    assert len(taxonomy["ordered_actuators"]) == 354
    assert taxonomy["hard_line_groups"] == []
    assert taxonomy["soft_compartment_groups"]
    assert all(group["training_enabled"] is False for group in taxonomy["soft_compartment_groups"])

    assert graph["graph_fingerprint"] == GRAPH_FINGERPRINT
    assert graph["taxonomy_binding"]["taxonomy_fingerprint"] == CURATED_FINGERPRINT
    assert len(graph["chains"]) == 28
    assert sum(len(chain["edges"]) for chain in graph["chains"]) == 140
    assert all(chain["review_status"] == "provisional" for chain in graph["chains"])
    assert all(chain["training_enabled"] is False for chain in graph["chains"])


def test_chinajump_continuity_diagnostics_is_explicit_in_run_manifest(tmp_path):
    experiment = _compose(CONFIG_NAME).experiment
    write_manifest(tmp_path, experiment, config_hash(experiment))
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    resolved = manifest["experiment_config"]
    consistency = resolved["env_params"]["reward_params"]["intra_muscle_consistency"]
    contract = resolved["continuity_diagnostics_contract"]

    assert resolved["run_id"] == RUN_ID
    assert consistency["mode"] == "diagnostics"
    assert consistency["signal"] == "activation"
    assert consistency["coefficient"] == 0.0
    assert consistency["candidate_reward_enabled"] is False
    assert contract["continuity_reward_enabled"] is False
    assert contract["verified_imr_reward_claim"] is False
    assert contract["asi_enabled"] is False
    assert "continuity_training_contract" not in manifest


def test_chinajump_continuity_diagnostics_is_a_stage1_bootstrap_config():
    config_name = f"config_specific_task/stage1_body/{CONFIG_NAME}"
    contract = _pipeline_config_contract(
        config_name,
        env_prefix="MUSCLEMIMIC_CHINAJUMP",
        readiness_mode="bootstrap",
    )

    assert contract["config_name"] == config_name
    assert contract["readiness_mode"] == "bootstrap"
    assert contract["expected_target_skill_id"] == "ChinaJump"
    assert contract["expected_underlying_action_dim"] == 354
    assert contract["max_policy_action_dim"] == 64
    assert contract["config_status"]["readiness"] == "bootstrap_only"
    assert "provisional_continuity_graph_diagnostics_only" in contract["config_status"]["evidence_limitations"]

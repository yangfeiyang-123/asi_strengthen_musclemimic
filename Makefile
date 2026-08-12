.PHONY: help install install-dev install-all sync-exact precommit-install check format lint test source-only asset-test gpu-smoke-test smoke ci clean

UV ?= $(shell if command -v uv >/dev/null 2>&1; then command -v uv; elif [ -x "$(HOME)/.local/bin/uv" ]; then printf '%s' "$(HOME)/.local/bin/uv"; else printf '%s' "uv"; fi)
VENV_BIN ?= .venv/bin
PYTHON ?= $(VENV_BIN)/python
PRECOMMIT ?= $(VENV_BIN)/pre-commit
PYTEST ?= $(VENV_BIN)/pytest
RUFF ?= $(VENV_BIN)/ruff
PYTEST_ARGS ?= -m "not integration"
SYNC_EXTRAS ?= --all-extras
SOURCE_ONLY_TESTS := \
	tests/source_only \
	tests/unit/test_json_contract.py \
	tests/unit/test_forehand_clear_ablation_report.py \
	tests/unit/test_ablation_report_v2.py \
	tests/unit/test_stage3_v2_contracts.py \
	tests/unit/test_stage3_paired_comparison.py \
	tests/unit/test_research_training_gates.py \
	tests/unit/test_forehand_clear_training_gates.py \
	tests/unit/test_forehand_clear_pipeline.py \
	tests/unit/test_event_reference_v2.py \
	tests/unit/test_contact_tracking_data.py \
	tests/unit/test_physical_distill_contract.py \
	tests/unit/test_physical_rollout_qc.py \
	tests/unit/test_synergy_core.py \
	tests/unit/test_synergy_fit.py \
	tests/unit/test_primitive_catalog.py \
	tests/unit/test_primitive_ingest.py \
	tests/unit/test_primitive_recording.py \
	tests/unit/test_primitive_manifest.py \
	tests/unit/test_chinajump_coverage_gate.py \
	tests/unit/test_chinajump_coverage_proxy.py \
	tests/unit/test_synergy_action_wrapper.py \
	tests/unit/test_synergy_exploration_scaling.py \
	tests/unit/test_synergy_residual_fit.py \
	tests/unit/test_chinajump_training_config.py \
	tests/unit/test_chinajump_synergy_bootstrap_config.py \
	tests/unit/test_stage1_synergy_pipeline.py \
	tests/unit/test_latent_synergy_decoder.py \
	tests/unit/test_latent_synergy_analysis.py \
	tests/unit/test_causal_rollout_artifact.py \
	tests/unit/test_causal_rollout_driver.py \
	tests/unit/test_stage2_causal_adapter.py \
	tests/unit/test_stage3_task_causal.py \
	tests/unit/test_emg_evaluation.py \
	tests/unit/test_emg_cohort_evaluation.py \
	tests/unit/test_jidian_emg_import.py \
	tests/unit/test_jidian_emg_mapping.py \
	tests/unit/test_physiology_contracts.py \
	tests/unit/test_physiology_taxonomy.py \
	tests/unit/test_physiology_taxonomy_v2.py \
	tests/unit/test_physiology_continuity_groups.py \
	tests/unit/test_continuity_loss_spec.py \
	tests/unit/test_continuity_candidate_graph.py \
	tests/unit/test_continuity_release.py \
	tests/unit/continuity_v3_fixtures.py \
	tests/unit/test_continuity_baseline_v3.py \
	tests/unit/test_continuity_calibration_v3.py \
	tests/unit/test_graph_nmf.py \
	tests/unit/test_graph_nmf_lambda_selection.py \
	tests/unit/test_continuity_training_smoke.py \
	tests/unit/test_fullbody_training_preflight.py \
	tests/unit/test_bc_checkpoint_roundtrip.py \
	tests/unit/test_ablation_evidence_builder.py \
	tests/unit/test_continuity_baseline_collection.py \
	tests/unit/test_continuity_ablation_report.py \
	tests/unit/test_intra_muscle_continuity.py \
	tests/unit/test_mimic_reward_continuity.py \
	tests/unit/test_forehand_continuity_config.py \
	tests/unit/test_metrics_handler_trajectory_binding.py \
	tests/unit/test_physiology_joint_report.py \
	tests/unit/test_physiology_evaluation.py \
	tests/unit/test_physiology_synergy_binding.py \
	tests/unit/test_stage3_signal_export.py \
	tests/unit/test_distill_dataset.py \
	tests/unit/test_emg_reference_tube.py \
	tests/unit/test_emg_anchor_loss.py \
	tests/unit/test_server_deployment.py
LINT_PATHS := \
	bimanual \
	tests/unit/test_enhanced_fullbody_terminal_handler.py \
	tests/unit/test_enhanced_fullbody_terminal_handler_integration.py \
	tests/test_muscle_observations.py \
	musclemimic/core/terminal_state_handler/enhanced_fullbody.py \
	musclemimic/core/terminal_state_handler/enhanced_bimanual.py \
	musclemimic/environments/humanoids/base_bimanual.py \
	musclemimic/environments/humanoids/bimanual.py \
	musclemimic/utils/metrics.py \
	tests/unit/test_metrics.py \
	loco_mujoco/smpl/retargeting.py
NEW_RESEARCH_LINT_PATHS := \
	analysis/latent_synergy \
	analysis/physiology_synergy \
	musclemimic/synergy \
	musclemimic/physiology \
	musclemimic/evaluation \
	musclemimic/badminton/data/event_qc.py \
	musclemimic/badminton/asi/contact_tracking_data.py \
	musclemimic/badminton/data/event_schema.py \
	musclemimic/badminton/data/racket_reference.py \
	musclemimic/badminton/data/event_lookup.py \
	musclemimic/badminton/data/reference_bundle.py \
	musclemimic/distill/physical.py \
	musclemimic/distill/physical_qc.py \
	musclemimic/distill/collect_teacher.py \
	fullbody/distill_collect.py \
	musclemimic/latent_muscle/analysis_export.py \
	musclemimic/latent_muscle/causal_rollout_artifact.py \
	musclemimic/latent_muscle/causal_rollout_driver.py \
	musclemimic/latent_muscle/stage2_causal_adapter.py \
	musclemimic/latent_muscle/decoder_factory.py \
	musclemimic/latent_muscle/synergy_decoder.py \
	musclemimic/badminton/racket_mass_curriculum.py \
	musclemimic/badminton/scripts/latent_synergy_sweep.py \
	musclemimic/badminton/scripts/run_incoming_shuttle_hit.py \
	musclemimic/badminton/stage3_paired_comparison.py \
	musclemimic/badminton/stage3_task_causal.py \
	musclemimic/badminton/scripts/build_forehand_clear_ablation_report.py \
	musclemimic/evaluation/physiology.py \
	fullbody/run_chinajump_synergy_pipeline.py \
	fullbody/run_forehand_clear_pipeline.py \
	fullbody/smoke_forehand_continuity_training.py \
	scripts/resolve_fullbody_training.py \
	scripts/build_training_asset_manifest.py \
	scripts/server_training_preflight.py \
	musclemimic/runner/continuity_smoke.py \
	musclemimic/runner/engine.py \
	musclemimic/runner/checkpointing.py \
	musclemimic/algorithms/ppo/runner.py \
	musclemimic/algorithms/ppo/checkpoint.py \
	musclemimic/algorithms/common/checkpoint_hooks.py \
	musclemimic/algorithms/common/checkpoint_manager.py \
	scripts/build_myofullbody_curated_taxonomy.py \
	scripts/build_myofullbody_fascicle_continuity.py \
	tests/unit/test_physiology_taxonomy_v2.py \
	tests/unit/test_physiology_continuity_groups.py \
	tests/unit/test_continuity_loss_spec.py \
	tests/unit/test_continuity_candidate_graph.py \
	tests/unit/test_continuity_release.py \
	tests/unit/continuity_v3_fixtures.py \
	tests/unit/test_continuity_baseline_v3.py \
	tests/unit/test_continuity_calibration_v3.py \
	tests/unit/test_graph_nmf.py \
	tests/unit/test_graph_nmf_lambda_selection.py \
	tests/unit/test_continuity_training_smoke.py \
	tests/unit/test_fullbody_training_preflight.py \
	tests/unit/test_server_deployment.py \
	tests/unit/test_bc_checkpoint_roundtrip.py \
	tests/unit/continuity_ablation_evidence_fixtures.py \
	tests/unit/test_ablation_evidence_builder.py \
	tests/unit/test_continuity_baseline_collection.py \
	tests/unit/test_continuity_ablation_report.py \
	tests/unit/test_intra_muscle_continuity.py \
	tests/unit/test_mimic_reward_continuity.py \
	tests/unit/test_forehand_continuity_config.py \
	tests/unit/test_metrics_handler_trajectory_binding.py \
	tests/unit/test_physiology_joint_report.py \
	tests/asset/test_myofullbody_354_continuity_binding.py \
	tests/asset/test_fascicle_continuity_numerical_smoke.py \
	tests/asset/test_chinajump_cache_contract.py \
	tests/asset/test_candidate_continuity_loss_binding.py \
	tests/asset/test_racket_muscle_channel_portability.py \
	tests/asset/test_continuity_release_runtime_binding.py \
	tests/gpu/_continuity_smoke.py \
	tests/gpu/test_forehand_continuity_ppo_smoke.py \
	tests/gpu/test_fixed_synergy_continuity_smoke.py \
	tests/gpu/test_graph_nmf_continuity_smoke.py \
	environment/overall_environment/src/stage3_target_bank_v2.py \
	environment/overall_environment/src/stage3_task_curriculum_v2.py
LINT_PATHS += $(NEW_RESEARCH_LINT_PATHS)

help:  ## Show this help message
	@echo "MuscleMimic - Development Commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# Every install target is additive (--inexact): uv sync is declarative by
# default and prunes anything outside the requested extras, so a plain
# `uv sync --extra dev` on a training box silently uninstalls the cuda, smpl,
# and gmr extras and breaks the next run.  Use `sync-exact` when pruning is
# actually what you want.
install:  ## Install runtime dependencies (additive; never uninstalls extras)
	$(UV) sync --inexact

install-dev:  ## Add developer tools without removing installed extras
	$(UV) sync --inexact --extra dev

install-all:  ## Install every extra (dev, cuda, smpl, gmr) for a training host
	$(UV) sync --inexact --all-extras

sync-exact:  ## Prune the venv to exactly the declared extras (destructive)
	$(UV) sync $(SYNC_EXTRAS)

precommit-install:  ## Install git pre-commit hooks
	$(PRECOMMIT) install

check: ci  ## Run the default verification suite

format:  ## Format files currently covered by repository linting
	$(RUFF) format $(LINT_PATHS)
	$(RUFF) check --fix --select I $(LINT_PATHS)

lint:  ## Run scoped lint checks without touching the rest of the repository
	$(RUFF) format --check $(LINT_PATHS)
	$(RUFF) check $(LINT_PATHS)

test:  ## Run pytest (override with PYTEST_ARGS=...)
	$(PYTEST) $(PYTEST_ARGS)

source-only:  ## Verify a clean source checkout without datasets/checkpoints/SMPL assets
	PYTHONDONTWRITEBYTECODE=1 $(PYTEST) -p no:cacheprovider -q $(SOURCE_ONLY_TESTS)

asset-test:  ## Run prepared model/dataset tests without GPU smoke evidence
	$(PYTEST) tests/asset

gpu-smoke-test:  ## Validate generated A1/B1/C1/G1 GPU smoke artifacts
	$(PYTEST) -q tests/gpu -m gpu

smoke:  ## Test critical package imports
	$(PYTHON) -c "from musclemimic import set_all_caches; from loco_mujoco import TaskFactory, ImitationFactory; print('Imports OK')"

ci: lint source-only  ## Run the default source-release CI checks

clean:  ## Clean cache files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true

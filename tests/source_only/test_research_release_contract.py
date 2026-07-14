from __future__ import annotations

from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir

from fullbody.run_forehand_clear_pipeline import PipelineArtifacts, build_pipeline_plan
from musclemimic.badminton.json_contract import load_json_strict

ROOT = Path(__file__).resolve().parents[2]


def test_canonical_yaml_is_portable_and_parseable():
    paths = [
        ROOT / "fullbody/conf_fullbody.yaml",
        ROOT / "fullbody/conf_fullbody_gmr.yaml",
        *sorted((ROOT / "fullbody/config_specific_task/base").glob("*.yaml")),
        *sorted((ROOT / "fullbody/config_specific_task/stage1_body").glob("*.yaml")),
        *sorted((ROOT / "fullbody/config_specific_task/stage2_racket").glob("*.yaml")),
        *sorted((ROOT / "fullbody/config_specific_task/stage2_racket_v2").glob("*.yaml")),
        ROOT / "fullbody/config_specific_task/distill/conf_fullbody_forehandclear_racket_student_phase_bc.yaml",
        ROOT / "fullbody/config_specific_task/distill/conf_fullbody_forehandclear_racket_student_phase_ppo.yaml",
        ROOT / "fullbody/config_specific_task/distill/latent_forehandclear_lab.yaml",
        ROOT / "fullbody/config_specific_task/distill/latent_forehandclear_synergy_v3.yaml",
        ROOT / "experiments/posttrain/incoming_shuttle_hit_v1.yaml",
        ROOT / "experiments/posttrain/incoming_shuttle_hit_impact_recovery_v2.yaml",
        ROOT / "loco_mujoco/smpl/robot_confs/defaults.yaml",
        ROOT / "loco_mujoco/smpl/robot_confs/MyoFullBody.yaml",
    ]
    assert paths
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "/data3/" not in text and "/home/" not in text and "/raid/" not in text
        assert isinstance(yaml.safe_load(text), dict)


def test_public_json_templates_are_strict_and_portable():
    paths = sorted((ROOT / "configs/public").glob("*.json"))
    assert paths
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "/data3/" not in text and "/home/" not in text and "/raid/" not in text
        assert isinstance(load_json_strict(path), dict)


def test_stage2_v2_mass_configs_compose_with_isolated_physics():
    fullbody = ROOT / "fullbody"
    with initialize_config_dir(version_base=None, config_dir=str(fullbody)):
        for suffix, expected_scale in (
            ("025", 0.25),
            ("050", 0.50),
            ("075", 0.75),
            ("100", 1.00),
        ):
            cfg = compose(
                config_name=(f"config_specific_task/stage2_racket_v2/conf_fullbody_forehand_clear_racket_mass_{suffix}")
            )
            assert float(cfg.experiment.env_params.racket_mass_scale) == expected_scale
            assert cfg.experiment.env_params.reward_params.racket_reference_source == "event_cache"
            assert cfg.experiment.auto_resume is False


def test_stage3_v2_uses_selected_checkpoint_as_latent_dimension_source():
    payload = yaml.safe_load(
        (ROOT / "experiments/posttrain/incoming_shuttle_hit_impact_recovery_v2.yaml").read_text(encoding="utf-8")
    )
    assert payload["stage3_lab"]["expected_latent_dim"] is None
    assert "latent_dim" not in payload["stage3_lab"]


def test_legacy_plan_is_unchanged_and_synergy_profile_is_opt_in(tmp_path):
    artifacts = PipelineArtifacts()
    implicit = build_pipeline_plan(tmp_path, artifacts)
    explicit = build_pipeline_plan(tmp_path, artifacts, profile="legacy_v2")
    assert implicit == explicit
    assert all("synergy_v3" not in str(arg) for step in implicit for arg in step.command)

    research = build_pipeline_plan(tmp_path, artifacts, profile="synergy_v3")
    required = {
        "physical_rollout_collect",
        "synergy_fit",
        "synergy_gate",
        "latent_dimension_sweep",
        "latent_dimension_execute",
        "latent_causal_evaluate",
        "latent_causal_finalize",
        "latent_synergy_analysis",
        "latent_synergy_gate",
        "stage3_static_target_train",
        "stage3_static_target_gate",
        "recovery_target",
        "stage3_v2_train",
        "stage3_v2_gate",
        "direct_stage3_v2_preflight",
        "direct_stage3_static_target_train",
        "direct_stage3_static_target_gate",
        "direct_stage3_v2_train",
        "direct_stage3_v2_gate",
        "stage3_paired_comparison",
        "stage3_task_causal_evaluate",
        "stage3_task_causal_gate",
        "stage3_signal_export",
        "emg_validation",
        "direct_baseline_train",
        "direct_baseline_evaluate",
    }
    assert required <= {step.name for step in research}
    names = [step.name for step in research]
    assert "stage2_train" not in names
    assert names.index("stage1r005_gate") < names.index("event_reference_qc")
    assert names.index("event_reference_gate") < names.index("racket_mass_025_physics")
    assert names.index("racket_mass_025_promote") < names.index("racket_mass_050_train")
    assert names.index("racket_mass_100_promote") < names.index("physical_rollout_collect")
    assert names.index("direct_baseline_train") < names.index("direct_baseline_evaluate")
    assert names.index("latent_dimension_sweep") < names.index("latent_dimension_execute")
    assert names.index("latent_dimension_execute") < names.index("latent_causal_evaluate")
    assert names.index("latent_causal_evaluate") < names.index("latent_causal_finalize")
    assert names.index("latent_causal_finalize") < names.index("latent_synergy_analysis")
    latent_plan = next(step for step in research if step.name == "latent_dimension_sweep")
    latent_causal = next(step for step in research if step.name == "latent_causal_evaluate")
    latent_analysis = next(step for step in research if step.name == "latent_synergy_analysis")
    assert "--require-causal-interventions" in latent_plan.command
    assert "--shared-config" in latent_causal.command
    assert latent_causal.required_artifacts == ("latent_causal_adapter_config",)
    assert "--require-causal-interventions" in latent_analysis.command
    assert names.index("stage3_v2_gate") < names.index("stage3_paired_comparison")
    assert names.index("direct_stage3_v2_gate") < names.index("stage3_paired_comparison")
    assert names.index("stage3_paired_comparison") < names.index("stage3_task_causal_evaluate")
    assert names.index("stage3_task_causal_evaluate") < names.index("stage3_task_causal_gate")
    assert names.index("stage3_task_causal_gate") < names.index("stage3_signal_export")
    assert names.index("stage3_signal_export") < names.index("emg_validation")
    task_causal = next(step for step in research if step.name == "stage3_task_causal_evaluate")
    assert task_causal.command[-2:] == (
        "--config",
        "<required:stage3_task_causal_config>",
    )
    assert "stage3_task_causal_config" in task_causal.required_artifacts
    signal_export = next(step for step in research if step.name == "stage3_signal_export")
    assert signal_export.required_artifacts == (
        "stage3_v2_checkpoint",
        "stage3_signal_identity_json",
        "recovery_train_feed_bank",
        "recovery_eval_feed_bank",
    )
    assert "--export-simulation-npz" in signal_export.command
    assert "--policy-evidence-json" in signal_export.command
    assert signal_export.command[signal_export.command.index("--feed-bank") + 1] == (
        "<required:recovery_train_feed_bank>"
    )
    assert signal_export.command[signal_export.command.index("--eval-feed-bank") + 1] == (
        "<required:recovery_eval_feed_bank>"
    )
    mass_025 = next(step for step in research if step.name == "racket_mass_025_train")
    assert "<required:stage1r005_checkpoint>" in mass_025.command
    assert "--launch-stage" in mass_025.command
    target = next(step for step in research if step.name == "recovery_target")
    assert "--event-reference-metrics" in target.command
    assert target.command[target.command.index("--reference-split") + 1] == "train"
    for synergy_name, direct_name in (
        ("stage3_static_target_train", "direct_stage3_static_target_train"),
        ("stage3_v2_train", "direct_stage3_v2_train"),
    ):
        synergy_step = next(step for step in research if step.name == synergy_name)
        direct_step = next(step for step in research if step.name == direct_name)
        for flag in (
            "--seed",
            "--feed-bank",
            "--eval-feed-bank",
            "--target-bank",
            "--eval-target-bank",
        ):
            assert (
                synergy_step.command[synergy_step.command.index(flag) + 1]
                == (direct_step.command[direct_step.command.index(flag) + 1])
            )
        assert synergy_step.command[synergy_step.command.index("--seed") + 1] == "0"
    for step in research:
        if "musclemimic.badminton.scripts.run_incoming_shuttle_hit" not in step.command:
            continue
        assert "recovery_train_feed_bank" in step.required_artifacts
        assert "recovery_eval_feed_bank" in step.required_artifacts
    assert all(
        str(tmp_path / "synergy_v3") in " ".join(step.command)
        or step.name
        in {
            "emg_validation",
            "physiology_validation",
            "stage3_task_causal_evaluate",
        }
        or step.name.startswith(("data_", "stage1", "stage2", "racket_mass"))
        or step.name.endswith("gate")
        for step in research
    )


def test_external_physiology_steps_bind_to_paired_selected_policy(tmp_path):
    artifacts = PipelineArtifacts(
        emg_simulation_npz="simulation.npz",
        emg_measurement_npz="emg.npz",
        emg_mapping_json="mapping.json",
        physiology_input_npz="physiology.npz",
        physiology_config_json="physiology.json",
        stage3_paired_metrics="paired.json",
        expected_policy_checkpoint_fingerprint="a" * 64,
        expected_policy_promotion_fingerprint="b" * 64,
        expected_formal_synergy_basis_fingerprint="c" * 64,
        expected_event_reference_fingerprint="d" * 64,
        expected_session_uid="session-heldout-01",
        expected_policy_decoder_type="synergy_residual",
    )
    plan = build_pipeline_plan(tmp_path, artifacts, profile="synergy_v3")
    emg = next(step for step in plan if step.name == "emg_validation")
    physiology = next(step for step in plan if step.name == "physiology_validation")
    for step in (emg, physiology):
        assert "--policy-evidence-json" in step.command
        assert "stage3_paired_metrics" in step.required_artifacts
    assert "--signal-identity-json" in physiology.command
    assert "stage3_signal_identity_json" in physiology.required_artifacts

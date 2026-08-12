from __future__ import annotations

from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir

import fullbody.run_forehand_clear_pipeline as pipeline_module
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
        ROOT / "experiments/posttrain/incoming_shuttle_hit_full354_v1.yaml",
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
    assert "direct_stage3_v2_base_only" not in names
    assert names.index("stage3_v2_gate") < names.index("stage3_task_causal_evaluate")
    assert names.index("stage3_task_causal_gate") < names.index("direct_stage3_v2_preflight")
    assert names.index("stage3_task_causal_evaluate") < names.index("stage3_task_causal_gate")
    assert names.index("stage3_task_causal_gate") < names.index("stage3_signal_export")
    assert names.index("stage3_signal_export") < names.index("emg_validation")
    task_causal = next(step for step in research if step.name == "stage3_task_causal_evaluate")
    assert task_causal.command[-2:] == (
        "--config",
        "<required:stage3_task_causal_config>",
    )
    assert task_causal.required_artifacts == (
        "stage3_task_causal_config",
        "stage3_v2_metrics",
        "latent_selection_manifest",
    )
    assert "stage3_paired_metrics" not in task_causal.required_artifacts
    assert "direct_stage3_v2_checkpoint" not in task_causal.required_artifacts
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
        assert "experiments/posttrain/incoming_shuttle_hit_impact_recovery_v2.yaml" in synergy_step.command
        assert "experiments/posttrain/incoming_shuttle_hit_full354_v1.yaml" in direct_step.command
        assert "--latent-checkpoint" in synergy_step.command
        assert "--latent-checkpoint" not in direct_step.command
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


def test_full354_preflight_prerequisite_has_no_latent_selection_dependency(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "_require_target_event_binding", lambda *_args, **_kwargs: None)

    pipeline_module._verify_upstream_gates(
        "direct_stage3_v2_preflight",
        PipelineArtifacts(),
        output_dir=tmp_path,
    )

    plan = build_pipeline_plan(tmp_path, PipelineArtifacts(), profile="synergy_v3")
    direct_steps = [step for step in plan if step.name.startswith("direct_stage3_")]
    assert direct_steps
    assert all("latent_synergy_checkpoint" not in step.required_artifacts for step in direct_steps)
    assert all("latent_direct_checkpoint" not in step.required_artifacts for step in direct_steps)


def test_formal_task_causal_upstream_is_synergy_only(tmp_path, monkeypatch):
    passed_labels = []
    bound_reports = []
    monkeypatch.setattr(
        pipeline_module,
        "_require_passed_report",
        lambda *_args, **kwargs: passed_labels.append(kwargs.get("label")),
    )
    monkeypatch.setattr(pipeline_module, "_require_latent_selection_binding", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline_module, "_require_target_event_binding", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        pipeline_module,
        "_require_stage3_artifact_binding",
        lambda report: bound_reports.append(Path(report)),
    )
    synergy_metrics = tmp_path / "selected_synergy_evaluation.json"
    artifacts = PipelineArtifacts(stage3_v2_metrics=str(synergy_metrics))

    pipeline_module._verify_upstream_gates(
        "stage3_task_causal_evaluate",
        artifacts,
        output_dir=tmp_path,
    )

    assert "synergy Stage-3 v2 gate" in passed_labels
    assert "direct Stage-3 v2 gate" not in passed_labels
    assert bound_reports == [synergy_metrics]


def test_formal_stage3_selection_requires_only_best_synergy(tmp_path, monkeypatch):
    from musclemimic.badminton.scripts import latent_synergy_sweep

    v3 = tmp_path / "synergy_v3"
    selected = v3 / "latent_synergy" / "selected" / "best_synergy"
    selected.mkdir(parents=True)
    manifest_path = v3 / "latent_synergy" / "selected" / "selection_manifest.json"
    promotion_path = v3 / "latent_synergy" / "promotion_metrics.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    promotion_path.write_text("{}\n", encoding="utf-8")
    manifest = {
        "checkpoints": {
            "best_synergy": {"stable_checkpoint_path": str(selected.resolve())},
        },
        "promotion_metrics_path": str(promotion_path.resolve()),
        "compatibility_alias": {
            "target_family": "best_synergy",
            "stable_checkpoint_path": str(selected.resolve()),
        },
    }
    monkeypatch.setattr(latent_synergy_sweep, "validate_selected_artifact", lambda _path: manifest)

    pipeline_module._require_latent_selection_binding(
        PipelineArtifacts(
            latent_synergy_checkpoint=str(selected),
            latent_selection_manifest=str(manifest_path),
            latent_synergy_metrics=str(promotion_path),
            latent_direct_checkpoint=None,
        ),
        v3=v3,
    )


def _sweep_command(tmp_path, **artifact_kwargs):
    plan = build_pipeline_plan(tmp_path, PipelineArtifacts(**artifact_kwargs), profile="synergy_v3")
    step = next(step for step in plan if step.name == "latent_dimension_sweep")
    return list(step.command)


def _peasd_sweep_command(tmp_path, monkeypatch, *, arm: str, **artifact_kwargs):
    from musclemimic.badminton import stage2_context_family

    promotion = tmp_path / "stage1_peasd_promotion.json"
    tube = tmp_path / "emg_reference_manifest.json"
    shared_path = tmp_path / "stage2_shared_inputs.json"
    architecture_lock = tmp_path / "stage2_s2b_architecture_lock.json"
    for path in (promotion, tube, shared_path, architecture_lock):
        path.write_text("{}\n", encoding="utf-8")
    teacher = tmp_path / "teacher_checkpoint"
    train = tmp_path / "shared_train"
    validation = tmp_path / "shared_validation"
    basis = tmp_path / "synergy_basis"
    decoder = tmp_path / "frozen_decoder"
    for path in (teacher, train, validation, basis, decoder):
        path.mkdir(exist_ok=True)
    shared = {
        "stage1_peasd": {
            "promotion": {"path": str(promotion.resolve())},
            "emg_reference": {"path": str(tube.resolve())},
        },
        "datasets": {
            "train": {"path": str(train.resolve())},
            "validation": {"path": str(validation.resolve())},
        },
        "teacher": {
            "checkpoint": {"resolved_path": str(teacher.resolve())},
            "promotion": {"path": str(promotion.resolve())},
        },
        "synergy": {
            "basis": {
                "path": str(basis.resolve()),
                "artifact_fingerprint": "b" * 64,
            },
            "frozen_body_decoder": {
                "path": str(decoder.resolve()),
                "artifact_fingerprint": "d" * 64,
                "body_synergy_contract_fingerprint": "c" * 64,
                "portable_decoder_core_fingerprint": "e" * 64,
            },
        },
        "direct_s2a_evidence": {"required": False},
    }
    monkeypatch.setattr(
        stage2_context_family,
        "validate_stage2_shared_inputs",
        lambda _path, expected_action=None: shared,
    )
    return _sweep_command(
        tmp_path,
        stage1_checkpoint=str(teacher),
        stage1_peasd_promotion_manifest=str(promotion),
        emg_reference_manifest=str(tube),
        stage1_peasd_latent_arm=arm,
        stage2_shared_inputs_manifest=str(shared_path),
        stage2_architecture_lock_manifest=(
            None if arm == "disabled" else str(architecture_lock)
        ),
        **artifact_kwargs,
    )


def test_baseline_latent_arm_emits_no_emg_flags(tmp_path):
    """The EMG-free arm must stay byte-identical to the historical baseline.

    A privileged-EMG comparison is only honest if the control arm is unchanged,
    so any ``emg`` token leaking into the default sweep command invalidates it.
    """

    assert [token for token in _sweep_command(tmp_path) if "emg" in token] == []


def test_privileged_latent_arm_differs_only_by_registered_treatment(tmp_path, monkeypatch):
    from musclemimic.badminton.scripts.latent_synergy_sweep import build_parser

    module = "musclemimic.badminton.scripts.latent_synergy_sweep"
    baseline = _peasd_sweep_command(tmp_path, monkeypatch, arm="disabled")
    privileged = _peasd_sweep_command(
        tmp_path,
        monkeypatch,
        arm="real",
        emg_synergy_dim=3,
    )
    baseline_args = build_parser().parse_args(baseline[baseline.index(module) + 1 :])
    privileged_args = build_parser().parse_args(
        privileged[privileged.index(module) + 1 :]
    )

    assert baseline_args.stage2_arm == "S2-B"
    assert privileged_args.stage2_arm == "S2-C"
    assert baseline_args.stage2_shared_inputs == privileged_args.stage2_shared_inputs
    assert baseline_args.dataset_dir == privileged_args.dataset_dir
    assert baseline_args.val_dataset_dir == privileged_args.val_dataset_dir
    assert baseline_args.teacher_ckpt == privileged_args.teacher_ckpt
    assert baseline_args.synergy_basis == privileged_args.synergy_basis
    assert baseline_args.frozen_body_decoder == privileged_args.frozen_body_decoder
    assert baseline_args.emg_privileged_enabled is False
    assert privileged_args.emg_privileged_enabled is True
    assert privileged_args.emg_synergy_dim == 3
    assert privileged_args.emg_synergy_loss_weight == 0.05


def test_launcher_sweep_command_is_accepted_by_the_sweep_parser(tmp_path, monkeypatch):
    """Guards the launcher/sweep flag contract at plan time.

    Without this, a renamed flag surfaces only after the GPU job is dispatched.
    """

    from musclemimic.badminton.scripts.latent_synergy_sweep import build_parser

    module = "musclemimic.badminton.scripts.latent_synergy_sweep"
    command = _peasd_sweep_command(
        tmp_path,
        monkeypatch,
        arm="real",
        emg_synergy_dim=3,
    )
    args = build_parser().parse_args(command[command.index(module) + 1 :])

    assert args.emg_privileged_enabled is True
    assert args.emg_synergy_dim == 3
    assert Path(args.emg_reference_manifest).name == "emg_reference_manifest.json"


def test_half_configured_privileged_latent_arm_is_refused(tmp_path, monkeypatch):
    import pytest

    with pytest.raises(ValueError, match=r"explicit Stage1 PEASD latent arm"):
        _sweep_command(tmp_path, emg_synergy_dim=3)
    with pytest.raises(ValueError, match=r"promotion_manifest"):
        _sweep_command(tmp_path, emg_reference_manifest="/refs/tube.json")
    with pytest.raises(ValueError, match=r"positive emg_synergy_dim"):
        _peasd_sweep_command(
            tmp_path, monkeypatch, arm="real", emg_synergy_dim=0
        )


def test_shuffled_context_control_arm_is_selectable_from_the_launcher(
    tmp_path, monkeypatch
):
    """§26.2 S2-D is a gate arm, so the launcher must be able to run it.

    If the real privileged context does not beat its shuffled twin, the
    privileged claim is unearned.  That comparison is only possible when the
    canonical launcher can dispatch the shuffled arm, not just the real one.
    """

    from musclemimic.badminton.scripts.latent_synergy_sweep import build_parser

    module = "musclemimic.badminton.scripts.latent_synergy_sweep"
    real = _peasd_sweep_command(
        tmp_path,
        monkeypatch,
        arm="real",
        emg_synergy_dim=3,
    )
    shuffled = _peasd_sweep_command(
        tmp_path,
        monkeypatch,
        arm="shuffled",
        emg_synergy_dim=3,
        emg_shuffle_context_ablation=True,
    )

    real_args = build_parser().parse_args(real[real.index(module) + 1 :])
    args = build_parser().parse_args(shuffled[shuffled.index(module) + 1 :])
    assert real_args.stage2_arm == "S2-C"
    assert args.stage2_arm == "S2-D"
    assert real_args.stage2_shared_inputs == args.stage2_shared_inputs
    assert real_args.stage2_architecture_lock == args.stage2_architecture_lock
    assert args.emg_shuffle_context_ablation is True
    assert args.emg_privileged_enabled is True
    assert args.emg_synergy_dim == 3


def test_shuffled_control_without_the_privileged_arm_is_refused(tmp_path):
    """A shuffled context with no EMG arm would plan a mislabelled EMG-free run."""

    import pytest

    with pytest.raises(ValueError, match=r"conflicts|promotion_manifest"):
        _sweep_command(tmp_path, emg_shuffle_context_ablation=True)


def _artifact_parser():
    """Rebuild the launcher's auto-generated artifact parser."""

    import argparse

    parser = argparse.ArgumentParser()
    for field_name, field_spec in PipelineArtifacts.__dataclass_fields__.items():
        pipeline_module._add_artifact_argument(parser, field_name, field_spec.type)
    return parser


def test_launcher_reads_a_written_out_false_as_false():
    """``bool("False")`` is True, which would invert this flag's meaning.

    Someone disabling the shuffled control by writing ``False`` would otherwise
    get the shuffled arm while believing it was off, and the gate would compare
    the shuffled arm against itself.
    """

    parser = _artifact_parser()

    for spelling in ("False", "false", "0", "no", "off"):
        args = parser.parse_args(["--emg_shuffle_context_ablation", spelling])
        assert args.emg_shuffle_context_ablation is False, spelling
    for spelling in ("True", "true", "1", "yes", "on"):
        args = parser.parse_args(["--emg_shuffle_context_ablation", spelling])
        assert args.emg_shuffle_context_ablation is True, spelling

    # Bare flag stays the natural spelling for "on", and unset means off.
    assert parser.parse_args(["--emg_shuffle_context_ablation"]).emg_shuffle_context_ablation is True
    assert parser.parse_args([]).emg_shuffle_context_ablation is False


def test_launcher_refuses_ambiguous_typed_artifact_values():
    """An unparseable value must fail at the command line, not mid-plan."""

    import pytest

    parser = _artifact_parser()
    for argv in (
        ["--emg_shuffle_context_ablation", "maybe"],
        ["--emg_synergy_dim", "three"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)

    assert parser.parse_args(["--emg_synergy_dim", "3"]).emg_synergy_dim == 3


def test_stage3_h3_residual_flag_reaches_all_four_stage3_steps(tmp_path):
    """§26.3 H3 must flow through base-only, both trains and evaluate.

    If the flag reached training but not evaluation, the evaluated policy would
    silently be the H2 baseline rather than the H3 corrected one.
    """

    plan = build_pipeline_plan(
        tmp_path,
        PipelineArtifacts(stage3_bounded_residual_groups_json="/refs/groups.json"),
        profile="synergy_v3",
    )
    for step in plan:
        if step.name in {
            "stage3_v2_base_only",
            "stage3_static_target_train",
            "stage3_v2_train",
            "stage3_v2_evaluate",
        }:
            assert "--bounded-residual-groups-json" in step.command, step.name
            assert "/refs/groups.json" in step.command, step.name
        else:
            assert all("bounded-residual-groups" not in token for token in step.command), (
                f"{step.name} leaked the residual flag"
            )


def test_stage3_baseline_has_no_residual_flag(tmp_path):
    plan = build_pipeline_plan(tmp_path, PipelineArtifacts(), profile="synergy_v3")
    for step in plan:
        assert all("bounded-residual-groups" not in token for token in step.command)


def test_s2e_no_dropout_arm_reaches_launcher(tmp_path, monkeypatch):
    """§26.2 S2-E: privileged latent with context dropout forced to 0."""

    from musclemimic.badminton.scripts.latent_synergy_sweep import build_parser

    module = "musclemimic.badminton.scripts.latent_synergy_sweep"
    real = _peasd_sweep_command(
        tmp_path,
        monkeypatch,
        arm="real",
        emg_synergy_dim=3,
    )
    no_dropout = _peasd_sweep_command(
        tmp_path,
        monkeypatch,
        arm="real_no_dropout",
        emg_synergy_dim=3,
        emg_no_context_dropout=True,
    )

    real_args = build_parser().parse_args(real[real.index(module) + 1 :])
    args = build_parser().parse_args(no_dropout[no_dropout.index(module) + 1 :])
    assert real_args.stage2_arm == "S2-C"
    assert args.stage2_arm == "S2-E"
    assert real_args.stage2_shared_inputs == args.stage2_shared_inputs
    assert real_args.stage2_architecture_lock == args.stage2_architecture_lock
    assert args.emg_context_dropout == 0.0
    assert args.emg_shuffle_context_ablation is False

    import pytest

    with pytest.raises(ValueError, match=r"promotion_manifest"):
        _sweep_command(tmp_path, emg_no_context_dropout=True)

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from fullbody.run_distill_experiment import (
    DistillExperimentConfig as _DistillExperimentConfig,
)
from fullbody.run_distill_experiment import (
    build_distill_experiment_plan,
    run_distill_experiment,
)


def distill_experiment_config(**kwargs):
    kwargs.setdefault("test_only_allow_unpromoted_teacher", True)
    return _DistillExperimentConfig(**kwargs)


def test_production_plan_requires_stage2_teacher_promotion_manifest(tmp_path):
    with pytest.raises(ValueError, match="teacher_promotion_manifest"):
        build_distill_experiment_plan(
            _DistillExperimentConfig(
                teacher_ckpt="/ckpt/teacher",
                student_config="student.yaml",
                out_dir=str(tmp_path),
                collect_train=True,
            )
        )


def test_production_plan_forwards_stage2_promotion_to_all_collectors(tmp_path):
    plan = build_distill_experiment_plan(
        _DistillExperimentConfig(
            teacher_ckpt="/ckpt/teacher",
            teacher_promotion_manifest="/evidence/stage2.json",
            student_config="student.yaml",
            out_dir=str(tmp_path),
            train_motion_path=["train"],
            val_motion_path=["val"],
            collect_train=True,
            collect_val=True,
            train_bc=True,
            run_dagger=1,
        )
    )
    for name in ("collect_train", "collect_val", "dagger"):
        command = plan.commands[name]
        assert command[command.index("--teacher_promotion_manifest") + 1] == (
            "/evidence/stage2.json"
        )


def test_distill_experiment_plan_uses_fixed_output_layout(tmp_path):
    cfg = distill_experiment_config(
        teacher_ckpt="/ckpt/teacher",
        student_config="student.yaml",
        motion_path=["motion/a"],
        val_motion_path=["motion/held_out"],
        out_dir=str(tmp_path / "run"),
        collect_train=True,
        train_bc=True,
        run_dagger=2,
        compare=True,
        seed=7,
        num_envs=8,
        num_steps=16,
        train_steps=32,
    )

    plan = build_distill_experiment_plan(cfg)

    assert plan.dataset_dir == str(tmp_path / "run" / "dataset")
    assert plan.bc_dir == str(tmp_path / "run" / "bc")
    assert plan.dagger_dir == str(tmp_path / "run" / "dagger")
    assert plan.ppo_dir == str(tmp_path / "run" / "ppo")
    assert plan.compare_dir == str(tmp_path / "run" / "compare")
    assert plan.commands["collect_train"][0:3][-1] == "fullbody.distill_collect"
    assert "--deterministic_teacher" in plan.commands["collect_train"]
    assert "--save_reference_features" in plan.commands["collect_train"]
    assert plan.commands["dagger"][plan.commands["dagger"].index("--num_iters") + 1] == "2"
    assert plan.commands["dagger"][plan.commands["dagger"].index("--motion_path") + 1 :] == ["motion/a"]
    compare_cmd = plan.commands["compare"]
    assert compare_cmd[compare_cmd.index("--dataset_dir") + 1] == plan.dataset_dir
    assert compare_cmd[compare_cmd.index("--convergence_metrics") + 1] == str(
        tmp_path / "run" / "dagger" / "iter_001" / "distill_metadata.json"
    )


def test_distill_experiment_dry_run_writes_manifest_and_report(tmp_path):
    cfg = distill_experiment_config(
        teacher_ckpt="/ckpt/teacher",
        student_config="student.yaml",
        motion_path=["motion/a"],
        out_dir=str(tmp_path / "run"),
        collect_train=True,
        train_bc=True,
        compare=False,
    )

    manifest_path = run_distill_experiment(cfg, dry_run=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = (tmp_path / "run" / "final_report.md").read_text(encoding="utf-8")
    assert manifest["schema_version"] == "distill_experiment_v1"
    assert manifest["teacher_ckpt"] == "/ckpt/teacher"
    assert manifest["student_config"] == "student.yaml"
    assert manifest["motion_path"] == ["motion/a"]
    assert "git_commit" in manifest
    assert "collect_train" in manifest["commands"]
    assert "# Distill Experiment Report" in report
    assert "teacher_ckpt" in report


def test_distill_plan_keeps_train_and_validation_motions_disjoint(tmp_path):
    cfg = distill_experiment_config(
        teacher_ckpt="/ckpt/teacher",
        student_config="student.yaml",
        out_dir=str(tmp_path / "run"),
        train_motion_path=["motion/train_a", "motion/train_b"],
        val_motion_path=["motion/held_out"],
        collect_train=True,
        collect_val=True,
        train_bc=True,
        compare=True,
    )

    plan = build_distill_experiment_plan(cfg)

    train_cmd = plan.commands["collect_train"]
    val_cmd = plan.commands["collect_val"]
    compare_cmd = plan.commands["compare"]
    assert train_cmd[train_cmd.index("--motion_path") + 1 :] == ["motion/train_a", "motion/train_b"]
    assert val_cmd[val_cmd.index("--motion_path") + 1 :] == ["motion/held_out"]
    assert compare_cmd[compare_cmd.index("--motion_path") + 1 : -1] == ["motion/held_out"]
    assert compare_cmd[-1] == "--require_pass"


def test_distill_plan_makes_only_first_train_collection_fresh(tmp_path):
    cfg = distill_experiment_config(
        teacher_ckpt="/ckpt/teacher",
        student_config="student.yaml",
        out_dir=str(tmp_path / "run"),
        train_motion_path=["motion/train"],
        val_motion_path=["motion/val"],
        collect_train=True,
        collect_val=True,
        train_bc=True,
        run_dagger=1,
    )

    plan = build_distill_experiment_plan(cfg)
    train = plan.commands["collect_train"]
    val = plan.commands["collect_val"]
    dagger = plan.commands["dagger"]
    assert "--resume_dataset" not in train
    assert "--resume_dataset" in val
    assert "--resume_dataset" in dagger
    run_uids = {
        command[command.index("--run_uid") + 1]
        for command in (train, val, dagger)
    }
    assert len(run_uids) == 1

    resumed = build_distill_experiment_plan(
        distill_experiment_config(**{**cfg.__dict__, "resume_dataset": True})
    )
    assert "--resume_dataset" in resumed.commands["collect_train"]


def test_distill_plan_chains_dagger_into_closed_loop_ppo_and_final_gate(tmp_path):
    cfg = distill_experiment_config(
        teacher_ckpt="/ckpt/teacher",
        student_config="fullbody/config_specific_task/distill/conf_fullbody_forehandclear_racket_student_phase_bc.yaml",
        student_ppo_config="fullbody/config_specific_task/distill/conf_fullbody_forehandclear_racket_student_phase_ppo.yaml",
        out_dir=str(tmp_path / "run"),
        train_motion_path=["motion/train"],
        val_motion_path=["motion/val"],
        train_bc=True,
        run_dagger=2,
        run_ppo=True,
        compare=True,
        ppo_total_timesteps=1234,
    )

    plan = build_distill_experiment_plan(cfg)

    ppo_cmd = plan.commands["ppo"]
    expected_dagger = str(
        tmp_path / "run" / "dagger" / "iter_001" / "checkpoints" / "checkpoint_200000"
    )
    assert f"experiment.resume_from={expected_dagger}" in ppo_cmd
    assert "--config-name=config_specific_task/distill/conf_fullbody_forehandclear_racket_student_phase_ppo" in ppo_cmd
    assert "experiment.total_timesteps=1234" in ppo_cmd
    compare_cmd = plan.commands["compare"]
    assert "--student_dagger_ckpt" in compare_cmd
    assert "--student_ppo_ckpt" in compare_cmd
    assert compare_cmd[compare_cmd.index("--promotion_policy") + 1] == "student_bc_ppo"
    assert "--deterministic" in compare_cmd
    assert compare_cmd[compare_cmd.index("--dataset_dir") + 1] == str(
        tmp_path / "run" / "dataset"
    )
    assert compare_cmd[compare_cmd.index("--convergence_metrics") + 1] == str(
        tmp_path / "run" / "dagger" / "iter_001" / "distill_metadata.json"
    )


def test_distill_plan_rejects_missing_student_source(tmp_path):
    cfg = distill_experiment_config(
        teacher_ckpt="/ckpt/teacher",
        student_config="student.yaml",
        student_ppo_config="config_specific_task/distill/student_ppo.yaml",
        out_dir=str(tmp_path / "run"),
        val_motion_path=["motion/held_out"],
        run_ppo=True,
        compare=True,
    )

    with pytest.raises(ValueError, match="initial_student_ckpt"):
        build_distill_experiment_plan(cfg)


def test_distill_plan_rejects_overlapping_train_and_heldout_motions(tmp_path):
    cfg = distill_experiment_config(
        teacher_ckpt="/ckpt/teacher",
        student_config="student.yaml",
        out_dir=str(tmp_path / "run"),
        train_motion_path=["motion/shared"],
        val_motion_path=["motion/shared"],
        train_bc=True,
        compare=True,
    )

    with pytest.raises(ValueError, match="must be disjoint"):
        build_distill_experiment_plan(cfg)


def test_distill_plan_requires_and_wires_external_convergence_evidence(tmp_path):
    missing = distill_experiment_config(
        teacher_ckpt="/ckpt/teacher",
        student_config="student.yaml",
        initial_student_ckpt="/ckpt/student",
        out_dir=str(tmp_path / "missing"),
        val_motion_path=["motion/held_out"],
        compare=True,
    )
    with pytest.raises(ValueError, match="convergence_metrics_path"):
        build_distill_experiment_plan(missing)

    supplied = distill_experiment_config(
        teacher_ckpt="/ckpt/teacher",
        student_config="student.yaml",
        initial_student_ckpt="/ckpt/student",
        convergence_metrics_path="/metrics/convergence.json",
        out_dir=str(tmp_path / "supplied"),
        val_motion_path=["motion/held_out"],
        compare=True,
    )
    compare_cmd = build_distill_experiment_plan(supplied).commands["compare"]
    assert compare_cmd[compare_cmd.index("--convergence_metrics") + 1] == (
        "/metrics/convergence.json"
    )


def test_distill_default_collection_budgets_are_total_transitions(tmp_path):
    cfg = distill_experiment_config(
        teacher_ckpt="/ckpt/teacher",
        student_config="student.yaml",
        out_dir=str(tmp_path / "run"),
        train_motion_path=["motion/train"],
        val_motion_path=["motion/val"],
        collect_train=True,
        collect_val=True,
        train_bc=True,
        run_dagger=1,
    )

    commands = build_distill_experiment_plan(cfg).commands
    assert commands["collect_train"][commands["collect_train"].index("--num_transitions") + 1] == "1000000"
    assert commands["collect_val"][commands["collect_val"].index("--num_transitions") + 1] == "200000"
    assert commands["dagger"][commands["dagger"].index("--num_transitions") + 1] == "500000"


def test_distill_execution_streams_long_subprocesses_without_capture(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="deadbeef\n")

    monkeypatch.setattr("fullbody.run_distill_experiment.subprocess.run", fake_run)
    cfg = distill_experiment_config(
        teacher_ckpt="/ckpt/teacher",
        student_config="student.yaml",
        out_dir=str(tmp_path / "run"),
        train_motion_path=["motion/train"],
        collect_train=True,
        train_bc=False,
    )

    run_distill_experiment(cfg)

    execution = next(call for call in calls if call[0][0:2] != ["git", "rev-parse"])
    assert execution[1]["check"] is True
    assert execution[1]["text"] is True
    assert "capture_output" not in execution[1]

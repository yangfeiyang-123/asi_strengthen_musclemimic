from __future__ import annotations

import json

from fullbody.run_distill_experiment import DistillExperimentConfig, build_distill_experiment_plan, run_distill_experiment


def test_distill_experiment_plan_uses_fixed_output_layout(tmp_path):
    cfg = DistillExperimentConfig(
        teacher_ckpt="/ckpt/teacher",
        student_config="student.yaml",
        motion_path=["motion/a"],
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
    assert plan.compare_dir == str(tmp_path / "run" / "compare")
    assert plan.commands["collect_train"][0:3][-1] == "fullbody.distill_collect"
    assert "--deterministic_teacher" in plan.commands["collect_train"]
    assert "--save_reference_features" in plan.commands["collect_train"]
    assert plan.commands["dagger"][plan.commands["dagger"].index("--num_iters") + 1] == "2"


def test_distill_experiment_dry_run_writes_manifest_and_report(tmp_path):
    cfg = DistillExperimentConfig(
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

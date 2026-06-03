"""Tests for iterative DAgger orchestration command planning."""

import json

from musclemimic.distill.dagger_loop import DaggerLoopConfig, build_iteration_plan, write_loop_manifest


def test_build_iteration_plan_chains_student_checkpoint_outputs(tmp_path):
    cfg = DaggerLoopConfig(
        teacher_ckpt="/ckpt/teacher",
        initial_student_ckpt="/ckpt/student0",
        student_config="fullbody/config_specific_task/conf_fullbody_badminton_student_gmr.yaml",
        dataset_dir=str(tmp_path / "dataset"),
        output_dir=str(tmp_path / "runs"),
        num_iters=2,
        num_envs=8,
        num_steps=16,
        shard_size=32,
        train_steps=4,
        batch_size=2,
        lr=1e-4,
        mix_teacher_action_prob=0.1,
        seed=11,
    )

    plan = build_iteration_plan(cfg)

    assert len(plan) == 2
    assert plan[0].student_ckpt_in == "/ckpt/student0"
    assert "--append" in plan[0].collect_command
    assert "--freeze_run_stats" in plan[0].collect_command
    assert "--split" in plan[0].collect_command
    assert "--mix_teacher_action_prob" in plan[0].collect_command
    assert "--gaussian_kl_weight" in plan[0].train_command
    assert plan[1].student_ckpt_in.endswith("iter_000/checkpoints/checkpoint_4")
    assert plan[1].train_output_dir.endswith("iter_001")


def test_write_loop_manifest_records_iterations(tmp_path):
    cfg = DaggerLoopConfig(
        teacher_ckpt="/ckpt/teacher",
        initial_student_ckpt="/ckpt/student0",
        student_config="student.yaml",
        dataset_dir=str(tmp_path / "dataset"),
        output_dir=str(tmp_path / "runs"),
        num_iters=1,
    )

    path = write_loop_manifest(cfg, build_iteration_plan(cfg), tmp_path / "manifest.json")

    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "dagger_loop_v1"
    assert manifest["iterations"][0]["dagger_iteration"] == 0
    assert manifest["iterations"][0]["student_checkpoint_in"] == "/ckpt/student0"

from pathlib import Path

from musclemimic.badminton.scripts.evaluate_posttrain_protocol import (
    build_delta_rows,
    build_metrics_command,
    latest_checkpoint,
    parse_validation_metrics,
    write_reports,
)


def test_build_metrics_command_uses_start_from_beginning_and_evaluate_all(tmp_path: Path):
    command = build_metrics_command(
        checkpoint=tmp_path / "checkpoint_10",
        motion="ForehandNetLift/best/video07_best_stage5_smpl",
        eval_seed=0,
        metrics_envs=1,
    )

    assert command[:3] == ["uv", "run", "fullbody/eval.py"]
    assert "--start_from_beginning" in command
    assert "--evaluate_all" in command
    assert "--metrics_deterministic" in command
    assert "--metrics_only" in command
    assert "ForehandNetLift/best/video07_best_stage5_smpl" in command


def test_latest_checkpoint_searches_config_hash_dirs(tmp_path: Path):
    root = tmp_path / "arm"
    (root / "hash_a" / "checkpoint_7907").mkdir(parents=True)
    (root / "hash_a" / "checkpoint_8002").mkdir(parents=True)

    assert latest_checkpoint(root) == root / "hash_a" / "checkpoint_8002"


def test_parse_validation_metrics_reads_metric_block():
    output = """
=== VALIDATION METRICS ===
val_early_termination_rate: 0.000000
val_frame_coverage: 1.000000
val_mean_episode_return: 123.500000
"""

    metrics = parse_validation_metrics(output)

    assert metrics["val_early_termination_rate"] == 0.0
    assert metrics["val_frame_coverage"] == 1.0
    assert metrics["val_mean_episode_return"] == 123.5


def test_build_delta_rows_marks_failed_hard_gates():
    rows = [
        {
            "split": "heldout-validation",
            "motion": "ForehandNetLift/best/video01_best_stage7_smpl",
            "arm": "baseline",
            "checkpoint": "/ckpt/base",
            "val_mean_episode_return": 100.0,
            "val_early_termination_rate": 0.0,
            "val_frame_coverage": 1.0,
            "val_err_joint_vel": 0.7,
            "val_err_rpos": 0.05,
        },
        {
            "split": "heldout-validation",
            "motion": "ForehandNetLift/best/video01_best_stage7_smpl",
            "arm": "E1c",
            "checkpoint": "/ckpt/e1c",
            "val_mean_episode_return": 90.0,
            "val_early_termination_rate": 0.0,
            "val_frame_coverage": 1.0,
            "val_err_joint_vel": 0.75,
            "val_err_rpos": 0.05,
        },
    ]

    delta_rows = build_delta_rows(rows)

    assert delta_rows[0]["delta_val_mean_episode_return"] == -10.0
    assert delta_rows[0]["pass_hard_gates"] == "false"


def test_write_reports_creates_expected_files(tmp_path: Path):
    rows = [
        {
            "split": "heldout-validation",
            "motion": "ForehandNetLift/best/video01_best_stage7_smpl",
            "arm": "baseline",
            "checkpoint": "/ckpt/base",
            "val_mean_episode_return": 100.0,
            "val_early_termination_rate": 0.0,
            "val_frame_coverage": 1.0,
        }
    ]
    delta_rows = [
        {
            "split": "heldout-validation",
            "motion": "ForehandNetLift/best/video01_best_stage7_smpl",
            "posttrain_arm": "E1c",
            "pass_hard_gates": "false",
        }
    ]

    write_reports(tmp_path, rows, delta_rows)

    assert (tmp_path / "metrics_table.csv").exists()
    assert (tmp_path / "metrics_delta.csv").exists()
    assert (tmp_path / "comparison_report.md").exists()

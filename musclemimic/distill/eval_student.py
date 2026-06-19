"""Helpers for comparing teacher and student evaluation metrics."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


METRIC_RE = re.compile(r"^([A-Za-z0-9_./-]+):\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)$")
REQUIRED_EVAL_METRICS = (
    "mean_episode_return",
    "mean_episode_length",
    "early_termination_rate",
    "err_rpos",
)
REPORT_METRICS = (
    "mean_episode_return",
    "completion_rate",
    "early_termination_rate",
    "mean_episode_length",
    "err_root_xyz",
    "err_root_yaw",
    "err_joint_pos",
    "err_joint_vel",
    "err_site_abs",
    "err_rpos",
    "reward_qpos",
    "reward_qvel",
    "reward_root_pos",
    "reward_root_vel",
    "reward_rpos",
    "reward_rquat",
    "reward_rvel_rot",
    "reward_rvel_lin",
)


def parse_eval_metrics_stdout(stdout: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in stdout.splitlines():
        match = METRIC_RE.match(line.strip())
        if match:
            metrics[match.group(1)] = float(match.group(2))
    return metrics


def validate_required_metrics(metrics: dict[str, float], required: tuple[str, ...] = REQUIRED_EVAL_METRICS) -> None:
    """Fail fast when eval output lacks metrics needed for comparison reports."""
    missing = []
    for metric in required:
        if metric not in metrics and f"val_{metric}" not in metrics:
            missing.append(metric)
    if missing:
        raise RuntimeError(f"missing eval metrics: {sorted(missing)}")


def run_eval_metrics(
    checkpoint: str,
    *,
    motion_paths: list[str] | None = None,
    metrics_envs: int = 20,
    metrics_steps: int = 500,
    eval_seed: int = 0,
    deterministic: bool = False,
    require_metrics: bool = True,
) -> dict[str, float]:
    with tempfile.TemporaryDirectory(prefix="distill_eval_metrics_") as tmpdir:
        metrics_json = str(Path(tmpdir) / "metrics.json")
        cmd = [
            sys.executable,
            "-m",
            "fullbody.eval",
            "--path",
            checkpoint,
            "--metrics",
            "--metrics_only",
            "--metrics_envs",
            str(metrics_envs),
            "--metrics_steps",
            str(metrics_steps),
            "--eval_seed",
            str(eval_seed),
            "--metrics_output_json",
            metrics_json,
        ]
        if deterministic:
            cmd.append("--metrics_deterministic")
        if motion_paths:
            cmd.append("--motion_path")
            cmd.extend(motion_paths)
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        metrics_path = Path(metrics_json)
        if metrics_path.is_file():
            metrics = {str(key): float(value) for key, value in json.loads(metrics_path.read_text()).items()}
        else:
            metrics = parse_eval_metrics_stdout(result.stdout)
    if require_metrics:
        validate_required_metrics(metrics)
    return metrics


def write_comparison_outputs(results: dict[str, dict[str, float]], output_dir: str | Path) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "comparison_metrics.json"
    csv_path = output_path / "comparison_table.csv"

    json_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    metric_names = sorted({metric for metrics in results.values() for metric in metrics})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", *metric_names])
        for policy, metrics in results.items():
            writer.writerow([policy, *[metrics.get(metric, "") for metric in metric_names]])
    return json_path, csv_path


def _ratio(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None or baseline == 0.0:
        return None
    return float(value) / float(baseline)


def write_summary_report(results: dict[str, dict[str, float]], output_dir: str | Path) -> Path:
    """Write a teacher-vs-student markdown report with acceptance ratios."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "summary.md"
    teacher = results.get("teacher", {})

    lines = [
        "# ForehandClear Distillation Evaluation",
        "",
        "## Required Metrics",
        "",
        "| Policy | Metric | Value | Teacher Ratio |",
        "|---|---|---:|---:|",
    ]
    for policy, metrics in results.items():
        for metric in REPORT_METRICS:
            if metric not in metrics:
                continue
            ratio = _ratio(metrics.get(metric), teacher.get(metric))
            ratio_text = "" if ratio is None else f"{ratio:.6f}"
            lines.append(f"| {policy} | {metric} | {metrics[metric]:.6f} | {ratio_text} |")

    lines.extend(
        [
            "",
            "## Acceptance Signals",
            "",
            "| Policy | return_ratio | completion_ratio | early_termination_delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for policy, metrics in results.items():
        if policy == "teacher":
            continue
        return_ratio = _ratio(metrics.get("mean_episode_return"), teacher.get("mean_episode_return"))
        completion_ratio = _ratio(metrics.get("completion_rate"), teacher.get("completion_rate"))
        early_delta = None
        if "early_termination_rate" in metrics and "early_termination_rate" in teacher:
            early_delta = float(metrics["early_termination_rate"]) - float(teacher["early_termination_rate"])
        lines.append(
            "| {policy} | {return_ratio} | {completion_ratio} | {early_delta} |".format(
                policy=policy,
                return_ratio="" if return_ratio is None else f"{return_ratio:.6f}",
                completion_ratio="" if completion_ratio is None else f"{completion_ratio:.6f}",
                early_delta="" if early_delta is None else f"{early_delta:.6f}",
            )
        )

    lines.extend(
        [
            "",
            "Initial v1 thresholds:",
            "",
            "- Student BC return ratio target before PPO fine-tune: >= 0.70.",
            "- Student BC+PPO return ratio target after fine-tune: >= 0.85.",
            "- Student rollout completion ratio target: >= 0.80 of teacher.",
            "- Early termination rate should not exceed teacher by more than 0.20 after PPO fine-tune.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path

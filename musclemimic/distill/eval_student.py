"""Helpers for comparing teacher and student evaluation metrics."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path


METRIC_RE = re.compile(r"^([A-Za-z0-9_./-]+):\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)$")


def parse_eval_metrics_stdout(stdout: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in stdout.splitlines():
        match = METRIC_RE.match(line.strip())
        if match:
            metrics[match.group(1)] = float(match.group(2))
    return metrics


def run_eval_metrics(
    checkpoint: str,
    *,
    motion_paths: list[str] | None = None,
    metrics_envs: int = 20,
    metrics_steps: int = 500,
    eval_seed: int = 0,
    deterministic: bool = False,
) -> dict[str, float]:
    cmd = [
        sys.executable,
        "fullbody/eval.py",
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
    ]
    if deterministic:
        cmd.append("--metrics_deterministic")
    if motion_paths:
        cmd.append("--motion_path")
        cmd.extend(motion_paths)
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return parse_eval_metrics_stdout(result.stdout)


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

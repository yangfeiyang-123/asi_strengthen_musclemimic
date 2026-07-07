#!/usr/bin/env python3
"""Run the standardized ForehandNetLift PostTrain evaluation protocol."""

from __future__ import annotations

import argparse
import csv
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from BadmintonMimic.scripts.run_posttrain_experiment import load_spec


METRIC_KEYS = (
    "val_mean_episode_return",
    "val_mean_episode_length",
    "val_early_termination_count",
    "val_early_termination_rate",
    "val_frame_coverage",
    "val_total_frame",
    "val_err_joint_pos",
    "val_err_joint_vel",
    "val_err_root_xyz",
    "val_err_root_yaw",
    "val_err_rpos",
    "val_err_site_abs",
    "val_reward_total",
)
DELTA_KEYS = (
    "val_mean_episode_return",
    "val_early_termination_rate",
    "val_frame_coverage",
    "val_err_joint_pos",
    "val_err_joint_vel",
    "val_err_root_xyz",
    "val_err_root_yaw",
    "val_err_rpos",
    "val_err_site_abs",
    "val_reward_total",
)
METRIC_RE = re.compile(r"^(val_[A-Za-z0-9_]+):\s+([-+0-9.eE]+)\s*$")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def quote_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    suffix = path.name.removeprefix("checkpoint_")
    try:
        return (int(suffix), path.name)
    except ValueError:
        return (-1, path.name)


def latest_checkpoint(root: Path) -> Path | None:
    if not root.exists():
        return None
    candidates = [item for item in root.rglob("checkpoint_*") if item.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=checkpoint_sort_key)


def build_metrics_command(
    *,
    checkpoint: Path,
    motion: str,
    eval_seed: int,
    metrics_envs: int,
) -> list[str]:
    return [
        "uv",
        "run",
        "fullbody/eval.py",
        "--path",
        str(checkpoint),
        "--motion_path",
        motion,
        "--use_mujoco",
        "--eval_seed",
        str(eval_seed),
        "--start_from_beginning",
        "--evaluate_all",
        "--metrics",
        "--metrics_only",
        "--metrics_deterministic",
        "--metrics_envs",
        str(metrics_envs),
    ]


def parse_validation_metrics(output: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    in_block = False
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line == "=== VALIDATION METRICS ===":
            in_block = True
            continue
        if not in_block:
            continue
        match = METRIC_RE.match(line)
        if match:
            metrics[match.group(1)] = float(match.group(2))
    return metrics


def motion_groups(spec: dict[str, Any], splits: list[str]) -> list[tuple[str, str]]:
    label_for_split = {
        "train": "train-seen",
        "validation": "heldout-validation",
        "stress_test": "stress-test",
    }
    groups: list[tuple[str, str]] = []
    for split in splits:
        for motion in spec["reference"].get(split, []):
            groups.append((label_for_split.get(split, split), motion))
    return groups


def baseline_checkpoint(spec: dict[str, Any]) -> Path:
    for arm in spec["arms"]:
        if arm.get("type") == "baseline" and arm.get("checkpoint"):
            return Path(arm["checkpoint"])
    return Path(spec["resume_from"])


def posttrain_checkpoint(spec: dict[str, Any], arm_id: str, explicit_checkpoint: str | None) -> Path:
    if explicit_checkpoint:
        return Path(explicit_checkpoint)
    checkpoint_root = Path(spec.get("checkpoint_root", Path(spec["output_root"]) / spec["action"] / spec["experiment_id"] / "checkpoints"))
    latest = latest_checkpoint(checkpoint_root / arm_id)
    if latest is None:
        raise FileNotFoundError(f"No checkpoint_* directory found under {checkpoint_root / arm_id}")
    return latest


def run_metrics_command(command: list[str]) -> dict[str, float]:
    completed = subprocess.run(
        command,
        cwd=project_root(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with exit code {completed.returncode}\n{completed.stdout}")
    metrics = parse_validation_metrics(completed.stdout)
    if not metrics:
        raise RuntimeError(f"command produced no validation metrics\n{completed.stdout}")
    return metrics


def evaluate_checkpoint_set(
    *,
    spec: dict[str, Any],
    arm_id: str,
    checkpoint: Path,
    splits: list[str],
    eval_seed: int,
    metrics_envs: int,
    execute: bool,
) -> list[dict[str, str | float]]:
    checkpoints = [
        ("baseline", baseline_checkpoint(spec)),
        (arm_id, checkpoint),
    ]
    rows: list[dict[str, str | float]] = []
    for split, motion in motion_groups(spec, splits):
        for arm_name, ckpt in checkpoints:
            command = build_metrics_command(
                checkpoint=ckpt,
                motion=motion,
                eval_seed=eval_seed,
                metrics_envs=metrics_envs,
            )
            print(quote_command(command), flush=True)
            metrics = run_metrics_command(command) if execute else {}
            row: dict[str, str | float] = {
                "split": split,
                "motion": motion,
                "arm": arm_name,
                "checkpoint": str(ckpt),
            }
            for key in METRIC_KEYS:
                row[key] = metrics.get(key, "")
            rows.append(row)
    return rows


def _metric(row: dict[str, str | float], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value == "":
        return default
    return float(value)


def build_delta_rows(rows: list[dict[str, str | float]]) -> list[dict[str, str | float]]:
    baselines = {(row["split"], row["motion"]): row for row in rows if row["arm"] == "baseline"}
    delta_rows: list[dict[str, str | float]] = []
    for row in rows:
        if row["arm"] == "baseline":
            continue
        base = baselines.get((row["split"], row["motion"]))
        if base is None:
            continue
        delta: dict[str, str | float] = {
            "split": row["split"],
            "motion": row["motion"],
            "posttrain_arm": row["arm"],
            "baseline_checkpoint": base["checkpoint"],
            "posttrain_checkpoint": row["checkpoint"],
        }
        for key in DELTA_KEYS:
            delta[f"baseline_{key}"] = base.get(key, "")
            delta[f"posttrain_{key}"] = row.get(key, "")
            if base.get(key, "") != "" and row.get(key, "") != "":
                delta[f"delta_{key}"] = _metric(row, key) - _metric(base, key)
            else:
                delta[f"delta_{key}"] = ""
        pass_gates = (
            _metric(row, "val_early_termination_rate", 1.0) == 0.0
            and _metric(row, "val_frame_coverage", 0.0) >= 0.95
            and _metric(row, "val_mean_episode_return", -1e9) >= _metric(base, "val_mean_episode_return", 1e9)
            and _metric(row, "val_err_joint_vel", 1e9) <= _metric(base, "val_err_joint_vel", 0.0) + 0.10
            and _metric(row, "val_err_rpos", 1e9) <= _metric(base, "val_err_rpos", 0.0) + 0.01
        )
        delta["pass_hard_gates"] = str(pass_gates).lower()
        delta_rows.append(delta)
    return delta_rows


def write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(path: Path, rows: list[dict[str, str | float]], delta_rows: list[dict[str, str | float]]) -> None:
    lines = [
        "# PostTrain Evaluation Protocol Report",
        "",
        "## Summary",
        "",
        f"- Motions evaluated: {len({(row['split'], row['motion']) for row in rows})}",
        f"- PostTrain comparisons: {len(delta_rows)}",
        f"- Hard-gate passes: {sum(1 for row in delta_rows if row.get('pass_hard_gates') == 'true')}",
        "",
        "## Deltas",
        "",
        "| split | motion | arm | return delta | early term | coverage | joint_vel delta | rpos delta | pass |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in delta_rows:
        lines.append(
            "| {split} | {motion} | {posttrain_arm} | {ret} | {early} | {coverage} | {jvel} | {rpos} | {passed} |".format(
                split=row.get("split", ""),
                motion=row.get("motion", ""),
                posttrain_arm=row.get("posttrain_arm", ""),
                ret=_format_cell(row.get("delta_val_mean_episode_return", "")),
                early=_format_cell(row.get("posttrain_val_early_termination_rate", "")),
                coverage=_format_cell(row.get("posttrain_val_frame_coverage", "")),
                jvel=_format_cell(row.get("delta_val_err_joint_vel", "")),
                rpos=_format_cell(row.get("delta_val_err_rpos", "")),
                passed=row.get("pass_hard_gates", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Recommendation Rule",
            "",
            "Keep a PostTrain checkpoint only if every required heldout motion passes hard gates. "
            "Root/site improvements do not override early termination, low coverage, or worse return.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def _format_cell(value: str | float) -> str:
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_reports(output_dir: Path, rows: list[dict[str, str | float]], delta_rows: list[dict[str, str | float]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "metrics_table.csv", rows)
    write_csv(output_dir / "metrics_delta.csv", delta_rows)
    write_markdown_report(output_dir / "comparison_report.md", rows, delta_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--run-name", default="posttrain_protocol_check")
    parser.add_argument("--splits", default="train,validation,stress_test")
    parser.add_argument("--metrics-envs", type=int, default=1)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = load_spec(args.spec)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    checkpoint = posttrain_checkpoint(spec, args.arm, args.checkpoint)
    rows = evaluate_checkpoint_set(
        spec=spec,
        arm_id=args.arm,
        checkpoint=checkpoint,
        splits=splits,
        eval_seed=args.eval_seed,
        metrics_envs=args.metrics_envs,
        execute=args.execute,
    )
    if args.execute:
        output_dir = (
            Path(spec["output_root"])
            / spec["action"]
            / spec["experiment_id"]
            / "metrics"
            / args.run_name
        )
        write_reports(output_dir, rows, build_delta_rows(rows))
        print(f"Wrote reports to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

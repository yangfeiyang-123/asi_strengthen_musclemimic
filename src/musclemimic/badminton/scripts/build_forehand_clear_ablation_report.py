"""Aggregate fail-aware, fingerprinted forehand-clear ablation JSONL."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "forehand_clear_ablation_summary_v2"
PAIR_KEYS = ("seed", "feed_uid", "motion_uid")
FINGERPRINT_KEYS = (
    "config_fingerprint",
    "checkpoint_fingerprint",
    "data_fingerprint",
    "basis_fingerprint",
    "feed_fingerprint",
)


def render_markdown_report(rows: list[dict]) -> str:
    """Keep the original compact renderer for existing callers/tests."""

    ranked = sorted(
        rows,
        key=lambda row: (float(row["success_rate"]), float(row["backcourt_rate"])),
        reverse=True,
    )
    lines = [
        "# ForehandClear Racket Ablation Summary",
        "",
        "| Arm | Success Rate | Backcourt Rate |",
        "| --- | ---: | ---: |",
    ]
    for row in ranked:
        lines.append(f"| {row['arm']} | {float(row['success_rate']):.3f} | {float(row['backcourt_rate']):.3f} |")
    lines.append("")
    return "\n".join(lines)


def load_ablation_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw, object_pairs_hook=_reject_duplicates)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid ablation JSONL line {line_no}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"ablation JSONL line {line_no} must be an object")
        records.append(_validate_record(row, line_no=line_no))
    if not records:
        raise ValueError("ablation JSONL contains no records")
    return records


def aggregate_ablation_records(
    records: list[dict[str, Any]],
    *,
    baseline_arm: str | None = None,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["arm"]].append(record)
    if baseline_arm is None:
        baseline_arm = sorted(grouped)[0]
    if baseline_arm not in grouped:
        raise ValueError(f"baseline arm {baseline_arm!r} is absent")

    rng = np.random.default_rng(seed)
    arms: dict[str, Any] = {}
    for arm in sorted(grouped):
        rows = grouped[arm]
        successful = [row for row in rows if row["status"] == "ok"]
        metric_names = sorted({name for row in successful for name in row["metrics"]})
        metrics: dict[str, Any] = {}
        for name in metric_names:
            values = np.asarray(
                [row["metrics"][name] for row in successful if name in row["metrics"]],
                dtype=float,
            )
            low, high = _bootstrap_ci(values, rng, bootstrap_samples)
            metrics[name] = {
                "n": int(values.size),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "ci95_low": low,
                "ci95_high": high,
            }
        arms[arm] = {
            "attempted_runs": len(rows),
            "successful_runs": len(successful),
            "failed_runs": len(rows) - len(successful),
            "failure_rate": (len(rows) - len(successful)) / len(rows),
            "failures": [
                {
                    "seed": row["seed"],
                    "feed_uid": row.get("feed_uid"),
                    "motion_uid": row.get("motion_uid"),
                    "error": row.get("error", "unspecified failure"),
                }
                for row in rows
                if row["status"] != "ok"
            ],
            "fingerprints": {
                key: sorted({str(row[key]) for row in rows if row.get(key) is not None}) for key in FINGERPRINT_KEYS
            },
            "metrics": metrics,
        }

    effects: dict[str, Any] = {}
    baseline = _successful_by_pair(grouped[baseline_arm])
    for arm in sorted(grouped):
        if arm == baseline_arm:
            continue
        candidate = _successful_by_pair(grouped[arm])
        common = sorted(set(baseline) & set(candidate), key=str)
        metric_names = sorted(
            {name for pair in common for name in set(baseline[pair]["metrics"]) & set(candidate[pair]["metrics"])}
        )
        arm_effects: dict[str, Any] = {}
        for name in metric_names:
            pairs = [
                pair for pair in common if name in baseline[pair]["metrics"] and name in candidate[pair]["metrics"]
            ]
            diff = np.asarray(
                [candidate[pair]["metrics"][name] - baseline[pair]["metrics"][name] for pair in pairs],
                dtype=float,
            )
            low, high = _bootstrap_ci(diff, rng, bootstrap_samples)
            std = float(diff.std(ddof=1)) if diff.size > 1 else 0.0
            arm_effects[name] = {
                "paired_n": int(diff.size),
                "mean_difference": float(diff.mean()),
                "ci95_low": low,
                "ci95_high": high,
                "paired_effect_size_dz": (float(diff.mean() / std) if std > 1e-12 else None),
            }
        effects[arm] = {
            "baseline_arm": baseline_arm,
            "paired_key_fields": list(PAIR_KEYS),
            "paired_record_count": len(common),
            "metrics": arm_effects,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "baseline_arm": baseline_arm,
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(seed),
        "record_count": len(records),
        "arms": arms,
        "paired_effects": effects,
    }


def write_ablation_report(summary: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    metrics_path = out / "metrics.csv"
    effects_path = out / "paired_effects.csv"
    markdown_path = out / "report.md"
    _atomic_text(
        summary_path,
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _write_csv(
        metrics_path,
        ("arm", "metric", "n", "mean", "std", "ci95_low", "ci95_high", "failed_runs"),
        (
            (
                arm,
                metric,
                values["n"],
                values["mean"],
                values["std"],
                values["ci95_low"],
                values["ci95_high"],
                arm_data["failed_runs"],
            )
            for arm, arm_data in summary["arms"].items()
            for metric, values in arm_data["metrics"].items()
        ),
    )
    _write_csv(
        effects_path,
        (
            "arm",
            "baseline_arm",
            "metric",
            "paired_n",
            "mean_difference",
            "ci95_low",
            "ci95_high",
            "paired_effect_size_dz",
        ),
        (
            (
                arm,
                values["baseline_arm"],
                metric,
                effect["paired_n"],
                effect["mean_difference"],
                effect["ci95_low"],
                effect["ci95_high"],
                effect["paired_effect_size_dz"],
            )
            for arm, values in summary["paired_effects"].items()
            for metric, effect in values["metrics"].items()
        ),
    )
    _atomic_text(markdown_path, _render_full_markdown(summary))
    return {
        "summary_json": summary_path,
        "metrics_csv": metrics_path,
        "effects_csv": effects_path,
        "markdown": markdown_path,
    }


def _validate_record(row: dict[str, Any], *, line_no: int) -> dict[str, Any]:
    result = dict(row)
    if not str(result.get("arm", "")).strip():
        raise ValueError(f"line {line_no}: arm is required")
    result["arm"] = str(result["arm"])
    if isinstance(result.get("seed"), bool):
        raise ValueError(f"line {line_no}: seed must be an integer")
    result["seed"] = int(result.get("seed"))
    status = str(result.get("status", "ok")).lower()
    status = "ok" if status in {"ok", "passed", "success"} else "failed"
    result["status"] = status
    metrics = result.get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError(f"line {line_no}: metrics must be an object")
    clean_metrics: dict[str, float] = {}
    for name, raw in metrics.items():
        if isinstance(raw, bool):
            value = float(raw)
        else:
            value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"line {line_no}: metric {name!r} is non-finite")
        clean_metrics[str(name)] = value
    if status == "ok" and not clean_metrics:
        raise ValueError(f"line {line_no}: successful record has no metrics")
    result["metrics"] = clean_metrics
    for key in ("config_fingerprint", "checkpoint_fingerprint", "data_fingerprint"):
        _validate_sha256(result.get(key), label=f"line {line_no} {key}")
    for key in ("basis_fingerprint", "feed_fingerprint"):
        if result.get(key) is not None:
            _validate_sha256(result[key], label=f"line {line_no} {key}")
    return result


def _successful_by_pair(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        key = tuple(row.get(field) for field in PAIR_KEYS)
        if key in result:
            raise ValueError(f"duplicate successful paired key within arm: {key}")
        result[key] = row
    return result


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, samples: int) -> tuple[float, float]:
    if values.size == 0:
        raise ValueError("cannot aggregate an empty metric")
    if values.size == 1:
        scalar = float(values[0])
        return scalar, scalar
    means = np.mean(rng.choice(values, size=(samples, values.size), replace=True), axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _render_full_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Forehand-clear ablation report",
        "",
        f"Baseline: `{summary['baseline_arm']}`. Records: {summary['record_count']}.",
        "",
        "| Arm | Attempted | Successful | Failed | Failure rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for arm, values in summary["arms"].items():
        lines.append(
            f"| {arm} | {values['attempted_runs']} | {values['successful_runs']} | "
            f"{values['failed_runs']} | {values['failure_rate']:.3f} |"
        )
    lines.extend(["", "## Metrics", ""])
    for arm, values in summary["arms"].items():
        lines.append(f"### {arm}")
        lines.append("")
        lines.append("| Metric | n | Mean | 95% CI |")
        lines.append("| --- | ---: | ---: | ---: |")
        for metric, item in values["metrics"].items():
            lines.append(
                f"| {metric} | {item['n']} | {item['mean']:.6g} | [{item['ci95_low']:.6g}, {item['ci95_high']:.6g}] |"
            )
        lines.append("")
    lines.extend(["## Paired effects", ""])
    for arm, values in summary["paired_effects"].items():
        lines.append(f"### {arm} vs {values['baseline_arm']}")
        lines.append("")
        lines.append("| Metric | Paired n | Mean difference | 95% CI | dz |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for metric, effect in values["metrics"].items():
            dz = effect["paired_effect_size_dz"]
            dz_text = "n/a" if dz is None else f"{dz:.4g}"
            lines.append(
                f"| {metric} | {effect['paired_n']} | {effect['mean_difference']:.6g} | "
                f"[{effect['ci95_low']:.6g}, {effect['ci95_high']:.6g}] | {dz_text} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _validate_sha256(value: Any, *, label: str) -> None:
    text = str(value or "")
    if len(text) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256")
    try:
        int(text, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be hexadecimal") from exc


def _reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, header: tuple[str, ...], rows) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-arm", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    source = Path(args.input_jsonl)
    records = load_ablation_jsonl(source)
    summary = aggregate_ablation_records(
        records,
        baseline_arm=args.baseline_arm,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    summary["input_jsonl_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    paths = write_ablation_report(summary, args.output_dir)
    print(json.dumps({key: str(value) for key, value in paths.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build a fingerprinted multi-seed latent/synergy comparison report."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_METRICS = (
    "posterior_action_mse",
    "prior_mean_action_mse",
    "participation_ratio_dimension",
    "physical_excitation_mse",
    "residual_energy_ratio",
)


def bootstrap_summary(
    values: Sequence[float],
    *,
    seed: int = 0,
    num_bootstrap: int = 2000,
) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("bootstrap values must be finite and non-empty")
    if int(num_bootstrap) <= 0:
        raise ValueError("num_bootstrap must be positive")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, array.size, size=(int(num_bootstrap), array.size))
    means = np.mean(array[indices], axis=1)
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def build_comparison_report(
    records: Sequence[Mapping[str, Any]],
    *,
    required_metrics: Sequence[str] = DEFAULT_METRICS,
    bootstrap_seed: int = 0,
    num_bootstrap: int = 2000,
) -> dict[str, Any]:
    normalized = [_normalize_record(record, required_metrics) for record in records]
    if not normalized:
        raise ValueError("comparison report requires at least one experiment record")
    run_names = [record["run_name"] for record in normalized]
    if len(set(run_names)) != len(run_names):
        raise ValueError("comparison report run_name values must be unique")
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in normalized:
        grouped[(record["latent_dim"], record["decoder_type"])].append(record)
    groups: list[dict[str, Any]] = []
    for group_index, ((latent_dim, decoder_type), items) in enumerate(sorted(grouped.items())):
        seeds = [item["seed"] for item in items]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"duplicate seeds in latent_dim={latent_dim}, decoder_type={decoder_type}")
        summaries = {
            metric: bootstrap_summary(
                [item["metrics"][metric] for item in items],
                seed=int(bootstrap_seed) + group_index,
                num_bootstrap=int(num_bootstrap),
            )
            for metric in required_metrics
        }
        groups.append(
            {
                "latent_dim": latent_dim,
                "decoder_type": decoder_type,
                "num_seeds": len(items),
                "seeds": sorted(seeds),
                "metrics": summaries,
                "checkpoint_fingerprints": sorted(item["checkpoint_fingerprint"] for item in items),
                "dataset_fingerprints": sorted({item["dataset_fingerprint"] for item in items}),
                "synergy_basis_fingerprints": sorted(
                    {
                        item["synergy_basis_fingerprint"]
                        for item in items
                        if item["synergy_basis_fingerprint"] is not None
                    }
                ),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": "latent_synergy_comparison_report_v1",
        "num_runs": len(normalized),
        "num_groups": len(groups),
        "required_metrics": list(required_metrics),
        "groups": groups,
        "runs": normalized,
    }
    payload["report_fingerprint"] = _json_sha256(payload)
    return payload


def report_markdown(report: Mapping[str, Any]) -> str:
    metrics = list(report["required_metrics"])
    header = ["latent dim", "decoder", "seeds", *metrics]
    rows = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for group in report["groups"]:
        cells = [
            str(group["latent_dim"]),
            str(group["decoder_type"]),
            str(group["num_seeds"]),
        ]
        for metric in metrics:
            summary = group["metrics"][metric]
            cells.append(
                f"{summary['mean']:.6g} ± {summary['std']:.3g} [{summary['ci95_low']:.6g}, {summary['ci95_high']:.6g}]"
            )
        rows.append("| " + " | ".join(cells) + " |")
    return (
        "# Latent-synergy comparison\n\n"
        f"Runs: {report['num_runs']}; groups: {report['num_groups']}; "
        f"fingerprint: `{report['report_fingerprint']}`.\n\n" + "\n".join(rows) + "\n"
    )


def load_records(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    if not paths:
        raise ValueError("at least one result JSON is required")
    result: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"comparison result does not exist: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            candidates = payload
        elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
            candidates = payload["records"]
        elif isinstance(payload, dict):
            candidates = [payload]
        else:
            raise ValueError(f"comparison result {path} is not a JSON object/list")
        if not candidates:
            raise ValueError(f"comparison result {path} contains no records")
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError(f"comparison result {path} contains a non-object record")
            result.append(dict(candidate) | {"source_json": str(path.resolve())})
    return result


def _normalize_record(
    record: Mapping[str, Any],
    required_metrics: Sequence[str],
) -> dict[str, Any]:
    experiment = record.get("experiment")
    config = record.get("config")
    metadata: dict[str, Any] = {}
    if isinstance(config, Mapping):
        metadata.update(config)
    if isinstance(experiment, Mapping):
        metadata.update(experiment)
    metadata.update(
        {
            key: record[key]
            for key in (
                "run_name",
                "latent_dim",
                "decoder_type",
                "seed",
                "checkpoint_fingerprint",
                "dataset_fingerprint",
                "synergy_basis_fingerprint",
            )
            if key in record
        }
    )
    metric_sources = [
        value
        for value in (
            record.get("metrics"),
            record.get("eval_metrics"),
            record.get("analysis_metrics"),
            record,
        )
        if isinstance(value, Mapping)
    ]
    metrics: dict[str, float] = {}
    missing: list[str] = []
    for metric in required_metrics:
        value = next((source[metric] for source in metric_sources if metric in source), None)
        if value is None:
            missing.append(metric)
            continue
        array = np.asarray(value)
        if array.size != 1 or not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"metric {metric!r} must be a numeric scalar")
        number = float(array.reshape(-1)[0])
        if not np.isfinite(number):
            raise ValueError(f"metric {metric!r} must be finite")
        metrics[str(metric)] = number
    if missing:
        raise ValueError(f"experiment record is missing required metrics: {missing}")
    required_metadata = (
        "run_name",
        "latent_dim",
        "decoder_type",
        "seed",
        "checkpoint_fingerprint",
        "dataset_fingerprint",
    )
    absent = [key for key in required_metadata if metadata.get(key) in (None, "")]
    if absent:
        raise ValueError(f"experiment record is missing required provenance: {absent}")
    decoder_type = str(metadata["decoder_type"])
    if decoder_type not in {"direct", "fixed_synergy", "synergy_residual"}:
        raise ValueError(f"unsupported decoder_type in report: {decoder_type!r}")
    synergy_fingerprint = metadata.get("synergy_basis_fingerprint")
    if decoder_type != "direct" and not synergy_fingerprint:
        raise ValueError("synergy experiment record is missing synergy_basis_fingerprint")
    return {
        "run_name": str(metadata["run_name"]),
        "latent_dim": int(metadata["latent_dim"]),
        "decoder_type": decoder_type,
        "seed": int(metadata["seed"]),
        "checkpoint_fingerprint": str(metadata["checkpoint_fingerprint"]),
        "dataset_fingerprint": str(metadata["dataset_fingerprint"]),
        "synergy_basis_fingerprint": (None if synergy_fingerprint is None else str(synergy_fingerprint)),
        "metrics": metrics,
        "source_json": record.get("source_json"),
    }


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, nargs="+", required=True)
    parser.add_argument("--required-metric", action="append", default=None)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--num-bootstrap", type=int, default=2000)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = build_comparison_report(
        load_records(args.input_json),
        required_metrics=(DEFAULT_METRICS if args.required_metric is None else args.required_metric),
        bootstrap_seed=int(args.bootstrap_seed),
        num_bootstrap=int(args.num_bootstrap),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(report_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

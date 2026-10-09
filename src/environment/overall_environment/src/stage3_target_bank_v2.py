"""Portable desired-impact target bank used only by Stage-3 v2 profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "incoming_hit_target_bank_v2"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class DesiredImpactTarget:
    feed_fingerprint: str
    impact_position_world: tuple[float, float, float]
    impact_time_s: float
    stringbed_normal_world: tuple[float, float, float]
    racket_linear_velocity_world: tuple[float, float, float]
    racket_angular_velocity_world: tuple[float, float, float]
    landing_target_xy: tuple[float, float]
    apex_height_m: float
    recovery_horizon_steps: int = 60
    provenance: str = "user_supplied"

    def __post_init__(self) -> None:
        if _SHA256_RE.fullmatch(self.feed_fingerprint) is None:
            raise ValueError("feed_fingerprint must be a lowercase 64-hex SHA-256")
        _finite_vector("impact_position_world", self.impact_position_world, 3)
        normal = _finite_vector("stringbed_normal_world", self.stringbed_normal_world, 3)
        if not np.isclose(np.linalg.norm(normal), 1.0, atol=1e-5):
            raise ValueError("stringbed_normal_world must be unit length")
        _finite_vector("racket_linear_velocity_world", self.racket_linear_velocity_world, 3)
        _finite_vector("racket_angular_velocity_world", self.racket_angular_velocity_world, 3)
        _finite_vector("landing_target_xy", self.landing_target_xy, 2)
        if not np.isfinite(self.impact_time_s) or self.impact_time_s <= 0.0:
            raise ValueError("impact_time_s must be finite and positive")
        if not np.isfinite(self.apex_height_m) or self.apex_height_m <= 0.0:
            raise ValueError("apex_height_m must be finite and positive")
        if self.recovery_horizon_steps <= 0:
            raise ValueError("recovery_horizon_steps must be positive")
        if not self.provenance:
            raise ValueError("provenance must not be empty")


@dataclass(frozen=True)
class Stage3TargetBank:
    targets: tuple[DesiredImpactTarget, ...]
    source_fingerprint: str
    metadata: dict[str, Any]
    bank_sha256: str

    def aligned_to_feeds(self, feed_fingerprints: Iterable[str]) -> Stage3TargetBank:
        expected = tuple(str(value) for value in feed_fingerprints)
        actual = tuple(target.feed_fingerprint for target in self.targets)
        if actual != expected:
            raise ValueError(
                "Stage-3 v2 target bank is not in exact feed-bank order; regenerate it from the bound feed manifest"
            )
        return self


def build_target_bank(
    targets: Iterable[DesiredImpactTarget],
    *,
    source_fingerprint: str,
    metadata: dict[str, Any] | None = None,
) -> Stage3TargetBank:
    target_tuple = tuple(targets)
    if not target_tuple:
        raise ValueError("target bank must contain at least one target")
    if _SHA256_RE.fullmatch(source_fingerprint) is None:
        raise ValueError("source_fingerprint must be a lowercase 64-hex SHA-256")
    fingerprints = [target.feed_fingerprint for target in target_tuple]
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("target bank contains duplicate feed_fingerprint values")
    meta = dict(metadata or {})
    canonical = {
        "schema_version": SCHEMA_VERSION,
        "source_fingerprint": source_fingerprint,
        "metadata": meta,
        "targets": [asdict(target) for target in target_tuple],
    }
    return Stage3TargetBank(
        targets=target_tuple,
        source_fingerprint=source_fingerprint,
        metadata=meta,
        bank_sha256=_sha256(canonical),
    )


def save_target_bank(path: str | Path, bank: Stage3TargetBank) -> Path:
    destination = Path(path)
    payload = _payload(bank)
    validate_target_bank_payload(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def load_target_bank(
    path: str | Path,
    *,
    expected_feed_fingerprints: Iterable[str] | None = None,
    expected_source_fingerprint: str | None = None,
) -> Stage3TargetBank:
    payload = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )
    bank = validate_target_bank_payload(payload)
    if expected_source_fingerprint is not None and (bank.source_fingerprint != expected_source_fingerprint):
        raise ValueError("target-bank source fingerprint mismatch")
    if expected_feed_fingerprints is not None:
        bank.aligned_to_feeds(expected_feed_fingerprints)
    return bank


def validate_target_bank_payload(payload: Any) -> Stage3TargetBank:
    if not isinstance(payload, dict):
        raise ValueError("target-bank root must be an object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("target-bank schema mismatch")
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, list):
        raise ValueError("target-bank targets must be a list")
    targets = tuple(DesiredImpactTarget(**row) for row in raw_targets)
    bank = build_target_bank(
        targets,
        source_fingerprint=str(payload.get("source_fingerprint", "")),
        metadata=dict(payload.get("metadata", {}) or {}),
    )
    if payload.get("bank_sha256") != bank.bank_sha256:
        raise ValueError("target-bank content fingerprint mismatch")
    return bank


def target_arrays(bank: Stage3TargetBank) -> dict[str, np.ndarray]:
    """Convert a validated bank to stable arrays consumed by CPU and MJX."""

    return {
        "impact_position_world": np.asarray(
            [target.impact_position_world for target in bank.targets], dtype=np.float32
        ),
        "impact_time_s": np.asarray([target.impact_time_s for target in bank.targets], dtype=np.float32),
        "stringbed_normal_world": np.asarray(
            [target.stringbed_normal_world for target in bank.targets], dtype=np.float32
        ),
        "racket_linear_velocity_world": np.asarray(
            [target.racket_linear_velocity_world for target in bank.targets], dtype=np.float32
        ),
        "racket_angular_velocity_world": np.asarray(
            [target.racket_angular_velocity_world for target in bank.targets], dtype=np.float32
        ),
        "landing_target_xy": np.asarray([target.landing_target_xy for target in bank.targets], dtype=np.float32),
        "apex_height_m": np.asarray([target.apex_height_m for target in bank.targets], dtype=np.float32),
        "recovery_horizon_steps": np.asarray(
            [target.recovery_horizon_steps for target in bank.targets], dtype=np.int32
        ),
    }


def _payload(bank: Stage3TargetBank) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_fingerprint": bank.source_fingerprint,
        "metadata": bank.metadata,
        "targets": [asdict(target) for target in bank.targets],
        "bank_sha256": bank.bank_sha256,
    }


def _finite_vector(name: str, values: Any, size: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return array


def _sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load_targets_jsonl(path: str | Path) -> tuple[DesiredImpactTarget, ...]:
    rows: list[DesiredImpactTarget] = []
    for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid target JSONL line {line_number}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"target JSONL line {line_number} must be an object")
        try:
            rows.append(DesiredImpactTarget(**payload))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid target JSONL line {line_number}: {exc}") from exc
    if not rows:
        raise ValueError("target JSONL contains no records")
    return tuple(rows)


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _feed_difficulty(feed: Any) -> float:
    point = np.asarray(feed.intercept_point, dtype=float)
    return (
        abs(float(point[1]))
        + 0.6 * abs(float(point[2]) - 1.8)
        + 0.25 * abs(float(feed.intercept_time_s) - 0.75)
        + 0.02 * float(np.linalg.norm(feed.intercept_velocity))
    )


def source_fingerprint_from_event_metrics(metrics_path: str | Path, *, split: str) -> tuple[str, dict[str, Any]]:
    """Resolve the target source identity from a passed event-reference artifact."""

    path = Path(metrics_path).expanduser().resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(payload, dict) or payload.get("schema_version") != ("event_reference_promotion_metrics_v1"):
        raise ValueError("target generation requires event-reference promotion metrics v1")
    recorded = payload.get("metrics_fingerprint")
    unsigned = dict(payload)
    unsigned.pop("metrics_fingerprint", None)
    if recorded != _sha256(unsigned):
        raise ValueError("event-reference metrics fingerprint is stale")
    if (
        float(payload.get("artifact_binding_verified", 0.0)) != 1.0
        or float(payload.get("event_bank_binding_verified", 0.0)) != 1.0
    ):
        raise ValueError("event-reference source/bank binding did not pass")
    split_name = str(split).strip().lower()
    key = {
        "train": "train_reference_set_fingerprint",
        "validation": "validation_reference_set_fingerprint",
    }.get(split_name)
    if key is None:
        raise ValueError("reference split must be 'train' or 'validation'")
    fingerprint = payload.get(key)
    if not isinstance(fingerprint, str) or _SHA256_RE.fullmatch(fingerprint) is None:
        raise ValueError(f"event-reference metrics has no valid {key}")
    return fingerprint, {
        "event_reference_metrics_path": str(path),
        "event_reference_metrics_content_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "event_reference_metrics_fingerprint": recorded,
        "reference_split": split_name,
        "reference_set_fingerprint_field": key,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl")
    parser.add_argument("--source-fingerprint")
    parser.add_argument("--event-reference-metrics")
    parser.add_argument("--reference-split", choices=("train", "validation"))
    parser.add_argument("--output")
    parser.add_argument("--expected-feed-fingerprints-json")
    parser.add_argument(
        "--feed-bank-path",
        help="Validated feed-bank artifact used to derive exact consumer order.",
    )
    parser.add_argument(
        "--consumer-order",
        choices=("stored", "difficulty_sorted"),
        default="stored",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "required_record_fields": [
                        field.name for field in DesiredImpactTarget.__dataclass_fields__.values()
                    ],
                    "feed_alignment": "exact_order",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    missing = [
        flag
        for flag, value in (
            ("--input-jsonl", args.input_jsonl),
            ("--event-reference-metrics", args.event_reference_metrics),
            ("--reference-split", args.reference_split),
            ("--output", args.output),
        )
        if not value
    ]
    if missing:
        parser.error("missing required arguments: " + ", ".join(missing))
    targets = load_targets_jsonl(args.input_jsonl)
    source_fingerprint, source_metadata = source_fingerprint_from_event_metrics(
        args.event_reference_metrics,
        split=args.reference_split,
    )
    if args.source_fingerprint is not None and args.source_fingerprint != source_fingerprint:
        raise ValueError("supplied source fingerprint differs from bound event-reference metrics")
    metadata = {
        "input_jsonl_sha256": hashlib.sha256(Path(args.input_jsonl).read_bytes()).hexdigest(),
        **source_metadata,
    }
    expected_feed_fingerprints: list[str] | None = None
    if args.feed_bank_path:
        from environment.overall_environment.src.shuttle_feeder import (
            feed_sample_fingerprint,
            load_feed_bank_with_manifest,
        )

        feeds, feed_manifest = load_feed_bank_with_manifest(args.feed_bank_path)
        if args.consumer_order == "difficulty_sorted":
            feeds = sorted(feeds, key=_feed_difficulty)
        expected_feed_fingerprints = [feed_sample_fingerprint(sample) for sample in feeds]
        metadata.update(
            {
                "feed_bank_path": str(Path(args.feed_bank_path).resolve()),
                "feed_bank_manifest_sha256": _sha256(feed_manifest),
                "consumer_order": args.consumer_order,
                "consumer_feed_fingerprints_sha256": _sha256({"sample_fingerprints": expected_feed_fingerprints}),
            }
        )
    bank = build_target_bank(
        targets,
        source_fingerprint=source_fingerprint,
        metadata=metadata,
    )
    if args.expected_feed_fingerprints_json and args.feed_bank_path:
        raise ValueError("use either --feed-bank-path or --expected-feed-fingerprints-json, not both")
    if args.expected_feed_fingerprints_json:
        expected = json.loads(
            Path(args.expected_feed_fingerprints_json).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
        if not isinstance(expected, list):
            raise ValueError("expected feed fingerprints JSON must be a list")
        expected_feed_fingerprints = [str(value) for value in expected]
    if expected_feed_fingerprints is not None:
        bank.aligned_to_feeds(expected_feed_fingerprints)
    save_target_bank(args.output, bank)
    print(json.dumps({"output": str(Path(args.output)), "bank_sha256": bank.bank_sha256}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

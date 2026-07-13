"""Fail-closed promotion gates for the forehand-clear training pipeline."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class GateCheck:
    name: str
    value: float | None
    operator: str
    threshold: float
    passed: bool
    source_key: str | None


@dataclass(frozen=True)
class PromotionReport:
    schema_version: str
    stage: str
    passed: bool
    consecutive_required: int
    consecutive_evaluated: int
    evaluations: tuple[tuple[GateCheck, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CANONICAL_PROMOTION_THRESHOLDS = MappingProxyType(
    {
        "stage1": MappingProxyType(
            {
                "max_early_termination_rate": 0.05,
                "min_frame_coverage": 0.95,
                "max_relative_site_position_error_m": 0.09,
                "max_action_saturation_fraction": 0.05,
                "max_activation_energy": 0.35,
            }
        ),
        "stage1r": MappingProxyType(
            {
                "max_body_site_error_relative_degradation": 0.05,
                "max_right_hand_error_relative_degradation": 0.05,
                "max_racket_head_position_error_relative_degradation": 0.05,
                "max_racket_head_rotation_error_relative_degradation": 0.05,
                "max_early_termination_rate_absolute_increase": 0.02,
            }
        ),
        "stage2": MappingProxyType(
            {
                "max_early_termination_rate": 0.05,
                "min_frame_coverage": 0.95,
                "max_racket_position_error_m": 0.05,
                "max_racket_rotation_error_rad": 0.20,
                "max_body_metric_relative_degradation": 0.10,
            }
        ),
    }
)


def validate_promotion_threshold_config(
    stage: str,
    promotion: Mapping[str, Any],
) -> None:
    """Reject missing or misleading YAML threshold overrides.

    Online Stage-1/Stage-2 stopping and the offline gate deliberately share one
    canonical threshold table.  YAML retains the values for readable resolved
    configs, but it is an asserted mirror rather than a second authority.
    """

    key = str(stage).lower()
    if key not in CANONICAL_PROMOTION_THRESHOLDS:
        raise ValueError(f"no canonical promotion threshold config for {stage!r}")
    for field, expected in CANONICAL_PROMOTION_THRESHOLDS[key].items():
        if field not in promotion:
            raise ValueError(
                f"{key} promotion config is missing canonical threshold {field!r}"
            )
        raw = promotion[field]
        if isinstance(raw, bool):
            raise ValueError(f"{key} promotion threshold {field!r} must be numeric")
        try:
            actual = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{key} promotion threshold {field!r} must be numeric"
            ) from exc
        if not math.isfinite(actual) or not math.isclose(
            actual,
            float(expected),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{key} promotion threshold drift for {field!r}: "
                f"configured={raw!r} canonical={expected!r}"
            )


_T = CANONICAL_PROMOTION_THRESHOLDS


_RULES: dict[str, tuple[tuple[str, str, float, tuple[str, ...]], ...]] = {
    "stage1": (
        ("early_termination_rate", "<=", _T["stage1"]["max_early_termination_rate"], ("val_early_termination_rate", "early_termination_rate")),
        ("frame_coverage", ">=", _T["stage1"]["min_frame_coverage"], ("val_frame_coverage", "frame_coverage")),
        ("relative_site_position_error_m", "<=", _T["stage1"]["max_relative_site_position_error_m"], ("val_err_rpos", "err_rpos", "relative_site_position_error_m")),
        (
            "action_saturation_fraction",
            "<=",
            _T["stage1"]["max_action_saturation_fraction"],
            ("val_action_saturation_fraction", "action_saturation_fraction"),
        ),
        (
            "activation_energy",
            "<=",
            _T["stage1"]["max_activation_energy"],
            ("val_activation_energy", "activation_energy"),
        ),
    ),
    "stage1r": (
        (
            "body_site_relative_degradation",
            "<=",
            _T["stage1r"]["max_body_site_error_relative_degradation"],
            (
                "body_site_relative_degradation",
                "metrics.body_site_error.relative_degradation",
            ),
        ),
        (
            "right_hand_relative_degradation",
            "<=",
            _T["stage1r"]["max_right_hand_error_relative_degradation"],
            (
                "right_hand_relative_degradation",
                "metrics.right_hand_site_error.relative_degradation",
            ),
        ),
        (
            "racket_head_position_relative_degradation",
            "<=",
            _T["stage1r"]["max_racket_head_position_error_relative_degradation"],
            (
                "racket_head_position_relative_degradation",
                "metrics.racket_head_position_error.relative_degradation",
            ),
        ),
        (
            "racket_head_rotation_relative_degradation",
            "<=",
            _T["stage1r"]["max_racket_head_rotation_error_relative_degradation"],
            (
                "racket_head_rotation_relative_degradation",
                "metrics.racket_head_rotation_error.relative_degradation",
            ),
        ),
        (
            "early_termination_gap",
            "<=",
            _T["stage1r"]["max_early_termination_rate_absolute_increase"],
            (
                "early_termination_gap",
                "metrics.early_termination.absolute_degradation",
            ),
        ),
        ("paired_seed_verified", ">=", 1.0, ("paired_seed_verified",)),
        (
            "new_root_hand_racket_spike_count",
            "<=",
            0.0,
            ("new_root_hand_racket_spike_count",),
        ),
    ),
    "stage2": (
        ("early_termination_rate", "<=", _T["stage2"]["max_early_termination_rate"], ("val_early_termination_rate", "early_termination_rate")),
        ("frame_coverage", ">=", _T["stage2"]["min_frame_coverage"], ("val_frame_coverage", "frame_coverage")),
        (
            "racket_position_error_m",
            "<=",
            _T["stage2"]["max_racket_position_error_m"],
            (
                "val_err_racket_pos",
                "err_racket_pos",
                "val_racket_position_error_m",
                "racket_position_error_m",
                "val_racket_pos_error",
            ),
        ),
        (
            "racket_rotation_error_rad",
            "<=",
            _T["stage2"]["max_racket_rotation_error_rad"],
            (
                "val_err_racket_rot",
                "err_racket_rot",
                "val_racket_rotation_error_rad",
                "racket_rotation_error_rad",
                "val_racket_rot_error",
            ),
        ),
        ("body_metric_relative_degradation", "<=", _T["stage2"]["max_body_metric_relative_degradation"], ("body_metric_relative_degradation",)),
    ),
    "stage3": (
        (
            "evaluated_feed_count",
            ">=",
            128.0,
            ("evaluated_feed_count", "heldout_feed_count", "feed_count"),
        ),
        ("no_fall_rate", ">=", 0.95, ("no_fall_rate", "val_no_fall_rate")),
        ("hit_rate", ">=", 0.90, ("hit_rate", "val_hit_rate")),
        ("crossed_net_rate", ">=", 0.85, ("crossed_net_rate", "val_crossed_net_rate")),
        ("opponent_back_landing_rate", ">=", 0.70, ("opponent_back_landing_rate", "val_opponent_back_landing_rate")),
        (
            "racket_head_speed_m_s",
            ">=",
            8.0,
            (
                "racket_head_speed_m_s",
                "mean_racket_head_speed_m_s",
                "val_racket_head_speed_m_s",
            ),
        ),
        (
            "net_clearance_m",
            ">=",
            0.25,
            ("net_clearance_m", "mean_net_clearance_m", "val_net_clearance_m"),
        ),
        ("control_finite", ">=", 1.0, ("control_finite",)),
        ("min_root_height_m", ">=", 0.55, ("min_root_height_m",)),
        (
            "body_action_saturation_fraction",
            "<=",
            0.01,
            ("body_action_saturation_fraction",),
        ),
        (
            "full_action_saturation_fraction",
            "<=",
            0.01,
            ("full_action_saturation_fraction",),
        ),
        (
            "normalized_control_energy",
            "<=",
            0.35,
            ("normalized_control_energy",),
        ),
        (
            "raw_latent_saturation",
            "<=",
            0.10,
            ("raw_latent_saturation",),
        ),
        (
            "lab_state_ood_fraction_p95",
            "<=",
            0.01,
            ("lab_state_ood_fraction_p95",),
        ),
        (
            "max_attachment_translation_drift_m",
            "<=",
            0.005,
            ("max_attachment_translation_drift_m",),
        ),
        (
            "max_attachment_rotation_drift_rad",
            "<=",
            0.05,
            ("max_attachment_rotation_drift_rad",),
        ),
        (
            "body_relative_deviation_to_prior",
            "<=",
            0.25,
            ("body_relative_deviation_to_prior", "naturalness.body_relative_deviation_to_prior"),
        ),
        (
            "right_hand_site_rmse_to_prior_m",
            "<=",
            0.12,
            ("right_hand_site_rmse_to_prior_m", "naturalness.right_hand_site_rmse_to_prior_m"),
        ),
        (
            "right_hand_site_relative_deviation_to_prior",
            "<=",
            0.25,
            (
                "right_hand_site_relative_deviation_to_prior",
                "naturalness.right_hand_site_relative_deviation_to_prior",
            ),
        ),
        (
            "racket_position_rmse_to_prior_m",
            "<=",
            0.12,
            ("racket_position_rmse_to_prior_m", "naturalness.racket_position_rmse_to_prior_m"),
        ),
        (
            "racket_position_relative_deviation_to_prior",
            "<=",
            0.25,
            (
                "racket_position_relative_deviation_to_prior",
                "naturalness.racket_position_relative_deviation_to_prior",
            ),
        ),
        (
            "racket_rotation_rmse_to_prior_rad",
            "<=",
            0.35,
            ("racket_rotation_rmse_to_prior_rad", "naturalness.racket_rotation_rmse_to_prior_rad"),
        ),
        (
            "racket_rotation_relative_deviation_to_prior",
            "<=",
            0.25,
            (
                "racket_rotation_relative_deviation_to_prior",
                "naturalness.racket_rotation_relative_deviation_to_prior",
            ),
        ),
        (
            "prior_vs_direct_body_racket_relative_degradation",
            "<=",
            0.10,
            (
                "prior_vs_direct_body_racket_relative_degradation",
                "naturalness.prior_vs_direct_body_racket_relative_degradation",
            ),
        ),
        (
            "stage3_vs_direct_naturalness_upper_bound",
            "<=",
            0.375,
            (
                "stage3_vs_direct_naturalness_upper_bound",
                "naturalness.stage3_vs_direct_naturalness_upper_bound",
            ),
        ),
        (
            "artifact_binding_verified",
            ">=",
            1.0,
            ("artifact_binding_verified",),
        ),
    ),
}


def evaluate_promotion(
    stage: str,
    metrics: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    *,
    consecutive: int | None = None,
    baseline_metrics: Mapping[str, Any] | None = None,
) -> PromotionReport:
    """Evaluate the latest N validation records; absent/non-finite metrics fail."""
    key = str(stage).lower()
    if key not in _RULES:
        raise ValueError(f"unsupported promotion stage {stage!r}; expected one of {sorted(_RULES)}")
    records = [metrics] if isinstance(metrics, Mapping) else list(metrics)
    required = int(consecutive if consecutive is not None else (3 if key in {"stage1", "stage2"} else 1))
    if required <= 0:
        raise ValueError("consecutive must be positive")
    selected = records[-required:]
    evaluations = tuple(
        _evaluate_record(key, record, baseline_metrics=baseline_metrics)
        for record in selected
    )
    passed = len(selected) == required and all(all(check.passed for check in checks) for checks in evaluations)
    return PromotionReport(
        schema_version="forehand_clear_promotion_v1",
        stage=key,
        passed=passed,
        consecutive_required=required,
        consecutive_evaluated=len(selected),
        evaluations=evaluations,
    )


def _evaluate_record(
    stage: str,
    metrics: Mapping[str, Any],
    *,
    baseline_metrics: Mapping[str, Any] | None = None,
) -> tuple[GateCheck, ...]:
    normalized = _flatten_metrics(metrics)
    if stage == "stage1r" and "paired_seed_verified" not in normalized:
        seed_hash = normalized.get("seed_hash")
        pair_count = _first_finite(normalized, ("pair_count",))
        normalized["paired_seed_verified"] = float(
            isinstance(seed_hash, str)
            and bool(seed_hash.strip())
            and pair_count is not None
            and pair_count[0] > 0.0
        )
    if stage == "stage3" and "no_fall_rate" not in normalized:
        fall = _first_finite(normalized, ("fall_rate", "val_fall_rate"))
        if fall is not None:
            normalized["no_fall_rate"] = 1.0 - fall[0]
    if stage == "stage2" and "body_metric_relative_degradation" not in normalized:
        degradation = _body_metric_relative_degradation(normalized, baseline_metrics)
        if degradation is not None:
            normalized["body_metric_relative_degradation"] = degradation
    checks: list[GateCheck] = []
    for name, operator, threshold, aliases in _RULES[stage]:
        found = _first_finite(normalized, aliases)
        value, source = (None, None) if found is None else found
        passed = False if value is None else (value <= threshold if operator == "<=" else value >= threshold)
        checks.append(GateCheck(name, value, operator, threshold, bool(passed), source))
    return tuple(checks)


_BODY_BASELINE_METRIC_ALIASES: tuple[tuple[str, ...], ...] = (
    ("val_err_rpos", "err_rpos"),
    ("val_err_joint_pos", "err_joint_pos"),
    ("val_err_joint_vel", "err_joint_vel"),
    ("val_err_site_abs", "err_site_abs"),
)


def _body_metric_relative_degradation(
    current: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any] | None,
) -> float | None:
    """Return the worst relative body-error increase over the Stage-1 baseline.

    The gate is fail-closed when no baseline is supplied.  Metrics with a zero
    baseline are accepted only when the current value is also numerically zero;
    otherwise their degradation is infinite.
    """
    if baseline_metrics is None:
        return None
    baseline = _flatten_metrics(baseline_metrics)
    degradations: list[float] = []
    for metric_index, aliases in enumerate(_BODY_BASELINE_METRIC_ALIASES):
        cur = _first_finite(current, aliases)
        ref = _first_finite(baseline, aliases)
        # Relative-site position is the mandatory Stage-1/Stage-2 body metric.
        # Other configured metrics must be present on both sides if either side
        # reports them; silently dropping one would understate degradation.
        if metric_index == 0 and (cur is None or ref is None):
            return None
        if (cur is None) != (ref is None):
            return None
        if cur is None or ref is None:
            continue
        cur_value, _ = cur
        ref_value, _ = ref
        if abs(ref_value) <= 1e-12:
            degradations.append(0.0 if abs(cur_value) <= 1e-12 else 1.0e30)
        else:
            degradations.append((cur_value - ref_value) / abs(ref_value))
    return max(degradations) if degradations else None


def _flatten_metrics(metrics: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in metrics.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            result.update(_flatten_metrics(value, full_key))
        else:
            result[str(key)] = value
            result[full_key] = value
    return result


def _first_finite(metrics: Mapping[str, Any], aliases: tuple[str, ...]) -> tuple[float, str] | None:
    for alias in aliases:
        if alias not in metrics:
            continue
        try:
            value = float(metrics[alias])
        except (TypeError, ValueError):
            # Alias order is a priority order.  Once the producer emitted the
            # preferred key, a malformed value must not be hidden by a legacy
            # fallback alias in the same payload.
            return None
        if math.isfinite(value):
            return value, alias
        return None
    return None


def extract_validation_records(payload: Any) -> list[Mapping[str, Any]]:
    """Normalize direct, offline, and online-progress metric payloads.

    ``promotion_progress.json`` stores audit-rich ``history`` entries while
    ordinary evaluators store ``validations``.  Both are accepted here so one
    canonical artifact can feed the offline gate and the Stage-2 baseline.
    """

    if isinstance(payload, Mapping):
        # An online progress file carries both history and a validations alias;
        # history is authoritative because it binds metrics to checkpoints.
        if "history" in payload:
            history = payload["history"]
            if not isinstance(history, list):
                raise ValueError("promotion history must be a JSON list")
            records = []
            for event in history:
                if not isinstance(event, Mapping) or not isinstance(
                    event.get("metrics"), Mapping
                ):
                    raise ValueError(
                        "each promotion history entry must contain a metrics object"
                    )
                records.append(event["metrics"])
            if not records:
                raise ValueError("promotion history is empty")
            return records
        if "validations" in payload:
            payload = payload["validations"]
        else:
            # Keep wrapper fields: Stage-1R paired reports contain a nested
            # ``metrics`` object plus seed/spike evidence at the root.
            return [payload]
    if isinstance(payload, list):
        if not payload:
            raise ValueError("validations list is empty")
        if not all(isinstance(record, Mapping) for record in payload):
            raise ValueError("each validation must be a JSON object")
        return list(payload)
    raise ValueError(
        "metrics must be a JSON object, validations list, or promotion history"
    )


def latest_validation_record(payload: Any) -> Mapping[str, Any]:
    """Resolve a Stage-1 baseline payload to its latest validation record."""

    if (
        isinstance(payload, Mapping)
        and "history" not in payload
        and "validations" not in payload
        and "metrics" in payload
    ):
        metrics = payload["metrics"]
        if not isinstance(metrics, Mapping):
            raise ValueError("baseline metrics must be a JSON object")
        return metrics
    return extract_validation_records(payload)[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=sorted(_RULES))
    parser.add_argument("--metrics", required=True, help="JSON object or list of validation objects")
    parser.add_argument("--consecutive", type=int, default=None)
    parser.add_argument(
        "--baseline-metrics",
        default=None,
        help="Stage-1 metrics JSON used to derive Stage-2 body degradation.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint bound to a production Stage-1R paired-rollout report.",
    )
    parser.add_argument(
        "--finger-perturb-qpos-scale",
        type=float,
        default=None,
        help="Expected Stage-1R perturbation rung; never inferred from the report.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--require_pass", action="store_true")
    args = parser.parse_args()
    if args.stage == "stage1r":
        if args.checkpoint is None or args.finger_perturb_qpos_scale is None:
            parser.error(
                "production stage1r gate requires --checkpoint and "
                "--finger-perturb-qpos-scale"
            )
        from musclemimic.badminton.stage1r_artifact import validate_stage1r_report

        validate_stage1r_report(
            args.metrics,
            expected_checkpoint=args.checkpoint,
            expected_perturb_qpos_scale=args.finger_perturb_qpos_scale,
        )
    payload = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    payload = extract_validation_records(payload)
    baseline = (
        None
        if args.baseline_metrics is None
        else json.loads(Path(args.baseline_metrics).read_text(encoding="utf-8"))
    )
    if baseline is not None:
        baseline = latest_validation_record(baseline)
    report = evaluate_promotion(
        args.stage,
        payload,
        consecutive=args.consecutive,
        baseline_metrics=baseline,
    )
    encoded = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 2 if args.require_pass and not report.passed else 0


if __name__ == "__main__":
    raise SystemExit(main())

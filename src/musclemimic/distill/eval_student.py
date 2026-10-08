"""Helpers for comparing teacher and student evaluation metrics."""

from __future__ import annotations

import csv
import json
import math
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


METRIC_RE = re.compile(r"^([A-Za-z0-9_./-]+):\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)$")
REQUIRED_EVAL_METRICS = (
    "mean_episode_return",
    "mean_episode_length",
    "early_termination_rate",
    "frame_coverage",
    "err_rpos",
    "err_racket_pos",
    "err_racket_rot",
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
    "err_racket_pos",
    "err_racket_rot",
    "reward_qpos",
    "reward_qvel",
    "reward_root_pos",
    "reward_root_vel",
    "reward_rpos",
    "reward_rquat",
    "reward_rvel_rot",
    "reward_rvel_lin",
)


@dataclass(frozen=True)
class DistillAcceptanceThresholds:
    """Closed-loop promotion thresholds for a Stage-2 distilled student."""

    min_return_ratio: float = 0.90
    target_return_ratio: float = 0.95
    min_completion_ratio: float = 0.90
    max_early_termination_delta: float = 0.05
    max_tracking_error_relative_degradation: float = 0.10
    plateau_min_points: int = 5
    plateau_window_points: int = 5
    max_plateau_normalized_abs_slope: float = 0.01
    max_plateau_normalized_span: float = 0.05
    temporal_search_max_lag_steps: int = 5
    max_abs_temporal_best_lag_steps: int = 2
    max_temporal_lag_mse_improvement_fraction: float = 0.05
    min_temporal_sequences: int = 5


DEFAULT_DISTILL_ACCEPTANCE_THRESHOLDS = DistillAcceptanceThresholds()


def _validated_thresholds(
    thresholds: DistillAcceptanceThresholds | None,
) -> DistillAcceptanceThresholds:
    limits = thresholds or DEFAULT_DISTILL_ACCEPTANCE_THRESHOLDS
    integer_fields = {
        "plateau_min_points": limits.plateau_min_points,
        "plateau_window_points": limits.plateau_window_points,
        "temporal_search_max_lag_steps": limits.temporal_search_max_lag_steps,
        "max_abs_temporal_best_lag_steps": limits.max_abs_temporal_best_lag_steps,
        "min_temporal_sequences": limits.min_temporal_sequences,
    }
    if any(
        isinstance(value, bool) or float(value) != int(value)
        for value in integer_fields.values()
    ):
        raise ValueError(f"distill integer thresholds must be integral: {integer_fields}")
    if int(limits.plateau_min_points) < 2:
        raise ValueError("plateau_min_points must be at least two")
    if int(limits.plateau_window_points) < 2:
        raise ValueError("plateau_window_points must be at least two")
    if int(limits.temporal_search_max_lag_steps) < 1:
        raise ValueError("temporal_search_max_lag_steps must be positive")
    if not 0 <= int(limits.max_abs_temporal_best_lag_steps) <= int(
        limits.temporal_search_max_lag_steps
    ):
        raise ValueError(
            "max_abs_temporal_best_lag_steps must lie inside the temporal search window"
        )
    if int(limits.min_temporal_sequences) <= 0:
        raise ValueError("min_temporal_sequences must be positive")
    nonnegative = {
        "min_return_ratio": limits.min_return_ratio,
        "target_return_ratio": limits.target_return_ratio,
        "min_completion_ratio": limits.min_completion_ratio,
        "max_early_termination_delta": limits.max_early_termination_delta,
        "max_tracking_error_relative_degradation": (
            limits.max_tracking_error_relative_degradation
        ),
        "max_plateau_normalized_abs_slope": limits.max_plateau_normalized_abs_slope,
        "max_plateau_normalized_span": limits.max_plateau_normalized_span,
        "max_temporal_lag_mse_improvement_fraction": (
            limits.max_temporal_lag_mse_improvement_fraction
        ),
    }
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in nonnegative.values()):
        raise ValueError(f"distill gate thresholds must be finite and non-negative: {nonnegative}")
    if float(limits.target_return_ratio) < float(limits.min_return_ratio):
        raise ValueError("target_return_ratio must be at least min_return_ratio")
    if float(limits.min_completion_ratio) > 1.0:
        raise ValueError("min_completion_ratio must not exceed one")
    if float(limits.max_early_termination_delta) > 1.0:
        raise ValueError("max_early_termination_delta must not exceed one")
    if float(limits.max_temporal_lag_mse_improvement_fraction) > 1.0:
        raise ValueError(
            "max_temporal_lag_mse_improvement_fraction must not exceed one"
        )
    return limits


def evaluate_mse_plateau(
    history: Sequence[Mapping[str, Any]] | None,
    thresholds: DistillAcceptanceThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate a deterministic held-out action-MSE history, failing closed.

    The normalized slope is the least-squares slope per validation point divided
    by the mean MSE in the final window.  The normalized span is
    ``(max-min)/mean`` over the same fixed window.  Both are scale independent.
    """

    limits = _validated_thresholds(thresholds)
    required = max(int(limits.plateau_min_points), int(limits.plateau_window_points))
    errors: list[str] = []
    rows: list[dict[str, float | int]] = []
    if history is None or isinstance(history, (str, bytes, Mapping)):
        errors.append("missing_or_invalid_history")
    else:
        previous_step: int | None = None
        for index, raw in enumerate(history):
            if not isinstance(raw, Mapping):
                errors.append(f"history[{index}]_not_object")
                continue
            try:
                step_value = float(raw.get("step"))
                mse = float(raw.get("action_mse"))
            except (TypeError, ValueError):
                errors.append(f"history[{index}]_missing_numeric_step_or_action_mse")
                continue
            if (
                not math.isfinite(step_value)
                or step_value < 0.0
                or step_value != math.floor(step_value)
            ):
                errors.append(f"history[{index}]_invalid_step")
                continue
            step = int(step_value)
            if previous_step is not None and step <= previous_step:
                errors.append("history_steps_not_strictly_increasing")
            previous_step = step
            if not math.isfinite(mse) or mse < 0.0:
                errors.append(f"history[{index}]_invalid_action_mse")
                continue
            rows.append({"step": step, "action_mse": mse})

    sufficient = len(rows) >= required and not errors
    normalized_slope = normalized_abs_slope = normalized_span = None
    window: list[dict[str, float | int]] = []
    if sufficient:
        window = rows[-int(limits.plateau_window_points) :]
        values = np.asarray([row["action_mse"] for row in window], dtype=np.float64)
        x = np.arange(values.size, dtype=np.float64)
        slope = float(np.polyfit(x, values, 1)[0])
        denominator = max(abs(float(np.mean(values))), 1e-12)
        normalized_slope = slope / denominator
        normalized_abs_slope = abs(normalized_slope)
        normalized_span = float(np.max(values) - np.min(values)) / denominator

    checks = {
        "enough_validation_points": bool(sufficient),
        "normalized_abs_slope": bool(
            normalized_abs_slope is not None
            and normalized_abs_slope <= float(limits.max_plateau_normalized_abs_slope)
        ),
        "normalized_span": bool(
            normalized_span is not None
            and normalized_span <= float(limits.max_plateau_normalized_span)
        ),
    }
    return {
        "schema_version": "direct_bc_convergence_v1",
        "metric": "heldout_deterministic_action_mse",
        "num_points": len(rows),
        "required_points": required,
        "window_points": int(limits.plateau_window_points),
        "history": rows,
        "window": window,
        "normalized_slope": normalized_slope,
        "normalized_abs_slope": normalized_abs_slope,
        "normalized_span": normalized_span,
        "errors": sorted(set(errors)),
        "checks": checks,
        "thresholds": {
            "min_points": int(limits.plateau_min_points),
            "window_points": int(limits.plateau_window_points),
            "max_normalized_abs_slope": float(
                limits.max_plateau_normalized_abs_slope
            ),
            "max_normalized_span": float(limits.max_plateau_normalized_span),
        },
        "plateau": bool(all(checks.values()) and not errors),
        "passed": bool(all(checks.values()) and not errors),
    }


def _best_lag(lag_mse: Mapping[int, float]) -> tuple[int, float]:
    finite = [
        (int(lag), float(value))
        for lag, value in lag_mse.items()
        if math.isfinite(float(value))
    ]
    if not finite:
        raise ValueError("temporal lag audit has no finite lag MSE")
    lag, value = min(finite, key=lambda item: (item[1], abs(item[0]), item[0]))
    return lag, value


def evaluate_temporal_drift(
    *,
    student_action: Any,
    teacher_action: Any,
    motion_uid: Any,
    traj_step: Any,
    rollout_uid: Any | None = None,
    actuator_names: Sequence[str] | None = None,
    checkpoint_actuator_names: Sequence[str] | None = None,
    thresholds: DistillAcceptanceThresholds | None = None,
) -> dict[str, Any]:
    """Measure action phase drift on held-out, name-aligned motion sequences.

    Rows are grouped by ``motion_uid`` and, when available, ``rollout_uid``;
    ``traj_step`` orders each group.  Every candidate lag uses the same central
    frames, so shorter overlap at large lags cannot bias the selected lag.
    """

    from musclemimic.distill.action_schema import actuator_schema_hash

    limits = _validated_thresholds(thresholds)
    student = np.asarray(student_action, dtype=np.float64)
    teacher = np.asarray(teacher_action, dtype=np.float64)
    motions = np.asarray(motion_uid)
    steps = np.asarray(traj_step)
    rollouts = None if rollout_uid is None else np.asarray(rollout_uid)
    errors: list[str] = []

    if student.ndim != 2 or teacher.shape != student.shape or student.shape[0] == 0:
        errors.append("student_teacher_action_shape_mismatch")
    row_count = int(student.shape[0]) if student.ndim >= 1 else 0
    for label, values in (("motion_uid", motions), ("traj_step", steps)):
        if values.ndim != 1 or int(values.shape[0]) != row_count:
            errors.append(f"{label}_shape_mismatch")
    if rollouts is not None and (rollouts.ndim != 1 or int(rollouts.shape[0]) != row_count):
        errors.append("rollout_uid_shape_mismatch")
    if not np.isfinite(student).all() or not np.isfinite(teacher).all():
        errors.append("nonfinite_action")

    names = [] if actuator_names is None else [str(name) for name in actuator_names]
    checkpoint_names = (
        []
        if checkpoint_actuator_names is None
        else [str(name) for name in checkpoint_actuator_names]
    )
    if not names or len(names) != (student.shape[1] if student.ndim == 2 else -1):
        errors.append("missing_or_invalid_dataset_actuator_names")
    if not checkpoint_names or checkpoint_names != names:
        errors.append("checkpoint_actuator_schema_mismatch")
    action_schema_hash = None
    if names:
        try:
            action_schema_hash = actuator_schema_hash(names)
        except ValueError:
            errors.append("invalid_actuator_names")

    search = int(limits.temporal_search_max_lag_steps)
    lags = tuple(range(-search, search + 1))
    global_sse = {lag: 0.0 for lag in lags}
    global_count = {lag: 0 for lag in lags}
    motion_sse: dict[int, dict[int, float]] = {}
    motion_count: dict[int, dict[int, int]] = {}
    usable_sequences = 0
    too_short_sequences = 0

    if not errors:
        if not np.issubdtype(motions.dtype, np.integer):
            if not np.all(np.isfinite(motions)) or not np.all(motions == np.floor(motions)):
                errors.append("motion_uid_not_integer")
        if not np.issubdtype(steps.dtype, np.integer):
            if not np.all(np.isfinite(steps)) or not np.all(steps == np.floor(steps)):
                errors.append("traj_step_not_integer")
        if rollouts is not None and not np.issubdtype(rollouts.dtype, np.integer):
            if not np.all(np.isfinite(rollouts)) or not np.all(rollouts == np.floor(rollouts)):
                errors.append("rollout_uid_not_integer")

    if not errors:
        motion_int = motions.astype(np.int64)
        step_int = steps.astype(np.int64)
        rollout_int = (
            np.zeros(row_count, dtype=np.int64)
            if rollouts is None
            else rollouts.astype(np.int64)
        )
        if np.any(motion_int < 0) or np.any(step_int < 0) or np.any(rollout_int < 0):
            errors.append("negative_sequence_identity")
        else:
            keys = sorted(set(zip(motion_int.tolist(), rollout_int.tolist(), strict=True)))
            for motion, rollout in keys:
                indices = np.flatnonzero((motion_int == motion) & (rollout_int == rollout))
                ordered = indices[np.argsort(step_int[indices], kind="stable")]
                ordered_steps = step_int[ordered]
                if len(np.unique(ordered_steps)) != len(ordered_steps):
                    errors.append(f"duplicate_traj_step:motion={motion}:rollout={rollout}")
                    continue
                boundaries = np.flatnonzero(np.diff(ordered_steps) != 1) + 1
                start = 0
                for end in (*boundaries.tolist(), len(ordered)):
                    segment = ordered[start:end]
                    start = end
                    if len(segment) <= 2 * search:
                        too_short_sequences += 1
                        continue
                    usable_sequences += 1
                    anchor = segment[search : len(segment) - search]
                    per_motion_sse = motion_sse.setdefault(
                        int(motion), {lag: 0.0 for lag in lags}
                    )
                    per_motion_count = motion_count.setdefault(
                        int(motion), {lag: 0 for lag in lags}
                    )
                    for lag in lags:
                        teacher_indices = segment[search + lag : len(segment) - search + lag]
                        diff = student[anchor] - teacher[teacher_indices]
                        sse = float(np.sum(np.square(diff)))
                        count = int(diff.size)
                        global_sse[lag] += sse
                        global_count[lag] += count
                        per_motion_sse[lag] += sse
                        per_motion_count[lag] += count

    lag_mse = {
        lag: (global_sse[lag] / global_count[lag] if global_count[lag] else float("nan"))
        for lag in lags
    }
    per_motion: dict[str, Any] = {}
    best_lag = None
    best_mse = zero_lag_mse = improvement = max_abs_motion_lag = None
    if usable_sequences and not errors:
        best_lag, best_mse = _best_lag(lag_mse)
        zero_lag_mse = float(lag_mse[0])
        improvement = max(0.0, zero_lag_mse - best_mse) / max(zero_lag_mse, 1e-12)
        motion_lags: list[int] = []
        for motion in sorted(motion_sse):
            current = {
                lag: motion_sse[motion][lag] / motion_count[motion][lag]
                for lag in lags
                if motion_count[motion][lag]
            }
            motion_best_lag, motion_best_mse = _best_lag(current)
            motion_zero = float(current[0])
            motion_improvement = max(0.0, motion_zero - motion_best_mse) / max(
                motion_zero, 1e-12
            )
            motion_lags.append(motion_best_lag)
            per_motion[str(motion)] = {
                "best_lag_steps": motion_best_lag,
                "zero_lag_mse": motion_zero,
                "best_lag_mse": motion_best_mse,
                "lag_mse_improvement_fraction": motion_improvement,
                "lag_mse": {str(lag): float(value) for lag, value in current.items()},
            }
        max_abs_motion_lag = max(abs(value) for value in motion_lags)

    checks = {
        "enough_sequences": usable_sequences >= int(limits.min_temporal_sequences),
        "global_best_lag": bool(
            best_lag is not None
            and abs(int(best_lag)) <= int(limits.max_abs_temporal_best_lag_steps)
        ),
        "per_motion_best_lag": bool(
            max_abs_motion_lag is not None
            and int(max_abs_motion_lag) <= int(limits.max_abs_temporal_best_lag_steps)
        ),
        "lag_mse_improvement": bool(
            improvement is not None
            and float(improvement)
            <= float(limits.max_temporal_lag_mse_improvement_fraction)
        ),
    }
    return {
        "schema_version": "direct_temporal_audit_v1",
        "action_semantics": "clipped_deterministic_student_mean_vs_teacher_action",
        "action_dim": int(student.shape[1]) if student.ndim == 2 else None,
        "actuator_names": names,
        "action_schema_hash": action_schema_hash,
        "row_count": row_count,
        "usable_sequence_count": usable_sequences,
        "too_short_sequence_count": too_short_sequences,
        "motion_count": len(per_motion),
        "sequence_identity": (
            ["motion_uid", "rollout_uid", "traj_step"]
            if rollout_uid is not None
            else ["motion_uid", "traj_step"]
        ),
        "search_max_lag_steps": search,
        "best_lag_steps": best_lag,
        "max_abs_motion_best_lag_steps": max_abs_motion_lag,
        "zero_lag_mse": zero_lag_mse,
        "best_lag_mse": best_mse,
        "lag_mse_improvement_fraction": improvement,
        "lag_mse": {
            str(lag): (float(value) if math.isfinite(float(value)) else None)
            for lag, value in lag_mse.items()
        },
        "per_motion": per_motion,
        "errors": sorted(set(errors)),
        "checks": checks,
        "thresholds": {
            "search_max_lag_steps": search,
            "max_abs_best_lag_steps": int(limits.max_abs_temporal_best_lag_steps),
            "max_lag_mse_improvement_fraction": float(
                limits.max_temporal_lag_mse_improvement_fraction
            ),
            "min_sequences": int(limits.min_temporal_sequences),
        },
        "passed": bool(all(checks.values()) and not errors),
    }


def canonicalize_eval_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Expose training-validation JSON keys through stable comparison names.

    ``fullbody.eval`` writes ``val_*`` keys while older distillation reports
    parsed unprefixed stdout.  Returning both keeps old consumers working and
    prevents a comparison report from silently containing no required rows.
    """

    result = {str(key): float(value) for key, value in metrics.items()}
    for key, value in list(result.items()):
        if key.startswith("val_"):
            result.setdefault(key.removeprefix("val_"), value)
    if "completion_rate" not in result:
        if "frame_coverage" in result:
            result["completion_rate"] = float(result["frame_coverage"])
        elif "early_termination_rate" in result:
            result["completion_rate"] = max(0.0, 1.0 - float(result["early_termination_rate"]))
    return result


def evaluate_distill_acceptance(
    teacher: dict[str, float],
    student: dict[str, float],
    thresholds: DistillAcceptanceThresholds | None = None,
    *,
    convergence: Mapping[str, Any] | None = None,
    temporal_audit: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Return a machine-readable, fail-closed teacher/student promotion result."""

    limits = _validated_thresholds(thresholds)
    teacher = canonicalize_eval_metrics(teacher)
    student = canonicalize_eval_metrics(student)
    failed: list[str] = []
    missing: list[str] = []
    values: dict[str, float] = {}

    def require(name: str, payload: dict[str, float]) -> float | None:
        value = payload.get(name)
        if value is None:
            missing.append(name)
            return None
        value = float(value)
        if not math.isfinite(value):
            missing.append(name)
            return None
        return value

    teacher_return = require("mean_episode_return", teacher)
    student_return = require("mean_episode_return", student)
    if teacher_return is not None and student_return is not None:
        if abs(teacher_return) < 1e-12:
            missing.append("nonzero_teacher_mean_episode_return")
        else:
            values["return_ratio"] = student_return / teacher_return
            if values["return_ratio"] < limits.min_return_ratio:
                failed.append("return_ratio")

    teacher_completion = require("completion_rate", teacher)
    student_completion = require("completion_rate", student)
    if teacher_completion is not None and student_completion is not None:
        if abs(teacher_completion) < 1e-12:
            missing.append("nonzero_teacher_completion_rate")
        else:
            values["completion_ratio"] = student_completion / teacher_completion
            if values["completion_ratio"] < limits.min_completion_ratio:
                failed.append("completion_ratio")

    teacher_et = require("early_termination_rate", teacher)
    student_et = require("early_termination_rate", student)
    if teacher_et is not None and student_et is not None:
        values["early_termination_delta"] = student_et - teacher_et
        if values["early_termination_delta"] > limits.max_early_termination_delta:
            failed.append("early_termination_delta")

    # A racket-aware direct student may not promote on body tracking alone.
    # All three core errors are mandatory; joint errors remain report-only.
    for metric in ("err_rpos", "err_racket_pos", "err_racket_rot"):
        teacher_value = require(metric, teacher)
        student_value = require(metric, student)
        if teacher_value is None or student_value is None:
            continue
        baseline = float(teacher_value)
        key = f"{metric}_relative_degradation"
        if abs(baseline) < 1e-12:
            # Keep acceptance.json strict/portable instead of serializing Infinity.
            values[key] = 0.0 if abs(float(student_value)) < 1e-12 else 1.0e30
        else:
            values[key] = (float(student_value) - baseline) / abs(baseline)
        if values[key] > limits.max_tracking_error_relative_degradation:
            failed.append(key)

    convergence_payload = (
        convergence.get("convergence")
        if isinstance(convergence, Mapping)
        and isinstance(convergence.get("convergence"), Mapping)
        else convergence
    )
    convergence_gate = _evaluate_convergence_evidence(convergence_payload, limits)
    if not convergence_gate["passed"]:
        failed.append("mse_plateau")
        missing.extend(convergence_gate["evidence_missing"])
    for name in ("normalized_abs_slope", "normalized_span"):
        value = convergence_gate.get(name)
        if value is not None and math.isfinite(float(value)):
            values[f"convergence_{name}"] = float(value)

    temporal_gate = _evaluate_temporal_audit_evidence(temporal_audit, limits)
    if not temporal_gate["passed"]:
        failed.append("temporal_drift")
        missing.extend(temporal_gate["missing"])
    for name, value in temporal_gate["values"].items():
        values[f"temporal_{name}"] = value

    passed = not failed and not missing
    return {
        "passed": passed,
        "failed": sorted(set(failed)),
        "missing": sorted(set(missing)),
        "values": values,
        "thresholds": asdict(limits),
        "convergence": convergence_gate,
        "temporal": temporal_gate,
    }


def _evaluate_convergence_evidence(
    convergence: Mapping[str, Any] | None,
    thresholds: DistillAcceptanceThresholds,
) -> dict[str, Any]:
    """Recompute the plateau and validate that it came from a fixed val split."""

    history = convergence.get("history") if isinstance(convergence, Mapping) else None
    result = evaluate_mse_plateau(history, thresholds)
    missing: list[str] = []
    checks = {
        "schema_version": False,
        "deterministic_action_metric": False,
        "heldout_split": False,
        "fixed_motion_split": False,
        "periodic_schedule": False,
    }
    if not isinstance(convergence, Mapping):
        missing.append("convergence")
    else:
        checks["schema_version"] = (
            convergence.get("schema_version") == "direct_bc_convergence_v1"
        )
        if not checks["schema_version"]:
            missing.append("convergence.schema_version")
        checks["deterministic_action_metric"] = bool(
            convergence.get("metric") == "heldout_deterministic_action_mse"
            and convergence.get("deterministic") is True
        )
        if not checks["deterministic_action_metric"]:
            missing.append("convergence.deterministic_action_metric")
        checks["heldout_split"] = convergence.get("split") == "val"
        if not checks["heldout_split"]:
            missing.append("convergence.split")

        try:
            interval = float(convergence.get("evaluation_interval_steps"))
            history_steps = [int(row["step"]) for row in result["history"]]
        except (KeyError, TypeError, ValueError):
            valid_schedule = False
        else:
            diffs = np.diff(np.asarray(history_steps, dtype=np.int64))
            valid_schedule = bool(
                math.isfinite(interval)
                and interval > 0.0
                and interval == math.floor(interval)
                and history_steps
                and history_steps[0] == 0
                and (
                    diffs.size == 0
                    or (
                        np.all(diffs > 0)
                        and np.all(diffs[:-1] == int(interval))
                        and diffs[-1] <= int(interval)
                    )
                )
            )
        checks["periodic_schedule"] = valid_schedule
        if not checks["periodic_schedule"]:
            missing.append("convergence.evaluation_interval_steps")

        split = convergence.get("motion_split")
        motion_field = convergence.get("motion_field")
        valid_split = isinstance(split, Mapping)
        if valid_split:
            def stable_ids(raw: Any) -> set[int]:
                if not isinstance(raw, (list, tuple)) or not raw:
                    raise ValueError("motion IDs must be a non-empty JSON array")
                values: set[int] = set()
                for item in raw:
                    value = float(item)
                    if (
                        isinstance(item, bool)
                        or not math.isfinite(value)
                        or value < 0.0
                        or value != math.floor(value)
                    ):
                        raise ValueError("motion IDs must be non-negative integers")
                    values.add(int(value))
                return values

            try:
                train_ids = stable_ids(split.get("train_motion_ids"))
                val_ids = stable_ids(split.get("val_motion_ids"))
                val_samples = float(split.get("val_num_samples"))
            except (TypeError, ValueError):
                valid_split = False
            else:
                valid_split = bool(
                    split.get("schema_version") == "motion_split_v1"
                    and split.get("mode")
                    in {"explicit_motion_shards", "deterministic_motion_holdout"}
                    and isinstance(motion_field, str)
                    and motion_field
                    and split.get("motion_field") == motion_field
                    and train_ids
                    and val_ids
                    and not (train_ids & val_ids)
                    and math.isfinite(val_samples)
                    and val_samples > 0.0
                    and val_samples == math.floor(val_samples)
                )
        checks["fixed_motion_split"] = bool(valid_split)
        if not checks["fixed_motion_split"]:
            missing.append("convergence.motion_split")

    missing.extend(f"convergence.{error}" for error in result.get("errors", []))
    result["evidence_checks"] = checks
    result["evidence_missing"] = sorted(set(missing))
    result["plateau"] = bool(result["plateau"])
    result["passed"] = bool(result["plateau"] and all(checks.values()) and not missing)
    return result


def _evaluate_temporal_audit_evidence(
    audit: Mapping[str, Any] | None,
    thresholds: DistillAcceptanceThresholds,
) -> dict[str, Any]:
    """Re-evaluate a persisted temporal audit instead of trusting its passed bit."""

    missing: list[str] = []
    values: dict[str, float] = {}
    if not isinstance(audit, Mapping):
        return {
            "passed": False,
            "missing": ["temporal_audit"],
            "values": values,
            "checks": {},
        }
    if audit.get("schema_version") != "direct_temporal_audit_v1":
        missing.append("temporal_audit.schema_version")
    if audit.get("errors"):
        missing.append("temporal_audit.errors")
    if audit.get("action_semantics") != (
        "clipped_deterministic_student_mean_vs_teacher_action"
    ):
        missing.append("temporal_audit.action_semantics")
    if audit.get("sequence_identity") != [
        "motion_uid",
        "rollout_uid",
        "traj_step",
    ]:
        missing.append("temporal_audit.sequence_identity")
    heldout_paths = audit.get("heldout_motion_paths")
    if (
        audit.get("dataset_split") != "val"
        or audit.get("traj_step_field") != "rollout_step"
        or not isinstance(heldout_paths, (list, tuple))
        or not heldout_paths
        or any(not isinstance(path, str) or not path for path in heldout_paths)
        or len(set(heldout_paths)) != len(heldout_paths)
    ):
        missing.append("temporal_audit.heldout_split")

    names = audit.get("actuator_names")
    try:
        from musclemimic.distill.action_schema import actuator_schema_hash

        if not isinstance(names, (list, tuple)):
            raise ValueError("actuator_names must be a JSON array")
        names = [str(name) for name in names]
        expected_hash = actuator_schema_hash(names)
        action_dim = float(audit.get("action_dim"))
        valid_action_schema = bool(
            names
            and action_dim == len(names)
            and action_dim == math.floor(action_dim)
            and audit.get("action_schema_hash") == expected_hash
        )
    except (TypeError, ValueError):
        valid_action_schema = False
    if not valid_action_schema:
        missing.append("temporal_audit.action_schema")

    def finite(name: str) -> float | None:
        raw = audit.get(name)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            missing.append(f"temporal_audit.{name}")
            return None
        if not math.isfinite(value):
            missing.append(f"temporal_audit.{name}")
            return None
        values[name] = value
        return value

    sequence_count = finite("usable_sequence_count")
    search = finite("search_max_lag_steps")
    best_lag = finite("best_lag_steps")
    max_motion_lag = finite("max_abs_motion_best_lag_steps")
    improvement = finite("lag_mse_improvement_fraction")
    zero_lag_mse = finite("zero_lag_mse")
    best_lag_mse = finite("best_lag_mse")
    integral_sequences = bool(
        sequence_count is not None
        and sequence_count >= 0.0
        and sequence_count == math.floor(sequence_count)
    )
    integral_lags = bool(
        best_lag is not None
        and best_lag == math.floor(best_lag)
        and max_motion_lag is not None
        and max_motion_lag >= 0.0
        and max_motion_lag == math.floor(max_motion_lag)
    )
    valid_search = bool(
        search is not None
        and search == int(thresholds.temporal_search_max_lag_steps)
    )
    valid_mse = bool(
        zero_lag_mse is not None
        and zero_lag_mse >= 0.0
        and best_lag_mse is not None
        and best_lag_mse >= 0.0
        and best_lag_mse <= zero_lag_mse + 1e-12
        and improvement is not None
        and 0.0 <= improvement <= 1.0
    )
    checks = {
        "enough_sequences": bool(
            integral_sequences
            and sequence_count >= int(thresholds.min_temporal_sequences)
        ),
        "search_window": valid_search,
        "numeric_schema": bool(integral_lags and valid_mse),
        "global_best_lag": bool(
            integral_lags
            and valid_search
            and abs(best_lag) <= search
            and abs(best_lag) <= int(thresholds.max_abs_temporal_best_lag_steps)
        ),
        "per_motion_best_lag": bool(
            integral_lags
            and valid_search
            and max_motion_lag <= search
            and max_motion_lag <= int(thresholds.max_abs_temporal_best_lag_steps)
        ),
        "lag_mse_improvement": bool(
            valid_mse
            and improvement
            <= float(thresholds.max_temporal_lag_mse_improvement_fraction)
        ),
    }
    return {
        "passed": bool(not missing and all(checks.values())),
        "missing": sorted(set(missing)),
        "values": values,
        "checks": checks,
        "thresholds": {
            "min_sequences": int(thresholds.min_temporal_sequences),
            "max_abs_best_lag_steps": int(
                thresholds.max_abs_temporal_best_lag_steps
            ),
            "max_lag_mse_improvement_fraction": float(
                thresholds.max_temporal_lag_mse_improvement_fraction
            ),
        },
    }


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
    non_finite = []
    for metric in required:
        key = metric if metric in metrics else f"val_{metric}"
        if key not in metrics:
            missing.append(metric)
            continue
        if not math.isfinite(float(metrics[key])):
            non_finite.append(metric)
    if missing or non_finite:
        details = []
        if missing:
            details.append(f"missing={sorted(missing)}")
        if non_finite:
            details.append(f"non_finite={sorted(non_finite)}")
        raise RuntimeError("invalid eval metrics: " + ", ".join(details))


def run_eval_metrics(
    checkpoint: str,
    *,
    motion_paths: list[str] | None = None,
    metrics_envs: int = 20,
    metrics_steps: int = 500,
    eval_seed: int = 0,
    deterministic: bool = False,
    evaluate_all: bool = False,
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
        if evaluate_all:
            cmd.append("--evaluate_all")
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
    metrics = canonicalize_eval_metrics(metrics)
    if require_metrics:
        validate_required_metrics(metrics)
    return metrics


def load_convergence_evidence(path: str | Path) -> dict[str, Any]:
    """Load the BC convergence record from a training metadata JSON."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"direct convergence metadata is unreadable: {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError("direct convergence metadata root must be a JSON object")
    convergence = payload.get("convergence", payload)
    if not isinstance(convergence, dict):
        raise ValueError("direct convergence metadata has no convergence object")
    return convergence


def run_checkpoint_temporal_audit(
    checkpoint: str,
    *,
    dataset_dir: str | Path,
    expected_motion_paths: Sequence[str] | None = None,
    thresholds: DistillAcceptanceThresholds | None = None,
    batch_size: int = 4096,
) -> dict[str, Any]:
    """Evaluate one final direct checkpoint on fixed held-out sequence shards."""

    import jax
    import jax.numpy as jnp
    from omegaconf import OmegaConf

    from musclemimic.algorithms import PPOJax
    from musclemimic.algorithms.common.env_utils import apply_policy_interface_wrappers
    from musclemimic.core.wrappers.finger_isolation import model_action_names
    from musclemimic.distill.dataset import DistillDataset
    from musclemimic.distill.losses import distribution_mean
    from musclemimic.distill.motion_identity import MotionIdentityMap
    from musclemimic.runner.engine import instantiate_env
    from musclemimic.runner.eval_utils import load_checkpoint

    limits = _validated_thresholds(thresholds)
    if int(batch_size) <= 0:
        raise ValueError("temporal audit batch_size must be positive")
    config, agent_state, _metadata = load_checkpoint(checkpoint)
    OmegaConf.set_struct(config, False)
    env = instantiate_env(config)
    try:
        policy_env = apply_policy_interface_wrappers(
            env, config.experiment, include_student=False
        )
        policy_names = getattr(policy_env, "policy_actuator_names", None)
        if policy_names is None:
            policy_names = getattr(policy_env, "policy_action_names", None)
        if policy_names is None:
            policy_names = model_action_names(policy_env)
        checkpoint_names = [str(name) for name in policy_names]
        agent_conf = PPOJax.init_agent_conf(env, config)
        dataset = DistillDataset(
            dataset_dir,
            split="val",
            strict_schema=True,
            required_optional_fields=("motion_uid", "rollout_uid", "rollout_step"),
            target_actuator_names=checkpoint_names,
        )
        if list(dataset.source_actuator_names) != checkpoint_names:
            raise ValueError(
                "held-out teacher action actuator names/order differ from final direct checkpoint"
            )
        expected_paths: list[str] = []
        if expected_motion_paths is not None:
            expected_identity = MotionIdentityMap.from_paths(expected_motion_paths)
            expected_paths = list(expected_identity.motion_paths)
            val_metadata = (dataset.metadata.get("split_metadata") or {}).get("val")
            identity_metadata = (
                val_metadata.get("motion_identity")
                if isinstance(val_metadata, Mapping)
                else None
            )
            actual_paths = (
                identity_metadata.get("motion_paths")
                if isinstance(identity_metadata, Mapping)
                else None
            )
            if actual_paths != expected_paths:
                raise ValueError(
                    "temporal-audit val shard motions differ from closed-loop held-out motions: "
                    f"dataset={actual_paths} expected={expected_paths}"
                )
            observed_uids = {
                int(value)
                for value in np.unique(dataset.arrays["motion_uid"]).tolist()
            }
            expected_uids = {
                int(value) for value in expected_identity.motion_uids.tolist()
            }
            if observed_uids != expected_uids:
                raise ValueError(
                    "temporal-audit val rows do not cover the exact closed-loop held-out "
                    f"motion set: observed_uids={sorted(observed_uids)} "
                    f"expected_uids={sorted(expected_uids)}"
                )
        expected_obs = int(
            apply_policy_interface_wrappers(env, config.experiment).info.observation_space.shape[0]
        )
        if int(dataset.student_obs_dim) != expected_obs:
            raise ValueError(
                "held-out student_obs dimension differs from final direct checkpoint: "
                f"dataset={dataset.student_obs_dim} checkpoint={expected_obs}"
            )

        train_state = agent_state.train_state
        if int(config.get("n_seeds", 1)) > 1:
            # Match fullbody.eval's default train_state_seed=0 so the temporal
            # and closed-loop gates audit the same member of a multi-seed file.
            train_state = jax.tree.map(lambda value: value[0], train_state)

        @jax.jit
        def predict(observation):
            pi, _value = agent_conf.network.apply(
                {"params": train_state.params, "run_stats": train_state.run_stats},
                observation,
            )
            return jnp.clip(distribution_mean(pi), -1.0, 1.0)

        predictions: list[np.ndarray] = []
        observations = np.asarray(dataset.arrays["student_obs"], dtype=np.float32)
        for start in range(0, int(dataset.num_samples), int(batch_size)):
            action = predict(jnp.asarray(observations[start : start + int(batch_size)]))
            predictions.append(np.asarray(jax.device_get(action), dtype=np.float32))
        student_actions = np.concatenate(predictions, axis=0)
        if student_actions.shape != dataset.arrays["teacher_action"].shape:
            raise ValueError(
                "final direct checkpoint action shape differs from held-out teacher target: "
                f"student={student_actions.shape} teacher={dataset.arrays['teacher_action'].shape}"
            )
        report = evaluate_temporal_drift(
            student_action=student_actions,
            teacher_action=dataset.arrays["teacher_action"],
            motion_uid=dataset.arrays["motion_uid"],
            rollout_uid=dataset.arrays["rollout_uid"],
            traj_step=dataset.arrays["rollout_step"],
            actuator_names=dataset.actuator_names,
            checkpoint_actuator_names=checkpoint_names,
            thresholds=limits,
        )
        report.update(
            {
                "checkpoint": str(checkpoint),
                "dataset_dir": str(Path(dataset_dir)),
                "dataset_split": "val",
                "heldout_motion_paths": expected_paths,
                "traj_step_field": "rollout_step",
            }
        )
        return report
    finally:
        stop = getattr(env, "stop", None)
        if callable(stop):
            stop()


def write_temporal_audit_outputs(
    audits: Mapping[str, Mapping[str, Any]], output_dir: str | Path
) -> Path:
    target = Path(output_dir) / "temporal_audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dict(audits), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return target


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


def write_acceptance_outputs(
    results: dict[str, dict[str, float]],
    output_dir: str | Path,
    thresholds: DistillAcceptanceThresholds | None = None,
    *,
    convergence: Mapping[str, Any] | None = None,
    temporal_audits: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    teacher = results.get("teacher")
    if teacher is None:
        raise ValueError("distill acceptance requires a teacher result")
    acceptance = {
        policy: evaluate_distill_acceptance(
            teacher,
            metrics,
            thresholds,
            convergence=convergence,
            temporal_audit=(temporal_audits or {}).get(policy),
        )
        for policy, metrics in results.items()
        if policy != "teacher"
    }
    target = output_path / "acceptance.json"
    target.write_text(json.dumps(acceptance, indent=2, sort_keys=True), encoding="utf-8")
    return target


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
            "Production promotion thresholds:",
            "",
            "- Student return ratio: >= 0.90 (target 0.95).",
            "- Student rollout completion ratio: >= 0.90 of teacher.",
            "- Early termination rate may exceed teacher by at most 0.05.",
            "- Relative-site position and both racket tracking errors may degrade by at most 10%.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path

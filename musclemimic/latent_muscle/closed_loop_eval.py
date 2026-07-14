"""Held-out prior-mean and LAB perturbation evaluation."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from musclemimic.distill.motion_identity import normalize_motion_path, stable_motion_uid
from musclemimic.distill.obs_filter import filter_student_obs
from musclemimic.distill.provenance import (
    canonical_json_sha256,
    checkpoint_fingerprint_matches,
    file_sha256,
    validate_direct_acceptance_record,
)

REQUIRED_TRACKING_METRICS = ("err_rpos", "err_racket_pos", "err_racket_rot")
OFFLINE_PROMOTION_METRICS = (
    "prior_posterior_mse_ratio",
    "active_latent_fraction",
    "prior_sigma_min_clamp_fraction",
    "prior_sigma_max_clamp_fraction",
    "decoder_saturation_fraction",
    "posterior_action_mse",
)


@dataclass(frozen=True)
class ClosedLoopEvalConfig:
    lambdas: tuple[float, ...] = (0.0, 0.25, 0.5)
    seed: int = 0
    # The canonical retarget/control cache runs at 100 Hz.  A 120-step window
    # therefore evaluates the approximately 1.2 s swing required by the plan.
    max_steps: int | None = 120
    motion_paths: tuple[str, ...] | None = None
    # Opt-in research evidence.  Defaults preserve the production-v2 ABI used
    # by currently running direct-latent experiments.
    phase_field: str | None = None
    require_all_phases: bool = False
    collect_decoder_usage: bool = False
    collect_jacobian_alignment: bool = False


def evaluate_latent_closed_loop(
    *,
    env: Any,
    runtime: Any,
    student_obs_spec: Any,
    config: ClosedLoopEvalConfig = ClosedLoopEvalConfig(),
    direct_bc_metrics: dict[str, Any] | None = None,
    alignment_synergy_basis: Any | None = None,
) -> dict[str, Any]:
    """Evaluate every loaded motion from frame zero with frozen prior/decoder."""
    lambdas = tuple(float(value) for value in config.lambdas)
    if not lambdas or any(value < 0.0 for value in lambdas):
        raise ValueError("closed-loop lambdas must be non-empty and non-negative")
    if 0.0 not in lambdas:
        raise ValueError("closed-loop evaluation must include lambda=0 prior mean")
    if config.max_steps is not None and int(config.max_steps) <= 0:
        raise ValueError("closed-loop max_steps must be positive when specified")
    if config.require_all_phases and config.phase_field is None:
        raise ValueError("require_all_phases requires a configured phase_field")
    if config.collect_jacobian_alignment and alignment_synergy_basis is None:
        alignment_synergy_basis = getattr(runtime, "synergy_basis", None)
    if config.collect_jacobian_alignment and alignment_synergy_basis is None:
        raise ValueError(
            "Jacobian alignment requires an explicit excitation synergy basis "
            "(direct checkpoints do not carry one)"
        )
    trajectory_handler = getattr(env, "th", None)
    if trajectory_handler is None:
        raise ValueError("closed-loop environment has no trajectory handler")
    n_trajectories = int(trajectory_handler.n_trajectories)
    if n_trajectories <= 0:
        raise ValueError("closed-loop environment has no trajectories")
    motion_paths = (
        tuple(f"trajectory_{index}" for index in range(n_trajectories))
        if config.motion_paths is None
        else tuple(normalize_motion_path(path) for path in config.motion_paths)
    )
    if len(motion_paths) != n_trajectories or len(set(motion_paths)) != n_trajectories:
        raise ValueError(
            "closed-loop motion_paths must uniquely identify every loaded trajectory"
        )

    results: dict[str, dict[str, float]] = {}
    for lambda_lab in lambdas:
        results[_lambda_key(lambda_lab)] = _evaluate_lambda(
            env=env,
            runtime=runtime,
            spec=student_obs_spec,
            lambda_lab=lambda_lab,
            seed=int(config.seed),
            max_steps=config.max_steps,
            n_trajectories=n_trajectories,
            motion_paths=motion_paths,
            phase_field=config.phase_field,
            require_all_phases=bool(config.require_all_phases),
            collect_decoder_usage=bool(config.collect_decoder_usage),
            collect_jacobian_alignment=bool(config.collect_jacobian_alignment),
            alignment_synergy_basis=alignment_synergy_basis,
        )

    prior = results[_lambda_key(0.0)]
    sweep_no_fall = [
        metrics["no_fall_rate"]
        for key, metrics in results.items()
        if key != _lambda_key(0.0)
    ]
    report: dict[str, Any] = {
        "schema_version": "latent_closed_loop_eval_v2",
        "checkpoint_fingerprint": runtime.checkpoint_fingerprint,
        "num_trajectories": n_trajectories,
        "heldout_motion_paths": list(motion_paths),
        "heldout_motion_uids": [int(stable_motion_uid(path)) for path in motion_paths],
        "heldout_motion_set_fingerprint": canonical_json_sha256(list(motion_paths)),
        "lambdas": list(lambdas),
        "max_steps": config.max_steps,
        "by_lambda": results,
        "fall_or_early_termination_rate": prior["fall_or_early_termination_rate"],
        "prior_mean_no_fall_rate": prior["no_fall_rate"],
        "lambda_025_050_no_fall_rate": min(sweep_no_fall) if sweep_no_fall else prior["no_fall_rate"],
        "prior_mean_frame_coverage": prior["frame_coverage"],
        "prior_mean_episode_return": prior["mean_episode_return"],
        "decoder_type": str(getattr(runtime, "decoder_type", "direct")),
        "config_synergy_basis_expected_fingerprint": (
            getattr(runtime, "config", {}) or {}
        ).get("synergy_basis_expected_fingerprint"),
    }
    runtime_basis = getattr(runtime, "synergy_basis", None)
    if runtime_basis is not None:
        report["runtime_synergy_basis_fingerprint"] = str(
            runtime_basis.fingerprint
        )
        report["runtime_synergy_basis_source_fingerprint"] = str(
            runtime_basis.manifest.get("source_fingerprint", "")
        )
    if config.phase_field is not None:
        report["phase_field"] = str(config.phase_field)
        report["require_all_phases"] = bool(config.require_all_phases)
    if config.collect_decoder_usage:
        report["decoder_usage_collected"] = True
    if config.collect_jacobian_alignment:
        report["jacobian_alignment_collected"] = True
        report["analysis_synergy_basis_fingerprint"] = str(
            getattr(alignment_synergy_basis, "fingerprint", "")
            or (alignment_synergy_basis.get("fingerprint") if isinstance(alignment_synergy_basis, dict) else "")
        )
        if len(report["analysis_synergy_basis_fingerprint"]) != 64:
            raise ValueError("Jacobian alignment basis lacks a valid fingerprint binding")
    degradation = body_racket_relative_degradation(prior, direct_bc_metrics or {})
    if degradation is not None:
        report["body_racket_relative_degradation"] = degradation
    return report


def body_racket_relative_degradation(
    latent_metrics: dict[str, Any],
    direct_metrics: dict[str, Any],
) -> float | None:
    """Worst relative increase across the mandatory body/racket error metrics.

    A partial comparison is unsafe: Stage 3 may not promote on body tracking
    while silently omitting racket position or orientation.  Returning ``None``
    makes the downstream YAML gate fail closed when any required metric is
    absent, non-finite, or negative.
    """
    baseline = _flatten_numeric(direct_metrics)
    candidates: list[float] = []
    for key in REQUIRED_TRACKING_METRICS:
        latent_value = latent_metrics.get(key, latent_metrics.get(f"val_{key}"))
        baseline_value = baseline.get(key, baseline.get(f"val_{key}"))
        if latent_value is None or baseline_value is None:
            return None
        base = float(baseline_value)
        current = float(latent_value)
        if not np.isfinite(base) or not np.isfinite(current) or base < 0.0 or current < 0.0:
            return None
        candidates.append(max(0.0, (current - base) / max(abs(base), 1e-8)))
    return float(max(candidates))


def select_direct_rollout_policy(
    payload: dict[str, Any],
    promotion_policy: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Select the promoted direct student's held-out rollout metrics."""
    policy_keys = ("student_bc_ppo", "student_bc_dagger", "student_bc")
    recorded_policy = payload.get("promotion_policy")
    if promotion_policy is None and isinstance(recorded_policy, str):
        promotion_policy = recorded_policy
    if promotion_policy is not None:
        if promotion_policy not in policy_keys:
            raise ValueError(f"unsupported direct promotion policy: {promotion_policy}")
        selected = promotion_policy
    else:
        selected = next((key for key in policy_keys if isinstance(payload.get(key), dict)), "")
    if selected:
        metrics = payload.get(selected)
        if not isinstance(metrics, dict):
            raise ValueError(f"direct rollout metrics do not contain policy {selected!r}")
        validate_direct_rollout_tracking_metrics(metrics)
        return selected, metrics
    # A single-policy held-out metrics JSON is also accepted explicitly.
    if any(str(key).startswith(("val_err_", "val_racket_", "err_", "racket_")) for key in payload):
        validate_direct_rollout_tracking_metrics(payload)
        return "single_policy", payload
    raise ValueError(
        "direct rollout metrics must contain student_bc_ppo, student_bc_dagger, "
        "student_bc, or a single policy's held-out tracking metrics"
    )


def validate_direct_rollout_tracking_metrics(payload: dict[str, Any]) -> dict[str, float]:
    """Return canonical mandatory direct-student metrics or fail closed."""
    flattened = _flatten_numeric(payload)
    result: dict[str, float] = {}
    missing: list[str] = []
    invalid: list[str] = []
    for key in REQUIRED_TRACKING_METRICS:
        value = flattened.get(key, flattened.get(f"val_{key}"))
        if value is None:
            missing.append(key)
            continue
        number = float(value)
        if not np.isfinite(number) or number < 0.0:
            invalid.append(key)
            continue
        result[key] = number
    if missing or invalid:
        raise ValueError(
            "direct rollout tracking metrics are incomplete or invalid: "
            f"missing={missing}, invalid={invalid}; required={list(REQUIRED_TRACKING_METRICS)}"
        )
    return result


def _evaluate_lambda(
    *,
    env: Any,
    runtime: Any,
    spec: Any,
    lambda_lab: float,
    seed: int,
    max_steps: int | None,
    n_trajectories: int,
    motion_paths: tuple[str, ...],
    phase_field: str | None,
    require_all_phases: bool,
    collect_decoder_usage: bool,
    collect_jacobian_alignment: bool,
    alignment_synergy_basis: Any | None,
) -> dict[str, Any]:
    total_return = 0.0
    total_length = 0
    total_frames = 0
    covered_frames = 0
    early = 0
    step_sums: dict[str, float] = {}
    step_counts: dict[str, int] = {}
    per_motion: list[dict[str, Any]] = []
    decoder_usage = _empty_decoder_usage()
    jacobian_metrics: list[dict[str, float]] = []
    phase_evidence: dict[int, dict[str, Any]] = {}

    for traj_index in range(n_trajectories):
        _set_fixed_start(env, traj_index)
        obs = _reset_obs(env)
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(traj_index), _lambda_seed(lambda_lab)]))
        episode_return = 0.0
        episode_length = 0
        last_info: dict[str, Any] = {}
        terminal = False
        absorbing_terminal = False
        motion_step_sums: dict[str, float] = {}
        motion_step_counts: dict[str, int] = {}
        while True:
            student_obs = np.asarray(filter_student_obs(obs, spec), dtype=np.float32)
            state = student_obs[None, :] if student_obs.ndim == 1 else student_obs
            prior_mu, prior_raw_sigma = runtime.prior_raw_numpy(state)
            if float(lambda_lab) == 0.0:
                latent = prior_mu
            else:
                sigma = np.clip(
                    _softplus(prior_raw_sigma),
                    runtime.sigma_min,
                    runtime.sigma_max,
                )
                raw_latent = rng.standard_normal(prior_mu.shape).astype(np.float32)
                latent = prior_mu + float(lambda_lab) * sigma * np.tanh(raw_latent)
            components = None
            if collect_decoder_usage:
                if not hasattr(runtime, "decode_components_numpy"):
                    raise ValueError("runtime lacks decoder-usage analysis interface")
                components = runtime.decode_components_numpy(state, latent)
                action = np.asarray(components.action)
            else:
                action = runtime.decoder_numpy(state, latent)
            jacobian_alignment = None
            if collect_jacobian_alignment:
                if not hasattr(runtime, "decoder_jacobian_numpy"):
                    raise ValueError("runtime lacks decoder-Jacobian analysis interface")
                jacobian = np.asarray(
                    runtime.decoder_jacobian_numpy(
                        state,
                        latent,
                        output="physical_excitation",
                    )
                )
                if jacobian.ndim == 3 and jacobian.shape[0] == 1:
                    jacobian = jacobian[0]
                from analysis.latent_synergy.jacobian_alignment import subspace_alignment

                aligned = subspace_alignment(
                    jacobian,
                    _alignment_basis_matrix(alignment_synergy_basis),
                )
                jacobian_alignment = {
                    "projection_score": float(aligned["projection_score"]),
                    "grassmann_distance": float(aligned["grassmann_distance"]),
                    "mean_canonical_correlation": float(
                        aligned["mean_canonical_correlation"]
                    ),
                }
            obs, reward, absorbing, done, info = env.step(action)
            last_info = dict(info or {})
            episode_return += float(np.asarray(reward).reshape(-1)[0])
            episode_length += 1
            _accumulate_numeric_step_metrics(last_info, step_sums, step_counts)
            _accumulate_numeric_step_metrics(
                last_info,
                motion_step_sums,
                motion_step_counts,
            )
            phase_id = None
            if phase_field is not None:
                phase_id = _phase_id_from_info(last_info, phase_field)
                phase_record = phase_evidence.setdefault(
                    phase_id,
                    {
                        "num_steps": 0,
                        "step_sums": {},
                        "step_counts": {},
                        "decoder_usage": _empty_decoder_usage(),
                        "jacobian": [],
                    },
                )
                phase_record["num_steps"] += 1
                _accumulate_numeric_step_metrics(
                    last_info,
                    phase_record["step_sums"],
                    phase_record["step_counts"],
                )
            if components is not None:
                observation = _decoder_usage_observation(
                    components,
                    decoder_type=str(getattr(runtime, "decoder_type", "direct")),
                )
                _accumulate_decoder_usage(decoder_usage, observation)
                if phase_id is not None:
                    _accumulate_decoder_usage(
                        phase_evidence[phase_id]["decoder_usage"], observation
                    )
            if jacobian_alignment is not None:
                jacobian_metrics.append(jacobian_alignment)
                if phase_id is not None:
                    phase_evidence[phase_id]["jacobian"].append(jacobian_alignment)
            terminal = bool(np.asarray(done).reshape(-1)[0])
            absorbing_terminal = bool(np.asarray(absorbing).reshape(-1)[0])
            if terminal:
                break
            if max_steps is not None and episode_length >= int(max_steps):
                break

        traj_len = int(np.asarray(last_info.get("traj_len", episode_length)).reshape(-1)[0])
        subtraj_step = int(
            np.asarray(last_info.get("subtraj_step_no", episode_length - 1)).reshape(-1)[0]
        )
        # Reaching the explicitly requested evaluation horizon is success, not
        # an early termination.  Only an environment terminal/absorbing state
        # before the reference end counts as fall/ET.
        terminated_early = (terminal or absorbing_terminal) and subtraj_step < traj_len - 1
        early += int(terminated_early)
        expected_frames = max(traj_len, 1)
        if max_steps is not None:
            expected_frames = min(expected_frames, max(int(max_steps), 1))
        total_frames += expected_frames
        covered_frames += min(episode_length, expected_frames)
        total_return += episode_return
        total_length += episode_length
        motion_metrics = {
            "traj_index": int(traj_index),
            "motion_path": motion_paths[traj_index],
            "motion_uid": int(stable_motion_uid(motion_paths[traj_index])),
            "episode_return": float(episode_return),
            "episode_length": int(episode_length),
            "terminated_early": bool(terminated_early),
            "no_fall": bool(not terminated_early),
            "frame_coverage": float(min(episode_length, expected_frames) / expected_frames),
        }
        for key, total in motion_step_sums.items():
            motion_metrics[key] = float(total / max(motion_step_counts[key], 1))
        per_motion.append(motion_metrics)

    metrics = {
        "mean_episode_return": total_return / n_trajectories,
        "mean_episode_length": total_length / n_trajectories,
        "fall_or_early_termination_rate": early / n_trajectories,
        "no_fall_rate": 1.0 - early / n_trajectories,
        "frame_coverage": covered_frames / max(total_frames, 1),
    }
    for key, total in step_sums.items():
        metrics[key] = total / max(step_counts[key], 1)
    if collect_decoder_usage:
        metrics.update(_finalize_decoder_usage(decoder_usage))
    if collect_jacobian_alignment:
        if not jacobian_metrics:
            raise ValueError("Jacobian alignment collected no samples")
        metrics.update(_finalize_jacobian_metrics(jacobian_metrics))
    result = {
        **{key: float(value) for key, value in metrics.items()},
        "per_motion": per_motion,
    }
    if phase_field is not None:
        missing_phases = [
            name
            for phase_id, name in enumerate(_PHASE_NAMES)
            if phase_id not in phase_evidence
        ]
        if require_all_phases and missing_phases:
            raise ValueError(
                f"closed-loop phase evidence is missing phases: {missing_phases}"
            )
        by_phase: dict[str, Any] = {}
        for phase_id, name in enumerate(_PHASE_NAMES):
            if phase_id not in phase_evidence:
                continue
            evidence = phase_evidence[phase_id]
            phase_metrics: dict[str, Any] = {"num_steps": int(evidence["num_steps"])}
            for key, total in evidence["step_sums"].items():
                phase_metrics[key] = float(total / max(evidence["step_counts"][key], 1))
            if collect_decoder_usage:
                phase_metrics.update(
                    _finalize_decoder_usage(evidence["decoder_usage"])
                )
            if collect_jacobian_alignment:
                if not evidence["jacobian"]:
                    raise ValueError(f"phase {name!r} has no Jacobian evidence")
                phase_metrics.update(
                    _finalize_jacobian_metrics(evidence["jacobian"])
                )
            by_phase[name] = phase_metrics
        result["by_phase"] = by_phase
        result["missing_phases"] = missing_phases
    return result


_PHASE_NAMES = (
    "ready",
    "backswing",
    "acceleration",
    "impact",
    "followthrough",
    "recovery",
)


def _phase_id_from_info(info: dict[str, Any], field: str) -> int:
    if field not in info:
        raise ValueError(f"closed-loop step info is missing required phase field {field!r}")
    value = np.asarray(info[field])
    if value.size != 1 or not np.issubdtype(value.dtype, np.number):
        raise ValueError(f"closed-loop phase field {field!r} must be a numeric scalar")
    number = float(value.reshape(-1)[0])
    if not np.isfinite(number) or number != np.floor(number):
        raise ValueError(f"closed-loop phase field {field!r} must be a finite integer")
    phase_id = int(number)
    if phase_id < 0 or phase_id >= len(_PHASE_NAMES):
        raise ValueError(f"closed-loop phase field {field!r} has unknown ID {phase_id}")
    return phase_id


def _empty_decoder_usage() -> dict[str, float | int]:
    return {
        "num_steps": 0,
        "physical_energy": 0.0,
        "residual_energy": 0.0,
        "baseline_energy": 0.0,
        "coefficient_abs_sum": 0.0,
        "coefficient_square_sum": 0.0,
        "coefficient_count": 0,
    }


def _decoder_usage_observation(components: Any, *, decoder_type: str) -> dict[str, float | int]:
    required = (
        "physical_excitation",
        "synergy_coefficients",
        "baseline_excitation",
        "residual_excitation",
    )
    missing = [name for name in required if not hasattr(components, name)]
    if missing:
        raise ValueError(f"decoder components are missing usage fields: {missing}")
    physical = np.asarray(components.physical_excitation, dtype=np.float64)
    coefficients = np.asarray(components.synergy_coefficients, dtype=np.float64)
    baseline = np.asarray(components.baseline_excitation, dtype=np.float64)
    residual = np.asarray(components.residual_excitation, dtype=np.float64)
    arrays = (physical, coefficients, baseline, residual)
    if any(not np.all(np.isfinite(array)) for array in arrays) or physical.size == 0:
        raise ValueError("decoder usage components must be finite and include physical excitation")
    if decoder_type != "direct" and coefficients.size == 0:
        raise ValueError("synergy decoder emitted no coefficient evidence")
    if decoder_type == "synergy_residual" and residual.size == 0:
        raise ValueError("synergy_residual decoder emitted no residual evidence")
    return {
        "num_steps": 1,
        "physical_energy": float(np.sum(np.square(physical))),
        "residual_energy": float(np.sum(np.square(residual))),
        "baseline_energy": float(np.sum(np.square(baseline))),
        "coefficient_abs_sum": float(np.sum(np.abs(coefficients))),
        "coefficient_square_sum": float(np.sum(np.square(coefficients))),
        "coefficient_count": int(coefficients.size),
    }


def _accumulate_decoder_usage(
    accumulator: dict[str, float | int],
    observation: dict[str, float | int],
) -> None:
    for key in accumulator:
        accumulator[key] = accumulator[key] + observation[key]


def _finalize_decoder_usage(accumulator: dict[str, float | int]) -> dict[str, float]:
    if int(accumulator["num_steps"]) <= 0:
        raise ValueError("decoder usage evidence is empty")
    physical_energy = float(accumulator["physical_energy"])
    coefficient_count = int(accumulator["coefficient_count"])
    result = {
        "decoder_usage_num_steps": float(accumulator["num_steps"]),
        "residual_energy_ratio": float(accumulator["residual_energy"])
        / max(physical_energy, 1e-12),
        "baseline_energy_ratio": float(accumulator["baseline_energy"])
        / max(physical_energy, 1e-12),
    }
    if coefficient_count > 0:
        result.update(
            {
                "synergy_coefficient_abs_mean": float(
                    accumulator["coefficient_abs_sum"]
                )
                / coefficient_count,
                "synergy_coefficient_rms": float(
                    np.sqrt(
                        float(accumulator["coefficient_square_sum"])
                        / coefficient_count
                    )
                ),
            }
        )
    return result


def _finalize_jacobian_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        raise ValueError("Jacobian evidence is empty")
    return {
        "jacobian_alignment_num_steps": float(len(items)),
        "jacobian_projection_score": float(
            np.mean([item["projection_score"] for item in items])
        ),
        "jacobian_grassmann_distance": float(
            np.mean([item["grassmann_distance"] for item in items])
        ),
        "jacobian_mean_canonical_correlation": float(
            np.mean([item["mean_canonical_correlation"] for item in items])
        ),
    }


def _alignment_basis_matrix(value: Any) -> np.ndarray:
    if value is None:
        raise ValueError("Jacobian alignment basis is missing")
    if hasattr(value, "basis"):
        matrix = np.asarray(value.basis, dtype=np.float64)
    elif isinstance(value, dict) and "basis" in value:
        matrix = np.asarray(value["basis"], dtype=np.float64)
    else:
        raise ValueError("Jacobian alignment basis has no matrix")
    if matrix.ndim != 2 or min(matrix.shape) <= 0 or not np.all(np.isfinite(matrix)):
        raise ValueError("Jacobian alignment basis must be a finite non-empty matrix")
    return matrix


def validate_closed_loop_promotion_report(
    report: dict[str, Any],
    *,
    checkpoint_dir: str | Path,
    require_seal: bool = True,
    verify_external_files: bool = True,
) -> dict[str, Any]:
    """Validate production promotion evidence; legacy scalar JSON is rejected."""
    from musclemimic.latent_muscle.checkpoint import latent_checkpoint_fingerprint

    path = Path(checkpoint_dir)
    if report.get("schema_version") != "latent_closed_loop_eval_v2":
        raise ValueError("production latent promotion requires latent_closed_loop_eval_v2")
    current_fingerprint = latent_checkpoint_fingerprint(path)
    if report.get("checkpoint_fingerprint") != current_fingerprint:
        raise ValueError("closed-loop report latent checkpoint fingerprint mismatch")
    if list(report.get("lambdas") or []) != [0.0, 0.25, 0.5]:
        raise ValueError("production closed-loop lambdas must be exactly [0.0, 0.25, 0.5]")
    if int(report.get("max_steps", -1)) != 120:
        raise ValueError("production closed-loop max_steps must be exactly 120")
    paths = report.get("heldout_motion_paths")
    uids = report.get("heldout_motion_uids")
    if not isinstance(paths, list) or len(paths) != 5 or len(set(paths)) != 5:
        raise ValueError("production closed-loop report requires exactly five unique held-out motions")
    normalized = [normalize_motion_path(item) for item in paths]
    expected_uids = [int(stable_motion_uid(item)) for item in normalized]
    if uids != expected_uids or int(report.get("num_trajectories", -1)) != 5:
        raise ValueError("closed-loop held-out motion identity mismatch")
    if report.get("heldout_motion_set_fingerprint") != canonical_json_sha256(normalized):
        raise ValueError("closed-loop held-out motion set fingerprint mismatch")
    offline = report.get("offline_eval_metrics")
    if not isinstance(offline, dict) or set(offline) != set(OFFLINE_PROMOTION_METRICS):
        raise ValueError("closed-loop report has incomplete offline latent gate metrics")
    if any(not np.isfinite(float(offline[key])) for key in OFFLINE_PROMOTION_METRICS):
        raise ValueError("closed-loop report offline latent gate metrics are non-finite")

    by_lambda = report.get("by_lambda")
    expected_keys = {"lambda_0p000", "lambda_0p250", "lambda_0p500"}
    if not isinstance(by_lambda, dict) or set(by_lambda) != expected_keys:
        raise ValueError("closed-loop report has an incomplete lambda sweep")
    required_aggregate = {
        "mean_episode_return",
        "mean_episode_length",
        "fall_or_early_termination_rate",
        "no_fall_rate",
        "frame_coverage",
        *REQUIRED_TRACKING_METRICS,
    }
    for lambda_key in sorted(expected_keys):
        metrics = by_lambda[lambda_key]
        if not isinstance(metrics, dict):
            raise ValueError(f"closed-loop {lambda_key} metrics must be an object")
        missing = sorted(required_aggregate - set(metrics))
        if missing:
            raise ValueError(f"closed-loop {lambda_key} is missing metrics: {missing}")
        for key in required_aggregate:
            value = float(metrics[key])
            if not np.isfinite(value):
                raise ValueError(f"closed-loop {lambda_key}.{key} is non-finite")
        per_motion = metrics.get("per_motion")
        if not isinstance(per_motion, list) or len(per_motion) != 5:
            raise ValueError(f"closed-loop {lambda_key} requires five per-motion records")
        for index, item in enumerate(per_motion):
            if not isinstance(item, dict):
                raise ValueError(f"closed-loop {lambda_key} per-motion record is invalid")
            if (
                item.get("traj_index") != index
                or item.get("motion_path") != normalized[index]
                or item.get("motion_uid") != expected_uids[index]
            ):
                raise ValueError(f"closed-loop {lambda_key} per-motion identity mismatch")
            for key in ("episode_return", "episode_length", "frame_coverage", *REQUIRED_TRACKING_METRICS):
                if key not in item or not np.isfinite(float(item[key])):
                    raise ValueError(f"closed-loop {lambda_key} per-motion {key} is invalid")
        if report.get("decoder_usage_collected") is True:
            for key in (
                "decoder_usage_num_steps",
                "residual_energy_ratio",
                "baseline_energy_ratio",
            ):
                if key not in metrics or not np.isfinite(float(metrics[key])):
                    raise ValueError(f"closed-loop {lambda_key} decoder usage {key} is invalid")
        if report.get("jacobian_alignment_collected") is True:
            for key in (
                "jacobian_alignment_num_steps",
                "jacobian_projection_score",
                "jacobian_grassmann_distance",
                "jacobian_mean_canonical_correlation",
            ):
                if key not in metrics or not np.isfinite(float(metrics[key])):
                    raise ValueError(f"closed-loop {lambda_key} Jacobian metric {key} is invalid")
        if report.get("phase_field") is not None:
            by_phase = metrics.get("by_phase")
            if not isinstance(by_phase, dict) or not by_phase:
                raise ValueError(f"closed-loop {lambda_key} phase evidence is empty")
            missing_phases = metrics.get("missing_phases")
            if not isinstance(missing_phases, list):
                raise ValueError(f"closed-loop {lambda_key} missing_phases contract is invalid")
            if report.get("require_all_phases") is True and missing_phases:
                raise ValueError(f"closed-loop {lambda_key} is missing required phase evidence")
            if report.get("decoder_usage_collected") is True:
                for phase_name, phase_metrics in by_phase.items():
                    if not isinstance(phase_metrics, dict):
                        raise ValueError(f"closed-loop {lambda_key} phase {phase_name!r} is invalid")
                    for usage_key in (
                        "decoder_usage_num_steps",
                        "residual_energy_ratio",
                        "baseline_energy_ratio",
                    ):
                        if usage_key not in phase_metrics or not np.isfinite(float(phase_metrics[usage_key])):
                            raise ValueError(
                                f"closed-loop {lambda_key} phase {phase_name!r} decoder usage "
                                f"{usage_key!r} is invalid"
                            )

    if report.get("jacobian_alignment_collected") is True:
        if len(str(report.get("analysis_synergy_basis_fingerprint", ""))) != 64:
            raise ValueError("closed-loop Jacobian evidence lacks a synergy basis fingerprint")

    config_payload = json.loads(
        (path / "latent_config.yaml").read_text(encoding="utf-8")
    )
    decoder_type = str(config_payload.get("decoder_type", "direct"))
    report_decoder_type = report.get("decoder_type")
    if report_decoder_type is not None and report_decoder_type != decoder_type:
        raise ValueError("closed-loop report decoder type differs from latent config")
    expected_basis = config_payload.get("synergy_basis_expected_fingerprint")
    if (
        decoder_type != "direct"
        or "config_synergy_basis_expected_fingerprint" in report
    ) and report.get("config_synergy_basis_expected_fingerprint") != expected_basis:
        raise ValueError(
            "closed-loop report formal synergy basis expectation differs from latent config"
        )
    if decoder_type != "direct":
        basis_contract_path = path / "synergy_basis.json"
        if not basis_contract_path.is_file():
            raise ValueError("synergy checkpoint lacks embedded basis contract")
        basis_contract = json.loads(
            basis_contract_path.read_text(encoding="utf-8")
        )
        runtime_basis = config_payload.get("synergy_basis_fingerprint")
        source_basis = (basis_contract.get("manifest") or {}).get(
            "source_fingerprint"
        )
        if not (
            report.get("runtime_synergy_basis_fingerprint")
            == runtime_basis
            == basis_contract.get("fingerprint")
            and report.get("runtime_synergy_basis_source_fingerprint")
            == expected_basis
            == source_basis
        ):
            raise ValueError(
                "closed-loop runtime/checkpoint/config synergy basis fingerprints differ"
            )
    elif report.get("runtime_synergy_basis_fingerprint") is not None:
        raise ValueError("direct closed-loop report must not claim an embedded synergy basis")

    training_path = path / "training_provenance.json"
    if not training_path.is_file():
        raise ValueError("production latent checkpoint is missing training_provenance.json")
    training = json.loads(training_path.read_text(encoding="utf-8"))
    dataset_manifest = training.get("dataset_manifest")
    teacher = training.get("teacher_checkpoint")
    teacher_promotion = training.get("teacher_promotion")
    if (
        not isinstance(dataset_manifest, dict)
        or not isinstance(teacher, dict)
        or not isinstance(teacher_promotion, dict)
    ):
        raise ValueError("latent training provenance lacks dataset/teacher evidence")
    if dataset_manifest.get("teacher_checkpoint", {}).get("sha256") != teacher.get("sha256"):
        raise ValueError("latent dataset and training teacher fingerprints differ")
    if report.get("dataset_manifest_fingerprint") != dataset_manifest.get("manifest_fingerprint"):
        raise ValueError("closed-loop report dataset manifest fingerprint mismatch")
    validation_manifest = training.get("validation_dataset_manifest")
    validation_fingerprint = training.get(
        "validation_dataset_manifest_fingerprint"
    )
    if validation_fingerprint is not None:
        if (
            not isinstance(validation_manifest, dict)
            or validation_manifest.get("manifest_fingerprint")
            != validation_fingerprint
            or report.get("validation_dataset_manifest_fingerprint")
            != validation_fingerprint
        ):
            raise ValueError(
                "closed-loop report validation dataset fingerprint mismatch"
            )
        split = json.loads(
            (path / "motion_split.json").read_text(encoding="utf-8")
        )
        if report.get("motion_split_fingerprint") != split.get(
            "split_fingerprint"
        ):
            raise ValueError("closed-loop report motion split fingerprint mismatch")
    if dataset_manifest.get("teacher_promotion") != teacher_promotion:
        raise ValueError("latent dataset and training teacher promotion bindings differ")
    if report.get("teacher_promotion") != teacher_promotion:
        raise ValueError("closed-loop report teacher promotion binding mismatch")
    from musclemimic.distill.provenance import validate_teacher_promotion_binding

    validate_teacher_promotion_binding(
        teacher_promotion,
        teacher_checkpoint=teacher,
        require_promoted=True,
    )
    report_teacher = report.get("teacher_checkpoint")
    if not isinstance(report_teacher, dict) or report_teacher.get("sha256") != teacher.get("sha256"):
        raise ValueError("closed-loop report teacher fingerprint differs from latent training")
    if verify_external_files and not checkpoint_fingerprint_matches(report_teacher):
        raise ValueError("closed-loop report teacher checkpoint no longer matches its content fingerprint")

    evidence_record = report.get("direct_promotion_evidence")
    if not isinstance(evidence_record, dict):
        raise ValueError("closed-loop report lacks direct promotion evidence")
    evidence_path = Path(str(evidence_record.get("path", "")))
    if not evidence_path.is_file() or file_sha256(evidence_path) != evidence_record.get("sha256"):
        raise ValueError("direct promotion evidence file/hash mismatch")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("schema_version") != "direct_distill_promotion_evidence_v2":
        raise ValueError("direct promotion evidence schema is invalid")
    evidence_without_hash = {
        key: value for key, value in evidence.items() if key != "evidence_fingerprint"
    }
    if evidence.get("evidence_fingerprint") != canonical_json_sha256(evidence_without_hash):
        raise ValueError("direct promotion evidence fingerprint mismatch")
    if evidence.get("deterministic") is not True:
        raise ValueError("direct promotion evidence must be deterministic")
    heldout = evidence.get("heldout") or {}
    if heldout.get("motion_paths") != normalized or heldout.get("motion_uids") != expected_uids:
        raise ValueError("direct and latent promotion held-out motions differ")
    if evidence.get("teacher_checkpoint", {}).get("sha256") != teacher.get("sha256"):
        raise ValueError("direct and latent teacher checkpoint fingerprints differ")
    if evidence.get("teacher_promotion") != teacher_promotion:
        raise ValueError("direct and latent teacher promotion bindings differ")
    student = evidence.get("student_checkpoint")
    if not isinstance(student, dict):
        raise ValueError("direct promotion evidence lacks selected student checkpoint")
    if verify_external_files and not checkpoint_fingerprint_matches(student):
        raise ValueError("selected direct checkpoint no longer matches promotion evidence")
    artifacts = evidence.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("direct promotion evidence artifact map is missing")
    for name in ("comparison_metrics", "acceptance", "convergence", "temporal_audit"):
        record = artifacts.get(name)
        artifact_path = Path(str((record or {}).get("path", "")))
        if not isinstance(record, dict) or not artifact_path.is_file():
            raise ValueError(f"direct promotion artifact is missing: {name}")
        if file_sha256(artifact_path) != record.get("sha256"):
            raise ValueError(f"direct promotion artifact was modified: {name}")
    acceptance = json.loads(Path(artifacts["acceptance"]["path"]).read_text(encoding="utf-8"))
    policy = evidence.get("promotion_policy")
    validate_direct_acceptance_record(acceptance.get(policy))
    if report.get("direct_rollout_policy") != policy:
        raise ValueError("closed-loop direct rollout policy differs from accepted direct checkpoint")
    rollout_record = report.get("direct_rollout_metrics")
    if not isinstance(rollout_record, dict):
        raise ValueError("closed-loop report lacks direct rollout metrics provenance")
    if (
        rollout_record.get("path") != artifacts["comparison_metrics"].get("path")
        or rollout_record.get("sha256") != artifacts["comparison_metrics"].get("sha256")
    ):
        raise ValueError("closed-loop direct rollout metrics differ from direct acceptance evidence")

    if require_seal:
        expected = canonical_json_sha256(
            {key: value for key, value in report.items() if key != "report_fingerprint"}
        )
        if report.get("report_fingerprint") != expected:
            raise ValueError("closed-loop promotion report fingerprint mismatch")
        promotion = report.get("promotion")
        if not isinstance(promotion, dict) or promotion.get("passed") is not True:
            raise ValueError("closed-loop promotion report is not passed")
        config_payload = json.loads((path / "latent_config.yaml").read_text(encoding="utf-8"))
        current_eval = json.loads((path / "eval_metrics.json").read_text(encoding="utf-8"))
        for key in OFFLINE_PROMOTION_METRICS:
            if float(current_eval.get(key, np.nan)) != float(offline[key]):
                raise ValueError(f"closed-loop report offline metric differs from checkpoint: {key}")
        from dataclasses import fields

        from musclemimic.latent_muscle.train_latent import (
            LatentTrainConfig,
            _evaluate_promotion_gates,
        )

        allowed = {item.name for item in fields(LatentTrainConfig)}
        train_config = LatentTrainConfig(
            **{key: value for key, value in config_payload.items() if key in allowed}
        )
        recompute_metrics = dict(offline)
        for key in (
            "fall_or_early_termination_rate",
            "body_racket_relative_degradation",
            "lambda_025_050_no_fall_rate",
        ):
            if key not in report or not np.isfinite(float(report[key])):
                raise ValueError(f"closed-loop report promotion scalar is invalid: {key}")
            recompute_metrics[f"closed_loop_{key}"] = float(report[key])
        recompute_metrics["closed_loop_evidence_kind"] = "verified_production_v2"
        if (
            report.get("teacher_promotion_evidence_kind")
            != "verified_stage2_promotion_v1"
            or current_eval.get("teacher_promotion_evidence_kind")
            != "verified_stage2_promotion_v1"
        ):
            raise ValueError("closed-loop report lacks verified Stage-2 teacher promotion")
        recompute_metrics["teacher_promotion_evidence_kind"] = (
            "verified_stage2_promotion_v1"
        )
        expected_promotion = _evaluate_promotion_gates(recompute_metrics, train_config)
        if promotion != expected_promotion:
            raise ValueError("closed-loop promotion result does not match recomputed gates")
    return report


def _set_fixed_start(env: Any, traj_index: int) -> None:
    th = env.th
    th.random_start = False
    th.use_fixed_start = True
    th.start_from_random_step = False
    th.fixed_start_conf = [int(traj_index), 0]


def _reset_obs(env: Any) -> np.ndarray:
    result = env.reset()
    obs = result[0] if isinstance(result, tuple) else result
    return np.asarray(obs, dtype=np.float32)


def _accumulate_numeric_step_metrics(
    info: dict[str, Any],
    sums: dict[str, float],
    counts: dict[str, int],
) -> None:
    for key, raw_value in info.items():
        if not (str(key).startswith("err_") or str(key).startswith("racket_")):
            continue
        value = np.asarray(raw_value)
        if value.size != 1 or not np.isfinite(value).all():
            continue
        name = str(key)
        sums[name] = sums.get(name, 0.0) + float(value.reshape(-1)[0])
        counts[name] = counts.get(name, 0) + 1


def _flatten_numeric(payload: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    stack: list[dict[str, Any]] = [payload]
    while stack:
        mapping = stack.pop()
        for key, value in mapping.items():
            if isinstance(value, dict):
                stack.append(value)
            else:
                array = np.asarray(value)
                if array.size == 1 and np.issubdtype(array.dtype, np.number):
                    result[str(key)] = float(array.reshape(-1)[0])
    return result


def _lambda_key(value: float) -> str:
    return f"lambda_{float(value):.3f}".replace(".", "p")


def _lambda_seed(value: float) -> int:
    return round(float(value) * 1_000_000)


def _softplus(value: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(value))) + np.maximum(value, 0.0)

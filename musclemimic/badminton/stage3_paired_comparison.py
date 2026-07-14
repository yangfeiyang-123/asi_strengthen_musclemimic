"""Build a sealed direct-vs-synergy Stage-3 paired comparison.

The producer accepts only complete ``incoming_shuttle_hit_evaluate_v3``
reports.  It revalidates their immutable evaluation bindings, proves that the
two branches used the same train/evaluation banks and seeds, and binds each
branch to the independently selected latent checkpoint family.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from musclemimic.badminton.json_contract import load_json_strict
from musclemimic.badminton.scripts.latent_synergy_sweep import (
    validate_selected_artifact,
)
from musclemimic.badminton.scripts.run_incoming_shuttle_hit import (
    _stage3_evaluation_content_sha256,
)

SCHEMA_VERSION = "stage3_direct_synergy_paired_comparison_v1"
DEFAULT_BOOTSTRAP_SAMPLES = 2_000
DEFAULT_BOOTSTRAP_SEED = 20_260_713

_COMMON_BINDING_FIELDS = (
    "spec_sha256",
    "scene_sha256",
    "training_feed_manifest_sha256",
    "evaluation_feed_manifest_sha256",
    "training_target_bank_sha256",
    "training_target_source_fingerprint",
    "training_target_file_sha256",
    "evaluation_target_bank_sha256",
    "evaluation_target_source_fingerprint",
    "evaluation_target_file_sha256",
    "training_seed",
    "evaluation_seed",
    "checkpoint_env_steps",
    "checkpoint_task_curriculum_max_stage",
    "checkpoint_task_curriculum_complete",
)

# Every metric is reported as synergy - direct, plus an improvement-oriented
# value whose sign is positive when the synergy branch is better.
_METRIC_DIRECTIONS = {
    "mean_return": "higher",
    "no_fall_rate": "higher",
    "hit_rate": "higher",
    "crossed_net_rate": "higher",
    "opponent_back_landing_rate": "higher",
    "mean_contact_racket_head_speed_m_s": "higher",
    "mean_net_clearance_m": "higher",
    "impact_position_error_m": "lower",
    "center_hit_rate": "higher",
    "impact_timing_mae_s": "lower",
    "stringbed_normal_error_rad": "lower",
    "racket_linear_velocity_rmse_m_s": "lower",
    "racket_angular_velocity_rmse_rad_s": "lower",
    "landing_rmse_m": "lower",
    "apex_mae_m": "lower",
    "recovery_ready_rate": "higher",
    "normalized_control_energy": "lower",
    "body_action_saturation_fraction": "lower",
    "full_action_saturation_fraction": "lower",
    "raw_latent_saturation": "lower",
    "lab_state_ood_fraction": "lower",
    "naturalness.body_relative_deviation_to_prior": "lower",
    "naturalness.right_hand_site_rmse_to_prior_m": "lower",
    "naturalness.racket_position_rmse_to_prior_m": "lower",
    "naturalness.racket_rotation_rmse_to_prior_rad": "lower",
}

_RMSE_METRICS = {
    "racket_linear_velocity_rmse_m_s",
    "racket_angular_velocity_rmse_rad_s",
    "landing_rmse_m",
}


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 fingerprint")
    return value


def _require_finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _mapping_at(report: Mapping[str, Any], dotted_name: str) -> Any:
    value: Any = report
    for component in dotted_name.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise ValueError(f"evaluation report is missing {dotted_name}")
        value = value[component]
    return value


def _validate_bound_file(binding: Mapping[str, Any], path_key: str, hash_key: str) -> Path:
    raw_path = binding.get(path_key)
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"Stage-3 artifact binding has no {path_key}")
    path = Path(raw_path).expanduser().resolve(strict=True)
    if not path.is_file() or binding.get(hash_key) != _file_sha256(path):
        raise ValueError(f"Stage-3 bound file changed: {path_key}")
    return path


def _validate_target_binding(binding: Mapping[str, Any], label: str) -> None:
    target_path = _validate_bound_file(
        binding,
        f"{label}_target_path",
        f"{label}_target_file_sha256",
    )
    target = load_json_strict(target_path)
    if not isinstance(target, dict):
        raise ValueError(f"{label} target bank must be a JSON object")
    for key in ("bank_sha256", "source_fingerprint"):
        expected = _require_sha256(
            binding.get(f"{label}_target_{key}"),
            f"{label}_target_{key}",
        )
        if target.get(key) != expected:
            raise ValueError(f"{label} target bank differs from its evaluation binding")


def _validate_evaluation_binding(
    report: dict[str, Any],
    *,
    report_path: Path,
    family: str,
    expected_latent_fingerprint: str,
) -> dict[str, Any]:
    if report.get("schema_version") != "incoming_shuttle_hit_evaluate_v3":
        raise ValueError(f"{family} report has an unsupported schema")
    if report.get("runner_stage") != "evaluate" or report.get("passed") is not True:
        raise ValueError(f"{family} Stage-3 evaluation has not passed")
    if report.get("artifact_binding_verified") != 1.0:
        raise ValueError(f"{family} Stage-3 report has no verified artifact binding")
    gates = report.get("promotion_gates")
    if not isinstance(gates, dict) or not gates or not all(value is True for value in gates.values()):
        raise ValueError(f"{family} Stage-3 promotion gates are incomplete")

    binding = report.get("artifact_binding")
    if not isinstance(binding, dict):
        raise ValueError(f"{family} Stage-3 report has no artifact binding")
    if (
        binding.get("schema_version") != "incoming_hit_evaluation_artifact_binding_v3"
        or binding.get("verified") is not True
    ):
        raise ValueError(f"{family} Stage-3 artifact binding is incompatible")
    supplied_binding_hash = binding.get("binding_sha256")
    unbound = dict(binding)
    unbound.pop("binding_sha256", None)
    if supplied_binding_hash != _canonical_sha256(unbound):
        raise ValueError(f"{family} Stage-3 artifact binding fingerprint mismatch")
    if binding.get("evaluation_content_sha256") != _stage3_evaluation_content_sha256(report):
        raise ValueError(f"{family} Stage-3 evaluation content changed after binding")
    if binding.get("latent_checkpoint_fingerprint") != expected_latent_fingerprint:
        raise ValueError(f"{family} Stage-3 branch uses the wrong selected latent checkpoint")
    control = report.get("control_manifest")
    if not isinstance(control, dict) or (control.get("latent_checkpoint_fingerprint") != expected_latent_fingerprint):
        raise ValueError(f"{family} Stage-3 control manifest uses the wrong latent checkpoint")

    for path_key, hash_key in (
        ("checkpoint_payload_path", "checkpoint_payload_sha256"),
        ("checkpoint_metadata_path", "checkpoint_metadata_sha256"),
        ("spec_path", "spec_sha256"),
        ("scene_path", "scene_sha256"),
        ("train_report_path", "train_report_sha256"),
    ):
        _validate_bound_file(binding, path_key, hash_key)
    for label in ("training", "evaluation"):
        _validate_target_binding(binding, label)

    training_feed = report.get("training_feed_manifest")
    evaluation_feed = report.get("evaluation_feed_manifest")
    if not isinstance(training_feed, dict) or not isinstance(evaluation_feed, dict):
        raise ValueError(f"{family} Stage-3 report has incomplete feed manifests")
    if binding.get("training_feed_manifest_sha256") != _canonical_sha256(training_feed) or binding.get(
        "evaluation_feed_manifest_sha256"
    ) != _canonical_sha256(evaluation_feed):
        raise ValueError(f"{family} Stage-3 feed manifests differ from their binding")

    metadata_path = Path(binding["checkpoint_metadata_path"])
    metadata = load_json_strict(metadata_path)
    if not isinstance(metadata, dict):
        raise ValueError(f"{family} checkpoint metadata must be a JSON object")
    config = metadata.get("config")
    if not isinstance(config, dict) or isinstance(config.get("seed"), bool):
        raise ValueError(f"{family} checkpoint metadata has no training seed")
    try:
        metadata_seed = int(config["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{family} checkpoint metadata has no training seed") from exc
    if float(config["seed"]) != float(metadata_seed) or binding.get("training_seed") != metadata_seed:
        raise ValueError(f"{family} training seed differs from checkpoint metadata")
    evaluation_seed = report.get("evaluation_seed")
    if isinstance(evaluation_seed, bool) or not isinstance(evaluation_seed, int):
        raise ValueError(f"{family} report has no exact evaluation seed")
    if binding.get("evaluation_seed") != evaluation_seed:
        raise ValueError(f"{family} evaluation seed differs from its binding")

    episodes = report.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"{family} Stage-3 report contains no paired episodes")
    episode_indices: list[int] = []
    for expected_index, episode in enumerate(episodes):
        if not isinstance(episode, dict) or isinstance(episode.get("episode"), bool):
            raise ValueError(f"{family} Stage-3 episode identity is invalid")
        try:
            index = int(episode["episode"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{family} Stage-3 episode identity is invalid") from exc
        if index != expected_index:
            raise ValueError(f"{family} Stage-3 episodes are not in exact feed-index order")
        episode_indices.append(index)
    if report.get("evaluated_feed_count") != len(episodes):
        raise ValueError(f"{family} evaluated feed count differs from its episodes")
    if report.get("required_heldout_feed_count") != len(episodes):
        raise ValueError(f"{family} report did not evaluate the complete held-out feed bank")
    sample_fingerprints = evaluation_feed.get("sample_fingerprints")
    if not isinstance(sample_fingerprints, list) or len(sample_fingerprints) < len(episodes):
        raise ValueError(f"{family} evaluation feed manifest cannot identify every episode")

    return {
        "family": family,
        "report_path": str(report_path.resolve()),
        "report_sha256": _file_sha256(report_path),
        "report": report,
        "binding": binding,
        "episode_indices": episode_indices,
    }


def _episode_metric(episode: Mapping[str, Any], name: str) -> float | None:
    if name == "mean_return":
        value: Any = episode.get("return")
    elif name == "no_fall_rate":
        value = 0.0 if bool(episode.get("body_fall", True)) else 1.0
    elif name == "hit_rate":
        value = 1.0 if bool(episode.get("hit", False)) else 0.0
    elif name == "crossed_net_rate":
        value = 1.0 if bool(episode.get("crossed_net", False)) else 0.0
    elif name == "opponent_back_landing_rate":
        value = 1.0 if episode.get("landing_region") == "opponent_back" else 0.0
    elif name == "mean_contact_racket_head_speed_m_s":
        value = episode.get("contact_racket_head_speed_m_s")
    elif name == "mean_net_clearance_m":
        value = episode.get("net_clearance_m")
    elif name.startswith("naturalness."):
        naturalness = episode.get("naturalness")
        value = naturalness.get(name.split(".", 1)[1]) if isinstance(naturalness, Mapping) else None
    elif name in {
        "normalized_control_energy",
        "body_action_saturation_fraction",
        "full_action_saturation_fraction",
        "raw_latent_saturation",
        "lab_state_ood_fraction",
    }:
        diagnostics = episode.get("lab_diagnostics")
        value = diagnostics.get(name) if isinstance(diagnostics, Mapping) else None
    else:
        v2 = episode.get("stage3_v2_metrics")
        if not isinstance(v2, Mapping):
            return None
        source_name = {
            "impact_position_error_m": "impact_position_error_m",
            "center_hit_rate": "impact_rho2",
            "impact_timing_mae_s": "impact_timing_error_s",
            "stringbed_normal_error_rad": "stringbed_normal_error_rad",
            "racket_linear_velocity_rmse_m_s": "racket_linear_velocity_error_m_s",
            "racket_angular_velocity_rmse_rad_s": "racket_angular_velocity_error_rad_s",
            "landing_rmse_m": "landing_error_m",
            "apex_mae_m": "apex_error_m",
            "recovery_ready_rate": "ready_pose_error",
        }.get(name)
        if source_name is None or source_name not in v2:
            return None
        value = v2[source_name]
        if name == "center_hit_rate":
            value = 1.0 if _require_finite(value, source_name) <= 0.25 else 0.0
        elif name == "recovery_ready_rate":
            value = (
                1.0
                if bool(episode.get("recovery_complete", False)) and _require_finite(value, source_name) <= 0.15
                else 0.0
            )
        elif name in {"impact_timing_mae_s", "apex_mae_m"}:
            value = abs(_require_finite(value, source_name))
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _aggregate(values: np.ndarray, name: str) -> float:
    if name in _RMSE_METRICS:
        return float(np.sqrt(np.mean(np.square(values))))
    return float(np.mean(values))


def _paired_metric(
    direct_report: Mapping[str, Any],
    synergy_report: Mapping[str, Any],
    *,
    name: str,
    direction: str,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    direct_value = _require_finite(_mapping_at(direct_report, name), f"direct {name}")
    synergy_value = _require_finite(_mapping_at(synergy_report, name), f"synergy {name}")
    delta = synergy_value - direct_value
    sign = 1.0 if direction == "higher" else -1.0
    direct_pairs: list[float] = []
    synergy_pairs: list[float] = []
    for direct_episode, synergy_episode in zip(direct_report["episodes"], synergy_report["episodes"], strict=True):
        direct_episode_value = _episode_metric(direct_episode, name)
        synergy_episode_value = _episode_metric(synergy_episode, name)
        if direct_episode_value is None or synergy_episode_value is None:
            continue
        direct_pairs.append(direct_episode_value)
        synergy_pairs.append(synergy_episode_value)
    if not direct_pairs:
        raise ValueError(f"paired Stage-3 reports have no comparable episodes for {name}")
    direct_array = np.asarray(direct_pairs, dtype=np.float64)
    synergy_array = np.asarray(synergy_pairs, dtype=np.float64)
    paired_improvements = sign * (synergy_array - direct_array)
    standard_deviation = float(np.std(paired_improvements, ddof=1)) if len(paired_improvements) > 1 else 0.0
    effect_size = float(np.mean(paired_improvements) / standard_deviation) if standard_deviation > 0.0 else None
    bootstrap = np.empty(int(bootstrap_samples), dtype=np.float64)
    for index in range(int(bootstrap_samples)):
        draw = rng.integers(0, len(direct_array), size=len(direct_array))
        bootstrap[index] = sign * (_aggregate(synergy_array[draw], name) - _aggregate(direct_array[draw], name))
    ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "direction": direction,
        "direct": direct_value,
        "synergy": synergy_value,
        "synergy_minus_direct": delta,
        "synergy_improvement": sign * delta,
        "paired_episode_count": len(direct_pairs),
        "paired_standardized_effect": effect_size,
        "paired_bootstrap_improvement_ci95": [float(ci_low), float(ci_high)],
    }


def _selection_identity(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = validate_selected_artifact(manifest_path)
    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, dict) or set(checkpoints) != {
        "best_direct",
        "best_synergy",
    }:
        raise ValueError("paired Stage-3 comparison requires best_direct and best_synergy")
    direct = checkpoints["best_direct"]
    synergy = checkpoints["best_synergy"]
    if direct.get("decoder_type") != "direct":
        raise ValueError("best_direct selection is not a direct decoder")
    if synergy.get("decoder_type") == "direct":
        raise ValueError("best_synergy selection is not a synergy decoder")
    for family, entry in (("best_direct", direct), ("best_synergy", synergy)):
        _require_sha256(entry.get("checkpoint_fingerprint"), f"{family} checkpoint")
        _require_sha256(
            entry.get("formal_synergy_basis_fingerprint"),
            f"{family} formal synergy basis",
        )
    return manifest, checkpoints


def build_paired_comparison(
    *,
    direct_report_path: str | Path,
    synergy_report_path: str | Path,
    selection_manifest_path: str | Path,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Validate both branches and return a self-fingerprinted paired report."""

    if int(bootstrap_samples) < 100:
        raise ValueError("paired bootstrap requires at least 100 samples")
    direct_path = Path(direct_report_path).expanduser().resolve(strict=True)
    synergy_path = Path(synergy_report_path).expanduser().resolve(strict=True)
    manifest_path = Path(selection_manifest_path).expanduser().resolve(strict=True)
    if len({direct_path, synergy_path, manifest_path}) != 3:
        raise ValueError("paired Stage-3 sources must be distinct files")
    manifest, checkpoints = _selection_identity(manifest_path)

    direct_report = load_json_strict(direct_path)
    synergy_report = load_json_strict(synergy_path)
    if not isinstance(direct_report, dict) or not isinstance(synergy_report, dict):
        raise ValueError("paired Stage-3 reports must be JSON objects")
    direct = _validate_evaluation_binding(
        direct_report,
        report_path=direct_path,
        family="best_direct",
        expected_latent_fingerprint=checkpoints["best_direct"]["checkpoint_fingerprint"],
    )
    synergy = _validate_evaluation_binding(
        synergy_report,
        report_path=synergy_path,
        family="best_synergy",
        expected_latent_fingerprint=checkpoints["best_synergy"]["checkpoint_fingerprint"],
    )
    if direct["episode_indices"] != synergy["episode_indices"]:
        raise ValueError("direct and synergy reports do not share exact feed indices")

    common_protocol: dict[str, Any] = {}
    for field in _COMMON_BINDING_FIELDS:
        direct_value = direct["binding"].get(field)
        synergy_value = synergy["binding"].get(field)
        if direct_value != synergy_value:
            raise ValueError(f"direct and synergy Stage-3 protocols differ on {field}")
        common_protocol[field] = direct_value
    if direct_report["evaluation_feed_manifest"] != synergy_report["evaluation_feed_manifest"]:
        raise ValueError("direct and synergy Stage-3 evaluation feed content differs")
    if direct_report["training_feed_manifest"] != synergy_report["training_feed_manifest"]:
        raise ValueError("direct and synergy Stage-3 training feed content differs")
    common_protocol.update(
        {
            "paired_episode_indices": direct["episode_indices"],
            "paired_episode_count": len(direct["episode_indices"]),
            "evaluation_feed_sample_fingerprints": direct_report["evaluation_feed_manifest"]["sample_fingerprints"][
                : len(direct["episode_indices"])
            ],
        }
    )

    rng = np.random.default_rng(int(bootstrap_seed))
    paired_metrics = {
        name: _paired_metric(
            direct_report,
            synergy_report,
            name=name,
            direction=direction,
            bootstrap_samples=int(bootstrap_samples),
            rng=rng,
        )
        for name, direction in _METRIC_DIRECTIONS.items()
    }
    promotion_fingerprint = _require_sha256(
        manifest.get("promotion_metrics_fingerprint"),
        "latent promotion metrics",
    )
    synergy_entry = checkpoints["best_synergy"]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "comparison_design": "paired_seed_feed_v1",
        "binding_verified": 1.0,
        "passed": True,
        "source_reports": {
            "best_direct": {
                "path": direct["report_path"],
                "sha256": direct["report_sha256"],
                "evaluation_binding_sha256": direct["binding"]["binding_sha256"],
            },
            "best_synergy": {
                "path": synergy["report_path"],
                "sha256": synergy["report_sha256"],
                "evaluation_binding_sha256": synergy["binding"]["binding_sha256"],
            },
        },
        "latent_selection": {
            "path": str(manifest_path),
            "sha256": _file_sha256(manifest_path),
            "selection_manifest_fingerprint": manifest["selection_manifest_fingerprint"],
            "promotion_metrics_fingerprint": promotion_fingerprint,
        },
        "shared_protocol": common_protocol,
        "branch_identities": {
            family: {
                "latent_checkpoint_fingerprint": branch["binding"]["latent_checkpoint_fingerprint"],
                "stage3_checkpoint_payload_sha256": branch["binding"]["checkpoint_payload_sha256"],
                "stage3_checkpoint_metadata_sha256": branch["binding"]["checkpoint_metadata_sha256"],
                "policy_abi_hash": branch["binding"]["policy_abi_hash"],
                "training_control_hash": branch["binding"]["training_control_hash"],
            }
            for family, branch in (
                ("best_direct", direct),
                ("best_synergy", synergy),
            )
        },
        "paired_metrics": paired_metrics,
        "bootstrap": {
            "samples": int(bootstrap_samples),
            "seed": int(bootstrap_seed),
            "unit": "heldout_feed_episode_pair",
        },
        # EMG validates the promoted low-level synergy controller used by the
        # final Stage-3 branch.  The final high-level policy identity remains
        # separately sealed above in branch_identities.
        "selected_policy_for_emg": {
            "family": "best_synergy",
            "policy_checkpoint_fingerprint": synergy_entry["checkpoint_fingerprint"],
            "policy_promotion_fingerprint": promotion_fingerprint,
            "formal_synergy_basis_fingerprint": synergy_entry["formal_synergy_basis_fingerprint"],
            "event_reference_fingerprint": common_protocol["evaluation_target_source_fingerprint"],
            "policy_decoder_type": synergy_entry["decoder_type"],
            "stage3_checkpoint_payload_sha256": synergy["binding"]["checkpoint_payload_sha256"],
        },
    }
    payload["paired_comparison_fingerprint"] = _canonical_sha256(payload)
    return payload


def validate_paired_comparison(path: str | Path) -> dict[str, Any]:
    """Recompute a persisted comparison from its sealed source artifacts."""

    report_path = Path(path).expanduser().resolve(strict=True)
    report = load_json_strict(report_path)
    if not isinstance(report, dict) or report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported Stage-3 paired comparison schema")
    supplied = report.get("paired_comparison_fingerprint")
    unbound = dict(report)
    unbound.pop("paired_comparison_fingerprint", None)
    if supplied != _canonical_sha256(unbound):
        raise ValueError("Stage-3 paired comparison fingerprint mismatch")
    try:
        source_reports = report["source_reports"]
        latent_selection = report["latent_selection"]
        bootstrap = report["bootstrap"]
        recomputed = build_paired_comparison(
            direct_report_path=source_reports["best_direct"]["path"],
            synergy_report_path=source_reports["best_synergy"]["path"],
            selection_manifest_path=latent_selection["path"],
            bootstrap_samples=int(bootstrap["samples"]),
            bootstrap_seed=int(bootstrap["seed"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Stage-3 paired comparison has incomplete source binding") from exc
    if report != recomputed:
        raise ValueError("Stage-3 paired comparison no longer matches its source artifacts")
    return report


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-report", required=True)
    parser.add_argument("--synergy-report", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_paired_comparison(
        direct_report_path=args.direct_report,
        synergy_report_path=args.synergy_report,
        selection_manifest_path=args.selection_manifest,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.seed,
    )
    output = Path(args.output)
    _write_atomic(output, payload)
    validate_paired_comparison(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

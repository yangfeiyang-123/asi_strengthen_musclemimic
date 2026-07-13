"""Fail-closed Stage-1R paired-rollout promotion evidence.

Stage-1R is evaluated outside the PPO training loop.  Its numerical report is
therefore useful for promotion only when it is bound to the exact checkpoint,
canonical held-out motions and the two source rollout payloads that produced
it.  Legacy offline comparisons remain supported by the evaluator, but this
module deliberately rejects them for production promotion.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from musclemimic.badminton.data_qc import VAL_MOTIONS
from musclemimic.badminton.promotion_artifact import checkpoint_identity, sha256_path

REPORT_SCHEMA_VERSION = "stage1r_paired_robustness_v3"
SOURCE_SCHEMA_VERSION = "stage1r_rollout_source_v1"
EVALUATION_CONTRACT_SCHEMA_VERSION = "stage1r_evaluation_contract_v1"
HELDOUT_IDENTITY_SCHEMA_VERSION = "stage1r_heldout_motion_identity_v1"
VERIFIED_EVIDENCE_KIND = "verified_checkpoint_rollout_v1"
LEGACY_EVIDENCE_KIND = "legacy_offline_v1"
CANONICAL_SEEDS = (0, 1, 2, 3, 4)
CANONICAL_METRICS_ENVS = 5
CANONICAL_METRICS_STEPS = 500
DATA_VARIANT = "raw_smooth_v1"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATASET_ROOT = _REPO_ROOT / "datasets" / "forehandClear_standard"
_RELEASE_MANIFEST = (
    _DATASET_ROOT / "manifests" / DATA_VARIANT / "release_manifest.json"
)
_CORE_REPORT_KEYS = (
    "clean_provenance",
    "finger_qpos_perturb_scale",
    "metrics",
    "new_root_hand_racket_spike_count",
    "pair_count",
    "passed",
    "perturbed_provenance",
    "seed_hash",
    "spike_checks",
)
_SOURCE_METRIC_KEYS = (
    "body_site_error",
    "right_hand_site_error",
    "racket_head_position_error",
    "racket_head_rotation_error",
    "early_termination",
    "val_max_err_root_xyz",
    "val_max_err_right_hand_pos",
    "val_max_err_racket_pos",
    "val_max_err_racket_rot",
)
_ROLLOUT_SOURCE_KEYS = {
    "body_site_error": "val_err_rpos",
    "right_hand_site_error": "val_err_right_hand_pos",
    "racket_head_position_error": "val_err_racket_pos",
    "racket_head_rotation_error": "val_err_racket_rot",
    "early_termination": "val_early_termination_rate",
    "val_max_err_root_xyz": "val_max_err_root_xyz",
    "val_max_err_right_hand_pos": "val_max_err_right_hand_pos",
    "val_max_err_racket_pos": "val_max_err_racket_pos",
    "val_max_err_racket_rot": "val_max_err_racket_rot",
}


def canonical_heldout_motion_paths() -> tuple[str, ...]:
    return tuple(
        f"forehandClear_standard/muscle_trajectory/{DATA_VARIANT}/{motion}"
        for motion in VAL_MOTIONS
    )


def canonical_mapping_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def canonical_heldout_motion_identity(
    motion_paths: Sequence[str],
) -> dict[str, Any]:
    """Bind the exact five held-out names to release-v3 source/cache hashes."""

    paths = tuple(str(path) for path in motion_paths)
    expected_paths = canonical_heldout_motion_paths()
    if paths != expected_paths:
        raise ValueError(
            "Stage-1R production evidence requires the canonical ordered five "
            f"held-out motions: expected={expected_paths!r} actual={paths!r}"
        )
    manifest = _load_mapping(_RELEASE_MANIFEST, "raw_smooth_v1 release manifest")
    if manifest.get("schema_version") != "forehand_clear_raw_smooth_release_v3":
        raise ValueError("Stage-1R requires the release-v3 manifest")
    release_sha256 = manifest.get("release_sha256")
    if not _is_sha256(release_sha256):
        raise ValueError("raw_smooth_v1 release manifest has no content identity")
    raw_motions = manifest.get("motions")
    if not isinstance(raw_motions, list):
        raise ValueError("raw_smooth_v1 release manifest has no motion inventory")
    by_motion = {
        str(item.get("motion")): item
        for item in raw_motions
        if isinstance(item, Mapping) and item.get("motion") is not None
    }
    records: list[dict[str, Any]] = []
    for motion, motion_path in zip(VAL_MOTIONS, paths, strict=True):
        item = by_motion.get(motion)
        if not isinstance(item, Mapping) or item.get("split") != "validation":
            raise ValueError(
                f"canonical held-out motion {motion!r} is absent from validation release"
            )
        cache = item.get("cache")
        source = item.get("source")
        if not isinstance(cache, Mapping) or not isinstance(source, Mapping):
            raise ValueError(f"canonical held-out motion {motion!r} lacks source/cache identity")
        cache_sha256 = cache.get("sha256")
        source_sha256 = source.get("sha256")
        if not _is_sha256(cache_sha256) or not _is_sha256(source_sha256):
            raise ValueError(f"canonical held-out motion {motion!r} has an invalid SHA")
        cache_path = _DATASET_ROOT / str(cache.get("path", ""))
        source_path = _DATASET_ROOT / str(source.get("path", ""))
        if sha256_path(cache_path) != cache_sha256:
            raise ValueError(
                f"canonical held-out motion {motion!r} cache differs from release"
            )
        if sha256_path(source_path) != source_sha256:
            raise ValueError(
                f"canonical held-out motion {motion!r} source differs from release"
            )
        records.append(
            {
                "motion": motion,
                "motion_path": motion_path,
                "cache_path": str(cache_path.resolve(strict=True)),
                "cache_sha256": str(cache_sha256),
                "source_path": str(source_path.resolve(strict=True)),
                "source_sha256": str(source_sha256),
            }
        )
    identity: dict[str, Any] = {
        "schema_version": HELDOUT_IDENTITY_SCHEMA_VERSION,
        "dataset": "forehandClear_standard",
        "variant": DATA_VARIANT,
        "release_manifest_path": str(_RELEASE_MANIFEST.resolve(strict=True)),
        "release_manifest_content_sha256": sha256_path(_RELEASE_MANIFEST),
        "release_sha256": str(release_sha256),
        "motion_paths": list(paths),
        "motions": records,
    }
    identity["identity_sha256"] = canonical_mapping_sha256(identity)
    return identity


def build_evaluation_contract(
    *,
    motion_paths: Sequence[str],
    seeds: Sequence[int],
    perturb_qpos_scale: float,
    perturb_qvel_scale: float,
    metrics_envs: int,
    metrics_steps: int,
) -> dict[str, Any]:
    seed_values = tuple(int(seed) for seed in seeds)
    if seed_values != CANONICAL_SEEDS or len(set(seed_values)) != len(seed_values):
        raise ValueError(
            "Stage-1R production evidence requires exact distinct seeds "
            f"{CANONICAL_SEEDS!r}; got {seed_values!r}"
        )
    qpos_scale = _finite_float("perturb_qpos_scale", perturb_qpos_scale)
    qvel_scale = _finite_float("perturb_qvel_scale", perturb_qvel_scale)
    if qpos_scale <= 0.0:
        raise ValueError("Stage-1R perturbed qpos scale must be positive")
    if qvel_scale != 0.0:
        raise ValueError("Stage-1R production evidence requires zero qvel perturbation")
    env_count = int(metrics_envs)
    step_count = int(metrics_steps)
    if env_count != CANONICAL_METRICS_ENVS:
        raise ValueError(
            "Stage-1R production evidence requires "
            f"metrics_envs={CANONICAL_METRICS_ENVS}"
        )
    if step_count != CANONICAL_METRICS_STEPS:
        raise ValueError(
            "Stage-1R production evidence requires "
            f"metrics_steps={CANONICAL_METRICS_STEPS}"
        )
    return {
        "schema_version": EVALUATION_CONTRACT_SCHEMA_VERSION,
        "heldout_motion_identity": canonical_heldout_motion_identity(motion_paths),
        "seeds": list(seed_values),
        "pair_count": len(seed_values),
        "deterministic_policy": True,
        "evaluate_all": True,
        "finger_perturb_rng_mode": "fold_in",
        "finger_perturb_side": "right",
        "clean_finger_qpos_perturb_scale": 0.0,
        "clean_finger_qvel_perturb_scale": 0.0,
        "perturbed_finger_qpos_perturb_scale": qpos_scale,
        "perturbed_finger_qvel_perturb_scale": qvel_scale,
        "metrics_envs": env_count,
        "metrics_steps": step_count,
    }


def build_verified_report(
    base_report: Mapping[str, Any],
    *,
    checkpoint: str | Path,
    evaluation_contract: Mapping[str, Any],
    clean_source_path: str | Path,
    perturbed_source_path: str | Path,
) -> dict[str, Any]:
    """Attach immutable source/checkpoint identities to a numerical report."""

    clean_path = Path(clean_source_path).expanduser().resolve(strict=True)
    perturbed_path = Path(perturbed_source_path).expanduser().resolve(strict=True)
    identity = checkpoint_identity(checkpoint)
    contract = dict(evaluation_contract)
    try:
        expected_contract = build_evaluation_contract(
            motion_paths=contract["heldout_motion_identity"]["motion_paths"],
            seeds=contract["seeds"],
            perturb_qpos_scale=contract[
                "perturbed_finger_qpos_perturb_scale"
            ],
            perturb_qvel_scale=contract[
                "perturbed_finger_qvel_perturb_scale"
            ],
            metrics_envs=contract["metrics_envs"],
            metrics_steps=contract["metrics_steps"],
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("Stage-1R evaluation contract is incomplete") from exc
    _require_equal_mapping(
        contract,
        expected_contract,
        "Stage-1R evaluation contract",
    )
    for condition, source_path in (
        ("clean", clean_path),
        ("perturbed", perturbed_path),
    ):
        _validate_source_payload(
            _load_mapping(source_path, f"Stage-1R {condition} source payload"),
            condition=condition,
            expected_checkpoint_identity=identity,
            expected_contract=expected_contract,
        )
    payload = dict(base_report)
    payload.update(
        {
            "schema_version": REPORT_SCHEMA_VERSION,
            "evidence_kind": VERIFIED_EVIDENCE_KIND,
            "production_eligible": True,
            "checkpoint_identity": identity,
            "evaluation_contract": expected_contract,
            "source_payloads": {
                "clean": {
                    "path": str(clean_path),
                    "content_sha256": sha256_path(clean_path),
                },
                "perturbed": {
                    "path": str(perturbed_path),
                    "content_sha256": sha256_path(perturbed_path),
                },
            },
        }
    )
    payload["binding_sha256"] = canonical_mapping_sha256(payload)
    return payload


def validate_stage1r_report(
    report_path: str | Path,
    *,
    expected_checkpoint: str | Path,
    expected_perturb_qpos_scale: float,
) -> dict[str, Any]:
    """Validate and recompute one production Stage-1R evidence artifact."""

    path = Path(report_path).expanduser().resolve(strict=True)
    report = _load_mapping(path, "Stage-1R paired robustness report")
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("Stage-1R production gate rejects legacy/offline report schema")
    if report.get("evidence_kind") != VERIFIED_EVIDENCE_KIND:
        raise ValueError("Stage-1R production gate requires verified checkpoint rollouts")
    if report.get("production_eligible") is not True:
        raise ValueError("Stage-1R report is not production eligible")

    recorded_binding = report.get("binding_sha256")
    unsigned = dict(report)
    unsigned.pop("binding_sha256", None)
    if not _is_sha256(recorded_binding) or recorded_binding != canonical_mapping_sha256(
        unsigned
    ):
        raise ValueError("Stage-1R report binding hash is missing or stale")

    expected_identity = checkpoint_identity(expected_checkpoint)
    _require_equal_mapping(
        report.get("checkpoint_identity"),
        expected_identity,
        "Stage-1R report checkpoint identity",
    )
    expected_contract = build_evaluation_contract(
        motion_paths=canonical_heldout_motion_paths(),
        seeds=CANONICAL_SEEDS,
        perturb_qpos_scale=expected_perturb_qpos_scale,
        perturb_qvel_scale=0.0,
        metrics_envs=CANONICAL_METRICS_ENVS,
        metrics_steps=CANONICAL_METRICS_STEPS,
    )
    _require_equal_mapping(
        report.get("evaluation_contract"),
        expected_contract,
        "Stage-1R evaluation contract",
    )

    references = report.get("source_payloads")
    if not isinstance(references, Mapping) or set(references) != {"clean", "perturbed"}:
        raise ValueError("Stage-1R report requires clean and perturbed source payloads")
    sources: dict[str, dict[str, Any]] = {}
    for condition in ("clean", "perturbed"):
        reference = references[condition]
        if not isinstance(reference, Mapping):
            raise ValueError(f"Stage-1R {condition} source reference is malformed")
        source_path = Path(str(reference.get("path", ""))).expanduser().resolve(
            strict=True
        )
        recorded_sha = reference.get("content_sha256")
        if not _is_sha256(recorded_sha) or recorded_sha != sha256_path(source_path):
            raise ValueError(f"Stage-1R {condition} source payload hash is stale")
        source = _load_mapping(source_path, f"Stage-1R {condition} source payload")
        _validate_source_payload(
            source,
            condition=condition,
            expected_checkpoint_identity=expected_identity,
            expected_contract=expected_contract,
        )
        sources[condition] = source

    # Lazy import avoids a package cycle: the evaluator uses this module to
    # construct the artifact, while validation recomputes its numerical core.
    from fullbody.eval_finger_robustness import compare_finger_robustness

    recomputed = compare_finger_robustness(
        sources["clean"], sources["perturbed"]
    )
    for key in _CORE_REPORT_KEYS:
        if key not in report or report[key] != recomputed[key]:
            raise ValueError(
                f"Stage-1R report field {key!r} differs from bound source payloads"
            )
    if report.get("passed") is not True:
        raise ValueError("Stage-1R paired robustness report did not pass")
    return report


def _validate_source_payload(
    source: Mapping[str, Any],
    *,
    condition: str,
    expected_checkpoint_identity: Mapping[str, Any],
    expected_contract: Mapping[str, Any],
) -> None:
    if source.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise ValueError(f"Stage-1R {condition} source schema is incompatible")
    if source.get("evidence_kind") != VERIFIED_EVIDENCE_KIND:
        raise ValueError(f"Stage-1R {condition} source is not verified rollout evidence")
    if source.get("condition") != condition:
        raise ValueError(f"Stage-1R {condition} source condition is mismatched")
    _require_equal_mapping(
        source.get("checkpoint_identity"),
        expected_checkpoint_identity,
        f"Stage-1R {condition} checkpoint identity",
    )
    _require_equal_mapping(
        source.get("evaluation_contract"),
        expected_contract,
        f"Stage-1R {condition} evaluation contract",
    )
    if source.get("seeds") != list(CANONICAL_SEEDS):
        raise ValueError(f"Stage-1R {condition} source seeds are not canonical")
    rollouts = source.get("rollouts")
    if not isinstance(rollouts, list) or len(rollouts) != len(CANONICAL_SEEDS):
        raise ValueError(f"Stage-1R {condition} source requires exactly five rollouts")
    if not all(isinstance(row, Mapping) for row in rollouts):
        raise ValueError(f"Stage-1R {condition} rollout rows are malformed")
    metrics = source.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"Stage-1R {condition} source metrics are missing")
    for key in _SOURCE_METRIC_KEYS:
        values = metrics.get(key)
        if not isinstance(values, list) or len(values) != len(CANONICAL_SEEDS):
            raise ValueError(
                f"Stage-1R {condition} source metric {key!r} requires five values"
            )
        try:
            finite = all(math.isfinite(float(value)) for value in values)
        except (TypeError, ValueError):
            finite = False
        if not finite:
            raise ValueError(
                f"Stage-1R {condition} source metric {key!r} is non-finite"
            )
        source_key = _ROLLOUT_SOURCE_KEYS[key]
        try:
            row_values = [float(row[source_key]) for row in rollouts]
            metric_values = [float(value) for value in values]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Stage-1R {condition} rollout rows lack metric {source_key!r}"
            ) from exc
        if row_values != metric_values:
            raise ValueError(
                f"Stage-1R {condition} metric {key!r} differs from rollout rows"
            )
    provenance = source.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"Stage-1R {condition} provenance is missing")
    expected_qpos = (
        expected_contract["clean_finger_qpos_perturb_scale"]
        if condition == "clean"
        else expected_contract["perturbed_finger_qpos_perturb_scale"]
    )
    expected_qvel = (
        expected_contract["clean_finger_qvel_perturb_scale"]
        if condition == "clean"
        else expected_contract["perturbed_finger_qvel_perturb_scale"]
    )
    expected_provenance = {
        "checkpoint": expected_checkpoint_identity["checkpoint_path"],
        "checkpoint_identity": dict(expected_checkpoint_identity),
        "motion_paths": expected_contract["heldout_motion_identity"]["motion_paths"],
        "heldout_motion_identity": expected_contract["heldout_motion_identity"],
        "deterministic_policy": True,
        "evaluate_all": True,
        "finger_perturb_rng_mode": "fold_in",
        "finger_perturb_side": "right",
        "finger_qpos_perturb_scale": expected_qpos,
        "finger_qvel_perturb_scale": expected_qvel,
        "metrics_envs": expected_contract["metrics_envs"],
        "metrics_steps": expected_contract["metrics_steps"],
    }
    _require_equal_mapping(
        provenance,
        expected_provenance,
        f"Stage-1R {condition} provenance",
    )


def _require_equal_mapping(actual: Any, expected: Mapping[str, Any], label: str) -> None:
    if not isinstance(actual, Mapping) or dict(actual) != dict(expected):
        raise ValueError(f"{label} is missing, stale, or belongs to another run")


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return dict(payload)


def _finite_float(label: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True

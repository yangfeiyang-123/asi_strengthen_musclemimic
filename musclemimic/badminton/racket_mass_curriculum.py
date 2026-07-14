"""Versioned racket-mass curriculum contracts for Stage-2 v2.

The legacy Stage-2 config remains untouched.  This module only describes and
validates an opt-in 0/25/50/75/100 percent continuation chain; callers decide
where checkpoints and generated plan artifacts are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "racket_mass_curriculum_v2"
PHYSICS_SCHEMA_VERSION = "racket_mass_physics_manifest_v1"
PROMOTION_SCHEMA_VERSION = "racket_mass_promoted_artifact_v1"
MASS_STAGE_SCALES = {
    "mass_025": 0.25,
    "mass_050": 0.50,
    "mass_075": 0.75,
    "mass_100": 1.00,
}


@dataclass(frozen=True)
class RacketMassStage:
    name: str
    mass_scale: float
    config_name: str
    parent_stage: str | None
    promotion_gate: str


def canonical_racket_mass_stages() -> tuple[RacketMassStage, ...]:
    """Return the immutable v2 stage order without touching runtime state."""

    base = "config_specific_task/stage2_racket_v2"
    return (
        RacketMassStage(
            name="mass_000_body_baseline",
            mass_scale=0.0,
            config_name=("config_specific_task/stage1_body/conf_fullbody_forehand_clear_body_finger_isolated_005"),
            parent_stage=None,
            promotion_gate="stage1r",
        ),
        RacketMassStage(
            name="mass_025",
            mass_scale=0.25,
            config_name=f"{base}/conf_fullbody_forehand_clear_racket_mass_025",
            parent_stage="mass_000_body_baseline",
            promotion_gate="stage2",
        ),
        RacketMassStage(
            name="mass_050",
            mass_scale=0.50,
            config_name=f"{base}/conf_fullbody_forehand_clear_racket_mass_050",
            parent_stage="mass_025",
            promotion_gate="stage2",
        ),
        RacketMassStage(
            name="mass_075",
            mass_scale=0.75,
            config_name=f"{base}/conf_fullbody_forehand_clear_racket_mass_075",
            parent_stage="mass_050",
            promotion_gate="stage2",
        ),
        RacketMassStage(
            name="mass_100",
            mass_scale=1.00,
            config_name=f"{base}/conf_fullbody_forehand_clear_racket_mass_100",
            parent_stage="mass_075",
            promotion_gate="stage2",
        ),
    )


def curriculum_plan() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "legacy_stage2_unchanged": True,
        "checkpoint_namespace": "stage2_racket_mass_v2",
        "stages": [asdict(stage) for stage in canonical_racket_mass_stages()],
    }
    payload["plan_sha256"] = _fingerprint(payload)
    return payload


def validate_curriculum_plan(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("racket mass curriculum schema mismatch")
    recorded = payload.get("plan_sha256")
    unbound = dict(payload)
    unbound.pop("plan_sha256", None)
    if recorded != _fingerprint(unbound):
        raise ValueError("racket mass curriculum fingerprint mismatch")
    expected = [asdict(stage) for stage in canonical_racket_mass_stages()]
    if payload.get("stages") != expected:
        raise ValueError("racket mass curriculum stage order/configuration drifted")
    return payload


def write_curriculum_plan(path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(curriculum_plan(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def build_racket_physics_manifest(stage: str) -> dict[str, Any]:
    """Materialize the actual rigid-racket model properties for one load rung.

    Importing and compiling MuJoCo is intentionally lazy: merely importing the
    pipeline or constructing its command plan remains source-only and cannot
    interfere with a running trainer.
    """

    stage_name, mass_scale = _mass_stage(stage)
    import mujoco

    from musclemimic.environments.humanoids.myofullbody_racket import (
        DEFAULT_RACKET_ATTACH_BODY,
        DEFAULT_RACKET_GRIP_POS,
        DEFAULT_RACKET_GRIP_QUAT,
        DEFAULT_RACKET_XML,
        RACKET_BODY_NAME,
        MyoFullBodyRacket,
    )

    environment = MyoFullBodyRacket(
        disable_fingers=True,
        racket_mass_scale=mass_scale,
    )
    model = environment._model
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, RACKET_BODY_NAME)
    if body_id < 0:
        raise ValueError(f"compiled model has no racket body {RACKET_BODY_NAME!r}")
    rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, np.asarray(model.body_iquat[body_id], dtype=float))
    rotation = rotation.reshape(3, 3)
    inertia_tensor = rotation @ np.diag(model.body_inertia[body_id]) @ rotation.T
    weld_ids = [
        index
        for index in range(int(model.neq))
        if int(model.eq_type[index]) == int(mujoco.mjtEq.mjEQ_WELD)
        and body_id in {int(model.eq_obj1id[index]), int(model.eq_obj2id[index])}
    ]
    if weld_ids:
        weld_parameters: dict[str, Any] = {
            "kind": "mujoco_equality_weld",
            "count": len(weld_ids),
            "solref": [model.eq_solref[index].tolist() for index in weld_ids],
            "solimp": [model.eq_solimp[index].tolist() for index in weld_ids],
        }
    else:
        # Stage-2 uses a jointless child-body attachment.  Recording the
        # absence of an equality weld is meaningful physics provenance, not a
        # missing value.
        weld_parameters = {
            "kind": "jointless_spec_attach",
            "count": 0,
            "solref": [],
            "solimp": [],
        }
    asset_path = Path(DEFAULT_RACKET_XML).resolve(strict=True)
    payload: dict[str, Any] = {
        "schema_version": PHYSICS_SCHEMA_VERSION,
        "stage": stage_name,
        "mass_scale": mass_scale,
        "racket_body_name": RACKET_BODY_NAME,
        "racket_mass_kg": float(model.body_mass[body_id]),
        "racket_center_of_mass_m": np.asarray(model.body_ipos[body_id], dtype=float).tolist(),
        "racket_inertia_tensor_kg_m2": inertia_tensor.tolist(),
        "attachment_transform": {
            "parent_body": DEFAULT_RACKET_ATTACH_BODY,
            "translation_m": [float(value) for value in DEFAULT_RACKET_GRIP_POS],
            "quaternion_wxyz": [float(value) for value in DEFAULT_RACKET_GRIP_QUAT],
            "joint_count": 0,
        },
        "weld_parameters": weld_parameters,
        "racket_asset": {
            "path": str(asset_path),
            "sha256": _sha256_file(asset_path),
        },
        "compiled_model_sha256": _compiled_model_sha256(model),
    }
    payload["manifest_sha256"] = _fingerprint(payload)
    return payload


def validate_racket_physics_manifest(
    manifest: str | Path | Mapping[str, Any],
    *,
    expected_stage: str | None = None,
    verify_compiled_model: bool = True,
) -> dict[str, Any]:
    payload = _load_mapping(manifest, "racket physics manifest")
    if payload.get("schema_version") != PHYSICS_SCHEMA_VERSION:
        raise ValueError("racket physics manifest schema mismatch")
    recorded = payload.get("manifest_sha256")
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    if recorded != _fingerprint(unsigned):
        raise ValueError("racket physics manifest fingerprint mismatch")
    stage_name, expected_scale = _mass_stage(
        expected_stage if expected_stage is not None else str(payload.get("stage", ""))
    )
    if payload.get("stage") != stage_name or not math.isclose(
        float(payload.get("mass_scale", -1.0)), expected_scale, abs_tol=1e-12
    ):
        raise ValueError("racket physics manifest stage/scale mismatch")
    mass = _finite_positive("racket_mass_kg", payload.get("racket_mass_kg"))
    center = _finite_array(
        "racket_center_of_mass_m",
        payload.get("racket_center_of_mass_m"),
        shape=(3,),
    )
    inertia = _finite_array(
        "racket_inertia_tensor_kg_m2",
        payload.get("racket_inertia_tensor_kg_m2"),
        shape=(3, 3),
    )
    if not np.allclose(inertia, inertia.T, rtol=0.0, atol=1e-12):
        raise ValueError("racket inertia tensor must be symmetric")
    if float(np.min(np.linalg.eigvalsh(inertia))) <= 0.0:
        raise ValueError("racket inertia tensor must be positive definite")
    if mass <= 0.0 or center.shape != (3,):  # shape branch documents invariants
        raise ValueError("racket physics values are invalid")
    attachment = payload.get("attachment_transform")
    if not isinstance(attachment, Mapping):
        raise ValueError("racket attachment transform is missing")
    _finite_array("attachment translation", attachment.get("translation_m"), shape=(3,))
    quaternion = _finite_array("attachment quaternion", attachment.get("quaternion_wxyz"), shape=(4,))
    if not math.isclose(float(np.linalg.norm(quaternion)), 1.0, abs_tol=1e-5):
        raise ValueError("attachment quaternion is not normalized")
    if int(attachment.get("joint_count", -1)) != 0:
        raise ValueError("racket mass curriculum requires a jointless attachment")
    weld = payload.get("weld_parameters")
    if not isinstance(weld, Mapping) or weld.get("kind") != "jointless_spec_attach":
        raise ValueError("Stage-2 physics manifest must bind the rigid attachment mode")
    if int(weld.get("count", -1)) != 0:
        raise ValueError("jointless Stage-2 racket must not add an equality weld")
    for key in ("compiled_model_sha256",):
        if not _is_sha256(payload.get(key)):
            raise ValueError(f"racket physics manifest has invalid {key}")
    asset = payload.get("racket_asset")
    if not isinstance(asset, Mapping) or not _is_sha256(asset.get("sha256")):
        raise ValueError("racket physics manifest has invalid asset identity")
    asset_path = Path(str(asset.get("path", ""))).expanduser().resolve(strict=True)
    if _sha256_file(asset_path) != asset.get("sha256"):
        raise ValueError("racket XML asset changed after physics manifest creation")
    if verify_compiled_model:
        rebuilt = build_racket_physics_manifest(stage_name)
        if rebuilt != payload:
            raise ValueError("compiled racket model differs from physics manifest")
    return payload


def write_racket_physics_manifest(path: str | Path, *, stage: str) -> Path:
    return _atomic_write_json(path, build_racket_physics_manifest(stage))


def launch_mass_training(
    *,
    stage: str,
    physics_manifest: str | Path,
    resume_from: str | Path,
    baseline_metrics: str | Path,
    train_event_bank: str | Path,
    val_event_bank: str | Path,
) -> None:
    """Launch one explicit rung after sealing physics into the resolved config."""

    stage_name, _ = _mass_stage(stage)
    physics_path = Path(physics_manifest).expanduser().resolve(strict=True)
    validate_racket_physics_manifest(
        physics_path,
        expected_stage=stage_name,
        verify_compiled_model=True,
    )
    scale_code = stage_name.removeprefix("mass_")
    previous_role = (
        "stage1r_005_promoted" if stage_name == "mass_025" else f"racket_{_previous_mass_stage(stage_name)}_promoted"
    )
    resolved_baseline = Path(baseline_metrics).expanduser().resolve(strict=True)
    if stage_name == "mass_025":
        from musclemimic.badminton.stage1r_artifact import validate_stage1r_report

        stage1r = validate_stage1r_report(
            resolved_baseline,
            expected_checkpoint=resume_from,
            expected_perturb_qpos_scale=0.05,
        )
        clean_source = stage1r.get("source_payloads", {}).get("clean", {})
        resolved_baseline = Path(str(clean_source.get("path", ""))).resolve(strict=True)
        if _sha256_file(resolved_baseline) != clean_source.get("content_sha256"):
            raise ValueError("Stage-1R clean baseline changed after report creation")
    command = (
        sys.executable,
        "fullbody/experiment.py",
        f"--config-name=config_specific_task/stage2_racket_v2/conf_fullbody_forehand_clear_racket_mass_{scale_code}",
        f"experiment.resume_from={resume_from}",
        f"experiment.parent_checkpoint_lineage.role={previous_role}",
        f"experiment.promotion.baseline_metrics_path={resolved_baseline}",
        f"experiment.contact_tracking.event_reference_bank_manifest={train_event_bank}",
        f"experiment.contact_tracking.validation_event_reference_bank_manifest={val_event_bank}",
        f"+experiment.racket_mass_stage={stage_name}",
        "+experiment.racket_mass_physics_manifest_path=" + str(physics_path),
        "+experiment.racket_mass_physics_manifest_sha256=" + _sha256_file(physics_path),
    )
    subprocess.run(command, check=True)


def build_mass_promoted_artifact(
    *,
    stage: str,
    checkpoint: str | Path,
    promotion_progress: str | Path,
    visual_review: str | Path,
    physics_manifest: str | Path,
    parent_checkpoint: str | Path | None = None,
    parent_stage1r_report: str | Path | None = None,
    parent_mass_promotion: str | Path | None = None,
) -> dict[str, Any]:
    """Bind one mass-rung checkpoint to its exact parent and physical model."""

    from musclemimic.badminton.promotion_artifact import (
        PROMOTION_PROGRESS_SCHEMA_VERSION,
        checkpoint_identity,
        sha256_path,
    )
    from musclemimic.badminton.visual_review import (
        STAGE2_REVIEW_KIND,
        validate_visual_review,
    )

    stage_name, scale = _mass_stage(stage)
    identity = checkpoint_identity(checkpoint)
    progress_path = Path(promotion_progress).expanduser().resolve(strict=True)
    review_path = Path(visual_review).expanduser().resolve(strict=True)
    physics_path = Path(physics_manifest).expanduser().resolve(strict=True)
    progress = _load_mapping(progress_path, "mass promotion progress")
    review = _load_mapping(review_path, "mass visual review")
    physics = validate_racket_physics_manifest(physics_path, expected_stage=stage_name, verify_compiled_model=True)
    _validate_checkpoint_physics_binding(
        identity,
        physics_path=physics_path,
        physics=physics,
    )
    if progress.get("schema_version") != PROMOTION_PROGRESS_SCHEMA_VERSION:
        raise ValueError("mass promotion progress schema is incompatible")
    if str(progress.get("stage", "")).lower() != "stage2":
        raise ValueError("mass promotion progress must use Stage-2 gates")
    if Path(str(progress.get("checkpoint_dir", ""))).expanduser().resolve() != Path(identity["checkpoint_dir"]):
        raise ValueError("mass promotion progress belongs to a different checkpoint run")
    if progress.get("config_hash") != identity["config_hash"]:
        raise ValueError("mass promotion progress config hash differs from checkpoint")
    history = progress.get("history")
    if not isinstance(history, list) or not history or not isinstance(history[-1], Mapping):
        raise ValueError("mass promotion progress requires non-empty history")
    tail = history[-1]
    if tail.get("passed") is not True or progress.get("stopped_early") is not True:
        raise ValueError("mass promotion numerical gate did not complete")
    _require_checkpoint_candidate(tail, identity, label="mass promotion progress")
    recorded_identity = tail.get("checkpoint_identity")
    if not isinstance(recorded_identity, Mapping):
        raise ValueError("mass promotion progress has no checkpoint identity")
    _require_checkpoint_identity(recorded_identity, identity, label="mass progress")
    review_report = validate_visual_review(
        review,
        required_clips=5,
        required_review_kind=STAGE2_REVIEW_KIND,
        expected_candidate=identity,
    )
    if review_report["passed"] is not True:
        raise ValueError("mass-rung structured visual review did not pass")

    baseline_value = progress.get("baseline_metrics_path")
    baseline_sha256 = progress.get("baseline_metrics_sha256")
    if not isinstance(baseline_value, str) or not baseline_value:
        raise ValueError("mass promotion has no parent-rung baseline path")
    baseline_path = Path(baseline_value).expanduser().resolve(strict=True)
    if baseline_sha256 != sha256_path(baseline_path):
        raise ValueError("mass promotion baseline content identity is stale")

    if stage_name == "mass_025":
        if parent_checkpoint is None or parent_stage1r_report is None:
            raise ValueError("mass_025 requires Stage-1R checkpoint/report parent")
        if parent_mass_promotion is not None:
            raise ValueError("mass_025 cannot declare a mass-rung parent")
        from musclemimic.badminton.stage1r_artifact import validate_stage1r_report

        parent_identity = checkpoint_identity(parent_checkpoint)
        parent_path = Path(parent_stage1r_report).expanduser().resolve(strict=True)
        parent_report = validate_stage1r_report(
            parent_path,
            expected_checkpoint=parent_checkpoint,
            expected_perturb_qpos_scale=0.05,
        )
        clean_source = parent_report.get("source_payloads", {}).get("clean", {})
        clean_path = Path(str(clean_source.get("path", ""))).resolve(strict=True)
        if (
            baseline_path != clean_path
            or baseline_sha256 != clean_source.get("content_sha256")
            or baseline_sha256 != sha256_path(clean_path)
        ):
            raise ValueError("mass_025 baseline is not the clean rollout bound by its Stage-1R report")
        parent_binding: dict[str, Any] = {
            "kind": "stage1r_005",
            "path": str(parent_path),
            "content_sha256": sha256_path(parent_path),
            "binding_sha256": parent_report.get("binding_sha256"),
            "checkpoint": parent_identity,
        }
        previous_physics = None
    else:
        if parent_mass_promotion is None or parent_checkpoint is not None:
            raise ValueError(f"{stage_name} requires only its previous mass promotion")
        parent_path = Path(parent_mass_promotion).expanduser().resolve(strict=True)
        parent_stage = _previous_mass_stage(stage_name)
        parent = validate_mass_promoted_artifact(parent_path, expected_stage=parent_stage)
        parent_identity = parent["checkpoint"]
        parent_progress = parent["promotion_progress"]
        if baseline_path != Path(parent_progress["path"]).resolve(strict=True):
            raise ValueError("mass-rung baseline is not its promoted parent progress")
        if baseline_sha256 != parent_progress["content_sha256"]:
            raise ValueError("mass-rung baseline differs from promoted parent progress")
        parent_binding = {
            "kind": "racket_mass",
            "stage": parent_stage,
            "path": str(parent_path),
            "content_sha256": sha256_path(parent_path),
            "binding_sha256": parent.get("binding_sha256"),
            "checkpoint": parent_identity,
        }
        previous_physics = parent["physics_manifest"]
    expected_parent_role = (
        "stage1r_005_promoted" if stage_name == "mass_025" else f"racket_{_previous_mass_stage(stage_name)}_promoted"
    )
    if not _direct_lineage_matches(
        identity.get("parent_checkpoint_lineage"),
        parent_identity,
        expected_role=expected_parent_role,
    ):
        raise ValueError("mass-rung checkpoint does not directly resume from its promoted parent")
    if previous_physics is not None:
        _validate_physics_transition(previous_physics, physics)

    payload: dict[str, Any] = {
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "stage": stage_name,
        "mass_scale": scale,
        "checkpoint": identity,
        "parent": parent_binding,
        "promotion_progress": {
            "path": str(progress_path),
            "content_sha256": sha256_path(progress_path),
            "validation_count": int(progress.get("validation_count", len(history))),
            "consecutive_pass_streak": int(progress.get("consecutive_pass_streak", 0)),
            "tail": dict(tail),
        },
        "visual_review": {
            "path": str(review_path),
            "content_sha256": sha256_path(review_path),
            "candidate": dict(review.get("candidate") or {}),
        },
        "physics_manifest": {
            **physics,
            "path": str(physics_path),
            "content_sha256": sha256_path(physics_path),
        },
    }
    payload["binding_sha256"] = _fingerprint(payload)
    return payload


def validate_mass_promoted_artifact(
    manifest: str | Path,
    *,
    expected_stage: str,
    expected_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(manifest).expanduser().resolve(strict=True)
    payload = _load_mapping(path, "mass promoted artifact")
    stage_name, _ = _mass_stage(expected_stage)
    if payload.get("schema_version") != PROMOTION_SCHEMA_VERSION:
        raise ValueError("mass promoted artifact schema mismatch")
    if payload.get("stage") != stage_name:
        raise ValueError("mass promoted artifact belongs to a different rung")
    checkpoint = payload.get("checkpoint")
    parent = payload.get("parent")
    if not isinstance(checkpoint, Mapping) or not isinstance(parent, Mapping):
        raise ValueError("mass promoted artifact is missing checkpoint/parent identity")
    checkpoint_path = Path(str(checkpoint.get("checkpoint_path", "")))
    if expected_checkpoint is not None and checkpoint_path.resolve(strict=True) != Path(
        expected_checkpoint
    ).expanduser().resolve(strict=True):
        raise ValueError("mass promoted artifact points to a different checkpoint")
    rebuilt = build_mass_promoted_artifact(
        stage=stage_name,
        checkpoint=checkpoint_path,
        promotion_progress=str(payload.get("promotion_progress", {}).get("path", "")),
        visual_review=str(payload.get("visual_review", {}).get("path", "")),
        physics_manifest=str(payload.get("physics_manifest", {}).get("path", "")),
        parent_checkpoint=(
            str(parent.get("checkpoint", {}).get("checkpoint_path", ""))
            if parent.get("kind") == "stage1r_005"
            else None
        ),
        parent_stage1r_report=(str(parent.get("path", "")) if parent.get("kind") == "stage1r_005" else None),
        parent_mass_promotion=(str(parent.get("path", "")) if parent.get("kind") == "racket_mass" else None),
    )
    if rebuilt != payload:
        raise ValueError("mass promoted artifact or one of its sources changed")
    return rebuilt


def write_mass_promoted_artifact(path: str | Path, payload: Mapping[str, Any]) -> Path:
    return _atomic_write_json(path, payload)


def _validate_physics_transition(previous: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    previous_scale = float(previous["mass_scale"])
    current_scale = float(current["mass_scale"])
    if current_scale <= previous_scale:
        raise ValueError("racket mass promotion scale must increase monotonically")
    for key in ("racket_center_of_mass_m", "attachment_transform", "weld_parameters"):
        if previous.get(key) != current.get(key):
            raise ValueError(f"racket physics {key} changed across a mass rung")
    ratio = current_scale / previous_scale
    if not math.isclose(
        float(current["racket_mass_kg"]),
        float(previous["racket_mass_kg"]) * ratio,
        rel_tol=1e-8,
        abs_tol=1e-12,
    ):
        raise ValueError("racket mass did not scale linearly across curriculum")
    previous_inertia = np.asarray(previous["racket_inertia_tensor_kg_m2"], dtype=float)
    current_inertia = np.asarray(current["racket_inertia_tensor_kg_m2"], dtype=float)
    if not np.allclose(current_inertia, previous_inertia * ratio, rtol=1e-8, atol=1e-12):
        raise ValueError("racket inertia did not scale linearly across curriculum")


def _validate_checkpoint_physics_binding(
    checkpoint_identity_payload: Mapping[str, Any],
    *,
    physics_path: Path,
    physics: Mapping[str, Any],
) -> None:
    manifest_path = Path(str(checkpoint_identity_payload.get("checkpoint_dir", ""))) / "manifest.json"
    run_manifest = _load_mapping(manifest_path, "mass-rung checkpoint run manifest")
    experiment = run_manifest.get("experiment_config")
    if not isinstance(experiment, Mapping):
        raise ValueError("mass-rung checkpoint has no resolved experiment config")
    env_params = experiment.get("env_params")
    if not isinstance(env_params, Mapping) or not math.isclose(
        float(env_params.get("racket_mass_scale", -1.0)),
        float(physics["mass_scale"]),
        abs_tol=1e-12,
    ):
        raise ValueError("checkpoint racket_mass_scale differs from physics manifest")
    if experiment.get("racket_mass_stage") != physics.get("stage"):
        raise ValueError("checkpoint mass stage is not bound to its physics manifest")
    recorded_path = experiment.get("racket_mass_physics_manifest_path")
    try:
        same_path = Path(str(recorded_path)).expanduser().resolve(strict=True) == (physics_path.resolve(strict=True))
    except (OSError, RuntimeError):
        same_path = False
    if not same_path:
        raise ValueError("checkpoint records a different physics manifest path")
    if experiment.get("racket_mass_physics_manifest_sha256") != _sha256_file(physics_path):
        raise ValueError("checkpoint physics manifest content hash is stale")


def _mass_stage(stage: str) -> tuple[str, float]:
    stage_name = str(stage).strip().lower()
    if stage_name not in MASS_STAGE_SCALES:
        raise ValueError(f"unsupported racket mass stage: {stage!r}")
    return stage_name, MASS_STAGE_SCALES[stage_name]


def _previous_mass_stage(stage: str) -> str:
    order = tuple(MASS_STAGE_SCALES)
    index = order.index(stage)
    if index <= 0:
        raise ValueError("mass_025 has no previous mass rung")
    return order[index - 1]


def _load_mapping(source: str | Path | Mapping[str, Any], label: str) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return dict(payload)


def _atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def _finite_positive(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _finite_array(name: str, value: Any, *, shape: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite array with shape {shape}") from exc
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    return array


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compiled_model_sha256(model: Any) -> str:
    digest = hashlib.sha256(b"mujoco-compiled-model-v1\0")
    for scalar in (model.nq, model.nv, model.nu, model.nbody, model.ngeom, model.neq):
        digest.update(int(scalar).to_bytes(8, "big", signed=False))
    for name in (
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_iquat",
        "body_parentid",
        "geom_bodyid",
        "geom_pos",
        "geom_quat",
        "geom_size",
        "geom_type",
        "eq_type",
        "eq_obj1id",
        "eq_obj2id",
        "eq_solref",
        "eq_solimp",
    ):
        value = np.ascontiguousarray(np.asarray(getattr(model, name)))
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(value.shape).encode("ascii") + b"\0")
        digest.update(value.tobytes())
    return digest.hexdigest()


def _require_checkpoint_candidate(candidate: Mapping[str, Any], identity: Mapping[str, Any], *, label: str) -> None:
    for key in ("update_number", "global_timestep"):
        try:
            matches = int(candidate[key]) == int(identity[key])
        except (KeyError, TypeError, ValueError):
            matches = False
        if not matches:
            raise ValueError(f"{label} {key} differs from checkpoint")


def _require_checkpoint_identity(actual: Mapping[str, Any], expected: Mapping[str, Any], *, label: str) -> None:
    keys = (
        "checkpoint_path",
        "checkpoint_dir",
        "checkpoint_content_sha256",
        "metadata_content_sha256",
        "run_manifest_content_sha256",
        "update_number",
        "global_timestep",
        "config_hash",
        "run_id",
    )
    if any(actual.get(key) != expected.get(key) for key in keys):
        raise ValueError(f"{label} checkpoint identity differs")
    if actual.get("parent_checkpoint_lineage") != expected.get("parent_checkpoint_lineage"):
        raise ValueError(f"{label} parent checkpoint lineage differs")


def _direct_lineage_matches(
    lineage: Any,
    expected_checkpoint: Mapping[str, Any],
    *,
    expected_role: str,
) -> bool:
    if not isinstance(lineage, Mapping):
        return False
    keys = (
        "checkpoint_content_sha256",
        "metadata_content_sha256",
        "run_manifest_content_sha256",
        "update_number",
        "global_timestep",
        "target_global_timestep",
        "config_hash",
        "run_id",
    )
    current = lineage.get("checkpoint")
    return (
        lineage.get("role") == expected_role
        and isinstance(current, Mapping)
        and all(current.get(key) == expected_checkpoint.get(key) for key in keys)
    )


def _fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--physics-stage", choices=tuple(MASS_STAGE_SCALES))
    parser.add_argument("--promote-stage", choices=tuple(MASS_STAGE_SCALES))
    parser.add_argument("--launch-stage", choices=tuple(MASS_STAGE_SCALES))
    parser.add_argument("--checkpoint")
    parser.add_argument("--promotion-progress")
    parser.add_argument("--visual-review")
    parser.add_argument("--physics-manifest")
    parser.add_argument("--parent-checkpoint")
    parser.add_argument("--parent-stage1r-report")
    parser.add_argument("--parent-mass-promotion")
    parser.add_argument("--resume-from")
    parser.add_argument("--baseline-metrics")
    parser.add_argument("--train-event-bank")
    parser.add_argument("--val-event-bank")
    args = parser.parse_args()
    selected = sum(value is not None for value in (args.physics_stage, args.promote_stage, args.launch_stage))
    if selected > 1:
        parser.error("--physics-stage, --promote-stage and --launch-stage are mutually exclusive")
    if args.physics_stage is not None:
        if args.output is None:
            parser.error("--physics-stage requires --output")
        print(write_racket_physics_manifest(args.output, stage=args.physics_stage))
        return 0
    if args.launch_stage is not None:
        required = {
            "physics_manifest": args.physics_manifest,
            "resume_from": args.resume_from,
            "baseline_metrics": args.baseline_metrics,
            "train_event_bank": args.train_event_bank,
            "val_event_bank": args.val_event_bank,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(f"--launch-stage requires: {', '.join(missing)}")
        launch_mass_training(
            stage=args.launch_stage,
            physics_manifest=args.physics_manifest,
            resume_from=args.resume_from,
            baseline_metrics=args.baseline_metrics,
            train_event_bank=args.train_event_bank,
            val_event_bank=args.val_event_bank,
        )
        return 0
    if args.promote_stage is not None:
        required = {
            "checkpoint": args.checkpoint,
            "promotion_progress": args.promotion_progress,
            "visual_review": args.visual_review,
            "physics_manifest": args.physics_manifest,
            "output": args.output,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(f"--promote-stage requires: {', '.join(missing)}")
        payload = build_mass_promoted_artifact(
            stage=args.promote_stage,
            checkpoint=args.checkpoint,
            promotion_progress=args.promotion_progress,
            visual_review=args.visual_review,
            physics_manifest=args.physics_manifest,
            parent_checkpoint=args.parent_checkpoint,
            parent_stage1r_report=args.parent_stage1r_report,
            parent_mass_promotion=args.parent_mass_promotion,
        )
        print(write_mass_promoted_artifact(args.output, payload))
        return 0
    extra = (
        args.checkpoint,
        args.promotion_progress,
        args.visual_review,
        args.physics_manifest,
        args.parent_checkpoint,
        args.parent_stage1r_report,
        args.parent_mass_promotion,
    )
    if any(value is not None for value in extra) or any(
        value is not None
        for value in (
            args.resume_from,
            args.baseline_metrics,
            args.train_event_bank,
            args.val_event_bank,
        )
    ):
        parser.error("promotion arguments require --promote-stage")
    if args.output is None:
        parser.error("curriculum plan generation requires --output")
    print(write_curriculum_plan(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

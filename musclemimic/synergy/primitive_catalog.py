"""Strict catalog and phase contracts for primitive-synergy ingestion.

The catalog is intentionally a *source description*, not scientific evidence by
itself.  Paths in it are resolved relative to the catalog file, while model,
controller, raw-trial, and phase-schema contents are fingerprinted again by the
ingest builder.  Consequently a path can never bless changed bytes silently.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from musclemimic.badminton.json_contract import load_json_strict
from musclemimic.distill.motion_identity import (
    normalize_relative_motion_path,
    stable_motion_uid,
)

PRIMITIVE_CATALOG_SCHEMA_VERSION = "primitive_synergy_catalog_v1"
PRIMITIVE_PHASE_SCHEMA_VERSION = "primitive_phase_schema_v1"

_CATALOG_FIELDS = {
    "schema_version",
    "catalog_id",
    "target_skill_id",
    "model_xml_path",
    "expected_action_dim",
    "regional_grouping_path",
    "tasks",
}
_TASK_FIELDS = {
    "task_id",
    "display_name",
    "enabled",
    "controller_artifact",
    "phase_schema_path",
    "trials",
}
_TRIAL_FIELDS = {
    "trial_id",
    "split",
    "motion_path",
    "raw_npz_path",
    "success",
    "quality_weight",
}
_PHASE_SCHEMA_FIELDS = {"schema_version", "task_id", "phases"}
_PHASE_FIELDS = {"id", "name", "definition"}


@dataclass(frozen=True)
class PrimitivePhaseSchema:
    """One task-specific, content-addressed event-phase definition."""

    path: Path
    payload: dict[str, Any]
    fingerprint: str
    task_id: str
    required_phase_ids: tuple[int, ...]


@dataclass(frozen=True)
class PrimitiveTrialSpec:
    """One complete raw rollout trial declared by the catalog."""

    trial_id: str
    split: str
    motion_path: str
    motion_uid: int
    raw_npz_path: Path
    success: bool
    quality_weight: float


@dataclass(frozen=True)
class PrimitiveTaskSpec:
    """One primitive task, its controller, phases, and independent trials."""

    task_id: str
    display_name: str
    enabled: bool
    controller_artifact: Path | None
    phase_schema: PrimitivePhaseSchema
    trials: tuple[PrimitiveTrialSpec, ...]


@dataclass(frozen=True)
class PrimitiveCatalog:
    """Validated catalog with paths resolved against its source file."""

    path: Path
    payload: dict[str, Any]
    fingerprint: str
    catalog_id: str
    target_skill_id: str
    model_xml_path: Path | None
    expected_action_dim: int
    regional_grouping_path: Path | None
    tasks: tuple[PrimitiveTaskSpec, ...]

    @property
    def enabled_tasks(self) -> tuple[PrimitiveTaskSpec, ...]:
        return tuple(task for task in self.tasks if task.enabled)

    @property
    def model_artifact_path(self) -> Path | None:
        """Return the compiled model declared by the legacy-named JSON field.

        ``model_xml_path`` is retained for catalog schema compatibility.  A
        production MyoFullBody catalog should point it at the exact 354-D
        runtime ``.mjb``; source XML remains supported for simpler models.
        """

        return self.model_xml_path


def canonical_json_sha256(payload: Any) -> str:
    """Return the lowercase SHA-256 of strict canonical JSON content."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_primitive_phase_schema(
    path: str | Path,
    *,
    expected_task_id: str | None = None,
) -> PrimitivePhaseSchema:
    """Load one exact phase schema and derive its identity from current bytes."""

    schema_path = Path(path).expanduser().resolve()
    if not schema_path.is_file():
        raise FileNotFoundError(f"primitive phase schema does not exist: {schema_path}")
    payload = load_json_strict(schema_path)
    if not isinstance(payload, Mapping):
        raise ValueError("primitive phase schema must contain a JSON object")
    _require_exact_fields(payload, _PHASE_SCHEMA_FIELDS, "primitive phase schema")
    if payload.get("schema_version") != PRIMITIVE_PHASE_SCHEMA_VERSION:
        raise ValueError("unsupported primitive phase schema_version")
    task_id = _nonempty_string(payload.get("task_id"), "phase schema task_id")
    if expected_task_id is not None and task_id != str(expected_task_id):
        raise ValueError(
            "primitive phase schema task_id differs from catalog task: "
            f"expected={expected_task_id!r} actual={task_id!r}"
        )
    phases = payload.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("primitive phase schema phases must be a non-empty JSON array")
    phase_ids: list[int] = []
    phase_names: list[str] = []
    canonical_phases: list[dict[str, Any]] = []
    for index, item in enumerate(phases):
        if not isinstance(item, Mapping):
            raise ValueError(f"primitive phase schema phases[{index}] must be an object")
        _require_exact_fields(item, _PHASE_FIELDS, f"primitive phase schema phases[{index}]")
        phase_id = item.get("id")
        if type(phase_id) is not int or phase_id < 0 or phase_id > np.iinfo(np.int32).max:
            raise ValueError("primitive phase ids must be non-negative int32 integers")
        name = _nonempty_string(item.get("name"), f"primitive phase {phase_id} name")
        definition = _nonempty_string(
            item.get("definition"),
            f"primitive phase {phase_id} definition",
        )
        phase_ids.append(phase_id)
        phase_names.append(name)
        canonical_phases.append({"id": phase_id, "name": name, "definition": definition})
    if phase_ids != sorted(phase_ids) or len(phase_ids) != len(set(phase_ids)):
        raise ValueError("primitive phase ids must be sorted and unique")
    if len(phase_names) != len(set(phase_names)):
        raise ValueError("primitive phase names must be unique")
    canonical = {
        "schema_version": PRIMITIVE_PHASE_SCHEMA_VERSION,
        "task_id": task_id,
        "phases": canonical_phases,
    }
    return PrimitivePhaseSchema(
        path=schema_path,
        payload=canonical,
        fingerprint=canonical_json_sha256(canonical),
        task_id=task_id,
        required_phase_ids=tuple(phase_ids),
    )


def load_primitive_catalog(
    path: str | Path,
    *,
    require_build_ready: bool = False,
) -> PrimitiveCatalog:
    """Load a strict catalog; optionally require every production input.

    ``require_build_ready=False`` deliberately permits the checked-in P01--P12
    template to retain empty model/controller/trial slots.  The ingest builder
    always passes ``True`` and therefore fails closed on any placeholder.
    """

    catalog_path = Path(path).expanduser().resolve()
    if not catalog_path.is_file():
        raise FileNotFoundError(f"primitive catalog does not exist: {catalog_path}")
    payload = load_json_strict(catalog_path)
    if not isinstance(payload, Mapping):
        raise ValueError("primitive catalog must contain a JSON object")
    _require_exact_fields(payload, _CATALOG_FIELDS, "primitive catalog")
    if payload.get("schema_version") != PRIMITIVE_CATALOG_SCHEMA_VERSION:
        raise ValueError("unsupported primitive catalog schema_version")
    catalog_id = _nonempty_string(payload.get("catalog_id"), "catalog_id")
    target_skill_id = _nonempty_string(payload.get("target_skill_id"), "target_skill_id")
    expected_action_dim = payload.get("expected_action_dim")
    if type(expected_action_dim) is not int or expected_action_dim <= 0:
        raise ValueError("expected_action_dim must be a positive integer")
    raw_model_path = payload.get("model_xml_path")
    if not isinstance(raw_model_path, str):
        raise ValueError("model_xml_path must be a string")
    model_xml_path = None if not raw_model_path.strip() else _resolve_catalog_path(catalog_path, raw_model_path)
    raw_grouping_path = payload.get("regional_grouping_path")
    if not isinstance(raw_grouping_path, str):
        raise ValueError("regional_grouping_path must be a string")
    regional_grouping_path = (
        None if not raw_grouping_path.strip() else _resolve_catalog_path(catalog_path, raw_grouping_path)
    )

    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("primitive catalog tasks must be a non-empty JSON array")
    tasks: list[PrimitiveTaskSpec] = []
    seen_task_ids: set[str] = set()
    seen_trial_ids: set[str] = set()
    seen_raw_paths: set[Path] = set()
    for index, raw_task in enumerate(raw_tasks):
        if not isinstance(raw_task, Mapping):
            raise ValueError(f"primitive catalog tasks[{index}] must be an object")
        _require_exact_fields(raw_task, _TASK_FIELDS, f"primitive catalog tasks[{index}]")
        task_id = _nonempty_string(raw_task.get("task_id"), f"tasks[{index}].task_id")
        if task_id in seen_task_ids:
            raise ValueError(f"primitive catalog contains duplicate task_id {task_id!r}")
        if task_id.casefold() == target_skill_id.casefold():
            raise ValueError("primitive task_id must not equal target_skill_id")
        seen_task_ids.add(task_id)
        display_name = _nonempty_string(
            raw_task.get("display_name"),
            f"tasks[{index}].display_name",
        )
        enabled = raw_task.get("enabled")
        if type(enabled) is not bool:
            raise ValueError(f"tasks[{index}].enabled must be boolean")
        raw_controller = raw_task.get("controller_artifact")
        if not isinstance(raw_controller, str):
            raise ValueError(f"tasks[{index}].controller_artifact must be a string")
        controller = None if not raw_controller.strip() else _resolve_catalog_path(catalog_path, raw_controller)
        raw_phase_path = _nonempty_string(
            raw_task.get("phase_schema_path"),
            f"tasks[{index}].phase_schema_path",
        )
        phase_schema = load_primitive_phase_schema(
            _resolve_catalog_path(catalog_path, raw_phase_path),
            expected_task_id=task_id,
        )
        raw_trials = raw_task.get("trials")
        if not isinstance(raw_trials, list):
            raise ValueError(f"tasks[{index}].trials must be a JSON array")
        trials: list[PrimitiveTrialSpec] = []
        for trial_index, raw_trial in enumerate(raw_trials):
            label = f"tasks[{index}].trials[{trial_index}]"
            if not isinstance(raw_trial, Mapping):
                raise ValueError(f"{label} must be an object")
            _require_exact_fields(raw_trial, _TRIAL_FIELDS, label)
            trial_id = _nonempty_string(raw_trial.get("trial_id"), f"{label}.trial_id")
            if trial_id in seen_trial_ids:
                raise ValueError(f"primitive catalog contains duplicate trial_id {trial_id!r}")
            seen_trial_ids.add(trial_id)
            split = raw_trial.get("split")
            if split not in {"train", "val"}:
                raise ValueError(f"{label}.split must be 'train' or 'val'")
            motion_path = normalize_relative_motion_path(
                _nonempty_string(raw_trial.get("motion_path"), f"{label}.motion_path")
            )
            if target_skill_id.casefold() in {
                component.casefold() for component in motion_path.replace("\\", "/").split("/") if component
            }:
                raise ValueError(
                    f"{label}.motion_path enters the target-skill namespace "
                    f"{target_skill_id!r}; primitive-only W fitting forbids target rollouts"
                )
            raw_npz_path = _resolve_catalog_path(
                catalog_path,
                _nonempty_string(raw_trial.get("raw_npz_path"), f"{label}.raw_npz_path"),
            )
            if raw_npz_path in seen_raw_paths:
                raise ValueError(f"primitive catalog reuses raw trial NPZ {raw_npz_path}")
            seen_raw_paths.add(raw_npz_path)
            success = raw_trial.get("success")
            if type(success) is not bool:
                raise ValueError(f"{label}.success must be boolean")
            quality_weight = raw_trial.get("quality_weight")
            if (
                isinstance(quality_weight, bool)
                or not isinstance(quality_weight, int | float)
                or not np.isfinite(float(quality_weight))
                or float(quality_weight) <= 0.0
            ):
                raise ValueError(f"{label}.quality_weight must be finite and positive")
            trials.append(
                PrimitiveTrialSpec(
                    trial_id=trial_id,
                    split=str(split),
                    motion_path=motion_path,
                    motion_uid=stable_motion_uid(motion_path),
                    raw_npz_path=raw_npz_path,
                    success=success,
                    quality_weight=float(quality_weight),
                )
            )
        tasks.append(
            PrimitiveTaskSpec(
                task_id=task_id,
                display_name=display_name,
                enabled=enabled,
                controller_artifact=controller,
                phase_schema=phase_schema,
                trials=tuple(trials),
            )
        )

    catalog = PrimitiveCatalog(
        path=catalog_path,
        payload=copy.deepcopy(dict(payload)),
        fingerprint=canonical_json_sha256(payload),
        catalog_id=catalog_id,
        target_skill_id=target_skill_id,
        model_xml_path=model_xml_path,
        expected_action_dim=expected_action_dim,
        regional_grouping_path=regional_grouping_path,
        tasks=tuple(tasks),
    )
    if require_build_ready:
        validate_build_ready_catalog(catalog)
    return catalog


def validate_build_ready_catalog(catalog: PrimitiveCatalog) -> None:
    """Require complete, leakage-free production inputs for ingestion."""

    if catalog.model_xml_path is None:
        raise ValueError(
            "build-ready primitive catalog requires a compiled model artifact "
            "in model_xml_path (.mjb preferred; .xml remains supported)"
        )
    if not catalog.model_xml_path.is_file():
        raise FileNotFoundError(f"primitive catalog compiled model artifact does not exist: {catalog.model_xml_path}")
    if catalog.regional_grouping_path is None:
        raise ValueError("build-ready primitive catalog requires regional_grouping_path")
    if not catalog.regional_grouping_path.is_file():
        raise FileNotFoundError(f"primitive catalog regional grouping does not exist: {catalog.regional_grouping_path}")
    enabled = catalog.enabled_tasks
    if not enabled:
        raise ValueError("build-ready primitive catalog requires at least one enabled task")
    split_motion_uids: dict[str, set[int]] = {"train": set(), "val": set()}
    split_trial_ids: dict[str, set[str]] = {"train": set(), "val": set()}
    for task in enabled:
        if task.controller_artifact is None:
            raise ValueError(f"enabled primitive task {task.task_id!r} requires controller_artifact")
        if not task.controller_artifact.exists():
            raise FileNotFoundError(f"primitive controller artifact does not exist: {task.controller_artifact}")
        by_split = {split: tuple(trial for trial in task.trials if trial.split == split) for split in ("train", "val")}
        if len(by_split["train"]) < 2 or len(by_split["val"]) < 1:
            raise ValueError(
                f"enabled primitive task {task.task_id!r} requires at least two train trials and one validation trial"
            )
        for split, trials in by_split.items():
            for trial in trials:
                if not trial.success:
                    raise ValueError(f"production primitive catalog contains unsuccessful trial {trial.trial_id!r}")
                if not trial.raw_npz_path.is_file():
                    raise FileNotFoundError(f"primitive raw trial NPZ does not exist: {trial.raw_npz_path}")
                split_motion_uids[split].add(trial.motion_uid)
                split_trial_ids[split].add(trial.trial_id)
    motion_overlap = split_motion_uids["train"] & split_motion_uids["val"]
    if motion_overlap:
        raise ValueError(f"primitive train/validation motion leakage detected: {sorted(motion_overlap)}")
    trial_overlap = split_trial_ids["train"] & split_trial_ids["val"]
    if trial_overlap:
        raise ValueError(f"primitive train/validation trial leakage detected: {sorted(trial_overlap)}")


def _resolve_catalog_path(catalog_path: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = catalog_path.parent / candidate
    return candidate.resolve()


def _require_exact_fields(
    payload: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    fields = set(payload)
    missing = sorted(expected - fields)
    unknown = sorted(fields - expected)
    if missing:
        raise ValueError(f"{label} is missing fields: {missing}")
    if unknown:
        raise ValueError(f"{label} has unknown fields: {unknown}")


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument(
        "--require-build-ready",
        action="store_true",
        help="also require concrete model/controller/raw-trial paths and split coverage",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    catalog = load_primitive_catalog(
        args.catalog,
        require_build_ready=bool(args.require_build_ready),
    )
    print(
        json.dumps(
            {
                "catalog": str(catalog.path),
                "catalog_fingerprint": catalog.fingerprint,
                "enabled_task_ids": [task.task_id for task in catalog.enabled_tasks],
                "build_ready_checked": bool(args.require_build_ready),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

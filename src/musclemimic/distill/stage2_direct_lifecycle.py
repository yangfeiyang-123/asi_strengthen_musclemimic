"""Sealed Stage-2 S2-A direct-student lifecycle.

The legacy direct experiment wrapper mixed collection, BC, DAgger, PPO, and
promotion in one nested process.  This module deliberately does not execute
training.  It provides:

* an immutable, seed-specific copy of the shared physical train/val dataset;
* a component plan whose BC, every DAgger retrain, and PPO jobs use the
  canonical fullbody launcher;
* per-seed evidence for BC -> three DAgger rounds -> fresh-optimizer PPO;
* a seed-family promotion over the exact seeds 0/1/2.

The source train/validation dataset is never mutated.  DAgger may append only
to the derived direct dataset rooted below one seed directory.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from musclemimic.badminton.action_registry import resolve
from musclemimic.distill.eval_student import (
    REQUIRED_EVAL_METRICS,
    canonicalize_eval_metrics,
)
from musclemimic.distill.motion_identity import normalize_motion_path, stable_motion_uid
from musclemimic.distill.provenance import (
    canonical_json_sha256,
    checkpoint_content_fingerprint,
    checkpoint_fingerprint_matches,
    file_sha256,
    validate_dataset_manifest,
    validate_direct_acceptance_record,
    validate_stage2_teacher_promotion,
)

DIRECT_DATASET_DERIVATION_SCHEMA_VERSION = "stage2_direct_dataset_derivation_v1"
DIRECT_FAMILY_PLAN_SCHEMA_VERSION = "stage2_direct_family_plan_v1"
DIRECT_SEED_EVIDENCE_SCHEMA_VERSION = "stage2_direct_seed_evidence_v1"
DIRECT_FAMILY_PROMOTION_SCHEMA_VERSION = "stage2_direct_family_promotion_v1"
EXACT_SEEDS = (0, 1, 2)
EXACT_DAGGER_ITERATIONS = 3
SELECTED_DEPLOYMENT_SEED = 0


def _load_object(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON: {source}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{label} must be a lowercase SHA-256 fingerprint")
    return text


def _path_record(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    return {
        "path": str(source),
        "content_sha256": file_sha256(source),
        "num_bytes": int(source.stat().st_size),
    }


def _write_immutable(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"immutable Stage-2 direct artifact already differs: {target}")
        return target
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, target)
    return target


def _split_contract(manifest: Mapping[str, Any], *, split: str) -> dict[str, Any]:
    contracts: list[Mapping[str, Any]] = []
    for item in manifest.get("collections", []):
        if not isinstance(item, Mapping):
            continue
        contract = item.get("contract")
        if not isinstance(contract, Mapping):
            continue
        if str(contract.get("split", "")) == split:
            contracts.append(contract)
    motions: list[str] = []
    collection_ids: list[str] = []
    for contract in contracts:
        collection_ids.append(str(contract.get("collection_id", "")))
        motions.extend(str(value) for value in contract.get("motion_paths", []))
    normalized = [normalize_motion_path(value) for value in motions]
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError(f"S2-A {split} motion split is empty or duplicated")
    return {
        "collection_ids": collection_ids,
        "motion_paths": normalized,
        "motion_uids": [int(stable_motion_uid(value)) for value in normalized],
        "motion_set_fingerprint": canonical_json_sha256(normalized),
    }


def _validate_shared_sources(
    *,
    action: str,
    shared_inputs: str | Path,
    source_train_dataset_dir: str | Path | None = None,
    source_val_dataset_dir: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    from musclemimic.badminton.stage2_context_family import (
        validate_stage2_shared_inputs,
    )

    spec = resolve(action)
    shared_path = Path(shared_inputs).expanduser().resolve(strict=True)
    shared = validate_stage2_shared_inputs(shared_path, expected_action=spec.slug)
    datasets = shared.get("datasets") or {}
    train_path = Path(str((datasets.get("train") or {}).get("path", ""))).expanduser().resolve(strict=True)
    val_path = Path(str((datasets.get("validation") or {}).get("path", ""))).expanduser().resolve(strict=True)
    if source_train_dataset_dir is not None and train_path != Path(source_train_dataset_dir).expanduser().resolve(
        strict=True
    ):
        raise ValueError("selected S2-A train source differs from shared inputs")
    if source_val_dataset_dir is not None and val_path != Path(source_val_dataset_dir).expanduser().resolve(
        strict=True
    ):
        raise ValueError("selected S2-A validation source differs from shared inputs")
    teacher = (shared.get("teacher") or {}).get("checkpoint")
    if not isinstance(teacher, Mapping):
        raise ValueError("Stage-2 shared inputs have no teacher checkpoint")
    manifests = {
        "train": validate_dataset_manifest(train_path, expected_teacher=teacher, require_promoted_teacher=True),
        "validation": validate_dataset_manifest(val_path, expected_teacher=teacher, require_promoted_teacher=True),
    }
    for name, key in (("train", "train"), ("validation", "validation")):
        if manifests[name].get("manifest_fingerprint") != (datasets.get(key) or {}).get("manifest_fingerprint"):
            raise ValueError(f"Stage-2 shared inputs no longer bind the {name} physical dataset")
    if manifests["train"].get("teacher_promotion") != manifests["validation"].get("teacher_promotion"):
        raise ValueError("S2-A train/validation sources use different teachers")
    split = {
        "train": _split_contract(manifests["train"], split="train"),
        "val": _split_contract(manifests["validation"], split="val"),
    }
    for name, required in (("train", "train"), ("validation", "val")):
        observed = {
            str((item.get("contract") or {}).get("split", "")) for item in manifests[name].get("collections", [])
        }
        if observed != {required}:
            raise ValueError(f"shared {name} physical root must contain only {required} collections")
    overlap = sorted(set(split["train"]["motion_paths"]) & set(split["val"]["motion_paths"]))
    if overlap:
        raise ValueError(f"S2-A train/val motions overlap: {overlap}")
    manifests["train"]["_source_path"] = str(train_path)
    manifests["validation"]["_source_path"] = str(val_path)
    return shared, manifests, split


def derive_direct_dataset(
    *,
    action: str,
    seed: int,
    shared_inputs: str | Path,
    source_train_dataset_dir: str | Path | None = None,
    source_val_dataset_dir: str | Path | None = None,
    output_dataset_dir: str | Path,
) -> Path:
    """Copy immutable physical train/val data into one seed-specific direct root."""

    spec = resolve(action)
    if int(seed) not in EXACT_SEEDS:
        raise ValueError("S2-A dataset derivation requires one of exact seeds 0/1/2")
    shared_path = Path(shared_inputs).expanduser().resolve(strict=True)
    output = Path(output_dataset_dir).expanduser().resolve()
    shared, source_manifests, split = _validate_shared_sources(
        action=spec.slug,
        shared_inputs=shared_path,
        source_train_dataset_dir=source_train_dataset_dir,
        source_val_dataset_dir=source_val_dataset_dir,
    )
    source_paths = {
        name: Path(str(manifest["_source_path"])).resolve(strict=True) for name, manifest in source_manifests.items()
    }
    for source in set(source_paths.values()):
        if source == output or source in output.parents or output in source.parents:
            raise ValueError("direct dataset must be disjoint from both shared physical roots")
    clean_manifests = {
        name: {key: value for key, value in manifest.items() if key != "_source_path"}
        for name, manifest in source_manifests.items()
    }
    teacher_checkpoint = clean_manifests["train"]["teacher_checkpoint"]
    teacher_promotion = clean_manifests["train"]["teacher_promotion"]
    derived_run_uid = str(clean_manifests["train"]["run_uid"])
    artifact_path = output / "direct_dataset_derivation.json"
    payload: dict[str, Any] = {
        "schema_version": DIRECT_DATASET_DERIVATION_SCHEMA_VERSION,
        "action": {"slug": spec.slug, "action_id": spec.action_id},
        "seed": int(seed),
        "shared_inputs": {
            **_path_record(shared_path),
            "binding_sha256": shared["binding_sha256"],
        },
        "source_datasets": {
            "train": {
                "path": str(source_paths["train"]),
                "manifest_fingerprint": clean_manifests["train"]["manifest_fingerprint"],
                "run_uid": clean_manifests["train"]["run_uid"],
                "split": split["train"],
            },
            "validation": {
                "path": str(source_paths["validation"]),
                "manifest_fingerprint": clean_manifests["validation"]["manifest_fingerprint"],
                "run_uid": clean_manifests["validation"]["run_uid"],
                "split": split["val"],
            },
            "teacher_checkpoint": teacher_checkpoint,
            "teacher_promotion": teacher_promotion,
        },
        "output_dataset": {"path": str(output), "run_uid": derived_run_uid},
        "copy_policy": {
            "mode": "byte_copy",
            "source_is_never_modified": True,
            "dagger_append_target": "output_dataset_only",
        },
    }
    payload["binding_sha256"] = canonical_json_sha256(payload)
    payload["artifact_path"] = str(artifact_path)

    if output.exists():
        if not artifact_path.is_file():
            raise FileExistsError("existing direct dataset has no immutable derivation manifest")
        validated = validate_direct_dataset_derivation(output)
        expected = dict(payload)
        if validated != expected:
            raise FileExistsError("existing direct dataset was derived under a different S2-A contract")
        return artifact_path

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.derive-", dir=str(output.parent)))
    try:
        for item in source_paths["train"].iterdir():
            if item.name == ".distill_staging":
                continue
            destination = staging / item.name
            if item.is_dir():
                shutil.copytree(item, destination)
            else:
                shutil.copy2(item, destination)
        staged_artifact = staging / artifact_path.name
        staged_payload = dict(payload)
        staged_payload["artifact_path"] = str(artifact_path)
        staged_artifact.write_text(
            json.dumps(staged_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        copied = validate_dataset_manifest(
            staging,
            expected_teacher=teacher_checkpoint,
            require_promoted_teacher=True,
        )
        if copied["run_uid"] != derived_run_uid:
            raise ValueError("derived direct dataset run identity is stale")
        if copied["manifest_fingerprint"] != clean_manifests["train"]["manifest_fingerprint"]:
            raise ValueError("derived direct train dataset differs from shared source")
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return artifact_path


def validate_direct_dataset_derivation(
    dataset_dir: str | Path,
    *,
    expected_action: str | None = None,
    expected_seed: int | None = None,
    expected_shared_inputs: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(dataset_dir).expanduser().resolve(strict=True)
    artifact_path = root / "direct_dataset_derivation.json"
    payload = _load_object(artifact_path, label="direct dataset derivation")
    if payload.get("schema_version") != DIRECT_DATASET_DERIVATION_SCHEMA_VERSION:
        raise ValueError("unsupported direct dataset derivation schema")
    supplied = _require_sha256(payload.get("binding_sha256"), label="direct dataset derivation binding")
    unsigned = {key: value for key, value in payload.items() if key not in {"binding_sha256", "artifact_path"}}
    if supplied != canonical_json_sha256(unsigned):
        raise ValueError("direct dataset derivation binding is stale")
    if Path(str(payload.get("artifact_path", ""))).resolve() != artifact_path:
        raise ValueError("direct dataset derivation artifact path changed")
    action = payload.get("action") or {}
    spec = resolve(expected_action or str(action.get("slug", "")))
    if action != {"slug": spec.slug, "action_id": spec.action_id}:
        raise ValueError("direct dataset derivation belongs to another action")
    seed = int(payload.get("seed", -1))
    if seed not in EXACT_SEEDS or (expected_seed is not None and seed != int(expected_seed)):
        raise ValueError("direct dataset derivation seed is not the requested exact seed")
    shared_record = payload.get("shared_inputs") or {}
    shared_path = Path(str(shared_record.get("path", ""))).resolve(strict=True)
    if expected_shared_inputs is not None and shared_path != Path(expected_shared_inputs).expanduser().resolve(
        strict=True
    ):
        raise ValueError("direct dataset derivation uses different shared inputs")
    source_records = payload.get("source_datasets") or {}
    train_record = source_records.get("train") or {}
    val_record = source_records.get("validation") or {}
    train_source = Path(str(train_record.get("path", ""))).resolve(strict=True)
    val_source = Path(str(val_record.get("path", ""))).resolve(strict=True)
    shared, source_manifests, split = _validate_shared_sources(
        action=spec.slug,
        shared_inputs=shared_path,
        source_train_dataset_dir=train_source,
        source_val_dataset_dir=val_source,
    )
    clean_manifests = {
        name: {key: value for key, value in manifest.items() if key != "_source_path"}
        for name, manifest in source_manifests.items()
    }
    if (
        shared_record.get("content_sha256") != file_sha256(shared_path)
        or shared_record.get("binding_sha256") != shared["binding_sha256"]
        or train_record.get("manifest_fingerprint") != clean_manifests["train"]["manifest_fingerprint"]
        or val_record.get("manifest_fingerprint") != clean_manifests["validation"]["manifest_fingerprint"]
        or train_record.get("run_uid") != clean_manifests["train"]["run_uid"]
        or val_record.get("run_uid") != clean_manifests["validation"]["run_uid"]
        or train_record.get("split") != split["train"]
        or val_record.get("split") != split["val"]
        or source_records.get("teacher_checkpoint") != clean_manifests["train"]["teacher_checkpoint"]
        or source_records.get("teacher_promotion") != clean_manifests["train"]["teacher_promotion"]
    ):
        raise ValueError("direct dataset source/shared provenance changed")
    if Path(str((payload.get("output_dataset") or {}).get("path", ""))).resolve() != root:
        raise ValueError("direct dataset derivation points to another output root")

    direct = validate_dataset_manifest(
        root,
        expected_teacher=clean_manifests["train"]["teacher_checkpoint"],
        require_promoted_teacher=True,
    )
    if direct.get("run_uid") != (payload.get("output_dataset") or {}).get("run_uid"):
        raise ValueError("direct dataset run_uid differs from its derivation")
    selected_source_collections = list(clean_manifests["train"]["collections"])
    selected_source_shards = list(clean_manifests["train"]["shards"])
    source_shards = {str(item["filename"]): item for item in selected_source_shards}
    direct_shards = {str(item["filename"]): item for item in direct["shards"]}
    if any(direct_shards.get(name) != record for name, record in source_shards.items()):
        raise ValueError("direct dataset mutated or replaced an immutable source shard")
    source_collections = {str(item["collection_id"]): item for item in selected_source_collections}
    direct_collections = {str(item["collection_id"]): item for item in direct["collections"]}
    if any(direct_collections.get(name) != record for name, record in source_collections.items()):
        raise ValueError("direct dataset mutated an immutable source collection")
    return payload


@dataclass(frozen=True)
class Stage2DirectFamilyConfig:
    action: str
    shared_inputs: str
    source_train_dataset_dir: str
    source_val_dataset_dir: str
    teacher_checkpoint: str
    teacher_promotion_manifest: str
    output_dir: str
    physical_gpu: int
    cache_key_prefix: str
    student_bc_config: str | None = None
    student_ppo_config: str | None = None
    seeds: tuple[int, ...] = EXACT_SEEDS
    selected_seed: int = SELECTED_DEPLOYMENT_SEED
    num_dagger_iterations: int = EXACT_DAGGER_ITERATIONS
    train_steps: int = 200_000
    dagger_num_transitions: int = 500_000
    ppo_total_timesteps: int | None = None
    num_envs: int = 256
    shard_size: int = 50_000
    batch_size: int = 4096
    bc_lr: float = 3e-4


@dataclass(frozen=True)
class Stage2DirectStep:
    name: str
    seed: int | None
    command: tuple[str, ...]
    environment: dict[str, str]
    output_artifact: str


def _hydra_name(path: str) -> str:
    value = Path(path)
    if value.parts and value.parts[0] == "fullbody":
        value = Path(*value.parts[1:])
    return value.with_suffix("").as_posix()


def _train_environment(*, config: Stage2DirectFamilyConfig, seed: int, phase: str, log_path: Path) -> dict[str, str]:
    cache_key = f"{config.cache_key_prefix}_{resolve(config.action).slug}_s{seed}_{phase}"
    cache_root = Path(
        os.environ.get(
            "MUSCLEMIMIC_JAX_CACHE_ROOT",
            "/data3/yangfeiyang/WorkSpace/ENV/jax-cache",
        )
    )
    return {
        "CUDA_VISIBLE_DEVICES": str(int(config.physical_gpu)),
        "MUSCLEMIMIC_JAX_CACHE_KEY": cache_key,
        "JAX_COMPILATION_CACHE_DIR": str(cache_root / cache_key),
        "MUSCLEMIMIC_TRAIN_LOG": str(log_path.resolve()),
        "MUSCLEMIMIC_ORBAX_SAVE_CONCURRENT_GB": "4",
        "MUSCLEMIMIC_ORBAX_RESTORE_CONCURRENT_GB": "4",
    }


def _validate_family_config(config: Stage2DirectFamilyConfig) -> tuple[Any, str, str]:
    spec = resolve(config.action)
    if not spec.racket_applicable or spec.student_bc_config is None or spec.student_ppo_config is None:
        raise ValueError(f"action {spec.slug} has no complete racket/direct S2-A contract")
    if tuple(config.seeds) != EXACT_SEEDS:
        raise ValueError("S2-A family requires exact ordered seeds (0, 1, 2)")
    if int(config.selected_seed) != SELECTED_DEPLOYMENT_SEED:
        raise ValueError("S2-A deployment seed is pre-registered as seed 0")
    if int(config.num_dagger_iterations) != EXACT_DAGGER_ITERATIONS:
        raise ValueError("S2-A requires exactly three DAgger iterations")
    if int(config.physical_gpu) < 0 or not str(config.cache_key_prefix).strip():
        raise ValueError("S2-A requires an explicit physical GPU and stable cache prefix")
    positive = {
        "train_steps": config.train_steps,
        "dagger_num_transitions": config.dagger_num_transitions,
        "num_envs": config.num_envs,
        "shard_size": config.shard_size,
        "batch_size": config.batch_size,
    }
    if (
        any(int(value) <= 0 for value in positive.values())
        or not math.isfinite(float(config.bc_lr))
        or float(config.bc_lr) <= 0.0
    ):
        raise ValueError(f"S2-A budgets must be finite and positive: {positive}")
    bc = config.student_bc_config or spec.student_bc_config
    ppo = config.student_ppo_config or spec.student_ppo_config
    return spec, str(bc), str(ppo)


def build_stage2_direct_family_plan(
    config: Stage2DirectFamilyConfig,
) -> tuple[dict[str, Any], tuple[Stage2DirectStep, ...]]:
    """Build, but never execute, the exact three-seed S2-A lifecycle."""

    spec, bc_config, ppo_config = _validate_family_config(config)
    shared, source_manifests, split = _validate_shared_sources(
        action=spec.slug,
        shared_inputs=config.shared_inputs,
        source_train_dataset_dir=config.source_train_dataset_dir,
        source_val_dataset_dir=config.source_val_dataset_dir,
    )
    teacher = checkpoint_content_fingerprint(config.teacher_checkpoint)
    if teacher != (shared.get("teacher") or {}).get("checkpoint"):
        raise ValueError("S2-A teacher checkpoint differs from Stage-2 shared inputs")
    teacher_promotion = validate_stage2_teacher_promotion(
        config.teacher_promotion_manifest,
        teacher_checkpoint=teacher,
    )
    clean_sources = {
        name: {key: value for key, value in manifest.items() if key != "_source_path"}
        for name, manifest in source_manifests.items()
    }
    if teacher_promotion != clean_sources["train"].get("teacher_promotion"):
        raise ValueError("S2-A teacher promotion differs from shared physical data")

    root = Path(config.output_dir)
    launcher = "scripts/run_fullbody_training.sh"
    steps: list[Stage2DirectStep] = []
    per_seed: dict[str, Any] = {}
    direct_run_uid = str(clean_sources["train"]["run_uid"])
    heldout_motions = list(split["val"]["motion_paths"])
    for seed in EXACT_SEEDS:
        seed_root = root / f"seed_{seed}"
        dataset = seed_root / "direct_dataset"
        derivation = dataset / "direct_dataset_derivation.json"
        bc_root = seed_root / "bc"
        bc_checkpoint = bc_root / "checkpoints" / f"checkpoint_{int(config.train_steps)}"
        dagger_root = seed_root / "dagger"
        dagger_checkpoint = (
            dagger_root
            / f"iter_{EXACT_DAGGER_ITERATIONS - 1:03d}"
            / "checkpoints"
            / f"checkpoint_{int(config.train_steps)}"
        )
        ppo_root = seed_root / "ppo"
        ppo_checkpoints = ppo_root / "checkpoints"
        compare_root = seed_root / "compare"
        evidence = seed_root / "stage2_direct_seed_evidence.json"
        logs = seed_root / "logs"
        seed_run_id = f"{spec.slug}_stage2_s2a_direct_s{seed}_ppo_v1"

        derive_command = (
            os.sys.executable,
            "-m",
            "fullbody.stage2_direct_lifecycle",
            "derive-dataset",
            "--action",
            spec.slug,
            "--seed",
            str(seed),
            "--shared-inputs",
            str(Path(config.shared_inputs).resolve()),
            "--source-train-dataset-dir",
            str(Path(config.source_train_dataset_dir).resolve()),
            "--source-val-dataset-dir",
            str(Path(config.source_val_dataset_dir).resolve()),
            "--output-dataset-dir",
            str(dataset),
        )
        steps.append(
            Stage2DirectStep(
                name=f"s2a_seed{seed}_derive_direct_dataset",
                seed=seed,
                command=derive_command,
                environment={},
                output_artifact=str(derivation),
            )
        )
        bc_command = (
            launcher,
            "--distill-bc",
            "--dataset_dir",
            str(dataset),
            "--student_config",
            bc_config,
            "--output_dir",
            str(bc_root),
            "--batch_size",
            str(int(config.batch_size)),
            "--num_steps",
            str(int(config.train_steps)),
            "--lr",
            str(float(config.bc_lr)),
            "--seed",
            str(seed),
            "--split_seed",
            str(seed),
            "--require_dataset_manifest",
        )
        steps.append(
            Stage2DirectStep(
                name=f"s2a_seed{seed}_bc",
                seed=seed,
                command=bc_command,
                environment=_train_environment(
                    config=config,
                    seed=seed,
                    phase="bc",
                    log_path=logs / "bc.log",
                ),
                output_artifact=str(bc_checkpoint),
            )
        )
        dagger_command = (
            launcher,
            "--distill-dagger",
            "--teacher_ckpt",
            str(config.teacher_checkpoint),
            "--teacher_promotion_manifest",
            str(config.teacher_promotion_manifest),
            "--initial_student_ckpt",
            str(bc_checkpoint),
            "--student_config",
            bc_config,
            "--dataset_dir",
            str(dataset),
            "--source_train_dataset_dir",
            str(Path(config.source_train_dataset_dir).resolve()),
            "--source_val_dataset_dir",
            str(Path(config.source_val_dataset_dir).resolve()),
            "--dataset_derivation_manifest",
            str(derivation),
            "--output_dir",
            str(dagger_root),
            "--num_iters",
            str(EXACT_DAGGER_ITERATIONS),
            "--num_envs",
            str(int(config.num_envs)),
            "--num_transitions",
            str(int(config.dagger_num_transitions)),
            "--shard_size",
            str(int(config.shard_size)),
            "--train_steps",
            str(int(config.train_steps)),
            "--batch_size",
            str(int(config.batch_size)),
            "--lr",
            str(float(config.bc_lr)),
            "--seed",
            str(seed),
            "--resume_dataset",
            "--run_uid",
            direct_run_uid,
            "--jax_cache_key_prefix",
            f"{config.cache_key_prefix}_{spec.slug}_s{seed}_dagger",
            "--train_log_dir",
            str(logs / "dagger"),
            "--motion_path",
            *split["train"]["motion_paths"],
        )
        steps.append(
            Stage2DirectStep(
                name=f"s2a_seed{seed}_dagger_3round",
                seed=seed,
                command=dagger_command,
                environment=_train_environment(
                    config=config,
                    seed=seed,
                    phase="dagger_driver",
                    log_path=logs / "dagger_driver.log",
                ),
                output_artifact=str(dagger_root / "dagger_loop_results.json"),
            )
        )
        ppo_command = (
            launcher,
            f"--config-name={_hydra_name(ppo_config)}",
            f"experiment.resume_from={dagger_checkpoint}",
            "experiment.auto_resume=false",
            "experiment.reset_optimizer_on_resume=true",
            "experiment.reset_lr_schedule_on_resume=true",
            f"experiment.run_id={seed_run_id}",
            f"wandb.name={seed_run_id}",
            "experiment.n_seeds=1",
            f"experiment.seeds=[{seed}]",
            f"experiment.checkpoint_dir={ppo_checkpoints}",
            f"experiment.training_root={ppo_root}",
        )
        if config.ppo_total_timesteps is not None:
            if int(config.ppo_total_timesteps) <= 0:
                raise ValueError("S2-A PPO total_timesteps must be positive")
            ppo_command += (f"experiment.total_timesteps={int(config.ppo_total_timesteps)}",)
        steps.append(
            Stage2DirectStep(
                name=f"s2a_seed{seed}_fresh_ppo",
                seed=seed,
                command=ppo_command,
                environment=_train_environment(
                    config=config,
                    seed=seed,
                    phase="ppo",
                    log_path=logs / "ppo.log",
                ),
                output_artifact=str(ppo_checkpoints),
            )
        )
        convergence = dagger_root / "iter_002" / "distill_metadata.json"
        compare_command = (
            launcher,
            "--distill-compare",
            "--teacher_ckpt",
            str(config.teacher_checkpoint),
            "--student_ckpt",
            str(bc_checkpoint),
            "--student_dagger_ckpt",
            str(dagger_checkpoint),
            "--student_ppo_ckpt",
            str(ppo_checkpoints),
            "--promotion_policy",
            "student_bc_ppo",
            "--output_dir",
            str(compare_root),
            "--dataset_dir",
            str(Path(config.source_val_dataset_dir).resolve()),
            "--convergence_metrics",
            str(convergence),
            "--eval_seed",
            str(seed),
            "--deterministic",
            "--motion_path",
            *heldout_motions,
            "--require_pass",
        )
        steps.append(
            Stage2DirectStep(
                name=f"s2a_seed{seed}_heldout_compare",
                seed=seed,
                command=compare_command,
                environment=_train_environment(
                    config=config,
                    seed=seed,
                    phase="heldout_eval",
                    log_path=logs / "heldout_compare.log",
                ),
                output_artifact=str(compare_root / "direct_promotion_evidence.json"),
            )
        )
        seal_command = (
            os.sys.executable,
            "-m",
            "fullbody.stage2_direct_lifecycle",
            "seal-seed",
            "--action",
            spec.slug,
            "--seed",
            str(seed),
            "--shared-inputs",
            str(Path(config.shared_inputs).resolve()),
            "--direct-dataset-dir",
            str(dataset),
            "--bc-checkpoint",
            str(bc_checkpoint),
            "--dagger-dir",
            str(dagger_root),
            "--ppo-checkpoint",
            str(ppo_checkpoints),
            "--compare-dir",
            str(compare_root),
            "--output",
            str(evidence),
        )
        steps.append(
            Stage2DirectStep(
                name=f"s2a_seed{seed}_seal",
                seed=seed,
                command=seal_command,
                environment={},
                output_artifact=str(evidence),
            )
        )
        per_seed[str(seed)] = {
            "root": str(seed_root),
            "direct_dataset": str(dataset),
            "dataset_derivation": str(derivation),
            "bc_checkpoint": str(bc_checkpoint),
            "dagger_checkpoint": str(dagger_checkpoint),
            "ppo_checkpoint_root": str(ppo_checkpoints),
            "seed_evidence": str(evidence),
        }

    family_promotion = root / "stage2_direct_family_promotion.json"
    family_command: list[str] = [
        os.sys.executable,
        "-m",
        "fullbody.stage2_direct_lifecycle",
        "promote-family",
        "--action",
        spec.slug,
        "--shared-inputs",
        str(Path(config.shared_inputs).resolve()),
        "--output",
        str(family_promotion),
    ]
    for seed in EXACT_SEEDS:
        family_command.extend(["--seed-evidence", f"{seed}:{per_seed[str(seed)]['seed_evidence']}"])
    steps.append(
        Stage2DirectStep(
            name="s2a_family_promotion",
            seed=None,
            command=tuple(family_command),
            environment={},
            output_artifact=str(family_promotion),
        )
    )
    contract_core = {
        "action_slug": spec.slug,
        "teacher_checkpoint_sha256": teacher["sha256"],
        "teacher_promotion_binding_sha256": teacher_promotion["binding_sha256"],
        "shared_inputs_binding_sha256": shared["binding_sha256"],
        "source_train_dataset_manifest_fingerprint": clean_sources["train"]["manifest_fingerprint"],
        "source_validation_dataset_manifest_fingerprint": clean_sources["validation"]["manifest_fingerprint"],
        "train_motion_set_fingerprint": split["train"]["motion_set_fingerprint"],
        "heldout_motion_set_fingerprint": split["val"]["motion_set_fingerprint"],
        "student_bc_config": bc_config,
        "student_ppo_config": ppo_config,
        "num_dagger_iterations": EXACT_DAGGER_ITERATIONS,
        "train_steps_per_bc": int(config.train_steps),
        "dagger_num_transitions_per_iteration": int(config.dagger_num_transitions),
        "ppo_total_timesteps": config.ppo_total_timesteps,
    }
    payload: dict[str, Any] = {
        "schema_version": DIRECT_FAMILY_PLAN_SCHEMA_VERSION,
        "config": asdict(config),
        "action": {"slug": spec.slug, "action_id": spec.action_id},
        "exact_seeds": list(EXACT_SEEDS),
        "selected_deployment_seed": SELECTED_DEPLOYMENT_SEED,
        "contract_core": contract_core,
        "contract_core_sha256": canonical_json_sha256(contract_core),
        "per_seed": per_seed,
        "steps": [asdict(step) for step in steps],
        "family_promotion": str(family_promotion),
        "execution_policy": {
            "training_started_by_planner": False,
            "canonical_launcher": launcher,
            "source_dataset_mutation_forbidden": True,
            "dagger_dataset_scope": "seed_specific_direct_dataset_only",
            "retired_monolithic_runner_allowed": False,
        },
    }
    payload["plan_fingerprint"] = canonical_json_sha256(payload)
    return payload, tuple(steps)


def _validate_file_record(record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} artifact record is missing")
    path = Path(str(record.get("path", ""))).expanduser().resolve(strict=True)
    if record.get("sha256", record.get("content_sha256")) != file_sha256(path):
        raise ValueError(f"{label} artifact content changed")
    return path


def _validate_direct_promotion_evidence(
    path: str | Path,
    *,
    teacher: Mapping[str, Any],
    student: Mapping[str, Any],
    dataset_manifest_fingerprint: str,
    heldout: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, float]], dict[str, Any]]:
    source = Path(path).expanduser().resolve(strict=True)
    evidence = _load_object(source, label="direct PPO promotion evidence")
    if evidence.get("schema_version") != "direct_distill_promotion_evidence_v2":
        raise ValueError("direct PPO promotion evidence schema is invalid")
    supplied = _require_sha256(evidence.get("evidence_fingerprint"), label="direct evidence fingerprint")
    unsigned = {key: value for key, value in evidence.items() if key != "evidence_fingerprint"}
    if supplied != canonical_json_sha256(unsigned):
        raise ValueError("direct PPO promotion evidence fingerprint is stale")
    if (
        evidence.get("promotion_policy") != "student_bc_ppo"
        or evidence.get("deterministic") is not True
        or evidence.get("teacher_checkpoint") != dict(teacher)
        or evidence.get("student_checkpoint") != dict(student)
        or evidence.get("dataset_manifest_fingerprint") != dataset_manifest_fingerprint
    ):
        raise ValueError("direct promotion is not the deterministic PPO endpoint for this lineage")
    expected_paths = list(heldout["motion_paths"])
    expected_uids = [int(value) for value in heldout["motion_uids"]]
    recorded_heldout = evidence.get("heldout") or {}
    if (
        recorded_heldout.get("motion_paths") != expected_paths
        or recorded_heldout.get("motion_uids") != expected_uids
        or recorded_heldout.get("motion_set_fingerprint") != heldout["motion_set_fingerprint"]
        or int(recorded_heldout.get("num_motions", -1)) != len(expected_paths)
    ):
        raise ValueError("direct promotion held-out split differs from shared inputs")

    artifacts = evidence.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("direct promotion evidence has no source artifacts")
    paths: dict[str, Path] = {}
    for name in ("comparison_metrics", "acceptance", "convergence", "temporal_audit"):
        paths[name] = _validate_file_record(artifacts.get(name), label=f"direct {name}")
    acceptance = _load_object(paths["acceptance"], label="direct acceptance")
    accepted = validate_direct_acceptance_record(acceptance.get("student_bc_ppo"))
    comparison_raw = _load_object(paths["comparison_metrics"], label="direct comparison")
    required_policies = ("teacher", "student_bc", "student_bc_dagger", "student_bc_ppo")
    comparison: dict[str, dict[str, float]] = {}
    for policy in required_policies:
        raw = comparison_raw.get(policy)
        if not isinstance(raw, Mapping):
            raise ValueError(f"direct comparison lacks {policy}")
        numeric: dict[str, float] = {}
        for key, value in raw.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                numeric[str(key)] = number
        canonical = canonicalize_eval_metrics(numeric)
        missing = [name for name in REQUIRED_EVAL_METRICS if name not in canonical]
        if missing:
            raise ValueError(f"direct comparison {policy} lacks metrics {missing}")
        comparison[policy] = canonical
    return evidence, comparison, accepted


def _checkpoint_run_manifest(checkpoint: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    leaf = Path(str(checkpoint.get("resolved_path", ""))).resolve(strict=True)
    candidates = (leaf.parent / "manifest.json", leaf.parent.parent / "manifest.json")
    manifest_path = next((value for value in candidates if value.is_file()), None)
    if manifest_path is None:
        raise FileNotFoundError(f"PPO checkpoint has no checkpoint-root manifest.json: {leaf}")
    return manifest_path.resolve(), _load_object(manifest_path, label="S2-A PPO run manifest")


def _same_checkpoint_path(left: Any, right: str | Path) -> bool:
    try:
        from musclemimic.runner.checkpointing import _canonicalize_resume_path

        return Path(_canonicalize_resume_path(str(left))).resolve(strict=True) == Path(right).resolve(strict=True)
    except (OSError, ValueError, TypeError):
        return False


def _validate_dagger_artifacts(
    *,
    dagger_dir: str | Path,
    seed: int,
    direct_dataset_dir: str | Path,
    source_train_dataset_dir: str | Path,
    source_val_dataset_dir: str | Path,
    bc_checkpoint: Mapping[str, Any],
    teacher_checkpoint: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    root = Path(dagger_dir).expanduser().resolve(strict=True)
    manifest_path = root / "dagger_loop_manifest.json"
    results_path = root / "dagger_loop_results.json"
    manifest = _load_object(manifest_path, label="DAgger loop manifest")
    results = _load_object(results_path, label="DAgger loop results")
    if manifest.get("schema_version") != "dagger_loop_v2":
        raise ValueError("S2-A requires canonical DAgger loop v2")
    config = manifest.get("config") or {}
    if (
        int(config.get("num_iters", -1)) != EXACT_DAGGER_ITERATIONS
        or int(config.get("seed", -1)) != int(seed)
        or config.get("num_steps") is not None
        or int(config.get("num_transitions", -1)) <= 0
        or Path(str(config.get("dataset_dir", ""))).resolve() != Path(direct_dataset_dir).resolve()
        or Path(str(config.get("source_train_dataset_dir", ""))).resolve() != Path(source_train_dataset_dir).resolve()
        or Path(str(config.get("source_val_dataset_dir", ""))).resolve() != Path(source_val_dataset_dir).resolve()
        or Path(str(config.get("output_dir", ""))).resolve() != root
        or str(config.get("canonical_launcher")) != "scripts/run_fullbody_training.sh"
    ):
        raise ValueError("DAgger loop manifest differs from the S2-A seed contract")
    iterations = manifest.get("iterations")
    if not isinstance(iterations, list) or [int(item.get("dagger_iteration", -1)) for item in iterations] != list(
        range(EXACT_DAGGER_ITERATIONS)
    ):
        raise ValueError("S2-A DAgger manifest lacks exact iterations 0/1/2")
    for index, item in enumerate(iterations):
        command = item.get("train_command")
        environment = item.get("train_environment")
        if (
            not isinstance(command, list)
            or command[:2] != ["scripts/run_fullbody_training.sh", "--distill-bc"]
            or not isinstance(environment, Mapping)
            or str(environment.get("CUDA_VISIBLE_DEVICES", "")).isdigit() is False
            or not str(environment.get("MUSCLEMIMIC_JAX_CACHE_KEY", ""))
            or not str(environment.get("JAX_COMPILATION_CACHE_DIR", ""))
            or not str(environment.get("MUSCLEMIMIC_TRAIN_LOG", ""))
        ):
            raise ValueError(f"DAgger iteration {index} bypasses the canonical BC launch contract")
    cache_keys = [str(item["train_environment"]["MUSCLEMIMIC_JAX_CACHE_KEY"]) for item in iterations]
    logs = [str(item["train_environment"]["MUSCLEMIMIC_TRAIN_LOG"]) for item in iterations]
    cache_dirs = [str(item["train_environment"]["JAX_COMPILATION_CACHE_DIR"]) for item in iterations]
    if (
        len(set(cache_keys)) != EXACT_DAGGER_ITERATIONS
        or len(set(cache_dirs)) != EXACT_DAGGER_ITERATIONS
        or len(set(logs)) != (EXACT_DAGGER_ITERATIONS)
    ):
        raise ValueError("DAgger BC retrains do not have independent cache/log identities")

    if results.get("schema_version") != "dagger_loop_results_v2":
        raise ValueError("S2-A DAgger results schema is invalid")
    supplied = _require_sha256(results.get("binding_sha256"), label="DAgger results binding")
    unsigned = {key: value for key, value in results.items() if key != "binding_sha256"}
    if supplied != canonical_json_sha256(unsigned):
        raise ValueError("DAgger results binding is stale")
    if Path(str(results.get("loop_manifest", ""))).resolve(strict=True) != manifest_path.resolve() or results.get(
        "loop_manifest_sha256"
    ) != file_sha256(manifest_path):
        raise ValueError("DAgger results belong to a different loop manifest")
    executed = results.get("iterations")
    if not isinstance(executed, list) or len(executed) != EXACT_DAGGER_ITERATIONS:
        raise ValueError("S2-A DAgger results lack three executed rounds")

    previous = dict(bc_checkpoint)
    previous_dataset_after: Mapping[str, Any] | None = None
    compact: list[dict[str, Any]] = []
    for index, record in enumerate(executed):
        if int(record.get("dagger_iteration", -1)) != index:
            raise ValueError("DAgger results iteration order changed")
        checkpoint_in = record.get("checkpoint_in_content")
        checkpoint_out = record.get("checkpoint_out_content")
        if checkpoint_in != previous:
            raise ValueError("DAgger checkpoint chain is broken")
        if not isinstance(checkpoint_out, Mapping) or not checkpoint_fingerprint_matches(checkpoint_out):
            raise ValueError(f"DAgger iteration {index} checkpoint changed")
        if (
            index == 0
            and (manifest.get("config") or {}).get("initial_student_ckpt") != bc_checkpoint.get("supplied_path")
            and not _same_checkpoint_path(
                (manifest.get("config") or {}).get("initial_student_ckpt"),
                str(bc_checkpoint.get("resolved_path")),
            )
        ):
            raise ValueError("DAgger does not start from the selected BC checkpoint")
        result_json = Path(str(record.get("result_json", ""))).resolve(strict=True)
        result_payload = _load_object(result_json, label=f"DAgger iteration {index} result")
        if (
            result_payload.get("checkpoint_in_content") != checkpoint_in
            or result_payload.get("checkpoint_out_content") != checkpoint_out
        ):
            raise ValueError(f"DAgger iteration {index} result checkpoint mismatch")
        # The compact loop result intentionally omits num_samples; compare
        # shared identity fields instead of trusting path-only data.
        for field in ("manifest_fingerprint", "num_collections"):
            if (result_payload.get("dataset_manifest_after") or {}).get(field) != (
                record.get("dataset_manifest_after") or {}
            ).get(field):
                raise ValueError(f"DAgger iteration {index} result/summary dataset mismatch")
        before = record.get("dataset_manifest_before") or {}
        after = record.get("dataset_manifest_after") or {}
        if int(after.get("num_collections", -1)) != int(before.get("num_collections", -1)) + 1:
            raise ValueError(f"DAgger iteration {index} did not append exactly one collection")
        if int(after.get("num_samples", -1)) - int(before.get("num_samples", -1)) != int(
            config.get("num_transitions", -1)
        ):
            raise ValueError(f"DAgger iteration {index} sample delta differs from its fixed budget")
        if previous_dataset_after is not None and any(
            before.get(field) != previous_dataset_after.get(field)
            for field in ("manifest_fingerprint", "num_collections")
        ):
            raise ValueError("DAgger dataset snapshots do not form one append-only chain")
        previous_dataset_after = after
        metadata_path = Path(str(checkpoint_out["resolved_path"])).parent.parent / "distill_metadata.json"
        metadata = _load_object(metadata_path, label=f"DAgger iteration {index} metadata")
        provenance = metadata.get("dataset_provenance") or {}
        after = record.get("dataset_manifest_after") or {}
        if provenance.get("manifest_fingerprint") != after.get("manifest_fingerprint"):
            raise ValueError(f"DAgger iteration {index} BC metadata uses another dataset snapshot")
        if index == 0 and checkpoint_in != bc_checkpoint:
            raise ValueError("DAgger iteration zero does not consume the selected BC")
        previous = dict(checkpoint_out)
        compact.append(
            {
                "iteration": index,
                "checkpoint_in": checkpoint_in,
                "checkpoint_out": checkpoint_out,
                "dataset_manifest_before": record.get("dataset_manifest_before"),
                "dataset_manifest_after": record.get("dataset_manifest_after"),
                "result": _path_record(result_json),
                "distill_metadata": _path_record(metadata_path),
            }
        )
    current_dataset = validate_dataset_manifest(direct_dataset_dir, require_promoted_teacher=True)
    if previous_dataset_after is None or previous_dataset_after.get("manifest_fingerprint") != current_dataset.get(
        "manifest_fingerprint"
    ):
        raise ValueError("final DAgger dataset differs from the sealed append chain")
    if previous.get("sha256") == teacher_checkpoint.get("sha256"):
        raise ValueError("DAgger student checkpoint is mislabeled as the teacher")
    return manifest, results, compact


def build_stage2_direct_seed_evidence(
    *,
    action: str,
    seed: int,
    shared_inputs: str | Path,
    direct_dataset_dir: str | Path,
    bc_checkpoint: str | Path,
    dagger_dir: str | Path,
    ppo_checkpoint: str | Path,
    compare_dir: str | Path,
) -> dict[str, Any]:
    """Seal one exact seed after deterministic PPO held-out acceptance."""

    spec = resolve(action)
    if int(seed) not in EXACT_SEEDS:
        raise ValueError("S2-A seed evidence requires exact seed 0/1/2")
    shared_path = Path(shared_inputs).expanduser().resolve(strict=True)
    derivation = validate_direct_dataset_derivation(
        direct_dataset_dir,
        expected_action=spec.slug,
        expected_seed=int(seed),
        expected_shared_inputs=shared_path,
    )
    from musclemimic.badminton.stage2_context_family import (
        validate_stage2_shared_inputs,
    )

    shared = validate_stage2_shared_inputs(shared_path, expected_action=spec.slug)
    teacher = (shared.get("teacher") or {}).get("checkpoint")
    if not isinstance(teacher, Mapping) or not checkpoint_fingerprint_matches(teacher):
        raise ValueError("S2-A shared teacher checkpoint changed")
    direct_manifest = validate_dataset_manifest(
        direct_dataset_dir,
        expected_teacher=teacher,
        require_promoted_teacher=True,
    )
    source_records = derivation["source_datasets"]
    train_source = source_records["train"]
    val_source = source_records["validation"]
    validation_manifest = validate_dataset_manifest(
        val_source["path"],
        expected_teacher=teacher,
        require_promoted_teacher=True,
    )
    if validation_manifest["manifest_fingerprint"] != val_source["manifest_fingerprint"]:
        raise ValueError("S2-A immutable held-out validation source changed")
    direct_collections = direct_manifest.get("collections") or []
    source_count = len(train_source["split"]["collection_ids"])
    appended = direct_collections[source_count:]
    if (
        len(appended) != EXACT_DAGGER_ITERATIONS
        or [int((item.get("contract") or {}).get("dagger_iteration", -1)) for item in appended]
        != list(range(EXACT_DAGGER_ITERATIONS))
        or any(
            (item.get("contract") or {}).get("collector") != "dagger_student_rollout_teacher_relabel"
            for item in appended
        )
    ):
        raise ValueError("direct dataset must contain only the shared source plus exact DAgger rounds 0/1/2")

    bc = checkpoint_content_fingerprint(bc_checkpoint)
    dagger_manifest, dagger_results, dagger_iterations = _validate_dagger_artifacts(
        dagger_dir=dagger_dir,
        seed=int(seed),
        direct_dataset_dir=direct_dataset_dir,
        source_train_dataset_dir=train_source["path"],
        source_val_dataset_dir=val_source["path"],
        bc_checkpoint=bc,
        teacher_checkpoint=teacher,
    )
    final_dagger = dagger_iterations[-1]["checkpoint_out"]
    ppo = checkpoint_content_fingerprint(ppo_checkpoint)
    run_manifest_path, run_manifest = _checkpoint_run_manifest(ppo)
    experiment = run_manifest.get("experiment_config")
    if not isinstance(experiment, Mapping):
        raise ValueError("S2-A PPO run manifest lacks resolved experiment config")
    expected_run_id = f"{spec.slug}_stage2_s2a_direct_s{int(seed)}_ppo_v1"
    if (
        str(experiment.get("run_id")) != expected_run_id
        or experiment.get("auto_resume") is not False
        or experiment.get("reset_optimizer_on_resume") is not True
        or experiment.get("reset_lr_schedule_on_resume") is not True
        or int(experiment.get("n_seeds", -1)) != 1
        or list(experiment.get("seeds") or []) != [int(seed)]
        or not _same_checkpoint_path(experiment.get("resume_from"), str(final_dagger["resolved_path"]))
    ):
        raise ValueError("S2-A PPO did not initialize from final DAgger weights with a fresh optimizer")
    git_sha = str(run_manifest.get("git_sha") or "").strip()
    config_hash = str(run_manifest.get("config_hash") or "").strip()
    if not git_sha or not config_hash:
        raise ValueError("S2-A PPO run manifest lacks source/config identity")

    compare_root = Path(compare_dir).expanduser().resolve(strict=True)
    evidence, comparison, accepted = _validate_direct_promotion_evidence(
        compare_root / "direct_promotion_evidence.json",
        teacher=teacher,
        student=ppo,
        dataset_manifest_fingerprint=validation_manifest["manifest_fingerprint"],
        heldout=val_source["split"],
    )
    if evidence.get("teacher_promotion") != validation_manifest.get("teacher_promotion"):
        raise ValueError("S2-A comparison uses another teacher promotion")

    bc_metrics = comparison["student_bc"]
    dagger_metrics = comparison["student_bc_dagger"]
    ppo_metrics = comparison["student_bc_ppo"]
    dagger_improvement = {
        "mean_episode_return_delta": float(dagger_metrics["mean_episode_return"] - bc_metrics["mean_episode_return"]),
        "completion_rate_delta": float(dagger_metrics["completion_rate"] - bc_metrics["completion_rate"]),
        "early_termination_rate_reduction": float(
            bc_metrics["early_termination_rate"] - dagger_metrics["early_termination_rate"]
        ),
        "err_rpos_reduction": float(bc_metrics["err_rpos"] - dagger_metrics["err_rpos"]),
        "err_racket_pos_reduction": float(bc_metrics["err_racket_pos"] - dagger_metrics["err_racket_pos"]),
        "err_racket_rot_reduction": float(bc_metrics["err_racket_rot"] - dagger_metrics["err_racket_rot"]),
    }
    failure_rates = {
        "bc_early_termination_rate": float(bc_metrics["early_termination_rate"]),
        "dagger_early_termination_rate": float(dagger_metrics["early_termination_rate"]),
        "ppo_early_termination_rate": float(ppo_metrics["early_termination_rate"]),
        "ppo_completion_failure_rate": float(max(0.0, 1.0 - ppo_metrics["completion_rate"])),
    }
    if any(not math.isfinite(value) for value in (*dagger_improvement.values(), *failure_rates.values())) or any(
        not 0.0 <= value <= 1.0 for value in failure_rates.values()
    ):
        raise ValueError("S2-A DAgger improvement/failure metrics are invalid")

    contract_core = {
        "action_slug": spec.slug,
        "teacher_checkpoint_sha256": teacher["sha256"],
        "teacher_promotion_binding_sha256": (direct_manifest.get("teacher_promotion") or {}).get("binding_sha256"),
        "shared_inputs_binding_sha256": shared["binding_sha256"],
        "source_train_dataset_manifest_fingerprint": train_source["manifest_fingerprint"],
        "source_validation_dataset_manifest_fingerprint": val_source["manifest_fingerprint"],
        "train_motion_set_fingerprint": train_source["split"]["motion_set_fingerprint"],
        "heldout_motion_set_fingerprint": val_source["split"]["motion_set_fingerprint"],
        "num_dagger_iterations": EXACT_DAGGER_ITERATIONS,
        "train_steps_per_bc": int((dagger_manifest.get("config") or {})["train_steps"]),
        "dagger_num_transitions_per_iteration": int((dagger_manifest.get("config") or {})["num_transitions"]),
        "student_bc_config": str((dagger_manifest.get("config") or {})["student_config"]),
        "ppo_total_timesteps": int(experiment.get("total_timesteps", 0)),
    }
    if contract_core["ppo_total_timesteps"] <= 0:
        raise ValueError("S2-A PPO run manifest has no fixed positive budget")
    payload: dict[str, Any] = {
        "schema_version": DIRECT_SEED_EVIDENCE_SCHEMA_VERSION,
        "action": {"slug": spec.slug, "action_id": spec.action_id},
        "seed": int(seed),
        "accepted": True,
        "shared_inputs": {
            **_path_record(shared_path),
            "binding_sha256": shared["binding_sha256"],
        },
        "dataset": {
            "derivation": _path_record(Path(direct_dataset_dir) / "direct_dataset_derivation.json"),
            "source_train_manifest_fingerprint": train_source["manifest_fingerprint"],
            "source_validation_manifest_fingerprint": val_source["manifest_fingerprint"],
            "direct_manifest_fingerprint": direct_manifest["manifest_fingerprint"],
            "heldout_validation_manifest_fingerprint": validation_manifest["manifest_fingerprint"],
            "run_uid": direct_manifest["run_uid"],
            "train_split": train_source["split"],
            "heldout_split": val_source["split"],
            "source_collection_count": source_count,
            "dagger_collection_count": len(appended),
        },
        "teacher": {
            "checkpoint": teacher,
            "promotion": direct_manifest["teacher_promotion"],
        },
        "checkpoints": {
            "bc": bc,
            "dagger_iterations": dagger_iterations,
            "final_dagger": final_dagger,
            "ppo": ppo,
        },
        "fresh_ppo": {
            "run_manifest": _path_record(run_manifest_path),
            "run_id": expected_run_id,
            "config_hash": config_hash,
            "git_sha": git_sha,
            "initialized_from_final_dagger_checkpoint_sha256": final_dagger["sha256"],
            "auto_resume": False,
            "optimizer_reset": True,
            "lr_schedule_reset": True,
        },
        "dagger": {
            "loop_manifest": _path_record(Path(dagger_dir) / "dagger_loop_manifest.json"),
            "loop_results": _path_record(Path(dagger_dir) / "dagger_loop_results.json"),
            "results_binding_sha256": dagger_results["binding_sha256"],
            "improvement_vs_bc": dagger_improvement,
        },
        "heldout": {
            "direct_promotion_evidence": _path_record(compare_root / "direct_promotion_evidence.json"),
            "promotion_evidence_fingerprint": evidence["evidence_fingerprint"],
            "accepted_policy": "student_bc_ppo",
            "accepted_values": accepted["values"],
            "failure_rates": failure_rates,
            "comparison_metrics": {
                policy: {
                    key: float(metrics[key])
                    for key in (
                        "mean_episode_return",
                        "completion_rate",
                        "early_termination_rate",
                        "err_rpos",
                        "err_racket_pos",
                        "err_racket_rot",
                    )
                }
                for policy, metrics in comparison.items()
            },
        },
        "contract_core": contract_core,
        "contract_core_sha256": canonical_json_sha256(contract_core),
        "claim_scope": {
            "supported": [
                "one deterministic held-out S2-A seed endpoint",
                "BC to three-round DAgger to fresh-optimizer PPO lineage",
            ],
            "excluded": [
                "single-seed family claim",
                "frame-level samples as independent statistical units",
            ],
        },
    }
    payload["binding_sha256"] = canonical_json_sha256(payload)
    return payload


def validate_stage2_direct_seed_evidence(
    source: str | Path | Mapping[str, Any],
    *,
    expected_action: str | None = None,
    expected_seed: int | None = None,
    expected_shared_inputs: str | Path | None = None,
) -> dict[str, Any]:
    payload = dict(source) if isinstance(source, Mapping) else _load_object(source, label="S2-A seed evidence")
    if payload.get("schema_version") != DIRECT_SEED_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("unsupported S2-A seed evidence schema")
    supplied = _require_sha256(payload.get("binding_sha256"), label="S2-A seed evidence binding")
    unsigned = {key: value for key, value in payload.items() if key != "binding_sha256"}
    if supplied != canonical_json_sha256(unsigned):
        raise ValueError("S2-A seed evidence binding is stale")
    action = payload.get("action") or {}
    spec = resolve(expected_action or str(action.get("slug", "")))
    seed = int(payload.get("seed", -1))
    if expected_seed is not None and seed != int(expected_seed):
        raise ValueError("S2-A seed evidence uses another seed")
    if seed not in EXACT_SEEDS or payload.get("accepted") is not True:
        raise ValueError("S2-A seed evidence is not an accepted exact seed")
    shared_path = Path(str((payload.get("shared_inputs") or {}).get("path", ""))).resolve(strict=True)
    if expected_shared_inputs is not None and shared_path != Path(expected_shared_inputs).expanduser().resolve(
        strict=True
    ):
        raise ValueError("S2-A seed evidence uses different shared inputs")
    derivation_path = Path(str((payload.get("dataset") or {}).get("derivation", {}).get("path", ""))).resolve(
        strict=True
    )
    bc = payload.get("checkpoints", {}).get("bc") or {}
    ppo = payload.get("checkpoints", {}).get("ppo") or {}
    dagger_manifest_path = Path(str((payload.get("dagger") or {}).get("loop_manifest", {}).get("path", ""))).resolve(
        strict=True
    )
    promotion_path = Path(
        str((payload.get("heldout") or {}).get("direct_promotion_evidence", {}).get("path", ""))
    ).resolve(strict=True)
    rebuilt = build_stage2_direct_seed_evidence(
        action=spec.slug,
        seed=seed,
        shared_inputs=shared_path,
        direct_dataset_dir=derivation_path.parent,
        bc_checkpoint=str(bc.get("supplied_path", bc.get("resolved_path", ""))),
        dagger_dir=dagger_manifest_path.parent,
        ppo_checkpoint=str(ppo.get("supplied_path", ppo.get("resolved_path", ""))),
        compare_dir=promotion_path.parent,
    )
    if rebuilt != payload:
        raise ValueError("S2-A seed evidence or one of its sources changed")
    return payload


def _descriptive(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (len(EXACT_SEEDS),) or not np.isfinite(array).all():
        raise ValueError("S2-A family statistics require three finite seed values")
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1))
    half_width = float(4.302652729911275 * std / math.sqrt(len(array)))
    return {
        "n_seeds": len(EXACT_SEEDS),
        "values_by_seed": {str(seed): float(value) for seed, value in zip(EXACT_SEEDS, array, strict=True)},
        "mean": mean,
        "sample_std": std,
        "cohen_dz_vs_zero": None if std <= 0.0 else float(mean / std),
        "student_t_95_ci": [mean - half_width, mean + half_width],
        "inference_limit": ("n=3 seed-level descriptive interval; motions/frames are not independent n"),
    }


def build_stage2_direct_family_promotion(
    *,
    action: str,
    shared_inputs: str | Path,
    seed_evidence: Mapping[int, str | Path],
) -> dict[str, Any]:
    """Promote seed-0 only after an exact all-seed paired S2-A gate."""

    spec = resolve(action)
    shared_path = Path(shared_inputs).expanduser().resolve(strict=True)
    if {int(value) for value in seed_evidence} != set(EXACT_SEEDS):
        raise ValueError("S2-A family promotion requires exact seeds 0/1/2")
    records: dict[int, dict[str, Any]] = {}
    artifacts: dict[str, Any] = {}
    for seed in EXACT_SEEDS:
        path = Path(seed_evidence[seed]).expanduser().resolve(strict=True)
        record = validate_stage2_direct_seed_evidence(
            path,
            expected_action=spec.slug,
            expected_seed=seed,
            expected_shared_inputs=shared_path,
        )
        records[seed] = record
        artifacts[str(seed)] = {
            **_path_record(path),
            "binding_sha256": record["binding_sha256"],
        }
    shared_bindings = {str(record["shared_inputs"]["binding_sha256"]) for record in records.values()}
    contract_cores = {canonical_json_sha256(record["contract_core"]) for record in records.values()}
    teacher_hashes = {str(record["teacher"]["checkpoint"]["sha256"]) for record in records.values()}
    teacher_promotions = {canonical_json_sha256(record["teacher"]["promotion"]) for record in records.values()}
    git_shas = {str(record["fresh_ppo"]["git_sha"]) for record in records.values()}
    if any(
        len(values) != 1
        for values in (
            shared_bindings,
            contract_cores,
            teacher_hashes,
            teacher_promotions,
            git_shas,
        )
    ):
        raise ValueError("S2-A seeds do not share one teacher/dataset/split/config/source lineage")

    improvement_fields = tuple(records[0]["dagger"]["improvement_vs_bc"].keys())
    improvement = {
        field: _descriptive([float(records[seed]["dagger"]["improvement_vs_bc"][field]) for seed in EXACT_SEEDS])
        for field in improvement_fields
    }
    failure_fields = tuple(records[0]["heldout"]["failure_rates"].keys())
    failure_rates = {
        field: _descriptive([float(records[seed]["heldout"]["failure_rates"][field]) for seed in EXACT_SEEDS])
        for field in failure_fields
    }
    paired_checks = {
        "exact_seed_pairs": list(records) == list(EXACT_SEEDS),
        "all_ppo_endpoints_pass_direct_acceptance": all(record["accepted"] is True for record in records.values()),
        "dagger_mean_return_improves_over_bc": (improvement["mean_episode_return_delta"]["mean"] > 0.0),
        "dagger_mean_completion_does_not_degrade": (improvement["completion_rate_delta"]["mean"] >= 0.0),
        "dagger_mean_early_termination_does_not_increase": (
            improvement["early_termination_rate_reduction"]["mean"] >= 0.0
        ),
        "one_shared_source_and_git_identity": True,
    }
    if not all(paired_checks.values()):
        failed = sorted(name for name, passed in paired_checks.items() if not passed)
        raise ValueError(f"S2-A paired held-out promotion gate failed: {failed}")

    selected = records[SELECTED_DEPLOYMENT_SEED]
    payload: dict[str, Any] = {
        "schema_version": DIRECT_FAMILY_PROMOTION_SCHEMA_VERSION,
        "action": {"slug": spec.slug, "action_id": spec.action_id},
        "passed": True,
        "exact_seeds": list(EXACT_SEEDS),
        "selected_deployment_seed": SELECTED_DEPLOYMENT_SEED,
        "selected_checkpoint": selected["checkpoints"]["ppo"],
        "shared_inputs": {
            **_path_record(shared_path),
            "binding_sha256": next(iter(shared_bindings)),
        },
        "teacher": selected["teacher"],
        "contract_core": selected["contract_core"],
        "contract_core_sha256": selected["contract_core_sha256"],
        "source_binding": {
            "git_sha": next(iter(git_shas)),
            "source_train_dataset_manifest_fingerprint": selected["dataset"]["source_train_manifest_fingerprint"],
            "source_validation_dataset_manifest_fingerprint": selected["dataset"][
                "source_validation_manifest_fingerprint"
            ],
            "train_motion_set_fingerprint": selected["dataset"]["train_split"]["motion_set_fingerprint"],
            "heldout_motion_set_fingerprint": selected["dataset"]["heldout_split"]["motion_set_fingerprint"],
        },
        "seed_evidence": artifacts,
        "paired_heldout_gate": {
            "checks": paired_checks,
            "dagger_improvement_vs_bc": improvement,
            "failure_rates": failure_rates,
            "failed_seed_fraction": 0.0,
            "statistical_unit": "seed",
            "n_seeds": len(EXACT_SEEDS),
            "confirmatory_p_value_claim": False,
        },
        "claim_scope": {
            "supported": [
                "complete S2-A BC-to-DAgger-to-fresh-PPO lifecycle",
                "paired held-out seed-level DAgger improvement",
                "seed-0 deployment selected only after all-seed gate",
            ],
            "excluded": [
                "population-level inference from three seeds",
                "latent shared dataset mutation",
                "per-seed checkpoint cherry-picking",
            ],
        },
    }
    payload["binding_sha256"] = canonical_json_sha256(payload)
    return payload


def validate_stage2_direct_family_promotion(
    source: str | Path | Mapping[str, Any],
    *,
    expected_action: str | None = None,
    expected_shared_inputs: str | Path | None = None,
) -> dict[str, Any]:
    """Rebuild a complete S2-A family promotion from every external source."""

    payload = dict(source) if isinstance(source, Mapping) else _load_object(source, label="S2-A family promotion")
    if payload.get("schema_version") != DIRECT_FAMILY_PROMOTION_SCHEMA_VERSION:
        raise ValueError("unsupported S2-A family promotion schema")
    supplied = _require_sha256(payload.get("binding_sha256"), label="S2-A family promotion binding")
    unsigned = {key: value for key, value in payload.items() if key != "binding_sha256"}
    if supplied != canonical_json_sha256(unsigned):
        raise ValueError("S2-A family promotion binding is stale")
    action = payload.get("action") or {}
    spec = resolve(expected_action or str(action.get("slug", "")))
    shared_path = Path(str((payload.get("shared_inputs") or {}).get("path", ""))).resolve(strict=True)
    if expected_shared_inputs is not None and shared_path != Path(expected_shared_inputs).expanduser().resolve(
        strict=True
    ):
        raise ValueError("S2-A family promotion uses different shared inputs")
    seed_records = payload.get("seed_evidence") or {}
    paths = {
        seed: Path(str((seed_records.get(str(seed)) or {}).get("path", ""))).resolve(strict=True)
        for seed in EXACT_SEEDS
    }
    rebuilt = build_stage2_direct_family_promotion(
        action=spec.slug,
        shared_inputs=shared_path,
        seed_evidence=paths,
    )
    if rebuilt != payload:
        raise ValueError("S2-A family promotion or one of its sources changed")
    return payload

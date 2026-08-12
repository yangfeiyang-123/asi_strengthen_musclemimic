"""Immutable Stage-2 shared inputs and paired context-family evidence.

The Stage-2 B/C/D/E comparison is meaningful only when every arm consumes the
same physical collection and the same promoted teacher.  This module seals
that shared lineage, freezes the architecture selected by S2-B, and evaluates
the pre-registered paired C/D negative control over seeds 0/1/2.

No command in this module starts training.  Training remains owned by the
latent sweep and its canonical ``scripts/run_fullbody_training.sh --latent``
launcher.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from musclemimic.badminton.action_registry import action_choices, resolve
from musclemimic.distill.provenance import (
    canonical_json_sha256,
    checkpoint_content_fingerprint,
    file_sha256,
    validate_dataset_manifest,
    validate_direct_acceptance_record,
    validate_teacher_promotion_manifest,
)
from musclemimic.synergy.basis_artifact import load_synergy_basis
from musclemimic.synergy.frozen_decoder import load_frozen_body_decoder

SHARED_INPUTS_SCHEMA_VERSION = "stage2_shared_inputs_v1"
ARCHITECTURE_LOCK_SCHEMA_VERSION = "stage2_s2b_architecture_lock_v1"
FAMILY_INDEX_SCHEMA_VERSION = "stage2_context_family_index_v1"
FAMILY_GATE_SCHEMA_VERSION = "stage2_context_family_gate_v1"

STAGE2_ARMS = ("S2-B", "S2-C", "S2-D", "S2-E")
EXACT_SEEDS = (0, 1, 2)
PRIMARY_METRIC = "eval_metrics.emg_synergy_head_loss"
RESPONSE_METRIC = "eval_metrics.emg_synergy_head_correlation"


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


def _path_record(path: str | Path, *, self_fingerprint: str | None = None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    result: dict[str, Any] = {
        "path": str(source),
        "content_sha256": file_sha256(source),
    }
    if self_fingerprint is not None:
        result["artifact_fingerprint"] = _require_sha256(
            self_fingerprint, label=f"{source.name} artifact fingerprint"
        )
    return result


def _self_fingerprint(payload: Mapping[str, Any]) -> str | None:
    for field in (
        "metrics_fingerprint",
        "report_fingerprint",
        "manifest_fingerprint",
        "binding_sha256",
        "artifact_binding_sha256",
        "evidence_fingerprint",
    ):
        value = payload.get(field)
        if isinstance(value, str):
            return value
    return None


def _verify_known_self_fingerprint(payload: Mapping[str, Any], *, label: str) -> str | None:
    """Verify a conventional top-level fingerprint when one is present."""

    for field in (
        "metrics_fingerprint",
        "report_fingerprint",
        "manifest_fingerprint",
        "evidence_fingerprint",
    ):
        value = payload.get(field)
        if value is None:
            continue
        supplied = _require_sha256(value, label=f"{label} {field}")
        unsigned = {key: item for key, item in payload.items() if key != field}
        if supplied != canonical_json_sha256(unsigned):
            raise ValueError(f"{label} {field} mismatch")
        return supplied
    # A binding_sha256 is often computed over a source graph rather than the
    # object alone.  Its dedicated validator must check it; retain it here.
    value = payload.get("binding_sha256")
    return None if value is None else _require_sha256(value, label=f"{label} binding_sha256")


def _validate_passed_source_gate(
    gate_path: str | Path,
    *,
    metrics_path: str | Path,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebuild the generic promotion-gate/source binding used by the pipeline."""

    gate_source = Path(gate_path).expanduser().resolve(strict=True)
    metrics_source = Path(metrics_path).expanduser().resolve(strict=True)
    gate = _load_object(gate_source, label=label)
    metrics = _load_object(metrics_source, label=f"{label} source metrics")
    if gate.get("passed") is not True:
        raise ValueError(f"{label} did not pass")
    binding = gate.get("source_binding")
    if not isinstance(binding, dict) or binding.get("schema_version") != "promotion_gate_source_binding_v1":
        raise ValueError(f"{label} has no source-bound gate evidence")
    if Path(str(binding.get("metrics_path", ""))).expanduser().resolve(strict=True) != metrics_source:
        raise ValueError(f"{label} is bound to different metrics")
    if binding.get("metrics_content_sha256") != file_sha256(metrics_source):
        raise ValueError(f"{label} source metrics changed after gating")
    if binding.get("metrics_schema_version") != metrics.get("schema_version"):
        raise ValueError(f"{label} source schema differs from its gate")
    expected_self = {
        key: metrics[key]
        for key in (
            "metrics_fingerprint",
            "report_fingerprint",
            "manifest_fingerprint",
            "binding_sha256",
            "bank_sha256",
            "artifact_binding_sha256",
        )
        if isinstance(metrics.get(key), str)
    }
    if binding.get("metrics_self_fingerprints") != expected_self:
        raise ValueError(f"{label} source self-fingerprint binding is stale")
    unsigned_gate = {key: value for key, value in gate.items() if key != "source_binding"}
    unsigned_source = {key: value for key, value in binding.items() if key != "binding_sha256"}
    if binding.get("binding_sha256") != canonical_json_sha256(
        {"gate": unsigned_gate, "source": unsigned_source}
    ):
        raise ValueError(f"{label} gate/source binding hash is stale")
    return gate, metrics


def _compact_dataset_manifest(path: str | Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    collections = manifest.get("collections")
    if not isinstance(collections, list) or not collections:
        raise ValueError("Stage-2 shared dataset has no physical collection provenance")
    collection_ids = sorted(
        str(item.get("collection_id", ""))
        for item in collections
        if isinstance(item, Mapping)
    )
    if not collection_ids or any(not value for value in collection_ids):
        raise ValueError("Stage-2 shared dataset collections lack collection identity")
    return {
        "path": str(Path(path).expanduser().resolve(strict=True)),
        "manifest_fingerprint": _require_sha256(
            manifest.get("manifest_fingerprint"), label="dataset manifest_fingerprint"
        ),
        "run_uid": manifest.get("run_uid"),
        "collection_ids": collection_ids,
        "teacher_checkpoint_sha256": (manifest.get("teacher_checkpoint") or {}).get("sha256"),
        "teacher_promotion": manifest.get("teacher_promotion"),
        "stage1_peasd_reference_promotions": [
            ((item.get("contract") or {}).get("request") or {}).get(
                "stage1_peasd_reference_promotion"
            )
            for item in collections
            if isinstance(item, Mapping)
        ],
        "body_synergy_contract_fingerprint": manifest.get("body_synergy_contract_fingerprint"),
        "body_synergy_portable_core_fingerprint": manifest.get(
            "body_synergy_portable_core_fingerprint"
        ),
        "frozen_body_decoder_fingerprint": manifest.get("frozen_body_decoder_fingerprint"),
    }


def _validate_direct_bc_evidence(
    *,
    bc_metrics: str | Path,
    rollout_metrics: str | Path,
    promotion_evidence: str | Path,
    teacher_checkpoint: Mapping[str, Any],
    validation_dataset_fingerprint: str,
) -> dict[str, Any]:
    """Validate the existing BC-only direct comparator without relabelling it S2-A."""

    bc_path = Path(bc_metrics).expanduser().resolve(strict=True)
    rollout_path = Path(rollout_metrics).expanduser().resolve(strict=True)
    evidence_path = Path(promotion_evidence).expanduser().resolve(strict=True)
    evidence = _load_object(evidence_path, label="direct BC promotion evidence")
    if evidence.get("schema_version") != "direct_distill_promotion_evidence_v2":
        raise ValueError("direct BC promotion evidence schema is invalid")
    supplied = _require_sha256(
        evidence.get("evidence_fingerprint"), label="direct BC evidence_fingerprint"
    )
    unsigned = {key: value for key, value in evidence.items() if key != "evidence_fingerprint"}
    if supplied != canonical_json_sha256(unsigned):
        raise ValueError("direct BC promotion evidence fingerprint is stale")
    if evidence.get("promotion_policy") != "student_bc" or evidence.get("deterministic") is not True:
        raise ValueError("direct evidence is not deterministic held-out BC")
    if evidence.get("teacher_checkpoint") != dict(teacher_checkpoint):
        raise ValueError("direct BC evidence belongs to a different teacher")
    if evidence.get("dataset_manifest_fingerprint") != validation_dataset_fingerprint:
        raise ValueError("direct BC evidence belongs to a different validation collection")
    artifacts = evidence.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("direct BC evidence has no bound source artifacts")
    expected_paths = {
        "comparison_metrics": rollout_path,
        "convergence": bc_path,
    }
    for name in ("comparison_metrics", "acceptance", "convergence", "temporal_audit"):
        record = artifacts.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"direct BC evidence is missing {name}")
        source = Path(str(record.get("path", ""))).expanduser().resolve(strict=True)
        if record.get("sha256") != file_sha256(source):
            raise ValueError(f"direct BC {name} content changed")
        if name in expected_paths and source != expected_paths[name]:
            raise ValueError(f"direct BC {name} path differs from selected evidence")
    acceptance_path = Path(str(artifacts["acceptance"]["path"])).resolve(strict=True)
    acceptance = _load_object(acceptance_path, label="direct BC acceptance")
    validate_direct_acceptance_record(acceptance.get("student_bc"))
    return {
        "required": True,
        "status": "verified_bc_only_direct_comparator",
        "evidence_scope": "bc_only_direct_comparator_evidence",
        "complete_s2a": False,
        "claim_limit": (
            "This binds the existing deterministic held-out BC comparator only; "
            "it is not a complete S2-A implementation and makes no DAgger/PPO claim."
        ),
        "bc_metrics": _path_record(bc_path),
        "rollout_metrics": _path_record(rollout_path),
        "promotion_evidence": _path_record(evidence_path, self_fingerprint=supplied),
    }


def build_stage2_shared_inputs(
    *,
    action: str,
    train_dataset_dir: str | Path,
    val_dataset_dir: str | Path,
    teacher_checkpoint: str | Path,
    teacher_promotion_manifest: str | Path,
    stage1_peasd_promotion_manifest: str | Path,
    emg_reference_manifest: str | Path,
    physical_qc_metrics: str | Path,
    physical_qc_gate: str | Path,
    synergy_basis: str | Path,
    frozen_body_decoder: str | Path,
    direct_bc_metrics: str | Path | None = None,
    direct_rollout_metrics: str | Path | None = None,
    direct_promotion_evidence: str | Path | None = None,
) -> dict[str, Any]:
    """Build and fully revalidate one immutable collection-level handoff."""

    spec = resolve(action)
    checkpoint = checkpoint_content_fingerprint(teacher_checkpoint)
    stage1_path = Path(stage1_peasd_promotion_manifest).expanduser().resolve(strict=True)
    emg_path = Path(emg_reference_manifest).expanduser().resolve(strict=True)

    from musclemimic.badminton.stage1_peasd_gate import (
        validate_stage1_peasd_teacher_promotion,
    )

    stage1_promotion = validate_stage1_peasd_teacher_promotion(
        stage1_path,
        expected_action=spec.slug,
        expected_tube=emg_path,
    )
    # A racket-mass teacher is Stage-2; a body-only teacher is the promoted
    # Stage-1 T3 checkpoint.  The direct evidence makes this distinction
    # explicit and never guesses another action's teacher role.
    direct_values = (direct_bc_metrics, direct_rollout_metrics, direct_promotion_evidence)
    any_direct = any(value is not None for value in direct_values)
    if any_direct and not all(value is not None for value in direct_values):
        raise ValueError("direct BC evidence must supply metrics, rollout, and promotion together")
    teacher_stage = "stage2" if any_direct else "stage1"
    teacher_role = "racket_mass_100" if any_direct else "body_only"
    teacher_promotion = validate_teacher_promotion_manifest(
        teacher_promotion_manifest,
        teacher_checkpoint=checkpoint,
        expected_stage=teacher_stage,
        teacher_role=teacher_role,
    )
    validation_kwargs = {
        "expected_teacher": checkpoint,
        "expected_teacher_promotion": teacher_promotion,
        "expected_stage1_peasd_promotion": stage1_path,
        "expected_emg_reference": emg_path,
        "require_promoted_teacher": True,
    }
    train = validate_dataset_manifest(train_dataset_dir, **validation_kwargs)
    val = validate_dataset_manifest(val_dataset_dir, **validation_kwargs)
    train_record = _compact_dataset_manifest(train_dataset_dir, train)
    val_record = _compact_dataset_manifest(val_dataset_dir, val)
    if train_record["teacher_checkpoint_sha256"] != val_record["teacher_checkpoint_sha256"]:
        raise ValueError("Stage-2 train/validation collections use different teachers")

    gate, qc = _validate_passed_source_gate(
        physical_qc_gate,
        metrics_path=physical_qc_metrics,
        label="Stage-2 physical rollout QC gate",
    )
    qc_fp = _verify_known_self_fingerprint(qc, label="physical rollout QC")

    basis = load_synergy_basis(synergy_basis)
    frozen = load_frozen_body_decoder(frozen_body_decoder)
    if frozen.body_synergy_contract.basis_fingerprint != basis.fingerprint:
        raise ValueError("frozen decoder and Stage-2 synergy basis differ")
    expected_contract = train_record.get("body_synergy_contract_fingerprint")
    if expected_contract is not None and expected_contract != frozen.body_synergy_contract.contract_fingerprint:
        raise ValueError("shared dataset and frozen decoder BodySynergyContract differ")

    if any_direct:
        direct = _validate_direct_bc_evidence(
            bc_metrics=direct_bc_metrics,  # type: ignore[arg-type]
            rollout_metrics=direct_rollout_metrics,  # type: ignore[arg-type]
            promotion_evidence=direct_promotion_evidence,  # type: ignore[arg-type]
            teacher_checkpoint=checkpoint,
            validation_dataset_fingerprint=val_record["manifest_fingerprint"],
        )
    else:
        direct = {
            "required": False,
            "status": "not_applicable",
            "reason": "body-only action has no racket/direct S2-A endpoint",
            "complete_s2a": False,
            "claim_limit": "No direct or racket S2-A claim is made for this action.",
        }

    payload: dict[str, Any] = {
        "schema_version": SHARED_INPUTS_SCHEMA_VERSION,
        "action": {
            "slug": spec.slug,
            "action_id": spec.action_id,
            "tube_action_id": spec.emg_trial_actions[0],
        },
        "claim_scope": {
            "supported": [
                "one immutable physical collection reused by S2-B/C/D/E",
                "one promoted teacher and one verified PEASD tube lineage",
                "paired context-family comparison under a frozen S2-B architecture",
            ],
            "excluded": [
                "BC-only evidence as a complete S2-A method",
                "independent per-arm architecture selection",
                "cross-action pooled treatment claims",
            ],
        },
        "stage1_peasd": {
            "promotion": _path_record(
                stage1_path,
                self_fingerprint=str(stage1_promotion["binding_sha256"]),
            ),
            "emg_reference": _path_record(emg_path),
            "promotion_binding_sha256": stage1_promotion["binding_sha256"],
            "emg_reference_binding": stage1_promotion["emg_reference_binding"],
        },
        "teacher": {
            "checkpoint": checkpoint,
            "promotion": _path_record(
                teacher_promotion_manifest,
                self_fingerprint=str(teacher_promotion["binding_sha256"]),
            ),
            "promotion_stage": teacher_stage,
            "teacher_role": teacher_role,
        },
        "datasets": {"train": train_record, "validation": val_record},
        "physical_qc": {
            "metrics": _path_record(physical_qc_metrics, self_fingerprint=qc_fp),
            "gate": _path_record(
                physical_qc_gate,
                self_fingerprint=str((gate.get("source_binding") or {}).get("binding_sha256")),
            ),
        },
        "direct_s2a_evidence": direct,
        "synergy": {
            "basis": {
                "path": str(basis.path.resolve()),
                "artifact_fingerprint": basis.fingerprint,
                "source_dataset_fingerprint": basis.manifest.get("source_dataset_fingerprint"),
                "teacher_checkpoint_fingerprint": basis.manifest.get("teacher_checkpoint_fingerprint"),
            },
            "frozen_body_decoder": {
                "path": str(Path(frozen_body_decoder).expanduser().resolve(strict=True)),
                "artifact_fingerprint": frozen.artifact_fingerprint,
                "body_synergy_contract_fingerprint": frozen.body_synergy_contract.contract_fingerprint,
                "portable_decoder_core_fingerprint": (
                    frozen.body_synergy_contract.portable_decoder_core_fingerprint
                ),
            },
        },
    }
    payload["binding_sha256"] = canonical_json_sha256(payload)
    return payload


def validate_stage2_shared_inputs(
    source: str | Path | Mapping[str, Any],
    *,
    expected_action: str | None = None,
) -> dict[str, Any]:
    """Rebuild a shared-input seal from its bound sources and compare exactly."""

    payload = dict(source) if isinstance(source, Mapping) else _load_object(source, label="Stage-2 shared inputs")
    if payload.get("schema_version") != SHARED_INPUTS_SCHEMA_VERSION:
        raise ValueError("unsupported Stage-2 shared-input schema")
    action = payload.get("action") or {}
    spec = resolve(expected_action or str(action.get("slug", "")))
    if action != {
        "slug": spec.slug,
        "action_id": spec.action_id,
        "tube_action_id": spec.emg_trial_actions[0],
    }:
        raise ValueError("Stage-2 shared inputs belong to another action")
    supplied = _require_sha256(payload.get("binding_sha256"), label="shared inputs binding_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "binding_sha256"}
    if supplied != canonical_json_sha256(unsigned):
        raise ValueError("Stage-2 shared-input binding mismatch")
    stage1 = payload.get("stage1_peasd") or {}
    teacher = payload.get("teacher") or {}
    datasets = payload.get("datasets") or {}
    physical = payload.get("physical_qc") or {}
    synergy = payload.get("synergy") or {}
    direct = payload.get("direct_s2a_evidence") or {}
    rebuilt = build_stage2_shared_inputs(
        action=spec.slug,
        train_dataset_dir=(datasets.get("train") or {}).get("path"),
        val_dataset_dir=(datasets.get("validation") or {}).get("path"),
        teacher_checkpoint=(teacher.get("checkpoint") or {}).get("resolved_path"),
        teacher_promotion_manifest=(teacher.get("promotion") or {}).get("path"),
        stage1_peasd_promotion_manifest=(stage1.get("promotion") or {}).get("path"),
        emg_reference_manifest=(stage1.get("emg_reference") or {}).get("path"),
        physical_qc_metrics=(physical.get("metrics") or {}).get("path"),
        physical_qc_gate=(physical.get("gate") or {}).get("path"),
        synergy_basis=(synergy.get("basis") or {}).get("path"),
        frozen_body_decoder=(synergy.get("frozen_body_decoder") or {}).get("path"),
        direct_bc_metrics=(direct.get("bc_metrics") or {}).get("path"),
        direct_rollout_metrics=(direct.get("rollout_metrics") or {}).get("path"),
        direct_promotion_evidence=(direct.get("promotion_evidence") or {}).get("path"),
    )
    if rebuilt != payload:
        raise ValueError("Stage-2 shared inputs or one of their sources changed")
    return payload


def _write_immutable(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"immutable Stage-2 artifact already exists with different content: {target}")
        return target
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, target)
    return target


def build_stage2_s2b_architecture_lock(
    *,
    shared_inputs: str | Path,
    s2b_output_dir: str | Path,
) -> dict[str, Any]:
    """Freeze S2-B's pre-registered best-synergy architecture, not a seed."""

    shared_path = Path(shared_inputs).expanduser().resolve(strict=True)
    shared = validate_stage2_shared_inputs(shared_path)
    from musclemimic.badminton.scripts.latent_synergy_sweep import (
        _load_and_validate_plan,
        validate_selected_artifact,
    )

    root = Path(s2b_output_dir).expanduser().resolve(strict=True)
    plan = _load_and_validate_plan(root)
    family = plan.get("stage2_context_family") or {}
    if family.get("arm") != "S2-B" or family.get("shared_inputs_binding_sha256") != shared["binding_sha256"]:
        raise ValueError("architecture lock requires an S2-B plan bound to these shared inputs")
    keys = sorted(
        (int(job["latent_dim"]), str(job["decoder_type"]), int(job["seed"]))
        for job in plan["jobs"]
    )
    if sorted({seed for _, _, seed in keys}) != list(EXACT_SEEDS):
        raise ValueError("S2-B architecture selection requires exact seeds 0/1/2")
    promotion_path = root / "promotion_metrics.json"
    selection_path = root / "selected" / "selection_manifest.json"
    promotion = _load_object(promotion_path, label="S2-B promotion metrics")
    selection = validate_selected_artifact(selection_path)
    selected_group = (promotion.get("selected_groups") or {}).get("best_synergy")
    selected_model = (promotion.get("selected_models") or {}).get("best_synergy")
    if not isinstance(selected_group, Mapping) or not isinstance(selected_model, Mapping):
        raise ValueError("S2-B has no promoted best_synergy architecture")
    architecture = {
        "latent_dim": int(selected_group.get("latent_dim", -1)),
        "decoder_type": str(selected_group.get("decoder_type", "")),
    }
    if (
        architecture["latent_dim"] <= 0
        or architecture["decoder_type"] == "direct"
        or int(selected_model.get("latent_dim", -1)) != architecture["latent_dim"]
        or str(selected_model.get("decoder_type", "")) != architecture["decoder_type"]
        or sorted(int(value) for value in selected_group.get("seed_set", [])) != list(EXACT_SEEDS)
    ):
        raise ValueError("S2-B best_synergy selection is malformed or seed-incomplete")
    payload: dict[str, Any] = {
        "schema_version": ARCHITECTURE_LOCK_SCHEMA_VERSION,
        "shared_inputs": _path_record(shared_path, self_fingerprint=shared["binding_sha256"]),
        "s2b": {
            "output_dir": str(root),
            "sweep_plan": _path_record(root / "sweep_plan.json", self_fingerprint=plan["plan_fingerprint"]),
            "promotion_metrics": _path_record(
                promotion_path,
                self_fingerprint=str(promotion["promotion_metrics_fingerprint"]),
            ),
            "selection_manifest": _path_record(
                selection_path,
                self_fingerprint=str(selection["selection_manifest_fingerprint"]),
            ),
        },
        "architecture": architecture,
        "training_seeds": list(EXACT_SEEDS),
        "selection_policy": {
            "source": "S2-B selected_groups.best_synergy",
            "architecture_only": True,
            "deployment_seed_is_not_reused_for_family_statistics": True,
            "per_seed_cherry_picking_forbidden": True,
        },
    }
    payload["binding_sha256"] = canonical_json_sha256(payload)
    return payload


def validate_stage2_s2b_architecture_lock(
    source: str | Path | Mapping[str, Any],
    *,
    expected_shared_inputs: str | Path | None = None,
) -> dict[str, Any]:
    payload = dict(source) if isinstance(source, Mapping) else _load_object(source, label="S2-B architecture lock")
    if payload.get("schema_version") != ARCHITECTURE_LOCK_SCHEMA_VERSION:
        raise ValueError("unsupported S2-B architecture-lock schema")
    supplied = _require_sha256(payload.get("binding_sha256"), label="architecture lock binding_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "binding_sha256"}
    if supplied != canonical_json_sha256(unsigned):
        raise ValueError("S2-B architecture-lock binding mismatch")
    shared_path = Path(str((payload.get("shared_inputs") or {}).get("path", ""))).resolve(strict=True)
    if expected_shared_inputs is not None and shared_path != Path(expected_shared_inputs).resolve(strict=True):
        raise ValueError("S2-B architecture lock uses different shared inputs")
    rebuilt = build_stage2_s2b_architecture_lock(
        shared_inputs=shared_path,
        s2b_output_dir=(payload.get("s2b") or {}).get("output_dir"),
    )
    if rebuilt != payload:
        raise ValueError("S2-B architecture lock or a bound source changed")
    return payload


def _finite_metric(metrics: Mapping[str, Any], key: str, *, arm: str, seed: int) -> float:
    try:
        value = float(metrics[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{arm}/seed-{seed} lacks finite eval_metrics.{key}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{arm}/seed-{seed} has non-finite eval_metrics.{key}")
    return value


def _context_response_metrics(
    metrics: Mapping[str, Any], *, arm: str, seed: int
) -> dict[str, float]:
    """Validate the per-seed EMG-head response contract for C/D/E.

    A finite scalar is not sufficient evidence that the privileged input was
    used.  The two real-context arms (C and E) must change the posterior when
    the context is blanked.  The shuffled arm remains a negative-control audit
    and therefore has no invented response-amplitude threshold, but all four
    diagnostics must still be present and physically valid.
    """

    values = {
        key: _finite_metric(metrics, key, arm=arm, seed=seed)
        for key in (
            "emg_synergy_head_loss",
            "emg_synergy_head_correlation",
            "emg_blank_context_posterior_mu_l2",
            "emg_blank_context_action_mse",
        )
    }
    correlation = values["emg_synergy_head_correlation"]
    if not -1.0 <= correlation <= 1.0:
        raise ValueError(
            f"{arm}/seed-{seed} has invalid emg_synergy_head_correlation"
        )
    for key in (
        "emg_synergy_head_loss",
        "emg_blank_context_posterior_mu_l2",
        "emg_blank_context_action_mse",
    ):
        if values[key] < 0.0:
            raise ValueError(f"{arm}/seed-{seed} has negative eval_metrics.{key}")
    if arm in {"S2-C", "S2-E"} and not (
        values["emg_blank_context_posterior_mu_l2"] > 0.0
    ):
        raise ValueError(
            f"{arm}/seed-{seed} has no positive blank-context posterior response"
        )
    return values


def _arm_records(
    *,
    arm: str,
    output_dir: Path,
    shared: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    from musclemimic.badminton.scripts.latent_synergy_sweep import (
        _load_and_validate_plan,
        _load_completed_run_record,
        validate_selected_artifact,
    )

    plan = _load_and_validate_plan(output_dir)
    family = plan.get("stage2_context_family") or {}
    if family.get("arm") != arm:
        raise ValueError(f"{arm} output contains a different Stage-2 arm")
    if family.get("shared_inputs_binding_sha256") != shared["binding_sha256"]:
        raise ValueError(f"{arm} does not use the shared family lineage")
    if arm == "S2-B":
        expected_architecture = lock["architecture"]
        promotion = _load_object(output_dir / "promotion_metrics.json", label="S2-B promotion metrics")
        selected = (promotion.get("selected_groups") or {}).get("best_synergy") or {}
        if {
            "latent_dim": int(selected.get("latent_dim", -1)),
            "decoder_type": str(selected.get("decoder_type", "")),
        } != expected_architecture:
            raise ValueError("S2-B promotion no longer matches its architecture lock")
        jobs = [
            job
            for job in plan["jobs"]
            if int(job["latent_dim"]) == int(expected_architecture["latent_dim"])
            and str(job["decoder_type"]) == str(expected_architecture["decoder_type"])
        ]
    else:
        if family.get("architecture_lock_binding_sha256") != lock["binding_sha256"]:
            raise ValueError(f"{arm} does not use the S2-B architecture lock")
        jobs = list(plan["jobs"])
        expected_cells = [
            (
                int(lock["architecture"]["latent_dim"]),
                str(lock["architecture"]["decoder_type"]),
                seed,
            )
            for seed in EXACT_SEEDS
        ]
        actual_cells = sorted(
            (int(job["latent_dim"]), str(job["decoder_type"]), int(job["seed"]))
            for job in jobs
        )
        if actual_cells != expected_cells:
            raise ValueError(f"{arm} is not the exact locked architecture x seeds 0/1/2")
    if sorted(int(job["seed"]) for job in jobs) != list(EXACT_SEEDS):
        raise ValueError(f"{arm} lacks exact seeds 0/1/2")
    validate_selected_artifact(output_dir / "selected" / "selection_manifest.json")
    basis = load_synergy_basis(plan["synergy_basis_path"])
    records: list[dict[str, Any]] = []
    for job in sorted(jobs, key=lambda value: int(value["seed"])):
        record = _load_completed_run_record(job, plan=plan, basis_artifact=basis)
        metrics = record["metrics"]
        seed = int(job["seed"])
        result = {
            "seed": seed,
            "run_name": str(job["run_name"]),
            "checkpoint_dir": str(Path(job["checkpoint_dir"]).resolve()),
            "checkpoint_fingerprint": record["checkpoint_fingerprint"],
            "eval_metrics": _path_record(Path(job["checkpoint_dir"]) / "eval_metrics.json"),
        }
        if arm in {"S2-C", "S2-D", "S2-E"}:
            result.update(
                _context_response_metrics(metrics, arm=arm, seed=seed)
            )
        else:
            for key in (
                "emg_synergy_head_loss",
                "emg_synergy_head_correlation",
                "emg_blank_context_posterior_mu_l2",
                "emg_blank_context_action_mse",
            ):
                if key in metrics:
                    result[key] = _finite_metric(metrics, key, arm=arm, seed=seed)
        records.append(result)
    return plan, records, family


def _assert_cd_only_shuffle(c_plan: Mapping[str, Any], d_plan: Mapping[str, Any]) -> None:
    """Reject any C/D scientific difference other than the shuffle treatment."""

    c_family = dict(c_plan.get("stage2_context_family") or {})
    d_family = dict(d_plan.get("stage2_context_family") or {})
    for value in (c_family, d_family):
        value.pop("arm", None)
    c_treatment = dict(c_family.pop("treatment", {}) or {})
    d_treatment = dict(d_family.pop("treatment", {}) or {})
    c_shuffle_treatment = c_treatment.pop("emg_shuffle_context_ablation", None)
    d_shuffle_treatment = d_treatment.pop("emg_shuffle_context_ablation", None)
    if c_family != d_family or c_treatment != d_treatment:
        raise ValueError("S2-C/S2-D family bindings differ beyond arm identity")
    if c_shuffle_treatment is not False or d_shuffle_treatment is not True:
        raise ValueError("S2-C/S2-D treatment does not encode the shuffle contrast")
    if c_plan.get("lifecycle_inputs") != d_plan.get("lifecycle_inputs"):
        raise ValueError("S2-C/S2-D lifecycle inputs are not identical")
    c_emg = dict(c_plan.get("emg_privileged") or {})
    d_emg = dict(d_plan.get("emg_privileged") or {})
    c_shuffle = c_emg.pop("shuffle_context_ablation", None)
    d_shuffle = d_emg.pop("shuffle_context_ablation", None)
    if c_shuffle is not False or d_shuffle is not True or c_emg != d_emg:
        raise ValueError("S2-C/S2-D must differ only by EMG context shuffle")
    c_jobs = sorted(c_plan["jobs"], key=lambda item: int(item["seed"]))
    d_jobs = sorted(d_plan["jobs"], key=lambda item: int(item["seed"]))
    for c_job, d_job in zip(c_jobs, d_jobs, strict=True):
        for field in (
            "latent_dim",
            "decoder_type",
            "seed",
            "synergy_basis_expected_fingerprint",
            "frozen_body_decoder_expected_fingerprint",
            "body_synergy_contract_expected_fingerprint",
            "body_synergy_portable_core_expected_fingerprint",
        ):
            if c_job.get(field) != d_job.get(field):
                raise ValueError(f"S2-C/S2-D job field {field} differs")
        c_command = list(c_job.get("training_command") or [])
        d_command = list(d_job.get("training_command") or [])

        def normalized_command(command: list[str]) -> list[str]:
            result: list[str] = []
            skip_value = False
            for token in command:
                if skip_value:
                    result.append("<arm-output-dir>")
                    skip_value = False
                elif token == "--output_dir":
                    result.append(token)
                    skip_value = True
                elif token != "--emg_shuffle_context_ablation":
                    result.append(token)
            return result

        if normalized_command(c_command) != normalized_command(d_command):
            raise ValueError(
                "S2-C/S2-D training commands differ beyond output identity and context shuffle"
            )


def build_stage2_context_family_index(
    *,
    shared_inputs: str | Path,
    architecture_lock: str | Path,
    s2b_output_dir: str | Path,
    s2c_output_dir: str | Path,
    s2d_output_dir: str | Path,
    s2e_output_dir: str | Path,
) -> dict[str, Any]:
    """Seal the exact completed B/C/D/E family before computing a gate."""

    shared_path = Path(shared_inputs).expanduser().resolve(strict=True)
    lock_path = Path(architecture_lock).expanduser().resolve(strict=True)
    shared = validate_stage2_shared_inputs(shared_path)
    lock = validate_stage2_s2b_architecture_lock(lock_path, expected_shared_inputs=shared_path)
    roots = {
        "S2-B": Path(s2b_output_dir).expanduser().resolve(strict=True),
        "S2-C": Path(s2c_output_dir).expanduser().resolve(strict=True),
        "S2-D": Path(s2d_output_dir).expanduser().resolve(strict=True),
        "S2-E": Path(s2e_output_dir).expanduser().resolve(strict=True),
    }
    arms: dict[str, Any] = {}
    plans: dict[str, dict[str, Any]] = {}
    families: dict[str, dict[str, Any]] = {}
    for arm in STAGE2_ARMS:
        plan, records, family = _arm_records(
            arm=arm,
            output_dir=roots[arm],
            shared=shared,
            lock=lock,
        )
        plans[arm] = plan
        families[arm] = family
        arms[arm] = {
            "output_dir": str(roots[arm]),
            "sweep_plan": _path_record(
                roots[arm] / "sweep_plan.json", self_fingerprint=plan["plan_fingerprint"]
            ),
            "promotion_metrics": _path_record(roots[arm] / "promotion_metrics.json"),
            "selection_manifest": _path_record(
                roots[arm] / "selected" / "selection_manifest.json"
            ),
            "seeds": records,
        }
    _assert_cd_only_shuffle(plans["S2-C"], plans["S2-D"])
    direct_contract = shared["direct_s2a_evidence"]
    if bool(direct_contract.get("required")):
        direct_paths = {
            str(family.get("direct_family_promotion_path"))
            for family in families.values()
        }
        direct_hashes = {
            str(family.get("direct_family_promotion_content_sha256"))
            for family in families.values()
        }
        direct_bindings = {
            str(family.get("direct_family_promotion_binding_sha256"))
            for family in families.values()
        }
        if (
            len(direct_paths) != 1
            or len(direct_hashes) != 1
            or len(direct_bindings) != 1
            or "None" in direct_paths | direct_hashes | direct_bindings
        ):
            raise ValueError(
                "Stage-2 B/C/D/E do not share one complete S2-A family promotion"
            )
        direct_path = Path(next(iter(direct_paths))).resolve(strict=True)
        from musclemimic.distill.stage2_direct_lifecycle import (
            validate_stage2_direct_family_promotion,
        )

        direct_family = validate_stage2_direct_family_promotion(
            direct_path,
            expected_action=str(shared["action"]["slug"]),
            expected_shared_inputs=shared_path,
        )
        if (
            file_sha256(direct_path) != next(iter(direct_hashes))
            or direct_family["binding_sha256"] != next(iter(direct_bindings))
        ):
            raise ValueError("Stage-2 S2-A family promotion binding changed")
        direct_family_record: dict[str, Any] = _path_record(
            direct_path,
            self_fingerprint=direct_family["binding_sha256"],
        )
    else:
        if any(
            family.get("direct_family_promotion_path") is not None
            or family.get("direct_family_promotion_content_sha256") is not None
            or family.get("direct_family_promotion_binding_sha256") is not None
            for family in families.values()
        ):
            raise ValueError(
                "body-only Stage-2 family contains an inapplicable S2-A promotion"
            )
        direct_family_record = {
            "required": False,
            "status": "not_applicable",
            "reason": "body-only action has no direct/racket S2-A endpoint",
        }
    payload: dict[str, Any] = {
        "schema_version": FAMILY_INDEX_SCHEMA_VERSION,
        "action": shared["action"],
        "shared_inputs": _path_record(shared_path, self_fingerprint=shared["binding_sha256"]),
        "architecture_lock": _path_record(lock_path, self_fingerprint=lock["binding_sha256"]),
        "architecture": lock["architecture"],
        "exact_arms": list(STAGE2_ARMS),
        "exact_seeds": list(EXACT_SEEDS),
        "primary_metric": PRIMARY_METRIC,
        "response_evidence_required_for": ["S2-C", "S2-D", "S2-E"],
        "direct_s2a_evidence": shared["direct_s2a_evidence"],
        "direct_s2a_family_promotion": direct_family_record,
        "arms": arms,
    }
    payload["binding_sha256"] = canonical_json_sha256(payload)
    return payload


def validate_stage2_context_family_index(
    source: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(source) if isinstance(source, Mapping) else _load_object(source, label="Stage-2 family index")
    if payload.get("schema_version") != FAMILY_INDEX_SCHEMA_VERSION:
        raise ValueError("unsupported Stage-2 family-index schema")
    supplied = _require_sha256(payload.get("binding_sha256"), label="family index binding_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "binding_sha256"}
    if supplied != canonical_json_sha256(unsigned):
        raise ValueError("Stage-2 family-index binding mismatch")
    arms = payload.get("arms") or {}
    if list(payload.get("exact_arms") or []) != list(STAGE2_ARMS) or set(arms) != set(STAGE2_ARMS):
        raise ValueError("Stage-2 family index does not contain exact B/C/D/E arms")
    rebuilt = build_stage2_context_family_index(
        shared_inputs=(payload.get("shared_inputs") or {}).get("path"),
        architecture_lock=(payload.get("architecture_lock") or {}).get("path"),
        s2b_output_dir=(arms.get("S2-B") or {}).get("output_dir"),
        s2c_output_dir=(arms.get("S2-C") or {}).get("output_dir"),
        s2d_output_dir=(arms.get("S2-D") or {}).get("output_dir"),
        s2e_output_dir=(arms.get("S2-E") or {}).get("output_dir"),
    )
    if rebuilt != payload:
        raise ValueError("Stage-2 family index or one of its sources changed")
    return payload


def build_stage2_context_family_gate(*, family_index: str | Path) -> dict[str, Any]:
    """Apply the pre-registered seed-paired C>D loss gate."""

    index_path = Path(family_index).expanduser().resolve(strict=True)
    index = validate_stage2_context_family_index(index_path)
    c_records = {int(item["seed"]): item for item in index["arms"]["S2-C"]["seeds"]}
    d_records = {int(item["seed"]): item for item in index["arms"]["S2-D"]["seeds"]}
    if set(c_records) != set(EXACT_SEEDS) or set(d_records) != set(EXACT_SEEDS):
        raise ValueError("paired Stage-2 gate requires exact C/D seeds 0/1/2")
    pairs = []
    deltas = []
    for seed in EXACT_SEEDS:
        c_value = float(c_records[seed]["emg_synergy_head_loss"])
        d_value = float(d_records[seed]["emg_synergy_head_loss"])
        delta = d_value - c_value
        if not all(math.isfinite(value) for value in (c_value, d_value, delta)):
            raise ValueError(f"C/D seed-{seed} primary metric is non-finite")
        deltas.append(delta)
        pairs.append(
            {
                "seed": seed,
                "s2c_real_context_loss": c_value,
                "s2d_shuffled_context_loss": d_value,
                "delta_d_minus_c": delta,
                "passed": delta > 0.0,
                "c_response": {
                    key: c_records[seed][key]
                    for key in (
                        "emg_synergy_head_correlation",
                        "emg_blank_context_posterior_mu_l2",
                        "emg_blank_context_action_mse",
                    )
                },
                "d_response": {
                    key: d_records[seed][key]
                    for key in (
                        "emg_synergy_head_correlation",
                        "emg_blank_context_posterior_mu_l2",
                        "emg_blank_context_action_mse",
                    )
                },
            }
        )
    mean = float(np.mean(deltas))
    std = float(np.std(deltas, ddof=1))
    sem = std / math.sqrt(len(deltas))
    t_value = math.inf if std == 0.0 and mean > 0.0 else (0.0 if std == 0.0 else mean / sem)
    dz = math.inf if std == 0.0 and mean > 0.0 else (0.0 if std == 0.0 else mean / std)
    critical_t_df2 = 4.302652729911275
    passed = all(delta > 0.0 for delta in deltas) and mean > 0.0
    payload: dict[str, Any] = {
        "schema_version": FAMILY_GATE_SCHEMA_VERSION,
        "action": index["action"],
        "passed": passed,
        "primary_hypothesis": {
            "metric": PRIMARY_METRIC,
            "direction": "S2-D shuffled loss minus S2-C real-context loss > 0",
            "unit_of_analysis": "seed",
            "exact_seeds": list(EXACT_SEEDS),
            "required_seed_wins": "3/3",
            "pairs": pairs,
            "statistics": {
                "n": len(deltas),
                "mean_delta_d_minus_c": mean,
                "sample_std_delta": std,
                "paired_cohens_dz": dz,
                "paired_t_statistic": t_value,
                "mean_delta_95pct_t_interval": [
                    mean - critical_t_df2 * sem,
                    mean + critical_t_df2 * sem,
                ],
                "exact_one_sided_sign_test_p": 0.125 if all(delta > 0.0 for delta in deltas) else None,
                "significance_claimed": False,
            },
        },
        "completion_contract": {
            "arms": list(STAGE2_ARMS),
            "seeds_per_arm": list(EXACT_SEEDS),
            "same_shared_lineage": True,
            "same_s2b_architecture": True,
            "c_d_only_treatment_difference": "emg_shuffle_context_ablation",
            "positive_blank_context_response_required_for": ["S2-C", "S2-E"],
            "finite_context_head_audit_required_for": ["S2-C", "S2-D", "S2-E"],
        },
        "descriptive_only": {
            "response_metrics": [
                RESPONSE_METRIC,
                "eval_metrics.emg_blank_context_posterior_mu_l2",
                "eval_metrics.emg_blank_context_action_mse",
            ],
            "action_and_closed_loop_metrics": "report only; no post-hoc acceptance threshold",
        },
        "direct_s2a_evidence": index["direct_s2a_evidence"],
        "direct_s2a_family_promotion": index["direct_s2a_family_promotion"],
        "family_index": _path_record(index_path, self_fingerprint=index["binding_sha256"]),
    }
    payload["binding_sha256"] = canonical_json_sha256(payload)
    return payload


def validate_stage2_context_family_gate(
    source: str | Path | Mapping[str, Any],
    *,
    require_pass: bool = True,
) -> dict[str, Any]:
    payload = dict(source) if isinstance(source, Mapping) else _load_object(source, label="Stage-2 family gate")
    if payload.get("schema_version") != FAMILY_GATE_SCHEMA_VERSION:
        raise ValueError("unsupported Stage-2 family-gate schema")
    supplied = _require_sha256(payload.get("binding_sha256"), label="family gate binding_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "binding_sha256"}
    if supplied != canonical_json_sha256(unsigned):
        raise ValueError("Stage-2 family-gate binding mismatch")
    rebuilt = build_stage2_context_family_gate(
        family_index=(payload.get("family_index") or {}).get("path")
    )
    if rebuilt != payload:
        raise ValueError("Stage-2 family gate or its family index changed")
    if require_pass and payload.get("passed") is not True:
        raise ValueError("Stage-2 paired C>D family gate did not pass")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    shared = sub.add_parser("seal-shared", help="seal immutable B/C/D/E physical inputs")
    shared.add_argument("--action", choices=action_choices(), required=True)
    shared.add_argument("--train-dataset-dir", type=Path, required=True)
    shared.add_argument("--val-dataset-dir", type=Path, required=True)
    shared.add_argument("--teacher-checkpoint", type=Path, required=True)
    shared.add_argument("--teacher-promotion-manifest", type=Path, required=True)
    shared.add_argument("--stage1-peasd-promotion-manifest", type=Path, required=True)
    shared.add_argument("--emg-reference-manifest", type=Path, required=True)
    shared.add_argument("--physical-qc-metrics", type=Path, required=True)
    shared.add_argument("--physical-qc-gate", type=Path, required=True)
    shared.add_argument("--synergy-basis", type=Path, required=True)
    shared.add_argument("--frozen-body-decoder", type=Path, required=True)
    shared.add_argument("--direct-bc-metrics", type=Path)
    shared.add_argument("--direct-rollout-metrics", type=Path)
    shared.add_argument("--direct-promotion-evidence", type=Path)
    shared.add_argument("--output", type=Path, required=True)

    lock = sub.add_parser("lock-architecture", help="freeze S2-B best_synergy architecture")
    lock.add_argument("--shared-inputs", type=Path, required=True)
    lock.add_argument("--s2b-output-dir", type=Path, required=True)
    lock.add_argument("--output", type=Path, required=True)

    index = sub.add_parser("index", help="seal exact completed B/C/D/E arms")
    index.add_argument("--shared-inputs", type=Path, required=True)
    index.add_argument("--architecture-lock", type=Path, required=True)
    for arm in ("s2b", "s2c", "s2d", "s2e"):
        index.add_argument(f"--{arm}-output-dir", type=Path, required=True)
    index.add_argument("--output", type=Path, required=True)

    gate = sub.add_parser("gate", help="apply the pre-registered paired C>D gate")
    gate.add_argument("--family-index", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)
    gate.add_argument("--require-pass", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "seal-shared":
        payload = build_stage2_shared_inputs(
            action=args.action,
            train_dataset_dir=args.train_dataset_dir,
            val_dataset_dir=args.val_dataset_dir,
            teacher_checkpoint=args.teacher_checkpoint,
            teacher_promotion_manifest=args.teacher_promotion_manifest,
            stage1_peasd_promotion_manifest=args.stage1_peasd_promotion_manifest,
            emg_reference_manifest=args.emg_reference_manifest,
            physical_qc_metrics=args.physical_qc_metrics,
            physical_qc_gate=args.physical_qc_gate,
            synergy_basis=args.synergy_basis,
            frozen_body_decoder=args.frozen_body_decoder,
            direct_bc_metrics=args.direct_bc_metrics,
            direct_rollout_metrics=args.direct_rollout_metrics,
            direct_promotion_evidence=args.direct_promotion_evidence,
        )
    elif args.command == "lock-architecture":
        payload = build_stage2_s2b_architecture_lock(
            shared_inputs=args.shared_inputs,
            s2b_output_dir=args.s2b_output_dir,
        )
    elif args.command == "index":
        payload = build_stage2_context_family_index(
            shared_inputs=args.shared_inputs,
            architecture_lock=args.architecture_lock,
            s2b_output_dir=args.s2b_output_dir,
            s2c_output_dir=args.s2c_output_dir,
            s2d_output_dir=args.s2d_output_dir,
            s2e_output_dir=args.s2e_output_dir,
        )
    else:
        payload = build_stage2_context_family_gate(family_index=args.family_index)
        if args.require_pass and payload.get("passed") is not True:
            raise ValueError("Stage-2 paired C>D family gate did not pass")
    _write_immutable(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

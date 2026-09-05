"""Seal and evaluate the pre-registered Stage-3 H1/H2/H3 PEASD family.

The statistical unit in this module is an independent training seed.  It does
not promote held-out feeds, episodes, frames, or timesteps to independent
replicates.  Before a comparison can be built it revalidates:

* one passed Stage-2 B/C/D/E context-family gate and its immutable shared
  inputs;
* H1 against the selected S2-B latent checkpoint and H2/H3 against the
  selected S2-C latent checkpoint;
* one complete ``stage3_reachability_release_v1`` lineage for every arm/seed;
* complete ``incoming_shuttle_hit_evaluate_v3`` reports for exact training
  seeds 0/1/2 under one target, feed, scene, and evaluation protocol; and
* H3's bounded right-arm residual treatment while H1/H2 keep it disabled.

The acceptance rules live in the checked-in public comparison contract and
are mirrored byte-for-byte by :data:`PRE_REGISTERED_COMPARISON_CONTRACT`.
Changing a result threshold therefore requires an explicit source change; a
result directory cannot silently supply a post-hoc threshold.

This module never launches training.  Pipeline integration should add the two
steps in :data:`PIPELINE_STEP_NAMES` only after all three branch runs finish.
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

from musclemimic.badminton.action_registry import resolve as resolve_action
from musclemimic.badminton.json_contract import load_json_strict
from musclemimic.badminton.scripts.latent_synergy_sweep import (
    validate_selected_artifact,
)
from musclemimic.badminton.stage2_context_family import (
    validate_stage2_context_family_gate,
    validate_stage2_context_family_index,
)
from musclemimic.badminton.stage3_paired_comparison import (
    _COMMON_BINDING_FIELDS,
    _environment_protocol,
    _validate_evaluation_binding,
)
from musclemimic.badminton.stage3_reachability_release import (
    validate_stage3_reachability_release,
    validate_static_ppo_entry,
    validate_successful_correction_dataset_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPARISON_CONTRACT = (
    REPO_ROOT / "configs/public/stage3_peasd_family_comparison_contract_v1.json"
)

CONTRACT_SCHEMA_VERSION = "stage3_peasd_comparison_contract_v1"
FAMILY_INDEX_SCHEMA_VERSION = "stage3_peasd_family_index_v1"
FAMILY_GATE_SCHEMA_VERSION = "stage3_peasd_family_gate_v1"

EXACT_ARMS = ("H1", "H2", "H3")
EXACT_SEEDS = (0, 1, 2)
ARM_TO_STAGE2 = {"H1": "S2-B", "H2": "S2-C", "H3": "S2-C"}

PIPELINE_ARTIFACT_FIELDS = (
    "stage3_peasd_family_index",
    "stage3_peasd_family_gate",
)
PIPELINE_STEP_NAMES = (
    "stage3_peasd_family_index",
    "stage3_peasd_family_gate",
)

# This is deliberately duplicated in a checked-in JSON file.  Validation
# requires exact equality so result-producing commands cannot choose their own
# endpoint or tolerance after observing outcomes.
PRE_REGISTERED_COMPARISON_CONTRACT: dict[str, Any] = {
    "schema_version": CONTRACT_SCHEMA_VERSION,
    "contract_id": "peasd_stage3_h1_h2_h3_seed_paired_v1",
    "frozen_before_results": True,
    "statistical_unit": "independent_training_seed",
    "exact_training_seeds": list(EXACT_SEEDS),
    "evaluation_protocol": {
        "evaluation_seed": 123,
        "heldout_feed_count": 128,
        "requires_complete_heldout_bank": True,
        "episode_or_frame_as_independent_n": False,
    },
    "arm_definitions": {
        "H1": {
            "stage2_source_arm": "S2-B",
            "latent_treatment": "non_emg_latent_baseline",
            "bounded_residual": "disabled",
        },
        "H2": {
            "stage2_source_arm": "S2-C",
            "latent_treatment": "peasd_latent",
            "bounded_residual": "disabled",
        },
        "H3": {
            "stage2_source_arm": "S2-C",
            "latent_treatment": "peasd_latent",
            "bounded_residual": "required_grouped_right_arm",
        },
    },
    "h2_vs_h1": {
        "primary": {
            "metric": "opponent_back_landing_rate",
            "direction": "higher",
            "per_seed_improvement_strictly_greater_than": 0.0,
            "mean_improvement_strictly_greater_than": 0.0,
        },
        "guardrails": [
            {
                "metric": "hit_rate",
                "direction": "higher",
                "per_seed_minimum_improvement": 0.0,
                "mean_minimum_improvement": 0.0,
            },
            {
                "metric": "no_fall_rate",
                "direction": "higher",
                "per_seed_minimum_improvement": 0.0,
                "mean_minimum_improvement": 0.0,
            },
        ],
    },
    "h3_vs_h2": {
        "primary": {
            "metric": "impact_position_error_m",
            "direction": "lower",
            "per_seed_improvement_strictly_greater_than": 0.0,
            "mean_improvement_strictly_greater_than": 0.0,
        },
        "guardrails": [
            {
                "metric": "hit_rate",
                "direction": "higher",
                "per_seed_minimum_improvement": 0.0,
                "mean_minimum_improvement": 0.0,
            },
            {
                "metric": "no_fall_rate",
                "direction": "higher",
                "per_seed_minimum_improvement": 0.0,
                "mean_minimum_improvement": 0.0,
            },
            {
                "metric": "opponent_back_landing_rate",
                "direction": "higher",
                "per_seed_minimum_improvement": 0.0,
                "mean_minimum_improvement": 0.0,
            },
        ],
    },
    "inference_policy": {
        "n_independent_seeds": 3,
        "report_mean": True,
        "report_sample_standard_deviation": True,
        "report_cohen_dz": True,
        "report_df2_t_interval": True,
        "report_failure_count": True,
        "claim_null_hypothesis_significance": False,
        "claim_population_level_effect": False,
    },
}

_RATE_METRICS = {
    "opponent_back_landing_rate",
    "hit_rate",
    "no_fall_rate",
}
_PROTOCOL_BINDING_FIELDS = tuple(
    field for field in _COMMON_BINDING_FIELDS if field != "training_seed"
)
# These fields implement H3's pre-registered bounded-residual treatment.  All
# other trainer fields, including optimizer, rollout, network, short-BC and
# curriculum settings, are part of the cross-arm matched core.  H1 and H2
# additionally require the complete non-seed config to be identical.
_H3_TREATMENT_CONFIG_FIELDS = frozenset(
    {
        "policy_update_mode",
        "policy_trainable_action_indices",
        "policy_correction_hidden",
        "correction_physical_scales",
        "correction_std_init",
        "correction_std_min",
        "correction_std_max",
        "reset_correction_std_on_actor_initialization",
    }
)
_T_DF2_975 = 4.302652729911275


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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
        raise ValueError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _load_object(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve(strict=True)
    value = load_json_strict(source)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return source, value


def _path_record(
    path: str | Path,
    *,
    artifact_fingerprint: str | None = None,
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    record: dict[str, Any] = {
        "path": str(source),
        "content_sha256": _file_sha256(source),
    }
    if artifact_fingerprint is not None:
        record["artifact_fingerprint"] = _require_sha256(
            artifact_fingerprint, f"{source.name} artifact fingerprint"
        )
    return record


def validate_comparison_contract(
    path: str | Path = DEFAULT_COMPARISON_CONTRACT,
) -> dict[str, Any]:
    """Require the exact public, source-mirrored pre-registration contract."""

    _, contract = _load_object(path, "Stage-3 PEASD comparison contract")
    if contract != PRE_REGISTERED_COMPARISON_CONTRACT:
        raise ValueError(
            "Stage-3 PEASD comparison contract differs from the pre-registered source contract"
        )
    return contract


def _selection_from_stage2_arm(
    index: Mapping[str, Any], arm: str
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    arm_record = (index.get("arms") or {}).get(arm)
    if not isinstance(arm_record, Mapping):
        raise ValueError(f"Stage-2 family index is missing {arm}")
    selection_record = arm_record.get("selection_manifest")
    if not isinstance(selection_record, Mapping):
        raise ValueError(f"{arm} has no selection-manifest binding")
    path = Path(str(selection_record.get("path", ""))).expanduser().resolve(strict=True)
    if selection_record.get("content_sha256") != _file_sha256(path):
        raise ValueError(f"{arm} selection manifest changed after Stage-2 sealing")
    selection = validate_selected_artifact(path)
    entry = (selection.get("checkpoints") or {}).get("best_synergy")
    if not isinstance(entry, Mapping):
        raise ValueError(f"{arm} has no selected best_synergy checkpoint")
    if str(entry.get("decoder_type", "")) == "direct":
        raise ValueError(f"{arm} selected checkpoint is not a synergy latent")
    _require_sha256(entry.get("checkpoint_fingerprint"), f"{arm} selected latent")
    return path, selection, dict(entry)


def _contains_value(value: Any, expected: str) -> bool:
    if value == expected:
        return True
    if isinstance(value, Mapping):
        return any(_contains_value(item, expected) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return any(_contains_value(item, expected) for item in value)
    return False


def _expected_reachability_entry(
    release_path: Path,
    release: Mapping[str, Any],
) -> dict[str, Any]:
    correction_binding = release.get("correction_dataset_manifest")
    short_bc = release.get("short_bc")
    if not isinstance(correction_binding, Mapping) or not isinstance(short_bc, Mapping):
        raise ValueError("Stage-3 reachability release has incomplete lineage")
    correction = validate_successful_correction_dataset_manifest(
        str(correction_binding.get("path", ""))
    )
    correction_dataset = correction.get("correction_dataset")
    checkpoint = short_bc.get("checkpoint")
    if not isinstance(correction_dataset, Mapping) or not isinstance(checkpoint, Mapping):
        raise ValueError("Stage-3 reachability release has incomplete BC inputs")
    payload_path = Path(str(checkpoint.get("payload_path", ""))).resolve(strict=True)
    return validate_static_ppo_entry(
        release_path=release_path,
        start_checkpoint=str(
            checkpoint.get("pointer_path") or checkpoint.get("payload_path") or ""
        ),
        teacher_dataset=str(correction_dataset.get("path", "")),
        runtime_run_dir=str(payload_path.parent.parent.parent),
        runtime_control_manifest=dict(short_bc.get("runtime_control_manifest") or {}),
        runtime_training_feed_manifest=dict(
            short_bc.get("runtime_training_feed_manifest") or {}
        ),
    )


def _verify_reachability_lineage(
    *,
    release_path: Path,
    expected_action: Any,
    expected_latent_fingerprint: str,
    report: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    release = validate_stage3_reachability_release(release_path)
    latent = release.get("latent_identity")
    if not isinstance(latent, Mapping) or latent != {
        "kind": "latent_checkpoint",
        "fingerprint": expected_latent_fingerprint,
    }:
        raise ValueError("Stage-3 reachability release uses the wrong latent identity")
    spec = release.get("spec_identity")
    if not isinstance(spec, Mapping):
        raise ValueError("Stage-3 reachability release has no spec identity")
    if (
        spec.get("spec_sha256") != binding.get("spec_sha256")
        or spec.get("scene_sha256") != binding.get("scene_sha256")
    ):
        raise ValueError("Stage-3 reachability and evaluation use different spec/scene")
    target = release.get("target_identity")
    if not isinstance(target, Mapping):
        raise ValueError("Stage-3 reachability release has no target identity")
    if (
        target.get("action") != expected_action.slug
        or target.get("dataset_action_id") != expected_action.action_id
    ):
        raise ValueError(
            "Stage-3 reachability release belongs to a different action"
        )
    single_feed = _require_sha256(
        target.get("single_feed_fingerprint"),
        "Stage-3 reachability single-feed target",
    )
    if not _contains_value(report.get("training_feed_manifest"), single_feed):
        raise ValueError("Stage-3 evaluation training feed omits its reachability target")

    metadata_path = Path(str(binding.get("checkpoint_metadata_path", ""))).resolve(
        strict=True
    )
    metadata = load_json_strict(metadata_path)
    if not isinstance(metadata, Mapping):
        raise ValueError("Stage-3 evaluation checkpoint metadata must be an object")
    prerequisite = metadata.get("training_prerequisite_binding")
    if not isinstance(prerequisite, Mapping):
        raise ValueError("Stage-3 evaluation checkpoint has no prerequisite lineage")
    supplied = prerequisite.get("binding_sha256")
    unsigned = dict(prerequisite)
    unsigned.pop("binding_sha256", None)
    if supplied != _canonical_sha256(unsigned):
        raise ValueError("Stage-3 evaluation prerequisite binding is stale")
    if binding.get("training_prerequisite_binding_sha256") != supplied:
        raise ValueError("Stage-3 evaluation report binds a different prerequisite lineage")
    expected_entry = _expected_reachability_entry(release_path, release)
    if prerequisite.get("stage3_reachability_release") != expected_entry:
        raise ValueError("Stage-3 evaluation checkpoint uses the wrong reachability release")
    correction_binding = release.get("correction_dataset_manifest") or {}
    correction = validate_successful_correction_dataset_manifest(
        str(correction_binding.get("path", ""))
    )
    source_checkpoint = correction.get("source_checkpoint") or {}
    cem = correction.get("cem") or {}
    cpu_audit = correction.get("cpu_audit") or {}
    correction_dataset = correction.get("correction_dataset") or {}
    short_checkpoint = (release.get("short_bc") or {}).get("checkpoint") or {}

    def recorded_path(record: Any, label: str) -> str:
        if not isinstance(record, Mapping):
            raise ValueError(f"Stage-3 reachability lineage has no {label}")
        value = record.get("path") or record.get("payload_path")
        if not isinstance(value, str) or not value:
            raise ValueError(f"Stage-3 reachability lineage has no {label} path")
        return str(Path(value).expanduser().resolve(strict=True))

    short_payload = recorded_path(short_checkpoint, "short-BC checkpoint")
    return {
        "release": release,
        "entry": expected_entry,
        "target_identity": dict(target),
        "spec_identity": dict(spec),
        "checkpoint_metadata": dict(metadata),
        "leaf_lineage": {
            "source_checkpoint": recorded_path(
                source_checkpoint, "source checkpoint"
            ),
            "cem_report": recorded_path(cem.get("report"), "CEM report"),
            "cem_candidate": recorded_path(
                cem.get("candidate"), "CEM candidate"
            ),
            "cpu_audit_trace": recorded_path(
                cpu_audit.get("trace"), "CPU audit trace"
            ),
            "correction_dataset": recorded_path(
                correction_dataset, "correction dataset"
            ),
            "short_bc_checkpoint": short_payload,
            "short_bc_run_root": str(
                Path(short_payload).parent.parent.parent.resolve(strict=True)
            ),
        },
    }


def _training_config_contract(
    metadata: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
) -> dict[str, Any]:
    config = metadata.get("config")
    if not isinstance(config, Mapping):
        raise ValueError(f"{arm}/seed-{seed} checkpoint has no training config")
    if isinstance(config.get("seed"), bool):
        raise ValueError(f"{arm}/seed-{seed} checkpoint has an invalid config seed")
    try:
        config_seed = int(config.get("seed"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{arm}/seed-{seed} checkpoint has an invalid config seed"
        ) from exc
    if config_seed != seed or float(config.get("seed")) != float(seed):
        raise ValueError(f"{arm}/seed-{seed} checkpoint config uses the wrong seed")
    arm_core = {key: value for key, value in config.items() if key != "seed"}
    shared_core = {
        key: value
        for key, value in arm_core.items()
        if key not in _H3_TREATMENT_CONFIG_FIELDS
    }
    return {
        "shared_core": shared_core,
        "shared_core_sha256": _canonical_sha256(shared_core),
        "arm_core": arm_core,
        "arm_core_sha256": _canonical_sha256(arm_core),
        "h3_treatment_fields": {
            key: arm_core.get(key)
            for key in sorted(_H3_TREATMENT_CONFIG_FIELDS)
            if key in arm_core
        },
    }


def _validate_residual_treatment(control: Mapping[str, Any], arm: str) -> dict[str, Any]:
    try:
        size = int(control.get("bounded_residual_dim", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{arm} has no bounded-residual dimension") from exc
    schema_hash = control.get("bounded_residual_schema_hash")
    groups = control.get("bounded_residual_groups")
    if arm in {"H1", "H2"}:
        if size != 0 or schema_hash is not None or groups is not None:
            raise ValueError(f"{arm} must disable the bounded residual")
        return {"enabled": False, "dimension": 0, "schema_sha256": None, "groups": None}
    if size <= 0 or not isinstance(groups, list) or not groups:
        raise ValueError("H3 must enable a non-empty grouped bounded residual")
    fingerprint = _require_sha256(schema_hash, "H3 bounded-residual schema")
    for group in groups:
        if not isinstance(group, Mapping):
            raise ValueError("H3 bounded-residual groups must be structured records")
        alpha = _require_finite(group.get("alpha"), "H3 bounded-residual alpha")
        names = group.get("actuator_names")
        if not 0.0 <= alpha <= 0.10 or not isinstance(names, list) or not names:
            raise ValueError("H3 bounded residual exceeds authority or has an empty group")
    return {
        "enabled": True,
        "dimension": size,
        "schema_sha256": fingerprint,
        "groups": groups,
    }


def _normalize_sources(
    values: Mapping[str, Mapping[int, str | Path]],
    *,
    label: str,
) -> dict[str, dict[int, Path]]:
    if set(values) != set(EXACT_ARMS):
        raise ValueError(f"{label} must contain exact H1/H2/H3 arms")
    result: dict[str, dict[int, Path]] = {}
    for arm in EXACT_ARMS:
        by_seed = values[arm]
        if set(by_seed) != set(EXACT_SEEDS):
            raise ValueError(f"{label} {arm} must contain exact seeds 0/1/2")
        result[arm] = {
            seed: Path(by_seed[seed]).expanduser().resolve(strict=True)
            for seed in EXACT_SEEDS
        }
    flattened = [path for arm in EXACT_ARMS for path in result[arm].values()]
    if len(set(flattened)) != len(flattened):
        raise ValueError(f"{label} paths must be unique per arm and seed")
    return result


def build_stage3_peasd_family_index(
    *,
    stage2_family_gate: str | Path,
    reports: Mapping[str, Mapping[int, str | Path]],
    reachability_releases: Mapping[str, Mapping[int, str | Path]],
    comparison_contract: str | Path = DEFAULT_COMPARISON_CONTRACT,
) -> dict[str, Any]:
    """Build a source-bound index for the exact H1/H2/H3 x seed family."""

    contract_path = Path(comparison_contract).expanduser().resolve(strict=True)
    contract = validate_comparison_contract(contract_path)
    gate_path = Path(stage2_family_gate).expanduser().resolve(strict=True)
    stage2_gate = validate_stage2_context_family_gate(gate_path, require_pass=True)
    family_index_record = stage2_gate.get("family_index")
    if not isinstance(family_index_record, Mapping):
        raise ValueError("Stage-2 family gate has no family-index binding")
    stage2_index_path = Path(str(family_index_record.get("path", ""))).resolve(
        strict=True
    )
    stage2_index = validate_stage2_context_family_index(stage2_index_path)
    if family_index_record.get("content_sha256") != _file_sha256(stage2_index_path):
        raise ValueError("Stage-2 family index changed after gating")
    if family_index_record.get("artifact_fingerprint") != stage2_index.get(
        "binding_sha256"
    ):
        raise ValueError("Stage-2 family gate binds a different family index")
    action = stage2_index.get("action")
    if not isinstance(action, Mapping):
        raise ValueError("Stage-2 family index has no action identity")
    action_spec = resolve_action(str(action.get("slug", "")))
    if not action_spec.stage3_applicable:
        raise ValueError(f"{action_spec.slug} has no Stage-3 hitting endpoint")

    selection_records: dict[str, Any] = {}
    selection_entries: dict[str, dict[str, Any]] = {}
    for stage2_arm in ("S2-B", "S2-C"):
        path, selection, entry = _selection_from_stage2_arm(
            stage2_index, stage2_arm
        )
        selection_records[stage2_arm] = {
            **_path_record(
                path,
                artifact_fingerprint=selection["selection_manifest_fingerprint"],
            ),
            "selected_checkpoint_fingerprint": entry["checkpoint_fingerprint"],
            "decoder_type": entry.get("decoder_type"),
            "latent_dim": entry.get("latent_dim"),
        }
        selection_entries[stage2_arm] = entry
    h1_latent = selection_entries["S2-B"]["checkpoint_fingerprint"]
    h2_latent = selection_entries["S2-C"]["checkpoint_fingerprint"]
    if h1_latent == h2_latent:
        raise ValueError("H1 and H2 resolve to the same latent checkpoint")

    report_paths = _normalize_sources(reports, label="Stage-3 evaluation reports")
    release_paths = _normalize_sources(
        reachability_releases, label="Stage-3 reachability releases"
    )
    all_sources = [
        *[path for arm in EXACT_ARMS for path in report_paths[arm].values()],
        *[path for arm in EXACT_ARMS for path in release_paths[arm].values()],
    ]
    if len(set(all_sources)) != len(all_sources):
        raise ValueError("Stage-3 reports and reachability releases must be distinct")

    protocol_reference: dict[str, Any] | None = None
    feed_reference: dict[str, Any] | None = None
    target_reference: dict[str, Any] | None = None
    spec_reference: dict[str, Any] | None = None
    shared_training_config_reference: dict[str, Any] | None = None
    non_residual_training_config_reference: dict[str, Any] | None = None
    leaf_lineages: list[dict[str, Any]] = []
    arms: dict[str, Any] = {}
    required_count = int(contract["evaluation_protocol"]["heldout_feed_count"])
    required_eval_seed = int(contract["evaluation_protocol"]["evaluation_seed"])
    for arm in EXACT_ARMS:
        stage2_arm = ARM_TO_STAGE2[arm]
        latent_fingerprint = selection_entries[stage2_arm][
            "checkpoint_fingerprint"
        ]
        seeds: list[dict[str, Any]] = []
        residual_reference: dict[str, Any] | None = None
        arm_training_config_reference: dict[str, Any] | None = None
        for seed in EXACT_SEEDS:
            report_path = report_paths[arm][seed]
            _, report = _load_object(report_path, f"{arm}/seed-{seed} evaluation")
            validated = _validate_evaluation_binding(
                report,
                report_path=report_path,
                family=f"{arm}/seed-{seed}",
                expected_action_family="fixed_synergy",
                expected_latent_fingerprint=latent_fingerprint,
            )
            binding = validated["binding"]
            if binding.get("training_seed") != seed:
                raise ValueError(f"{arm}/seed-{seed} uses the wrong training seed")
            if report.get("evaluation_seed") != required_eval_seed:
                raise ValueError(f"{arm}/seed-{seed} uses the wrong evaluation seed")
            if report.get("evaluation_feed_source") != "heldout_evaluation_bank":
                raise ValueError(f"{arm}/seed-{seed} is not a held-out evaluation")
            if (
                report.get("evaluated_feed_count") != required_count
                or report.get("required_heldout_feed_count") != required_count
                or len(validated["episode_indices"]) != required_count
            ):
                raise ValueError(
                    f"{arm}/seed-{seed} did not evaluate the complete fixed held-out bank"
                )
            control = report.get("control_manifest")
            if not isinstance(control, Mapping):
                raise ValueError(f"{arm}/seed-{seed} has no control manifest")
            residual = _validate_residual_treatment(control, arm)
            if residual_reference is None:
                residual_reference = residual
            elif residual != residual_reference:
                raise ValueError(f"{arm} changes bounded-residual treatment across seeds")

            reachability = _verify_reachability_lineage(
                release_path=release_paths[arm][seed],
                expected_action=action_spec,
                expected_latent_fingerprint=latent_fingerprint,
                report=report,
                binding=binding,
            )
            training_config = _training_config_contract(
                reachability["checkpoint_metadata"],
                arm=arm,
                seed=seed,
            )
            if arm_training_config_reference is None:
                arm_training_config_reference = training_config["arm_core"]
            elif training_config["arm_core"] != arm_training_config_reference:
                raise ValueError(f"{arm} changes training config across seeds")
            if shared_training_config_reference is None:
                shared_training_config_reference = training_config["shared_core"]
            elif training_config["shared_core"] != shared_training_config_reference:
                raise ValueError(
                    "H1/H2/H3 change matched PPO or network hyperparameters"
                )
            if arm in {"H1", "H2"}:
                if non_residual_training_config_reference is None:
                    non_residual_training_config_reference = training_config[
                        "arm_core"
                    ]
                elif (
                    training_config["arm_core"]
                    != non_residual_training_config_reference
                ):
                    raise ValueError(
                        "H1/H2 non-residual training configs differ beyond seed"
                    )
            leaf_lineages.append(
                {
                    "arm": arm,
                    "seed": seed,
                    **reachability["leaf_lineage"],
                }
            )
            current_protocol = {
                "binding": {
                    field: binding.get(field)
                    for field in _PROTOCOL_BINDING_FIELDS
                },
                "spec_sha256": binding.get("spec_sha256"),
                "environment": _environment_protocol(report, f"{arm}/seed-{seed}"),
                "training_feed_manifest": report.get("training_feed_manifest"),
                "evaluation_feed_manifest": report.get("evaluation_feed_manifest"),
                "evaluation_feed_sample_fingerprints": list(
                    report["evaluation_feed_manifest"]["sample_fingerprints"][
                        :required_count
                    ]
                ),
                "episode_indices": list(validated["episode_indices"]),
            }
            if protocol_reference is None:
                protocol_reference = current_protocol
                feed_reference = {
                    "training": report.get("training_feed_manifest"),
                    "evaluation": report.get("evaluation_feed_manifest"),
                }
                target_reference = reachability["target_identity"]
                spec_reference = reachability["spec_identity"]
            elif current_protocol != protocol_reference:
                raise ValueError(
                    "H1/H2/H3 Stage-3 target, feed, or evaluation protocols differ"
                )
            if reachability["target_identity"] != target_reference:
                raise ValueError("H1/H2/H3 reachability releases use different target feeds")
            if reachability["spec_identity"] != spec_reference:
                raise ValueError("H1/H2/H3 reachability releases use different specs")

            seeds.append(
                {
                    "seed": seed,
                    "evaluation_report": _path_record(
                        report_path,
                        artifact_fingerprint=binding["binding_sha256"],
                    ),
                    "stage3_checkpoint_payload_sha256": binding[
                        "checkpoint_payload_sha256"
                    ],
                    "stage3_checkpoint_metadata_sha256": binding[
                        "checkpoint_metadata_sha256"
                    ],
                    "latent_checkpoint_fingerprint": latent_fingerprint,
                    "reachability_release": _path_record(
                        release_paths[arm][seed],
                        artifact_fingerprint=reachability["release"][
                            "release_binding_sha256"
                        ],
                    ),
                    "reachability_entry_binding_sha256": reachability["entry"][
                        "binding_sha256"
                    ],
                    "training_config": {
                        "shared_core_sha256": training_config[
                            "shared_core_sha256"
                        ],
                        "arm_core_sha256": training_config["arm_core_sha256"],
                        "h3_treatment_fields": training_config[
                            "h3_treatment_fields"
                        ],
                    },
                    "reachability_lineage": reachability["leaf_lineage"],
                    "metrics": {
                        name: _require_finite(
                            report.get(name), f"{arm}/seed-{seed} {name}"
                        )
                        for name in _contract_metric_names(contract)
                    },
                }
            )
        arms[arm] = {
            "stage2_source_arm": stage2_arm,
            "selection": selection_records[stage2_arm],
            "latent_checkpoint_fingerprint": latent_fingerprint,
            "bounded_residual": residual_reference,
            "training_config_core_sha256": _canonical_sha256(
                arm_training_config_reference or {}
            ),
            "seeds": seeds,
        }

    assert protocol_reference is not None
    assert feed_reference is not None
    assert shared_training_config_reference is not None
    for field in (
        "source_checkpoint",
        "cem_report",
        "cem_candidate",
        "cpu_audit_trace",
        "correction_dataset",
        "short_bc_checkpoint",
        "short_bc_run_root",
    ):
        values = [str(record[field]) for record in leaf_lineages]
        if len(values) != len(set(values)):
            raise ValueError(
                f"Stage-3 PEASD leaves reuse internal reachability lineage: {field}"
            )
    payload: dict[str, Any] = {
        "schema_version": FAMILY_INDEX_SCHEMA_VERSION,
        "action": dict(action),
        "exact_arms": list(EXACT_ARMS),
        "exact_training_seeds": list(EXACT_SEEDS),
        "stage2_context_family": {
            "gate": _path_record(
                gate_path,
                artifact_fingerprint=stage2_gate["binding_sha256"],
            ),
            "index": _path_record(
                stage2_index_path,
                artifact_fingerprint=stage2_index["binding_sha256"],
            ),
            "shared_inputs": stage2_index["shared_inputs"],
            "selections": selection_records,
        },
        "comparison_contract": {
            **_path_record(contract_path),
            "contract_sha256": _canonical_sha256(contract),
        },
        "shared_evaluation_protocol": protocol_reference,
        "shared_training_config": {
            "matched_core": shared_training_config_reference,
            "matched_core_sha256": _canonical_sha256(
                shared_training_config_reference
            ),
            "h3_only_allowed_treatment_fields": sorted(
                _H3_TREATMENT_CONFIG_FIELDS
            ),
            "h1_h2_exact_non_seed_config_sha256": _canonical_sha256(
                non_residual_training_config_reference or {}
            ),
        },
        "shared_feed_manifests": feed_reference,
        "shared_reachability_target": target_reference,
        "shared_spec_identity": spec_reference,
        "arms": arms,
        "independence_statement": {
            "unit": "independent_training_seed",
            "n": len(EXACT_SEEDS),
            "episodes_are_repeated_measurements_not_independent_n": True,
        },
    }
    payload["binding_sha256"] = _canonical_sha256(payload)
    return payload


def _contract_metric_names(contract: Mapping[str, Any]) -> tuple[str, ...]:
    names: set[str] = set()
    for contrast in ("h2_vs_h1", "h3_vs_h2"):
        spec = contract.get(contrast)
        if not isinstance(spec, Mapping):
            raise ValueError(f"comparison contract is missing {contrast}")
        primary = spec.get("primary")
        guardrails = spec.get("guardrails")
        if not isinstance(primary, Mapping) or not isinstance(guardrails, list):
            raise ValueError(f"comparison contract has malformed {contrast}")
        names.add(str(primary.get("metric", "")))
        for guardrail in guardrails:
            if not isinstance(guardrail, Mapping):
                raise ValueError(f"comparison contract has malformed {contrast} guardrail")
            names.add(str(guardrail.get("metric", "")))
    if "" in names:
        raise ValueError("comparison contract has an empty metric")
    return tuple(sorted(names))


def validate_stage3_peasd_family_index(
    path: str | Path,
) -> dict[str, Any]:
    source, payload = _load_object(path, "Stage-3 PEASD family index")
    if payload.get("schema_version") != FAMILY_INDEX_SCHEMA_VERSION:
        raise ValueError("unsupported Stage-3 PEASD family-index schema")
    supplied = _require_sha256(
        payload.get("binding_sha256"), "Stage-3 PEASD family-index binding"
    )
    unsigned = dict(payload)
    unsigned.pop("binding_sha256", None)
    if supplied != _canonical_sha256(unsigned):
        raise ValueError("Stage-3 PEASD family-index binding mismatch")
    if list(payload.get("exact_arms") or []) != list(EXACT_ARMS):
        raise ValueError("Stage-3 PEASD family index has the wrong arms")
    arms = payload.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != set(EXACT_ARMS):
        raise ValueError("Stage-3 PEASD family index lacks an exact arm")
    reports: dict[str, dict[int, str]] = {}
    releases: dict[str, dict[int, str]] = {}
    for arm in EXACT_ARMS:
        records = arms[arm].get("seeds") if isinstance(arms[arm], Mapping) else None
        if not isinstance(records, list) or [item.get("seed") for item in records] != list(
            EXACT_SEEDS
        ):
            raise ValueError(f"Stage-3 PEASD family index has invalid {arm} seeds")
        reports[arm] = {
            int(item["seed"]): item["evaluation_report"]["path"] for item in records
        }
        releases[arm] = {
            int(item["seed"]): item["reachability_release"]["path"]
            for item in records
        }
    stage2 = payload.get("stage2_context_family") or {}
    contract = payload.get("comparison_contract") or {}
    rebuilt = build_stage3_peasd_family_index(
        stage2_family_gate=(stage2.get("gate") or {}).get("path"),
        reports=reports,
        reachability_releases=releases,
        comparison_contract=contract.get("path"),
    )
    if rebuilt != payload:
        raise ValueError(
            f"Stage-3 PEASD family index or a bound source changed: {source}"
        )
    return payload


def _seed_summary(
    *,
    index: Mapping[str, Any],
    candidate_arm: str,
    reference_arm: str,
    metric_contract: Mapping[str, Any],
    primary: bool,
) -> dict[str, Any]:
    metric = str(metric_contract.get("metric", ""))
    direction = str(metric_contract.get("direction", ""))
    if direction not in {"higher", "lower"}:
        raise ValueError(f"invalid direction for {metric}")
    sign = 1.0 if direction == "higher" else -1.0
    candidate = {
        int(item["seed"]): item
        for item in index["arms"][candidate_arm]["seeds"]
    }
    reference = {
        int(item["seed"]): item
        for item in index["arms"][reference_arm]["seeds"]
    }
    pairs: list[dict[str, Any]] = []
    improvements: list[float] = []
    if primary:
        per_seed_threshold = _require_finite(
            metric_contract.get("per_seed_improvement_strictly_greater_than"),
            f"{metric} per-seed threshold",
        )
        mean_threshold = _require_finite(
            metric_contract.get("mean_improvement_strictly_greater_than"),
            f"{metric} mean threshold",
        )
        def predicate(value: float, threshold: float) -> bool:
            return value > threshold

        rule = "strictly_greater_than"
    else:
        per_seed_threshold = _require_finite(
            metric_contract.get("per_seed_minimum_improvement"),
            f"{metric} per-seed guardrail",
        )
        mean_threshold = _require_finite(
            metric_contract.get("mean_minimum_improvement"),
            f"{metric} mean guardrail",
        )
        def predicate(value: float, threshold: float) -> bool:
            return value >= threshold

        rule = "greater_than_or_equal"
    for seed in EXACT_SEEDS:
        candidate_value = _require_finite(
            candidate[seed]["metrics"].get(metric),
            f"{candidate_arm}/seed-{seed} {metric}",
        )
        reference_value = _require_finite(
            reference[seed]["metrics"].get(metric),
            f"{reference_arm}/seed-{seed} {metric}",
        )
        if metric in _RATE_METRICS and not (
            0.0 <= candidate_value <= 1.0 and 0.0 <= reference_value <= 1.0
        ):
            raise ValueError(f"{metric} must lie in [0,1]")
        if metric.endswith("_m") and (
            candidate_value < 0.0 or reference_value < 0.0
        ):
            raise ValueError(f"{metric} cannot be negative")
        raw_delta = candidate_value - reference_value
        improvement = sign * raw_delta
        passed = predicate(improvement, per_seed_threshold)
        improvements.append(improvement)
        pairs.append(
            {
                "seed": seed,
                "reference": reference_value,
                "candidate": candidate_value,
                "candidate_minus_reference": raw_delta,
                "improvement": improvement,
                "passed": passed,
            }
        )
    values = np.asarray(improvements, dtype=np.float64)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    sem = std / math.sqrt(len(values))
    effect = None if std == 0.0 else mean / std
    interval = [mean - _T_DF2_975 * sem, mean + _T_DF2_975 * sem]
    seed_failures = sum(not pair["passed"] for pair in pairs)
    mean_passed = predicate(mean, mean_threshold)
    passed = seed_failures == 0 and mean_passed
    return {
        "metric": metric,
        "direction": direction,
        "candidate_arm": candidate_arm,
        "reference_arm": reference_arm,
        "threshold_rule": rule,
        "per_seed_threshold": per_seed_threshold,
        "mean_threshold": mean_threshold,
        "seed_pairs": pairs,
        "statistics": {
            "unit": "independent_training_seed",
            "n": 3,
            "mean_improvement": mean,
            "sample_standard_deviation": std,
            "cohen_dz": effect,
            "degrees_of_freedom": 2,
            "t_df2_95_interval": interval,
            "seed_failure_count": seed_failures,
            "mean_rule_passed": mean_passed,
            "significance_claimed": False,
        },
        "passed": passed,
    }


def _contrast(
    *,
    index: Mapping[str, Any],
    contract: Mapping[str, Any],
    candidate_arm: str,
    reference_arm: str,
) -> dict[str, Any]:
    primary = _seed_summary(
        index=index,
        candidate_arm=candidate_arm,
        reference_arm=reference_arm,
        metric_contract=contract["primary"],
        primary=True,
    )
    guardrails = [
        _seed_summary(
            index=index,
            candidate_arm=candidate_arm,
            reference_arm=reference_arm,
            metric_contract=item,
            primary=False,
        )
        for item in contract["guardrails"]
    ]
    return {
        "candidate_arm": candidate_arm,
        "reference_arm": reference_arm,
        "primary": primary,
        "guardrails": guardrails,
        "passed": primary["passed"] and all(item["passed"] for item in guardrails),
    }


def build_stage3_peasd_family_gate(*, family_index: str | Path) -> dict[str, Any]:
    index_path = Path(family_index).expanduser().resolve(strict=True)
    index = validate_stage3_peasd_family_index(index_path)
    contract_path = Path(index["comparison_contract"]["path"])
    contract = validate_comparison_contract(contract_path)
    h2_vs_h1 = _contrast(
        index=index,
        contract=contract["h2_vs_h1"],
        candidate_arm="H2",
        reference_arm="H1",
    )
    h3_vs_h2 = _contrast(
        index=index,
        contract=contract["h3_vs_h2"],
        candidate_arm="H3",
        reference_arm="H2",
    )
    payload: dict[str, Any] = {
        "schema_version": FAMILY_GATE_SCHEMA_VERSION,
        "passed": h2_vs_h1["passed"] and h3_vs_h2["passed"],
        "primary_scientific_claim_gate": "H2_vs_H1",
        "h2_vs_h1": h2_vs_h1,
        "h3_vs_h2": h3_vs_h2,
        "family_index": _path_record(
            index_path, artifact_fingerprint=index["binding_sha256"]
        ),
        "comparison_contract": index["comparison_contract"],
        "statistical_scope": {
            "independent_unit": "training_seed",
            "n": 3,
            "degrees_of_freedom": 2,
            "episodes_and_frames_are_not_counted_as_n": True,
            "significance_claimed": False,
            "population_level_claim_allowed": False,
        },
    }
    payload["binding_sha256"] = _canonical_sha256(payload)
    return payload


def validate_stage3_peasd_family_gate(
    path: str | Path,
    *,
    require_pass: bool = True,
) -> dict[str, Any]:
    source, payload = _load_object(path, "Stage-3 PEASD family gate")
    if payload.get("schema_version") != FAMILY_GATE_SCHEMA_VERSION:
        raise ValueError("unsupported Stage-3 PEASD family-gate schema")
    supplied = _require_sha256(
        payload.get("binding_sha256"), "Stage-3 PEASD family-gate binding"
    )
    unsigned = dict(payload)
    unsigned.pop("binding_sha256", None)
    if supplied != _canonical_sha256(unsigned):
        raise ValueError("Stage-3 PEASD family-gate binding mismatch")
    rebuilt = build_stage3_peasd_family_gate(
        family_index=(payload.get("family_index") or {}).get("path")
    )
    if rebuilt != payload:
        raise ValueError(f"Stage-3 PEASD family gate is stale: {source}")
    if require_pass and payload.get("passed") is not True:
        raise ValueError("Stage-3 PEASD H1/H2/H3 gate did not pass")
    return payload


def _parse_seed_paths(values: Sequence[str], label: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        seed_text, separator, path_text = str(value).partition("=")
        if not separator or not path_text:
            raise ValueError(f"{label} entries must use SEED=PATH")
        try:
            seed = int(seed_text)
        except ValueError as exc:
            raise ValueError(f"{label} has a non-integer seed") from exc
        if seed in result:
            raise ValueError(f"{label} repeats seed {seed}")
        result[seed] = Path(path_text)
    if set(result) != set(EXACT_SEEDS):
        raise ValueError(f"{label} requires exact seeds 0/1/2")
    return result


def _write_immutable(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output.exists():
        if output.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(
                f"refusing to replace immutable Stage-3 family artifact: {output}"
            )
        return output
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    index = sub.add_parser("index", help="seal exact H1/H2/H3 x seeds 0/1/2")
    index.add_argument("--stage2-family-gate", type=Path, required=True)
    index.add_argument(
        "--comparison-contract",
        type=Path,
        default=DEFAULT_COMPARISON_CONTRACT,
    )
    for arm in ("h1", "h2", "h3"):
        index.add_argument(
            f"--{arm}-report",
            action="append",
            default=[],
            metavar="SEED=PATH",
        )
        index.add_argument(
            f"--{arm}-reachability-release",
            action="append",
            default=[],
            metavar="SEED=PATH",
        )
    index.add_argument("--output", type=Path, required=True)
    gate = sub.add_parser("gate", help="apply the frozen seed-level comparisons")
    gate.add_argument("--family-index", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)
    gate.add_argument(
        "--require-pass", action=argparse.BooleanOptionalAction, default=True
    )
    validate = sub.add_parser("validate", help="revalidate a family artifact")
    validate.add_argument("--family-index", type=Path)
    validate.add_argument("--family-gate", type=Path)
    validate.add_argument(
        "--require-pass", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "index":
        reports = {
            arm.upper(): _parse_seed_paths(
                getattr(args, f"{arm}_report"), f"{arm.upper()} reports"
            )
            for arm in ("h1", "h2", "h3")
        }
        releases = {
            arm.upper(): _parse_seed_paths(
                getattr(args, f"{arm}_reachability_release"),
                f"{arm.upper()} reachability releases",
            )
            for arm in ("h1", "h2", "h3")
        }
        payload = build_stage3_peasd_family_index(
            stage2_family_gate=args.stage2_family_gate,
            reports=reports,
            reachability_releases=releases,
            comparison_contract=args.comparison_contract,
        )
        output = _write_immutable(args.output, payload)
        validate_stage3_peasd_family_index(output)
    elif args.command == "gate":
        payload = build_stage3_peasd_family_gate(family_index=args.family_index)
        if args.require_pass and payload.get("passed") is not True:
            raise ValueError("Stage-3 PEASD H1/H2/H3 gate did not pass")
        output = _write_immutable(args.output, payload)
        validate_stage3_peasd_family_gate(output, require_pass=args.require_pass)
    else:
        if bool(args.family_index) == bool(args.family_gate):
            raise ValueError("validate requires exactly one family artifact")
        if args.family_index:
            output = Path(args.family_index)
            validate_stage3_peasd_family_index(output)
        else:
            output = Path(args.family_gate)
            validate_stage3_peasd_family_gate(
                output, require_pass=args.require_pass
            )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

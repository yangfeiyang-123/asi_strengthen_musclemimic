"""Strict, model-bound anatomical muscle taxonomy contracts.

The four taxonomies in this module are deliberately non-interchangeable:

``hard_line_group``
    Verified numerical lines of one anatomical muscle.  Only this relationship
    can ever declare ``training_enabled=true``.
``soft_compartment_group``
    Anatomical compartments or heads retained for diagnostics only.
``observation_aggregate``
    Projection channels such as surface EMG; never a training group.
``functional_synergy_region``
    Functional/NMF partitions; never anatomical equality constraints.

Loading and runtime binding happen outside JIT.  The compiled
:class:`~musclemimic.physiology.intra_muscle.IntraMuscleSpec` contains only
fixed-shape arrays.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

from musclemimic.distill.action_schema import actuator_schema_hash
from musclemimic.distill.physical import MUSCLE_CHANNEL_CORE_ABI_SCHEMA_VERSION
from musclemimic.physiology.effective_excitation import (
    EFFECTIVE_EXCITATION_SEMANTICS,
    MUSCLE_ACTIVATION_SEMANTICS,
    UNIT_MUSCLE_CTRLRANGE,
    actuator_transmission_target,
    resolve_muscle_channel_layout,
)
from musclemimic.physiology.intra_muscle import IntraMuscleSpec

ANATOMICAL_TAXONOMY_V1_SCHEMA_VERSION = "anatomical_muscle_grouping_v1"
ANATOMICAL_TAXONOMY_SCHEMA_VERSION = "anatomical_muscle_grouping_v2"
COMPILED_RUNTIME_HASH_SEMANTICS = "mujoco_model_getstate_same_build_diagnostic_only"
_SUPPORTED_TAXONOMY_SCHEMA_VERSIONS = frozenset(
    {
        ANATOMICAL_TAXONOMY_V1_SCHEMA_VERSION,
        ANATOMICAL_TAXONOMY_SCHEMA_VERSION,
    }
)
# NMF basis-conditioning partition (class-D functional_synergy_region prior).  It
# is intentionally NOT loadable as an anatomical taxonomy; see the guard in
# validate_anatomical_taxonomy.
REGIONAL_GROUPING_SCHEMA_VERSION = "regional_muscle_grouping_v1"
DEFAULT_TRAINING_BEHAVIOR = "diagnostics_only_no_reward"
HARD_LINE_RELATIONSHIP = "hard_line_group"
SOFT_COMPARTMENT_RELATIONSHIP = "soft_compartment_group"
OBSERVATION_AGGREGATE_RELATIONSHIP = "observation_aggregate"
FUNCTIONAL_SYNERGY_RELATIONSHIP = "functional_synergy_region"
EXACT_RUNTIME_MODEL_COMPATIBILITY = "exact_runtime_model"
PORTABLE_MUSCLE_CHANNEL_ABI_COMPATIBILITY = "portable_muscle_channel_abi"
RuntimeCompatibility = Literal[
    "exact_runtime_model",
    "portable_muscle_channel_abi",
]

_GROUP_COLLECTIONS = (
    "hard_line_groups",
    "soft_compartment_groups",
    "observation_aggregates",
    "functional_synergy_regions",
)
_SIDES = frozenset({"left", "right", "midline", "unspecified"})
_SHA256_CHARACTERS = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class AnatomicalTaxonomy:
    """Validated immutable identity plus normalized taxonomy payload."""

    schema_version: str
    taxonomy_id: str
    model_binding: dict[str, Any]
    signal_contract: dict[str, Any]
    ordered_actuators: tuple[dict[str, Any], ...]
    hard_line_groups: tuple[dict[str, Any], ...]
    soft_compartment_groups: tuple[dict[str, Any], ...]
    observation_aggregates: tuple[dict[str, Any], ...]
    functional_synergy_regions: tuple[dict[str, Any], ...]
    fingerprint: str
    notes: str
    generation: dict[str, Any] | None = None
    source_path: Path | None = None

    @property
    def actuator_names(self) -> tuple[str, ...]:
        return tuple(str(row["name"]) for row in self.ordered_actuators)

    @property
    def stable_model_binding(self) -> dict[str, Any]:
        """Return the cross-runner binding without a compiled-model hash."""

        if self.schema_version == ANATOMICAL_TAXONOMY_SCHEMA_VERSION:
            return self.model_binding["stable_model_binding"]
        return {key: value for key, value in self.model_binding.items() if key != "runtime_model_hash"}

    @property
    def compiled_runtime_audit(self) -> dict[str, Any]:
        """Return the local-build diagnostic identity.

        V1 is read-only compatible and is normalized into the v2 audit shape at
        access time.  New training releases must bind a v2 taxonomy.
        """

        if self.schema_version == ANATOMICAL_TAXONOMY_SCHEMA_VERSION:
            return self.model_binding["compiled_runtime_audit"]
        return {
            "runtime_model_hash": self.model_binding["runtime_model_hash"],
            "hash_semantics": COMPILED_RUNTIME_HASH_SEMANTICS,
        }

    @property
    def release_eligible(self) -> bool:
        return self.schema_version == ANATOMICAL_TAXONOMY_SCHEMA_VERSION

    def groups(self, collection: str) -> tuple[dict[str, Any], ...]:
        if collection not in _GROUP_COLLECTIONS:
            raise ValueError(f"unsupported taxonomy collection: {collection!r}")
        return getattr(self, collection)

    def to_manifest(self) -> dict[str, Any]:
        manifest: dict[str, Any] = {
            "schema_version": self.schema_version,
            "taxonomy_id": self.taxonomy_id,
            "model_binding": copy.deepcopy(self.model_binding),
            "signal_contract": copy.deepcopy(self.signal_contract),
            "ordered_actuators": copy.deepcopy(list(self.ordered_actuators)),
            "hard_line_groups": copy.deepcopy(list(self.hard_line_groups)),
            "soft_compartment_groups": copy.deepcopy(list(self.soft_compartment_groups)),
            "observation_aggregates": copy.deepcopy(list(self.observation_aggregates)),
            "functional_synergy_regions": copy.deepcopy(list(self.functional_synergy_regions)),
            "notes": self.notes,
        }
        if self.generation is not None:
            manifest["generation"] = copy.deepcopy(self.generation)
        manifest["taxonomy_fingerprint"] = self.fingerprint
        return manifest


def taxonomy_fingerprint(payload: Mapping[str, Any]) -> str:
    """Hash every taxonomy field except its self-referential fingerprint."""

    unsigned = copy.deepcopy(dict(payload))
    unsigned.pop("taxonomy_fingerprint", None)
    encoded = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_anatomical_taxonomy(path: str | Path) -> AnatomicalTaxonomy:
    """Load UTF-8 JSON while rejecting duplicate keys and schema drift."""

    source = Path(path).expanduser().resolve(strict=True)
    payload = json.loads(
        source.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_pairs,
    )
    return validate_anatomical_taxonomy(payload, source_path=source)


def validate_anatomical_taxonomy(
    payload: Mapping[str, Any],
    *,
    source_path: Path | None = None,
) -> AnatomicalTaxonomy:
    """Validate the complete taxonomy and return its normalized contract."""

    if not isinstance(payload, Mapping):
        raise ValueError("anatomical taxonomy must be a JSON object")
    # Fail closed with a specific, legible message before the generic key check so
    # an anatomy-derived NMF region partition can never be mistaken for a taxonomy.
    if payload.get("schema_version") == REGIONAL_GROUPING_SCHEMA_VERSION or "index_ranges" in str(
        payload.get("regions", "")
    ):
        raise ValueError(
            "refusing to load a regional_muscle_grouping_v1 partition as an anatomical taxonomy: "
            "anatomy-derived NMF rank/region partitions are class-D functional_synergy_region priors "
            "and must never become hard_line_group, soft_compartment_group, or observation_aggregate"
        )
    _require_keys(
        payload,
        required={
            "schema_version",
            "taxonomy_id",
            "model_binding",
            "signal_contract",
            "ordered_actuators",
            *_GROUP_COLLECTIONS,
            "taxonomy_fingerprint",
            "notes",
        },
        optional={"generation"},
        context="anatomical taxonomy",
    )
    schema_version = str(payload["schema_version"])
    if schema_version not in _SUPPORTED_TAXONOMY_SCHEMA_VERSIONS:
        raise ValueError(
            f"anatomical taxonomy schema_version must be one of {sorted(_SUPPORTED_TAXONOMY_SCHEMA_VERSIONS)!r}"
        )
    supplied_fingerprint = _require_sha256(
        payload["taxonomy_fingerprint"],
        "taxonomy_fingerprint",
    )
    computed_fingerprint = taxonomy_fingerprint(payload)
    if supplied_fingerprint != computed_fingerprint:
        raise ValueError("anatomical taxonomy fingerprint is stale")

    taxonomy_id = _nonempty_text(payload["taxonomy_id"], "taxonomy_id")
    model_binding = _validate_model_binding(
        payload["model_binding"],
        schema_version=schema_version,
    )
    stable_model_binding = (
        model_binding["stable_model_binding"] if schema_version == ANATOMICAL_TAXONOMY_SCHEMA_VERSION else model_binding
    )
    signal_contract = _validate_signal_contract(payload["signal_contract"])
    ordered_actuators = _validate_ordered_actuators(
        payload["ordered_actuators"],
        model_binding=stable_model_binding,
    )
    if "muscle_channel_core_fingerprint" in stable_model_binding and stable_model_binding[
        "muscle_channel_core_fingerprint"
    ] != _ordered_rows_core_fingerprint(ordered_actuators):
        raise ValueError("model binding muscle_channel_core_fingerprint is stale")
    actuator_by_name = {row["name"]: row for row in ordered_actuators}
    hard = _validate_line_groups(
        payload["hard_line_groups"],
        relationship=HARD_LINE_RELATIONSHIP,
        actuator_by_name=actuator_by_name,
        allow_training=True,
    )
    soft = _validate_line_groups(
        payload["soft_compartment_groups"],
        relationship=SOFT_COMPARTMENT_RELATIONSHIP,
        actuator_by_name=actuator_by_name,
        allow_training=False,
    )
    observation = _validate_observation_aggregates(
        payload["observation_aggregates"],
        actuator_by_name=actuator_by_name,
    )
    functional = _validate_functional_regions(
        payload["functional_synergy_regions"],
        actuator_by_name=actuator_by_name,
    )
    all_ids = [
        *(group["group_id"] for group in hard),
        *(group["group_id"] for group in soft),
        *(group["channel_id"] for group in observation),
        *(group["region_id"] for group in functional),
    ]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("taxonomy identifiers must be unique across all relationships")
    _reject_overlapping_hard_groups(hard)

    generation = payload.get("generation")
    if generation is not None:
        if not isinstance(generation, Mapping):
            raise ValueError("generation must be an object when present")
        generation = copy.deepcopy(dict(generation))
    return AnatomicalTaxonomy(
        schema_version=schema_version,
        taxonomy_id=taxonomy_id,
        model_binding=model_binding,
        signal_contract=signal_contract,
        ordered_actuators=tuple(ordered_actuators),
        hard_line_groups=tuple(hard),
        soft_compartment_groups=tuple(soft),
        observation_aggregates=tuple(observation),
        functional_synergy_regions=tuple(functional),
        fingerprint=supplied_fingerprint,
        notes=str(payload["notes"]),
        generation=generation,
        source_path=source_path,
    )


def validate_taxonomy_against_model(
    taxonomy: AnatomicalTaxonomy,
    model: Any,
    *,
    require_unit_ctrlrange: bool = True,
    validate_package_version: bool = True,
    compatibility: RuntimeCompatibility = EXACT_RUNTIME_MODEL_COMPATIBILITY,
) -> None:
    """Fail closed on any runtime model or ordered-channel drift."""

    binding = taxonomy.stable_model_binding
    runtime_audit = taxonomy.compiled_runtime_audit
    if compatibility not in {
        EXACT_RUNTIME_MODEL_COMPATIBILITY,
        PORTABLE_MUSCLE_CHANNEL_ABI_COMPATIBILITY,
    }:
        raise ValueError(f"unsupported taxonomy runtime compatibility: {compatibility!r}")
    if validate_package_version:
        try:
            installed_version = metadata.version(binding["package"])
        except metadata.PackageNotFoundError as exc:
            raise ValueError(f"taxonomy model package is not installed: {binding['package']!r}") from exc
        if installed_version != binding["version"]:
            raise ValueError(
                "installed model package version differs from anatomical taxonomy: "
                f"installed={installed_version!r}, taxonomy={binding['version']!r}"
            )
    layout = resolve_muscle_channel_layout(
        model,
        taxonomy.actuator_names,
        require_unit_ctrlrange=require_unit_ctrlrange,
        require_scalar_activation=True,
    )
    if (
        compatibility == EXACT_RUNTIME_MODEL_COMPATIBILITY
        and layout.runtime_model_hash != runtime_audit["runtime_model_hash"]
    ):
        raise ValueError("runtime MuJoCo model hash differs from anatomical taxonomy")
    expected_core_fingerprint = binding.get("muscle_channel_core_fingerprint")
    if compatibility == PORTABLE_MUSCLE_CHANNEL_ABI_COMPATIBILITY and expected_core_fingerprint is None:
        raise ValueError(
            "portable muscle-channel ABI validation requires model_binding.muscle_channel_core_fingerprint"
        )
    if expected_core_fingerprint is not None and layout.muscle_channel_core_fingerprint != expected_core_fingerprint:
        raise ValueError("runtime muscle-channel core fingerprint differs from anatomical taxonomy")
    if layout.actuator_schema_hash != binding["actuator_schema_hash"]:
        raise ValueError("runtime actuator schema hash differs from anatomical taxonomy")
    for ordered_index, row in enumerate(taxonomy.ordered_actuators):
        actuator_id = int(layout.actuator_ids[ordered_index])
        if actuator_id != row["actuator_id"]:
            raise ValueError(f"runtime actuator id differs for {row['name']!r}")
        if int(layout.activation_addresses[ordered_index]) != row["actadr"]:
            raise ValueError(f"runtime activation address differs for {row['name']!r}")
        if int(layout.activation_counts[ordered_index]) != row["actnum"]:
            raise ValueError(f"runtime activation count differs for {row['name']!r}")
        if int(layout.dyntype_ids[ordered_index]) != row["dyntype_id"]:
            raise ValueError(f"runtime dyntype differs for {row['name']!r}")
        if not np.array_equal(
            layout.ctrlrange[ordered_index],
            np.asarray(row["ctrlrange"], dtype=np.float64),
        ):
            raise ValueError(f"runtime ctrlrange differs for {row['name']!r}")
        runtime_target = actuator_transmission_target(model, actuator_id)
        if runtime_target != row["target"]:
            raise ValueError(f"runtime transmission target differs for {row['name']!r}")
        for field, runtime_values in (
            ("dynprm", model.actuator_dynprm[actuator_id]),
            ("gainprm", model.actuator_gainprm[actuator_id]),
            ("biasprm", model.actuator_biasprm[actuator_id]),
        ):
            if not np.array_equal(
                np.asarray(runtime_values, dtype=np.float64),
                np.asarray(row[field], dtype=np.float64),
            ):
                raise ValueError(f"runtime {field} differs for {row['name']!r}")


def taxonomy_muscle_channel_core_fingerprint(
    taxonomy: AnatomicalTaxonomy,
) -> str:
    """Hash the exact ordered muscle rows while excluding scene-only state."""

    return _ordered_rows_core_fingerprint(taxonomy.ordered_actuators)


def _ordered_rows_core_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = {
        "schema_version": MUSCLE_CHANNEL_CORE_ABI_SCHEMA_VERSION,
        "actuator_names": [row["name"] for row in rows],
        "actuator_ids": [row["actuator_id"] for row in rows],
        "actuator_dyntype_ids": [row["dyntype_id"] for row in rows],
        "actuator_actnum": [row["actnum"] for row in rows],
        "actuator_actadr": [row["actadr"] for row in rows],
        "actuator_ctrlrange": [row["ctrlrange"] for row in rows],
        "actuator_targets": [row["target"] for row in rows],
        "actuator_dynprm": [row["dynprm"] for row in rows],
        "actuator_gainprm": [row["gainprm"] for row in rows],
        "actuator_biasprm": [row["biasprm"] for row in rows],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_intra_muscle_reward_gate(
    taxonomy: AnatomicalTaxonomy,
    *,
    enabled: bool,
) -> tuple[bool, str]:
    """Fail-closed gate deciding whether an IMR reward term may be enabled.

    The physiology contract keeps intra-muscle consistency diagnostics-only until
    at least one ``hard_line_group`` is ``training_enabled`` with a verified review
    status and non-empty provenance.  A caller that requests ``enabled=True`` while
    no such group exists is rejected here rather than silently penalizing an empty
    set (or, worse, an unverified group).  Returns ``(active, reason)`` where
    ``active`` is only ``True`` when a reward term should actually be applied.
    """

    training_groups = [group for group in taxonomy.hard_line_groups if bool(group["training_enabled"])]
    if not enabled:
        return False, "intra_muscle_reward_disabled_by_config"
    if not training_groups:
        raise ValueError(
            "intra_muscle reward requested but no training-enabled hard_line_group exists; "
            "the contract keeps IMR diagnostics-only until a verified hard-line group with "
            "provenance is added (soft/observation/functional groups can never drive reward)"
        )
    unverified = [
        group["group_id"]
        for group in training_groups
        if group["review_status"] != "verified" or not group["provenance"]
    ]
    if unverified:
        raise ValueError(
            f"intra_muscle reward requested with unverified hard groups {unverified}; "
            "every training-enabled hard_line_group must be review_status=verified with provenance"
        )
    return True, "intra_muscle_reward_active_verified_hard_groups"


def build_intra_muscle_spec(
    taxonomy: AnatomicalTaxonomy,
    *,
    collection: str = "hard_line_groups",
    training_enabled_only: bool = False,
) -> IntraMuscleSpec:
    """Compile hard/soft diagnostic groups into fixed-shape JAX arrays.

    Observation aggregates and functional synergy regions are intentionally
    rejected here so they cannot silently become anatomical constraints.
    """

    if collection not in {"hard_line_groups", "soft_compartment_groups"}:
        raise ValueError("IMR specs may be built only from hard_line_groups or soft_compartment_groups")
    groups = list(taxonomy.groups(collection))
    if training_enabled_only:
        groups = [group for group in groups if bool(group["training_enabled"])]
    max_members = max((len(group["members"]) for group in groups), default=1)
    group_indices = np.zeros((len(groups), max_members), dtype=np.int32)
    member_mask = np.zeros((len(groups), max_members), dtype=np.float32)
    member_weights = np.zeros((len(groups), max_members), dtype=np.float32)
    name_to_index = {row["name"]: int(row["ordered_index"]) for row in taxonomy.ordered_actuators}
    for group_index, group in enumerate(groups):
        member_count = len(group["members"])
        group_indices[group_index, :member_count] = [name_to_index[name] for name in group["members"]]
        member_mask[group_index, :member_count] = 1.0
        member_weights[group_index, :member_count] = np.asarray(
            group["member_weights"],
            dtype=np.float32,
        )
    return IntraMuscleSpec(
        group_indices=jnp.asarray(group_indices),
        member_mask=jnp.asarray(member_mask),
        member_weights=jnp.asarray(member_weights),
        group_weights=jnp.asarray(
            [group["group_weight"] for group in groups],
            dtype=jnp.float32,
        ),
        deadband=jnp.asarray(
            [group["deadband"] for group in groups],
            dtype=jnp.float32,
        ),
        activity_off=jnp.asarray(
            [group["activity_off"] for group in groups],
            dtype=jnp.float32,
        ),
        activity_on=jnp.asarray(
            [group["activity_on"] for group in groups],
            dtype=jnp.float32,
        ),
        activation_addresses=jnp.asarray(
            [row["actadr"] for row in taxonomy.ordered_actuators],
            dtype=jnp.int32,
        ),
        body_actuator_ids=jnp.asarray(
            [row["actuator_id"] for row in taxonomy.ordered_actuators],
            dtype=jnp.int32,
        ),
        group_ids=tuple(group["group_id"] for group in groups),
        relationship=(HARD_LINE_RELATIONSHIP if collection == "hard_line_groups" else SOFT_COMPARTMENT_RELATIONSHIP),
    )


def _validate_model_binding(
    value: Any,
    *,
    schema_version: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("model_binding must be an object")
    if schema_version == ANATOMICAL_TAXONOMY_SCHEMA_VERSION:
        _require_keys(
            value,
            required={"stable_model_binding", "compiled_runtime_audit"},
            context="model_binding",
        )
        stable = _validate_stable_model_binding(value["stable_model_binding"])
        audit = value["compiled_runtime_audit"]
        if not isinstance(audit, Mapping):
            raise ValueError("model_binding.compiled_runtime_audit must be an object")
        _require_keys(
            audit,
            required={"runtime_model_hash", "hash_semantics"},
            context="model_binding.compiled_runtime_audit",
        )
        semantics = _nonempty_text(
            audit["hash_semantics"],
            "model_binding.compiled_runtime_audit.hash_semantics",
        )
        if semantics != COMPILED_RUNTIME_HASH_SEMANTICS:
            raise ValueError("compiled runtime audit hash_semantics must be diagnostic-only")
        return {
            "stable_model_binding": stable,
            "compiled_runtime_audit": {
                "runtime_model_hash": _require_sha256(
                    audit["runtime_model_hash"],
                    "model_binding.compiled_runtime_audit.runtime_model_hash",
                ),
                "hash_semantics": semantics,
            },
        }

    return _validate_v1_model_binding(value)


def _validate_v1_model_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    _require_keys(
        value,
        required={
            "package",
            "version",
            "source_tag_hint",
            "source_tag_status",
            "xml_path",
            "xml_sha256",
            "xml_bundle_sha256",
            "runtime_model_hash",
            "actuator_schema_hash",
            "ordered_action_dim",
            "target",
        },
        optional={"project_urls", "muscle_channel_core_fingerprint"},
        context="model_binding",
    )
    target = value["target"]
    if not isinstance(target, Mapping):
        raise ValueError("model_binding.target must be an object")
    _require_keys(
        target,
        required={"environment", "disable_fingers", "expected_action_dim"},
        context="model_binding.target",
    )
    if not isinstance(target["disable_fingers"], bool):
        raise ValueError("model_binding.target.disable_fingers must be boolean")
    expected_action_dim = _positive_int(
        target["expected_action_dim"],
        "model_binding.target.expected_action_dim",
    )
    ordered_action_dim = _positive_int(
        value["ordered_action_dim"],
        "model_binding.ordered_action_dim",
    )
    if ordered_action_dim != expected_action_dim:
        raise ValueError("model binding action dimensions disagree")
    result = {
        "package": _nonempty_text(value["package"], "model_binding.package"),
        "version": _nonempty_text(value["version"], "model_binding.version"),
        "source_tag_hint": _nonempty_text(
            value["source_tag_hint"],
            "model_binding.source_tag_hint",
        ),
        "source_tag_status": _nonempty_text(
            value["source_tag_status"],
            "model_binding.source_tag_status",
        ),
        "xml_path": _nonempty_text(value["xml_path"], "model_binding.xml_path"),
        "xml_sha256": _require_sha256(
            value["xml_sha256"],
            "model_binding.xml_sha256",
        ),
        "xml_bundle_sha256": _require_sha256(
            value["xml_bundle_sha256"],
            "model_binding.xml_bundle_sha256",
        ),
        "runtime_model_hash": _require_sha256(
            value["runtime_model_hash"],
            "model_binding.runtime_model_hash",
        ),
        "actuator_schema_hash": _require_sha256(
            value["actuator_schema_hash"],
            "model_binding.actuator_schema_hash",
        ),
        "ordered_action_dim": ordered_action_dim,
        "target": {
            "environment": _nonempty_text(
                target["environment"],
                "model_binding.target.environment",
            ),
            "disable_fingers": target["disable_fingers"],
            "expected_action_dim": expected_action_dim,
        },
    }
    if "project_urls" in value:
        if not isinstance(value["project_urls"], list) or not all(
            isinstance(item, str) and item.strip() for item in value["project_urls"]
        ):
            raise ValueError("model_binding.project_urls must be a list of non-empty strings")
        result["project_urls"] = [str(item) for item in value["project_urls"]]
    if "muscle_channel_core_fingerprint" in value:
        result["muscle_channel_core_fingerprint"] = _require_sha256(
            value["muscle_channel_core_fingerprint"],
            "model_binding.muscle_channel_core_fingerprint",
        )
    return result


def _validate_stable_model_binding(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("model_binding.stable_model_binding must be an object")
    _require_keys(
        value,
        required={
            "package",
            "version",
            "source_tag_hint",
            "source_tag_status",
            "xml_path",
            "xml_sha256",
            "xml_bundle_sha256",
            "actuator_schema_hash",
            "muscle_channel_core_fingerprint",
            "ordered_action_dim",
            "target",
        },
        optional={"project_urls"},
        context="model_binding.stable_model_binding",
    )
    legacy_payload = dict(value)
    # Reuse the exact v1 field validation, supplying a syntactically valid local
    # audit hash that is removed from the normalized stable result below.
    legacy_payload["runtime_model_hash"] = "0" * 64
    result = _validate_v1_model_binding(legacy_payload)
    result.pop("runtime_model_hash")
    return result


def _validate_signal_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("signal_contract must be an object")
    _require_keys(
        value,
        required={
            "primary",
            "secondary",
            "control_contract",
            "default_training_behavior",
        },
        context="signal_contract",
    )
    expected = {
        "primary": MUSCLE_ACTIVATION_SEMANTICS,
        "secondary": EFFECTIVE_EXCITATION_SEMANTICS,
        "control_contract": "physical_muscle_ctrlrange_0_1_policy_abi_minus1_1",
        "default_training_behavior": DEFAULT_TRAINING_BEHAVIOR,
    }
    result = {key: str(value[key]) for key in expected}
    if result != expected:
        raise ValueError(f"signal_contract must equal {expected}")
    return result


def _validate_ordered_actuators(
    value: Any,
    *,
    model_binding: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("ordered_actuators must be a non-empty list")
    rows: list[dict[str, Any]] = []
    for expected_index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise ValueError("ordered actuator entries must be objects")
        _require_keys(
            raw,
            required={
                "ordered_index",
                "actuator_id",
                "name",
                "side",
                "dyntype",
                "dyntype_id",
                "actadr",
                "actnum",
                "ctrlrange",
                "target",
                "dynprm",
                "gainprm",
                "biasprm",
            },
            context=f"ordered actuator {expected_index}",
        )
        ordered_index = _nonnegative_int(
            raw["ordered_index"],
            f"ordered actuator {expected_index}.ordered_index",
        )
        if ordered_index != expected_index:
            raise ValueError("ordered actuator indices must be contiguous and exact")
        name = _nonempty_text(raw["name"], f"ordered actuator {expected_index}.name")
        side = _validate_side(raw["side"], f"ordered actuator {name}.side")
        if raw["dyntype"] != "mjDYN_MUSCLE":
            raise ValueError(f"ordered actuator {name!r} must be mjDYN_MUSCLE")
        actadr = _nonnegative_int(raw["actadr"], f"ordered actuator {name}.actadr")
        actnum = _positive_int(raw["actnum"], f"ordered actuator {name}.actnum")
        if actnum != 1:
            raise ValueError(f"ordered actuator {name!r} must have actnum == 1")
        ctrlrange = _finite_vector(
            raw["ctrlrange"],
            length=2,
            context=f"ordered actuator {name}.ctrlrange",
        )
        if ctrlrange != list(UNIT_MUSCLE_CTRLRANGE):
            raise ValueError(f"ordered actuator {name!r} ctrlrange must be [0,1]")
        target = _validate_target(raw["target"], actuator_name=name)
        rows.append(
            {
                "ordered_index": ordered_index,
                "actuator_id": _nonnegative_int(
                    raw["actuator_id"],
                    f"ordered actuator {name}.actuator_id",
                ),
                "name": name,
                "side": side,
                "dyntype": "mjDYN_MUSCLE",
                "dyntype_id": _nonnegative_int(
                    raw["dyntype_id"],
                    f"ordered actuator {name}.dyntype_id",
                ),
                "actadr": actadr,
                "actnum": actnum,
                "ctrlrange": ctrlrange,
                "target": target,
                "dynprm": _finite_vector(
                    raw["dynprm"],
                    length=10,
                    context=f"ordered actuator {name}.dynprm",
                ),
                "gainprm": _finite_vector(
                    raw["gainprm"],
                    length=10,
                    context=f"ordered actuator {name}.gainprm",
                ),
                "biasprm": _finite_vector(
                    raw["biasprm"],
                    length=10,
                    context=f"ordered actuator {name}.biasprm",
                ),
            }
        )
    names = [row["name"] for row in rows]
    ids = [row["actuator_id"] for row in rows]
    actadrs = [row["actadr"] for row in rows]
    if len(set(names)) != len(names):
        raise ValueError("ordered actuator names must be unique")
    if len(set(ids)) != len(ids):
        raise ValueError("ordered actuator ids must be unique")
    if len(set(actadrs)) != len(actadrs):
        raise ValueError("ordered actuator activation addresses must be unique")
    if len(rows) != model_binding["ordered_action_dim"]:
        raise ValueError("ordered actuator count differs from model binding")
    if actuator_schema_hash(names) != model_binding["actuator_schema_hash"]:
        raise ValueError("model binding actuator_schema_hash is stale")
    return rows


def _validate_target(value: Any, *, actuator_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"ordered actuator {actuator_name!r} target must be an object")
    _require_keys(
        value,
        required={
            "transmission",
            "transmission_id",
            "object_type",
            "object_id",
            "name",
        },
        context=f"ordered actuator {actuator_name}.target",
    )
    name = value["name"]
    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise ValueError(f"ordered actuator {actuator_name!r} target name is invalid")
    return {
        "transmission": _nonempty_text(
            value["transmission"],
            f"ordered actuator {actuator_name}.target.transmission",
        ),
        "transmission_id": _nonnegative_int(
            value["transmission_id"],
            f"ordered actuator {actuator_name}.target.transmission_id",
        ),
        "object_type": _nonempty_text(
            value["object_type"],
            f"ordered actuator {actuator_name}.target.object_type",
        ),
        "object_id": _nonnegative_int(
            value["object_id"],
            f"ordered actuator {actuator_name}.target.object_id",
        ),
        "name": name,
    }


def _validate_line_groups(
    value: Any,
    *,
    relationship: str,
    actuator_by_name: Mapping[str, Mapping[str, Any]],
    allow_training: bool,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{relationship} collection must be a list")
    result: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{relationship} entries must be objects")
        _require_keys(
            raw,
            required={
                "group_id",
                "side",
                "anatomical_muscle",
                "members",
                "relationship",
                "review_status",
                "training_enabled",
                "member_weights",
                "deadband",
                "group_weight",
                "activity_off",
                "activity_on",
                "provenance",
            },
            optional={"notes"},
            context=relationship,
        )
        group_id = _nonempty_text(raw["group_id"], f"{relationship}.group_id")
        if raw["relationship"] != relationship:
            raise ValueError(f"group {group_id!r} relationship must be {relationship!r}")
        if not isinstance(raw["training_enabled"], bool):
            raise ValueError(f"group {group_id!r} training_enabled must be boolean")
        training_enabled = bool(raw["training_enabled"])
        if training_enabled and not allow_training:
            raise ValueError(f"{relationship} groups are diagnostics-only")
        review_status = _nonempty_text(
            raw["review_status"],
            f"group {group_id}.review_status",
        )
        provenance = _validate_provenance(
            raw["provenance"],
            context=f"group {group_id}.provenance",
        )
        if training_enabled and (review_status != "verified" or not provenance):
            raise ValueError(f"training-enabled hard group {group_id!r} requires verified review and provenance")
        members = _validate_members(
            raw["members"],
            actuator_by_name=actuator_by_name,
            minimum=2,
            context=f"group {group_id}",
        )
        side = _validate_side(raw["side"], f"group {group_id}.side")
        _validate_group_side(
            side,
            members,
            actuator_by_name=actuator_by_name,
            training_enabled=training_enabled,
            context=f"group {group_id}",
        )
        weights = _finite_vector(
            raw["member_weights"],
            length=len(members),
            context=f"group {group_id}.member_weights",
        )
        if any(weight <= 0.0 for weight in weights):
            raise ValueError(f"group {group_id!r} member weights must be positive")
        deadband = _finite_float(raw["deadband"], f"group {group_id}.deadband")
        group_weight = _finite_float(
            raw["group_weight"],
            f"group {group_id}.group_weight",
        )
        activity_off = _finite_float(
            raw["activity_off"],
            f"group {group_id}.activity_off",
        )
        activity_on = _finite_float(
            raw["activity_on"],
            f"group {group_id}.activity_on",
        )
        if deadband < 0.0 or group_weight <= 0.0:
            raise ValueError(f"group {group_id!r} deadband/weight is invalid")
        if not 0.0 <= activity_off < activity_on <= 1.0:
            raise ValueError(f"group {group_id!r} activity gate is invalid")
        normalized = {
            "group_id": group_id,
            "side": side,
            "anatomical_muscle": _nonempty_text(
                raw["anatomical_muscle"],
                f"group {group_id}.anatomical_muscle",
            ),
            "members": members,
            "relationship": relationship,
            "review_status": review_status,
            "training_enabled": training_enabled,
            "member_weights": weights,
            "deadband": deadband,
            "group_weight": group_weight,
            "activity_off": activity_off,
            "activity_on": activity_on,
            "provenance": provenance,
        }
        if "notes" in raw:
            normalized["notes"] = str(raw["notes"])
        result.append(normalized)
    return result


def _validate_observation_aggregates(
    value: Any,
    *,
    actuator_by_name: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("observation_aggregates must be a list")
    result: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ValueError("observation aggregate entries must be objects")
        _require_keys(
            raw,
            required={
                "channel_id",
                "side",
                "members",
                "weights",
                "relationship",
                "training_enabled",
                "provenance",
            },
            optional={"notes"},
            context=OBSERVATION_AGGREGATE_RELATIONSHIP,
        )
        channel_id = _nonempty_text(
            raw["channel_id"],
            "observation aggregate channel_id",
        )
        if raw["relationship"] != OBSERVATION_AGGREGATE_RELATIONSHIP:
            raise ValueError(f"observation aggregate {channel_id!r} has the wrong relationship")
        if raw["training_enabled"] is not False:
            raise ValueError("observation aggregates can never be training-enabled")
        members = _validate_members(
            raw["members"],
            actuator_by_name=actuator_by_name,
            minimum=1,
            context=f"observation aggregate {channel_id}",
        )
        weights = _finite_vector(
            raw["weights"],
            length=len(members),
            context=f"observation aggregate {channel_id}.weights",
        )
        if any(weight < 0.0 for weight in weights) or not np.isclose(
            sum(weights),
            1.0,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(f"observation aggregate {channel_id!r} weights must be non-negative and sum to one")
        normalized = {
            "channel_id": channel_id,
            "side": _validate_side(
                raw["side"],
                f"observation aggregate {channel_id}.side",
            ),
            "members": members,
            "weights": weights,
            "relationship": OBSERVATION_AGGREGATE_RELATIONSHIP,
            "training_enabled": False,
            "provenance": _validate_provenance(
                raw["provenance"],
                context=f"observation aggregate {channel_id}.provenance",
            ),
        }
        if "notes" in raw:
            normalized["notes"] = str(raw["notes"])
        result.append(normalized)
    return result


def _validate_functional_regions(
    value: Any,
    *,
    actuator_by_name: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("functional_synergy_regions must be a list")
    result: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ValueError("functional synergy region entries must be objects")
        _require_keys(
            raw,
            required={
                "region_id",
                "side",
                "members",
                "relationship",
                "training_enabled",
                "provenance",
            },
            optional={"notes"},
            context=FUNCTIONAL_SYNERGY_RELATIONSHIP,
        )
        region_id = _nonempty_text(
            raw["region_id"],
            "functional synergy region_id",
        )
        if raw["relationship"] != FUNCTIONAL_SYNERGY_RELATIONSHIP:
            raise ValueError(f"functional region {region_id!r} has the wrong relationship")
        if raw["training_enabled"] is not False:
            raise ValueError("functional synergy regions can never be training-enabled")
        normalized = {
            "region_id": region_id,
            "side": _validate_side(
                raw["side"],
                f"functional region {region_id}.side",
            ),
            "members": _validate_members(
                raw["members"],
                actuator_by_name=actuator_by_name,
                minimum=1,
                context=f"functional region {region_id}",
            ),
            "relationship": FUNCTIONAL_SYNERGY_RELATIONSHIP,
            "training_enabled": False,
            "provenance": _validate_provenance(
                raw["provenance"],
                context=f"functional region {region_id}.provenance",
            ),
        }
        if "notes" in raw:
            normalized["notes"] = str(raw["notes"])
        result.append(normalized)
    return result


def _validate_members(
    value: Any,
    *,
    actuator_by_name: Mapping[str, Mapping[str, Any]],
    minimum: int,
    context: str,
) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{context} members must be a list")
    members = [_nonempty_text(item, f"{context} member") for item in value]
    if len(members) < minimum or len(set(members)) != len(members):
        raise ValueError(f"{context} members are empty, duplicated, or too short")
    missing = [name for name in members if name not in actuator_by_name]
    if missing:
        raise ValueError(f"{context} contains unknown actuators: {missing}")
    return members


def _validate_group_side(
    side: str,
    members: Sequence[str],
    *,
    actuator_by_name: Mapping[str, Mapping[str, Any]],
    training_enabled: bool,
    context: str,
) -> None:
    member_sides = {
        str(actuator_by_name[name]["side"]) for name in members if actuator_by_name[name]["side"] != "unspecified"
    }
    lateral_sides = member_sides & {"left", "right"}
    if len(lateral_sides) > 1:
        raise ValueError(f"{context} crosses left/right sides")
    if side in {"left", "right"} and lateral_sides and lateral_sides != {side}:
        raise ValueError(f"{context} side label disagrees with its members")
    if training_enabled:
        exact_sides = {str(actuator_by_name[name]["side"]) for name in members}
        if exact_sides != {side}:
            raise ValueError(f"{context} cannot be training-enabled with unresolved or mismatched member sides")


def _validate_provenance(value: Any, *, context: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    result: list[dict[str, Any]] = []
    for entry in value:
        if not isinstance(entry, Mapping):
            raise ValueError(f"{context} entries must be objects")
        _require_keys(
            entry,
            required={"kind", "reference"},
            optional={"sha256"},
            context=context,
        )
        normalized = {
            "kind": _nonempty_text(entry["kind"], f"{context}.kind"),
            "reference": _nonempty_text(
                entry["reference"],
                f"{context}.reference",
            ),
        }
        if "sha256" in entry:
            normalized["sha256"] = _require_sha256(
                entry["sha256"],
                f"{context}.sha256",
            )
        result.append(normalized)
    return result


def _reject_overlapping_hard_groups(groups: Sequence[Mapping[str, Any]]) -> None:
    owner: dict[str, str] = {}
    for group in groups:
        for member in group["members"]:
            if member in owner:
                raise ValueError(f"hard line groups overlap on {member!r}: {owner[member]!r} and {group['group_id']!r}")
            owner[member] = str(group["group_id"])


def _validate_side(value: Any, context: str) -> str:
    side = str(value)
    if side not in _SIDES:
        raise ValueError(f"{context} must be one of {sorted(_SIDES)}")
    return side


def _require_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
    context: str,
) -> None:
    keys = set(value)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - (optional or set()))
    if missing or unknown:
        raise ValueError(f"{context} keys differ from schema: missing={missing}, unknown={unknown}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _nonempty_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _finite_vector(value: Any, *, length: int, context: str) -> list[float]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be a finite vector of length {length}") from exc
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{context} must be a finite vector of length {length}")
    return [float(item) for item in array]


def _finite_float(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be a finite number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{context} must be a finite number")
    return result


def _nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be a non-negative integer")
    try:
        result = int(value)
        exact = float(value) == float(result)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{context} must be a non-negative integer") from exc
    if result < 0 or not exact:
        raise ValueError(f"{context} must be a non-negative integer")
    return result


def _positive_int(value: Any, context: str) -> int:
    result = _nonnegative_int(value, context)
    if result <= 0:
        raise ValueError(f"{context} must be positive")
    return result


def _require_sha256(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value

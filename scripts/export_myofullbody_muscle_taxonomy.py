#!/usr/bin/env python3
"""Export the exact no-finger MyoFullBody muscle taxonomy for manual audit.

The exporter records runtime facts only.  It never infers or enables
``hard_line_group`` relationships; anatomical curation and provenance must be
added in a separate reviewed change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from importlib import metadata
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import mujoco  # noqa: E402
import musclemimic_models  # noqa: E402
import numpy as np  # noqa: E402

from musclemimic.distill.action_schema import actuator_schema_hash  # noqa: E402
from musclemimic.environments.humanoids.myofullbody import MyoFullBody  # noqa: E402
from musclemimic.physiology.anatomical_groups import (  # noqa: E402
    ANATOMICAL_TAXONOMY_SCHEMA_VERSION,
    COMPILED_RUNTIME_HASH_SEMANTICS,
    DEFAULT_TRAINING_BEHAVIOR,
    taxonomy_fingerprint,
    validate_anatomical_taxonomy,
    validate_taxonomy_against_model,
)
from musclemimic.physiology.effective_excitation import (  # noqa: E402
    EFFECTIVE_EXCITATION_SEMANTICS,
    MUSCLE_ACTIVATION_SEMANTICS,
    actuator_transmission_target,
    resolve_muscle_channel_layout,
)

TAXONOMY_ID = "myofullbody_354_muscle_taxonomy_audit_v2"
EXPECTED_ACTION_DIM = 354
PACKAGE_DISTRIBUTION = "musclemimic-models"


def build_taxonomy_manifest(
    *,
    expected_package_version: str | None = None,
) -> dict[str, Any]:
    """Build and self-validate a runtime-bound taxonomy manifest."""

    package_version = metadata.version(PACKAGE_DISTRIBUTION)
    if expected_package_version is not None and package_version != str(expected_package_version):
        raise ValueError(
            "installed musclemimic-models version differs from the requested "
            f"version: installed={package_version!r}, "
            f"expected={str(expected_package_version)!r}"
        )
    xml_path = Path(musclemimic_models.get_xml_path("myofullbody")).resolve(strict=True)
    package_root = Path(musclemimic_models.__file__).resolve(strict=True).parent

    env = MyoFullBody(disable_fingers=True)
    model = env._model
    actuator_ids = np.asarray(env._action_indices, dtype=np.int32)
    if actuator_ids.shape != (EXPECTED_ACTION_DIM,):
        raise ValueError(
            f"MyoFullBody(disable_fingers=True) must expose exactly 354 ordered actuators, got {actuator_ids.shape}"
        )
    actuator_names = tuple(
        _required_mujoco_name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            int(actuator_id),
            context="actuator",
        )
        for actuator_id in actuator_ids
    )
    layout = resolve_muscle_channel_layout(
        model,
        actuator_names,
        require_unit_ctrlrange=True,
        require_scalar_activation=True,
    )
    rows = [
        _actuator_row(
            model,
            ordered_index=ordered_index,
            actuator_id=int(actuator_id),
            name=name,
            known_actuator_names=frozenset(actuator_names),
        )
        for ordered_index, (actuator_id, name) in enumerate(zip(actuator_ids, actuator_names, strict=True))
    ]
    project_urls = metadata.metadata(PACKAGE_DISTRIBUTION).get_all("Project-URL")
    payload: dict[str, Any] = {
        "schema_version": ANATOMICAL_TAXONOMY_SCHEMA_VERSION,
        "taxonomy_id": TAXONOMY_ID,
        "model_binding": {
            "stable_model_binding": {
                "package": PACKAGE_DISTRIBUTION,
                "version": package_version,
                "source_tag_hint": f"v{package_version}",
                "source_tag_status": ("derived_from_installed_version_not_independently_verified"),
                "xml_path": xml_path.relative_to(package_root).as_posix(),
                "xml_sha256": _file_sha256(xml_path),
                "xml_bundle_sha256": _xml_bundle_sha256(
                    xml_path,
                    package_root=package_root,
                ),
                "actuator_schema_hash": actuator_schema_hash(actuator_names),
                "muscle_channel_core_fingerprint": (layout.muscle_channel_core_fingerprint),
                "ordered_action_dim": EXPECTED_ACTION_DIM,
                "target": {
                    "environment": "MyoFullBody",
                    "disable_fingers": True,
                    "expected_action_dim": EXPECTED_ACTION_DIM,
                },
                "project_urls": list(project_urls or ()),
            },
            "compiled_runtime_audit": {
                "runtime_model_hash": layout.runtime_model_hash,
                "hash_semantics": COMPILED_RUNTIME_HASH_SEMANTICS,
            },
        },
        "signal_contract": {
            "primary": MUSCLE_ACTIVATION_SEMANTICS,
            "secondary": EFFECTIVE_EXCITATION_SEMANTICS,
            "control_contract": "physical_muscle_ctrlrange_0_1_policy_abi_minus1_1",
            "default_training_behavior": DEFAULT_TRAINING_BEHAVIOR,
        },
        "ordered_actuators": rows,
        "hard_line_groups": [],
        "soft_compartment_groups": [],
        "observation_aggregates": [],
        "functional_synergy_regions": [],
        "generation": {
            "tool": "scripts/export_myofullbody_muscle_taxonomy.py",
            "method": ("stable_portable_muscle_abi_plus_local_runtime_audit_no_anatomical_inference"),
            "hard_line_group_policy": "empty_until_manual_review_with_provenance",
            "side_policy": "suffix_hint_or_unsuffixed_right_when_exact_left_counterpart_exists",
        },
        "notes": (
            "Audit inventory only. No hard-line, soft-compartment, EMG aggregate, "
            "or functional-synergy relationship is inferred by this exporter. "
            "All training constraints remain disabled."
        ),
    }
    payload["taxonomy_fingerprint"] = taxonomy_fingerprint(payload)
    taxonomy = validate_anatomical_taxonomy(payload)
    # Schema validation only proves the manifest agrees with itself.  Bind it back
    # to the very model it was exported from, so an exporter bug cannot ship a
    # self-consistent manifest whose rows no longer describe the runtime channels
    # that every downstream IMR and synergy hash is keyed to.
    validate_taxonomy_against_model(taxonomy, model)
    return payload


def _actuator_row(
    model: Any,
    *,
    ordered_index: int,
    actuator_id: int,
    name: str,
    known_actuator_names: frozenset[str],
) -> dict[str, Any]:
    dyntype_id = int(model.actuator_dyntype[actuator_id])
    return {
        "ordered_index": int(ordered_index),
        "actuator_id": int(actuator_id),
        "name": str(name),
        "side": _side_hint(name, known_actuator_names=known_actuator_names),
        "dyntype": mujoco.mjtDyn(dyntype_id).name,
        "dyntype_id": dyntype_id,
        "actadr": int(model.actuator_actadr[actuator_id]),
        "actnum": int(model.actuator_actnum[actuator_id]),
        "ctrlrange": np.asarray(
            model.actuator_ctrlrange[actuator_id],
            dtype=np.float64,
        ).tolist(),
        "target": actuator_transmission_target(model, actuator_id),
        "dynprm": np.asarray(
            model.actuator_dynprm[actuator_id],
            dtype=np.float64,
        ).tolist(),
        "gainprm": np.asarray(
            model.actuator_gainprm[actuator_id],
            dtype=np.float64,
        ).tolist(),
        "biasprm": np.asarray(
            model.actuator_biasprm[actuator_id],
            dtype=np.float64,
        ).tolist(),
    }


def _side_hint(name: str, *, known_actuator_names: frozenset[str]) -> str:
    lowered = str(name).lower()
    if lowered.endswith("_left") or lowered.endswith("_l"):
        return "left"
    if lowered.endswith("_right") or lowered.endswith("_r"):
        return "right"
    # MyoFullBody's upper-limb convention uses an unsuffixed right actuator
    # and an exact ``*_left`` partner.  Treat this only as a name-pair fact;
    # do not guess a side when the counterpart is absent.
    if f"{name}_left" in known_actuator_names:
        return "right"
    return "unspecified"


def _required_mujoco_name(
    model: Any,
    object_type: Any,
    object_id: int,
    *,
    context: str,
) -> str:
    name = mujoco.mj_id2name(model, object_type, int(object_id))
    if not name:
        raise ValueError(f"{context} id={int(object_id)} has no stable name")
    return str(name)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _xml_bundle_sha256(root_xml: Path, *, package_root: Path) -> str:
    files = _included_xml_files(root_xml)
    digest = hashlib.sha256(b"musclemimic-xml-include-bundle-v1\0")
    for path in sorted(files, key=lambda item: item.as_posix()):
        try:
            label = path.relative_to(package_root).as_posix()
        except ValueError as exc:
            raise ValueError(f"included XML escapes the model package root: {path}") from exc
        digest.update(label.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _included_xml_files(root_xml: Path) -> set[Path]:
    pending = [root_xml.resolve(strict=True)]
    visited: set[Path] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        try:
            root = ET.parse(current).getroot()
        except ET.ParseError as exc:
            raise ValueError(f"cannot parse model XML include {current}") from exc
        for include in root.iter("include"):
            raw_file = include.get("file")
            if not raw_file:
                raise ValueError(f"XML include has no file attribute: {current}")
            pending.append((current.parent / raw_file).resolve(strict=True))
    return visited


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export exact MyoFullBody-354 runtime muscle metadata without guessing anatomical training groups."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write JSON to this path; stdout is used when omitted.",
    )
    parser.add_argument(
        "--expected-package-version",
        help="Fail if the installed musclemimic-models version differs.",
    )
    parser.add_argument(
        "--disable-fingers",
        action="store_true",
        default=True,
        help=(
            "Export the canonical finger-disabled 354-actuator target. "
            "This flag is accepted explicitly for reproducible commands."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = build_taxonomy_manifest(
        expected_package_version=args.expected_package_version,
    )
    text = json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    if args.output is None:
        sys.stdout.write(text + "\n")
    else:
        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.write_text(text + "\n", encoding="utf-8")
        temporary.replace(output)
        print(f"Wrote {len(manifest['ordered_actuators'])} ordered muscles to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

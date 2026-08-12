"""Strict schema and runtime-binding tests for anatomical taxonomy."""

from __future__ import annotations

import copy
import hashlib
from importlib import metadata
from pathlib import Path

import mujoco
import numpy as np
import pytest

from musclemimic.distill.action_schema import actuator_schema_hash
from musclemimic.physiology.anatomical_groups import (
    ANATOMICAL_TAXONOMY_V1_SCHEMA_VERSION,
    build_intra_muscle_spec,
    load_anatomical_taxonomy,
    taxonomy_fingerprint,
    validate_anatomical_taxonomy,
    validate_taxonomy_against_model,
)
from musclemimic.physiology.effective_excitation import (
    EFFECTIVE_EXCITATION_SEMANTICS,
    MUSCLE_ACTIVATION_SEMANTICS,
    actuator_transmission_target,
)


def _model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(
        """
        <mujoco model="taxonomy_test">
          <worldbody>
            <body name="body">
              <joint name="joint" type="hinge"/>
              <geom type="capsule" size=".02" fromto="0 0 0 0 0 .2" mass="1"/>
              <site name="origin" pos="0 0 .02"/>
              <site name="insertion" pos="0 0 .18"/>
            </body>
          </worldbody>
          <tendon>
            <spatial name="line_a_tendon"><site site="origin"/><site site="insertion"/></spatial>
            <spatial name="line_b_tendon"><site site="origin"/><site site="insertion"/></spatial>
            <spatial name="line_left_tendon"><site site="origin"/><site site="insertion"/></spatial>
          </tendon>
          <actuator>
            <general name="line_a_r" tendon="line_a_tendon"
              ctrllimited="true" ctrlrange="0 1"
              dyntype="muscle" gaintype="muscle" biastype="muscle"
              dynprm=".01 .04" gainprm=".75 1.05 -1 200 .5 1.6 1.5 1.3 1.2 0"
              biasprm=".75 1.05 -1 200 .5 1.6 1.5 1.3 1.2 0"
              lengthrange=".05 .2"/>
            <general name="line_b_r" tendon="line_b_tendon"
              ctrllimited="true" ctrlrange="0 1"
              dyntype="muscle" gaintype="muscle" biastype="muscle"
              dynprm=".01 .04" gainprm=".75 1.05 -1 200 .5 1.6 1.5 1.3 1.2 0"
              biasprm=".75 1.05 -1 200 .5 1.6 1.5 1.3 1.2 0"
              lengthrange=".05 .2"/>
            <general name="line_c_l" tendon="line_left_tendon"
              ctrllimited="true" ctrlrange="0 1"
              dyntype="muscle" gaintype="muscle" biastype="muscle"
              dynprm=".01 .04" gainprm=".75 1.05 -1 200 .5 1.6 1.5 1.3 1.2 0"
              biasprm=".75 1.05 -1 200 .5 1.6 1.5 1.3 1.2 0"
              lengthrange=".05 .2"/>
          </actuator>
        </mujoco>
        """
    )


def _seal(payload: dict) -> dict:
    payload["taxonomy_fingerprint"] = taxonomy_fingerprint(payload)
    return payload


def _payload() -> tuple[dict, mujoco.MjModel]:
    model = _model()
    names = ["line_a_r", "line_b_r", "line_c_l"]
    rows = []
    for ordered_index, name in enumerate(names):
        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )
        rows.append(
            {
                "ordered_index": ordered_index,
                "actuator_id": actuator_id,
                "name": name,
                "side": "left" if name.endswith("_l") else "right",
                "dyntype": "mjDYN_MUSCLE",
                "dyntype_id": int(model.actuator_dyntype[actuator_id]),
                "actadr": int(model.actuator_actadr[actuator_id]),
                "actnum": int(model.actuator_actnum[actuator_id]),
                "ctrlrange": np.asarray(
                    model.actuator_ctrlrange[actuator_id],
                    dtype=float,
                ).tolist(),
                "target": actuator_transmission_target(model, actuator_id),
                "dynprm": np.asarray(model.actuator_dynprm[actuator_id]).tolist(),
                "gainprm": np.asarray(model.actuator_gainprm[actuator_id]).tolist(),
                "biasprm": np.asarray(model.actuator_biasprm[actuator_id]).tolist(),
            }
        )
    model_hash = hashlib.sha256(model.__getstate__()).hexdigest()
    payload = {
        "schema_version": ANATOMICAL_TAXONOMY_V1_SCHEMA_VERSION,
        "taxonomy_id": "test_taxonomy",
        "model_binding": {
            "package": "musclemimic-models",
            "version": metadata.version("musclemimic-models"),
            "source_tag_hint": "test-only",
            "source_tag_status": "fixture",
            "xml_path": "fixture.xml",
            "xml_sha256": "1" * 64,
            "xml_bundle_sha256": "2" * 64,
            "runtime_model_hash": model_hash,
            "actuator_schema_hash": actuator_schema_hash(names),
            "ordered_action_dim": len(names),
            "target": {
                "environment": "test",
                "disable_fingers": True,
                "expected_action_dim": len(names),
            },
            "project_urls": [],
        },
        "signal_contract": {
            "primary": MUSCLE_ACTIVATION_SEMANTICS,
            "secondary": EFFECTIVE_EXCITATION_SEMANTICS,
            "control_contract": "physical_muscle_ctrlrange_0_1_policy_abi_minus1_1",
            "default_training_behavior": "diagnostics_only_no_reward",
        },
        "ordered_actuators": rows,
        "hard_line_groups": [],
        "soft_compartment_groups": [],
        "observation_aggregates": [],
        "functional_synergy_regions": [],
        "notes": "test",
    }
    return _seal(payload), model


def _hard_group(*, enabled: bool = False, provenance: list | None = None) -> dict:
    return {
        "group_id": "right_verified_lines",
        "side": "right",
        "anatomical_muscle": "fixture muscle",
        "members": ["line_a_r", "line_b_r"],
        "relationship": "hard_line_group",
        "review_status": "verified" if enabled else "provisional",
        "training_enabled": enabled,
        "member_weights": [1.0, 1.0],
        "deadband": 0.1,
        "group_weight": 1.0,
        "activity_off": 0.02,
        "activity_on": 0.1,
        "provenance": [] if provenance is None else provenance,
    }


def _soft_group() -> dict:
    return {
        **_hard_group(),
        "group_id": "right_soft_compartments",
        "relationship": "soft_compartment_group",
        "review_status": "verified_compartments_not_shared_line",
        "training_enabled": False,
    }


def test_empty_hard_groups_are_valid_and_compile_to_stable_empty_spec():
    payload, model = _payload()
    taxonomy = validate_anatomical_taxonomy(payload)
    assert taxonomy.hard_line_groups == ()
    assert taxonomy.model_binding["ordered_action_dim"] == 3
    validate_taxonomy_against_model(taxonomy, model)

    spec = build_intra_muscle_spec(taxonomy, training_enabled_only=True)
    assert spec.group_indices.shape == (0, 1)
    assert spec.member_mask.shape == (0, 1)
    assert spec.group_ids == ()


def test_training_enabled_hard_group_requires_verified_review_and_provenance():
    payload, _ = _payload()
    payload["hard_line_groups"] = [_hard_group(enabled=True)]
    _seal(payload)
    with pytest.raises(ValueError, match="requires verified review and provenance"):
        validate_anatomical_taxonomy(payload)

    payload["hard_line_groups"] = [
        _hard_group(
            enabled=True,
            provenance=[
                {
                    "kind": "manual_anatomical_review",
                    "reference": "fixture-review-1",
                }
            ],
        )
    ]
    taxonomy = validate_anatomical_taxonomy(_seal(payload))
    spec = build_intra_muscle_spec(
        taxonomy,
        training_enabled_only=True,
    )
    assert spec.group_ids == ("right_verified_lines",)
    assert spec.group_indices.tolist() == [[0, 1]]
    assert spec.activation_addresses.tolist() == [0, 1, 2]


def test_nonhard_relationships_are_separate_and_can_never_enable_training():
    payload, _ = _payload()
    payload["soft_compartment_groups"] = [_soft_group()]
    payload["observation_aggregates"] = [
        {
            "channel_id": "right_surface_channel",
            "side": "right",
            "members": ["line_a_r", "line_b_r"],
            "weights": [0.5, 0.5],
            "relationship": "observation_aggregate",
            "training_enabled": False,
            "provenance": [],
        }
    ]
    payload["functional_synergy_regions"] = [
        {
            "region_id": "functional_right",
            "side": "right",
            "members": ["line_a_r", "line_b_r"],
            "relationship": "functional_synergy_region",
            "training_enabled": False,
            "provenance": [],
        }
    ]
    taxonomy = validate_anatomical_taxonomy(_seal(payload))
    soft_spec = build_intra_muscle_spec(
        taxonomy,
        collection="soft_compartment_groups",
    )
    assert soft_spec.relationship == "soft_compartment_group"
    with pytest.raises(ValueError, match="only from hard_line_groups or"):
        build_intra_muscle_spec(
            taxonomy,
            collection="observation_aggregates",
        )

    payload["observation_aggregates"][0]["training_enabled"] = True
    with pytest.raises(ValueError, match="never be training-enabled"):
        validate_anatomical_taxonomy(_seal(payload))


def test_manifest_rejects_overlap_cross_side_singleton_unknown_and_stale_hash():
    payload, _ = _payload()
    stale = copy.deepcopy(payload)
    stale["notes"] = "tampered"
    with pytest.raises(ValueError, match="fingerprint is stale"):
        validate_anatomical_taxonomy(stale)

    payload["hard_line_groups"] = [
        _hard_group(),
        {**_hard_group(), "group_id": "overlap"},
    ]
    with pytest.raises(ValueError, match="overlap"):
        validate_anatomical_taxonomy(_seal(payload))

    payload, _ = _payload()
    cross_side = _hard_group()
    cross_side["members"] = ["line_a_r", "line_c_l"]
    payload["hard_line_groups"] = [cross_side]
    with pytest.raises(ValueError, match="crosses left/right"):
        validate_anatomical_taxonomy(_seal(payload))

    payload, _ = _payload()
    singleton = _hard_group()
    singleton["members"] = ["line_a_r"]
    singleton["member_weights"] = [1.0]
    payload["hard_line_groups"] = [singleton]
    with pytest.raises(ValueError, match="too short"):
        validate_anatomical_taxonomy(_seal(payload))

    payload, _ = _payload()
    unknown = _hard_group()
    unknown["members"] = ["line_a_r", "unknown"]
    payload["hard_line_groups"] = [unknown]
    with pytest.raises(ValueError, match="unknown actuators"):
        validate_anatomical_taxonomy(_seal(payload))


def test_runtime_binding_rejects_model_hash_package_version_and_channel_drift():
    payload, model = _payload()
    taxonomy = validate_anatomical_taxonomy(payload)

    drift = copy.deepcopy(payload)
    drift["model_binding"]["runtime_model_hash"] = "f" * 64
    with pytest.raises(ValueError, match="runtime MuJoCo model hash"):
        validate_taxonomy_against_model(
            validate_anatomical_taxonomy(_seal(drift)),
            model,
        )

    drift = copy.deepcopy(payload)
    drift["ordered_actuators"][0]["actadr"] = 2
    drift["ordered_actuators"][2]["actadr"] = 0
    with pytest.raises(ValueError, match="runtime activation address"):
        validate_taxonomy_against_model(
            validate_anatomical_taxonomy(_seal(drift)),
            model,
        )

    drift = copy.deepcopy(payload)
    drift["model_binding"]["version"] = "0.0.0-not-installed"
    with pytest.raises(ValueError, match="installed model package version differs"):
        validate_taxonomy_against_model(
            validate_anatomical_taxonomy(_seal(drift)),
            model,
        )

    assert taxonomy.model_binding["version"] == metadata.version("musclemimic-models")


def test_exporter_emits_exact_354_inventory_and_no_inferred_groups():
    from scripts.export_myofullbody_muscle_taxonomy import (
        build_taxonomy_manifest,
    )

    manifest = build_taxonomy_manifest()
    taxonomy = validate_anatomical_taxonomy(manifest)
    checked_in_path = (
        Path(__file__).resolve().parents[2] / "configs" / "physiology" / "myofullbody_354_muscle_taxonomy_audit_v2.json"
    )
    checked_in = load_anatomical_taxonomy(checked_in_path)
    assert checked_in.stable_model_binding == taxonomy.stable_model_binding
    assert checked_in.ordered_actuators == taxonomy.ordered_actuators
    assert len(taxonomy.ordered_actuators) == 354
    assert taxonomy.stable_model_binding["version"] == metadata.version("musclemimic-models")
    assert taxonomy.hard_line_groups == ()
    assert taxonomy.soft_compartment_groups == ()
    assert taxonomy.observation_aggregates == ()
    assert taxonomy.functional_synergy_regions == ()
    assert all(
        row["dyntype"] == "mjDYN_MUSCLE" and row["actnum"] == 1 and row["ctrlrange"] == [0.0, 1.0]
        for row in taxonomy.ordered_actuators
    )
    by_name = {row["name"]: row for row in taxonomy.ordered_actuators}
    assert by_name["DELT1"]["side"] == "right"
    assert by_name["DELT1_left"]["side"] == "left"


def test_regional_grouping_partition_cannot_load_as_anatomical_taxonomy():
    # The anatomy-derived NMF region partition is a class-D functional_synergy_region
    # basis prior; it must never be loadable as an anatomical taxonomy.
    regions_path = (
        Path(__file__).resolve().parents[2]
        / "experiments"
        / "synergy"
        / "forehand_clear_myofullbody_354_regions_v1.json"
    )
    with pytest.raises(ValueError, match="regional_muscle_grouping_v1"):
        load_anatomical_taxonomy(regions_path)

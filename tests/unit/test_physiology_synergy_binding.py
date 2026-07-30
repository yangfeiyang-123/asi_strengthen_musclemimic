"""Cross-module binding between synergy artifacts and the anatomy taxonomy.

The synergy stack and the anatomy taxonomy each pin their own ordered channel
schema, so both can validate individually while disagreeing about what channel
*k* means.  Intra-muscle consistency addresses muscles by position, so that
disagreement would silently relabel group members instead of failing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from musclemimic.physiology.anatomical_groups import validate_anatomical_taxonomy
from musclemimic.physiology.synergy_binding import (
    SYNERGY_TAXONOMY_BINDING_SCHEMA_VERSION,
    assert_synergy_artifact_matches_taxonomy,
    assert_taxonomy_matches_ordered_muscles,
    ordered_muscle_schema_sha256,
    taxonomy_ordered_muscle_schema_hash,
)
from musclemimic.synergy.basis_artifact import _json_sha256
from musclemimic.synergy.grouping import ordered_muscle_schema_hash
from tests.unit.test_physiology_taxonomy import _payload

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED_TAXONOMY = REPO_ROOT / "configs/physiology/myofullbody_354_muscle_taxonomy_audit_v2.json"
SHIPPED_REGIONS = REPO_ROOT / "experiments/synergy/forehand_clear_myofullbody_354_regions_v1.json"


def _taxonomy():
    payload, _ = _payload()
    return validate_anatomical_taxonomy(payload)


@pytest.mark.parametrize(
    "names",
    [
        ("glut_max1_r", "glut_max2_r", "vasmed_l"),
        ("胸大肌_r", "三角肌_l"),
        ("biceps_brachii_é_r", "tríceps_l"),
    ],
    ids=["ascii", "cjk", "accented"],
)
def test_all_three_ordered_muscle_hash_implementations_agree(names):
    """Defend the invariant the whole cross-check rests on.

    ``synergy_binding`` re-states the canonical JSON instead of importing it, and
    ``basis_artifact._json_sha256`` is shared with unrelated fingerprints.  If any
    of the three drifts (an added ``indent``, a dropped ``ensure_ascii=False``),
    every valid artifact would start being rejected -- so assert the agreement
    directly rather than trusting three independent ``json.dumps`` calls.
    """

    local = ordered_muscle_schema_sha256(names)
    assert local == ordered_muscle_schema_hash(names)
    assert local == _json_sha256({"muscle_names": list(names)})


def test_taxonomy_hash_is_the_hash_of_its_ordered_actuator_names():
    taxonomy = _taxonomy()
    assert taxonomy_ordered_muscle_schema_hash(taxonomy) == ordered_muscle_schema_hash(taxonomy.actuator_names)


def test_matching_order_returns_a_binding_record():
    taxonomy = _taxonomy()
    record = assert_taxonomy_matches_ordered_muscles(
        taxonomy,
        taxonomy.actuator_names,
        context="probe",
    )
    assert record["schema_version"] == SYNERGY_TAXONOMY_BINDING_SCHEMA_VERSION
    assert record["actuator_count"] == len(taxonomy.actuator_names)
    assert record["ordered_muscle_schema_sha256"] == ordered_muscle_schema_hash(taxonomy.actuator_names)
    assert record["taxonomy_fingerprint"] == taxonomy.fingerprint


def test_reordered_names_report_the_divergent_index():
    taxonomy = _taxonomy()
    names = list(taxonomy.actuator_names)
    names[0], names[1] = names[1], names[0]
    with pytest.raises(ValueError, match="diverge at index 0"):
        assert_taxonomy_matches_ordered_muscles(taxonomy, names, context="probe")


def test_truncated_names_report_both_counts():
    taxonomy = _taxonomy()
    with pytest.raises(ValueError, match="but the synergy schema describes"):
        assert_taxonomy_matches_ordered_muscles(
            taxonomy,
            list(taxonomy.actuator_names)[:-1],
            context="probe",
        )


def test_duplicate_and_empty_names_are_rejected():
    taxonomy = _taxonomy()
    names = list(taxonomy.actuator_names)
    with pytest.raises(ValueError, match="must be unique"):
        assert_taxonomy_matches_ordered_muscles(
            taxonomy,
            [names[0]] * len(names),
            context="probe",
        )
    with pytest.raises(ValueError, match="must be non-empty"):
        assert_taxonomy_matches_ordered_muscles(taxonomy, [], context="probe")


def test_artifact_hash_mismatch_fails_closed():
    taxonomy = _taxonomy()
    with pytest.raises(ValueError, match="does not match the taxonomy"):
        assert_synergy_artifact_matches_taxonomy(
            {"muscle_schema_sha256": "f" * 64},
            taxonomy=taxonomy,
            context="probe",
        )


def test_artifact_without_any_declared_hash_is_rejected():
    taxonomy = _taxonomy()
    with pytest.raises(ValueError, match="must publish one of"):
        assert_synergy_artifact_matches_taxonomy(
            {"muscle_names": list(taxonomy.actuator_names)},
            taxonomy=taxonomy,
            context="probe",
        )


def test_artifact_names_are_checked_element_wise_not_only_by_hash():
    taxonomy = _taxonomy()
    names = list(taxonomy.actuator_names)
    names[0], names[1] = names[1], names[0]
    manifest = {
        "muscle_names": names,
        # A self-consistent artifact: the hash matches its own reordered names,
        # so only a comparison against the taxonomy can catch the reordering.
        "muscle_schema_sha256": ordered_muscle_schema_hash(names),
    }
    with pytest.raises(ValueError, match="diverge at index 0"):
        assert_synergy_artifact_matches_taxonomy(
            manifest,
            taxonomy=taxonomy,
            context="probe",
        )


@pytest.mark.parametrize(
    "grouping_class",
    [
        # The three anatomy classes, singular.
        "hard_line_group",
        "soft_compartment_group",
        "observation_aggregate",
        # The taxonomy's own top-level keys are plural, and build_intra_muscle_spec
        # takes the plural names, so a copied plural key is the single most likely
        # mislabeling.  A blacklist of singular spellings would accept all of these.
        "hard_line_groups",
        "soft_compartment_groups",
        "observation_aggregates",
        # Case variants and near-misses must not slip past either.
        "Hard_Line_Group",
        "HARD_LINE_GROUP",
        "anatomical_hard_line",
        "functional_synergy_regions",
    ],
)
def test_only_the_functional_class_may_be_bound(grouping_class):
    taxonomy = _taxonomy()
    with pytest.raises(ValueError, match="may be bound here"):
        assert_synergy_artifact_matches_taxonomy(
            {
                "grouping_class": grouping_class,
                "muscle_schema_sha256": taxonomy_ordered_muscle_schema_hash(taxonomy),
            },
            taxonomy=taxonomy,
            context="probe",
        )


@pytest.mark.parametrize(
    "grouping_class",
    ["functional_synergy_region", "Functional_Synergy_Region"],
)
def test_the_functional_class_is_accepted_case_insensitively(grouping_class):
    taxonomy = _taxonomy()
    record = assert_synergy_artifact_matches_taxonomy(
        {
            "grouping_class": grouping_class,
            "muscle_schema_sha256": taxonomy_ordered_muscle_schema_hash(taxonomy),
        },
        taxonomy=taxonomy,
        context="probe",
    )
    assert record["grouping_class"] == grouping_class


def test_verified_hash_fields_are_reported():
    taxonomy = _taxonomy()
    expected = taxonomy_ordered_muscle_schema_hash(taxonomy)
    record = assert_synergy_artifact_matches_taxonomy(
        {
            "grouping_class": "functional_synergy_region",
            "muscle_schema_sha256": expected,
            "ordered_muscle_schema_sha256": expected,
        },
        taxonomy=taxonomy,
        context="probe",
    )
    assert record["verified_synergy_hash_fields"] == [
        "muscle_schema_sha256",
        "ordered_muscle_schema_sha256",
    ]
    assert record["grouping_class"] == "functional_synergy_region"


@pytest.mark.skipif(
    not (SHIPPED_TAXONOMY.exists() and SHIPPED_REGIONS.exists()),
    reason="shipped MyoFullBody taxonomy or synergy region asset is absent",
)
def test_shipped_forehand_clear_regions_bind_to_the_shipped_taxonomy():
    """The checked-in assets really are the same 354-channel schema."""

    from musclemimic.physiology.anatomical_groups import load_anatomical_taxonomy

    taxonomy = load_anatomical_taxonomy(SHIPPED_TAXONOMY)
    regions = json.loads(SHIPPED_REGIONS.read_text(encoding="utf-8"))
    record = assert_synergy_artifact_matches_taxonomy(
        regions,
        taxonomy=taxonomy,
        context="forehand_clear_myofullbody_354_regions_v1",
    )
    assert record["actuator_count"] == 354
    assert record["verified_synergy_hash_fields"] == ["ordered_muscle_schema_sha256"]
    assert record["grouping_class"] == "functional_synergy_region"


def test_binding_record_publishes_real_model_provenance():
    """A successful binding must never report empty provenance."""

    taxonomy = _taxonomy()
    record = assert_taxonomy_matches_ordered_muscles(
        taxonomy,
        taxonomy.actuator_names,
        context="probe",
    )
    assert record["actuator_schema_hash"] == taxonomy.stable_model_binding["actuator_schema_hash"]
    assert record["runtime_model_hash"] == taxonomy.compiled_runtime_audit["runtime_model_hash"]
    assert record["actuator_schema_hash"]
    assert record["runtime_model_hash"]


def _grouping_file(tmp_path: Path, names, *, schema_names=None) -> Path:
    """Write an indexed regional grouping over ``names``, hashed over its own order."""

    path = tmp_path / "regional_grouping.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "regional_muscle_grouping_v1",
                "grouping_class": "functional_synergy_region",
                "ordered_muscle_schema_sha256": ordered_muscle_schema_sha256(
                    schema_names if schema_names is not None else names
                ),
                "regions": [
                    {
                        "name": "whole",
                        "index_ranges": [[0, len(names)]],
                        "boundary_names": [[names[0], names[-1]]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_load_grouping_json_binds_to_the_taxonomy_when_one_is_supplied(tmp_path):
    """The production loader is the gate, not just the validator's own tests."""

    from musclemimic.synergy.grouping import load_grouping_json

    taxonomy = _taxonomy()
    names = list(taxonomy.actuator_names)
    groups = load_grouping_json(
        _grouping_file(tmp_path, names),
        muscle_names=names,
        taxonomy=taxonomy,
    )
    assert groups == {"whole": tuple(range(len(names)))}


def test_load_grouping_json_rejects_a_dataset_order_the_taxonomy_disagrees_with(tmp_path):
    """A self-consistent grouping over a permuted dataset order must not load.

    Without ``taxonomy`` this file is perfectly valid: its hash matches the dataset
    order it is applied to.  Only the taxonomy reveals that the order is not the
    compiled model's, which is what every intra-muscle group index means.
    """

    from musclemimic.synergy.grouping import load_grouping_json

    taxonomy = _taxonomy()
    permuted = list(taxonomy.actuator_names)
    permuted[0], permuted[1] = permuted[1], permuted[0]
    path = _grouping_file(tmp_path, permuted)

    assert load_grouping_json(path, muscle_names=permuted) == {"whole": tuple(range(len(permuted)))}
    with pytest.raises(ValueError, match="does not match the taxonomy ordered muscle"):
        load_grouping_json(path, muscle_names=permuted, taxonomy=taxonomy)

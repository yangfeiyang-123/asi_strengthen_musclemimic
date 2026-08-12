from __future__ import annotations

import pytest

from musclemimic.badminton.action_release import validate_action_release


@pytest.mark.parametrize(
    ("action", "train_count", "val_count", "formal"),
    (
        ("forehand_clear", 22, 5, True),
        ("forehand_lift", 12, 4, True),
        ("chinajump", 8, 2, False),
    ),
)
def test_registered_action_release_binds_exact_split_and_files(
    action,
    train_count,
    val_count,
    formal,
):
    report = validate_action_release(action)

    assert report["passed"] is True, report["errors"]
    assert len(report["train_motions"]) == train_count
    assert len(report["validation_motions"]) == val_count
    assert len(report["file_inventory"]) == train_count + val_count
    assert report["formal_release_manifest"] is formal
    assert len(report["release_binding_sha256"]) == 64
    for row in report["file_inventory"]:
        assert len(row["source_sha256"]) == 64
        assert len(row["cache_sha256"]) == 64


def test_chinajump_release_does_not_overstate_legacy_evidence() -> None:
    report = validate_action_release("chinajump")

    assert report["review_evidence_kind"] == "legacy_qc_decision_document"
    assert report["evidence_limitations"] == [
        "no_structured_json_release_manifest",
        "no_content_bound_structured_visual_qc_report",
    ]

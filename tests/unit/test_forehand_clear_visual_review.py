from musclemimic.badminton.visual_review import (
    LEGACY_REVIEW_SCHEMA_VERSION,
    REVIEW_SCHEMA_VERSION,
    STAGE1_REVIEW_KIND,
    STAGE2_REVIEW_KIND,
    validate_visual_review,
)


def _review(kind=STAGE1_REVIEW_KIND):
    clips = []
    for index in range(5):
        clip = {
            "review_kind": kind,
            "motion": f"heldout/{index}",
            "artifact": f"videos/{index}.mp4",
            "major_swing_complete": True,
            "root_tracking_spike_free": True,
            "right_hand_tracking_spike_free": True,
            "passed": True,
            "notes": f"motion {index} reviewed frame by frame",
        }
        if kind == STAGE2_REVIEW_KIND:
            clip.update(
                {
                    "racket_head_trajectory_ok": True,
                    "racket_face_orientation_ok": True,
                }
            )
        clips.append(clip)
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "review_kind": kind,
        "passed": True,
        "clips": clips,
    }


def test_stage1_visual_review_accepts_five_structured_distinct_clips():
    report = validate_visual_review(
        _review(),
        required_clips=5,
        required_review_kind=STAGE1_REVIEW_KIND,
    )

    assert report["passed"] is True
    assert report["production_eligible"] is True
    assert report["distinct_motion_count"] == 5


def test_stage1_missing_or_false_structured_field_fails_closed():
    missing = _review()
    missing["clips"][1].pop("major_swing_complete")
    missing["clips"][2]["notes"] = ""
    report = validate_visual_review(
        missing,
        required_review_kind=STAGE1_REVIEW_KIND,
    )
    assert report["passed"] is False
    assert any("major_swing_complete must be true" in error for error in report["errors"])
    assert any("notes must be a non-empty string" in error for error in report["errors"])

    false_spike = _review()
    false_spike["clips"][0]["root_tracking_spike_free"] = False
    false_spike["clips"][3]["right_hand_tracking_spike_free"] = False
    report = validate_visual_review(
        false_spike,
        required_review_kind=STAGE1_REVIEW_KIND,
    )
    assert report["passed"] is False
    assert any("root_tracking_spike_free must be true" in error for error in report["errors"])
    assert any("right_hand_tracking_spike_free must be true" in error for error in report["errors"])


def test_stage2_requires_racket_trajectory_and_face_orientation_signoff():
    payload = _review(STAGE2_REVIEW_KIND)
    assert validate_visual_review(
        payload,
        required_review_kind=STAGE2_REVIEW_KIND,
    )["passed"] is True

    payload["clips"][0].pop("racket_head_trajectory_ok")
    payload["clips"][1]["racket_face_orientation_ok"] = False
    report = validate_visual_review(
        payload,
        required_review_kind=STAGE2_REVIEW_KIND,
    )
    assert report["passed"] is False
    assert any("racket_head_trajectory_ok must be true" in error for error in report["errors"])
    assert any("racket_face_orientation_ok must be true" in error for error in report["errors"])


def test_visual_review_fails_on_duplicate_motion_or_artifact():
    payload = _review()
    payload["clips"][4]["motion"] = payload["clips"][0]["motion"]
    payload["clips"][3]["artifact"] = payload["clips"][0]["artifact"]
    report = validate_visual_review(
        payload,
        required_review_kind=STAGE1_REVIEW_KIND,
    )
    assert report["passed"] is False
    assert any("motions must be distinct" in error for error in report["errors"])
    assert any("artifacts must be distinct" in error for error in report["errors"])


def test_wrong_review_kind_cannot_cross_stage_gate():
    payload = _review(STAGE1_REVIEW_KIND)
    report = validate_visual_review(
        payload,
        required_review_kind=STAGE2_REVIEW_KIND,
    )
    assert report["passed"] is False
    assert any("review_kind must be 'stage2_racket'" in error for error in report["errors"])


def test_legacy_schema_is_readable_but_never_production_eligible():
    payload = {
        "schema_version": LEGACY_REVIEW_SCHEMA_VERSION,
        "passed": True,
        "clips": [
            {
                "motion": f"heldout/{index}",
                "artifact": f"videos/{index}.mp4",
                "passed": True,
            }
            for index in range(5)
        ],
    }
    compatibility = validate_visual_review(payload)
    assert compatibility["passed"] is True
    assert compatibility["legacy_compatible"] is True
    assert compatibility["production_eligible"] is False

    production = validate_visual_review(
        payload,
        required_review_kind=STAGE1_REVIEW_KIND,
    )
    assert production["passed"] is False
    assert any("legacy schema is non-production only" in error for error in production["errors"])


def test_visual_review_requires_exact_heldout_motion_set():
    payload = _review()
    expected = [f"heldout/{index}.npz" for index in range(5)]
    assert validate_visual_review(
        payload,
        expected_motions=expected,
        required_review_kind=STAGE1_REVIEW_KIND,
    )["passed"] is True

    payload["clips"][4]["motion"] = "train/not-heldout"
    report = validate_visual_review(
        payload,
        expected_motions=expected,
        required_review_kind=STAGE1_REVIEW_KIND,
    )
    assert report["passed"] is False
    assert report["missing_expected_motions"] == ["4"]
    assert report["unexpected_motions"] == ["not-heldout"]


def test_visual_review_candidate_requires_every_clip_from_same_checkpoint():
    candidate = {
        "checkpoint_path": "/run/checkpoint_30",
        "checkpoint_content_sha256": "a" * 64,
        "update_number": 30,
        "global_timestep": 3000,
    }
    payload = _review()
    payload["candidate"] = candidate
    for clip in payload["clips"]:
        clip["candidate"] = candidate.copy()
    assert validate_visual_review(
        payload,
        required_review_kind=STAGE1_REVIEW_KIND,
        expected_candidate=candidate,
    )["passed"] is True

    payload["clips"][2]["candidate"]["update_number"] = 20
    report = validate_visual_review(
        payload,
        required_review_kind=STAGE1_REVIEW_KIND,
        expected_candidate=candidate,
    )
    assert report["passed"] is False
    assert any("differs from the promoted checkpoint" in error for error in report["errors"])

"""Structured, fail-closed human visual gates for Stage 1 and Stage 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


LEGACY_REVIEW_SCHEMA_VERSION = "forehand_clear_visual_review_v1"
REVIEW_SCHEMA_VERSION = "forehand_clear_visual_review_v2"
REPORT_SCHEMA_VERSION = "forehand_clear_visual_review_validation_v2"
CANDIDATE_SCHEMA_VERSION = "forehand_clear_visual_review_candidate_v2"

STAGE1_REVIEW_KIND = "stage1_body"
STAGE2_REVIEW_KIND = "stage2_racket"
REVIEW_KINDS = (STAGE1_REVIEW_KIND, STAGE2_REVIEW_KIND)

_COMMON_REQUIRED_TRUE_FIELDS = (
    "major_swing_complete",
    "root_tracking_spike_free",
    "right_hand_tracking_spike_free",
    "passed",
)
_STAGE2_REQUIRED_TRUE_FIELDS = (
    "racket_head_trajectory_ok",
    "racket_face_orientation_ok",
)


def _motion_id(value: str) -> str:
    """Normalize a review label without weakening exact held-out membership checks."""

    name = value.strip().replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return name[:-4] if name.endswith(".npz") else name


def validate_visual_review(
    payload: Mapping[str, Any],
    *,
    required_clips: int = 5,
    expected_motions: Iterable[str] | None = None,
    required_review_kind: str | None = None,
    expected_candidate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a human-authored visual review artifact.

    Supplying ``required_review_kind`` activates the production contract.  The
    legacy v1 schema remains readable without it for historical/non-production
    analysis, but can never satisfy a Stage-1 or Stage-2 production gate.
    """

    if required_clips <= 0:
        raise ValueError("required_clips must be positive")
    if required_review_kind is not None and required_review_kind not in REVIEW_KINDS:
        raise ValueError(f"unsupported review kind: {required_review_kind!r}")
    expected = tuple(dict.fromkeys(_motion_id(str(item)) for item in (expected_motions or ())))
    clips = payload.get("clips")
    errors: list[str] = []
    source_schema = payload.get("schema_version")
    is_legacy = source_schema == LEGACY_REVIEW_SCHEMA_VERSION
    if source_schema not in {REVIEW_SCHEMA_VERSION, LEGACY_REVIEW_SCHEMA_VERSION}:
        errors.append(
            f"schema_version must be {REVIEW_SCHEMA_VERSION!r} "
            f"(or legacy {LEGACY_REVIEW_SCHEMA_VERSION!r} outside production)"
        )
    if required_review_kind is not None and source_schema != REVIEW_SCHEMA_VERSION:
        errors.append(
            f"production {required_review_kind} gate requires schema_version "
            f"{REVIEW_SCHEMA_VERSION!r}; legacy schema is non-production only"
        )

    payload_kind = payload.get("review_kind")
    if source_schema == REVIEW_SCHEMA_VERSION:
        if payload_kind not in REVIEW_KINDS:
            errors.append(f"review_kind must be one of {list(REVIEW_KINDS)!r}")
        if required_review_kind is not None and payload_kind != required_review_kind:
            errors.append(
                f"review_kind must be {required_review_kind!r}, got {payload_kind!r}"
            )
    review_kind = required_review_kind or (
        str(payload_kind) if payload_kind in REVIEW_KINDS else None
    )
    if not isinstance(clips, list):
        clips = []
        errors.append("clips must be a JSON list")
    if len(clips) != required_clips:
        errors.append(f"requires exactly {required_clips} clips, got {len(clips)}")

    reviewed: list[dict[str, Any]] = []
    motions: list[str] = []
    artifacts: list[str] = []
    top_candidate = payload.get("candidate")
    if expected_candidate is not None:
        if not isinstance(top_candidate, Mapping):
            errors.append("top-level candidate checkpoint identity is required")
        elif dict(top_candidate) != dict(expected_candidate):
            errors.append("top-level candidate differs from the promoted checkpoint")
    for index, row in enumerate(clips):
        if not isinstance(row, Mapping):
            errors.append(f"clips[{index}] must be a JSON object")
            continue
        motion = str(row.get("motion", "")).strip()
        artifact = str(row.get("artifact", "")).strip()
        if not motion:
            errors.append(f"clips[{index}].motion is required")
        if not artifact:
            errors.append(f"clips[{index}].artifact is required")
        structured: dict[str, Any] = {}
        if source_schema == REVIEW_SCHEMA_VERSION:
            if row.get("review_kind") != review_kind:
                errors.append(
                    f"clips[{index}].review_kind must be {review_kind!r}"
                )
            required_true_fields = list(_COMMON_REQUIRED_TRUE_FIELDS)
            if review_kind == STAGE2_REVIEW_KIND:
                required_true_fields.extend(_STAGE2_REQUIRED_TRUE_FIELDS)
            for field in required_true_fields:
                value = row.get(field)
                structured[field] = value
                if value is not True:
                    errors.append(f"clips[{index}].{field} must be true")
            notes = row.get("notes")
            structured["notes"] = notes
            if not isinstance(notes, str) or not notes.strip():
                errors.append(f"clips[{index}].notes must be a non-empty string")
            if expected_candidate is not None:
                clip_candidate = row.get("candidate")
                if not isinstance(clip_candidate, Mapping):
                    errors.append(
                        f"clips[{index}].candidate checkpoint identity is required"
                    )
                elif dict(clip_candidate) != dict(expected_candidate):
                    errors.append(
                        f"clips[{index}].candidate differs from the promoted checkpoint"
                    )
        else:
            structured["passed"] = row.get("passed")
            if row.get("passed") is not True:
                errors.append(f"clips[{index}].passed must be true")
        if motion:
            motions.append(_motion_id(motion))
        if artifact:
            artifacts.append(artifact)
        reviewed.append(
            {
                "motion": motion,
                "artifact": artifact,
                "review_kind": row.get("review_kind"),
                **structured,
            }
        )

    distinct = sorted(set(motions))
    if len(distinct) != required_clips:
        errors.append(
            f"requires exactly {required_clips} distinct reviewed motions, got {len(distinct)}"
        )
    if len(distinct) != len(motions):
        errors.append("visual review motions must be distinct")
    if len(set(artifacts)) != len(artifacts):
        errors.append("visual review artifacts must be distinct")
    missing_expected = sorted(set(expected) - set(distinct))
    unexpected = sorted(set(distinct) - set(expected)) if expected else []
    if missing_expected:
        errors.append(f"missing expected validation motions: {missing_expected}")
    if unexpected:
        errors.append(f"review contains non-validation motions: {unexpected}")
    if payload.get("passed") is not True:
        errors.append("top-level passed must be true")

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "source_schema_version": source_schema,
        "review_kind": review_kind,
        "legacy_compatible": bool(is_legacy),
        "production_eligible": bool(
            source_schema == REVIEW_SCHEMA_VERSION
            and payload_kind in REVIEW_KINDS
            and payload_kind == review_kind
        ),
        "passed": not errors,
        "required_clips": int(required_clips),
        "reviewed_clip_count": len(reviewed),
        "distinct_motion_count": len(distinct),
        "distinct_motions": distinct,
        "expected_motions": list(expected),
        "missing_expected_motions": missing_expected,
        "unexpected_motions": unexpected,
        "clips": reviewed,
        "errors": errors,
        "candidate": None if not isinstance(top_candidate, Mapping) else dict(top_candidate),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", required=True)
    parser.add_argument("--required_clips", type=int, default=5)
    parser.add_argument(
        "--review-kind",
        choices=REVIEW_KINDS,
        default=None,
        help=(
            "Production gate kind. Omit only to inspect legacy/non-production "
            "review files."
        ),
    )
    parser.add_argument(
        "--expected_motion",
        nargs="*",
        default=None,
        help="Optional exact held-out motion names required by this gate.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Production candidate checkpoint; binds every reviewed clip to one update.",
    )
    parser.add_argument("--require_pass", action="store_true")
    args = parser.parse_args()
    payload = json.loads(Path(args.review).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SystemExit("visual review root must be a JSON object")
    expected_candidate = None
    if args.checkpoint is not None:
        from musclemimic.badminton.promotion_artifact import checkpoint_identity

        expected_candidate = checkpoint_identity(args.checkpoint)
    report = validate_visual_review(
        payload,
        required_clips=args.required_clips,
        expected_motions=args.expected_motion,
        required_review_kind=args.review_kind,
        expected_candidate=expected_candidate,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 2 if args.require_pass and not report["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

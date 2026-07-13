from __future__ import annotations

import json
from pathlib import Path

from musclemimic.badminton.scripts.finalize_raw_smooth_visual_qc import (
    VISUAL_REPORT_RELATIVE_PATH,
    validate_report,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_final_visual_qc_binds_all_semantic_metrics_and_reviewed_artifacts(
    tmp_path: Path,
) -> None:
    canonical = REPO_ROOT / VISUAL_REPORT_RELATIVE_PATH
    validation = validate_report(REPO_ROOT, canonical)

    assert validation["passed"] is True
    assert validation["motion_count"] == 27

    payload = json.loads(canonical.read_text(encoding="utf-8"))
    assert all(
        all(value is not None for value in motion["semantic_swing_metrics"].values())
        for motion in payload["motions"]
    )
    tampered = tmp_path / "visual_qc_report.json"
    payload["motions"][0]["semantic_swing_metrics"][
        "right_hand_path_length_m"
    ] = None
    tampered.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    failed = validate_report(REPO_ROOT, tampered)
    assert failed["passed"] is False
    assert any("right_hand_path_length_m" in error for error in failed["errors"])

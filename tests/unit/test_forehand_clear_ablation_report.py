from __future__ import annotations

from BadmintonMimic.scripts.build_forehand_clear_ablation_report import render_markdown_report


def test_render_markdown_report_contains_ranked_arms():
    rows = [
        {"arm": "A0", "success_rate": 0.1, "backcourt_rate": 0.0},
        {"arm": "A1", "success_rate": 0.4, "backcourt_rate": 0.2},
    ]

    report = render_markdown_report(rows)

    assert "# ForehandClear Racket Ablation Summary" in report
    assert "| A1 |" in report
    assert report.index("| A1 |") < report.index("| A0 |")

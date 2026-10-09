from __future__ import annotations

import json

from musclemimic.badminton.scripts.build_forehand_clear_ablation_report import (
    aggregate_ablation_records,
    load_ablation_jsonl,
    write_ablation_report,
)


def test_ablation_report_tracks_pairing_ci_failures_and_fingerprints(tmp_path):
    rows = []
    for arm, delta in (("A0", 0.0), ("A1", 0.1)):
        for seed in (0, 1, 2):
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "feed_uid": "feed",
                    "motion_uid": f"motion-{seed}",
                    "status": "ok",
                    "metrics": {"hit_rate": 0.7 + delta + 0.01 * seed, "landing_error_m": 1.0 - delta},
                    "config_fingerprint": "a" * 64,
                    "checkpoint_fingerprint": ("b" if arm == "A0" else "c") * 64,
                    "data_fingerprint": "d" * 64,
                    "basis_fingerprint": "e" * 64,
                    "feed_fingerprint": "f" * 64,
                }
            )
    rows.append(
        {
            "arm": "A1",
            "seed": 9,
            "status": "failed",
            "error": "numerical failure",
            "metrics": {},
            "config_fingerprint": "a" * 64,
            "checkpoint_fingerprint": "c" * 64,
            "data_fingerprint": "d" * 64,
        }
    )
    source = tmp_path / "runs.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    summary = aggregate_ablation_records(
        load_ablation_jsonl(source), baseline_arm="A0", bootstrap_samples=200, seed=3
    )
    assert summary["arms"]["A1"]["failed_runs"] == 1
    assert summary["paired_effects"]["A1"]["metrics"]["hit_rate"]["paired_n"] == 3
    assert summary["paired_effects"]["A1"]["metrics"]["hit_rate"]["mean_difference"] > 0
    paths = write_ablation_report(summary, tmp_path / "report")
    assert all(path.is_file() for path in paths.values())

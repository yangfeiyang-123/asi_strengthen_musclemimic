from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mocap_to_moshpp"))

from mocap_to_moshpp.preprocess import preprocess_marker_npz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npz", required=True)
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--report_json", default="outputs/preprocess_report.json")
    ap.add_argument("--max_gap", type=int, default=10)
    ap.add_argument("--outlier_speed_percentile", type=float, default=99.5)
    ap.add_argument("--outlier_action", choices=["report", "nan"], default="report")
    ap.add_argument("--smooth", action="store_true")
    ap.add_argument("--smooth_window", type=int, default=5)
    args = ap.parse_args()
    report = preprocess_marker_npz(
        args.in_npz,
        args.out_npz,
        args.report_json,
        max_gap=args.max_gap,
        outlier_speed_percentile=args.outlier_speed_percentile,
        outlier_action=args.outlier_action,
        smooth=args.smooth,
        smooth_window=args.smooth_window,
    )
    print(report)


if __name__ == "__main__":
    main()

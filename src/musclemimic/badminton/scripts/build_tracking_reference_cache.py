#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from musclemimic.badminton.asi.tracking_cache import build_tracking_reference_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a contact-aware tracking cache from an Optimized-WHAM reference bundle.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--control-dt", required=True, type=float)
    parser.add_argument("--allow-low-quality", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_tracking_reference_cache(
        args.manifest,
        args.out_dir,
        control_dt=args.control_dt,
        allow_low_quality=args.allow_low_quality,
    )
    print(f"tracking_cache: {result.cache_npz}")
    print(f"retarget_report: {result.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

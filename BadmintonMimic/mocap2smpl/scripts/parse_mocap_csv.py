from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mocap_to_moshpp"))

from mocap_to_moshpp.io_csv import parse_mocap_csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name_map", required=True)
    ap.add_argument("--meters_out")
    ap.add_argument("--min_valid_frames", type=int, default=10)
    ap.add_argument("--no_merge_duplicate_names", action="store_true")
    args = ap.parse_args()
    result = parse_mocap_csv(
        args.csv,
        args.out,
        args.name_map,
        args.meters_out,
        min_valid_frames=args.min_valid_frames,
        merge_duplicate_names=not args.no_merge_duplicate_names,
    )
    print(result)


if __name__ == "__main__":
    main()

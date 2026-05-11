from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mocap_to_moshpp"))

from mocap_to_moshpp.marker_names import normalize_marker_npz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markers_npz", required=True)
    ap.add_argument("--out_map", required=True)
    ap.add_argument("--out_npz", required=True)
    args = ap.parse_args()
    mapping = normalize_marker_npz(args.markers_npz, args.out_map, args.out_npz)
    print(f"normalized {len(mapping['markers'])} marker labels")


if __name__ == "__main__":
    main()

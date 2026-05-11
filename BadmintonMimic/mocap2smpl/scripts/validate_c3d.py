from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mocap_to_moshpp"))

from mocap_to_moshpp.c3d_validator import validate_c3d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c3d", required=True)
    ap.add_argument("--ref_npz")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()
    report = validate_c3d(args.c3d, args.ref_npz, args.out_json)
    print(report)


if __name__ == "__main__":
    main()

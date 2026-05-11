from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mocap_to_moshpp"))

from mocap_to_moshpp.c3d_writer import write_c3d_from_npz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npz", required=True)
    ap.add_argument("--out_c3d", required=True)
    ap.add_argument("--units", default="mm")
    args = ap.parse_args()
    print(write_c3d_from_npz(args.in_npz, args.out_c3d, args.units))


if __name__ == "__main__":
    main()

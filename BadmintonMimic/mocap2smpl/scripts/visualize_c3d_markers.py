from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mocap_to_moshpp"))

from mocap_to_moshpp.visualize import visualize_c3d_markers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c3d", required=True)
    ap.add_argument("--out_mp4", required=True)
    ap.add_argument("--out_dir")
    args = ap.parse_args()
    print(visualize_c3d_markers(args.c3d, args.out_mp4, args.out_dir))


if __name__ == "__main__":
    main()

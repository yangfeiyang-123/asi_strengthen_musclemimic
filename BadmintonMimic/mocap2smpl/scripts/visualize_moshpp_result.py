from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mocap_to_moshpp"))

from mocap_to_moshpp.visualize import visualize_amass_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--amass_npz", required=True)
    ap.add_argument("--c3d")
    ap.add_argument("--body_model_dir")
    ap.add_argument("--out_mp4", required=True)
    args = ap.parse_args()
    print(visualize_amass_result(args.amass_npz, args.c3d, args.out_mp4))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mocap_to_moshpp"))

from mocap_to_moshpp.moshpp_config import prepare_moshpp_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c3d", required=True)
    ap.add_argument("--marker_name_map", required=True)
    ap.add_argument("--moshpp_dir", required=True)
    ap.add_argument("--body_model_dir", required=True)
    ap.add_argument("--model_type", choices=["smpl", "smplh", "smplx"], default="smplh")
    ap.add_argument("--gender", choices=["neutral", "male", "female"], default="neutral")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    print(prepare_moshpp_config(**vars(args)))


if __name__ == "__main__":
    main()

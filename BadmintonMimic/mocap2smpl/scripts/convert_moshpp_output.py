from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mocap_to_moshpp"))

from mocap_to_moshpp.output_convert import convert_moshpp_output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--moshpp_out", required=True)
    ap.add_argument("--out_npz", required=True)
    args = ap.parse_args()
    print(convert_moshpp_output(args.moshpp_out, args.out_npz))


if __name__ == "__main__":
    main()

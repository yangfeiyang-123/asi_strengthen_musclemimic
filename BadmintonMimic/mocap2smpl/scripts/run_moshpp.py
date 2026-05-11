from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mocap_to_moshpp"))

from mocap_to_moshpp.moshpp_runner import run_moshpp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c3d", required=True)
    ap.add_argument("--config_dir", required=True)
    ap.add_argument("--moshpp_dir", required=True)
    ap.add_argument("--conda_env")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    try:
        print(run_moshpp(**vars(args)))
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(f"See {Path(args.out_dir) / 'candidate_entrypoints.txt'} and saved logs.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()

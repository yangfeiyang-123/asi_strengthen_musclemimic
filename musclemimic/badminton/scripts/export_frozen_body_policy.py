#!/usr/bin/env python3
"""Export a legacy body checkpoint into a lightweight frozen-policy artifact."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Legacy Orbax checkpoint directory.")
    parser.add_argument("--out", required=True, help="Output artifact directory.")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Write manifest/schema files only. Does not restore heavy checkpoint tensors.",
    )
    parser.add_argument(
        "--restore-tensors",
        action="store_true",
        help="Restore actor params/run_stats from Orbax and write params.npz/run_stats.npz.",
    )
    args = parser.parse_args()

    if args.metadata_only and args.restore_tensors:
        parser.error("--metadata-only and --restore-tensors are mutually exclusive")

    from environment.overall_environment.src.frozen_body_policy import export_frozen_body_policy

    manifest = export_frozen_body_policy(
        args.checkpoint,
        args.out,
        restore_tensors=bool(args.restore_tensors and not args.metadata_only),
    )
    print(json.dumps(asdict(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

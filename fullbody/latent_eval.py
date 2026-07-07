"""Inspect a latent checkpoint's saved metrics and action-mask contract."""

from __future__ import annotations

import argparse
import json

from musclemimic.latent_muscle.checkpoint import load_latent_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output_json", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    checkpoint = load_latent_checkpoint(args.checkpoint_dir)
    report = {
        "checkpoint_dir": args.checkpoint_dir,
        "action_mask": checkpoint["action_mask"],
        "config": checkpoint["config"],
        "eval_metrics": checkpoint["eval_metrics"],
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

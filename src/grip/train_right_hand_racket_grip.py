from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.grip.evaluate_right_hand_racket_grip import evaluate
from src.grip.paths import REPO_ROOT, reference_json_path, scene_xml_path, target_config_path

DEFAULT_BASELINE_OUT = REPO_ROOT / "outputs" / "right_hand_racket_grip" / "baseline_metrics.json"


def run_baseline(
    xml: str | Path = scene_xml_path(),
    targets: str | Path = target_config_path(),
    reference: str | Path = reference_json_path(),
    out: str | Path | None = None,
    *,
    steps: int = 1000,
) -> dict[str, Any]:
    """Run the deterministic baseline and write JSON metrics."""
    out_path = Path(out) if out is not None else DEFAULT_BASELINE_OUT
    metrics = evaluate(xml, targets, reference, episodes=1, steps=steps)
    metrics = {
        **metrics,
        "out": str(out_path),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the right-hand racket grip zero-action baseline.")
    parser.add_argument("--xml", type=Path, default=scene_xml_path(), help="MuJoCo XML scene path.")
    parser.add_argument("--targets", type=Path, default=target_config_path(), help="Grip target JSON path.")
    parser.add_argument("--reference", type=Path, default=reference_json_path(), help="Grip reference JSON path.")
    parser.add_argument("--out", type=Path, default=DEFAULT_BASELINE_OUT, help="Output metrics JSON path.")
    parser.add_argument("--steps", type=int, default=1000, help="Maximum zero-action steps.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    metrics = run_baseline(args.xml, args.targets, args.reference, args.out, steps=args.steps)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

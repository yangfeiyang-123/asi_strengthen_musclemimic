"""Build the incoming-shuttle hit scene.

The person stands at the center of their own half court (x=-3.35, facing the
net), the racket is a jointless rigid child of the right hand, and the shuttle starts
airborne on the opposite half as a placeholder (each episode reset overwrites
the shuttle free-joint state with a feeder sample).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from environment.overall_environment.src.build_overall_environment import build_overall_scene
from environment.overall_environment.src.paths import default_incoming_scene_path
from environment.overall_environment.src.racket_attachment import RacketAttachmentContract

INCOMING_HUMAN_ROOT_POS = np.array([-3.35, 0.0, 1.0], dtype=float)
# Placeholder pose: airborne over the opposite half, nose (+Z) pointing toward -x.
INCOMING_SHUTTLE_HOLD_QPOS = np.array(
    [3.0, 0.0, 2.5, np.sqrt(0.5), 0.0, -np.sqrt(0.5), 0.0], dtype=float
)
def build_incoming_hit_scene(
    output_xml: str | Path | None = None,
    *,
    grip_seed: str | Path | None = None,
    human_root_xy: tuple[float, float] = (-3.35, 0.0),
    shuttle_hold_qpos: np.ndarray = INCOMING_SHUTTLE_HOLD_QPOS,
    racket_attachment_contract: str | Path | RacketAttachmentContract | None = None,
) -> Path:
    out_path = Path(output_xml) if output_xml is not None else default_incoming_scene_path()
    human_root_pos = np.array(
        [float(human_root_xy[0]), float(human_root_xy[1]), float(INCOMING_HUMAN_ROOT_POS[2])],
        dtype=float,
    )
    return build_overall_scene(
        out_path,
        grip_seed=grip_seed,
        mode="training",
        attachment_mode="exact_child",
        finger_mode="removed",
        racket_attachment_contract=racket_attachment_contract,
        # The contract treats the racket as a rigid tool extension; native
        # hand-racket contacts would be redundant internal constraints.
        enable_person_racket_contact=False,
        enable_soft_weld=False,
        human_root_pos=human_root_pos,
        shuttle_qpos=np.asarray(shuttle_hold_qpos, dtype=float),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the incoming-shuttle hit MuJoCo scene.")
    parser.add_argument("--out", type=Path, default=None, help="Output XML path.")
    parser.add_argument("--grip-seed", type=Path, default=None, help="Optional grip seed JSON.")
    parser.add_argument(
        "--racket-attachment-contract",
        type=Path,
        default=None,
        help="Versioned exact-child racket attachment contract JSON.",
    )
    parser.add_argument("--human-root-x", type=float, default=-3.35)
    parser.add_argument("--human-root-y", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    out = build_incoming_hit_scene(
        args.out,
        grip_seed=args.grip_seed,
        human_root_xy=(args.human_root_x, args.human_root_y),
        racket_attachment_contract=args.racket_attachment_contract,
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

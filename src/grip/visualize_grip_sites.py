from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.grip.hand_racket_model_map import HAND_SITE_CANDIDATES, RACKET_SITE_NAMES
from src.grip.paths import scene_xml_path

HAND_SITE_NAMES = {key: names[0] for key, names in HAND_SITE_CANDIDATES.items()}
REQUIRED_SITE_NAMES = tuple(HAND_SITE_NAMES.values()) + RACKET_SITE_NAMES


def collect_site_positions(xml: str | Path) -> dict[str, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    positions: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for site_name in REQUIRED_SITE_NAMES:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            missing.append(site_name)
            continue
        positions[site_name] = np.array(data.site_xpos[site_id], dtype=float)

    if missing:
        raise ValueError(f"missing required grip sites: {', '.join(missing)}")

    return positions


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print right-hand racket grip site positions.")
    parser.add_argument("--xml", type=Path, default=scene_xml_path(), help="MuJoCo XML scene to inspect.")
    parser.add_argument("--no-viewer", action="store_true", help="Print site positions without launching the viewer.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    positions = collect_site_positions(args.xml)
    for site_name, xyz in positions.items():
        print(f"{site_name}: {xyz[0]: .6f} {xyz[1]: .6f} {xyz[2]: .6f}")

    if not args.no_viewer:
        import mujoco.viewer

        model = mujoco.MjModel.from_xml_path(str(args.xml))
        data = mujoco.MjData(model)
        mujoco.viewer.launch(model, data)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

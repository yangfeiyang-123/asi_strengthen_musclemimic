#!/usr/bin/env python3
"""Pre-generate BadmintonMimic GMR retarget caches."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_manifest(path: Path) -> list[str]:
    motions: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        motions.append(line.removesuffix(".npz"))
    if not motions:
        raise ValueError(f"No motions found in manifest: {path}")
    return motions


def _configure_env(project_root: Path, repo_root: Path) -> None:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MPLCONFIGDIR", str(project_root / "outputs" / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(project_root / "outputs" / "cache"))
    os.environ.setdefault("MUSCLEMIMIC_AMASS_PATH", str(project_root / "data" / "amass_npz"))
    os.environ.setdefault("AMASS_PATH", os.environ["MUSCLEMIMIC_AMASS_PATH"])
    os.environ.setdefault("MUSCLEMIMIC_CONVERTED_AMASS_PATH", str(repo_root / "caches" / "AMASS"))
    os.environ.setdefault("CONVERTED_AMASS_PATH", os.environ["MUSCLEMIMIC_CONVERTED_AMASS_PATH"])
    os.environ.setdefault("MUSCLEMIMIC_SMPL_MODEL_PATH", str(repo_root / "smpl_models" / "smplh"))
    os.environ.setdefault("SMPL_MODEL_PATH", os.environ["MUSCLEMIMIC_SMPL_MODEL_PATH"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--target-fps", type=int, default=30)
    args = parser.parse_args()

    project_root = _project_root()
    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root))
    _configure_env(project_root, repo_root)

    manifest = args.manifest or project_root / "manifests" / f"{args.split}_list.txt"
    motions = _load_manifest(manifest)

    from loco_mujoco.smpl.retargeting import load_retargeted_amass_trajectory

    gmr_config = {
        "src_human": "smplh",
        "target_fps": args.target_fps,
        "solver": "daqp",
        "damping": 0.5,
        "offset_to_ground": False,
        "use_velocity_limit": False,
        "use_fitted_shape": True,
        "shape_fitting_iterations": 500,
    }

    print(f"AMASS_PATH={os.environ['AMASS_PATH']}")
    print(f"CONVERTED_AMASS_PATH={os.environ['CONVERTED_AMASS_PATH']}")
    print(f"Retargeting {len(motions)} {args.split} motions")

    load_retargeted_amass_trajectory(
        "MyoFullBody",
        motions,
        retargeting_method="gmr",
        gmr_config=gmr_config,
        clear_cache=args.clear_cache,
    )
    print("[OK] Retarget cache generation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

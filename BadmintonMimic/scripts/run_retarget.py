#!/usr/bin/env python3
"""Pre-generate BadmintonMimic GMR retarget caches."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


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


def _read_motion_fps(path: Path) -> float:
    with np.load(path, allow_pickle=True) as data:
        has_mocap_framerate = "mocap_framerate" in data
        has_mocap_frame_rate = "mocap_frame_rate" in data
        if not has_mocap_framerate and not has_mocap_frame_rate:
            raise ValueError(f"Framerate not found in {path}")

        fps = None
        if has_mocap_framerate:
            fps = float(np.asarray(data["mocap_framerate"]).reshape(-1)[0])
        if has_mocap_frame_rate:
            frame_rate = float(np.asarray(data["mocap_frame_rate"]).reshape(-1)[0])
            if fps is not None and not np.isclose(fps, frame_rate, rtol=0.0, atol=1e-6):
                raise ValueError(
                    f"Inconsistent framerate fields in {path}: "
                    f"mocap_framerate={fps}, mocap_frame_rate={frame_rate}"
                )
            fps = frame_rate
        assert fps is not None
        return fps


def _validate_motion_fps(motions: list[str], amass_root: Path, expected_fps: float) -> None:
    for motion in motions:
        path = amass_root / f"{motion}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Manifest motion not found: {path}")
        fps = _read_motion_fps(path)
        if not np.isclose(fps, float(expected_fps), rtol=0.0, atol=1e-6):
            raise ValueError(f"{path} fps={fps:g}; expected {expected_fps:g}. Reconvert with --fps {expected_fps:g} --force-fps.")


def _build_gmr_config(
    target_fps: int,
    damping: float | None,
    use_velocity_limit: bool,
    ik_config_path: Path | None,
) -> dict:
    config = {
        "src_human": "smplh",
        "target_fps": target_fps,
        "solver": "daqp",
        "damping": 0.5 if damping is None else float(damping),
        "offset_to_ground": False,
        "use_velocity_limit": bool(use_velocity_limit),
        "use_fitted_shape": True,
        "shape_fitting_iterations": 500,
    }
    if ik_config_path is not None:
        config["ik_config_path"] = str(ik_config_path)
    return config


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--damping", type=float, default=None)
    parser.add_argument("--use-velocity-limit", action="store_true")
    parser.add_argument("--gmr-ik-config", type=Path, default=None)
    parser.add_argument(
        "--target-fps",
        "--fps",
        dest="target_fps",
        type=int,
        default=30,
        help="Expected AMASS-style input fps and GMR output fps. --fps is an alias.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    project_root = _project_root()
    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root))
    _configure_env(project_root, repo_root)

    manifest = args.manifest or project_root / "manifests" / f"{args.split}_list.txt"
    motions = _load_manifest(manifest)
    _validate_motion_fps(motions, project_root / "data" / "amass_npz", args.target_fps)

    from loco_mujoco.smpl.retargeting import load_retargeted_amass_trajectory

    gmr_config = _build_gmr_config(
        target_fps=args.target_fps,
        damping=args.damping,
        use_velocity_limit=args.use_velocity_limit,
        ik_config_path=args.gmr_ik_config,
    )

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

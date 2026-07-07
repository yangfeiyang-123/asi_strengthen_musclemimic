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


def _autodetect_motion_fps(motions: list[str], amass_root: Path) -> float:
    """Derive the target fps from the manifest's npz files (their mocap_framerate).

    Used when --fps is omitted: the AMASS npzs are self-describing (convert_wham_to_amass
    writes the true rate), so retargeting should honour it rather than assume a default.
    All motions in one manifest must agree.
    """
    rates: dict[str, float] = {}
    for motion in motions:
        path = amass_root / f"{motion}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Manifest motion not found: {path}")
        rates[motion] = _read_motion_fps(path)
    unique = sorted({round(r, 6) for r in rates.values()})
    if len(unique) > 1:
        raise ValueError(
            "Manifest mixes multiple frame rates; pass --fps explicitly. Per-motion: "
            + ", ".join(f"{m}={r:g}" for m, r in rates.items())
        )
    return float(next(iter(rates.values())))


def _build_gmr_config(
    target_fps: float,
    damping: float | None,
    use_velocity_limit: bool,
    ik_config_path: Path | None,
    stance_projection: dict | None = None,
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
    if stance_projection is not None:
        config["stance_projection"] = stance_projection
    return config


def _set_env_pair(primary: str, alias: str, value: Path, *, override: bool) -> None:
    if override:
        os.environ[primary] = str(value)
        os.environ[alias] = str(value)
        return
    os.environ.setdefault(primary, str(value))
    os.environ.setdefault(alias, os.environ[primary])


def _configure_env(
    project_root: Path,
    repo_root: Path,
    amass_root: Path | None = None,
    converted_amass_root: Path | None = None,
    gmr_cache_root: Path | None = None,
) -> None:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MPLCONFIGDIR", str(project_root / "outputs" / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(project_root / "outputs" / "cache"))
    _set_env_pair(
        "MUSCLEMIMIC_AMASS_PATH",
        "AMASS_PATH",
        amass_root or repo_root / "datasets" / "_global" / "amass_npz",
        override=amass_root is not None,
    )
    _set_env_pair(
        "MUSCLEMIMIC_CONVERTED_AMASS_PATH",
        "CONVERTED_AMASS_PATH",
        converted_amass_root or repo_root / "datasets" / "_global" / "muscle_trajectory" / "gmr_cache",
        override=converted_amass_root is not None,
    )
    if gmr_cache_root is not None:
        os.environ["MUSCLEMIMIC_GMR_CACHE_PATH"] = str(gmr_cache_root)
    os.environ.setdefault("MUSCLEMIMIC_SMPL_MODEL_PATH", str(repo_root / "smpl_models" / "smplh"))
    os.environ.setdefault("SMPL_MODEL_PATH", os.environ["MUSCLEMIMIC_SMPL_MODEL_PATH"])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--amass-root",
        type=Path,
        default=None,
        help="AMASS-style NPZ root. Defaults to datasets/_global/amass_npz.",
    )
    parser.add_argument(
        "--converted-amass-root",
        type=Path,
        default=None,
        help="Converted AMASS cache root used by legacy non-GMR cache paths.",
    )
    parser.add_argument(
        "--gmr-cache-root",
        type=Path,
        default=None,
        help="Direct MyoFullBody GMR cache root, e.g. datasets/<action>/muscle_trajectory/gmr_cache.",
    )
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--damping", type=float, default=None)
    parser.add_argument("--use-velocity-limit", action="store_true")
    parser.add_argument("--gmr-ik-config", type=Path, default=None)
    parser.add_argument(
        "--stance-bundle",
        type=Path,
        default=None,
        help=(
            "reference_bundle/manifest.json to project stance contacts onto the "
            "MyoFullBody skeleton (OmniRetarget-style foot-sticking). Single-motion "
            "manifests only (one bundle per motion)."
        ),
    )
    parser.add_argument("--stance-iters", type=int, default=None, help="Stance projection IK iterations.")
    parser.add_argument("--stance-damping", type=float, default=None, help="Stance projection IK damping.")
    parser.add_argument("--stance-max-step", type=float, default=None, help="Stance projection IK max step (rad).")
    parser.add_argument(
        "--target-fps",
        "--fps",
        dest="target_fps",
        type=float,
        default=None,
        help="Expected AMASS input / GMR output fps (float; e.g. 29.97 ok). "
        "If omitted, auto-detected from the manifest npz files' mocap_framerate. --fps is an alias.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    project_root = _project_root()
    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root))
    _configure_env(
        project_root,
        repo_root,
        amass_root=args.amass_root,
        converted_amass_root=args.converted_amass_root,
        gmr_cache_root=args.gmr_cache_root,
    )

    manifest = (
        args.manifest
        or repo_root / "datasets" / "_global" / "manifests" / "badmintonmimic" / f"{args.split}_list.txt"
    )
    motions = _load_manifest(manifest)
    amass_root = Path(os.environ["AMASS_PATH"])
    if args.target_fps is None:
        target_fps = _autodetect_motion_fps(motions, amass_root)
        print(f"[fps] auto-detected target_fps={target_fps:g} from manifest npz files")
    else:
        target_fps = float(args.target_fps)
        _validate_motion_fps(motions, amass_root, target_fps)

    from loco_mujoco.smpl.retargeting import load_retargeted_amass_trajectory

    stance_projection = None
    if args.stance_bundle is not None:
        if len(motions) != 1:
            raise SystemExit(
                "--stance-bundle requires a single-motion manifest (one bundle per motion); "
                f"got {len(motions)} motions in {manifest}."
            )
        if not args.stance_bundle.exists():
            raise SystemExit(f"--stance-bundle not found: {args.stance_bundle}")
        stance_projection = {"manifest": str(args.stance_bundle)}
        for key, value in (
            ("iters", args.stance_iters),
            ("damping", args.stance_damping),
            ("max_step", args.stance_max_step),
        ):
            if value is not None:
                stance_projection[key] = value

    gmr_config = _build_gmr_config(
        target_fps=target_fps,
        damping=args.damping,
        use_velocity_limit=args.use_velocity_limit,
        ik_config_path=args.gmr_ik_config,
        stance_projection=stance_projection,
    )

    print(f"AMASS_PATH={os.environ['AMASS_PATH']}")
    print(f"CONVERTED_AMASS_PATH={os.environ['CONVERTED_AMASS_PATH']}")
    if "MUSCLEMIMIC_GMR_CACHE_PATH" in os.environ:
        print(f"MUSCLEMIMIC_GMR_CACHE_PATH={os.environ['MUSCLEMIMIC_GMR_CACHE_PATH']}")
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

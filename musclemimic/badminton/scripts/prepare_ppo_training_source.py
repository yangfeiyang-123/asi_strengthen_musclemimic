#!/usr/bin/env python3
"""Prepare comparable PPO training configs for existing AMASS and reference-bundle sources."""

import argparse
import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from musclemimic.badminton.data.reference_bundle import load_reference_bundle
from musclemimic.badminton.scripts.build_config_from_manifests import _read_manifest, build_config


@dataclass(frozen=True)
class PreparedPPOTrainingSource:
    source_mode: str
    train_motions: list[str]
    val_motions: list[str]
    train_manifest: Path
    val_manifest: Path
    output_config: Path
    metadata_json: Path | None = None


@dataclass(frozen=True)
class _ReferenceBundleEntry:
    manifest_path: Path
    sequence: str
    quality_tier: str
    num_frames: int
    fps: float
    motion_npz: Path


def prepare_existing_source(
    *,
    train_manifest: str | Path,
    val_manifest: str | Path,
    output_config: str | Path,
    fps: int,
    num_envs: int,
    total_timesteps: int,
    action: str | None = None,
) -> PreparedPPOTrainingSource:
    train_path = Path(train_manifest).resolve()
    val_path = Path(val_manifest).resolve()
    output_path = Path(output_config).resolve()
    train = _read_manifest(train_path)
    val = _read_manifest(val_path)
    build_config(
        train,
        val,
        output_path,
        num_envs=num_envs,
        total_timesteps=total_timesteps,
        target_fps=int(fps),
        source_mode="existing_ppo",
        action=action,
    )
    return PreparedPPOTrainingSource(
        source_mode="existing_ppo",
        train_motions=train,
        val_motions=val,
        train_manifest=train_path,
        val_manifest=val_path,
        output_config=output_path,
    )


def prepare_reference_bundle_source(
    *,
    reference_root: str | Path | None = None,
    reference_manifest_jsonl: str | Path | None = None,
    namespace: str,
    output_config: str | Path,
    fps: int,
    num_envs: int,
    total_timesteps: int,
    include_tiers: set[str] | None = None,
    min_frames: int = 60,
    allow_low_quality: bool = False,
    val_ratio: float = 0.2,
    val_count: int | None = None,
    link_mode: str = "copy",
    amass_root: str | Path | None = None,
    manifest_dir: str | Path | None = None,
    metadata_out: str | Path | None = None,
    train_manifest: str | Path | None = None,
    val_manifest: str | Path | None = None,
    action: str | None = None,
) -> PreparedPPOTrainingSource:
    if reference_root is None and reference_manifest_jsonl is None:
        raise ValueError("reference_bundle mode requires --reference-root or --reference-manifest-jsonl.")
    if link_mode not in {"copy", "symlink"}:
        raise ValueError("link_mode must be 'copy' or 'symlink'.")

    tiers = include_tiers or {"A", "B"}
    amass_path = Path(amass_root or PROJECT_ROOT / "data" / "amass_npz").resolve()
    manifest_path = Path(manifest_dir or PROJECT_ROOT / "manifests").resolve()
    namespace_clean = _clean_namespace(namespace)
    manifest_slug = _slug(namespace_clean.replace("/", "_"))

    selected, skipped = _select_reference_bundles(
        reference_root=Path(reference_root).resolve() if reference_root is not None else None,
        reference_manifest_jsonl=Path(reference_manifest_jsonl).resolve() if reference_manifest_jsonl is not None else None,
        include_tiers=tiers,
        min_frames=int(min_frames),
        allow_low_quality=allow_low_quality,
        expected_fps=float(fps),
    )
    if len(selected) < 2:
        raise ValueError(
            "reference_bundle mode needs at least two usable bundles so train and validation are both non-empty. "
            f"selected={len(selected)} skipped={len(skipped)}"
        )

    materialized = _materialize_reference_motions(
        selected,
        amass_root=amass_path,
        namespace=namespace_clean,
        link_mode=link_mode,
    )
    train_motions, val_motions = _split_motions(materialized, val_ratio=val_ratio, val_count=val_count)

    train_manifest_path = Path(train_manifest or manifest_path / f"{manifest_slug}_train_list.txt").resolve()
    val_manifest_path = Path(val_manifest or manifest_path / f"{manifest_slug}_val_list.txt").resolve()
    _write_manifest(train_manifest_path, train_motions)
    _write_manifest(val_manifest_path, val_motions)

    output_path = Path(output_config).resolve()
    build_config(
        train_motions,
        val_motions,
        output_path,
        num_envs=num_envs,
        total_timesteps=total_timesteps,
        target_fps=int(fps),
        source_mode="reference_bundle",
        action=action,
    )

    metadata_path = Path(metadata_out or manifest_path / f"{manifest_slug}_reference_bundle_metadata.json").resolve()
    _write_metadata(
        metadata_path,
        namespace=namespace_clean,
        fps=float(fps),
        selected=selected,
        skipped=skipped,
        materialized=materialized,
        train_motions=train_motions,
        val_motions=val_motions,
        train_manifest=train_manifest_path,
        val_manifest=val_manifest_path,
        output_config=output_path,
    )

    return PreparedPPOTrainingSource(
        source_mode="reference_bundle",
        train_motions=train_motions,
        val_motions=val_motions,
        train_manifest=train_manifest_path,
        val_manifest=val_manifest_path,
        output_config=output_path,
        metadata_json=metadata_path,
    )


def _select_reference_bundles(
    *,
    reference_root: Path | None,
    reference_manifest_jsonl: Path | None,
    include_tiers: set[str],
    min_frames: int,
    allow_low_quality: bool,
    expected_fps: float,
) -> tuple[list[_ReferenceBundleEntry], list[dict[str, Any]]]:
    selected: list[_ReferenceBundleEntry] = []
    skipped: list[dict[str, Any]] = []
    for manifest_path in _iter_reference_manifest_paths(reference_root, reference_manifest_jsonl):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("version") != "contact_reference_bundle_v1":
                skipped.append(_skip(manifest_path, "unsupported_version"))
                continue
            if manifest.get("coordinate_system") != "amass_zup":
                skipped.append(_skip(manifest_path, "wrong_coordinate_system"))
                continue

            quality = dict(manifest.get("quality", {}))
            tier = str(quality.get("quality_tier", ""))
            usable = bool(quality.get("usable_for_training", False))
            if tier not in include_tiers:
                skipped.append(_skip(manifest_path, f"tier_{tier}_excluded"))
                continue
            if not usable and not allow_low_quality:
                skipped.append(_skip(manifest_path, "not_usable_for_training"))
                continue
            if int(manifest.get("num_frames", 0)) < int(min_frames):
                skipped.append(_skip(manifest_path, "too_short"))
                continue

            bundle = load_reference_bundle(manifest_path, allow_low_quality=allow_low_quality)
            if not np.isclose(bundle.fps, expected_fps, rtol=0.0, atol=1e-6):
                raise ValueError(f"{manifest_path} fps={bundle.fps:g}; expected {expected_fps:g}.")
            selected.append(
                _ReferenceBundleEntry(
                    manifest_path=manifest_path,
                    sequence=str(manifest.get("sequence", manifest_path.parent.parent.name)),
                    quality_tier=tier,
                    num_frames=int(manifest["num_frames"]),
                    fps=float(bundle.fps),
                    motion_npz=manifest_path.parent / manifest["motion_npz"],
                )
            )
        except Exception as exc:
            skipped.append(_skip(manifest_path, f"invalid:{exc}"))
    selected.sort(key=lambda item: (_natural_key(item.sequence), str(item.manifest_path)))
    return selected, skipped


def _iter_reference_manifest_paths(reference_root: Path | None, reference_manifest_jsonl: Path | None) -> Iterable[Path]:
    paths: list[Path] = []
    if reference_root is not None:
        paths.extend(reference_root.rglob("manifest.json"))
    if reference_manifest_jsonl is not None:
        for line in reference_manifest_jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("{"):
                path_value = json.loads(line)["manifest"]
            else:
                path_value = line
            paths.append(Path(path_value))
    seen: set[Path] = set()
    for path in sorted((p.resolve() for p in paths), key=lambda p: str(p)):
        if path in seen:
            continue
        seen.add(path)
        yield path


def _materialize_reference_motions(
    entries: list[_ReferenceBundleEntry],
    *,
    amass_root: Path,
    namespace: str,
    link_mode: str,
) -> list[str]:
    motions: list[str] = []
    for index, entry in enumerate(entries, start=1):
        stem = _safe_motion_stem(index, entry)
        rel_motion = f"{namespace}/{stem}"
        dst = amass_root / f"{rel_motion}.npz"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if link_mode == "symlink":
            dst.symlink_to(entry.motion_npz.resolve())
        else:
            shutil.copy2(entry.motion_npz, dst)
        motions.append(rel_motion)
    return motions


def _split_motions(motions: list[str], *, val_ratio: float, val_count: int | None) -> tuple[list[str], list[str]]:
    if len(motions) < 2:
        raise ValueError("Need at least two motions for non-empty train and val manifests.")
    if val_count is None:
        if not (0.0 < float(val_ratio) < 1.0):
            raise ValueError("val_ratio must be between 0 and 1.")
        val_count = max(1, int(round(len(motions) * float(val_ratio))))
    val_count = max(1, min(int(val_count), len(motions) - 1))
    return motions[:-val_count], motions[-val_count:]


def _write_manifest(path: Path, motions: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{motion}\n" for motion in motions), encoding="utf-8")


def _write_metadata(
    path: Path,
    *,
    namespace: str,
    fps: float,
    selected: list[_ReferenceBundleEntry],
    skipped: list[dict[str, Any]],
    materialized: list[str],
    train_motions: list[str],
    val_motions: list[str],
    train_manifest: Path,
    val_manifest: Path,
    output_config: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_mode": "reference_bundle",
        "namespace": namespace,
        "fps": fps,
        "selected_count": len(selected),
        "skipped_count": len(skipped),
        "train_count": len(train_motions),
        "val_count": len(val_motions),
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "output_config": str(output_config),
        "materialized_motions": materialized,
        "selected": [
            {
                "sequence": entry.sequence,
                "quality_tier": entry.quality_tier,
                "num_frames": entry.num_frames,
                "fps": entry.fps,
                "manifest": str(entry.manifest_path),
                "motion": materialized[index],
            }
            for index, entry in enumerate(selected)
        ],
        "skipped": skipped,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _skip(path: Path, reason: str) -> dict[str, Any]:
    return {"manifest": str(path), "reason": reason}


def _safe_motion_stem(index: int, entry: _ReferenceBundleEntry) -> str:
    digest = hashlib.sha1(str(entry.manifest_path).encode("utf-8")).hexdigest()[:8]
    return f"{index:03d}_{_slug(entry.sequence)}_{digest}"


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-")
    return slug or "motion"


def _clean_namespace(namespace: str) -> str:
    parts = [_slug(part) for part in namespace.replace("\\", "/").split("/") if part.strip()]
    if not parts:
        raise ValueError("namespace must contain at least one non-empty path component.")
    return "/".join(parts)


def _natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def _parse_tiers(value: str) -> set[str]:
    tiers = {part.strip() for part in value.split(",") if part.strip()}
    if not tiers:
        raise ValueError("--include-tiers must not be empty.")
    return tiers


def _default_existing_config() -> Path:
    return REPO_ROOT / "fullbody" / "config_specific_task" / "conf_fullbody_badminton_existing_gmr.yaml"


def _default_reference_config(namespace: str) -> Path:
    slug = _slug(_clean_namespace(namespace).replace("/", "_"))
    return REPO_ROOT / "fullbody" / "config_specific_task" / f"conf_fullbody_{slug}_reference_gmr.yaml"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-mode", choices=["existing", "existing_ppo", "reference_bundle"], default="existing_ppo")
    parser.add_argument("--fps", "--target-fps", dest="fps", type=int, default=30)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--total-timesteps", type=int, default=20480000)
    parser.add_argument("--output-config", type=Path, default=None)
    parser.add_argument(
        "--action",
        default=None,
        help="Dataset action name. Generated configs route training artifacts to datasets/<action>/training.",
    )

    parser.add_argument("--train-manifest", type=Path, default=PROJECT_ROOT / "manifests" / "train_list.txt")
    parser.add_argument("--val-manifest", type=Path, default=PROJECT_ROOT / "manifests" / "val_list.txt")

    parser.add_argument("--reference-root", type=Path, default=None)
    parser.add_argument("--reference-manifest-jsonl", type=Path, default=None)
    parser.add_argument("--namespace", default="reference_bundle/ppo")
    parser.add_argument("--include-tiers", default="A,B")
    parser.add_argument("--min-frames", type=int, default=60)
    parser.add_argument("--allow-low-quality", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--val-count", type=int, default=None)
    parser.add_argument("--link-mode", choices=["copy", "symlink"], default="copy")
    parser.add_argument("--amass-root", type=Path, default=PROJECT_ROOT / "data" / "amass_npz")
    parser.add_argument("--manifest-dir", type=Path, default=PROJECT_ROOT / "manifests")
    parser.add_argument("--metadata-out", type=Path, default=None)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    source_mode = "existing_ppo" if args.source_mode == "existing" else args.source_mode
    if source_mode == "reference_bundle":
        output_config = args.output_config or _default_reference_config(args.namespace)
        result = prepare_reference_bundle_source(
            reference_root=args.reference_root,
            reference_manifest_jsonl=args.reference_manifest_jsonl,
            namespace=args.namespace,
            output_config=output_config,
            fps=args.fps,
            num_envs=args.num_envs,
            total_timesteps=args.total_timesteps,
            include_tiers=_parse_tiers(args.include_tiers),
            min_frames=args.min_frames,
            allow_low_quality=args.allow_low_quality,
            val_ratio=args.val_ratio,
            val_count=args.val_count,
            link_mode=args.link_mode,
            amass_root=args.amass_root,
            manifest_dir=args.manifest_dir,
            metadata_out=args.metadata_out,
            action=args.action,
        )
    else:
        result = prepare_existing_source(
            train_manifest=args.train_manifest,
            val_manifest=args.val_manifest,
            output_config=args.output_config or _default_existing_config(),
            fps=args.fps,
            num_envs=args.num_envs,
            total_timesteps=args.total_timesteps,
            action=args.action,
        )

    print(f"[OK] source_mode={result.source_mode}")
    print(f"     train_manifest={result.train_manifest} ({len(result.train_motions)} motions)")
    print(f"     val_manifest={result.val_manifest} ({len(result.val_motions)} motions)")
    print(f"     config={result.output_config}")
    if result.metadata_json is not None:
        print(f"     metadata={result.metadata_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_contact_tracking_manifest(
    *,
    reference_root: str | Path,
    out: str | Path,
    include_tiers: set[str] | None = None,
    min_frames: int = 60,
    max_foot_penetration_cm: float | None = None,
    max_stance_sliding_cm: float | None = None,
) -> list[dict[str, Any]]:
    root = Path(reference_root).resolve()
    tiers = include_tiers or {"A", "B"}
    entries: list[dict[str, Any]] = []
    for manifest_path in sorted(root.rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("version") != "contact_reference_bundle_v1":
            continue
        if manifest.get("coordinate_system") != "amass_zup":
            continue
        quality = dict(manifest.get("quality", {}))
        tier = str(quality.get("quality_tier", ""))
        if tier not in tiers:
            continue
        if not bool(quality.get("usable_for_training", False)):
            continue
        num_frames = int(manifest.get("num_frames", 0))
        if num_frames < int(min_frames):
            continue
        if max_foot_penetration_cm is not None:
            value = float(quality.get("foot_penetration_max_cm_after", 0.0) or 0.0)
            if value > float(max_foot_penetration_cm):
                continue
        if max_stance_sliding_cm is not None:
            value = float(quality.get("stance_sliding_max_cm_after", 0.0) or 0.0)
            if value > float(max_stance_sliding_cm):
                continue

        entries.append(
            _with_cache_metadata(
                manifest_path,
                {
                    "manifest": str(manifest_path),
                    "sequence": str(manifest.get("sequence", manifest_path.parent.name)),
                    "quality_tier": tier,
                    "num_frames": num_frames,
                    "fps": float(manifest.get("fps", 0.0)),
                    "coordinate_system": str(manifest.get("coordinate_system", "")),
                },
            )
        )

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries),
        encoding="utf-8",
    )
    return entries


def _with_cache_metadata(manifest_path: Path, entry: dict[str, Any]) -> dict[str, Any]:
    cache_path = manifest_path.parent / "tracking_reference_cache.npz"
    report_path = manifest_path.parent / "retarget_report.json"
    if cache_path.exists():
        entry["cache_path"] = str(cache_path)
    if report_path.exists():
        entry["retarget_report_json"] = str(report_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if "effective_ref_stride" in report:
            entry["effective_ref_stride"] = float(report["effective_ref_stride"])
        if "status" in report:
            entry["retarget_status"] = str(report["status"])
    return entry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build JSONL manifest for contact-aware tracking references.")
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--include-tiers", default="A,B")
    parser.add_argument("--min-frames", type=int, default=60)
    parser.add_argument("--max-foot-penetration-cm", type=float, default=None)
    parser.add_argument("--max-stance-sliding-cm", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    entries = build_contact_tracking_manifest(
        reference_root=args.reference_root,
        out=args.out,
        include_tiers={tier.strip() for tier in args.include_tiers.split(",") if tier.strip()},
        min_frames=args.min_frames,
        max_foot_penetration_cm=args.max_foot_penetration_cm,
        max_stance_sliding_cm=args.max_stance_sliding_cm,
    )
    print(f"wrote {len(entries)} reference entries to {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

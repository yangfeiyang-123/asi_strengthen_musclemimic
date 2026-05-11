from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mocap_to_moshpp"))

from mocap_to_moshpp.utils import ensure_dir, finite_range, read_marker_npz, write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out = ensure_dir(args.out_dir)
    data = read_marker_npz(args.npz)
    markers = data["markers"].astype(float)
    names = data["marker_names"]
    fps = float(data["fps"])
    mask = np.isfinite(markers).all(axis=2)
    speeds = np.linalg.norm(np.diff(markers, axis=0), axis=2) * fps

    rows = []
    for i, name in enumerate(names):
        valid = mask[:, i]
        coords = markers[valid, i, :]
        speed = speeds[:, i]
        speed_valid = speed[np.isfinite(speed)]
        coord_range = finite_range(coords)
        max_speed = float(np.nanmax(speed_valid)) if speed_valid.size else None
        mean_speed = float(np.nanmean(speed_valid)) if speed_valid.size else None
        missing_rate = 1.0 - float(valid.mean())
        broken = bool(missing_rate > 0.2 or (max_speed is not None and mean_speed and max_speed > mean_speed * 12))
        rows.append(
            {
                "marker": name,
                "valid_frames": int(valid.sum()),
                "missing_frames": int((~valid).sum()),
                "missing_rate": missing_rate,
                "coord_min": coord_range["min"],
                "coord_max": coord_range["max"],
                "coord_mean": coord_range["mean"],
                "mean_speed": mean_speed,
                "max_speed": max_speed,
                "possible_broken_track": broken,
            }
        )

    with (out / "marker_report.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    missing_report = {
        "npz": args.npz,
        "frames": int(markers.shape[0]),
        "markers": int(markers.shape[1]),
        "fps": fps,
        "overall_missing_rate": float(1.0 - mask.mean()) if mask.size else 0.0,
        "top_missing": sorted(rows, key=lambda r: r["missing_rate"], reverse=True)[:20],
    }
    write_json(out / "marker_missing_report.json", missing_report)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(~mask.T, aspect="auto", interpolation="nearest", cmap="magma")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Marker")
    ax.set_title("Missing marker heatmap")
    fig.tight_layout()
    fig.savefig(out / "missing_heatmap.png", dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 6))
    ax3 = fig.add_subplot(111, projection="3d")
    frame_idx = markers.shape[0] // 2
    pts = markers[frame_idx]
    valid = np.isfinite(pts).all(axis=1)
    ax3.scatter(pts[valid, 0], pts[valid, 1], pts[valid, 2], s=10)
    ax3.set_title(f"Marker preview frame {frame_idx}")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    fig.tight_layout()
    fig.savefig(out / "marker_3d_preview.png", dpi=150)
    plt.close(fig)

    print("Top 20 missing markers:")
    for item in missing_report["top_missing"]:
        print(f"{item['marker']}: {item['missing_frames']} missing ({item['missing_rate']:.2%})")


if __name__ == "__main__":
    main()

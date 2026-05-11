from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .c3d_validator import read_c3d_points
from .utils import ensure_parent


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _axis_limits(markers: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    valid = np.isfinite(markers).all(axis=2)
    if not np.any(valid):
        return ((-1, 1), (-1, 1), (-1, 1))
    pts = markers[valid]
    mins = np.nanmin(pts, axis=0)
    maxs = np.nanmax(pts, axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 1.0)
    return tuple((float(c - radius), float(c + radius)) for c in center)  # type: ignore[return-value]


def save_marker_frame(markers: np.ndarray, frame: int, out_png: str | Path, title: str = "") -> None:
    plt = _setup_matplotlib()
    out_png = ensure_parent(out_png)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    pts = markers[frame]
    valid = np.isfinite(pts).all(axis=1)
    ax.scatter(pts[valid, 0], pts[valid, 1], pts[valid, 2], s=10, c="#2563eb")
    (xlim, ylim, zlim) = _axis_limits(markers)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title or f"Frame {frame}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _save_mp4_with_cv2(fig: Any, update: Any, frames: list[int], out_mp4: str | Path, fps: float) -> None:
    import cv2

    out_mp4 = str(out_mp4)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open video writer for {out_mp4}")
    try:
        for frame_idx in frames:
            update(frame_idx)
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
            rgb = buf[:, :, :3]
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def visualize_c3d_markers(c3d_path: str | Path, out_mp4: str | Path, out_dir: str | Path | None = None) -> dict[str, Any]:
    import matplotlib.animation as animation

    plt = _setup_matplotlib()
    data = read_c3d_points(c3d_path)
    markers = data["markers"]
    out_mp4 = ensure_parent(out_mp4)
    out_dir = Path(out_dir) if out_dir else out_mp4.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    save_marker_frame(markers, 0, out_dir / "c3d_frame_000.png", "C3D markers frame 0")
    save_marker_frame(markers, markers.shape[0] // 2, out_dir / "c3d_frame_mid.png", "C3D markers mid frame")

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    (xlim, ylim, zlim) = _axis_limits(markers)
    scat = ax.scatter([], [], [], s=10, c="#2563eb")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    step = max(1, markers.shape[0] // 220)
    frames = list(range(0, markers.shape[0], step))

    def update(frame_idx: int):
        pts = markers[frame_idx]
        valid = np.isfinite(pts).all(axis=1)
        vv = pts[valid]
        scat._offsets3d = (vv[:, 0], vv[:, 1], vv[:, 2])
        ax.set_title(f"Frame {frame_idx}")
        return (scat,)

    video_fps = min(max(float(data["fps"]), 1), 30)
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000.0 / max(data["fps"], 1), blit=False)
    try:
        ani.save(out_mp4, fps=video_fps)
    except Exception:
        _save_mp4_with_cv2(fig, update, frames, out_mp4, video_fps)
    plt.close(fig)
    return {"out_mp4": str(out_mp4), "frames_rendered": len(frames), "fps": data["fps"]}


def visualize_amass_result(amass_npz: str | Path, c3d: str | Path | None, out_mp4: str | Path) -> dict[str, Any]:
    import matplotlib.animation as animation

    plt = _setup_matplotlib()
    out_mp4 = ensure_parent(out_mp4)
    markers = None
    fps = 30.0
    if c3d:
        c3d_data = read_c3d_points(c3d)
        markers = c3d_data["markers"]
        fps = float(c3d_data["fps"])
    data = np.load(amass_npz, allow_pickle=True)
    trans = data["trans"] if "trans" in data.files else None
    if markers is None:
        if trans is None:
            raise ValueError("Need either --c3d markers or trans in AMASS npz for fallback visualization.")
        markers = trans[:, None, :] * 1000.0

    out_dir = out_mp4.parent
    save_marker_frame(markers, 0, out_dir / "moshpp_frame_000.png", "MoSh++ result frame 0")
    save_marker_frame(markers, markers.shape[0] // 2, out_dir / "moshpp_frame_mid.png", "MoSh++ result mid frame")

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    (xlim, ylim, zlim) = _axis_limits(markers)
    marker_scat = ax.scatter([], [], [], s=9, c="#2563eb", label="markers")
    trans_scat = ax.scatter([], [], [], s=30, c="#dc2626", label="trans") if trans is not None else None
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.legend(loc="upper right")
    frames = list(range(0, markers.shape[0], max(1, markers.shape[0] // 220)))

    def update(frame_idx: int):
        pts = markers[frame_idx]
        valid = np.isfinite(pts).all(axis=1)
        vv = pts[valid]
        marker_scat._offsets3d = (vv[:, 0], vv[:, 1], vv[:, 2])
        artists = [marker_scat]
        if trans_scat is not None:
            p = trans[min(frame_idx, len(trans) - 1)] * 1000.0
            trans_scat._offsets3d = ([p[0]], [p[1]], [p[2]])
            artists.append(trans_scat)
        ax.set_title(f"Frame {frame_idx}")
        return artists

    video_fps = min(max(fps, 1), 30)
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000.0 / max(fps, 1), blit=False)
    try:
        ani.save(out_mp4, fps=video_fps)
    except Exception:
        _save_mp4_with_cv2(fig, update, frames, out_mp4, video_fps)
    plt.close(fig)
    return {"out_mp4": str(out_mp4), "frames_rendered": len(frames)}

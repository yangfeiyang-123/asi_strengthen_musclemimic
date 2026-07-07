"""Frame-rate detection + resolution shared across the video→SMPL→muscle pipeline.

The single source of truth for a clip's frame rate is the raw video. WHAM reads it
but historically discarded it, so every downstream step had to be told the fps by hand
— and a wrong value silently time-scales the whole motion (a 30 fps clip processed at
60 fps comes out half-length / 2× speed). These helpers let every step resolve the fps
with one consistent priority:

    explicit override  >  fps stored in the input record (pkl / npz)  >  probe the video

so the pipeline is correct with zero configuration and no per-dataset patching.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional, Union

# Keys under which a frame rate may be stored in a WHAM pkl track or an AMASS-style npz.
FPS_RECORD_KEYS = ("mocap_framerate", "mocap_frame_rate", "fps", "frame_rate")
VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")


def _parse_rate(text: str) -> Optional[float]:
    """Parse an ffprobe rate string, which may be a fraction like '30000/1001'."""
    text = str(text).strip()
    if not text or text in {"0", "0/0", "N/A"}:
        return None
    try:
        if "/" in text:
            num, den = text.split("/", 1)
            den_f = float(den)
            if den_f == 0.0:
                return None
            return float(num) / den_f
        return float(text)
    except (ValueError, ZeroDivisionError):
        return None


def detect_video_fps(video_path: Union[str, Path]) -> Optional[float]:
    """Return the true frame rate of a video file, or None if it cannot be determined.

    Prefers ffprobe's ``r_frame_rate`` (an exact fraction, e.g. 30000/1001 for 29.97),
    falling back to ``avg_frame_rate`` and then to OpenCV's ``CAP_PROP_FPS``.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        return None

    for entry in ("r_frame_rate", "avg_frame_rate"):
        try:
            out = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", f"stream={entry}", "-of", "csv=p=0", str(video_path),
                ],
                capture_output=True, text=True, timeout=30, check=False,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            break
        rate = _parse_rate(out.stdout)
        if rate and rate > 0:
            return rate

    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        rate = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        if rate and rate > 0:
            return rate
    except Exception:
        pass
    return None


def fps_from_record(record: Any) -> Optional[float]:
    """Return the frame rate stored in a WHAM track / AMASS npz record, if any."""
    if not isinstance(record, Mapping) and not hasattr(record, "files") and not hasattr(record, "keys"):
        return None
    for key in FPS_RECORD_KEYS:
        try:
            has = key in record
        except TypeError:
            has = False
        if not has:
            continue
        value = record[key]
        try:
            import numpy as np

            flat = np.asarray(value).reshape(-1)
            if flat.size == 0:
                continue
            rate = float(flat[0])
        except Exception:
            try:
                rate = float(value)
            except (TypeError, ValueError):
                continue
        if rate and rate > 0:
            return rate
    return None


def resolve_fps(
    explicit: Optional[float] = None,
    *,
    record: Any = None,
    video_path: Optional[Union[str, Path]] = None,
    what: str = "input",
) -> float:
    """Resolve fps by priority: explicit override > record fps field > video probe.

    Raises ValueError with an actionable message if none of the sources yields a rate.
    """
    if explicit is not None and float(explicit) > 0:
        return float(explicit)

    rate = fps_from_record(record) if record is not None else None
    if rate:
        return rate

    if video_path is not None:
        rate = detect_video_fps(video_path)
        if rate:
            return rate

    raise ValueError(
        f"Could not determine fps for {what}: no explicit --fps, no fps field in the "
        f"record, and no readable video at {video_path!r}. Pass --fps explicitly or "
        f"backfill the source pkl (see backfill_pkl_fps.py)."
    )

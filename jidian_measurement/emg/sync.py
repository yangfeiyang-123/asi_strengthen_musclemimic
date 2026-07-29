from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from typing import Protocol


class Synchronizer(Protocol):
    sync_method: str

    def event(self, event_name: str, sample_index: int | None, notes: str = "") -> dict: ...


class SoftwareCueSynchronizer:
    sync_method = "software_cue_only"

    def __init__(self, sample_rate_hz: float) -> None:
        self.sample_rate_hz = sample_rate_hz

    def event(
        self,
        event_name: str,
        sample_index: int | None,
        notes: str = "",
        source: str = "software",
        confidence: float = 0.5,
    ) -> dict:
        return {
            "event_name": event_name,
            "sample_index": "" if sample_index is None else int(sample_index),
            "emg_time_s": "" if sample_index is None else float(sample_index / self.sample_rate_hz),
            "monotonic_time_ns": time.monotonic_ns(),
            "wall_clock_iso": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "confidence": float(confidence),
            "notes": notes,
        }

    def cue(self, sample_index: int | None = None, text: str = "开始动作") -> dict:
        print(f"\n*** {text} ***", flush=True)
        try:
            if sys.platform == "win32":
                import winsound

                winsound.Beep(1000, 180)
            else:
                print("\a", end="", flush=True)
        except (ImportError, RuntimeError):
            print("\a", end="", flush=True)
        return self.event("movement_cue", sample_index, notes=text, source="software_audio_visual", confidence=0.5)


class TTLTriggerSynchronizer:
    sync_method = "ttl_unavailable"

    def __init__(self, *_: object, **__: object) -> None:
        raise NotImplementedError("No verified TTL interface exists in this repository; hardware events are not fabricated")


class AudioVisualSynchronizer(SoftwareCueSynchronizer):
    """Explicit alias for the implemented audio/visual software cue path."""


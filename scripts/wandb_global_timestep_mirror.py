#!/usr/bin/env python3
"""Mirror a W&B run using Current Timestep as the real history step.

This is intended for runs whose remote ``_step`` was corrupted by delayed or
multi-writer uploads.  W&B history is append-only, so the safe repair is a
separate view with the source rows replayed in physical-timestep order.
"""

from __future__ import annotations

import argparse
import math
import re
import signal
import time
from pathlib import Path
from typing import Any

import wandb


VIDEO_DIR_RE = re.compile(
    r"validation_(?P<index>\d+)_traj(?P<trajectory>\d+)_t(?P<timestep>\d+)_"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, help="entity/project/run_id")
    parser.add_argument("--target-run-id", required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--run-dir", type=Path, default=Path("wandb"))
    parser.add_argument("--history-samples", type=int, default=4_000)
    parser.add_argument("--source-step-lookback", type=int, default=12_000)
    parser.add_argument("--safe-lag", type=int, default=20_480)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def _finite_scalar(value: Any) -> bool:
    if isinstance(value, bool | int | str):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def _clean_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if not key.startswith("_")
        and key not in {"Validation/Video", "validation_video_latest", "validation_videos_all"}
        and _finite_scalar(value)
    }


def _validation_keys(source) -> list[str]:
    history_keys = source._attrs.get("historyKeys", {}).get("keys", {})
    keys = [
        key
        for key in history_keys
        if key.startswith("Validation/") or key.startswith("Validation Measures/")
    ]
    keys = [key for key in keys if key != "Validation/Video"]
    if "Metric for Sweep" in history_keys:
        keys.append("Metric for Sweep")
    return ["Current Timestep", *sorted(set(keys))]


def _validation_rows(source) -> list[dict[str, Any]]:
    keys = _validation_keys(source)
    if len(keys) == 1:
        return []
    return source.history(keys=keys, samples=50_000, pandas=False)


def _initial_rows(source, samples: int) -> list[dict[str, Any]]:
    rows = source.history(samples=samples, pandas=False)
    rows.extend(_validation_rows(source))
    return rows


def _recent_rows(source, lookback: int) -> list[dict[str, Any]]:
    last_source_step = int(getattr(source, "lastHistoryStep", 0) or 0)
    min_step = max(0, last_source_step - lookback)
    try:
        return list(source.scan_history(min_step=min_step, page_size=2_000))
    except Exception as exc:
        print(f"recent scan failed ({type(exc).__name__}: {exc}); using sampled history", flush=True)
        return source.history(samples=10_000, pandas=False)


def _local_videos(video_root: Path | None) -> dict[int, tuple[int, int, Path]]:
    videos: dict[int, tuple[int, int, Path]] = {}
    if video_root is None or not video_root.is_dir():
        return videos
    for path in video_root.glob("validation_*/MyoFullBody.mp4"):
        match = VIDEO_DIR_RE.match(path.parent.name)
        if match is None:
            continue
        timestep = int(match.group("timestep"))
        videos[timestep] = (
            int(match.group("index")),
            int(match.group("trajectory")),
            path.resolve(),
        )
    return videos


def _merge_rows(
    rows: list[dict[str, Any]],
    *,
    after_timestep: int,
    cutoff_timestep: int,
    videos: dict[int, tuple[int, int, Path]],
) -> dict[int, dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for row in rows:
        raw_timestep = row.get("Current Timestep")
        if raw_timestep is None:
            continue
        timestep = int(raw_timestep)
        if timestep <= after_timestep or timestep > cutoff_timestep:
            continue
        merged.setdefault(timestep, {}).update(_clean_row(row))
        merged[timestep]["Current Timestep"] = timestep

    for timestep, (index, trajectory, _path) in videos.items():
        if timestep <= after_timestep or timestep > cutoff_timestep:
            continue
        payload = merged.setdefault(timestep, {"Current Timestep": timestep})
        payload["Validation/Video Index"] = index
        payload["Validation/Trajectory"] = trajectory
        payload["Validation/Video Timestep"] = timestep
    return merged


def _target_last_timestep(api: wandb.Api, target_path: str) -> int:
    try:
        target = api.run(target_path)
    except Exception:
        return -1
    return int(
        target.summary.get(
            "Mirror/Last Timestep",
            target.summary.get("Current Timestep", -1),
        )
        or -1
    )


def main() -> int:
    args = parse_args()
    entity, project, _source_id = args.source_run.split("/", 2)
    target_path = f"{entity}/{project}/{args.target_run_id}"
    args.run_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api(timeout=120)
    source = api.run(args.source_run)
    target_last = _target_last_timestep(api, target_path)
    source_config = dict(source.config)
    source_config["_timestep_mirror"] = {
        "source_run": args.source_run,
        "source_url": source.url,
        "step_metric": "Current Timestep",
    }

    run = wandb.init(
        entity=entity,
        project=project,
        id=args.target_run_id,
        name=args.target_name,
        config=source_config,
        tags=[*source.tags, "corrected-global-timestep", "history-mirror"],
        job_type="monitoring_mirror",
        dir=str(args.run_dir.resolve()),
        resume="allow",
        settings=wandb.Settings(console="off", x_update_finish_state=False),
    )
    run.define_metric("Current Timestep")
    run.define_metric("*", step_metric="Current Timestep", step_sync=True)
    run.summary["Mirror/Source Run"] = args.source_run
    run.summary["Mirror/Source URL"] = source.url
    run.summary["Mirror/Status"] = "running"

    stopping = False

    def _stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    first_poll = target_last < 0
    while not stopping:
        try:
            # Public API objects cache run attributes.  Recreate the client on
            # every poll so Current Timestep and lastHistoryStep reflect the
            # live source rather than the first observation.
            api = wandb.Api(timeout=120)
            source = api.run(args.source_run)
            current_timestep = int(source.summary.get("Current Timestep", 0) or 0)
            source_finished = source.state in {"finished", "failed", "crashed", "killed"}
            cutoff = current_timestep if source_finished else max(0, current_timestep - args.safe_lag)
            rows = (
                _initial_rows(source, args.history_samples)
                if first_poll
                else _recent_rows(source, args.source_step_lookback)
            )
            videos = _local_videos(args.video_root)
            merged = _merge_rows(
                rows,
                after_timestep=target_last,
                cutoff_timestep=cutoff,
                videos=videos,
            )

            for timestep in sorted(merged):
                payload = merged[timestep]
                video_meta = videos.get(timestep)
                if video_meta is not None:
                    index, trajectory, video_path = video_meta
                    payload["Validation/Video"] = wandb.Video(
                        str(video_path),
                        format="mp4",
                        caption=(
                            f"validation {index} | trajectory {trajectory} | "
                            f"timestep {timestep:,}"
                        ),
                    )
                run.log(payload, step=timestep)
                target_last = timestep

            run.summary["Mirror/Last Timestep"] = target_last
            run.summary["Mirror/Source Current Timestep"] = current_timestep
            run.summary["Mirror/Video Count"] = sum(t <= target_last for t in videos)
            run.summary["Mirror/Status"] = "source-finished" if source_finished else "running"
            print(
                f"mirrored rows={len(merged)} target_last={target_last:,} "
                f"source_current={current_timestep:,} videos={sum(t <= target_last for t in videos)}",
                flush=True,
            )
            first_poll = False
            if args.once or source_finished:
                break
        except Exception as exc:
            print(f"mirror poll failed: {type(exc).__name__}: {exc}", flush=True)
            if args.once:
                raise

        deadline = time.monotonic() + args.poll_seconds
        while not stopping and time.monotonic() < deadline:
            time.sleep(min(1.0, deadline - time.monotonic()))

    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

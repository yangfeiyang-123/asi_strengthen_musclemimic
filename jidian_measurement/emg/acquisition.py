from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .models import (
    ActionProtocol,
    ActionSpec,
    ChannelProfile,
    SessionMetadata,
    TrialMetadata,
    TrignoConfig,
    merge_session_metadata_overrides,
    utc_now_iso,
)
from .profiles import require_collection_profile
from .protocols import get_protocol
from .qc import assess_signal_quality, save_preview
from .storage import (
    atomic_write_json,
    git_commit_hash,
    initialize_session,
    read_json,
    save_trial_bundle,
    session_path,
    trial_path,
)
from .sync import SoftwareCueSynchronizer
from .synthetic import generate_synthetic_emg
from .trigno import TrignoClient


ERROR_LABELS = (
    "correct",
    "missed_shuttle",
    "wrong_footwork",
    "wrong_takeoff",
    "wrong_landing",
    "incomplete_motion",
    "sensor_motion_artifact",
    "sensor_detached",
    "signal_clipping",
    "sync_failed",
    "other",
)


def _refresh_completed_action_statistics(
    session_dir: Path,
    action: ActionSpec,
    profile: ChannelProfile,
    show: bool,
) -> None:
    """Refresh the across-trial action summary without risking saved trial data."""

    # Keep analytical/plotting dependencies out of the acquisition import path.
    # scikit-learn is used only by synergy extraction and is not an acquisition
    # dependency.
    from .action_statistics import InsufficientTrialsError, save_action_statistics

    try:
        outputs = save_action_statistics(
            session_dir / "trials" / action.action_id,
            profile,
            action,
            only_valid=True,
            show=show,
        )
    except InsufficientTrialsError as exc:
        print(f"Action statistics skipped: {exc}")
        return
    print(f"Action statistics saved: {outputs['figure']}")


def print_channel_table(profile: ChannelProfile) -> None:
    print(f"\nChannel profile: {profile.profile_id} (v{profile.version})")
    print("Sensor | Side  | Muscle slug                     | 中文名称             | Abbr")
    print("-------+-------+---------------------------------+----------------------+-----")
    for channel in profile.channels:
        print(
            f"{channel.sensor_id:>6} | {channel.side:<5} | {channel.muscle_slug:<31} | "
            f"{channel.name_zh:<20} | {channel.abbreviation}"
        )


def _session_metadata(
    participant_id: str,
    session_id: str,
    profile: ChannelProfile,
    protocol: ActionProtocol,
    handedness: str,
    dominant_leg: str,
    metadata_overrides: dict[str, Any] | None,
) -> SessionMetadata:
    canonical = SessionMetadata(
        participant_id=participant_id,
        session_id=session_id,
        channel_profile_id=profile.profile_id,
        protocol_id=protocol.protocol_id,
        handedness=handedness,
        dominant_leg=dominant_leg,
    ).to_dict()
    payload = merge_session_metadata_overrides(canonical, metadata_overrides)
    return SessionMetadata(**payload)


def _existing_trials(session_dir: Path, action_id: str) -> tuple[int, int]:
    action_dir = session_dir / "trials" / action_id
    max_index = 0
    valid = 0
    for metadata_path in sorted(action_dir.glob("trial_*/metadata.json")):
        try:
            metadata = read_json(metadata_path)
            max_index = max(max_index, int(metadata.get("trial_index", 0)))
            valid += int(bool(metadata.get("valid_for_analysis")))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return max_index, valid


def _review_trial(
    interactive: bool,
    default_error_label: str,
    default_notes: str,
) -> tuple[bool, list[str], str]:
    if default_error_label not in ERROR_LABELS:
        raise ValueError(f"Unknown error label {default_error_label!r}; choose from {ERROR_LABELS}")
    default_notes = default_notes.strip()
    if not interactive:
        valid = default_error_label == "correct"
        if not valid and not default_notes:
            raise ValueError(
                "A non-empty --operator-notes reason is required for every invalid/other trial"
            )
        return valid, [default_error_label], default_notes
    valid_raw = input("动作已完成。该 trial 是否有效？[Y/n]: ").strip().lower()
    valid = valid_raw not in {"n", "no", "0"}
    if valid:
        labels = ["correct"]
    else:
        print("错误标签：" + ", ".join(ERROR_LABELS[1:]))
        label_raw = input("输入一个或多个标签（逗号分隔）: ").strip()
        labels = [item.strip() for item in label_raw.split(",") if item.strip()] or ["other"]
        unknown = [label for label in labels if label not in ERROR_LABELS]
        if unknown:
            raise ValueError(f"Unknown error labels: {unknown}")
    notes = input("操作员备注（有效 trial 可空）: ").strip() or default_notes
    while not valid and not notes:
        notes = input("无效 trial 必须填写原因，请输入备注: ").strip()
    return valid, labels, notes


def _append_optional_event_slots(
    events: list[dict[str, Any]], action: ActionSpec, sync: SoftwareCueSynchronizer
) -> None:
    """Declare annotatable slots without inventing sample times or hardware evidence."""
    if action.requires_racket_contact:
        events.append(sync.event("racket_contact", None, "Awaiting manual/video annotation", "unannotated", 0.0))
    if action.requires_foot_contact:
        events.append(sync.event("foot_contact", None, "Awaiting manual/video annotation", "unannotated", 0.0))
    if action.includes_jump:
        events.extend(
            [
                sync.event("takeoff", None, "Awaiting manual/video annotation", "unannotated", 0.0),
                sync.event("landing", None, "Awaiting manual/video annotation", "unannotated", 0.0),
            ]
        )


def run_sensor_check(
    profile: ChannelProfile,
    dry_run: bool,
    trigno_config: TrignoConfig,
    duration_s: float = 1.5,
    seed: int = 20260720,
) -> dict[str, Any]:
    require_collection_profile(profile.profile_id)
    print("\nRunning sensor signal check...")
    if dry_run:
        result = generate_synthetic_emg(profile.channel_ids, duration_s, trigno_config.sample_rate_hz, seed=seed)
    else:
        with TrignoClient(trigno_config) as client:
            result = client.record(duration_s, profile.channel_ids)
    qc = assess_signal_quality(
        result.emg_mV,
        result.fs_hz,
        result.expected_samples,
        baseline_samples=result.received_samples,
        action_start_sample=0,
        action_end_sample=result.received_samples,
    )
    fatal = [
        (channel.sensor_id, metrics["warnings"])
        for channel, metrics in zip(profile.channels, qc["channels"])
        if any(label in metrics["warnings"] for label in ("all_zero", "flatline", "nan_or_inf"))
    ]
    if fatal:
        print(f"Sensor check needs attention: {fatal}")
    else:
        print(f"Sensor check: all {len(profile.channels)} channels are finite and non-flat.")
    return {"fatal_channel_warnings": fatal, "qc": qc, "record_result": result}


def collect_action(
    dataset_root: Path,
    participant_id: str,
    session_id: str,
    profile_id: str,
    protocol_id: str,
    action_id: str,
    handedness: str = "right",
    dominant_leg: str = "unknown",
    metadata_overrides: dict[str, Any] | None = None,
    dry_run: bool = False,
    interactive: bool = True,
    target_valid_trials: int | None = None,
    action_duration_s: float | None = None,
    rest_seconds: float | None = None,
    block_rest_seconds: float | None = None,
    show_preview: bool = False,
    save_legacy_csv: bool = True,
    error_label: str = "correct",
    operator_notes: str = "",
    seed: int = 20260720,
    synthetic_powerline_50hz: float = 0.0,
    synthetic_clipping_mV: float | None = None,
    synthetic_dropped_samples: int = 0,
    trigno_config: TrignoConfig | None = None,
) -> list[Path]:
    profile = require_collection_profile(profile_id)
    protocol = get_protocol(protocol_id)
    profile.validate_handedness(handedness)
    action = protocol.action(action_id)
    if action_duration_s is not None:
        action = replace(action, duration_s=action_duration_s)
    if rest_seconds is not None:
        action = replace(action, rest_between_trials_s=rest_seconds)
    if block_rest_seconds is not None:
        action = replace(action, rest_between_blocks_s=block_rest_seconds)
    trigno = trigno_config or TrignoConfig()
    print_channel_table(profile)
    print(f"Action: {action.display_name_zh} ({action.action_id})")
    print(f"Instruction: {action.instruction_zh}")
    session_metadata = _session_metadata(
        participant_id, session_id, profile, protocol, handedness, dominant_leg, metadata_overrides
    )
    session_dir = initialize_session(
        dataset_root,
        participant_id,
        session_id,
        session_metadata.to_dict(),
        profile,
        protocol.to_dict(),
    )
    sensor_check = run_sensor_check(profile, dry_run, trigno, seed=seed)
    if sensor_check["fatal_channel_warnings"] and not dry_run:
        raise RuntimeError("Sensor check found all-zero/flat/non-finite channels; correct placement before collection")
    last_index, valid_count = _existing_trials(session_dir, action.action_id)
    target = target_valid_trials or action.target_valid_trials
    if target < 1:
        raise ValueError("target_valid_trials must be >= 1")
    if valid_count >= target:
        print(f"Session already has {valid_count}/{target} valid trials for {action.action_id}; nothing to collect.")
        _refresh_completed_action_statistics(session_dir, action, profile, show_preview)
        return []
    saved: list[Path] = []
    attempt_limit = max(target * 3, target + 2)
    attempts = 0
    current_index = last_index
    trials_per_block = max(1, int(np.ceil(target / action.blocks)))
    rested_block_boundaries: set[int] = set()
    while valid_count < target and attempts < attempt_limit:
        attempts += 1
        current_index += 1
        print(f"\nTrial {current_index} ({valid_count}/{target} valid complete)")
        if interactive:
            input("检查传感器、场地和球后，按 Enter 开始记录...")
        total_duration = action.total_recording_s
        sync = SoftwareCueSynchronizer(trigno.sample_rate_hz)
        events: list[dict[str, Any]] = [
            sync.event("recording_start", 0, "Software timestamp immediately before acquisition", confidence=0.5),
            sync.event("baseline_start", 0, "Quiet baseline begins", confidence=0.5),
        ]
        if dry_run:
            result = generate_synthetic_emg(
                profile.channel_ids,
                total_duration,
                trigno.sample_rate_hz,
                baseline_s=action.baseline_s,
                cue_s=action.cue_s,
                post_s=action.post_s,
                seed=seed + current_index,
                powerline_50hz=synthetic_powerline_50hz,
                clipping_mV=synthetic_clipping_mV,
                dropped_samples=synthetic_dropped_samples,
            )
            cue_sample = min(
                int(round((action.baseline_s + action.cue_s) * result.fs_hz)), result.received_samples
            )
            events.append(
                sync.event(
                    "movement_cue",
                    cue_sample,
                    f"Synthetic dry-run cue: {action.display_name_zh}",
                    "synthetic",
                    1.0,
                )
            )
        else:
            cue_sent = False

            def progress(received_samples: int) -> None:
                nonlocal cue_sent
                cue_at = int(round((action.baseline_s + action.cue_s) * trigno.sample_rate_hz))
                if not cue_sent and received_samples >= cue_at:
                    events.append(sync.cue(received_samples, f"开始：{action.display_name_zh}"))
                    cue_sent = True

            with TrignoClient(trigno) as client:
                result = client.record(total_duration, profile.channel_ids, progress_callback=progress)
            if not cue_sent:
                events.append(
                    sync.event(
                        "movement_cue",
                        None,
                        "Cue threshold was not reached before stream ended",
                        "software",
                        0.0,
                    )
                )
        events.append(
            sync.event(
                "movement_start_manual",
                None,
                "Reserved for operator/video annotation; not inferred from the cue",
                "unannotated",
                0.0,
            )
        )
        events.append(
            sync.event(
                "recording_stop",
                result.received_samples,
                "Software timestamp immediately after acquisition",
                confidence=0.5,
            )
        )
        _append_optional_event_slots(events, action, sync)
        action_start = int(round((action.baseline_s + action.cue_s) * result.fs_hz))
        action_end = int(round((action.baseline_s + action.cue_s + action.duration_s) * result.fs_hz))
        qc = assess_signal_quality(
            result.emg_mV,
            result.fs_hz,
            result.expected_samples,
            int(round(action.baseline_s * result.fs_hz)),
            action_start,
            action_end,
        )
        trial_id = f"{action.action_id}_trial_{current_index:03d}"
        metadata = TrialMetadata(
            participant_id=participant_id,
            session_id=session_id,
            trial_id=trial_id,
            trial_index=current_index,
            action_id=action.action_id,
            protocol_id=protocol.protocol_id,
            protocol_version=protocol.version,
            channel_profile_id=profile.profile_id,
            channel_profile_version=profile.version,
            channel_profile_snapshot=profile.to_dict(),
            handedness=handedness,
            dominant_leg=dominant_leg,
            sample_rate_hz=result.fs_hz,
            requested_duration_s=total_duration,
            actual_duration_s=result.actual_duration_s,
            expected_samples=result.expected_samples,
            received_samples=result.received_samples,
            dropped_samples=result.dropped_samples,
            start_time=result.start_time,
            valid_for_analysis=False,
            error_labels=["pending_review"],
            operator_notes="Trial saved before post-trial review",
            racket={
                "included": action.includes_racket,
                "mass_g": session_metadata.racket_mass_g,
                "grip_size": session_metadata.grip_size,
                "string_tension_lb": session_metadata.string_tension_lb,
            },
            includes_shuttle=action.includes_shuttle,
            sync_method="software_cue_only",
            software_version=__version__,
            git_commit_hash=git_commit_hash(Path.cwd()),
            qc_pass=bool(qc["qc_pass"]),
            interrupted=result.interrupted,
            receive_error=result.receive_error,
        )
        output_dir = trial_path(dataset_root, participant_id, session_id, action.action_id, current_index)

        def preview_writer(path: Path) -> None:
            qc_label = "QC PASS" if qc["qc_pass"] else f"QC WARNING ({len(qc['warnings'])})"
            save_preview(
                path,
                result.emg_mV,
                result.fs_hz,
                profile,
                f"{trial_id} | {qc_label}",
                show=show_preview,
            )

        save_trial_bundle(
            output_dir,
            result,
            metadata,
            events,
            qc,
            profile,
            preview_writer,
            legacy_csv=save_legacy_csv,
        )
        valid, labels, notes = _review_trial(interactive, error_label, operator_notes)
        if result.incomplete or result.interrupted or result.receive_error is not None:
            valid = False
            if "sync_failed" not in labels:
                labels = [label for label in labels if label != "correct"] + ["sync_failed"]
            automatic_reason = (
                f"Automatic invalidation: received {result.received_samples}/"
                f"{result.expected_samples} samples; interrupted={result.interrupted}; "
                f"receive_error={result.receive_error or 'none'}"
            )
            notes = f"{notes}; {automatic_reason}" if notes else automatic_reason
        metadata.valid_for_analysis = valid
        metadata.error_labels = labels
        metadata.operator_notes = notes
        atomic_write_json(output_dir / "metadata.json", metadata.to_dict())
        saved.append(output_dir)
        valid_count += int(valid)
        print(f"Saved {output_dir} | valid={valid} | qc_pass={qc['qc_pass']} | labels={labels}")
        if valid_count >= target:
            metadata.post_trial_rest = _rest_not_required(
                reason="target_reached",
                rest_kind="none",
            )
            atomic_write_json(output_dir / "metadata.json", metadata.to_dict())
            break
        next_valid_number = valid_count + 1
        at_block_boundary = (
            valid
            and valid_count > 0
            and valid_count % trials_per_block == 0
            and valid_count not in rested_block_boundaries
        )
        if at_block_boundary:
            rested_block_boundaries.add(valid_count)
        rest_kind = "block" if at_block_boundary else "trial"
        planned_rest_s = (
            action.rest_between_blocks_s if at_block_boundary else action.rest_between_trials_s
        )
        if dry_run:
            metadata.post_trial_rest = _rest_not_required(
                reason="dry_run",
                rest_kind=rest_kind,
                planned_seconds=planned_rest_s,
            )
        elif planned_rest_s <= 0:
            metadata.post_trial_rest = _rest_not_required(
                reason="protocol_zero_rest",
                rest_kind=rest_kind,
            )
        else:
            metadata.post_trial_rest = _timed_rest_record(
                planned_rest_s,
                f"下一有效 trial {next_valid_number}",
                rest_kind=rest_kind,
            )
        atomic_write_json(output_dir / "metadata.json", metadata.to_dict())
    if valid_count < target:
        print(f"Stopped at attempt limit with {valid_count}/{target} valid trials; session can be resumed.")
    else:
        _refresh_completed_action_statistics(session_dir, action, profile, show_preview)
    return saved


def _enter_pressed() -> bool:
    """Consume a pending Enter key without blocking the acquisition process."""
    if not sys.stdin or not sys.stdin.isatty():
        return False
    if os.name == "nt":
        import msvcrt

        while msvcrt.kbhit():
            key = msvcrt.getwch()
            if key in {"\r", "\n"}:
                return True
            if key in {"\x00", "\xe0"} and msvcrt.kbhit():
                msvcrt.getwch()
        return False
    try:
        import select

        readable, _, _ = select.select([sys.stdin], [], [], 0)
        if readable:
            return sys.stdin.readline().endswith("\n")
    except (OSError, ValueError):
        return False
    return False


def _rest_countdown(seconds: float, label: str) -> bool:
    """Run a responsive rest timer; return True when the operator skips it."""
    deadline = time.monotonic() + max(float(seconds), 0.0)
    last_displayed: int | None = None
    while True:
        remaining = max(0, int(math.ceil(deadline - time.monotonic())))
        if remaining != last_displayed:
            print(
                f"\rRest: {remaining:3d}s | {label} | 按 Enter 跳过休息",
                end="",
                flush=True,
            )
            last_displayed = remaining
        if _enter_pressed():
            print("\rRest skipped by operator.                              ")
            return True
        if remaining <= 0:
            print("\rRest complete.                                         ")
            return False
        time.sleep(0.05)


def _rest_not_required(
    *,
    reason: str,
    rest_kind: str,
    planned_seconds: float = 0.0,
) -> dict[str, Any]:
    """Build an explicit non-rest record; an absent record means unknown."""

    planned = float(planned_seconds)
    if not math.isfinite(planned) or planned < 0:
        raise ValueError("Rest seconds must be finite and non-negative")
    return {
        "schema_version": "emg_rest_record_v1",
        "rest_kind": str(rest_kind),
        "status": "not_performed",
        "reason": str(reason),
        "planned_seconds": planned,
        "actual_seconds": 0.0,
        "skipped": False,
        "started_at": None,
        "completed_at": None,
    }


def _timed_rest_record(seconds: float, label: str, *, rest_kind: str) -> dict[str, Any]:
    """Run a rest countdown and retain planned, elapsed, and skip provenance."""

    planned = float(seconds)
    if not math.isfinite(planned) or planned <= 0:
        raise ValueError("Timed rest seconds must be finite and positive")
    started_at = utc_now_iso()
    started_monotonic = time.monotonic()
    skipped = _rest_countdown(planned, label)
    actual = max(0.0, time.monotonic() - started_monotonic)
    return {
        "schema_version": "emg_rest_record_v1",
        "rest_kind": str(rest_kind),
        "status": "skipped" if skipped else "completed",
        "reason": "operator_skip" if skipped else "countdown_complete",
        "planned_seconds": planned,
        "actual_seconds": actual,
        "skipped": bool(skipped),
        "started_at": started_at,
        "completed_at": utc_now_iso(),
    }


def load_metadata_overrides(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return read_json(path)

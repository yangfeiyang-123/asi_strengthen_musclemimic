from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .acquisition import ERROR_LABELS, collect_action, load_metadata_overrides, print_channel_table, run_sensor_check
from .models import CHANNEL_NORMALIZATIONS, SessionMetadata, TrignoConfig, merge_session_metadata_overrides
from .mvc import collect_mvc
from .profiles import PROFILE_REGISTRY, PROFILES, get_profile, require_collection_profile
from .protocols import PROTOCOLS, get_protocol
from .storage import read_json, session_path


DEFAULT_PROFILE = "badminton_synergy_16_v2"
DEFAULT_PROTOCOL = "badminton_primitive_protocol_v1"


def _trigno_config(args: argparse.Namespace) -> TrignoConfig:
    return TrignoConfig(
        host=args.host,
        command_port=args.command_port,
        emg_port=args.emg_port,
        sample_rate_hz=args.fs,
        stream_scale_to_mV=args.stream_scale_to_mv,
    )


def _add_hardware_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="127.0.0.1", help="Trigno Control Utility host")
    parser.add_argument("--command-port", type=int, default=50040)
    parser.add_argument("--emg-port", type=int, default=50041)
    parser.add_argument("--fs", type=float, default=2000.0, help="EMG sampling rate in Hz")
    parser.add_argument("--stream-scale-to-mV", dest="stream_scale_to_mv", type=float, default=1000.0)


def _add_session_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--participant", required=True, help="Pseudonymous participant ID")
    parser.add_argument("--session", required=True, help="Session ID")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, choices=sorted(PROFILES))
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL, choices=sorted(PROTOCOLS))
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset_root"))
    parser.add_argument("--handedness", choices=("right", "left"), default="right")
    parser.add_argument("--dominant-leg", default="unknown")
    parser.add_argument("--session-metadata", type=Path, help="UTF-8 JSON with anthropometrics/equipment/operator fields")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m emg.cli",
        description="Delsys Trigno badminton EMG acquisition, QC, preprocessing, and synergy extraction",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-profiles", help="List immutable channel profile versions")

    audit_profile = subparsers.add_parser(
        "audit-session-profile",
        help="Read-only audit of raw channel order, metadata semantics, and migration safety",
    )
    audit_profile.add_argument("--session-path", type=Path, required=True)
    audit_profile.add_argument("--actual-profile", required=True, choices=sorted(PROFILES))
    audit_profile.add_argument("--json", action="store_true", help="Print the complete machine-readable audit")

    migrate_profile = subparsers.add_parser(
        "migrate-session-profile",
        help="Back up and correct historical profile semantics without writing raw EMG",
    )
    migrate_profile.add_argument("--session-path", type=Path, required=True)
    migrate_profile.add_argument("--actual-profile", required=True, choices=sorted(PROFILES))
    migrate_profile.add_argument("--apply", action="store_true", help="Apply the audited migration; otherwise dry-run only")
    migrate_profile.add_argument("--skip-preview-regeneration", action="store_true")
    migrate_profile.add_argument("--skip-action-statistics-regeneration", action="store_true")

    annotate = subparsers.add_parser(
        "annotate-event",
        help="Atomically fill one evidence-backed event slot and append a hash audit trail",
    )
    annotate.add_argument("--trial-path", type=Path, required=True)
    annotate.add_argument("--event-name", required=True)
    annotate.add_argument("--sample-index", type=int)
    annotate.add_argument("--emg-time-s", type=float)
    annotate.add_argument("--source", required=True)
    annotate.add_argument("--confidence", type=float, required=True)
    annotate.add_argument("--annotator", required=True, help="Pseudonymous operator ID")
    annotate.add_argument("--evidence-reference", required=True, help="Video/frame, hardware, or field-record reference")
    annotate.add_argument(
        "--evidence-sha256",
        required=True,
        help="SHA-256 of the referenced evidence object (64 hexadecimal characters)",
    )
    annotate.add_argument("--notes", default="")
    annotate.add_argument("--overwrite", action="store_true")
    annotate.add_argument(
        "--expected-before-sha256",
        help="Optional compare-and-swap guard copied from a prior review",
    )

    actions = subparsers.add_parser("list-actions", help="List configured primitive and complete actions")
    actions.add_argument("--protocol", default=DEFAULT_PROTOCOL, choices=sorted(PROTOCOLS))

    sensor = subparsers.add_parser("sensor-check", help="Record a short signal-quality check")
    sensor.add_argument("--profile", default=DEFAULT_PROFILE, choices=sorted(PROFILES))
    sensor.add_argument("--duration", type=float, default=1.5)
    sensor.add_argument("--dry-run", action="store_true")
    sensor.add_argument("--seed", type=int, default=20260720)
    _add_hardware_args(sensor)

    collect = subparsers.add_parser("collect", help="Collect/resume one action until target valid trials are saved")
    _add_session_args(collect)
    _add_hardware_args(collect)
    collect.add_argument("--action", required=True)
    collect.add_argument("--trials", type=int, help="Override target number of valid trials")
    collect.add_argument("--duration", type=float, help="Override movement duration only")
    collect.add_argument("--rest", type=float, help="Override inter-trial rest")
    collect.add_argument("--block-rest", type=float, help="Override inter-block rest")
    collect.add_argument("--dry-run", action="store_true", help="Generate deterministic synthetic EMG and do not connect hardware")
    collect.add_argument("--non-interactive", action="store_true", help="Use --error-label without prompts")
    collect.add_argument("--interactive-review", action="store_true", help="Keep post-trial review prompts in dry-run")
    collect.add_argument("--error-label", default="correct", choices=ERROR_LABELS)
    collect.add_argument("--operator-notes", default="")
    collect.add_argument("--seed", type=int, default=20260720)
    collect.add_argument("--synthetic-50hz-mV", type=float, default=0.0)
    collect.add_argument("--synthetic-clipping-mV", type=float)
    collect.add_argument("--synthetic-dropped-samples", type=int, default=0)
    collect.add_argument("--no-legacy-csv", action="store_true")
    preview_group = collect.add_mutually_exclusive_group()
    preview_group.add_argument("--show-preview", dest="show_preview", action="store_true")
    preview_group.add_argument("--no-show-preview", dest="show_preview", action="store_false")
    collect.set_defaults(show_preview=None)

    protocol_collect = subparsers.add_parser("collect-protocol", help="Collect multiple configured actions in one resumable session")
    _add_session_args(protocol_collect)
    _add_hardware_args(protocol_collect)
    protocol_collect.add_argument("--actions", nargs="+", help="Action IDs; default is all protocol actions")
    protocol_collect.add_argument("--trials-per-action", type=int)
    protocol_collect.add_argument("--duration", type=float)
    protocol_collect.add_argument("--rest", type=float)
    protocol_collect.add_argument("--dry-run", action="store_true")
    protocol_collect.add_argument("--non-interactive", action="store_true")
    protocol_collect.add_argument("--seed", type=int, default=20260720)
    protocol_collect.add_argument("--no-legacy-csv", action="store_true")
    protocol_preview = protocol_collect.add_mutually_exclusive_group()
    protocol_preview.add_argument("--show-preview", dest="show_preview", action="store_true")
    protocol_preview.add_argument("--no-show-preview", dest="show_preview", action="store_false")
    protocol_collect.set_defaults(show_preview=None)

    mvc = subparsers.add_parser("mvc", help="Collect repeated muscle-specific MVC recordings")
    _add_session_args(mvc)
    _add_hardware_args(mvc)
    mvc.add_argument("--repetitions", type=int, default=3)
    mvc.add_argument("--contraction-duration", type=float, default=4.0)
    mvc.add_argument("--rest", type=float, default=60.0)
    mvc.add_argument("--window-ms", type=float, default=500.0)
    mvc.add_argument("--muscle", action="append", help="Recollect only this muscle_slug; repeat option as needed")
    mvc.add_argument("--dry-run", action="store_true")
    mvc.add_argument("--non-interactive", action="store_true")
    mvc.add_argument("--seed", type=int, default=20260720)
    mvc.add_argument(
        "--allow-retrospective-profile-mvc",
        action="store_true",
        help="Explicitly allow MVC collection for a registry profile marked retrospective_mvc_allowed=explicit_only",
    )

    preprocess = subparsers.add_parser("preprocess", help="Offline filter/envelope/normalize all trials in a session")
    preprocess.add_argument("--session-path", type=Path, required=True)
    preprocess.add_argument("--profile", default=DEFAULT_PROFILE, choices=sorted(PROFILES))
    preprocess.add_argument("--config", type=Path, help="JSON preprocessing configuration")
    preprocess.add_argument("--normalization", choices=("mvc", "dynamic_p95", "none"))
    preprocess.add_argument("--fallback-normalization", choices=("dynamic_p95", "none"))
    preprocess_notch = preprocess.add_mutually_exclusive_group()
    preprocess_notch.add_argument("--notch-50hz", dest="notch_50hz", action="store_true", help="Enable the configured 50 Hz notch")
    preprocess_notch.add_argument("--no-notch-50hz", dest="notch_50hz", action="store_false", help="Disable the notch for this run")
    preprocess.set_defaults(notch_50hz=None)
    preprocess.add_argument("--mvc-scope", choices=("session", "participant"), default="participant")
    preprocess.add_argument("--no-figures", action="store_true")
    preprocess.add_argument("--continue-on-error", action="store_true")

    preprocess_all = subparsers.add_parser("preprocess-dataset", help="Batch preprocess matching participants, sessions, actions, and trials")
    preprocess_all.add_argument("--dataset-root", type=Path, required=True)
    preprocess_all.add_argument("--profile", default=DEFAULT_PROFILE, choices=sorted(PROFILES))
    preprocess_all.add_argument("--config", type=Path, default=Path("config/semg_preprocessing.json"))
    preprocess_all.add_argument("--normalization", choices=("mvc", "dynamic_p95", "none"))
    preprocess_all.add_argument("--fallback-normalization", choices=("dynamic_p95", "none"))
    preprocess_all.add_argument("--participant", action="append", help="Participant ID; repeat to select multiple")
    preprocess_all.add_argument("--session", action="append", help="Session ID; repeat to select multiple")
    preprocess_all.add_argument("--mvc-scope", choices=("session", "participant"), default="participant")
    preprocess_all.add_argument("--no-figures", action="store_true")
    preprocess_all.add_argument("--fail-fast", action="store_true")

    qc = subparsers.add_parser("qc", help="Regenerate per-trial QC JSON and nonblocking previews")
    qc.add_argument("--session-path", type=Path, required=True)
    qc.add_argument("--profile", default=DEFAULT_PROFILE, choices=sorted(PROFILES))
    qc.add_argument("--show-preview", action="store_true")

    action_stats = subparsers.add_parser(
        "action-stats",
        help="Compute repeated-trial mean/sample variance and save a 4x4 sensor figure",
    )
    action_stats.add_argument("--session-path", type=Path, required=True)
    action_stats.add_argument("--profile", default=DEFAULT_PROFILE, choices=sorted(PROFILES))
    action_stats.add_argument("--action", action="append", help="Action ID; repeat to select multiple")
    action_stats.add_argument("--config", type=Path, help="JSON preprocessing configuration")
    action_stats.add_argument("--include-invalid", action="store_true")
    action_stats.add_argument("--show", action="store_true")

    dataset = subparsers.add_parser("build-synergy-dataset", help="Build a validated nonnegative V=[channels,time] matrix")
    dataset.add_argument("--dataset-root", type=Path, required=True)
    dataset.add_argument("--output", type=Path, required=True)
    dataset.add_argument("--profile", default=DEFAULT_PROFILE, choices=sorted(PROFILES))
    dataset.add_argument("--protocol", default=DEFAULT_PROTOCOL, choices=sorted(PROTOCOLS))
    validity = dataset.add_mutually_exclusive_group()
    validity.add_argument("--only-valid", dest="only_valid", action="store_true", default=True)
    validity.add_argument("--include-invalid", dest="only_valid", action="store_false")
    dataset.add_argument("--scope", choices=("all", "primitive", "complete"), default="all")
    dataset.add_argument("--action", action="append")
    dataset.add_argument("--participant", action="append")
    dataset.add_argument(
        "--crop-mode",
        choices=("annotated_movement_events", "software_cue_exploratory", "full_trial"),
        default="annotated_movement_events",
    )
    dataset.add_argument("--time-normalize-samples", type=int)

    synergy = subparsers.add_parser("extract-synergy", help="Fit NMF, select K by VAF, assess stability, and export artifact")
    synergy.add_argument("--dataset", type=Path, required=True)
    synergy.add_argument("--output", type=Path, required=True)
    synergy.add_argument("--k-min", type=int, default=1)
    synergy.add_argument("--k-max", type=int, default=8)
    synergy.add_argument("--n-init", type=int, default=50)
    synergy.add_argument("--seed", type=int, default=20260720)
    synergy.add_argument("--global-vaf", type=float, default=0.90)
    synergy.add_argument("--local-vaf", type=float, default=0.75)
    synergy.add_argument("--local-fraction", type=float, default=0.8)
    synergy.add_argument("--omit-h", action="store_true")
    synergy.add_argument("--skip-stability", action="store_true")
    synergy.add_argument(
        "--channel-normalization",
        choices=CHANNEL_NORMALIZATIONS,
        default="unit_variance",
        help=(
            "Per-channel divisor applied before NMF so loud electrodes do not dominate the "
            "least-squares fit. 'none' reproduces an unbalanced fit and generally returns "
            "single-muscle components rather than synergies."
        ),
    )
    synergy.add_argument("--split-half-repeats", type=int, default=20)
    return parser


def _new_session_payload(args: argparse.Namespace, overrides: dict[str, Any] | None) -> dict[str, Any]:
    profile = get_profile(args.profile)
    protocol = get_protocol(args.protocol)
    payload = SessionMetadata(
        participant_id=args.participant,
        session_id=args.session,
        channel_profile_id=profile.profile_id,
        protocol_id=protocol.protocol_id,
        handedness=args.handedness,
        dominant_leg=args.dominant_leg,
    ).to_dict()
    return merge_session_metadata_overrides(payload, overrides)


def main(argv: list[str] | None = None) -> int:
    # Keep Chinese protocol/muscle names printable in redirected Windows shells.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except (OSError, ValueError):
                pass
    args = build_parser().parse_args(argv)
    if args.command == "list-profiles":
        for registration in PROFILE_REGISTRY.values():
            profile = registration.profile
            print(
                f"{profile.profile_id} v{profile.version} | status={registration.status} | "
                f"collection_allowed={registration.collection_allowed} | "
                f"analysis_allowed={registration.analysis_allowed} | "
                f"retrospective_mvc_allowed={registration.retrospective_mvc_allowed} | "
                f"channels={len(profile.channels)} | handedness={profile.intended_handedness}"
            )
            if registration.reason:
                print(f"  reason: {registration.reason}")
        return 0
    if args.command == "audit-session-profile":
        from .profile_migration import audit_session_profile, format_audit_report

        audit = audit_session_profile(args.session_path, args.actual_profile)
        print(json.dumps(audit, ensure_ascii=False, indent=2) if args.json else format_audit_report(audit))
        return 0 if audit["migration_safe"] else 2
    if args.command == "migrate-session-profile":
        from .profile_migration import format_audit_report, migrate_session_profile

        result = migrate_session_profile(
            args.session_path,
            args.actual_profile,
            apply=args.apply,
            regenerate_previews=not args.skip_preview_regeneration,
            regenerate_action_statistics=not args.skip_action_statistics_regeneration,
        )
        print(f"Migration status: {result['status']}")
        audit = result.get("audit_after") or result.get("audit")
        if audit:
            print(format_audit_report(audit))
        manifest = result.get("manifest") or {}
        if manifest:
            print(f"Backup: {manifest.get('backup_path')}")
            print(f"Raw NPZ unchanged: {manifest.get('raw_npz_files_unchanged')}")
            print(f"CSV numeric bodies unchanged: {manifest.get('csv_numeric_bodies_unchanged')}")
        elif not args.apply:
            print("Dry-run only; pass --apply after reviewing the audit.")
        return 0
    if args.command == "annotate-event":
        from .event_annotation import annotate_event

        result = annotate_event(
            args.trial_path,
            args.event_name,
            sample_index=args.sample_index,
            emg_time_s=args.emg_time_s,
            source=args.source,
            confidence=args.confidence,
            annotator=args.annotator,
            evidence_reference=args.evidence_reference,
            evidence_sha256=args.evidence_sha256,
            notes=args.notes,
            overwrite=args.overwrite,
            expected_before_sha256=args.expected_before_sha256,
        )
        print(
            f"Annotated {result['event_name']} at sample={result['sample_index']} "
            f"time={result['emg_time_s']:.9f}s"
        )
        print(f"events.csv SHA-256: {result['before_sha256']} -> {result['after_sha256']}")
        print(f"Annotation manifest SHA-256: {result['annotation_manifest_sha256']}")
        print(f"Audit: {result['audit_path']} | annotation_id={result['annotation_id']}")
        return 0
    if args.command == "list-actions":
        protocol = get_protocol(args.protocol)
        for action in protocol.actions:
            print(
                f"{action.action_id:<38} {action.category:<9} {action.display_name_zh} | "
                f"target={action.target_valid_trials}, blocks={action.blocks}, rest={action.rest_between_trials_s:.0f}s"
            )
        return 0
    if args.command == "sensor-check":
        profile = require_collection_profile(args.profile)
        print_channel_table(profile)
        result = run_sensor_check(profile, args.dry_run, _trigno_config(args), args.duration, args.seed)
        return 1 if result["fatal_channel_warnings"] else 0
    if args.command in {"collect", "collect-protocol"}:
        overrides = load_metadata_overrides(args.session_metadata)
        actions = [args.action] if args.command == "collect" else (
            args.actions or [item.action_id for item in get_protocol(args.protocol).actions]
        )
        all_saved: list[Path] = []
        if args.command == "collect":
            interactive_collection = args.interactive_review if args.dry_run else not args.non_interactive
        else:
            interactive_collection = False if args.dry_run else not args.non_interactive
        show_preview = interactive_collection if args.show_preview is None else args.show_preview
        for offset, action_id in enumerate(actions):
            saved = collect_action(
                dataset_root=args.dataset_root,
                participant_id=args.participant,
                session_id=args.session,
                profile_id=args.profile,
                protocol_id=args.protocol,
                action_id=action_id,
                handedness=args.handedness,
                dominant_leg=args.dominant_leg,
                metadata_overrides=overrides,
                dry_run=args.dry_run,
                interactive=interactive_collection,
                target_valid_trials=args.trials if args.command == "collect" else args.trials_per_action,
                action_duration_s=args.duration,
                rest_seconds=args.rest,
                block_rest_seconds=args.block_rest if args.command == "collect" else None,
                show_preview=show_preview,
                save_legacy_csv=not args.no_legacy_csv,
                error_label=args.error_label if args.command == "collect" else "correct",
                operator_notes=args.operator_notes if args.command == "collect" else "",
                seed=args.seed + offset * 1000,
                synthetic_powerline_50hz=args.synthetic_50hz_mV if args.command == "collect" else 0.0,
                synthetic_clipping_mV=args.synthetic_clipping_mV if args.command == "collect" else None,
                synthetic_dropped_samples=args.synthetic_dropped_samples if args.command == "collect" else 0,
                trigno_config=_trigno_config(args),
            )
            all_saved.extend(saved)
        print(f"Saved {len(all_saved)} trial bundle(s).")
        return 0
    if args.command == "mvc":
        overrides = load_metadata_overrides(args.session_metadata)
        candidate = session_path(args.dataset_root, args.participant, args.session) / "session.json"
        payload = read_json(candidate) if candidate.exists() else _new_session_payload(args, overrides)
        collect_mvc(
            args.dataset_root,
            args.participant,
            args.session,
            args.profile,
            args.protocol,
            payload,
            repetitions=args.repetitions,
            contraction_s=args.contraction_duration,
            rest_s=args.rest,
            window_s=args.window_ms / 1000.0,
            muscle_slugs=args.muscle,
            dry_run=args.dry_run,
            interactive=False if args.dry_run else not args.non_interactive,
            seed=args.seed,
            trigno_config=_trigno_config(args),
            allow_retrospective_profile=args.allow_retrospective_profile_mvc,
        )
        return 0
    if args.command == "preprocess":
        from .offline import load_processing_config, preprocess_session

        config = load_processing_config(args.config, normalization=args.normalization)
        outputs = preprocess_session(
            args.session_path,
            args.profile,
            fallback=args.fallback_normalization,
            notch_50hz=args.notch_50hz,
            config=config,
            mvc_scope=args.mvc_scope,
            save_figures=not args.no_figures,
            continue_on_error=args.continue_on_error,
        )
        print(f"Preprocessed {len(outputs)} trial(s).")
        return 0
    if args.command == "preprocess-dataset":
        from .offline import load_processing_config, preprocess_dataset

        config = load_processing_config(args.config, normalization=args.normalization)
        summary = preprocess_dataset(
            args.dataset_root,
            args.profile,
            config,
            fallback=args.fallback_normalization,
            participant_ids=args.participant,
            session_ids=args.session,
            mvc_scope=args.mvc_scope,
            save_figures=not args.no_figures,
            continue_on_error=not args.fail_fast,
        )
        print(
            f"Batch preprocessing complete: sessions={summary['session_count']}, "
            f"completed={summary['completed_sessions']}, failed={summary['failed_sessions']}"
        )
        return 1 if summary["failed_sessions"] else 0
    if args.command == "qc":
        from .offline import qc_session

        outputs = qc_session(args.session_path, args.profile, args.show_preview)
        print(f"Regenerated QC for {len(outputs)} trial(s).")
        return 0
    if args.command == "action-stats":
        from .action_statistics import summarize_session_actions
        from .offline import load_processing_config

        config = load_processing_config(args.config, normalization="none") if args.config else None
        outputs = summarize_session_actions(
            args.session_path,
            args.profile,
            action_ids=args.action,
            only_valid=not args.include_invalid,
            show=args.show,
            processing_config=config,
        )
        print(f"Generated repeated-trial statistics for {len(outputs)} action(s).")
        for output in outputs:
            print(output["figure"])
        return 0
    if args.command == "build-synergy-dataset":
        from .dataset import build_synergy_dataset

        output = build_synergy_dataset(
            args.dataset_root,
            args.output,
            args.profile,
            args.protocol,
            only_valid=args.only_valid,
            scope=args.scope,
            action_ids=args.action,
            participant_ids=args.participant,
            crop_mode=args.crop_mode,
            time_normalize_samples=args.time_normalize_samples,
        )
        print(f"Synergy dataset saved: {output}")
        return 0
    if args.command == "extract-synergy":
        from .synergy import extract_synergy

        output = extract_synergy(
            args.dataset,
            args.output,
            args.k_min,
            args.k_max,
            args.n_init,
            args.seed,
            args.global_vaf,
            args.local_vaf,
            args.local_fraction,
            include_h=not args.omit_h,
            calculate_stability=not args.skip_stability,
            channel_normalization=args.channel_normalization,
            split_half_repeats=args.split_half_repeats,
        )
        print(f"Synergy artifact saved: {output}")
        return 0
    raise AssertionError(f"Unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

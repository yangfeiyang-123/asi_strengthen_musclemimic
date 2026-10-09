import json

import numpy as np
import pytest

from musclemimic.badminton.asi.contact_tracking_data import load_contact_tracking_bank
from musclemimic.badminton.asi.tracking_cache import build_tracking_reference_cache
from musclemimic.badminton.data.event_lookup import (
    EventReferenceLookup,
    select_transition_coordinates,
    write_event_reference_bank_manifest,
)
from musclemimic.badminton.data.event_lookup import main as event_bank_main
from musclemimic.badminton.data.event_qc import (
    build_event_reference_metrics,
)
from musclemimic.badminton.data.event_qc import main as event_qc_main
from musclemimic.badminton.data.event_schema import load_event_timeline
from musclemimic.badminton.data.racket_reference import load_racket_reference
from musclemimic.badminton.data.reference_bundle import (
    load_reference_bundle,
    reference_bundle_fingerprint,
)
from musclemimic.badminton.data_qc import (
    inspect_event_racket_bundle,
    validate_session_split,
)
from musclemimic.core.reward.trajectory_based import _select_reference_coordinates
from musclemimic.distill.motion_identity import MotionIdentityMap, stable_motion_uid

FPS = 60.0
NUM_FRAMES = 20


def _event_payload():
    frames = {
        "ready_start": 0,
        "backswing_onset": 2,
        "backswing_apex": 4,
        "acceleration_onset": 6,
        "impact": 8,
        "followthrough_end": 14,
        "recovery_end": 19,
    }
    return {
        "schema_version": "forehand_clear_events_v1",
        "num_frames": NUM_FRAMES,
        "fps": FPS,
        "events": {
            name: {
                "frame": frame,
                "time_s": frame / FPS,
                "confidence": 0.95 if name == "impact" else 0.9,
                "source": "manual_review",
            }
            for name, frame in frames.items()
        },
    }


def _write_racket(path, *, convention="wxyz"):
    position = np.zeros((NUM_FRAMES, 3), dtype=np.float32)
    position[:, 0] = np.linspace(0.0, 1.0, NUM_FRAMES)
    quaternion = np.zeros((NUM_FRAMES, 4), dtype=np.float32)
    quaternion[:, 0] = 1.0
    normal = np.zeros((NUM_FRAMES, 3), dtype=np.float32)
    normal[:, 2] = 1.0
    np.savez(
        path,
        schema_version=np.asarray("forehand_clear_racket_reference_v1"),
        racket_reference_source=np.asarray("fused"),
        quaternion_convention=np.asarray(convention),
        coordinate_system=np.asarray("amass_zup"),
        fps=np.asarray(FPS, dtype=np.float32),
        racket_position_world=position,
        racket_quaternion_world=quaternion,
        racket_linear_velocity_world=np.ones((NUM_FRAMES, 3), dtype=np.float32),
        racket_angular_velocity_world=np.ones((NUM_FRAMES, 3), dtype=np.float32) * 2.0,
        stringbed_normal_world=normal,
        stringbed_center_world=position.copy(),
        racket_reference_confidence=np.full(NUM_FRAMES, 0.9, dtype=np.float32),
    )


def _write_v2_bundle(
    root,
    *,
    subject_id="subject-01",
    session_id="session-01",
    trial_id="trial-01",
    motion_path=None,
    source_video_id=None,
    manual_review_status="passed",
):
    root.mkdir(parents=True, exist_ok=True)
    np.savez(
        root / "motion.npz",
        poses=np.zeros((NUM_FRAMES, 72), dtype=np.float32),
        trans=np.zeros((NUM_FRAMES, 3), dtype=np.float32),
        betas=np.zeros((10,), dtype=np.float32),
        mocap_framerate=np.asarray(FPS, dtype=np.float32),
        mocap_frame_rate=np.asarray(FPS, dtype=np.float32),
        frame_ids=np.arange(NUM_FRAMES, dtype=np.int32),
        coordinate_system=np.asarray("amass_zup"),
    )
    np.savez(
        root / "contact.npz",
        contact_confidence=np.ones((NUM_FRAMES, 2), dtype=np.float32),
        stance_mask=np.ones((NUM_FRAMES, 2), dtype=np.bool_),
        foot_points=np.zeros((NUM_FRAMES, 2, 3), dtype=np.float32),
        foot_labels=np.asarray(["left_foot", "right_foot"]),
        frame_ids=np.arange(NUM_FRAMES, dtype=np.int32),
        coordinate_system=np.asarray("amass_zup"),
    )
    (root / "body_graph.json").write_text(json.dumps({"version": "body_graph_v1"}), encoding="utf-8")
    (root / "quality.json").write_text(
        json.dumps({"usable_for_training": True, "quality_tier": "A"}),
        encoding="utf-8",
    )
    (root / "events.json").write_text(json.dumps(_event_payload()), encoding="utf-8")
    _write_racket(root / "racket.npz")
    motion_path = motion_path or f"motions/{subject_id}-{session_id}-{trial_id}.npz"
    source_video_id = source_video_id or f"video-{session_id}-{trial_id}"
    provenance = {
        "subject_id": subject_id,
        "session_id": session_id,
        "trial_id": trial_id,
        "motion_path": motion_path,
        "motion_uid": stable_motion_uid(motion_path),
        "source_video_id": source_video_id,
        "source_kind": "video_fused",
        "retarget_pipeline_version": "optimized_wham_v2",
        "cache_kind": "gmr_100hz",
        "quality_tier": "A",
        "manual_review_status": manual_review_status,
        "legacy_fallback": False,
        "kinematic_confidence": 0.85,
        "racket_confidence": 0.9,
        "impact_confidence": 0.95,
        "impact_position_uncertainty_m": 0.02,
        "impact_timing_uncertainty_s": 0.01,
    }
    manifest = {
        "version": "event_reference_bundle_v2",
        "sequence": "demo",
        "num_frames": NUM_FRAMES,
        "fps": FPS,
        "coordinate_system": "amass_zup",
        "unit": "meter",
        "motion_npz": "motion.npz",
        "contact_npz": "contact.npz",
        "body_graph_json": "body_graph.json",
        "quality_report_json": "quality.json",
        "event_json": "events.json",
        "racket_npz": "racket.npz",
        "provenance": provenance,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest["content_fingerprint"] = reference_bundle_fingerprint(manifest_path, manifest=manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _write_bank_for_manifests(root, manifests):
    entries = []
    for index, manifest in enumerate(manifests):
        provenance = load_reference_bundle(manifest).provenance
        assert provenance is not None
        tracking = build_tracking_reference_cache(
            manifest,
            root / f"cache-{index}",
            control_dt=1.0 / 30.0,
        )
        entries.append(
            {
                "traj_no": index,
                "motion_uid": int(provenance["motion_uid"]),
                "motion_path": str(provenance["motion_path"]),
                "tracking_cache_npz": tracking.cache_npz,
            }
        )
    return write_event_reference_bank_manifest(root / "event_bank.json", entries=entries)


def test_v2_bundle_tracking_cache_bank_and_multimotion_lookup_roundtrip(tmp_path):
    manifest_path = _write_v2_bundle(tmp_path / "bundle")
    bundle = load_reference_bundle(manifest_path)
    qc = inspect_event_racket_bundle(bundle, min_racket_confidence=0.8)

    assert qc["passed"] is True
    assert bundle.events is not None and bundle.events.impact.frame == 8
    assert bundle.racket is not None and bundle.racket.quaternion_convention == "wxyz"
    tracking = build_tracking_reference_cache(
        manifest_path,
        tmp_path / "tracking",
        control_dt=1.0 / 30.0,
    )
    with np.load(tracking.cache_npz, allow_pickle=False) as cache:
        assert cache["phase_id"].shape == (NUM_FRAMES,)
        assert cache["reference_confidence"].shape == (NUM_FRAMES,)
        assert float(cache["reference_confidence"][8]) == pytest.approx(0.85)
        assert str(cache["racket_quaternion_convention"]) == "wxyz"
        assert str(cache["reference_motion_path"]) == bundle.provenance["motion_path"]
        assert int(cache["reference_motion_uid"]) == bundle.provenance["motion_uid"]

    identity = MotionIdentityMap.from_paths([bundle.provenance["motion_path"]])
    bank_path = write_event_reference_bank_manifest(
        tmp_path / "manifests" / "event_bank.json",
        entries=[
            {
                "traj_no": 0,
                "motion_uid": int(identity.motion_uids[0]),
                "motion_path": identity.motion_paths[0],
                # Deliberately lives outside the manifest subtree.
                "tracking_cache_npz": tracking.cache_npz,
            }
        ],
    )
    lookup = EventReferenceLookup.from_manifest(bank_path, motion_identity_map=identity)
    assert lookup.validate_control_dt(1.0 / 30.0) == pytest.approx(1.0 / 30.0)
    with pytest.raises(ValueError, match="control_dt differs from policy runtime"):
        lookup.validate_control_dt(0.01)
    assert lookup.manifest["entries"][0]["control_dt"] == pytest.approx(1.0 / 30.0)
    values = lookup.lookup_batch(
        traj_no=np.asarray([0, 0]),
        subtraj_step_no=np.asarray([0, 4]),
        motion_uid=np.asarray([identity.motion_uids[0], identity.motion_uids[0]]),
        include_racket=True,
    )

    np.testing.assert_array_equal(values["event_reference_frame"], [0, 8])
    assert values["impact_flag"].tolist() == [False, True]
    assert values["racket_position_world"].shape == (2, 3)
    assert len(lookup.fingerprint) == 64


def test_terminal_lookup_requires_and_uses_pre_reset_coordinates():
    with pytest.raises(ValueError, match="final_traj_no and final_subtraj_step_no"):
        select_transition_coordinates(
            np.asarray([1]),
            np.asarray([0]),
            np.asarray([True]),
            final_traj_no=np.asarray([0]),
            final_subtraj_step_no=None,
        )
    traj, step = select_transition_coordinates(
        np.asarray([1, 1]),
        np.asarray([0, 2]),
        np.asarray([True, False]),
        final_traj_no=np.asarray([0, 9]),
        final_subtraj_step_no=np.asarray([4, 9]),
    )
    np.testing.assert_array_equal(traj, [0, 1])
    np.testing.assert_array_equal(step, [4, 2])


def test_event_bank_cli_dry_run_validates_without_writing(tmp_path, capsys):
    manifest_path = _write_v2_bundle(tmp_path / "bundle")
    tracking = build_tracking_reference_cache(
        manifest_path,
        tmp_path / "cache",
        control_dt=1.0 / 30.0,
    )
    bundle = load_reference_bundle(manifest_path)
    assert bundle.provenance is not None
    identity = MotionIdentityMap.from_paths([bundle.provenance["motion_path"]])
    entries = tmp_path / "entries.json"
    entries.write_text(
        json.dumps(
            [
                {
                    "traj_no": 0,
                    "motion_uid": int(identity.motion_uids[0]),
                    "motion_path": identity.motion_paths[0],
                    "tracking_cache_npz": str(tracking.cache_npz.relative_to(tmp_path)),
                }
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "bank.json"
    assert event_bank_main(["--entries-json", str(entries), "--output", str(output), "--dry-run"]) == 0
    assert not output.exists()
    assert '"dry_run": true' in capsys.readouterr().out


def test_v2_fingerprint_detects_tampered_referenced_content(tmp_path):
    manifest_path = _write_v2_bundle(tmp_path)
    (tmp_path / "quality.json").write_text(
        json.dumps({"usable_for_training": False, "quality_tier": "rejected"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="content_fingerprint mismatch"):
        load_reference_bundle(manifest_path)


def test_event_order_and_quaternion_convention_fail_closed(tmp_path):
    events = _event_payload()
    events["events"]["impact"]["frame"] = 5
    events["events"]["impact"]["time_s"] = 5 / FPS
    with pytest.raises(ValueError, match="strictly increasing"):
        load_event_timeline(events)

    racket = tmp_path / "racket.npz"
    _write_racket(racket, convention="xyzw")
    with pytest.raises(ValueError, match="quaternion_convention must be 'wxyz'"):
        load_racket_reference(racket, num_frames=NUM_FRAMES, fps=FPS)


def test_recovery_phase_reaches_end_annotation_and_clamps_trailing_frames():
    events = _event_payload()
    events["events"]["recovery_end"]["frame"] = 16
    events["events"]["recovery_end"]["time_s"] = 16 / FPS

    phase = load_event_timeline(events).phase_arrays()

    np.testing.assert_allclose(phase.phase_local[14:17], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(phase.phase_local[16:], 1.0)
    np.testing.assert_allclose(phase.phase_global[16:], 1.0)


def test_event_bank_rejects_cache_bound_to_another_motion(tmp_path):
    manifest = _write_v2_bundle(
        tmp_path / "bundle",
        motion_path="motions/cache-owner.npz",
    )
    tracking = build_tracking_reference_cache(
        manifest,
        tmp_path / "tracking",
        control_dt=1.0 / 30.0,
    )
    wrong = MotionIdentityMap.from_paths(["motions/different-motion.npz"])

    with pytest.raises(ValueError, match="cache motion identity differs"):
        write_event_reference_bank_manifest(
            tmp_path / "bank.json",
            entries=[
                {
                    "traj_no": 0,
                    "motion_uid": int(wrong.motion_uids[0]),
                    "motion_path": wrong.motion_paths[0],
                    "tracking_cache_npz": tracking.cache_npz,
                }
            ],
        )


def test_session_split_qc_rejects_recording_leakage():
    train = [{"subject_id": "s1", "session_id": "day1"}]
    leaked = [{"subject_id": "s1", "session_id": "day1"}]
    clean = [{"subject_id": "s1", "session_id": "day2"}]

    assert validate_session_split(train, leaked)["passed"] is False
    assert validate_session_split(train, clean)["passed"] is True


def test_event_reference_promotion_metrics_bind_heldout_sessions(tmp_path):
    train = [
        _write_v2_bundle(
            tmp_path / f"train-{index}",
            session_id=f"train-session-{index}",
            trial_id=f"train-trial-{index}",
        )
        for index in range(5)
    ]
    validation = [
        _write_v2_bundle(
            tmp_path / f"val-{index}",
            session_id=f"val-session-{index}",
            trial_id=f"val-trial-{index}",
        )
        for index in range(5)
    ]
    train_bank = _write_bank_for_manifests(tmp_path / "train-bank", train)
    val_bank = _write_bank_for_manifests(tmp_path / "val-bank", validation)
    metrics = build_event_reference_metrics(
        train,
        validation,
        min_racket_confidence=0.8,
        train_event_bank=train_bank,
        val_event_bank=val_bank,
    )

    assert metrics["reference_count"] == 5
    assert metrics["event_valid_rate"] == 1.0
    assert metrics["racket_state_finite_rate"] == 1.0
    assert metrics["artifact_binding_verified"] == 1.0
    assert metrics["event_bank_binding_verified"] == 1.0
    assert metrics["manual_review_passed"] == 1.0
    assert metrics["source_video_split_disjoint_verified"] == 1.0
    assert len(metrics["metrics_fingerprint"]) == 64


def test_event_reference_qc_cli_dry_run_and_split_leakage(tmp_path, capsys):
    train = _write_v2_bundle(
        tmp_path / "train",
        session_id="same-session",
        trial_id="train-trial",
    )
    validation = _write_v2_bundle(
        tmp_path / "validation",
        session_id="same-session",
        trial_id="val-trial",
    )
    (tmp_path / "train.json").write_text(json.dumps([str(train.relative_to(tmp_path))]), encoding="utf-8")
    (tmp_path / "val.json").write_text(
        json.dumps({"manifests": [str(validation.relative_to(tmp_path))]}), encoding="utf-8"
    )
    train_bank = _write_bank_for_manifests(tmp_path / "train-bank", [train])
    val_bank = _write_bank_for_manifests(tmp_path / "val-bank", [validation])
    output = tmp_path / "event_metrics.json"
    assert (
        event_qc_main(
            [
                "--train-manifests-json",
                str(tmp_path / "train.json"),
                "--val-manifests-json",
                str(tmp_path / "val.json"),
                "--output",
                str(output),
                "--train-event-bank",
                str(train_bank),
                "--val-event-bank",
                str(val_bank),
                "--dry-run",
            ]
        )
        == 0
    )
    assert not output.exists()
    payload = json.loads(capsys.readouterr().out)
    assert payload["metrics"]["artifact_binding_verified"] == 0.0


@pytest.mark.parametrize(
    ("manual_status", "reuse_video", "failed_field"),
    [("failed", False, "manual_review_passed"), ("passed", True, "source_video_split_disjoint_verified")],
)
def test_event_reference_qc_rejects_failed_review_or_source_video_reuse(
    tmp_path,
    manual_status,
    reuse_video,
    failed_field,
):
    shared_video = "shared-recording"
    train = _write_v2_bundle(
        tmp_path / "train",
        session_id="train-session",
        trial_id="train-trial",
        source_video_id=shared_video,
        manual_review_status=manual_status,
    )
    validation = _write_v2_bundle(
        tmp_path / "validation",
        session_id="val-session",
        trial_id="val-trial",
        source_video_id=shared_video if reuse_video else "different-recording",
    )
    train_bank = _write_bank_for_manifests(tmp_path / "train-bank", [train])
    val_bank = _write_bank_for_manifests(tmp_path / "val-bank", [validation])

    metrics = build_event_reference_metrics(
        [train],
        [validation],
        train_event_bank=train_bank,
        val_event_bank=val_bank,
    )

    assert metrics["artifact_binding_verified"] == 0.0
    assert metrics[failed_field] == 0.0


def test_multimotion_contact_bank_is_exactly_trajectory_indexed(tmp_path, monkeypatch):
    motion_paths = ["motions/forehand-a.npz", "motions/forehand-b.npz"]
    identity = MotionIdentityMap.from_paths(motion_paths)
    entries = []
    for index, motion_path in enumerate(motion_paths):
        manifest = _write_v2_bundle(
            tmp_path / f"bundle-{index}",
            session_id=f"session-{index}",
            trial_id=f"trial-{index}",
            motion_path=motion_path,
        )
        tracking = build_tracking_reference_cache(
            manifest,
            tmp_path / f"tracking-{index}",
            control_dt=1.0 / 30.0,
        )
        entries.append(
            {
                "traj_no": index,
                "motion_uid": int(identity.motion_uids[index]),
                "motion_path": motion_path,
                "tracking_cache_npz": tracking.cache_npz,
            }
        )
    bank_manifest = write_event_reference_bank_manifest(
        tmp_path / "bank" / "event_bank.json",
        entries=entries,
    )

    bank = load_contact_tracking_bank(
        bank_manifest,
        control_dt=1.0 / 30.0,
        motion_identity_map=identity,
    )

    assert bank.num_trajectories == 2
    assert bank.stance_mask.shape == (2, NUM_FRAMES, 2)
    assert bank.phase_id.shape == (2, NUM_FRAMES)
    assert bank.racket_position_world.shape == (2, NUM_FRAMES, 3)
    np.testing.assert_array_equal(bank.motion_uids, identity.motion_uids)
    assert bank.frame_at_traj_step(1, 4) == 8
    assert len(bank.event_reference_bank_fingerprint) == 64

    # Regression: attaching a bank must preserve [trajectory, frame, foot]
    # when selecting the z coordinate.  Indexing ``[:, :, 2]`` would instead
    # select foot 2 (or crash for this two-foot fixture) and lose the foot axis.
    import types

    import musclemimic.core.reward.trajectory_based as trajectory_reward

    fake_mujoco = types.SimpleNamespace(
        mjtObj=types.SimpleNamespace(mjOBJ_SITE=object()),
        mj_name2id=lambda _model, _kind, name: bank.foot_labels.index(name),
    )
    monkeypatch.setattr(trajectory_reward, "mujoco", fake_mujoco)
    reward = object.__new__(trajectory_reward.MimicReward)
    reward.attach_contact_tracking(bank, bank.foot_labels, object())
    assert tuple(reward._ctd_foot_z.shape) == (2, NUM_FRAMES, 2)

    reversed_identity = MotionIdentityMap.from_paths(reversed(motion_paths))
    with pytest.raises(ValueError, match="identity mismatch"):
        load_contact_tracking_bank(
            bank_manifest,
            control_dt=1.0 / 30.0,
            motion_identity_map=reversed_identity,
        )


def test_reward_reference_coordinates_select_per_motion_stride_and_length():
    trajectory, frame = _select_reference_coordinates(
        np.asarray(1, dtype=np.int32),
        np.asarray(4, dtype=np.int32),
        np.asarray([1.0, 2.0], dtype=np.float32),
        np.asarray([5, 20], dtype=np.int32),
        is_bank=True,
        backend=np,
    )

    assert int(trajectory) == 1
    assert int(frame) == 8

    _trajectory, clipped = _select_reference_coordinates(
        np.asarray(0, dtype=np.int32),
        np.asarray(99, dtype=np.int32),
        np.asarray([1.0], dtype=np.float32),
        np.asarray([5], dtype=np.int32),
        is_bank=True,
        backend=np,
    )
    assert int(clipped) == 4

import hashlib
import json

import numpy as np

from musclemimic.badminton.data.event_lookup import (
    EventReferenceLookup,
    write_event_reference_bank_manifest,
)
from musclemimic.distill.action_schema import ordered_schema_hash
from musclemimic.distill.dataset import write_split_shard
from musclemimic.distill.motion_identity import stable_motion_uid
from musclemimic.distill.physical import (
    MUSCLE_CHANNEL_CONTRACT_SCHEMA_VERSION,
    PHYSICAL_CAPTURE_SCHEMA_VERSION,
    physical_signal_metadata,
)
from musclemimic.distill.physical_qc import build_physical_rollout_metrics

TEACHER_SHA256 = "a" * 64


def _teacher_content():
    return {
        "schema_version": "checkpoint_content_fingerprint_v1",
        "supplied_path": "fixture",
        "resolved_path": "/fixture/checkpoint",
        "sha256": TEACHER_SHA256,
        "num_files": 1,
        "num_bytes": 1,
        "files": [{"path": "params", "sha256": "b" * 64, "num_bytes": 1}],
    }


def _write_physical_split(
    root,
    *,
    split,
    motion_path,
    bundle_fingerprint,
    tamper_event=False,
):
    samples = 16
    muscles = 2
    motion_uid = stable_motion_uid(motion_path)
    phase_id = np.zeros(samples, dtype=np.int32)
    phase_id[8] = 3
    impact = np.zeros(samples, dtype=bool)
    impact[8] = True
    time_from = (np.arange(samples, dtype=np.float32) - 8.0) * 0.01
    phase_local = np.linspace(0.0, 1.0, samples, dtype=np.float32)
    cache_path = root / "event_evidence" / "tracking_reference_cache.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        phase_global=np.linspace(0.0, 1.0, samples, dtype=np.float32),
        phase_id=phase_id,
        phase_local=phase_local,
        time_to_impact_s=-time_from,
        time_from_impact_s=time_from,
        impact_flag=impact,
        reference_confidence=np.full(samples, 0.9, dtype=np.float32),
        racket_position_world=np.zeros((samples, 3), dtype=np.float32),
        racket_quaternion_world=np.tile(
            np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            (samples, 1),
        ),
        racket_linear_velocity_world=np.zeros((samples, 3), dtype=np.float32),
        racket_angular_velocity_world=np.zeros((samples, 3), dtype=np.float32),
        stringbed_normal_world=np.tile(
            np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            (samples, 1),
        ),
        stringbed_center_world=np.zeros((samples, 3), dtype=np.float32),
        racket_reference_confidence=np.full(samples, 0.9, dtype=np.float32),
        reference_fps=np.asarray(100.0, dtype=np.float32),
        control_dt=np.asarray(0.01, dtype=np.float32),
        effective_ref_stride=np.asarray(1.0, dtype=np.float32),
        reference_bundle_content_fingerprint=np.asarray(bundle_fingerprint),
        reference_motion_path=np.asarray(motion_path),
        reference_motion_uid=np.asarray(motion_uid, dtype=np.int64),
    )
    bank_manifest = write_event_reference_bank_manifest(
        root / "event_evidence" / "event_bank.json",
        entries=[
            {
                "traj_no": 0,
                "motion_uid": motion_uid,
                "motion_path": motion_path,
                "tracking_cache_npz": cache_path,
            }
        ],
    )
    lookup = EventReferenceLookup.from_manifest(bank_manifest)
    event = lookup.lookup_batch(
        traj_no=np.zeros(samples, dtype=np.int32),
        subtraj_step_no=np.arange(samples, dtype=np.int32),
        motion_uid=np.full(samples, motion_uid, dtype=np.int64),
    )
    if tamper_event:
        event["phase_local"] = event["phase_local"].copy()
        event["phase_local"][0] = 0.25
    unit = np.full((samples, muscles), 0.4, dtype=np.float32)
    names = ["a", "b"]
    ctrlrange = np.tile(np.asarray([[0.0, 1.0]]), (muscles, 1))
    data = {
        "student_obs": np.zeros((samples, 3), dtype=np.float32),
        "teacher_action": np.zeros((samples, muscles), dtype=np.float32),
        "teacher_ctrl_physical": unit.copy(),
        "muscle_excitation": unit,
        "muscle_activation": 0.8 * unit,
        "muscle_force": np.ones((samples, muscles), dtype=np.float32),
        "muscle_tendon_length": np.ones((samples, muscles), dtype=np.float32),
        "muscle_tendon_velocity": np.zeros((samples, muscles), dtype=np.float32),
        "actuator_power": np.zeros((samples, muscles), dtype=np.float32),
        "qfrc_actuator": np.zeros((samples, 4), dtype=np.float32),
        **event,
        "traj_no": np.zeros(samples, dtype=np.int32),
        "subtraj_step_no": np.arange(samples, dtype=np.int32),
        "rollout_uid": np.repeat(np.arange(4, dtype=np.int64), 4)
        + (100 if split == "val" else 0),
        "motion_uid": np.full(samples, motion_uid, dtype=np.int64),
    }
    metadata = {
        "actuator_names": names,
        "actuator_ctrlrange": ctrlrange.tolist(),
        "ctrlrange_schema_hash": ordered_schema_hash(
            kind="actuator_ctrlrange",
            payload={"actuator_names": names, "ctrlrange": ctrlrange.tolist()},
        ),
        "physical_signal_semantics": physical_signal_metadata(),
        "physical_capture": {
            "schema_version": PHYSICAL_CAPTURE_SCHEMA_VERSION,
            "actuator_names": names,
            "activation_valid_mask": [True] * len(names),
            "muscle_channel_contract": {
                "schema_version": MUSCLE_CHANNEL_CONTRACT_SCHEMA_VERSION,
                "actuator_names": names,
                "actuator_ids": list(range(muscles)),
                "actuator_dyntype": ["muscle"] * muscles,
                "actuator_actnum": [1] * muscles,
                "actuator_actadr": list(range(muscles)),
                "model_na": muscles,
            },
        },
        "teacher_checkpoint_fingerprint": TEACHER_SHA256,
        "teacher_checkpoint_content": _teacher_content(),
        "event_reference_bank_manifest": str(bank_manifest),
        "event_reference_bank_fingerprint": lookup.fingerprint,
        "event_reference_control_dt": 0.01,
        "event_reference_bundle_fingerprints": [bundle_fingerprint],
        "event_reference_bank_motion_uids": [motion_uid],
        "event_reference_bank_motion_paths": [motion_path],
    }
    write_split_shard(root, data, split=split, metadata=metadata)
    return {
        "bank_fingerprint": lookup.fingerprint,
        "bundle_fingerprint": bundle_fingerprint,
        "motion_uid": motion_uid,
    }


def test_physical_rollout_qc_emits_gate_ready_bound_metrics(tmp_path):
    train = tmp_path / "train"
    val = tmp_path / "val"
    train_bundle = "e" * 64
    val_bundle = "f" * 64
    train_evidence = _write_physical_split(
        train,
        split="train",
        motion_path="motions/train-1.npz",
        bundle_fingerprint=train_bundle,
    )
    val_evidence = _write_physical_split(
        val,
        split="val",
        motion_path="motions/val-1.npz",
        bundle_fingerprint=val_bundle,
    )
    event_metrics = {
        "schema_version": "event_reference_promotion_metrics_v1",
        "artifact_binding_verified": 1.0,
        "event_bank_binding_verified": 1.0,
        "train_reference_bundle_fingerprints": [train_bundle],
        "validation_reference_bundle_fingerprints": [val_bundle],
        "train_event_reference_bank_fingerprint": train_evidence["bank_fingerprint"],
        "validation_event_reference_bank_fingerprint": val_evidence["bank_fingerprint"],
    }
    event_metrics["metrics_fingerprint"] = hashlib.sha256(
        json.dumps(
            event_metrics,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    metrics = build_physical_rollout_metrics(
        train,
        val,
        teacher_checkpoint_fingerprint=TEACHER_SHA256,
        event_reference_metrics=event_metrics,
    )

    assert metrics["rollout_count"] == 4
    assert metrics["finite_rate"] == 1.0
    assert metrics["reference_alignment_rate"] == 1.0
    assert metrics["exact_event_reference_rate"] == 1.0
    assert metrics["action_saturation_fraction"] == 0.0
    assert metrics["checkpoint_binding_verified"] == 1.0
    assert metrics["split_disjoint_verified"] == 1.0
    assert metrics["event_reference_binding_verified"] == 1.0
    assert len(metrics["metrics_fingerprint"]) == 64

    wrong_event_metrics = dict(event_metrics)
    wrong_event_metrics.pop("metrics_fingerprint")
    wrong_event_metrics["train_reference_bundle_fingerprints"] = ["9" * 64]
    wrong_event_metrics["metrics_fingerprint"] = hashlib.sha256(
        json.dumps(
            wrong_event_metrics,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    rejected = build_physical_rollout_metrics(
        train,
        val,
        teacher_checkpoint_fingerprint=TEACHER_SHA256,
        event_reference_metrics=wrong_event_metrics,
    )
    assert rejected["event_reference_binding_verified"] == 0.0
    assert rejected["checkpoint_binding_verified"] == 0.0


def test_physical_rollout_qc_detects_semantically_plausible_event_tampering(tmp_path):
    train_bundle = "e" * 64
    val_bundle = "f" * 64
    train_evidence = _write_physical_split(
        tmp_path / "train",
        split="train",
        motion_path="motions/train.npz",
        bundle_fingerprint=train_bundle,
        tamper_event=True,
    )
    val_evidence = _write_physical_split(
        tmp_path / "val",
        split="val",
        motion_path="motions/val.npz",
        bundle_fingerprint=val_bundle,
    )
    event_metrics = {
        "schema_version": "event_reference_promotion_metrics_v1",
        "artifact_binding_verified": 1.0,
        "event_bank_binding_verified": 1.0,
        "train_reference_bundle_fingerprints": [train_bundle],
        "validation_reference_bundle_fingerprints": [val_bundle],
        "train_event_reference_bank_fingerprint": train_evidence["bank_fingerprint"],
        "validation_event_reference_bank_fingerprint": val_evidence["bank_fingerprint"],
    }
    event_metrics["metrics_fingerprint"] = hashlib.sha256(
        json.dumps(
            event_metrics,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    metrics = build_physical_rollout_metrics(
        tmp_path / "train",
        tmp_path / "val",
        teacher_checkpoint_fingerprint=TEACHER_SHA256,
        event_reference_metrics=event_metrics,
    )

    assert metrics["exact_event_reference_rate"] < 1.0
    assert metrics["event_reference_binding_verified"] == 0.0
    assert metrics["checkpoint_binding_verified"] == 0.0

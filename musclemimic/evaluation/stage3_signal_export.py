"""Fail-closed physical-signal export from final Stage-3 evaluation.

The exporter is deliberately attached to the CPU replay used by the canonical
Stage-3 evaluator.  A checkpoint trained with MJX/Warp is therefore replayed
through the same policy/LAB stack while MuJoCo's transition state is available
for exact ``data.ctrl``, scalar muscle activation and generalized-force reads.

Only successful, non-fall trials with a complete window around the first real
``hit_this_step`` event are accepted.  Trial/subject/session identities are
never synthesized: callers must provide a feed-fingerprint-bound identity
manifest for the held-out trials.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from musclemimic.badminton.json_contract import load_json_strict
from musclemimic.distill.physical import (
    MUSCLE_ACTIVATION_SEMANTICS,
    MUSCLE_ACTIVATION_SOURCE,
    MUSCLE_CHANNEL_CONTRACT_SCHEMA_VERSION,
    MUSCLE_EXCITATION_FORMULA,
    MUSCLE_EXCITATION_ROUNDOFF_POLICY,
    MUSCLE_EXCITATION_SEMANTICS,
    MUSCLE_EXCITATION_SOURCE,
    PHYSICAL_SIGNAL_SCHEMA_VERSION,
    UNIT_EXCITATION_TRANSFORM,
    UNIT_INTERVAL_ROUNDOFF_POLICY,
    MuscleChannelContract,
    physical_ctrl_to_effective_muscle_excitation,
    resolve_muscle_channel_contract,
    validate_unit_muscle_activation,
    validate_unit_muscle_ctrlrange,
)
from musclemimic.physiology.synergy_binding import ordered_muscle_schema_sha256

LEGACY_TRIAL_IDENTITY_SCHEMA_VERSION = "stage3_signal_trial_identity_v1"
TRIAL_IDENTITY_SCHEMA_VERSION = "stage3_signal_trial_identity_v2"
PAIRED_EMG_COMPARISON_DESIGN = "paired_same_reference_v1"
UNPAIRED_EMG_COMPARISON_DESIGN = "unpaired_action_cohort_v1"
SIGNAL_EXPORT_SCHEMA_VERSION = "stage3_policy_physical_signals_v2"
SIGNAL_EXPORT_MANIFEST_SCHEMA_VERSION = "stage3_policy_physical_signals_manifest_v2"
PAIRED_COMPARISON_SCHEMA_VERSION = "stage3_direct_synergy_paired_comparison_v2"

_SHA256_CHARS = frozenset("0123456789abcdef")
_EVENT_STATE_CODES = {
    "INCOMING": 0,
    "HIT": 1,
    "FLIGHT": 2,
    "RECOVERY": 3,
    "DONE": 4,
}
_FOREHAND_PHASE_NAMES = (
    "ready",
    "backswing",
    "acceleration",
    "impact",
    "followthrough",
    "recovery",
)


@dataclass(frozen=True)
class TrialIdentity:
    feed_index: int
    feed_fingerprint: str
    trial_uid: str
    subject_uid: str
    session_uid: str
    reference_trial_fingerprint: str | None = None


@dataclass(frozen=True)
class TrialIdentityManifest:
    schema_version: str
    dataset_split: str
    training_session_uids: tuple[str, ...]
    trials_by_feed: dict[int, TrialIdentity]
    manifest_fingerprint: str
    source_path: Path
    source_sha256: str
    action_id: str | None
    handedness: str | None
    comparison_design: str | None
    comparison_set_uid: str | None
    model_taxonomy_id: str | None
    model_taxonomy_fingerprint: str | None
    runtime_model_hash: str | None
    actuator_schema_hash: str | None
    taxonomy_source_path: Path | None
    taxonomy_source_sha256: str | None
    taxonomy_ordered_actuators: tuple[Mapping[str, Any], ...]

    @property
    def is_emg_v2(self) -> bool:
        return self.schema_version == TRIAL_IDENTITY_SCHEMA_VERSION

    def require(self, *, feed_index: int, feed_fingerprint: str) -> TrialIdentity:
        try:
            identity = self.trials_by_feed[int(feed_index)]
        except KeyError as exc:
            raise ValueError(f"trial identity manifest has no held-out feed_index={int(feed_index)}") from exc
        if identity.feed_fingerprint != _require_sha256(feed_fingerprint, "evaluation feed fingerprint"):
            raise ValueError(f"trial identity feed fingerprint differs for feed_index={int(feed_index)}")
        return identity


@dataclass(frozen=True)
class Stage3PolicyEvidence:
    family: str
    decoder_type: str
    policy_checkpoint_fingerprint: str
    policy_promotion_fingerprint: str
    formal_synergy_basis_fingerprint: str
    event_reference_fingerprint: str
    stage3_checkpoint_payload_sha256: str
    paired_comparison_fingerprint: str
    source_path: Path
    source_sha256: str


@dataclass(frozen=True)
class Stage3SignalLayout:
    actuator_names: tuple[str, ...]
    actuator_ids: np.ndarray
    activation_addresses: np.ndarray
    actuator_ctrlrange: np.ndarray
    activation_valid_mask: np.ndarray
    muscle_channel_contract: MuscleChannelContract
    joint_names: tuple[str, ...]
    joint_dof_addresses: np.ndarray
    scene_runtime_model_hash: str | None = None

    @classmethod
    def from_environment(
        cls,
        env: Any,
        *,
        body_actuator_names: Sequence[str],
    ) -> Stage3SignalLayout:
        """Resolve the exact ordered muscle and scalar-hinge signal layout."""

        import mujoco

        model = env.model
        model_state = model.__getstate__()
        if not isinstance(model_state, bytes) or not model_state:
            raise ValueError("Stage-3 runtime MuJoCo scene has no canonical byte state")
        names = tuple(str(name) for name in body_actuator_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("Stage-3 signal export requires unique ordered body actuator names")
        channel_contract = resolve_muscle_channel_contract(model, names)
        ids = np.asarray(channel_contract.actuator_ids, dtype=np.int32)
        ctrlrange = validate_unit_muscle_ctrlrange(
            names,
            np.asarray(model.actuator_ctrlrange, dtype=float)[ids],
        )

        joint_names: list[str] = []
        joint_dof_addresses: list[int] = []
        for joint_id in range(int(model.njnt)):
            if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if not name:
                raise ValueError(f"hinge joint id={joint_id} has no stable MuJoCo name")
            joint_names.append(str(name))
            joint_dof_addresses.append(int(model.jnt_dofadr[joint_id]))
        if not joint_names or len(set(joint_names)) != len(joint_names):
            raise ValueError("Stage-3 signal export found no unique scalar hinge-joint channels")
        return cls(
            actuator_names=names,
            actuator_ids=ids,
            activation_addresses=np.asarray(channel_contract.actuator_actadr, dtype=np.int32),
            actuator_ctrlrange=ctrlrange,
            activation_valid_mask=np.ones((len(names),), dtype=bool),
            muscle_channel_contract=channel_contract,
            joint_names=tuple(joint_names),
            joint_dof_addresses=np.asarray(joint_dof_addresses, dtype=np.int32),
            scene_runtime_model_hash=hashlib.sha256(model_state).hexdigest(),
        )

    def capture_transition(self, env: Any, info: Mapping[str, Any]) -> dict[str, Any]:
        """Read one post-step transition without relabeling action as activation."""

        data = env.data
        ctrl = np.asarray(data.ctrl, dtype=np.float64)[self.actuator_ids].copy()
        excitation = physical_ctrl_to_effective_muscle_excitation(
            ctrl,
            channel_contract=self.muscle_channel_contract,
        )
        activation = validate_unit_muscle_activation(np.asarray(data.act, dtype=np.float64)[self.activation_addresses])
        dofs = self.joint_dof_addresses
        state_name = str(getattr(getattr(env, "state", None), "value", getattr(env, "state", "")))
        if state_name not in _EVENT_STATE_CODES:
            raise ValueError(f"unsupported Stage-3 event state during signal capture: {state_name!r}")
        control_dt = float(env.control_substeps) * float(env.model.opt.timestep)
        elapsed = float(env.step_index) * control_dt
        return {
            "teacher_ctrl_physical": ctrl.astype(np.float32),
            "muscle_excitation": excitation.astype(np.float32),
            "muscle_activation": activation.astype(np.float32),
            "joint_torque": np.asarray(data.qfrc_actuator, dtype=np.float64)[dofs].astype(np.float32),
            "joint_angular_velocity": np.asarray(data.qvel, dtype=np.float64)[dofs].astype(np.float32),
            "step_index": int(env.step_index),
            "elapsed_time_s": elapsed,
            "swing_phase": float(env._swing_phase()),
            "event_state_code": int(_EVENT_STATE_CODES[state_name]),
            "hit_this_step": bool(info.get("hit_this_step", False)),
            "body_fall": bool(info.get("body_fall", False)),
            "recovery_complete": bool(info.get("recovery_complete", False)),
        }


@dataclass
class _EpisodeBuffer:
    episode_index: int
    feed_index: int
    feed_fingerprint: str
    identity: TrialIdentity
    transitions: list[dict[str, Any]]


class Stage3SignalCollector:
    """Collect dense, real-impact-aligned Stage-3 physical signal trials."""

    def __init__(
        self,
        *,
        layout: Stage3SignalLayout,
        identities: TrialIdentityManifest,
        policy_evidence: Stage3PolicyEvidence,
        control_dt_s: float,
        pre_impact_s: float,
        post_impact_s: float,
        expected_episode_count: int,
        runtime: Any,
        event_reference_fingerprint: str,
        stage3_checkpoint_payload_sha256: str,
        evaluation_feed_manifest_fingerprint: str,
        evaluation_seed: int,
    ) -> None:
        self.layout = layout
        self.identities = identities
        self.policy_evidence = policy_evidence
        self.control_dt_s = _positive_float(control_dt_s, "control_dt_s")
        self.pre_steps = _seconds_to_steps(pre_impact_s, self.control_dt_s, "pre_impact_s")
        self.post_steps = _seconds_to_steps(post_impact_s, self.control_dt_s, "post_impact_s")
        self.expected_episode_count = int(expected_episode_count)
        if self.expected_episode_count <= 0:
            raise ValueError("Stage-3 signal export expected_episode_count must be positive")
        self.evaluation_seed = int(evaluation_seed)
        self.event_reference_fingerprint = _require_sha256(
            event_reference_fingerprint,
            "event_reference_fingerprint",
        )
        self.stage3_checkpoint_payload_sha256 = _require_sha256(
            stage3_checkpoint_payload_sha256,
            "stage3_checkpoint_payload_sha256",
        )
        self.evaluation_feed_manifest_fingerprint = _require_sha256(
            evaluation_feed_manifest_fingerprint,
            "evaluation_feed_manifest_fingerprint",
        )
        if _file_sha256(policy_evidence.source_path) != _require_sha256(
            policy_evidence.source_sha256,
            "paired comparison file fingerprint",
        ):
            raise ValueError("paired Stage-3 policy evidence changed before signal collection")
        if _file_sha256(identities.source_path) != identities.source_sha256:
            raise ValueError("Stage-3 trial identity manifest changed before signal collection")
        self._runtime_basis_fields = _validate_policy_runtime_binding(
            policy_evidence,
            runtime=runtime,
            event_reference_fingerprint=self.event_reference_fingerprint,
            stage3_checkpoint_payload_sha256=self.stage3_checkpoint_payload_sha256,
        )
        _validate_identity_taxonomy_layout(identities, layout)
        self._current: _EpisodeBuffer | None = None
        self._trials: list[dict[str, Any]] = []

    def begin_episode(
        self,
        *,
        episode_index: int,
        feed_index: int,
        feed_fingerprint: str,
    ) -> None:
        if self._current is not None:
            raise RuntimeError("previous Stage-3 signal episode was not finalized")
        if int(episode_index) != len(self._trials):
            raise ValueError("Stage-3 signal episodes must be collected in exact contiguous order")
        identity = self.identities.require(
            feed_index=int(feed_index),
            feed_fingerprint=feed_fingerprint,
        )
        self._current = _EpisodeBuffer(
            episode_index=int(episode_index),
            feed_index=int(feed_index),
            feed_fingerprint=_require_sha256(feed_fingerprint, "feed_fingerprint"),
            identity=identity,
            transitions=[],
        )

    def record_transition(self, transition: Mapping[str, Any]) -> None:
        if self._current is None:
            raise RuntimeError("begin_episode() must precede Stage-3 signal transitions")
        self._current.transitions.append(_validate_transition(transition, self.layout))

    def end_episode(self) -> None:
        if self._current is None:
            raise RuntimeError("no Stage-3 signal episode is active")
        buffer = self._current
        self._current = None
        transitions = buffer.transitions
        if not transitions:
            raise ValueError(f"Stage-3 signal episode {buffer.episode_index} contains no transitions")
        if any(bool(record["body_fall"]) for record in transitions):
            raise ValueError(
                f"Stage-3 signal episode {buffer.episode_index} had a body fall; physiology export fails closed"
            )
        hit_indices = [index for index, record in enumerate(transitions) if bool(record["hit_this_step"])]
        if not hit_indices:
            raise ValueError(
                f"Stage-3 signal episode {buffer.episode_index} has no real hit_this_step; physiology export fails closed"
            )
        impact_index = int(hit_indices[0])
        start = impact_index - self.pre_steps
        stop = impact_index + self.post_steps + 1
        if start < 0 or stop > len(transitions):
            raise ValueError(
                "Stage-3 signal episode lacks the complete requested real-impact window: "
                f"episode={buffer.episode_index} impact={impact_index} available={len(transitions)} "
                f"required_pre={self.pre_steps} required_post={self.post_steps}"
            )
        phase = _derive_phase_arrays(transitions, impact_index=impact_index)
        selected = transitions[start:stop]
        trial: dict[str, Any] = {
            key: np.stack([np.asarray(record[key]) for record in selected], axis=0)
            for key in (
                "teacher_ctrl_physical",
                "muscle_excitation",
                "muscle_activation",
                "joint_torque",
                "joint_angular_velocity",
            )
        }
        for key, dtype in (
            ("step_index", np.int32),
            ("elapsed_time_s", np.float32),
            ("swing_phase", np.float32),
            ("event_state_code", np.int8),
            ("body_fall", bool),
            ("recovery_complete", bool),
        ):
            trial[key] = np.asarray([record[key] for record in selected], dtype=dtype)
        for key, values in phase.items():
            trial[key] = np.asarray(values[start:stop])
        trial.update(
            {
                "impact_frame": np.int32(self.pre_steps),
                "episode_index": np.int32(buffer.episode_index),
                "feed_index": np.int32(buffer.feed_index),
                "feed_fingerprint": buffer.feed_fingerprint,
                "trial_uid": buffer.identity.trial_uid,
                "subject_uid": buffer.identity.subject_uid,
                "session_uid": buffer.identity.session_uid,
            }
        )
        if buffer.identity.reference_trial_fingerprint is not None:
            trial["reference_trial_fingerprint"] = buffer.identity.reference_trial_fingerprint
        self._trials.append(trial)

    def finalize_arrays(self, *, evaluation_binding_sha256: str) -> dict[str, np.ndarray]:
        if self._current is not None:
            raise RuntimeError("cannot finalize Stage-3 signal export with an active episode")
        if len(self._trials) != self.expected_episode_count:
            raise ValueError(
                f"Stage-3 signal export collected {len(self._trials)} trials; "
                f"expected exactly {self.expected_episode_count}"
            )
        evaluation_binding = _require_sha256(
            evaluation_binding_sha256,
            "evaluation_binding_sha256",
        )
        arrays: dict[str, np.ndarray] = {}
        time_fields = (
            "teacher_ctrl_physical",
            "muscle_excitation",
            "muscle_activation",
            "joint_torque",
            "joint_angular_velocity",
            "step_index",
            "elapsed_time_s",
            "swing_phase",
            "event_state_code",
            "body_fall",
            "recovery_complete",
            "phase_global",
            "phase_id",
            "phase_local",
            "time_to_impact_s",
            "time_from_impact_s",
            "impact_flag",
        )
        for key in time_fields:
            arrays[key] = np.stack([np.asarray(trial[key]) for trial in self._trials], axis=0)
        for key, dtype in (
            ("impact_frame", np.int32),
            ("episode_index", np.int32),
            ("feed_index", np.int32),
        ):
            arrays[key] = np.asarray([trial[key] for trial in self._trials], dtype=dtype)
        for key in ("feed_fingerprint", "trial_uid", "subject_uid", "session_uid"):
            arrays[key] = _string_array([str(trial[key]) for trial in self._trials])
        if self.identities.is_emg_v2:
            arrays.update(
                {
                    "trial_identity_schema_version": np.asarray(self.identities.schema_version),
                    "action_id": np.asarray(self.identities.action_id),
                    "handedness": np.asarray(self.identities.handedness),
                    "comparison_design": np.asarray(self.identities.comparison_design),
                    "comparison_set_uid": np.asarray(self.identities.comparison_set_uid),
                    "model_taxonomy_id": np.asarray(self.identities.model_taxonomy_id),
                    "model_taxonomy_fingerprint": np.asarray(self.identities.model_taxonomy_fingerprint),
                    "runtime_model_hash": np.asarray(self.identities.runtime_model_hash),
                    "actuator_schema_hash": np.asarray(self.identities.actuator_schema_hash),
                    "taxonomy_source_sha256": np.asarray(self.identities.taxonomy_source_sha256),
                    "scene_runtime_model_hash": np.asarray(self.layout.scene_runtime_model_hash),
                }
            )
            if self.identities.comparison_design == PAIRED_EMG_COMPARISON_DESIGN:
                arrays["reference_trial_fingerprint"] = _string_array(
                    [str(trial["reference_trial_fingerprint"]) for trial in self._trials]
                )
        evidence = self.policy_evidence
        arrays.update(
            {
                "stage3_signal_export_schema_version": np.asarray(SIGNAL_EXPORT_SCHEMA_VERSION),
                "physical_signal_schema_version": np.asarray(PHYSICAL_SIGNAL_SCHEMA_VERSION),
                "muscle_excitation_source": np.asarray(MUSCLE_EXCITATION_SOURCE),
                "muscle_excitation_semantics": np.asarray(MUSCLE_EXCITATION_SEMANTICS),
                "muscle_excitation_transform": np.asarray(UNIT_EXCITATION_TRANSFORM),
                "muscle_excitation_formula": np.asarray(MUSCLE_EXCITATION_FORMULA),
                "muscle_excitation_roundoff_policy": np.asarray(MUSCLE_EXCITATION_ROUNDOFF_POLICY),
                "muscle_activation_source": np.asarray(MUSCLE_ACTIVATION_SOURCE),
                "muscle_activation_semantics": np.asarray(MUSCLE_ACTIVATION_SEMANTICS),
                "muscle_activation_roundoff_policy": np.asarray(UNIT_INTERVAL_ROUNDOFF_POLICY),
                "activation_valid_mask": self.layout.activation_valid_mask.astype(bool),
                "actuator_names": _string_array(self.layout.actuator_names),
                # Publish the exported channel order under the synergy stack's own
                # hash so the offline physiology report can bind these channels to
                # the anatomy taxonomy by hash, not just by name comparison.
                "ordered_muscle_schema_sha256": np.asarray(ordered_muscle_schema_sha256(self.layout.actuator_names)),
                "actuator_ctrlrange": self.layout.actuator_ctrlrange.astype(np.float32),
                "muscle_channel_contract_schema_version": np.asarray(MUSCLE_CHANNEL_CONTRACT_SCHEMA_VERSION),
                "actuator_ids": np.asarray(
                    self.layout.muscle_channel_contract.actuator_ids,
                    dtype=np.int32,
                ),
                "actuator_dyntype": _string_array(self.layout.muscle_channel_contract.actuator_dyntype),
                "actuator_actnum": np.asarray(
                    self.layout.muscle_channel_contract.actuator_actnum,
                    dtype=np.int32,
                ),
                "actuator_actadr": np.asarray(
                    self.layout.muscle_channel_contract.actuator_actadr,
                    dtype=np.int32,
                ),
                "model_na": np.asarray(
                    self.layout.muscle_channel_contract.model_na,
                    dtype=np.int32,
                ),
                "joint_names": _string_array(self.layout.joint_names),
                "sampling_rate_hz": np.asarray(1.0 / self.control_dt_s, dtype=np.float64),
                "control_dt_s": np.asarray(self.control_dt_s, dtype=np.float64),
                "forehand_phase_names": _string_array(_FOREHAND_PHASE_NAMES),
                "stage3_event_state_names": _string_array(tuple(_EVENT_STATE_CODES)),
                "dataset_split": np.asarray(self.identities.dataset_split),
                "training_session_uid": _string_array(self.identities.training_session_uids),
                "trial_identity_manifest_fingerprint": np.asarray(self.identities.manifest_fingerprint),
                "policy_decoder_type": np.asarray(evidence.decoder_type),
                "policy_checkpoint_fingerprint": np.asarray(evidence.policy_checkpoint_fingerprint),
                "policy_promotion_fingerprint": np.asarray(evidence.policy_promotion_fingerprint),
                "formal_synergy_basis_fingerprint": np.asarray(evidence.formal_synergy_basis_fingerprint),
                "analysis_synergy_basis_fingerprint": np.asarray(evidence.formal_synergy_basis_fingerprint),
                "event_reference_fingerprint": np.asarray(evidence.event_reference_fingerprint),
                "stage3_checkpoint_payload_sha256": np.asarray(self.stage3_checkpoint_payload_sha256),
                "stage3_evaluation_binding_sha256": np.asarray(evaluation_binding),
                "stage3_paired_comparison_fingerprint": np.asarray(evidence.paired_comparison_fingerprint),
                "evaluation_feed_manifest_fingerprint": np.asarray(self.evaluation_feed_manifest_fingerprint),
                "evaluation_seed": np.asarray(self.evaluation_seed, dtype=np.int64),
                "impact_window_pre_steps": np.asarray(self.pre_steps, dtype=np.int32),
                "impact_window_post_steps": np.asarray(self.post_steps, dtype=np.int32),
                "valid_time_mask": np.ones(arrays["impact_flag"].shape, dtype=bool),
            }
        )
        arrays.update(self._runtime_basis_fields)
        _validate_export_arrays(arrays)
        fingerprint = _arrays_fingerprint(arrays)
        arrays["signal_export_fingerprint"] = np.asarray(fingerprint)
        return arrays


def load_trial_identity_manifest(path: str | Path) -> TrialIdentityManifest:
    source_path = Path(path).expanduser().resolve(strict=True)
    payload = load_json_strict(source_path)
    if not isinstance(payload, dict):
        raise ValueError("trial identity manifest must be a JSON object")
    schema_version = str(payload.get("schema_version", ""))
    supported_versions = {
        LEGACY_TRIAL_IDENTITY_SCHEMA_VERSION,
        TRIAL_IDENTITY_SCHEMA_VERSION,
    }
    if schema_version not in supported_versions:
        raise ValueError(f"trial identity manifest schema_version must be one of {sorted(supported_versions)!r}")
    split = str(payload.get("dataset_split", "")).strip().lower()
    if split not in {"heldout", "validation", "test"}:
        raise ValueError("Stage-3 signal identity dataset_split must be heldout/validation/test")
    training_sessions = _nonempty_unique_strings(
        payload.get("training_session_uids"),
        "training_session_uids",
    )
    raw_trials = payload.get("trials")
    if not isinstance(raw_trials, list) or not raw_trials:
        raise ValueError("trial identity manifest requires a non-empty trials list")
    action_id: str | None = None
    handedness: str | None = None
    comparison_design: str | None = None
    comparison_set_uid: str | None = None
    model_taxonomy_id: str | None = None
    model_taxonomy_fingerprint: str | None = None
    runtime_model_hash: str | None = None
    actuator_schema_hash: str | None = None
    taxonomy_source_path: Path | None = None
    taxonomy_source_sha256: str | None = None
    taxonomy_ordered_actuators: tuple[Mapping[str, Any], ...] = ()
    if schema_version == TRIAL_IDENTITY_SCHEMA_VERSION:
        action_id = _identity_text(payload.get("action_id"), "action_id")
        handedness = str(payload.get("handedness", "")).strip().lower()
        if handedness not in {"right", "left"}:
            raise ValueError("trial identity handedness must be explicitly right or left")
        comparison_design = _identity_text(
            payload.get("comparison_design"),
            "comparison_design",
        )
        if comparison_design not in {
            PAIRED_EMG_COMPARISON_DESIGN,
            UNPAIRED_EMG_COMPARISON_DESIGN,
        }:
            raise ValueError("trial identity comparison_design is unsupported")
        comparison_set_uid = _identity_text(
            payload.get("comparison_set_uid"),
            "comparison_set_uid",
        )
        taxonomy_value = _identity_text(
            payload.get("model_taxonomy_path"),
            "model_taxonomy_path",
        )
        taxonomy_source_path = Path(taxonomy_value).expanduser()
        if not taxonomy_source_path.is_absolute():
            taxonomy_source_path = source_path.parent / taxonomy_source_path
        taxonomy_source_path = taxonomy_source_path.resolve(strict=True)
        from musclemimic.physiology import load_anatomical_taxonomy

        taxonomy = load_anatomical_taxonomy(taxonomy_source_path)
        model_taxonomy_id = taxonomy.taxonomy_id
        model_taxonomy_fingerprint = taxonomy.fingerprint
        runtime_model_hash = str(taxonomy.model_binding["runtime_model_hash"])
        actuator_schema_hash = str(taxonomy.model_binding["actuator_schema_hash"])
        taxonomy_source_sha256 = _file_sha256(taxonomy_source_path)
        taxonomy_ordered_actuators = tuple(taxonomy.ordered_actuators)

    identities: list[TrialIdentity] = []
    for row in raw_trials:
        if not isinstance(row, dict):
            raise ValueError("trial identity entries must be objects")
        try:
            feed_index = int(row["feed_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("trial identity feed_index must be a non-negative integer") from exc
        try:
            exact_feed_index = float(row["feed_index"]) == float(feed_index)
        except (KeyError, TypeError, ValueError):
            exact_feed_index = False
        if feed_index < 0 or isinstance(row.get("feed_index"), bool) or not exact_feed_index:
            raise ValueError("trial identity feed_index must be a non-negative integer")
        reference_trial_fingerprint = None
        if schema_version == TRIAL_IDENTITY_SCHEMA_VERSION:
            has_reference = "reference_trial_fingerprint" in row
            if comparison_design == PAIRED_EMG_COMPARISON_DESIGN:
                if not has_reference:
                    raise ValueError("paired Stage-3 identity rows require reference_trial_fingerprint")
                reference_trial_fingerprint = _require_sha256(
                    row.get("reference_trial_fingerprint"),
                    "reference_trial_fingerprint",
                )
            elif has_reference:
                raise ValueError("unpaired Stage-3 identity rows must not claim reference_trial_fingerprint")
        identities.append(
            TrialIdentity(
                feed_index=feed_index,
                feed_fingerprint=_require_sha256(row.get("feed_fingerprint"), "feed_fingerprint"),
                trial_uid=_identity_text(row.get("trial_uid"), "trial_uid"),
                subject_uid=_identity_text(row.get("subject_uid"), "subject_uid"),
                session_uid=_identity_text(row.get("session_uid"), "session_uid"),
                reference_trial_fingerprint=reference_trial_fingerprint,
            )
        )
    feed_indices = [identity.feed_index for identity in identities]
    trial_uids = [identity.trial_uid for identity in identities]
    feed_fingerprints = [identity.feed_fingerprint for identity in identities]
    if len(set(feed_indices)) != len(feed_indices):
        raise ValueError("trial identity manifest contains duplicate feed_index values")
    if len(set(trial_uids)) != len(trial_uids):
        raise ValueError("trial identity manifest contains duplicate trial_uid values")
    if len(set(feed_fingerprints)) != len(feed_fingerprints):
        raise ValueError("trial identity manifest contains duplicate feed fingerprints")
    heldout_sessions = {identity.session_uid for identity in identities}
    if heldout_sessions & set(training_sessions):
        raise ValueError("Stage-3 signal identity manifest leaks a held-out session into training_session_uids")
    if len(heldout_sessions) != 1:
        raise ValueError(
            "one Stage-3 signal NPZ must contain exactly one held-out session; "
            "export separate files for separate sessions"
        )
    unsigned = dict(payload)
    supplied = unsigned.pop("manifest_fingerprint", None)
    computed = _canonical_sha256(unsigned)
    if supplied is not None and supplied != computed:
        raise ValueError("trial identity manifest fingerprint is stale")
    return TrialIdentityManifest(
        schema_version=schema_version,
        dataset_split=split,
        training_session_uids=tuple(training_sessions),
        trials_by_feed={identity.feed_index: identity for identity in identities},
        manifest_fingerprint=computed,
        source_path=source_path,
        source_sha256=_file_sha256(source_path),
        action_id=action_id,
        handedness=handedness,
        comparison_design=comparison_design,
        comparison_set_uid=comparison_set_uid,
        model_taxonomy_id=model_taxonomy_id,
        model_taxonomy_fingerprint=model_taxonomy_fingerprint,
        runtime_model_hash=runtime_model_hash,
        actuator_schema_hash=actuator_schema_hash,
        taxonomy_source_path=taxonomy_source_path,
        taxonomy_source_sha256=taxonomy_source_sha256,
        taxonomy_ordered_actuators=taxonomy_ordered_actuators,
    )


def load_paired_policy_evidence(
    path: str | Path,
    *,
    stage3_checkpoint_payload_sha256: str | None = None,
) -> Stage3PolicyEvidence:
    """Load the sealed policy selected by the paired Stage-3 comparison.

    Signal capture supplies ``stage3_checkpoint_payload_sha256`` and therefore
    proves that the replayed high-level checkpoint is the selected one.  EMG
    and physiology consumers may omit it when they only need the already
    sealed low-level policy/basis/event bindings from the paired report.
    """

    from musclemimic.badminton.stage3_paired_comparison import validate_paired_comparison

    source_path = Path(path).expanduser().resolve(strict=True)
    report = validate_paired_comparison(source_path)
    if report.get("schema_version") != PAIRED_COMPARISON_SCHEMA_VERSION:
        raise ValueError("unsupported Stage-3 paired comparison schema for signal export")
    selected = report.get("selected_policy_for_emg")
    if not isinstance(selected, dict):
        raise ValueError("paired Stage-3 report has no selected policy evidence")
    selected_stage3 = _require_sha256(
        selected.get("stage3_checkpoint_payload_sha256"),
        "selected stage3 checkpoint payload",
    )
    actual_stage3 = (
        selected_stage3
        if stage3_checkpoint_payload_sha256 is None
        else _require_sha256(
            stage3_checkpoint_payload_sha256,
            "stage3_checkpoint_payload_sha256",
        )
    )
    if selected_stage3 != actual_stage3:
        raise ValueError("signal-export Stage-3 checkpoint differs from the paired selected policy")
    decoder_type = str(selected.get("policy_decoder_type", ""))
    if decoder_type not in {"direct", "fixed_synergy", "synergy_residual"}:
        raise ValueError("paired selected policy has an unsupported decoder type")
    return Stage3PolicyEvidence(
        family=_identity_text(selected.get("family"), "selected policy family"),
        decoder_type=decoder_type,
        policy_checkpoint_fingerprint=_require_sha256(
            selected.get("policy_checkpoint_fingerprint"),
            "policy_checkpoint_fingerprint",
        ),
        policy_promotion_fingerprint=_require_sha256(
            selected.get("policy_promotion_fingerprint"),
            "policy_promotion_fingerprint",
        ),
        formal_synergy_basis_fingerprint=_require_sha256(
            selected.get("formal_synergy_basis_fingerprint"),
            "formal_synergy_basis_fingerprint",
        ),
        event_reference_fingerprint=_require_sha256(
            selected.get("event_reference_fingerprint"),
            "event_reference_fingerprint",
        ),
        stage3_checkpoint_payload_sha256=selected_stage3,
        paired_comparison_fingerprint=_require_sha256(
            report.get("paired_comparison_fingerprint"),
            "paired_comparison_fingerprint",
        ),
        source_path=source_path,
        source_sha256=_file_sha256(source_path),
    )


def write_stage3_signal_export(
    output_npz: str | Path,
    arrays: Mapping[str, np.ndarray],
    *,
    collector: Stage3SignalCollector,
    sidecar_json: str | Path | None = None,
) -> dict[str, Any]:
    """Atomically persist the NPZ plus a content-addressed JSON sidecar."""

    output = Path(output_npz).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar = Path(sidecar_json).expanduser() if sidecar_json is not None else output.with_suffix(".manifest.json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    validated = {str(key): np.asarray(value) for key, value in arrays.items()}
    _validate_export_arrays(validated, require_fingerprint=True)
    expected_fingerprint = str(validated["signal_export_fingerprint"].reshape(-1)[0])
    if _arrays_fingerprint(validated) != expected_fingerprint:
        raise ValueError("Stage-3 signal arrays changed after finalization")
    if _file_sha256(collector.identities.source_path) != collector.identities.source_sha256:
        raise ValueError("Stage-3 trial identity manifest changed during signal collection")
    if (
        collector.identities.taxonomy_source_path is not None
        and _file_sha256(collector.identities.taxonomy_source_path) != collector.identities.taxonomy_source_sha256
    ):
        raise ValueError("Stage-3 model taxonomy changed during signal collection")

    temporary_npz = output.with_name(f".{output.stem}.{os.getpid()}.tmp.npz")
    temporary_sidecar = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.tmp")
    try:
        np.savez_compressed(temporary_npz, **validated)
        npz_sha256 = _file_sha256(temporary_npz)
        manifest: dict[str, Any] = {
            "schema_version": SIGNAL_EXPORT_MANIFEST_SCHEMA_VERSION,
            "signal_schema_version": SIGNAL_EXPORT_SCHEMA_VERSION,
            "signal_export_fingerprint": expected_fingerprint,
            "npz_sha256": npz_sha256,
            "npz_shape": {
                "trials": int(validated["muscle_activation"].shape[0]),
                "time_steps": int(validated["muscle_activation"].shape[1]),
                "muscle_channels": int(validated["muscle_activation"].shape[2]),
                "joint_channels": int(validated["joint_torque"].shape[2]),
            },
            "capture": {
                "source": "canonical_stage3_cpu_checkpoint_replay_post_step_transition",
                "impact_event": "first_real_hit_this_step",
                "trial_acceptance": "complete_window_and_no_body_fall_fail_closed",
                "control_dt_s": float(collector.control_dt_s),
                "sampling_rate_hz": float(1.0 / collector.control_dt_s),
                "pre_impact_steps": int(collector.pre_steps),
                "post_impact_steps": int(collector.post_steps),
            },
            "identity": {
                "identity_manifest_path": str(collector.identities.source_path),
                "identity_manifest_sha256": collector.identities.source_sha256,
                "identity_manifest_fingerprint": collector.identities.manifest_fingerprint,
                "dataset_split": collector.identities.dataset_split,
                "session_uid": str(validated["session_uid"][0]),
                "trial_uids": validated["trial_uid"].astype(str).tolist(),
                "feed_fingerprints": validated["feed_fingerprint"].astype(str).tolist(),
            },
            "policy_evidence": {
                "paired_comparison_path": str(collector.policy_evidence.source_path),
                "paired_comparison_sha256": collector.policy_evidence.source_sha256,
                "paired_comparison_fingerprint": collector.policy_evidence.paired_comparison_fingerprint,
                "family": collector.policy_evidence.family,
                "decoder_type": collector.policy_evidence.decoder_type,
                "policy_checkpoint_fingerprint": collector.policy_evidence.policy_checkpoint_fingerprint,
                "policy_promotion_fingerprint": collector.policy_evidence.policy_promotion_fingerprint,
                "formal_synergy_basis_fingerprint": (collector.policy_evidence.formal_synergy_basis_fingerprint),
                "event_reference_fingerprint": collector.policy_evidence.event_reference_fingerprint,
                "stage3_checkpoint_payload_sha256": collector.stage3_checkpoint_payload_sha256,
                "stage3_evaluation_binding_sha256": str(validated["stage3_evaluation_binding_sha256"].reshape(-1)[0]),
            },
            "array_fingerprints": {
                name: _array_sha256(value)
                for name, value in sorted(validated.items())
                if name != "signal_export_fingerprint"
            },
        }
        if collector.identities.is_emg_v2:
            manifest["identity"].update(
                {
                    "schema_version": collector.identities.schema_version,
                    "action_id": collector.identities.action_id,
                    "handedness": collector.identities.handedness,
                    "comparison_design": collector.identities.comparison_design,
                    "comparison_set_uid": collector.identities.comparison_set_uid,
                    "model_taxonomy_id": collector.identities.model_taxonomy_id,
                    "model_taxonomy_fingerprint": (collector.identities.model_taxonomy_fingerprint),
                    "runtime_model_hash": collector.identities.runtime_model_hash,
                    "scene_runtime_model_hash": str(validated["scene_runtime_model_hash"]),
                    "actuator_schema_hash": collector.identities.actuator_schema_hash,
                    "taxonomy_source_path": str(collector.identities.taxonomy_source_path),
                    "taxonomy_source_sha256": collector.identities.taxonomy_source_sha256,
                }
            )
            if collector.identities.comparison_design == PAIRED_EMG_COMPARISON_DESIGN:
                manifest["identity"]["reference_trial_fingerprints"] = (
                    validated["reference_trial_fingerprint"].astype(str).tolist()
                )
        manifest["manifest_fingerprint"] = _canonical_sha256(manifest)
        temporary_sidecar.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_npz, output)
        os.replace(temporary_sidecar, sidecar)
    finally:
        if temporary_npz.exists():
            temporary_npz.unlink()
        if temporary_sidecar.exists():
            temporary_sidecar.unlink()

    with np.load(output, allow_pickle=False) as stored:
        reloaded = {name: np.asarray(stored[name]) for name in stored.files}
    _validate_export_arrays(reloaded, require_fingerprint=True)
    if _file_sha256(output) != manifest["npz_sha256"] or _arrays_fingerprint(reloaded) != expected_fingerprint:
        raise RuntimeError("persisted Stage-3 signal export failed content verification")
    persisted_manifest = load_json_strict(sidecar)
    if not isinstance(persisted_manifest, dict):
        raise RuntimeError("persisted Stage-3 signal sidecar is not an object")
    unsigned = dict(persisted_manifest)
    supplied = unsigned.pop("manifest_fingerprint", None)
    if supplied != _canonical_sha256(unsigned):
        raise RuntimeError("persisted Stage-3 signal sidecar fingerprint mismatch")
    return persisted_manifest


def canonical_mapping_fingerprint(value: Mapping[str, Any]) -> str:
    """Public helper for binding the evaluation feed manifest exactly."""

    return _canonical_sha256(dict(value))


def _validate_policy_runtime_binding(
    evidence: Stage3PolicyEvidence,
    *,
    runtime: Any,
    event_reference_fingerprint: str,
    stage3_checkpoint_payload_sha256: str,
) -> dict[str, np.ndarray]:
    runtime_checkpoint = _require_sha256(
        getattr(runtime, "checkpoint_fingerprint", None),
        "runtime checkpoint fingerprint",
    )
    if runtime_checkpoint != evidence.policy_checkpoint_fingerprint:
        raise ValueError("Stage-3 LAB runtime differs from the paired selected low-level policy")
    decoder = str(getattr(runtime, "decoder_type", ""))
    if decoder != evidence.decoder_type:
        raise ValueError("Stage-3 LAB runtime decoder differs from the paired selected policy")
    if evidence.event_reference_fingerprint != _require_sha256(
        event_reference_fingerprint,
        "event_reference_fingerprint",
    ):
        raise ValueError("signal-export event reference differs from paired held-out Stage-3 evaluation")
    if evidence.stage3_checkpoint_payload_sha256 != _require_sha256(
        stage3_checkpoint_payload_sha256,
        "stage3_checkpoint_payload_sha256",
    ):
        raise ValueError("signal-export high-level Stage-3 policy differs from paired selection")
    basis = getattr(runtime, "synergy_basis", None)
    if decoder == "direct":
        if basis is not None:
            raise ValueError("direct Stage-3 runtime must not embed a synergy basis")
        return {}
    if basis is None:
        raise ValueError(f"{decoder} Stage-3 runtime is missing its embedded fixed synergy basis")
    runtime_fingerprint = _require_sha256(getattr(basis, "fingerprint", None), "runtime synergy basis")
    manifest = getattr(basis, "manifest", None)
    if not isinstance(manifest, Mapping):
        raise ValueError("runtime synergy basis is missing its source manifest")
    source_fingerprint = _require_sha256(
        manifest.get("source_fingerprint"),
        "runtime synergy basis source",
    )
    if source_fingerprint != evidence.formal_synergy_basis_fingerprint:
        raise ValueError("runtime synergy basis source differs from paired formal analysis basis")
    return {
        "runtime_synergy_basis_fingerprint": np.asarray(runtime_fingerprint),
        "runtime_synergy_basis_source_fingerprint": np.asarray(source_fingerprint),
    }


def _validate_identity_taxonomy_layout(
    identities: TrialIdentityManifest,
    layout: Stage3SignalLayout,
) -> None:
    """Bind a v2 EMG export to the audited base-model actuator taxonomy.

    The Stage-3 scene contains racket/shuttle additions, so its whole-model
    MuJoCo byte hash is intentionally different from the base MyoFullBody XML
    hash recorded by the taxonomy.  We instead verify every ordered body
    actuator field exposed by the signal layout.  Only then may the export
    carry the taxonomy's base-model binding hashes used by the EMG mapping.
    """

    if not identities.is_emg_v2:
        return
    if (
        identities.taxonomy_source_path is None
        or identities.taxonomy_source_sha256 is None
        or _file_sha256(identities.taxonomy_source_path) != identities.taxonomy_source_sha256
    ):
        raise ValueError("Stage-3 EMG model taxonomy changed before signal collection")
    _require_sha256(
        layout.scene_runtime_model_hash,
        "Stage-3 scene_runtime_model_hash",
    )
    rows = identities.taxonomy_ordered_actuators
    if len(rows) != len(layout.actuator_names):
        raise ValueError("Stage-3 body actuator width differs from the EMG model taxonomy")
    row_names = tuple(str(row["name"]) for row in rows)
    if row_names != layout.actuator_names:
        raise ValueError("Stage-3 ordered body actuators differ from the EMG model taxonomy")
    contract = layout.muscle_channel_contract
    checks = (
        (contract.actuator_ids, "actuator_id", "actuator id"),
        (contract.actuator_actadr, "actadr", "activation address"),
        (contract.actuator_actnum, "actnum", "activation count"),
    )
    for values, row_key, label in checks:
        expected = tuple(int(row[row_key]) for row in rows)
        actual = tuple(int(value) for value in values)
        if actual != expected:
            raise ValueError(f"Stage-3 ordered {label} differs from the EMG model taxonomy")
    expected_ctrlrange = np.asarray(
        [row["ctrlrange"] for row in rows],
        dtype=np.float64,
    )
    if not np.array_equal(layout.actuator_ctrlrange, expected_ctrlrange):
        raise ValueError("Stage-3 ordered actuator ctrlrange differs from the EMG model taxonomy")
    if int(contract.model_na) != len(rows):
        raise ValueError("Stage-3 activation-state width differs from the EMG model taxonomy")


def _validate_transition(
    transition: Mapping[str, Any],
    layout: Stage3SignalLayout,
) -> dict[str, Any]:
    required = {
        "teacher_ctrl_physical",
        "muscle_excitation",
        "muscle_activation",
        "joint_torque",
        "joint_angular_velocity",
        "step_index",
        "elapsed_time_s",
        "swing_phase",
        "event_state_code",
        "hit_this_step",
        "body_fall",
        "recovery_complete",
    }
    if missing := sorted(required - set(transition)):
        raise ValueError(f"Stage-3 physical transition is missing fields: {missing}")
    result = {key: transition[key] for key in required}
    muscle_width = len(layout.actuator_names)
    joint_width = len(layout.joint_names)
    for key, width in (
        ("teacher_ctrl_physical", muscle_width),
        ("muscle_excitation", muscle_width),
        ("muscle_activation", muscle_width),
        ("joint_torque", joint_width),
        ("joint_angular_velocity", joint_width),
    ):
        values = np.asarray(result[key], dtype=np.float64)
        if values.shape != (width,) or not np.all(np.isfinite(values)):
            raise ValueError(f"Stage-3 transition {key} must be finite [{width}]")
        result[key] = values.astype(np.float32)
    result["muscle_activation"] = validate_unit_muscle_activation(result["muscle_activation"])
    excitation = np.asarray(result["muscle_excitation"], dtype=np.float64)
    if np.any(excitation < -1e-6) or np.any(excitation > 1.0 + 1e-6):
        raise ValueError("Stage-3 transition muscle_excitation lies outside [0,1]")
    result["muscle_excitation"] = np.clip(excitation, 0.0, 1.0).astype(np.float32)
    for key in ("elapsed_time_s", "swing_phase"):
        value = float(result[key])
        if not np.isfinite(value):
            raise ValueError(f"Stage-3 transition {key} must be finite")
        result[key] = value
    result["step_index"] = int(result["step_index"])
    result["event_state_code"] = int(result["event_state_code"])
    if result["step_index"] <= 0 or result["event_state_code"] not in set(_EVENT_STATE_CODES.values()):
        raise ValueError("Stage-3 transition step/event state is invalid")
    for key in ("hit_this_step", "body_fall", "recovery_complete"):
        result[key] = bool(result[key])
    return result


def _derive_phase_arrays(
    transitions: Sequence[Mapping[str, Any]],
    *,
    impact_index: int,
) -> dict[str, np.ndarray]:
    count = len(transitions)
    if not 0 <= int(impact_index) < count:
        raise ValueError("impact index is outside Stage-3 signal trace")
    swing = np.asarray([record["swing_phase"] for record in transitions], dtype=np.float64)
    backswing_candidates = np.flatnonzero(swing > 1e-8)
    backswing = int(backswing_candidates[0]) if backswing_candidates.size else max(0, impact_index - 2)
    backswing = min(backswing, max(0, impact_index - 2))
    acceleration = max(backswing + 1, backswing + max(1, (impact_index - backswing) // 2))
    acceleration = min(acceleration, max(backswing + 1, impact_index - 1))
    follow_candidates = np.flatnonzero((np.arange(count) > impact_index) & (swing >= 1.0 - 1e-8))
    follow_end = int(follow_candidates[0]) if follow_candidates.size else count
    follow_end = min(count, max(impact_index + 1, follow_end))
    recovery_candidates = [
        index for index, record in enumerate(transitions) if index >= follow_end and bool(record["recovery_complete"])
    ]
    recovery_end = int(recovery_candidates[0]) if recovery_candidates else count - 1

    phase_id = np.full((count,), 5, dtype=np.int16)
    phase_id[:backswing] = 0
    phase_id[backswing:acceleration] = 1
    phase_id[acceleration:impact_index] = 2
    phase_id[impact_index : impact_index + 1] = 3
    phase_id[impact_index + 1 : follow_end] = 4
    phase_id[follow_end:] = 5
    phase_local = np.zeros((count,), dtype=np.float32)
    intervals = (
        (0, backswing),
        (backswing, acceleration),
        (acceleration, impact_index),
        (impact_index, impact_index + 1),
        (impact_index + 1, follow_end),
        (follow_end, recovery_end + 1),
    )
    for start, stop in intervals:
        if stop <= start:
            continue
        denominator = max(stop - start - 1, 1)
        phase_local[start:stop] = np.arange(stop - start, dtype=np.float32) / float(denominator)
    if recovery_end + 1 < count:
        phase_local[recovery_end + 1 :] = 1.0
    anchors = np.asarray(
        [0, backswing, acceleration, impact_index, follow_end, recovery_end],
        dtype=np.float64,
    )
    # Repeated/late boundaries are legal for short phases.  ``maximum.accumulate``
    # and a tiny stable offset keep interpolation deterministic and monotone.
    anchors = np.maximum.accumulate(anchors)
    anchors += np.arange(anchors.size, dtype=np.float64) * 1e-9
    global_phase = np.interp(
        np.arange(count, dtype=np.float64),
        anchors,
        np.linspace(0.0, 1.0, anchors.size),
    ).astype(np.float32)
    time_from = (
        np.asarray([record["elapsed_time_s"] for record in transitions], dtype=np.float64)
        - float(transitions[impact_index]["elapsed_time_s"])
    ).astype(np.float32)
    impact_flag = np.zeros((count,), dtype=bool)
    impact_flag[impact_index] = True
    return {
        "phase_global": global_phase,
        "phase_id": phase_id,
        "phase_local": phase_local,
        "time_to_impact_s": -time_from,
        "time_from_impact_s": time_from,
        "impact_flag": impact_flag,
    }


def _validate_export_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    require_fingerprint: bool = False,
) -> None:
    required = {
        "muscle_excitation",
        "muscle_activation",
        "teacher_ctrl_physical",
        "actuator_names",
        "activation_valid_mask",
        "joint_torque",
        "joint_angular_velocity",
        "joint_names",
        "phase_id",
        "impact_flag",
        "impact_frame",
        "trial_uid",
        "subject_uid",
        "session_uid",
        "dataset_split",
        "training_session_uid",
        "policy_decoder_type",
        "policy_checkpoint_fingerprint",
        "policy_promotion_fingerprint",
        "formal_synergy_basis_fingerprint",
        "analysis_synergy_basis_fingerprint",
        "event_reference_fingerprint",
    }
    if require_fingerprint:
        required.add("signal_export_fingerprint")
    if missing := sorted(required - set(arrays)):
        raise ValueError(f"Stage-3 signal export is missing arrays: {missing}")
    excitation = np.asarray(arrays["muscle_excitation"])
    activation = np.asarray(arrays["muscle_activation"])
    if excitation.ndim != 3 or excitation.shape != activation.shape or min(excitation.shape) <= 0:
        raise ValueError("Stage-3 muscle excitation/activation must be same-shape [trial,time,muscle]")
    if not np.all(np.isfinite(excitation)) or not np.all(np.isfinite(activation)):
        raise ValueError("Stage-3 muscle signals contain non-finite values")
    if np.any(excitation < -1e-6) or np.any(excitation > 1.0 + 1e-6):
        raise ValueError("Stage-3 muscle excitation lies outside [0,1]")
    validate_unit_muscle_activation(activation)
    names = np.asarray(arrays["actuator_names"])
    mask = np.asarray(arrays["activation_valid_mask"])
    if names.shape != (excitation.shape[2],) or names.dtype.kind not in {"U", "S"}:
        raise ValueError("Stage-3 actuator_names do not match muscle signal width")
    if mask.shape != names.shape or mask.dtype.kind != "b" or not np.all(mask):
        raise ValueError("Stage-3 physiology export requires every activation channel to be valid")
    joint_torque = np.asarray(arrays["joint_torque"])
    joint_velocity = np.asarray(arrays["joint_angular_velocity"])
    if joint_torque.ndim != 3 or joint_torque.shape != joint_velocity.shape:
        raise ValueError("Stage-3 joint torque/velocity must be same-shape [trial,time,joint]")
    if joint_torque.shape[:2] != excitation.shape[:2] or not np.all(np.isfinite(joint_torque)):
        raise ValueError("Stage-3 joint signals do not align with muscle trial/time dimensions")
    joint_names = np.asarray(arrays["joint_names"])
    if joint_names.shape != (joint_torque.shape[2],) or joint_names.dtype.kind not in {"U", "S"}:
        raise ValueError("Stage-3 joint_names do not match joint signal width")
    impact = np.asarray(arrays["impact_frame"])
    flag = np.asarray(arrays["impact_flag"])
    if impact.shape != (excitation.shape[0],) or flag.shape != excitation.shape[:2] or flag.dtype.kind != "b":
        raise ValueError("Stage-3 impact_frame/impact_flag dimensions are invalid")
    for trial, frame in enumerate(impact.astype(int).tolist()):
        if frame < 0 or frame >= flag.shape[1] or np.flatnonzero(flag[trial]).tolist() != [frame]:
            raise ValueError("Stage-3 signal trial must contain exactly one named impact frame")
    for key in ("trial_uid", "subject_uid", "session_uid"):
        value = np.asarray(arrays[key])
        if value.shape != (excitation.shape[0],) or value.dtype.kind not in {"U", "S"}:
            raise ValueError(f"Stage-3 {key} must contain one string per trial")
    if len(set(np.asarray(arrays["trial_uid"]).astype(str).tolist())) != excitation.shape[0]:
        raise ValueError("Stage-3 trial_uid values must be unique")
    if "trial_identity_schema_version" in arrays:
        identity_schema = _array_identity_scalar(
            arrays["trial_identity_schema_version"],
            "trial_identity_schema_version",
        )
        if identity_schema != TRIAL_IDENTITY_SCHEMA_VERSION:
            raise ValueError("Stage-3 signal export has an unsupported v2 identity schema")
        v2_fields = {
            "action_id",
            "handedness",
            "comparison_design",
            "comparison_set_uid",
            "model_taxonomy_id",
            "model_taxonomy_fingerprint",
            "runtime_model_hash",
            "actuator_schema_hash",
            "taxonomy_source_sha256",
            "scene_runtime_model_hash",
        }
        if missing := sorted(v2_fields - set(arrays)):
            raise ValueError(f"Stage-3 EMG-v2 identity is missing arrays: {missing}")
        for field in ("action_id", "comparison_set_uid", "model_taxonomy_id"):
            _array_identity_scalar(arrays[field], field)
        handedness = _array_identity_scalar(arrays["handedness"], "handedness")
        if handedness not in {"right", "left"}:
            raise ValueError("Stage-3 handedness must be explicitly right or left")
        for field in (
            "model_taxonomy_fingerprint",
            "runtime_model_hash",
            "actuator_schema_hash",
            "taxonomy_source_sha256",
            "scene_runtime_model_hash",
        ):
            _require_sha256(_array_identity_scalar(arrays[field], field), field)
        design = _array_identity_scalar(arrays["comparison_design"], "comparison_design")
        if design == PAIRED_EMG_COMPARISON_DESIGN:
            if "reference_trial_fingerprint" not in arrays:
                raise ValueError("paired Stage-3 signal export requires reference_trial_fingerprint")
            references = np.asarray(arrays["reference_trial_fingerprint"])
            if references.shape != (excitation.shape[0],) or references.dtype.kind not in {
                "U",
                "S",
            }:
                raise ValueError("Stage-3 reference_trial_fingerprint must contain one string per trial")
            for value in references.astype(str).tolist():
                _require_sha256(value, "reference_trial_fingerprint")
        elif design == UNPAIRED_EMG_COMPARISON_DESIGN:
            if "reference_trial_fingerprint" in arrays:
                raise ValueError("unpaired Stage-3 signal export must not claim reference_trial_fingerprint")
        else:
            raise ValueError("Stage-3 comparison_design is unsupported")


def _seconds_to_steps(seconds: float, control_dt_s: float, field: str) -> int:
    value = _positive_float(seconds, field)
    ratio = value / float(control_dt_s)
    steps = round(ratio)
    if steps <= 0 or not np.isclose(ratio, float(steps), atol=1e-8, rtol=0.0):
        raise ValueError(f"{field} must be a positive integer multiple of control_dt_s")
    return steps


def _positive_float(value: Any, field: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return result


def _identity_text(value: Any, field: str) -> str:
    result = str(value).strip()
    if not result or result.lower() == "none":
        raise ValueError(f"{field} must be a non-empty stable identity")
    return result


def _array_identity_scalar(value: Any, field: str) -> str:
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"Stage-3 {field} must be one scalar string")
    return _identity_text(array.reshape(-1)[0], field)


def _nonempty_unique_strings(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a non-empty list")
    result = [_identity_text(item, field) for item in value]
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{field} must be non-empty and unique")
    return result


def _require_sha256(value: Any, field: str) -> str:
    result = str(value)
    if len(result) != 64 or any(character not in _SHA256_CHARS for character in result):
        raise ValueError(f"{field} must be a lowercase SHA-256 fingerprint")
    return result


def _string_array(values: Sequence[str]) -> np.ndarray:
    strings = [str(value) for value in values]
    width = max(1, *(len(value) for value in strings))
    return np.asarray(strings, dtype=f"<U{width}")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + np.ascontiguousarray(array).tobytes()).hexdigest()


def _arrays_fingerprint(arrays: Mapping[str, np.ndarray]) -> str:
    return _canonical_sha256(
        {
            name: _array_sha256(np.asarray(value))
            for name, value in sorted(arrays.items())
            if name != "signal_export_fingerprint"
        }
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

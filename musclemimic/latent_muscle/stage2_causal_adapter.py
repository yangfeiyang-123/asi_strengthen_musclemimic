"""Exact-restart CPU adapter for Stage-2 latent intervention rollouts.

The adapter consumes physical teacher rows collected with
``mujoco_mjx_pre_transition_state_v1``.  It injects the recorded numeric
integration state into the matching LocoMujoco CPU trajectory, reconstructs
the trajectory carry, and accepts the state only when the live filtered
student observation reproduces the recorded ``student_obs``.  Missing state
fields or a cross-backend mismatch fail closed with a re-collection message;
there is no frame-reset or nearest-state fallback.

After acceptance, paired baseline/perturbed rollouts use a byte-stable snapshot
of MuJoCo ``mjSTATE_INTEGRATION`` plus the complete Loco carry, observation, and
info dictionary.  The supplied posterior latent (or its perturbation) is used
for exactly the first action; later actions use the frozen conditional-prior
mean.  This implements an impulse intervention at fixed ``(s_t, phase_t)``.

Stage-2 has no simulated shuttle contact or flight.  Accordingly
``impact_outcome`` and ``landing_outcome`` are explicitly unavailable and
represented by empty vectors.  They are excluded from the sealed causal-effect
tensor by the driver/artifact availability contract rather than filled with
fake zeros or reference-tracking proxies.
"""

from __future__ import annotations

import hashlib
import json
import random
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import mujoco
import numpy as np
from flax import serialization
from omegaconf import OmegaConf

from fullbody.latent_closed_loop_eval import _make_cpu_env
from musclemimic.algorithms.common.env_utils import apply_policy_interface_wrappers
from musclemimic.distill.body_obs_schema import build_body_obs_schema
from musclemimic.distill.collect_teacher import (
    SIMULATOR_PRE_STATE_FIELDS,
    SIMULATOR_PRE_STATE_SCHEMA_VERSION,
    _build_physical_capture_spec,
    _capture_physical_transition,
    _resolve_actuator_ctrlrange,
    _resolve_actuator_names,
    _student_state_schema,
)
from musclemimic.distill.config_overrides import apply_collection_overrides
from musclemimic.distill.dataset import SequenceDistillDataset
from musclemimic.distill.motion_identity import MotionIdentityMap
from musclemimic.distill.obs_filter import (
    build_student_obs_indices,
    filter_student_obs,
)
from musclemimic.distill.physical import physical_signal_metadata
from musclemimic.distill.provenance import (
    canonical_json_sha256,
    checkpoint_content_fingerprint,
    file_sha256,
    validate_dataset_manifest,
)
from musclemimic.latent_muscle.analysis_export import (
    ANALYSIS_INPUT_SCHEMA_VERSION,
    stable_sample_uids,
)
from musclemimic.latent_muscle.causal_rollout_artifact import REQUIRED_OUTCOMES
from musclemimic.latent_muscle.causal_rollout_driver import (
    ADAPTER_SCHEMA_VERSION,
    RolloutRequest,
)
from musclemimic.latent_muscle.runtime import load_latent_runtime
from musclemimic.runner.eval_utils import apply_temporal_params, load_checkpoint

STAGE2_ADAPTER_SCHEMA_VERSION = "stage2_cpu_causal_adapter_v1"
SNAPSHOT_SCHEMA_VERSION = "stage2_cpu_exact_snapshot_v1"
_SNAPSHOT_MAGIC = b"MM-ST2-SNAPSHOT\x01"
_HEX = frozenset("0123456789abcdef")
_RECOLLECT = (
    "re-collect the physical train/validation rows with the current "
    "fullbody.distill_collect --save-physical-muscle-state pipeline; it emits "
    "CPU-compatible causal replay state fields automatically"
)


@dataclass
class _Source:
    name: str
    dataset: SequenceDistillDataset
    dataset_manifest: dict[str, Any]
    motion_map: MotionIdentityMap
    env: Any
    base_env: Any
    student_spec: Any
    physical_spec: dict[str, Any]
    model_fingerprint: str
    uid_to_row: dict[str, int]


def create_adapter(config: Mapping[str, Any]) -> Stage2CausalRolloutAdapter:
    """Factory used by :mod:`causal_rollout_driver` job files."""

    return Stage2CausalRolloutAdapter(config)


class Stage2CausalRolloutAdapter:
    """Real LocoMujoco CPU evaluator implementing ``CausalRolloutAdapter``."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = _validated_config(config)
        self.horizon = int(self.config["rollout_horizon_steps"])
        self.state_match_atol = float(self.config["state_match_atol"])
        self.runtime = load_latent_runtime(self.config["latent_checkpoint"])
        self._analysis_latents, analysis_sidecar = _load_analysis_bindings(
            self.config["analysis_inputs"], self.config["analysis_manifest"]
        )
        if analysis_sidecar["checkpoint_fingerprint"] != self.runtime.checkpoint_fingerprint:
            raise ValueError("Stage-2 causal adapter latent checkpoint differs from analysis inputs")

        teacher_config, _teacher_state, _teacher_metadata = load_checkpoint(self.config["teacher_ckpt"])
        OmegaConf.set_struct(teacher_config, False)
        apply_temporal_params(teacher_config)
        teacher_fingerprint = checkpoint_content_fingerprint(self.config["teacher_ckpt"])
        expected_teacher = (self.runtime.training_provenance or {}).get("teacher_checkpoint") or {}
        if expected_teacher.get("sha256") != teacher_fingerprint["sha256"]:
            raise ValueError("Stage-2 causal adapter teacher differs from latent training provenance")

        self._sources = _build_sources(
            config=self.config,
            teacher_config=teacher_config,
            runtime=self.runtime,
        )
        self._uid_source: dict[str, tuple[int, int]] = {}
        for source_index, source in enumerate(self._sources):
            for uid, row in source.uid_to_row.items():
                if uid in self._uid_source:
                    raise ValueError(f"causal sample UID occurs in multiple dataset sources: {uid}")
                self._uid_source[uid] = (source_index, row)
        missing = sorted(set(self._analysis_latents) - set(self._uid_source))
        extra = sorted(set(self._uid_source) & set(self._analysis_latents), key=str)
        if missing:
            raise ValueError(f"analysis sample UIDs are absent from the bound physical datasets: {missing[:5]}")
        if len(extra) != len(self._analysis_latents):
            raise RuntimeError("internal analysis/dataset UID binding invariant failed")

        self._active_source_index: int | None = None
        self._active_uid: str | None = None
        self._seed: int | None = None
        self._rng = np.random.default_rng(0)
        first = self._sources[0]
        self._muscle_names = [str(name) for name in self.runtime.body_actuator_names]
        per_actuator_activation_valid = [
            bool(value) for value in first.physical_spec["metadata"]["activation_valid_mask"]
        ]
        if not all(per_actuator_activation_valid):
            invalid = [
                name
                for name, valid in zip(
                    self._muscle_names,
                    per_actuator_activation_valid,
                    strict=True,
                )
                if not valid
            ]
            raise ValueError(f"Stage-2 causal activation channels are not all scalar MuJoCo states: {invalid}")
        # Outcome vectors are time-major [step, actuator] traces.  The generic
        # evidence ABI requires one validity bit per flattened outcome feature,
        # not one bit per unrolled actuator name.
        self._activation_valid = _time_major_activation_valid_mask(
            per_actuator_activation_valid,
            horizon=self.horizon,
        )
        self._qpos_names, self._qvel_names = _joint_coordinate_names(first.base_env._model)
        for source in self._sources[1:]:
            qpos_names, qvel_names = _joint_coordinate_names(source.base_env._model)
            source_activation_valid = [
                bool(value) for value in source.physical_spec["metadata"]["activation_valid_mask"]
            ]
            if (
                qpos_names != self._qpos_names
                or qvel_names != self._qvel_names
                or source_activation_valid != per_actuator_activation_valid
            ):
                raise ValueError("Stage-2 CPU sources expose different physical outcome ABIs")
        (
            self._trunk_qpos,
            self._trunk_qvel,
            self._trunk_names,
            self._trunk_units,
        ) = _trunk_state_spec(first.base_env)
        for source in self._sources[1:]:
            qpos, qvel, names, units = _trunk_state_spec(source.base_env)
            if (
                not np.array_equal(qpos, self._trunk_qpos)
                or not np.array_equal(qvel, self._trunk_qvel)
                or names != self._trunk_names
                or units != self._trunk_units
            ):
                raise ValueError("Stage-2 CPU sources expose different trunk-state ABIs")
        # Stage-2 event annotations describe the reference motion, not a
        # simulated racket/shuttle contact.  Treating them as an observed task
        # impact would turn a tracking target into a fabricated causal outcome.
        self._impact_available = False
        self._availability = {
            "muscle_excitation": True,
            "muscle_activation": True,
            "joint_position": True,
            "joint_velocity": True,
            "trunk_state": True,
            "racket_state": True,
            "impact_outcome": self._impact_available,
            # A Stage-2 reference-tracking environment contains no shuttle
            # flight or court landing state.  Do not synthesize one.
            "landing_outcome": False,
        }
        self._outcome_schemas = _build_outcome_schemas(
            horizon=self.horizon,
            muscle_names=self._muscle_names,
            qpos_names=self._qpos_names,
            qvel_names=self._qvel_names,
            trunk_names=self._trunk_names,
            trunk_units=self._trunk_units,
        )
        environment_payload = {
            "schema_version": STAGE2_ADAPTER_SCHEMA_VERSION,
            "teacher_checkpoint": teacher_fingerprint,
            "datasets": [
                {
                    "name": source.name,
                    "manifest_fingerprint": source.dataset_manifest["manifest_fingerprint"],
                    "motion_identity": source.motion_map.to_manifest(),
                    "model_fingerprint": source.model_fingerprint,
                }
                for source in self._sources
            ],
            "rollout_horizon_steps": self.horizon,
            "state_match_atol": self.state_match_atol,
            "intervention_steps": 1,
            "terminal_padding": "hold_last_measured_diagnostic_state",
            "outcome_availability": self._availability,
        }
        self._descriptor = {
            "schema_version": ADAPTER_SCHEMA_VERSION,
            "checkpoint_fingerprint": self.runtime.checkpoint_fingerprint,
            "synergy_basis_fingerprint": analysis_sidecar["formal_synergy_basis_fingerprint"],
            "environment_fingerprint": canonical_json_sha256(environment_payload),
            "policy_abi_hash": canonical_json_sha256(self.runtime.control_manifest),
            "rollout_engine": "loco_mujoco_cpu_stage2_exact_snapshot_v1",
            "physical_signal_semantics": physical_signal_metadata(),
            "activation_valid_mask": self._activation_valid,
            "outcome_schemas": self._outcome_schemas,
            "outcome_availability": self._availability,
            "stage2_diagnostic_outcomes_complete": True,
            "task_outcomes_complete": False,
            "adapter_details": environment_payload,
            "limitations": {
                "landing_outcome": "unavailable: Stage-2 has no simulated shuttle or landing",
                "impact_outcome": "unavailable: Stage-2 has no simulated racket/shuttle contact",
            },
        }

    def descriptor(self) -> Mapping[str, Any]:
        return dict(self._descriptor)

    def prepare_analysis_sample(self, sample_uid: str) -> bytes:
        uid = str(sample_uid)
        if uid not in self._uid_source:
            raise ValueError(f"physical datasets do not uniquely contain sample UID {uid!r}")
        source_index, row = self._uid_source[uid]
        source = self._sources[source_index]
        self._active_source_index = source_index
        self._active_uid = uid
        _inject_dataset_state(
            source,
            row=row,
            state_match_atol=self.state_match_atol,
        )
        snapshot = self.capture_snapshot()
        # A serializer that changes without an environment step cannot establish
        # the byte-exact restoration contract used by the generic driver.
        if snapshot != self.capture_snapshot():
            raise ValueError("Stage-2 CPU snapshot serialization is not deterministic")
        return snapshot

    def snapshot_to_bytes(self, snapshot: Any) -> bytes:
        if not isinstance(snapshot, bytes) or not snapshot.startswith(_SNAPSHOT_MAGIC):
            raise ValueError("Stage-2 snapshot must be canonical adapter bytes")
        _decode_snapshot(snapshot)
        return snapshot

    def restore_snapshot(self, snapshot: Any) -> None:
        payload = self.snapshot_to_bytes(snapshot)
        header, parts = _decode_snapshot(payload)
        source_index = int(header["source_index"])
        if source_index < 0 or source_index >= len(self._sources):
            raise ValueError("snapshot source index is outside the configured CPU environments")
        source = self._sources[source_index]
        if header["model_fingerprint"] != source.model_fingerprint:
            raise ValueError("snapshot model fingerprint differs from the configured CPU environment")
        if _model_fingerprint(source.base_env._model) != source.model_fingerprint:
            raise ValueError("CPU model mutated between paired rollouts; exact restore is unsupported")
        state = np.frombuffer(parts["mujoco_state"], dtype="<f8").astype(np.float64, copy=True)
        expected = int(mujoco.mj_stateSize(source.base_env._model, mujoco.mjtState.mjSTATE_INTEGRATION))
        if state.shape != (expected,):
            raise ValueError("snapshot MuJoCo integration-state width is invalid")
        mujoco.mj_setState(
            source.base_env._model,
            source.base_env._data,
            state,
            mujoco.mjtState.mjSTATE_INTEGRATION,
        )
        mujoco.mj_forward(source.base_env._model, source.base_env._data)
        source.base_env._additional_carry = serialization.from_bytes(
            source.base_env._additional_carry,
            parts["carry"],
        )
        source.base_env._obs = _decode_array(parts["observation"])
        source.base_env._info = serialization.msgpack_restore(parts["info"])
        self._active_source_index = source_index
        self._active_uid = str(header["sample_uid"])
        self._seed = None

    def capture_snapshot(self) -> bytes:
        source = self._active_source()
        model = source.base_env._model
        state = np.empty(
            int(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION)),
            dtype=np.float64,
        )
        mujoco.mj_getState(
            model,
            source.base_env._data,
            state,
            mujoco.mjtState.mjSTATE_INTEGRATION,
        )
        parts = {
            "mujoco_state": np.asarray(state, dtype="<f8").tobytes(order="C"),
            "carry": serialization.to_bytes(source.base_env._additional_carry),
            "observation": _encode_array(np.asarray(source.base_env._obs)),
            "info": serialization.msgpack_serialize(_canonical_tree(source.base_env._info)),
        }
        return _encode_snapshot(
            {
                "schema_version": SNAPSHOT_SCHEMA_VERSION,
                "source_index": int(self._active_source_index),
                "sample_uid": str(self._active_uid),
                "model_fingerprint": source.model_fingerprint,
                "state_spec": "mjSTATE_INTEGRATION",
            },
            parts,
        )

    def set_common_random_seed(self, seed: int) -> None:
        value = int(seed)
        if value < 0:
            raise ValueError("common random seed must be non-negative")
        random.seed(value)
        np.random.seed(value & 0xFFFFFFFF)
        self._rng = np.random.default_rng(value)
        source = self._active_source()
        carry = source.base_env._additional_carry
        if not hasattr(carry, "key") or not hasattr(carry, "replace"):
            raise ValueError("CPU Loco carry has no explicit JAX RNG key")
        source.base_env._additional_carry = carry.replace(key=jax.random.PRNGKey(value & 0xFFFFFFFF))
        self._seed = value

    def random_state_fingerprint(self) -> str:
        if self._seed is None:
            raise RuntimeError("set_common_random_seed must precede RNG fingerprinting")
        carry_key = np.asarray(self._active_source().base_env._additional_carry.key)
        numpy_state = np.random.get_state()
        payload = {
            "seed": self._seed,
            "python_random": repr(random.getstate()),
            "numpy_random": {
                "algorithm": str(numpy_state[0]),
                "state": np.asarray(numpy_state[1], dtype=np.uint32).tolist(),
                "position": int(numpy_state[2]),
                "has_gauss": int(numpy_state[3]),
                "cached_gaussian": float(numpy_state[4]),
            },
            "adapter_generator": _canonical_tree(self._rng.bit_generator.state),
            "loco_carry_key": carry_key.astype(np.uint32).tolist(),
        }
        return canonical_json_sha256(payload)

    def evaluate_rollout(self, request: RolloutRequest) -> Mapping[str, Any]:
        if self._seed is None or self._active_uid != request.sample_uid:
            raise RuntimeError("rollout must follow exact restore, common seeding, and matching sample preparation")
        expected_latent = self._analysis_latents[request.sample_uid]
        if not np.array_equal(np.asarray(request.baseline_latent, dtype=np.float32), expected_latent):
            raise ValueError("rollout request baseline latent differs from analysis inputs")
        source = self._active_source()
        env = source.env
        obs = _policy_observation(source)
        traces: dict[str, list[np.ndarray]] = {
            name: []
            for name in (
                "muscle_excitation",
                "muscle_activation",
                "joint_position",
                "joint_velocity",
                "trunk_state",
                "racket_state",
            )
        }
        terminal = False
        for step in range(self.horizon):
            student = np.asarray(filter_student_obs(obs, source.student_spec), dtype=np.float32)
            state = student[None, :] if student.ndim == 1 else student
            if step == 0:
                latent = np.asarray(request.evaluated_latent, dtype=np.float32)
                latent = latent[None, :] if latent.ndim == 1 else latent
            else:
                latent, _raw_sigma = self.runtime.prior_raw_numpy(state)
            action = np.asarray(self.runtime.decoder_numpy(state, latent), dtype=np.float32)
            if not np.all(np.isfinite(action)) or np.any(np.abs(action) > 1.0 + 1e-5):
                raise ValueError("latent decoder produced an invalid normalized CPU action")
            action = np.clip(action, -1.0, 1.0)
            obs, _reward, absorbing, done, _info = env.step(action)
            physical = {
                key: np.asarray(value)
                for key, value in _capture_physical_transition(source.base_env._data, source.physical_spec).items()
            }
            traces["muscle_excitation"].append(np.asarray(physical["muscle_excitation"], dtype=np.float32))
            traces["muscle_activation"].append(np.asarray(physical["muscle_activation"], dtype=np.float32))
            traces["joint_position"].append(np.asarray(source.base_env._data.qpos, dtype=np.float32).copy())
            traces["joint_velocity"].append(np.asarray(source.base_env._data.qvel, dtype=np.float32).copy())
            traces["trunk_state"].append(
                _trunk_state(
                    source,
                    qpos_indices=self._trunk_qpos,
                    qvel_indices=self._trunk_qvel,
                )
            )
            traces["racket_state"].append(_racket_state(physical))
            terminal = bool(np.asarray(done).reshape(-1)[0]) or bool(np.asarray(absorbing).reshape(-1)[0])
            if terminal:
                break
        valid_steps = len(traces["joint_position"])
        if valid_steps == 0:
            raise RuntimeError("CPU causal rollout produced no transition")
        for values in traces.values():
            while len(values) < self.horizon:
                values.append(values[-1].copy())
        result = {name: np.stack(values, axis=0).reshape(-1).astype(np.float32) for name, values in traces.items()}
        result["impact_outcome"] = np.empty((0,), dtype=np.float32)
        result["landing_outcome"] = np.empty((0,), dtype=np.float32)
        return result

    def _active_source(self) -> _Source:
        if self._active_source_index is None:
            raise RuntimeError("prepare_analysis_sample must select a CPU source first")
        return self._sources[self._active_source_index]


def _validated_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("Stage-2 causal adapter config must be an object")
    required = {
        "latent_checkpoint",
        "teacher_ckpt",
        "dataset_dir",
        "analysis_inputs",
        "analysis_manifest",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Stage-2 causal adapter config is missing {missing}")
    allowed = required | {"val_dataset_dir", "rollout_horizon_steps", "state_match_atol"}
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"Stage-2 causal adapter config has unsupported keys: {unknown}")
    result = dict(config)
    for key in required:
        if not str(result[key]).strip():
            raise ValueError(f"Stage-2 causal adapter config {key} must be non-empty")
    result["val_dataset_dir"] = None if result.get("val_dataset_dir") in (None, "") else str(result["val_dataset_dir"])
    result["rollout_horizon_steps"] = int(result.get("rollout_horizon_steps", 120))
    result["state_match_atol"] = float(result.get("state_match_atol", 1e-5))
    if result["rollout_horizon_steps"] <= 0:
        raise ValueError("rollout_horizon_steps must be positive")
    if (
        not np.isfinite(result["state_match_atol"])
        or result["state_match_atol"] < 0.0
        or result["state_match_atol"] > 1e-5
    ):
        raise ValueError("state_match_atol must be finite and within the strict [0, 1e-5] replay gate")
    return result


def _time_major_activation_valid_mask(
    per_actuator: Sequence[bool],
    *,
    horizon: int,
) -> list[bool]:
    if int(horizon) <= 0 or not per_actuator or any(type(value) is not bool for value in per_actuator):
        raise ValueError("activation validity requires booleans and a positive horizon")
    return list(per_actuator) * int(horizon)


def _load_analysis_bindings(inputs: str, manifest: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    input_path = Path(inputs)
    manifest_path = Path(manifest)
    if not input_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Stage-2 causal analysis inputs/manifest are missing")
    sidecar = json.loads(manifest_path.read_text(encoding="utf-8"))
    content = {key: value for key, value in sidecar.items() if key != "manifest_fingerprint"}
    if sidecar.get("manifest_fingerprint") != canonical_json_sha256(content):
        raise ValueError("Stage-2 causal analysis manifest fingerprint mismatch")
    if sidecar.get("schema_version") != ANALYSIS_INPUT_SCHEMA_VERSION:
        raise ValueError("Stage-2 causal adapter requires analysis_inputs_v2")
    if sidecar.get("npz_sha256") != file_sha256(input_path):
        raise ValueError("Stage-2 causal analysis NPZ differs from its manifest")
    for key in ("checkpoint_fingerprint", "formal_synergy_basis_fingerprint"):
        value = str(sidecar.get(key, ""))
        if len(value) != 64 or any(char not in _HEX for char in value):
            raise ValueError(f"analysis manifest {key} must be lowercase 64-hex")
    with np.load(input_path, allow_pickle=False) as data:
        uids = np.asarray(data["sample_uids"]).astype(str)
        latents = np.asarray(data["latents"], dtype=np.float32)
    if (
        uids.ndim != 1
        or uids.shape[0] == 0
        or len(set(uids.tolist())) != len(uids)
        or any(not uid for uid in uids.tolist())
        or latents.ndim != 2
        or latents.shape[0] != len(uids)
        or not np.all(np.isfinite(latents))
    ):
        raise ValueError("Stage-2 causal analysis UID/latent arrays are malformed")
    return {uid: latents[index] for index, uid in enumerate(uids.tolist())}, sidecar


def _build_sources(*, config: dict[str, Any], teacher_config: Any, runtime: Any) -> list[_Source]:
    roots: list[tuple[str, Path, str, str | None]] = [("train", Path(config["dataset_dir"]), "train", "dataset")]
    val_root = None if config["val_dataset_dir"] is None else Path(config["val_dataset_dir"])
    if val_root is not None:
        roots.append(("val", val_root, "val", "validation"))
    elif sorted(Path(config["dataset_dir"]).glob("val_*.npz")):
        roots.append(("val", Path(config["dataset_dir"]), "val", "dataset"))
    sources: list[_Source] = []
    seen_root_manifest: dict[Path, dict[str, Any]] = {}
    for name, root, split, provenance_kind in roots:
        resolved = root.resolve()
        manifest = seen_root_manifest.get(resolved)
        if manifest is None:
            manifest = validate_dataset_manifest(root)
            seen_root_manifest[resolved] = manifest
        expected_key = (
            "validation_dataset_manifest_fingerprint"
            if provenance_kind == "validation"
            else "dataset_manifest_fingerprint"
        )
        expected = (runtime.training_provenance or {}).get(expected_key)
        if expected is None:
            raise ValueError(f"latent training provenance does not bind the {name} causal dataset")
        if manifest["manifest_fingerprint"] != expected:
            raise ValueError(f"{name} causal dataset differs from latent training provenance")
        dataset = SequenceDistillDataset(
            root,
            split=split,
            seed=int(runtime.config.get("seed", 0)),
            target_actuator_names=runtime.body_actuator_names,
            require_stable_ids=True,
        )
        _validate_simulator_state_dataset(dataset, name=name)
        identity = dataset.metadata.get("motion_identity")
        if not isinstance(identity, Mapping):
            raise ValueError(f"{name} physical dataset lacks stable motion_identity")
        motion_map = MotionIdentityMap.from_paths(identity.get("motion_paths", ()))
        if [int(value) for value in motion_map.motion_uids] != [
            int(value) for value in identity.get("motion_uids", ())
        ]:
            raise ValueError(f"{name} physical dataset motion identity is inconsistent")
        _validate_row_motion_mapping(dataset, motion_map, name=name)
        source_config = OmegaConf.create(OmegaConf.to_container(teacher_config, resolve=True))
        OmegaConf.set_struct(source_config, False)
        apply_collection_overrides(source_config, motion_path=list(motion_map.motion_paths))
        source_config.experiment.env_params["headless"] = True
        source_config.experiment.env_params.pop("num_envs", None)
        env = _make_cpu_env(source_config)
        policy_env = apply_policy_interface_wrappers(env, source_config.experiment, include_student=False)
        spec = build_student_obs_indices(policy_env, dict(runtime.state_schema.get("student_obs_filter", {})))
        live_state = _student_state_schema(
            spec,
            dict(runtime.state_schema.get("student_obs_filter", {})),
            {"teacher_ckpt": config["teacher_ckpt"]},
            env=policy_env,
        )
        if live_state["schema_hash"] != runtime.schema_hash:
            raise ValueError(f"{name} CPU state schema differs from latent checkpoint")
        live_names = _resolve_actuator_names(policy_env, None)
        if live_names != list(runtime.body_actuator_names):
            raise ValueError(f"{name} CPU actuator order differs from latent checkpoint")
        live_ctrlrange = _resolve_actuator_ctrlrange(policy_env, live_names)
        if runtime.body_ctrlrange is None or not np.array_equal(live_ctrlrange, runtime.body_ctrlrange):
            raise ValueError(f"{name} CPU actuator ctrlrange differs from latent checkpoint")
        live_body = build_body_obs_schema(
            env=policy_env,
            spec=spec,
            actuator_names=live_names,
            channels=live_state["channels"],
            provenance={"teacher_ckpt": config["teacher_ckpt"]},
        )
        if live_body["semantic_hash"] != runtime.body_obs_schema_hash:
            raise ValueError(f"{name} CPU BodyObsSchema differs from latent checkpoint")
        base = _resolve_cpu_base(policy_env)
        physical_spec = _build_physical_capture_spec(
            policy_env,
            live_names,
            live_ctrlrange,
            racket_site_name="racket_stringbed_center_site",
        )
        uids = stable_sample_uids(dataset.arrays).astype(str)
        if len(set(uids.tolist())) != len(uids):
            raise ValueError(f"{name} physical dataset contains duplicate sample UIDs")
        sources.append(
            _Source(
                name=name,
                dataset=dataset,
                dataset_manifest=manifest,
                motion_map=motion_map,
                env=policy_env,
                base_env=base,
                student_spec=spec,
                physical_spec=physical_spec,
                model_fingerprint=_model_fingerprint(base._model),
                uid_to_row={uid: index for index, uid in enumerate(uids.tolist())},
            )
        )
    return sources


def _validate_simulator_state_dataset(dataset: SequenceDistillDataset, *, name: str) -> None:
    record = dataset.metadata.get("simulator_pre_transition_state")
    expected_fields = [f"sim_pre_{field}" for field in SIMULATOR_PRE_STATE_FIELDS]
    if (
        not isinstance(record, Mapping)
        or record.get("schema_version") != SIMULATOR_PRE_STATE_SCHEMA_VERSION
        or record.get("fields") != expected_fields
        or record.get("timing") != "same_s_t_as_student_obs_before_teacher_action"
    ):
        raise ValueError(f"{name} physical dataset lacks {SIMULATOR_PRE_STATE_SCHEMA_VERSION}; {_RECOLLECT}")
    missing = sorted(set(expected_fields) - set(dataset.arrays))
    if missing:
        raise ValueError(f"{name} causal simulator state fields are missing {missing}; {_RECOLLECT}")
    n = dataset.num_samples
    for field in expected_fields:
        value = np.asarray(dataset.arrays[field])
        if value.ndim < 1 or value.shape[0] != n or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} causal simulator field {field!r} is malformed")


def _validate_row_motion_mapping(dataset: SequenceDistillDataset, motion_map: MotionIdentityMap, *, name: str) -> None:
    traj = np.asarray(dataset.arrays["traj_no"], dtype=np.int64)
    motion = np.asarray(dataset.arrays["motion_uid"], dtype=np.int64)
    expected = motion_map.map_traj_no(traj)
    if not np.array_equal(motion, expected):
        raise ValueError(f"{name} dataset traj_no/motion_uid mapping differs from its manifest")


def _inject_dataset_state(source: _Source, *, row: int, state_match_atol: float) -> None:
    arrays = source.dataset.arrays
    traj = int(arrays["traj_no"][row])
    recorded_step = int(arrays["subtraj_step_no"][row])
    # Collector transition coordinates describe the post-action frame while
    # sim_pre_* and student_obs describe the pre-action state.  Test the only
    # two admissible coordinates and require a unique observation match.
    candidates = sorted({max(0, recorded_step - 1), recorded_step})
    matches: list[int] = []
    for step in candidates:
        live = _inject_candidate(source, row=row, traj=traj, step=step)
        recorded = np.asarray(arrays["student_obs"][row], dtype=np.float32)
        if live.shape == recorded.shape and np.allclose(live, recorded, rtol=0.0, atol=state_match_atol):
            matches.append(step)
    if len(matches) != 1:
        raise ValueError(
            "CPU state injection did not uniquely reproduce the exact dataset student_obs "
            f"for source={source.name} row={row} candidates={candidates} matches={matches}; {_RECOLLECT}"
        )
    # The final inspected candidate need not be the uniquely matching one.
    # Re-inject the accepted candidate so snapshot capture observes exactly the
    # state that passed the cross-backend observation contract.
    selected = matches[0]
    live = _inject_candidate(source, row=row, traj=traj, step=selected)
    recorded = np.asarray(arrays["student_obs"][row], dtype=np.float32)
    if live.shape != recorded.shape or not np.allclose(live, recorded, rtol=0.0, atol=state_match_atol):
        raise RuntimeError("selected CPU causal state was not reproducible on deterministic re-injection")


def _inject_candidate(source: _Source, *, row: int, traj: int, step: int) -> np.ndarray:
    arrays = source.dataset.arrays
    th = source.base_env.th
    th.random_start = False
    th.use_fixed_start = True
    th.start_from_random_step = False
    th.fixed_start_conf = [traj, step]
    source.env.reset(jax.random.PRNGKey(0))
    data = source.base_env._data
    for field in SIMULATOR_PRE_STATE_FIELDS:
        target = getattr(data, field, None)
        value = np.asarray(arrays[f"sim_pre_{field}"][row])
        if target is None:
            if value.size:
                raise ValueError(f"CPU MjData lacks recorded state field {field!r}; {_RECOLLECT}")
            continue
        if field == "time":
            data.time = float(value.reshape(-1)[0])
        else:
            target_array = np.asarray(target)
            if target_array.shape != value.shape:
                raise ValueError(
                    f"CPU/MJX state shape mismatch for {field}: cpu={target_array.shape} data={value.shape}; "
                    f"{_RECOLLECT}"
                )
            target[...] = value.astype(target_array.dtype, copy=False)
    mujoco.mj_forward(source.base_env._model, data)
    observation, carry = source.base_env._create_observation(
        source.base_env._model,
        data,
        source.base_env._additional_carry,
    )
    source.base_env._obs = np.asarray(observation)
    source.base_env._additional_carry = carry
    return np.asarray(
        filter_student_obs(_policy_observation(source), source.student_spec),
        dtype=np.float32,
    )


def _policy_observation(source: _Source) -> np.ndarray:
    raw = np.asarray(source.base_env._obs)
    if source.env is source.base_env:
        return raw
    filter_fn = getattr(source.env, "filter_observation", None)
    if callable(filter_fn):
        return np.asarray(filter_fn(raw))
    raise ValueError("CPU policy wrapper cannot reconstruct its observation from the base environment")


def _resolve_cpu_base(env: Any) -> Any:
    current = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if all(hasattr(current, name) for name in ("_model", "_data", "_additional_carry")):
            if bool(getattr(current, "mjx_enabled", False)):
                raise ValueError("Stage-2 causal adapter requires the MuJoCo CPU environment")
            return current
        current = getattr(current, "env", None)
    raise ValueError("Stage-2 causal adapter cannot resolve a native LocoMujoco CPU environment")


def _model_fingerprint(model: mujoco.MjModel) -> str:
    # MuJoCo's pickle state is the complete compiled MjModel, including solver
    # options and every writable model array (terrain, equality constraints,
    # actuator parameters, etc.).  Hashing only a hand-picked subset would let
    # an untracked model mutation leak between paired rollouts.
    payload = model.__getstate__()
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("MuJoCo model does not expose a canonical complete byte state")
    return hashlib.sha256(payload).hexdigest()


def _encode_snapshot(header: dict[str, Any], parts: Mapping[str, bytes]) -> bytes:
    ordered = sorted((str(name), bytes(value)) for name, value in parts.items())
    header = dict(header)
    header["parts"] = [{"name": name, "length": len(value)} for name, value in ordered]
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return (
        _SNAPSHOT_MAGIC
        + struct.pack(">Q", len(header_bytes))
        + header_bytes
        + b"".join(value for _name, value in ordered)
    )


def _decode_snapshot(payload: bytes) -> tuple[dict[str, Any], dict[str, bytes]]:
    if not payload.startswith(_SNAPSHOT_MAGIC) or len(payload) < len(_SNAPSHOT_MAGIC) + 8:
        raise ValueError("invalid Stage-2 snapshot magic")
    cursor = len(_SNAPSHOT_MAGIC)
    header_size = struct.unpack(">Q", payload[cursor : cursor + 8])[0]
    cursor += 8
    stop = cursor + int(header_size)
    try:
        header = json.loads(payload[cursor:stop].decode("utf-8"))
    except Exception as exc:
        raise ValueError("invalid Stage-2 snapshot header") from exc
    required_header = {
        "schema_version",
        "source_index",
        "sample_uid",
        "model_fingerprint",
        "state_spec",
        "parts",
    }
    if not isinstance(header, dict) or set(header) != required_header:
        raise ValueError("Stage-2 snapshot header fields are not canonical")
    fingerprint = str(header.get("model_fingerprint", ""))
    if (
        header.get("schema_version") != SNAPSHOT_SCHEMA_VERSION
        or header.get("state_spec") != "mjSTATE_INTEGRATION"
        or isinstance(header.get("source_index"), bool)
        or not isinstance(header.get("source_index"), int)
        or int(header["source_index"]) < 0
        or not str(header.get("sample_uid", ""))
        or len(fingerprint) != 64
        or any(character not in _HEX for character in fingerprint)
    ):
        raise ValueError("unsupported Stage-2 snapshot schema")
    cursor = stop
    parts: dict[str, bytes] = {}
    if not isinstance(header["parts"], list):
        raise ValueError("malformed Stage-2 snapshot part table")
    for item in header["parts"]:
        if not isinstance(item, dict) or set(item) != {"name", "length"}:
            raise ValueError("malformed Stage-2 snapshot part table")
        name = str(item.get("name", ""))
        raw_length = item.get("length")
        if isinstance(raw_length, bool) or not isinstance(raw_length, int):
            raise ValueError("malformed Stage-2 snapshot part table")
        length = int(raw_length)
        if not name or name in parts or length < 0 or cursor + length > len(payload):
            raise ValueError("malformed Stage-2 snapshot part table")
        parts[name] = payload[cursor : cursor + length]
        cursor += length
    if cursor != len(payload) or set(parts) != {"mujoco_state", "carry", "observation", "info"}:
        raise ValueError("Stage-2 snapshot parts are incomplete or have trailing bytes")
    return header, parts


def _encode_array(value: np.ndarray) -> bytes:
    array = np.ascontiguousarray(value)
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return struct.pack(">Q", len(header)) + header + array.tobytes(order="C")


def _decode_array(payload: bytes) -> np.ndarray:
    if len(payload) < 8:
        raise ValueError("snapshot array payload is truncated")
    size = struct.unpack(">Q", payload[:8])[0]
    header = json.loads(payload[8 : 8 + size].decode("utf-8"))
    dtype = np.dtype(header["dtype"])
    shape = tuple(int(value) for value in header["shape"])
    expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    raw = payload[8 + size :]
    if len(raw) != expected:
        raise ValueError("snapshot array byte count differs from its header")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def _canonical_tree(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical_tree(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, tuple | list):
        return [_canonical_tree(item) for item in value]
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return np.asarray(value)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise ValueError(f"snapshot info/RNG tree contains unsupported value type {type(value).__name__}")


def _joint_coordinate_names(model: mujoco.MjModel) -> tuple[list[str], list[str]]:
    qpos = [""] * int(model.nq)
    qvel = [""] * int(model.nv)
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_{joint_id}"
        qadr = int(model.jnt_qposadr[joint_id])
        dadr = int(model.jnt_dofadr[joint_id])
        joint_type = int(model.jnt_type[joint_id])
        if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
            qlabels = ("x", "y", "z", "qw", "qx", "qy", "qz")
            vlabels = ("vx", "vy", "vz", "wx", "wy", "wz")
        elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
            qlabels = ("qw", "qx", "qy", "qz")
            vlabels = ("wx", "wy", "wz")
        else:
            qlabels = ("position",)
            vlabels = ("velocity",)
        for offset, label in enumerate(qlabels):
            qpos[qadr + offset] = f"{name}:{label}"
        for offset, label in enumerate(vlabels):
            qvel[dadr + offset] = f"{name}:{label}"
    if any(not name for name in qpos + qvel):
        raise ValueError("MuJoCo joint coordinate naming did not cover qpos/qvel")
    return qpos, qvel


def _root_indices(base: Any) -> tuple[np.ndarray, np.ndarray]:
    name = str(getattr(base, "root_free_joint_xml_name", ""))
    joint = mujoco.mj_name2id(base._model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint < 0 or int(base._model.jnt_type[joint]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError("Stage-2 CPU environment lacks the expected free root joint")
    qadr = int(base._model.jnt_qposadr[joint])
    dadr = int(base._model.jnt_dofadr[joint])
    return np.arange(qadr, qadr + 7), np.arange(dadr, dadr + 6)


def _trunk_state_spec(
    base: Any,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Resolve pelvis root state plus rotational lumbar-chain velocities.

    The upper-body body is walked back to the body carrying the free root
    joint.  Only hinge/ball joints on that ancestry are admitted, so arm,
    lower-body, and auxiliary abdominal slide coordinates cannot silently enter
    the trunk outcome.  Models without an explicit rotational torso chain fail
    closed instead of relabeling pelvis-only motion as whole-trunk motion.
    """

    model = base._model
    root_qpos, root_qvel = _root_indices(base)
    root_name = str(getattr(base, "root_free_joint_xml_name", ""))
    root_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, root_name)
    root_body = int(model.jnt_bodyid[root_joint])
    upper_name = str(getattr(base, "upper_body_xml_name", ""))
    upper_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, upper_name)
    if upper_body < 0:
        raise ValueError("Stage-2 CPU environment lacks its declared upper-body torso")

    ancestry: set[int] = set()
    body = int(upper_body)
    while body > 0:
        ancestry.add(body)
        if body == root_body:
            break
        body = int(model.body_parentid[body])
    if root_body not in ancestry:
        raise ValueError("declared Stage-2 torso is not descended from the free pelvis root")

    rotational_qvel: list[int] = []
    rotational_names: list[str] = []
    for joint_id in range(int(model.njnt)):
        if joint_id == root_joint or int(model.jnt_bodyid[joint_id]) not in ancestry:
            continue
        joint_type = int(model.jnt_type[joint_id])
        if joint_type == int(mujoco.mjtJoint.mjJNT_HINGE):
            width = 1
        elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
            width = 3
        else:
            continue
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"trunk_joint_{joint_id}"
        address = int(model.jnt_dofadr[joint_id])
        labels = ("angular_velocity",) if width == 1 else ("wx", "wy", "wz")
        rotational_qvel.extend(range(address, address + width))
        rotational_names.extend(f"lumbar_joint:{joint_name}:{label}" for label in labels)
    if not rotational_qvel:
        raise ValueError(
            "Stage-2 torso ancestry has no rotational lumbar DOFs; pelvis-only trunk relabeling is forbidden"
        )

    names = [
        "pelvis_root:position_x",
        "pelvis_root:position_y",
        "pelvis_root:position_z",
        "pelvis_root:quaternion_w",
        "pelvis_root:quaternion_x",
        "pelvis_root:quaternion_y",
        "pelvis_root:quaternion_z",
        "pelvis_root:linear_velocity_x",
        "pelvis_root:linear_velocity_y",
        "pelvis_root:linear_velocity_z",
        "pelvis_root:angular_velocity_x",
        "pelvis_root:angular_velocity_y",
        "pelvis_root:angular_velocity_z",
        *rotational_names,
    ]
    units = [
        "meter",
        "meter",
        "meter",
        "unit_quaternion",
        "unit_quaternion",
        "unit_quaternion",
        "unit_quaternion",
        "meter_per_second",
        "meter_per_second",
        "meter_per_second",
        "radian_per_second",
        "radian_per_second",
        "radian_per_second",
        *(["radian_per_second"] * len(rotational_qvel)),
    ]
    qvel = np.concatenate([root_qvel, np.asarray(rotational_qvel, dtype=np.int64)])
    return root_qpos, qvel, names, units


def _trunk_state(
    source: _Source,
    *,
    qpos_indices: np.ndarray,
    qvel_indices: np.ndarray,
) -> np.ndarray:
    data = source.base_env._data
    return np.concatenate(
        [
            np.asarray(data.qpos[qpos_indices], dtype=np.float32),
            np.asarray(data.qvel[qvel_indices], dtype=np.float32),
        ]
    )


def _racket_state(physical: Mapping[str, np.ndarray]) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    matrix = np.asarray(physical["racket_rotation_matrix"], dtype=np.float64).reshape(3, 3)
    xyzw = Rotation.from_matrix(matrix).as_quat(canonical=True)
    wxyz = np.concatenate([xyzw[3:], xyzw[:3]])
    return np.concatenate(
        [
            np.asarray(physical["racket_position"]).reshape(3),
            wxyz,
            np.asarray(physical["racket_linear_velocity"]).reshape(3),
            np.asarray(physical["racket_angular_velocity"]).reshape(3),
            np.asarray(physical["stringbed_normal"]).reshape(3),
        ]
    ).astype(np.float32)


def _build_outcome_schemas(
    *,
    horizon: int,
    muscle_names: Sequence[str],
    qpos_names: Sequence[str],
    qvel_names: Sequence[str],
    trunk_names: Sequence[str],
    trunk_units: Sequence[str],
) -> dict[str, dict[str, Any]]:
    def timed(names: Sequence[str]) -> list[str]:
        return [f"step_{step:03d}:{name}" for step in range(horizon) for name in names]

    muscle_features = timed(muscle_names)
    if len(trunk_names) != len(trunk_units) or not trunk_names:
        raise ValueError("Stage-2 trunk schema names/units must form one non-empty ABI")
    racket_names = (
        "position_x",
        "position_y",
        "position_z",
        "quaternion_w",
        "quaternion_x",
        "quaternion_y",
        "quaternion_z",
        "linear_velocity_x",
        "linear_velocity_y",
        "linear_velocity_z",
        "angular_velocity_x",
        "angular_velocity_y",
        "angular_velocity_z",
        "stringbed_normal_x",
        "stringbed_normal_y",
        "stringbed_normal_z",
    )
    result = {
        "muscle_excitation": _schema(
            muscle_features,
            ["unit_interval"] * len(muscle_features),
            "ordered_model_actuators_over_cpu_rollout",
            "unit_interval_excitation",
        ),
        "muscle_activation": _schema(
            muscle_features,
            ["unit_interval"] * len(muscle_features),
            "ordered_model_actuators_over_cpu_rollout",
            "mujoco_unit_interval_activation_state",
        ),
        "joint_position": _schema(
            timed(qpos_names),
            ["mixed_joint_coordinate"] * (horizon * len(qpos_names)),
            "mujoco_joint_coordinates_world_root",
            "ordered_joint_qpos",
        ),
        "joint_velocity": _schema(
            timed(qvel_names),
            ["mixed_joint_velocity"] * (horizon * len(qvel_names)),
            "mujoco_joint_coordinates_world_root",
            "ordered_joint_qvel",
        ),
        "trunk_state": _schema(
            timed(trunk_names),
            list(trunk_units) * horizon,
            "world_frame_pelvis_root_plus_local_lumbar_joint_rates",
            "ordered_trunk_state",
        ),
        "racket_state": _schema(
            timed(racket_names),
            ["mixed_racket_state"] * (horizon * len(racket_names)),
            "world_frame_stringbed_center",
            "ordered_racket_state",
        ),
        "impact_outcome": _schema(
            (),
            (),
            "unavailable_stage2_no_simulated_shuttle_contact",
            "ordered_impact_outcome",
            available=False,
        ),
        "landing_outcome": _schema(
            (),
            (),
            "unavailable_stage2_no_shuttle_flight",
            "ordered_landing_outcome",
            available=False,
        ),
    }
    return result


def _schema(
    names: Sequence[str],
    units: Sequence[str],
    frame: str,
    semantics: str,
    *,
    available: bool = True,
) -> dict[str, Any]:
    return {
        "feature_names": [str(value) for value in names],
        "units": [str(value) for value in units],
        "coordinate_frame": frame,
        "semantics": semantics,
        "available": bool(available),
        "terminal_padding": "hold_last_measured_state" if available else "not_applicable",
    }


__all__ = [
    "SNAPSHOT_SCHEMA_VERSION",
    "STAGE2_ADAPTER_SCHEMA_VERSION",
    "Stage2CausalRolloutAdapter",
    "create_adapter",
]

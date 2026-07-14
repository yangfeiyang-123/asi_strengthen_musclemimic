"""Produce exact-state paired latent-intervention rollout records.

The driver owns pairing, provenance, validation, and atomic publication.  It
does *not* guess how an environment maps an ``analysis_inputs`` sample UID back
to simulator state.  A production adapter must implement
``CausalRolloutAdapter`` and must serialize/restore the complete dynamical
snapshot deterministically.  Environments that cannot do that are unsupported
until an adapter is supplied; there is intentionally no permissive fallback.

An adapter factory is loaded as ``module:attribute`` and called with the
``adapter_config`` object from a job file.  A factory may reuse the environment
construction helpers in :mod:`fullbody.latent_closed_loop_eval`, but it must
add sample-UID lookup and exact snapshot support itself.  The built-in
``replay-record`` adapter only replays a separately sealed evaluator export; it
never synthesizes an environment outcome.

Typical workflow::

    python -m musclemimic.latent_muscle.causal_rollout_driver template \
      --output rollout_job.json
    # Fill in paths and an adapter factory, then inspect without rollouts.
    python -m musclemimic.latent_muscle.causal_rollout_driver evaluate \
      --job-config rollout_job.json --dry-run
    # Explicit evaluation publishes baseline_records.npz,
    # perturbed_records.npz, and paired_rollout_manifest.json atomically.
    python -m musclemimic.latent_muscle.causal_rollout_driver evaluate \
      --job-config rollout_job.json

The resulting triplet is input to ``causal_rollout_artifact``.  Merely running
this producer is evaluation, not evidence that an adapter is scientifically
correct; adapter source and its environment/policy fingerprints remain part of
the audit surface.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from musclemimic.badminton.json_contract import load_json_strict
from musclemimic.distill.physical import (
    physical_signal_metadata,
    validate_activation_valid_mask,
    validate_physical_signal_semantics,
    validate_unit_muscle_activation,
    validate_unit_muscle_excitation,
)
from musclemimic.distill.provenance import canonical_json_sha256, file_sha256
from musclemimic.latent_muscle.analysis_export import ANALYSIS_INPUT_SCHEMA_VERSION
from musclemimic.latent_muscle.causal_rollout_artifact import (
    PAIRED_ROLLOUT_SOURCE_SCHEMA_VERSION,
    REQUIRED_OUTCOMES,
)

ADAPTER_SCHEMA_VERSION = "latent_causal_rollout_adapter_v1"
JOB_SCHEMA_VERSION = "latent_causal_rollout_job_v1"
REPLAY_SOURCE_SCHEMA_VERSION = "latent_causal_replay_records_v1"
DRIVER_SCHEMA_VERSION = "latent_causal_rollout_driver_v1"
BASELINE_FILENAME = "baseline_records.npz"
PERTURBED_FILENAME = "perturbed_records.npz"
MANIFEST_FILENAME = "paired_rollout_manifest.json"
_HEX = frozenset("0123456789abcdef")
_OUTCOME_SEMANTICS = {
    "muscle_excitation": "unit_interval_excitation",
    "muscle_activation": "mujoco_unit_interval_activation_state",
    "joint_position": "ordered_joint_qpos",
    "joint_velocity": "ordered_joint_qvel",
    "trunk_state": "ordered_trunk_state",
    "racket_state": "ordered_racket_state",
    "impact_outcome": "ordered_impact_outcome",
    "landing_outcome": "ordered_landing_outcome",
}


@dataclass(frozen=True)
class RolloutRequest:
    """One baseline or latent-perturbed evaluation from a restored snapshot."""

    sample_uid: str
    rollout_seed: int
    baseline_latent: np.ndarray
    intervention_direction: np.ndarray | None
    intervention_epsilon: float
    direction_index: int | None
    epsilon_index: int | None

    @property
    def is_baseline(self) -> bool:
        return self.intervention_direction is None

    @property
    def evaluated_latent(self) -> np.ndarray:
        if self.intervention_direction is None:
            return np.asarray(self.baseline_latent, dtype=np.float32)
        return np.asarray(
            self.baseline_latent + self.intervention_epsilon * self.intervention_direction,
            dtype=np.float32,
        )


@runtime_checkable
class CausalRolloutAdapter(Protocol):
    """Required boundary between the generic sealer and a real evaluator.

    ``snapshot_to_bytes`` must be deterministic and cover all dynamical state
    required for an exact restart.  ``set_common_random_seed`` must reset every
    random source used by the rollout.  The driver verifies the serialized state
    after every restore and compares the adapter's random-state fingerprint for
    baseline and all paired perturbations.
    """

    def descriptor(self) -> Mapping[str, Any]:
        """Return immutable environment, policy, signal, and outcome ABI data."""

    def prepare_analysis_sample(self, sample_uid: str) -> Any:
        """Position the evaluator at ``sample_uid`` and return its exact snapshot."""

    def snapshot_to_bytes(self, snapshot: Any) -> bytes:
        """Serialize a complete snapshot to deterministic bytes."""

    def restore_snapshot(self, snapshot: Any) -> None:
        """Restore the complete dynamical snapshot without advancing time."""

    def capture_snapshot(self) -> Any:
        """Capture the current state for byte-for-byte restore verification."""

    def set_common_random_seed(self, seed: int) -> None:
        """Reset all rollout RNG streams to the supplied non-negative seed."""

    def random_state_fingerprint(self) -> str:
        """Fingerprint all RNG state after ``set_common_random_seed``."""

    def evaluate_rollout(self, request: RolloutRequest) -> Mapping[str, Any]:
        """Run evaluation and return exactly the eight ordered measured outcomes."""


@dataclass(frozen=True)
class _AnalysisBindings:
    sample_uids: np.ndarray
    latents: np.ndarray
    directions: np.ndarray
    epsilons: np.ndarray
    sidecar: dict[str, Any]
    inputs_sha256: str


def produce_paired_rollouts(
    *,
    analysis_inputs: str | Path,
    analysis_manifest: str | Path,
    adapter: CausalRolloutAdapter,
    output_dir: str | Path,
    base_seed: int = 0,
    adapter_import: str = "direct-python-object",
    adapter_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate all exact pairs and atomically publish the source triplet."""

    bindings = _load_analysis_bindings(analysis_inputs, analysis_manifest)
    descriptor = _validate_adapter(adapter, bindings=bindings)
    seed0 = int(base_seed)
    if seed0 < 0:
        raise ValueError("base_seed must be non-negative")
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"rollout output directory already exists (refusing partial overwrite): {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    n = int(bindings.sample_uids.shape[0])
    d = int(bindings.directions.shape[0])
    e = int(bindings.epsilons.shape[0])
    baseline_outcomes: dict[str, list[np.ndarray]] = {name: [] for name in REQUIRED_OUTCOMES}
    perturbed_outcomes: dict[str, list[list[list[np.ndarray]]]] = {name: [] for name in REQUIRED_OUTCOMES}
    snapshot_fingerprints: list[str] = []
    seeds: list[int] = []
    random_fingerprints: list[str] = []
    canonical_shapes: dict[str, tuple[int, ...]] | None = None

    for sample_index, uid_value in enumerate(bindings.sample_uids.tolist()):
        uid = str(uid_value)
        snapshot = adapter.prepare_analysis_sample(uid)
        snapshot_bytes = _snapshot_bytes(adapter, snapshot)
        snapshot_fingerprint = hashlib.sha256(snapshot_bytes).hexdigest()
        seed = _sample_seed(seed0, uid)
        if seed in seeds:
            raise ValueError("stable common-random-number seed collision")
        seeds.append(seed)
        snapshot_fingerprints.append(snapshot_fingerprint)

        baseline_random = _restore_and_seed(
            adapter,
            snapshot=snapshot,
            expected_snapshot_bytes=snapshot_bytes,
            seed=seed,
        )
        baseline_request = RolloutRequest(
            sample_uid=uid,
            rollout_seed=seed,
            baseline_latent=np.asarray(bindings.latents[sample_index], dtype=np.float32),
            intervention_direction=None,
            intervention_epsilon=0.0,
            direction_index=None,
            epsilon_index=None,
        )
        baseline_result = _validate_outcomes(
            adapter.evaluate_rollout(baseline_request),
            descriptor=descriptor,
            expected_shapes=canonical_shapes,
        )
        if canonical_shapes is None:
            canonical_shapes = {name: value.shape for name, value in baseline_result.items()}
            _validate_outcome_schemas(
                descriptor["outcome_schemas"],
                shapes=canonical_shapes,
                activation_valid_mask=descriptor["activation_valid_mask"],
                outcome_availability=descriptor["outcome_availability"],
            )
        for name in REQUIRED_OUTCOMES:
            baseline_outcomes[name].append(baseline_result[name])

        sample_perturbations: dict[str, list[list[np.ndarray]]] = {name: [] for name in REQUIRED_OUTCOMES}
        for direction_index, direction in enumerate(bindings.directions):
            direction_results: dict[str, list[np.ndarray]] = {name: [] for name in REQUIRED_OUTCOMES}
            for epsilon_index, epsilon in enumerate(bindings.epsilons):
                paired_random = _restore_and_seed(
                    adapter,
                    snapshot=snapshot,
                    expected_snapshot_bytes=snapshot_bytes,
                    seed=seed,
                )
                if paired_random != baseline_random:
                    raise ValueError(f"adapter did not restore the same common-random-number state for sample {uid!r}")
                request = RolloutRequest(
                    sample_uid=uid,
                    rollout_seed=seed,
                    baseline_latent=np.asarray(bindings.latents[sample_index], dtype=np.float32),
                    intervention_direction=np.asarray(direction, dtype=np.float32),
                    intervention_epsilon=float(epsilon),
                    direction_index=direction_index,
                    epsilon_index=epsilon_index,
                )
                result = _validate_outcomes(
                    adapter.evaluate_rollout(request),
                    descriptor=descriptor,
                    expected_shapes=canonical_shapes,
                )
                for name in REQUIRED_OUTCOMES:
                    direction_results[name].append(result[name])
            for name in REQUIRED_OUTCOMES:
                sample_perturbations[name].append(direction_results[name])
        for name in REQUIRED_OUTCOMES:
            perturbed_outcomes[name].append(sample_perturbations[name])
        random_fingerprints.append(baseline_random)

    if canonical_shapes is None:
        raise ValueError("analysis inputs contain no samples")
    baseline_arrays: dict[str, np.ndarray] = {
        "sample_uids": np.asarray(bindings.sample_uids, dtype=np.str_),
        "initial_state_fingerprints": np.asarray(snapshot_fingerprints, dtype=np.str_),
        "rollout_seeds": np.asarray(seeds, dtype=np.int64),
    }
    perturbed_arrays: dict[str, np.ndarray] = {
        "sample_uids": np.asarray(bindings.sample_uids, dtype=np.str_),
        "initial_state_fingerprints": np.broadcast_to(
            np.asarray(snapshot_fingerprints, dtype=np.str_)[:, None, None], (n, d, e)
        ).copy(),
        "rollout_seeds": np.broadcast_to(np.asarray(seeds, dtype=np.int64)[:, None, None], (n, d, e)).copy(),
        "intervention_directions": np.asarray(bindings.directions),
        "intervention_epsilons": np.asarray(bindings.epsilons),
    }
    for name in REQUIRED_OUTCOMES:
        baseline_arrays[name] = np.stack(baseline_outcomes[name], axis=0)
        perturbed_arrays[name] = np.asarray(perturbed_outcomes[name])

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=str(destination.parent),
        )
    )
    published = False
    try:
        baseline_path = staging / BASELINE_FILENAME
        perturbed_path = staging / PERTURBED_FILENAME
        np.savez_compressed(baseline_path, **baseline_arrays)
        np.savez_compressed(perturbed_path, **perturbed_arrays)
        manifest = {
            "schema_version": PAIRED_ROLLOUT_SOURCE_SCHEMA_VERSION,
            "producer_schema_version": DRIVER_SCHEMA_VERSION,
            "evidence_kind": "environment_rollout",
            "checkpoint_fingerprint": descriptor["checkpoint_fingerprint"],
            "synergy_basis_fingerprint": descriptor["synergy_basis_fingerprint"],
            "analysis_inputs_sha256": bindings.inputs_sha256,
            "analysis_manifest_fingerprint": bindings.sidecar["manifest_fingerprint"],
            "sample_uid_fingerprint": canonical_json_sha256(bindings.sample_uids.tolist()),
            "intervention_direction_fingerprint": canonical_json_sha256(
                np.asarray(bindings.directions, dtype=np.float64).tolist()
            ),
            "intervention_epsilon_fingerprint": canonical_json_sha256(
                np.asarray(bindings.epsilons, dtype=np.float64).tolist()
            ),
            "baseline_records_sha256": file_sha256(baseline_path),
            "perturbed_records_sha256": file_sha256(perturbed_path),
            "environment_fingerprint": descriptor["environment_fingerprint"],
            "policy_abi_hash": descriptor["policy_abi_hash"],
            "rollout_engine": descriptor["rollout_engine"],
            "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
            "adapter_import": str(adapter_import),
            "adapter_config_fingerprint": canonical_json_sha256(dict(adapter_config or {})),
            "fixed_state_initialization": "exact_snapshot_restore",
            "common_random_numbers": True,
            "snapshot_fingerprints": snapshot_fingerprints,
            "snapshot_fingerprint_set": canonical_json_sha256(snapshot_fingerprints),
            "rollout_seeds": seeds,
            "random_state_fingerprints": random_fingerprints,
            "sample_snapshot_bindings": [
                {
                    "sample_uid": str(uid),
                    "snapshot_fingerprint": snapshot_fingerprints[index],
                    "rollout_seed": seeds[index],
                    "random_state_fingerprint": random_fingerprints[index],
                }
                for index, uid in enumerate(bindings.sample_uids.tolist())
            ],
            "physical_signal_semantics": descriptor["physical_signal_semantics"],
            "activation_valid_mask": descriptor["activation_valid_mask"],
            "outcome_schemas": descriptor["outcome_schemas"],
            "outcome_availability": descriptor["outcome_availability"],
            "stage2_diagnostic_outcomes_complete": descriptor["stage2_diagnostic_outcomes_complete"],
            "task_outcomes_complete": descriptor["task_outcomes_complete"],
            "required_outcomes": list(REQUIRED_OUTCOMES),
            "num_samples": n,
            "num_directions": d,
            "num_epsilons": e,
            "publication": "atomic_directory_rename_no_overwrite",
            "limitations": (
                "The generic driver verifies pairing and adapter receipts, but scientific "
                "validity still depends on the registered adapter serializing the complete "
                "environment state and reporting genuinely measured outcomes."
            ),
        }
        manifest["manifest_fingerprint"] = canonical_json_sha256(manifest)
        manifest_path = staging / MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _fsync_file(baseline_path)
        _fsync_file(perturbed_path)
        _fsync_file(manifest_path)
        _fsync_directory(staging)
        os.replace(staging, destination)
        published = True
        _fsync_directory(destination.parent)
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)
    return manifest


def validate_job(*, job: Mapping[str, Any], instantiate_adapter: bool = True) -> dict[str, Any]:
    """Validate a job and adapter descriptor without capturing state or evaluating."""

    payload = _validate_job_shape(job)
    bindings = _load_analysis_bindings(payload["analysis_inputs"], payload["analysis_manifest"])
    result: dict[str, Any] = {
        "schema_version": "latent_causal_rollout_dry_run_v1",
        "rollouts_executed": False,
        "output_published": False,
        "num_samples": int(bindings.sample_uids.shape[0]),
        "num_directions": int(bindings.directions.shape[0]),
        "num_epsilons": int(bindings.epsilons.shape[0]),
        "job_config_fingerprint": canonical_json_sha256(payload),
        "limitation": (
            "Dry-run does not call prepare_analysis_sample, restore_snapshot, or "
            "evaluate_rollout; exact restoration is proven only by a completed evaluation."
        ),
    }
    if instantiate_adapter:
        adapter = load_adapter(payload["adapter_import"], payload["adapter_config"])
        result["adapter_descriptor"] = _validate_adapter(adapter, bindings=bindings)
    return result


def evaluate_job(job: Mapping[str, Any]) -> dict[str, Any]:
    payload = _validate_job_shape(job)
    adapter = load_adapter(payload["adapter_import"], payload["adapter_config"])
    return produce_paired_rollouts(
        analysis_inputs=payload["analysis_inputs"],
        analysis_manifest=payload["analysis_manifest"],
        adapter=adapter,
        output_dir=payload["output_dir"],
        base_seed=int(payload["base_seed"]),
        adapter_import=payload["adapter_import"],
        adapter_config=payload["adapter_config"],
    )


def load_adapter(import_path: str, config: Mapping[str, Any]) -> CausalRolloutAdapter:
    """Instantiate an adapter factory from ``module:attribute`` or replay shorthand."""

    spec = str(import_path).strip()
    if spec == "replay-record":
        candidate: Any = ReplayRecordAdapter(config)
    else:
        if ":" not in spec:
            raise ValueError("adapter_import must be 'module:attribute' or 'replay-record'")
        module_name, attribute_name = spec.split(":", 1)
        if not module_name or not attribute_name:
            raise ValueError("adapter_import has an empty module or attribute")
        module = importlib.import_module(module_name)
        factory = getattr(module, attribute_name)
        candidate = factory(dict(config))
    if not isinstance(candidate, CausalRolloutAdapter):
        raise TypeError("adapter does not implement the complete CausalRolloutAdapter protocol")
    return candidate


class ReplayRecordAdapter:
    """Strictly replay a content-addressed export from an external evaluator.

    Config keys are ``records_npz`` and ``records_manifest``.  Required NPZ
    fields are documented by :func:`replay_record_template`.  This adapter is a
    transport/testing bridge, not a simulator and not independent causal
    evidence; the upstream evaluator remains responsible for the recorded data.
    """

    def __init__(self, config: Mapping[str, Any]):
        self._records_path = Path(str(config.get("records_npz", "")))
        self._manifest_path = Path(str(config.get("records_manifest", "")))
        if not self._records_path.is_file() or not self._manifest_path.is_file():
            raise FileNotFoundError("replay-record config points to missing records or manifest")
        self._manifest = _load_self_fingerprinted_json(self._manifest_path)
        if self._manifest.get("schema_version") != REPLAY_SOURCE_SCHEMA_VERSION:
            raise ValueError("replay source schema is unsupported")
        if self._manifest.get("records_sha256") != file_sha256(self._records_path):
            raise ValueError("replay records differ from the replay manifest")
        descriptor = self._manifest.get("adapter_descriptor")
        if not isinstance(descriptor, dict):
            raise ValueError("replay manifest lacks adapter_descriptor")
        self._descriptor = descriptor
        with np.load(self._records_path, allow_pickle=False) as raw:
            self._data = {name: np.asarray(raw[name]) for name in raw.files}
        required = {
            "sample_uids",
            "baseline_latents",
            "intervention_directions",
            "intervention_epsilons",
            "snapshot_bytes",
            "snapshot_lengths",
            "snapshot_fingerprints",
            "rollout_seeds",
            "random_state_fingerprints",
        }
        required.update(f"baseline__{name}" for name in REQUIRED_OUTCOMES)
        required.update(f"perturbed__{name}" for name in REQUIRED_OUTCOMES)
        missing = sorted(required - set(self._data))
        if missing:
            raise ValueError(f"replay records are incomplete: {missing}")
        self._validate_replay_records()
        self._active_index: int | None = None
        self._active_snapshot: bytes | None = None
        self._seeded = False

    def descriptor(self) -> Mapping[str, Any]:
        return dict(self._descriptor)

    def prepare_analysis_sample(self, sample_uid: str) -> bytes:
        matches = np.flatnonzero(self._data["sample_uids"].astype(str) == str(sample_uid))
        if matches.shape != (1,):
            raise ValueError(f"replay source does not uniquely contain sample UID {sample_uid!r}")
        index = int(matches[0])
        length = int(self._data["snapshot_lengths"][index])
        snapshot = bytes(np.asarray(self._data["snapshot_bytes"][index, :length], dtype=np.uint8))
        self._active_index = index
        self._active_snapshot = snapshot
        self._seeded = False
        return snapshot

    def snapshot_to_bytes(self, snapshot: Any) -> bytes:
        if not isinstance(snapshot, bytes) or not snapshot:
            raise ValueError("replay snapshot must be non-empty bytes")
        return snapshot

    def restore_snapshot(self, snapshot: Any) -> None:
        payload = self.snapshot_to_bytes(snapshot)
        if self._active_index is None:
            raise RuntimeError("prepare_analysis_sample must precede replay restore")
        expected = self.prepare_analysis_sample(str(self._data["sample_uids"][self._active_index]))
        if payload != expected:
            raise ValueError("replay restore snapshot differs from its recorded sample")
        self._active_snapshot = payload

    def capture_snapshot(self) -> bytes:
        if self._active_snapshot is None:
            raise RuntimeError("replay has no active snapshot")
        return self._active_snapshot

    def set_common_random_seed(self, seed: int) -> None:
        if self._active_index is None:
            raise RuntimeError("replay has no active sample")
        if int(self._data["rollout_seeds"][self._active_index]) != int(seed):
            raise ValueError("replay rollout seed differs from the driver common-random-number seed")
        self._seeded = True

    def random_state_fingerprint(self) -> str:
        if self._active_index is None or not self._seeded:
            raise RuntimeError("replay random state was not seeded")
        return str(self._data["random_state_fingerprints"][self._active_index])

    def evaluate_rollout(self, request: RolloutRequest) -> Mapping[str, Any]:
        if self._active_index is None or not self._seeded:
            raise RuntimeError("replay rollout was not restored and seeded")
        index = self._active_index
        if str(self._data["sample_uids"][index]) != request.sample_uid:
            raise ValueError("replay request sample UID mismatch")
        if not np.array_equal(
            np.asarray(request.baseline_latent),
            np.asarray(self._data["baseline_latents"][index]),
        ):
            raise ValueError("replay request baseline latent mismatch")
        if request.is_baseline:
            return {name: self._data[f"baseline__{name}"][index] for name in REQUIRED_OUTCOMES}
        if request.direction_index is None or request.epsilon_index is None:
            raise ValueError("replay perturbed request lacks direction/epsilon indices")
        if not np.array_equal(
            np.asarray(request.intervention_direction),
            np.asarray(self._data["intervention_directions"][request.direction_index]),
        ) or float(request.intervention_epsilon) != float(self._data["intervention_epsilons"][request.epsilon_index]):
            raise ValueError("replay request direction/epsilon binding mismatch")
        return {
            name: self._data[f"perturbed__{name}"][index, request.direction_index, request.epsilon_index]
            for name in REQUIRED_OUTCOMES
        }

    def _validate_replay_records(self) -> None:
        uids = np.asarray(self._data["sample_uids"]).astype(str)
        latents = np.asarray(self._data["baseline_latents"])
        directions = np.asarray(self._data["intervention_directions"])
        epsilons = np.asarray(self._data["intervention_epsilons"])
        snapshots = np.asarray(self._data["snapshot_bytes"])
        lengths = np.asarray(self._data["snapshot_lengths"])
        n = len(uids)
        if (
            uids.shape != (n,)
            or n == 0
            or len(set(uids.tolist())) != n
            or latents.ndim != 2
            or latents.shape[0] != n
            or directions.ndim != 2
            or directions.shape[1] != latents.shape[1]
            or epsilons.ndim != 1
            or directions.shape[0] == 0
            or epsilons.shape[0] == 0
            or np.any(epsilons == 0.0)
            or not np.all(np.isfinite(latents))
            or not np.all(np.isfinite(directions))
            or not np.all(np.isfinite(epsilons))
            or snapshots.ndim != 2
            or snapshots.dtype != np.uint8
            or snapshots.shape[0] != n
            or lengths.shape != (n,)
            or np.any(lengths <= 0)
            or np.any(lengths > snapshots.shape[1])
        ):
            raise ValueError("replay analysis/snapshot arrays are malformed")
        computed = []
        for index, length in enumerate(lengths.tolist()):
            if np.any(snapshots[index, int(length) :] != 0):
                raise ValueError("replay snapshot padding must be canonical zero bytes")
            computed.append(hashlib.sha256(bytes(snapshots[index, : int(length)])).hexdigest())
        if np.asarray(self._data["snapshot_fingerprints"]).astype(str).tolist() != computed:
            raise ValueError("replay snapshot fingerprints do not match serialized bytes")
        seeds = np.asarray(self._data["rollout_seeds"])
        random_fingerprints = np.asarray(self._data["random_state_fingerprints"]).astype(str)
        if (
            seeds.shape != (n,)
            or random_fingerprints.shape != (n,)
            or not np.issubdtype(seeds.dtype, np.integer)
            or np.any(seeds < 0)
        ):
            raise ValueError("replay rollout seeds are malformed")
        for value in random_fingerprints.tolist():
            _require_hex64("replay random state fingerprint", value)
        d, e = directions.shape[0], epsilons.shape[0]
        for name in REQUIRED_OUTCOMES:
            baseline = np.asarray(self._data[f"baseline__{name}"])
            perturbed = np.asarray(self._data[f"perturbed__{name}"])
            if baseline.shape[0] != n or perturbed.shape[:3] != (n, d, e):
                raise ValueError(f"replay outcome {name!r} has invalid leading dimensions")
            if perturbed.shape[3:] != baseline.shape[1:]:
                raise ValueError(f"replay outcome {name!r} shape mismatch")


def replay_record_template() -> dict[str, Any]:
    """Return the strict field contract for external replay exports."""

    return {
        "schema_version": REPLAY_SOURCE_SCHEMA_VERSION,
        "records_npz_fields": {
            "sample_uids": "str[N]",
            "baseline_latents": "float32[N,Z]",
            "intervention_directions": "float32[D,Z]",
            "intervention_epsilons": "float32[E]",
            "snapshot_bytes": "uint8[N,B] zero-padded canonical full-state bytes",
            "snapshot_lengths": "int64[N]",
            "snapshot_fingerprints": "lowercase_hex64[N]",
            "rollout_seeds": "int64[N]",
            "random_state_fingerprints": "lowercase_hex64[N]",
            **{f"baseline__{name}": f"measured [N,...] {name}" for name in REQUIRED_OUTCOMES},
            **{f"perturbed__{name}": f"measured [N,D,E,...] {name}" for name in REQUIRED_OUTCOMES},
        },
        "records_manifest_fields": {
            "schema_version": REPLAY_SOURCE_SCHEMA_VERSION,
            "records_sha256": "lowercase_hex64",
            "adapter_descriptor": f"{ADAPTER_SCHEMA_VERSION} object",
            "manifest_fingerprint": "canonical self fingerprint",
        },
        "warning": (
            "Replay records must come from a real evaluator. The adapter only validates and "
            "replays them; it cannot convert offline decoder deltas into causal task outcomes."
        ),
    }


def job_template() -> dict[str, Any]:
    return {
        "schema_version": JOB_SCHEMA_VERSION,
        "analysis_inputs": "artifacts/analysis_inputs.npz",
        "analysis_manifest": "artifacts/analysis_inputs.json",
        "output_dir": "artifacts/paired_causal_rollouts",
        "base_seed": 20260713,
        "adapter_import": ("musclemimic.latent_muscle.stage2_causal_adapter:create_adapter"),
        "adapter_config": {
            "latent_checkpoint": "artifacts/latent_checkpoint",
            "teacher_ckpt": "artifacts/stage2_teacher_checkpoint",
            "dataset_dir": "artifacts/physical_rollout_train",
            "val_dataset_dir": "artifacts/physical_rollout_val",
            "analysis_inputs": "artifacts/analysis_inputs.npz",
            "analysis_manifest": "artifacts/analysis_inputs.json",
            "rollout_horizon_steps": 120,
            "state_match_atol": 1e-5,
        },
        "adapter_contract": {
            "protocol": "musclemimic.latent_muscle.causal_rollout_driver:CausalRolloutAdapter",
            "factory_signature": "create_adapter(adapter_config: dict) -> CausalRolloutAdapter",
            "exact_state": "deterministic complete snapshot bytes and byte-identical restore",
            "randomness": "reset all RNGs; equal random-state fingerprint for every pair",
            "outcomes": list(REQUIRED_OUTCOMES),
            "stage2_builtin": (
                "The built-in adapter requires physical rows collected with "
                "mujoco_mjx_pre_transition_state_v1, injects them into the CPU environment, "
                "and accepts only an exact live student_obs match."
            ),
            "stage2_scope": (
                "Six diagnostic outcomes are measured. Stage-2 has no shuttle landing, so "
                "task_outcomes_complete is false and cannot support the final task-causal claim."
            ),
        },
        "replay_record_adapter": {
            "adapter_import": "replay-record",
            "adapter_config": {
                "records_npz": "external_evaluator/replay_records.npz",
                "records_manifest": "external_evaluator/replay_records.json",
            },
            "contract": replay_record_template(),
        },
    }


def _load_analysis_bindings(
    analysis_inputs: str | Path,
    analysis_manifest: str | Path,
) -> _AnalysisBindings:
    inputs = Path(analysis_inputs)
    sidecar_path = Path(analysis_manifest)
    if not inputs.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError("analysis inputs or manifest is missing")
    sidecar = _load_self_fingerprinted_json(sidecar_path)
    if sidecar.get("schema_version") != ANALYSIS_INPUT_SCHEMA_VERSION:
        raise ValueError("causal rollout driver requires analysis_inputs_v2")
    inputs_hash = file_sha256(inputs)
    if sidecar.get("npz_sha256") != inputs_hash:
        raise ValueError("analysis input NPZ differs from its sidecar")
    for key in ("checkpoint_fingerprint", "formal_synergy_basis_fingerprint"):
        _require_hex64(key, sidecar.get(key))
    with np.load(inputs, allow_pickle=False) as data:
        required = {
            "sample_uids",
            "latents",
            "intervention_directions",
            "intervention_epsilons",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"analysis inputs lack rollout bindings: {missing}")
        uids = np.asarray(data["sample_uids"]).astype(str)
        latents = np.asarray(data["latents"], dtype=np.float32)
        directions = np.asarray(data["intervention_directions"], dtype=np.float32)
        epsilons = np.asarray(data["intervention_epsilons"], dtype=np.float32)
    if (
        uids.ndim != 1
        or uids.shape[0] == 0
        or len(set(uids.tolist())) != uids.shape[0]
        or any(not uid for uid in uids.tolist())
        or latents.ndim != 2
        or latents.shape[0] != uids.shape[0]
        or directions.ndim != 2
        or directions.shape[0] == 0
        or directions.shape[1] != latents.shape[1]
        or epsilons.ndim != 1
        or epsilons.shape[0] == 0
        or np.any(epsilons == 0.0)
        or not np.all(np.isfinite(latents))
        or not np.all(np.isfinite(directions))
        or not np.all(np.isfinite(epsilons))
    ):
        raise ValueError("analysis rollout bindings are malformed")
    return _AnalysisBindings(uids, latents, directions, epsilons, sidecar, inputs_hash)


def _validate_adapter(
    adapter: CausalRolloutAdapter,
    *,
    bindings: _AnalysisBindings,
) -> dict[str, Any]:
    if not isinstance(adapter, CausalRolloutAdapter):
        raise TypeError("object does not implement CausalRolloutAdapter")
    raw = adapter.descriptor()
    if not isinstance(raw, Mapping):
        raise ValueError("adapter descriptor must be a mapping")
    descriptor = dict(raw)
    if descriptor.get("schema_version") != ADAPTER_SCHEMA_VERSION:
        raise ValueError(f"adapter descriptor schema must be {ADAPTER_SCHEMA_VERSION}")
    if not str(descriptor.get("rollout_engine", "")).strip():
        raise ValueError("adapter rollout_engine must be non-empty")
    for key in (
        "checkpoint_fingerprint",
        "synergy_basis_fingerprint",
        "environment_fingerprint",
        "policy_abi_hash",
    ):
        descriptor[key] = _require_hex64(key, descriptor.get(key))
    if descriptor["checkpoint_fingerprint"] != bindings.sidecar["checkpoint_fingerprint"]:
        raise ValueError("adapter policy checkpoint differs from analysis inputs")
    if descriptor["synergy_basis_fingerprint"] != bindings.sidecar["formal_synergy_basis_fingerprint"]:
        raise ValueError("adapter synergy basis differs from analysis inputs")
    semantics = validate_physical_signal_semantics(descriptor.get("physical_signal_semantics"))
    if semantics != physical_signal_metadata():
        raise ValueError("adapter physical signal semantics are not exact")
    descriptor["physical_signal_semantics"] = semantics
    schemas = descriptor.get("outcome_schemas")
    if not isinstance(schemas, dict) or set(schemas) != set(REQUIRED_OUTCOMES):
        raise ValueError("adapter must describe exactly all required outcomes")
    muscle_names = (schemas.get("muscle_activation") or {}).get("feature_names")
    if not isinstance(muscle_names, list) or not muscle_names:
        raise ValueError("adapter outcome schemas lack ordered muscle names")
    mask = validate_activation_valid_mask(
        descriptor.get("activation_valid_mask"),
        expected_width=len(muscle_names),
    )
    if not np.any(mask):
        raise ValueError("adapter has no activation-valid muscle channels")
    descriptor["activation_valid_mask"] = mask.tolist()
    availability = descriptor.get("outcome_availability")
    if availability is None:
        # Compatibility for already-sealed third-party adapters.  Newly built
        # production adapters must publish this field explicitly; the driver
        # always writes the normalized mapping into its source manifest.
        availability = dict.fromkeys(REQUIRED_OUTCOMES, True)
    if (
        not isinstance(availability, Mapping)
        or set(availability) != set(REQUIRED_OUTCOMES)
        or any(type(availability[name]) is not bool for name in REQUIRED_OUTCOMES)
    ):
        raise ValueError("adapter outcome_availability must contain exact boolean entries for all outcomes")
    if not availability["muscle_excitation"] or not availability["muscle_activation"]:
        raise ValueError("a causal rollout adapter must measure excitation and activation")
    descriptor["outcome_availability"] = dict(availability)
    for name in REQUIRED_OUTCOMES:
        schema = schemas[name]
        if not isinstance(schema, Mapping):
            raise ValueError(f"adapter outcome schema {name!r} must be an object")
        declared = schema.get("available", availability[name])
        if type(declared) is not bool or declared is not availability[name]:
            raise ValueError(f"adapter outcome schema {name!r} availability differs from outcome_availability")
    diagnostic_names = (
        "muscle_excitation",
        "muscle_activation",
        "joint_position",
        "joint_velocity",
        "trunk_state",
        "racket_state",
    )
    diagnostic_complete = all(descriptor["outcome_availability"][name] for name in diagnostic_names)
    task_complete = all(descriptor["outcome_availability"][name] for name in REQUIRED_OUTCOMES)
    for key, expected in (
        ("stage2_diagnostic_outcomes_complete", diagnostic_complete),
        ("task_outcomes_complete", task_complete),
    ):
        supplied = descriptor.get(key, expected)
        if type(supplied) is not bool or supplied is not expected:
            raise ValueError(f"adapter {key} does not match outcome_availability")
        descriptor[key] = expected
    return descriptor


def _validate_outcomes(
    raw: Mapping[str, Any],
    *,
    descriptor: Mapping[str, Any],
    expected_shapes: Mapping[str, tuple[int, ...]] | None,
) -> dict[str, np.ndarray]:
    if not isinstance(raw, Mapping) or set(raw) != set(REQUIRED_OUTCOMES):
        raise ValueError("adapter rollout must return exactly the required measured outcomes")
    result: dict[str, np.ndarray] = {}
    for name in REQUIRED_OUTCOMES:
        value = np.asarray(raw[name])
        available = bool(descriptor["outcome_availability"][name])
        if value.ndim != 1 or not np.all(np.isfinite(value)):
            raise ValueError(f"adapter outcome {name!r} must be a finite ordered vector")
        if available and value.size == 0:
            raise ValueError(f"available adapter outcome {name!r} cannot be empty")
        if not available and value.size != 0:
            raise ValueError(f"unavailable adapter outcome {name!r} must be an empty vector, not a numeric placeholder")
        if expected_shapes is not None and value.shape != expected_shapes[name]:
            raise ValueError(f"adapter outcome {name!r} shape changed between paired rollouts")
        if name == "muscle_excitation":
            value = validate_unit_muscle_excitation(value)
        if name == "muscle_activation":
            value = validate_unit_muscle_activation(value)
        result[name] = value.astype(np.float32, copy=False)
    if result["muscle_excitation"].shape != result["muscle_activation"].shape:
        raise ValueError("muscle excitation and activation must share one ordered muscle ABI")
    return result


def _validate_outcome_schemas(
    schemas: Mapping[str, Any],
    *,
    shapes: Mapping[str, tuple[int, ...]],
    activation_valid_mask: Any,
    outcome_availability: Mapping[str, bool],
) -> None:
    if set(schemas) != set(REQUIRED_OUTCOMES):
        raise ValueError("outcome schemas are incomplete")
    for name in REQUIRED_OUTCOMES:
        schema = schemas[name]
        width = int(np.prod(shapes[name], dtype=np.int64))
        if not isinstance(schema, Mapping):
            raise ValueError(f"outcome schema {name!r} must be an object")
        feature_names = schema.get("feature_names")
        units = schema.get("units")
        available = outcome_availability[name]
        declared = schema.get("available", available)
        if type(declared) is not bool or declared is not available or available != (width > 0):
            raise ValueError(
                f"outcome schema {name!r} availability must agree with descriptor and measured vector width"
            )
        if (
            not isinstance(feature_names, list)
            or len(feature_names) != width
            or len({str(value) for value in feature_names}) != width
            or any(not str(value).strip() for value in feature_names)
            or not isinstance(units, list)
            or len(units) != width
            or any(not str(value).strip() for value in units)
            or not str(schema.get("coordinate_frame", "")).strip()
            or schema.get("semantics") != _OUTCOME_SEMANTICS[name]
        ):
            raise ValueError(
                f"outcome schema {name!r} must bind every ordered value and use semantics {_OUTCOME_SEMANTICS[name]!r}"
            )
    excitation = schemas["muscle_excitation"]
    activation = schemas["muscle_activation"]
    if excitation["feature_names"] != activation["feature_names"]:
        raise ValueError("excitation and activation ordered muscle names differ")
    width = len(excitation["feature_names"])
    if excitation["units"] != ["unit_interval"] * width or activation["units"] != ["unit_interval"] * width:
        raise ValueError("muscle outcome units must be unit_interval")
    validate_activation_valid_mask(activation_valid_mask, expected_width=width)


def _restore_and_seed(
    adapter: CausalRolloutAdapter,
    *,
    snapshot: Any,
    expected_snapshot_bytes: bytes,
    seed: int,
) -> str:
    adapter.restore_snapshot(snapshot)
    captured = adapter.capture_snapshot()
    if _snapshot_bytes(adapter, captured) != expected_snapshot_bytes:
        raise ValueError("adapter exact snapshot restore verification failed")
    adapter.set_common_random_seed(int(seed))
    return _require_hex64("random_state_fingerprint", adapter.random_state_fingerprint())


def _snapshot_bytes(adapter: CausalRolloutAdapter, snapshot: Any) -> bytes:
    value = adapter.snapshot_to_bytes(snapshot)
    if not isinstance(value, bytes) or not value:
        raise ValueError("adapter snapshot serialization must return non-empty bytes")
    return value


def _sample_seed(base_seed: int, sample_uid: str) -> int:
    payload = f"{int(base_seed)}\0{sample_uid}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _validate_job_shape(job: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(job, Mapping) or job.get("schema_version") != JOB_SCHEMA_VERSION:
        raise ValueError(f"job schema must be {JOB_SCHEMA_VERSION}")
    required = {
        "schema_version",
        "analysis_inputs",
        "analysis_manifest",
        "output_dir",
        "base_seed",
        "adapter_import",
        "adapter_config",
    }
    missing = sorted(required - set(job))
    if missing:
        raise ValueError(f"causal rollout job is missing {missing}")
    payload = {key: job[key] for key in required}
    if not isinstance(payload["adapter_config"], Mapping):
        raise ValueError("adapter_config must be an object")
    payload["adapter_config"] = dict(payload["adapter_config"])
    if int(payload["base_seed"]) < 0:
        raise ValueError("base_seed must be non-negative")
    for key in ("analysis_inputs", "analysis_manifest", "output_dir", "adapter_import"):
        if not str(payload[key]).strip():
            raise ValueError(f"job field {key} must be non-empty")
    return payload


def _load_self_fingerprinted_json(path: Path) -> dict[str, Any]:
    payload = load_json_strict(path)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    supplied = payload.get("manifest_fingerprint")
    content = {key: value for key, value in payload.items() if key != "manifest_fingerprint"}
    if supplied != canonical_json_sha256(content):
        raise ValueError(f"JSON fingerprint mismatch: {path}")
    return payload


def _require_hex64(name: str, value: Any) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise ValueError(f"{name} must be lowercase 64-hex")
    return text


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_template(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"template target already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(job_template(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _fsync_file(temporary)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    template_parser = commands.add_parser("template", help="write an explicit adapter job template")
    template_parser.add_argument("--output", type=Path, required=True)
    replay_parser = commands.add_parser("replay-contract", help="print the strict replay source contract")
    replay_parser.add_argument("--output", type=Path, default=None)
    evaluate_parser = commands.add_parser("evaluate", help="run paired evaluator calls and publish records")
    evaluate_parser.add_argument("--job-config", type=Path, required=True)
    evaluate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate bindings/descriptor only; never capture state, evaluate, or publish",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "template":
        _write_template(args.output)
        print(json.dumps({"template": str(args.output.resolve()), "rollouts_executed": False}, indent=2))
        return 0
    if args.command == "replay-contract":
        payload = replay_record_template()
        text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output is None:
            print(text, end="")
        else:
            if args.output.exists():
                raise FileExistsError(f"replay contract target already exists: {args.output}")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        return 0
    job = load_json_strict(args.job_config)
    result = validate_job(job=job) if args.dry_run else evaluate_job(job)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

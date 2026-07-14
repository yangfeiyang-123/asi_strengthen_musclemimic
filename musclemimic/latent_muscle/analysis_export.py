"""Export checkpoint-bound latent/synergy analysis inputs from held-out data.

The exporter is deliberately offline: it can measure posterior representations,
decoder Jacobians, NMF coefficients, and decoder interventions without creating
environment-level causal claims.  Optional causal rollout evidence is accepted
only through a separately fingerprinted production artifact.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from musclemimic.distill.dataset import SequenceDistillDataset
from musclemimic.distill.provenance import (
    canonical_json_sha256,
    file_sha256,
    validate_dataset_manifest,
)
from musclemimic.latent_muscle.checkpoint import load_latent_checkpoint
from musclemimic.latent_muscle.networks import PosteriorEncoder
from musclemimic.latent_muscle.runtime import LatentMuscleRuntime
from musclemimic.synergy.basis_artifact import load_synergy_basis
from musclemimic.synergy.nmf import transform_nmf
from musclemimic.synergy.schema import EXCITATION_SIGNAL_KIND

ANALYSIS_INPUT_SCHEMA_VERSION = "latent_synergy_analysis_inputs_v2"
CAUSAL_EVIDENCE_SCHEMA_VERSION = "latent_synergy_causal_interventions_v1"
CORE_ARRAY_FIELDS = (
    "latents",
    "synergy_coefficients",
    "target_synergy_coefficients",
    "decoder_jacobians",
    "phase_ids",
    "train_mask",
    "sample_uids",
    "teacher_physical_excitation",
    "baseline_physical_excitation",
    "perturbed_physical_excitation",
    "intervention_epsilons",
    "intervention_directions",
    "intervention_direction_names",
)
OPTIONAL_CAUSAL_FIELDS = (
    "causal_effects",
    "causal_effect_names",
)
STAGE2_DIAGNOSTIC_CAUSAL_OUTCOMES = (
    "muscle_excitation",
    "muscle_activation",
    "joint_position",
    "joint_velocity",
    "trunk_state",
    "racket_state",
)


def export_analysis_inputs(
    *,
    latent_checkpoint: str | Path,
    dataset_dir: str | Path,
    val_dataset_dir: str | Path | None,
    synergy_basis: str | Path,
    synergy_basis_fingerprint: str,
    output_npz: str | Path,
    max_samples: int = 1024,
    max_intervention_directions: int = 8,
    epsilons: Sequence[float] = (-1.0, -0.5, 0.5, 1.0),
    batch_size: int = 64,
    require_all_phases: bool = False,
    causal_interventions_npz: str | Path | None = None,
    causal_interventions_manifest: str | Path | None = None,
    require_causal_interventions: bool = False,
) -> dict[str, Any]:
    """Write ``analysis_inputs.npz`` and a strict provenance sidecar."""

    if int(max_samples) < 5:
        raise ValueError("analysis export requires max_samples >= 5")
    if int(max_intervention_directions) <= 0:
        raise ValueError("max_intervention_directions must be positive")
    if int(batch_size) <= 0:
        raise ValueError("analysis export batch_size must be positive")
    epsilon = np.asarray(tuple(float(value) for value in epsilons), dtype=np.float32)
    if epsilon.size == 0 or not np.all(np.isfinite(epsilon)) or np.any(epsilon == 0.0):
        raise ValueError("intervention epsilons must be finite, non-zero, and non-empty")

    checkpoint = load_latent_checkpoint(latent_checkpoint)
    runtime = LatentMuscleRuntime(checkpoint)
    formal_basis = load_synergy_basis(synergy_basis)
    supplied_fingerprint = str(synergy_basis_fingerprint)
    if formal_basis.fingerprint != supplied_fingerprint:
        raise ValueError("analysis synergy basis fingerprint differs from the formal artifact")
    if formal_basis.manifest.get("signal_kind") != EXCITATION_SIGNAL_KIND:
        raise ValueError("analysis export requires a physical_excitation_unit basis")
    if tuple(formal_basis.muscle_names) != runtime.body_actuator_names:
        raise ValueError("analysis synergy basis actuator names/order differ from the latent runtime")
    basis_binding = validate_runtime_basis_binding(
        runtime,
        formal_basis_fingerprint=formal_basis.fingerprint,
    )

    training = checkpoint.get("training_provenance")
    if not isinstance(training, Mapping):
        raise ValueError("analysis export requires latent training_provenance.json")
    dataset_manifest = validate_dataset_manifest(dataset_dir)
    dataset_fingerprint = dataset_manifest.get("manifest_fingerprint")
    if dataset_fingerprint != training.get("dataset_manifest_fingerprint"):
        raise ValueError("analysis dataset fingerprint differs from latent training provenance")
    validation_dataset_manifest = None
    if val_dataset_dir is not None:
        validation_dataset_manifest = validate_dataset_manifest(val_dataset_dir)
        if validation_dataset_manifest.get("manifest_fingerprint") != training.get(
            "validation_dataset_manifest_fingerprint"
        ):
            raise ValueError("analysis validation dataset fingerprint differs from latent training provenance")
    elif training.get("validation_dataset_manifest_fingerprint") is not None:
        raise ValueError("latent checkpoint used an explicit validation dataset; --val-dataset-dir is required")
    teacher_sha256 = (training.get("teacher_checkpoint") or {}).get("sha256")
    if not teacher_sha256 or str(formal_basis.manifest.get("teacher_checkpoint_fingerprint")) != str(teacher_sha256):
        raise ValueError("formal synergy basis and latent training use different teacher checkpoints")

    arrays, train_mask = _load_checkpoint_split_arrays(
        dataset_dir,
        val_dataset_dir=val_dataset_dir,
        checkpoint=checkpoint,
        target_actuator_names=runtime.body_actuator_names,
    )
    sample_uids = stable_sample_uids(arrays)
    selected = _stable_sample_selection(sample_uids, int(max_samples))
    arrays = {name: np.asarray(value)[selected] for name, value in arrays.items()}
    train_mask = np.asarray(train_mask, dtype=bool)[selected]
    sample_uids = np.asarray(sample_uids)[selected]
    _validate_selected_evidence(
        phase_ids=arrays["phase_id"],
        train_mask=train_mask,
        sample_uids=sample_uids,
        require_all_phases=bool(require_all_phases),
    )

    states = np.asarray(arrays["student_obs"], dtype=np.float32)
    references = np.asarray(arrays["reference_features"], dtype=np.float32)
    physical_target = np.asarray(arrays["muscle_excitation"], dtype=np.float32)
    posterior = PosteriorEncoder(
        latent_dim=int(runtime.latent_dim),
        hidden_layer_dims=tuple(int(value) for value in runtime.config.get("hidden_layer_dims", (512, 256))),
        sigma_min=float(runtime.sigma_min),
        sigma_max=float(runtime.sigma_max),
    )
    latents, physical, decoder_coefficients, jacobians = _evaluate_checkpoint_batches(
        runtime=runtime,
        posterior=posterior,
        encoder_variables=checkpoint["encoder_variables"],
        states=states,
        references=references,
        batch_size=int(batch_size),
    )
    target_coefficients, _target_reconstruction = transform_nmf(
        physical_target,
        formal_basis.basis,
    )
    directions = _principal_intervention_directions(
        latents,
        max_directions=int(max_intervention_directions),
    )
    perturbed_physical = _evaluate_physical_interventions(
        runtime=runtime,
        states=states,
        latents=latents,
        directions=directions,
        epsilons=epsilon,
        batch_size=int(batch_size),
    )
    direction_names = np.asarray(
        [f"latent_covariance_pc_{index}" for index in range(directions.shape[0])],
        dtype=np.str_,
    )
    output_arrays: dict[str, np.ndarray] = {
        "latents": latents.astype(np.float32),
        # This is the NNLS transform of observed teacher excitation under the
        # formal W, and is comparable for direct and constrained decoders.
        "synergy_coefficients": target_coefficients.astype(np.float32),
        "target_synergy_coefficients": target_coefficients.astype(np.float32),
        "decoder_jacobians": jacobians.astype(np.float32),
        "phase_ids": np.asarray(arrays["phase_id"], dtype=np.int32),
        "train_mask": train_mask,
        "sample_uids": np.asarray(sample_uids, dtype=np.str_),
        "teacher_physical_excitation": physical_target,
        "baseline_physical_excitation": physical.astype(np.float32),
        "perturbed_physical_excitation": perturbed_physical.astype(np.float32),
        "intervention_epsilons": epsilon,
        "intervention_directions": directions.astype(np.float32),
        "intervention_direction_names": direction_names,
    }
    if decoder_coefficients.shape[1] > 0:
        output_arrays["decoder_synergy_coefficients"] = decoder_coefficients.astype(np.float32)

    causal_status: dict[str, Any] = {
        "status": "not_provided_optional",
        "required_for_analysis": bool(require_causal_interventions),
        "offline_intervention_verified": True,
        "causal_rollout_verified": False,
        "stage2_diagnostic_outcomes_complete": False,
        "task_outcomes_complete": False,
        "reason": ("offline decoder interventions do not establish joint/racket/task causal effects"),
    }
    if (causal_interventions_npz is None) != (causal_interventions_manifest is None):
        raise ValueError("causal evidence requires both --causal-interventions-npz and manifest")
    if require_causal_interventions and causal_interventions_npz is None:
        raise ValueError("required environment-rollout causal evidence was not supplied")
    if causal_interventions_npz is not None:
        causal_arrays, causal_status = load_optional_causal_evidence(
            causal_interventions_npz,
            causal_interventions_manifest,
            checkpoint_fingerprint=runtime.checkpoint_fingerprint,
            synergy_basis_fingerprint=formal_basis.fingerprint,
            sample_uids=sample_uids,
            directions=directions,
            epsilons=epsilon,
            require_stage2_diagnostic_outcomes=bool(require_causal_interventions),
        )
        causal_status["required_for_analysis"] = bool(require_causal_interventions)
        causal_status["offline_intervention_verified"] = True
        causal_status["causal_rollout_verified"] = True
        output_arrays.update(causal_arrays)

    output = Path(output_npz)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **output_arrays)
    manifest = {
        "schema_version": ANALYSIS_INPUT_SCHEMA_VERSION,
        "npz_path": str(output.resolve()),
        "npz_sha256": file_sha256(output),
        "checkpoint_dir": str(Path(latent_checkpoint).resolve()),
        "checkpoint_fingerprint": runtime.checkpoint_fingerprint,
        "decoder_type": runtime.decoder_type,
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "dataset_manifest_fingerprint": dataset_fingerprint,
        "validation_dataset_dir": (None if val_dataset_dir is None else str(Path(val_dataset_dir).resolve())),
        "validation_dataset_manifest_fingerprint": (
            None if validation_dataset_manifest is None else validation_dataset_manifest.get("manifest_fingerprint")
        ),
        "teacher_checkpoint_sha256": teacher_sha256,
        "formal_synergy_basis_path": str(formal_basis.path.resolve()),
        "formal_synergy_basis_fingerprint": formal_basis.fingerprint,
        "basis_binding": basis_binding,
        "num_samples": int(latents.shape[0]),
        "latent_dim": int(latents.shape[1]),
        "synergy_dim": int(target_coefficients.shape[1]),
        "num_intervention_directions": int(directions.shape[0]),
        "core_fields": list(CORE_ARRAY_FIELDS),
        "optional_fields_present": sorted(set(output_arrays) - set(CORE_ARRAY_FIELDS)),
        "sample_uid_fingerprint": canonical_json_sha256(sample_uids.tolist()),
        "causal_evidence": causal_status,
    }
    manifest["manifest_fingerprint"] = canonical_json_sha256(manifest)
    sidecar = output.with_suffix(".json")
    sidecar.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def validate_runtime_basis_binding(
    runtime: LatentMuscleRuntime,
    *,
    formal_basis_fingerprint: str,
) -> dict[str, Any]:
    """Validate config/embedded/runtime W against the formal artifact."""

    expected = str(runtime.config.get("synergy_basis_expected_fingerprint", ""))
    if expected != str(formal_basis_fingerprint):
        raise ValueError("latent config expected synergy fingerprint differs from the formal artifact")
    if runtime.decoder_type == "direct":
        if runtime.synergy_basis is not None:
            raise ValueError("direct runtime unexpectedly embeds a synergy basis")
        return {
            "verified": True,
            "decoder_type": "direct",
            "formal_synergy_basis_fingerprint": str(formal_basis_fingerprint),
            "config_synergy_basis_expected_fingerprint": expected,
            "runtime_synergy_basis_fingerprint": None,
            "runtime_synergy_basis_source_fingerprint": None,
        }
    if runtime.synergy_basis is None:
        raise ValueError("synergy runtime is missing its embedded fixed basis")
    runtime_fingerprint = str(runtime.synergy_basis.fingerprint)
    configured_runtime = str(runtime.config.get("synergy_basis_fingerprint", ""))
    source_fingerprint = str(runtime.synergy_basis.manifest.get("source_fingerprint", ""))
    control_fingerprint = str(runtime.control_manifest.get("synergy_basis_fingerprint", ""))
    if not (
        runtime_fingerprint == configured_runtime == control_fingerprint
        and source_fingerprint == str(formal_basis_fingerprint)
    ):
        raise ValueError("runtime/checkpoint/config synergy basis fingerprints are not mutually bound")
    return {
        "verified": True,
        "decoder_type": runtime.decoder_type,
        "formal_synergy_basis_fingerprint": str(formal_basis_fingerprint),
        "config_synergy_basis_expected_fingerprint": expected,
        "runtime_synergy_basis_fingerprint": runtime_fingerprint,
        "runtime_synergy_basis_source_fingerprint": source_fingerprint,
    }


def load_optional_causal_evidence(
    npz_path: str | Path,
    manifest_path: str | Path | None,
    *,
    checkpoint_fingerprint: str,
    synergy_basis_fingerprint: str,
    sample_uids: np.ndarray,
    directions: np.ndarray,
    epsilons: np.ndarray,
    require_stage2_diagnostic_outcomes: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load genuine environment-rollout effects or fail closed."""

    source = Path(npz_path)
    manifest_file = Path(manifest_path) if manifest_path is not None else source.with_suffix(".json")
    if not source.is_file() or not manifest_file.is_file():
        raise ValueError("causal intervention artifact or manifest is missing")
    # Import lazily to avoid a module cycle: the builder imports the public
    # analysis schema constants but never invokes this loader.
    from musclemimic.latent_muscle.causal_rollout_artifact import (
        validate_causal_rollout_artifact,
    )

    manifest = validate_causal_rollout_artifact(source, manifest_file)
    availability = manifest.get("outcome_availability")
    if not isinstance(availability, dict):
        raise ValueError("causal intervention manifest lacks outcome availability")
    missing_diagnostic = [name for name in STAGE2_DIAGNOSTIC_CAUSAL_OUTCOMES if availability.get(name) is not True]
    stage2_diagnostic_complete = not missing_diagnostic
    task_outcomes_complete = bool(availability) and all(value is True for value in availability.values())
    if require_stage2_diagnostic_outcomes and not stage2_diagnostic_complete:
        raise ValueError(f"required Stage-2 causal diagnostic outcomes are unavailable: {missing_diagnostic}")
    if manifest.get("checkpoint_fingerprint") != str(checkpoint_fingerprint):
        raise ValueError("causal intervention checkpoint fingerprint mismatch")
    if manifest.get("synergy_basis_fingerprint") != str(synergy_basis_fingerprint):
        raise ValueError("causal intervention synergy basis fingerprint mismatch")
    with np.load(source, allow_pickle=False) as data:
        required = {
            "causal_effects",
            "sample_uids",
            "intervention_directions",
            "intervention_epsilons",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"causal intervention artifact is missing {missing}")
        effects = np.asarray(data["causal_effects"], dtype=np.float64)
        causal_uids = np.asarray(data["sample_uids"]).astype(str)
        causal_directions = np.asarray(data["intervention_directions"], dtype=np.float64)
        causal_epsilons = np.asarray(data["intervention_epsilons"], dtype=np.float64)
        names = np.asarray(data["causal_effect_names"]).astype(str) if "causal_effect_names" in data.files else None
    if not np.array_equal(causal_uids, np.asarray(sample_uids).astype(str)):
        raise ValueError("causal intervention sample_uids are not aligned")
    if not np.array_equal(causal_directions, np.asarray(directions, dtype=np.float64)):
        raise ValueError("causal intervention directions are not aligned")
    if not np.array_equal(causal_epsilons, np.asarray(epsilons, dtype=np.float64)):
        raise ValueError("causal intervention epsilons are not aligned")
    expected_leading = (len(sample_uids), directions.shape[0], len(epsilons))
    if (
        effects.ndim < 4
        or effects.shape[:3] != expected_leading
        or effects.shape[-1] <= 0
        or not np.all(np.isfinite(effects))
    ):
        raise ValueError("causal_effects must be finite [sample,direction,epsilon,outcome...]")
    result = {"causal_effects": effects.astype(np.float32)}
    if names is not None:
        if names.ndim != 1 or names.shape[0] != effects.shape[-1]:
            raise ValueError("causal_effect_names must match the final outcome axis")
        result["causal_effect_names"] = names.astype(np.str_)
    return result, {
        "status": "verified_environment_rollout",
        "offline_intervention_verified": True,
        "causal_rollout_verified": True,
        "outcome_availability": dict(availability),
        "stage2_diagnostic_outcomes_complete": stage2_diagnostic_complete,
        "task_outcomes_complete": task_outcomes_complete,
        "artifact_path": str(source.resolve()),
        "artifact_sha256": file_sha256(source),
        "manifest_path": str(manifest_file.resolve()),
        "manifest_sha256": file_sha256(manifest_file),
    }


def stable_sample_uids(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    required = ("motion_uid", "rollout_uid", "rollout_step", "env_index")
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError(f"analysis export requires stable ID fields {missing}")
    columns = [np.asarray(arrays[name]) for name in required]
    size = int(columns[0].shape[0])
    if any(column.shape != (size,) for column in columns):
        raise ValueError("stable ID fields must be aligned rank-1 arrays")
    if any(np.any(column < 0) for column in columns):
        raise ValueError("stable ID fields cannot contain negative values")
    values = []
    for row in zip(*(column.astype(np.int64).tolist() for column in columns), strict=True):
        values.append(hashlib.sha256(json.dumps(row, separators=(",", ":")).encode("utf-8")).hexdigest())
    if len(set(values)) != len(values):
        raise ValueError("stable sample identity is not unique")
    return np.asarray(values, dtype=np.str_)


def _load_checkpoint_split_arrays(
    dataset_dir: str | Path,
    *,
    val_dataset_dir: str | Path | None,
    checkpoint: Mapping[str, Any],
    target_actuator_names: Sequence[str],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    root = Path(dataset_dir)
    kwargs = {
        "seed": int(checkpoint["config"].get("seed", 0)),
        "target_actuator_names": tuple(target_actuator_names),
        "require_stable_ids": True,
    }
    train = SequenceDistillDataset(root, split="train", **kwargs)
    split = checkpoint.get("split_manifest")
    if not isinstance(split, Mapping):
        raise ValueError("latent checkpoint is missing motion_split.json")
    motion_field = str(split.get("motion_field", "motion_uid"))
    validation_root = (
        Path(val_dataset_dir) if val_dataset_dir is not None else (root if sorted(root.glob("val_*.npz")) else None)
    )
    if validation_root is not None:
        validation = SequenceDistillDataset(validation_root, split="val", **kwargs)
        fields = _required_export_fields(motion_field)
        _require_dataset_fields(train, fields)
        _require_dataset_fields(validation, fields)
        arrays = {
            name: np.concatenate(
                [np.asarray(train.arrays[name]), np.asarray(validation.arrays[name])],
                axis=0,
            )
            for name in fields
        }
        train_mask = np.concatenate(
            [
                np.ones(train.num_samples, dtype=bool),
                np.zeros(validation.num_samples, dtype=bool),
            ]
        )
    else:
        fields = _required_export_fields(motion_field)
        _require_dataset_fields(train, fields)
        arrays = {name: np.asarray(train.arrays[name]) for name in fields}
        train_ids = {int(value) for value in split.get("train_motion_ids", ())}
        validation_ids = {int(value) for value in split.get("val_motion_ids", ())}
        if not train_ids or not validation_ids or train_ids & validation_ids:
            raise ValueError("analysis export requires a non-empty leakage-free train/validation motion split")
        motion_ids = np.asarray(arrays[motion_field], dtype=np.int64)
        if set(np.unique(motion_ids).tolist()) != train_ids | validation_ids:
            raise ValueError("checkpoint motion split differs from analysis dataset")
        train_mask = np.isin(motion_ids, sorted(train_ids))
    actual_train_ids = set(np.unique(np.asarray(arrays[motion_field])[train_mask]).astype(int).tolist())
    actual_val_ids = set(np.unique(np.asarray(arrays[motion_field])[~train_mask]).astype(int).tolist())
    if actual_train_ids != {int(value) for value in split.get("train_motion_ids", ())}:
        raise ValueError("analysis train motions differ from checkpoint split")
    if actual_val_ids != {int(value) for value in split.get("val_motion_ids", ())}:
        raise ValueError("analysis validation motions differ from checkpoint split")
    return arrays, train_mask


def _required_export_fields(motion_field: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                "student_obs",
                "reference_features",
                "muscle_excitation",
                "phase_id",
                "motion_uid",
                "rollout_uid",
                "rollout_step",
                "env_index",
                str(motion_field),
            )
        )
    )


def _require_dataset_fields(dataset: SequenceDistillDataset, fields: Sequence[str]) -> None:
    missing = [name for name in fields if name not in dataset.arrays]
    if missing:
        raise ValueError(f"analysis dataset split {dataset.split!r} is missing {missing}")


def _stable_sample_selection(sample_uids: np.ndarray, max_samples: int) -> np.ndarray:
    values = np.asarray(sample_uids).astype(str)
    if values.shape[0] <= int(max_samples):
        return np.argsort(values, kind="stable")
    scores = np.asarray([hashlib.sha256(f"latent-analysis-v1:{value}".encode()).hexdigest() for value in values])
    selected = np.argsort(scores, kind="stable")[: int(max_samples)]
    return selected[np.argsort(values[selected], kind="stable")]


def _validate_selected_evidence(
    *,
    phase_ids: np.ndarray,
    train_mask: np.ndarray,
    sample_uids: np.ndarray,
    require_all_phases: bool,
) -> None:
    phases = np.asarray(phase_ids)
    mask = np.asarray(train_mask)
    uids = np.asarray(sample_uids)
    size = len(uids)
    if size < 5 or phases.shape != (size,) or mask.shape != (size,):
        raise ValueError("selected analysis arrays are not sample-aligned")
    if len(set(uids.astype(str).tolist())) != size:
        raise ValueError("selected analysis sample_uids are not unique")
    if not np.all(np.isfinite(phases)) or not np.all(phases == np.floor(phases)):
        raise ValueError("phase_ids must be finite integers")
    unknown = sorted(set(phases.astype(int).tolist()) - set(range(6)))
    if unknown:
        raise ValueError(f"phase_ids contain unknown values: {unknown}")
    if np.sum(mask) < 2 or np.sum(~mask.astype(bool)) < 2:
        raise ValueError(
            "stable analysis sample selection needs at least two train and two validation rows; increase --max-samples"
        )
    missing = sorted(set(range(6)) - set(phases.astype(int).tolist()))
    if require_all_phases and missing:
        raise ValueError(f"analysis export is missing phase IDs {missing}")


def _evaluate_checkpoint_batches(
    *,
    runtime: LatentMuscleRuntime,
    posterior: PosteriorEncoder,
    encoder_variables: Any,
    states: np.ndarray,
    references: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    latents: list[np.ndarray] = []
    physical: list[np.ndarray] = []
    coefficients: list[np.ndarray] = []
    jacobians: list[np.ndarray] = []
    for start in range(0, len(states), int(batch_size)):
        stop = min(start + int(batch_size), len(states))
        state = states[start:stop]
        reference = references[start:stop]
        normalized = runtime.normalize_jax(jnp.asarray(state))
        mu, _raw_sigma = posterior.apply(
            encoder_variables,
            normalized,
            jnp.asarray(reference, dtype=jnp.float32),
        )
        mu_numpy = np.asarray(jax.device_get(mu), dtype=np.float32)
        components = runtime.decode_components_numpy(state, mu_numpy)
        jacobian = runtime.decoder_jacobian_numpy(
            state,
            mu_numpy,
            output="physical_excitation",
        )
        latents.append(mu_numpy)
        physical.append(np.asarray(components.physical_excitation, dtype=np.float32))
        coefficients.append(np.asarray(components.synergy_coefficients, dtype=np.float32))
        jacobians.append(np.asarray(jacobian, dtype=np.float32))
    return tuple(np.concatenate(values, axis=0) for values in (latents, physical, coefficients, jacobians))


def _principal_intervention_directions(
    latents: np.ndarray,
    *,
    max_directions: int,
) -> np.ndarray:
    z = np.asarray(latents, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] < 2 or z.shape[1] <= 0 or not np.all(np.isfinite(z)):
        raise ValueError("latents must be a finite non-empty rank-2 matrix")
    _u, _singular, vh = np.linalg.svd(
        z - np.mean(z, axis=0, keepdims=True),
        full_matrices=False,
    )
    count = min(int(max_directions), z.shape[1], vh.shape[0])
    directions = vh[:count]
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    if count <= 0 or np.any(norms <= 1e-12):
        raise ValueError("latent covariance produced no valid intervention direction")
    return directions / norms


def _evaluate_physical_interventions(
    *,
    runtime: LatentMuscleRuntime,
    states: np.ndarray,
    latents: np.ndarray,
    directions: np.ndarray,
    epsilons: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    output: list[np.ndarray] = []
    count = int(directions.shape[0] * len(epsilons))
    for start in range(0, len(states), int(batch_size)):
        stop = min(start + int(batch_size), len(states))
        state = states[start:stop]
        z = latents[start:stop]
        perturbed = z[:, None, None, :] + directions[None, :, None, :] * epsilons[None, None, :, None]
        flat_z = perturbed.reshape((-1, z.shape[1]))
        repeated_state = np.repeat(state, count, axis=0)
        components = runtime.decode_components_numpy(repeated_state, flat_z)
        physical = np.asarray(components.physical_excitation, dtype=np.float32)
        output.append(physical.reshape((len(state), directions.shape[0], len(epsilons), physical.shape[-1])))
    return np.concatenate(output, axis=0)

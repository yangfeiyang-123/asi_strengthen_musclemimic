"""Checkpoint IO for latent posterior/prior/decoder training."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from flax import serialization

from musclemimic.distill.action_schema import actuator_schema_hash, ordered_schema_hash
from musclemimic.distill.body_obs_schema import validate_body_obs_schema
from musclemimic.distill.provenance import canonical_json_sha256
from musclemimic.latent_muscle.action_mask import ActionMask


def save_latent_checkpoint(
    checkpoint_dir: str | Path,
    *,
    encoder_variables: Any,
    prior_variables: Any,
    decoder_variables: Any,
    optimizer_state: Any,
    action_mask: ActionMask,
    config: dict[str, Any],
    train_metrics: list[dict[str, Any]],
    eval_metrics: dict[str, Any],
    obs_norm: dict[str, Any] | None = None,
    action_norm: dict[str, Any] | None = None,
    action_schema: dict[str, Any] | None = None,
    state_schema: dict[str, Any] | None = None,
    body_obs_schema: dict[str, Any] | None = None,
    split_manifest: dict[str, Any] | None = None,
    training_provenance: dict[str, Any] | None = None,
    synergy_basis: Any | None = None,
    frozen_body_decoder: Any | None = None,
) -> Path:
    """Persist a complete latent distillation checkpoint directory."""
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    config_payload = dict(config)
    synergy_payload = _canonical_synergy_payload(synergy_basis)
    from musclemimic.synergy.frozen_decoder import FrozenBodyDecoder

    if frozen_body_decoder is not None and not isinstance(
        frozen_body_decoder, FrozenBodyDecoder
    ):
        raise TypeError("frozen_body_decoder must be a FrozenBodyDecoder")
    portable_files = (
        "frozen_body_decoder.npz",
        "frozen_body_decoder.json",
        "body_synergy_contract.json",
    )
    stale_synergy_files = [
        filename
        for filename in ("synergy_basis.npz", "synergy_basis.json")
        if (path / filename).is_file()
    ]
    if synergy_payload is None and stale_synergy_files:
        raise ValueError(
            "refusing to save a direct checkpoint over stale synergy artifacts; "
            f"use a fresh output directory: {stale_synergy_files}"
        )
    stale_portable_files = [
        filename for filename in portable_files if (path / filename).is_file()
    ]
    if frozen_body_decoder is None and stale_portable_files:
        raise ValueError(
            "refusing to save without a portable decoder over stale frozen artifacts; "
            f"use a fresh output directory: {stale_portable_files}"
        )
    if frozen_body_decoder is not None and synergy_payload is not None:
        raise ValueError(
            "portable frozen decoder and legacy standalone synergy basis cannot be "
            "stored in the same latent checkpoint"
        )
    if frozen_body_decoder is not None and stale_synergy_files:
        raise ValueError(
            "refusing to mix a portable decoder with stale legacy synergy files; "
            f"use a fresh output directory: {stale_synergy_files}"
        )
    if synergy_payload is not None and stale_portable_files:
        raise ValueError(
            "refusing to mix a legacy synergy checkpoint with stale portable files; "
            f"use a fresh output directory: {stale_portable_files}"
        )
    if synergy_payload is not None:
        config_payload["synergy_basis_fingerprint"] = synergy_payload["fingerprint"]
        config_payload["synergy_dim"] = int(synergy_payload["basis"].shape[1])
    if frozen_body_decoder is not None:
        contract = frozen_body_decoder.body_synergy_contract
        config_payload.update(
            {
                "frozen_body_decoder_fingerprint": (
                    frozen_body_decoder.artifact_fingerprint
                ),
                "body_synergy_contract_fingerprint": contract.contract_fingerprint,
                "body_synergy_portable_core_fingerprint": (
                    contract.portable_decoder_core_fingerprint
                ),
                "synergy_basis_fingerprint": contract.runtime_basis_fingerprint,
                "synergy_dim": frozen_body_decoder.synergy_dim,
                "synergy_residual_dim": frozen_body_decoder.residual_dim,
                "legacy_synergy_decoder_ablation": False,
            }
        )
    _write_msgpack(path / "encoder.msgpack", encoder_variables)
    _write_msgpack(path / "prior.msgpack", prior_variables)
    _write_msgpack(path / "decoder.msgpack", decoder_variables)
    _write_msgpack(path / "optimizer_state.msgpack", optimizer_state)
    (path / "action_mask.json").write_text(
        json.dumps(_action_mask_manifest(action_mask), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (path / "latent_config.yaml").write_text(
        json.dumps(_jsonable(config_payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if obs_norm is not None:
        (path / "obs_norm.json").write_text(json.dumps(_jsonable(obs_norm), indent=2, sort_keys=True), encoding="utf-8")
    if action_norm is not None:
        (path / "action_norm.json").write_text(
            json.dumps(_jsonable(action_norm), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if action_schema is not None:
        (path / "action_schema.json").write_text(
            json.dumps(_jsonable(action_schema), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if state_schema is not None:
        (path / "state_schema.json").write_text(
            json.dumps(_jsonable(state_schema), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if body_obs_schema is not None:
        (path / "body_obs_schema.json").write_text(
            json.dumps(_jsonable(body_obs_schema), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if split_manifest is not None:
        (path / "motion_split.json").write_text(
            json.dumps(_jsonable(split_manifest), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if training_provenance is not None:
        (path / "training_provenance.json").write_text(
            json.dumps(_jsonable(training_provenance), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if synergy_payload is not None:
        np.savez_compressed(
            path / "synergy_basis.npz",
            basis=np.asarray(synergy_payload["basis"], dtype=np.float32),
            actuator_names=np.asarray(synergy_payload["actuator_names"], dtype=np.str_),
            excitation_bounds=np.asarray(
                synergy_payload["excitation_bounds"], dtype=np.float32
            ),
        )
        (path / "synergy_basis.json").write_text(
            json.dumps(
                _jsonable(
                    {
                        "fingerprint": synergy_payload["fingerprint"],
                        "actuator_names": synergy_payload["actuator_names"],
                        "manifest": synergy_payload["manifest"],
                        "source_path": synergy_payload.get("source_path"),
                    }
                ),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
    if frozen_body_decoder is not None:
        frozen_body_decoder.save(path)
    _write_metrics_csv(path / "train_metrics.csv", train_metrics)
    (path / "eval_metrics.json").write_text(
        json.dumps(_jsonable(eval_metrics), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (path / "checkpoint_fingerprint.txt").write_text(
        latent_checkpoint_fingerprint(path) + "\n",
        encoding="utf-8",
    )
    return path


def load_latent_checkpoint(
    checkpoint_dir: str | Path,
    *,
    expected_state_schema_hash: str | None = None,
    expected_body_actuator_names: list[str] | tuple[str, ...] | None = None,
    runtime_only: bool = False,
) -> dict[str, Any]:
    """Load a latent checkpoint saved by :func:`save_latent_checkpoint`."""
    path = Path(checkpoint_dir)
    result = {
        "prior_variables": _read_msgpack(path / "prior.msgpack"),
        "decoder_variables": _read_msgpack(path / "decoder.msgpack"),
        "action_mask": json.loads((path / "action_mask.json").read_text(encoding="utf-8")),
        "config": json.loads((path / "latent_config.yaml").read_text(encoding="utf-8")),
        "eval_metrics": json.loads((path / "eval_metrics.json").read_text(encoding="utf-8")),
    }
    if runtime_only:
        result["encoder_variables"] = None
        result["optimizer_state"] = None
    else:
        result["encoder_variables"] = _read_msgpack(path / "encoder.msgpack")
        result["optimizer_state"] = _read_msgpack(path / "optimizer_state.msgpack")
    for key, filename in (
        ("obs_norm", "obs_norm.json"),
        ("action_norm", "action_norm.json"),
        ("action_schema", "action_schema.json"),
        ("state_schema", "state_schema.json"),
        ("body_obs_schema", "body_obs_schema.json"),
        ("split_manifest", "motion_split.json"),
        ("training_provenance", "training_provenance.json"),
    ):
        file_path = path / filename
        result[key] = json.loads(file_path.read_text(encoding="utf-8")) if file_path.is_file() else None
    result["synergy_basis"] = _load_checkpoint_synergy_basis(path)
    result["frozen_body_decoder"] = _load_checkpoint_frozen_body_decoder(path)
    result["body_synergy_contract"] = (
        None
        if result["frozen_body_decoder"] is None
        else result["frozen_body_decoder"].body_synergy_contract
    )
    result["body_synergy_contract_fingerprint"] = (
        None
        if result["body_synergy_contract"] is None
        else result["body_synergy_contract"].contract_fingerprint
    )
    result["body_synergy_portable_core_fingerprint"] = (
        None
        if result["body_synergy_contract"] is None
        else result["body_synergy_contract"].portable_decoder_core_fingerprint
    )
    result["checkpoint_dir"] = str(path)
    result["checkpoint_fingerprint"] = latent_checkpoint_fingerprint(path)
    stored_fingerprint = path / "checkpoint_fingerprint.txt"
    if stored_fingerprint.is_file():
        expected_fingerprint = stored_fingerprint.read_text(encoding="utf-8").strip()
        if expected_fingerprint != result["checkpoint_fingerprint"]:
            raise ValueError(
                "latent checkpoint content fingerprint mismatch: "
                f"stored={expected_fingerprint} computed={result['checkpoint_fingerprint']}"
            )
    _validate_checkpoint_manifests(
        result,
        expected_state_schema_hash=expected_state_schema_hash,
        expected_body_actuator_names=expected_body_actuator_names,
    )
    if (
        bool(result["config"].get("require_closed_loop_metrics", False))
        and result["eval_metrics"].get("promotion", {}).get("passed") is True
    ):
        report_path = path / "closed_loop_metrics.json"
        if not report_path.is_file():
            raise ValueError(
                "passed production latent checkpoint is missing closed_loop_metrics.json"
            )
        from musclemimic.latent_muscle.closed_loop_eval import (
            validate_closed_loop_promotion_report,
        )

        validate_closed_loop_promotion_report(
            json.loads(report_path.read_text(encoding="utf-8")),
            checkpoint_dir=path,
            require_seal=True,
        )
    return result


def _validate_checkpoint_manifests(
    checkpoint: dict[str, Any],
    *,
    expected_state_schema_hash: str | None,
    expected_body_actuator_names: list[str] | tuple[str, ...] | None,
) -> None:
    mask_payload = checkpoint["action_mask"]
    mask = ActionMask.from_partitions(
        all_actuator_names=list(mask_payload["all_actuator_names"]),
        body_actuator_names=list(mask_payload["body_actuator_names"]),
        correction_actuator_names=list(mask_payload["correction_actuator_names"]),
        neutral_actuator_names=list(mask_payload.get("neutral_actuator_names") or []),
        neutral_values=np.asarray(mask_payload.get("neutral_values") or [], dtype=float),
    )
    supplied_mask_hash = mask_payload.get("schema_hash")
    if supplied_mask_hash is not None and str(supplied_mask_hash) != mask.schema_hash:
        raise ValueError("latent checkpoint action_mask schema hash mismatch")
    config_action_dim = checkpoint["config"].get("action_dim")
    if config_action_dim is not None and int(config_action_dim) != mask.body_size:
        raise ValueError(
            f"latent checkpoint action_dim={config_action_dim} != action mask body size={mask.body_size}"
        )

    from musclemimic.latent_muscle.decoder_factory import canonical_decoder_type

    decoder_type = canonical_decoder_type(checkpoint["config"].get("decoder_type"))
    synergy_basis = checkpoint.get("synergy_basis")
    frozen_decoder = checkpoint.get("frozen_body_decoder")
    if decoder_type == "direct":
        if synergy_basis is not None:
            raise ValueError("direct latent checkpoint must not carry a synergy basis")
        if frozen_decoder is not None:
            raise ValueError("direct latent checkpoint must not carry a frozen synergy decoder")
    else:
        if frozen_decoder is not None:
            if synergy_basis is not None:
                raise ValueError(
                    "portable latent checkpoint must not mix a standalone legacy synergy basis"
                )
            if bool(checkpoint["config"].get("legacy_synergy_decoder_ablation", False)):
                raise ValueError(
                    "portable latent checkpoint is incorrectly marked as a legacy ablation"
                )
            contract = frozen_decoder.body_synergy_contract
            if list(contract.actuator_names) != mask.body_actuator_names:
                raise ValueError(
                    "latent frozen decoder actuator names differ from body action mask"
                )
            expected_mode = (
                "fixed_synergy"
                if decoder_type == "fixed_synergy"
                else "fixed_synergy_residual"
            )
            if contract.mode != expected_mode:
                raise ValueError(
                    "latent frozen decoder mode differs from checkpoint decoder_type"
                )
            bindings = {
                "frozen_body_decoder_fingerprint": (
                    frozen_decoder.artifact_fingerprint
                ),
                "body_synergy_contract_fingerprint": contract.contract_fingerprint,
                "body_synergy_portable_core_fingerprint": (
                    contract.portable_decoder_core_fingerprint
                ),
                "synergy_basis_fingerprint": contract.runtime_basis_fingerprint,
            }
            mismatched = [
                key
                for key, value in bindings.items()
                if str(checkpoint["config"].get(key)) != str(value)
            ]
            if mismatched:
                raise ValueError(
                    "latent portable decoder checkpoint bindings differ: "
                    f"fields={mismatched}"
                )
        else:
            if not bool(
                checkpoint["config"].get("legacy_synergy_decoder_ablation", False)
            ):
                raise ValueError(
                    f"{decoder_type} latent checkpoint is missing its portable frozen "
                    "decoder; W-only checkpoints require "
                    "legacy_synergy_decoder_ablation=true"
                )
            if synergy_basis is None:
                raise ValueError(
                    f"legacy {decoder_type} latent checkpoint is missing its fixed synergy basis"
                )
            names = list(synergy_basis["actuator_names"])
            if names != mask.body_actuator_names:
                raise ValueError("latent checkpoint synergy basis names differ from body action mask")
            configured = checkpoint["config"].get("synergy_basis_fingerprint")
            if str(configured) != str(synergy_basis["fingerprint"]):
                raise ValueError("latent checkpoint synergy basis fingerprint mismatch")

    action_schema = checkpoint.get("action_schema")
    if action_schema is not None:
        names = list(action_schema.get("target_actuator_names") or action_schema.get("actuator_names") or [])
        if names != mask.body_actuator_names:
            raise ValueError("latent checkpoint action schema names differ from action mask body partition")
        schema_hash = action_schema.get("target_schema_hash", action_schema.get("action_schema_hash"))
        if schema_hash is not None and str(schema_hash) != actuator_schema_hash(names):
            raise ValueError("latent checkpoint action schema hash mismatch")
        ctrlrange_payload = action_schema.get("target_ctrlrange")
        if ctrlrange_payload is not None:
            ctrlrange = np.asarray(ctrlrange_payload, dtype=np.float64)
            if ctrlrange.shape != (len(names), 2) or not np.all(np.isfinite(ctrlrange)):
                raise ValueError("latent checkpoint target_ctrlrange is invalid")
            actual_ctrlrange_hash = ordered_schema_hash(
                kind="actuator_ctrlrange",
                payload={"actuator_names": names, "ctrlrange": ctrlrange.tolist()},
            )
            if str(action_schema.get("ctrlrange_schema_hash")) != actual_ctrlrange_hash:
                raise ValueError("latent checkpoint ctrlrange schema hash mismatch")
    if expected_body_actuator_names is not None and list(expected_body_actuator_names) != mask.body_actuator_names:
        raise ValueError("runtime body actuator names/order differ from latent checkpoint")

    state_schema = checkpoint.get("state_schema")
    if state_schema is not None:
        payload = {
            key: value
            for key, value in state_schema.items()
            if key not in {"schema_hash", "provenance"}
        }
        actual_hash = ordered_schema_hash(kind="student_state", payload=payload)
        supplied_hash = state_schema.get("schema_hash")
        if supplied_hash is not None and str(supplied_hash) != actual_hash:
            raise ValueError("latent checkpoint state schema hash mismatch")
        config_state_dim = checkpoint["config"].get("student_obs_dim")
        if config_state_dim is not None and int(state_schema.get("state_dim", -1)) != int(config_state_dim):
            raise ValueError("latent checkpoint state schema dimension differs from config")
        if expected_state_schema_hash is not None and str(expected_state_schema_hash) != str(supplied_hash):
            raise ValueError("runtime state schema hash differs from latent checkpoint")
    elif expected_state_schema_hash is not None:
        raise ValueError("latent checkpoint has no state schema to validate")

    body_obs_schema = checkpoint.get("body_obs_schema")
    if body_obs_schema is not None:
        validate_body_obs_schema(
            body_obs_schema,
            state_dim=checkpoint["config"].get("student_obs_dim"),
        )
        if list(body_obs_schema.get("actuator_names") or []) != mask.body_actuator_names:
            raise ValueError(
                "latent checkpoint BodyObsSchema actuator names/order differ from action mask body partition"
            )
        if int(body_obs_schema.get("action_size", -1)) != mask.body_size:
            raise ValueError(
                "latent checkpoint BodyObsSchema action_size differs from action mask body size"
            )

    obs_norm = checkpoint.get("obs_norm")
    if obs_norm is not None and checkpoint["config"].get("student_obs_dim") is not None:
        if len(obs_norm.get("mean", [])) != int(checkpoint["config"]["student_obs_dim"]):
            raise ValueError("latent checkpoint observation normalizer dimension differs from config")
    if bool(checkpoint["config"].get("require_dataset_provenance", False)):
        training = checkpoint.get("training_provenance")
        if not isinstance(training, dict):
            raise ValueError("production latent checkpoint is missing training provenance")
        dataset = training.get("dataset_manifest")
        teacher = training.get("teacher_checkpoint")
        if not isinstance(dataset, dict) or not isinstance(teacher, dict):
            raise ValueError("production latent checkpoint has incomplete dataset/teacher provenance")
        if training.get("dataset_manifest_fingerprint") != dataset.get("manifest_fingerprint"):
            raise ValueError("latent training dataset manifest fingerprint mismatch")
        if dataset.get("teacher_checkpoint", {}).get("sha256") != teacher.get("sha256"):
            raise ValueError("latent training dataset teacher fingerprint mismatch")
        promotion = training.get("teacher_promotion")
        if dataset.get("teacher_promotion") != promotion:
            raise ValueError("latent training dataset teacher promotion binding mismatch")
        from musclemimic.distill.provenance import validate_teacher_promotion_binding

        validate_teacher_promotion_binding(
            promotion,
            teacher_checkpoint=teacher,
            require_promoted=not bool(
                checkpoint["config"].get("test_only_allow_unpromoted_teacher", False)
            ),
        )
        if checkpoint["config"].get("val_dataset_dir") is not None:
            validation = training.get("validation_dataset_manifest")
            if not isinstance(validation, dict):
                raise ValueError(
                    "production latent checkpoint is missing validation dataset provenance"
                )
            if training.get(
                "validation_dataset_manifest_fingerprint"
            ) != validation.get("manifest_fingerprint"):
                raise ValueError(
                    "latent validation dataset manifest fingerprint mismatch"
                )
            if validation.get("teacher_checkpoint", {}).get("sha256") != teacher.get(
                "sha256"
            ):
                raise ValueError(
                    "latent validation dataset teacher fingerprint mismatch"
                )
            if validation.get("teacher_promotion") != promotion:
                raise ValueError(
                    "latent validation dataset teacher promotion binding mismatch"
                )
            split = checkpoint.get("split_manifest")
            if (
                not isinstance(split, dict)
                or split.get("schema_version") != "motion_split_v2"
                or split.get("mode") != "explicit_dataset_directories"
                or split.get("split_seed") is not None
            ):
                raise ValueError(
                    "production latent checkpoint lacks a seed-independent explicit validation split"
                )
            split_payload = {
                key: value for key, value in split.items() if key != "split_fingerprint"
            }
            if split.get("split_fingerprint") != canonical_json_sha256(split_payload):
                raise ValueError("latent explicit validation split fingerprint mismatch")
            if split.get("train_dataset_manifest_fingerprint") != dataset.get(
                "manifest_fingerprint"
            ) or split.get("val_dataset_manifest_fingerprint") != validation.get(
                "manifest_fingerprint"
            ):
                raise ValueError(
                    "latent explicit split dataset fingerprints differ from training provenance"
                )
            expected_count = checkpoint["config"].get("expected_val_motion_count")
            if expected_count is not None and len(split.get("val_motion_ids") or []) != int(
                expected_count
            ):
                raise ValueError("latent explicit validation motion count mismatch")


_RUNTIME_FINGERPRINT_FILES = (
    "prior.msgpack",
    "decoder.msgpack",
    "obs_norm.json",
    "action_norm.json",
    "state_schema.json",
    "body_obs_schema.json",
    "action_schema.json",
    "action_mask.json",
    "latent_config.yaml",
    "training_provenance.json",
    # Optional files are appended so legacy direct-checkpoint fingerprints do
    # not change: absent files contribute no bytes to the digest.
    "synergy_basis.npz",
    "synergy_basis.json",
    "frozen_body_decoder.npz",
    "frozen_body_decoder.json",
    "body_synergy_contract.json",
)


def latent_checkpoint_fingerprint(checkpoint_dir: str | Path) -> str:
    """Hash every tensor/contract that determines frozen controller output."""
    path = Path(checkpoint_dir)
    digest = hashlib.sha256()
    found = 0
    for filename in _RUNTIME_FINGERPRINT_FILES:
        file_path = path / filename
        if not file_path.is_file():
            continue
        payload = file_path.read_bytes()
        encoded_name = filename.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        found += 1
    if found < 2 or not (path / "prior.msgpack").is_file() or not (path / "decoder.msgpack").is_file():
        raise FileNotFoundError("latent runtime fingerprint requires prior.msgpack and decoder.msgpack")
    return digest.hexdigest()


def _write_msgpack(path: Path, value: Any) -> None:
    path.write_bytes(serialization.msgpack_serialize(serialization.to_state_dict(value)))


def _read_msgpack(path: Path) -> Any:
    return serialization.msgpack_restore(path.read_bytes())


def _action_mask_manifest(mask: ActionMask) -> dict[str, Any]:
    return {
        "all_actuator_names": list(mask.all_actuator_names),
        "body_actuator_names": list(mask.body_actuator_names),
        "correction_actuator_names": list(mask.correction_actuator_names),
        "neutral_actuator_names": list(mask.neutral_actuator_names),
        "body_indices": mask.body_indices.tolist(),
        "correction_indices": mask.correction_indices.tolist(),
        "neutral_indices": mask.neutral_indices.tolist(),
        "neutral_values": mask.neutral_values.tolist(),
        "decoder_action_dim": int(mask.body_size),
        "correction_action_dim": int(mask.correction_size),
        "neutral_action_dim": int(mask.neutral_size),
        "full_action_dim": int(mask.action_size),
        "schema_hash": mask.schema_hash,
    }


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fieldnames})


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _canonical_synergy_payload(value: Any | None) -> dict[str, Any] | None:
    if value is None:
        return None
    from musclemimic.latent_muscle.synergy_decoder import coerce_synergy_basis

    return coerce_synergy_basis(value).to_checkpoint_payload()


def _load_checkpoint_synergy_basis(path: Path) -> dict[str, Any] | None:
    array_path = path / "synergy_basis.npz"
    manifest_path = path / "synergy_basis.json"
    if not array_path.is_file() and not manifest_path.is_file():
        return None
    if not array_path.is_file() or not manifest_path.is_file():
        raise ValueError("latent checkpoint synergy basis contract is incomplete")
    with np.load(array_path, allow_pickle=False) as data:
        required = {"basis", "actuator_names", "excitation_bounds"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"latent checkpoint synergy_basis.npz is missing {missing}")
        raw = {
            "basis": np.asarray(data["basis"], dtype=np.float32),
            "actuator_names": [
                str(name) for name in np.asarray(data["actuator_names"]).tolist()
            ],
            "excitation_bounds": np.asarray(
                data["excitation_bounds"], dtype=np.float32
            ),
        }
    contract = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw["actuator_names"] != list(contract.get("actuator_names") or []):
        raise ValueError("latent checkpoint synergy basis name contracts disagree")
    raw.update(
        {
            "fingerprint": contract.get("fingerprint"),
            "manifest": contract.get("manifest") or {},
            "source_path": contract.get("source_path"),
        }
    )
    from musclemimic.latent_muscle.synergy_decoder import coerce_synergy_basis

    loaded = coerce_synergy_basis(raw)
    if str(contract.get("fingerprint")) != loaded.fingerprint:
        raise ValueError("latent checkpoint synergy basis content fingerprint mismatch")
    return loaded.to_checkpoint_payload()


def _load_checkpoint_frozen_body_decoder(path: Path):
    from musclemimic.synergy.frozen_decoder import (
        BODY_SYNERGY_CONTRACT_FILENAME,
        FROZEN_BODY_DECODER_ARRAYS,
        FROZEN_BODY_DECODER_MANIFEST,
        load_frozen_body_decoder,
    )

    filenames = (
        FROZEN_BODY_DECODER_ARRAYS,
        FROZEN_BODY_DECODER_MANIFEST,
        BODY_SYNERGY_CONTRACT_FILENAME,
    )
    present = [filename for filename in filenames if (path / filename).is_file()]
    if not present:
        return None
    if len(present) != len(filenames):
        raise ValueError(
            "latent checkpoint frozen body decoder contract is incomplete: "
            f"present={present}"
        )
    return load_frozen_body_decoder(path)

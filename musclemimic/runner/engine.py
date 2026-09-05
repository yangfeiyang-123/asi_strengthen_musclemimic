from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf, open_dict

from loco_mujoco.task_factories import TaskFactory
from musclemimic.algorithms.common.checkpoint_hooks import create_jax_checkpoint_host_callback
from musclemimic.utils.logging import setup_logger
from musclemimic.utils.metrics import MetricsHandler

from .checkpointing import (
    bind_explicit_parent_checkpoint,
    config_hash,
    find_latest_checkpoint,
    infer_training_action,
    resolve_checkpoint_dir,
    resolve_training_root,
    resume_or_fresh,
    validate_checkpoint_body_action_contract,
    validate_checkpoint_compatibility,
    validate_checkpoint_continuity_training_contract,
    validate_checkpoint_parent_lineage,
    validate_explicit_parent_checkpoint,
    write_manifest,
)
from .logging import ExperimentHooks

logger = setup_logger(__name__)


_CANONICAL_FOREHAND_VARIANT = "raw_smooth_v1"
_NONPRODUCTION_CONFIG_STATUSES = frozenset({"legacy", "experimental"})


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_experiment_config_status(config: Any) -> dict[str, Any] | None:
    """Block declared legacy/experimental Hydra configs by default.

    The opt-in lives in the selected config itself and must be set by an
    explicit Hydra override.  Accepted non-production runs record that choice
    inside ``experiment`` so it enters their config hash and run manifest.
    """

    status_config = config.get("config_status", None)
    if status_config is None:
        return None
    status = str(status_config.get("status", "")).strip().lower()
    canonical = status_config.get("canonical", None)
    if status not in _NONPRODUCTION_CONFIG_STATUSES and canonical is not False:
        return None
    replacement = str(status_config.get("replacement", "")).strip()
    if status_config.get("allow_nonproduction_runtime", False) is not True:
        replacement_hint = f"; use {replacement}" if replacement else ""
        raise ValueError(
            f"refusing non-production {status or 'noncanonical'} experiment config"
            f"{replacement_hint}. For an isolated legacy/experimental ablation only, "
            "set config_status.allow_nonproduction_runtime=true explicitly"
        )
    evidence = {
        "schema_version": "nonproduction_runtime_opt_in_v1",
        "status": status or "noncanonical",
        "canonical": False,
        "replacement": replacement or None,
        "explicit_opt_in": True,
    }
    with open_dict(config.experiment):
        config.experiment.nonproduction_runtime_opt_in = evidence
    return evidence


def bind_continuity_training_release(
    config: Any,
    *,
    launch_dir: str | Path,
    result_dir: str | Path | None = None,
) -> dict[str, Any] | None:
    """Validate the single continuity release before W&B, env, or GPU work."""

    experiment = config.experiment
    reward_params = experiment.get("env_params", {}).get("reward_params", {})
    continuity = reward_params.get("intra_muscle_consistency", {})
    mode = str(continuity.get("mode", "off")).strip().lower()
    release_value = continuity.get("release_path", None)
    release_text = str(release_value or "").strip()
    if mode != "reward":
        if release_text:
            raise ValueError("continuity release_path requires reward mode")
        return None
    if not release_text:
        raise ValueError("continuity reward mode requires one immutable release_path")

    from musclemimic.physiology.release import (
        load_continuity_training_release,
        resolve_continuity_training_release,
        validate_release_against_runtime,
    )

    release_path = Path(release_text).expanduser()
    if not release_path.is_absolute():
        release_path = Path(launch_dir) / release_path
    release = load_continuity_training_release(release_path)
    expected = str(continuity.get("expected_release_fingerprint", "") or "").strip()
    if not expected:
        raise ValueError("continuity reward mode requires expected_release_fingerprint")
    if expected != release.release_fingerprint:
        raise ValueError("configured continuity release fingerprint differs from artifact")
    artifacts = resolve_continuity_training_release(release)

    action = experiment.get("action_representation", {})
    if not bool(action.get("enabled", False)):
        action_mode = "full_354"
    else:
        action_mode = str(action.get("mode", "")).strip()
    ablation = experiment.get("continuity_ablation", {})
    declared_action_mode = str(ablation.get("action_mode", action_mode) or "").strip()
    if declared_action_mode != action_mode:
        raise ValueError("continuity ablation action_mode differs from action representation")
    validate_release_against_runtime(
        release,
        taxonomy=artifacts.taxonomy,
        graph=artifacts.candidate_graph,
        runtime_loss_identity=artifacts.loss_identity,
        action_mode=action_mode,
    )

    contract = {
        "schema_version": "continuity_training_runtime_contract_v1",
        "release_path": str(release_path.resolve()),
        "release_id": release.release_id,
        "release_fingerprint": release.release_fingerprint,
        "taxonomy_fingerprint": release.taxonomy["taxonomy_fingerprint"],
        "diagnostic_graph_fingerprint": release.diagnostic_graph["graph_fingerprint"],
        "candidate_graph_fingerprint": release.candidate_graph["graph_fingerprint"],
        "loss_spec_fingerprint": release.loss_spec["loss_spec_fingerprint"],
        "calibration_fingerprint": release.calibration["calibration_fingerprint"],
        "selected_reward_coefficient": release.reward["coefficient"],
        "target_chain_count": release.loss_spec["target_chain_count"],
        "target_edge_count": release.loss_spec["target_edge_count"],
        "action_mode": action_mode,
    }
    contract["binding_sha256"] = _canonical_json_sha256(contract)
    with open_dict(experiment):
        continuity.action_mode = action_mode
        experiment.continuity_training_contract = contract
    if result_dir is not None:
        _atomic_write_json(
            Path(result_dir) / "continuity_training_preflight.json",
            contract,
        )
    logger.info(
        "Continuity release preflight: id=%s fingerprint=%s action_mode=%s",
        release.release_id,
        release.release_fingerprint,
        action_mode,
    )
    return contract


def bind_emg_consistency_training_reference(
    config: Any,
    *,
    launch_dir: str | Path,
    result_dir: str | Path | None = None,
) -> dict[str, Any] | None:
    """Validate the reviewed Stage-1 EMG bundle before environment work."""

    from musclemimic.physiology.emg_consistency_runtime import (
        build_emg_consistency_preflight_contract,
    )

    experiment = config.experiment
    reward_params = experiment.get("env_params", {}).get("reward_params", {})
    raw_config = reward_params.get("emg_consistency", None)
    budget_contract = bind_stage1_peasd_fixed_budget_contract(config)
    bind_stage1_peasd_action_release(config, result_dir=result_dir)
    contract = build_emg_consistency_preflight_contract(
        raw_config,
        base_dir=launch_dir,
    )
    if contract is None:
        if budget_contract is not None and budget_contract["arm"] != "T0":
            raise ValueError("active Stage1 PEASD arm did not bind an EMG reference")
        return None

    if budget_contract is None:
        raise ValueError("EMG training reward requires an experiment.stage1_peasd contract")
    if contract.get("mode") != "reward" or contract.get("training_signal_enabled") is not True:
        raise ValueError("training launch refuses diagnostics-only EMG runtime mode")
    if contract.get("arm") != budget_contract["arm"]:
        raise ValueError("Stage1 PEASD arm differs from reward_params.emg_consistency.arm")
    if contract.get("action_id") != budget_contract["tube_action_id"]:
        raise ValueError("Stage1 PEASD tube action differs from the matched action contract")

    action = experiment.get("action_representation", {})
    if bool(action.get("enabled", False)) and str(action.get("mode", "")) != "full_354":
        raise ValueError("Stage1 PEASD-Lite is defined only for the full-354 action ABI")
    if not bool(experiment.get("env_params", {}).get("disable_fingers", False)):
        raise ValueError("Stage1 PEASD-Lite requires the no-finger 354-muscle environment")

    with open_dict(experiment):
        experiment.emg_consistency_preflight_contract = contract
    if result_dir is not None:
        _atomic_write_json(
            Path(result_dir) / "emg_consistency_preflight.json",
            contract,
        )
    logger.info(
        "PEASD-Lite artifact preflight: arm=%s reference=%s fingerprint=%s",
        contract["arm"],
        contract["reference_id"],
        contract["reference_fingerprint"],
    )
    return contract


def bind_stage1_peasd_action_release(
    config: Any,
    *,
    result_dir: str | Path | None = None,
) -> dict[str, Any] | None:
    """Bind the registered trajectory release/QC bytes for every T0--T4 arm."""

    stage1 = config.experiment.get("stage1_peasd", None)
    if stage1 is None:
        return None
    action_id = str(stage1.get("action_id", ""))
    training_source = config.experiment.get("training_source", {})
    declared_variant = str(training_source.get("variant", ""))
    if declared_variant == "raw_smooth_v1_aug100":
        from musclemimic.badminton.aug100_release import (
            ACTION_ID,
            ACTION_SLUG,
            CACHE_VARIANT,
            SOURCE_NAMESPACE,
            SOURCE_VARIANT,
            inspect_forehand_clear_aug100_dataset,
            motion_names_from_relative_paths,
            validate_forehand_clear_aug100_release,
        )

        if action_id != ACTION_ID:
            raise ValueError("Forehand Clear Aug100 release requires forehandClear_standard")
        train_conf = config.experiment.task_factory.params.amass_dataset_conf
        val_conf = config.experiment.get("validation", {}).get("amass_dataset_conf", None)
        if val_conf is None:
            raise ValueError("Forehand Clear Aug100 requires a held-out validation dataset")
        train_motions = motion_names_from_relative_paths(
            list(train_conf.get("rel_dataset_path", ()) or ())
        )
        validation_motions = motion_names_from_relative_paths(
            list(val_conf.get("rel_dataset_path", ()) or ())
        )
        split_contract = str(training_source.get("split_contract", "") or "") or None
        report = validate_forehand_clear_aug100_release(
            train_motions,
            validation_motions,
            split_contract=split_contract,
        )
        numeric_report = inspect_forehand_clear_aug100_dataset(
            train_motions,
            validation_motions,
            release_report=report,
            split_contract=split_contract,
        )
        numeric_identity = {
            "action_id": ACTION_ID,
            "action_slug": ACTION_SLUG,
            "source_namespace": SOURCE_NAMESPACE,
            "source_variant": SOURCE_VARIANT,
            "cache_variant": CACHE_VARIANT,
        }
    else:
        from musclemimic.badminton.action_registry import resolve
        from musclemimic.badminton.action_release import validate_action_release
        from musclemimic.badminton.data_qc import inspect_canonical_dataset

        spec = resolve(action_id)
        report = validate_action_release(action_id)
        numeric_report = inspect_canonical_dataset(
            spec.dataset_root,
            source_variant=spec.source_namespace,
            cache_variant=spec.cache_variant,
            action=spec.slug,
        )
        numeric_identity = {
            "action_id": spec.action_id,
            "action_slug": spec.slug,
            "source_namespace": spec.source_namespace,
            "source_variant": spec.source_variant,
            "cache_variant": spec.cache_variant,
        }
    if report.get("passed") is not True:
        raise ValueError(
            "Stage1 PEASD action release/QC preflight failed: "
            + "; ".join(str(item) for item in report.get("errors", ()))
        )
    if numeric_report.get("clean_passed") is not True:
        failures = [
            *[str(item) for item in numeric_report.get("hard_errors", ())],
            *[str(item) for item in numeric_report.get("warnings", ())],
        ]
        raise ValueError("Stage1 PEASD numeric data QC must be a warning-free clean pass: " + "; ".join(failures))
    # The manifest is JSON, so normalize dataclass tuple fields now.  Without
    # this round trip an in-memory tuple would become a list on disk and make
    # otherwise identical post-training validation fail equality checks.
    numeric_report = json.loads(json.dumps(numeric_report, sort_keys=True, allow_nan=False))
    numeric_unsigned = {
        "schema_version": "stage1_peasd_numeric_data_qc_contract_v1",
        **numeric_identity,
        "report_sha256": _canonical_json_sha256(numeric_report),
        "report": numeric_report,
    }
    numeric_contract = {
        **numeric_unsigned,
        "binding_sha256": _canonical_json_sha256(numeric_unsigned),
    }
    with open_dict(config.experiment):
        config.experiment.stage1_peasd_action_release_contract = report
        config.experiment.stage1_peasd_numeric_data_qc_contract = numeric_contract
    if result_dir is not None:
        _atomic_write_json(Path(result_dir) / "stage1_peasd_action_release_preflight.json", report)
        _atomic_write_json(
            Path(result_dir) / "stage1_peasd_numeric_data_qc_preflight.json",
            numeric_contract,
        )
    return report


def bind_stage1_peasd_fixed_budget_contract(config: Any) -> dict[str, Any] | None:
    """Fail closed on matched-action identity, fresh state and fixed budget.

    This gate deliberately also runs for T0, which has no tube at training
    time.  It prevents a wrong-action or early-stopped baseline from reaching
    the much later paired-results gate.
    """

    from musclemimic.badminton.action_registry import resolve

    experiment = config.experiment
    raw_stage1 = experiment.get("stage1_peasd", None)
    reward = experiment.get("env_params", {}).get("reward_params", {}).get("emg_consistency", {})
    reward_enabled = bool(reward.get("enabled", False))
    if raw_stage1 is None:
        if reward_enabled:
            raise ValueError("EMG training reward requires experiment.stage1_peasd")
        return None
    if not isinstance(raw_stage1, Mapping) and not hasattr(raw_stage1, "items"):
        raise ValueError("experiment.stage1_peasd must be a mapping")
    stage1 = {str(key): value for key, value in raw_stage1.items()}
    if stage1.get("schema_version") != "stage1_peasd_lite_matched_arm_v1":
        raise ValueError("unsupported experiment.stage1_peasd schema_version")

    arm = str(stage1.get("arm", "")).strip().upper()
    if arm not in {"T0", "T1", "T2", "T3", "T4"}:
        raise ValueError("experiment.stage1_peasd.arm must be T0--T4")
    reward_arm = str(reward.get("arm", "T0")).strip().upper()
    if reward_arm != arm:
        raise ValueError("experiment.stage1_peasd.arm differs from the EMG reward arm")
    if reward_enabled != (arm != "T0"):
        raise ValueError("Stage1 PEASD enabled state does not match its T0--T4 arm")
    reward_mode = str(reward.get("mode", "reward" if reward_enabled else "off")).strip().lower()
    if reward_mode != ("off" if arm == "T0" else "reward"):
        raise ValueError("training launch permits only T0/off or T1--T4/reward EMG mode")

    dataset_action_id = str(stage1.get("action_id", "") or "").strip()
    training_action = str(experiment.get("training_action", "") or "").strip()
    if not dataset_action_id or not training_action:
        raise ValueError("Stage1 PEASD requires explicit dataset and training action identities")
    spec = resolve(dataset_action_id)
    if dataset_action_id != spec.action_id or training_action != spec.action_id:
        raise ValueError(
            "experiment.training_action and stage1_peasd.action_id must equal one registry dataset action_id"
        )
    tube_action_id = str(reward.get("action_id", "") or "").strip()
    if not tube_action_id or tube_action_id != spec.emg_trial_actions[0]:
        raise ValueError(
            "reward_params.emg_consistency.action_id must be the registry's primary tube action; "
            "shadow/foreign actions are not accepted"
        )
    if resolve(tube_action_id).slug != spec.slug:
        raise ValueError("Stage1 PEASD tube action resolves to another registry action")

    canonical_seeds = [int(value) for value in stage1.get("canonical_seeds", ())]
    if canonical_seeds != [0, 1, 2]:
        raise ValueError("Stage1 PEASD canonical_seeds must be [0, 1, 2]")
    seeds = [int(value) for value in experiment.get("seeds", ())]
    if int(experiment.get("n_seeds", 0)) != 1 or len(seeds) != 1 or seeds[0] not in canonical_seeds:
        raise ValueError("one Stage1 PEASD run must select exactly one canonical seed")
    if (
        stage1.get("fresh_optimizer_required") is not True
        or stage1.get("parent_initialization_checkpoint") is not None
        or bool(experiment.get("auto_resume", True))
        or experiment.get("resume_from", None) is not None
        or experiment.get("reset_optimizer_on_resume", False) is not True
        or experiment.get("resume_lr_override", None) is not None
        or bool(experiment.get("extend_completed_run", False))
    ):
        raise ValueError("Stage1 PEASD matched arms require a fresh optimizer with no parent/resume/extension")
    promotion = experiment.get("promotion", {})
    if bool(promotion.get("auto_stop", False)):
        raise ValueError("Stage1 PEASD matched arms require promotion.auto_stop=false")

    total_timesteps = int(experiment.get("total_timesteps", 0))
    ppo_config = experiment.get("ppo_config", {})
    top_level_num_steps = experiment.get("num_steps", None)
    nested_num_steps = ppo_config.get("num_steps", None)
    num_steps = int(
        top_level_num_steps
        if top_level_num_steps is not None
        else (nested_num_steps if nested_num_steps is not None else 0)
    )
    num_envs = int(experiment.get("num_envs", 0))
    rollout_batch = num_steps * num_envs
    if total_timesteps <= 0 or rollout_batch <= 0 or total_timesteps % rollout_batch:
        raise ValueError("Stage1 PEASD total_timesteps must be an exact positive number of PPO rollouts")
    num_updates = total_timesteps // rollout_batch
    configured_updates = experiment.get("num_updates", None)
    if configured_updates is not None and int(configured_updates) != num_updates:
        raise ValueError("Stage1 PEASD configured num_updates differs from the fixed budget")

    validation = experiment.get("validation", {})
    if (
        not bool(validation.get("active", False))
        or not bool(validation.get("deterministic", False))
        or not bool(validation.get("start_from_beginning", False))
    ):
        raise ValueError("Stage1 PEASD requires active deterministic validation from frame zero")
    requested_validations = int(validation.get("num", 0))
    if requested_validations <= 0:
        raise ValueError("Stage1 PEASD fixed schedule requires validation.num > 0")
    if arm == "T3" and seeds == [0]:
        if (
            validation.get("visual_review_kind") != "stage1_body"
            or not bool(validation.get("cycle_video_trajectories", False))
            or int(validation.get("video_length", 0)) <= 0
        ):
            raise ValueError(
                "pre-registered Stage1 T3/seed0 teacher requires complete endpoint visual review recording"
            )
    validation_interval = max(1, num_updates // requested_validations)
    scheduled_validation_count = num_updates // validation_interval
    endpoint_requires_independent_validation = num_updates % validation_interval != 0
    expected_history_count = scheduled_validation_count + int(endpoint_requires_independent_validation)
    if not bool(experiment.get("save_checkpoints", False)) or not bool(
        experiment.get(
            "save_checkpoints_on_validation",
            experiment.get("checkpoints_on_validation", True),
        )
    ):
        raise ValueError("Stage1 PEASD requires checkpointing at every validation boundary")

    start_update = int(reward.get("start_update", 0))
    ramp_updates = int(reward.get("ramp_updates", 0))
    if start_update < 0 or ramp_updates < 0:
        raise ValueError("Stage1 PEASD EMG curriculum updates must be non-negative")
    full_weight_update = start_update + ramp_updates
    if arm != "T0" and full_weight_update >= num_updates:
        raise ValueError("Stage1 PEASD budget must include at least one fully exposed EMG training rollout")

    unsigned = {
        "schema_version": "stage1_peasd_fixed_budget_contract_v1",
        "action_slug": spec.slug,
        "action_id": spec.action_id,
        "tube_action_id": tube_action_id,
        "arm": arm,
        "seed": seeds[0],
        "canonical_seeds": canonical_seeds,
        "fresh_optimizer": True,
        "promotion_auto_stop": False,
        "total_timesteps": total_timesteps,
        "num_updates": num_updates,
        "num_steps": num_steps,
        "num_envs": num_envs,
        "rollout_batch_size": rollout_batch,
        "expected_endpoint_update_number": num_updates,
        "expected_endpoint_global_timestep": total_timesteps,
        "validation_interval_updates": validation_interval,
        "requested_validation_count": requested_validations,
        "scheduled_validation_count": scheduled_validation_count,
        "endpoint_requires_independent_validation": endpoint_requires_independent_validation,
        "expected_history_count": expected_history_count,
        "emg_curriculum": {
            "mode": reward_mode,
            "anchor_weight_max": float(reward.get("anchor_weight_max", 0.0)),
            "synergy_weight_max": float(reward.get("synergy_weight_max", 0.0)),
            "start_update": start_update,
            "ramp_updates": ramp_updates,
            "full_weight_update": full_weight_update,
            "fully_exposed_within_budget": arm == "T0" or full_weight_update < num_updates,
        },
    }
    contract = {**unsigned, "binding_sha256": _canonical_json_sha256(unsigned)}
    with open_dict(experiment):
        experiment.stage1_peasd_fixed_budget_contract = contract
    return contract


def bind_emg_consistency_runtime_model(
    config: Any,
    *,
    env: Any,
    val_env: Any | None,
    result_dir: str | Path | None = None,
) -> dict[str, Any] | None:
    """Bind train/validation rewards to one identical compiled muscle ABI."""

    from musclemimic.physiology.emg_consistency_runtime import (
        validate_emg_runtime_against_preflight,
    )

    preflight = config.experiment.get("emg_consistency_preflight_contract", None)
    train_contract = getattr(
        getattr(env, "_reward_function", None),
        "emg_consistency_runtime_contract",
        None,
    )
    if callable(train_contract):
        train_contract = train_contract()
    val_contract = None
    if val_env is not None:
        val_contract = getattr(
            getattr(val_env, "_reward_function", None),
            "emg_consistency_runtime_contract",
            None,
        )
        if callable(val_contract):
            val_contract = val_contract()

    if preflight is None:
        if train_contract is not None or val_contract is not None:
            raise ValueError("EMG reward compiled without an artifact preflight contract")
        return None
    if train_contract is None:
        raise ValueError("EMG artifact preflight passed but the training reward did not compile it")
    validate_emg_runtime_against_preflight(train_contract, preflight)
    if val_env is not None:
        if val_contract is None:
            raise ValueError("active validation did not compile the Stage1 EMG reward")
        if val_contract != train_contract:
            raise ValueError("training and validation compiled different Stage1 EMG runtime contracts")

    with open_dict(config.experiment):
        config.experiment.emg_consistency_runtime_contract = train_contract
    if result_dir is not None:
        _atomic_write_json(
            Path(result_dir) / "emg_consistency_runtime_preflight.json",
            train_contract,
        )
    logger.info(
        "PEASD-Lite runtime preflight: arm=%s model=%s spec=%s",
        train_contract["arm"],
        train_contract["runtime_model_hash"],
        train_contract["anchor_loss_spec_fingerprint"],
    )
    return train_contract


def _validate_aug100_training_source_preflight(
    config: Any,
    *,
    launch_dir: str | Path,
    result_dir: str | Path,
) -> dict[str, Any]:
    from musclemimic.badminton.aug100_release import (
        ACTION_ID,
        CACHE_NAMESPACE,
        CACHE_VARIANT,
        DATASET_MANIFEST,
        EXPECTED_AUGMENTATION_SEED,
        EXPECTED_AUGMENTATION_TYPE,
        EXPECTED_FPS,
        EXPECTED_TRANSFER_MANIFEST_FINGERPRINT,
        SOURCE_MODE,
        SOURCE_NAMESPACE,
        TRANSFER_MANIFEST,
        inspect_forehand_clear_aug100_dataset,
        motion_names_from_relative_paths,
        validate_forehand_clear_aug100_release,
    )

    source = config.experiment.training_source
    action = str(config.experiment.get("training_action", ""))
    if action != ACTION_ID:
        raise ValueError("raw_smooth_v1_aug100 training_source requires forehandClear_standard")
    contract = {
        "source_mode": SOURCE_MODE,
        "variant": CACHE_VARIANT,
        "source_namespace": SOURCE_NAMESPACE,
        "cache_namespace": CACHE_NAMESPACE,
        "transfer_manifest": TRANSFER_MANIFEST.as_posix(),
        "dataset_manifest": DATASET_MANIFEST.as_posix(),
        "augmentation_type": EXPECTED_AUGMENTATION_TYPE,
        "augmentation_seed": EXPECTED_AUGMENTATION_SEED,
        "source_fps": EXPECTED_FPS,
        "cache_fps": EXPECTED_FPS,
    }
    for key, expected in contract.items():
        actual = source.get(key, None)
        try:
            matches = float(actual) == expected if isinstance(expected, float) else actual == expected
        except (TypeError, ValueError):
            matches = False
        if not matches:
            raise ValueError(f"training_source.{key} must be {expected!r}; got {actual!r}")

    cache_root_value = os.environ.get("MUSCLEMIMIC_GMR_CACHE_PATH")
    if not cache_root_value:
        raise ValueError("MUSCLEMIMIC_GMR_CACHE_PATH is unset; source configs/env.sh before training")
    dataset_root = (Path(cache_root_value).expanduser().resolve() / ACTION_ID).resolve()
    expected_dataset_root = (Path(launch_dir).resolve() / "datasets" / ACTION_ID).resolve()
    if dataset_root != expected_dataset_root:
        raise ValueError(
            "Aug100 runtime cache root differs from the content-bound repository dataset: "
            f"{dataset_root} != {expected_dataset_root}"
        )

    train_conf = config.experiment.task_factory.params.amass_dataset_conf
    val_conf = config.experiment.get("validation", {}).get("amass_dataset_conf", None)
    if val_conf is None:
        raise ValueError("raw_smooth_v1_aug100 production training requires held-out data")
    train_motions = motion_names_from_relative_paths(
        list(train_conf.get("rel_dataset_path", ()) or ())
    )
    validation_motions = motion_names_from_relative_paths(
        list(val_conf.get("rel_dataset_path", ()) or ())
    )
    split_contract = str(source.get("split_contract", "") or "") or None

    def _validate_gmr(conf: Any, *, label: str) -> None:
        if bool(conf.get("clear_cache", True)):
            raise ValueError(f"{label}.clear_cache must remain false for released caches")
        if str(conf.get("retargeting_method", "")) != "gmr":
            raise ValueError(f"{label}.retargeting_method must be gmr")
        gmr = conf.get("gmr_config", {})
        requirements = {
            "target_fps": 60.0,
            "solver": "daqp",
            "damping": 1.0,
            "use_velocity_limit": True,
        }
        for key, expected in requirements.items():
            actual = gmr.get(key, None)
            try:
                matches = float(actual) == expected if isinstance(expected, float) else actual == expected
            except (TypeError, ValueError):
                matches = False
            if not matches:
                raise ValueError(f"{label}.gmr_config.{key} changed from {expected!r}")

    _validate_gmr(train_conf, label="training dataset")
    _validate_gmr(val_conf, label="validation dataset")
    release = validate_forehand_clear_aug100_release(
        train_motions,
        validation_motions,
        split_contract=split_contract,
    )
    if release.get("passed") is not True:
        raise ValueError(
            "raw_smooth_v1_aug100 release validation failed: "
            + "; ".join(str(value) for value in release.get("errors", ()))
        )
    qc = inspect_forehand_clear_aug100_dataset(
        train_motions,
        validation_motions,
        release_report=release,
        split_contract=split_contract,
    )
    if qc.get("clean_passed") is not True:
        details = [*list(qc.get("hard_errors", ()) or ()), *list(qc.get("warnings", ()) or ())]
        raise ValueError("raw_smooth_v1_aug100 strict data QC failed: " + "; ".join(map(str, details)))

    transfer_path = (Path(launch_dir).resolve() / TRANSFER_MANIFEST).resolve()
    dataset_manifest_path = (Path(launch_dir).resolve() / DATASET_MANIFEST).resolve()
    identity = {
        "transfer_manifest_fingerprint": EXPECTED_TRANSFER_MANIFEST_FINGERPRINT,
        "transfer_manifest_content_sha256": hashlib.sha256(transfer_path.read_bytes()).hexdigest(),
        "dataset_manifest_content_sha256": hashlib.sha256(
            dataset_manifest_path.read_bytes()
        ).hexdigest(),
        "action_release_binding_sha256": release["release_binding_sha256"],
        "qc_contract_sha256": _canonical_json_sha256(qc),
    }
    identity["preflight_binding_sha256"] = _canonical_json_sha256(identity)
    with open_dict(config.experiment):
        for key, value in identity.items():
            config.experiment.training_source[key] = value

    report: dict[str, Any] = {
        "schema_version": "raw_smooth_v1_aug100_training_source_preflight_v1",
        "dataset_root": str(dataset_root),
        "source_variant": "raw_smooth_v1",
        "cache_variant": CACHE_VARIANT,
        "source_fps": EXPECTED_FPS,
        "cache_fps": EXPECTED_FPS,
        "split_contract": split_contract or "reviewed_grouped_80_train_20_validation_v1",
        "transfer_manifest": str(transfer_path),
        "dataset_manifest": str(dataset_manifest_path),
        **identity,
        "train_motions": list(train_motions),
        "validation_motions": list(validation_motions),
        "train_source_groups": list(release["train_source_groups"]),
        "validation_source_groups": list(release["validation_source_groups"]),
        "clean_passed": True,
        "passed": True,
    }
    _atomic_write_json(Path(result_dir) / "training_source_preflight.json", report)
    return report


def validate_training_source_preflight(
    config: Any,
    *,
    launch_dir: str | Path,
    result_dir: str | Path,
) -> dict[str, Any] | None:
    """Fail closed before PPO can consume the canonical smooth release.

    The YAML ``training_source`` block is a declaration, not evidence.  This
    runtime gate recomputes both the immutable release and warning-free QC so a
    direct/manual ``fullbody/experiment.py`` launch cannot bypass Stage 0.
    Other source modes retain their existing behavior.
    """

    source = config.experiment.get("training_source", None)
    if source is None:
        return None
    declared_mode = str(source.get("source_mode", ""))
    declared_variant = str(source.get("variant", ""))
    if declared_mode == "verified_augmented_cache" or declared_variant == "raw_smooth_v1_aug100":
        return _validate_aug100_training_source_preflight(
            config,
            launch_dir=launch_dir,
            result_dir=result_dir,
        )
    if declared_mode != "existing_ppo":
        return None
    action = str(config.experiment.get("training_action", ""))
    # ``existing_ppo`` is also used by the generic manifest builder.  Claim
    # the strict badminton release contract only when the canonical action or
    # variant is explicitly selected; generic AMASS configs remain outside it.
    if action != "forehandClear_standard" and declared_variant != _CANONICAL_FOREHAND_VARIANT:
        return None
    from musclemimic.badminton.data_qc import (
        TRAIN_MOTIONS,
        VAL_MOTIONS,
        inspect_canonical_dataset,
    )
    from musclemimic.badminton.scripts.data_release import validate_release_manifest
    from musclemimic.badminton.scripts.finalize_raw_smooth_visual_qc import (
        validate_report as validate_visual_qc_report,
    )

    contract = {
        "variant": _CANONICAL_FOREHAND_VARIANT,
        "source_namespace": f"temp/{_CANONICAL_FOREHAND_VARIANT}",
        "cache_namespace": f"muscle_trajectory/{_CANONICAL_FOREHAND_VARIANT}",
        "source_fps": 60.0,
        "cache_fps": 100.0,
    }
    for key, expected in contract.items():
        actual = source.get(key, None)
        try:
            matches = float(actual) == expected if isinstance(expected, float) else actual == expected
        except (TypeError, ValueError):
            matches = False
        if not matches:
            raise ValueError(f"training_source.{key} must be {expected!r}; got {actual!r}")

    if action != "forehandClear_standard":
        raise ValueError("raw_smooth_v1 training_source requires forehandClear_standard")
    cache_root_value = os.environ.get("MUSCLEMIMIC_GMR_CACHE_PATH")
    if not cache_root_value:
        raise ValueError("MUSCLEMIMIC_GMR_CACHE_PATH is unset; source configs/env.sh before training")
    cache_root = Path(cache_root_value).expanduser().resolve()
    dataset_root = (cache_root / action).resolve()
    release_value = source.get("release_manifest", None)
    if not release_value:
        raise ValueError("training_source.release_manifest is required")
    release_path = Path(str(release_value)).expanduser()
    if not release_path.is_absolute():
        release_path = Path(launch_dir).resolve() / release_path
    release_path = release_path.resolve()
    expected_release = (dataset_root / "manifests" / _CANONICAL_FOREHAND_VARIANT / "release_manifest.json").resolve()
    if release_path != expected_release:
        raise ValueError(
            "training_source release manifest and runtime cache root differ: "
            f"release={release_path} expected={expected_release}"
        )

    recipe_value = source.get("source_recipe", None)
    if not recipe_value:
        raise ValueError("training_source.source_recipe is required")
    recipe_path = Path(str(recipe_value)).expanduser()
    if not recipe_path.is_absolute():
        recipe_path = Path(launch_dir).resolve() / recipe_path
    recipe_path = recipe_path.resolve()

    train_conf = config.experiment.task_factory.params.amass_dataset_conf
    validation = config.experiment.get("validation", {})
    val_conf = validation.get("amass_dataset_conf", None)
    if val_conf is None:
        raise ValueError("raw_smooth_v1 production training requires a held-out dataset")
    prefix = f"{action}/muscle_trajectory/{_CANONICAL_FOREHAND_VARIANT}"
    expected_train = [f"{prefix}/{motion}" for motion in TRAIN_MOTIONS]
    expected_val = [f"{prefix}/{motion}" for motion in VAL_MOTIONS]
    if list(train_conf.get("rel_dataset_path", ()) or ()) != expected_train:
        raise ValueError("training dataset is not the canonical ordered 22-motion split")
    if list(val_conf.get("rel_dataset_path", ()) or ()) != expected_val:
        raise ValueError("validation dataset is not the canonical ordered 5-motion split")

    def _validate_gmr(conf: Any, *, label: str) -> Path:
        if bool(conf.get("clear_cache", True)):
            raise ValueError(f"{label}.clear_cache must remain false for released caches")
        if str(conf.get("retargeting_method", "")) != "gmr":
            raise ValueError(f"{label}.retargeting_method must be gmr")
        gmr = conf.get("gmr_config", {})
        requirements = {
            "target_fps": 60.0,
            "solver": "daqp",
            "damping": 1.0,
            "use_velocity_limit": True,
        }
        for key, expected in requirements.items():
            actual = gmr.get(key, None)
            if isinstance(expected, float):
                try:
                    matches = float(actual) == expected
                except (TypeError, ValueError):
                    matches = False
            else:
                matches = actual == expected
            if not matches:
                raise ValueError(f"{label}.gmr_config.{key} changed from {expected!r}")
        ik_value = gmr.get("ik_config_path", None)
        if not ik_value:
            raise ValueError(f"{label}.gmr_config.ik_config_path is required")
        ik_path = Path(str(ik_value)).expanduser()
        if not ik_path.is_absolute():
            ik_path = Path(launch_dir).resolve() / ik_path
        return ik_path.resolve()

    train_ik = _validate_gmr(train_conf, label="training dataset")
    val_ik = _validate_gmr(val_conf, label="validation dataset")
    if train_ik != val_ik:
        raise ValueError("training and validation use different smooth IK contracts")

    release_validation = validate_release_manifest(
        dataset_root,
        release_path,
        recipe_path=recipe_path,
        ik_config_path=train_ik,
    )
    if release_validation.get("passed") is not True:
        raise ValueError(
            "raw_smooth_v1 release validation failed: "
            + "; ".join(str(item) for item in release_validation.get("errors", ()))
        )
    qc = inspect_canonical_dataset(
        dataset_root,
        source_variant=_CANONICAL_FOREHAND_VARIANT,
        cache_variant=_CANONICAL_FOREHAND_VARIANT,
    )
    if qc.get("passed") is not True or qc.get("clean_passed") is not True:
        details = [*list(qc.get("hard_errors", ()) or ()), *list(qc.get("warnings", ()) or ())]
        raise ValueError("raw_smooth_v1 strict data QC failed: " + "; ".join(map(str, details)))
    qc_sha256 = _canonical_json_sha256(qc)
    release_file_sha256 = hashlib.sha256(release_path.read_bytes()).hexdigest()
    visual_qc_path = release_path.with_name("visual_qc_report.json")
    visual_validation = validate_visual_qc_report(dataset_root.parents[1], visual_qc_path)
    if visual_validation.get("passed") is not True:
        raise ValueError(
            "raw_smooth_v1 visual QC validation failed: "
            + "; ".join(str(item) for item in visual_validation.get("errors", ()))
        )
    identity = {
        "release_sha256": release_validation.get("release_sha256"),
        "release_manifest_content_sha256": release_file_sha256,
        "qc_contract_sha256": qc_sha256,
        "visual_qc_report_content_sha256": visual_validation.get("report_sha256"),
    }
    identity["preflight_binding_sha256"] = _canonical_json_sha256(identity)
    with open_dict(config.experiment):
        for key, value in identity.items():
            config.experiment.training_source[key] = value

    report: dict[str, Any] = {
        "schema_version": "raw_smooth_v1_training_source_preflight_v1",
        "dataset_root": str(dataset_root),
        "source_variant": _CANONICAL_FOREHAND_VARIANT,
        "cache_variant": _CANONICAL_FOREHAND_VARIANT,
        "source_fps": 60.0,
        "cache_fps": 100.0,
        "release_manifest": str(release_path),
        "visual_qc_report": str(visual_qc_path),
        **identity,
        "train_motions": list(TRAIN_MOTIONS),
        "validation_motions": list(VAL_MOTIONS),
        "clean_passed": True,
        "passed": True,
    }
    _atomic_write_json(Path(result_dir) / "training_source_preflight.json", report)
    return report


def setup_jax_cache() -> None:
    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR") or os.path.join(Path.home(), ".musclemimic", ".jax_cache")
    cache_dir = str(Path(cache_dir).expanduser().resolve())
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_dir
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    logger.info(f"JAX compilation cache enabled at: {cache_dir}")


def setup_wandb(config) -> tuple[bool, Any]:
    import wandb

    use_wandb = config.wandb.get("mode", "online") != "disabled"
    if not use_wandb:
        logger.info("Wandb logging disabled")
        return False, None
    wandb.login()
    config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    params = {"project": config.wandb.project, "config": config_dict}
    if "dir" in config.wandb and config.wandb.dir:
        Path(str(config.wandb.dir)).mkdir(parents=True, exist_ok=True)
        params["dir"] = str(config.wandb.dir)
    if "name" in config.wandb and config.wandb.name:
        params["name"] = str(config.wandb.name)
    if "id" in config.wandb and config.wandb.id:
        params["id"] = str(config.wandb.id)
    if "resume" in config.wandb and config.wandb.resume:
        params["resume"] = config.wandb.resume
    if "tags" in config.wandb and config.wandb.tags:
        params["tags"] = config.wandb.tags
    run = wandb.init(**params)
    # Keep charts and media keyed to the physical environment timestep even
    # when W&B has to replay buffered history or multiple writers append to a
    # run.  W&B's internal ``_step`` is an ingestion-order field in those
    # cases and can diverge from the actual training progress.
    run.define_metric("Current Timestep")
    run.define_metric("*", step_metric="Current Timestep", step_sync=True)
    return True, run


def configure_action_training_outputs(config, launch_dir: str) -> str | None:
    """Route local training artifacts into datasets/<action>/training.

    This is intentionally runtime-level rather than config-file-only so older
    action configs with hard-coded dataset paths also get action-scoped outputs.
    """
    training_root = resolve_training_root(config.experiment, launch_dir)
    if training_root is None:
        return None

    action = infer_training_action(config.experiment)
    training_root_path = Path(training_root)
    training_root_path.mkdir(parents=True, exist_ok=True)

    checkpoint_root = training_root_path / "checkpoints"
    validation_video_dir = training_root_path / "validation_videos"
    wandb_dir = training_root_path / "wandb"
    for directory in (checkpoint_root, validation_video_dir, wandb_dir):
        directory.mkdir(parents=True, exist_ok=True)

    with open_dict(config.experiment):
        config.experiment.training_root = str(training_root_path)
        config.experiment.checkpoint_root = str(checkpoint_root)
        config.experiment.validation_video_dir = str(validation_video_dir)

    if hasattr(config, "wandb"):
        with open_dict(config.wandb):
            config.wandb.dir = str(wandb_dir)
            if action and not config.wandb.get("name", None):
                run_id = config.experiment.get("run_id", None)
                config.wandb.name = f"{action}-{run_id}" if run_id else action
            tags = list(config.wandb.get("tags", []) or [])
            if action and action not in tags:
                tags.append(action)
            config.wandb.tags = tags

    logger.info(f"Training artifact root: {training_root_path}")
    logger.info(f"Checkpoint root: {checkpoint_root}")
    logger.info(f"Validation videos: {validation_video_dir}")
    return str(training_root_path)


# Dataset config keys that may appear in validation config
_VAL_DATASET_CONF_KEYS = ("amass_dataset_conf", "lafan1_dataset_conf", "custom_dataset_conf")

# Body-level validation quantities that require full body data
_BODY_QUANTITIES = ("BodyPosition", "BodyOrientation", "BodyVelocity")


def _validation_needs_body_data(config) -> bool:
    """Check if validation metrics require body-level trajectory data."""
    val_cfg = config.experiment.get("validation", None)
    if not val_cfg or not val_cfg.get("active", False):
        return False
    quantities = val_cfg.get("quantities", None)
    if not quantities:
        return False
    for q in quantities:
        if q in _BODY_QUANTITIES:
            return True
    return False


def _auto_set_skip_body_data(config) -> None:
    """Auto-set skip_body_data=True if validation doesn't need body quantities.

    When validation doesn't use BodyPosition/BodyOrientation/BodyVelocity metrics,
    we can skip storing xpos_parent/xquat_parent and only keep cvel_parent/subtree_com_root
    for site velocity computation. This saves additional memory.

    Note: sparse_body_data=True is already the default, which skips full body arrays.
    This function enables skip_body_data for even more savings when body metrics aren't needed.
    """
    if _validation_needs_body_data(config):
        return

    # Check if amass_dataset_conf exists
    task_params = config.experiment.get("task_factory", {}).get("params", {})
    amass_conf = task_params.get("amass_dataset_conf", None)
    if amass_conf is None:
        return

    # Skip if already explicitly set
    if amass_conf.get("skip_body_data", None) is not None:
        return

    # Also skip if lite mode is enabled (lite already implies minimal data)
    if amass_conf.get("lite", False):
        return

    # Auto-enable skip_body_data
    with open_dict(config):
        config.experiment.task_factory.params.amass_dataset_conf.skip_body_data = True
    logger.info(
        "[Trajectory] Auto-enabled skip_body_data (validation doesn't use BodyPosition/BodyOrientation/BodyVelocity)."
    )


def _has_validation_dataset_override(val_cfg) -> bool:
    """Check if validation config has its own dataset configuration."""
    if val_cfg is None:
        return False
    for key in _VAL_DATASET_CONF_KEYS:
        if val_cfg.get(key, None) is not None:
            return True
    # Also check nested validation.task_factory.params
    val_task = val_cfg.get("task_factory", None)
    if val_task is None:
        return False
    if OmegaConf.is_config(val_task):
        val_task = OmegaConf.to_container(val_task, resolve=True)
    if isinstance(val_task, dict):
        params = val_task.get("params")
        if OmegaConf.is_config(params):
            params = OmegaConf.to_container(params, resolve=True)
        if isinstance(params, dict):
            for key in _VAL_DATASET_CONF_KEYS:
                if params.get(key, None) is not None:
                    return True
    return False


def _can_share_trajectory_handler(config) -> bool:
    """Check if validation env can share trajectory with training env."""
    val_cfg = config.experiment.get("validation", None)
    if not val_cfg:
        return False
    if not val_cfg.get("share_trajectory_handler", True):
        return False
    if _has_validation_dataset_override(val_cfg):
        return False
    return True


def _maybe_share_validation_trajectory(env, val_env, config) -> None:
    """Share trajectory data from training env to validation env if possible."""
    if not _can_share_trajectory_handler(config):
        if val_env is not None and getattr(val_env, "th", None) is not None:
            logger.info("Using separate validation trajectory (share disabled or dataset override).")
        return
    if env is None or val_env is None:
        return
    if getattr(env, "th", None) is None:
        return

    # val_env.th is None when we skipped loading - create handler using shared trajectory
    if getattr(val_env, "th", None) is None:
        val_env.load_trajectory(traj=env.th.traj, warn=False)
        # Convert to JAX if training env's trajectory is in JAX
        if getattr(env, "mjx_enabled", False) and not env.th.is_numpy and val_env.th.is_numpy:
            val_env.th.to_jax()
        logger.info("Created validation trajectory handler using shared trajectory data.")
        return

    # val_env already has th - just share the trajectory data reference
    if env.th.traj is val_env.th.traj:
        logger.info("Validation trajectory data already shared.")
        return
    val_env.th.traj = env.th.traj
    val_env.th._is_numpy = env.th.is_numpy
    if hasattr(val_env, "_finalize_traj_load"):
        val_env._finalize_traj_load()
    logger.info("Sharing validation trajectory data with training env.")


def instantiate_env(config) -> Any:
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

    # Convert trajectory to JAX before JIT compilation. This must happen here
    # because mjx_reset() runs inside JIT-traced functions. Calling to_jax()
    # inside JIT triggers TracerBoolConversionError due to Python control flow
    # in TrajectoryInfo.__post_init__.
    if getattr(env, "mjx_enabled", False) and getattr(env, "th", None) is not None and env.th.is_numpy:
        env.th.to_jax()

    return env


def instantiate_validation_env(config, share_trajectory: bool = False) -> Any:
    """Create a separate validation environment with validation-specific terminal state handler.

    Args:
        config: Experiment configuration.
        share_trajectory: If True, skip trajectory loading (will share from training env later).
    """
    if not config.experiment.get("validation", {}).get("active", False):
        return None

    # Get validation terminal state type, default to NoTerminalStateHandler
    val_terminal_state_type = config.experiment.validation.get("terminal_state_type", "NoTerminalStateHandler")

    # Create a copy of env_params with validation-specific settings
    val_env_params = OmegaConf.to_container(config.experiment.env_params, resolve=True)

    # Override terminal state handler for validation
    val_env_params["terminal_state_type"] = val_terminal_state_type
    val_env_params["terminal_state_params"] = config.experiment.validation.get("terminal_state_params", {})

    # Start from beginning of each trajectory (random trajectory, but step 0)
    if config.experiment.validation.get("start_from_beginning", False):
        if "th_params" not in val_env_params:
            val_env_params["th_params"] = {}
        val_env_params["th_params"]["start_from_random_step"] = False

    # Use validation num_envs if specified
    val_env_params["num_envs"] = config.experiment.validation.get("num_envs", config.experiment.env_params.num_envs)

    # Override th_params for validation (e.g., random_start: false for deterministic eval)
    val_th_params = config.experiment.validation.get("th_params", None)
    if val_th_params is not None:
        base_th_params = val_env_params.get("th_params", {}) or {}
        val_env_params["th_params"] = {**base_th_params, **OmegaConf.to_container(val_th_params, resolve=True)}

    # Prepare task_factory params, optionally skipping trajectory loading if sharing
    task_factory_params = OmegaConf.to_container(config.experiment.task_factory.params, resolve=True)
    if share_trajectory:
        for key in _VAL_DATASET_CONF_KEYS:
            task_factory_params.pop(key, None)
        logger.info("Skipping validation trajectory load (will share from training env).")
    else:
        # Apply validation-specific dataset overrides if present
        val_cfg = config.experiment.validation
        for key in _VAL_DATASET_CONF_KEYS:
            val_dataset = val_cfg.get(key, None)
            if val_dataset is not None:
                task_factory_params[key] = (
                    OmegaConf.to_container(val_dataset, resolve=True)
                    if OmegaConf.is_config(val_dataset)
                    else val_dataset
                )
                logger.info(f"Using validation-specific {key}.")

    # Create validation environment
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    val_env = factory.make(**val_env_params, **task_factory_params)

    # Convert trajectory to JAX before JIT (see instantiate_env for details)
    if getattr(val_env, "mjx_enabled", False) and getattr(val_env, "th", None) is not None and val_env.th.is_numpy:
        val_env.th.to_jax()

    return val_env


def pick_algorithm(config) -> Any:
    from musclemimic.algorithms import PPOJax

    algorithm_name = config.experiment.get("algorithm", "PPOJax")
    if algorithm_name == "PPOJax":
        return PPOJax
    raise ValueError(f"Unknown algorithm: {algorithm_name}")


def build_agent_conf(algorithm_cls, env, config):
    return algorithm_cls.init_agent_conf(env, config)


def build_metrics_handler(config, env):
    active = getattr(config.experiment, "validation", {}).get("active", False)
    return MetricsHandler(config, env) if active else None


def build_logging_callback(env, config, agent_conf, use_wandb, hooks: ExperimentHooks):
    import wandb as _wandb

    algorithm_name = config.experiment.get("algorithm", "PPOJax")

    def _cb(metrics_dict: dict[str, Any]):
        if metrics_dict.get("_log_only", False):
            log_dict: dict[str, Any] = {}
            for k, v in metrics_dict.items():
                if k.startswith("model/"):
                    label = k.split("/", 1)[1]
                    log_dict[f"Model/{label}"] = v
            if log_dict:
                if use_wandb:
                    _wandb.log(log_dict, step=0)
                else:
                    logger.info(log_dict)
            return

        # Determine which timestep to use for logging
        # reset_logging_timestep=True: Start from 0
        # reset_logging_timestep=False: Continue from checkpoint (resume the
        # timestep logging as well)
        reset_logging = config.experiment.get("reset_logging_timestep", False)

        if reset_logging:
            # Use timestep since resume start (jax_raw_timestep)
            current_timestep = int(metrics_dict.get("jax_raw_timestep", 0.0))
        else:
            # Use global timestep including checkpoint offset (max_timestep)
            current_timestep = int(metrics_dict.get("max_timestep", metrics_dict.get("jax_raw_timestep", 0.0)))

        jax_timestep = metrics_dict.get("jax_raw_timestep", metrics_dict.get("max_timestep", 0.0))
        # Host-side promotion validation callbacks intentionally contain only
        # validation metrics.  Do not manufacture zero-valued training metrics
        # for those callbacks: logging them at the checkpoint timestep
        # overwrites the real training return/length and creates periodic zero
        # spikes in W&B.
        log_dict: dict[str, Any] = {
            "Current Timestep": current_timestep,
            "Raw JAX Timestep": int(jax_timestep),
        }
        if "mean_episode_return" in metrics_dict:
            log_dict["Mean Episode Return"] = metrics_dict["mean_episode_return"]
        if "mean_episode_length" in metrics_dict:
            log_dict["Mean Episode Length"] = metrics_dict["mean_episode_length"]
        if "learning_rate" in metrics_dict:
            log_dict["Learning Rate"] = metrics_dict["learning_rate"]

        # Pass through all prefixed metrics (ppo/, reward/, adaptive/, etc.)
        # Define display names for metric prefixes
        prefix_display = {"ppo": "PPO"}
        for k, v in metrics_dict.items():
            if "/" in k:
                prefix, suffix = k.split("/", 1)
                display_prefix = prefix_display.get(prefix, prefix)
                log_dict[f"{display_prefix}/{suffix}"] = v

        # Forward validation metrics from ValidationSummary.
        if metrics_dict.get("has_validation_update", False):
            for k, v in metrics_dict.items():
                if not k.startswith("val_"):
                    continue
                key_body = k[len("val_") :]
                # Summary metrics.
                if key_body == "mean_episode_return":
                    log_dict["Validation/Mean Episode Return"] = v
                    continue
                if key_body == "mean_episode_length":
                    log_dict["Validation/Mean Episode Length"] = v
                    continue
                # Detailed validation measures.
                log_dict[f"Validation Measures/{key_body}"] = v

            # Sweep metric for HPO: sum Euclidean site measures
            s_rpos = metrics_dict.get("val_euclidean_distance_site_rpos", None)
            s_rrot = metrics_dict.get("val_euclidean_distance_site_rrotvec", None)
            s_rvel = metrics_dict.get("val_euclidean_distance_site_rvel", None)
            if s_rpos is not None and s_rrot is not None and s_rvel is not None:
                sweep_metric = float(s_rpos) + float(s_rrot) + float(s_rvel)
                log_dict["Metric for Sweep"] = sweep_metric
                logger.info(f"Combined sweep metric: {sweep_metric:.4f}")

        # Experiment-specific log enrichment.
        hooks.enrich_log(log_dict, metrics_dict, env)

        if use_wandb:
            _wandb.log(log_dict, step=current_timestep)

        # Trigger validation video recording.
        if (
            algorithm_name == "PPOJax"
            and metrics_dict.get("has_validation_update", False)
            and "_train_params" in metrics_dict
        ):
            # Build a temporary agent state from the current params.
            from musclemimic.algorithms import PPOJax, TrainState

            cur = metrics_dict["_train_params"]
            temp_state = TrainState(
                apply_fn=agent_conf.network.apply,
                tx=agent_conf.tx,
                params=cur["params"],
                run_stats=cur["run_stats"],
                opt_state=None,
                step=0,
            )
            temp_agent_state = PPOJax._agent_state(train_state=temp_state)
            recorder = getattr(hooks, "_video_recorder", None)
            review_set_required = bool(metrics_dict.get("_promotion_review_set_required", False))
            if recorder is None and review_set_required:
                raise RuntimeError("required endpoint visual review set has no configured recorder")
            if recorder is not None:
                try:
                    validation_number = getattr(hooks, "_validation_counter", 0) + 1
                    if metrics_dict.get("_promotion_review_set", False):
                        candidate = metrics_dict.get("_promotion_candidate")
                        if not isinstance(candidate, dict):
                            raise ValueError("promotion review-set callback has no checkpoint identity")
                        video_paths = recorder.record_review_set(
                            agent_conf=agent_conf,
                            agent_state=temp_agent_state,
                            validation_number=validation_number,
                            timestep=current_timestep,
                            candidate_identity=candidate,
                        )
                        video_path = video_paths[0] if video_paths else None
                    else:
                        video_path = recorder.record_episode(
                            agent_conf=agent_conf,
                            agent_state=temp_agent_state,
                            validation_number=validation_number,
                            timestep=current_timestep,
                        )
                    hooks._validation_counter = getattr(hooks, "_validation_counter", 0) + 1
                    if video_path:
                        logger.info(f"Validation video recorded: {video_path}")
                    hooks.on_validation_video(use_wandb, _wandb, video_path, current_timestep)
                except Exception as e:
                    if review_set_required:
                        raise RuntimeError("required endpoint visual review-set recording failed") from e
                    # Ordinary monitoring-video failures should not interrupt training.
                    logger.warning(f"Video recording failed: {e}")

    return _cb


def compute_training_rngs(config):
    n_seeds = int(config.experiment.get("n_seeds", 1))
    if "seeds" in config.experiment and config.experiment.seeds is not None:
        seeds = list(config.experiment.seeds)
        if len(seeds) != n_seeds:
            raise ValueError(f"Length of seeds ({len(seeds)}) must match n_seeds ({n_seeds})")
    else:
        seeds = list(range(n_seeds))
    keys = [jax.random.PRNGKey(int(s)) for s in seeds]
    return jnp.squeeze(jnp.vstack(keys)) if len(keys) > 1 else keys[0]


def build_train_fn(algorithm_cls, env, agent_conf, mh, logging_cb, logging_interval=1, val_env=None):
    return algorithm_cls.build_train_fn(
        env,
        agent_conf,
        mh=mh,
        online_logging_callback=logging_cb,
        logging_interval=logging_interval,
        val_env=val_env,
    )


def run_training(train_fn, rngs, *, host_controlled: bool = False):
    # vmap if multiple seeds
    if hasattr(rngs, "ndim") and rngs.ndim > 1:  # jnp array
        if host_controlled:
            raise ValueError("host-controlled promotion early stopping requires one seed")
        train_fn = jax.jit(jax.vmap(train_fn))
    elif host_controlled:
        # Stage-1/Stage-2 promotion inspects validation results and persists a
        # streak between compiled chunks.  The chunk scans are JIT-compiled in
        # the PPO runner; wrapping this host loop in a second outer JIT would
        # trace away the filesystem and early-break decisions.
        logger.info("Starting host-controlled validation-boundary training...")
        return train_fn(rngs)
    else:
        train_fn = jax.jit(train_fn)
    logger.info("Starting training...")
    return train_fn(rngs)


def validate_auto_resume_config(
    checkpoint_dir: str | Path,
    current_hash: str,
    *,
    strict: bool,
    expected_parent_lineage: dict[str, Any] | None = None,
) -> bool:
    """Apply the legacy warning or the production fail-fast hash policy."""

    compatible = validate_checkpoint_compatibility(
        checkpoint_dir,
        current_hash,
        require_manifest=strict or expected_parent_lineage is not None,
    )
    lineage_compatible = True
    if expected_parent_lineage is not None:
        lineage_compatible = validate_checkpoint_parent_lineage(
            checkpoint_dir,
            expected_parent_lineage,
        )
        if not lineage_compatible:
            raise ValueError(
                "auto-resume parent checkpoint lineage mismatch; the fixed run_id "
                "already belongs to a different or unbound parent checkpoint"
            )
    if strict and not compatible:
        raise ValueError(
            "auto-resume config hash mismatch for a strict training run; "
            "use a new run_id/checkpoint directory or restore the exact config"
        )
    return compatible and lineage_compatible


def _generate_run_suffix() -> str:
    """Return a per-run unique suffix for checkpoint directories."""
    ts = datetime.now(UTC).strftime("%y%m%dT%H%M%S")
    pid = os.getpid()
    job_num = os.environ.get("HYDRA_JOB_NUM")
    job_part = f"-job{job_num}" if job_num is not None else ""
    return f"{ts}-pid{pid}{job_part}-{uuid.uuid4().hex[:6]}"


def run_experiment(config, hooks: ExperimentHooks):
    # XLA flags
    os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True "

    from hydra.core.hydra_config import HydraConfig

    hydra_runtime = HydraConfig.get().runtime
    result_dir = hydra_runtime.output_dir
    launch_dir = hydra_runtime.cwd

    smoke_gate_config = config.experiment.get("continuity_smoke_gate", None)
    smoke_execution_config = config.experiment.get("training_smoke", None)
    formal_resolved_config_sha256 = None
    if smoke_gate_config is not None or smoke_execution_config is not None:
        from musclemimic.runner.continuity_smoke import resolved_training_config_sha256

        # Capture the exact Hydra resolution before runtime contracts and output
        # paths are injected.  The canonical smoke driver hashes the same
        # formal configuration before applying its short-run overrides.
        formal_resolved_config_sha256 = resolved_training_config_sha256(config)

    # Both gates run before W&B, environment construction, checkpoint creation,
    # or GPU work. Parent content identity is injected before the config hash is
    # computed, and declared legacy/experimental configs require an explicit
    # auditable opt-in.
    validate_experiment_config_status(config)
    parent_lineage = bind_explicit_parent_checkpoint(
        config,
        launch_dir=launch_dir,
    )

    # This runs before W&B, environment construction, checkpoint creation, or
    # any GPU work.  It also adds stable data identities to the experiment
    # config, so the checkpoint config hash is bound to the validated release.
    validate_training_source_preflight(
        config,
        launch_dir=launch_dir,
        result_dir=result_dir,
    )
    bind_continuity_training_release(
        config,
        launch_dir=launch_dir,
        result_dir=result_dir,
    )
    bind_emg_consistency_training_reference(
        config,
        launch_dir=launch_dir,
        result_dir=result_dir,
    )

    training_root = configure_action_training_outputs(config, launch_dir)

    setup_jax_cache()

    # Auto-optimize trajectory storage based on validation requirements
    _auto_set_skip_body_data(config)

    # Env and algo - share trajectory if possible to save memory
    env = instantiate_env(config)
    can_share = _can_share_trajectory_handler(config) and getattr(env, "th", None) is not None
    val_env = instantiate_validation_env(config, share_trajectory=can_share)
    _maybe_share_validation_trajectory(env, val_env, config)
    bind_emg_consistency_runtime_model(
        config,
        env=env,
        val_env=val_env,
        result_dir=result_dir,
    )

    # Contact tracking setup
    contact_tracking_cfg = config.experiment.get("contact_tracking", {})
    if contact_tracking_cfg.get("enabled", False):
        from musclemimic.badminton.asi.contact_tracking_data import (
            load_contact_tracking_bank,
            load_contact_tracking_data,
        )
        from musclemimic.distill.motion_identity import (
            MotionIdentityMap,
            resolve_config_motion_paths,
        )

        contract_version = str(contact_tracking_cfg.get("contract_version", "legacy_v1"))
        if contract_version not in {"legacy_v1", "event_reference_v2"}:
            raise ValueError(f"unknown contact_tracking.contract_version={contract_version!r}")
        train_bank_manifest = contact_tracking_cfg.get("event_reference_bank_manifest")
        validation_bank_manifest = contact_tracking_cfg.get("validation_event_reference_bank_manifest")
        cache_dir = contact_tracking_cfg.get("tracking_cache_dir")
        if contract_version == "legacy_v1":
            if train_bank_manifest is not None or validation_bank_manifest is not None:
                raise ValueError("event reference banks require contact_tracking.contract_version=event_reference_v2")
            if cache_dir is None:
                raise ValueError("contact_tracking.tracking_cache_dir must be set when contact_tracking.enabled=true")
            # This call deliberately retains the pre-v2 permissive cache
            # contract and never changes the validation environment.
            contact_data = load_contact_tracking_data(cache_dir, control_dt=env.dt)
        elif train_bank_manifest is not None:
            if cache_dir is not None:
                raise ValueError(
                    "contact_tracking must choose one event_reference_bank_manifest or tracking_cache_dir, not both"
                )
            train_identity = MotionIdentityMap.from_paths(resolve_config_motion_paths(config))
            contact_data = load_contact_tracking_bank(
                train_bank_manifest,
                control_dt=env.dt,
                motion_identity_map=train_identity,
            )
        else:
            if cache_dir is None:
                raise ValueError(
                    "contact_tracking requires tracking_cache_dir or event_reference_bank_manifest when enabled"
                )
            contact_data = load_contact_tracking_data(
                cache_dir,
                control_dt=env.dt,
                strict_contract=True,
            )
        foot_sites = list(
            contact_tracking_cfg.get(
                "foot_sites",
                ["left_toes_mimic", "right_toes_mimic", "left_ankle_mimic", "right_ankle_mimic"],
            )
        )
        env._reward_function.attach_contact_tracking(contact_data, foot_sites, env._model)
        # The separately fingerprinted validation bank belongs to the new
        # event-reference mode.  Preserve the legacy single-cache behavior so
        # existing contact-tracking jobs and checkpoints remain resumable.
        if contract_version == "event_reference_v2" and val_env is not None:
            if validation_bank_manifest is None:
                raise ValueError(
                    "active validation with contact tracking requires a separately bound "
                    "validation_event_reference_bank_manifest"
                )
            val_conf = config.experiment.validation.get("amass_dataset_conf", None)
            if val_conf is None:
                raise ValueError("contact tracking validation has no held-out motion config")
            val_paths = val_conf.get("rel_dataset_path", None)
            if not val_paths:
                raise ValueError("contact tracking validation requires explicit ordered rel_dataset_path")
            val_identity = MotionIdentityMap.from_paths(val_paths)
            val_contact_data = load_contact_tracking_bank(
                validation_bank_manifest,
                control_dt=val_env.dt,
                motion_identity_map=val_identity,
            )
            val_env._reward_function.attach_contact_tracking(
                val_contact_data,
                foot_sites,
                val_env._model,
            )
        logger.info(
            "Contact tracking: %s trajectory cache(s), %s foot sites",
            int(getattr(contact_data, "num_trajectories", 1)),
            len(foot_sites),
        )

    # NOTE: Wrapping is now handled entirely by algorithm._wrap_env methods
    # to avoid conflicts and ensure correct wrapper ordering
    algorithm_cls = pick_algorithm(config)
    agent_conf = build_agent_conf(algorithm_cls, env, config)

    if formal_resolved_config_sha256 is not None:
        from musclemimic.runner.continuity_smoke import validate_continuity_smoke_launch_gate

        validated_smoke = validate_continuity_smoke_launch_gate(
            config,
            formal_resolved_config_sha256=formal_resolved_config_sha256,
            repo_root=launch_dir,
        )
        if validated_smoke is not None:
            artifact_path = Path(str(config.experiment.continuity_smoke_gate.artifact_path)).expanduser()
            if not artifact_path.is_absolute():
                artifact_path = Path(launch_dir) / artifact_path
            smoke_contract = {
                "schema_version": "continuity_smoke_runtime_contract_v1",
                "artifact_path": str(artifact_path.resolve()),
                "artifact_fingerprint": validated_smoke["artifact_fingerprint"],
                "git_commit_sha": validated_smoke["git_commit_sha"],
                "formal_resolved_config_sha256": validated_smoke["formal_config"]["resolved_config_sha256"],
                "condition": validated_smoke["formal_config"]["condition"],
                "release_fingerprint": validated_smoke["contracts"]["release_fingerprint"],
                "basis_fingerprint": validated_smoke["contracts"]["basis_fingerprint"],
                "completed_at_utc": validated_smoke["completed_at_utc"],
            }
            smoke_contract["binding_sha256"] = _canonical_json_sha256(smoke_contract)
            with open_dict(config.experiment):
                config.experiment.continuity_smoke_contract = smoke_contract
            logger.info(
                "Continuity GPU smoke gate passed: artifact=%s fingerprint=%s",
                artifact_path,
                validated_smoke["artifact_fingerprint"],
            )

    # W&B starts only after every source/release/action/smoke gate has passed,
    # so a rejected launch cannot create a misleading remote run.
    use_wandb, run = setup_wandb(config)

    # Report total motion duration (post-concatenation), before training starts
    for label, _env in [("Train", env), ("Validation", val_env)]:
        if (
            _env is not None
            and hasattr(_env, "th")
            and _env.th is not None
            and hasattr(_env.th, "traj")
            and _env.th.traj is not None
            and hasattr(_env.th.traj, "info")
            and hasattr(_env.th.traj, "data")
            and hasattr(_env.th.traj.info, "frequency")
            and hasattr(_env.th.traj.data, "qpos")
        ):
            freq = float(_env.th.traj.info.frequency)
            frames = int(_env.th.traj.data.qpos.shape[0])
            if freq > 0.0 and frames > 0:
                duration_sec = (frames - 1) / freq
                duration_min = duration_sec / 60.0
                logger.info(
                    f"[{label}] total concatenated motion duration: {duration_min:.2f} minutes "
                    f"({duration_sec:.1f} s, {frames} frames @ {freq:.1f} Hz)"
                )

    # Metrics + hooks + video
    metrics_env = val_env if val_env is not None else env
    mh = build_metrics_handler(config, metrics_env)
    recorder_dir = config.experiment.get("validation_video_dir", result_dir)
    recorder = hooks.build_video_recorder(result_dir=recorder_dir, config=config)
    hooks._video_recorder = recorder

    # Logging callback
    logging_cb = build_logging_callback(env, config, agent_conf, use_wandb, hooks)

    # Checkpoint resume or fresh with auto-resume support
    explicit_resume = getattr(config.experiment, "resume_from", None)
    auto_resume = getattr(config.experiment, "auto_resume", True)
    run_id = getattr(config.experiment, "run_id", None)
    checkpoint_root = getattr(config.experiment, "checkpoint_root", None)

    exp_config_hash = config_hash(config.experiment)
    experiment_id = run_id or exp_config_hash

    # Resolve checkpoint directory path
    configured_ckpt_dir = getattr(config.experiment, "checkpoint_dir", "checkpoints") or "checkpoints"
    resolved_ckpt_dir = resolve_checkpoint_dir(
        configured_ckpt_dir=configured_ckpt_dir,
        launch_dir=launch_dir,
        result_dir=result_dir,
        experiment_id=experiment_id,
        auto_resume=auto_resume,
        checkpoint_root=checkpoint_root,
        training_root=training_root,
    )
    if not auto_resume:
        resolved_ckpt_dir = os.path.join(resolved_ckpt_dir, _generate_run_suffix())

    with open_dict(config.experiment):
        config.experiment.checkpoint_dir = resolved_ckpt_dir
    os.makedirs(resolved_ckpt_dir, exist_ok=True)

    # Determine resume path (auto-detect > explicit > fresh when auto_resume=true)
    resume_from = None
    apply_resume_resets = True
    if auto_resume:
        latest = find_latest_checkpoint(resolved_ckpt_dir)
        if latest:
            logger.info(f"Auto-resume: found checkpoint: {latest}")
            resume_from = latest
            apply_resume_resets = False
            validate_auto_resume_config(
                resolved_ckpt_dir,
                exp_config_hash,
                strict=bool(config.experiment.get("strict_auto_resume_config_hash", False)),
                expected_parent_lineage=parent_lineage,
            )
        elif explicit_resume:
            logger.info(f"Auto-resume: no local checkpoint, using explicit: {explicit_resume}")
            resume_from = explicit_resume
        else:
            logger.info(f"Auto-resume: no checkpoint in {resolved_ckpt_dir}, starting fresh")
    elif explicit_resume:
        logger.info(f"Resuming from explicit path: {explicit_resume}")
        resume_from = explicit_resume

    # A long preflight/environment build can overlap accidental filesystem
    # changes. Re-hash the explicit upstream checkpoint immediately before the
    # run manifest is committed and before the first restore. This ordering
    # prevents a failed startup from leaving a stale empty-run manifest.
    if resume_from is not None and apply_resume_resets and parent_lineage is not None:
        validate_explicit_parent_checkpoint(config.experiment, resume_from)

    # A matching tensor shape is not a sufficient restore contract: direct
    # 354-D and fixed-synergy policies can otherwise be silently interchanged,
    # and two synergy runs can share a latent rank while using different W/R.
    # Same-run resumes require the complete stage binding; an explicitly bound
    # cross-stage parent is allowed to rebind only its runtime/coverage layer.
    body_action_contract = config.experiment.get("body_synergy_contract", None)
    continuity_training_contract = config.experiment.get(
        "continuity_training_contract",
        None,
    )
    muscle_control_contract = config.experiment.get(
        "muscle_control_contract",
        None,
    )
    if resume_from is not None and continuity_training_contract is not None:
        validate_checkpoint_continuity_training_contract(
            resume_from,
            continuity_training_contract,
        )
    if resume_from is not None and muscle_control_contract is not None:
        from musclemimic.runner.checkpointing import (
            validate_checkpoint_muscle_control_contract,
        )

        validate_checkpoint_muscle_control_contract(
            resume_from,
            muscle_control_contract,
        )
    if resume_from is not None and body_action_contract is not None:
        from musclemimic.synergy.multistage_contract import (
            EXACT_RUNTIME_COMPATIBILITY,
            PORTABLE_COMPATIBILITY,
        )

        compatibility = (
            PORTABLE_COMPATIBILITY
            if apply_resume_resets and parent_lineage is not None
            else EXACT_RUNTIME_COMPATIBILITY
        )
        validate_checkpoint_body_action_contract(
            resume_from,
            body_action_contract,
            compatibility=compatibility,
            legacy_attestation=config.experiment.get(
                "legacy_parent_body_action_attestation",
                None,
            ),
        )

    # Write or validate the immutable run manifest only after all parent
    # identity checks pass. Existing manifests are checked even if the run has
    # not produced a checkpoint yet.
    write_manifest(resolved_ckpt_dir, config.experiment, exp_config_hash)

    # Update config with detected resume path for resume_or_fresh
    with open_dict(config.experiment):
        config.experiment.resume_from = resume_from

    logger.info(f"Checkpoint directory: {resolved_ckpt_dir}")
    logger.info(f"Experiment ID (config hash): {exp_config_hash}")

    train_fn = resume_or_fresh(
        env,
        agent_conf,
        algorithm_cls,
        config,
        mh,
        logging_cb,
        logging_interval=config.experiment.get("online_logging_interval", 1),
        val_env=val_env,
        apply_resume_resets=apply_resume_resets,
    )

    # Seeds and training
    rngs = compute_training_rngs(config)
    promotion_cfg = config.experiment.get("promotion", {})
    stage1_fixed_schedule = config.experiment.get("stage1_peasd", None) is not None
    training_result = run_training(
        train_fn,
        rngs,
        host_controlled=bool(promotion_cfg.get("auto_stop", False)) or stage1_fixed_schedule,
    )

    # Close any cached checkpoint manager created during training (host-side cleanup)
    cache_entry = getattr(create_jax_checkpoint_host_callback, "__cached_instance__", None)
    if cache_entry is not None:
        ckpt_manager = cache_entry[2]  # (cache_key, ckpt_cb, ckpt_mgr)
        ckpt_manager.close()
        delattr(create_jax_checkpoint_host_callback, "__cached_instance__")

    smoke_execution = config.experiment.get("training_smoke", {})
    if bool(smoke_execution.get("enabled", False)):
        from musclemimic.runner.continuity_smoke import (
            build_continuity_training_smoke_artifact,
            write_continuity_training_smoke,
        )

        output_path = str(smoke_execution.get("output_json", "") or "").strip()
        if not output_path:
            raise ValueError("training_smoke.output_json is required")
        smoke_artifact = build_continuity_training_smoke_artifact(
            config=config,
            env=env,
            training_result=training_result,
            checkpoint_dir=resolved_ckpt_dir,
            agent_conf=agent_conf,
            repo_root=launch_dir,
        )
        write_continuity_training_smoke(output_path, smoke_artifact)
        if smoke_artifact["passed"] is not True:
            raise RuntimeError(
                "continuity GPU smoke failed: " + "; ".join(str(error) for error in smoke_artifact["errors"])
            )
        logger.info(
            "Continuity GPU smoke passed: artifact=%s fingerprint=%s",
            output_path,
            smoke_artifact["artifact_fingerprint"],
        )

    if use_wandb and run is not None:
        import wandb as _wandb

        _wandb.finish()

    return training_result

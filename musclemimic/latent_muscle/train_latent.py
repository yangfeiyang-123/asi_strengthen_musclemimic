"""Minimal latent distillation trainer for posterior/prior/decoder modules."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from musclemimic.distill.action_schema import ordered_schema_hash
from musclemimic.distill.dataset import SequenceDistillDataset, motion_split_datasets
from musclemimic.distill.provenance import (
    checkpoint_content_fingerprint,
    file_sha256,
    test_only_unpromoted_teacher_binding,
    validate_dataset_manifest,
    validate_stage2_teacher_promotion,
)
from musclemimic.latent_muscle.action_mask import ActionMask
from musclemimic.latent_muscle.checkpoint import save_latent_checkpoint
from musclemimic.latent_muscle.losses import latent_distillation_loss, positive_sigma
from musclemimic.latent_muscle.networks import (
    ConditionalPrior,
    LatentDecoder,
    PosteriorEncoder,
    reparameterize_gaussian,
)
from musclemimic.latent_muscle.normalization import ObservationNormalizer


@dataclass(frozen=True)
class LatentTrainConfig:
    dataset_dir: str
    output_dir: str
    latent_dim: int = 32
    hidden_layer_dims: tuple[int, ...] = (512, 256)
    batch_size: int = 256
    horizon: int = 8
    num_steps: int = 100_000
    learning_rate: float = 3e-4
    seed: int = 0
    kl_weight: float = 1e-3
    kl_warmup_steps: int = 10_000
    free_bits: float = 0.0
    smooth_weight: float = 0.0
    bound_weight: float = 0.0
    action_min: float = -1.0
    action_max: float = 1.0
    sigma_min: float = 0.05
    sigma_max: float = 2.0
    log_interval: int = 100
    action_mask: dict[str, Any] | None = None
    val_fraction: float = 0.0
    motion_field: str = "motion_uid"
    strict_motion_identity: bool = False
    normalizer_epsilon: float = 1e-6
    normalizer_clip: float = 10.0
    max_eval_samples: int = 65_536
    direct_bc_action_mse: float | None = None
    direct_bc_metrics_path: str | None = None
    teacher_ckpt: str | None = None
    require_direct_bc_baseline: bool = False
    require_closed_loop_metrics: bool = False
    require_dataset_provenance: bool = False
    teacher_promotion_manifest: str | None = None
    test_only_allow_unpromoted_teacher: bool = False
    promotion_gates: dict[str, float] | None = None
    closed_loop_evaluator: Callable[[dict[str, Any]], dict[str, Any]] | None = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class LatentTrainResult:
    checkpoint_dir: str
    final_total_loss: float
    final_action_mse: float


def kl_warmup_weight(*, step: int, target: float, warmup_steps: int) -> float:
    if int(warmup_steps) <= 0:
        return float(target)
    return float(target) * min(1.0, max(0.0, float(step) / float(warmup_steps)))


def teacher_delta_smooth_mse(predicted_sequence, teacher_sequence):
    """MSE between student and teacher temporal action deltas."""
    predicted = jnp.asarray(predicted_sequence)
    teacher = jnp.asarray(teacher_sequence)
    if predicted.shape != teacher.shape or predicted.ndim < 2:
        raise ValueError(
            "predicted/teacher sequences must have the same [..., time, action] shape"
        )
    if predicted.shape[-2] < 2:
        return jnp.asarray(0.0, dtype=predicted.dtype)
    predicted_delta = predicted[..., 1:, :] - predicted[..., :-1, :]
    teacher_delta = teacher[..., 1:, :] - teacher[..., :-1, :]
    return jnp.mean(jnp.square(predicted_delta - teacher_delta))


def train_latent(config: LatentTrainConfig) -> LatentTrainResult:
    dataset_manifest = None
    teacher_fingerprint = None
    teacher_promotion = None
    teacher_promotion_evidence_kind = None
    dataset_path = Path(config.dataset_dir)
    if config.require_dataset_provenance or (dataset_path / "dataset_manifest.json").is_file():
        if config.require_dataset_provenance and not config.teacher_ckpt:
            raise ValueError("production latent training requires teacher_ckpt provenance")
        teacher_fingerprint = (
            None
            if config.teacher_ckpt is None
            else checkpoint_content_fingerprint(config.teacher_ckpt)
        )
        if config.teacher_promotion_manifest is not None:
            if config.test_only_allow_unpromoted_teacher:
                raise ValueError(
                    "teacher_promotion_manifest and test-only bypass are mutually exclusive"
                )
            teacher_promotion = validate_stage2_teacher_promotion(
                config.teacher_promotion_manifest,
                teacher_checkpoint=teacher_fingerprint,
            )
            teacher_promotion_evidence_kind = "verified_stage2_promotion_v1"
        elif config.test_only_allow_unpromoted_teacher:
            teacher_promotion = test_only_unpromoted_teacher_binding(
                teacher_fingerprint
            )
            teacher_promotion_evidence_kind = "test_only_unpromoted_teacher"
        elif config.require_dataset_provenance:
            raise ValueError(
                "production latent training requires teacher_promotion_manifest"
            )
        dataset_manifest = validate_dataset_manifest(
            dataset_path,
            expected_teacher=teacher_fingerprint,
            expected_teacher_promotion=teacher_promotion,
            require_promoted_teacher=bool(
                config.require_dataset_provenance
                and not config.test_only_allow_unpromoted_teacher
            ),
        )
        if teacher_promotion is None:
            teacher_promotion = dataset_manifest["teacher_promotion"]
            teacher_promotion_evidence_kind = (
                "test_only_unpromoted_teacher"
                if teacher_promotion.get("test_only") is True
                else "verified_stage2_promotion_v1"
            )
    target_body_names = _target_body_names(config.action_mask)
    split_motion_field = str(config.motion_field)
    if not config.strict_motion_identity and split_motion_field == "motion_uid":
        # Legacy train-only shards can still be used for small regression tests,
        # but production YAML enables strict identities and never takes this path.
        candidate = Path(config.dataset_dir)
        first_shard = next(iter(sorted(candidate.glob("train_*.npz")) or sorted(candidate.glob("shard_*.npz"))), None)
        if first_shard is not None:
            with np.load(first_shard) as shard:
                if "motion_uid" not in shard.files:
                    split_motion_field = "traj_no"
    dataset, val_dataset, split_manifest = motion_split_datasets(
        config.dataset_dir,
        dataset_cls=SequenceDistillDataset,
        seed=int(config.seed),
        val_fraction=float(config.val_fraction),
        motion_field=split_motion_field,
        target_actuator_names=target_body_names,
        require_stable_ids=bool(config.strict_motion_identity),
    )
    action_dim = int(dataset.action_dim)
    action_mask = _build_action_mask(config.action_mask, action_dim, dataset.actuator_names)
    if action_mask.body_size != action_dim:
        raise ValueError(
            f"decoder action_dim={action_dim} must equal action mask body size={action_mask.body_size}"
        )
    if list(dataset.actuator_names) != list(action_mask.body_actuator_names):
        raise ValueError(
            "latent decoder target actuator names/order differ from ActionMask body partition"
        )
    if config.strict_motion_identity and dataset.actuator_ctrlrange is None:
        raise ValueError("production latent training requires ordered teacher actuator_ctrlrange metadata")
    normalizer = ObservationNormalizer.fit(
        dataset.arrays["student_obs"],
        epsilon=float(config.normalizer_epsilon),
        clip=float(config.normalizer_clip),
    )

    posterior = PosteriorEncoder(
        latent_dim=int(config.latent_dim),
        hidden_layer_dims=tuple(config.hidden_layer_dims),
        sigma_min=float(config.sigma_min),
        sigma_max=float(config.sigma_max),
    )
    prior = ConditionalPrior(
        latent_dim=int(config.latent_dim),
        hidden_layer_dims=tuple(config.hidden_layer_dims),
        sigma_min=float(config.sigma_min),
        sigma_max=float(config.sigma_max),
    )
    decoder = LatentDecoder(
        action_dim=action_dim,
        hidden_layer_dims=tuple(config.hidden_layer_dims),
        bounded_action=True,
    )

    rng = jax.random.PRNGKey(int(config.seed))
    init_batch = next(dataset.iter_sequence_batches(batch_size=1, horizon=config.horizon, shuffle=False))
    init_state = normalizer.normalize_jax(
        init_batch["student_obs"].reshape(-1, dataset.student_obs_dim)
    )
    init_ref = jnp.asarray(init_batch["reference_features"].reshape(-1, dataset.reference_features_dim), dtype=jnp.float32)
    init_latent = jnp.zeros((init_state.shape[0], int(config.latent_dim)), dtype=jnp.float32)
    rng, post_rng, prior_rng, dec_rng = jax.random.split(rng, 4)
    variables = {
        "encoder": posterior.init(post_rng, init_state, init_ref),
        "prior": prior.init(prior_rng, init_state),
        "decoder": decoder.init(dec_rng, init_state, init_latent),
    }
    tx = optax.adam(float(config.learning_rate))
    opt_state = tx.init(variables)
    batch_iter = dataset.iter_sequence_batches(
        batch_size=int(config.batch_size),
        horizon=int(config.horizon),
        shuffle=True,
        repeat=True,
    )
    metrics_history: list[dict[str, Any]] = []

    def loss_fn(params, batch, step_rng, kl_weight):
        state = normalizer.normalize_jax(batch["student_obs"])
        reference = jnp.asarray(batch["reference_features"], dtype=jnp.float32)
        teacher_action = jnp.asarray(batch["teacher_action"], dtype=jnp.float32)
        batch_size, horizon = state.shape[:2]
        flat_state = state.reshape((batch_size * horizon, state.shape[-1]))
        flat_reference = reference.reshape((batch_size * horizon, reference.shape[-1]))
        flat_teacher_action = teacher_action.reshape((batch_size * horizon, teacher_action.shape[-1]))

        posterior_mu, posterior_raw_sigma = posterior.apply(params["encoder"], flat_state, flat_reference)
        prior_mu, prior_raw_sigma = prior.apply(params["prior"], flat_state)
        z = reparameterize_gaussian(
            step_rng,
            posterior_mu,
            posterior_raw_sigma,
            sigma_min=float(config.sigma_min),
            sigma_max=float(config.sigma_max),
        )
        predicted_action = decoder.apply(params["decoder"], flat_state, z)
        losses = latent_distillation_loss(
            predicted_action=predicted_action,
            teacher_action=flat_teacher_action,
            posterior_mu=posterior_mu,
            posterior_raw_sigma=posterior_raw_sigma,
            prior_mu=prior_mu,
            prior_raw_sigma=prior_raw_sigma,
            action_weight=1.0,
            kl_weight=kl_weight,
            free_bits=float(config.free_bits),
            smooth_weight=0.0,
            bound_weight=float(config.bound_weight),
            action_min=float(config.action_min),
            action_max=float(config.action_max),
            sigma_min=float(config.sigma_min),
            sigma_max=float(config.sigma_max),
        )
        pred_seq = predicted_action.reshape((batch_size, horizon, teacher_action.shape[-1]))
        if horizon > 1 and float(config.smooth_weight):
            teacher_seq = teacher_action.reshape(
                (batch_size, horizon, teacher_action.shape[-1])
            )
            # Penalize temporal tracking error, not motion itself.  The old
            # ``||a_t-a_{t-1}||`` objective suppressed the teacher's genuine
            # high acceleration around contact and biased the decoder toward
            # a flat action.  This term preserves the teacher delta while
            # damping only student-specific chatter.
            smooth_mse = teacher_delta_smooth_mse(pred_seq, teacher_seq)
        else:
            smooth_mse = jnp.asarray(0.0, dtype=losses["total_loss"].dtype)
        posterior_sigma = positive_sigma(
            posterior_raw_sigma,
            sigma_min=float(config.sigma_min),
            sigma_max=float(config.sigma_max),
        )
        prior_sigma = positive_sigma(
            prior_raw_sigma,
            sigma_min=float(config.sigma_min),
            sigma_max=float(config.sigma_max),
        )
        total = losses["total_loss"] + float(config.smooth_weight) * smooth_mse
        diagnostics = losses | {
            "total_loss": total,
            "smooth_mse": smooth_mse,
            "posterior_sigma_mean": jnp.mean(posterior_sigma),
            "posterior_sigma_min": jnp.min(posterior_sigma),
            "posterior_sigma_max": jnp.max(posterior_sigma),
            "prior_sigma_mean": jnp.mean(prior_sigma),
            "prior_sigma_min": jnp.min(prior_sigma),
            "prior_sigma_max": jnp.max(prior_sigma),
            "posterior_prior_mu_l2": jnp.mean(jnp.linalg.norm(posterior_mu - prior_mu, axis=-1)),
            "latent_dim_active_count": jnp.sum(jnp.std(z, axis=0) > 1e-3),
            "decoder_action_mean": jnp.mean(predicted_action),
            "decoder_action_std": jnp.std(predicted_action),
            "decoder_action_min": jnp.min(predicted_action),
            "decoder_action_max": jnp.max(predicted_action),
        }
        return total, diagnostics

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    final_metrics: dict[str, Any] | None = None
    for step in range(1, int(config.num_steps) + 1):
        batch = _batch_to_jax(next(batch_iter))
        rng, step_rng = jax.random.split(rng)
        current_kl_weight = kl_warmup_weight(
            step=step,
            target=float(config.kl_weight),
            warmup_steps=int(config.kl_warmup_steps),
        )
        (_loss, metrics), grads = grad_fn(variables, batch, step_rng, current_kl_weight)
        updates, opt_state = tx.update(grads, opt_state, variables)
        variables = optax.apply_updates(variables, updates)
        final_metrics = _metrics_to_float(metrics | {"step": step, "kl_weight": current_kl_weight})
        metrics_history.append(final_metrics)
        if config.log_interval and step % int(config.log_interval) == 0:
            print(
                f"[latent] step={step} loss={final_metrics['total_loss']:.6f} "
                f"action_mse={final_metrics['action_mse']:.6f} kl={final_metrics['kl']:.6f}"
            )

    evaluation_dataset = val_dataset if val_dataset is not None else dataset
    eval_metrics = _evaluate_once(
        variables=variables,
        posterior=posterior,
        prior=prior,
        decoder=decoder,
        dataset=evaluation_dataset,
        config=config,
        normalizer=normalizer,
    )
    eval_metrics["evaluation_split"] = "val" if val_dataset is not None else "train"
    eval_metrics["teacher_promotion_evidence_kind"] = teacher_promotion_evidence_kind
    if config.closed_loop_evaluator is not None:
        closed_loop = dict(
            config.closed_loop_evaluator(
                {
                    "variables": variables,
                    "posterior": posterior,
                    "prior": prior,
                    "decoder": decoder,
                    "normalizer": normalizer,
                    "action_mask": action_mask,
                    "dataset": evaluation_dataset,
                    "config": config,
                }
            )
        )
        eval_metrics["closed_loop"] = _metrics_to_float(closed_loop)
        # In-process injection exists only for small unit/regression tests.  A
        # production gate is granted later by latent_closed_loop_eval after it
        # validates the checkpoint-bound v2 evidence report.
        eval_metrics["closed_loop_evidence_kind"] = "test_only_injected"
        for key, value in closed_loop.items():
            if np.ndim(np.asarray(value)) == 0:
                eval_metrics[f"closed_loop_{key}"] = float(value)
    eval_metrics["promotion"] = _evaluate_promotion_gates(eval_metrics, config)
    checkpoint_dir = Path(config.output_dir) / "latent_checkpoint"
    state_schema = _state_schema_from_dataset(dataset)
    action_schema = _action_schema_from_dataset(dataset)
    body_obs_schema = dataset.metadata.get("body_obs_schema")
    if config.strict_motion_identity and body_obs_schema is None:
        raise ValueError("production latent training requires a self-contained body_obs_schema")
    checkpoint_config = asdict(config)
    checkpoint_config.pop("closed_loop_evaluator", None)
    training_provenance = {
        "schema_version": "latent_training_provenance_v1",
        "dataset_manifest": dataset_manifest,
        "dataset_manifest_fingerprint": (
            None if dataset_manifest is None else dataset_manifest["manifest_fingerprint"]
        ),
        "teacher_checkpoint": teacher_fingerprint,
        "teacher_promotion": teacher_promotion,
        "direct_bc_metrics": (
            None
            if config.direct_bc_metrics_path is None
            else {
                "path": str(Path(config.direct_bc_metrics_path).resolve()),
                "sha256": file_sha256(config.direct_bc_metrics_path),
            }
        ),
    }
    save_latent_checkpoint(
        checkpoint_dir,
        encoder_variables=variables["encoder"],
        prior_variables=variables["prior"],
        decoder_variables=variables["decoder"],
        optimizer_state=opt_state,
        action_mask=action_mask,
        config=checkpoint_config | {
            "student_obs_dim": int(dataset.student_obs_dim),
            "reference_features_dim": int(dataset.reference_features_dim),
            "action_dim": action_dim,
        },
        train_metrics=metrics_history,
        eval_metrics=eval_metrics,
        obs_norm=normalizer.to_manifest(source_split="train"),
        action_norm={"min": float(config.action_min), "max": float(config.action_max)},
        action_schema=action_schema,
        state_schema=state_schema,
        body_obs_schema=body_obs_schema,
        split_manifest=split_manifest,
        training_provenance=training_provenance,
    )
    final_metrics = final_metrics or {"total_loss": float("nan"), "action_mse": float("nan")}
    return LatentTrainResult(
        checkpoint_dir=str(checkpoint_dir),
        final_total_loss=float(final_metrics["total_loss"]),
        final_action_mse=float(final_metrics["action_mse"]),
    )


def _evaluate_once(
    *,
    variables,
    posterior,
    prior,
    decoder,
    dataset,
    config: LatentTrainConfig,
    normalizer: ObservationNormalizer,
) -> dict[str, Any]:
    num_samples = min(int(dataset.num_samples), int(config.max_eval_samples))
    if num_samples <= 0:
        raise ValueError("latent validation dataset is empty")
    flat_state = normalizer.normalize_jax(dataset.arrays["student_obs"][:num_samples])
    flat_reference = jnp.asarray(dataset.arrays["reference_features"][:num_samples], dtype=jnp.float32)
    flat_teacher_action = jnp.asarray(dataset.arrays["teacher_action"][:num_samples], dtype=jnp.float32)
    posterior_mu, posterior_raw_sigma = posterior.apply(variables["encoder"], flat_state, flat_reference)
    prior_mu, prior_raw_sigma = prior.apply(variables["prior"], flat_state)
    posterior_action = decoder.apply(variables["decoder"], flat_state, posterior_mu)
    prior_mean_action = decoder.apply(variables["decoder"], flat_state, prior_mu)
    losses = latent_distillation_loss(
        predicted_action=posterior_action,
        teacher_action=flat_teacher_action,
        posterior_mu=posterior_mu,
        posterior_raw_sigma=posterior_raw_sigma,
        prior_mu=prior_mu,
        prior_raw_sigma=prior_raw_sigma,
        kl_weight=float(config.kl_weight),
        free_bits=float(config.free_bits),
        bound_weight=float(config.bound_weight),
        action_min=float(config.action_min),
        action_max=float(config.action_max),
        sigma_min=float(config.sigma_min),
        sigma_max=float(config.sigma_max),
    )
    posterior_sigma = positive_sigma(
        posterior_raw_sigma,
        sigma_min=float(config.sigma_min),
        sigma_max=float(config.sigma_max),
    )
    prior_sigma = positive_sigma(
        prior_raw_sigma,
        sigma_min=float(config.sigma_min),
        sigma_max=float(config.sigma_max),
    )
    posterior_mse = jnp.mean(jnp.square(posterior_action - flat_teacher_action))
    prior_mse = jnp.mean(jnp.square(prior_mean_action - flat_teacher_action))
    active_count = jnp.sum(jnp.std(posterior_mu, axis=0) > 1e-3)
    sigma_tolerance = 1e-6
    metrics = _metrics_to_float(
        losses
        | {
            "posterior_action_mse": posterior_mse,
            "prior_mean_action_mse": prior_mse,
            "prior_posterior_mse_ratio": prior_mse / jnp.maximum(posterior_mse, 1e-12),
            "active_latent_dimensions": active_count,
            "active_latent_fraction": active_count / float(config.latent_dim),
            "posterior_sigma_mean": jnp.mean(posterior_sigma),
            "prior_sigma_mean": jnp.mean(prior_sigma),
            "prior_sigma_min_clamp_fraction": jnp.mean(
                prior_sigma <= float(config.sigma_min) + sigma_tolerance
            ),
            "prior_sigma_max_clamp_fraction": jnp.mean(
                prior_sigma >= float(config.sigma_max) - sigma_tolerance
            ),
            "decoder_saturation_fraction": jnp.mean(jnp.abs(posterior_action) > 0.98),
            "prior_decoder_saturation_fraction": jnp.mean(jnp.abs(prior_mean_action) > 0.98),
            "action_min": float(config.action_min),
            "action_max": float(config.action_max),
            "num_eval_samples": int(flat_state.shape[0]),
        }
    )
    # Preserve the historical key while making its semantics unambiguous.
    metrics["action_mse"] = metrics["posterior_action_mse"]
    return metrics


def _batch_to_jax(batch: dict[str, np.ndarray]) -> dict[str, jax.Array]:
    return {key: jnp.asarray(value) for key, value in batch.items()}


def _metrics_to_float(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: float(value) if np.ndim(np.asarray(value)) == 0 else np.asarray(value).tolist() for key, value in metrics.items()}


def _build_action_mask(
    config: dict[str, Any] | None,
    action_dim: int,
    target_actuator_names: list[str] | tuple[str, ...] | None = None,
) -> ActionMask:
    if config is not None and config.get("preset") == "myofullbody_finger_partition":
        from musclemimic_models import load as load_model

        from musclemimic.runner.export_metadata import model_actuator_names
        from musclemimic.utils.finger_isolation import FingerActuatorPartition

        body_names = list(target_actuator_names or [])
        if len(body_names) != int(action_dim):
            raise ValueError("myofullbody_finger_partition requires exact decoder target actuator names")
        model, _data = load_model("myofullbody")
        partition = FingerActuatorPartition.from_actuator_names(model_actuator_names(model))
        if body_names != list(partition.body_actuator_names):
            raise ValueError(
                "decoder body actuator schema differs from canonical full MyoFullBody partition"
            )
        return ActionMask.from_partitions(
            all_actuator_names=list(partition.all_actuator_names),
            body_actuator_names=body_names,
            correction_actuator_names=list(partition.right_grip_actuator_names),
            neutral_actuator_names=list(partition.left_neutral_actuator_names),
            neutral_values=np.zeros(partition.left_neutral_size, dtype=float),
        )
    if config is None:
        all_names = list(target_actuator_names or [f"action_{index}" for index in range(int(action_dim))])
        correction_names: list[str] = []
    else:
        all_names = list(config.get("all_actuator_names") or [f"action_{index}" for index in range(int(action_dim))])
        correction_names = list(config.get("correction_actuator_names") or [])
        neutral_names = list(config.get("neutral_actuator_names") or [])
        neutral_values = config.get("neutral_values")
    if config is None:
        neutral_names = []
        neutral_values = None
    return ActionMask.from_correction_actuators(
        all_actuator_names=all_names,
        correction_actuator_names=correction_names,
        neutral_actuator_names=neutral_names,
        neutral_values=neutral_values,
    )


def _target_body_names(config: dict[str, Any] | None) -> list[str] | None:
    if config is None:
        return None
    if config.get("preset") == "myofullbody_finger_partition":
        return None
    all_names = list(config.get("all_actuator_names") or [])
    if not all_names:
        return None
    correction = {str(name) for name in (config.get("correction_actuator_names") or [])}
    neutral = {str(name) for name in (config.get("neutral_actuator_names") or [])}
    return [str(name) for name in all_names if str(name) not in correction and str(name) not in neutral]


def _state_schema_from_dataset(dataset: SequenceDistillDataset) -> dict[str, Any]:
    supplied = dataset.metadata.get("student_state_schema")
    if supplied is not None:
        schema = dict(supplied)
        if int(schema.get("state_dim", -1)) != int(dataset.student_obs_dim):
            raise ValueError(
                "dataset student_state_schema dimension does not match student_obs: "
                f"schema={schema.get('state_dim')} obs={dataset.student_obs_dim}"
            )
        payload = {key: value for key, value in schema.items() if key not in {"schema_hash", "provenance"}}
        actual_hash = ordered_schema_hash(kind="student_state", payload=payload)
        if schema.get("schema_hash") not in {None, actual_hash}:
            raise ValueError("dataset student_state_schema hash mismatch")
        schema["schema_hash"] = actual_hash
        return schema
    payload = {
        "schema_version": "latent_student_state_v1",
        "state_dim": int(dataset.student_obs_dim),
        "feature_names": [f"student_obs[{index}]" for index in range(int(dataset.student_obs_dim))],
    }
    return payload | {
        "schema_hash": ordered_schema_hash(kind="student_state", payload=payload),
        "provenance": {},
    }


def _action_schema_from_dataset(dataset: SequenceDistillDataset) -> dict[str, Any]:
    schema = dataset.action_selection.to_manifest()
    if dataset.actuator_ctrlrange is None:
        return schema
    ctrlrange = np.asarray(dataset.actuator_ctrlrange, dtype=np.float64)
    schema["target_ctrlrange"] = ctrlrange.tolist()
    schema["ctrlrange_schema_hash"] = ordered_schema_hash(
        kind="actuator_ctrlrange",
        payload={
            "actuator_names": list(dataset.actuator_names),
            "ctrlrange": ctrlrange.tolist(),
        },
    )
    return schema


def _evaluate_promotion_gates(metrics: dict[str, Any], config: LatentTrainConfig) -> dict[str, Any]:
    thresholds = {
        "max_prior_posterior_mse_ratio": 1.25,
        "min_active_latent_fraction": 0.25,
        "max_prior_sigma_min_clamp_fraction": 0.10,
        "max_prior_sigma_max_clamp_fraction": 0.10,
        "max_decoder_saturation_fraction": 0.01,
    }
    thresholds.update(config.promotion_gates or {})
    checks = {
        "prior_vs_posterior": float(metrics["prior_posterior_mse_ratio"])
        <= float(thresholds["max_prior_posterior_mse_ratio"]),
        "active_latent_dimensions": float(metrics["active_latent_fraction"])
        >= float(thresholds["min_active_latent_fraction"]),
        "prior_sigma_min_clamp": float(metrics["prior_sigma_min_clamp_fraction"])
        < float(thresholds["max_prior_sigma_min_clamp_fraction"]),
        "prior_sigma_max_clamp": float(metrics["prior_sigma_max_clamp_fraction"])
        < float(thresholds["max_prior_sigma_max_clamp_fraction"]),
        "decoder_saturation": float(metrics["decoder_saturation_fraction"])
        < float(thresholds["max_decoder_saturation_fraction"]),
    }
    if config.direct_bc_action_mse is not None:
        checks["posterior_vs_direct_bc"] = float(metrics["posterior_action_mse"]) <= (
            1.10 * float(config.direct_bc_action_mse)
        )
    elif config.require_direct_bc_baseline:
        checks["posterior_vs_direct_bc"] = False
    for threshold_name, threshold in thresholds.items():
        if threshold_name.startswith("closed_loop_min_"):
            source_key = f"closed_loop_{threshold_name[len('closed_loop_min_') :]}"
            if source_key in metrics:
                checks[source_key] = float(metrics[source_key]) >= float(threshold)
            elif config.require_closed_loop_metrics:
                checks[source_key] = False
        elif threshold_name.startswith("closed_loop_max_"):
            source_key = f"closed_loop_{threshold_name[len('closed_loop_max_') :]}"
            if source_key in metrics:
                checks[source_key] = float(metrics[source_key]) <= float(threshold)
            elif config.require_closed_loop_metrics:
                checks[source_key] = False
    if config.require_closed_loop_metrics:
        checks["production_closed_loop_evidence"] = (
            metrics.get("closed_loop_evidence_kind") == "verified_production_v2"
        )
    if config.require_dataset_provenance:
        checks["stage2_teacher_promotion_evidence"] = (
            metrics.get("teacher_promotion_evidence_kind")
            == "verified_stage2_promotion_v1"
        )
    if metrics.get("closed_loop_evidence_kind") == "test_only_injected":
        checks["test_only_evidence_not_promotable"] = False
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "thresholds": thresholds,
    }

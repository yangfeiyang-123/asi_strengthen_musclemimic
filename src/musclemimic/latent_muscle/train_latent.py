"""Minimal latent distillation trainer for posterior/prior/decoder modules."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from musclemimic.distill.action_schema import ordered_schema_hash
from musclemimic.distill.dataset import (
    PhysicalDistillDataset,
    SequenceDistillDataset,
    motion_split_datasets,
)
from musclemimic.distill.physical import (
    MUSCLE_EXCITATION_FORMULA,
    MUSCLE_EXCITATION_ROUNDOFF_POLICY,
    PHYSICAL_CAPTURE_SCHEMA_VERSION,
    PHYSICAL_SIGNAL_SCHEMA_VERSION,
    UNIT_EXCITATION_TRANSFORM,
    MuscleChannelContract,
    physical_ctrl_to_effective_muscle_excitation,
    validate_activation_valid_mask,
    validate_muscle_channel_contract,
    validate_physical_signal_semantics,
    validate_unit_muscle_activation,
    validate_unit_muscle_ctrlrange,
    validate_unit_muscle_excitation,
)
from musclemimic.distill.provenance import (
    DEFAULT_TEACHER_PROMOTION_STAGE,
    STAGE1_BODY_ONLY_TEACHER_ROLE,
    VERIFIED_STAGE1_PEASD_PROMOTION_EVIDENCE,
    VERIFIED_STAGE1_PROMOTION_EVIDENCE,
    VERIFIED_STAGE2_PROMOTION_EVIDENCE,
    canonical_json_sha256,
    checkpoint_content_fingerprint,
    file_sha256,
    test_only_unpromoted_teacher_binding,
    validate_dataset_manifest,
    validate_teacher_promotion_binding,
    validate_teacher_promotion_manifest,
)
from musclemimic.distill.provenance import (
    teacher_promotion_evidence_kind as teacher_promotion_evidence_kind_from_binding,
)
from musclemimic.latent_muscle.action_mask import ActionMask
from musclemimic.latent_muscle.checkpoint import (
    LATENT_MUSCLE_ACTION_SCHEMA_VERSION,
    save_latent_checkpoint,
)
from musclemimic.latent_muscle.decoder_factory import (
    SYNERGY_RESIDUAL_DECODER,
    DecoderBundle,
    apply_decoder,
    build_decoder_bundle,
    init_decoder,
)
from musclemimic.latent_muscle.losses import latent_distillation_loss, positive_sigma
from musclemimic.latent_muscle.networks import (
    ConditionalPrior,
    PosteriorEncoder,
    SynergyHead,
    emg_context_dropout,
    reparameterize_gaussian,
)
from musclemimic.latent_muscle.normalization import ObservationNormalizer
from musclemimic.latent_muscle.phase_contract import (
    FOREHAND_PHASE_NAMES,
    normalize_phase_contract,
    phase_items,
)


@dataclass(frozen=True)
class LatentTrainConfig:
    dataset_dir: str
    output_dir: str
    # Production sweeps use the canonical five motions from a separate
    # immutable collection.  This split must not depend on the optimizer seed.
    val_dataset_dir: str | None = None
    expected_val_motion_count: int | None = None
    # Optional §11.4 latent DAgger input.  Collection is intentionally outside
    # this trainer; only an immutable student-rollout/teacher-relabel dataset
    # is accepted and appended to the training split.
    closed_loop_correction_dataset_dir: str | None = None
    closed_loop_correction_manifest: str | None = None
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
    closed_loop_tracking_metrics: tuple[str, ...] = (
        "err_rpos",
        "err_racket_pos",
        "err_racket_rot",
    )
    require_dataset_provenance: bool = False
    teacher_promotion_manifest: str | None = None
    # Stage-2 remains the default ABI.  Body-only Stage-1 distillation must opt
    # in to both stage1 and the body_only role; the two artifacts are never
    # treated as interchangeable.
    teacher_promotion_stage: str = DEFAULT_TEACHER_PROMOTION_STAGE
    teacher_promotion_role: str | None = None
    test_only_allow_unpromoted_teacher: bool = False
    promotion_gates: dict[str, float] | None = None
    # Decoder extensions are opt-in; missing fields retain the historical MLP.
    decoder_type: str = "direct"
    # Scientific primary: a self-contained Stage-1 frozen decoder artifact.
    # The latent network predicts only raw c/rho and cannot change W/tonic/R.
    frozen_body_decoder_path: str | None = None
    frozen_body_decoder_expected_fingerprint: str | None = None
    body_synergy_contract_expected_fingerprint: str | None = None
    body_synergy_portable_core_expected_fingerprint: str | None = None
    # Historical W + softplus/direct-residual decoder, retained only as a
    # conspicuous ablation/checkpoint migration route.
    legacy_synergy_decoder_ablation: bool = False
    synergy_basis_path: str | None = None
    synergy_basis_expected_fingerprint: str | None = None
    test_only_allow_legacy_synergy_basis: bool = False
    # False is the scientific primary: an explicit True is a full-dimensional
    # state-baseline ablation and must not be mistaken for fixed-W control.
    synergy_include_baseline: bool = False
    synergy_baseline_init: float = 0.01
    synergy_residual_actuator_names: tuple[str, ...] = ()
    synergy_residual_alpha: float = 0.0
    synergy_residual_l1_weight: float = 0.0
    synergy_residual_l2_weight: float = 0.0
    synergy_residual_smooth_weight: float = 0.0
    synergy_baseline_l1_weight: float = 0.0
    synergy_baseline_l2_weight: float = 0.0
    phase_field: str | None = "phase_id"
    phase_balance_weights: dict[str, float] | None = None
    phase_contract: dict[str, Any] | None = None
    physical_excitation_field: str = "muscle_excitation"
    physical_excitation_weight: float = 0.0
    physical_excitation_min: float = 0.0
    physical_excitation_max: float = 1.0
    emg_privileged_enabled: bool = False
    emg_synergy_dim: int = 0
    emg_context_dropout: float = 0.30
    emg_synergy_loss_weight: float = 0.0
    emg_tube_kappa: float = 1.0
    emg_include_scale: bool = True
    emg_include_confidence: bool = True
    emg_require_reference_hash: bool = True
    emg_reference_manifest: str | None = None
    emg_shuffle_context_ablation: bool = False
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


def _teacher_promotion_expectation(
    config: LatentTrainConfig,
) -> tuple[str, str | None]:
    stage = str(config.teacher_promotion_stage).strip().lower()
    role = (
        None
        if config.teacher_promotion_role in (None, "")
        else str(config.teacher_promotion_role).strip().lower()
    )
    if stage not in {"stage1", "stage2"}:
        raise ValueError("teacher_promotion_stage must be 'stage1' or 'stage2'")
    if stage == "stage1" and role != STAGE1_BODY_ONLY_TEACHER_ROLE:
        raise ValueError(
            "Stage-1 latent distillation requires "
            "teacher_promotion_role='body_only'"
        )
    return stage, role


def expected_teacher_promotion_evidence_kind(
    config: LatentTrainConfig,
) -> str:
    """Return the only formal teacher evidence token allowed by a config."""

    stage, _role = _teacher_promotion_expectation(config)
    if stage == "stage1":
        if config.teacher_promotion_manifest is not None:
            try:
                payload = json.loads(
                    Path(config.teacher_promotion_manifest).read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("Stage-1 teacher promotion manifest is unreadable") from exc
            from musclemimic.badminton.stage1_peasd_gate import (
                PEASD_TEACHER_PROMOTION_SCHEMA_VERSION,
            )

            if isinstance(payload, dict) and payload.get("schema_version") == (
                PEASD_TEACHER_PROMOTION_SCHEMA_VERSION
            ):
                return VERIFIED_STAGE1_PEASD_PROMOTION_EVIDENCE
        return VERIFIED_STAGE1_PROMOTION_EVIDENCE
    return VERIFIED_STAGE2_PROMOTION_EVIDENCE


def kl_warmup_weight(*, step: int, target: float, warmup_steps: int) -> float:
    if int(warmup_steps) <= 0:
        return float(target)
    return float(target) * min(1.0, max(0.0, float(step) / float(warmup_steps)))


def teacher_delta_smooth_mse(predicted_sequence, teacher_sequence):
    """MSE between student and teacher temporal action deltas."""
    predicted = jnp.asarray(predicted_sequence)
    teacher = jnp.asarray(teacher_sequence)
    if predicted.shape != teacher.shape or predicted.ndim < 2:
        raise ValueError("predicted/teacher sequences must have the same [..., time, action] shape")
    if predicted.shape[-2] < 2:
        return jnp.asarray(0.0, dtype=predicted.dtype)
    predicted_delta = predicted[..., 1:, :] - predicted[..., :-1, :]
    teacher_delta = teacher[..., 1:, :] - teacher[..., :-1, :]
    return jnp.mean(jnp.square(predicted_delta - teacher_delta))


def train_latent(config: LatentTrainConfig) -> LatentTrainResult:
    dataset_manifest = None
    validation_dataset_manifest = None
    teacher_fingerprint = None
    teacher_promotion = None
    teacher_promotion_evidence_kind = None
    correction_dataset_manifest = None
    dataset_path = Path(config.dataset_dir)
    if config.require_dataset_provenance or (dataset_path / "dataset_manifest.json").is_file():
        expected_promotion_stage, expected_promotion_role = (
            _teacher_promotion_expectation(config)
        )
        if config.require_dataset_provenance and not config.teacher_ckpt:
            raise ValueError("production latent training requires teacher_ckpt provenance")
        teacher_fingerprint = (
            None if config.teacher_ckpt is None else checkpoint_content_fingerprint(config.teacher_ckpt)
        )
        if config.teacher_promotion_manifest is not None:
            if config.test_only_allow_unpromoted_teacher:
                raise ValueError("teacher_promotion_manifest and test-only bypass are mutually exclusive")
            if teacher_fingerprint is None:
                raise ValueError(
                    "teacher_promotion_manifest requires teacher_ckpt provenance"
                )
            teacher_promotion = validate_teacher_promotion_manifest(
                config.teacher_promotion_manifest,
                teacher_checkpoint=teacher_fingerprint,
                expected_stage=expected_promotion_stage,
                teacher_role=expected_promotion_role,
            )
            teacher_promotion_evidence_kind = (
                teacher_promotion_evidence_kind_from_binding(teacher_promotion)
            )
        elif config.test_only_allow_unpromoted_teacher:
            if teacher_fingerprint is None:
                raise ValueError(
                    "test-only unpromoted teacher bypass requires teacher_ckpt provenance"
                )
            teacher_promotion = test_only_unpromoted_teacher_binding(teacher_fingerprint)
            teacher_promotion_evidence_kind = "test_only_unpromoted_teacher"
        elif config.require_dataset_provenance:
            raise ValueError("production latent training requires teacher_promotion_manifest")
        dataset_manifest = validate_dataset_manifest(
            dataset_path,
            expected_teacher=teacher_fingerprint,
            expected_teacher_promotion=teacher_promotion,
            require_promoted_teacher=bool(
                config.require_dataset_provenance and not config.test_only_allow_unpromoted_teacher
            ),
        )
        if teacher_promotion is None:
            embedded_teacher = dataset_manifest["teacher_checkpoint"]
            teacher_promotion = validate_teacher_promotion_binding(
                dataset_manifest["teacher_promotion"],
                teacher_checkpoint=embedded_teacher,
                require_promoted=False,
                expected_stage=expected_promotion_stage,
                expected_teacher_role=expected_promotion_role,
            )
            teacher_promotion_evidence_kind = (
                teacher_promotion_evidence_kind_from_binding(teacher_promotion)
            )
        if config.val_dataset_dir is not None:
            validation_dataset_manifest = validate_dataset_manifest(
                config.val_dataset_dir,
                expected_teacher=teacher_fingerprint,
                expected_teacher_promotion=teacher_promotion,
                require_promoted_teacher=bool(
                    config.require_dataset_provenance and not config.test_only_allow_unpromoted_teacher
                ),
            )
    correction_values = (
        config.closed_loop_correction_dataset_dir,
        config.closed_loop_correction_manifest,
    )
    if (correction_values[0] is None) != (correction_values[1] is None):
        raise ValueError("closed-loop correction training requires both dataset_dir and manifest")
    if correction_values[0] is not None:
        correction_dir = Path(str(correction_values[0])).resolve()
        correction_manifest_path = Path(str(correction_values[1])).resolve()
        if correction_manifest_path != correction_dir / "dataset_manifest.json":
            raise ValueError("closed-loop correction manifest must be the dataset's immutable dataset_manifest.json")
        if dataset_manifest is None or teacher_fingerprint is None or teacher_promotion is None:
            raise ValueError(
                "closed-loop correction training requires fully validated production dataset/teacher provenance"
            )
        correction_dataset_manifest = validate_dataset_manifest(
            correction_dir,
            expected_teacher=teacher_fingerprint,
            expected_teacher_promotion=teacher_promotion,
            require_promoted_teacher=True,
        )
        _validate_closed_loop_correction_manifest(
            correction_dataset_manifest,
            expected_teacher_sha256=str(teacher_fingerprint["sha256"]),
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
    dataset, val_dataset, split_manifest = _load_latent_train_validation_datasets(
        config,
        motion_field=split_motion_field,
        target_body_names=target_body_names,
        dataset_manifest=dataset_manifest,
        validation_dataset_manifest=validation_dataset_manifest,
    )
    if correction_dataset_manifest is not None:
        dataset, split_manifest = _append_closed_loop_correction_dataset(
            dataset,
            validation=val_dataset,
            split_manifest=split_manifest,
            correction_dataset_dir=str(config.closed_loop_correction_dataset_dir),
            correction_manifest=correction_dataset_manifest,
            motion_field=split_motion_field,
            target_body_names=target_body_names,
            seed=int(config.seed),
            require_stable_ids=bool(config.strict_motion_identity),
        )
    train_channel_contract = _validate_latent_physical_dataset_contract(
        dataset,
        split="train",
    )
    if val_dataset is not None:
        _validate_latent_physical_dataset_contract(
            val_dataset,
            split="val",
        )
    action_dim = int(dataset.action_dim)
    action_mask = _build_action_mask(config.action_mask, action_dim, dataset.actuator_names)
    if action_mask.body_size != action_dim:
        raise ValueError(f"decoder action_dim={action_dim} must equal action mask body size={action_mask.body_size}")
    if list(dataset.actuator_names) != list(action_mask.body_actuator_names):
        raise ValueError("latent decoder target actuator names/order differ from ActionMask body partition")
    decoder_bundle = build_decoder_bundle(
        asdict(config),
        action_dim=action_dim,
        hidden_layer_dims=tuple(config.hidden_layer_dims),
        actuator_names=dataset.actuator_names,
    )
    _validate_decoder_training_config(config, decoder_bundle)
    _validate_phase_contract(config)
    _validate_emg_privileged_config(config)
    _validate_optional_training_fields(dataset, config, decoder_bundle, split="train")
    _validate_portable_decoder_dataset_contract(dataset, decoder_bundle, split="train")
    if val_dataset is not None:
        _validate_optional_training_fields(val_dataset, config, decoder_bundle, split="val")
        _validate_portable_decoder_dataset_contract(val_dataset, decoder_bundle, split="val")
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
    decoder = decoder_bundle.module
    emg_privileged = bool(config.emg_privileged_enabled)
    synergy_head = SynergyHead(synergy_dim=int(config.emg_synergy_dim)) if emg_privileged else None

    rng = jax.random.PRNGKey(int(config.seed))
    init_batch = next(dataset.iter_sequence_batches(batch_size=1, horizon=config.horizon, shuffle=False))
    init_state = normalizer.normalize_jax(init_batch["student_obs"].reshape(-1, dataset.student_obs_dim))
    init_ref = jnp.asarray(
        init_batch["reference_features"].reshape(-1, dataset.reference_features_dim), dtype=jnp.float32
    )
    init_latent = jnp.zeros((init_state.shape[0], int(config.latent_dim)), dtype=jnp.float32)
    init_emg_context = (
        jnp.zeros(
            (
                init_state.shape[0],
                emg_context_width(
                    synergy_dim=int(config.emg_synergy_dim),
                    include_scale=bool(config.emg_include_scale),
                    include_confidence=bool(config.emg_include_confidence),
                ),
            ),
            dtype=jnp.float32,
        )
        if emg_privileged
        else None
    )
    rng, post_rng, prior_rng, dec_rng, head_rng = jax.random.split(rng, 5)
    variables = {
        "encoder": posterior.init(post_rng, init_state, init_ref, init_emg_context),
        "prior": prior.init(prior_rng, init_state),
        "decoder": init_decoder(decoder_bundle, dec_rng, init_state, init_latent),
    }
    if synergy_head is not None:
        variables["synergy_head"] = synergy_head.init(head_rng, init_latent)
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
        sample_weight = _phase_sample_weights_from_batch(
            batch,
            phase_field=str(config.phase_field),
            phase_balance_weights=config.phase_balance_weights,
            phase_contract=config.phase_contract,
        )
        teacher_physical = _physical_target_from_batch(
            batch,
            field=str(config.physical_excitation_field),
            enabled=float(config.physical_excitation_weight) > 0.0,
        )
        emg_bundle = _emg_context_from_batch(
            batch,
            enabled=emg_privileged,
            synergy_dim=int(config.emg_synergy_dim),
            include_scale=bool(config.emg_include_scale),
            include_confidence=bool(config.emg_include_confidence),
        )

        sample_rng, context_rng, shuffle_rng = jax.random.split(step_rng, 3)
        flat_emg_context = None
        emg_keep_fraction = None
        if emg_bundle is not None:
            flat_emg_context, emg_mean, emg_scale, emg_valid = emg_bundle
            if bool(config.emg_shuffle_context_ablation):
                # Negative control: keep the marginal distribution of the context
                # but destroy its correspondence to the state it was queried for.
                permutation = jax.random.permutation(shuffle_rng, flat_emg_context.shape[0])
                flat_emg_context = flat_emg_context[permutation]
            flat_emg_context, keep = emg_context_dropout(
                context_rng,
                flat_emg_context,
                dropout_rate=float(config.emg_context_dropout),
            )
            emg_keep_fraction = jnp.mean(keep)

        posterior_mu, posterior_raw_sigma = posterior.apply(
            params["encoder"], flat_state, flat_reference, flat_emg_context
        )
        prior_mu, prior_raw_sigma = prior.apply(params["prior"], flat_state)
        z = reparameterize_gaussian(
            sample_rng,
            posterior_mu,
            posterior_raw_sigma,
            sigma_min=float(config.sigma_min),
            sigma_max=float(config.sigma_max),
        )
        decoder_output = apply_decoder(
            decoder_bundle,
            params["decoder"],
            flat_state,
            z,
            return_aux=True,
        )
        predicted_action = decoder_output.action
        residual_excitation = (
            decoder_output.residual_excitation if decoder_bundle.decoder_type == SYNERGY_RESIDUAL_DECODER else None
        )
        baseline_excitation = decoder_output.baseline_excitation if decoder_bundle.is_synergy else None
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
            sample_weight=sample_weight,
            predicted_physical_excitation=(
                decoder_output.physical_excitation if teacher_physical is not None else None
            ),
            teacher_physical_excitation=teacher_physical,
            physical_excitation_weight=float(config.physical_excitation_weight),
            residual_excitation=residual_excitation,
            residual_l1_weight=float(config.synergy_residual_l1_weight),
            residual_l2_weight=float(config.synergy_residual_l2_weight),
            baseline_excitation=baseline_excitation,
            baseline_l1_weight=float(config.synergy_baseline_l1_weight),
            baseline_l2_weight=float(config.synergy_baseline_l2_weight),
        )
        pred_seq = predicted_action.reshape((batch_size, horizon, teacher_action.shape[-1]))
        if horizon > 1 and float(config.smooth_weight):
            teacher_seq = teacher_action.reshape((batch_size, horizon, teacher_action.shape[-1]))
            # Penalize temporal tracking error, not motion itself.  The old
            # ``||a_t-a_{t-1}||`` objective suppressed the teacher's genuine
            # high acceleration around contact and biased the decoder toward
            # a flat action.  This term preserves the teacher delta while
            # damping only student-specific chatter.
            smooth_mse = teacher_delta_smooth_mse(pred_seq, teacher_seq)
        else:
            smooth_mse = jnp.asarray(0.0, dtype=losses["total_loss"].dtype)
        if (
            horizon > 1
            and decoder_bundle.decoder_type == SYNERGY_RESIDUAL_DECODER
            and float(config.synergy_residual_smooth_weight) > 0.0
        ):
            residual_sequence = decoder_output.residual_excitation.reshape((batch_size, horizon, action_dim))
            residual_smooth_mse = jnp.mean(jnp.square(residual_sequence[:, 1:, :] - residual_sequence[:, :-1, :]))
        else:
            residual_smooth_mse = jnp.asarray(0.0, dtype=losses["total_loss"].dtype)
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
        emg_synergy_loss = jnp.asarray(0.0, dtype=losses["total_loss"].dtype)
        emg_diagnostics: dict[str, Any] = {}
        if emg_bundle is not None and synergy_head is not None:
            predicted_synergy = synergy_head.apply(params["synergy_head"], z)
            emg_synergy_loss = emg_synergy_tube_loss(
                predicted_synergy,
                emg_mean,
                emg_scale,
                emg_valid,
                kappa=float(config.emg_tube_kappa),
            )
            emg_diagnostics = {
                "emg_synergy_loss": emg_synergy_loss,
                "emg_synergy_head_mean": jnp.mean(predicted_synergy),
                "emg_synergy_head_std": jnp.std(predicted_synergy),
                "emg_synergy_reference_mean": jnp.mean(emg_mean),
                "emg_synergy_correlation": _masked_correlation(predicted_synergy, emg_mean, emg_valid),
                "emg_context_keep_fraction": emg_keep_fraction,
                "emg_valid_fraction": jnp.mean(emg_valid),
            }

        total = (
            losses["total_loss"]
            + float(config.smooth_weight) * smooth_mse
            + float(config.synergy_residual_smooth_weight) * residual_smooth_mse
            + float(config.emg_synergy_loss_weight) * emg_synergy_loss
        )
        residual_energy_ratio = _energy_ratio(
            decoder_output.residual_excitation,
            decoder_output.physical_excitation,
        )
        diagnostics = losses | {
            "total_loss": total,
            "smooth_mse": smooth_mse,
            "residual_smooth_mse": residual_smooth_mse,
            "residual_energy_ratio": residual_energy_ratio,
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
        if emg_diagnostics:
            diagnostics |= emg_diagnostics
        if decoder_bundle.is_synergy:
            diagnostics |= {
                "synergy_coefficient_mean": jnp.mean(decoder_output.synergy_coefficients),
                "synergy_coefficient_std": jnp.std(decoder_output.synergy_coefficients),
                "baseline_energy_ratio": _energy_ratio(
                    decoder_output.baseline_excitation,
                    decoder_output.physical_excitation,
                ),
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
        decoder_bundle=decoder_bundle,
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
                    "decoder_bundle": decoder_bundle,
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
    action_schema = _action_schema_from_dataset(
        dataset,
        muscle_channel_contract=train_channel_contract,
    )
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
        "validation_dataset_manifest": validation_dataset_manifest,
        "validation_dataset_manifest_fingerprint": (
            None if validation_dataset_manifest is None else validation_dataset_manifest["manifest_fingerprint"]
        ),
        "closed_loop_correction_dataset_manifest": correction_dataset_manifest,
        "closed_loop_correction_dataset_manifest_fingerprint": (
            None if correction_dataset_manifest is None else correction_dataset_manifest["manifest_fingerprint"]
        ),
        "teacher_checkpoint": teacher_fingerprint,
        "teacher_promotion": teacher_promotion,
        "synergy_basis_fingerprint": (
            None if decoder_bundle.synergy_basis is None else decoder_bundle.synergy_basis.fingerprint
        ),
        "frozen_body_decoder_fingerprint": (
            None
            if decoder_bundle.frozen_body_decoder is None
            else decoder_bundle.frozen_body_decoder.artifact_fingerprint
        ),
        "body_synergy_contract": (
            None
            if decoder_bundle.frozen_body_decoder is None
            else decoder_bundle.frozen_body_decoder.body_synergy_contract.to_manifest()
        ),
        "body_synergy_contract_fingerprint": (
            None
            if decoder_bundle.frozen_body_decoder is None
            else decoder_bundle.frozen_body_decoder.body_synergy_contract.contract_fingerprint
        ),
        "body_synergy_portable_core_fingerprint": (
            None
            if decoder_bundle.frozen_body_decoder is None
            else decoder_bundle.frozen_body_decoder.body_synergy_contract.portable_decoder_core_fingerprint
        ),
        "direct_bc_metrics": (
            None
            if config.direct_bc_metrics_path is None
            else {
                "path": str(Path(config.direct_bc_metrics_path).resolve()),
                "sha256": file_sha256(config.direct_bc_metrics_path),
            }
        ),
        # Runtime never loads EMG data, but the checkpoint must record which
        # human reference shaped this decoder, and under what training terms.
        "emg_privileged": _emg_privileged_provenance(config, dataset_metadata=dataset.metadata),
    }
    save_latent_checkpoint(
        checkpoint_dir,
        encoder_variables=variables["encoder"],
        prior_variables=variables["prior"],
        decoder_variables=variables["decoder"],
        optimizer_state=opt_state,
        action_mask=action_mask,
        config=checkpoint_config
        | {
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
        synergy_basis=(decoder_bundle.synergy_basis if decoder_bundle.frozen_body_decoder is None else None),
        frozen_body_decoder=decoder_bundle.frozen_body_decoder,
    )
    final_metrics = final_metrics or {"total_loss": float("nan"), "action_mse": float("nan")}
    return LatentTrainResult(
        checkpoint_dir=str(checkpoint_dir),
        final_total_loss=float(final_metrics["total_loss"]),
        final_action_mse=float(final_metrics["action_mse"]),
    )


def _load_latent_train_validation_datasets(
    config: LatentTrainConfig,
    *,
    motion_field: str,
    target_body_names: list[str] | None,
    dataset_manifest: dict[str, Any] | None,
    validation_dataset_manifest: dict[str, Any] | None,
) -> tuple[SequenceDistillDataset, SequenceDistillDataset | None, dict[str, Any]]:
    """Load a seed-independent explicit validation collection when supplied."""

    if config.val_dataset_dir is None:
        return motion_split_datasets(
            config.dataset_dir,
            dataset_cls=SequenceDistillDataset,
            seed=int(config.seed),
            val_fraction=float(config.val_fraction),
            motion_field=motion_field,
            target_actuator_names=target_body_names,
            require_stable_ids=bool(config.strict_motion_identity),
        )
    if float(config.val_fraction) != 0.0:
        raise ValueError(
            "explicit val_dataset_dir requires val_fraction=0 so the optimizer seed cannot change held-out motions"
        )
    train_path = Path(config.dataset_dir).resolve()
    validation_path = Path(config.val_dataset_dir).resolve()
    if train_path == validation_path:
        raise ValueError("val_dataset_dir must be distinct from dataset_dir")
    kwargs = {
        "seed": int(config.seed),
        "target_actuator_names": target_body_names,
        "require_stable_ids": bool(config.strict_motion_identity),
    }
    train = SequenceDistillDataset(train_path, split="train", **kwargs)
    validation = SequenceDistillDataset(validation_path, split="val", **kwargs)
    _require_same_dataset_abi(train, validation)
    train_ids = _dataset_motion_ids(train, motion_field)
    validation_ids = _dataset_motion_ids(validation, motion_field)
    overlap = sorted(train_ids & validation_ids)
    if overlap:
        raise ValueError(
            f"explicit latent train/validation motion leakage detected: motion_field={motion_field!r} overlap={overlap}"
        )
    expected_count = config.expected_val_motion_count
    if expected_count is not None and len(validation_ids) != int(expected_count):
        raise ValueError(
            "explicit validation motion count differs from the production contract: "
            f"expected={int(expected_count)} actual={len(validation_ids)}"
        )
    manifest = {
        "schema_version": "motion_split_v2",
        "mode": "explicit_dataset_directories",
        "motion_field": str(motion_field),
        # Deliberately no split seed: all optimization seeds share these IDs.
        "split_seed": None,
        "train_dataset_dir": str(train_path),
        "val_dataset_dir": str(validation_path),
        "train_dataset_manifest_fingerprint": (
            None if dataset_manifest is None else dataset_manifest.get("manifest_fingerprint")
        ),
        "val_dataset_manifest_fingerprint": (
            None if validation_dataset_manifest is None else validation_dataset_manifest.get("manifest_fingerprint")
        ),
        "train_motion_ids": sorted(train_ids),
        "val_motion_ids": sorted(validation_ids),
        "train_num_samples": int(train.num_samples),
        "val_num_samples": int(validation.num_samples),
    }
    manifest["split_fingerprint"] = canonical_json_sha256(manifest)
    return train, validation, manifest


def _validate_closed_loop_correction_manifest(
    manifest: dict[str, Any],
    *,
    expected_teacher_sha256: str,
) -> None:
    """Require genuine student-rollout/teacher-relabel collection contracts."""

    collections = manifest.get("collections")
    if not isinstance(collections, list) or not collections:
        raise ValueError("closed-loop correction dataset has no committed collections")
    source_checkpoints: set[str] = set()
    for collection in collections:
        contract = collection.get("contract") if isinstance(collection, dict) else None
        if not isinstance(contract, dict):
            raise ValueError("closed-loop correction collection lacks an immutable contract")
        student_sha = str(contract.get("student_checkpoint_sha256", ""))
        request = contract.get("request")
        student_checkpoint = contract.get("student_checkpoint")
        if (
            contract.get("schema_version") != "distill_collection_contract_v2"
            or contract.get("collector") != "dagger_student_rollout_teacher_relabel"
            or contract.get("split") != "train"
            or contract.get("dagger_iteration") is None
            or contract.get("teacher_checkpoint_sha256") != expected_teacher_sha256
            or len(student_sha) != 64
            or any(character not in "0123456789abcdef" for character in student_sha)
            or not isinstance(student_checkpoint, dict)
            or student_checkpoint.get("sha256") != student_sha
            or not isinstance(request, dict)
            or request.get("student_policy_kind") != "latent_checkpoint_prior_mean_lab"
            or request.get("teacher_relabel_target") != "normalized_body_action"
            or request.get("closed_loop_state_source") != "environment_student_visited_state"
            or int(collection.get("num_samples", 0)) <= 0
        ):
            raise ValueError(
                "closed-loop correction data must be non-empty train-split DAgger "
                "student rollouts relabeled by the bound Stage-2 teacher"
            )
        source_checkpoints.add(student_sha)
    if not source_checkpoints:
        raise ValueError("closed-loop correction data bind no source latent checkpoint")


def _append_closed_loop_correction_dataset(
    train: SequenceDistillDataset,
    *,
    validation: SequenceDistillDataset | None,
    split_manifest: dict[str, Any],
    correction_dataset_dir: str,
    correction_manifest: dict[str, Any],
    motion_field: str,
    target_body_names: list[str] | None,
    seed: int,
    require_stable_ids: bool,
) -> tuple[SequenceDistillDataset, dict[str, Any]]:
    """Append sealed correction rows while preserving validation isolation."""

    correction = SequenceDistillDataset(
        correction_dataset_dir,
        split="train",
        seed=int(seed),
        target_actuator_names=target_body_names,
        require_stable_ids=bool(require_stable_ids),
    )
    _validate_latent_physical_dataset_contract(
        correction,
        split="closed-loop correction",
    )
    _require_same_dataset_abi(train, correction)
    missing = sorted(set(train.arrays) - set(correction.arrays))
    if missing:
        raise ValueError(f"closed-loop correction shards omit fields present in the offline training ABI: {missing}")
    merged_arrays: dict[str, np.ndarray] = {}
    for name, base in train.arrays.items():
        extra = np.asarray(correction.arrays[name])
        base_array = np.asarray(base)
        if extra.shape[1:] != base_array.shape[1:]:
            raise ValueError(f"closed-loop correction field {name!r} has a different trailing shape")
        merged_arrays[name] = np.concatenate([base_array, extra], axis=0)
    if require_stable_ids:
        identity_fields = ("motion_uid", "rollout_uid", "rollout_step", "env_index")
        identities = list(
            zip(
                *(np.asarray(merged_arrays[name]).astype(np.int64).tolist() for name in identity_fields),
                strict=True,
            )
        )
        if len(set(identities)) != len(identities):
            raise ValueError("closed-loop correction rows collide with existing stable sample identities")
    correction_ids = _dataset_motion_ids(correction, motion_field)
    if validation is not None:
        validation_ids = _dataset_motion_ids(validation, motion_field)
        overlap = sorted(correction_ids & validation_ids)
        if overlap:
            raise ValueError(f"closed-loop correction data leak held-out validation motions: {overlap}")
    clone = object.__new__(type(train))
    clone.__dict__ = dict(train.__dict__)
    clone.arrays = merged_arrays
    clone.num_samples = int(next(iter(merged_arrays.values())).shape[0])
    clone.metadata = dict(train.metadata) | {
        "latent_closed_loop_correction_manifest_fingerprint": correction_manifest["manifest_fingerprint"]
    }
    clone.shard_paths = list(train.shard_paths) + list(correction.shard_paths)

    updated = {key: value for key, value in split_manifest.items() if key != "split_fingerprint"}
    train_ids = {int(value) for value in updated.get("train_motion_ids", [])}
    updated["train_motion_ids"] = sorted(train_ids | correction_ids)
    updated["train_num_samples"] = int(clone.num_samples)
    updated["closed_loop_correction"] = {
        "evidence_kind": "student_closed_loop_rollout_stage2_teacher_relabel",
        "dataset_dir": str(Path(correction_dataset_dir).resolve()),
        "dataset_manifest_fingerprint": correction_manifest["manifest_fingerprint"],
        "num_samples": int(correction.num_samples),
        "source_latent_checkpoint_fingerprints": sorted(
            {
                str(collection["contract"]["student_checkpoint_sha256"])
                for collection in correction_manifest["collections"]
            }
        ),
        "collection_performed_by_trainer": False,
    }
    updated["split_fingerprint"] = canonical_json_sha256(updated)
    return clone, updated


def _dataset_motion_ids(
    dataset: SequenceDistillDataset,
    motion_field: str,
) -> set[int]:
    if motion_field not in dataset.arrays:
        raise ValueError(f"explicit latent split is missing motion field {motion_field!r}")
    values = np.asarray(dataset.arrays[motion_field])
    if values.ndim != 1 or not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
        raise ValueError("explicit latent motion identities must be finite integers")
    result = {int(value) for value in values.tolist()}
    if not result or any(value < 0 for value in result):
        raise ValueError("explicit latent motion identities must be non-empty and non-negative")
    return result


def _require_same_dataset_abi(
    train: SequenceDistillDataset,
    validation: SequenceDistillDataset,
) -> None:
    if (
        train.student_obs_dim != validation.student_obs_dim
        or train.reference_features_dim != validation.reference_features_dim
        or train.action_dim != validation.action_dim
        or list(train.actuator_names) != list(validation.actuator_names)
        or train.action_schema_hash != validation.action_schema_hash
    ):
        raise ValueError("explicit latent train/validation action or observation ABI differs")
    if (train.actuator_ctrlrange is None) != (validation.actuator_ctrlrange is None):
        raise ValueError("explicit latent train/validation ctrlrange provenance differs")
    if train.actuator_ctrlrange is not None and not np.array_equal(
        train.actuator_ctrlrange,
        validation.actuator_ctrlrange,
    ):
        raise ValueError("explicit latent train/validation ordered ctrlrange differs")
    for key in (
        "student_state_schema_hash",
        "body_obs_schema_hash",
        "student_obs_filter",
        "physical_signal_semantics",
        "physical_capture",
    ):
        if train.metadata.get(key) != validation.metadata.get(key):
            raise ValueError(f"explicit latent train/validation metadata ABI differs for {key!r}")


def _evaluate_once(
    *,
    variables,
    posterior,
    prior,
    decoder_bundle: DecoderBundle,
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
    eval_emg_context = None
    eval_emg_reference = None
    if bool(config.emg_privileged_enabled):
        legs = [jnp.asarray(dataset.arrays["emg_synergy_mean"][:num_samples], dtype=jnp.float32)]
        eval_emg_reference = (
            legs[0],
            jnp.asarray(dataset.arrays["emg_synergy_scale"][:num_samples], dtype=jnp.float32),
            jnp.asarray(dataset.arrays["emg_synergy_valid"][:num_samples], dtype=jnp.float32),
        )
        if bool(config.emg_include_scale):
            legs.append(jnp.log(eval_emg_reference[1] + 1e-6))
        if bool(config.emg_include_confidence):
            legs.append(eval_emg_reference[2])
        eval_emg_context = jnp.concatenate(legs, axis=-1)
    posterior_mu, posterior_raw_sigma = posterior.apply(
        variables["encoder"], flat_state, flat_reference, eval_emg_context
    )
    prior_mu, prior_raw_sigma = prior.apply(variables["prior"], flat_state)
    posterior_output = apply_decoder(
        decoder_bundle,
        variables["decoder"],
        flat_state,
        posterior_mu,
        return_aux=True,
    )
    prior_output = apply_decoder(
        decoder_bundle,
        variables["decoder"],
        flat_state,
        prior_mu,
        return_aux=True,
    )
    posterior_action = posterior_output.action
    prior_mean_action = prior_output.action
    phase_values = dataset.arrays.get(str(config.phase_field))
    sample_weight = _phase_sample_weights(
        None if phase_values is None else phase_values[:num_samples],
        config.phase_balance_weights,
        phase_contract=config.phase_contract,
    )
    teacher_physical = None
    if float(config.physical_excitation_weight) > 0.0:
        teacher_physical = jnp.asarray(
            dataset.arrays[str(config.physical_excitation_field)][:num_samples],
            dtype=jnp.float32,
        )
    residual_excitation = (
        posterior_output.residual_excitation if decoder_bundle.decoder_type == SYNERGY_RESIDUAL_DECODER else None
    )
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
        sample_weight=sample_weight,
        predicted_physical_excitation=(posterior_output.physical_excitation if teacher_physical is not None else None),
        teacher_physical_excitation=teacher_physical,
        physical_excitation_weight=float(config.physical_excitation_weight),
        residual_excitation=residual_excitation,
        residual_l1_weight=float(config.synergy_residual_l1_weight),
        residual_l2_weight=float(config.synergy_residual_l2_weight),
        baseline_excitation=(posterior_output.baseline_excitation if decoder_bundle.is_synergy else None),
        baseline_l1_weight=float(config.synergy_baseline_l1_weight),
        baseline_l2_weight=float(config.synergy_baseline_l2_weight),
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
    posterior_mse = _weighted_mse(posterior_action, flat_teacher_action, sample_weight)
    prior_mse = _weighted_mse(prior_mean_action, flat_teacher_action, sample_weight)
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
            "prior_sigma_min_clamp_fraction": jnp.mean(prior_sigma <= float(config.sigma_min) + sigma_tolerance),
            "prior_sigma_max_clamp_fraction": jnp.mean(prior_sigma >= float(config.sigma_max) - sigma_tolerance),
            "decoder_saturation_fraction": jnp.mean(jnp.abs(posterior_action) > 0.98),
            "prior_decoder_saturation_fraction": jnp.mean(jnp.abs(prior_mean_action) > 0.98),
            "action_min": float(config.action_min),
            "action_max": float(config.action_max),
            "num_eval_samples": int(flat_state.shape[0]),
            "residual_energy_ratio": _energy_ratio(
                posterior_output.residual_excitation,
                posterior_output.physical_excitation,
            ),
        }
    )
    if decoder_bundle.is_synergy:
        metrics.update(
            _metrics_to_float(
                {
                    "synergy_coefficient_mean": jnp.mean(posterior_output.synergy_coefficients),
                    "synergy_coefficient_std": jnp.std(posterior_output.synergy_coefficients),
                    "baseline_energy_ratio": _energy_ratio(
                        posterior_output.baseline_excitation,
                        posterior_output.physical_excitation,
                    ),
                }
            )
        )
    if eval_emg_context is not None and eval_emg_reference is not None:
        # Zeroing the context is exactly the deployment condition (prior-only,
        # no EMG).  A posterior that barely moves has not become dependent on a
        # signal the deployed model will not have.
        blank_context = jnp.zeros_like(eval_emg_context)
        blank_mu, _ = posterior.apply(variables["encoder"], flat_state, flat_reference, blank_context)
        blank_action = apply_decoder(decoder_bundle, variables["decoder"], flat_state, blank_mu, return_aux=True).action
        emg_metrics = {
            "emg_context_width": int(eval_emg_context.shape[-1]),
            "emg_valid_fraction": jnp.mean(eval_emg_reference[2]),
            "emg_blank_context_posterior_mu_l2": jnp.mean(jnp.linalg.norm(posterior_mu - blank_mu, axis=-1)),
            "emg_blank_context_action_mse": _weighted_mse(blank_action, flat_teacher_action, sample_weight),
        }
        if "synergy_head" in variables:
            predicted_synergy = SynergyHead(synergy_dim=int(config.emg_synergy_dim)).apply(
                variables["synergy_head"], posterior_mu
            )
            emg_metrics |= {
                "emg_synergy_head_loss": emg_synergy_tube_loss(
                    predicted_synergy,
                    eval_emg_reference[0],
                    eval_emg_reference[1],
                    eval_emg_reference[2],
                    kappa=float(config.emg_tube_kappa),
                ),
                "emg_synergy_head_correlation": _masked_correlation(
                    predicted_synergy, eval_emg_reference[0], eval_emg_reference[2]
                ),
                "emg_synergy_head_mean": jnp.mean(predicted_synergy),
            }
        metrics.update(_metrics_to_float(emg_metrics))
    if phase_values is not None:
        metrics.update(
            _per_phase_evaluation_metrics(
                phase_ids=np.asarray(phase_values[:num_samples]),
                predicted_action=np.asarray(jax.device_get(posterior_action)),
                teacher_action=np.asarray(jax.device_get(flat_teacher_action)),
                residual_excitation=np.asarray(jax.device_get(posterior_output.residual_excitation)),
                physical_excitation=np.asarray(jax.device_get(posterior_output.physical_excitation)),
                phase_contract=config.phase_contract,
            )
        )
    # Preserve the historical key while making its semantics unambiguous.
    metrics["action_mse"] = metrics["posterior_action_mse"]
    return metrics


_FOREHAND_PHASE_NAMES = FOREHAND_PHASE_NAMES


def _validate_phase_contract(config: LatentTrainConfig) -> None:
    if config.phase_contract is None:
        return
    contract = normalize_phase_contract(config.phase_contract)
    if str(config.phase_field) != str(contract["phase_field"]):
        raise ValueError("latent phase_field differs from phase_contract.phase_field")


def _validate_decoder_training_config(
    config: LatentTrainConfig,
    decoder_bundle: DecoderBundle,
) -> None:
    if (
        str(config.physical_excitation_field) != "muscle_excitation"
        or float(config.physical_excitation_min) != 0.0
        or float(config.physical_excitation_max) != 1.0
    ):
        raise ValueError(
            "latent physical excitation config must use canonical muscle_excitation with exact [0,1] bounds"
        )
    nonnegative = {
        "physical_excitation_weight": config.physical_excitation_weight,
        "synergy_residual_l1_weight": config.synergy_residual_l1_weight,
        "synergy_residual_l2_weight": config.synergy_residual_l2_weight,
        "synergy_residual_smooth_weight": config.synergy_residual_smooth_weight,
        "synergy_baseline_l1_weight": config.synergy_baseline_l1_weight,
        "synergy_baseline_l2_weight": config.synergy_baseline_l2_weight,
    }
    invalid = {
        name: float(value) for name, value in nonnegative.items() if not np.isfinite(float(value)) or float(value) < 0.0
    }
    if invalid:
        raise ValueError(f"latent decoder loss weights must be finite and non-negative: {invalid}")
    residual_weights = (
        float(config.synergy_residual_l1_weight),
        float(config.synergy_residual_l2_weight),
        float(config.synergy_residual_smooth_weight),
    )
    if any(value > 0.0 for value in residual_weights) and (decoder_bundle.decoder_type != SYNERGY_RESIDUAL_DECODER):
        raise ValueError("synergy residual penalties require decoder_type='synergy_residual'")
    baseline_weights = (
        float(config.synergy_baseline_l1_weight),
        float(config.synergy_baseline_l2_weight),
    )
    if any(value > 0.0 for value in baseline_weights) and not (
        decoder_bundle.is_synergy and bool(config.synergy_include_baseline)
    ):
        raise ValueError("synergy baseline penalties require a synergy decoder with baseline enabled")


def _validate_portable_decoder_dataset_contract(
    dataset: SequenceDistillDataset,
    decoder_bundle: DecoderBundle,
    *,
    split: str,
) -> None:
    """Require train/val data to name the exact Stage-1 decoder core."""

    frozen = decoder_bundle.frozen_body_decoder
    if frozen is None:
        return
    manifest = dataset.metadata.get("body_synergy_contract")
    artifact_fingerprint = dataset.metadata.get("frozen_body_decoder_fingerprint")
    if not isinstance(manifest, Mapping) or artifact_fingerprint in (None, ""):
        raise ValueError(
            f"portable latent synergy training requires {split} dataset metadata "
            "to contain body_synergy_contract and frozen_body_decoder_fingerprint"
        )
    from musclemimic.synergy.multistage_contract import BodySynergyContractV2

    dataset_contract = BodySynergyContractV2.from_manifest(manifest)
    frozen.body_synergy_contract.assert_portable_compatible(dataset_contract)
    if str(artifact_fingerprint) != frozen.artifact_fingerprint:
        raise ValueError(f"{split} dataset frozen decoder fingerprint differs from latent artifact")
    supplied_contract_fingerprint = dataset.metadata.get("body_synergy_contract_fingerprint")
    if (
        supplied_contract_fingerprint not in (None, "")
        and str(supplied_contract_fingerprint) != dataset_contract.contract_fingerprint
    ):
        raise ValueError(f"{split} dataset BodySynergyContractV2 fingerprint is invalid")
    supplied_portable_fingerprint = dataset.metadata.get("body_synergy_portable_core_fingerprint")
    if (
        supplied_portable_fingerprint not in (None, "")
        and str(supplied_portable_fingerprint) != dataset_contract.portable_decoder_core_fingerprint
    ):
        raise ValueError(f"{split} dataset portable decoder core fingerprint is invalid")


def _validate_optional_training_fields(
    dataset: SequenceDistillDataset,
    config: LatentTrainConfig,
    decoder_bundle: DecoderBundle,
    *,
    split: str,
) -> None:
    if config.phase_balance_weights is not None:
        field = str(config.phase_field)
        if field not in dataset.arrays:
            raise ValueError(f"phase-balanced latent training requires {field!r} in every {split} shard")
        phase_ids = np.asarray(dataset.arrays[field])
        if phase_ids.ndim != 1 or not np.all(np.isfinite(phase_ids)):
            raise ValueError(f"{split} {field!r} must be a finite rank-1 phase ID array")
        if not np.all(phase_ids == np.floor(phase_ids)):
            raise ValueError(f"{split} {field!r} must contain integer phase IDs")
        phase_ids = phase_ids.astype(np.int64)
        phases_and_names = phase_items(config.phase_contract)
        names_by_id = dict(phases_and_names)
        unknown = sorted(set(phase_ids.tolist()) - set(names_by_id))
        if unknown:
            raise ValueError(f"{split} {field!r} contains unknown phase IDs: {unknown}")
        configured = _canonical_phase_weights(
            config.phase_balance_weights,
            phase_contract=config.phase_contract,
        )
        missing = [names_by_id[phase_id] for phase_id in configured if not np.any(phase_ids == phase_id)]
        if missing:
            raise ValueError(f"phase-balanced {split} data has no samples for configured phases: {missing}")

    if float(config.physical_excitation_weight) > 0.0:
        field = str(config.physical_excitation_field)
        if field not in dataset.arrays:
            raise ValueError(f"physical excitation loss requires {field!r} in every {split} shard")
        target = np.asarray(dataset.arrays[field], dtype=np.float64)
        expected_shape = (int(dataset.num_samples), int(dataset.action_dim))
        if target.shape != expected_shape:
            raise ValueError(f"{split} {field!r} must have shape {expected_shape}, got {target.shape}")
        if not np.all(np.isfinite(target)):
            raise ValueError(f"{split} {field!r} contains non-finite values")
        bounds = np.asarray(decoder_bundle.excitation_bounds, dtype=np.float64)
        if np.any(target < bounds[:, 0] - 1e-6) or np.any(target > bounds[:, 1] + 1e-6):
            raise ValueError(f"{split} {field!r} lies outside the decoder's declared physical excitation bounds")

    if bool(config.emg_privileged_enabled):
        synergy_dim = int(config.emg_synergy_dim)
        expected_shape = (int(dataset.num_samples), synergy_dim)
        for name in EMG_CONTEXT_FIELDS:
            if name not in dataset.arrays:
                raise ValueError(
                    f"privileged EMG distillation requires {name!r} in every {split} shard; "
                    "collect the teacher dataset with --save_emg_reference"
                )
            values = np.asarray(dataset.arrays[name], dtype=np.float64)
            if values.shape != expected_shape:
                raise ValueError(f"{split} {name!r} must have shape {expected_shape}, got {values.shape}")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{split} {name!r} contains non-finite values")
        scale = np.asarray(dataset.arrays["emg_synergy_scale"], dtype=np.float64)
        if np.any(scale <= 0.0):
            raise ValueError(
                f"{split} 'emg_synergy_scale' must be strictly positive so a phase bin "
                "never degenerates into a hard label"
            )
        valid = np.asarray(dataset.arrays["emg_synergy_valid"], dtype=np.float64)
        if np.any(valid < 0.0) or np.any(valid > 1.0):
            raise ValueError(f"{split} 'emg_synergy_valid' must lie inside [0, 1]")
        if not np.any(valid > 0.0):
            raise ValueError(
                f"{split} EMG reference has no valid synergy component; the privileged context would be pure noise"
            )
        mean = np.asarray(dataset.arrays["emg_synergy_mean"], dtype=np.float64)
        if np.any(mean < 0.0):
            raise ValueError(f"{split} 'emg_synergy_mean' must be non-negative")
        if bool(config.emg_require_reference_hash):
            from musclemimic.physiology import (
                load_emg_phase_reference_tube,
                resolve_emg_reference_reward_gate,
            )

            tube = load_emg_phase_reference_tube(
                str(config.emg_reference_manifest)
            )
            resolve_emg_reference_reward_gate(tube, enabled=True)
            semantics = dataset.metadata.get("emg_reference_semantics")
            if not isinstance(semantics, Mapping):
                raise ValueError(
                    f"{split} privileged EMG dataset lacks emg_reference_semantics"
                )
            expected = {
                "schema_version": "emg_reference_capture_v1",
                "reference_id": tube.reference_id,
                "reference_fingerprint": tube.reference_fingerprint,
                "review_status": "verified",
                "training_enabled": True,
                "mapping_sha256": str(
                    tube.mapping_binding.get("mapping_sha256", "")
                ),
                "mapping_review_status": "verified",
                "trial_qc_review_schema_version": str(
                    tube.provenance["trial_qc_review"]["schema_version"]
                ),
                "trial_qc_review_sha256": str(
                    tube.provenance["trial_qc_review"]["review_sha256"]
                ),
                "synergy_count": int(tube.synergy_count),
            }
            mismatches = {
                key: (semantics.get(key), value)
                for key, value in expected.items()
                if semantics.get(key) != value
            }
            if mismatches:
                raise ValueError(
                    f"{split} privileged EMG dataset was collected from a "
                    f"different reference tube: {mismatches}"
                )


def _canonical_phase_weights(
    weights: dict[str, float] | None,
    *,
    phase_contract: dict[str, Any] | None = None,
) -> dict[int, float]:
    if weights is None:
        return {}
    if not isinstance(weights, dict) or not weights:
        raise ValueError("phase_balance_weights must be a non-empty mapping")
    phases_and_names = phase_items(phase_contract)
    by_name = {name: phase_id for phase_id, name in phases_and_names}
    known_ids = set(by_name.values())
    result: dict[int, float] = {}
    for raw_key, raw_value in weights.items():
        key = str(raw_key).strip().lower()
        if key in by_name:
            phase_id = by_name[key]
        else:
            try:
                phase_id = int(key)
            except ValueError as exc:
                raise ValueError(f"unknown phase weight key: {raw_key!r}") from exc
        if phase_id not in known_ids:
            raise ValueError(f"phase weight ID is outside the declared phase_contract: {phase_id}")
        value = float(raw_value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"phase weight for {raw_key!r} must be finite and non-negative")
        if phase_id in result:
            raise ValueError(f"duplicate phase weight for ID {phase_id}")
        result[phase_id] = value
    if not any(value > 0.0 for value in result.values()):
        raise ValueError("phase_balance_weights must contain at least one positive weight")
    return result


def _phase_sample_weights_from_batch(
    batch: dict[str, Any],
    *,
    phase_field: str,
    phase_balance_weights: dict[str, float] | None,
    phase_contract: dict[str, Any] | None = None,
):
    if phase_balance_weights is None:
        return None
    if phase_field not in batch:
        raise ValueError(f"phase-balanced batch is missing {phase_field!r}")
    return _phase_sample_weights(
        batch[phase_field],
        phase_balance_weights,
        phase_contract=phase_contract,
    )


def _phase_sample_weights(
    phase_ids,
    weights: dict[str, float] | None,
    *,
    phase_contract: dict[str, Any] | None = None,
):
    if weights is None:
        return None
    phases = jnp.asarray(phase_ids).reshape(-1)
    sample_weights = jnp.ones(phases.shape, dtype=jnp.float32)
    for phase_id, weight in _canonical_phase_weights(
        weights,
        phase_contract=phase_contract,
    ).items():
        sample_weights = jnp.where(phases == int(phase_id), float(weight), sample_weights)
    return sample_weights


def _physical_target_from_batch(
    batch: dict[str, Any],
    *,
    field: str,
    enabled: bool,
):
    if not enabled:
        return None
    if field not in batch:
        raise ValueError(f"physical excitation batch is missing {field!r}")
    value = jnp.asarray(batch[field], dtype=jnp.float32)
    return value.reshape((-1, value.shape[-1]))


EMG_CONTEXT_FIELDS = ("emg_synergy_mean", "emg_synergy_scale", "emg_synergy_valid")
"""Batch fields assembled into the privileged posterior context."""


def emg_context_width(*, synergy_dim: int, include_scale: bool, include_confidence: bool) -> int:
    """Width of the assembled EMG context vector for a given configuration."""

    legs = 1 + int(bool(include_scale)) + int(bool(include_confidence))
    return int(synergy_dim) * legs


def _emg_context_from_batch(
    batch: dict[str, Any],
    *,
    enabled: bool,
    synergy_dim: int,
    include_scale: bool,
    include_confidence: bool,
):
    """Assemble ``[mu_H, log(s_H + eps), c_H]`` from the batch, flattened to rank 2.

    Returns ``None`` when the feature is off, mirroring
    :func:`_physical_target_from_batch`, so a disabled run takes the original
    code path with no EMG tensors in the graph at all.
    """

    if not enabled:
        return None
    missing = [name for name in EMG_CONTEXT_FIELDS if name not in batch]
    if missing:
        raise ValueError(
            f"privileged EMG distillation requires batch fields {missing}; "
            "collect the teacher dataset with --save_emg_reference"
        )
    mean = jnp.asarray(batch["emg_synergy_mean"], dtype=jnp.float32)
    scale = jnp.asarray(batch["emg_synergy_scale"], dtype=jnp.float32)
    valid = jnp.asarray(batch["emg_synergy_valid"], dtype=jnp.float32)
    width = mean.shape[-1]
    if int(width) != int(synergy_dim):
        raise ValueError(f"emg_synergy_mean has width {int(width)} but emg_synergy_dim is {int(synergy_dim)}")
    for name, array in (("emg_synergy_scale", scale), ("emg_synergy_valid", valid)):
        if array.shape[-1] != width:
            raise ValueError(f"{name} width {array.shape[-1]} differs from emg_synergy_mean {width}")

    legs = [mean]
    if include_scale:
        legs.append(jnp.log(scale + 1e-6))
    if include_confidence:
        legs.append(valid)
    context = jnp.concatenate(legs, axis=-1)
    flat_context = context.reshape((-1, context.shape[-1]))
    flat_mean = mean.reshape((-1, width))
    flat_scale = scale.reshape((-1, width))
    flat_valid = valid.reshape((-1, width))
    return flat_context, flat_mean, flat_scale, flat_valid


def emg_synergy_tube_loss(
    predicted,
    mean,
    scale,
    valid,
    *,
    kappa: float = 1.0,
    eps: float = 1e-6,
):
    """Uncertainty-aware synergy loss: charge only what leaves the human tube.

    Reuses the Stage 1 anchoring convention so the latent head is judged on the
    same footing as the tracking reward: deviations within ``kappa`` robust
    scales of the human centre are free, the excess is measured in scale units,
    and invalid components contribute nothing.
    """

    values = jnp.asarray(predicted, dtype=jnp.float32)
    centre = jnp.asarray(mean, dtype=jnp.float32)
    width = jnp.maximum(jnp.asarray(scale, dtype=jnp.float32), eps)
    mask = jnp.asarray(valid, dtype=jnp.float32)
    excess = jnp.abs(values - centre) - float(kappa) * width
    outside = jnp.maximum(excess, 0.0) / width
    element = mask * jnp.square(outside)
    return jnp.sum(element) / jnp.maximum(jnp.sum(mask), 1e-12)


def _masked_correlation(predicted, target, valid):
    """Pearson correlation over valid entries only; 0 when nothing is valid.

    This is the metric that separates "the latent carries EMG structure" from
    "the tube loss happens to be small", so it is reported whether or not the
    loss weight is armed.
    """

    values = jnp.asarray(predicted, dtype=jnp.float32)
    reference = jnp.asarray(target, dtype=jnp.float32)
    mask = jnp.asarray(valid, dtype=jnp.float32)
    weight = jnp.maximum(jnp.sum(mask), 1e-12)
    value_mean = jnp.sum(values * mask) / weight
    reference_mean = jnp.sum(reference * mask) / weight
    value_dev = (values - value_mean) * mask
    reference_dev = (reference - reference_mean) * mask
    covariance = jnp.sum(value_dev * reference_dev)
    denominator = jnp.sqrt(jnp.sum(jnp.square(value_dev)) * jnp.sum(jnp.square(reference_dev)))
    return jnp.where(denominator > 1e-12, covariance / jnp.maximum(denominator, 1e-12), 0.0)


def _emg_privileged_provenance(
    config: LatentTrainConfig,
    *,
    dataset_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Record the privileged-EMG training terms and the reference they came from."""

    if not bool(config.emg_privileged_enabled):
        return None
    manifest_path = str(config.emg_reference_manifest or "").strip()
    manifest: dict[str, Any] | None = None
    if manifest_path:
        resolved = Path(manifest_path)
        manifest = {"path": str(resolved.resolve()), "sha256": file_sha256(resolved)}
    semantics = None
    if dataset_metadata is not None:
        semantics = dataset_metadata.get("emg_reference_semantics")
    return {
        "synergy_dim": int(config.emg_synergy_dim),
        "context_dropout": float(config.emg_context_dropout),
        "synergy_loss_weight": float(config.emg_synergy_loss_weight),
        "tube_kappa": float(config.emg_tube_kappa),
        "include_scale": bool(config.emg_include_scale),
        "include_confidence": bool(config.emg_include_confidence),
        "context_width": emg_context_width(
            synergy_dim=int(config.emg_synergy_dim),
            include_scale=bool(config.emg_include_scale),
            include_confidence=bool(config.emg_include_confidence),
        ),
        "shuffle_context_ablation": bool(config.emg_shuffle_context_ablation),
        "reference_manifest": manifest,
        "dataset_emg_reference_semantics": semantics,
        "deployment_note": (
            "prior and decoder are EMG-free at runtime; the posterior that read this "
            "context is not exported for deployment"
        ),
    }


def _validate_emg_privileged_config(config: LatentTrainConfig) -> None:
    """Fail closed on an incoherent privileged-EMG configuration."""

    enabled = bool(config.emg_privileged_enabled)
    synergy_dim = int(config.emg_synergy_dim)
    weight = float(config.emg_synergy_loss_weight)
    dropout = float(config.emg_context_dropout)

    if not enabled:
        for name, value in (
            ("emg_synergy_loss_weight", weight),
            ("emg_synergy_dim", synergy_dim),
        ):
            if value:
                raise ValueError(f"{name} requires emg_privileged_enabled=true")
        if bool(config.emg_shuffle_context_ablation):
            raise ValueError("emg_shuffle_context_ablation requires emg_privileged_enabled=true")
        return

    if synergy_dim <= 0:
        raise ValueError("emg_privileged_enabled requires a positive emg_synergy_dim")
    if not 0.0 <= dropout <= 1.0:
        raise ValueError("emg_context_dropout must lie in [0, 1]")
    if weight < 0.0 or not np.isfinite(weight):
        raise ValueError("emg_synergy_loss_weight must be finite and non-negative")
    if float(config.emg_tube_kappa) <= 0.0:
        raise ValueError("emg_tube_kappa must be positive")
    if bool(config.emg_require_reference_hash):
        manifest = str(config.emg_reference_manifest or "").strip()
        if not manifest:
            raise ValueError(
                "emg_require_reference_hash=true requires emg_reference_manifest so the "
                "trained decoder records which EMG reference shaped it"
            )
        from musclemimic.physiology import (
            load_emg_phase_reference_tube,
            resolve_emg_reference_reward_gate,
        )

        tube = load_emg_phase_reference_tube(manifest)
        resolve_emg_reference_reward_gate(tube, enabled=True)
        if int(tube.synergy_count) != synergy_dim:
            raise ValueError(
                "emg_synergy_dim differs from the verified reference tube: "
                f"config={synergy_dim} tube={int(tube.synergy_count)}"
            )


def _weighted_mse(predicted, target, sample_weight=None):
    error = jnp.mean(jnp.square(jnp.asarray(predicted) - jnp.asarray(target)), axis=-1)
    if sample_weight is None:
        return jnp.mean(error)
    weights = jnp.maximum(jnp.asarray(sample_weight, dtype=error.dtype), 0.0)
    return jnp.sum(error * weights) / jnp.maximum(jnp.sum(weights), 1e-12)


def _energy_ratio(component, total):
    numerator = jnp.sum(jnp.square(jnp.asarray(component)))
    denominator = jnp.sum(jnp.square(jnp.asarray(total)))
    return numerator / jnp.maximum(denominator, 1e-12)


def _per_phase_evaluation_metrics(
    *,
    phase_ids: np.ndarray,
    predicted_action: np.ndarray,
    teacher_action: np.ndarray,
    residual_excitation: np.ndarray,
    physical_excitation: np.ndarray,
    phase_contract: dict[str, Any] | None = None,
) -> dict[str, float]:
    phases = np.asarray(phase_ids).reshape(-1)
    predicted = np.asarray(predicted_action)
    teacher = np.asarray(teacher_action)
    residual = np.asarray(residual_excitation)
    physical = np.asarray(physical_excitation)
    if predicted.shape != teacher.shape or predicted.shape[0] != phases.shape[0]:
        raise ValueError("per-phase evaluation arrays have inconsistent sample dimensions")
    if physical.shape[0] != phases.shape[0] or residual.shape[0] != phases.shape[0]:
        raise ValueError("per-phase decoder diagnostics have inconsistent sample dimensions")
    result: dict[str, float] = {}
    for phase_id, phase_name in phase_items(phase_contract):
        mask = phases == phase_id
        if not np.any(mask):
            continue
        result[f"action_mse_{phase_name}"] = float(np.mean(np.square(predicted[mask] - teacher[mask])))
        residual_energy = float(np.sum(np.square(residual[mask])))
        physical_energy = float(np.sum(np.square(physical[mask])))
        result[f"residual_energy_ratio_{phase_name}"] = residual_energy / max(physical_energy, 1e-12)
        result[f"num_eval_samples_{phase_name}"] = int(np.sum(mask))
    return result


def _batch_to_jax(batch: dict[str, np.ndarray]) -> dict[str, jax.Array]:
    return {key: jnp.asarray(value) for key, value in batch.items()}


def _metrics_to_float(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: float(value) if np.ndim(np.asarray(value)) == 0 else np.asarray(value).tolist()
        for key, value in metrics.items()
    }


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
            raise ValueError("decoder body actuator schema differs from canonical full MyoFullBody partition")
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


def _validate_latent_physical_dataset_contract(
    dataset: SequenceDistillDataset,
    *,
    split: str,
) -> MuscleChannelContract:
    """Fail closed on legacy affine/signed latent training datasets."""

    validate_physical_signal_semantics(dataset.metadata.get("physical_signal_semantics"))
    capture = dataset.metadata.get("physical_capture")
    if not isinstance(capture, Mapping) or capture.get("schema_version") != PHYSICAL_CAPTURE_SCHEMA_VERSION:
        raise ValueError(
            f"{split} latent dataset requires physical_capture_spec_v2; legacy affine datasets are rejected"
        )
    source_names = [str(name) for name in capture.get("actuator_names", ())]
    if source_names != list(dataset.source_actuator_names):
        raise ValueError(
            f"{split} latent dataset physical_capture actuator order differs from the source action schema"
        )
    source_contract = validate_muscle_channel_contract(
        capture.get("muscle_channel_contract"),
        expected_names=dataset.source_actuator_names,
    )
    selected_contract = source_contract.subset(dataset.action_selection.source_indices.tolist())
    activation_valid = validate_activation_valid_mask(
        capture.get("activation_valid_mask"),
        expected_width=dataset.source_action_dim,
    )[dataset.action_selection.source_indices]
    if not np.all(activation_valid):
        raise ValueError(f"{split} latent dataset contains muscle channels without one valid scalar activation state")
    if dataset.actuator_ctrlrange is None:
        raise ValueError(f"{split} latent dataset is missing ordered muscle actuator_ctrlrange")
    validate_unit_muscle_ctrlrange(
        dataset.actuator_names,
        dataset.actuator_ctrlrange,
    )
    missing_fields = sorted(set(PhysicalDistillDataset.REQUIRED_PHYSICAL_FIELDS) - set(dataset.arrays))
    if missing_fields:
        raise ValueError(f"{split} latent dataset lacks required physical fields: {missing_fields}")
    raw_ctrl = np.asarray(
        dataset.arrays["teacher_ctrl_physical"],
        dtype=np.float64,
    )
    excitation = validate_unit_muscle_excitation(dataset.arrays["muscle_excitation"])
    activation = validate_unit_muscle_activation(dataset.arrays["muscle_activation"])
    expected_shape = (dataset.num_samples, dataset.action_dim)
    if raw_ctrl.shape != expected_shape or excitation.shape != expected_shape or activation.shape != expected_shape:
        raise ValueError(f"{split} latent physical muscle arrays must have shape {expected_shape}")
    expected_excitation = physical_ctrl_to_effective_muscle_excitation(
        raw_ctrl,
        channel_contract=selected_contract,
    )
    if not np.allclose(
        excitation,
        expected_excitation,
        rtol=0.0,
        atol=1e-6,
    ):
        difference = np.abs(excitation - expected_excitation)
        first_bad = np.argwhere(difference > 1e-6)[0]
        sample_index, channel_index = (
            int(first_bad[0]),
            int(first_bad[1]),
        )
        raise ValueError(
            f"{split} latent muscle_excitation differs from "
            "clip(raw data.ctrl,0,1): "
            f"sample={sample_index} channel={channel_index} "
            f"observed={excitation[sample_index, channel_index]!r} "
            f"expected={expected_excitation[sample_index, channel_index]!r}"
        )
    for physical_field in PhysicalDistillDataset.REQUIRED_PHYSICAL_FIELDS:
        if not np.all(np.isfinite(np.asarray(dataset.arrays[physical_field]))):
            raise ValueError(f"{split} latent physical field {physical_field!r} contains non-finite values")
    return selected_contract


def _action_schema_from_dataset(
    dataset: SequenceDistillDataset,
    *,
    muscle_channel_contract: MuscleChannelContract,
) -> dict[str, Any]:
    schema = dataset.action_selection.to_manifest()
    schema["selection_schema_version"] = schema.pop("schema_version")
    schema["schema_version"] = LATENT_MUSCLE_ACTION_SCHEMA_VERSION
    ctrlrange = np.asarray(dataset.actuator_ctrlrange, dtype=np.float64)
    schema["target_ctrlrange"] = ctrlrange.tolist()
    schema["ctrlrange_schema_hash"] = ordered_schema_hash(
        kind="actuator_ctrlrange",
        payload={
            "actuator_names": list(dataset.actuator_names),
            "ctrlrange": ctrlrange.tolist(),
        },
    )
    schema.update(
        {
            "physical_signal_schema_version": PHYSICAL_SIGNAL_SCHEMA_VERSION,
            "physical_capture_schema_version": PHYSICAL_CAPTURE_SCHEMA_VERSION,
            "muscle_excitation_transform": UNIT_EXCITATION_TRANSFORM,
            "muscle_excitation_formula": MUSCLE_EXCITATION_FORMULA,
            "muscle_excitation_roundoff_policy": (MUSCLE_EXCITATION_ROUNDOFF_POLICY),
            "muscle_channel_contract": muscle_channel_contract.to_metadata(),
        }
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
        checks["production_closed_loop_evidence"] = metrics.get("closed_loop_evidence_kind") == "verified_production_v2"
    if config.require_dataset_provenance:
        stage, _role = _teacher_promotion_expectation(config)
        checks[f"{stage}_teacher_promotion_evidence"] = (
            metrics.get("teacher_promotion_evidence_kind")
            == expected_teacher_promotion_evidence_kind(config)
        )
    if metrics.get("closed_loop_evidence_kind") == "test_only_injected":
        checks["test_only_evidence_not_promotable"] = False
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "thresholds": thresholds,
    }

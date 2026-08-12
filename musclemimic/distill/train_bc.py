"""Offline behavior cloning trainer for distilled student policies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from omegaconf import OmegaConf, open_dict

from musclemimic.algorithms import PPOJax
from musclemimic.algorithms.common.checkpoint_manager import CheckpointMetadata, UnifiedCheckpointManager
from musclemimic.algorithms.common.dataclasses import TrainState
from musclemimic.algorithms.common.env_utils import apply_policy_interface_wrappers, wrap_env
from musclemimic.algorithms.ppo.config import PPOAgentState
from musclemimic.distill.action_schema import actuator_schema_hash
from musclemimic.distill.dataset import DistillDataset, motion_split_datasets
from musclemimic.distill.eval_student import (
    DistillAcceptanceThresholds,
    evaluate_mse_plateau,
)
from musclemimic.distill.losses import bc_loss, distribution_log_std, distribution_mean
from musclemimic.distill.provenance import validate_dataset_manifest
from musclemimic.runner.engine import instantiate_env
from musclemimic.runner.eval_utils import load_checkpoint
from musclemimic.runner.export_metadata import model_actuator_names


@dataclass
class BCTrainResult:
    checkpoint_path: str
    metadata_path: str
    final_train_action_mse: float
    final_val_action_mse: float | None
    convergence_path: str
    convergence_passed: bool


def _batch_to_jax(batch: dict[str, np.ndarray]) -> dict[str, jax.Array]:
    return {name: jnp.asarray(value) for name, value in batch.items()}


def _ensure_student_filter(config: Any) -> None:
    with open_dict(config):
        if "student_obs_filter" not in config.experiment:
            config.experiment.student_obs_filter = {}
        config.experiment.student_obs_filter.enabled = True
        config.experiment.student_obs_filter.drop_goal_lookahead = True
        config.experiment.student_obs_filter.keep_motion_phase = True
        config.experiment.student_obs_filter.require_goal_group = True
        config.experiment.student_obs_filter.require_motion_phase = True


def validate_dataset_matches_student_env(*, dataset: DistillDataset, env: Any, config: Any) -> int:
    """Validate that BC shards match the configured student policy input shape."""
    wrapped_env = wrap_env(env, config.experiment)
    expected_dim = int(wrapped_env.info.observation_space.shape[0])
    if int(dataset.student_obs_dim) != expected_dim:
        raise ValueError(
            "distill dataset student_obs_dim does not match configured student env: "
            f"dataset student_obs_dim={int(dataset.student_obs_dim)} expected={expected_dim}"
        )
    expected_action_dim = int(wrapped_env.info.action_space.shape[0])
    dataset_action_dim = getattr(dataset, "action_dim", None)
    if dataset_action_dim is not None and int(dataset_action_dim) != expected_action_dim:
        raise ValueError(
            "distill dataset action_dim does not match configured student env: "
            f"dataset action_dim={int(dataset_action_dim)} expected={expected_action_dim}"
        )
    policy_names = getattr(
        wrapped_env,
        "policy_actuator_names",
        getattr(wrapped_env, "policy_action_names", None),
    )
    dataset_names = getattr(dataset, "actuator_names", None)
    if policy_names is not None and dataset_names is not None:
        policy_names = [str(name) for name in policy_names]
        dataset_names = [str(name) for name in dataset_names]
        if dataset_names != policy_names:
            raise ValueError(
                "distill dataset actuator names/order do not match configured student policy interface"
            )
    return expected_dim


def evaluate_bc_loss(
    train_state: TrainState,
    network: Any,
    dataset: DistillDataset,
    batch_size: int,
    gaussian_kl_weight: float = 0.0,
) -> dict[str, float]:
    sums = {
        "total_loss": 0.0,
        "action_mse": 0.0,
        "deterministic_action_mse": 0.0,
        "mse_to_teacher_action": 0.0,
        "mse_to_teacher_mu": 0.0,
        "value_mse": 0.0,
        "gaussian_kl": 0.0,
    }
    count = 0
    for batch in dataset.iter_batches(batch_size=batch_size, shuffle=False, repeat=False):
        jbatch = _batch_to_jax(batch)
        pi, value = network.apply(
            {"params": train_state.params, "run_stats": train_state.run_stats},
            jbatch["student_obs"],
        )
        student_mu = distribution_mean(pi)
        losses = bc_loss(
            student_mu=student_mu,
            teacher_action=jbatch["teacher_action"],
            student_value=value,
            teacher_value=jbatch.get("teacher_value"),
            student_log_std=distribution_log_std(pi) if float(gaussian_kl_weight) else None,
            teacher_mu=jbatch.get("teacher_mu"),
            teacher_log_std=jbatch.get("teacher_log_std"),
            gaussian_kl_weight=float(gaussian_kl_weight),
        )
        teacher_mu = jbatch.get("teacher_mu")
        mse_to_teacher_mu = (
            jnp.mean(jnp.square(student_mu - teacher_mu))
            if teacher_mu is not None
            else losses["action_mse"]
        )
        losses = losses | {
            # Promotion observes the action actually emitted by deterministic
            # inference.  The environment clips normalized controls to [-1, 1],
            # so this metric must not silently use the unbounded Gaussian mean.
            "deterministic_action_mse": jnp.mean(
                jnp.square(jnp.clip(student_mu, -1.0, 1.0) - jbatch["teacher_action"])
            ),
            "mse_to_teacher_action": losses["action_mse"],
            "mse_to_teacher_mu": mse_to_teacher_mu,
        }
        n = int(batch["student_obs"].shape[0])
        for key in sums:
            sums[key] += float(losses[key]) * n
        count += n
    return {key: value / max(count, 1) for key, value in sums.items()}


def train_bc(
    *,
    config: Any,
    dataset_dir: str | Path,
    output_dir: str | Path,
    batch_size: int = 4096,
    num_steps: int = 200_000,
    lr: float = 3e-4,
    seed: int = 0,
    value_distill_weight: float = 0.1,
    gaussian_kl_weight: float = 0.0,
    init_ckpt: str | Path | None = None,
    log_interval: int = 100,
    val_fraction: float = 0.2,
    split_seed: int | None = None,
    motion_field: str = "motion_uid",
    strict_motion_identity: bool = True,
    convergence_eval_interval: int = 10_000,
    acceptance_thresholds: DistillAcceptanceThresholds | None = None,
    require_dataset_manifest: bool = False,
) -> BCTrainResult:
    """Train a PPO-compatible student checkpoint from teacher rollout shards."""
    _ensure_student_filter(config)
    if int(num_steps) <= 0:
        raise ValueError("BC num_steps must be positive")
    if int(convergence_eval_interval) <= 0:
        raise ValueError("convergence_eval_interval must be positive")
    env = instantiate_env(config)
    policy_action_names = _policy_actuator_names(env, config)
    dataset_path = Path(dataset_dir)
    dataset_provenance = None
    if require_dataset_manifest or (dataset_path / "dataset_manifest.json").is_file():
        dataset_provenance = validate_dataset_manifest(dataset_path)
    if not strict_motion_identity and sorted(dataset_path.glob("val_*.npz")):
        raise ValueError(
            "explicit train/val shards require stable motion_uid; local traj_no values overlap across environments"
        )
    effective_motion_field = str(motion_field) if strict_motion_identity else "traj_no"
    dataset, val_dataset, split_manifest = motion_split_datasets(
        dataset_dir,
        dataset_cls=DistillDataset,
        seed=int(seed if split_seed is None else split_seed),
        val_fraction=float(val_fraction),
        motion_field=effective_motion_field,
        target_actuator_names=policy_action_names,
    )
    _validate_gaussian_kl_action_semantics(
        train_dataset=dataset,
        val_dataset=val_dataset,
        gaussian_kl_weight=gaussian_kl_weight,
    )
    expected_student_obs_dim = validate_dataset_matches_student_env(dataset=dataset, env=env, config=config)
    agent_conf = PPOJax.init_agent_conf(env, config)
    tx = optax.adam(float(lr))

    rng = jax.random.PRNGKey(int(seed))
    init_obs = jnp.zeros((dataset.student_obs_dim,), dtype=jnp.float32)
    if init_ckpt:
        _loaded_config, loaded_agent_state, _loaded_metadata = load_checkpoint(str(init_ckpt))
        loaded_ts = loaded_agent_state.train_state
        agent_conf.network.apply(
            {"params": loaded_ts.params, "run_stats": loaded_ts.run_stats},
            init_obs,
        )
        train_state = TrainState.create(
            apply_fn=agent_conf.network.apply,
            params=loaded_ts.params,
            run_stats=loaded_ts.run_stats,
            tx=tx,
        )
    else:
        init_vars = agent_conf.network.init(rng, init_obs)
        train_state = TrainState.create(
            apply_fn=agent_conf.network.apply,
            params=init_vars["params"],
            run_stats=init_vars["run_stats"],
            tx=tx,
        )

    @jax.jit
    def train_step(ts: TrainState, batch: dict[str, jax.Array]):
        def loss_fn(params, run_stats):
            (pi, value), updates = agent_conf.network.apply(
                {"params": params, "run_stats": run_stats},
                batch["student_obs"],
                mutable=["run_stats"],
            )
            losses = bc_loss(
                student_mu=distribution_mean(pi),
                teacher_action=batch["teacher_action"],
                student_value=value,
                teacher_value=batch.get("teacher_value"),
                student_log_std=distribution_log_std(pi),
                teacher_mu=batch.get("teacher_mu"),
                teacher_log_std=batch.get("teacher_log_std"),
                action_mse_weight=1.0,
                value_distill_weight=float(value_distill_weight),
                gaussian_kl_weight=float(gaussian_kl_weight),
            )
            return losses["total_loss"], (losses, updates["run_stats"])

        (loss, (losses, new_run_stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            ts.params,
            ts.run_stats,
        )
        ts = ts.apply_gradients(grads=grads)
        ts = ts.replace(run_stats=new_run_stats)
        return ts, losses | {"total_loss": loss}

    last_losses = None
    batch_iter = dataset.iter_batches(batch_size=batch_size, shuffle=True, repeat=True)
    val_action_mse_history: list[dict[str, float | int]] = []
    last_val_metrics: dict[str, float] | None = None

    def record_heldout_mse(step: int) -> None:
        nonlocal last_val_metrics
        if val_dataset is None:
            return
        last_val_metrics = evaluate_bc_loss(
            train_state,
            agent_conf.network,
            val_dataset,
            batch_size=batch_size,
            gaussian_kl_weight=gaussian_kl_weight,
        )
        val_action_mse_history.append(
            {
                "step": int(step),
                "action_mse": float(last_val_metrics["deterministic_action_mse"]),
            }
        )

    # The same fixed held-out motion split and deterministic batch order are
    # reused at every point, including the untrained baseline at step zero.
    record_heldout_mse(0)
    for step in range(1, int(num_steps) + 1):
        batch = _batch_to_jax(next(batch_iter))
        train_state, last_losses = train_step(train_state, batch)
        if log_interval and step % int(log_interval) == 0:
            print(
                f"[bc] step={step} "
                f"loss={float(last_losses['total_loss']):.6f} "
                f"action_mse={float(last_losses['action_mse']):.6f} "
                f"value_mse={float(last_losses['value_mse']):.6f} "
                f"gaussian_kl={float(last_losses['gaussian_kl']):.6f}"
            )
        if step % int(convergence_eval_interval) == 0 or step == int(num_steps):
            record_heldout_mse(step)

    output_path = Path(output_dir)
    checkpoint_dir = output_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with open_dict(config.experiment):
        config.experiment.checkpoint_dir = str(checkpoint_dir)
        config.experiment.bc_dataset_dir = str(dataset_dir)

    agent_conf = PPOJax._agent_conf(config, agent_conf.network, tx)
    agent_state = PPOAgentState(train_state=train_state)
    ckpt_metadata = CheckpointMetadata(
        step=int(train_state.step),
        update_number=int(num_steps),
        global_timestep=0,
        target_global_timestep=0,
        learning_rate=float(lr),
        num_envs=int(config.experiment.get("num_envs", 1)),
        num_steps=int(config.experiment.get("num_steps", config.experiment.get("ppo_config", {}).get("num_steps", 1))),
        num_minibatches=int(config.experiment.get("num_minibatches", config.experiment.get("ppo_config", {}).get("num_minibatches", 1))),
        update_epochs=int(config.experiment.get("update_epochs", config.experiment.get("ppo_config", {}).get("update_epochs", 1))),
        backend=str(config.experiment.env_params.get("mjx_backend", "jax")),
        env_name=str(config.experiment.env_params.get("env_name", "")),
    )
    manager = UnifiedCheckpointManager(str(checkpoint_dir), max_to_keep=5, async_save=False)
    try:
        checkpoint_path = manager.save_checkpoint(int(train_state.step), agent_conf, agent_state, ckpt_metadata)
    finally:
        manager.close()

    train_metrics = evaluate_bc_loss(
        train_state,
        agent_conf.network,
        dataset,
        batch_size=batch_size,
        gaussian_kl_weight=gaussian_kl_weight,
    )
    val_metrics = last_val_metrics
    convergence = evaluate_mse_plateau(
        val_action_mse_history,
        thresholds=acceptance_thresholds,
    )
    convergence.update(
        {
            "evaluation_interval_steps": int(convergence_eval_interval),
            "split": "val",
            "motion_field": effective_motion_field,
            "deterministic": True,
            "motion_split": split_manifest,
        }
    )

    metadata = {
        "dataset_dir": str(dataset_dir),
        "student_obs_dim": dataset.student_obs_dim,
        "expected_student_obs_dim": expected_student_obs_dim,
        "action_dim": dataset.action_dim,
        "bc_steps": int(num_steps),
        "lr": float(lr),
        "batch_size": int(batch_size),
        "value_distill_weight": float(value_distill_weight),
        "gaussian_kl_weight": float(gaussian_kl_weight),
        "init_ckpt": None if init_ckpt is None else str(init_ckpt),
        "train": train_metrics,
        "val": val_metrics,
        "convergence": convergence,
        "dataset_metadata": dataset.metadata,
        "dataset_provenance": dataset_provenance,
        "motion_split": split_manifest,
        "action_schema_hash": actuator_schema_hash(dataset.actuator_names),
        "actuator_names": list(dataset.actuator_names),
        "student_obs_filter": OmegaConf.to_container(config.experiment.student_obs_filter, resolve=True),
    }
    if "teacher_raw_mean_saturation_fraction" in dataset.arrays:
        metadata["teacher_raw_mean_saturation_fraction"] = float(
            np.mean(dataset.arrays["teacher_raw_mean_saturation_fraction"])
        )
    if "teacher_raw_target_saturation_fraction" in dataset.arrays:
        metadata["teacher_raw_target_saturation_fraction"] = float(
            np.mean(dataset.arrays["teacher_raw_target_saturation_fraction"])
        )
    metadata_path = output_path / "distill_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    convergence_path = output_path / "convergence.json"
    convergence_path.write_text(
        json.dumps(convergence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_path / "motion_split.json").write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_bc_contract_manifests(
        config=config,
        checkpoint_path=Path(checkpoint_path),
        output_path=output_path,
        actuator_names=list(dataset.actuator_names),
        obs_size=int(dataset.student_obs_dim),
        state_schema=dataset.metadata.get("student_state_schema"),
        dataset_provenance=dataset_provenance,
    )

    return BCTrainResult(
        checkpoint_path=checkpoint_path,
        metadata_path=str(metadata_path),
        final_train_action_mse=float(train_metrics["action_mse"]),
        final_val_action_mse=float(val_metrics["action_mse"]) if val_metrics else None,
        convergence_path=str(convergence_path),
        convergence_passed=bool(convergence["passed"]),
    )


def _validate_gaussian_kl_action_semantics(
    *,
    train_dataset: Any,
    val_dataset: Any | None,
    gaussian_kl_weight: float,
) -> None:
    """Fail closed when decoded body targets do not define a Gaussian ABI."""

    if float(gaussian_kl_weight) <= 0.0:
        return
    unavailable = {
        # Current strict schema omits teacher_log_std entirely in decoded
        # body-action space.  Keep the historical explicit-placeholder label
        # fail-closed for already collected development shards.
        "unavailable_for_nonlinear_decoded_body_action",
        "unavailable_for_nonlinear_decoded_body_action_zero_placeholder",
    }
    for split_name, split_dataset in (
        ("train", train_dataset),
        ("val", val_dataset),
    ):
        if split_dataset is not None and split_dataset.metadata.get(
            "teacher_log_std_semantics"
        ) in unavailable:
            raise ValueError(
                "gaussian_kl_weight must be zero for early-synergy decoded "
                f"{split_name} targets: no diagonal Gaussian exists in the "
                "nonlinear 354-D body-action space; distill the saved c/rho "
                "policy Gaussian explicitly instead"
            )


def _policy_actuator_names(env: Any, config: Any) -> list[str]:
    policy_env = apply_policy_interface_wrappers(env, config.experiment, include_student=False)
    names = getattr(
        policy_env,
        "policy_actuator_names",
        getattr(policy_env, "policy_action_names", None),
    )
    if names is None:
        model = getattr(env, "_model", None) or getattr(env, "model", None)
        if model is None:
            raise ValueError("student environment does not expose a model or policy_action_names")
        names = model_actuator_names(model)
    names = [str(name) for name in names]
    expected = int(policy_env.info.action_space.shape[0])
    if len(names) != expected:
        raise ValueError(f"policy actuator name count {len(names)} does not match action dimension {expected}")
    return names


def _write_bc_contract_manifests(
    *,
    config: Any,
    checkpoint_path: Path,
    output_path: Path,
    actuator_names: list[str],
    obs_size: int,
    state_schema: dict[str, Any] | None,
    dataset_provenance: dict[str, Any] | None,
) -> None:
    from environment.overall_environment.src.action_manifest import (
        ActionManifest,
        write_action_manifest,
    )

    env_params = OmegaConf.to_container(config.experiment.env_params, resolve=True)
    manifest = ActionManifest.from_env_params(
        env_params,
        actuator_names=actuator_names,
        obs_size=int(obs_size),
        obs_fields=[] if state_schema is None else ["student_state_schema"],
    )
    for root in (output_path, checkpoint_path):
        write_action_manifest(root / "action_manifest.json", manifest)
        if state_schema is not None:
            (root / "student_state_schema.json").write_text(
                json.dumps(state_schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        if dataset_provenance is not None:
            (root / "distill_provenance.json").write_text(
                json.dumps(
                    dataset_provenance,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )


def load_config_for_bc(config_path: str | Path):
    """Load a Hydra fullbody config from a path, falling back to plain YAML."""
    config_name = str(config_path)
    config_path = Path(config_name)
    fullbody_dir = Path(__file__).resolve().parents[2] / "fullbody"
    try:
        import hydra
        from hydra import compose, initialize_config_dir

        if config_path.is_file():
            rel = config_path.resolve().relative_to(fullbody_dir.resolve()).with_suffix("")
            config_name = str(rel)
        with initialize_config_dir(version_base=None, config_dir=str(fullbody_dir.resolve())):
            return compose(config_name=config_name.removesuffix(".yaml"))
    except Exception:
        return OmegaConf.load(config_path)

#!/usr/bin/env python3
"""Multi-skill (or single-skill) behavior cloning into a frozen-policy export.

Consumes distill shards (``train_*.npz`` with ``student_obs``/``teacher_action``)
collected per action from the per-action tracking experts, appends a skill
one-hot to the student observation, and trains one conditioned base policy.
The output directory is a **frozen body policy export** (same file layout as
``export_frozen_body_policy``) plus ``skill_manifest.json``, so
``BaseSwingBridge`` and the hitting environments load single- and multi-skill
bases through one code path.

Why dataset-level conditioning: the mainline ``multi_action``/``skill_id``
config keys have no consumer in the codebase (the env obs_container has no
skill group). Distillation is offline, so appending the one-hot to the shard
observations is exactly equivalent and requires no mainline env changes.

Example:

    .venv/bin/python musclemimic/badminton/skill_pipeline/train_multi_skill_bc.py \
        --dataset forehandClear_standard=datasets/_global/distill/forehandClear_standard \
        --dataset smash=datasets/_global/distill/smash \
        --schema-from checkpoints/<expert>/checkpoint_XXXX \
        --output-dir outputs/skill_pipeline/base_multi
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_shards(dataset_dir: Path, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    shards = sorted(dataset_dir.rglob(f"{prefix}_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no {prefix}_*.npz shards under {dataset_dir}")
    obs_list, act_list = [], []
    for shard in shards:
        with np.load(shard, allow_pickle=False) as payload:
            obs_list.append(np.asarray(payload["student_obs"], dtype=np.float32))
            act_list.append(np.asarray(payload["teacher_action"], dtype=np.float32))
    return np.concatenate(obs_list), np.concatenate(act_list)


def _mlp_init(rng: np.random.Generator, sizes: list[int]) -> list[dict[str, np.ndarray]]:
    layers = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        scale = np.sqrt(2.0 / fan_in)
        layers.append(
            {
                "kernel": rng.normal(0.0, scale, (fan_in, fan_out)).astype(np.float32),
                "bias": np.zeros(fan_out, dtype=np.float32),
            }
        )
    return layers


def train_bc(
    obs: np.ndarray,
    actions: np.ndarray,
    *,
    hidden: tuple[int, ...],
    activation: str,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    val_obs: np.ndarray | None = None,
    val_actions: np.ndarray | None = None,
    log_every: int = 500,
) -> tuple[list[dict[str, np.ndarray]], dict[str, np.ndarray], list[dict[str, float]]]:
    import jax
    import jax.numpy as jnp
    import optax

    obs_mean = obs.mean(axis=0)
    obs_var = obs.var(axis=0) + 1e-8
    stats = {
        "mean": obs_mean.astype(np.float32),
        "var": obs_var.astype(np.float32),
        "count": np.asarray(float(obs.shape[0]), dtype=np.float32),
    }

    rng = np.random.default_rng(seed)
    params = _mlp_init(rng, [obs.shape[1], *hidden, actions.shape[1]])
    params = jax.tree_util.tree_map(jnp.asarray, params)
    act_fn = {"tanh": jnp.tanh, "relu": jax.nn.relu, "elu": jax.nn.elu, "swish": jax.nn.silu}[activation]

    mean_j = jnp.asarray(stats["mean"])
    var_j = jnp.asarray(stats["var"])

    def forward(p, x):
        x = (x - mean_j) / jnp.sqrt(var_j)
        for layer in p[:-1]:
            x = act_fn(x @ layer["kernel"] + layer["bias"])
        return x @ p[-1]["kernel"] + p[-1]["bias"]

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def update(p, opt_state, batch_obs, batch_act):
        def loss_fn(p):
            pred = forward(p, batch_obs)
            return jnp.mean(jnp.square(pred - batch_act))

        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(p, updates), opt_state, loss

    @jax.jit
    def mse(p, x, y):
        return jnp.mean(jnp.square(forward(p, x) - y))

    history: list[dict[str, float]] = []
    n = obs.shape[0]
    obs_j = jnp.asarray(obs)
    act_j = jnp.asarray(actions)
    for step in range(1, steps + 1):
        idx = rng.integers(0, n, size=min(batch_size, n))
        params, opt_state, loss = update(params, opt_state, obs_j[idx], act_j[idx])
        if step % log_every == 0 or step == steps:
            record = {"step": step, "train_mse": float(loss)}
            if val_obs is not None and val_actions is not None:
                record["val_mse"] = float(mse(params, jnp.asarray(val_obs), jnp.asarray(val_actions)))
            history.append(record)
            print(
                f"bc step {step}/{steps} train_mse={record['train_mse']:.6f}"
                + (f" val_mse={record.get('val_mse'):.6f}" if "val_mse" in record else ""),
                flush=True,
            )
    params_np = jax.tree_util.tree_map(lambda x: np.asarray(x), params)
    return params_np, stats, history


def export_frozen_artifact(
    output_dir: Path,
    *,
    params: list[dict[str, np.ndarray]],
    stats: dict[str, np.ndarray],
    hidden: tuple[int, ...],
    activation: str,
    obs_size: int,
    action_size: int,
    schema_payload: dict,
    actuator_names: list[str],
    skill_actions: list[str],
    metadata: dict,
) -> None:
    sys.path.insert(0, str(REPO_ROOT / "environment" / "overall_environment" / "src"))
    from frozen_body_policy import (
        ActorCheckpointShapeReport,
        ActorCheckpointSpec,
        FrozenBodyPolicyManifest,
        _manifest_to_dict,
        _save_npz_tree,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    actor_tree = {f"Dense_{i}": layer for i, layer in enumerate(params)}
    _save_npz_tree(output_dir / "params.npz", {"actor": actor_tree})
    _save_npz_tree(output_dir / "run_stats.npz", {"RunningMeanStd_0": stats})

    spec = ActorCheckpointSpec(
        obs_size=obs_size,
        action_size=action_size,
        actor_hidden_layers=tuple(hidden),
        critic_hidden_layers=tuple(hidden),
        activation=activation,
        init_std=0.1,
        learnable_std=False,
        use_layernorm=False,
        layernorm_eps=1e-6,
    )
    report = ActorCheckpointShapeReport(
        valid=True,
        actor_output_kernel_shape=(hidden[-1], action_size),
        log_std_shape=(action_size,),
        run_stats_mean_shape=(obs_size,),
        reason="multi-skill BC export",
    )
    manifest = FrozenBodyPolicyManifest(
        schema_version=1,
        source_checkpoint=str(metadata.get("schema_from", "multi_skill_bc")),
        tensor_format="npz",
        has_tensors=True,
        actor_spec=spec,
        shape_report=report,
        params_file="params.npz",
        run_stats_file="run_stats.npz",
        body_obs_schema_file="body_obs_schema.json",
        action_manifest_file="action_manifest.json",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(_manifest_to_dict(manifest), indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "body_obs_schema.json").write_text(
        json.dumps(schema_payload, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "action_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "env_name": "MjxMyoFullBody",
                "disable_fingers": True,
                "action_size": action_size,
                "actuator_names": actuator_names,
                "obs_size": obs_size,
                "obs_fields": [],
                "control_min": -1.0,
                "control_max": 1.0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "skill_manifest.json").write_text(
        json.dumps(
            {
                "actions": skill_actions,
                "condition_size": len(skill_actions) if len(skill_actions) > 1 else 0,
                "condition_layout": "onehot_appended_after_student_obs",
                **metadata,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="<action_name>=<shard_dir>, repeatable; order defines the one-hot layout",
    )
    parser.add_argument(
        "--schema-from",
        required=True,
        help="teacher checkpoint dir OR existing frozen export dir providing body-obs schema + actuator names",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden", type=int, nargs="+", default=[512, 512])
    parser.add_argument("--activation", default="tanh")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pairs = []
    for spec in args.dataset:
        name, _, path = spec.partition("=")
        if not path:
            raise ValueError(f"--dataset must be <action>=<dir>, got {spec!r}")
        pairs.append((name, Path(path)))
    actions = [name for name, _ in pairs]
    condition_size = len(actions) if len(actions) > 1 else 0

    all_obs, all_act = [], []
    val_obs_list, val_act_list = [], []
    per_action_counts: dict[str, int] = {}
    for index, (name, shard_dir) in enumerate(pairs):
        obs, act = _load_shards(shard_dir, "train")
        if condition_size:
            onehot = np.zeros((obs.shape[0], condition_size), dtype=np.float32)
            onehot[:, index] = 1.0
            obs = np.concatenate([obs, onehot], axis=1)
        all_obs.append(obs)
        all_act.append(act)
        per_action_counts[name] = int(obs.shape[0])
        try:
            vobs, vact = _load_shards(shard_dir, "val")
            if condition_size:
                vhot = np.zeros((vobs.shape[0], condition_size), dtype=np.float32)
                vhot[:, index] = 1.0
                vobs = np.concatenate([vobs, vhot], axis=1)
            val_obs_list.append(vobs)
            val_act_list.append(vact)
        except FileNotFoundError:
            pass

    obs = np.concatenate(all_obs)
    actions_arr = np.concatenate(all_act)
    val_obs = np.concatenate(val_obs_list) if val_obs_list else None
    val_act = np.concatenate(val_act_list) if val_act_list else None
    print(
        f"multi-skill BC: {len(actions)} skills, obs {obs.shape}, actions {actions_arr.shape}, "
        f"per-action {per_action_counts}"
    )

    t0 = time.time()
    params, stats, history = train_bc(
        obs,
        actions_arr,
        hidden=tuple(args.hidden),
        activation=args.activation,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        val_obs=val_obs,
        val_actions=val_act,
    )

    # schema + actuator names from the teacher checkpoint or a prior export
    schema_source = Path(args.schema_from)
    if (schema_source / "body_obs_schema.json").is_file():
        schema_payload = json.loads((schema_source / "body_obs_schema.json").read_text(encoding="utf-8"))
        actuator_names = json.loads(
            (schema_source / "action_manifest.json").read_text(encoding="utf-8")
        )["actuator_names"]
    else:
        sys.path.insert(0, str(REPO_ROOT / "environment" / "overall_environment" / "src"))
        from action_manifest import reconstruct_action_manifest
        from body_obs_adapter import reconstruct_body_obs_schema

        schema_payload = asdict(reconstruct_body_obs_schema(schema_source))
        actuator_names = list(reconstruct_action_manifest(schema_source).actuator_names)

    # The distilled student sees the filtered observation: teacher body state
    # with the goal lookahead dropped and only the motion phase kept, so the
    # exported schema's goal segment shrinks to a single phase slot.
    student_dim = int(obs.shape[1]) - condition_size
    body_dim = (
        int(schema_payload["total_size"]) - int(schema_payload["goal_size"])
    )
    if student_dim != body_dim + 1:
        raise ValueError(
            "distill dataset obs does not look like [teacher body obs + phase]: "
            f"student_dim={student_dim}, teacher body dim={body_dim} (+1 phase expected)"
        )
    schema_payload = dict(schema_payload)
    schema_payload["goal_size"] = 1
    schema_payload["total_size"] = body_dim + 1

    export_frozen_artifact(
        args.output_dir,
        params=params,
        stats=stats,
        hidden=tuple(args.hidden),
        activation=args.activation,
        obs_size=int(obs.shape[1]),
        action_size=int(actions_arr.shape[1]),
        schema_payload=schema_payload,
        actuator_names=list(actuator_names),
        skill_actions=actions,
        metadata={
            "schema_from": str(schema_source),
            "per_action_samples": per_action_counts,
            "train_history_tail": history[-3:],
            "wall_seconds": round(time.time() - t0, 1),
        },
    )
    print(f"exported multi-skill base to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

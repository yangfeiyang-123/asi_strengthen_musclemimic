"""Run held-out Stage-2 prior-mean/LAB sweeps and update latent promotion."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from loco_mujoco.task_factories import TaskFactory
from musclemimic.algorithms.common.env_utils import apply_policy_interface_wrappers
from musclemimic.distill.body_obs_schema import build_body_obs_schema
from musclemimic.distill.collect_teacher import (
    _resolve_actuator_ctrlrange,
    _resolve_actuator_names,
    _student_state_schema,
)
from musclemimic.distill.config_overrides import apply_collection_overrides
from musclemimic.distill.motion_identity import normalize_motion_path, resolve_config_motion_paths
from musclemimic.distill.obs_filter import build_student_obs_indices
from musclemimic.distill.provenance import (
    canonical_json_sha256,
    checkpoint_content_fingerprint,
    file_sha256,
)
from musclemimic.latent_muscle.closed_loop_eval import (
    OFFLINE_PROMOTION_METRICS,
    ClosedLoopEvalConfig,
    evaluate_latent_closed_loop,
    select_direct_rollout_policy,
    validate_closed_loop_promotion_report,
)
from musclemimic.latent_muscle.runtime import load_latent_runtime
from musclemimic.latent_muscle.train_latent import (
    LatentTrainConfig,
    _evaluate_promotion_gates,
)
from musclemimic.runner.eval_utils import apply_temporal_params, load_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent_checkpoint", required=True)
    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--motion_path", nargs="+", default=None)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=120,
        help="Per-motion held-out horizon; 120 control steps is about 1.2 s at the canonical 100 Hz.",
    )
    parser.add_argument(
        "--direct_rollout_metrics",
        type=Path,
        default=None,
        help="comparison_metrics.json from deterministic held-out direct-student rollouts.",
    )
    parser.add_argument(
        "--direct_bc_metrics",
        type=Path,
        default=None,
        help="Deprecated compatibility alias; use --direct_rollout_metrics.",
    )
    parser.add_argument(
        "--promotion_policy",
        choices=("student_bc", "student_bc_dagger", "student_bc_ppo"),
        default=None,
    )
    parser.add_argument(
        "--direct_promotion_evidence",
        type=Path,
        default=None,
        help="Strict direct_promotion_evidence.json; defaults beside comparison_metrics.json.",
    )
    parser.add_argument("--require_pass", action="store_true", default=False)
    parser.add_argument(
        "--phase_field",
        default=None,
        help="Optional required integer phase ID key in env step info.",
    )
    parser.add_argument("--require_all_phases", action="store_true", default=False)
    parser.add_argument("--collect_decoder_usage", action="store_true", default=False)
    parser.add_argument("--collect_jacobian_alignment", action="store_true", default=False)
    parser.add_argument(
        "--alignment_synergy_basis",
        type=Path,
        default=None,
        help="Formal excitation basis for direct-decoder Jacobian alignment.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if [float(value) for value in args.lambdas] != [0.0, 0.25, 0.5]:
        raise ValueError("production latent promotion requires lambdas exactly 0, 0.25, 0.5")
    if int(args.max_steps) != 120:
        raise ValueError("production latent promotion requires max_steps=120")
    runtime = load_latent_runtime(args.latent_checkpoint)
    alignment_basis = None
    if args.alignment_synergy_basis is not None:
        from musclemimic.latent_muscle.synergy_decoder import load_fixed_synergy_basis

        alignment_basis = load_fixed_synergy_basis(
            args.alignment_synergy_basis,
            expected_actuator_names=runtime.body_actuator_names,
        )
    teacher_config, _teacher_state, _metadata = load_checkpoint(args.teacher_ckpt)
    OmegaConf.set_struct(teacher_config, False)
    _configure_heldout_dataset(teacher_config, args.motion_path)
    heldout_motion_paths = [
        normalize_motion_path(path) for path in resolve_config_motion_paths(teacher_config)
    ]
    if len(heldout_motion_paths) != 5:
        raise ValueError(
            "production latent promotion requires exactly five held-out motions; "
            f"resolved={len(heldout_motion_paths)}"
        )
    apply_temporal_params(teacher_config)
    env = _make_cpu_env(teacher_config)
    policy_env = apply_policy_interface_wrappers(
        env,
        teacher_config.experiment,
        include_student=False,
    )

    filter_cfg = dict(runtime.state_schema.get("student_obs_filter", {}))
    spec = build_student_obs_indices(policy_env, filter_cfg)
    live_state_schema = _student_state_schema(
        spec,
        filter_cfg,
        {"teacher_ckpt": args.teacher_ckpt},
        env=policy_env,
    )
    if live_state_schema["schema_hash"] != runtime.schema_hash:
        raise ValueError(
            "held-out environment state schema differs from latent checkpoint: "
            f"env={live_state_schema['schema_hash']} checkpoint={runtime.schema_hash}"
        )
    live_names = _resolve_actuator_names(policy_env, None)
    if live_names != list(runtime.body_actuator_names):
        raise ValueError("held-out environment ordered actuator names differ from latent checkpoint")
    live_ctrlrange = _resolve_actuator_ctrlrange(policy_env, live_names)
    if runtime.body_ctrlrange is None or not np.array_equal(live_ctrlrange, runtime.body_ctrlrange):
        raise ValueError("held-out environment ctrlrange differs from latent checkpoint")
    live_body_schema = build_body_obs_schema(
        env=policy_env,
        spec=spec,
        actuator_names=live_names,
        channels=live_state_schema["channels"],
        provenance={"teacher_ckpt": args.teacher_ckpt},
    )
    if live_body_schema["semantic_hash"] != runtime.body_obs_schema_hash:
        raise ValueError("held-out environment BodyObsSchema differs from latent checkpoint")

    rollout_metrics_path = args.direct_rollout_metrics or args.direct_bc_metrics
    if rollout_metrics_path is None:
        raise ValueError(
            "closed-loop promotion requires --direct_rollout_metrics from deterministic held-out comparison"
        )
    direct_payload = json.loads(rollout_metrics_path.read_text(encoding="utf-8"))
    direct_policy, direct_metrics = select_direct_rollout_policy(
        direct_payload,
        args.promotion_policy,
    )
    report = evaluate_latent_closed_loop(
        env=policy_env,
        runtime=runtime,
        student_obs_spec=spec,
        config=ClosedLoopEvalConfig(
            lambdas=tuple(args.lambdas),
            seed=int(args.seed),
            max_steps=args.max_steps,
            motion_paths=tuple(heldout_motion_paths),
            phase_field=args.phase_field,
            require_all_phases=bool(args.require_all_phases),
            collect_decoder_usage=bool(
                args.collect_decoder_usage or runtime.synergy_basis is not None
            ),
            collect_jacobian_alignment=bool(args.collect_jacobian_alignment),
        ),
        direct_bc_metrics=direct_metrics,
        alignment_synergy_basis=alignment_basis,
    )
    report["direct_rollout_policy"] = direct_policy
    rollout_metrics_path = rollout_metrics_path.resolve()
    evidence_path = (
        args.direct_promotion_evidence.resolve()
        if args.direct_promotion_evidence is not None
        else rollout_metrics_path.parent / "direct_promotion_evidence.json"
    )
    if not evidence_path.is_file():
        raise ValueError(
            "closed-loop promotion requires direct_promotion_evidence.json from strict direct acceptance"
        )
    training_path = Path(args.latent_checkpoint) / "training_provenance.json"
    if not training_path.is_file():
        raise ValueError("latent checkpoint is missing training_provenance.json")
    training_provenance = json.loads(training_path.read_text(encoding="utf-8"))
    saved_eval = json.loads(
        (Path(args.latent_checkpoint) / "eval_metrics.json").read_text(encoding="utf-8")
    )
    offline_eval_metrics = {}
    for key in OFFLINE_PROMOTION_METRICS:
        value = saved_eval.get(key)
        if value is None or not np.isfinite(float(value)):
            raise ValueError(f"latent checkpoint is missing finite offline promotion metric: {key}")
        offline_eval_metrics[key] = float(value)
    teacher_fingerprint = checkpoint_content_fingerprint(args.teacher_ckpt)
    report.update(
        {
            "teacher_checkpoint": teacher_fingerprint,
            "dataset_manifest_fingerprint": training_provenance.get(
                "dataset_manifest_fingerprint"
            ),
            "validation_dataset_manifest_fingerprint": training_provenance.get(
                "validation_dataset_manifest_fingerprint"
            ),
            "motion_split_fingerprint": (
                runtime.control_manifest.get("motion_split_fingerprint")
            ),
            "teacher_promotion": training_provenance.get("teacher_promotion"),
            "teacher_promotion_evidence_kind": saved_eval.get(
                "teacher_promotion_evidence_kind"
            ),
            "offline_eval_metrics": offline_eval_metrics,
            "direct_rollout_metrics": {
                "path": str(rollout_metrics_path),
                "sha256": file_sha256(rollout_metrics_path),
            },
            "direct_promotion_evidence": {
                "path": str(evidence_path),
                "sha256": file_sha256(evidence_path),
            },
        }
    )
    promotion = _merge_and_update_promotion(Path(args.latent_checkpoint), report)
    report["promotion"] = promotion
    report["report_fingerprint"] = canonical_json_sha256(report)
    validate_closed_loop_promotion_report(
        report,
        checkpoint_dir=args.latent_checkpoint,
        require_seal=True,
    )
    canonical_output = Path(args.latent_checkpoint) / "closed_loop_metrics.json"
    canonical_output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if args.output_json is not None and Path(args.output_json).resolve() != canonical_output.resolve():
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if args.require_pass and not promotion["passed"] else 0


def _configure_heldout_dataset(config, motion_paths: list[str] | None) -> None:
    if motion_paths:
        apply_collection_overrides(config, motion_path=list(motion_paths))
    else:
        validation = config.experiment.get("validation", {})
        heldout = validation.get("amass_dataset_conf")
        if heldout is None:
            raise ValueError("teacher checkpoint has no held-out validation AMASS dataset")
        config.experiment.task_factory.params.amass_dataset_conf = OmegaConf.create(
            OmegaConf.to_container(heldout, resolve=True)
        )
    env_params = config.experiment.env_params
    env_params["headless"] = True
    env_params["th_params"] = {
        **dict(env_params.get("th_params", {}) or {}),
        "random_start": False,
        "fixed_start_conf": [0, 0],
        "start_from_random_step": False,
    }


def _make_cpu_env(config):
    env_params = OmegaConf.to_container(config.experiment.env_params, resolve=True)
    env_name = str(env_params.get("env_name", ""))
    if env_name.startswith("Mjx"):
        env_params["env_name"] = env_name.removeprefix("Mjx")
    env_params.pop("num_envs", None)
    task_params = OmegaConf.to_container(config.experiment.task_factory.params, resolve=True)
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    return factory.make(**{**env_params, **task_params})


def _merge_and_update_promotion(checkpoint_dir: Path, closed_loop: dict) -> dict:
    eval_path = checkpoint_dir / "eval_metrics.json"
    config_path = checkpoint_dir / "latent_config.yaml"
    eval_metrics = json.loads(eval_path.read_text(encoding="utf-8"))
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    eval_metrics["closed_loop"] = closed_loop
    for key, value in closed_loop.items():
        array = np.asarray(value)
        if array.size == 1 and np.issubdtype(array.dtype, np.number):
            eval_metrics[f"closed_loop_{key}"] = float(array.reshape(-1)[0])
    allowed = {item.name for item in fields(LatentTrainConfig)}
    train_config = LatentTrainConfig(**{key: value for key, value in config_payload.items() if key in allowed})
    if train_config.require_closed_loop_metrics:
        validate_closed_loop_promotion_report(
            closed_loop,
            checkpoint_dir=checkpoint_dir,
            require_seal=False,
        )
        eval_metrics["closed_loop_evidence_kind"] = "verified_production_v2"
    promotion = _evaluate_promotion_gates(eval_metrics, train_config)
    eval_metrics["promotion"] = promotion
    eval_path.write_text(json.dumps(eval_metrics, indent=2, sort_keys=True), encoding="utf-8")
    return promotion


if __name__ == "__main__":
    raise SystemExit(main())

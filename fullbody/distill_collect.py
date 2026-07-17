"""Command line entrypoint for collecting teacher rollouts for distillation."""

from __future__ import annotations

import argparse

from omegaconf import OmegaConf

from loco_mujoco.task_factories import TaskFactory
from musclemimic.algorithms import PPOJax
from musclemimic.badminton.data.event_lookup import EventReferenceLookup
from musclemimic.distill.collect_teacher import collect_teacher_dataset
from musclemimic.distill.collection_budget import resolve_collection_budget
from musclemimic.distill.config_overrides import apply_collection_overrides
from musclemimic.distill.motion_identity import (
    MotionIdentityMap,
    resolve_config_motion_paths,
    validate_environment_motion_identity,
)
from musclemimic.distill.provenance import (
    begin_collection,
    canonical_json_sha256,
    checkpoint_content_fingerprint,
    validate_stage2_teacher_promotion,
)
from musclemimic.runner.eval_utils import apply_temporal_params, load_checkpoint
from musclemimic.synergy.frozen_decoder import load_frozen_body_decoder
from musclemimic.synergy.multistage_contract import (
    FIXED_SYNERGY_MODE,
    FIXED_SYNERGY_RESIDUAL_MODE,
    BodySynergyContractV2,
    canonical_action_mode,
)
from musclemimic.utils.runtime_env import reexec_with_configured_cuda_env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect teacher rollout shards for student distillation.")
    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument(
        "--frozen_body_decoder_path",
        "--frozen-body-decoder-path",
        default=None,
        help=(
            "Required for an early-synergy teacher: the exact portable decoder "
            "exported by its Stage-1 release."
        ),
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_envs", type=int, default=256)
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument(
        "--num_transitions",
        type=int,
        default=None,
        help="Exact total samples across all vector environments (default: 1,000,000).",
    )
    budget.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Legacy vector-step count; produces num_steps*num_envs samples.",
    )
    parser.add_argument("--shard_size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic_teacher", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--teacher_action_target", choices=["mean", "sample"], default="mean")
    parser.add_argument("--save_full_obs", action="store_true", default=False)
    parser.add_argument("--save_reference_features", action="store_true", default=False)
    parser.add_argument("--include_reference_phase", action="store_true", default=False)
    parser.add_argument(
        "--save_physical_muscle_state",
        "--save-physical-muscle-state",
        action="store_true",
        default=False,
        help="Persist applied ctrl, activation, force, length, velocity, power, and qfrc from the pre-reset transition.",
    )
    parser.add_argument(
        "--save_event_features",
        "--save-event-features",
        action="store_true",
        default=False,
        help="Require and persist event phase fields supplied by the environment.",
    )
    parser.add_argument(
        "--event_reference_manifest",
        "--event-reference-manifest",
        "--event-reference-bank",
        default=None,
        help=(
            "Optional forehand_clear_event_reference_bank_v1 manifest for exact host-side "
            "motion/transition-step event lookup; never falls back to linear phase."
        ),
    )
    parser.add_argument(
        "--physical_racket_site_name",
        "--physical-racket-site-name",
        default=None,
        help="Optional racket/stringbed site captured with physical transitions.",
    )
    parser.add_argument("--freeze_run_stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default=None)
    parser.add_argument("--motion_path", nargs="+", default=None)
    parser.add_argument("--motion_group", default=None)
    parser.add_argument("--traj_index", type=int, default=None)
    parser.add_argument("--traj_start_step", type=int, default=None)
    parser.add_argument(
        "--resume_dataset",
        "--resume-dataset",
        action="store_true",
        default=False,
        help="Resume only when the exact immutable dataset manifest matches.",
    )
    parser.add_argument(
        "--run_uid",
        "--run-uid",
        default=None,
        help="Stable run identity shared by train/val/DAgger collection commands.",
    )
    parser.add_argument(
        "--teacher_promotion_manifest",
        "--teacher-promotion-manifest",
        dest="teacher_promotion_manifest",
        default=None,
    )
    parser.add_argument(
        "--test_only_allow_unpromoted_teacher",
        "--test-only-allow-unpromoted-teacher",
        dest="test_only_allow_unpromoted_teacher",
        action="store_true",
        default=False,
    )
    return parser


def main() -> int:
    reexec_with_configured_cuda_env()
    parser = build_parser()
    args = parser.parse_args()
    collection_budget = resolve_collection_budget(
        num_envs=args.num_envs,
        num_transitions=args.num_transitions,
        num_steps=args.num_steps,
        default_transitions=1_000_000,
    )
    print(
        "[distill_collect] budget "
        f"transitions={collection_budget.requested_transitions} "
        f"vector_steps={collection_budget.vector_steps} num_envs={collection_budget.num_envs} "
        f"pretrim={collection_budget.planned_transitions_before_trim}"
    )
    deterministic_teacher = bool(args.deterministic_teacher)
    if args.teacher_action_target == "sample":
        deterministic_teacher = False
    elif args.teacher_action_target == "mean" and not deterministic_teacher:
        parser.error("--teacher_action_target=mean conflicts with --no-deterministic_teacher")

    config, agent_state, metadata = load_checkpoint(args.teacher_ckpt)
    OmegaConf.set_struct(config, False)
    config.experiment.env_params["headless"] = True
    config.experiment.env_params["num_envs"] = int(args.num_envs)
    apply_collection_overrides(
        config,
        motion_path=args.motion_path,
        motion_group=args.motion_group,
        traj_index=args.traj_index,
        traj_start_step=args.traj_start_step,
    )
    action_mode = canonical_action_mode(
        config.experiment.get("action_representation", {}) or {}
    )
    frozen_body_decoder = None
    collection_body_synergy_contract = None
    if action_mode in {FIXED_SYNERGY_MODE, FIXED_SYNERGY_RESIDUAL_MODE}:
        if args.frozen_body_decoder_path is None:
            parser.error(
                "early-synergy teacher collection requires "
                "--frozen-body-decoder-path from the Stage-1 release"
            )
        frozen_body_decoder = load_frozen_body_decoder(
            args.frozen_body_decoder_path
        )
        checkpoint_contract_payload = config.experiment.get(
            "body_synergy_contract", None
        )
        if checkpoint_contract_payload is None:
            raise ValueError(
                "early-synergy teacher checkpoint lacks BodySynergyContractV2"
            )
        if OmegaConf.is_config(checkpoint_contract_payload):
            checkpoint_contract_payload = OmegaConf.to_container(
                checkpoint_contract_payload, resolve=True
            )
        checkpoint_contract = BodySynergyContractV2.from_manifest(
            checkpoint_contract_payload
        )
        checkpoint_contract.assert_portable_compatible(
            frozen_body_decoder.body_synergy_contract
        )
        # The dataset records the teacher's current Stage-2 runtime contract;
        # the Stage-1 artifact is accepted at portable-core compatibility.
        collection_body_synergy_contract = checkpoint_contract
    elif args.frozen_body_decoder_path is not None:
        parser.error(
            "--frozen-body-decoder-path is valid only for an early-synergy teacher"
        )
    control_dt = apply_temporal_params(config)
    motion_identity_map = MotionIdentityMap.from_paths(resolve_config_motion_paths(config))
    event_reference_lookup = (
        None
        if args.event_reference_manifest is None
        else EventReferenceLookup.from_manifest(
            args.event_reference_manifest,
            motion_identity_map=motion_identity_map,
        )
    )
    if event_reference_lookup is not None:
        event_reference_lookup.validate_control_dt(control_dt)

    teacher_fingerprint = checkpoint_content_fingerprint(args.teacher_ckpt)
    teacher_promotion = (
        None
        if args.teacher_promotion_manifest is None
        else validate_stage2_teacher_promotion(
            args.teacher_promotion_manifest,
            teacher_checkpoint=teacher_fingerprint,
        )
    )
    transaction = begin_collection(
        dataset_dir=args.output_dir,
        teacher_checkpoint=teacher_fingerprint,
        collector="teacher_lookahead_rollout",
        split=args.split,
        seed=int(args.seed),
        motion_paths=motion_identity_map.motion_paths,
        config_payload=OmegaConf.to_container(config, resolve=True),
        request_payload={
            "num_envs": int(args.num_envs),
            "num_transitions": int(collection_budget.requested_transitions),
            "shard_size": int(args.shard_size),
            "deterministic_teacher": bool(deterministic_teacher),
            "teacher_action_target": str(args.teacher_action_target),
            "save_full_obs": bool(args.save_full_obs),
            "save_reference_features": bool(args.save_reference_features),
            "include_reference_phase": bool(args.include_reference_phase),
            "save_physical_muscle_state": bool(args.save_physical_muscle_state),
            "save_event_features": bool(args.save_event_features),
            "event_reference_manifest": args.event_reference_manifest,
            "event_reference_bank_fingerprint": (
                None if event_reference_lookup is None else event_reference_lookup.fingerprint
            ),
            "event_reference_control_dt": (None if event_reference_lookup is None else float(control_dt)),
            "event_reference_bundle_fingerprints": (
                None
                if event_reference_lookup is None
                else [
                    entry.reference_bundle_content_fingerprint
                    for entry in sorted(event_reference_lookup.entries, key=lambda value: value.traj_no)
                ]
            ),
            "event_reference_bank_motion_uids": (
                None
                if event_reference_lookup is None
                else [
                    int(entry.motion_uid)
                    for entry in sorted(event_reference_lookup.entries, key=lambda value: value.traj_no)
                ]
            ),
            "physical_racket_site_name": args.physical_racket_site_name,
            "freeze_run_stats": bool(args.freeze_run_stats),
        },
        resume=bool(args.resume_dataset),
        run_uid=args.run_uid,
        teacher_promotion=teacher_promotion,
        allow_test_only_unpromoted_teacher=bool(args.test_only_allow_unpromoted_teacher),
        body_synergy_contract=(
            None
            if collection_body_synergy_contract is None
            else collection_body_synergy_contract.to_manifest()
        ),
        frozen_body_decoder_fingerprint=(
            None
            if frozen_body_decoder is None
            else frozen_body_decoder.artifact_fingerprint
        ),
    )
    if transaction.already_complete:
        print(f"[distill_collect] idempotent collection already complete: {transaction.collection_id}")
        for shard in transaction.existing_paths:
            print(shard)
        return 0

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env = factory.make(
        **OmegaConf.to_container(config.experiment.env_params, resolve=True),
        **OmegaConf.to_container(config.experiment.task_factory.params, resolve=True),
    )
    validate_environment_motion_identity(env, motion_identity_map)
    if event_reference_lookup is not None:
        event_reference_lookup.validate_control_dt(float(env.dt))
    if getattr(env, "mjx_enabled", False) and getattr(env, "th", None) is not None and env.th.is_numpy:
        env.th.to_jax()

    agent_conf = PPOJax.init_agent_conf(env, config)
    shards = collect_teacher_dataset(
        env=env,
        agent_conf=agent_conf,
        agent_state=agent_state,
        output_dir=transaction.output_dir,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_transitions=args.num_transitions,
        shard_size=args.shard_size,
        deterministic_teacher=deterministic_teacher,
        seed=args.seed,
        save_full_obs=bool(args.save_full_obs),
        save_reference_features=bool(args.save_reference_features),
        include_reference_phase=bool(args.include_reference_phase),
        save_physical_muscle_state=bool(args.save_physical_muscle_state),
        save_event_features=bool(args.save_event_features),
        event_reference_manifest=args.event_reference_manifest,
        physical_racket_site_name=args.physical_racket_site_name,
        freeze_run_stats=bool(args.freeze_run_stats),
        split=args.split,
        metadata={
            "teacher_ckpt": args.teacher_ckpt,
            # Keep the compact identity and the auditable inventory separate.
            # Downstream physical QC/NMF artifacts require a canonical SHA-256,
            # while the immutable dataset manifest retains the full record.
            "teacher_checkpoint_fingerprint": teacher_fingerprint["sha256"],
            "teacher_checkpoint_content": teacher_fingerprint,
            "teacher_promotion": transaction.manifest["teacher_promotion"],
            "distill_run_uid": transaction.manifest["run_uid"],
            "collection_contract_fingerprint": canonical_json_sha256(transaction.contract),
            "teacher_checkpoint_step": int(getattr(metadata, "step", 0) or 0),
            "teacher_action_target": "mean" if deterministic_teacher else "sample",
            "teacher_config": OmegaConf.to_container(config, resolve=True),
        },
        motion_identity_map=motion_identity_map,
    )
    shards = transaction.commit(shards)
    for shard in shards:
        print(shard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

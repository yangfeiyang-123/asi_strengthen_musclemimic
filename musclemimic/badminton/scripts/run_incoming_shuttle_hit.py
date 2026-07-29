#!/usr/bin/env python3
"""Run incoming-shuttle Stage-3 preflight, LAB training and held-out evaluation.

The environment owns its MuJoCo scene and physics loop, but production training
is downstream of the promoted Stage-2 latent skill and requires bound preflight,
base-only and feed-check evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class IncomingHitPaths:
    spec_path: Path
    scene_xml: Path
    build_if_missing: bool
    human_root_xy: tuple[float, float]
    feed_bank_path: Path
    feed_bank_size: int
    feed_seed: int
    eval_feed_bank_path: Path
    eval_feed_bank_size: int
    eval_feed_seed: int
    feed_kwargs: dict[str, Any]
    hit_window_kwargs: dict[str, Any]
    control_substeps: int
    max_episode_steps: int
    reward_weights: dict[str, float]
    ppo_overrides: dict[str, Any]
    stage3_lab: dict[str, Any]
    evaluation: dict[str, Any]
    task_profile: str
    target_bank_path: Path | None
    eval_target_bank_path: Path | None
    recovery_horizon_steps: int
    output_dir: Path


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def load_incoming_hit_spec(spec_path: str | Path) -> IncomingHitPaths:
    resolved_spec = _resolve(spec_path)
    data = yaml.safe_load(resolved_spec.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{resolved_spec} must contain a mapping")
    if data.get("runner_type") != "incoming_shuttle_hit":
        raise ValueError(f"unsupported runner_type: {data.get('runner_type')!r}")

    scene = data.get("scene", {})
    feed = dict(data.get("feed", {}))
    eval_feed = dict(data.get("eval_feed", {}))
    window = dict(data.get("hit_window", {}))
    episode = dict(data.get("episode", {}))
    reward = dict(data.get("reward", {}))
    ppo = dict(data.get("ppo", {}))
    stage3_lab = dict(data.get("stage3_lab", {}))
    evaluation = dict(data.get("evaluation", {}))
    stage3_v2 = dict(data.get("stage3_v2", {}) or {})
    for section, name in ((scene, "scene"), (feed, "feed"), (episode, "episode")):
        if not isinstance(section, dict):
            raise ValueError(f"{name} must contain a mapping")

    output_dir = _resolve(data.get("output_root", "outputs/posttrain")) / data["action"] / data["experiment_id"]
    human_root_xy = tuple(float(v) for v in scene.get("human_root_xy", (-3.35, 0.0)))

    feed_bank_path = _resolve(feed.pop("bank_path", "outputs/incoming_shuttle_hit/feed_bank.npz"))
    feed_bank_size = int(feed.pop("bank_size", 512))
    feed_seed = int(feed.pop("seed", 7))
    eval_feed_bank_path = _resolve(eval_feed.pop("bank_path", str(feed_bank_path) + ".eval.npz"))
    eval_feed_bank_size = int(eval_feed.pop("bank_size", 128))
    eval_feed_seed = int(eval_feed.pop("seed", feed_seed + 1000))

    feed_kwargs = {key: tuple(value) if isinstance(value, list) else value for key, value in feed.items()}
    window_kwargs = {key: tuple(value) if isinstance(value, list) else value for key, value in window.items()}

    return IncomingHitPaths(
        spec_path=resolved_spec,
        scene_xml=_resolve(scene["xml"]),
        build_if_missing=bool(scene.get("build_if_missing", True)),
        human_root_xy=human_root_xy,
        feed_bank_path=feed_bank_path,
        feed_bank_size=feed_bank_size,
        feed_seed=feed_seed,
        eval_feed_bank_path=eval_feed_bank_path,
        eval_feed_bank_size=eval_feed_bank_size,
        eval_feed_seed=eval_feed_seed,
        feed_kwargs=feed_kwargs,
        hit_window_kwargs=window_kwargs,
        control_substeps=int(episode.get("control_substeps", 10)),
        max_episode_steps=int(episode.get("max_episode_steps", 300)),
        reward_weights={str(key): float(value) for key, value in reward.items()},
        ppo_overrides=ppo,
        stage3_lab=stage3_lab,
        evaluation=evaluation,
        task_profile=str(stage3_v2.get("profile", "legacy_v1")),
        target_bank_path=(None if not stage3_v2.get("target_bank_path") else _resolve(stage3_v2["target_bank_path"])),
        eval_target_bank_path=(
            None if not stage3_v2.get("eval_target_bank_path") else _resolve(stage3_v2["eval_target_bank_path"])
        ),
        recovery_horizon_steps=int(stage3_v2.get("recovery_horizon_steps", 60)),
        output_dir=output_dir,
    )


@dataclass(frozen=True)
class Stage3LabComponents:
    controller: Any
    state_builder: Any
    curriculum: Any
    latent_checkpoint_dir: Path


@dataclass(frozen=True)
class FeedBankArtifact:
    bank: list[Any]
    manifest: dict[str, Any]


def _validate_checkpoint_latent_dim(config: dict[str, Any], runtime_latent_dim: int) -> int:
    """Keep checkpoint metadata as the latent-dimension source of truth.

    ``expected_latent_dim`` is an optional pre-registration assertion.  The
    legacy ``latent_dim`` key remains supported for old specs, but a research
    spec may leave the expectation null so a dimension-sweep selection can be
    attached without generating a second, inconsistent configuration.
    """

    actual = int(runtime_latent_dim)
    if actual <= 0:
        raise ValueError("latent checkpoint runtime has a non-positive latent_dim")
    configured = config.get("expected_latent_dim") if "expected_latent_dim" in config else config.get("latent_dim")
    if configured is None:
        return actual
    if isinstance(configured, bool):
        raise ValueError("Stage-3 expected_latent_dim must be a positive integer or null")
    try:
        expected = int(configured)
        numeric = float(configured)
    except (TypeError, ValueError) as exc:
        raise ValueError("Stage-3 expected_latent_dim must be a positive integer or null") from exc
    if expected <= 0 or numeric != float(expected):
        raise ValueError("Stage-3 expected_latent_dim must be a positive integer or null")
    if actual != expected:
        raise ValueError(f"Stage-3 expected_latent_dim={expected} != checkpoint={actual}")
    return actual


def _build_stage3_lab_components(
    paths: IncomingHitPaths,
    *,
    latent_checkpoint: str | Path | None = None,
    lambda_lab: float | None = None,
    allow_unpromoted: bool = False,
) -> Stage3LabComponents | None:
    """Build the one shared LAB/runtime/router contract used by train and eval."""
    config = dict(paths.stage3_lab)
    if not bool(config.get("enabled", False)):
        return None
    checkpoint_value = latent_checkpoint if latent_checkpoint is not None else config.get("latent_checkpoint_dir")
    if not checkpoint_value:
        raise ValueError("stage3_lab.latent_checkpoint_dir is required when LAB is enabled")
    checkpoint_dir = _resolve(checkpoint_value)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Stage-3 latent checkpoint directory not found: {checkpoint_dir}")

    from musclemimic.latent_muscle.checkpoint import load_latent_checkpoint
    from musclemimic.latent_muscle.runtime import load_latent_runtime

    checkpoint = load_latent_checkpoint(checkpoint_dir)
    promotion = dict(checkpoint.get("eval_metrics", {}).get("promotion", {}) or {})
    permit_unpromoted = bool(config.get("allow_unpromoted_for_testing", False)) or bool(allow_unpromoted)
    if promotion.get("passed") is not True and not permit_unpromoted:
        raise ValueError(
            "Stage-3 LAB refuses an unpromoted latent checkpoint; eval_metrics.promotion.passed must be true"
        )

    import mujoco

    from environment.overall_environment.src.stage3_lab import (
        DEFAULT_RIGHT_WRIST_FOREARM_RESIDUAL_NAMES,
        BoundedResidualMask,
        Stage3ActionRouter,
        Stage3Curriculum,
        Stage3LABController,
        Stage3LabStateBuilder,
    )

    model = mujoco.MjModel.from_xml_path(str(paths.scene_xml))
    _validate_stage3_mainline_scene(model=model, paths=paths, config=config)
    router = Stage3ActionRouter.from_model(model)
    if router.fixture_mode != "rigid_tool_fingerless":
        raise ValueError(
            f"Stage-3 production requires the 354-D fingerless rigid-tool fixture; got {router.fixture_mode!r}"
        )
    runtime = load_latent_runtime(
        checkpoint_dir,
        runtime_body_actuator_names=router.body_actuator_names,
    )
    if len(str(getattr(runtime, "checkpoint_fingerprint", ""))) != 64:
        raise ValueError("latent runtime is missing its content checkpoint fingerprint")
    if getattr(runtime, "body_obs_schema", None) is None:
        raise ValueError("latent runtime is missing the self-contained body observation schema")
    if getattr(runtime, "body_ctrlrange", None) is None:
        raise ValueError("latent runtime is missing the ordered Stage-2 teacher ctrlrange")
    _validate_checkpoint_latent_dim(config, runtime.latent_dim)
    runtime.checkpoint_dir = str(checkpoint_dir)
    fixture = dict(config.get("hand_fixture", {}) or {})
    if str(fixture.get("mode", "")) != "removed":
        raise ValueError("Stage-3 production requires hand_fixture.mode=removed")
    if bool(fixture.get("policy_enabled", False)):
        raise ValueError("fingerless rigid-tool Stage-3 cannot enable a hand policy")
    if bool(fixture.get("observations_enabled", False)):
        raise ValueError("fingerless rigid-tool Stage-3 cannot expose hand observations")
    if config.get("right_grip") is not None or config.get("left_neutral_value") is not None:
        raise ValueError("fingerless rigid-tool Stage-3 must not configure hand providers")
    residual_config = dict(config.get("bounded_residual", {}) or {})
    bounded_residual = None
    if bool(residual_config.get("enabled", False)):
        residual_names = tuple(residual_config.get("actuator_names", DEFAULT_RIGHT_WRIST_FOREARM_RESIDUAL_NAMES))
        if residual_names != DEFAULT_RIGHT_WRIST_FOREARM_RESIDUAL_NAMES:
            raise ValueError("production bounded residual must use the exact 10-D right wrist/forearm actuator list")
        bounded_residual = BoundedResidualMask(
            body_actuator_names=router.body_actuator_names,
            residual_actuator_names=residual_names,
            alpha=float(residual_config.get("alpha", 0.05)),
        )

    curriculum_cfg = dict(config.get("curriculum", {}) or {})
    curriculum = Stage3Curriculum(
        lambda_start=float(curriculum_cfg.get("lambda_start", 0.25)),
        lambda_end=float(curriculum_cfg.get("lambda_end", 0.5)),
        fixed_feed_steps=int(curriculum_cfg.get("fixed_feed_steps", 2_000_000)),
        jitter_feed_count=int(curriculum_cfg.get("jitter_feed_count", 16)),
        jitter_expand_steps=int(curriculum_cfg.get("jitter_expand_steps", 4_000_000)),
        full_bank_expand_steps=int(curriculum_cfg.get("full_bank_expand_steps", 8_000_000)),
        lambda_expand_steps=int(curriculum_cfg.get("lambda_expand_steps", 4_000_000)),
        gate_min_no_fall_rate=float(curriculum_cfg.get("gate_min_no_fall_rate", 0.95)),
        fixed_min_hit_rate=float(curriculum_cfg.get("fixed_min_hit_rate", 0.50)),
        jitter_min_hit_rate=float(curriculum_cfg.get("jitter_min_hit_rate", 0.70)),
        jitter_min_crossed_net_rate=float(curriculum_cfg.get("jitter_min_crossed_net_rate", 0.50)),
        full_bank_min_hit_rate=float(curriculum_cfg.get("full_bank_min_hit_rate", 0.85)),
        full_bank_min_crossed_net_rate=float(curriculum_cfg.get("full_bank_min_crossed_net_rate", 0.75)),
    )
    controller = Stage3LABController(
        runtime=runtime,
        router=router,
        lambda_lab=(float(lambda_lab) if lambda_lab is not None else float(curriculum.lambda_start)),
        sigma_min=float(config.get("sigma_min", runtime.sigma_min)),
        sigma_max=float(config.get("sigma_max", runtime.sigma_max)),
        bounded_residual_mask=bounded_residual,
    )
    state_builder = Stage3LabStateBuilder.from_runtime(model=model, runtime=runtime)
    return Stage3LabComponents(controller, state_builder, curriculum, checkpoint_dir)


def _validate_stage3_mainline_scene(*, model: Any, paths: IncomingHitPaths, config: dict[str, Any]) -> None:
    """Fail if production drifts from the 354-D exact-child fixture."""
    import mujoco

    from environment.overall_environment.src.stage3_lab import (
        Stage3ActionRouter,
        stage3_attachment_report,
    )
    from musclemimic.utils.finger_isolation import finger_joint_side

    router = Stage3ActionRouter.from_model(model)
    if router.fixture_mode != "rigid_tool_fingerless" or router.expected_sizes != (354, 0, 0):
        raise ValueError("Stage-3 production requires the exact 354+0+0 actuator partition")
    if bool(config.get("filter_finger_observation", False)):
        raise ValueError("fingerless Stage-3 must not rely on an observation-only finger filter")
    attachment_config = dict(config.get("racket_attachment", {}) or {})
    if str(attachment_config.get("mode", "")) != "exact_child":
        raise ValueError("Stage-3 production requires racket_attachment.mode=exact_child")
    if bool(attachment_config.get("hand_racket_contact_enabled", False)):
        raise ValueError("Stage-3 production requires hand-racket contact to remain disabled")
    contract_value = attachment_config.get("contract_path")
    if not contract_value:
        raise ValueError("Stage-3 exact-child fixture requires racket_attachment.contract_path")
    attachment_report = stage3_attachment_report(
        model,
        paths.scene_xml,
        contract_path=_resolve(contract_value),
    )
    if attachment_report["contract_passed"] is not True:
        failed = sorted(name for name, passed in attachment_report["contract_checks"].items() if passed is not True)
        raise ValueError(f"Stage-3 exact-child attachment contract failed: {failed}")

    finger_joints = []
    for joint_id in range(int(model.njnt)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if finger_joint_side(name) is not None:
            finger_joints.append(name)
    if finger_joints:
        raise ValueError(f"fingerless Stage-3 still contains finger joints: {finger_joints}")


def _feed_config(paths: IncomingHitPaths):
    from environment.overall_environment.src.shuttle_feeder import FeedConfig

    return FeedConfig(**paths.feed_kwargs)


def _hit_window(paths: IncomingHitPaths):
    from environment.overall_environment.src.shuttle_feeder import HitWindow

    return HitWindow(**paths.hit_window_kwargs)


def _ensure_scene(paths: IncomingHitPaths) -> None:
    import mujoco

    validation_error: Exception | None = None
    if paths.scene_xml.is_file():
        try:
            existing_model = mujoco.MjModel.from_xml_path(str(paths.scene_xml))
            _validate_stage3_mainline_scene(
                model=existing_model,
                paths=paths,
                config=dict(paths.stage3_lab),
            )
            return
        except (OSError, RuntimeError, ValueError) as exc:
            validation_error = exc
            if not paths.build_if_missing:
                raise ValueError(f"existing Stage-3 scene violates the production contract: {paths.scene_xml}") from exc
    elif not paths.build_if_missing:
        raise FileNotFoundError(f"scene XML missing and build_if_missing is false: {paths.scene_xml}")

    from environment.overall_environment.src.incoming_scene import build_incoming_hit_scene

    attachment = dict(paths.stage3_lab.get("racket_attachment", {}) or {})
    contract_value = attachment.get("contract_path")
    if not contract_value:
        raise ValueError("Stage-3 scene build requires racket_attachment.contract_path")
    build_incoming_hit_scene(
        paths.scene_xml,
        human_root_xy=paths.human_root_xy,
        racket_attachment_contract=_resolve(contract_value),
    )
    rebuilt_model = mujoco.MjModel.from_xml_path(str(paths.scene_xml))
    try:
        _validate_stage3_mainline_scene(
            model=rebuilt_model,
            paths=paths,
            config=dict(paths.stage3_lab),
        )
    except (RuntimeError, ValueError) as exc:
        context = "" if validation_error is None else f"; stale-scene reason was: {validation_error}"
        raise ValueError(f"rebuilt Stage-3 scene still violates the production contract{context}") from exc


def _ensure_feed_bank_artifact(paths: IncomingHitPaths, *, evaluation: bool = False) -> FeedBankArtifact:
    from environment.overall_environment.src.shuttle_feeder import (
        FeedBankValidationError,
        build_feed_bank,
        feed_bank_contract,
        load_feed_bank_with_manifest,
        save_feed_bank,
    )

    bank_path = paths.eval_feed_bank_path if evaluation else paths.feed_bank_path
    bank_size = paths.eval_feed_bank_size if evaluation else paths.feed_bank_size
    seed = paths.eval_feed_seed if evaluation else paths.feed_seed
    feed_config = _feed_config(paths)
    hit_window = _hit_window(paths)
    expected_contract = feed_bank_contract(
        seed=seed,
        sample_count=bank_size,
        cfg=feed_config,
        window=hit_window,
    )
    try:
        bank, manifest = load_feed_bank_with_manifest(bank_path, expected_contract=expected_contract)
    except (FeedBankValidationError, OSError):
        # Old files without provenance, config/seed drift and any physical or
        # semantic hash mismatch are never reused.  Generation is deterministic
        # for the exact contract, so rebuilding is safe and reproducible.
        bank = build_feed_bank(bank_size, seed, feed_config, hit_window)
        save_feed_bank(
            bank_path,
            bank,
            seed=seed,
            cfg=feed_config,
            window=hit_window,
        )
        bank, manifest = load_feed_bank_with_manifest(bank_path, expected_contract=expected_contract)
    if len(bank) != bank_size:
        raise RuntimeError(f"feed bank exact-count invariant failed: {len(bank)} != {bank_size}")
    return FeedBankArtifact(bank=bank, manifest=manifest)


def _ensure_feed_bank(paths: IncomingHitPaths, *, evaluation: bool = False) -> list[Any]:
    return _ensure_feed_bank_artifact(paths, evaluation=evaluation).bank


def _make_env(paths: IncomingHitPaths, *, feed_bank: list[Any] | None, seed: int = 0, **overrides: Any):
    from environment.overall_environment.src.incoming_shuttle_hit_env import IncomingShuttleHitEnv

    kwargs: dict[str, Any] = {
        "feed_bank": feed_bank,
        "feed_config": _feed_config(paths),
        "hit_window": _hit_window(paths),
        "control_substeps": paths.control_substeps,
        "max_episode_steps": paths.max_episode_steps,
        "reward_weights": paths.reward_weights,
        "task_profile": paths.task_profile,
        "impact_target_bank": paths.target_bank_path,
        "recovery_horizon_steps": paths.recovery_horizon_steps,
        "seed": seed,
    }
    kwargs.update(overrides)
    return IncomingShuttleHitEnv(paths.scene_xml, **kwargs)


def preflight(paths: IncomingHitPaths, *, out_dir: str | Path | None = None) -> dict[str, Any]:
    import mujoco

    out_path = Path(out_dir) if out_dir is not None else paths.output_dir / "preflight"
    out_path.mkdir(parents=True, exist_ok=True)
    _ensure_scene(paths)

    model = mujoco.MjModel.from_xml_path(str(paths.scene_xml))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "overall_ready")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    from environment.overall_environment.src.stage3_lab import (
        Stage3ActionRouter,
        stage3_attachment_report,
    )
    from musclemimic.utils.finger_isolation import (
        finger_actuator_side,
        finger_joint_side,
    )

    action_router = Stage3ActionRouter.from_model(model)
    attachment_config = dict(paths.stage3_lab.get("racket_attachment", {}) or {})
    contract_value = attachment_config.get("contract_path")
    if not contract_value:
        raise ValueError("Stage-3 preflight requires racket_attachment.contract_path")
    attachment_report = stage3_attachment_report(
        model,
        paths.scene_xml,
        contract_path=_resolve(contract_value),
    )
    root_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    root_adr = int(model.jnt_qposadr[root_joint])
    required_sites = ["overall_stringbed_center_site", "overall_cork_contact_site", "rh_palm_grip_site"]
    missing_sites = [name for name in required_sites if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) < 0]
    finger_joint_names = [
        name
        for joint_id in range(int(model.njnt))
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
        and finger_joint_side(name) is not None
    ]
    finger_actuator_names = [
        name
        for actuator_id in range(int(model.nu))
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id))
        and finger_actuator_side(name) is not None
    ]
    fixture_config = dict(paths.stage3_lab.get("hand_fixture", {}) or {})
    configuration_contract_passed = bool(
        str(attachment_config.get("mode", "")) == "exact_child"
        and attachment_config.get("hand_racket_contact_enabled", False) is False
        and str(fixture_config.get("mode", "")) == "removed"
        and fixture_config.get("policy_enabled", False) is False
        and fixture_config.get("observations_enabled", False) is False
        and paths.stage3_lab.get("right_grip") is None
        and paths.stage3_lab.get("left_neutral_value") is None
        and bool(paths.stage3_lab.get("filter_finger_observation", False)) is False
    )

    report = {
        "runner_type": "incoming_shuttle_hit",
        "spec_path": str(paths.spec_path),
        "scene_xml": str(paths.scene_xml),
        "scene_exists": paths.scene_xml.is_file(),
        "output_dir": str(out_path),
        "keyframe_found": key_id >= 0,
        "racket_attachment": attachment_report,
        "attachment_contract_passed": attachment_report["contract_passed"],
        "configuration_contract_passed": configuration_contract_passed,
        "root_pos": [float(v) for v in data.qpos[root_adr : root_adr + 3]],
        "expected_root_xy": list(paths.human_root_xy),
        "missing_sites": missing_sites,
        "actuator_count": int(model.nu),
        "finger_joint_count": len(finger_joint_names),
        "finger_actuator_count": len(finger_actuator_names),
        "finger_joint_names": finger_joint_names,
        "finger_actuator_names": finger_actuator_names,
        "action_router": action_router.manifest(),
        "timestep_s": float(model.opt.timestep),
        "reward_weights": paths.reward_weights,
        "task_profile": paths.task_profile,
        "target_bank_path": (None if paths.target_bank_path is None else str(paths.target_bank_path)),
        "feed_bank_path": str(paths.feed_bank_path),
        "eval_feed_bank_path": str(paths.eval_feed_bank_path),
    }
    report["passed"] = bool(
        report["scene_exists"]
        and report["keyframe_found"]
        and report["attachment_contract_passed"]
        and report["configuration_contract_passed"]
        and list(action_router.expected_sizes) == [354, 0, 0]
        and not finger_joint_names
        and not finger_actuator_names
        and not missing_sites
        and abs(report["root_pos"][0] - paths.human_root_xy[0]) < 1e-6
    )
    (out_path / "preflight_report.json").write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def feed_check(paths: IncomingHitPaths, *, out_dir: str | Path | None = None) -> dict[str, Any]:
    out_path = Path(out_dir) if out_dir is not None else paths.output_dir / "feed_check"
    out_path.mkdir(parents=True, exist_ok=True)
    window = _hit_window(paths)

    report: dict[str, Any] = {"runner_stage": "feed-check"}
    manifests: dict[str, dict[str, Any]] = {}
    for label, evaluation in (("train", False), ("eval", True)):
        artifact = _ensure_feed_bank_artifact(paths, evaluation=evaluation)
        bank = artifact.bank
        manifest = artifact.manifest
        manifests[label] = manifest
        points = np.stack([sample.intercept_point for sample in bank])
        times = np.array([sample.intercept_time_s for sample in bank])
        speeds = np.array([np.linalg.norm(sample.intercept_velocity) for sample in bank])
        inside = window.contains(points)
        expected_count = paths.eval_feed_bank_size if evaluation else paths.feed_bank_size
        fingerprints = tuple(str(value) for value in manifest["sample_fingerprints"])
        unique_count = len(set(fingerprints))
        report[label] = {
            "bank_size": len(bank),
            "expected_bank_size": int(expected_count),
            "exact_count": len(bank) == int(expected_count),
            "unique_sample_count": unique_count,
            "all_samples_unique": unique_count == len(bank),
            "all_in_window": bool(inside.all()),
            "intercept_point_mean": points.mean(axis=0).tolist(),
            "intercept_point_min": points.min(axis=0).tolist(),
            "intercept_point_max": points.max(axis=0).tolist(),
            "intercept_time_mean_s": float(times.mean()),
            "intercept_time_range_s": [float(times.min()), float(times.max())],
            "intercept_speed_mean_m_s": float(speeds.mean()),
            "intercept_speed_range_m_s": [float(speeds.min()), float(speeds.max())],
            "manifest": manifest,
        }
    identity = _feed_bank_identity_qc(
        manifests["train"],
        manifests["eval"],
        paths_distinct=paths.feed_bank_path.resolve() != paths.eval_feed_bank_path.resolve(),
    )
    report.update(identity)
    report["passed"] = bool(
        report["train"]["exact_count"]
        and report["eval"]["exact_count"]
        and report["train"]["all_samples_unique"]
        and report["eval"]["all_samples_unique"]
        and report["train"]["all_in_window"]
        and report["eval"]["all_in_window"]
        and identity["bank_paths_distinct"]
        and identity["train_eval_fingerprint_overlap_count"] == 0
    )
    (out_path / "feed_check_report.json").write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def _feed_bank_identity_qc(
    train_manifest: dict[str, Any],
    eval_manifest: dict[str, Any],
    *,
    paths_distinct: bool = True,
) -> dict[str, Any]:
    """Return fail-closed duplicate/leakage evidence from persisted identities."""
    train_values = train_manifest.get("sample_fingerprints")
    eval_values = eval_manifest.get("sample_fingerprints")
    if not isinstance(train_values, list) or not isinstance(eval_values, list):
        raise ValueError("feed manifests must contain sample_fingerprints lists")
    train = [str(value) for value in train_values]
    evaluation = [str(value) for value in eval_values]
    overlap = sorted(set(train) & set(evaluation))
    return {
        "bank_paths_distinct": bool(paths_distinct),
        "train_unique_sample_count": len(set(train)),
        "eval_unique_sample_count": len(set(evaluation)),
        "train_duplicate_count": len(train) - len(set(train)),
        "eval_duplicate_count": len(evaluation) - len(set(evaluation)),
        "train_eval_fingerprint_overlap_count": len(overlap),
        "train_eval_fingerprint_overlap": overlap,
    }


def _site_relative_transform(env: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return the stringbed pose expressed in the right-palm site frame."""
    palm_pos = np.asarray(env.data.site_xpos[env._palm_site], dtype=float)
    palm_rot = np.asarray(env.data.site_xmat[env._palm_site], dtype=float).reshape(3, 3)
    racket_pos = np.asarray(env.data.site_xpos[env._stringbed_site], dtype=float)
    racket_rot = np.asarray(env.data.site_xmat[env._stringbed_site], dtype=float).reshape(3, 3)
    return palm_rot.T @ (racket_pos - palm_pos), palm_rot.T @ racket_rot


def _rotation_distance_rad(first: np.ndarray, second: np.ndarray) -> float:
    delta = np.asarray(first, dtype=float).T @ np.asarray(second, dtype=float)
    cosine = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(cosine))


def _base_only_summary(
    episodes: list[dict[str, Any]],
    *,
    thresholds: dict[str, float],
    required_steps: int,
) -> dict[str, Any]:
    """Aggregate the fail-closed Stage-3 prior-mean stability protocol."""
    if not episodes:
        raise ValueError("base-only check requires at least one rollout")

    def finite_values(name: str) -> list[float]:
        values: list[float] = []
        for episode in episodes:
            try:
                value = float(episode[name])
            except (KeyError, TypeError, ValueError):
                return []
            if not math.isfinite(value):
                return []
            values.append(value)
        return values

    completion_rate = float(
        np.mean([int(episode.get("completed_steps", -1)) >= int(required_steps) for episode in episodes])
    )
    finite_rate = float(np.mean([bool(episode.get("finite", False)) for episode in episodes]))
    no_fall_rate = float(np.mean([not bool(episode.get("body_fall", True)) for episode in episodes]))
    aggregations = {
        "min_root_height_m": (finite_values("min_root_height_m"), min, float("-inf")),
        "max_body_action_saturation_fraction": (
            finite_values("max_body_action_saturation_fraction"),
            max,
            float("inf"),
        ),
        "max_full_action_saturation_fraction": (
            finite_values("max_full_action_saturation_fraction"),
            max,
            float("inf"),
        ),
        "max_normalized_control_energy": (
            finite_values("max_normalized_control_energy"),
            max,
            float("inf"),
        ),
        "max_lab_state_ood_fraction": (
            finite_values("max_lab_state_ood_fraction"),
            max,
            float("inf"),
        ),
        "min_control_finite": (
            finite_values("min_control_finite"),
            min,
            float("-inf"),
        ),
        "max_attachment_translation_drift_m": (
            finite_values("max_attachment_translation_drift_m"),
            max,
            float("inf"),
        ),
        "max_attachment_rotation_drift_rad": (
            finite_values("max_attachment_rotation_drift_rad"),
            max,
            float("inf"),
        ),
    }
    metrics: dict[str, float] = {
        "rollout_count": float(len(episodes)),
        "completion_rate": completion_rate,
        "finite_rate": finite_rate,
        "no_fall_rate": no_fall_rate,
    }
    for name, (values, reducer, default) in aggregations.items():
        metrics[name] = float(reducer(values)) if len(values) == len(episodes) else default

    gates = {
        "rollout_count": len(episodes) >= int(thresholds["min_rollout_count"]),
        "completion_rate": completion_rate >= float(thresholds["min_completion_rate"]),
        "finite_rate": finite_rate >= float(thresholds["min_finite_rate"]),
        "no_fall_rate": no_fall_rate >= float(thresholds["min_no_fall_rate"]),
        "min_root_height_m": metrics["min_root_height_m"] >= float(thresholds["min_root_height_m"]),
        "body_action_saturation": metrics["max_body_action_saturation_fraction"]
        <= float(thresholds["max_body_action_saturation_fraction"]),
        "full_action_saturation": metrics["max_full_action_saturation_fraction"]
        <= float(thresholds["max_full_action_saturation_fraction"]),
        "normalized_control_energy": metrics["max_normalized_control_energy"]
        <= float(thresholds["max_normalized_control_energy"]),
        "lab_state_ood_fraction": metrics["max_lab_state_ood_fraction"]
        <= float(thresholds["max_lab_state_ood_fraction"]),
        "control_finite": metrics["min_control_finite"] >= float(thresholds["min_control_finite"]),
        "attachment_translation_drift": metrics["max_attachment_translation_drift_m"]
        <= float(thresholds["max_attachment_translation_drift_m"]),
        "attachment_rotation_drift": metrics["max_attachment_rotation_drift_rad"]
        <= float(thresholds["max_attachment_rotation_drift_rad"]),
    }
    return {
        **metrics,
        "required_steps": int(required_steps),
        "thresholds": thresholds,
        "gates": gates,
        "passed": all(gates.values()),
    }


def base_only_check(
    paths: IncomingHitPaths,
    *,
    out_dir: str | Path | None = None,
    latent_checkpoint: str | Path | None = None,
    episodes: int | None = None,
    steps: int | None = None,
) -> dict[str, Any]:
    """Validate frozen prior-mean control in the final Stage-3 full scene.

    A synthetic shuttle is parked high above the court so it cannot affect the
    body.  The task action is exactly zero and LAB lambda is zero, hence the
    only body command is ``decoder(state, prior_mean(state))``.  This closes the
    gap between Stage-2 latent rollout QC and the final 354-actuator rigid-tool scene.
    """
    from environment.overall_environment.src.shuttle_feeder import FeedSample

    out_path = Path(out_dir) if out_dir is not None else paths.output_dir / "base_only"
    out_path.mkdir(parents=True, exist_ok=True)
    _ensure_scene(paths)
    config = dict(paths.stage3_lab.get("base_only_check", {}) or {})
    rollout_count = int(config.get("rollouts", 5) if episodes is None else episodes)
    rollout_steps = int(config.get("steps", 120) if steps is None else steps)
    if rollout_count <= 0 or rollout_steps <= 0:
        raise ValueError("base-only rollouts and steps must be positive")

    lab = _build_stage3_lab_components(
        paths,
        latent_checkpoint=latent_checkpoint,
        lambda_lab=0.0,
    )
    if lab is None:
        raise ValueError("base-only check requires stage3_lab.enabled=true")
    contact_phase = float(paths.stage3_lab.get("contact_phase", 0.55))
    swing_duration = float(paths.stage3_lab.get("swing_duration_s", 1.2))
    parked_position = np.array([0.0, 0.0, 100.0], dtype=float)
    parked_velocity = np.zeros(3, dtype=float)
    parked_feed = FeedSample(
        launch_pos=parked_position,
        launch_vel=parked_velocity,
        trajectory=np.asarray([[0.0, *parked_position, *parked_velocity]], dtype=float),
        intercept_index=0,
        intercept_point=np.asarray([-2.7, 0.0, 1.8], dtype=float),
        intercept_velocity=parked_velocity,
        intercept_time_s=contact_phase * swing_duration,
    )
    base_only_target_bank = paths.target_bank_path
    if getattr(paths, "task_profile", "legacy_v1") == "impact_recovery_v2":
        from environment.overall_environment.src.shuttle_feeder import (
            feed_sample_fingerprint,
        )
        from environment.overall_environment.src.stage3_target_bank_v2 import (
            build_target_bank,
            load_target_bank,
        )

        production_bank = load_target_bank(paths.target_bank_path)
        base_only_target = replace(
            production_bank.targets[0],
            feed_fingerprint=feed_sample_fingerprint(parked_feed),
            impact_time_s=parked_feed.intercept_time_s,
            provenance="base_only_contract_target",
        )
        base_only_target_bank = build_target_bank(
            [base_only_target],
            source_fingerprint=production_bank.source_fingerprint,
            metadata={"purpose": "base_only_policy_abi_check"},
        )
    env = _make_env(
        paths,
        feed_bank=[parked_feed],
        seed=0,
        terminate_on_body_fall=True,
        lab_controller=lab.controller,
        lab_state_builder=lab.state_builder,
        curriculum=lab.curriculum,
        filter_finger_observation=False,
        swing_duration_s=swing_duration,
        contact_phase=contact_phase,
        impact_target_bank=base_only_target_bank,
    )
    zero_task_action = np.zeros(env.action_size, dtype=float)
    episode_reports: list[dict[str, Any]] = []
    for rollout in range(rollout_count):
        obs, _ = env.reset(feed_index=0)
        initial_pos, initial_rot = _site_relative_transform(env)
        minima = {"root_height": float(env._root_height())}
        maxima = {
            "body_action_saturation_fraction": 0.0,
            "full_action_saturation_fraction": 0.0,
            "normalized_control_energy": 0.0,
            "lab_state_ood_fraction": 0.0,
            "attachment_translation_drift_m": 0.0,
            "attachment_rotation_drift_rad": 0.0,
        }
        min_control_finite = 1.0
        finite = bool(np.isfinite(obs).all())
        body_fall = False
        termination_reason = None
        completed_steps = 0
        for _ in range(rollout_steps):
            obs, reward, terminated, truncated, info = env.step(zero_task_action)
            completed_steps += 1
            finite = finite and bool(np.isfinite(obs).all()) and math.isfinite(float(reward))
            body_fall = body_fall or bool(info.get("body_fall", False))
            minima["root_height"] = min(minima["root_height"], float(env._root_height()))
            current_pos, current_rot = _site_relative_transform(env)
            maxima["attachment_translation_drift_m"] = max(
                maxima["attachment_translation_drift_m"],
                float(np.linalg.norm(current_pos - initial_pos)),
            )
            maxima["attachment_rotation_drift_rad"] = max(
                maxima["attachment_rotation_drift_rad"],
                _rotation_distance_rad(initial_rot, current_rot),
            )
            for name in (
                "body_action_saturation_fraction",
                "full_action_saturation_fraction",
                "normalized_control_energy",
                "lab_state_ood_fraction",
            ):
                value = float(info.get(name, float("nan")))
                finite = finite and math.isfinite(value)
                if math.isfinite(value):
                    maxima[name] = max(maxima[name], value)
            control_finite = float(info.get("control_finite", float("nan")))
            finite = finite and math.isfinite(control_finite)
            if math.isfinite(control_finite):
                min_control_finite = min(min_control_finite, control_finite)
            if terminated or truncated:
                termination_reason = info.get("termination_reason")
                break
        episode_reports.append(
            {
                "rollout": rollout,
                "completed_steps": completed_steps,
                "finite": finite,
                "body_fall": body_fall,
                "termination_reason": termination_reason,
                "min_root_height_m": minima["root_height"],
                "max_body_action_saturation_fraction": maxima["body_action_saturation_fraction"],
                "max_full_action_saturation_fraction": maxima["full_action_saturation_fraction"],
                "max_normalized_control_energy": maxima["normalized_control_energy"],
                "max_lab_state_ood_fraction": maxima["lab_state_ood_fraction"],
                "min_control_finite": min_control_finite,
                "max_attachment_translation_drift_m": maxima["attachment_translation_drift_m"],
                "max_attachment_rotation_drift_rad": maxima["attachment_rotation_drift_rad"],
            }
        )

    thresholds = {
        "min_rollout_count": float(config.get("min_rollout_count", 5)),
        "min_completion_rate": float(config.get("min_completion_rate", 1.0)),
        "min_finite_rate": float(config.get("min_finite_rate", 1.0)),
        "min_no_fall_rate": float(config.get("min_no_fall_rate", 0.95)),
        "min_root_height_m": float(config.get("min_root_height_m", 0.55)),
        "max_body_action_saturation_fraction": float(config.get("max_body_action_saturation_fraction", 0.01)),
        "max_full_action_saturation_fraction": float(config.get("max_full_action_saturation_fraction", 0.01)),
        "max_normalized_control_energy": float(config.get("max_normalized_control_energy", 0.35)),
        "max_lab_state_ood_fraction": float(config.get("max_lab_state_ood_fraction", 0.01)),
        "min_control_finite": float(config.get("min_control_finite", 1.0)),
        "max_attachment_translation_drift_m": float(config.get("max_attachment_translation_drift_m", 0.005)),
        "max_attachment_rotation_drift_rad": float(config.get("max_attachment_rotation_drift_rad", 0.05)),
    }
    report = {
        "schema_version": "stage3_base_only_v1",
        "runner_stage": "base-only-check",
        "latent_checkpoint": str(lab.latent_checkpoint_dir),
        "lambda_lab": 0.0,
        "task_action": "all_zero_raw_latent",
        "shuttle_mode": "parked_out_of_scene",
        "episodes": episode_reports,
        **_base_only_summary(
            episode_reports,
            thresholds=thresholds,
            required_steps=rollout_steps,
        ),
        "control_manifest": env.control_manifest,
    }
    (out_path / "base_only_report.json").write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def physics_smoke(
    paths: IncomingHitPaths,
    *,
    out_dir: str | Path | None = None,
    episodes: int = 3,
    record_video: bool = False,
) -> dict[str, Any]:
    out_path = Path(out_dir) if out_dir is not None else paths.output_dir / "physics_smoke"
    out_path.mkdir(parents=True, exist_ok=True)
    _ensure_scene(paths)
    feed_artifact = _ensure_feed_bank_artifact(paths)
    bank = feed_artifact.bank

    # body_fall termination is disabled so the full shuttle flight is observable
    # under zero muscle activity.
    env = _make_env(paths, feed_bank=bank, seed=0, terminate_on_body_fall=False)
    zero_action = np.zeros(env.action_size, dtype=float)
    episode_reports = []
    finite = True
    frames: list[np.ndarray] = []
    renderer = None
    if record_video:
        import mujoco

        os.environ.setdefault("MUJOCO_GL", "egl")
        renderer = mujoco.Renderer(env.model, height=480, width=640)
        camera = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "overall_view")

    for episode in range(int(episodes)):
        obs, info = env.reset(feed_index=episode)
        start_x = float(info["feed_intercept_point"][0])
        launch_x = float(env.feed.launch_pos[0])
        min_speed = math.inf
        max_speed = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(zero_action)
            finite = finite and bool(np.isfinite(obs).all()) and math.isfinite(float(reward))
            speed = float(np.linalg.norm(info["flight"]["shuttle_velocity"]))
            min_speed = min(min_speed, speed)
            max_speed = max(max_speed, speed)
            if renderer is not None and episode == 0:
                renderer.update_scene(env.data, camera=camera)
                frames.append(renderer.render())
        flight = info["flight"]
        episode_reports.append(
            {
                "launch_x": launch_x,
                "intercept_x": start_x,
                "termination_reason": info.get("termination_reason"),
                "final_shuttle_xyz": np.asarray(flight["shuttle_xyz"]).tolist(),
                "final_state": info["state"],
                "steps": int(info["step_count"]),
                "max_speed_m_s": max_speed,
                "landing_speed_m_s": speed,
                "crossed_to_player_half": bool(float(flight["shuttle_xyz"][0]) < 0.0),
            }
        )

    if frames:
        import imageio.v2 as imageio

        video_path = out_path / "physics_smoke.mp4"
        imageio.mimsave(
            video_path, frames, fps=int(1.0 / (env.model.opt.timestep * env.control_substeps)), macro_block_size=None
        )

    landed = all(r["termination_reason"] in {"miss", "landed"} for r in episode_reports)
    crossed = all(r["crossed_to_player_half"] for r in episode_reports)
    # drag caps the shuttle speed near terminal velocity by touchdown
    aero_effective = all(r["landing_speed_m_s"] <= 7.5 for r in episode_reports)
    report = {
        "runner_stage": "physics-smoke",
        "episodes": episode_reports,
        "finite": bool(finite),
        "all_landed": landed,
        "all_crossed_to_player_half": crossed,
        "aero_effective": aero_effective,
        "passed": bool(finite and landed and crossed and aero_effective),
    }
    (out_path / "physics_smoke_report.json").write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def _run_ppo(
    paths: IncomingHitPaths,
    *,
    out_dir: Path,
    total_steps: int,
    rollout_steps: int,
    ppo_overrides: dict[str, Any] | None = None,
    seed: int = 0,
    device: str = "cpu",
    checkpoint_every_updates: int = 20,
) -> dict[str, Any]:
    if bool(paths.stage3_lab.get("enabled", False)):
        raise ValueError(
            "Stage-3 LAB specs cannot use the legacy CPU full-muscle PPO path; "
            "use --stage train-gpu so the task policy action is latent-only"
        )
    import torch

    from src.grip.train_right_hand_racket_grip_policy import (
        PolicyValueNet,
        PPOConfig,
        RunningMeanStd,
        _empty_rollout,
        _gae,
        _mean_last,
        _ppo_update,
        _sample_action,
        _tensor,
    )

    torch.set_num_threads(max(1, os.cpu_count() // 2 if os.cpu_count() else 1))
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    bank = _ensure_feed_bank(paths)
    env = _make_env(paths, feed_bank=bank, seed=seed)
    obs, last_info = env.reset()
    obs_size = int(obs.size)
    action_size = int(env.action_size)
    obs_rms = RunningMeanStd((obs_size,))
    obs_rms.update(obs)

    config_kwargs: dict[str, Any] = {
        "total_steps": int(total_steps),
        "rollout_steps": int(rollout_steps),
        "minibatch_size": min(int(rollout_steps), 256),
        "seed": int(seed),
    }
    for key, value in (ppo_overrides or {}).items():
        if key in {"total_steps", "rollout_steps"}:
            continue
        if key == "hidden_sizes":
            value = tuple(int(v) for v in value)
        config_kwargs[key] = value
    ppo_config = PPOConfig(**config_kwargs)

    model = PolicyValueNet(obs_size, action_size, ppo_config.hidden_sizes, ppo_config.action_std_init).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo_config.learning_rate)

    global_step = 0
    update_index = 0
    episode_return = 0.0
    episode_length = 0
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    episode_hits: list[float] = []
    episode_crossed: list[float] = []
    episode_landing_scores: list[float] = []
    current_hit = 0.0
    current_crossed = 0.0
    current_landing = 0.0
    summaries: list[dict[str, float]] = []
    finite = True

    while global_step < ppo_config.total_steps:
        rollout_target = min(ppo_config.rollout_steps, ppo_config.total_steps - global_step)
        rollout = _empty_rollout(rollout_target, obs_size, action_size)
        for step in range(rollout_target):
            obs_rms.update(obs)
            obs_norm = obs_rms.normalize(obs)
            action, logprob, value = _sample_action(torch, model, obs_norm, device, rng)
            next_obs, reward, terminated, truncated, last_info = env.step(action)
            done = bool(terminated or truncated)
            finite = finite and bool(np.isfinite(next_obs).all()) and bool(np.isfinite(reward))

            rollout["obs"][step] = obs_norm
            rollout["actions"][step] = action
            rollout["logprobs"][step] = logprob
            rollout["rewards"][step] = float(reward)
            rollout["dones"][step] = float(done)
            rollout["values"][step] = value

            terms = last_info.get("reward_terms", {})
            current_hit = max(current_hit, 1.0 if terms.get("hit_bonus", 0.0) > 0.0 else 0.0)
            current_crossed = max(current_crossed, 1.0 if terms.get("crossed_net", 0.0) > 0.0 else 0.0)
            if terms.get("landing_region", 0.0) != 0.0:
                current_landing = float(terms["landing_region"])

            episode_return += float(reward)
            episode_length += 1
            global_step += 1
            obs = next_obs
            if done:
                completed_returns.append(episode_return)
                completed_lengths.append(episode_length)
                episode_hits.append(current_hit)
                episode_crossed.append(current_crossed)
                episode_landing_scores.append(current_landing)
                episode_return = 0.0
                episode_length = 0
                current_hit = current_crossed = current_landing = 0.0
                obs, last_info = env.reset()
            if global_step >= ppo_config.total_steps:
                break

        next_obs_norm = obs_rms.normalize(obs)
        with torch.no_grad():
            next_value = float(model.value(_tensor(torch, next_obs_norm, device).unsqueeze(0)).item())
        advantages, returns = _gae(rollout["rewards"], rollout["dones"], rollout["values"], next_value, ppo_config)
        update_summary = _ppo_update(torch, model, optimizer, rollout, advantages, returns, ppo_config, device)
        update_index += 1
        update_summary.update(
            {
                "update": float(update_index),
                "global_step": float(global_step),
                "mean_rollout_reward": float(np.mean(rollout["rewards"])),
                "mean_episode_return": _mean_last(completed_returns, 10),
                "mean_episode_length": _mean_last(completed_lengths, 10),
                "hit_rate": _mean_last(episode_hits, 20),
                "crossed_net_rate": _mean_last(episode_crossed, 20),
                "mean_landing_score": _mean_last(episode_landing_scores, 20),
            }
        )
        summaries.append(update_summary)
        if update_index % max(1, int(checkpoint_every_updates)) == 0 or global_step >= ppo_config.total_steps:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "obs_rms_mean": obs_rms.mean,
                    "obs_rms_var": obs_rms.var,
                    "obs_size": obs_size,
                    "action_size": action_size,
                    "hidden_sizes": list(ppo_config.hidden_sizes),
                    "global_step": global_step,
                },
                out_dir / "policy_latest.pt",
            )

    report = {
        "runner_stage": "train",
        "global_steps": int(global_step),
        "updates": int(update_index),
        "finite": bool(finite),
        "episodes_completed": len(completed_returns),
        "mean_episode_return_last10": _mean_last(completed_returns, 10),
        "hit_rate_last20": _mean_last(episode_hits, 20),
        "crossed_net_rate_last20": _mean_last(episode_crossed, 20),
        "mean_landing_score_last20": _mean_last(episode_landing_scores, 20),
        "last_update": summaries[-1] if summaries else {},
        "checkpoint": str(out_dir / "policy_latest.pt"),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(
            _json_safe({"summaries": summaries, "report": report}),
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def train_tiny(
    paths: IncomingHitPaths,
    *,
    out_dir: str | Path | None = None,
    total_steps: int = 256,
    rollout_steps: int = 64,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    out_path = Path(out_dir) if out_dir is not None else paths.output_dir / "tiny_train"
    _ensure_scene(paths)
    report = _run_ppo(
        paths,
        out_dir=out_path,
        total_steps=total_steps,
        rollout_steps=rollout_steps,
        ppo_overrides={"update_epochs": 1, "hidden_sizes": (64, 64), "action_std_init": 0.25},
        seed=seed,
        device=device,
    )
    report["runner_stage"] = "train-tiny"
    (out_path / "tiny_train_report.json").write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def train(
    paths: IncomingHitPaths,
    *,
    out_dir: str | Path | None = None,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    out_path = Path(out_dir) if out_dir is not None else paths.output_dir / "train"
    _ensure_scene(paths)
    overrides = dict(paths.ppo_overrides)
    total_steps = int(overrides.get("total_steps", 2_000_000))
    rollout_steps = int(overrides.get("rollout_steps", 1024))
    return _run_ppo(
        paths,
        out_dir=out_path,
        total_steps=total_steps,
        rollout_steps=rollout_steps,
        ppo_overrides=overrides,
        seed=seed,
        device=device,
    )


def train_gpu(
    paths: IncomingHitPaths,
    *,
    out_dir: str | Path | None = None,
    num_envs: int = 512,
    rollout_steps: int = 64,
    total_env_steps: int | None = None,
    impl: str = "warp",
    base_policy_artifact: str | None = None,
    residual_scale: float = 0.3,
    base_skill: str | None = None,
    latent_checkpoint: str | Path | None = None,
    allow_unpromoted_latent: bool = False,
    resume_from: str | Path | None = None,
    curriculum_max_stage: str | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """GPU-parallel PPO on the MJX badminton env (warp backend by default).

    Requires the sanitized GPU environment (source configs/env.sh,
    which also prepends the cuda-compat 12.4 libraries needed by Warp).
    Production uses the promoted latent checkpoint and a 16D LAB task action.
    ``base_policy_artifact`` remains only for legacy residual ablations.
    """
    from environment.overall_environment.src.incoming_shuttle_hit_mjx_env import (
        IncomingHitMjxEnv,
    )
    from environment.overall_environment.src.train_incoming_hit_mjx import (
        TrainConfig,
        validate_stage3_direct_training_prerequisites,
        validate_stage3_training_prerequisites,
    )
    from environment.overall_environment.src.train_incoming_hit_mjx import (
        train as train_mjx,
    )

    out_path = Path(out_dir) if out_dir is not None else paths.output_dir / "train_gpu"
    _ensure_scene(paths)
    feed_artifact = _ensure_feed_bank_artifact(paths)
    bank = feed_artifact.bank
    lab = _build_stage3_lab_components(
        paths,
        latent_checkpoint=latent_checkpoint,
        allow_unpromoted=allow_unpromoted_latent,
    )
    task_profile = getattr(paths, "task_profile", "legacy_v1")

    env = IncomingHitMjxEnv(
        xml=paths.scene_xml,
        feed_bank=bank,
        control_substeps=paths.control_substeps,
        max_episode_steps=paths.max_episode_steps,
        reward_weights=paths.reward_weights,
        task_profile=task_profile,
        impact_target_bank=getattr(paths, "target_bank_path", None),
        recovery_horizon_steps=getattr(paths, "recovery_horizon_steps", 60),
        impl=impl,
        base_policy_artifact=base_policy_artifact,
        residual_scale=residual_scale,
        base_skill=base_skill,
        lab_controller=None if lab is None else lab.controller,
        lab_state_builder=None if lab is None else lab.state_builder,
        curriculum=None if lab is None else lab.curriculum,
        filter_finger_observation=None if lab is None else False,
        feed_bank_manifest=feed_artifact.manifest,
        swing_duration_s=float(paths.stage3_lab.get("swing_duration_s", 1.2)),
        contact_phase=float(paths.stage3_lab.get("contact_phase", 0.55)),
        task_curriculum_max_stage=(
            curriculum_max_stage
            if curriculum_max_stage is not None
            else ("C7_recovery" if task_profile == "impact_recovery_v2" else None)
        ),
    )
    if lab is not None:
        env.training_prerequisite_binding = validate_stage3_training_prerequisites(
            out_path,
            paths=paths,
            latent_checkpoint_dir=Path(lab.latent_checkpoint_dir),
            control_manifest=env.control_manifest,
            training_feed_manifest=env.feed_bank_manifest,
        )
    elif task_profile == "impact_recovery_v2":
        env.training_prerequisite_binding = validate_stage3_direct_training_prerequisites(
            out_path,
            paths=paths,
            control_manifest=env.control_manifest,
            training_feed_manifest=env.feed_bank_manifest,
        )
    ppo = dict(paths.ppo_overrides)
    if total_env_steps is None:
        total_env_steps = int(ppo.get("total_steps", 2_000_000))
    cfg = TrainConfig(
        num_envs=int(num_envs),
        rollout_steps=int(rollout_steps),
        total_env_steps=int(total_env_steps),
        update_epochs=int(ppo.get("update_epochs", 4)),
        minibatch_size=int(ppo.get("minibatch_size", 0)),
        hidden=tuple(ppo.get("hidden_sizes", (256, 256))),
        action_std_init=float(ppo.get("action_std_init", 0.35)),
        learning_rate=float(ppo.get("learning_rate", 3e-4)),
        seed=int(seed),
    )
    report = train_mjx(
        env,
        cfg,
        out_path,
        resume_from=None if resume_from is None else Path(resume_from),
    )
    report["runner_stage"] = "train-gpu"
    report["impl"] = impl
    report["control_manifest"] = env.control_manifest
    return report


def evaluate(
    paths: IncomingHitPaths,
    *,
    checkpoint: str | Path | None = None,
    out_dir: str | Path | None = None,
    episodes: int = 8,
    record_video: bool = False,
    export_simulation_npz: str | Path | None = None,
    signal_identity_json: str | Path | None = None,
    policy_evidence_json: str | Path | None = None,
    signal_sidecar_json: str | Path | None = None,
    signal_pre_impact_s: float = 0.5,
    signal_post_impact_s: float = 0.8,
) -> dict[str, Any]:
    """Replay a checkpoint with its exact training-time action/control stack."""
    out_path = Path(out_dir) if out_dir is not None else paths.output_dir / "evaluate"
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(checkpoint) if checkpoint is not None else paths.output_dir / "train_gpu" / "policy_latest.npz"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    from environment.overall_environment.src.train_incoming_hit_mjx import (
        load_training_checkpoint_metadata,
    )

    meta = load_training_checkpoint_metadata(ckpt_path)
    control_manifest = dict(meta.get("control_manifest", {}) or {})
    is_lab = control_manifest.get("schema_version") == "stage3_lab_control_v1"
    curriculum_state = dict(meta.get("curriculum_state", {}) or {})
    task_curriculum_state = dict(meta.get("task_curriculum_state", {}) or {})
    evaluation_task_stage = (
        str(task_curriculum_state.get("stage"))
        if paths.task_profile == "impact_recovery_v2" and task_curriculum_state.get("stage")
        else None
    )
    latent_checkpoint = control_manifest.get("latent_checkpoint_dir")
    lab = _build_stage3_lab_components(
        paths,
        latent_checkpoint=latent_checkpoint,
        lambda_lab=curriculum_state.get("lambda_lab"),
    )
    if is_lab != (lab is not None):
        raise ValueError("evaluation spec LAB mode does not match the training checkpoint control manifest")

    hidden = tuple(int(h) for h in meta.get("hidden", meta["config"]["hidden"]))
    if meta.get("checkpoint_version") in {
        "incoming_hit_training_v2",
        "incoming_hit_training_v3",
    }:
        import jax
        import jax.numpy as jnp
        import optax

        from environment.overall_environment.src.train_incoming_hit_mjx import (
            _mlp,
            init_agent,
            load_training_checkpoint,
        )

        config = dict(meta["config"])
        template = init_agent(
            jax.random.PRNGKey(0),
            obs_size=int(meta["obs_size"]),
            action_size=int(meta["action_size"]),
            hidden=hidden,
            action_std_init=float(config.get("action_std_init", 0.35)),
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(float(config.get("max_grad_norm", 0.5))),
            optax.adam(float(config.get("learning_rate", 3e-4))),
        )
        restored = load_training_checkpoint(
            ckpt_path,
            agent_template=template,
            optimizer_state_template=optimizer.init(template),
        )
        obs_mean = np.asarray(restored.obs_rms.mean)
        obs_var = np.asarray(restored.obs_rms.var)

        def mlp_forward(obs_norm: np.ndarray) -> np.ndarray:
            mean = np.asarray(jax.device_get(_mlp(restored.agent["policy"], jnp.asarray(obs_norm))))
            return mean if is_lab else np.tanh(mean)

    else:
        with np.load(ckpt_path) as payload:
            params = [
                payload[key]
                for key in sorted(payload.files, key=lambda value: (len(value), value))
                if key.startswith("param_")
            ]
            obs_mean = payload["obs_mean"]
            obs_var = payload["obs_var"]
        n_policy = len(hidden) + 1

        def mlp_forward(obs_norm: np.ndarray) -> np.ndarray:
            x = obs_norm
            for index in range(n_policy):
                bias, weight = params[1 + 2 * index], params[2 + 2 * index]
                x = x @ weight + bias
                if index < n_policy - 1:
                    x = np.tanh(x)
            return x if is_lab else np.tanh(x)

    from environment.overall_environment.src.incoming_shuttle_hit_env import IncomingShuttleHitEnv

    evaluation_feed_artifact = _ensure_feed_bank_artifact(paths, evaluation=True)
    bank = evaluation_feed_artifact.bank
    checkpoint_training_feed_manifest = meta.get("training_feed_manifest")
    heldout_feed_identity: dict[str, Any] | None = None
    if paths.task_profile == "impact_recovery_v2":
        if not isinstance(checkpoint_training_feed_manifest, dict):
            raise ValueError("Stage-3 checkpoint is missing its training feed manifest")
        heldout_feed_identity = _feed_bank_identity_qc(
            checkpoint_training_feed_manifest,
            evaluation_feed_artifact.manifest,
            paths_distinct=(paths.feed_bank_path.resolve() != paths.eval_feed_bank_path.resolve()),
        )
        if not (
            heldout_feed_identity["bank_paths_distinct"]
            and heldout_feed_identity["train_duplicate_count"] == 0
            and heldout_feed_identity["eval_duplicate_count"] == 0
            and heldout_feed_identity["train_eval_fingerprint_overlap_count"] == 0
        ):
            raise ValueError("Stage-3 evaluation feed is not a unique held-out bank relative to training")

    def make_evaluation_env(seed: int) -> IncomingShuttleHitEnv:
        return IncomingShuttleHitEnv(
            paths.scene_xml,
            feed_bank=bank,
            control_substeps=paths.control_substeps,
            max_episode_steps=paths.max_episode_steps,
            reward_weights=paths.reward_weights,
            task_profile=paths.task_profile,
            impact_target_bank=(
                paths.eval_target_bank_path if paths.eval_target_bank_path is not None else paths.target_bank_path
            ),
            recovery_horizon_steps=paths.recovery_horizon_steps,
            task_curriculum_stage=evaluation_task_stage,
            terminate_on_body_fall=True,
            lab_controller=None if lab is None else lab.controller,
            lab_state_builder=None if lab is None else lab.state_builder,
            curriculum=None if lab is None else lab.curriculum,
            filter_finger_observation=None if lab is None else False,
            swing_duration_s=float(paths.stage3_lab.get("swing_duration_s", 1.2)),
            contact_phase=float(paths.stage3_lab.get("contact_phase", 0.55)),
            seed=int(seed),
        )

    evaluation_seed = 123
    env = make_evaluation_env(evaluation_seed)
    prior_env = make_evaluation_env(evaluation_seed) if is_lab else None
    if int(meta["obs_size"]) != env.observation_size or int(meta["action_size"]) != env.action_size:
        raise ValueError("evaluation environment observation/action dimensions differ from checkpoint")
    checkpoint_policy_abi = control_manifest.get("policy_abi_hash")
    if is_lab:
        if paths.task_profile == "impact_recovery_v2":
            if checkpoint_policy_abi != env.policy_abi_hash:
                raise ValueError("evaluation policy/control/observation ABI differs from training checkpoint")
        elif meta.get("control_hash") != env.control_hash:
            raise ValueError("evaluation LAB runtime/router/grip/state contract differs from training checkpoint")

    signal_collector = None
    signal_episode_ordinals: dict[int, int] = {}
    if export_simulation_npz is not None:
        if signal_identity_json is None or policy_evidence_json is None:
            raise ValueError("Stage-3 signal export requires both signal_identity_json and policy_evidence_json")
        if lab is None or not is_lab:
            raise ValueError("Stage-3 signal export requires the final LAB low-level policy runtime")
        if paths.task_profile != "impact_recovery_v2" or int(env._v2_environment_mode_code) != 3:
            raise ValueError(
                "Stage-3 signal export is valid only for the final dynamic impact/recovery curriculum stage"
            )
        from environment.overall_environment.src.train_incoming_hit_mjx import (
            resolve_training_checkpoint,
        )
        from musclemimic.evaluation.stage3_signal_export import (
            Stage3SignalCollector,
            Stage3SignalLayout,
            canonical_mapping_fingerprint,
            load_paired_policy_evidence,
            load_trial_identity_manifest,
        )

        payload_path, _ = resolve_training_checkpoint(ckpt_path)
        stage3_payload_sha256 = hashlib.sha256(payload_path.read_bytes()).hexdigest()
        identities = load_trial_identity_manifest(signal_identity_json)
        signal_feed_indices = tuple(identities.trials_by_feed)
        if any(index < 0 or index >= int(episodes) for index in signal_feed_indices):
            raise ValueError(
                "Stage-3 signal identity feeds must all be covered by this deterministic evaluation; "
                f"episodes={int(episodes)} feeds={list(signal_feed_indices)}"
            )
        signal_episode_ordinals = {int(feed_index): ordinal for ordinal, feed_index in enumerate(signal_feed_indices)}
        evidence = load_paired_policy_evidence(
            policy_evidence_json,
            stage3_checkpoint_payload_sha256=stage3_payload_sha256,
        )
        event_reference_fingerprint = str(env.impact_target_bank.source_fingerprint)
        layout = Stage3SignalLayout.from_environment(
            env,
            body_actuator_names=lab.controller.router.body_actuator_names,
        )
        signal_collector = Stage3SignalCollector(
            layout=layout,
            identities=identities,
            policy_evidence=evidence,
            control_dt_s=float(paths.control_substeps) * float(env.model.opt.timestep),
            pre_impact_s=float(signal_pre_impact_s),
            post_impact_s=float(signal_post_impact_s),
            expected_episode_count=len(signal_episode_ordinals),
            runtime=lab.controller.runtime,
            event_reference_fingerprint=event_reference_fingerprint,
            stage3_checkpoint_payload_sha256=stage3_payload_sha256,
            evaluation_feed_manifest_fingerprint=canonical_mapping_fingerprint(evaluation_feed_artifact.manifest),
            evaluation_seed=evaluation_seed,
        )
    renderer = None
    frames: list[Any] = []
    if record_video:
        import mujoco

        os.environ.setdefault("MUJOCO_GL", "egl")
        env.model.vis.global_.offwidth = 1280
        env.model.vis.global_.offheight = 720
        renderer = mujoco.Renderer(env.model, height=720, width=1280)
        camera = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "overall_view")

    results = []
    heldout_lab_state_ood_values: list[float] = []
    for episode in range(int(episodes)):
        prior_trace = _rollout_prior_naturalness(prior_env, feed_index=episode) if prior_env is not None else None
        obs, info = env.reset(feed_index=episode)
        collect_signals = signal_collector is not None and episode in signal_episode_ordinals
        if collect_signals:
            feed_fingerprints = evaluation_feed_artifact.manifest.get("sample_fingerprints")
            if not isinstance(feed_fingerprints, list) or env._feed_index >= len(feed_fingerprints):
                raise ValueError("evaluation feed manifest has no signal-export fingerprint for active feed")
            signal_collector.begin_episode(
                episode_index=signal_episode_ordinals[episode],
                feed_index=int(env._feed_index),
                feed_fingerprint=str(feed_fingerprints[env._feed_index]),
            )
        initial_attachment_pos, initial_attachment_rot = _site_relative_transform(env)
        max_attachment_translation_drift = 0.0
        max_attachment_rotation_drift = 0.0
        min_root_height = float(env._root_height())
        any_body_fall = False
        episode_return = 0.0
        max_racket_speed = 0.0
        contact_racket_speed = 0.0
        max_net_clearance = float("-inf")
        previous_shuttle_x = float(env.data.qpos[env._shuttle_qadr])
        crossed = False
        lab_diagnostics: dict[str, list[float]] = {
            name: []
            for name in (
                "control_finite",
                "raw_latent_rms",
                "raw_latent_saturation",
                "latent_norm",
                "prior_sigma_mean",
                "lab_state_unclipped_z_rms",
                "lab_state_ood_fraction",
                "body_action_rms",
                "right_grip_action_rms",
                "raw_action_rate_rms",
                "normalized_control_energy",
                "body_action_saturation_fraction",
                "full_action_saturation_fraction",
                "muscle_power_abs_mean",
                "bounded_residual_rms",
            )
        }
        terminated = truncated = False
        naturalness_trace: list[dict[str, np.ndarray]] = []
        while not (terminated or truncated):
            obs_norm = np.clip((obs - obs_mean) / np.sqrt(obs_var + 1e-8), -10.0, 10.0)
            action = mlp_forward(obs_norm)
            obs, reward, terminated, truncated, info = env.step(action)
            if collect_signals:
                signal_collector.record_transition(signal_collector.layout.capture_transition(env, info))
            if is_lab:
                naturalness_trace.append(_naturalness_snapshot(env))
            episode_return += float(reward)
            min_root_height = min(min_root_height, float(env._root_height()))
            any_body_fall = any_body_fall or bool(info.get("body_fall", False))
            attachment_pos, attachment_rot = _site_relative_transform(env)
            max_attachment_translation_drift = max(
                max_attachment_translation_drift,
                float(np.linalg.norm(attachment_pos - initial_attachment_pos)),
            )
            max_attachment_rotation_drift = max(
                max_attachment_rotation_drift,
                _rotation_distance_rad(initial_attachment_rot, attachment_rot),
            )
            current_racket_speed = float(np.linalg.norm(env._stringbed_velocity()))
            max_racket_speed = max(max_racket_speed, current_racket_speed)
            if bool(info.get("hit_this_step", False)):
                contact_racket_speed = max(contact_racket_speed, current_racket_speed)
            shuttle = np.asarray(info["flight"]["shuttle_xyz"], dtype=float)
            clearance = _return_net_clearance(
                previous_shuttle_x=previous_shuttle_x,
                shuttle_xyz=shuttle,
                hit_registered=env._hit_rewarded,
            )
            if clearance is not None:
                max_net_clearance = max(max_net_clearance, clearance)
            previous_shuttle_x = float(shuttle[0])
            crossed = crossed or bool(env._crossed_net_rewarded)
            for name, values in lab_diagnostics.items():
                if name in info:
                    value = float(info[name])
                    values.append(value)
                    if name == "lab_state_ood_fraction":
                        heldout_lab_state_ood_values.append(value)
            if renderer is not None and episode == 0:
                renderer.update_scene(env.data, camera=camera)
                frames.append(renderer.render())
        if collect_signals:
            signal_collector.end_episode()
        episode_result = {
            "episode": episode,
            "return": episode_return,
            "steps": int(info["step_count"]),
            "termination_reason": info.get("termination_reason"),
            "hit": bool(env._hit_rewarded),
            "crossed_net": crossed,
            "landing_region": info.get("landing_region"),
            "body_fall": any_body_fall,
            "min_root_height_m": min_root_height,
            "max_attachment_translation_drift_m": max_attachment_translation_drift,
            "max_attachment_rotation_drift_rad": max_attachment_rotation_drift,
            "max_racket_head_speed_m_s": max_racket_speed,
            "contact_racket_head_speed_m_s": contact_racket_speed,
            "net_clearance_m": (None if not np.isfinite(max_net_clearance) else max_net_clearance),
            "lab_diagnostics": {name: float(np.mean(values)) for name, values in lab_diagnostics.items() if values},
            "stage3_v2_metrics": dict(info.get("stage3_v2_metrics", {}) or {}),
            "recovery_complete": bool(info.get("recovery_complete", False)),
            "flight_resolved": bool(info.get("flight_resolved", False)),
        }
        if prior_trace is not None:
            episode_result["naturalness"] = _compare_naturalness_to_prior(
                naturalness_trace,
                prior_trace,
            )
        results.append(episode_result)

    if frames:
        import imageio.v2 as imageio

        imageio.mimsave(out_path / "evaluate_episode0.mp4", frames, fps=100, macro_block_size=None)

    gate_config = dict(paths.evaluation.get("promotion_gates", {}) or {})
    prior_direct_baseline = _load_prior_direct_naturalness_baseline(lab) if is_lab else None
    evaluation_summary = _stage3_evaluation_summary(
        results,
        gate_config=gate_config,
        required_feed_count=int(paths.evaluation.get("heldout_feed_count", paths.eval_feed_bank_size)),
        lab_state_ood_values=heldout_lab_state_ood_values,
        prior_direct_baseline=prior_direct_baseline,
        task_profile=paths.task_profile,
        action_family=_stage3_action_family(env.control_manifest),
    )
    report = {
        "schema_version": "incoming_shuttle_hit_evaluate_v3",
        "runner_stage": "evaluate",
        "checkpoint": str(ckpt_path),
        "evaluation_seed": evaluation_seed,
        "episodes": results,
        "mean_return": float(np.mean([r["return"] for r in results])),
        **evaluation_summary,
        "control_manifest": env.control_manifest,
        "training_feed_manifest": checkpoint_training_feed_manifest,
        "evaluation_feed_manifest": evaluation_feed_artifact.manifest,
        "heldout_feed_identity": heldout_feed_identity,
        "prior_direct_naturalness_baseline": prior_direct_baseline,
        "lab_diagnostics": {
            name: float(
                np.mean([result["lab_diagnostics"][name] for result in results if name in result["lab_diagnostics"]])
            )
            for name in (
                "control_finite",
                "raw_latent_rms",
                "raw_latent_saturation",
                "latent_norm",
                "prior_sigma_mean",
                "lab_state_unclipped_z_rms",
                "lab_state_ood_fraction",
                "body_action_rms",
                "right_grip_action_rms",
                "raw_action_rate_rms",
                "normalized_control_energy",
                "body_action_saturation_fraction",
                "full_action_saturation_fraction",
                "muscle_power_abs_mean",
                "bounded_residual_rms",
            )
            if any(name in result["lab_diagnostics"] for result in results)
        },
    }
    evaluation_content_sha256 = _stage3_evaluation_content_sha256(report)
    artifact_binding = _build_stage3_artifact_binding(
        paths=paths,
        checkpoint_path=ckpt_path,
        checkpoint_metadata=meta,
        control_manifest=env.control_manifest,
        training_feed_manifest=checkpoint_training_feed_manifest,
        evaluation_feed_manifest=evaluation_feed_artifact.manifest,
        evaluation_seed=evaluation_seed,
        evaluation_content_sha256=evaluation_content_sha256,
    )
    report["artifact_binding"] = artifact_binding
    report["artifact_binding_verified"] = 1.0
    report["promotion_gates"]["artifact_binding_verified"] = True
    report["promotion_thresholds"]["artifact_binding_verified"] = 1.0
    report["passed"] = bool(report["passed"] and artifact_binding["verified"])
    if signal_collector is not None:
        from musclemimic.evaluation.stage3_signal_export import (
            write_stage3_signal_export,
        )

        signal_arrays = signal_collector.finalize_arrays(evaluation_binding_sha256=artifact_binding["binding_sha256"])
        write_stage3_signal_export(
            export_simulation_npz,
            signal_arrays,
            collector=signal_collector,
            sidecar_json=signal_sidecar_json,
        )
    candidate_path = out_path / "stage3_visual_review_candidate.json"
    if frames:
        report["visual_review_candidate"] = str(candidate_path)
    report_path = out_path / "evaluate_report.json"
    report_path.write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    if frames:
        video_path = out_path / "evaluate_episode0.mp4"
        candidate = {
            "schema_version": "incoming_hit_visual_review_candidate_v1",
            "review_kind": "stage3_incoming_hit",
            "checkpoint_binding_sha256": artifact_binding["binding_sha256"],
            "evaluate_report": str(report_path.resolve()),
            "evaluate_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "artifact": str(video_path.resolve()),
            "artifact_sha256": hashlib.sha256(video_path.read_bytes()).hexdigest(),
            "feed_index": 0,
            "body_motion_natural": None,
            "right_hand_site_motion_natural": None,
            "racket_head_trajectory_natural": None,
            "racket_face_orientation_natural": None,
            "major_swing_complete": None,
            "passed": None,
            "notes": None,
        }
        candidate_path.write_text(
            json.dumps(candidate, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return report


def _stage3_action_family(control_manifest: dict[str, Any]) -> str:
    """Return the explicit Stage-3 policy family represented by a control ABI."""

    schema = control_manifest.get("schema_version")
    if schema == "incoming_hit_direct_action_impact_recovery_v2":
        return "full_354"
    if schema == "stage3_lab_control_v1":
        return "latent_direct_ablation" if control_manifest.get("decoder_type") == "direct" else "fixed_synergy"
    return "legacy_unspecified"


def _build_stage3_artifact_binding(
    *,
    paths: IncomingHitPaths,
    checkpoint_path: Path,
    checkpoint_metadata: dict[str, Any],
    control_manifest: dict[str, Any],
    training_feed_manifest: dict[str, Any] | None,
    evaluation_feed_manifest: dict[str, Any],
    evaluation_content_sha256: str,
    evaluation_seed: int = 123,
) -> dict[str, Any]:
    """Bind evaluation to every immutable Stage-3 training dependency."""

    from environment.overall_environment.src.train_incoming_hit_mjx import (
        resolve_training_checkpoint,
    )

    payload_path, metadata_path = resolve_training_checkpoint(checkpoint_path)
    if not isinstance(training_feed_manifest, dict):
        raise ValueError("Stage-3 artifact binding requires a training feed manifest")
    checkpoint_control_manifest = dict(checkpoint_metadata.get("control_manifest", {}) or {})
    impact_recovery_v2 = getattr(paths, "task_profile", "legacy_v1") == "impact_recovery_v2"
    action_family = _stage3_action_family(control_manifest)
    checkpoint_action_family = _stage3_action_family(checkpoint_control_manifest)
    if impact_recovery_v2 and action_family not in {"full_354", "fixed_synergy", "latent_direct_ablation"}:
        raise ValueError("Stage-3 checkpoint has no explicit impact/recovery action family")
    if impact_recovery_v2 and checkpoint_action_family != action_family:
        raise ValueError("Stage-3 checkpoint and evaluation action families differ")
    if impact_recovery_v2 and checkpoint_control_manifest.get("policy_abi_hash") != control_manifest.get(
        "policy_abi_hash"
    ):
        raise ValueError("Stage-3 checkpoint and evaluation policy ABIs differ")
    if not impact_recovery_v2 and checkpoint_metadata.get("control_hash") != control_manifest.get("control_hash"):
        raise ValueError("Stage-3 checkpoint and evaluation control hashes differ")
    if checkpoint_metadata.get("training_feed_manifest") != training_feed_manifest:
        raise ValueError("Stage-3 checkpoint training feed manifest changed during evaluation")
    training_seed: int | None = None
    if impact_recovery_v2:
        checkpoint_config = checkpoint_metadata.get("config")
        if not isinstance(checkpoint_config, dict) or isinstance(checkpoint_config.get("seed"), bool):
            raise ValueError("Stage-3 checkpoint has no exact training seed")
        try:
            training_seed = int(checkpoint_config["seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Stage-3 checkpoint has no exact training seed") from exc
        if float(checkpoint_config["seed"]) != float(training_seed):
            raise ValueError("Stage-3 checkpoint training seed is not an integer")
    train_report_path = _find_train_report(checkpoint_path, payload_path)
    train_report = json.loads(train_report_path.read_text(encoding="utf-8"))
    if not isinstance(train_report, dict):
        raise ValueError("Stage-3 train report must be a JSON object")
    task_state = dict(checkpoint_metadata.get("task_curriculum_state", {}) or {})
    static_target_checkpoint = bool(
        task_state.get("max_stage") == "C3_static_velocity" and task_state.get("complete") is True
    )
    if train_report.get("curriculum_complete") is not True and not static_target_checkpoint:
        raise ValueError("Stage-3 artifact binding rejects incomplete curriculum training")
    if train_report.get("promotion_eligible") is not True and not static_target_checkpoint:
        raise ValueError("Stage-3 train report is not promotion eligible")
    if checkpoint_metadata.get("curriculum_complete") is not True and not static_target_checkpoint:
        raise ValueError("Stage-3 checkpoint metadata records an incomplete curriculum")
    if checkpoint_metadata.get("promotion_eligible") is not True and not static_target_checkpoint:
        raise ValueError("Stage-3 checkpoint metadata is not promotion eligible")
    prerequisite_binding = _validate_stage3_training_prerequisite_binding(
        checkpoint_metadata.get("training_prerequisite_binding")
    )
    if train_report.get("training_prerequisite_binding") != prerequisite_binding:
        raise ValueError("Stage-3 train report and checkpoint disagree on prerequisite evidence")
    training_control_manifest = checkpoint_control_manifest if impact_recovery_v2 else control_manifest
    if impact_recovery_v2 and prerequisite_binding.get("action_family") != action_family:
        raise ValueError("Stage-3 prerequisite action family changed")
    if prerequisite_binding.get("control_hash") != training_control_manifest.get(
        "control_hash"
    ) or prerequisite_binding.get("latent_checkpoint_fingerprint") != control_manifest.get(
        "latent_checkpoint_fingerprint"
    ):
        raise ValueError("Stage-3 prerequisite control/latent identity changed")
    if prerequisite_binding.get("training_feed_manifest_sha256") != _mapping_sha256(training_feed_manifest):
        raise ValueError("Stage-3 prerequisite training feed identity changed")
    curriculum_state = checkpoint_metadata.get("curriculum_state")
    if not isinstance(curriculum_state, dict):
        raise ValueError("Stage-3 checkpoint metadata has no curriculum state")
    consistency_fields = (
        ("iterations", "iteration", checkpoint_metadata),
        ("env_steps", "env_steps", checkpoint_metadata),
        ("curriculum_effective_steps", "effective_steps", curriculum_state),
        ("curriculum_phase", "phase", curriculum_state),
        ("curriculum_complete", "curriculum_complete", checkpoint_metadata),
        ("promotion_eligible", "promotion_eligible", checkpoint_metadata),
    )
    for report_key, checkpoint_key, source in consistency_fields:
        if train_report.get(report_key) != source.get(checkpoint_key):
            raise ValueError(f"Stage-3 train report and checkpoint metadata disagree on {report_key}")
    report_checkpoint_value = train_report.get("checkpoint")
    if not isinstance(report_checkpoint_value, str) or not report_checkpoint_value:
        raise ValueError("Stage-3 train report has no checkpoint pointer")
    report_checkpoint = Path(report_checkpoint_value).expanduser()
    if not report_checkpoint.is_absolute():
        report_checkpoint = REPO_ROOT / report_checkpoint
    report_payload, report_metadata = resolve_training_checkpoint(report_checkpoint)
    if report_payload != payload_path or report_metadata != metadata_path:
        raise ValueError("Stage-3 train report points to a different checkpoint")
    payload = {
        "schema_version": "incoming_hit_evaluation_artifact_binding_v3",
        "checkpoint_payload_path": str(payload_path),
        "checkpoint_payload_sha256": hashlib.sha256(payload_path.read_bytes()).hexdigest(),
        "checkpoint_metadata_path": str(metadata_path),
        "checkpoint_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "action_family": action_family,
        "latent_checkpoint_fingerprint": control_manifest.get("latent_checkpoint_fingerprint"),
        "spec_path": str(paths.spec_path.resolve()),
        "spec_sha256": hashlib.sha256(paths.spec_path.read_bytes()).hexdigest(),
        "scene_path": str(paths.scene_xml.resolve()),
        "scene_sha256": hashlib.sha256(paths.scene_xml.read_bytes()).hexdigest(),
        "train_report_path": str(train_report_path.resolve()),
        "train_report_sha256": hashlib.sha256(train_report_path.read_bytes()).hexdigest(),
        "training_feed_manifest_sha256": _mapping_sha256(training_feed_manifest),
        "evaluation_feed_manifest_sha256": _mapping_sha256(evaluation_feed_manifest),
        "evaluation_content_sha256": str(evaluation_content_sha256),
        "training_prerequisite_binding_sha256": prerequisite_binding["binding_sha256"],
        "curriculum_effective_steps": int(train_report.get("curriculum_effective_steps", -1)),
        "curriculum_phase": train_report.get("curriculum_phase"),
        "checkpoint_iteration": int(checkpoint_metadata["iteration"]),
        "checkpoint_env_steps": int(checkpoint_metadata["env_steps"]),
        "checkpoint_curriculum_complete": bool(checkpoint_metadata.get("curriculum_complete") is True),
        "checkpoint_promotion_eligible": bool(checkpoint_metadata.get("promotion_eligible") is True),
        "checkpoint_task_curriculum_complete": bool(task_state.get("complete") is True),
        "checkpoint_task_curriculum_max_stage": task_state.get("max_stage"),
        "verified": True,
    }
    if impact_recovery_v2:
        payload.update(
            {
                "training_control_hash": checkpoint_control_manifest.get("control_hash"),
                "evaluation_control_hash": control_manifest.get("control_hash"),
                "policy_abi_hash": control_manifest.get("policy_abi_hash"),
                "training_seed": training_seed,
                "evaluation_seed": int(evaluation_seed),
            }
        )
        target_paths = {
            "training": paths.target_bank_path,
            "evaluation": paths.eval_target_bank_path,
        }
        for label, target_path in target_paths.items():
            if target_path is None or not Path(target_path).is_file():
                raise ValueError(f"Stage-3 v2 {label} target bank is missing")
            target_payload = json.loads(Path(target_path).read_text(encoding="utf-8"))
            if not isinstance(target_payload, dict):
                raise ValueError(f"Stage-3 v2 {label} target bank is invalid")
            for key in ("bank_sha256", "source_fingerprint"):
                value = target_payload.get(key)
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                ):
                    raise ValueError(f"Stage-3 v2 {label} target bank has no valid {key}")
                payload[f"{label}_target_{key}"] = value
            payload[f"{label}_target_path"] = str(Path(target_path).resolve())
            payload[f"{label}_target_file_sha256"] = hashlib.sha256(Path(target_path).read_bytes()).hexdigest()
        if payload["training_target_bank_sha256"] == payload["evaluation_target_bank_sha256"]:
            raise ValueError("Stage-3 v2 train/evaluation target banks must differ")
        if action_family == "full_354":
            for key in (
                "training_target_bank_sha256",
                "training_target_source_fingerprint",
                "training_target_file_sha256",
            ):
                if prerequisite_binding.get(key) != payload[key]:
                    raise ValueError(f"Stage-3 direct prerequisite changed {key}")
            if prerequisite_binding.get("scene_sha256") != payload["scene_sha256"]:
                raise ValueError("Stage-3 direct prerequisite scene changed")
            if prerequisite_binding.get("spec_sha256") != payload["spec_sha256"]:
                raise ValueError("Stage-3 direct prerequisite spec changed")
    required_strings = [
        "action_family",
        "curriculum_phase",
        "evaluation_content_sha256",
        "training_prerequisite_binding_sha256",
    ]
    if action_family != "full_354":
        required_strings.append("latent_checkpoint_fingerprint")
    elif payload["latent_checkpoint_fingerprint"] is not None:
        raise ValueError("Stage-3 full_354 artifact binding must record latent_checkpoint_fingerprint=null")
    if impact_recovery_v2:
        required_strings.extend(["training_control_hash", "evaluation_control_hash", "policy_abi_hash"])
    else:
        payload["control_hash"] = control_manifest.get("control_hash")
        required_strings.append("control_hash")
    if any(not isinstance(payload[name], str) or not payload[name] for name in required_strings):
        raise ValueError("Stage-3 artifact binding has an empty identity field")
    payload["binding_sha256"] = _mapping_sha256(payload)
    return payload


def _validate_stage3_training_prerequisite_binding(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Stage-3 checkpoint has no training prerequisite binding")
    schema = value.get("schema_version")
    if (
        schema
        not in {
            "stage3_training_prerequisite_binding_v1",
            "stage3_direct_training_prerequisite_binding_v1",
        }
        or value.get("verified") is not True
    ):
        raise ValueError("Stage-3 training prerequisite binding is incompatible")
    recorded = value.get("binding_sha256")
    unbound = dict(value)
    unbound.pop("binding_sha256", None)
    if recorded != _mapping_sha256(unbound):
        raise ValueError("Stage-3 training prerequisite binding hash mismatch")
    report_bindings = [
        ("preflight_report_path", "preflight_report_sha256"),
        ("feed_check_report_path", "feed_check_report_sha256"),
    ]
    if schema == "stage3_training_prerequisite_binding_v1":
        report_bindings.append(("base_only_report_path", "base_only_report_sha256"))
    else:
        if (
            value.get("action_family") != "full_354"
            or value.get("policy_action_size") != 354
            or value.get("latent_checkpoint_fingerprint") is not None
        ):
            raise ValueError("Stage-3 direct prerequisite action contract is incompatible")
        report_bindings.extend(
            (
                ("spec_path", "spec_sha256"),
                ("scene_path", "scene_sha256"),
                ("training_target_path", "training_target_file_sha256"),
            )
        )
    for path_key, hash_key in report_bindings:
        path = Path(str(value.get(path_key, ""))).expanduser()
        if not path.is_file() or value.get(hash_key) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f"Stage-3 prerequisite report changed: {path}")
    return value


def _stage3_evaluation_content_sha256(report: dict[str, Any]) -> str:
    """Hash all evaluation results while excluding only self-binding fields."""

    # Failed evaluations legitimately use +/-inf internally for metrics with no
    # observed hit/landing.  Hash the same JSON-safe representation that is
    # persisted, so failure evidence is writable and validator/generator agree.
    content = _json_safe(report)
    if not isinstance(content, dict):
        raise ValueError("Stage-3 evaluation report must be a JSON object")
    content = json.loads(json.dumps(content, allow_nan=False))
    content.pop("artifact_binding", None)
    content.pop("artifact_binding_verified", None)
    content.pop("visual_review_candidate", None)
    for key in ("promotion_gates", "promotion_thresholds"):
        value = content.get(key)
        if isinstance(value, dict):
            value.pop("artifact_binding_verified", None)
    return _mapping_sha256(content)


def _find_train_report(checkpoint_path: Path, payload_path: Path) -> Path:
    candidates = [
        checkpoint_path.parent / "train_report.json",
        payload_path.parent.parent.parent / "train_report.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError("Stage-3 checkpoint has no colocated train_report.json")


def _mapping_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _naturalness_snapshot(env: Any) -> dict[str, np.ndarray]:
    """Capture body/right-hand/racket state in comparable physical units."""

    if env.lab_state is None:
        raise ValueError("naturalness snapshot requires an active LAB state")
    kinematic_size = int(env.lab_state_builder.schema.kinematic_size)
    return {
        "body": np.asarray(env.lab_state[:kinematic_size], dtype=np.float64).copy(),
        "right_hand_site": np.asarray(env.data.site_xpos[env._palm_site], dtype=np.float64).copy(),
        "racket_position": np.asarray(env.data.site_xpos[env._stringbed_site], dtype=np.float64).copy(),
        "racket_rotation": np.asarray(env.data.site_xmat[env._stringbed_site], dtype=np.float64).reshape(3, 3).copy(),
    }


def _rollout_prior_naturalness(env: Any, *, feed_index: int) -> list[dict[str, np.ndarray]]:
    """Roll out the frozen prior mean on the same held-out feed."""

    obs, _ = env.reset(feed_index=int(feed_index))
    del obs
    trace: list[dict[str, np.ndarray]] = []
    terminated = truncated = False
    zero_task_action = np.zeros(env.action_size, dtype=np.float32)
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(zero_task_action)
        trace.append(_naturalness_snapshot(env))
    return trace


def _compare_naturalness_to_prior(
    task_trace: list[dict[str, np.ndarray]],
    prior_trace: list[dict[str, np.ndarray]],
) -> dict[str, float]:
    """Return absolute and motion-scale-normalized paired deviations."""

    shared = min(len(task_trace), len(prior_trace))
    if shared < 2:
        return {
            "paired_step_count": float(shared),
            "body_relative_deviation_to_prior": float("inf"),
            "right_hand_site_rmse_to_prior_m": float("inf"),
            "right_hand_site_relative_deviation_to_prior": float("inf"),
            "racket_position_rmse_to_prior_m": float("inf"),
            "racket_position_relative_deviation_to_prior": float("inf"),
            "racket_rotation_rmse_to_prior_rad": float("inf"),
            "racket_rotation_relative_deviation_to_prior": float("inf"),
        }
    task = task_trace[:shared]
    prior = prior_trace[:shared]

    def stack(name: str) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.stack([row[name] for row in task]),
            np.stack([row[name] for row in prior]),
        )

    body_task, body_prior = stack("body")
    body_rmse = float(np.sqrt(np.mean(np.square(body_task - body_prior))))
    body_scale = float(np.sqrt(np.mean(np.square(body_prior - body_prior[0]))))
    hand_task, hand_prior = stack("right_hand_site")
    hand_rmse = float(np.sqrt(np.mean(np.square(hand_task - hand_prior))))
    hand_scale = float(np.sqrt(np.mean(np.square(hand_prior - hand_prior[0]))))
    racket_task, racket_prior = stack("racket_position")
    racket_rmse = float(np.sqrt(np.mean(np.square(racket_task - racket_prior))))
    racket_scale = float(np.sqrt(np.mean(np.square(racket_prior - racket_prior[0]))))
    task_rot = [row["racket_rotation"] for row in task]
    prior_rot = [row["racket_rotation"] for row in prior]
    rotation_errors = [
        _rotation_distance_rad(current, baseline) for current, baseline in zip(task_rot, prior_rot, strict=True)
    ]
    rotation_excursion = [_rotation_distance_rad(prior_rot[0], current) for current in prior_rot]
    rotation_rmse = float(np.sqrt(np.mean(np.square(rotation_errors))))
    rotation_scale = float(np.sqrt(np.mean(np.square(rotation_excursion))))
    return {
        "paired_step_count": float(shared),
        "body_state_rmse_to_prior": body_rmse,
        "body_relative_deviation_to_prior": body_rmse / max(body_scale, 0.05),
        "right_hand_site_rmse_to_prior_m": hand_rmse,
        "right_hand_site_relative_deviation_to_prior": hand_rmse / max(hand_scale, 0.02),
        "racket_position_rmse_to_prior_m": racket_rmse,
        "racket_position_relative_deviation_to_prior": racket_rmse / max(racket_scale, 0.02),
        "racket_rotation_rmse_to_prior_rad": rotation_rmse,
        "racket_rotation_relative_deviation_to_prior": rotation_rmse / max(rotation_scale, 0.10),
    }


def _load_prior_direct_naturalness_baseline(lab: Any) -> dict[str, Any]:
    """Load the promoted prior-vs-direct tracking degradation evidence."""

    if lab is None:
        raise ValueError("prior/direct naturalness baseline requires LAB components")
    path = Path(lab.checkpoint_dir) / "closed_loop_metrics.json"
    if not path.is_file():
        raise ValueError(f"latent closed-loop naturalness baseline is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_fingerprint = lab.controller.control_manifest.get("latent_checkpoint_fingerprint")
    if payload.get("checkpoint_fingerprint") != expected_fingerprint:
        raise ValueError("latent closed-loop baseline belongs to another checkpoint")
    try:
        degradation = float(payload["body_racket_relative_degradation"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("latent closed-loop baseline lacks prior-vs-direct degradation") from exc
    if not math.isfinite(degradation) or degradation < 0.0:
        raise ValueError("prior-vs-direct degradation must be finite and non-negative")
    return {
        "schema_version": "stage3_prior_direct_naturalness_baseline_v1",
        "path": str(path.resolve()),
        "content_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "latent_checkpoint_fingerprint": expected_fingerprint,
        "prior_vs_direct_body_racket_relative_degradation": degradation,
        "direct_rollout_policy": payload.get("direct_rollout_policy"),
        "direct_rollout_metrics_path": payload.get("direct_rollout_metrics_path"),
    }


def _stage3_evaluation_summary(
    results: list[dict[str, Any]],
    *,
    gate_config: dict[str, Any],
    required_feed_count: int,
    lab_state_ood_values: list[float] | None = None,
    prior_direct_baseline: dict[str, Any] | None = None,
    task_profile: str = "legacy_v1",
    action_family: str = "fixed_synergy",
) -> dict[str, Any]:
    """Aggregate the exact report schema consumed by the pipeline gate."""
    if not results:
        raise ValueError("Stage-3 evaluation requires at least one episode")
    if action_family not in {"full_354", "fixed_synergy", "latent_direct_ablation"}:
        raise ValueError(f"unsupported Stage-3 action family: {action_family!r}")
    lab_metrics_applicable = action_family != "full_354"
    hit_rate = float(np.mean([1.0 if result["hit"] else 0.0 for result in results]))
    crossed_rate = float(np.mean([1.0 if result["crossed_net"] else 0.0 for result in results]))
    no_fall_rate = float(np.mean([0.0 if result["body_fall"] else 1.0 for result in results]))
    back_rate = float(np.mean([1.0 if result["landing_region"] == "opponent_back" else 0.0 for result in results]))
    contact_speeds = [float(result.get("contact_racket_head_speed_m_s", float("nan"))) for result in results]
    racket_speed = (
        float(np.mean(contact_speeds))
        if all(math.isfinite(value) and value >= 0.0 for value in contact_speeds)
        else float("-inf")
    )
    clearances = [result["net_clearance_m"] for result in results if result["net_clearance_m"] is not None]
    mean_clearance = float(np.mean(clearances)) if clearances else float("-inf")
    minimum_thresholds = {
        "evaluated_feed_count": float(required_feed_count),
        "no_fall_rate": float(gate_config.get("min_no_fall_rate", 0.95)),
        "hit_rate": float(gate_config.get("min_hit_rate", 0.90)),
        "crossed_net_rate": float(gate_config.get("min_crossed_net_rate", 0.85)),
        "opponent_back_landing_rate": float(gate_config.get("min_opponent_back_landing_rate", 0.70)),
        "racket_head_speed_m_s": float(gate_config.get("min_racket_head_speed_m_s", 8.0)),
        "net_clearance_m": float(gate_config.get("min_net_clearance_m", 0.25)),
        "control_finite": float(gate_config.get("min_control_finite", 1.0)),
        "min_root_height_m": float(gate_config.get("min_root_height_m", 0.55)),
    }
    maximum_thresholds = {
        "body_action_saturation_fraction": float(gate_config.get("max_body_action_saturation_fraction", 0.01)),
        "full_action_saturation_fraction": float(gate_config.get("max_full_action_saturation_fraction", 0.01)),
        "normalized_control_energy": float(gate_config.get("max_normalized_control_energy", 0.35)),
        "max_attachment_translation_drift_m": float(gate_config.get("max_attachment_translation_drift_m", 0.005)),
        "max_attachment_rotation_drift_rad": float(gate_config.get("max_attachment_rotation_drift_rad", 0.05)),
    }
    if lab_metrics_applicable:
        maximum_thresholds.update(
            {
                "raw_latent_saturation": float(gate_config.get("max_raw_latent_saturation_fraction", 0.10)),
                "lab_state_ood_fraction_p95": float(
                    gate_config.get(
                        "max_lab_state_ood_fraction_p95",
                        gate_config.get("max_lab_state_ood_fraction", 0.01),
                    )
                ),
                "body_relative_deviation_to_prior": float(
                    gate_config.get("max_body_relative_deviation_to_prior", 0.25)
                ),
                "right_hand_site_rmse_to_prior_m": float(gate_config.get("max_right_hand_site_rmse_to_prior_m", 0.12)),
                "right_hand_site_relative_deviation_to_prior": float(
                    gate_config.get("max_right_hand_site_relative_deviation_to_prior", 0.25)
                ),
                "racket_position_rmse_to_prior_m": float(gate_config.get("max_racket_position_rmse_to_prior_m", 0.12)),
                "racket_position_relative_deviation_to_prior": float(
                    gate_config.get("max_racket_position_relative_deviation_to_prior", 0.25)
                ),
                "racket_rotation_rmse_to_prior_rad": float(
                    gate_config.get("max_racket_rotation_rmse_to_prior_rad", 0.35)
                ),
                "racket_rotation_relative_deviation_to_prior": float(
                    gate_config.get("max_racket_rotation_relative_deviation_to_prior", 0.25)
                ),
                "prior_vs_direct_body_racket_relative_degradation": float(
                    gate_config.get("max_prior_vs_direct_body_racket_relative_degradation", 0.10)
                ),
                "stage3_vs_direct_naturalness_upper_bound": float(
                    gate_config.get("max_stage3_vs_direct_naturalness_upper_bound", 0.375)
                ),
            }
        )
    if task_profile == "impact_recovery_v2":
        minimum_thresholds.update(
            {
                "center_hit_rate": float(gate_config.get("min_center_hit_rate", 0.75)),
                "recovery_ready_rate": float(gate_config.get("min_recovery_ready_rate", 0.85)),
            }
        )
        maximum_thresholds.update(
            {
                "impact_position_error_m": float(gate_config.get("max_impact_position_error_m", 0.12)),
                "impact_timing_mae_s": float(gate_config.get("max_impact_timing_mae_s", 0.08)),
                "stringbed_normal_error_rad": float(gate_config.get("max_stringbed_normal_error_rad", 0.35)),
                "racket_linear_velocity_rmse_m_s": float(gate_config.get("max_racket_linear_velocity_rmse_m_s", 2.0)),
                "racket_angular_velocity_rmse_rad_s": float(
                    gate_config.get("max_racket_angular_velocity_rmse_rad_s", 8.0)
                ),
                "landing_rmse_m": float(gate_config.get("max_landing_rmse_m", 0.85)),
                "apex_mae_m": float(gate_config.get("max_apex_mae_m", 0.40)),
            }
        )

    def diagnostic_mean(name: str, *, missing: float = float("inf")) -> float:
        values: list[float] = []
        for result in results:
            diagnostics = result.get("lab_diagnostics")
            if not isinstance(diagnostics, dict) or name not in diagnostics:
                return missing
            try:
                value = float(diagnostics[name])
            except (TypeError, ValueError):
                return missing
            if not math.isfinite(value):
                return missing
            values.append(value)
        return float(np.mean(values))

    def episode_max(name: str) -> float:
        values: list[float] = []
        for result in results:
            try:
                value = float(result[name])
            except (KeyError, TypeError, ValueError):
                return float("inf")
            if not math.isfinite(value):
                return float("inf")
            values.append(value)
        return float(max(values))

    def episode_min(name: str) -> float:
        values: list[float] = []
        for result in results:
            try:
                value = float(result[name])
            except (KeyError, TypeError, ValueError):
                return float("-inf")
            if not math.isfinite(value):
                return float("-inf")
            values.append(value)
        return float(min(values))

    if not lab_metrics_applicable:
        ood_values = []
    elif lab_state_ood_values is None:
        ood_values = []
        for result in results:
            diagnostics = result.get("lab_diagnostics")
            if not isinstance(diagnostics, dict) or "lab_state_ood_fraction" not in diagnostics:
                ood_values = []
                break
            try:
                value = float(diagnostics["lab_state_ood_fraction"])
            except (TypeError, ValueError):
                ood_values = []
                break
            if not math.isfinite(value):
                ood_values = []
                break
            ood_values.append(value)
    else:
        ood_values = [float(value) for value in lab_state_ood_values]
        if not ood_values or not all(math.isfinite(value) for value in ood_values):
            ood_values = []
    ood_p95 = float(np.quantile(ood_values, 0.95)) if ood_values else (float("inf") if lab_metrics_applicable else None)
    ood_max = float(max(ood_values)) if ood_values else (float("inf") if lab_metrics_applicable else None)

    promotion_metrics = {
        "evaluated_feed_count": float(len(results)),
        "no_fall_rate": no_fall_rate,
        "hit_rate": hit_rate,
        "crossed_net_rate": crossed_rate,
        "opponent_back_landing_rate": back_rate,
        "racket_head_speed_m_s": racket_speed,
        "net_clearance_m": mean_clearance,
        "control_finite": diagnostic_mean("control_finite", missing=float("-inf")),
        "min_root_height_m": episode_min("min_root_height_m"),
        "body_action_saturation_fraction": diagnostic_mean("body_action_saturation_fraction"),
        "full_action_saturation_fraction": diagnostic_mean("full_action_saturation_fraction"),
        "normalized_control_energy": diagnostic_mean("normalized_control_energy"),
        "raw_latent_saturation": (diagnostic_mean("raw_latent_saturation") if lab_metrics_applicable else None),
        "lab_state_ood_fraction": (diagnostic_mean("lab_state_ood_fraction") if lab_metrics_applicable else None),
        "lab_state_ood_fraction_p95": ood_p95,
        "lab_state_ood_fraction_max": ood_max,
        "lab_state_unclipped_z_rms": (diagnostic_mean("lab_state_unclipped_z_rms") if lab_metrics_applicable else None),
        "max_attachment_translation_drift_m": episode_max("max_attachment_translation_drift_m"),
        "max_attachment_rotation_drift_rad": episode_max("max_attachment_rotation_drift_rad"),
    }
    if task_profile == "impact_recovery_v2":

        def v2_values(name: str, *, require_hit: bool = False) -> list[float]:
            values: list[float] = []
            for result in results:
                if require_hit and not bool(result.get("hit", False)):
                    continue
                metrics = result.get("stage3_v2_metrics")
                if not isinstance(metrics, dict) or name not in metrics:
                    continue
                try:
                    value = float(metrics[name])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    values.append(value)
            return values

        impact_position = v2_values("impact_position_error_m", require_hit=True)
        impact_rho2 = v2_values("impact_rho2", require_hit=True)
        impact_timing = v2_values("impact_timing_error_s", require_hit=True)
        normal_error = v2_values("stringbed_normal_error_rad", require_hit=True)
        linear_error = v2_values("racket_linear_velocity_error_m_s", require_hit=True)
        angular_error = v2_values("racket_angular_velocity_error_rad_s", require_hit=True)
        landing_error = v2_values("landing_error_m")
        apex_error = v2_values("apex_error_m")
        ready_error = v2_values("ready_pose_error")
        missing = float("inf")
        promotion_metrics.update(
            {
                "evaluated_episode_count": float(len(results)),
                "impact_position_error_m": (float(np.mean(impact_position)) if impact_position else missing),
                "center_hit_rate": float(np.mean([value <= 0.25 for value in impact_rho2])) if impact_rho2 else 0.0,
                "impact_timing_mae_s": (float(np.mean(impact_timing)) if impact_timing else missing),
                "stringbed_normal_error_rad": (float(np.mean(normal_error)) if normal_error else missing),
                "racket_linear_velocity_rmse_m_s": (
                    float(np.sqrt(np.mean(np.square(linear_error)))) if linear_error else missing
                ),
                "racket_angular_velocity_rmse_rad_s": (
                    float(np.sqrt(np.mean(np.square(angular_error)))) if angular_error else missing
                ),
                "landing_rmse_m": (float(np.sqrt(np.mean(np.square(landing_error)))) if landing_error else missing),
                "apex_mae_m": (float(np.mean(apex_error)) if apex_error else missing),
                "recovery_ready_rate": float(
                    np.mean(
                        [
                            bool(result.get("recovery_complete", False))
                            and index < len(ready_error)
                            and ready_error[index] <= 0.15
                            for index, result in enumerate(results)
                        ]
                    )
                )
                if len(ready_error) == len(results)
                else 0.0,
            }
        )
    naturalness_names = (
        "body_relative_deviation_to_prior",
        "right_hand_site_rmse_to_prior_m",
        "right_hand_site_relative_deviation_to_prior",
        "racket_position_rmse_to_prior_m",
        "racket_position_relative_deviation_to_prior",
        "racket_rotation_rmse_to_prior_rad",
        "racket_rotation_relative_deviation_to_prior",
    )
    for name in naturalness_names:
        if not lab_metrics_applicable:
            promotion_metrics[name] = None
            continue
        values: list[float] = []
        for result in results:
            naturalness = result.get("naturalness")
            if not isinstance(naturalness, dict):
                values = []
                break
            try:
                value = float(naturalness[name])
            except (KeyError, TypeError, ValueError):
                values = []
                break
            if not math.isfinite(value):
                values = []
                break
            values.append(value)
        promotion_metrics[name] = float(np.mean(values)) if values else float("inf")
    if lab_metrics_applicable:
        try:
            prior_vs_direct = float((prior_direct_baseline or {})["prior_vs_direct_body_racket_relative_degradation"])
        except (KeyError, TypeError, ValueError):
            prior_vs_direct = float("inf")
        stage3_vs_prior = max(
            promotion_metrics["body_relative_deviation_to_prior"],
            promotion_metrics["right_hand_site_relative_deviation_to_prior"],
            promotion_metrics["racket_position_relative_deviation_to_prior"],
            promotion_metrics["racket_rotation_relative_deviation_to_prior"],
        )
        stage3_vs_direct = (1.0 + stage3_vs_prior) * (1.0 + prior_vs_direct) - 1.0
    else:
        prior_vs_direct = None
        stage3_vs_direct = None
    promotion_metrics["prior_vs_direct_body_racket_relative_degradation"] = prior_vs_direct
    promotion_metrics["stage3_vs_direct_naturalness_upper_bound"] = stage3_vs_direct
    gates = {name: promotion_metrics[name] >= threshold for name, threshold in minimum_thresholds.items()}
    gates.update({name: promotion_metrics[name] <= threshold for name, threshold in maximum_thresholds.items()})
    thresholds = {**minimum_thresholds, **maximum_thresholds}
    return {
        "action_family": action_family,
        "lab_metrics_applicable": lab_metrics_applicable,
        "not_applicable_metrics": (
            []
            if lab_metrics_applicable
            else [
                "raw_latent_saturation",
                "lab_state_ood_fraction",
                "lab_state_ood_fraction_p95",
                "lab_state_ood_fraction_max",
                "lab_state_unclipped_z_rms",
                *naturalness_names,
                "prior_vs_direct_body_racket_relative_degradation",
                "stage3_vs_direct_naturalness_upper_bound",
            ]
        ),
        "evaluated_feed_count": len(results),
        "required_heldout_feed_count": int(required_feed_count),
        "no_fall_rate": no_fall_rate,
        "hit_rate": hit_rate,
        "crossed_net_rate": crossed_rate,
        "opponent_back_landing_rate": back_rate,
        "mean_racket_head_speed_m_s": racket_speed,
        "mean_contact_racket_head_speed_m_s": racket_speed,
        "mean_net_clearance_m": mean_clearance,
        # Canonical aliases consumed by musclemimic.badminton.training_gates.
        "racket_head_speed_m_s": racket_speed,
        "net_clearance_m": mean_clearance,
        **(
            {
                name: promotion_metrics[name]
                for name in (
                    "evaluated_episode_count",
                    "impact_position_error_m",
                    "center_hit_rate",
                    "impact_timing_mae_s",
                    "stringbed_normal_error_rad",
                    "racket_linear_velocity_rmse_m_s",
                    "racket_angular_velocity_rmse_rad_s",
                    "landing_rmse_m",
                    "apex_mae_m",
                    "recovery_ready_rate",
                )
            }
            if task_profile == "impact_recovery_v2"
            else {}
        ),
        **{name: promotion_metrics[name] for name in maximum_thresholds},
        "control_finite": promotion_metrics["control_finite"],
        "min_root_height_m": promotion_metrics["min_root_height_m"],
        "raw_latent_saturation": promotion_metrics["raw_latent_saturation"],
        "lab_state_unclipped_z_rms": promotion_metrics["lab_state_unclipped_z_rms"],
        "lab_state_ood_fraction": promotion_metrics["lab_state_ood_fraction"],
        "lab_state_ood_fraction_max": promotion_metrics["lab_state_ood_fraction_max"],
        "naturalness": {
            name: promotion_metrics[name]
            for name in (
                *naturalness_names,
                "prior_vs_direct_body_racket_relative_degradation",
                "stage3_vs_direct_naturalness_upper_bound",
            )
        },
        "promotion_gates": gates,
        "promotion_thresholds": thresholds,
        "passed": all(gates.values()),
    }


def _return_net_clearance(
    *,
    previous_shuttle_x: float,
    shuttle_xyz: np.ndarray,
    hit_registered: bool,
) -> float | None:
    """Measure only the player-to-opponent crossing after a valid hit."""
    shuttle = np.asarray(shuttle_xyz, dtype=float)
    if shuttle.shape != (3,):
        raise ValueError(f"shuttle_xyz must have shape (3,), got {shuttle.shape}")
    if not bool(hit_registered):
        return None
    if float(previous_shuttle_x) <= 0.0 < float(shuttle[0]):
        return float(shuttle[2] - 1.55)
    return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.floating):
        scalar = float(value)
        return scalar if math.isfinite(scalar) else None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        default="experiments/posttrain/incoming_shuttle_hit_v1.yaml",
    )
    parser.add_argument(
        "--stage",
        choices=(
            "preflight",
            "feed-check",
            "base-only-check",
            "physics-smoke",
            "train-tiny",
            "train",
            "train-gpu",
            "evaluate",
        ),
        default="preflight",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--checkpoint", default=None, help="checkpoint npz for evaluate")
    parser.add_argument(
        "--target-bank",
        default=None,
        help="override Stage-3 v2 training target bank without editing the canonical spec",
    )
    parser.add_argument(
        "--eval-target-bank",
        default=None,
        help="override Stage-3 v2 held-out target bank without editing the canonical spec",
    )
    parser.add_argument(
        "--feed-bank",
        default=None,
        help="override the deterministic training feed bank without editing the canonical spec",
    )
    parser.add_argument(
        "--eval-feed-bank",
        default=None,
        help="override the deterministic held-out feed bank without editing the canonical spec",
    )
    parser.add_argument("--steps", type=int, default=256, help="total PPO steps for train-tiny")
    parser.add_argument(
        "--base-only-steps",
        type=int,
        default=None,
        help="prior-mean control steps per base-only rollout (default: spec)",
    )
    parser.add_argument("--rollout-steps", type=int, default=64, help="rollout length for train-tiny/train-gpu")
    parser.add_argument("--episodes", type=int, default=None, help="episode count override")
    parser.add_argument("--num-envs", type=int, default=512, help="parallel envs for train-gpu")
    parser.add_argument(
        "--total-env-steps", type=int, default=None, help="env steps for train-gpu (default: spec ppo.total_steps)"
    )
    parser.add_argument("--impl", choices=("jax", "warp"), default="warp", help="MJX backend for train-gpu")
    parser.add_argument(
        "--base-policy-artifact", default=None, help="frozen base policy export dir (Stage 3 residual mode)"
    )
    parser.add_argument("--residual-scale", type=float, default=0.3)
    parser.add_argument("--base-skill", default=None, help="skill name for a multi-skill base")
    parser.add_argument("--latent-checkpoint", default=None, help="override Stage-3 latent checkpoint dir")
    parser.add_argument(
        "--allow-unpromoted-latent",
        action="store_true",
        help="testing only: bypass the latent promotion gate",
    )
    parser.add_argument("--resume-from", default=None, help="complete train-gpu checkpoint to resume")
    parser.add_argument(
        "--curriculum-max-stage",
        default=None,
        help="Clamp impact_recovery_v2 training at a canonical C0--C7 stage.",
    )
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument(
        "--export-simulation-npz",
        default=None,
        help=(
            "during final Stage-3 evaluate, export real-impact-aligned physical "
            "muscle/joint signals for EMG and physiology validation"
        ),
    )
    parser.add_argument(
        "--signal-identity-json",
        default=None,
        help=(
            "held-out feed/trial/subject/session manifest; only its listed feeds are captured "
            "while the canonical held-out evaluation still runs in full"
        ),
    )
    parser.add_argument(
        "--policy-evidence-json",
        default=None,
        help="sealed Stage-3 paired comparison selecting the policy to export",
    )
    parser.add_argument(
        "--signal-sidecar-json",
        default=None,
        help="optional signal export manifest path (default: NPZ basename.manifest.json)",
    )
    parser.add_argument("--signal-pre-impact-s", type=float, default=0.5)
    parser.add_argument("--signal-post-impact-s", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    signal_export_arguments = (
        args.export_simulation_npz,
        args.signal_identity_json,
        args.policy_evidence_json,
    )
    if any(value is not None for value in signal_export_arguments) and not all(
        value is not None for value in signal_export_arguments
    ):
        parser.error(
            "--export-simulation-npz, --signal-identity-json and --policy-evidence-json must be supplied together"
        )
    if args.export_simulation_npz is not None and args.stage != "evaluate":
        parser.error("--export-simulation-npz is valid only with --stage evaluate")

    paths = load_incoming_hit_spec(args.spec)
    if any(
        value is not None
        for value in (
            args.target_bank,
            args.eval_target_bank,
            args.feed_bank,
            args.eval_feed_bank,
        )
    ):
        paths = replace(
            paths,
            feed_bank_path=(paths.feed_bank_path if args.feed_bank is None else _resolve(args.feed_bank)),
            eval_feed_bank_path=(
                paths.eval_feed_bank_path if args.eval_feed_bank is None else _resolve(args.eval_feed_bank)
            ),
            target_bank_path=(paths.target_bank_path if args.target_bank is None else _resolve(args.target_bank)),
            eval_target_bank_path=(
                paths.eval_target_bank_path if args.eval_target_bank is None else _resolve(args.eval_target_bank)
            ),
        )
    if args.stage == "preflight":
        report = preflight(paths, out_dir=args.out_dir)
    elif args.stage == "feed-check":
        report = feed_check(paths, out_dir=args.out_dir)
    elif args.stage == "base-only-check":
        report = base_only_check(
            paths,
            out_dir=args.out_dir,
            latent_checkpoint=args.latent_checkpoint,
            episodes=args.episodes,
            steps=args.base_only_steps,
        )
    elif args.stage == "physics-smoke":
        report = physics_smoke(
            paths,
            out_dir=args.out_dir,
            episodes=3 if args.episodes is None else args.episodes,
            record_video=args.record_video,
        )
    elif args.stage == "train-tiny":
        report = train_tiny(
            paths,
            out_dir=args.out_dir,
            total_steps=args.steps,
            rollout_steps=args.rollout_steps,
            seed=args.seed,
            device=args.device,
        )
    elif args.stage == "evaluate":
        report = evaluate(
            paths,
            checkpoint=args.checkpoint,
            out_dir=args.out_dir,
            episodes=(
                int(paths.evaluation.get("heldout_feed_count", paths.eval_feed_bank_size))
                if args.episodes is None
                else args.episodes
            ),
            record_video=args.record_video,
            export_simulation_npz=args.export_simulation_npz,
            signal_identity_json=args.signal_identity_json,
            policy_evidence_json=args.policy_evidence_json,
            signal_sidecar_json=args.signal_sidecar_json,
            signal_pre_impact_s=args.signal_pre_impact_s,
            signal_post_impact_s=args.signal_post_impact_s,
        )
    elif args.stage == "train-gpu":
        report = train_gpu(
            paths,
            out_dir=args.out_dir,
            num_envs=args.num_envs,
            rollout_steps=args.rollout_steps,
            total_env_steps=args.total_env_steps,
            impl=args.impl,
            base_policy_artifact=args.base_policy_artifact,
            residual_scale=args.residual_scale,
            base_skill=args.base_skill,
            latent_checkpoint=args.latent_checkpoint,
            allow_unpromoted_latent=args.allow_unpromoted_latent,
            resume_from=args.resume_from,
            curriculum_max_stage=args.curriculum_max_stage,
            seed=args.seed,
        )
    else:
        report = train(paths, out_dir=args.out_dir, seed=args.seed, device=args.device)

    print(json.dumps(_json_safe(report), indent=2, sort_keys=True, allow_nan=False))
    return 0 if report.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())

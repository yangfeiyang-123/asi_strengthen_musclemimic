from __future__ import annotations

import numpy as np
import pytest

from musclemimic.distill.obs_filter import StudentObsSpec
from musclemimic.latent_muscle.closed_loop_eval import (
    ClosedLoopEvalConfig,
    evaluate_latent_closed_loop,
    select_direct_rollout_policy,
)


class _TrajectoryHandler:
    n_trajectories = 2
    random_start = True
    use_fixed_start = False
    start_from_random_step = True
    fixed_start_conf = [0, 0]


class _FakeEnv:
    def __init__(self):
        self.th = _TrajectoryHandler()
        self.step_no = 0

    def reset(self):
        self.step_no = 0
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def step(self, action):
        assert np.asarray(action).shape == (1, 2)
        self.step_no += 1
        done = self.step_no == 3
        return (
            np.array([float(self.step_no), 0.0, self.step_no / 3.0], dtype=np.float32),
            1.0,
            done,
            done,
            {
                "traj_len": 3,
                "subtraj_step_no": self.step_no - 1,
                "err_rpos": 0.1,
                "err_racket_pos": 0.02,
                "err_racket_rot": 0.04,
            },
        )


class _FakeRuntime:
    checkpoint_fingerprint = "a" * 64
    sigma_min = 0.05
    sigma_max = 2.0

    def prior_raw_numpy(self, state):
        batch = np.asarray(state).shape[0]
        return np.zeros((batch, 2), dtype=np.float32), np.zeros((batch, 2), dtype=np.float32)

    def decoder_numpy(self, state, latent):
        return np.zeros((np.asarray(state).shape[0], 2), dtype=np.float32)


def test_closed_loop_evaluator_runs_prior_mean_and_lambda_sweeps():
    spec = StudentObsSpec(
        raw_obs_dim=3,
        goal_indices=np.array([2]),
        state_indices=np.array([0, 1]),
        student_indices=np.array([0, 1, 2]),
        phase_index=2,
    )
    report = evaluate_latent_closed_loop(
        env=_FakeEnv(),
        runtime=_FakeRuntime(),
        student_obs_spec=spec,
        config=ClosedLoopEvalConfig(lambdas=(0.0, 0.25, 0.5), seed=7),
        direct_bc_metrics={
            "val_err_rpos": 0.1,
            "val_err_racket_pos": 0.02,
            "val_err_racket_rot": 0.04,
        },
    )

    assert report["num_trajectories"] == 2
    assert report["fall_or_early_termination_rate"] == 0.0
    assert report["lambda_025_050_no_fall_rate"] == 1.0
    assert report["prior_mean_frame_coverage"] == 1.0
    assert report["body_racket_relative_degradation"] == 0.0
    assert set(report["by_lambda"]) == {"lambda_0p000", "lambda_0p250", "lambda_0p500"}


def test_closed_loop_report_keys_match_production_promotion_thresholds():
    from musclemimic.latent_muscle.train_latent import (
        LatentTrainConfig,
        _evaluate_promotion_gates,
    )

    metrics = {
        "prior_posterior_mse_ratio": 1.0,
        "active_latent_fraction": 0.5,
        "prior_sigma_min_clamp_fraction": 0.0,
        "prior_sigma_max_clamp_fraction": 0.0,
        "decoder_saturation_fraction": 0.0,
        "posterior_action_mse": 0.01,
        "closed_loop_fall_or_early_termination_rate": 0.01,
        "closed_loop_body_racket_relative_degradation": 0.05,
        "closed_loop_lambda_025_050_no_fall_rate": 0.99,
        "closed_loop_evidence_kind": "verified_production_v2",
    }
    config = LatentTrainConfig(
        dataset_dir="unused",
        output_dir="unused",
        direct_bc_action_mse=0.01,
        require_direct_bc_baseline=True,
        require_closed_loop_metrics=True,
        promotion_gates={
            "closed_loop_max_fall_or_early_termination_rate": 0.05,
            "closed_loop_max_body_racket_relative_degradation": 0.10,
            "closed_loop_min_lambda_025_050_no_fall_rate": 0.95,
        },
    )
    promotion = _evaluate_promotion_gates(metrics, config)

    assert promotion["passed"] is True
    assert promotion["checks"]["closed_loop_fall_or_early_termination_rate"] is True
    assert promotion["checks"]["closed_loop_body_racket_relative_degradation"] is True
    assert promotion["checks"]["closed_loop_lambda_025_050_no_fall_rate"] is True

    metrics["closed_loop_evidence_kind"] = "test_only_injected"
    rejected = _evaluate_promotion_gates(metrics, config)
    assert rejected["passed"] is False
    assert rejected["checks"]["production_closed_loop_evidence"] is False

    nonproduction_config = LatentTrainConfig(
        dataset_dir="unused",
        output_dir="unused",
        direct_bc_action_mse=0.01,
    )
    still_rejected = _evaluate_promotion_gates(metrics, nonproduction_config)
    assert still_rejected["passed"] is False
    assert still_rejected["checks"]["test_only_evidence_not_promotable"] is False


def test_direct_rollout_metrics_select_promoted_policy_in_priority_order():
    payload = {
        "teacher": {"val_err_rpos": 0.01, "val_err_racket_pos": 0.01, "val_err_racket_rot": 0.01},
        "student_bc": {"val_err_rpos": 0.03, "val_err_racket_pos": 0.03, "val_err_racket_rot": 0.03},
        "student_bc_dagger": {"val_err_rpos": 0.02, "val_err_racket_pos": 0.02, "val_err_racket_rot": 0.02},
        "student_bc_ppo": {"val_err_rpos": 0.015, "val_err_racket_pos": 0.015, "val_err_racket_rot": 0.015},
    }
    name, metrics = select_direct_rollout_policy(payload)
    assert name == "student_bc_ppo"
    assert metrics["val_err_rpos"] == 0.015

    name, metrics = select_direct_rollout_policy(payload, "student_bc_dagger")
    assert name == "student_bc_dagger"
    assert metrics["val_err_rpos"] == 0.02

    payload["promotion_policy"] = "student_bc"
    name, metrics = select_direct_rollout_policy(payload)
    assert name == "student_bc"
    assert metrics["val_err_rpos"] == 0.03


def test_evaluation_horizon_is_not_counted_as_early_termination():
    class LongEnv(_FakeEnv):
        def step(self, action):
            assert np.asarray(action).shape == (1, 2)
            self.step_no += 1
            return (
                np.array([float(self.step_no), 0.0, 0.1], dtype=np.float32),
                1.0,
                False,
                False,
                {
                    "traj_len": 1000,
                    "subtraj_step_no": self.step_no - 1,
                    "err_rpos": 0.1,
                    "err_racket_pos": 0.02,
                    "err_racket_rot": 0.04,
                },
            )

    spec = StudentObsSpec(
        raw_obs_dim=3,
        goal_indices=np.array([2]),
        state_indices=np.array([0, 1]),
        student_indices=np.array([0, 1, 2]),
        phase_index=2,
    )
    report = evaluate_latent_closed_loop(
        env=LongEnv(),
        runtime=_FakeRuntime(),
        student_obs_spec=spec,
        config=ClosedLoopEvalConfig(lambdas=(0.0,), max_steps=3),
        direct_bc_metrics={
            "err_rpos": 0.1,
            "err_racket_pos": 0.02,
            "err_racket_rot": 0.04,
        },
    )

    assert report["fall_or_early_termination_rate"] == 0.0
    assert report["prior_mean_frame_coverage"] == 1.0
    assert report["max_steps"] == 3


def test_body_racket_degradation_requires_all_three_metrics():
    from musclemimic.latent_muscle.closed_loop_eval import body_racket_relative_degradation

    assert body_racket_relative_degradation(
        {"err_rpos": 0.1, "err_racket_pos": 0.02},
        {"err_rpos": 0.1, "err_racket_pos": 0.02},
    ) is None


def test_comparison_metrics_missing_racket_channel_fails_closed():
    import pytest

    with pytest.raises(ValueError, match="err_racket_rot"):
        select_direct_rollout_policy(
            {"student_bc": {"err_rpos": 0.1, "err_racket_pos": 0.02}}
        )


def test_bare_closed_loop_scalar_json_cannot_drive_production_gate(tmp_path):
    import json

    from omegaconf import OmegaConf

    from fullbody.latent_closed_loop_eval import _merge_and_update_promotion

    payload = OmegaConf.to_container(
        OmegaConf.load("fullbody/config_specific_task/distill/latent_forehandclear_lab.yaml"),
        resolve=True,
    )["latent_distill"]
    payload["direct_bc_action_mse"] = 0.01
    (tmp_path / "latent_config.yaml").write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "eval_metrics.json").write_text(
        json.dumps(
            {
                "prior_posterior_mse_ratio": 1.0,
                "active_latent_fraction": 0.5,
                "prior_sigma_min_clamp_fraction": 0.0,
                "prior_sigma_max_clamp_fraction": 0.0,
                "decoder_saturation_fraction": 0.0,
                "posterior_action_mse": 0.01,
            }
        ),
        encoding="utf-8",
    )
    import pytest

    with pytest.raises(ValueError, match="latent_closed_loop_eval_v2"):
        _merge_and_update_promotion(
            tmp_path,
            {
                "fall_or_early_termination_rate": 0.01,
                "body_racket_relative_degradation": 0.05,
                "lambda_025_050_no_fall_rate": 0.99,
            },
        )


def _strict_promotion_fixture(tmp_path):
    import copy
    import json

    from musclemimic.distill.motion_identity import stable_motion_uid
    from musclemimic.distill.provenance import (
        canonical_json_sha256,
        checkpoint_content_fingerprint,
        file_sha256,
    )
    from musclemimic.latent_muscle.checkpoint import latent_checkpoint_fingerprint
    from musclemimic.latent_muscle.train_latent import (
        LatentTrainConfig,
        _evaluate_promotion_gates,
    )

    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    (teacher_dir / "weights.bin").write_bytes(b"teacher")
    teacher = checkpoint_content_fingerprint(teacher_dir, canonicalize=False)
    teacher_promotion = {
        "schema_version": "stage2_teacher_promotion_binding_v1",
        "path": str((tmp_path / "stage2-promotion.json").resolve()),
        "content_sha256": "e" * 64,
        "binding_sha256": "f" * 64,
        "stage": "stage2",
        "teacher_checkpoint_sha256": teacher["sha256"],
        "artifact": {},
    }
    direct_dir = tmp_path / "direct"
    direct_dir.mkdir()
    (direct_dir / "weights.bin").write_bytes(b"direct")
    direct = checkpoint_content_fingerprint(direct_dir, canonicalize=False)

    comparison = tmp_path / "comparison_metrics.json"
    acceptance = tmp_path / "acceptance.json"
    convergence = tmp_path / "convergence.json"
    temporal = tmp_path / "temporal_audit.json"
    comparison.write_text("{}", encoding="utf-8")
    acceptance.write_text(
        json.dumps(
            {
                "student_bc": {
                    "passed": True,
                    "failed": [],
                    "missing": [],
                    "values": {
                        "return_ratio": 1.0,
                        "completion_ratio": 1.0,
                        "early_termination_delta": 0.0,
                        "err_rpos_relative_degradation": 0.0,
                        "err_racket_pos_relative_degradation": 0.0,
                        "err_racket_rot_relative_degradation": 0.0,
                        "convergence_normalized_abs_slope": 0.0,
                        "convergence_normalized_span": 0.0,
                        "temporal_usable_sequence_count": 5.0,
                        "temporal_best_lag_steps": 0.0,
                        "temporal_max_abs_motion_best_lag_steps": 0.0,
                        "temporal_lag_mse_improvement_fraction": 0.0,
                    },
                    "thresholds": {"min_return_ratio": 0.9},
                    "convergence": {
                        "passed": True,
                        "evidence_checks": {"fixed_split": True},
                    },
                    "temporal": {"passed": True, "checks": {"zero_lag": True}},
                }
            }
        ),
        encoding="utf-8",
    )
    convergence.write_text("{}", encoding="utf-8")
    temporal.write_text("{}", encoding="utf-8")

    motions = [f"ForehandClear/raw/heldout_{index}" for index in range(5)]
    uids = [int(stable_motion_uid(path)) for path in motions]
    evidence = {
        "schema_version": "direct_distill_promotion_evidence_v2",
        "promotion_policy": "student_bc",
        "deterministic": True,
        "teacher_checkpoint": teacher,
        "teacher_promotion": teacher_promotion,
        "student_checkpoint": direct,
        "heldout": {"motion_paths": motions, "motion_uids": uids},
        "artifacts": {
            "comparison_metrics": {
                "path": str(comparison.resolve()),
                "sha256": file_sha256(comparison),
            },
            "acceptance": {
                "path": str(acceptance.resolve()),
                "sha256": file_sha256(acceptance),
            },
            "convergence": {
                "path": str(convergence.resolve()),
                "sha256": file_sha256(convergence),
            },
            "temporal_audit": {
                "path": str(temporal.resolve()),
                "sha256": file_sha256(temporal),
            },
        },
    }
    evidence["evidence_fingerprint"] = canonical_json_sha256(evidence)
    evidence_path = tmp_path / "direct_promotion_evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    latent = tmp_path / "latent"
    latent.mkdir()
    (latent / "prior.msgpack").write_bytes(b"prior")
    (latent / "decoder.msgpack").write_bytes(b"decoder")
    dataset_manifest = {
        "manifest_fingerprint": "d" * 64,
        "teacher_checkpoint": teacher,
        "teacher_promotion": teacher_promotion,
    }
    (latent / "training_provenance.json").write_text(
        json.dumps(
            {
                "schema_version": "latent_training_provenance_v1",
                "dataset_manifest": dataset_manifest,
                "dataset_manifest_fingerprint": dataset_manifest["manifest_fingerprint"],
                "teacher_checkpoint": teacher,
                "teacher_promotion": teacher_promotion,
            }
        ),
        encoding="utf-8",
    )
    config = LatentTrainConfig(
        dataset_dir="unused",
        output_dir="unused",
        require_closed_loop_metrics=True,
        promotion_gates={
            "closed_loop_max_fall_or_early_termination_rate": 0.05,
            "closed_loop_max_body_racket_relative_degradation": 0.10,
            "closed_loop_min_lambda_025_050_no_fall_rate": 0.95,
        },
    )
    config_payload = {
        "dataset_dir": config.dataset_dir,
        "output_dir": config.output_dir,
        "require_closed_loop_metrics": True,
        "promotion_gates": config.promotion_gates,
    }
    (latent / "latent_config.yaml").write_text(json.dumps(config_payload), encoding="utf-8")
    offline = {
        "prior_posterior_mse_ratio": 1.0,
        "active_latent_fraction": 0.5,
        "prior_sigma_min_clamp_fraction": 0.0,
        "prior_sigma_max_clamp_fraction": 0.0,
        "decoder_saturation_fraction": 0.0,
        "posterior_action_mse": 0.01,
    }
    current_eval = dict(offline)
    current_eval.update(
        {
            "closed_loop_fall_or_early_termination_rate": 0.0,
            "closed_loop_body_racket_relative_degradation": 0.0,
            "closed_loop_lambda_025_050_no_fall_rate": 1.0,
            "closed_loop_evidence_kind": "verified_production_v2",
            "teacher_promotion_evidence_kind": "verified_stage2_promotion_v1",
        }
    )
    expected_promotion = _evaluate_promotion_gates(current_eval, config)
    current_eval["promotion"] = expected_promotion
    (latent / "eval_metrics.json").write_text(json.dumps(current_eval), encoding="utf-8")
    latent_fingerprint = latent_checkpoint_fingerprint(latent)

    per_motion = [
        {
            "traj_index": index,
            "motion_path": path,
            "motion_uid": uids[index],
            "episode_return": 1.0,
            "episode_length": 120,
            "terminated_early": False,
            "no_fall": True,
            "frame_coverage": 1.0,
            "err_rpos": 0.1,
            "err_racket_pos": 0.02,
            "err_racket_rot": 0.04,
        }
        for index, path in enumerate(motions)
    ]
    aggregate = {
        "mean_episode_return": 1.0,
        "mean_episode_length": 120.0,
        "fall_or_early_termination_rate": 0.0,
        "no_fall_rate": 1.0,
        "frame_coverage": 1.0,
        "err_rpos": 0.1,
        "err_racket_pos": 0.02,
        "err_racket_rot": 0.04,
        "per_motion": per_motion,
    }
    report = {
        "schema_version": "latent_closed_loop_eval_v2",
        "checkpoint_fingerprint": latent_fingerprint,
        "num_trajectories": 5,
        "heldout_motion_paths": motions,
        "heldout_motion_uids": uids,
        "heldout_motion_set_fingerprint": canonical_json_sha256(motions),
        "lambdas": [0.0, 0.25, 0.5],
        "max_steps": 120,
        "by_lambda": {
            "lambda_0p000": copy.deepcopy(aggregate),
            "lambda_0p250": copy.deepcopy(aggregate),
            "lambda_0p500": copy.deepcopy(aggregate),
        },
        "teacher_checkpoint": teacher,
        "teacher_promotion": teacher_promotion,
        "teacher_promotion_evidence_kind": "verified_stage2_promotion_v1",
        "dataset_manifest_fingerprint": dataset_manifest["manifest_fingerprint"],
        "offline_eval_metrics": offline,
        "fall_or_early_termination_rate": 0.0,
        "body_racket_relative_degradation": 0.0,
        "lambda_025_050_no_fall_rate": 1.0,
        "direct_rollout_policy": "student_bc",
        "direct_rollout_metrics": evidence["artifacts"]["comparison_metrics"],
        "direct_promotion_evidence": {
            "path": str(evidence_path.resolve()),
            "sha256": file_sha256(evidence_path),
        },
        "promotion": expected_promotion,
    }
    report["report_fingerprint"] = canonical_json_sha256(report)
    return latent, report, acceptance


def test_strict_closed_loop_report_detects_forgery_and_artifact_tampering(
    monkeypatch, tmp_path
):
    import copy

    from musclemimic.distill.provenance import canonical_json_sha256
    from musclemimic.latent_muscle.closed_loop_eval import (
        validate_closed_loop_promotion_report,
    )

    latent, report, acceptance = _strict_promotion_fixture(tmp_path)
    monkeypatch.setattr(
        "musclemimic.distill.provenance.validate_teacher_promotion_binding",
        lambda binding, **_: binding,
    )
    validate_closed_loop_promotion_report(report, checkpoint_dir=latent)

    forged = copy.deepcopy(report)
    forged["by_lambda"]["lambda_0p250"]["per_motion"][3].pop("err_racket_rot")
    with pytest.raises(ValueError, match="per-motion err_racket_rot"):
        validate_closed_loop_promotion_report(
            forged,
            checkpoint_dir=latent,
            require_seal=False,
        )

    forged_promotion = copy.deepcopy(report)
    forged_promotion["promotion"]["checks"] = {"forged": True}
    forged_promotion["report_fingerprint"] = canonical_json_sha256(
        {
            key: value
            for key, value in forged_promotion.items()
            if key != "report_fingerprint"
        }
    )
    with pytest.raises(ValueError, match="does not match recomputed gates"):
        validate_closed_loop_promotion_report(forged_promotion, checkpoint_dir=latent)

    acceptance.write_text('{"student_bc":{"passed":true,"checks":{"all":false}}}', encoding="utf-8")
    with pytest.raises(ValueError, match="artifact was modified: acceptance"):
        validate_closed_loop_promotion_report(report, checkpoint_dir=latent)

"""Source-only tests for the formal continuity GPU smoke gate."""

from __future__ import annotations

import copy
import json
import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from musclemimic.runner.continuity_smoke import (
    CONTINUITY_TRAINING_SMOKE_SCHEMA_VERSION,
    REQUIRED_SMOKE_CHECKS,
    continuity_training_smoke_fingerprint,
    load_continuity_training_smoke,
    resolved_training_config_sha256,
    validate_continuity_smoke_launch_gate,
    validate_continuity_training_smoke,
)

ROOT = Path(__file__).resolve().parents[2]


def _artifact(*, commit: str, config_hash: str, completed: datetime | None = None):
    completed = completed or datetime.now(UTC)
    payload = {
        "schema_version": CONTINUITY_TRAINING_SMOKE_SCHEMA_VERSION,
        "created_at_utc": (completed - timedelta(minutes=10)).isoformat(),
        "completed_at_utc": completed.isoformat(),
        "git_commit_sha": commit,
        "formal_config": {
            "config_name": "config_specific_task/stage1_body/continuity_ablation_v1/conf_forehand_continuity_a1_s0",
            "resolved_config_sha256": config_hash,
            "condition": "A1",
            "seed": 0,
            "run_id": "forehand_clear_continuity_ablation_v1_a1_s0",
        },
        "runtime": {
            "jax_backend": "gpu",
            "jax_devices": ["cuda(id=0)"],
            "cuda_visible_devices": "2",
            "environment_class": "MyoFullBody",
            "disable_fingers": True,
            "ordered_muscle_channels": 354,
            "activation_address_count": 354,
            "ctrlrange_min": 0.0,
            "ctrlrange_max": 1.0,
            "muscle_channel_core_fingerprint": "a" * 64,
            "racket_muscle_channel_core_fingerprint": "a" * 64,
        },
        "contracts": {
            "release_fingerprint": "b" * 64,
            "taxonomy_fingerprint": "c" * 64,
            "diagnostic_graph_fingerprint": "d" * 64,
            "candidate_graph_fingerprint": "e" * 64,
            "loss_spec_fingerprint": "f" * 64,
            "calibration_fingerprint": "1" * 64,
            "muscle_channel_core_fingerprint": "a" * 64,
            "global_chain_count": 28,
            "global_edge_count": 140,
            "target_chain_count": 4,
            "target_edge_count": 20,
            "action_mode": "full_354",
            "basis_family": "direct_354",
            "basis_fingerprint": None,
            "residual_basis_fingerprint": None,
            "basis_factor_contract_fingerprint": None,
            "graph_regularization_lineage_fingerprint": None,
        },
        "execution": {
            "num_updates": 3,
            "num_envs": 8,
            "num_steps": 20,
            "total_timesteps": 480,
            "checkpoint_dir": "/tmp/smoke/checkpoints",
            "checkpoint_path": "/tmp/smoke/checkpoints/checkpoint_3",
            "restored_update_number": 3,
            "restored_release_fingerprint": "b" * 64,
            "restored_body_contract_matches": True,
            "final_optimizer_step": 24,
        },
        "checks": dict.fromkeys(REQUIRED_SMOKE_CHECKS, True),
        "measurements": {
            "reward_total": [0.2, 0.3, 0.4],
            "continuity_global_loss": [0.1, 0.2, 0.3],
            "continuity_global_chain_count": [28, 28, 28],
            "continuity_global_edge_count": [140, 140, 140],
            "continuity_target_loss": [0.02, 0.03, 0.04],
            "continuity_target_chain_count": [4, 4, 4],
            "continuity_target_edge_count": [20, 20, 20],
            "penalty_continuity_raw": [-0.001, -0.002, -0.003],
            "penalty_continuity_after_local_clip": [-0.001, -0.002, -0.003],
            "penalty_continuity_effective_after_total_clip": [-0.001, -0.002, -0.003],
            "continuity_penalty_masked_fraction": [0.0, 0.0, 0.0],
            "ppo_total_loss": [0.4, 0.3, 0.2],
            "gradient_l2_norm": [1.0, 0.8, 0.6],
            "gradients_all_finite": [1.0, 1.0, 1.0],
            "parameters_all_finite": [1.0, 1.0, 1.0],
            "parameter_update_l2_norm": [0.01, 0.01, 0.01],
            "optimizer_step": [8, 16, 24],
        },
        "errors": [],
        "passed": True,
    }
    payload["artifact_fingerprint"] = continuity_training_smoke_fingerprint(payload)
    return payload


def _git_commit(root):
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_smoke_artifact_is_self_fingerprinted_and_expectation_bound():
    commit = _git_commit(ROOT)
    payload = _artifact(commit=commit, config_hash="2" * 64)

    validated = validate_continuity_training_smoke(
        payload,
        expected_commit_sha=commit,
        expected_resolved_config_sha256="2" * 64,
        expected_release_fingerprint="b" * 64,
        expected_basis_fingerprint=None,
        expected_action_mode="full_354",
        expected_condition="A1",
        max_age_hours=24,
    )
    assert validated["passed"] is True


def test_intrinsic_loader_accepts_synergy_basis_until_launch_expectation_is_supplied(tmp_path):
    payload = _artifact(commit=_git_commit(ROOT), config_hash="2" * 64)
    payload["formal_config"]["condition"] = "B1"
    payload["contracts"].update(
        {
            "action_mode": "fixed_synergy",
            "basis_family": "standard_nmf",
            "basis_fingerprint": "3" * 64,
            "basis_factor_contract_fingerprint": "4" * 64,
        }
    )
    payload["artifact_fingerprint"] = continuity_training_smoke_fingerprint(payload)
    artifact_path = tmp_path / "b1_smoke.json"
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_continuity_training_smoke(artifact_path)["contracts"]["basis_fingerprint"] == "3" * 64
    with pytest.raises(ValueError, match="basis fingerprint differs"):
        validate_continuity_training_smoke(payload, expected_basis_fingerprint=None)


def test_smoke_artifact_rejects_tamper_failure_and_staleness():
    commit = _git_commit(ROOT)
    payload = _artifact(commit=commit, config_hash="2" * 64)
    tampered = copy.deepcopy(payload)
    tampered["measurements"]["reward_total"][0] = 99.0
    with pytest.raises(ValueError, match="fingerprint is stale"):
        validate_continuity_training_smoke(tampered)

    failed = copy.deepcopy(payload)
    failed["checks"]["gradient_finite"] = False
    failed["measurements"]["gradients_all_finite"][0] = 0.0
    failed["passed"] = False
    failed["errors"] = ["failed check: gradient_finite"]
    failed["artifact_fingerprint"] = continuity_training_smoke_fingerprint(failed)
    with pytest.raises(ValueError, match="failed checks"):
        validate_continuity_training_smoke(failed)

    stale = _artifact(
        commit=commit,
        config_hash="2" * 64,
        completed=datetime.now(UTC) - timedelta(hours=25),
    )
    with pytest.raises(ValueError, match="too old"):
        validate_continuity_training_smoke(stale, max_age_hours=24)


def test_formal_gate_binds_commit_config_release_action_and_basis(tmp_path):
    repo_root = ROOT
    commit = _git_commit(repo_root)
    formal_hash = "2" * 64
    payload = _artifact(commit=commit, config_hash=formal_hash)
    artifact_path = tmp_path / "a1_smoke.json"
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")
    config = OmegaConf.create(
        {
            "experiment": {
                "continuity_smoke_gate": {
                    "required": True,
                    "artifact_path": str(artifact_path),
                    "expected_artifact_fingerprint": payload["artifact_fingerprint"],
                    "max_age_hours": 24.0,
                    "require_clean_git": False,
                },
                "training_smoke": {"enabled": False},
                "continuity_training_contract": {"release_fingerprint": "b" * 64},
                "body_synergy_contract": {"mode": "full_354", "basis_fingerprint": None},
                "continuity_ablation": {"condition": "A1"},
            }
        }
    )

    validated = validate_continuity_smoke_launch_gate(
        config,
        formal_resolved_config_sha256=formal_hash,
        repo_root=repo_root,
    )
    assert validated["artifact_fingerprint"] == payload["artifact_fingerprint"]

    config.experiment.continuity_training_contract.release_fingerprint = "9" * 64
    with pytest.raises(ValueError, match="release fingerprint differs"):
        validate_continuity_smoke_launch_gate(
            config,
            formal_resolved_config_sha256=formal_hash,
            repo_root=repo_root,
        )


def test_resolved_config_hash_changes_with_training_semantics():
    config = OmegaConf.create(
        {
            "experiment": {
                "total_timesteps": 320_000_000,
                "auto_resume": False,
                "reward": {"coefficient": 0.01},
            },
            "wandb": {"mode": "online"},
        }
    )
    original = resolved_training_config_sha256(config)
    config.experiment.reward.coefficient = 0.02
    assert resolved_training_config_sha256(config) != original

    config.experiment.continuity_smoke_gate = {
        "artifact_path": "/tmp/smoke.json",
        "expected_artifact_fingerprint": "",
    }
    without_expected_artifact = resolved_training_config_sha256(config)
    config.experiment.continuity_smoke_gate.expected_artifact_fingerprint = "f" * 64
    assert resolved_training_config_sha256(config) == without_expected_artifact


def test_smoke_generation_bypass_is_limited_to_exact_fresh_short_run():
    config = OmegaConf.create(
        {
            "experiment": {
                "num_updates": 3,
                "num_envs": 8,
                "auto_resume": False,
                "resume_from": None,
                "promotion": {"auto_stop": False},
                "validation": {"active": False},
                "continuity_smoke_gate": {"required": True},
                "training_smoke": {
                    "enabled": True,
                    "formal_resolved_config_sha256": "2" * 64,
                },
            }
        }
    )
    assert (
        validate_continuity_smoke_launch_gate(
            config,
            formal_resolved_config_sha256="9" * 64,
            repo_root=ROOT,
        )
        is None
    )

    config.experiment.auto_resume = True
    with pytest.raises(ValueError, match="fresh optimizer"):
        validate_continuity_smoke_launch_gate(
            config,
            formal_resolved_config_sha256="9" * 64,
            repo_root=ROOT,
        )


def test_canonical_launcher_refuses_reward_run_without_smoke_artifact(tmp_path):
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "MUSCLEMIMIC_JAX_CACHE_KEY": "unit_continuity_gate",
            "MUSCLEMIMIC_TRAIN_LOG": str(tmp_path / "training.log"),
            "MUSCLEMIMIC_CONTINUITY_RELEASE": str(tmp_path / "release.json"),
        }
    )
    environment.pop("MUSCLEMIMIC_CONTINUITY_SMOKE_ARTIFACT", None)
    completed = subprocess.run(
        [
            str(ROOT / "scripts/run_fullbody_training.sh"),
            "--config-name=config_specific_task/stage1_body/continuity_ablation_v1/conf_forehand_continuity_a1_s0",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "MUSCLEMIMIC_CONTINUITY_SMOKE_ARTIFACT is required" in completed.stderr

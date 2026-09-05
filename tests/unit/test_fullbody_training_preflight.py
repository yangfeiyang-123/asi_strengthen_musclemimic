"""Read-only Hydra launch summary contracts."""

from __future__ import annotations

from types import SimpleNamespace

from omegaconf import OmegaConf

import scripts.resolve_fullbody_training as preflight
from scripts.resolve_fullbody_training import build_training_preflight_summary


def test_a0_dry_run_resolves_direct_fresh_reward_neutral_contract():
    summary = build_training_preflight_summary(
        "config_specific_task/stage1_body/continuity_ablation_v1/conf_forehand_continuity_a0_s0",
        ["wandb.mode=disabled"],
    )

    assert summary["schema_version"] == "fullbody_training_dry_run_preflight_v1"
    assert summary["condition"] == "A0"
    assert summary["action_mode"] == "full_354"
    assert summary["basis_family"] == "direct_354"
    assert summary["continuity_mode"] == "off"
    assert summary["continuity_release"] is None
    assert summary["ordered_muscle_channels"] == 354
    assert summary["optimizer_state"] == "fresh"
    assert summary["auto_resume"] is False
    assert summary["total_timesteps"] == 320_000_000
    assert summary["promotion"]["auto_stop"] is True
    assert summary["resolved_config_sha256"]
    assert summary["continuity_smoke_validated"] is False


def test_b0_cd_dry_run_reports_diagnostics_condition_and_fresh_optimizer():
    summary = build_training_preflight_summary(
        "config_specific_task/stage1_body/conf_fullbody_chinajump_early_synergy_bootstrap_continuity_diag",
        ["config_status.allow_nonproduction_runtime=true", "wandb.mode=disabled"],
    )

    assert summary["condition"] == "B0-CD"
    assert summary["action_mode"] == "fixed_synergy"
    assert summary["continuity_mode"] == "diagnostics"
    assert summary["optimizer_state"] == "fresh"
    assert summary["auto_resume"] is False
    assert summary["resume_from"] is None
    assert summary["total_timesteps"] == 640_000_000
    assert summary["reward_weights"]["intra_muscle_consistency"]["coefficient"] == 0.0


def test_reward_preflight_binds_smoke_before_training_process(monkeypatch, tmp_path):
    release_path = tmp_path / "release.json"
    smoke_path = tmp_path / "smoke.json"
    captured = {}
    monkeypatch.setattr(
        "musclemimic.physiology.release.load_continuity_training_release",
        lambda path: SimpleNamespace(release_fingerprint="b" * 64),
    )
    monkeypatch.setattr(preflight, "repository_git_commit", lambda *args, **kwargs: "a" * 40)
    monkeypatch.setattr(preflight, "load_continuity_training_smoke", lambda path: {"intrinsic": "valid"})

    def _validate(artifact, **expectations):
        captured.update(expectations)
        return {"artifact_fingerprint": "c" * 64}

    monkeypatch.setattr(preflight, "validate_continuity_training_smoke", _validate)
    config = OmegaConf.create(
        {
            "experiment": {
                "env_params": {
                    "reward_params": {
                        "intra_muscle_consistency": {
                            "release_path": str(release_path),
                            "expected_release_fingerprint": "b" * 64,
                        }
                    }
                },
                "action_representation": {
                    "expected_basis_fingerprint": "d" * 64,
                },
                "continuity_smoke_gate": {
                    "required": True,
                    "artifact_path": str(smoke_path),
                    "expected_artifact_fingerprint": "c" * 64,
                    "max_age_hours": 24.0,
                    "require_clean_git": True,
                },
            }
        }
    )

    artifact = preflight._validate_formal_continuity_smoke(
        config,
        resolved_config_sha256="e" * 64,
        action_mode="fixed_synergy",
        condition="B1",
    )

    assert artifact["artifact_fingerprint"] == "c" * 64
    assert captured["expected_commit_sha"] == "a" * 40
    assert captured["expected_resolved_config_sha256"] == "e" * 64
    assert captured["expected_release_fingerprint"] == "b" * 64
    assert captured["expected_basis_fingerprint"] == "d" * 64
    assert captured["expected_condition"] == "B1"

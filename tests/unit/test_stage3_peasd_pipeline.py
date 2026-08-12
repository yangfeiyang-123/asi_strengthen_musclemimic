from __future__ import annotations

from pathlib import Path

import pytest

from fullbody.run_forehand_clear_pipeline import (
    PipelineArtifacts,
    _canonical_training_launch_command,
    build_pipeline_plan,
)
from musclemimic.badminton.action_registry import CHINA_JUMP


def _stage3_leaf_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    arm: str = "H2",
    seed: int = 1,
) -> PipelineArtifacts:
    gate_path = tmp_path / "stage2_family_gate.json"
    index_path = tmp_path / "stage2_family_index.json"
    selection_path = tmp_path / "selection.json"
    checkpoint = tmp_path / "selected_latent"
    checkpoint.mkdir(exist_ok=True)
    gate_path.write_text("{}\n", encoding="utf-8")
    index_path.write_text("{}\n", encoding="utf-8")
    selection_path.write_text("{}\n", encoding="utf-8")
    selected = {
        "checkpoints": {
            "best_synergy": {
                "stable_checkpoint_path": str(checkpoint),
                "checkpoint_fingerprint": "a" * 64,
            }
        }
    }
    arms = {
        stage2_arm: {
            "selection_manifest": {"path": str(selection_path)}
        }
        for stage2_arm in ("S2-B", "S2-C", "S2-D", "S2-E")
    }
    monkeypatch.setattr(
        "musclemimic.badminton.stage2_context_family.validate_stage2_context_family_gate",
        lambda path, require_pass=True: {
            "action": {"slug": "forehand_clear"},
            "family_index": {"path": str(index_path)},
        },
    )
    monkeypatch.setattr(
        "musclemimic.badminton.stage2_context_family.validate_stage2_context_family_index",
        lambda path: {"arms": arms},
    )
    monkeypatch.setattr(
        "musclemimic.badminton.scripts.latent_synergy_sweep.validate_selected_artifact",
        lambda path: selected,
    )
    return PipelineArtifacts(
        stage2_context_family_gate=str(gate_path),
        stage3_peasd_arm=arm,
        stage3_training_seed=seed,
        stage3_physical_gpu=2,
        stage3_cache_key_prefix="peasd_stage3",
        stage3_reachability_source_checkpoint="/sealed/source_policy.npz",
        stage3_expected_feed_fingerprint="b" * 64,
        stage3_expected_control_hash="c" * 64,
        recovery_target_bank="/sealed/target_train.json",
        recovery_eval_target_bank="/sealed/target_eval.json",
        recovery_train_feed_bank="/sealed/feed_train.npz",
        recovery_eval_feed_bank="/sealed/feed_eval.npz",
    )


def test_stage3_peasd_leaf_has_reachability_before_positive_ppo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _stage3_leaf_artifacts(tmp_path, monkeypatch)
    steps = build_pipeline_plan(
        tmp_path / "h2_s1",
        artifacts,
        profile="stage3_peasd_arm",
    )
    names = [step.name for step in steps]
    assert names == [
        "stage3_v2_preflight",
        "stage3_v2_feed_check",
        "stage3_v2_base_only",
        "stage3_single_feed_cem",
        "stage3_candidate_cpu_audit",
        "stage3_cross_backend_seal",
        "stage3_correction_dataset_seal",
        "stage3_short_bc",
        "stage3_reachability_release",
        "stage3_static_target_train",
        "stage3_static_target_evaluate",
        "stage3_static_target_gate",
        "stage3_v2_train",
        "stage3_v2_evaluate",
        "stage3_v2_gate",
    ]
    assert names.index("stage3_reachability_release") < names.index(
        "stage3_static_target_train"
    )

    cem = next(step for step in steps if step.name == "stage3_single_feed_cem")
    assert cem.command[:2] == (
        str(Path.cwd() / "scripts" / "run_fullbody_training.sh"),
        "--incoming-hit-cem",
    )
    assert cem.command[cem.command.index("--seed") + 1] == "1"
    assert dict(cem.environment) == {
        "CUDA_VISIBLE_DEVICES": "2",
        "MUSCLEMIMIC_JAX_CACHE_KEY": "peasd_stage3_h2_s1",
        "MUSCLEMIMIC_TRAIN_LOG": str(
            tmp_path / "h2_s1" / "stage3_peasd_arm" / "training.log"
        ),
    }

    short_bc = next(step for step in steps if step.name == "stage3_short_bc")
    launched = _canonical_training_launch_command(short_bc)
    assert launched[1] == "--incoming-hit"
    assert launched[launched.index("--total-env-steps") + 1] == "0"
    static = next(step for step in steps if step.name == "stage3_static_target_train")
    assert static.command[static.command.index("--resume-from") + 1] == (
        "<required:stage3_short_bc_checkpoint>"
    )
    assert "--teacher-dataset" in static.command
    assert "--stage3-reachability-release" in static.command
    final = next(step for step in steps if step.name == "stage3_v2_train")
    assert final.command[final.command.index("--curriculum-max-stage") + 1] == (
        "C7_recovery"
    )
    assert final.command[final.command.index("--seed") + 1] == "1"
    assert "static_target_metrics" in final.required_artifacts


def test_stage3_peasd_h3_requires_residual_and_h1_h2_forbid_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h3 = _stage3_leaf_artifacts(tmp_path, monkeypatch, arm="H3")
    with pytest.raises(ValueError, match="H3 requires"):
        build_pipeline_plan(tmp_path / "h3", h3, profile="stage3_peasd_arm")

    residual = tmp_path / "residual.json"
    residual.write_text("{}\n", encoding="utf-8")
    h3 = PipelineArtifacts(
        **{
            **h3.__dict__,
            "stage3_bounded_residual_groups_json": str(residual),
        }
    )
    steps = build_pipeline_plan(tmp_path / "h3", h3, profile="stage3_peasd_arm")
    assert all(
        "--bounded-residual-groups-json" in step.command
        for step in steps
        if "musclemimic.badminton.scripts.run_incoming_shuttle_hit" in step.command
    )

    h1 = _stage3_leaf_artifacts(tmp_path, monkeypatch, arm="H1")
    h1 = PipelineArtifacts(
        **{
            **h1.__dict__,
            "stage3_bounded_residual_groups_json": str(residual),
        }
    )
    with pytest.raises(ValueError, match="H1 must disable"):
        build_pipeline_plan(tmp_path / "h1", h1, profile="stage3_peasd_arm")


def test_stage3_peasd_leaf_is_not_applicable_to_chinajump(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="not applicable"):
        build_pipeline_plan(
            tmp_path,
            PipelineArtifacts(stage3_peasd_arm="H1"),
            profile="stage3_peasd_arm",
            spec=CHINA_JUMP,
        )

"""Matched action-owned config and planner contracts for Stage-1 PEASD-Lite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from fullbody.run_forehand_clear_pipeline import (
    PipelineArtifacts,
    build_pipeline_plan,
    execute_pipeline_step,
)
from musclemimic.badminton.action_registry import (
    ACTIONS,
    STAGE1_PEASD_ARMS,
)
from musclemimic.badminton.data_qc import inspect_canonical_dataset
from musclemimic.badminton.training_gates import CANONICAL_PROMOTION_THRESHOLDS

ROOT = Path(__file__).resolve().parents[2]
FULLBODY = ROOT / "fullbody"


def _compose(config_name: str):
    with initialize_config_dir(version_base=None, config_dir=str(FULLBODY)):
        return compose(config_name=config_name)


def _native(value):
    return OmegaConf.to_container(value, resolve=True)


@pytest.mark.parametrize("spec", list(ACTIONS.values()), ids=lambda spec: spec.slug)
def test_every_action_owns_a_complete_t0_to_t4_config_family(spec) -> None:
    assert spec.stage1_peasd_ready is True
    assert tuple(label for label, _path in spec.stage1_peasd_configs) == STAGE1_PEASD_ARMS
    configs = {}
    for arm in STAGE1_PEASD_ARMS:
        path = spec.stage1_peasd_config(arm)
        assert spec.slug in path
        assert (FULLBODY / f"{path}.yaml").is_file()
        configs[arm] = _compose(path)

    run_ids = [str(configs[arm].experiment.run_id) for arm in STAGE1_PEASD_ARMS]
    assert len(set(run_ids)) == len(STAGE1_PEASD_ARMS)
    for arm, config in configs.items():
        experiment = config.experiment
        contract = experiment.stage1_peasd
        reward = experiment.env_params.reward_params.emg_consistency
        assert contract.schema_version == "stage1_peasd_lite_matched_arm_v1"
        assert contract.arm == arm
        assert contract.action_id == spec.action_id
        assert list(contract.canonical_seeds) == [0, 1, 2]
        assert contract.fresh_optimizer_required is True
        assert contract.parent_initialization_checkpoint is None
        assert experiment.auto_resume is False
        assert experiment.resume_from is None
        assert experiment.reset_optimizer_on_resume is True
        assert experiment.promotion.auto_stop is False
        for field, expected in CANONICAL_PROMOTION_THRESHOLDS["stage1"].items():
            assert float(experiment.promotion[field]) == float(expected)
        assert experiment.n_seeds == 1
        assert list(experiment.seeds) == [0]
        assert reward.action_id == spec.emg_trial_actions[0]
        assert reward.reference_cache is None
        assert reward.mapping_path is None
        assert list(experiment.task_factory.params.amass_dataset_conf.rel_dataset_path) == list(
            spec.train_motion_paths
        )
        assert list(experiment.validation.amass_dataset_conf.rel_dataset_path) == list(
            spec.val_motion_paths
        )

    rewards = {
        arm: configs[arm].experiment.env_params.reward_params.emg_consistency
        for arm in STAGE1_PEASD_ARMS
    }
    assert rewards["T0"].enabled is False
    assert (rewards["T1"].anchor_weight_max, rewards["T1"].synergy_weight_max) == (0.02, 0.0)
    assert (rewards["T2"].anchor_weight_max, rewards["T2"].synergy_weight_max) == (0.0, 0.05)
    assert (rewards["T3"].anchor_weight_max, rewards["T3"].synergy_weight_max) == (0.02, 0.05)
    assert (rewards["T4"].anchor_weight_max, rewards["T4"].synergy_weight_max) == (0.02, 0.05)
    assert rewards["T3"].synergy_phase_shuffle_offset_bins == 0
    assert rewards["T4"].synergy_phase_shuffle_offset_bins == 10
    assert (
        configs["T4"].experiment.stage1_peasd.control_kind
        == "deterministic_half_cycle_circular_phase_shift"
    )
    t4_keys = set(_native(rewards["T4"]))
    assert not any("event" in key.lower() for key in t4_keys)


@pytest.mark.parametrize("spec", list(ACTIONS.values()), ids=lambda spec: spec.slug)
def test_t3_t4_are_matched_except_arm_and_synergy_phase_shuffle(spec) -> None:
    t3 = _native(_compose(spec.stage1_peasd_config("T3")).experiment)
    t4 = _native(_compose(spec.stage1_peasd_config("T4")).experiment)

    for payload in (t3, t4):
        payload.pop("run_id")
        contract = payload["stage1_peasd"]
        contract.pop("arm")
        contract.pop("control_kind")
        reward = payload["env_params"]["reward_params"]["emg_consistency"]
        reward.pop("arm")
        reward.pop("synergy_phase_shuffle_offset_bins")
    assert t3 == t4


@pytest.mark.parametrize("spec", list(ACTIONS.values()), ids=lambda spec: spec.slug)
def test_stage1_peasd_profile_runs_t0_before_the_fail_closed_tube_gate(spec, tmp_path) -> None:
    tube = tmp_path / spec.slug / "emg_reference_manifest.json"
    artifacts = PipelineArtifacts(
        emg_reference_manifest=str(tube),
        stage1_peasd_pairwise_metrics=str(tmp_path / "paired.json"),
    )
    steps = build_pipeline_plan(
        tmp_path,
        artifacts,
        profile="stage1_peasd",
        spec=spec,
    )
    names = [step.name for step in steps]
    assert names[:2] == ["data_release_validate", "data_qc"]
    assert names[2:5] == [
        "stage1_peasd_t0_s0_train",
        "stage1_peasd_t0_s1_train",
        "stage1_peasd_t0_s2_train",
    ]
    assert names[5] == "stage1_peasd_tube_gate"
    assert names[6:9] == [
        "stage1_peasd_t0_s0_posthoc_physiology",
        "stage1_peasd_t0_s1_posthoc_physiology",
        "stage1_peasd_t0_s2_posthoc_physiology",
    ]
    assert names[-2:] == [
        "stage1_peasd_evidence_index",
        "stage1_peasd_pairwise_gate",
    ]
    assert names[-1] == "stage1_peasd_pairwise_gate"
    train = [step for step in steps if step.name.endswith("_train")]
    assert len(train) == 15
    assert not any(
        name.startswith(("stage1r", "stage2", "racket_mass", "stage3", "latent"))
        for name in names
    )

    run_ids = []
    for step in train:
        assert "experiment.auto_resume=false" in step.command
        assert "experiment.resume_from=null" in step.command
        assert "experiment.reset_optimizer_on_resume=true" in step.command
        run_token = next(token for token in step.command if token.startswith("experiment.run_id="))
        run_ids.append(run_token.split("=", 1)[1])
        config_token = next(token for token in step.command if token.startswith("--config-name="))
        assert spec.slug in config_token
        if "_t0_" in step.name:
            assert step.required_artifacts == ()
            assert not any("reference_cache=" in token for token in step.command)
            assert not any(str(tube) in token for token in step.command)
        else:
            assert step.required_artifacts == ("emg_reference_manifest",)
            assert (
                "experiment.env_params.reward_params.emg_consistency.reference_cache="
                f"{tube}"
            ) in step.command
            assert (
                "experiment.env_params.reward_params.emg_consistency.action_id="
                f"{spec.emg_trial_actions[0]}"
            ) in step.command
    assert len(set(run_ids)) == 15

    index = next(step for step in steps if step.name == "stage1_peasd_evidence_index")
    assert index.command.count("--evidence") == 15
    assert len(index.required_artifacts) == 15
    for arm in STAGE1_PEASD_ARMS:
        for seed in (0, 1, 2):
            selector = (
                f"{arm}:{seed}:"
                f"<required:stage1_peasd_{arm.lower()}_s{seed}_validation_evidence>"
            )
            assert selector in index.command
    pairwise = next(
        step for step in steps if step.name == "stage1_peasd_pairwise_gate"
    )
    assert "--visual-review" not in pairwise.command
    assert pairwise.command[pairwise.command.index("--blind-review") + 1] == (
        "<required:stage1_peasd_blind_review>"
    )
    assert pairwise.command[pairwise.command.index("--blind-mapping") + 1] == (
        "<required:stage1_peasd_blind_private_mapping>"
    )
    assert pairwise.required_artifacts == (
        "emg_reference_manifest",
        "stage1_peasd_blind_review",
        "stage1_peasd_blind_private_mapping",
    )


def test_registry_rejects_cross_action_stage1_peasd_asset() -> None:
    from dataclasses import replace

    source = ACTIONS["chinajump"]
    borrowed = tuple(
        (arm, path.replace("chinajump", "forehand_clear"))
        for arm, path in source.stage1_peasd_configs
    )
    with pytest.raises(ValueError, match="action-owned"):
        replace(source, stage1_peasd_configs=borrowed).validate()


def test_direct_peasd_train_execution_binds_qc_release_and_tube_gate(
    monkeypatch, tmp_path
) -> None:
    with pytest.raises(ValueError, match="requires completed data QC"):
        execute_pipeline_step(
            "stage1_peasd_t0_s0_train",
            output_dir=tmp_path / "missing",
            artifacts=PipelineArtifacts(),
            profile="stage1_peasd",
        )

    report = inspect_canonical_dataset(
        ACTIONS["forehand_clear"].dataset_root,
        source_variant=ACTIONS["forehand_clear"].source_variant,
        cache_variant=ACTIONS["forehand_clear"].cache_variant,
        action="forehand_clear",
    )
    (tmp_path / "data_qc.json").write_text(
        json.dumps(report, sort_keys=True) + "\n", encoding="utf-8"
    )
    monkeypatch.setenv(
        "MUSCLEMIMIC_GMR_CACHE_PATH",
        str(ACTIONS["forehand_clear"].dataset_root.parent),
    )
    launched: list[list[str]] = []
    monkeypatch.setattr(
        "fullbody.run_forehand_clear_pipeline.subprocess.run",
        lambda command, **_kwargs: launched.append(command),
    )
    execute_pipeline_step(
        "stage1_peasd_t0_s0_train",
        output_dir=tmp_path,
        artifacts=PipelineArtifacts(),
        profile="stage1_peasd",
    )
    assert launched and launched[0][0].endswith("scripts/run_fullbody_training.sh")
    assert (tmp_path / "stage1_peasd" / "data_preflight_binding.json").is_file()

    tube = tmp_path / "tube" / "emg_reference_manifest.json"
    with pytest.raises((FileNotFoundError, ValueError)):
        execute_pipeline_step(
            "stage1_peasd_t1_s0_train",
            output_dir=tmp_path,
            artifacts=PipelineArtifacts(emg_reference_manifest=str(tube)),
            profile="stage1_peasd",
        )
    assert len(launched) == 1

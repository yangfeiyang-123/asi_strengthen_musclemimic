from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from musclemimic.badminton.aug100_release import (
    GROUPED_SUBSET_40_10_CONTRACT,
    REPO_ROOT,
    TRANSFER_MANIFEST,
    expected_forehand_clear_aug100_split,
    motion_names_from_relative_paths,
    validate_forehand_clear_aug100_release,
)
from musclemimic.runner.engine import (
    bind_stage1_peasd_action_release,
    validate_training_source_preflight,
)
from musclemimic.runner.stage1_peasd_validation import (
    _action_release_contract,
    _numeric_data_qc_contract,
)

FULLBODY = REPO_ROOT / "fullbody"
CONFIG_NAME = (
    "config_specific_task/stage1_body/peasd_lite_v1/"
    "conf_fullbody_forehand_clear_aug100_peasd_t1"
)
SUBSET_CONFIG_TEMPLATE = (
    "config_specific_task/stage1_body/peasd_lite_v1/"
    "conf_fullbody_forehand_clear_aug100_40train10val_peasd_{arm}"
)
PRIVATE_ASSETS_AVAILABLE = (REPO_ROOT / TRANSFER_MANIFEST).is_file()


def _compose():
    with initialize_config_dir(version_base=None, config_dir=str(FULLBODY)):
        return compose(config_name=CONFIG_NAME)


def _compose_subset(arm: str):
    with initialize_config_dir(version_base=None, config_dir=str(FULLBODY)):
        return compose(config_name=SUBSET_CONFIG_TEMPLATE.format(arm=arm.lower()))


@pytest.mark.skipif(not PRIVATE_ASSETS_AVAILABLE, reason="private Aug100 release is unavailable")
def test_aug100_release_is_content_bound_and_source_grouped() -> None:
    train, validation = expected_forehand_clear_aug100_split()
    report = validate_forehand_clear_aug100_release(train, validation)

    assert report["passed"] is True
    assert report["errors"] == []
    assert len(train) == 80
    assert len(validation) == 20
    assert len(report["file_inventory"]) == 100
    assert set(report["train_source_groups"]).isdisjoint(
        report["validation_source_groups"]
    )
    assert sum(row["is_augmented"] for row in report["file_inventory"]) == 73
    assert all(
        row.get("rotation_qc")
        for row in report["file_inventory"]
        if row["is_augmented"]
    )


@pytest.mark.skipif(not PRIVATE_ASSETS_AVAILABLE, reason="private Aug100 release is unavailable")
def test_aug100_t1_config_matches_the_metadata_grouped_split(
    monkeypatch, tmp_path
) -> None:
    config = _compose()
    train = motion_names_from_relative_paths(
        config.experiment.task_factory.params.amass_dataset_conf.rel_dataset_path
    )
    validation = motion_names_from_relative_paths(
        config.experiment.validation.amass_dataset_conf.rel_dataset_path
    )
    assert (train, validation) == expected_forehand_clear_aug100_split()
    assert config.experiment.training_root == (
        "datasets/forehandClear_standard/training_aug100"
    )
    assert config.experiment.auto_resume is False
    assert config.experiment.resume_from is None
    assert config.experiment.reset_optimizer_on_resume is True
    assert config.experiment.stage1_peasd.arm == "T1"
    assert config.experiment.total_timesteps == 320_000_000

    report = bind_stage1_peasd_action_release(config)
    assert report is not None and report["passed"] is True
    assert report["data_variant"] == "raw_smooth_v1_aug100"
    assert config.experiment.stage1_peasd_numeric_data_qc_contract.report.clean_passed
    experiment = OmegaConf.to_container(config.experiment, resolve=True)
    assert _action_release_contract(experiment)["passed"] is True
    assert _numeric_data_qc_contract(experiment)["report"]["clean_passed"] is True

    monkeypatch.setenv("MUSCLEMIMIC_GMR_CACHE_PATH", str(REPO_ROOT / "datasets"))
    source_report = validate_training_source_preflight(
        config,
        launch_dir=REPO_ROOT,
        result_dir=tmp_path,
    )
    assert source_report is not None and source_report["passed"] is True
    assert len(source_report["train_motions"]) == 80
    assert len(source_report["validation_motions"]) == 20
    assert (tmp_path / "training_source_preflight.json").is_file()


@pytest.mark.skipif(not PRIVATE_ASSETS_AVAILABLE, reason="private Aug100 release is unavailable")
def test_aug100_release_rejects_a_file_level_leakage_split() -> None:
    train, validation = expected_forehand_clear_aug100_split()
    leaked_train = (*train, validation[5])
    leaked_validation = validation[0:5] + validation[6:]
    report = validate_forehand_clear_aug100_release(leaked_train, leaked_validation)
    assert report["passed"] is False
    assert any("metadata-grouped release" in error for error in report["errors"])


def test_aug100_runtime_paths_reject_foreign_namespaces() -> None:
    with pytest.raises(ValueError, match="outside"):
        motion_names_from_relative_paths(
            ["forehandClear_standard/muscle_trajectory/raw_smooth_v1/video1"]
        )


@pytest.mark.skipif(not PRIVATE_ASSETS_AVAILABLE, reason="private Aug100 release is unavailable")
def test_aug100_40_train_10_validation_t2_t4_family_is_grouped_and_matched() -> None:
    configs = {arm: _compose_subset(arm) for arm in ("T2", "T3", "T4")}
    split_pairs = []
    for arm, config in configs.items():
        train = motion_names_from_relative_paths(
            config.experiment.task_factory.params.amass_dataset_conf.rel_dataset_path
        )
        validation = motion_names_from_relative_paths(
            config.experiment.validation.amass_dataset_conf.rel_dataset_path
        )
        split_pairs.append((train, validation))
        assert len(train) == 40
        assert len(validation) == 10
        assert config.experiment.training_source.split_contract == (
            GROUPED_SUBSET_40_10_CONTRACT
        )
        assert config.experiment.validation.num_envs == 10
        assert config.experiment.promotion.require_visual_validation_clips == 10
        assert config.experiment.stage1_peasd.arm == arm
        assert config.experiment.auto_resume is False
        assert config.experiment.resume_from is None
        assert config.experiment.reset_optimizer_on_resume is True
        assert config.experiment.total_timesteps == 320_000_000

    assert split_pairs[0] == split_pairs[1] == split_pairs[2]
    train, validation = split_pairs[0]
    report = validate_forehand_clear_aug100_release(
        train,
        validation,
        split_contract=GROUPED_SUBSET_40_10_CONTRACT,
    )
    assert report["passed"] is True
    assert set(report["train_source_groups"]).isdisjoint(
        report["validation_source_groups"]
    )
    assert sum(row["split"] == "train" for row in report["file_inventory"]) == 40
    assert sum(row["split"] == "validation" for row in report["file_inventory"]) == 10
    assert sum(row["split"] == "unused" for row in report["file_inventory"]) == 50

    bound = bind_stage1_peasd_action_release(configs["T2"])
    assert bound is not None and bound["passed"] is True
    assert bound["split_contract"] == GROUPED_SUBSET_40_10_CONTRACT

    rewards = {
        arm: configs[arm].experiment.env_params.reward_params.emg_consistency
        for arm in configs
    }
    assert (rewards["T2"].anchor_weight_max, rewards["T2"].synergy_weight_max) == (
        0.0,
        0.05,
    )
    assert (rewards["T3"].anchor_weight_max, rewards["T3"].synergy_weight_max) == (
        0.02,
        0.05,
    )
    assert (rewards["T4"].anchor_weight_max, rewards["T4"].synergy_weight_max) == (
        0.02,
        0.05,
    )
    assert rewards["T3"].synergy_phase_shuffle_offset_bins == 0
    assert rewards["T4"].synergy_phase_shuffle_offset_bins == 10


@pytest.mark.skipif(not PRIVATE_ASSETS_AVAILABLE, reason="private Aug100 release is unavailable")
def test_aug100_40_10_contract_rejects_partial_source_group() -> None:
    config = _compose_subset("T2")
    train = list(
        motion_names_from_relative_paths(
            config.experiment.task_factory.params.amass_dataset_conf.rel_dataset_path
        )
    )
    validation = motion_names_from_relative_paths(
        config.experiment.validation.amass_dataset_conf.rel_dataset_path
    )
    train[-1] = "video8"
    report = validate_forehand_clear_aug100_release(
        train,
        validation,
        split_contract=GROUPED_SUBSET_40_10_CONTRACT,
    )
    assert report["passed"] is False
    assert any("partially selected" in error for error in report["errors"])

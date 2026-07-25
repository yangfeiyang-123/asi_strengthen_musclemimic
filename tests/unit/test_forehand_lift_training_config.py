from __future__ import annotations

from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from musclemimic.badminton.data_qc import _inspect_motion


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_NAME = (
    "config_specific_task/stage1_body/"
    "conf_fullbody_forehand_lift_optimized_root_first"
)


def _compose():
    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "fullbody")):
        return compose(config_name=CONFIG_NAME)


def test_forehand_lift_config_binds_all_optimized_caches_once():
    cfg = _compose()
    train = list(cfg.experiment.task_factory.params.amass_dataset_conf.rel_dataset_path)
    val = list(cfg.experiment.validation.amass_dataset_conf.rel_dataset_path)

    assert len(train) == 12
    assert len(val) == 4
    assert not set(train) & set(val)
    assert len(set(train + val)) == 16
    manifest = {
        line.strip()
        for line in (
            REPO_ROOT / "datasets" / "forehandLift" / "manifests" / "optimized_list.txt"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    expected = {
        motion.replace("forehandLift/optimized/", "forehandLift/muscle_trajectory/optimized/")
        for motion in manifest
    }
    assert set(train + val) == expected

    for motion in train + val:
        cache = REPO_ROOT / "datasets" / f"{motion}.npz"
        assert cache.is_file(), cache
        with np.load(cache, allow_pickle=True) as data:
            assert float(data["frequency"]) == 100.0
            assert data["qpos"].shape[1] == 89
            assert data["qvel"].shape[1] == 88
            assert np.isfinite(data["qpos"]).all()
            assert np.isfinite(data["qvel"]).all()


def test_forehand_lift_config_has_fresh_640m_root_first_contract():
    cfg = _compose()
    exp = cfg.experiment

    assert exp.training_action == "forehandLift"
    assert exp.run_id == "forehand_lift_optimized_root_first_stage1_body_640m_v1"
    assert exp.resume_from is None
    assert exp.auto_resume is True
    assert exp.strict_auto_resume_config_hash is True
    assert exp.total_timesteps == 640_000_000
    assert exp.training_source.source_fps == 60
    assert exp.training_source.cache_fps == 100
    assert exp.promotion.auto_stop is False

    reward = exp.env_params.reward_params
    assert reward.root_pos_w_sum == 0.35
    assert reward.root_vel_w_sum == 0.25
    assert reward.root_pos_w_sum > reward.qpos_w_sum
    assert reward.root_vel_w_sum > reward.qvel_w_sum
    assert list(reward.absolute_site_reward_sites) == ["right_hand_mimic"]
    assert reward.absolute_site_w_sum == 0.10

    for terminal in (
        exp.env_params.terminal_state_params,
        exp.validation.terminal_state_params,
    ):
        assert terminal.mean_site_deviation_threshold == 0.45
        assert terminal.root_deviation_threshold == 0.30
        assert terminal.root_orientation_threshold == 0.70
        assert terminal.enable_site_check is True

    assert exp.env_params.terminal_state_type == (
        "MeanRelativeSiteDeviationWithRootTerminalStateHandler"
    )
    assert exp.validation.terminal_state_type == (
        "MeanRelativeSiteDeviationWithRootTerminalStateHandler"
    )


def test_forehand_lift_config_reuses_existing_retarget_release_without_mutation():
    cfg = _compose()
    train_conf = cfg.experiment.task_factory.params.amass_dataset_conf
    val_conf = cfg.experiment.validation.amass_dataset_conf

    for conf in (train_conf, val_conf):
        assert conf.retargeting_method == "gmr"
        assert conf.clear_cache is False
        assert conf.gmr_config.target_fps == 60
        assert conf.gmr_config.damping == 0.5
        assert conf.gmr_config.use_velocity_limit is False
        assert conf.gmr_config.ik_config_path is None


def test_forehand_lift_optimized_release_passes_cache_level_pre_ppo_qc():
    cfg = _compose()
    motions = [
        *cfg.experiment.task_factory.params.amass_dataset_conf.rel_dataset_path,
        *cfg.experiment.validation.amass_dataset_conf.rel_dataset_path,
    ]

    for motion in motions:
        name = Path(str(motion)).name
        source = (
            REPO_ROOT
            / "datasets"
            / "forehandLift"
            / "wham"
            / "optimized_wham"
            / f"{name}.npz"
        )
        cache = REPO_ROOT / "datasets" / f"{motion}.npz"
        row, hard_errors = _inspect_motion(name, source, cache)

        assert hard_errors == []
        assert row.warnings == ()

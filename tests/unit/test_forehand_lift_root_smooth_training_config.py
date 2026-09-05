from __future__ import annotations

from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_NAME = (
    "config_specific_task/stage1_body/"
    "conf_fullbody_forehand_lift_optimized_root_smooth_v2"
)
RUN_ID = "forehand_lift_optimized_root_smooth_stage1_body_640m_v2"


def _compose():
    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "fullbody")):
        return compose(config_name=CONFIG_NAME)


def test_root_smooth_v2_config_has_fresh_640m_contract():
    exp = _compose().experiment

    assert exp.training_action == "forehandLift"
    assert exp.run_id == RUN_ID
    assert exp.resume_from is None
    assert exp.auto_resume is True
    assert exp.strict_auto_resume_config_hash is True
    assert exp.total_timesteps == 640_000_000
    assert exp.checkpoint_interval == 250
    assert exp.promotion.auto_stop is False
    assert exp.training_source.variant == "optimized_root_smooth_v2"
    assert exp.training_source.source_fps == 60
    assert exp.training_source.cache_fps == 100

    reward = exp.env_params.reward_params
    assert reward.root_pos_w_sum == 0.35
    assert reward.root_vel_w_sum == 0.25
    assert list(reward.absolute_site_reward_sites) == ["right_hand_mimic"]
    for terminal in (exp.env_params.terminal_state_params, exp.validation.terminal_state_params):
        assert terminal.mean_site_deviation_threshold == 0.45
        assert terminal.root_deviation_threshold == 0.30
        assert terminal.root_orientation_threshold == 0.70


def test_root_smooth_v2_config_binds_unique_12_train_4_val_release():
    exp = _compose().experiment
    train = list(exp.task_factory.params.amass_dataset_conf.rel_dataset_path)
    val = list(exp.validation.amass_dataset_conf.rel_dataset_path)

    assert len(train) == 12
    assert len(val) == 4
    assert len(set(train + val)) == 16
    assert not set(train) & set(val)
    assert train == [
        line.strip()
        for line in (REPO_ROOT / "datasets/forehandLift/manifests/optimized_root_smooth_v2/train_list.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert val == [
        line.strip()
        for line in (REPO_ROOT / "datasets/forehandLift/manifests/optimized_root_smooth_v2/val_list.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    for motion in train + val:
        cache = REPO_ROOT / "datasets" / f"{motion}.npz"
        assert cache.is_file(), cache
        with np.load(cache, allow_pickle=True) as data:
            assert float(data["frequency"]) == 100.0
            assert data["qpos"].shape[1] == 89
            assert data["qvel"].shape[1] == 88
            assert np.isfinite(data["qpos"]).all()
            assert np.isfinite(data["qvel"]).all()


def test_root_smooth_v2_config_uses_immutable_global_grounded_caches():
    exp = _compose().experiment
    for conf in (
        exp.task_factory.params.amass_dataset_conf,
        exp.validation.amass_dataset_conf,
    ):
        assert conf.retargeting_method == "gmr"
        assert conf.clear_cache is False
        assert conf.gmr_config.target_fps == 60
        assert conf.gmr_config.damping == 0.5
        assert conf.gmr_config.offset_to_ground is False
        assert conf.gmr_config.grounding_mode == "global"
        assert conf.gmr_config.use_velocity_limit is False

    release = REPO_ROOT / str(exp.training_source.release_manifest)
    assert release.is_file()

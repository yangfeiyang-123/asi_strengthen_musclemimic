from pathlib import Path

from musclemimic.badminton.scripts.run_retarget import _build_gmr_config


def test_build_gmr_config_defaults_keep_baseline_behavior():
    config = _build_gmr_config(target_fps=60, damping=None, use_velocity_limit=False, ik_config_path=None)

    assert config["target_fps"] == 60
    assert config["damping"] == 0.5
    assert config["use_velocity_limit"] is False
    assert "ik_config_path" not in config


def test_build_gmr_config_accepts_smooth_overrides():
    config = _build_gmr_config(
        target_fps=60,
        damping=1.0,
        use_velocity_limit=True,
        ik_config_path=Path("loco_mujoco/smpl/gmr_configs/smplh_to_myofullbody_smooth_train.json"),
    )

    assert config["target_fps"] == 60
    assert config["damping"] == 1.0
    assert config["use_velocity_limit"] is True
    assert config["ik_config_path"] == "loco_mujoco/smpl/gmr_configs/smplh_to_myofullbody_smooth_train.json"

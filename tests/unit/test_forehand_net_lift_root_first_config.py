from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "fullbody" / "config_specific_task" / "conf_fullbody_forehand_net_lift_root_first.yaml"


def test_root_first_config_exists_and_uses_current_data_paths():
    cfg = OmegaConf.load(CONFIG)
    paths = list(cfg.experiment.task_factory.params.amass_dataset_conf.rel_dataset_path)
    env_set = cfg.hydra.job.env_set

    assert paths == [
        "ForehandNetLift/best/video01_best_stage7_smpl",
        "ForehandNetLift/best/video02_best_stage7_smpl",
        "ForehandNetLift/best/video03_best_stage7_smpl",
        "ForehandNetLift/best/video04_best_stage7_smpl",
        "ForehandNetLift/best/video05_best_stage7_smpl",
        "ForehandNetLift/best/video06_best_stage7_smpl",
        "ForehandNetLift/best/video07_best_stage5_smpl",
        "ForehandNetLift/best/video08_best_stage5_smpl",
    ]
    assert env_set.MUSCLEMIMIC_AMASS_PATH == str(
        REPO_ROOT / "datasets" / "forehandLift" / "muscle_trajectory" / "amass_npz"
    )
    assert env_set.AMASS_PATH == env_set.MUSCLEMIMIC_AMASS_PATH
    assert env_set.MUSCLEMIMIC_GMR_CACHE_PATH == str(
        REPO_ROOT / "datasets" / "forehandLift" / "muscle_trajectory" / "gmr_cache"
    )


def test_root_first_config_uses_root_heavy_reward_and_strict_validation():
    cfg = OmegaConf.load(CONFIG)
    reward = cfg.experiment.env_params.reward_params
    validation = cfg.experiment.validation

    assert reward.root_pos_w_sum > reward.qpos_w_sum
    assert reward.root_vel_w_sum > reward.qvel_w_sum
    assert reward.absolute_site_reward_sites == ["right_hand_mimic"]
    assert validation.terminal_state_type == "MeanRelativeSiteDeviationWithRootTerminalStateHandler"
    assert validation.terminal_state_params.root_deviation_threshold == 0.30


def test_root_first_config_points_to_existing_checkpoint_root():
    cfg = OmegaConf.load(CONFIG)
    checkpoint_root = Path(cfg.experiment.checkpoint_root)

    assert checkpoint_root == REPO_ROOT / "checkpoints" / "ForehandNetLift" / "forehand_net_lift_best_ppo"
    assert cfg.experiment.resume_from == str(checkpoint_root / "checkpoint_7812")

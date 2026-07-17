from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError

from musclemimic.badminton.training_gates import CANONICAL_PROMOTION_THRESHOLDS
from musclemimic.runner.engine import validate_experiment_config_status

REPO_ROOT = Path(__file__).resolve().parents[2]
FULLBODY_DIR = REPO_ROOT / "fullbody"


def _compose(name: str):
    with initialize_config_dir(version_base=None, config_dir=str(FULLBODY_DIR)):
        return compose(config_name=name)


def _assert_canonical_promotion_thresholds(cfg, stage: str):
    assert {
        field: float(cfg.experiment.promotion[field])
        for field in CANONICAL_PROMOTION_THRESHOLDS[stage]
    } == dict(CANONICAL_PROMOTION_THRESHOLDS[stage])


def test_stage1_is_canonical_fingerless_raw_smooth_v1_354_contract():
    cfg = _compose(
        "config_specific_task/stage1_body/"
        "conf_fullbody_forehand_clear_body_local"
    )

    assert cfg.experiment.env_params.env_name == "MjxMyoFullBody"
    assert cfg.experiment.env_params.disable_fingers is True
    assert cfg.experiment.training_source.variant == "raw_smooth_v1"
    assert cfg.experiment.training_source.source_namespace == "temp/raw_smooth_v1"
    assert cfg.experiment.training_source.cache_namespace == "muscle_trajectory/raw_smooth_v1"
    assert cfg.experiment.run_id == "forehand_clear_raw_smooth_v1_22_stage1_body_v1"
    assert cfg.experiment.training_source.source_fps == 60
    assert cfg.experiment.training_source.cache_fps == 100
    assert cfg.experiment.total_timesteps == 320_000_000
    assert len(cfg.experiment.task_factory.params.amass_dataset_conf.rel_dataset_path) == 22
    assert len(cfg.experiment.validation.amass_dataset_conf.rel_dataset_path) == 5
    assert all(
        "/raw_smooth_v1/" in path
        for path in cfg.experiment.task_factory.params.amass_dataset_conf.rel_dataset_path
    )
    assert all(
        "/raw_smooth_v1/" in path
        for path in cfg.experiment.validation.amass_dataset_conf.rel_dataset_path
    )
    reward = cfg.experiment.env_params.reward_params
    assert reward.qpos_w_sum == 0.10
    assert reward.qvel_w_sum == 0.10
    assert reward.root_pos_w_sum == 0.10
    assert reward.root_vel_w_sum == 0.10
    assert reward.rpos_w_sum == 0.60
    assert reward.rquat_w_sum == 0.01
    assert reward.rvel_w_sum == 0.10
    train_gmr = cfg.experiment.task_factory.params.amass_dataset_conf.gmr_config
    assert train_gmr.damping == 1.0
    assert train_gmr.use_velocity_limit is True
    assert train_gmr.ik_config_path.endswith("smplh_to_myofullbody_smooth_train.json")
    assert cfg.experiment.promotion.stage == "stage1"
    assert cfg.experiment.promotion.auto_stop is True
    assert cfg.experiment.promotion.consecutive_validations == 3
    assert cfg.experiment.promotion.min_validations_before_stop == 5
    assert cfg.experiment.promotion.progress_path is None
    assert cfg.experiment.strict_auto_resume_config_hash is True
    assert cfg.experiment.validation.cover_all_trajectories is True
    assert cfg.experiment.validation.visual_review_kind == "stage1_body"
    assert cfg.experiment.promotion.max_early_termination_rate == 0.05
    assert cfg.experiment.promotion.max_action_saturation_fraction == 0.05
    assert cfg.experiment.promotion.max_activation_energy == 0.35
    assert cfg.experiment.validation.num // cfg.experiment.validation.video_frequency >= 5
    assert cfg.experiment.validation.video_frequency == 1
    assert cfg.experiment.validation.video_max_recordings is None
    assert cfg.experiment.validation.cycle_video_trajectories is True
    _assert_canonical_promotion_thresholds(cfg, "stage1")


def test_stage1_repair_v2_is_conservative_incremental_finetune(monkeypatch):
    parent = "/tmp/forehand-stage1-checkpoint-31250"
    monkeypatch.setenv("FOREHAND_STAGE1_REPAIR_PARENT_CHECKPOINT", parent)
    cfg = _compose(
        "config_specific_task/stage1_body/"
        "conf_fullbody_forehand_clear_body_repair_v2"
    )

    assert cfg.experiment.run_id == (
        "forehand_clear_raw_smooth_v1_22_stage1_repair_20m_v2"
    )
    assert cfg.experiment.resume_from == parent
    assert cfg.experiment.parent_checkpoint_lineage.required is True
    assert cfg.experiment.parent_checkpoint_lineage.role == "stage1_repair_parent_640m"
    assert cfg.experiment.legacy_parent_body_action_attestation.endswith(
        "forehand_clear_raw_smooth_v1_22_stage1_repair_20m_v1/manifest.json"
    )
    assert cfg.experiment.total_timesteps == 20_480_000
    assert cfg.experiment.total_timesteps_is_absolute is False
    assert cfg.experiment.reset_optimizer_on_resume is True
    assert cfg.experiment.reset_lr_schedule_on_resume is False
    assert cfg.experiment.reset_std_on_resume is None
    assert cfg.experiment.lr == pytest.approx(5.0e-6)
    assert cfg.experiment.anneal_lr is False
    assert cfg.experiment.policy_anchor.enabled is True
    assert cfg.experiment.policy_anchor.type == "linear_hinge_action_mse"
    assert cfg.experiment.policy_anchor.margin == pytest.approx(0.0025)
    assert cfg.experiment.validation.num == 5
    assert cfg.experiment.validation.visual_review_kind == "stage1_body"
    assert cfg.experiment.adaptive_sampling.enabled is True
    assert cfg.experiment.adaptive_sampling.floor_mix == 0.25
    reward = cfg.experiment.env_params.reward_params
    assert reward.action_out_of_bounds_coeff == pytest.approx(0.01)
    assert reward.action_rate_coeff == 0.0
    assert reward.action_saturation_coeff == 0.01
    assert reward.action_saturation_margin_fraction == 0.02
    assert reward.activation_energy_coeff == 0.001
    assert len(cfg.experiment.task_factory.params.amass_dataset_conf.rel_dataset_path) == 22
    assert len(cfg.experiment.validation.amass_dataset_conf.rel_dataset_path) == 5
    assert cfg.experiment.validation.amass_dataset_conf.rel_dataset_path[-1].endswith(
        "/video10"
    )
    _assert_canonical_promotion_thresholds(cfg, "stage1")


def test_stage1r_uses_full_finger_physics_but_canonical_354_policy(monkeypatch):
    monkeypatch.setenv("STAGE1_PROMOTED_CHECKPOINT", "/tmp/promoted-stage1")
    cfg = _compose(
        "config_specific_task/stage1_body/"
        "conf_fullbody_forehand_clear_body_finger_isolated"
    )

    assert cfg.experiment.env_params.disable_fingers is False
    assert cfg.experiment.env_params.init_state_type == "FingerPerturbInitialStateHandler"
    assert cfg.experiment.env_params.init_state_params.finger_perturb_side == "right"
    assert cfg.experiment.env_params.init_state_params.finger_qpos_perturb_scale == 0.03
    assert cfg.experiment.env_params.reward_params.exclude_finger_joints is True
    assert cfg.experiment.env_params.reward_params.finger_grip_w_sum == 0.0
    assert cfg.experiment.finger_isolation.enabled is True
    assert list(cfg.experiment.finger_isolation.expected_partition) == [354, 31, 31]
    assert cfg.experiment.finger_isolation.expected_removed_observation_dim == 390
    assert cfg.experiment.finger_isolation.expected_policy_observation_dim == 2418
    assert len(cfg.experiment.finger_isolation.expected_actuator_partition_hash) == 64
    assert len(cfg.experiment.finger_isolation.expected_policy_observation_schema_hash) == 64
    assert len(cfg.experiment.finger_isolation.expected_policy_interface_hash) == 64
    assert cfg.experiment.total_timesteps == 20_480_000
    assert cfg.experiment.resume_from == "/tmp/promoted-stage1"
    assert cfg.experiment.parent_checkpoint_lineage.required is True
    assert cfg.experiment.parent_checkpoint_lineage.role == "stage1_promoted"
    assert cfg.experiment.reset_lr_schedule_on_resume is True
    assert cfg.experiment.reset_std_on_resume == 0.25
    assert cfg.experiment.promotion.eligible_as_stage2_teacher is False
    assert cfg.experiment.promotion.auto_stop is False
    assert cfg.experiment.validation.visual_review_kind is None
    _assert_canonical_promotion_thresholds(cfg, "stage1r")


def test_stage1r_005_has_independent_run_and_requires_accepted_003_checkpoint(monkeypatch):
    monkeypatch.delenv("STAGE1_PROMOTED_CHECKPOINT", raising=False)
    monkeypatch.setenv("STAGE1R_003_PROMOTED_CHECKPOINT", "/tmp/accepted-stage1r-003")
    cfg = _compose(
        "config_specific_task/stage1_body/"
        "conf_fullbody_forehand_clear_body_finger_isolated_005"
    )

    assert cfg.experiment.run_id == "forehand_clear_raw_smooth_v1_22_stage1r_finger_isolated_005_v1"
    assert cfg.experiment.run_id != "forehand_clear_raw_smooth_v1_22_stage1r_finger_isolated_003_v1"
    assert cfg.experiment.resume_from == "/tmp/accepted-stage1r-003"
    assert cfg.experiment.parent_checkpoint_lineage.required is True
    assert cfg.experiment.parent_checkpoint_lineage.role == "stage1r_003_accepted"
    assert cfg.experiment.env_params.init_state_params.finger_qpos_perturb_scale == 0.05
    assert cfg.experiment.promotion.prerequisite_perturbation_qpos_rad == 0.03
    assert cfg.experiment.promotion.perturbation_qpos_rad == 0.05
    assert cfg.experiment.promotion.next_perturbation_qpos_rad is None
    assert cfg.experiment.promotion.eligible_as_stage2_teacher is False
    assert cfg.experiment.promotion.auto_stop is False


def test_stage1r_005_fails_closed_without_accepted_003_checkpoint(monkeypatch):
    monkeypatch.delenv("STAGE1_PROMOTED_CHECKPOINT", raising=False)
    monkeypatch.delenv("STAGE1R_003_PROMOTED_CHECKPOINT", raising=False)
    cfg = _compose(
        "config_specific_task/stage1_body/"
        "conf_fullbody_forehand_clear_body_finger_isolated_005"
    )

    with pytest.raises(InterpolationResolutionError, match="STAGE1R_003_PROMOTED_CHECKPOINT"):
        _ = cfg.experiment.resume_from


def test_stage2_is_80m_fingerless_rigid_racket_adaptation():
    cfg = _compose(
        "config_specific_task/stage2_racket/"
        "conf_fullbody_badminton_racket_local"
    )

    assert cfg.experiment.env_params.env_name == "MjxMyoFullBodyRacket"
    assert cfg.experiment.env_params.disable_fingers is True
    assert cfg.experiment.run_id == "forehand_clear_raw_smooth_v1_22_stage2_racket_v1"
    assert cfg.experiment.training_source.variant == "raw_smooth_v1"
    assert cfg.experiment.total_timesteps == 80_000_000
    assert cfg.experiment.reset_lr_schedule_on_resume is True
    assert cfg.experiment.reset_std_on_resume == 0.5
    assert cfg.experiment.parent_checkpoint_lineage.required is True
    assert cfg.experiment.parent_checkpoint_lineage.role == "stage1_promoted"
    assert cfg.experiment.promotion.stage == "stage2"
    assert cfg.experiment.promotion.auto_stop is True
    assert cfg.experiment.promotion.consecutive_validations == 3
    assert cfg.experiment.promotion.min_validations_before_stop == 5
    assert cfg.experiment.promotion.baseline_metrics_path is None
    assert cfg.experiment.strict_auto_resume_config_hash is True
    assert cfg.experiment.validation.cover_all_trajectories is True
    assert cfg.experiment.validation.visual_review_kind == "stage2_racket"
    _assert_canonical_promotion_thresholds(cfg, "stage2")
    assert cfg.experiment.promotion.max_racket_position_error_m == 0.05
    assert cfg.experiment.promotion.max_racket_rotation_error_rad == 0.20


def test_stage2_extension_declares_stage2_80m_parent_lineage(monkeypatch):
    monkeypatch.setenv("STAGE2_080M_CHECKPOINT", "/tmp/accepted-stage2-80m")
    cfg = _compose(
        "config_specific_task/stage2_racket/"
        "conf_fullbody_badminton_racket_local_extend_160m"
    )

    assert cfg.experiment.resume_from == "/tmp/accepted-stage2-80m"
    assert cfg.experiment.parent_checkpoint_lineage.required is True
    assert cfg.experiment.parent_checkpoint_lineage.role == "stage2_080m_checkpoint"


def test_racket_student_bc_uses_same_354_racket_environment_contract():
    cfg = _compose(
        "config_specific_task/distill/"
        "conf_fullbody_forehandclear_racket_student_phase_bc"
    )

    assert cfg.experiment.env_params.env_name == "MjxMyoFullBodyRacket"
    assert cfg.experiment.env_params.disable_fingers is True
    assert cfg.experiment.run_id == "forehand_clear_raw_smooth_v1_stage2_racket_student_bc_v1"
    assert "raw_smooth_v1" in cfg.experiment.checkpoint_dir
    assert cfg.experiment.distill_contract.source_variant == "raw_smooth_v1"
    assert cfg.experiment.student_obs_filter.drop_goal_lookahead is True
    assert cfg.experiment.student_obs_filter.keep_motion_phase is True
    assert cfg.experiment.distill_contract.teacher_stage == "stage2_rigid_racket"
    assert cfg.experiment.distill_contract.action_dim == 354
    assert cfg.experiment.distill_contract.include_fingers is False


def test_racket_student_ppo_has_closed_loop_promotion_contract():
    cfg = _compose(
        "config_specific_task/distill/"
        "conf_fullbody_forehandclear_racket_student_phase_ppo"
    )

    assert cfg.experiment.env_params.env_name == "MjxMyoFullBodyRacket"
    assert cfg.experiment.run_id == "forehand_clear_raw_smooth_v1_stage2_racket_student_ppo_v1"
    assert "raw_smooth_v1" in cfg.experiment.checkpoint_dir
    assert cfg.experiment.total_timesteps == 102_400_000
    assert cfg.experiment.promotion.min_teacher_return_fraction == 0.90
    assert cfg.experiment.promotion.target_teacher_return_fraction == 0.95
    assert cfg.experiment.promotion.max_early_termination_gap == 0.05


def test_legacy_forehand_student_names_are_racket_aware_aliases():
    bc = _compose("config_specific_task/distill/conf_fullbody_forehandclear_student_phase_bc")
    ppo = _compose("config_specific_task/distill/conf_fullbody_forehandclear_student_phase_ppo")

    assert bc.experiment.env_params.env_name == "MjxMyoFullBodyRacket"
    assert ppo.experiment.env_params.env_name == "MjxMyoFullBodyRacket"
    assert bc.experiment.env_params.disable_fingers is True
    assert ppo.experiment.env_params.disable_fingers is True


@pytest.mark.parametrize(
    "name",
    [
        "conf_fullbody_badminton_student_gmr",
        "conf_fullbody_badminton_student_action_conditioned",
        "conf_fullbody_badminton_student_bc_eval",
        "conf_fullbody_forehandclear_racket_student_gmr",
    ],
)
def test_legacy_and_experimental_distill_configs_require_explicit_runtime_opt_in(
    name,
):
    cfg = _compose(f"config_specific_task/distill/{name}")
    assert cfg.config_status.canonical is False
    assert cfg.config_status.allow_nonproduction_runtime is False
    with pytest.raises(ValueError, match="refusing non-production"):
        validate_experiment_config_status(cfg)

    cfg.config_status.allow_nonproduction_runtime = True
    evidence = validate_experiment_config_status(cfg)
    assert evidence["explicit_opt_in"] is True
    assert cfg.experiment.nonproduction_runtime_opt_in.explicit_opt_in is True


def test_canonical_direct_and_latent_configs_are_not_blocked_by_legacy_gate():
    direct = _compose(
        "config_specific_task/distill/"
        "conf_fullbody_forehandclear_racket_student_phase_ppo"
    )
    latent = OmegaConf.load(
        FULLBODY_DIR
        / "config_specific_task"
        / "distill"
        / "latent_forehandclear_lab.yaml"
    )

    assert validate_experiment_config_status(direct) is None
    assert validate_experiment_config_status(latent) is None

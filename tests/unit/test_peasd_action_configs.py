"""Action-specific PEASD config composition and latent asset contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from musclemimic.badminton.action_registry import (
    CHINA_JUMP,
    FOREHAND_CLEAR,
    FOREHAND_LIFT,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FULLBODY_DIR = REPO_ROOT / "fullbody"


def _compose(name: str):
    with initialize_config_dir(version_base=None, config_dir=str(FULLBODY_DIR)):
        return compose(config_name=name)


def _latent(path: str) -> dict:
    payload = OmegaConf.to_container(OmegaConf.load(REPO_ROOT / path), resolve=True)
    return dict(payload["latent_distill"])


@pytest.fixture
def lift_parent_checkpoints(monkeypatch):
    monkeypatch.setenv("STAGE1_PROMOTED_CHECKPOINT", "/tmp/accepted-forehand-lift-stage1")
    monkeypatch.setenv("STAGE1R_003_PROMOTED_CHECKPOINT", "/tmp/accepted-forehand-lift-stage1r003")
    monkeypatch.setenv("STAGE2_080M_CHECKPOINT", "/tmp/accepted-forehand-lift-stage2-080m")


@pytest.mark.usefixtures("lift_parent_checkpoints")
def test_forehand_lift_stage1r_composes_two_fresh_action_matched_rungs() -> None:
    rung003 = _compose(
        "config_specific_task/stage1_body/"
        "conf_fullbody_forehand_lift_body_finger_isolated"
    ).experiment
    rung005 = _compose(
        "config_specific_task/stage1_body/"
        "conf_fullbody_forehand_lift_body_finger_isolated_005"
    ).experiment

    assert rung003.run_id != rung005.run_id
    assert "forehand_lift" in rung003.run_id
    assert "forehand_lift" in rung005.run_id
    assert rung003.auto_resume is False
    assert rung005.auto_resume is False
    assert rung003.reset_optimizer_on_resume is True
    assert rung005.reset_optimizer_on_resume is True
    assert rung003.resume_from == "/tmp/accepted-forehand-lift-stage1"
    assert rung005.resume_from == "/tmp/accepted-forehand-lift-stage1r003"
    assert rung003.env_params.init_state_params.finger_qpos_perturb_scale == 0.03
    assert rung005.env_params.init_state_params.finger_qpos_perturb_scale == 0.05
    assert rung003.promotion.perturbation_qpos_rad == 0.03
    assert rung005.promotion.perturbation_qpos_rad == 0.05
    assert rung003.env_params.env_name == "MjxMyoFullBodyRacket"
    assert rung003.env_params.disable_fingers is False
    assert list(rung003.finger_isolation.expected_partition) == [354, 31, 31]

    for experiment in (rung003, rung005):
        train = list(experiment.task_factory.params.amass_dataset_conf.rel_dataset_path)
        val = list(experiment.validation.amass_dataset_conf.rel_dataset_path)
        assert train == list(FOREHAND_LIFT.train_motion_paths)
        assert val == list(FOREHAND_LIFT.val_motion_paths)
        assert experiment.env_params.reward_params.root_pos_w_sum == 0.35
        assert experiment.env_params.reward_params.root_vel_w_sum == 0.25
        for terminal in (
            experiment.env_params.terminal_state_params,
            experiment.validation.terminal_state_params,
        ):
            assert terminal.mean_site_deviation_threshold == 0.45
            assert terminal.root_deviation_threshold == 0.30
            assert terminal.root_orientation_threshold == 0.70


@pytest.mark.usefixtures("lift_parent_checkpoints")
def test_forehand_lift_legacy_stage2_and_extension_preserve_lift_contract() -> None:
    stage2 = _compose(
        "config_specific_task/stage2_racket/"
        "conf_fullbody_forehand_lift_racket_local"
    ).experiment
    extension = _compose(
        "config_specific_task/stage2_racket/"
        "conf_fullbody_forehand_lift_racket_local_extend_160m"
    ).experiment

    assert stage2.run_id == "forehand_lift_optimized_root_smooth_v2_12_stage2_racket_080m_v1"
    assert extension.run_id.endswith("stage2_racket_extend_160m_v1")
    assert stage2.run_id != extension.run_id
    assert stage2.auto_resume is False
    assert stage2.resume_from is None
    assert stage2.reset_optimizer_on_resume is True
    assert extension.auto_resume is False
    assert extension.resume_from == "/tmp/accepted-forehand-lift-stage2-080m"
    assert extension.total_timesteps_is_absolute is False
    assert extension.reset_optimizer_on_resume is False
    assert stage2.total_timesteps == extension.total_timesteps == 80_000_000

    for experiment in (stage2, extension):
        assert experiment.env_params.env_name == "MjxMyoFullBodyRacket"
        assert experiment.env_params.disable_fingers is True
        assert list(experiment.task_factory.params.amass_dataset_conf.rel_dataset_path) == list(
            FOREHAND_LIFT.train_motion_paths
        )
        assert list(experiment.validation.amass_dataset_conf.rel_dataset_path) == list(
            FOREHAND_LIFT.val_motion_paths
        )
        assert experiment.env_params.reward_params.root_pos_w_sum == 0.35
        assert experiment.env_params.terminal_state_params.mean_site_deviation_threshold == 0.45
        assert experiment.validation.terminal_state_params.mean_site_deviation_threshold == 0.45
        assert experiment.promotion.max_racket_position_error_m == 0.05
        assert experiment.promotion.max_racket_rotation_error_rad == 0.20
        assert experiment.promotion.max_body_metric_relative_degradation == 0.10


@pytest.mark.usefixtures("lift_parent_checkpoints")
def test_forehand_lift_student_configs_compose_with_action_specific_split() -> None:
    bc = _compose(
        "config_specific_task/distill/"
        "conf_fullbody_forehandlift_racket_student_phase_bc"
    ).experiment
    ppo = _compose(
        "config_specific_task/distill/"
        "conf_fullbody_forehandlift_racket_student_phase_ppo"
    ).experiment

    assert bc.run_id != ppo.run_id
    assert bc.run_id.endswith("student_bc_v1")
    assert ppo.run_id.endswith("student_ppo_v1")
    assert bc.auto_resume is ppo.auto_resume is False
    assert bc.resume_from is ppo.resume_from is None
    assert "forehandlift" in bc.checkpoint_dir
    assert "forehandlift" in ppo.checkpoint_dir
    assert bc.distill_contract.action_id == "forehandLift"
    assert bc.distill_contract.source_variant == "optimized_root_smooth_v2"
    assert bc.distill_contract.validation_motion_count == 4
    assert bc.distill_contract.action_dim == 354
    assert list(bc.task_factory.params.amass_dataset_conf.rel_dataset_path) == list(
        FOREHAND_LIFT.train_motion_paths
    )
    assert list(bc.validation.amass_dataset_conf.rel_dataset_path) == list(
        FOREHAND_LIFT.val_motion_paths
    )
    assert ppo.total_timesteps == 102_400_000
    assert ppo.reset_optimizer_on_resume is True


@pytest.mark.parametrize(
    ("spec", "lab_path", "synergy_path", "dataset_prefix"),
    [
        (
            FOREHAND_LIFT,
            "fullbody/config_specific_task/distill/latent_forehandlift_lab.yaml",
            "fullbody/config_specific_task/distill/latent_forehandlift_synergy_v3.yaml",
            "datasets/forehandLift/",
        ),
        (
            CHINA_JUMP,
            "fullbody/config_specific_task/distill/latent_chinajump_lab.yaml",
            "fullbody/config_specific_task/distill/latent_chinajump_synergy_v3.yaml",
            "datasets/ChinaJump/",
        ),
    ],
    ids=("forehand_lift", "chinajump"),
)
def test_generalization_latent_configs_match_canonical_gates_and_action_paths(
    spec,
    lab_path: str,
    synergy_path: str,
    dataset_prefix: str,
) -> None:
    canonical_lab = _latent(FOREHAND_CLEAR.latent_lab_config)
    canonical_synergy = _latent(FOREHAND_CLEAR.latent_synergy_config)
    lab = _latent(lab_path)
    synergy = _latent(synergy_path)

    if spec.racket_applicable:
        assert lab["promotion_gates"] == canonical_lab["promotion_gates"]
        assert synergy["promotion_gates"] == canonical_synergy["promotion_gates"]
    else:
        # ChinaJump keeps every action-neutral threshold but cannot claim a
        # racket-relative direct-student gate for evidence it never produces.
        for config, canonical in ((lab, canonical_lab), (synergy, canonical_synergy)):
            assert "closed_loop_max_body_racket_relative_degradation" not in config["promotion_gates"]
            expected = dict(canonical["promotion_gates"])
            expected.pop("closed_loop_max_body_racket_relative_degradation")
            assert config["promotion_gates"] == expected
            assert config["require_direct_bc_baseline"] is False
            assert config["closed_loop_tracking_metrics"] == ["err_rpos"]
            assert config["teacher_promotion_stage"] == "stage1"
            assert config["teacher_promotion_role"] == "body_only"
            assert "CHINAJUMP_STAGE1_PROMOTION_MANIFEST" in config[
                "teacher_promotion_manifest"
            ]
    for key in (
        "latent_dim",
        "hidden_layer_dims",
        "batch_size",
        "horizon",
        "num_steps",
        "learning_rate",
        "kl_weight",
        "smooth_weight",
        "bound_weight",
    ):
        assert synergy[key] == canonical_synergy[key]
    for config in (lab, synergy):
        assert config["dataset_dir"].startswith(dataset_prefix)
        assert config["val_dataset_dir"].startswith(dataset_prefix)
        assert config["output_dir"].startswith(dataset_prefix)
        assert config["expected_val_motion_count"] == len(spec.val_motions)
        assert config["val_fraction"] == 0.0

    expected_phase_contract = {
        "phase_field": spec.latent_phase_field,
        "phases": [
            {"id": phase_id, "name": name}
            for phase_id, name in spec.latent_phases
        ],
        "require_all_phases": spec.latent_require_all_phases,
    }
    assert synergy["phase_contract"] == expected_phase_contract
    assert synergy["phase_field"] is None
    assert synergy["phase_balance_weights"] is None


def test_canonical_latent_phase_contract_matches_registry() -> None:
    synergy = _latent(FOREHAND_CLEAR.latent_synergy_config)
    assert synergy["expected_val_motion_count"] == FOREHAND_CLEAR.latent_expected_val_motion_count
    assert synergy["phase_contract"] == {
        "phase_field": FOREHAND_CLEAR.latent_phase_field,
        "phases": [
            {"id": phase_id, "name": name}
            for phase_id, name in FOREHAND_CLEAR.latent_phases
        ],
        "require_all_phases": FOREHAND_CLEAR.latent_require_all_phases,
    }

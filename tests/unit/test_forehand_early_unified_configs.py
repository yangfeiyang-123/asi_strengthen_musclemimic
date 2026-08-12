from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
FULLBODY_DIR = REPO_ROOT / "fullbody"

CONFIGS = (
    "config_specific_task/stage1_body/"
    "conf_fullbody_forehand_clear_early_unified_synergy_v4",
    "config_specific_task/stage2_racket_v2/"
    "conf_fullbody_forehand_clear_racket_mass_025_early_unified_synergy_v4",
    "config_specific_task/stage2_racket_v2/"
    "conf_fullbody_forehand_clear_racket_mass_050_early_unified_synergy_v4",
    "config_specific_task/stage2_racket_v2/"
    "conf_fullbody_forehand_clear_racket_mass_075_early_unified_synergy_v4",
    "config_specific_task/stage2_racket_v2/"
    "conf_fullbody_forehand_clear_racket_mass_100_early_unified_synergy_v4",
)


def _compose_all():
    with initialize_config_dir(version_base=None, config_dir=str(FULLBODY_DIR)):
        return [compose(config_name=name) for name in CONFIGS]


def test_early_unified_stage1_and_mass_rungs_share_one_action_contract():
    configs = _compose_all()
    action_configs = [
        OmegaConf.to_container(
            cfg.experiment.action_representation,
            resolve=False,
        )
        for cfg in configs
    ]

    assert all(action == action_configs[0] for action in action_configs[1:])
    action = action_configs[0]
    assert action["mode"] == "fixed_synergy"
    assert action["expected_underlying_action_dim"] == 354
    assert action["expected_basis_region"] == "hybrid_global_regional"
    assert action["require_primitive_source_contract"] is True
    assert action["primitive_runtime_model_compatibility"] == "portable_body_action_abi"
    assert action["require_hybrid_dynamic_coverage"] is True
    assert action["require_all_basis_gates"] is True
    assert action["forbid_fallback_selected_basis"] is True
    assert action["learned_full_dimensional_baseline"] is False
    assert action["tonic_baseline"]["learned_full_dimensional"] is False
    assert action["residual"] == {"enabled": False, "alpha": 0.0}
    assert action["basis_path"] == "${oc.env:MUSCLEMIMIC_FOREHAND_UNIFIED_SYNERGY_BASIS,''}"
    assert action["coefficient_transform"]["stats_path"] == (
        "${oc.env:MUSCLEMIMIC_FOREHAND_UNIFIED_COEFFICIENT_STATS,''}"
    )


def test_early_unified_variants_have_fresh_runs_and_preserve_mass_curriculum():
    configs = _compose_all()
    run_ids = [str(cfg.experiment.run_id) for cfg in configs]

    assert len(run_ids) == len(set(run_ids)) == 5
    assert all("early_unified_synergy_v4" in run_id for run_id in run_ids)
    assert all(cfg.experiment.auto_resume is False for cfg in configs)
    assert configs[0].experiment.env_params.env_name == "MjxMyoFullBody"
    assert configs[0].experiment.env_params.disable_fingers is True
    assert [
        float(cfg.experiment.env_params.racket_mass_scale) for cfg in configs[1:]
    ] == [0.25, 0.50, 0.75, 1.0]
    assert all(
        cfg.experiment.env_params.env_name == "MjxMyoFullBodyRacket"
        for cfg in configs[1:]
    )
    assert all(cfg.experiment.env_params.disable_fingers is True for cfg in configs[1:])

    stage2_experiments = [
        OmegaConf.to_container(cfg.experiment, resolve=False) for cfg in configs[1:]
    ]
    assert [experiment["resume_from"] for experiment in stage2_experiments] == [
        "${oc.env:FOREHAND_EARLY_UNIFIED_STAGE1_PROMOTED_CHECKPOINT}",
        "${oc.env:FOREHAND_EARLY_UNIFIED_STAGE2_MASS_025_PROMOTED_CHECKPOINT}",
        "${oc.env:FOREHAND_EARLY_UNIFIED_STAGE2_MASS_050_PROMOTED_CHECKPOINT}",
        "${oc.env:FOREHAND_EARLY_UNIFIED_STAGE2_MASS_075_PROMOTED_CHECKPOINT}",
    ]
    assert [
        experiment["parent_checkpoint_lineage"]["role"]
        for experiment in stage2_experiments
    ] == [
        "stage1_early_unified_synergy_v4_promoted",
        "stage2_mass_025_early_unified_synergy_v4_promoted",
        "stage2_mass_050_early_unified_synergy_v4_promoted",
        "stage2_mass_075_early_unified_synergy_v4_promoted",
    ]
    assert all(
        experiment["parent_checkpoint_lineage"]["required"] is True
        for experiment in stage2_experiments
    )
    assert all(experiment["reset_optimizer_on_resume"] is True for experiment in stage2_experiments)
    assert all(experiment["reset_lr_schedule_on_resume"] is True for experiment in stage2_experiments)

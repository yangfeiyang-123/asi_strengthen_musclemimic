"""Contracts for the three-action registry behind PEASD generalization.

The registry is the single place that knows which assets each action owns.
These tests pin the two properties that keep a multi-action run honest:
forehand clear's identity is unchanged, and a missing asset fails loudly
instead of silently borrowing another action's file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from musclemimic.badminton.action_registry import (
    ACTIONS,
    CHINA_JUMP,
    FOREHAND_CLEAR,
    FOREHAND_LIFT,
    action_choices,
    emg_trial_action_choices,
    resolve,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_forehand_clear_identity_is_unchanged() -> None:
    """The canonical action's sealed values must survive the refactor."""
    spec = FOREHAND_CLEAR
    assert spec.action_id == "forehandClear_standard"
    assert spec.source_variant == "raw_smooth_v1"
    assert spec.cache_variant == "raw_smooth_v1"
    assert spec.source_bucket == "temp"
    assert len(spec.train_motions) == 22
    assert len(spec.val_motions) == 5
    assert spec.val_motions == (
        "6月2日(1)-3",
        "6月2日(1)-5",
        "6月2日-1",
        "6月2日-5",
        "video10",
    )


def test_motion_paths_match_training_config_spelling() -> None:
    assert FOREHAND_CLEAR.motion_path("video1") == (
        "forehandClear_standard/muscle_trajectory/raw_smooth_v1/video1"
    )
    assert CHINA_JUMP.motion_path("forehandJump-1") == (
        "ChinaJump/muscle_trajectory/optimized/forehandJump-1"
    )
    assert FOREHAND_LIFT.motion_path("forehandLift-1") == (
        "forehandLift/muscle_trajectory/optimized_root_smooth_v2/forehandLift-1"
    )


def test_chinajump_reads_from_wham_not_temp() -> None:
    """ChinaJump is the reason source_bucket exists; clear/lift use temp."""
    assert CHINA_JUMP.source_bucket == "wham"
    assert CHINA_JUMP.source_variant == "optimized_wham"
    assert CHINA_JUMP.cache_variant == "optimized"
    assert FOREHAND_LIFT.source_bucket == "temp"


@pytest.mark.parametrize("spec", list(ACTIONS.values()), ids=lambda s: s.slug)
def test_every_split_is_disjoint_and_non_empty(spec) -> None:
    spec.validate()
    assert not set(spec.train_motions) & set(spec.val_motions)


@pytest.mark.parametrize("spec", list(ACTIONS.values()), ids=lambda s: s.slug)
def test_every_action_declares_stage1_and_emg(spec) -> None:
    """Step (1) and Stage1 are the aligned floor for all three actions."""
    assert spec.stage1_config
    assert spec.emg_trial_actions
    assert spec.env_prefix.startswith("MUSCLEMIMIC_")


def test_missing_asset_names_the_action_and_field() -> None:
    """A gap must fail closed with an actionable message, never fall back."""
    with pytest.raises(ValueError) as excinfo:
        CHINA_JUMP.require("stage3_v2_spec")
    message = str(excinfo.value)
    assert "ChinaJump" in message
    assert "stage3_v2_spec" in message
    assert "chinajump" in message


def test_require_rejects_unknown_field_name() -> None:
    with pytest.raises(AttributeError):
        FOREHAND_CLEAR.require("not_a_field")


def test_require_returns_declared_assets() -> None:
    assert FOREHAND_CLEAR.require("stage3_v2_spec").endswith(".yaml")
    assert FOREHAND_CLEAR.require("stage1_config")


def test_resolve_accepts_slug_and_action_id() -> None:
    assert resolve("forehand_clear") is FOREHAND_CLEAR
    assert resolve("forehandClear_standard") is FOREHAND_CLEAR
    assert resolve("chinajump") is CHINA_JUMP
    assert resolve("ChinaJump") is CHINA_JUMP


def test_resolve_rejects_unknown_action_with_choices() -> None:
    with pytest.raises(ValueError) as excinfo:
        resolve("backhand_smash")
    message = str(excinfo.value)
    assert "backhand_smash" in message
    assert "forehand_clear" in message


def test_action_choices_covers_the_registry() -> None:
    assert set(action_choices()) == set(ACTIONS)
    assert len(action_choices()) == 3


def test_emg_trial_choices_are_resolvable_actions() -> None:
    """The tube builder's --action choices must all resolve."""
    choices = emg_trial_action_choices()
    assert choices
    for name in choices:
        assert resolve(name).emg_trial_actions


def test_env_prefixes_are_unique_per_action() -> None:
    """Shared prefixes would let one action's basis satisfy another's gate."""
    prefixes = [spec.env_prefix for spec in ACTIONS.values()]
    assert len(set(prefixes)) == len(prefixes)


def test_env_var_composes_suffix() -> None:
    assert CHINA_JUMP.env_var("SYNERGY_BASIS").endswith("_SYNERGY_BASIS")
    assert CHINA_JUMP.env_var("SYNERGY_BASIS").startswith("MUSCLEMIMIC_")


def test_stage_applicability_is_distinct_from_asset_readiness() -> None:
    assert (
        FOREHAND_CLEAR.stage1r_applicable,
        FOREHAND_CLEAR.racket_applicable,
        FOREHAND_CLEAR.stage3_applicable,
    ) == (True, True, True)
    assert (
        FOREHAND_LIFT.stage1r_applicable,
        FOREHAND_LIFT.racket_applicable,
        FOREHAND_LIFT.stage3_applicable,
    ) == (True, True, True)
    assert (
        CHINA_JUMP.stage1r_applicable,
        CHINA_JUMP.racket_applicable,
        CHINA_JUMP.stage3_applicable,
    ) == (False, False, False)

    # Lift hitting is meaningful but is not ready until a calibrated lift task
    # spec exists.  The unrelated legacy net-lift spec must never satisfy it.
    assert FOREHAND_LIFT.stage3_spec is None
    assert FOREHAND_LIFT.stage3_v2_spec is None
    assert FOREHAND_LIFT.stage3_direct_spec is None
    assert "forehand_net_lift" not in repr(FOREHAND_LIFT)


def test_racket_mass_v2_is_bound_only_where_event_calibration_exists() -> None:
    assert FOREHAND_CLEAR.racket_event_bank_config
    assert FOREHAND_CLEAR.racket_mass_v2_configs is not None
    assert tuple(
        config.rsplit("mass_", 1)[-1]
        for config in FOREHAND_CLEAR.racket_mass_v2_configs
    ) == ("025", "050", "075", "100")
    assert FOREHAND_LIFT.racket_event_bank_config is None
    assert FOREHAND_LIFT.racket_mass_v2_configs is None
    assert CHINA_JUMP.racket_event_bank_config is None
    assert CHINA_JUMP.racket_mass_v2_configs is None


def test_latent_phase_and_validation_contracts_are_per_action() -> None:
    assert FOREHAND_CLEAR.latent_phase_ready is True
    assert FOREHAND_CLEAR.latent_phase_field == "phase_id"
    assert FOREHAND_CLEAR.latent_phases == (
        (0, "ready"),
        (1, "backswing"),
        (2, "acceleration"),
        (3, "impact"),
        (4, "followthrough"),
        (5, "recovery"),
    )
    assert FOREHAND_CLEAR.latent_require_all_phases is True

    for spec, expected_val_count in ((FOREHAND_LIFT, 4), (CHINA_JUMP, 2)):
        assert spec.latent_expected_val_motion_count == expected_val_count
        assert spec.latent_phase_ready is False
        assert spec.latent_phase_field is None
        assert spec.latent_phases == ()
        assert spec.latent_require_all_phases is False


def test_registry_uses_audited_action_neutral_354_grouping() -> None:
    # Preserve the sealed Clear path while new actions use the audited neutral
    # identity.  Both files declare the same complete 354 index partition.
    assert "forehand_clear" in FOREHAND_CLEAR.synergy_grouping
    paths = {FOREHAND_LIFT.synergy_grouping, CHINA_JUMP.synergy_grouping}
    assert len(paths) == 1
    relative = paths.pop()
    assert "forehand_clear" not in relative
    payload = json.loads((REPO_ROOT / relative).read_text(encoding="utf-8"))
    assert payload["action_scope"] == "action_neutral"
    assert set(payload["audit"]["applies_to_action_ids"]) == {
        "forehandClear_standard",
        "forehandLift",
        "ChinaJump",
    }
    covered: list[int] = []
    for region in payload["regions"]:
        for start, stop in region["index_ranges"]:
            covered.extend(range(start, stop))
    assert sorted(covered) == list(range(354))
    assert len(covered) == len(set(covered))

    sealed = json.loads((REPO_ROOT / FOREHAND_CLEAR.synergy_grouping).read_text(encoding="utf-8"))
    assert [(row["name"], row["index_ranges"]) for row in payload["regions"]] == [
        (row["name"], row["index_ranges"]) for row in sealed["regions"]
    ]

"""Contracts for the three-action registry behind PEASD generalization.

The registry is the single place that knows which assets each action owns.
These tests pin the two properties that keep a multi-action run honest:
forehand clear's identity is unchanged, and a missing asset fails loudly
instead of silently borrowing another action's file.
"""

from __future__ import annotations

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

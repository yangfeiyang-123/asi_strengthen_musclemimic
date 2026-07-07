from musclemimic.badminton.asi.curriculum import build_default_contact_tracking_curriculum, stage_for_update


def test_default_contact_tracking_curriculum_enables_terms_progressively():
    stages = build_default_contact_tracking_curriculum()

    assert [stage.name for stage in stages] == [
        "short_clean",
        "joint_tracking",
        "contact_tracking",
        "long_clips",
        "full_finetune",
    ]
    assert stages[0].max_quality_tier == "A"
    assert "foot_contact_height" not in stages[0].reward_terms
    assert "foot_contact_height" in stages[2].reward_terms
    assert stages[-1].max_quality_tier == "C"


def test_stage_for_update_selects_last_matching_stage():
    stages = build_default_contact_tracking_curriculum()

    assert stage_for_update(stages, 0).name == "short_clean"
    assert stage_for_update(stages, stages[2].start_update).name == "contact_tracking"
    assert stage_for_update(stages, 10**9).name == "full_finetune"

import pytest


def test_stage_transitions():
    from musclemimic.badminton.asi.contact_curriculum import (
        ContactCurriculumState,
        create_contact_curriculum,
        update_contact_curriculum,
    )

    cfg = {
        "stages": [
            {"name": "body_only", "start_update": 0, "foot_h": 0.0, "foot_v": 0.0, "graph": 0.0},
            {"name": "contact", "start_update": 100, "foot_h": 0.3, "foot_v": 0.3, "graph": 0.0},
            {"name": "full", "start_update": 500, "foot_h": 0.45, "foot_v": 0.45, "graph": 0.0},
        ]
    }
    state = create_contact_curriculum(cfg)
    assert state.foot_contact_height_w == 0.0

    state = update_contact_curriculum(state, update_count=50)
    assert state.foot_contact_height_w == 0.0
    assert state.stage_name == "body_only"

    state = update_contact_curriculum(state, update_count=100)
    assert state.foot_contact_height_w == pytest.approx(0.3)
    assert state.stage_name == "contact"

    state = update_contact_curriculum(state, update_count=999)
    assert state.foot_contact_height_w == pytest.approx(0.45)
    assert state.stage_name == "full"
    assert state.body_graph_w == 0.0


def test_nonzero_graph_reward_is_rejected_until_implemented():
    from musclemimic.badminton.asi.contact_curriculum import create_contact_curriculum

    config = {
        "stages": [
            {
                "name": "unsupported_graph",
                "start_update": 0,
                "foot_h": 0.0,
                "foot_v": 0.0,
                "graph": 0.1,
            }
        ]
    }

    with pytest.raises(ValueError, match="body-graph Laplacian reward is not implemented"):
        create_contact_curriculum(config)


def test_empty_stages_uses_defaults():
    from musclemimic.badminton.asi.contact_curriculum import create_contact_curriculum

    state = create_contact_curriculum({})
    assert state.foot_contact_height_w == 0.0

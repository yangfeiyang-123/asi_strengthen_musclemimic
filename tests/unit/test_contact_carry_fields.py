import dataclasses


def test_loco_carry_has_contact_fields():
    from musclemimic.environments.base import LocoCarry

    field_names = {f.name for f in dataclasses.fields(LocoCarry)}
    assert "foot_contact_height_w_sum" in field_names
    assert "foot_contact_velocity_w_sum" in field_names
    assert "body_graph_w_sum" in field_names

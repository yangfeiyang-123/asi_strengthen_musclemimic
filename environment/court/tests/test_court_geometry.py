from __future__ import annotations

import math

from court_geometry import CourtParams


def test_nominal_dimensions_and_derived_edges() -> None:
    court = CourtParams()

    assert court.full_court_length == 13.40
    assert court.doubles_width == 6.10
    assert court.singles_width == 5.18
    assert court.line_width == 0.040
    assert math.isclose(court.half_length, 6.70, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(court.half_width_doubles, 3.05, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(court.half_width_singles, 2.59, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(court.short_service_center_abs_x, 2.00, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(court.doubles_long_service_outer_edge_abs_x, 5.94, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(court.doubles_long_service_center_abs_x, 5.92, rel_tol=0.0, abs_tol=1e-12)


def test_rally_bounds_include_lines_and_exclude_one_mm_outside() -> None:
    court = CourtParams()

    assert court.inside_rally(6.70, 3.05, "doubles")
    assert court.inside_rally(-6.70, -3.05, "doubles")
    assert not court.inside_rally(6.701, 0.0, "doubles")
    assert not court.inside_rally(0.0, -3.051, "doubles")

    assert court.inside_rally(6.70, 2.59, "singles")
    assert court.inside_rally(-6.70, -2.59, "singles")
    assert not court.inside_rally(0.0, 2.591, "singles")
    assert court.inside_rally(0.0, 2.591, "doubles")


def test_service_bounds_include_relevant_lines() -> None:
    court = CourtParams()

    assert court.inside_service(1.98, 0.00, "doubles", "+x", "+y")
    assert court.inside_service(5.94, 3.05, "doubles", "+x", "+y")
    assert not court.inside_service(5.941, 2.00, "doubles", "+x", "+y")

    assert court.inside_service(-1.98, -0.01, "doubles", "-x", "-y")
    assert court.inside_service(-5.94, -3.05, "doubles", "-x", "-y")
    assert not court.inside_service(-5.941, -2.00, "doubles", "-x", "-y")

    assert court.inside_service(6.70, -2.59, "singles", "+x", "-y")
    assert not court.inside_service(6.701, -2.00, "singles", "+x", "-y")


def test_visual_line_rectangles_are_edge_correct() -> None:
    court = CourtParams()
    rects = {rect["name"]: rect for rect in court.visual_line_rectangles()}

    assert math.isclose(rects["doubles_sideline_pos_y"]["y"], 3.03, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(rects["doubles_sideline_neg_y"]["y"], -3.03, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(rects["singles_sideline_pos_y"]["y"], 2.57, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(rects["singles_sideline_neg_y"]["y"], -2.57, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(rects["back_boundary_pos_x"]["x"], 6.68, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(rects["back_boundary_neg_x"]["x"], -6.68, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(rects["short_service_line_pos_x"]["x"], 2.00, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(rects["short_service_line_neg_x"]["x"], -2.00, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(rects["doubles_long_service_line_pos_x"]["x"], 5.92, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(rects["doubles_long_service_line_neg_x"]["x"], -5.92, rel_tol=0.0, abs_tol=1e-12)


def test_net_height_profile_matches_center_and_sidelines() -> None:
    court = CourtParams()

    assert court.net_top_height(0.0) == 1.524
    assert court.net_top_height(3.05) == 1.550
    assert court.net_top_height(-3.05) == 1.550
    assert court.net_bottom_height(0.0) == 0.764
    assert math.isclose(court.net_top_height(1.525), 1.5305, rel_tol=0.0, abs_tol=1e-12)

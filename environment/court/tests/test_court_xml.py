from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VISUAL_ASSET = ROOT / "assets" / "badminton_court_bwf_visual.xml"
COLLISION_ASSET = ROOT / "assets" / "badminton_court_bwf_collision_net.xml"


def _root(path: Path) -> ET.Element:
    return ET.parse(path).getroot()


def _named(root: ET.Element, tag: str, name: str) -> ET.Element:
    for elem in root.iter(tag):
        if elem.attrib.get("name") == name:
            return elem
    raise AssertionError(f"{tag} not found: {name}")


def _geoms(root: ET.Element, prefix: str) -> list[ET.Element]:
    return [elem for elem in root.iter("geom") if elem.attrib.get("name", "").startswith(prefix)]


def test_required_visual_asset_elements_exist() -> None:
    root = _root(VISUAL_ASSET)

    assert root.tag == "mujoco"
    for name in [
        "floor_collision",
        "doubles_sideline_pos_y",
        "doubles_sideline_neg_y",
        "singles_sideline_pos_y",
        "singles_sideline_neg_y",
        "back_boundary_pos_x",
        "back_boundary_neg_x",
        "short_service_line_pos_x",
        "short_service_line_neg_x",
        "doubles_long_service_line_pos_x",
        "doubles_long_service_line_neg_x",
        "centre_service_line_pos_x_half",
        "centre_service_line_neg_x_half",
        "net_post_pos_y",
        "net_post_neg_y",
    ]:
        _named(root, "geom", name)

    for name in [
        "net_midpoint_site",
        "post_official_pos_y_site",
        "post_official_neg_y_site",
        "court_back_pos_x_site",
        "court_back_neg_x_site",
    ]:
        _named(root, "site", name)


def test_visual_asset_has_lines_without_collision_and_no_net_proxy() -> None:
    root = _root(VISUAL_ASSET)

    floor = _named(root, "geom", "floor_collision")
    assert floor.attrib["contype"] == "1"
    assert floor.attrib["conaffinity"] == "1"

    for prefix in [
        "doubles_sideline_",
        "singles_sideline_",
        "back_boundary_",
        "short_service_line_",
        "doubles_long_service_line_",
        "centre_service_line_",
    ]:
        for geom in _geoms(root, prefix):
            assert geom.attrib["contype"] == "0"
            assert geom.attrib["conaffinity"] == "0"

    assert _geoms(root, "net_collision_proxy_") == []


def test_collision_net_asset_enables_only_net_proxy_and_top_cord_collision() -> None:
    root = _root(COLLISION_ASSET)

    proxies = _geoms(root, "net_collision_proxy_")
    assert len(proxies) == 40
    assert {proxy.attrib["contype"] for proxy in proxies} == {"2"}
    assert {proxy.attrib["conaffinity"] for proxy in proxies} == {"1"}

    top_cords = _geoms(root, "net_top_cord_")
    assert len(top_cords) == 40
    assert {cord.attrib["contype"] for cord in top_cords} == {"2"}
    assert {cord.attrib["conaffinity"] for cord in top_cords} == {"1"}

    allowed_collision_names = {
        "floor_collision",
        "net_post_neg_y",
        "net_post_pos_y",
        *[proxy.attrib["name"] for proxy in proxies],
        *[cord.attrib["name"] for cord in top_cords],
    }
    collision_enabled_names = {
        geom.attrib.get("name", "<unnamed>")
        for geom in root.iter("geom")
        if geom.attrib.get("contype", "0") != "0" or geom.attrib.get("conaffinity", "0") != "0"
    }
    assert collision_enabled_names == allowed_collision_names

    visual_cords = _geoms(root, "net_vertical_cord_") + _geoms(root, "net_horizontal_cord_")
    assert visual_cords
    assert {cord.attrib["contype"] for cord in visual_cords} == {"0"}
    assert {cord.attrib["conaffinity"] for cord in visual_cords} == {"0"}

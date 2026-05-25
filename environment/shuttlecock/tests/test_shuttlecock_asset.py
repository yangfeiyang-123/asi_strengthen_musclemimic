from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "assets" / "shuttlecock_mujoco.xml"
PARAMS = ROOT / "params" / "shuttlecock_nominal.json"


def _body(root: ET.Element, name: str) -> ET.Element:
    for body in root.iter("body"):
        if body.attrib.get("name") == name:
            return body
    raise AssertionError(f"body not found: {name}")


def _geom(body: ET.Element, name: str) -> ET.Element:
    for geom in body.iter("geom"):
        if geom.attrib.get("name") == name:
            return geom
    raise AssertionError(f"geom not found: {name}")


def _site(body: ET.Element, name: str) -> ET.Element:
    for site in body.iter("site"):
        if site.attrib.get("name") == name:
            return site
    raise AssertionError(f"site not found: {name}")


def _vec(value: str) -> tuple[float, ...]:
    return tuple(float(part) for part in value.split())


def test_cork_contact_site_matches_cork_collision_center():
    root = ET.parse(ASSET).getroot()
    shuttle = _body(root, "shuttle")

    cork = _geom(shuttle, "cork_collision")
    contact = _site(shuttle, "cork_contact_site")

    assert _vec(contact.attrib["pos"]) == _vec(cork.attrib["pos"])
    assert contact.attrib["size"] == "0.0020"


def test_feathers_and_threads_are_visual_only():
    root = ET.parse(ASSET).getroot()
    shuttle = _body(root, "shuttle")

    for geom in shuttle.iter("geom"):
        name = geom.attrib.get("name", "")
        if name.startswith(("feather_", "thread_")):
            assert geom.attrib.get("class") == "visual"


def test_nominal_params_include_randomization_and_impact_sections():
    data = json.loads(PARAMS.read_text(encoding="utf-8"))

    assert data["randomization"]["mass_kg"] == [0.00474, 0.0055]
    assert data["randomization"]["terminal_velocity_m_s"] == [6.5, 6.9]
    assert data["racket_impact"]["shuttle_contact_site_name"] == "cork_contact_site"
    assert data["racket_impact"]["event_restitution_normal_range"] == [0.45, 0.6]

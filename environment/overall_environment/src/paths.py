from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
OVERALL_ROOT = REPO_ROOT / "environment" / "overall_environment"


def court_xml_path() -> Path:
    return REPO_ROOT / "environment" / "court" / "assets" / "badminton_court_bwf_collision_net.xml"


def racket_xml_path() -> Path:
    return REPO_ROOT / "environment" / "racket" / "assets" / "badminton_racket_rigid.xml"


def shuttlecock_xml_path() -> Path:
    return REPO_ROOT / "environment" / "shuttlecock" / "assets" / "shuttlecock_mujoco.xml"


def grip_reference_xml_path() -> Path:
    return REPO_ROOT / "assets" / "right_hand_racket_grip_scene.xml"


def grip_reference_json_path() -> Path:
    return REPO_ROOT / "configs" / "right_hand_racket_grip_reference.json"


def default_overall_scene_path() -> Path:
    return OVERALL_ROOT / "assets" / "overall_badminton_scene.xml"

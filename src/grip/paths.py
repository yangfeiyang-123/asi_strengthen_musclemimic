from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def racket_xml_path() -> Path:
    return REPO_ROOT / "environment" / "racket" / "assets" / "badminton_racket_rigid.xml"


def target_config_path() -> Path:
    return REPO_ROOT / "configs" / "right_hand_racket_grip_targets.json"


def scene_xml_path() -> Path:
    return REPO_ROOT / "assets" / "right_hand_racket_grip_scene.xml"


def reference_json_path() -> Path:
    return REPO_ROOT / "configs" / "right_hand_racket_grip_reference.json"


def grip_seed_json_path() -> Path:
    return REPO_ROOT / "outputs" / "right_hand_racket_grip" / "reference" / "right_hand_racket_grip_seed.json"


def grip_seed_reference_dir() -> Path:
    return grip_seed_json_path().parent

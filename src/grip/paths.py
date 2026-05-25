from __future__ import annotations

from pathlib import Path


_CHECKOUT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = _CHECKOUT_ROOT.parents[1] if _CHECKOUT_ROOT.parent.name == ".worktrees" else _CHECKOUT_ROOT


def racket_xml_path() -> Path:
    return REPO_ROOT / "environment" / "racket" / "assets" / "badminton_racket_rigid.xml"


def target_config_path() -> Path:
    return REPO_ROOT / "configs" / "right_hand_racket_grip_targets.json"


def scene_xml_path() -> Path:
    return REPO_ROOT / "assets" / "right_hand_racket_grip_scene.xml"


def reference_json_path() -> Path:
    return REPO_ROOT / "configs" / "right_hand_racket_grip_reference.json"

from __future__ import annotations

import pytest
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment.overall_environment.src.training_scene import (
    TrainingSceneReport,
    build_training_scene_report,
    default_training_scene_path,
    validate_training_scene_report,
)


def test_validate_training_scene_report_requires_training_keyframe_and_actuators():
    report = TrainingSceneReport(
        xml_path="scene.xml",
        keyframes=["overall_ready", "training_start"],
        actuator_count=416,
        required_sites=["right_hand_mimic", "racket_head_site"],
        missing_sites=[],
        required_geoms=["racket_stringbed"],
        missing_geoms=[],
        has_fullbody_racket_exclude=False,
    )

    validate_training_scene_report(report)


def test_validate_training_scene_report_rejects_missing_site():
    report = TrainingSceneReport(
        xml_path="scene.xml",
        keyframes=["overall_ready", "training_start"],
        actuator_count=416,
        required_sites=["right_hand_mimic"],
        missing_sites=["right_hand_mimic"],
        required_geoms=[],
        missing_geoms=[],
        has_fullbody_racket_exclude=False,
    )

    with pytest.raises(ValueError, match="missing sites"):
        validate_training_scene_report(report)


def test_validate_training_scene_report_rejects_fullbody_racket_exclude():
    report = TrainingSceneReport(
        xml_path="scene.xml",
        keyframes=["overall_ready", "training_start"],
        actuator_count=416,
        required_sites=[],
        missing_sites=[],
        required_geoms=[],
        missing_geoms=[],
        has_fullbody_racket_exclude=True,
    )

    with pytest.raises(ValueError, match="Full Body - overall_racket contact"):
        validate_training_scene_report(report)


def test_training_scene_xml_loads_with_mujoco():
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(default_training_scene_path()))

    assert model.nu > 0
    assert model.nkey >= 1
    assert not (model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_ACTUATION)


def test_build_training_scene_report_validates_required_objects():
    report = build_training_scene_report(default_training_scene_path())

    validate_training_scene_report(report)
    assert "overall_ready" in report.keyframes
    assert report.actuator_count > 0
    assert report.missing_sites == []
    assert report.missing_geoms == []
    assert report.has_fullbody_racket_exclude is False

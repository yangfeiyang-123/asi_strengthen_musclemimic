from __future__ import annotations

import json
from pathlib import Path

import pytest

from environment.overall_environment.src.action_manifest import (
    ActionManifest,
    load_action_manifest,
    write_action_manifest,
)


def test_write_and_load_action_manifest(tmp_path: Path):
    manifest = ActionManifest(
        schema_version=1,
        env_name="MjxMyoFullBody",
        disable_fingers=True,
        action_size=2,
        actuator_names=["hip", "shoulder"],
        obs_size=3,
        obs_fields=["root", "joint", "goal"],
        control_min=-1.0,
        control_max=1.0,
    )

    path = tmp_path / "action_manifest.json"
    write_action_manifest(path, manifest)
    loaded = load_action_manifest(path)

    assert loaded == manifest
    assert json.loads(path.read_text(encoding="utf-8"))["actuator_names"] == ["hip", "shoulder"]


def test_action_manifest_rejects_action_size_mismatch():
    with pytest.raises(ValueError, match="action_size"):
        ActionManifest(
            schema_version=1,
            env_name="MjxMyoFullBody",
            disable_fingers=True,
            action_size=3,
            actuator_names=["hip", "shoulder"],
            obs_size=3,
            obs_fields=["root"],
            control_min=-1.0,
            control_max=1.0,
        )


def test_reconstruct_legacy_manifest_from_env_params():
    manifest = ActionManifest.from_env_params(
        {
            "env_name": "MjxMyoFullBody",
            "disable_fingers": True,
        },
        actuator_names=["hip", "shoulder"],
        obs_size=5,
        obs_fields=["obs"],
    )

    assert manifest.disable_fingers is True
    assert manifest.action_size == 2
    assert manifest.actuator_names == ["hip", "shoulder"]
    assert manifest.obs_size == 5

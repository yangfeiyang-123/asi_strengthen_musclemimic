from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from environment.overall_environment.src.racket_attachment import (
    canonical_contract_fingerprint,
    load_racket_attachment_contract,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "name", ["musclemimic", "loco_mujoco", "environment", "fullbody", "analysis", "musclemimic.grip"]
)
def test_project_packages_load_from_src(name):
    spec = importlib.util.find_spec(name)
    assert spec is not None
    assert Path(spec.origin).is_relative_to(ROOT / "src")


@pytest.mark.parametrize(
    "filename",
    ["forehand_clear_rigid_v2.json", "forehand_clear_rigid_v3_custom.json", "forehand_clear_rigid_v4_custom.json"],
)
def test_sealed_racket_contract_keeps_its_identity_after_source_move(filename):
    path = ROOT / "configs" / "racket_attachment" / filename
    document = json.loads(path.read_text())
    contract = load_racket_attachment_contract(path)
    assert contract.fingerprint == document["fingerprint"] == canonical_contract_fingerprint(document)
    assert contract.racket_asset_path == document["racket_asset"]["path"]
    assert contract.verify_asset() == ROOT / "src" / document["racket_asset"]["path"]

from __future__ import annotations

from pathlib import Path

from court_geometry import CourtParams
from generate_court_mjcf import generate_mjcf

ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "params" / "court_bwf_nominal.json"
VISUAL_ASSET = ROOT / "assets" / "badminton_court_bwf_visual.xml"
COLLISION_ASSET = ROOT / "assets" / "badminton_court_bwf_collision_net.xml"


def test_committed_visual_asset_matches_generator_output() -> None:
    court = CourtParams.from_json(PARAMS)

    assert VISUAL_ASSET.read_text(encoding="utf-8") == generate_mjcf(
        court,
        enable_net_collision=False,
    )


def test_committed_collision_asset_matches_generator_output() -> None:
    court = CourtParams.from_json(PARAMS)

    assert COLLISION_ASSET.read_text(encoding="utf-8") == generate_mjcf(
        court,
        enable_net_collision=True,
    )

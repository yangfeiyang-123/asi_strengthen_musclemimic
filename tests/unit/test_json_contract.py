from __future__ import annotations

import json
from pathlib import Path

import pytest

from musclemimic.badminton.json_contract import DuplicateJsonKeyError, loads_json_strict


def test_strict_json_rejects_duplicate_keys() -> None:
    with pytest.raises(DuplicateJsonKeyError, match="duplicate JSON key"):
        loads_json_strict('{"value": 1, "value": 2}')


def test_raw_smooth_recipe_has_no_duplicate_keys() -> None:
    path = Path("musclemimic/badminton/scripts/raw_smooth_v1_recipe.json")
    payload = loads_json_strict(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "raw_smooth_source_recipe_v1"
    assert json.loads(path.read_text(encoding="utf-8")) == payload

"""Strict JSON loading helpers for versioned experiment contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DuplicateJsonKeyError(ValueError):
    """Raised when a JSON object contains a duplicate member name."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKeyError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def loads_json_strict(text: str) -> Any:
    """Decode JSON while rejecting the silent last-value-wins behaviour."""
    return json.loads(text, object_pairs_hook=_reject_duplicate_pairs)


def load_json_strict(path: str | Path) -> Any:
    """Read a UTF-8 JSON contract and reject duplicate object keys."""
    return loads_json_strict(Path(path).read_text(encoding="utf-8"))

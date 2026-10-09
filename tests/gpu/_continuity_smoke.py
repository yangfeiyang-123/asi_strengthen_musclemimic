from __future__ import annotations

import os

import pytest

from musclemimic.runner.continuity_smoke import load_continuity_training_smoke


def artifact_from_env(variable: str):
    path = os.environ.get(variable, "").strip()
    if not path:
        pytest.skip(f"{variable} must point to a generated GPU smoke artifact")
    return load_continuity_training_smoke(path)

"""Tests for distillation packaging and command registration."""

import tomllib
from pathlib import Path


def test_badmintonmimic_package_is_included_for_console_scripts():
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    include = data["tool"]["setuptools"]["packages"]["find"]["include"]
    scripts = data["project"]["scripts"]

    # badminton code now lives under the musclemimic package (musclemimic.badminton.*),
    # so the musclemimic* glob covers the console-script entry points.
    assert "musclemimic*" in include
    assert scripts["forehand-clear-distill-collect-teacher"].startswith("musclemimic.badminton.")
    assert scripts["musclemimic-distill-inspect-dataset"] == "musclemimic.distill.inspect_dataset:main"

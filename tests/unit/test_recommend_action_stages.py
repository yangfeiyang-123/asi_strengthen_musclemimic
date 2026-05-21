import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "BadmintonMimic" / "scripts" / "recommend_action_stages.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_cache(cache_root: Path, motion: str, root_end_x: float, peak_speed: float = 0.0, yaw: float = 0.0):
    path = cache_root / f"{motion}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    qpos = np.array(
        [
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [root_end_x, 0.0, 1.0, np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)],
        ],
        dtype=np.float32,
    )
    qvel = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [peak_speed, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    site_xpos = np.zeros((2, 9, 3), dtype=np.float32)
    site_xpos[1, 8, 0] = root_end_x + 0.2
    np.savez(path, qpos=qpos, qvel=qvel, site_xpos=site_xpos, frequency=np.asarray(100.0, dtype=np.float32))


def test_read_manifest_skips_comments_and_suffixes(tmp_path):
    module = _load_module(SCRIPT, "recommend_action_stages_manifest_for_test")
    manifest = tmp_path / "list.txt"
    manifest.write_text("# comment\nForehandClear/raw/video1.npz\n\nBackhand/best/video2\n", encoding="utf-8")

    rows = module._read_manifest(manifest)

    assert rows == ["ForehandClear/raw/video1", "Backhand/best/video2"]


def test_load_hints_matches_motion_and_action_prefix(tmp_path):
    module = _load_module(SCRIPT, "recommend_action_stages_hints_for_test")
    hints_file = tmp_path / "hints.yaml"
    hints_file.write_text(
        """
defaults:
  ForehandNetLift:
    expected_large_motion: true
motions:
  ForehandNetLift/best/video01:
    has_jump_or_lunge: true
""",
        encoding="utf-8",
    )

    hints = module._load_hints(hints_file)

    assert hints.for_motion("ForehandNetLift/best/video01").expected_large_motion is True
    assert hints.for_motion("ForehandNetLift/best/video01").has_jump_or_lunge is True
    assert hints.for_motion("ForehandNetLift/best/video02").expected_large_motion is True
    assert hints.for_motion("Backhand/best/video01").expected_large_motion is False


def test_resolve_cache_file_rejects_parent_traversal_inside_cache_root(tmp_path):
    module = _load_module(SCRIPT, "recommend_action_stages_traversal_for_test")
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    np.savez(cache_root / "bar.npz", qpos=np.zeros((2, 7)))

    with pytest.raises(ValueError, match="parent traversal|stay under cache root"):
        module._resolve_cache_file(cache_root, "foo/../bar")


def test_main_writes_recommendation_json(tmp_path):
    module = _load_module(SCRIPT, "recommend_action_stages_main_for_test")
    cache_root = tmp_path / "cache"
    manifest = tmp_path / "manifest.txt"
    output = tmp_path / "recommendations.json"
    manifest.write_text("ForehandNetLift/best/video01\nForehandClear/raw/video01\n", encoding="utf-8")
    _write_cache(cache_root, "ForehandNetLift/best/video01", root_end_x=0.75)
    _write_cache(cache_root, "ForehandClear/raw/video01", root_end_x=0.20)

    code = module.main(
        [
            "--cache-root",
            str(cache_root),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ]
    )

    assert code == 0
    rows = json.loads(output.read_text(encoding="utf-8"))
    assert rows[0]["motion"] == "ForehandNetLift/best/video01"
    assert rows[0]["stage"] == "posttrain"
    assert rows[1]["motion"] == "ForehandClear/raw/video01"
    assert rows[1]["stage"] == "base"

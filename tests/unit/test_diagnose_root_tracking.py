import importlib.util
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "BadmintonMimic" / "scripts" / "diagnose_root_tracking.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_resolve_cache_file_accepts_motion_without_suffix(tmp_path):
    diagnose = _load_module(SCRIPT, "diagnose_root_tracking_for_test")
    cache_root = tmp_path / "cache"
    cache_file = cache_root / "ForehandNetLift" / "best" / "video01.npz"
    cache_file.parent.mkdir(parents=True)
    np.savez(cache_file, qpos=np.zeros((2, 7)), qvel=np.zeros((2, 6)))

    resolved = diagnose._resolve_cache_file(cache_root, "ForehandNetLift/best/video01")

    assert resolved == cache_file


def test_diagnose_cache_file_outputs_reference_metrics(tmp_path):
    diagnose = _load_module(SCRIPT, "diagnose_root_tracking_metrics_for_test")
    cache_file = tmp_path / "motion.npz"
    qpos = np.array(
        [
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    qvel = np.zeros((2, 6), dtype=np.float32)
    np.savez(cache_file, qpos=qpos, qvel=qvel, frequency=np.asarray(100.0, dtype=np.float32))

    metrics = diagnose._diagnose_cache_file(cache_file, right_hand_site_index=None)

    assert metrics["motion"] == "motion"
    assert metrics["frames"] == 2
    assert metrics["frequency"] == 100.0
    assert metrics["reference_root_xy_total_displacement"] == 1.0


def test_write_json_report(tmp_path):
    diagnose = _load_module(SCRIPT, "diagnose_root_tracking_json_for_test")
    output = tmp_path / "report.json"
    rows = [{"motion": "a", "reference_root_xy_total_displacement": 1.0}]

    diagnose._write_json_report(output, rows)

    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded == rows

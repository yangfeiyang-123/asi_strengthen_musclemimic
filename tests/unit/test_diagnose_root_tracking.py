import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


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


def test_resolve_cache_file_rejects_absolute_motion_path(tmp_path):
    diagnose = _load_module(SCRIPT, "diagnose_root_tracking_absolute_path_for_test")

    with pytest.raises(ValueError, match="motion path must be relative"):
        diagnose._resolve_cache_file(tmp_path / "cache", str(tmp_path / "motion.npz"))


def test_resolve_cache_file_rejects_parent_traversal_outside_cache_root(tmp_path):
    diagnose = _load_module(SCRIPT, "diagnose_root_tracking_traversal_for_test")
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    outside = tmp_path / "outside.npz"
    np.savez(outside, qpos=np.zeros((2, 7)))

    with pytest.raises(ValueError, match="motion path must stay under cache root"):
        diagnose._resolve_cache_file(cache_root, "../outside")


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


def test_diagnose_cache_file_allows_missing_optional_qvel_and_frequency(tmp_path):
    diagnose = _load_module(SCRIPT, "diagnose_root_tracking_optional_arrays_for_test")
    cache_file = tmp_path / "motion.npz"
    qpos = np.array(
        [
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.savez(cache_file, qpos=qpos)

    metrics = diagnose._diagnose_cache_file(cache_file, right_hand_site_index=None)

    assert metrics["frames"] == 2
    assert metrics["frequency"] == 0.0
    assert metrics["reference_root_xy_total_displacement"] == 0.5
    assert metrics["reference_root_xy_peak_speed"] == 0.0


def test_main_reports_expected_errors_without_traceback(tmp_path, capsys):
    diagnose = _load_module(SCRIPT, "diagnose_root_tracking_main_error_for_test")

    code = diagnose.main(
        [
            "--cache-root",
            str(tmp_path / "cache"),
            "--motion",
            str(tmp_path / "outside.npz"),
        ]
    )

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert captured.err.startswith("error: motion path must be relative")
    assert "Traceback" not in captured.err


def test_module_loads_root_tracking_without_importing_utils_package(monkeypatch):
    for module_name in list(sys.modules):
        if module_name == "musclemimic.utils" or module_name.startswith("musclemimic.utils."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.delitem(sys.modules, "diagnose_root_tracking_import_for_test", raising=False)

    diagnose = _load_module(SCRIPT, "diagnose_root_tracking_import_for_test")

    assert "musclemimic.utils" not in sys.modules
    assert callable(diagnose.compute_root_reference_metrics)


def test_write_json_report(tmp_path):
    diagnose = _load_module(SCRIPT, "diagnose_root_tracking_json_for_test")
    output = tmp_path / "report.json"
    rows = [{"motion": "a", "reference_root_xy_total_displacement": 1.0}]

    diagnose._write_json_report(output, rows)

    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded == rows

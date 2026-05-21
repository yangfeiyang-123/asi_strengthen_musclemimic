import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "BadmintonMimic" / "scripts" / "build_stage_manifests.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_stage_manifests_groups_by_stage_and_family(tmp_path):
    module = _load_module(SCRIPT, "build_stage_manifests_for_test")
    report = tmp_path / "recommendations.json"
    output_dir = tmp_path / "generated"
    report.write_text(
        """
[
  {"motion": "ForehandClear/raw/video01", "stage": "base", "family": "general"},
  {"motion": "ForehandNetLift/best/video01", "stage": "posttrain", "family": "net_frontcourt"},
  {"motion": "Smash/best/video01", "stage": "posttrain", "family": "smash"},
  {"motion": "Backhand/best/video07", "stage": "repair", "family": "rotation"},
  {"motion": "NetTumble/raw/video01", "stage": "exclude", "family": "fine_hand"}
]
""",
        encoding="utf-8",
    )

    code = module.main(["--recommendations", str(report), "--output-dir", str(output_dir)])

    assert code == 0
    assert (output_dir / "base_general_list.txt").read_text(encoding="utf-8") == "ForehandClear/raw/video01\n"
    assert (
        (output_dir / "posttrain_net_frontcourt_list.txt").read_text(encoding="utf-8")
        == "ForehandNetLift/best/video01\n"
    )
    assert (output_dir / "posttrain_smash_list.txt").read_text(encoding="utf-8") == "Smash/best/video01\n"
    assert (output_dir / "repair_list.txt").read_text(encoding="utf-8") == "Backhand/best/video07\n"
    assert (output_dir / "exclude_list.txt").read_text(encoding="utf-8") == "NetTumble/raw/video01\n"


def test_empty_groups_are_not_written(tmp_path):
    module = _load_module(SCRIPT, "build_stage_manifests_empty_for_test")
    report = tmp_path / "recommendations.json"
    output_dir = tmp_path / "generated"
    report.write_text(
        '[{"motion": "ForehandClear/raw/video01", "stage": "base", "family": "general"}]',
        encoding="utf-8",
    )

    code = module.main(["--recommendations", str(report), "--output-dir", str(output_dir)])

    assert code == 0
    assert (output_dir / "base_general_list.txt").exists()
    assert not (output_dir / "repair_list.txt").exists()
    assert not (output_dir / "exclude_list.txt").exists()

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "BadmintonMimic" / "scripts" / "build_claim_evidence_template.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_claim_evidence_template_writes_expected_claims(tmp_path):
    module = _load_module(SCRIPT, "build_claim_evidence_template_for_test")
    output = tmp_path / "claim_evidence.json"

    code = module.main(["--output", str(output)])

    assert code == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert [claim["id"] for claim in data["claims"]] == [
        "staging_improves_stability",
        "posttrain_helps_large_motion",
        "repair_gate_protects_training",
        "metric_gated_beats_action_name_grouping",
    ]
    assert "all_mix" in data["required_experiments"]
    assert "no_repair_gate" in data["required_ablations"]


def test_template_output_is_deterministic(tmp_path):
    module = _load_module(SCRIPT, "build_claim_evidence_template_deterministic_for_test")
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    assert module.main(["--output", str(first)]) == 0
    assert module.main(["--output", str(second)]) == 0

    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")

# General Badminton Action Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a metric-gated action-stage workflow that recommends whether badminton motions belong in base training, post-training, repair, or exclusion.

**Architecture:** Add a small pure-Python classifier in `musclemimic/utils` and keep project-specific file traversal in `BadmintonMimic/scripts`. The first implementation writes diagnostic JSON and generated manifest files; training config templates remain a later, separate step after stage assignments are reviewed.

**Tech Stack:** Python 3.11, NumPy, PyYAML, argparse, pytest, existing retarget cache `.npz` files, existing `musclemimic.utils.root_tracking` metrics.

---

## File Structure

- Create `musclemimic/utils/action_stage.py`: pure metric-based classifier with dataclasses and no file I/O.
- Create `tests/unit/test_action_stage.py`: unit tests for threshold behavior and reason strings.
- Create `BadmintonMimic/scripts/recommend_action_stages.py`: CLI that reads manifests, loads GMR cache metrics, applies optional manual YAML hints, and writes stage recommendations.
- Create `tests/unit/test_recommend_action_stages.py`: CLI/unit tests using temporary manifests and cache files.
- Create `BadmintonMimic/scripts/build_stage_manifests.py`: CLI that reads recommendation JSON and writes `base_general_list.txt`, post-train family lists, `repair_list.txt`, and `exclude_list.txt`.
- Create `tests/unit/test_build_stage_manifests.py`: verifies grouping and deterministic manifest output.
- Modify `doc/PostTrain_Advice.md`: add a concise action-stage policy section that points to the new workflow.

The implementation intentionally avoids changing PPO, reward code, or Hydra training configs in this plan. Those should be a follow-up after the generated stage recommendations have been inspected.

---

### Task 1: Pure Action-Stage Classifier

**Files:**
- Create: `musclemimic/utils/action_stage.py`
- Create: `tests/unit/test_action_stage.py`

- [ ] **Step 1: Write failing tests for the classifier**

Create `tests/unit/test_action_stage.py`:

```python
import pytest

from musclemimic.utils.action_stage import MotionHints, classify_motion_stage


def test_stationary_visible_motion_is_base_candidate():
    metrics = {
        "reference_root_xy_total_displacement": 0.20,
        "reference_root_xy_peak_speed": 0.45,
        "reference_root_yaw_change": 0.10,
        "right_hand_world_path_length": 0.55,
    }

    result = classify_motion_stage(metrics, MotionHints(action_label="ForehandClear"))

    assert result.stage == "base"
    assert result.family == "general"
    assert "stationary_or_small_step" in result.reasons


def test_small_root_when_large_motion_expected_requires_repair():
    metrics = {
        "reference_root_xy_total_displacement": 0.12,
        "reference_root_xy_peak_speed": 0.40,
        "reference_root_yaw_change": 0.05,
        "right_hand_world_path_length": 0.60,
    }

    result = classify_motion_stage(
        metrics,
        MotionHints(action_label="ForehandNetLift", expected_large_motion=True),
    )

    assert result.stage == "repair"
    assert result.family == "net_frontcourt"
    assert "expected_large_motion_but_root_is_small" in result.reasons


def test_large_displacement_prefers_posttrain():
    metrics = {
        "reference_root_xy_total_displacement": 0.75,
        "reference_root_xy_peak_speed": 0.90,
        "reference_root_yaw_change": 0.20,
        "right_hand_world_path_length": 1.10,
    }

    result = classify_motion_stage(metrics, MotionHints(action_label="ForehandNetLift"))

    assert result.stage == "posttrain"
    assert result.family == "net_frontcourt"
    assert "large_root_displacement" in result.reasons


def test_high_speed_prefers_posttrain_even_with_medium_displacement():
    metrics = {
        "reference_root_xy_total_displacement": 0.42,
        "reference_root_xy_peak_speed": 1.35,
        "reference_root_yaw_change": 0.20,
        "right_hand_world_path_length": 1.00,
    }

    result = classify_motion_stage(metrics, MotionHints(action_label="Backhand"))

    assert result.stage == "posttrain"
    assert result.family == "rotation"
    assert "high_root_peak_speed" in result.reasons


def test_large_yaw_prefers_rotation_posttrain():
    metrics = {
        "reference_root_xy_total_displacement": 0.35,
        "reference_root_xy_peak_speed": 0.80,
        "reference_root_yaw_change": 1.10,
        "right_hand_world_path_length": 0.95,
    }

    result = classify_motion_stage(metrics, MotionHints(action_label="Smash"))

    assert result.stage == "posttrain"
    assert result.family == "rotation"
    assert "large_yaw_change" in result.reasons


def test_jump_or_lunge_hint_prefers_posttrain_family():
    metrics = {
        "reference_root_xy_total_displacement": 0.38,
        "reference_root_xy_peak_speed": 0.95,
        "reference_root_yaw_change": 0.25,
        "right_hand_world_path_length": 0.90,
    }

    result = classify_motion_stage(
        metrics,
        MotionHints(action_label="Smash", has_jump_or_lunge=True),
    )

    assert result.stage == "posttrain"
    assert result.family == "smash"
    assert "jump_or_lunge_hint" in result.reasons


def test_fine_hand_dominant_hint_excludes_motion():
    metrics = {
        "reference_root_xy_total_displacement": 0.08,
        "reference_root_xy_peak_speed": 0.25,
        "reference_root_yaw_change": 0.05,
        "right_hand_world_path_length": 0.25,
    }

    result = classify_motion_stage(
        metrics,
        MotionHints(action_label="NetTumble", fine_hand_dominant=True),
    )

    assert result.stage == "exclude"
    assert result.family == "fine_hand"
    assert "fine_hand_dominant" in result.reasons


def test_missing_required_metric_raises_clear_error():
    with pytest.raises(KeyError, match="reference_root_xy_total_displacement"):
        classify_motion_stage({}, MotionHints(action_label="ForehandClear"))
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
pytest tests/unit/test_action_stage.py -v
```

Expected: FAIL because `musclemimic.utils.action_stage` does not exist.

- [ ] **Step 3: Implement the classifier**

Create `musclemimic/utils/action_stage.py`:

```python
"""Classify badminton motions into training stages from reference metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


STAGE_BASE = "base"
STAGE_POSTTRAIN = "posttrain"
STAGE_REPAIR = "repair"
STAGE_EXCLUDE = "exclude"


@dataclass(frozen=True)
class MotionHints:
    action_label: str = ""
    expected_large_motion: bool = False
    has_jump_or_lunge: bool = False
    contact_unreliable: bool = False
    endpoint_unreliable: bool = False
    fine_hand_dominant: bool = False


@dataclass(frozen=True)
class StageDecision:
    stage: str
    family: str
    reasons: tuple[str, ...]


def _metric(metrics: Mapping[str, float], name: str) -> float:
    if name not in metrics:
        raise KeyError(name)
    return float(metrics[name])


def _label_family(label: str) -> str:
    normalized = label.lower()
    if "net" in normalized or "lift" in normalized or "front" in normalized:
        return "net_frontcourt"
    if "smash" in normalized:
        return "smash"
    if "backhand" in normalized or "turn" in normalized or "rotation" in normalized:
        return "rotation"
    if "footwork" in normalized or "step" in normalized or "lunge" in normalized:
        return "footwork"
    if "tumble" in normalized or "slice" in normalized or "wrist" in normalized:
        return "fine_hand"
    return "general"


def _posttrain_family(label_family: str, reasons: list[str]) -> str:
    if "large_yaw_change" in reasons:
        return "rotation"
    if label_family in {"net_frontcourt", "smash", "rotation", "footwork"}:
        return label_family
    return "general"


def classify_motion_stage(metrics: Mapping[str, float], hints: MotionHints | None = None) -> StageDecision:
    """Return a stage recommendation for one retargeted badminton motion.

    The thresholds are intentionally simple and conservative. They match the
    first design pass and should be tuned after real training outcomes exist.
    """

    hints = hints or MotionHints()
    root_disp = _metric(metrics, "reference_root_xy_total_displacement")
    root_peak_speed = _metric(metrics, "reference_root_xy_peak_speed")
    yaw_change = abs(_metric(metrics, "reference_root_yaw_change"))
    _metric(metrics, "right_hand_world_path_length")

    label_family = _label_family(hints.action_label)
    reasons: list[str] = []

    if hints.fine_hand_dominant:
        return StageDecision(
            stage=STAGE_EXCLUDE,
            family="fine_hand",
            reasons=("fine_hand_dominant",),
        )

    if hints.contact_unreliable:
        reasons.append("contact_unreliable")
    if hints.endpoint_unreliable:
        reasons.append("endpoint_unreliable")
    if hints.expected_large_motion and root_disp < 0.25:
        reasons.append("expected_large_motion_but_root_is_small")

    if any(reason in reasons for reason in ("contact_unreliable", "endpoint_unreliable", "expected_large_motion_but_root_is_small")):
        return StageDecision(
            stage=STAGE_REPAIR,
            family=label_family,
            reasons=tuple(reasons),
        )

    if root_disp < 0.25:
        reasons.append("stationary_or_small_step")
    elif root_disp <= 0.60:
        reasons.append("medium_root_displacement")
    else:
        reasons.append("large_root_displacement")

    if root_peak_speed > 1.20:
        reasons.append("high_root_peak_speed")
    if yaw_change > 0.80:
        reasons.append("large_yaw_change")
    if hints.has_jump_or_lunge:
        reasons.append("jump_or_lunge_hint")

    posttrain_reasons = {"large_root_displacement", "high_root_peak_speed", "large_yaw_change", "jump_or_lunge_hint"}
    if posttrain_reasons.intersection(reasons):
        return StageDecision(
            stage=STAGE_POSTTRAIN,
            family=_posttrain_family(label_family, reasons),
            reasons=tuple(reasons),
        )

    return StageDecision(
        stage=STAGE_BASE,
        family="general" if label_family == "fine_hand" else label_family,
        reasons=tuple(reasons),
    )
```

- [ ] **Step 4: Run classifier tests**

Run:

```bash
pytest tests/unit/test_action_stage.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add musclemimic/utils/action_stage.py tests/unit/test_action_stage.py
git commit -m "feat: add badminton action stage classifier"
```

---

### Task 2: Recommendation CLI

**Files:**
- Create: `BadmintonMimic/scripts/recommend_action_stages.py`
- Create: `tests/unit/test_recommend_action_stages.py`

- [ ] **Step 1: Write failing tests for manifest and hint loading**

Create `tests/unit/test_recommend_action_stages.py`:

```python
import importlib.util
import json
from pathlib import Path

import numpy as np


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
```

- [ ] **Step 2: Run the new CLI tests and verify they fail**

Run:

```bash
pytest tests/unit/test_recommend_action_stages.py -v
```

Expected: FAIL because `BadmintonMimic/scripts/recommend_action_stages.py` does not exist.

- [ ] **Step 3: Implement the recommendation CLI**

Create `BadmintonMimic/scripts/recommend_action_stages.py`:

```python
#!/usr/bin/env python3
"""Recommend base/posttrain/repair/exclude stages for badminton motion manifests."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from musclemimic.utils.action_stage import MotionHints, classify_motion_stage


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_TRACKING = REPO_ROOT / "musclemimic" / "utils" / "root_tracking.py"
EXPECTED_USER_ERRORS = (FileNotFoundError, KeyError, ValueError, IndexError)


def _load_compute_root_reference_metrics():
    spec = importlib.util.spec_from_file_location(
        "_recommend_action_stages_root_tracking",
        ROOT_TRACKING,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load root tracking module: {ROOT_TRACKING}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_root_reference_metrics


compute_root_reference_metrics = _load_compute_root_reference_metrics()


class HintTable:
    def __init__(self, defaults: dict[str, dict[str, Any]] | None = None, motions: dict[str, dict[str, Any]] | None = None):
        self._defaults = defaults or {}
        self._motions = motions or {}

    def for_motion(self, motion: str) -> MotionHints:
        action_label = motion.split("/", 1)[0]
        values: dict[str, Any] = {"action_label": action_label}
        values.update(self._defaults.get(action_label, {}))
        values.update(self._motions.get(motion, {}))
        return MotionHints(**values)


def _read_manifest(path: Path) -> list[str]:
    motions: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        motions.append(line.removesuffix(".npz"))
    return motions


def _load_hints(path: Path | None) -> HintTable:
    if path is None:
        return HintTable()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return HintTable(
        defaults=data.get("defaults", {}),
        motions=data.get("motions", {}),
    )


def _resolve_cache_file(cache_root: Path, motion: str) -> Path:
    motion_path = Path(motion)
    if motion_path.is_absolute():
        raise ValueError(f"motion path must be relative to cache root: {motion}")
    if motion_path.suffix == "":
        motion_path = motion_path.with_suffix(".npz")

    resolved_cache_root = cache_root.resolve()
    cache_file = (cache_root / motion_path).resolve()
    if cache_file != resolved_cache_root and resolved_cache_root not in cache_file.parents:
        raise ValueError(f"motion path must stay under cache root: {motion}")
    if not cache_file.exists():
        raise FileNotFoundError(f"cache file does not exist: {cache_file}")
    return cache_file


def _float_scalar(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    array = np.asarray(value)
    if array.size == 0:
        return float(default)
    return float(array.reshape(-1)[0])


def _metrics_for_cache(cache_file: Path, right_hand_site_index: int | None) -> dict[str, float]:
    with np.load(cache_file) as data:
        qpos = data["qpos"]
        qvel = data["qvel"] if "qvel" in data else None
        site_xpos = data["site_xpos"] if "site_xpos" in data else None
        frequency = _float_scalar(data["frequency"]) if "frequency" in data else None
        return compute_root_reference_metrics(
            qpos=qpos,
            qvel=qvel,
            site_xpos=site_xpos,
            right_hand_site_index=right_hand_site_index,
            frequency=frequency,
        )


def _recommend_motion(cache_root: Path, motion: str, hints: MotionHints, right_hand_site_index: int | None) -> dict[str, Any]:
    cache_file = _resolve_cache_file(cache_root, motion)
    metrics = _metrics_for_cache(cache_file, right_hand_site_index=right_hand_site_index)
    decision = classify_motion_stage(metrics, hints)
    return {
        "motion": motion,
        "cache_file": str(cache_file),
        "stage": decision.stage,
        "family": decision.family,
        "reasons": list(decision.reasons),
        "hints": asdict(hints),
        "metrics": metrics,
    }


def _write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path("caches/AMASS/MyoFullBody/gmr"))
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--hints", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--right-hand-site-index", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    hint_table = _load_hints(args.hints)

    try:
        rows: list[dict[str, Any]] = []
        for manifest in args.manifest:
            for motion in _read_manifest(manifest):
                row = _recommend_motion(
                    args.cache_root,
                    motion,
                    hint_table.for_motion(motion),
                    right_hand_site_index=args.right_hand_site_index,
                )
                rows.append(row)
                print(f"{row['motion']}: stage={row['stage']} family={row['family']} reasons={','.join(row['reasons'])}")
        _write_json(args.output, rows)
        print(f"wrote JSON report: {args.output}")
    except EXPECTED_USER_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI tests**

Run:

```bash
pytest tests/unit/test_recommend_action_stages.py -v
```

Expected: PASS.

- [ ] **Step 5: Run combined classifier and CLI tests**

Run:

```bash
pytest tests/unit/test_action_stage.py tests/unit/test_recommend_action_stages.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

Run:

```bash
git add BadmintonMimic/scripts/recommend_action_stages.py tests/unit/test_recommend_action_stages.py
git commit -m "feat: recommend badminton action training stages"
```

---

### Task 3: Generated Stage Manifests

**Files:**
- Create: `BadmintonMimic/scripts/build_stage_manifests.py`
- Create: `tests/unit/test_build_stage_manifests.py`

- [ ] **Step 1: Write failing tests for manifest generation**

Create `tests/unit/test_build_stage_manifests.py`:

```python
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
    assert (output_dir / "posttrain_net_frontcourt_list.txt").read_text(encoding="utf-8") == "ForehandNetLift/best/video01\n"
    assert (output_dir / "posttrain_smash_list.txt").read_text(encoding="utf-8") == "Smash/best/video01\n"
    assert (output_dir / "repair_list.txt").read_text(encoding="utf-8") == "Backhand/best/video07\n"
    assert (output_dir / "exclude_list.txt").read_text(encoding="utf-8") == "NetTumble/raw/video01\n"


def test_empty_groups_are_not_written(tmp_path):
    module = _load_module(SCRIPT, "build_stage_manifests_empty_for_test")
    report = tmp_path / "recommendations.json"
    output_dir = tmp_path / "generated"
    report.write_text('[{"motion": "ForehandClear/raw/video01", "stage": "base", "family": "general"}]', encoding="utf-8")

    code = module.main(["--recommendations", str(report), "--output-dir", str(output_dir)])

    assert code == 0
    assert (output_dir / "base_general_list.txt").exists()
    assert not (output_dir / "repair_list.txt").exists()
    assert not (output_dir / "exclude_list.txt").exists()
```

- [ ] **Step 2: Run the manifest-generation tests and verify they fail**

Run:

```bash
pytest tests/unit/test_build_stage_manifests.py -v
```

Expected: FAIL because `BadmintonMimic/scripts/build_stage_manifests.py` does not exist.

- [ ] **Step 3: Implement manifest generation**

Create `BadmintonMimic/scripts/build_stage_manifests.py`:

```python
#!/usr/bin/env python3
"""Build stage-specific motion manifests from action-stage recommendations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("recommendations JSON must contain a list")
    return data


def _manifest_name(stage: str, family: str) -> str:
    if stage == "base":
        return "base_general_list.txt"
    if stage == "posttrain":
        return f"posttrain_{family}_list.txt"
    if stage == "repair":
        return "repair_list.txt"
    if stage == "exclude":
        return "exclude_list.txt"
    raise ValueError(f"unsupported stage: {stage}")


def _group_rows(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        motion = str(row["motion"])
        stage = str(row["stage"])
        family = str(row.get("family", "general"))
        grouped[_manifest_name(stage, family)].append(motion)
    return dict(grouped)


def _write_manifests(output_dir: Path, grouped: dict[str, list[str]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, motions in sorted(grouped.items()):
        unique_sorted = sorted(dict.fromkeys(motions))
        content = "".join(f"{motion}\n" for motion in unique_sorted)
        (output_dir / name).write_text(content, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recommendations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    rows = _load_rows(args.recommendations)
    grouped = _group_rows(rows)
    _write_manifests(args.output_dir, grouped)
    for name, motions in sorted(grouped.items()):
        print(f"{name}: {len(set(motions))} motions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run manifest-generation tests**

Run:

```bash
pytest tests/unit/test_build_stage_manifests.py -v
```

Expected: PASS.

- [ ] **Step 5: Run all action-stage tests**

Run:

```bash
pytest tests/unit/test_action_stage.py tests/unit/test_recommend_action_stages.py tests/unit/test_build_stage_manifests.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

Run:

```bash
git add BadmintonMimic/scripts/build_stage_manifests.py tests/unit/test_build_stage_manifests.py
git commit -m "feat: build badminton stage manifests"
```

---

### Task 4: Default Hint File And Real-Data Smoke Run

**Files:**
- Create: `BadmintonMimic/manifests/action_stage_hints.yaml`
- No tests required beyond smoke commands because this is data/configuration, but the file must be used by Task 2 CLI.

- [ ] **Step 1: Create the default hint file**

Create `BadmintonMimic/manifests/action_stage_hints.yaml`:

```yaml
defaults:
  ForehandClear:
    expected_large_motion: false
  Backhand:
    expected_large_motion: false
  ForehandNetLift:
    expected_large_motion: true
  Smash:
    expected_large_motion: false

motions: {}
```

- [ ] **Step 2: Run recommendation CLI on current manifests**

Run:

```bash
python BadmintonMimic/scripts/recommend_action_stages.py \
  --cache-root caches/AMASS/MyoFullBody/gmr \
  --manifest BadmintonMimic/manifests/ForehandClear/raw_list.txt \
  --manifest BadmintonMimic/manifests/Backhand/best_list.txt \
  --manifest BadmintonMimic/manifests/ForehandNetLift/best_list.txt \
  --manifest BadmintonMimic/manifests/Smash/best_list.txt \
  --hints BadmintonMimic/manifests/action_stage_hints.yaml \
  --output outputs/action_stage/recommendations.json
```

Expected: command exits `0`, prints one line per motion, and writes `outputs/action_stage/recommendations.json`.

- [ ] **Step 3: Generate stage manifests**

Run:

```bash
python BadmintonMimic/scripts/build_stage_manifests.py \
  --recommendations outputs/action_stage/recommendations.json \
  --output-dir BadmintonMimic/manifests/generated
```

Expected: command exits `0` and writes some subset of:

```text
BadmintonMimic/manifests/generated/base_general_list.txt
BadmintonMimic/manifests/generated/posttrain_general_list.txt
BadmintonMimic/manifests/generated/posttrain_net_frontcourt_list.txt
BadmintonMimic/manifests/generated/posttrain_rotation_list.txt
BadmintonMimic/manifests/generated/posttrain_smash_list.txt
BadmintonMimic/manifests/generated/repair_list.txt
BadmintonMimic/manifests/generated/exclude_list.txt
```

- [ ] **Step 4: Inspect generated counts**

Run:

```bash
for f in BadmintonMimic/manifests/generated/*.txt; do printf "%s " "$f"; wc -l < "$f"; done
```

Expected: each generated file prints a non-negative line count. A zero-line file should not exist.

- [ ] **Step 5: Commit Task 4**

Run:

```bash
git add BadmintonMimic/manifests/action_stage_hints.yaml BadmintonMimic/manifests/generated
git commit -m "data: add badminton action stage manifests"
```

---

### Task 5: Documentation Update

**Files:**
- Modify: `doc/PostTrain_Advice.md`

- [ ] **Step 1: Add the action-stage policy section**

Append this section to `doc/PostTrain_Advice.md`:

```markdown

---

# 动作分层：哪些动作适合初训，哪些适合后训练

第一版不要只按动作名字决定训练阶段，而要按“SMPL 是否可观测 + root/步法/接触复杂度”来分。

适合放进初训/base training 的动作：

- 正手高远球、ForehandClear。
- Root 位移中等、脚步稳定的 Backhand。
- 原地或小步吊球。
- 不含明显起跳和重落地的站立杀球。
- 基础侧移、启动、恢复、split-step、小后撤，前提是 retarget 后脚步和 root 稳定。

适合用已有 checkpoint 后训练/post-train 的动作：

- ForehandNetLift、上网挑球、网前上步。
- 弓步、跨步、并步、急停、恢复步法。
- 后场后撤接吊、后撤高远、后撤后恢复。
- 起跳杀球、重落地杀球。
- 大幅转体的 backhand 或 smash。

暂时不作为主目标的动作：

- 很细小的搓球。
- 主要由手腕、手指、拍面角度决定的网前小技术。
- 身体几乎不动、SMPL 无法观测关键差异的假动作。

推荐先运行：

```bash
python BadmintonMimic/scripts/recommend_action_stages.py \
  --cache-root caches/AMASS/MyoFullBody/gmr \
  --manifest BadmintonMimic/manifests/ForehandClear/raw_list.txt \
  --manifest BadmintonMimic/manifests/Backhand/best_list.txt \
  --manifest BadmintonMimic/manifests/ForehandNetLift/best_list.txt \
  --manifest BadmintonMimic/manifests/Smash/best_list.txt \
  --hints BadmintonMimic/manifests/action_stage_hints.yaml \
  --output outputs/action_stage/recommendations.json
```

再生成训练阶段 manifest：

```bash
python BadmintonMimic/scripts/build_stage_manifests.py \
  --recommendations outputs/action_stage/recommendations.json \
  --output-dir BadmintonMimic/manifests/generated
```

解释标准：

- `base`：适合初训，用来学习通用肌骨控制和基础羽毛球身体模式。
- `posttrain`：适合从已有 checkpoint 微调，通常需要更强 root、右手/球拍末端、足底接触和自然性约束。
- `repair`：动作有价值，但 reference root、脚步、接触或手部末端不可信，应先修数据。
- `exclude`：当前 SMPL 表达不了关键技术细节，不适合作为主要训练目标。
```

- [ ] **Step 2: Run documentation grep to verify commands exist**

Run:

```bash
rg -n "recommend_action_stages|build_stage_manifests|base|posttrain|repair|exclude" doc/PostTrain_Advice.md
```

Expected: output includes the new section and both command names.

- [ ] **Step 3: Commit Task 5**

Run:

```bash
git add doc/PostTrain_Advice.md
git commit -m "docs: describe badminton action stage policy"
```

---

### Task 6: Final Verification

**Files:**
- No new files.

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
pytest tests/unit/test_action_stage.py tests/unit/test_recommend_action_stages.py tests/unit/test_build_stage_manifests.py tests/unit/test_root_tracking_metrics.py tests/unit/test_diagnose_root_tracking.py -v
```

Expected: PASS.

- [ ] **Step 2: Run a no-write CLI parse check**

Run:

```bash
python BadmintonMimic/scripts/recommend_action_stages.py --help
python BadmintonMimic/scripts/build_stage_manifests.py --help
```

Expected: both commands print argparse help and exit `0`.

- [ ] **Step 3: Inspect git status**

Run:

```bash
git status --short
```

Expected: only pre-existing unrelated untracked files remain, or a clean worktree if those files were separately handled.

- [ ] **Step 4: Summarize generated stage counts**

Run:

```bash
for f in BadmintonMimic/manifests/generated/*.txt; do printf "%s " "$f"; wc -l < "$f"; done
```

Expected: line counts give the user a concrete view of how many motions landed in each stage.

---

## Self-Review

- Spec coverage: The plan implements metric-driven action-stage assignment, distinguishes base/posttrain/repair/exclude, avoids fine hand-only claims, and uses existing root/right-hand metrics. It does not implement training configs because the spec framed those as optional later work after recommendations are reviewed.
- Placeholder scan: No placeholder markers or unspecified implementation steps remain.
- Type consistency: The classifier uses `MotionHints`, `StageDecision`, `classify_motion_stage`, and the CLIs consume those exact names.

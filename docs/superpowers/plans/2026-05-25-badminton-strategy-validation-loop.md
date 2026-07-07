# Badminton Strategy Validation Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add validation infrastructure that turns badminton action staging from a reproducible heuristic into an auditable method with confidence labels, failure modes, summary reports, and claim-evidence templates.

**Architecture:** Keep the existing pure-classifier plus script pipeline. `musclemimic/utils/action_stage.py` owns stage, confidence, failure-mode, and required-action decisions; `musclemimic/utils/root_tracking.py` owns numeric diagnostics; `musclemimic/badminton/scripts/recommend_action_stages.py` emits per-motion validation rows and summaries; `musclemimic/badminton/scripts/build_stage_manifests.py` remains backwards compatible and only consumes stage/family/motion fields. A small claim-template script creates the paper-level evidence checklist without launching PPO jobs.

**Tech Stack:** Python dataclasses, NumPy, argparse, JSON, PyYAML, pytest, existing badminton manifests and cache format.

---

## File Structure

- Modify `musclemimic/utils/action_stage.py`: enrich `StageDecision` with `confidence`, `failure_modes`, `review_required`, and `required_action`; add threshold confidence bands while preserving current `stage`, `family`, and `reasons`.
- Modify `musclemimic/utils/root_tracking.py`: add reference-quality diagnostics such as frame count, path/displacement ratio, qpos/qvel max step, and site discontinuity proxy.
- Modify `musclemimic/badminton/scripts/recommend_action_stages.py`: include new validation fields in each recommendation row and optionally write a summary JSON report.
- Modify `musclemimic/badminton/scripts/build_stage_manifests.py`: explicitly tolerate the enriched recommendation schema and reject invalid new field types when present.
- Create `musclemimic/badminton/scripts/build_claim_evidence_template.py`: generate a static claim-to-evidence JSON template for baseline and ablation tracking.
- Modify `doc/PostTrain_Advice.md`: document the validation loop usage and how to interpret confidence/failure fields.
- Modify tests:
  - `tests/unit/test_action_stage.py`
  - `tests/unit/test_root_tracking_metrics.py`
  - `tests/unit/test_recommend_action_stages.py`
  - `tests/unit/test_build_stage_manifests.py`
  - Create `tests/unit/test_build_claim_evidence_template.py`

This plan does not modify PPO reward code or Hydra training configs. It creates the validation layer that decides which experiments are worth running and how results should support claims.

---

### Task 1: Enrich Stage Decisions With Confidence and Required Actions

**Files:**
- Modify: `musclemimic/utils/action_stage.py`
- Modify: `tests/unit/test_action_stage.py`

- [ ] **Step 1: Write failing tests for enriched decision fields**

Append these tests to `tests/unit/test_action_stage.py`:

```python
def test_borderline_peak_speed_marks_medium_confidence_and_review():
    metrics = {
        "reference_root_xy_total_displacement": 0.42,
        "reference_root_xy_peak_speed": 1.16,
        "reference_root_yaw_change": 0.20,
        "right_hand_world_path_length": 0.90,
    }

    result = classify_motion_stage(metrics, MotionHints(action_label="ForehandClear"))

    assert result.stage == "base"
    assert result.confidence == "medium"
    assert result.review_required is True
    assert result.required_action == "train"
    assert "borderline_root_peak_speed" in result.failure_modes


def test_repair_decision_is_low_confidence_and_repair_first():
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
    assert result.confidence == "low"
    assert result.review_required is True
    assert result.required_action == "repair_first"
    assert "expected_large_motion_but_root_is_small" in result.failure_modes


def test_high_confidence_posttrain_has_posttrain_required_action():
    metrics = {
        "reference_root_xy_total_displacement": 0.85,
        "reference_root_xy_peak_speed": 1.55,
        "reference_root_yaw_change": 0.10,
        "right_hand_world_path_length": 1.10,
    }

    result = classify_motion_stage(metrics, MotionHints(action_label="ForehandNetLift"))

    assert result.stage == "posttrain"
    assert result.confidence == "high"
    assert result.review_required is False
    assert result.required_action == "posttrain"
    assert result.failure_modes == ()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/unit/test_action_stage.py -v
```

Expected: existing tests pass and the three new tests fail with `AttributeError` for missing `confidence`, `review_required`, `required_action`, or `failure_modes`.

- [ ] **Step 3: Implement enriched `StageDecision`**

In `musclemimic/utils/action_stage.py`, replace the existing `StageDecision` dataclass and add constants/helpers above `classify_motion_stage`:

```python
CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"

REQUIRED_TRAIN = "train"
REQUIRED_POSTTRAIN = "posttrain"
REQUIRED_REPAIR_FIRST = "repair_first"
REQUIRED_EXCLUDE = "exclude"
REQUIRED_MANUAL_REVIEW = "manual_review"

ROOT_SMALL_DISPLACEMENT = 0.25
ROOT_LARGE_DISPLACEMENT = 0.60
ROOT_HIGH_PEAK_SPEED = 1.20
ROOT_LARGE_YAW_CHANGE = 0.80

ROOT_DISPLACEMENT_MARGIN = 0.05
ROOT_PEAK_SPEED_MARGIN = 0.10
ROOT_YAW_MARGIN = 0.05


@dataclass(frozen=True)
class StageDecision:
    stage: str
    family: str
    reasons: tuple[str, ...]
    confidence: str = CONFIDENCE_HIGH
    failure_modes: tuple[str, ...] = ()
    review_required: bool = False
    required_action: str = REQUIRED_TRAIN


def _near(value: float, threshold: float, margin: float) -> bool:
    return abs(value - threshold) <= margin


def _borderline_failure_modes(root_disp: float, root_peak_speed: float, yaw_change: float) -> list[str]:
    failures: list[str] = []
    if _near(root_disp, ROOT_SMALL_DISPLACEMENT, ROOT_DISPLACEMENT_MARGIN):
        failures.append("borderline_small_root_displacement")
    if _near(root_disp, ROOT_LARGE_DISPLACEMENT, ROOT_DISPLACEMENT_MARGIN):
        failures.append("borderline_large_root_displacement")
    if _near(root_peak_speed, ROOT_HIGH_PEAK_SPEED, ROOT_PEAK_SPEED_MARGIN):
        failures.append("borderline_root_peak_speed")
    if _near(yaw_change, ROOT_LARGE_YAW_CHANGE, ROOT_YAW_MARGIN):
        failures.append("borderline_root_yaw_change")
    return failures


def _confidence_from_failures(failure_modes: list[str]) -> tuple[str, bool]:
    if any(not failure.startswith("borderline_") for failure in failure_modes):
        return CONFIDENCE_LOW, True
    if failure_modes:
        return CONFIDENCE_MEDIUM, True
    return CONFIDENCE_HIGH, False
```

Then update `classify_motion_stage` so every return constructs `StageDecision` with the new fields:

```python
    borderline_failures = _borderline_failure_modes(root_disp, root_peak_speed, yaw_change)

    if hints.fine_hand_dominant:
        return StageDecision(
            stage=STAGE_EXCLUDE,
            family="fine_hand",
            reasons=("fine_hand_dominant",),
            confidence=CONFIDENCE_LOW,
            failure_modes=("fine_hand_dominant",),
            review_required=True,
            required_action=REQUIRED_EXCLUDE,
        )
```

For repair returns, use:

```python
        failure_modes = list(dict.fromkeys(reasons + borderline_failures))
        confidence, review_required = _confidence_from_failures(failure_modes)
        return StageDecision(
            stage=STAGE_REPAIR,
            family=label_family,
            reasons=tuple(reasons),
            confidence=confidence,
            failure_modes=tuple(failure_modes),
            review_required=review_required,
            required_action=REQUIRED_REPAIR_FIRST,
        )
```

For posttrain returns, use:

```python
        confidence, review_required = _confidence_from_failures(borderline_failures)
        return StageDecision(
            stage=STAGE_POSTTRAIN,
            family=_posttrain_family(label_family, reasons),
            reasons=tuple(reasons),
            confidence=confidence,
            failure_modes=tuple(borderline_failures),
            review_required=review_required,
            required_action=REQUIRED_POSTTRAIN,
        )
```

For base returns, use:

```python
    confidence, review_required = _confidence_from_failures(borderline_failures)
    return StageDecision(
        stage=STAGE_BASE,
        family="general" if label_family == "fine_hand" else label_family,
        reasons=tuple(reasons),
        confidence=confidence,
        failure_modes=tuple(borderline_failures),
        review_required=review_required,
        required_action=REQUIRED_TRAIN,
    )
```

Also replace raw threshold literals inside `classify_motion_stage` with the named constants.

- [ ] **Step 4: Run focused tests**

Run:

```bash
pytest tests/unit/test_action_stage.py -v
```

Expected: all tests in `tests/unit/test_action_stage.py` pass.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add musclemimic/utils/action_stage.py tests/unit/test_action_stage.py
git commit -m "feat: add action stage confidence fields"
```

---

### Task 2: Add Reference Data Quality Diagnostics

**Files:**
- Modify: `musclemimic/utils/root_tracking.py`
- Modify: `tests/unit/test_root_tracking_metrics.py`

- [ ] **Step 1: Write failing tests for quality metrics**

Append these tests to `tests/unit/test_root_tracking_metrics.py`:

```python
def test_compute_root_reference_metrics_reports_quality_diagnostics():
    qpos = np.array(
        [
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.4, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    qvel = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    site_xpos = np.zeros((3, 2, 3), dtype=np.float32)
    site_xpos[1, 1, 0] = 0.2
    site_xpos[2, 1, 0] = 1.0

    metrics = compute_root_reference_metrics(
        qpos=qpos,
        qvel=qvel,
        site_xpos=site_xpos,
        right_hand_site_index=1,
        frequency=50.0,
    )

    assert metrics["reference_frame_count"] == 3
    assert metrics["reference_frequency"] == pytest.approx(50.0)
    assert metrics["reference_root_xy_path_displacement_ratio"] == pytest.approx(1.0)
    assert metrics["reference_qpos_max_abs_step"] == pytest.approx(0.3)
    assert metrics["reference_qvel_max_abs"] == pytest.approx(2.0)
    assert metrics["reference_site_xpos_max_step"] == pytest.approx(0.8)


def test_path_displacement_ratio_is_infinite_for_looping_motion():
    qpos = np.array(
        [
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    metrics = compute_root_reference_metrics(qpos=qpos)

    assert metrics["reference_root_xy_total_displacement"] == pytest.approx(0.0)
    assert metrics["reference_root_xy_path_length"] == pytest.approx(2.0)
    assert metrics["reference_root_xy_path_displacement_ratio"] == float("inf")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/unit/test_root_tracking_metrics.py -v
```

Expected: new tests fail with missing metric keys.

- [ ] **Step 3: Implement quality metrics**

In `musclemimic/utils/root_tracking.py`, update `compute_root_reference_metrics`:

```python
    total_displacement = _total_displacement(root_xy)
    path_length = _path_length(root_xy)

    metrics = {
        "reference_root_xy_total_displacement": total_displacement,
        "reference_root_xy_path_length": path_length,
        "reference_root_xy_peak_speed": float(np.max(speed)) if speed.size else 0.0,
        "reference_root_yaw_change": _yaw_change(yaw),
        "right_hand_world_path_length": right_hand_path_length,
        "reference_frame_count": float(qpos_array.shape[0]),
        "reference_frequency": float(frequency) if frequency is not None else 0.0,
        "reference_root_xy_path_displacement_ratio": _safe_ratio(path_length, total_displacement),
        "reference_qpos_max_abs_step": _max_abs_step(qpos_array),
        "reference_qvel_max_abs": _max_abs_value(qvel),
        "reference_site_xpos_max_step": _site_xpos_max_step(site_xpos),
    }
    return metrics
```

Add helpers near the other private helpers:

```python
def _max_abs_step(values: np.ndarray) -> float:
    if values.shape[0] < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(values, axis=0))))


def _max_abs_value(values) -> float:
    if values is None:
        return 0.0
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return 0.0
    return float(np.max(np.abs(array)))


def _site_xpos_max_step(site_xpos) -> float:
    if site_xpos is None:
        return 0.0
    site_xpos_array = np.asarray(site_xpos, dtype=np.float64)
    if site_xpos_array.ndim != 3:
        raise ValueError("site_xpos must be a 3D array of shape (frames, sites, xyz)")
    if site_xpos_array.shape[2] < 3:
        raise ValueError("site_xpos must contain xyz coordinates")
    if site_xpos_array.shape[0] < 2:
        return 0.0
    frame_steps = np.linalg.norm(np.diff(site_xpos_array[:, :, :3], axis=0), axis=2)
    return float(np.max(frame_steps))
```

Keep existing metric keys unchanged so current scripts and tests remain compatible.

- [ ] **Step 4: Run focused tests**

Run:

```bash
pytest tests/unit/test_root_tracking_metrics.py tests/unit/test_diagnose_root_tracking.py -v
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add musclemimic/utils/root_tracking.py tests/unit/test_root_tracking_metrics.py
git commit -m "feat: add reference quality diagnostics"
```

---

### Task 3: Emit Validation Fields and Summary Reports

**Files:**
- Modify: `musclemimic/badminton/scripts/recommend_action_stages.py`
- Modify: `tests/unit/test_recommend_action_stages.py`

- [ ] **Step 1: Write failing tests for enriched recommendation rows**

Update `test_main_writes_recommendation_json` in `tests/unit/test_recommend_action_stages.py` so `expected_keys` is:

```python
    expected_keys = {
        "motion",
        "cache_file",
        "stage",
        "family",
        "reasons",
        "confidence",
        "failure_modes",
        "review_required",
        "required_action",
        "hints",
        "metrics",
    }
```

Add these assertions inside that test after loading `rows`:

```python
    assert rows[0]["confidence"] == "high"
    assert rows[0]["failure_modes"] == []
    assert rows[0]["review_required"] is False
    assert rows[0]["required_action"] == "posttrain"
    assert rows[1]["required_action"] == "train"
```

Add a new test:

```python
def test_main_writes_summary_json(tmp_path):
    module = _load_module(SCRIPT, "recommend_action_stages_summary_for_test")
    cache_root = tmp_path / "cache"
    manifest = tmp_path / "manifest.txt"
    output = tmp_path / "recommendations.json"
    summary = tmp_path / "summary.json"
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
            "--summary-output",
            str(summary),
        ]
    )

    assert code == 0
    data = json.loads(summary.read_text(encoding="utf-8"))
    assert data["total_motions"] == 2
    assert data["stage_counts"] == {"base": 1, "posttrain": 1}
    assert data["confidence_counts"] == {"high": 2}
    assert data["review_required_count"] == 0
    assert data["required_action_counts"] == {"posttrain": 1, "train": 1}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/unit/test_recommend_action_stages.py -v
```

Expected: tests fail because row fields and `--summary-output` do not exist yet.

- [ ] **Step 3: Add validation fields to recommendation rows**

In `_recommend_motion`, extend the returned dictionary:

```python
    return {
        "motion": motion,
        "cache_file": str(cache_file),
        "stage": decision.stage,
        "family": decision.family,
        "reasons": list(decision.reasons),
        "confidence": decision.confidence,
        "failure_modes": list(decision.failure_modes),
        "review_required": decision.review_required,
        "required_action": decision.required_action,
        "hints": asdict(hints),
        "metrics": metrics,
    }
```

- [ ] **Step 4: Add summary generation**

Add imports:

```python
from collections import Counter
```

Add this helper above `_write_json_report`:

```python
def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stage_counts = Counter(row["stage"] for row in rows)
    family_counts = Counter(row["family"] for row in rows)
    confidence_counts = Counter(row["confidence"] for row in rows)
    required_action_counts = Counter(row["required_action"] for row in rows)
    failure_mode_counts = Counter(
        failure_mode
        for row in rows
        for failure_mode in row.get("failure_modes", [])
    )
    return {
        "total_motions": len(rows),
        "stage_counts": dict(sorted(stage_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "required_action_counts": dict(sorted(required_action_counts.items())),
        "failure_mode_counts": dict(sorted(failure_mode_counts.items())),
        "review_required_count": sum(1 for row in rows if row.get("review_required") is True),
    }
```

Add the parser argument:

```python
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optional JSON path for aggregate validation counts.",
    )
```

After `_write_json_report(args.output, rows)` in `main`, add:

```python
        if args.summary_output is not None:
            _write_json_report(args.summary_output, _summarize_rows(rows))
            print(f"wrote summary JSON report: {args.summary_output}")
```

Change `_write_json_report` type to accept dictionaries and lists:

```python
def _write_json_report(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
```

- [ ] **Step 5: Update CLI summary text**

Update `_format_summary`:

```python
def _format_summary(row: dict[str, Any]) -> str:
    reasons = ",".join(row["reasons"])
    failures = ",".join(row["failure_modes"])
    return (
        f"{row['motion']}: "
        f"stage={row['stage']} "
        f"family={row['family']} "
        f"confidence={row['confidence']} "
        f"required_action={row['required_action']} "
        f"review_required={row['review_required']} "
        f"reasons={reasons} "
        f"failure_modes={failures}"
    )
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
pytest tests/unit/test_action_stage.py tests/unit/test_recommend_action_stages.py -v
```

Expected: selected tests pass.

- [ ] **Step 7: Commit Task 3**

Run:

```bash
git add musclemimic/badminton/scripts/recommend_action_stages.py tests/unit/test_recommend_action_stages.py
git commit -m "feat: emit action stage validation reports"
```

---

### Task 4: Validate Enriched Rows in Manifest Builder

**Files:**
- Modify: `musclemimic/badminton/scripts/build_stage_manifests.py`
- Modify: `tests/unit/test_build_stage_manifests.py`

- [ ] **Step 1: Write failing tests for enriched schema validation**

Append these tests to `tests/unit/test_build_stage_manifests.py`:

```python
def test_enriched_recommendation_rows_are_accepted(tmp_path):
    module = _load_module(SCRIPT, "build_stage_manifests_enriched_for_test")
    report = tmp_path / "recommendations.json"
    output_dir = tmp_path / "generated"
    report.write_text(
        """
[
  {
    "motion": "ForehandNetLift/best/video01",
    "stage": "posttrain",
    "family": "net_frontcourt",
    "confidence": "medium",
    "failure_modes": ["borderline_root_peak_speed"],
    "review_required": true,
    "required_action": "posttrain"
  }
]
""",
        encoding="utf-8",
    )

    code = module.main(["--recommendations", str(report), "--output-dir", str(output_dir)])

    assert code == 0
    assert (
        output_dir / "posttrain_net_frontcourt_list.txt"
    ).read_text(encoding="utf-8") == "ForehandNetLift/best/video01\n"


def test_invalid_enriched_review_field_returns_user_facing_error(tmp_path, capsys):
    module = _load_module(SCRIPT, "build_stage_manifests_bad_review_for_test")
    report = tmp_path / "recommendations.json"
    output_dir = tmp_path / "generated"
    report.write_text(
        """
[
  {
    "motion": "ForehandNetLift/best/video01",
    "stage": "posttrain",
    "family": "net_frontcourt",
    "review_required": "yes"
  }
]
""",
        encoding="utf-8",
    )

    code = module.main(["--recommendations", str(report), "--output-dir", str(output_dir)])

    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.startswith("error:")
    assert "review_required" in captured.err
    assert not output_dir.exists()
```

- [ ] **Step 2: Run tests to verify the invalid-field test fails**

Run:

```bash
pytest tests/unit/test_build_stage_manifests.py -v
```

Expected: enriched row acceptance already passes, and invalid `review_required` fails because the script currently ignores that field.

- [ ] **Step 3: Add optional enriched-field validation**

In `musclemimic/badminton/scripts/build_stage_manifests.py`, add constants near `GENERATED_MANIFEST_NAMES`:

```python
VALID_CONFIDENCE_VALUES = frozenset({"high", "medium", "low"})
VALID_REQUIRED_ACTION_VALUES = frozenset({"train", "posttrain", "repair_first", "exclude", "manual_review"})
```

Add this helper:

```python
def _validate_optional_validation_fields(row: Mapping[str, Any], row_index: int) -> None:
    if "confidence" in row:
        confidence = row["confidence"]
        if confidence not in VALID_CONFIDENCE_VALUES:
            raise ValueError(f"row {row_index} field 'confidence' must be high, medium, or low")
    if "required_action" in row:
        required_action = row["required_action"]
        if required_action not in VALID_REQUIRED_ACTION_VALUES:
            raise ValueError(f"row {row_index} field 'required_action' is invalid")
    if "review_required" in row and type(row["review_required"]) is not bool:
        raise ValueError(f"row {row_index} field 'review_required' must be a boolean")
    if "failure_modes" in row:
        failure_modes = row["failure_modes"]
        if not isinstance(failure_modes, list) or not all(isinstance(item, str) for item in failure_modes):
            raise ValueError(f"row {row_index} field 'failure_modes' must be a list of strings")
```

Call it inside `_group_rows` immediately after the `Mapping` check:

```python
        _validate_optional_validation_fields(row, row_index)
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
pytest tests/unit/test_build_stage_manifests.py -v
```

Expected: all manifest-builder tests pass.

- [ ] **Step 5: Commit Task 4**

Run:

```bash
git add musclemimic/badminton/scripts/build_stage_manifests.py tests/unit/test_build_stage_manifests.py
git commit -m "fix: validate action stage report fields"
```

---

### Task 5: Add Claim-to-Evidence Template Generator

**Files:**
- Create: `musclemimic/badminton/scripts/build_claim_evidence_template.py`
- Create: `tests/unit/test_build_claim_evidence_template.py`

- [ ] **Step 1: Write failing tests for the template script**

Create `tests/unit/test_build_claim_evidence_template.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/unit/test_build_claim_evidence_template.py -v
```

Expected: tests fail because `musclemimic/badminton/scripts/build_claim_evidence_template.py` does not exist.

- [ ] **Step 3: Create the template script**

Create `musclemimic/badminton/scripts/build_claim_evidence_template.py`:

```python
#!/usr/bin/env python3
"""Write a claim-to-evidence template for badminton validation experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


EXPECTED_USER_ERRORS = (OSError, ValueError)


def _template() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "required_experiments": [
            "all_mix",
            "action_name_grouping",
            "metric_gated_staging",
        ],
        "required_ablations": [
            "no_repair_gate",
            "no_rotation_speed_gate",
            "no_posttrain_root_focus",
        ],
        "metrics": [
            "root_xy_rmse",
            "root_displacement_ratio",
            "root_speed_rmse",
            "heading_yaw_error",
            "right_hand_position_error",
            "relative_body_pose_error",
            "early_termination_rate",
            "foot_slip_proxy",
            "control_action_rate_cost",
        ],
        "claims": [
            {
                "id": "staging_improves_stability",
                "claim": "Staging improves training stability.",
                "required_evidence": [
                    "metric_gated_staging has lower early termination or fewer catastrophic failures than all_mix",
                    "metric_gated_staging does not regress aggregate tracking metrics relative to action_name_grouping",
                ],
                "status": "not_evaluated",
                "decision_rule": "support only if stability improves without a severe tracking regression",
            },
            {
                "id": "posttrain_helps_large_motion",
                "claim": "Posttraining helps movement-heavy badminton actions.",
                "required_evidence": [
                    "posttrain actions improve root displacement ratio or root_xy_rmse",
                    "improvement appears on held-out or repeated movement-heavy clips",
                ],
                "status": "not_evaluated",
                "decision_rule": "downgrade to action-specific fine-tuning if only one clip improves",
            },
            {
                "id": "repair_gate_protects_training",
                "claim": "Repair/exclusion prevents corrupted references from hurting training.",
                "required_evidence": [
                    "no_repair_gate performs worse on reliability or tracking metrics",
                    "flagged repair clips show concrete data-quality failure modes",
                ],
                "status": "not_evaluated",
                "decision_rule": "treat as data hygiene if performance evidence is neutral",
            },
            {
                "id": "metric_gated_beats_action_name_grouping",
                "claim": "Metric-gated staging is better than action-name grouping.",
                "required_evidence": [
                    "metric_gated_staging beats action_name_grouping on at least one aggregate metric",
                    "metric_gated_staging does not create a severe family-level regression",
                ],
                "status": "not_evaluated",
                "decision_rule": "reframe as diagnostics if action-name grouping is equal or better",
            },
        ],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        _write_json(args.output, _template())
    except EXPECTED_USER_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote claim evidence template: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
pytest tests/unit/test_build_claim_evidence_template.py -v
```

Expected: all claim-template tests pass.

- [ ] **Step 5: Commit Task 5**

Run:

```bash
git add musclemimic/badminton/scripts/build_claim_evidence_template.py tests/unit/test_build_claim_evidence_template.py
git commit -m "feat: add badminton claim evidence template"
```

---

### Task 6: Document the Validation Workflow

**Files:**
- Modify: `doc/PostTrain_Advice.md`

- [ ] **Step 1: Update the documented command sequence**

In `doc/PostTrain_Advice.md`, find the existing action-stage command block around the recommendation JSON path. Extend the command so it includes summary output:

```bash
.venv/bin/python musclemimic/badminton/scripts/recommend_action_stages.py \
  --cache-root caches/AMASS/MyoFullBody/gmr \
  --manifest manifests/source/badminton_best_list.txt \
  --manifest manifests/source/badminton_raw_list.txt \
  --hints manifests/action_stage_hints.yaml \
  --output outputs/action_stage/recommendations.json \
  --summary-output outputs/action_stage/summary.json
```

- [ ] **Step 2: Add the claim-template command**

Add this command after the manifest build command:

```bash
.venv/bin/python musclemimic/badminton/scripts/build_claim_evidence_template.py \
  --output outputs/action_stage/claim_evidence_template.json
```

- [ ] **Step 3: Add interpretation text**

Add this paragraph near the existing explanation of `outputs/action_stage/recommendations.json`:

```markdown
The enriched recommendation report now includes `confidence`, `failure_modes`,
`review_required`, and `required_action`. Treat `confidence=high` as safe for the
current automated stage rule, `confidence=medium` as a threshold-borderline clip
that should not drive a strong paper claim by itself, and `confidence=low` as a
repair/manual-review/exclusion candidate. The summary report is the quick audit
surface: if `review_required_count` is high, inspect those motions before using
the manifests for training.
```

Add this paragraph near the paper-method discussion:

```markdown
For paper claims, use `outputs/action_stage/claim_evidence_template.json` as the
contract between training runs and claims. Do not claim that staged training is
better than simpler alternatives until `all_mix`, `action_name_grouping`,
`metric_gated_staging`, and the listed ablations have metrics attached. If an
ablation contradicts the hypothesis, weaken the claim instead of tuning only the
thresholds.
```

- [ ] **Step 4: Verify documentation references**

Run:

```bash
rg -n "summary-output|confidence|failure_modes|claim_evidence_template|build_claim_evidence_template" doc/PostTrain_Advice.md
```

Expected: output shows the new command flag, the new fields, and the claim-template script reference.

- [ ] **Step 5: Commit Task 6**

Run:

```bash
git add doc/PostTrain_Advice.md
git commit -m "docs: document badminton validation reports"
```

---

### Task 7: End-to-End Verification

**Files:**
- No planned source edits.
- Possible generated outputs in `/tmp` only.

- [ ] **Step 1: Run the full focused test suite**

Run:

```bash
pytest tests/unit/test_action_stage.py tests/unit/test_recommend_action_stages.py tests/unit/test_build_stage_manifests.py tests/unit/test_root_tracking_metrics.py tests/unit/test_diagnose_root_tracking.py tests/unit/test_build_claim_evidence_template.py -v
```

Expected: all selected tests pass.

- [ ] **Step 2: Check CLI help**

Run:

```bash
python musclemimic/badminton/scripts/recommend_action_stages.py --help
python musclemimic/badminton/scripts/build_stage_manifests.py --help
python musclemimic/badminton/scripts/build_claim_evidence_template.py --help
```

Expected: each command exits `0`; `recommend_action_stages.py --help` includes `--summary-output`.

- [ ] **Step 3: Generate reports into `/tmp`**

Run:

```bash
python musclemimic/badminton/scripts/recommend_action_stages.py \
  --cache-root caches/AMASS/MyoFullBody/gmr \
  --manifest manifests/source/badminton_best_list.txt \
  --manifest manifests/source/badminton_raw_list.txt \
  --hints manifests/action_stage_hints.yaml \
  --output /tmp/badminton_action_stage_recommendations.json \
  --summary-output /tmp/badminton_action_stage_summary.json
```

Expected: command exits `0`, prints one line per motion, writes both JSON files under `/tmp`, and includes validation fields in the recommendation rows.

- [ ] **Step 4: Regenerate manifests into `/tmp`**

Run:

```bash
python musclemimic/badminton/scripts/build_stage_manifests.py \
  --recommendations /tmp/badminton_action_stage_recommendations.json \
  --output-dir /tmp/badminton_action_stage_generated
```

Expected: command exits `0` and prints counts for `base_general_list.txt`, one or more `posttrain_*_list.txt` files, and any repair/exclude lists present in the data.

- [ ] **Step 5: Generate claim evidence template into `/tmp`**

Run:

```bash
python musclemimic/badminton/scripts/build_claim_evidence_template.py \
  --output /tmp/badminton_claim_evidence_template.json
```

Expected: command exits `0` and writes a deterministic JSON template with four claims.

- [ ] **Step 6: Confirm committed generated manifests are unchanged unless intentionally updated**

Run:

```bash
diff -ru manifests/generated /tmp/badminton_action_stage_generated
```

Expected: either no diff, or a diff caused only by intentionally changed classification behavior from confidence-band work. If classifications changed, inspect the changed motion IDs and decide whether to commit updated manifests in a separate data commit.

- [ ] **Step 7: Check git status**

Run:

```bash
git status --short
```

Expected: only the unrelated pre-existing untracked files remain unless generated manifests were intentionally updated.

- [ ] **Step 8: Commit updated generated manifests only if classification output changed**

If Step 6 produced an intentional manifest diff, run:

```bash
cp -r /tmp/badminton_action_stage_generated/. manifests/generated/
git add manifests/generated
git commit -m "data: refresh badminton validation manifests"
```

Expected: commit contains only generated manifest changes.

---

## Self-Review Notes

- Spec coverage: Tasks 1-3 implement confidence, failure modes, required actions, diagnostics, and validation summaries. Task 4 preserves manifest compatibility. Task 5 implements the paper claim gate template. Task 6 documents usage and limitations. Task 7 verifies tests, CLI behavior, reproducibility, and generated output handling.
- Scope check: This plan is one implementation unit because it only adds validation infrastructure around the existing action-stage pipeline. It does not include large PPO training sweeps or reward redesign.
- Type consistency: `confidence`, `failure_modes`, `review_required`, and `required_action` are introduced in `StageDecision`, serialized by `recommend_action_stages.py`, and optionally validated by `build_stage_manifests.py`.

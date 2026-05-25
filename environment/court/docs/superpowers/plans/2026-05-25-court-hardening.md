# Court Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for completed tracking.

**Completion note:** This plan is retained as a historical implementation artifact. The hardening pass was completed in commit `4107918695c8ebc0bc98413ba7a7a3cf6bc41000`; validation and the full court test suite passed at completion.

**Goal:** Make `environment/court` integration-ready by cleaning repository hygiene, adding generated-asset drift checks, adding focused geometry/XML tests, and adding optional MuJoCo compile validation.

**Architecture:** Keep the existing `params -> court_geometry -> generate_court_mjcf -> assets` pipeline. Add tests around the current public behavior instead of restructuring the package. Keep MuJoCo runtime validation optional so the court package remains testable in lightweight environments.

**Tech Stack:** Python 3.11+, pytest, stdlib `xml.etree.ElementTree`, optional `mujoco`, existing MuJoCo XML assets.

---

## File Structure

- Modify: `.gitignore`
  - Responsibility: ignore local visual companion state and Python cache files.
- Delete from working tree only: `environment/court/src/__pycache__/court_geometry.cpython-313.pyc`
  - Responsibility: remove generated cache artifact from the package directory.
- Create: `environment/court/tests/conftest.py`
  - Responsibility: expose `environment/court/src` imports to court tests.
- Create: `environment/court/tests/test_court_geometry.py`
  - Responsibility: test official dimensions, line inclusion semantics, service bounds, line rectangle placement, and net height profile.
- Create: `environment/court/tests/test_court_xml.py`
  - Responsibility: test XML key elements and visual/collision separation.
- Create: `environment/court/tests/test_generated_assets.py`
  - Responsibility: ensure committed XML assets match fresh generator output.
- Modify: `environment/court/src/validate_court_params.py`
  - Responsibility: add optional real MuJoCo compile validation while preserving current behavior when `mujoco` is unavailable.

## Task 1: Repository Hygiene

**Files:**
- Modify: `.gitignore`
- Delete from working tree only: `environment/court/src/__pycache__/court_geometry.cpython-313.pyc`

- [x] **Step 1: Inspect current cache artifact**

Run:

```bash
find environment/court -path '*__pycache__*' -print
```

Expected: output includes `environment/court/src/__pycache__/court_geometry.cpython-313.pyc`.

- [x] **Step 2: Add local visual companion ignore rule**

Append this block to `.gitignore`:

```gitignore

# Superpowers local brainstorming visual companion state
.superpowers/
**/.superpowers/
```

Do not remove the existing `__pycache__/` rule; it already covers Python cache directories.

- [x] **Step 3: Remove the generated Python cache file**

Run:

```bash
rm -rf environment/court/src/__pycache__
```

Expected: `find environment/court -path '*__pycache__*' -print` prints nothing.

- [x] **Step 4: Verify intended status**

Run:

```bash
git status --short -- .gitignore environment/court
```

Expected: `.gitignore` is modified, `environment/court/.superpowers/` is not shown, and no `__pycache__` path is shown.

- [x] **Step 5: Commit**

Run:

```bash
git add .gitignore
git commit -m "chore: ignore court local artifacts"
```

Expected: commit succeeds with only `.gitignore` staged.

## Task 2: Geometry Unit Tests

**Files:**
- Create: `environment/court/tests/conftest.py`
- Create: `environment/court/tests/test_court_geometry.py`

- [x] **Step 1: Create test import setup**

Create `environment/court/tests/conftest.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
```

- [x] **Step 2: Write geometry tests**

Create `environment/court/tests/test_court_geometry.py`:

```python
from __future__ import annotations

import math

from court_geometry import CourtParams


def test_nominal_dimensions_and_derived_edges() -> None:
    court = CourtParams()

    assert court.full_court_length == 13.40
    assert court.doubles_width == 6.10
    assert court.singles_width == 5.18
    assert court.line_width == 0.040
    assert court.half_length == 6.70
    assert court.half_width_doubles == 3.05
    assert court.half_width_singles == 2.59
    assert court.short_service_center_abs_x == 2.00
    assert court.doubles_long_service_outer_edge_abs_x == 5.94
    assert court.doubles_long_service_center_abs_x == 5.92


def test_rally_bounds_include_lines_and_exclude_one_mm_outside() -> None:
    court = CourtParams()

    assert court.inside_rally(6.70, 3.05, "doubles")
    assert court.inside_rally(-6.70, -3.05, "doubles")
    assert not court.inside_rally(6.701, 0.0, "doubles")
    assert not court.inside_rally(0.0, -3.051, "doubles")

    assert court.inside_rally(6.70, 2.59, "singles")
    assert court.inside_rally(-6.70, -2.59, "singles")
    assert not court.inside_rally(0.0, 2.591, "singles")
    assert court.inside_rally(0.0, 2.591, "doubles")


def test_service_bounds_include_relevant_lines() -> None:
    court = CourtParams()

    assert court.inside_service(1.98, 0.00, "doubles", "+x", "+y")
    assert court.inside_service(5.94, 3.05, "doubles", "+x", "+y")
    assert not court.inside_service(5.941, 2.00, "doubles", "+x", "+y")

    assert court.inside_service(-1.98, -0.01, "doubles", "-x", "-y")
    assert court.inside_service(-5.94, -3.05, "doubles", "-x", "-y")
    assert not court.inside_service(-5.941, -2.00, "doubles", "-x", "-y")

    assert court.inside_service(6.70, -2.59, "singles", "+x", "-y")
    assert not court.inside_service(6.701, -2.00, "singles", "+x", "-y")


def test_visual_line_rectangles_are_edge_correct() -> None:
    court = CourtParams()
    rects = {rect["name"]: rect for rect in court.visual_line_rectangles()}

    assert rects["doubles_sideline_pos_y"]["y"] == 3.03
    assert rects["doubles_sideline_neg_y"]["y"] == -3.03
    assert rects["singles_sideline_pos_y"]["y"] == 2.57
    assert rects["singles_sideline_neg_y"]["y"] == -2.57
    assert rects["back_boundary_pos_x"]["x"] == 6.68
    assert rects["back_boundary_neg_x"]["x"] == -6.68
    assert rects["short_service_line_pos_x"]["x"] == 2.00
    assert rects["short_service_line_neg_x"]["x"] == -2.00
    assert rects["doubles_long_service_line_pos_x"]["x"] == 5.92
    assert rects["doubles_long_service_line_neg_x"]["x"] == -5.92


def test_net_height_profile_matches_center_and_sidelines() -> None:
    court = CourtParams()

    assert court.net_top_height(0.0) == 1.524
    assert court.net_top_height(3.05) == 1.550
    assert court.net_top_height(-3.05) == 1.550
    assert court.net_bottom_height(0.0) == 0.764
    assert math.isclose(court.net_top_height(1.525), 1.5305, rel_tol=0.0, abs_tol=1e-12)
```

- [x] **Step 3: Run geometry tests**

Run:

```bash
pytest environment/court/tests/test_court_geometry.py -v
```

Expected: all 5 tests pass.

- [x] **Step 4: Commit**

Run:

```bash
git add environment/court/tests/conftest.py environment/court/tests/test_court_geometry.py
git commit -m "test: cover court geometry semantics"
```

Expected: commit succeeds with only the two test files staged.

## Task 3: XML Asset Tests

**Files:**
- Create: `environment/court/tests/test_court_xml.py`

- [x] **Step 1: Write XML tests**

Create `environment/court/tests/test_court_xml.py`:

```python
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VISUAL_ASSET = ROOT / "assets" / "badminton_court_bwf_visual.xml"
COLLISION_ASSET = ROOT / "assets" / "badminton_court_bwf_collision_net.xml"


def _root(path: Path) -> ET.Element:
    return ET.parse(path).getroot()


def _named(root: ET.Element, tag: str, name: str) -> ET.Element:
    for elem in root.iter(tag):
        if elem.attrib.get("name") == name:
            return elem
    raise AssertionError(f"{tag} not found: {name}")


def _geoms(root: ET.Element, prefix: str) -> list[ET.Element]:
    return [
        elem
        for elem in root.iter("geom")
        if elem.attrib.get("name", "").startswith(prefix)
    ]


def test_required_visual_asset_elements_exist() -> None:
    root = _root(VISUAL_ASSET)

    assert root.tag == "mujoco"
    for name in [
        "floor_collision",
        "doubles_sideline_pos_y",
        "doubles_sideline_neg_y",
        "singles_sideline_pos_y",
        "singles_sideline_neg_y",
        "back_boundary_pos_x",
        "back_boundary_neg_x",
        "short_service_line_pos_x",
        "short_service_line_neg_x",
        "doubles_long_service_line_pos_x",
        "doubles_long_service_line_neg_x",
        "centre_service_line_pos_x_half",
        "centre_service_line_neg_x_half",
        "net_post_pos_y",
        "net_post_neg_y",
    ]:
        _named(root, "geom", name)

    for name in [
        "net_midpoint_site",
        "post_official_pos_y_site",
        "post_official_neg_y_site",
        "court_back_pos_x_site",
        "court_back_neg_x_site",
    ]:
        _named(root, "site", name)


def test_visual_asset_has_lines_without_collision_and_no_net_proxy() -> None:
    root = _root(VISUAL_ASSET)

    floor = _named(root, "geom", "floor_collision")
    assert floor.attrib["contype"] == "1"
    assert floor.attrib["conaffinity"] == "1"

    for prefix in [
        "doubles_sideline_",
        "singles_sideline_",
        "back_boundary_",
        "short_service_line_",
        "doubles_long_service_line_",
        "centre_service_line_",
    ]:
        for geom in _geoms(root, prefix):
            assert geom.attrib["contype"] == "0"
            assert geom.attrib["conaffinity"] == "0"

    assert _geoms(root, "net_collision_proxy_") == []


def test_collision_net_asset_enables_only_net_proxy_and_top_cord_collision() -> None:
    root = _root(COLLISION_ASSET)

    proxies = _geoms(root, "net_collision_proxy_")
    assert len(proxies) == 40
    assert {proxy.attrib["contype"] for proxy in proxies} == {"2"}
    assert {proxy.attrib["conaffinity"] for proxy in proxies} == {"1"}

    top_cords = _geoms(root, "net_top_cord_")
    assert len(top_cords) == 40
    assert {cord.attrib["contype"] for cord in top_cords} == {"2"}
    assert {cord.attrib["conaffinity"] for cord in top_cords} == {"1"}

    visual_cords = _geoms(root, "net_vertical_cord_") + _geoms(root, "net_horizontal_cord_")
    assert visual_cords
    assert {cord.attrib["contype"] for cord in visual_cords} == {"0"}
    assert {cord.attrib["conaffinity"] for cord in visual_cords} == {"0"}
```

- [x] **Step 2: Run XML tests**

Run:

```bash
pytest environment/court/tests/test_court_xml.py -v
```

Expected: all 3 tests pass.

- [x] **Step 3: Commit**

Run:

```bash
git add environment/court/tests/test_court_xml.py
git commit -m "test: cover court XML asset semantics"
```

Expected: commit succeeds with only `test_court_xml.py` staged.

## Task 4: Generated Asset Drift Check

**Files:**
- Create: `environment/court/tests/test_generated_assets.py`

- [x] **Step 1: Write drift tests**

Create `environment/court/tests/test_generated_assets.py`:

```python
from __future__ import annotations

from pathlib import Path

from court_geometry import CourtParams
from generate_court_mjcf import generate_mjcf

ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "params" / "court_bwf_nominal.json"
VISUAL_ASSET = ROOT / "assets" / "badminton_court_bwf_visual.xml"
COLLISION_ASSET = ROOT / "assets" / "badminton_court_bwf_collision_net.xml"


def test_committed_visual_asset_matches_generator_output() -> None:
    court = CourtParams.from_json(PARAMS)

    assert VISUAL_ASSET.read_text(encoding="utf-8") == generate_mjcf(
        court,
        enable_net_collision=False,
    )


def test_committed_collision_asset_matches_generator_output() -> None:
    court = CourtParams.from_json(PARAMS)

    assert COLLISION_ASSET.read_text(encoding="utf-8") == generate_mjcf(
        court,
        enable_net_collision=True,
    )
```

- [x] **Step 2: Run drift tests**

Run:

```bash
pytest environment/court/tests/test_generated_assets.py -v
```

Expected: both tests pass. If a test fails, run `python environment/court/src/generate_court_mjcf.py` from the repository root, inspect the XML diff, and commit the intended regenerated assets with the source change that caused the drift.

- [x] **Step 3: Commit**

Run:

```bash
git add environment/court/tests/test_generated_assets.py
git commit -m "test: detect stale court generated assets"
```

Expected: commit succeeds with only `test_generated_assets.py` staged.

## Task 5: Optional MuJoCo Compile Validation

**Files:**
- Modify: `environment/court/src/validate_court_params.py`

- [x] **Step 1: Add optional MuJoCo import helper**

In `environment/court/src/validate_court_params.py`, add this import near the existing imports:

```python
import importlib.util
```

Then add this function below `check`:

```python
def check_optional_mujoco_compile(xml_paths: list[Path], failures: list[str]) -> None:
    """Compile MJCF assets with MuJoCo when the optional dependency is installed."""
    if importlib.util.find_spec("mujoco") is None:
        print("SKIP  MuJoCo compile validation (mujoco package not installed)")
        return

    import mujoco

    for xml_path in xml_paths:
        try:
            model = mujoco.MjModel.from_xml_path(str(xml_path))
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
        except Exception as exc:
            print(f"FAIL  {xml_path.name} compiles with MuJoCo: {exc}")
            failures.append(f"{xml_path.name} compiles with MuJoCo")
        else:
            print(
                f"PASS  {xml_path.name} compiles with MuJoCo "
                f"(ngeom={model.ngeom}, nbody={model.nbody})"
            )
```

- [x] **Step 2: Call the optional compile helper**

In `main()`, replace the inline asset list loop header:

```python
    for asset_name in [
        "badminton_court_bwf_visual.xml",
        "badminton_court_bwf_collision_net.xml",
    ]:
```

with:

```python
    xml_paths = [
        root / "assets" / "badminton_court_bwf_visual.xml",
        root / "assets" / "badminton_court_bwf_collision_net.xml",
    ]

    for xml_path in xml_paths:
        asset_name = xml_path.name
```

Inside that loop, remove this line:

```python
        xml_path = root / "assets" / asset_name
```

After the loop, before `if failures:`, add:

```python
    check_optional_mujoco_compile(xml_paths, failures)
```

- [x] **Step 3: Run validator in current lightweight environment**

Run:

```bash
python environment/court/src/validate_court_params.py
```

Expected: all existing checks pass, output includes `SKIP  MuJoCo compile validation (mujoco package not installed)`, and the script exits successfully.

- [x] **Step 4: Run full court tests**

Run:

```bash
pytest environment/court/tests -v
```

Expected: all court tests pass.

- [x] **Step 5: Commit**

Run:

```bash
git add environment/court/src/validate_court_params.py
git commit -m "test: add optional court MuJoCo compile validation"
```

Expected: commit succeeds with only `validate_court_params.py` staged.

## Task 6: Final Verification And Court Package Commit

**Files:**
- Add to tracking: `environment/court/README.md`
- Add to tracking: `environment/court/badminton_court_design_dossier.md`
- Add to tracking: `environment/court/assets/badminton_court_bwf_visual.xml`
- Add to tracking: `environment/court/assets/badminton_court_bwf_collision_net.xml`
- Add to tracking: `environment/court/docs/codex_tasks.md`
- Add to tracking: `environment/court/docs/validation_protocol.md`
- Add to tracking: `environment/court/params/court_bwf_nominal.json`
- Add to tracking: `environment/court/src/court_geometry.py`
- Add to tracking: `environment/court/src/generate_court_mjcf.py`
- Add to tracking: `environment/court/src/validate_court_params.py`
- Add to tracking: `environment/court/tests/`

- [x] **Step 1: Run static validator**

Run:

```bash
python environment/court/src/validate_court_params.py
```

Expected: all court design checks pass. If `mujoco` is unavailable, the optional MuJoCo compile check prints `SKIP` and the script still exits successfully.

- [x] **Step 2: Run all court tests**

Run:

```bash
pytest environment/court/tests -v
```

Expected: all court tests pass.

- [x] **Step 3: Confirm generated assets are current**

Run:

```bash
pytest environment/court/tests/test_generated_assets.py -v
```

Expected: both drift tests pass.

- [x] **Step 4: Stage only court package files**

Run:

```bash
git add environment/court/README.md \
  environment/court/badminton_court_design_dossier.md \
  environment/court/assets/badminton_court_bwf_visual.xml \
  environment/court/assets/badminton_court_bwf_collision_net.xml \
  environment/court/docs/codex_tasks.md \
  environment/court/docs/validation_protocol.md \
  environment/court/params/court_bwf_nominal.json \
  environment/court/src/court_geometry.py \
  environment/court/src/generate_court_mjcf.py \
  environment/court/src/validate_court_params.py \
  environment/court/tests
```

- [x] **Step 5: Inspect staged files**

Run:

```bash
git diff --cached --name-only
```

Expected: staged files are limited to `.gitignore` if not already committed, the court package source/assets/docs/tests, and no `.superpowers`, `__pycache__`, shuttlecock, racket, or fullbody files.

- [x] **Step 6: Commit tracked court package**

Run:

```bash
git commit -m "feat: add hardened court asset package"
```

Expected: commit succeeds and records the court package with tests and validation.

- [x] **Step 7: Final status check**

Run:

```bash
git status --short
```

Expected: no `environment/court/.superpowers/` or `environment/court/src/__pycache__/` paths are shown. Pre-existing unrelated changes may still appear and should be left untouched.

## Self-Review Notes

- Spec coverage: this plan covers cache hygiene, generated-asset policy, MuJoCo compile validation, focused tests, collision-net usage boundaries, and downstream helper usage guidance.
- Scope: this is one coherent hardening pass for the court package; it does not modify shuttlecock, racket, or SMPL integration code.
- Type consistency: tests import existing `CourtParams`, `generate_mjcf`, and existing XML names from the reviewed assets.

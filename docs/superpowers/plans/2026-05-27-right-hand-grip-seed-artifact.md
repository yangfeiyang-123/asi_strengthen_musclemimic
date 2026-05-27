# Right-Hand Grip Seed Artifact Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible grip seed artifact so the standalone grip environment and Overall badminton scene reset from the same visually reviewed right-hand racket grip.

**Architecture:** Add a small seed loader module that validates a seed JSON and copies named right-hand joints by name. Add a seed builder CLI that generates a stronger curled forehand V-grip seed plus verification renders. Switch Overall scene generation from `configs/right_hand_racket_grip_reference.json` to the seed JSON as the default source, while preserving a compatibility fallback.

**Tech Stack:** Python 3.11/3.13, MuJoCo Python bindings, NumPy/SciPy, PIL for PNG output, pytest.

---

## File Structure

- Create `src/grip/grip_seed.py`: seed schema validation, loading, joint-name copy helpers, joint shape metrics.
- Create `src/grip/build_right_hand_racket_grip_seed.py`: deterministic seed generation CLI, render output, report writing.
- Modify `src/grip/paths.py`: default seed path helpers.
- Modify `src/grip/right_hand_racket_grip_env.py`: allow the existing reference path to point at the seed JSON without changing env semantics.
- Modify `environment/overall_environment/src/paths.py`: expose default seed JSON path.
- Modify `environment/overall_environment/src/build_overall_environment.py`: load the seed by default, copy right-hand joints from seed, place the racket from seed orientation.
- Modify `tests/test_right_hand_racket_grip.py`: seed loader and builder smoke tests.
- Modify `environment/overall_environment/tests/test_overall_environment.py`: Overall uses seed tests.
- Generate `outputs/right_hand_racket_grip/reference/*`: seed JSON, seed scene XML, report, and verification renders.
- Update `docs/right_hand_racket_grip.md`: document the seed artifact and command.

## Task 1: Add Seed Paths And Loader

**Files:**
- Modify: `src/grip/paths.py`
- Create: `src/grip/grip_seed.py`
- Modify: `tests/test_right_hand_racket_grip.py`

- [ ] **Step 1: Write failing seed path and loader tests**

Append these imports in `tests/test_right_hand_racket_grip.py`:

```python
from src.grip.grip_seed import (
    GripSeed,
    apply_seed_right_hand_joints,
    joint_shape_metrics,
    load_grip_seed,
)
from src.grip.paths import grip_seed_json_path
```

Append these tests:

```python
def _write_seed_from_reference(tmp_path, scene: Path, reference: Path) -> Path:
    raw = json.loads(reference.read_text(encoding="utf-8"))
    seed = {
        "schema_version": 1,
        "source_xml": str(scene),
        "target_config": str(target_config_path()),
        "qpos": raw["qpos"],
        "qvel": raw["qvel"],
        "right_hand_joint_names": raw["right_hand_joint_names"],
        "racket_freejoint_name": "racket_free",
        "racket_freejoint_qpos": raw["racket_freejoint_qpos"],
        "site_errors_m": raw["site_errors_m"],
        "joint_shape_metrics": {},
        "contact_metrics": {},
        "visualization_paths": [],
        "generation_command": ["pytest"],
    }
    path = tmp_path / "right_hand_racket_grip_seed.json"
    path.write_text(json.dumps(seed), encoding="utf-8")
    return path


def test_grip_seed_default_path_is_under_outputs_reference():
    assert grip_seed_json_path() == REPO_ROOT / "outputs" / "right_hand_racket_grip" / "reference" / "right_hand_racket_grip_seed.json"


def test_load_grip_seed_validates_schema_and_dimensions(tmp_path):
    scene, _, reference = _build_smoke_paths(tmp_path)
    seed_path = _write_seed_from_reference(tmp_path, scene, reference)

    seed = load_grip_seed(seed_path)

    model = mujoco.MjModel.from_xml_path(str(scene))
    assert isinstance(seed, GripSeed)
    assert seed.qpos.shape == (model.nq,)
    assert seed.qvel.shape == (model.nv,)
    assert seed.racket_freejoint_qpos.shape == (7,)
    assert "mcp2_flexion_r" in seed.right_hand_joint_names


def test_load_grip_seed_rejects_wrong_qpos_length(tmp_path):
    scene, _, reference = _build_smoke_paths(tmp_path)
    seed_path = _write_seed_from_reference(tmp_path, scene, reference)
    raw = json.loads(seed_path.read_text(encoding="utf-8"))
    raw["qpos"] = raw["qpos"][:-1]
    seed_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="seed qpos"):
        load_grip_seed(seed_path)


def test_apply_seed_right_hand_joints_copies_by_joint_name(tmp_path):
    scene, _, reference = _build_smoke_paths(tmp_path)
    seed = load_grip_seed(_write_seed_from_reference(tmp_path, scene, reference))
    source_model = mujoco.MjModel.from_xml_path(str(scene))
    target_model = mujoco.MjModel.from_xml_path(str(scene))
    qpos = np.array(target_model.qpos0, dtype=float)

    apply_seed_right_hand_joints(seed, target_model, qpos)

    for joint_name in seed.right_hand_joint_names:
        source_id = mujoco.mj_name2id(source_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        target_id = mujoco.mj_name2id(target_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        source_adr = int(source_model.jnt_qposadr[source_id])
        target_adr = int(target_model.jnt_qposadr[target_id])
        assert qpos[target_adr] == pytest.approx(seed.qpos[source_adr])


def test_joint_shape_metrics_reports_extended_ring_and_pinky(tmp_path):
    scene, _, reference = _build_smoke_paths(tmp_path)
    seed = load_grip_seed(_write_seed_from_reference(tmp_path, scene, reference))
    model = mujoco.MjModel.from_xml_path(str(scene))

    metrics = joint_shape_metrics(model, seed.qpos, seed.right_hand_joint_names)

    assert metrics["mcp4_flexion_r"]["value"] >= 0.0
    assert metrics["pm5_flexion_r"]["lower_margin"] >= 0.0
    assert metrics["md5_flexion_r"]["lower_margin"] >= 0.0
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_right_hand_racket_grip.py::test_grip_seed_default_path_is_under_outputs_reference \
  tests/test_right_hand_racket_grip.py::test_load_grip_seed_validates_schema_and_dimensions \
  tests/test_right_hand_racket_grip.py::test_load_grip_seed_rejects_wrong_qpos_length \
  tests/test_right_hand_racket_grip.py::test_apply_seed_right_hand_joints_copies_by_joint_name \
  tests/test_right_hand_racket_grip.py::test_joint_shape_metrics_reports_extended_ring_and_pinky -q
```

Expected: FAIL because `src.grip.grip_seed` and `grip_seed_json_path` do not exist.

- [ ] **Step 3: Implement seed path helper**

In `src/grip/paths.py`, add:

```python
def grip_seed_json_path() -> Path:
    return REPO_ROOT / "outputs" / "right_hand_racket_grip" / "reference" / "right_hand_racket_grip_seed.json"


def grip_seed_reference_dir() -> Path:
    return grip_seed_json_path().parent
```

- [ ] **Step 4: Implement `src/grip/grip_seed.py`**

Create:

```python
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from src.grip.paths import grip_seed_json_path, scene_xml_path


REQUIRED_SEED_KEYS = {
    "schema_version",
    "source_xml",
    "target_config",
    "qpos",
    "qvel",
    "right_hand_joint_names",
    "racket_freejoint_name",
    "racket_freejoint_qpos",
}


@dataclass(frozen=True)
class GripSeed:
    path: Path
    raw: dict[str, Any]
    source_xml: Path
    qpos: np.ndarray
    qvel: np.ndarray
    right_hand_joint_names: tuple[str, ...]
    racket_freejoint_name: str
    racket_freejoint_qpos: np.ndarray


def load_grip_seed(path: str | Path | None = None) -> GripSeed:
    seed_path = Path(path) if path is not None else grip_seed_json_path()
    with seed_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("grip seed JSON root must be an object")
    missing = sorted(REQUIRED_SEED_KEYS - set(raw))
    if missing:
        raise ValueError(f"grip seed missing required keys: {missing}")

    source_xml = Path(str(raw["source_xml"]))
    if not source_xml.is_absolute():
        source_xml = seed_path.parent.parent.parent.parent / source_xml
    if not source_xml.is_file():
        source_xml = scene_xml_path()
    model = mujoco.MjModel.from_xml_path(str(source_xml))

    qpos = _finite_vector(raw["qpos"], model.nq, "seed qpos")
    qvel = _finite_vector(raw["qvel"], model.nv, "seed qvel")
    racket_freejoint_qpos = _finite_vector(raw["racket_freejoint_qpos"], 7, "seed racket_freejoint_qpos")

    right_hand_joint_names = tuple(str(name) for name in raw["right_hand_joint_names"])
    if not right_hand_joint_names:
        raise ValueError("grip seed right_hand_joint_names must be non-empty")
    for joint_name in right_hand_joint_names:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name) < 0:
            raise ValueError(f"grip seed joint {joint_name!r} is missing from source model")

    racket_freejoint_name = str(raw["racket_freejoint_name"])
    racket_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, racket_freejoint_name)
    if racket_id < 0:
        raise ValueError(f"grip seed racket freejoint {racket_freejoint_name!r} is missing from source model")
    if int(model.jnt_type[racket_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError(f"grip seed racket joint {racket_freejoint_name!r} is not a freejoint")

    return GripSeed(
        path=seed_path,
        raw=raw,
        source_xml=source_xml,
        qpos=qpos,
        qvel=qvel,
        right_hand_joint_names=right_hand_joint_names,
        racket_freejoint_name=racket_freejoint_name,
        racket_freejoint_qpos=racket_freejoint_qpos,
    )


def apply_seed_right_hand_joints(seed: GripSeed, target_model: mujoco.MjModel, qpos: np.ndarray) -> None:
    source_model = mujoco.MjModel.from_xml_path(str(seed.source_xml))
    for joint_name in seed.right_hand_joint_names:
        source_id = mujoco.mj_name2id(source_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        target_id = mujoco.mj_name2id(target_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if source_id < 0:
            raise ValueError(f"seed source missing joint {joint_name!r}")
        if target_id < 0:
            raise ValueError(f"target model missing seed joint {joint_name!r}")
        width = _joint_qpos_width(source_model, source_id)
        if width != _joint_qpos_width(target_model, target_id):
            raise ValueError(f"joint {joint_name!r} qpos width differs between seed and target model")
        source_adr = int(source_model.jnt_qposadr[source_id])
        target_adr = int(target_model.jnt_qposadr[target_id])
        qpos[target_adr : target_adr + width] = seed.qpos[source_adr : source_adr + width]


def joint_shape_metrics(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    joint_names: tuple[str, ...] | list[str],
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0 or _joint_qpos_width(model, joint_id) != 1:
            continue
        adr = int(model.jnt_qposadr[joint_id])
        value = float(qpos[adr])
        if bool(model.jnt_limited[joint_id]):
            lower = float(model.jnt_range[joint_id, 0])
            upper = float(model.jnt_range[joint_id, 1])
            metrics[joint_name] = {
                "value": value,
                "lower": lower,
                "upper": upper,
                "lower_margin": value - lower,
                "upper_margin": upper - value,
            }
        else:
            metrics[joint_name] = {"value": value}
    return metrics


def _finite_vector(value: object, expected_size: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (expected_size,):
        raise ValueError(f"{label} must have shape ({expected_size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be finite")
    return array.copy()


def _joint_qpos_width(model: mujoco.MjModel, joint_id: int) -> int:
    joint_type = int(model.jnt_type[joint_id])
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return 4
    return 1
```

- [ ] **Step 5: Run seed loader tests**

Run the same pytest command from Step 2.

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/grip/paths.py src/grip/grip_seed.py tests/test_right_hand_racket_grip.py
git commit -m "feat: add right hand grip seed loader"
```

## Task 2: Add Seed Builder And Verification Renders

**Files:**
- Create: `src/grip/build_right_hand_racket_grip_seed.py`
- Modify: `tests/test_right_hand_racket_grip.py`

- [ ] **Step 1: Write failing builder smoke test**

Append this import:

```python
from src.grip.build_right_hand_racket_grip_seed import build_grip_seed
```

Append this test:

```python
def test_build_grip_seed_writes_artifact_report_and_renders(tmp_path):
    scene = tmp_path / "grip_scene.xml"
    initial_reference = tmp_path / "reference.json"
    out = tmp_path / "reference" / "right_hand_racket_grip_seed.json"
    build_scene(scene)
    solve_reference(scene, target_config_path(), initial_reference, max_nfev=2)

    result = build_grip_seed(
        xml=scene,
        targets=target_config_path(),
        out=out,
        initial_reference=initial_reference,
        max_nfev=4,
        render=False,
    )

    seed = load_grip_seed(out)
    assert result["out"] == str(out)
    assert out.is_file()
    assert (out.parent / "right_hand_racket_grip_seed_report.json").is_file()
    assert (out.parent / "right_hand_racket_grip_seed_scene.xml").is_file()
    assert seed.raw["schema_version"] == 1
    assert "joint_shape_metrics" in seed.raw
    assert "contact_metrics" in seed.raw
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
pytest tests/test_right_hand_racket_grip.py::test_build_grip_seed_writes_artifact_report_and_renders -q
```

Expected: FAIL because `src.grip.build_right_hand_racket_grip_seed` does not exist.

- [ ] **Step 3: Implement seed builder**

Create `src/grip/build_right_hand_racket_grip_seed.py` with:

```python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any

import mujoco
import numpy as np
from scipy.optimize import least_squares

from src.grip.grip_seed import joint_shape_metrics
from src.grip.hand_racket_model_map import load_model_map
from src.grip.paths import grip_seed_json_path, scene_xml_path, target_config_path
from src.grip.solve_right_hand_racket_grip import (
    hand_site_positions,
    handle_max_penetration,
    racket_freejoint_qpos,
    racket_freejoint_qpos_address,
    racket_local_targets_to_world,
    right_hand_qpos_indices_and_bounds,
)
from src.grip.target_config import load_grip_target_config


CURL_PRIORS = {
    "mcp4_flexion_r": 0.35,
    "pm4_flexion_r": 1.05,
    "md4_flexion_r": 0.65,
    "mcp5_flexion_r": 0.30,
    "pm5_flexion_r": 0.85,
    "md5_flexion_r": 0.55,
}


def build_grip_seed(
    xml: str | Path = scene_xml_path(),
    targets: str | Path = target_config_path(),
    out: str | Path = grip_seed_json_path(),
    *,
    initial_reference: str | Path | None = None,
    max_nfev: int = 120,
    render: bool = True,
) -> dict[str, Any]:
    xml_path = Path(xml)
    targets_path = Path(targets)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    model_map = load_model_map(model)
    if not model_map.ok:
        raise ValueError(f"unresolved MuJoCo model names: {model_map.missing}")

    target_config = load_grip_target_config(targets_path)
    qpos = _initial_qpos(model, initial_reference)
    qvel = np.zeros(model.nv, dtype=float)
    hand_indices, lower, upper = right_hand_qpos_indices_and_bounds(model, model_map.right_hand_joint_names)
    racket_adr = racket_freejoint_qpos_address(model, model_map)
    initial_values = np.concatenate([qpos[hand_indices], qpos[racket_adr : racket_adr + 7]])

    def residual(values: np.ndarray) -> np.ndarray:
        qpos[hand_indices] = values[: len(hand_indices)]
        qpos[racket_adr : racket_adr + 7] = _normalized_freejoint(values[len(hand_indices) :])
        current_sites = hand_site_positions(model, data, qpos, model_map)
        target_sites = racket_local_targets_to_world(model, data, qpos, target_config, model_map)
        site_residuals = np.concatenate(
            [(current_sites[name] - target_sites[name]) * target_config.target_weight(name) for name in sorted(target_sites)]
        )
        curl_residuals = _curl_prior_residuals(model, qpos, weight=0.35)
        lower_bound_residuals = _lower_bound_residuals(model, qpos, model_map.right_hand_joint_names, weight=0.08)
        penetration = handle_max_penetration(model, data, qpos, model_map)
        penetration_residual = np.array([max(0.0, penetration - 0.003) * 8.0], dtype=float)
        return np.concatenate([site_residuals, curl_residuals, lower_bound_residuals, penetration_residual])

    variable_lower = np.concatenate([lower, np.full(7, -np.inf, dtype=float)])
    variable_upper = np.concatenate([upper, np.full(7, np.inf, dtype=float)])
    result = least_squares(
        residual,
        initial_values,
        bounds=(variable_lower, variable_upper),
        max_nfev=max_nfev,
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
    )
    qpos[hand_indices] = result.x[: len(hand_indices)]
    qpos[racket_adr : racket_adr + 7] = _normalized_freejoint(result.x[len(hand_indices) :])
    mujoco.mj_forward(model, data)

    site_errors = _site_errors(model, data, qpos, target_config, model_map)
    contact_metrics = _contact_metrics(model, data, model_map)
    shape_metrics = joint_shape_metrics(model, qpos, model_map.right_hand_joint_names)
    seed_scene = out_path.parent / "right_hand_racket_grip_seed_scene.xml"
    shutil.copyfile(xml_path, seed_scene)
    _write_keyframe(seed_scene, "right_hand_racket_grip_seed", qpos)

    visualization_paths: list[str] = []
    if render:
        visualization_paths = _render_seed_views(model, data, out_path.parent / "visualization")

    raw = {
        "schema_version": 1,
        "source_xml": str(xml_path),
        "target_config": str(targets_path),
        "qpos": _float_list(qpos),
        "qvel": _float_list(qvel),
        "right_hand_joint_names": list(model_map.right_hand_joint_names),
        "racket_freejoint_name": str(model_map.racket_freejoint),
        "racket_freejoint_qpos": racket_freejoint_qpos(model, qpos, model_map),
        "site_errors_m": site_errors,
        "joint_shape_metrics": shape_metrics,
        "contact_metrics": contact_metrics,
        "visualization_paths": visualization_paths,
        "generation_command": _generation_command(xml_path, targets_path, out_path, initial_reference, max_nfev),
        "objective": {
            "success": bool(result.success),
            "cost": float(result.cost),
            "nfev": int(result.nfev),
            "message": str(result.message),
        },
    }
    out_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = out_path.parent / "right_hand_racket_grip_seed_report.json"
    report_path.write_text(json.dumps({"out": str(out_path), **raw["objective"], "site_errors_m": site_errors, "contact_metrics": contact_metrics}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"out": str(out_path), "report": str(report_path), "seed_scene": str(seed_scene), "nfev": int(result.nfev)}
```

Also include helper functions in the same file:

```python
def _initial_qpos(model: mujoco.MjModel, initial_reference: str | Path | None) -> np.ndarray:
    if initial_reference is None:
        return np.array(model.qpos0, dtype=float)
    raw = json.loads(Path(initial_reference).read_text(encoding="utf-8"))
    qpos = np.asarray(raw["qpos"], dtype=float)
    if qpos.shape != (model.nq,):
        raise ValueError(f"initial reference qpos must have shape ({model.nq},), got {qpos.shape}")
    return qpos.copy()


def _normalized_freejoint(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    quat = result[3:7]
    norm = float(np.linalg.norm(quat))
    if norm == 0.0 or not np.isfinite(norm):
        result[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        result[3:7] = quat / norm
    return result


def _curl_prior_residuals(model: mujoco.MjModel, qpos: np.ndarray, *, weight: float) -> np.ndarray:
    residuals = []
    for joint_name, target in CURL_PRIORS.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            continue
        adr = int(model.jnt_qposadr[joint_id])
        residuals.append((float(qpos[adr]) - target) * weight)
    return np.asarray(residuals, dtype=float)


def _lower_bound_residuals(model: mujoco.MjModel, qpos: np.ndarray, joint_names: tuple[str, ...], *, weight: float) -> np.ndarray:
    residuals = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0 or not bool(model.jnt_limited[joint_id]):
            continue
        adr = int(model.jnt_qposadr[joint_id])
        lower = float(model.jnt_range[joint_id, 0])
        residuals.append(max(0.0, 0.03 - (float(qpos[adr]) - lower)) * weight)
    return np.asarray(residuals, dtype=float)


def _site_errors(model, data, qpos, target_config, model_map) -> dict[str, float]:
    current_sites = hand_site_positions(model, data, qpos, model_map)
    target_sites = racket_local_targets_to_world(model, data, qpos, target_config, model_map)
    return {name: float(np.linalg.norm(current_sites[name] - target_sites[name])) for name in sorted(target_sites)}


def _contact_metrics(model: mujoco.MjModel, data: mujoco.MjData, model_map) -> dict[str, float | int]:
    handle_ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in model_map.handle_geoms}
    handle_ids.discard(-1)
    contacts = 0
    max_penetration = 0.0
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        if int(contact.geom1) in handle_ids or int(contact.geom2) in handle_ids:
            contacts += 1
            max_penetration = max(max_penetration, max(0.0, -float(contact.dist)))
    return {"raw_handle_contacts": contacts, "max_handle_penetration_m": max_penetration}


def _write_keyframe(xml_path: Path, key_name: str, qpos: np.ndarray) -> None:
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()
    keyframe = root.find("keyframe")
    if keyframe is None:
        keyframe = ET.SubElement(root, "keyframe")
    for key in list(keyframe.findall("key")):
        if key.attrib.get("name") == key_name:
            keyframe.remove(key)
    ET.SubElement(keyframe, "key", {"name": key_name, "qpos": " ".join(f"{value:.17g}" for value in qpos)})
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)


def _render_seed_views(model: mujoco.MjModel, data: mujoco.MjData, out_dir: Path) -> list[str]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=480, width=640)
    paths = []
    for name, azimuth in (("seed_grip_closeup_front.png", 70), ("seed_grip_closeup_side.png", 20)):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "rh_palm_grip_site")]
        cam.distance = 0.35
        cam.azimuth = azimuth
        cam.elevation = -8
        renderer.update_scene(data, camera=cam)
        path = out_dir / name
        Image.fromarray(renderer.render()).save(path)
        paths.append(str(path))
    renderer.close()
    return paths


def _generation_command(xml: Path, targets: Path, out: Path, initial_reference: str | Path | None, max_nfev: int) -> list[str]:
    command = ["python", "-m", "src.grip.build_right_hand_racket_grip_seed", "--xml", str(xml), "--targets", str(targets), "--out", str(out), "--max-nfev", str(max_nfev)]
    if initial_reference is not None:
        command.extend(["--initial-reference", str(initial_reference)])
    return command


def _float_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=float).tolist()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reproducible right-hand racket grip seed artifact.")
    parser.add_argument("--xml", type=Path, default=scene_xml_path())
    parser.add_argument("--targets", type=Path, default=target_config_path())
    parser.add_argument("--out", type=Path, default=grip_seed_json_path())
    parser.add_argument("--initial-reference", type=Path, default=None)
    parser.add_argument("--max-nfev", type=int, default=120)
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    print(json.dumps(build_grip_seed(args.xml, args.targets, args.out, initial_reference=args.initial_reference, max_nfev=args.max_nfev, render=not args.no_render), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run builder smoke test**

Run:

```bash
pytest tests/test_right_hand_racket_grip.py::test_build_grip_seed_writes_artifact_report_and_renders -q
```

Expected: PASS.

- [ ] **Step 5: Run loader and builder tests together**

Run:

```bash
pytest tests/test_right_hand_racket_grip.py::test_grip_seed_default_path_is_under_outputs_reference \
  tests/test_right_hand_racket_grip.py::test_load_grip_seed_validates_schema_and_dimensions \
  tests/test_right_hand_racket_grip.py::test_load_grip_seed_rejects_wrong_qpos_length \
  tests/test_right_hand_racket_grip.py::test_apply_seed_right_hand_joints_copies_by_joint_name \
  tests/test_right_hand_racket_grip.py::test_joint_shape_metrics_reports_extended_ring_and_pinky \
  tests/test_right_hand_racket_grip.py::test_build_grip_seed_writes_artifact_report_and_renders -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/grip/build_right_hand_racket_grip_seed.py tests/test_right_hand_racket_grip.py
git commit -m "feat: build right hand grip seed artifact"
```

## Task 3: Make Overall Use The Seed

**Files:**
- Modify: `environment/overall_environment/src/paths.py`
- Modify: `environment/overall_environment/src/build_overall_environment.py`
- Modify: `environment/overall_environment/tests/test_overall_environment.py`

- [ ] **Step 1: Write failing Overall seed tests**

Add imports in `environment/overall_environment/tests/test_overall_environment.py`:

```python
from src.grip.build_right_hand_racket_grip_seed import build_grip_seed
from src.grip.grip_seed import load_grip_seed
```

Append:

```python
def test_overall_ready_can_use_explicit_grip_seed(tmp_path):
    grip_scene = tmp_path / "grip_scene.xml"
    reference = tmp_path / "reference.json"
    seed_path = tmp_path / "right_hand_racket_grip_seed.json"
    overall = tmp_path / "overall_badminton_scene.xml"
    from src.grip.build_right_hand_racket_grip_scene import build_scene
    from src.grip.solve_right_hand_racket_grip import solve_reference
    from src.grip.paths import target_config_path

    build_scene(grip_scene)
    solve_reference(grip_scene, target_config_path(), reference, max_nfev=2)
    build_grip_seed(grip_scene, target_config_path(), seed_path, initial_reference=reference, max_nfev=4, render=False)
    build_overall_scene(overall, grip_seed=seed_path)

    model = mujoco.MjModel.from_xml_path(str(overall))
    seed = load_grip_seed(seed_path)
    seed_model = mujoco.MjModel.from_xml_path(str(seed.source_xml))
    for joint_name in seed.right_hand_joint_names:
        seed_joint = _name_id(seed_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        overall_joint = _name_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        seed_adr = int(seed_model.jnt_qposadr[seed_joint])
        overall_adr = int(model.jnt_qposadr[overall_joint])
        assert model.qpos0[overall_adr] == pytest.approx(seed.qpos[seed_adr])
```

Also update imports to include `pytest`.

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
pytest environment/overall_environment/tests/test_overall_environment.py::test_overall_ready_can_use_explicit_grip_seed -q
```

Expected: FAIL because `build_overall_scene()` does not accept `grip_seed`.

- [ ] **Step 3: Add Overall seed path helper**

In `environment/overall_environment/src/paths.py`, add:

```python
def grip_seed_json_path() -> Path:
    return REPO_ROOT / "outputs" / "right_hand_racket_grip" / "reference" / "right_hand_racket_grip_seed.json"
```

- [ ] **Step 4: Modify Overall builder signature and qpos source**

In `environment/overall_environment/src/build_overall_environment.py`:

Add imports:

```python
from src.grip.grip_seed import GripSeed, apply_seed_right_hand_joints, load_grip_seed
```

Import the new path:

```python
    grip_seed_json_path,
```

Change the signature:

```python
def build_overall_scene(output_xml: str | Path | None = None, *, grip_seed: str | Path | None = None) -> Path:
```

Change the qpos call:

```python
        qpos = _overall_ready_qpos(raw_xml, grip_seed)
```

Replace `_overall_ready_qpos` with:

```python
def _overall_ready_qpos(xml_path: Path, grip_seed: str | Path | None = None) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    qpos = np.array(model.qpos0, dtype=float)
    seed_path = Path(grip_seed) if grip_seed is not None else grip_seed_json_path()
    seed = load_grip_seed(seed_path) if seed_path.is_file() else None

    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, HUMAN_ROOT_FREEJOINT)
    if root_id < 0:
        raise ValueError(f"missing joint {HUMAN_ROOT_FREEJOINT!r}")
    root_adr = int(model.jnt_qposadr[root_id])
    qpos[root_adr : root_adr + 3] = INITIAL_HUMAN_ROOT_POS

    if seed is None:
        reference = json.loads(grip_reference_json_path().read_text(encoding="utf-8"))
        _copy_legacy_reference_hand_qpos(model, qpos, reference)
        _place_racket_at_right_hand(model, qpos, reference)
    else:
        apply_seed_right_hand_joints(seed, model, qpos)
        _place_seed_racket_at_right_hand(model, qpos, seed)

    shuttle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, SHUTTLE_FREEJOINT)
    if shuttle_id < 0:
        raise ValueError(f"missing joint {SHUTTLE_FREEJOINT!r}")
    shuttle_adr = int(model.jnt_qposadr[shuttle_id])
    qpos[shuttle_adr : shuttle_adr + 7] = np.concatenate([INITIAL_SHUTTLE_POS, INITIAL_SHUTTLE_QUAT])
    return qpos
```

Add:

```python
def _copy_legacy_reference_hand_qpos(model: mujoco.MjModel, qpos: np.ndarray, reference: dict[str, object]) -> None:
    reference_qpos = np.asarray(reference["qpos"], dtype=float)
    reference_model = mujoco.MjModel.from_xml_path(str(grip_reference_xml_path()))
    right_hand_joint_names = set(reference["right_hand_joint_names"])
    for joint_id in range(reference_model.njnt):
        joint_name = mujoco.mj_id2name(reference_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name not in right_hand_joint_names:
            continue
        target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if target_id < 0:
            continue
        width = _joint_qpos_width(reference_model, joint_id)
        source_adr = int(reference_model.jnt_qposadr[joint_id])
        target_adr = int(model.jnt_qposadr[target_id])
        qpos[target_adr : target_adr + width] = reference_qpos[source_adr : source_adr + width]
```

Add:

```python
def _place_seed_racket_at_right_hand(model: mujoco.MjModel, qpos: np.ndarray, seed: GripSeed) -> None:
    reference = {"racket_freejoint_qpos": seed.racket_freejoint_qpos.tolist()}
    _place_racket_at_right_hand(model, qpos, reference)
```

- [ ] **Step 5: Update `_apply_ready_as_initial_pose` to use seed joint names when present**

In `_apply_ready_as_initial_pose`, replace the reference load block with:

```python
    seed_path = grip_seed_json_path()
    if seed_path.is_file():
        right_hand_joint_names = set(load_grip_seed(seed_path).right_hand_joint_names)
    else:
        reference = json.loads(grip_reference_json_path().read_text(encoding="utf-8"))
        right_hand_joint_names = set(reference["right_hand_joint_names"])
```

- [ ] **Step 6: Run Overall seed test**

Run:

```bash
pytest environment/overall_environment/tests/test_overall_environment.py::test_overall_ready_can_use_explicit_grip_seed -q
```

Expected: PASS.

- [ ] **Step 7: Run Overall test file**

Run:

```bash
pytest environment/overall_environment/tests/test_overall_environment.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add environment/overall_environment/src/paths.py environment/overall_environment/src/build_overall_environment.py environment/overall_environment/tests/test_overall_environment.py
git commit -m "feat: initialize overall scene from grip seed"
```

## Task 4: Generate Seed Artifact And Regenerate Overall Assets

**Files:**
- Generate: `outputs/right_hand_racket_grip/reference/right_hand_racket_grip_seed.json`
- Generate: `outputs/right_hand_racket_grip/reference/right_hand_racket_grip_seed_scene.xml`
- Generate: `outputs/right_hand_racket_grip/reference/right_hand_racket_grip_seed_report.json`
- Generate: `outputs/right_hand_racket_grip/reference/visualization/*.png`
- Modify: `environment/overall_environment/assets/overall_badminton_scene.xml`

- [ ] **Step 1: Build the seed artifact**

Run:

```bash
python -m src.grip.build_right_hand_racket_grip_seed \
  --xml assets/right_hand_racket_grip_scene.xml \
  --targets configs/right_hand_racket_grip_targets.json \
  --initial-reference configs/right_hand_racket_grip_reference.json \
  --out outputs/right_hand_racket_grip/reference/right_hand_racket_grip_seed.json \
  --max-nfev 160
```

Expected: prints JSON with `out`, `report`, and `seed_scene`.

- [ ] **Step 2: Inspect seed shape metrics**

Run:

```bash
python - <<'PY'
import json
raw=json.load(open("outputs/right_hand_racket_grip/reference/right_hand_racket_grip_seed.json"))
for name in ("mcp4_flexion_r","pm5_flexion_r","md5_flexion_r","mcp5_flexion_r"):
    print(name, raw["joint_shape_metrics"][name])
print("site_errors", raw["site_errors_m"])
print("contact", raw["contact_metrics"])
PY
```

Expected: ring/pinky flexion values are not all at their lower bounds; metrics are finite.

- [ ] **Step 3: Rebuild Overall scene using the seed**

Run:

```bash
python -m environment.overall_environment.src.build_overall_environment \
  --out environment/overall_environment/assets/overall_badminton_scene.xml
```

Expected: prints `environment/overall_environment/assets/overall_badminton_scene.xml`.

- [ ] **Step 4: Verify Overall right-hand qpos matches seed**

Run:

```bash
python - <<'PY'
import mujoco, numpy as np
from src.grip.grip_seed import load_grip_seed
seed=load_grip_seed("outputs/right_hand_racket_grip/reference/right_hand_racket_grip_seed.json")
seed_model=mujoco.MjModel.from_xml_path(str(seed.source_xml))
overall=mujoco.MjModel.from_xml_path("environment/overall_environment/assets/overall_badminton_scene.xml")
diffs=[]
for joint_name in seed.right_hand_joint_names:
    sj=mujoco.mj_name2id(seed_model,mujoco.mjtObj.mjOBJ_JOINT,joint_name)
    oj=mujoco.mj_name2id(overall,mujoco.mjtObj.mjOBJ_JOINT,joint_name)
    diffs.append(abs(float(seed.qpos[int(seed_model.jnt_qposadr[sj])])-float(overall.qpos0[int(overall.jnt_qposadr[oj])])))
print("max_right_hand_seed_diff", max(diffs))
PY
```

Expected: `max_right_hand_seed_diff 0.0` or a value below `1e-12`.

- [ ] **Step 5: Commit generated seed and Overall asset**

```bash
git add outputs/right_hand_racket_grip/reference environment/overall_environment/assets/overall_badminton_scene.xml environment/overall_environment/assets/mimic_msk_model
git commit -m "data: add right hand grip seed artifact"
```

## Task 5: Document And Run Final Verification

**Files:**
- Modify: `docs/right_hand_racket_grip.md`

- [ ] **Step 1: Update documentation**

Add a `Grip Seed Artifact` section to `docs/right_hand_racket_grip.md`:

```markdown
## Grip Seed Artifact

The training reset and Overall initial scene now use a first-class seed artifact:

`outputs/right_hand_racket_grip/reference/right_hand_racket_grip_seed.json`

Generate it with:

```bash
python -m src.grip.build_right_hand_racket_grip_seed \
  --xml assets/right_hand_racket_grip_scene.xml \
  --targets configs/right_hand_racket_grip_targets.json \
  --initial-reference configs/right_hand_racket_grip_reference.json \
  --out outputs/right_hand_racket_grip/reference/right_hand_racket_grip_seed.json \
  --max-nfev 160
```

The seed JSON is the source of truth for the initial right-hand joint values and racket orientation. The PNGs in `outputs/right_hand_racket_grip/reference/visualization/` are verification renders only.
```

- [ ] **Step 2: Run targeted grip tests**

Run:

```bash
pytest tests/test_right_hand_racket_grip.py -q
```

Expected: PASS.

- [ ] **Step 3: Run Overall tests**

Run:

```bash
pytest environment/overall_environment/tests/test_overall_environment.py -q
```

Expected: PASS.

- [ ] **Step 4: Run seed validation CLI**

Run:

```bash
python -m src.grip.validate_right_hand_racket_grip \
  --xml assets/right_hand_racket_grip_scene.xml \
  --reference outputs/right_hand_racket_grip/reference/right_hand_racket_grip_seed.json \
  --steps 1
```

Expected: JSON prints with `"finite": true`; full `acceptance_pass` may still be false until a trained grip controller exists.

- [ ] **Step 5: Commit docs**

```bash
git add docs/right_hand_racket_grip.md
git commit -m "docs: document right hand grip seed artifact"
```

- [ ] **Step 6: Final status check**

Run:

```bash
git status --short
git log --oneline -5
```

Expected: only unrelated pre-existing dirty files remain; new seed artifact commits appear in the recent log.


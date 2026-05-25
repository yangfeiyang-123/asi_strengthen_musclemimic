# RL Natural Shuttlecock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current natural shuttlecock design into a tested RL-ready MuJoCo asset with explicit cork impact site, aerodynamic diagnostics, racket rebound helper, randomizable parameters, and staged validation docs.

**Architecture:** Keep the existing single rigid shuttle body and visual feather geometry. `src/shuttlecock_aero.py` owns free-flight force computation and diagnostics, `src/shuttlecock_racket_impact.py` owns event-style rebound velocity helpers, `assets/shuttlecock_mujoco.xml` owns geometry/contact sites, and `params/shuttlecock_nominal.json` owns nominal values plus randomization ranges. Tests live inside `environment/shuttlecock/tests` and avoid importing the larger `musclemimic` package.

**Tech Stack:** Python 3, dataclasses, NumPy, stdlib `xml.etree.ElementTree`, JSON, pytest, optional MuJoCo Python package.

---

## File Structure

- Modify `assets/shuttlecock_mujoco.xml`: add `cork_contact_site` at the cork collision sphere center and keep feather/thread geoms visual-only.
- Modify `src/shuttlecock_aero.py`: add a pure aerodynamic computation helper and diagnostics dataclass, then make `apply_shuttlecock_aero` optionally return diagnostics.
- Create `src/shuttlecock_racket_impact.py`: pure rebound velocity math plus a helper that writes rebound linear velocity into a MuJoCo freejoint.
- Modify `params/shuttlecock_nominal.json`: add explicit `randomization` and `racket_impact` sections.
- Modify `validation_protocol.md`: expand into the three-stage validation protocol from the approved spec.
- Create `tests/test_shuttlecock_asset.py`: static XML and parameter tests.
- Create `tests/test_shuttlecock_aero.py`: pure aerodynamic force and diagnostics tests.
- Create `tests/test_shuttlecock_racket_impact.py`: pure event rebound and freejoint velocity setter tests.

---

### Task 1: Add Cork Contact Site and Asset Tests

**Files:**
- Modify: `assets/shuttlecock_mujoco.xml`
- Create: `tests/test_shuttlecock_asset.py`

- [ ] **Step 1: Write failing XML tests**

Create `tests/test_shuttlecock_asset.py`:

```python
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "assets" / "shuttlecock_mujoco.xml"
PARAMS = ROOT / "params" / "shuttlecock_nominal.json"


def _body(root: ET.Element, name: str) -> ET.Element:
    for body in root.iter("body"):
        if body.attrib.get("name") == name:
            return body
    raise AssertionError(f"body not found: {name}")


def _geom(body: ET.Element, name: str) -> ET.Element:
    for geom in body.iter("geom"):
        if geom.attrib.get("name") == name:
            return geom
    raise AssertionError(f"geom not found: {name}")


def _site(body: ET.Element, name: str) -> ET.Element:
    for site in body.iter("site"):
        if site.attrib.get("name") == name:
            return site
    raise AssertionError(f"site not found: {name}")


def _vec(value: str) -> tuple[float, ...]:
    return tuple(float(part) for part in value.split())


def test_cork_contact_site_matches_cork_collision_center():
    root = ET.parse(ASSET).getroot()
    shuttle = _body(root, "shuttle")

    cork = _geom(shuttle, "cork_collision")
    contact = _site(shuttle, "cork_contact_site")

    assert _vec(contact.attrib["pos"]) == _vec(cork.attrib["pos"])
    assert contact.attrib["size"] == "0.0020"


def test_feathers_and_threads_are_visual_only():
    root = ET.parse(ASSET).getroot()
    shuttle = _body(root, "shuttle")

    for geom in shuttle.iter("geom"):
        name = geom.attrib.get("name", "")
        if name.startswith(("feather_", "thread_")):
            assert geom.attrib.get("class") == "visual"


def test_nominal_params_include_randomization_and_impact_sections():
    data = json.loads(PARAMS.read_text(encoding="utf-8"))

    assert data["randomization"]["mass_kg"] == [0.00474, 0.0055]
    assert data["randomization"]["terminal_velocity_m_s"] == [6.5, 6.9]
    assert data["racket_impact"]["shuttle_contact_site_name"] == "cork_contact_site"
    assert data["racket_impact"]["event_restitution_normal_range"] == [0.45, 0.6]
```

- [ ] **Step 2: Run tests to verify the cork site and params assertions fail**

Run:

```bash
pytest tests/test_shuttlecock_asset.py -v
```

Expected: `test_cork_contact_site_matches_cork_collision_center` fails because `cork_contact_site` does not exist, and `test_nominal_params_include_randomization_and_impact_sections` fails because the new JSON sections do not exist.

- [ ] **Step 3: Add `cork_contact_site` to the MJCF**

In `assets/shuttlecock_mujoco.xml`, add this site immediately after `shuttle_nose`:

```xml
      <site name="cork_contact_site" pos="0 0 0.011154" size="0.0020" rgba="1 0.5 0 1"/>
```

Do not change feather or thread geoms in this task; they are already visual-only through `class="visual"`.

- [ ] **Step 4: Run the focused asset tests**

Run:

```bash
pytest tests/test_shuttlecock_asset.py::test_cork_contact_site_matches_cork_collision_center tests/test_shuttlecock_asset.py::test_feathers_and_threads_are_visual_only -v
```

Expected: both selected XML tests pass. The params test still fails until Task 4.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add assets/shuttlecock_mujoco.xml tests/test_shuttlecock_asset.py
git commit -m "feat: add shuttlecock cork contact site"
```

---

### Task 2: Add Aerodynamic Diagnostics

**Files:**
- Modify: `src/shuttlecock_aero.py`
- Create: `tests/test_shuttlecock_aero.py`

- [ ] **Step 1: Write failing pure aero tests**

Create `tests/test_shuttlecock_aero.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from src.shuttlecock_aero import (
    ShuttlecockAeroConfig,
    compute_shuttlecock_aero,
    equivalent_cd,
    expected_drag_constant,
)


def test_expected_drag_constant_and_equivalent_cd_match_nominal_design():
    k = expected_drag_constant()

    assert k == pytest.approx(0.0010819, rel=5e-4)
    assert equivalent_cd(k) == pytest.approx(0.532, rel=5e-3)


def test_drag_force_opposes_relative_velocity_and_reports_diagnostics():
    cfg = ShuttlecockAeroConfig()
    force, torque, cp, diag = compute_shuttlecock_aero(
        mass_kg=0.00519,
        gravity=np.array([0.0, 0.0, -9.81]),
        wind=np.zeros(3),
        v_world=np.array([10.0, 0.0, 0.0]),
        omega_world=np.zeros(3),
        nose_axis_world=np.array([1.0, 0.0, 0.0]),
        com_world=np.array([0.0, 0.0, 1.0]),
        cfg=cfg,
    )

    assert np.dot(force, np.array([10.0, 0.0, 0.0])) < 0.0
    assert diag.speed_m_s == pytest.approx(10.0)
    assert diag.angle_of_attack_rad == pytest.approx(0.0)
    assert diag.force_clipped is False
    assert diag.torque_clipped is False
    assert cp == pytest.approx(np.array([-0.035, 0.0, 1.0]))
    assert torque == pytest.approx(np.zeros(3))


def test_sideways_flight_increases_effective_drag_constant():
    cfg = ShuttlecockAeroConfig(angle_drag_gain=0.5)
    aligned = compute_shuttlecock_aero(
        mass_kg=0.00519,
        gravity=np.array([0.0, 0.0, -9.81]),
        wind=np.zeros(3),
        v_world=np.array([12.0, 0.0, 0.0]),
        omega_world=np.zeros(3),
        nose_axis_world=np.array([1.0, 0.0, 0.0]),
        com_world=np.zeros(3),
        cfg=cfg,
    )[3]
    sideways = compute_shuttlecock_aero(
        mass_kg=0.00519,
        gravity=np.array([0.0, 0.0, -9.81]),
        wind=np.zeros(3),
        v_world=np.array([12.0, 0.0, 0.0]),
        omega_world=np.zeros(3),
        nose_axis_world=np.array([0.0, 0.0, 1.0]),
        com_world=np.zeros(3),
        cfg=cfg,
    )[3]

    assert sideways.effective_drag_constant_kg_m > aligned.effective_drag_constant_kg_m
    assert sideways.angle_of_attack_rad == pytest.approx(np.pi / 2.0)


def test_force_and_torque_clipping_are_reported():
    cfg = ShuttlecockAeroConfig(max_force_n=0.01, max_torque_nm=0.001)
    force, torque, _cp, diag = compute_shuttlecock_aero(
        mass_kg=0.00519,
        gravity=np.array([0.0, 0.0, -9.81]),
        wind=np.zeros(3),
        v_world=np.array([100.0, 0.0, 0.0]),
        omega_world=np.array([100.0, 0.0, 0.0]),
        nose_axis_world=np.array([1.0, 0.0, 0.0]),
        com_world=np.zeros(3),
        cfg=cfg,
    )

    assert np.linalg.norm(force) == pytest.approx(0.01)
    assert np.linalg.norm(torque) == pytest.approx(0.001)
    assert diag.force_clipped is True
    assert diag.torque_clipped is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_shuttlecock_aero.py -v
```

Expected: import fails with `ImportError: cannot import name 'compute_shuttlecock_aero'`.

- [ ] **Step 3: Add diagnostics dataclass and pure computation helper**

In `src/shuttlecock_aero.py`, add this dataclass below `ShuttlecockAeroConfig`:

```python
@dataclass(frozen=True)
class ShuttlecockAeroDiagnostics:
    speed_m_s: float
    angle_of_attack_rad: float
    drag_constant_kg_m: float
    effective_drag_constant_kg_m: float
    force_world_n: np.ndarray
    damping_torque_world_nm: np.ndarray
    center_of_pressure_world_m: np.ndarray
    force_clipped: bool
    torque_clipped: bool
```

Replace `_clip_norm` with this implementation:

```python
def _clip_norm_with_flag(vec: np.ndarray, max_norm: float) -> tuple[np.ndarray, bool]:
    norm = float(np.linalg.norm(vec))
    if norm > max_norm > 0:
        return vec * (max_norm / norm), True
    return vec, False
```

Add this pure helper above `apply_shuttlecock_aero`:

```python
def compute_shuttlecock_aero(
    *,
    mass_kg: float,
    gravity: np.ndarray,
    wind: np.ndarray,
    v_world: np.ndarray,
    omega_world: np.ndarray,
    nose_axis_world: np.ndarray,
    com_world: np.ndarray,
    cfg: ShuttlecockAeroConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ShuttlecockAeroDiagnostics]:
    """Compute force, damping torque, center of pressure, and diagnostics."""
    v_rel = np.asarray(v_world, dtype=float) - np.asarray(wind, dtype=float)
    speed = float(np.linalg.norm(v_rel))
    nose_axis = np.asarray(nose_axis_world, dtype=float)
    nose_norm = float(np.linalg.norm(nose_axis))
    if nose_norm < 1e-12:
        raise ValueError("nose_axis_world must be non-zero")
    nose_axis = nose_axis / nose_norm

    gravity_norm = float(np.linalg.norm(gravity))
    if gravity_norm <= 0.0:
        gravity_norm = 9.81

    k = float(mass_kg) * gravity_norm / (cfg.terminal_velocity_m_s**2)
    cp_world = np.asarray(com_world, dtype=float) - cfg.center_of_pressure_offset_m * nose_axis

    if speed < 1e-8:
        zero = np.zeros(3, dtype=float)
        diag = ShuttlecockAeroDiagnostics(
            speed_m_s=0.0,
            angle_of_attack_rad=0.0,
            drag_constant_kg_m=k,
            effective_drag_constant_kg_m=k,
            force_world_n=zero.copy(),
            damping_torque_world_nm=zero.copy(),
            center_of_pressure_world_m=cp_world,
            force_clipped=False,
            torque_clipped=False,
        )
        return zero.copy(), zero.copy(), cp_world, diag

    v_hat = v_rel / speed
    cos_alpha = float(np.clip(np.dot(nose_axis, v_hat), -1.0, 1.0))
    angle_of_attack = float(np.arccos(cos_alpha))
    sin2_alpha = max(0.0, 1.0 - cos_alpha * cos_alpha)
    k_eff = k * (1.0 + cfg.angle_drag_gain * sin2_alpha)

    force_world = -k_eff * speed * v_rel
    force_world, force_clipped = _clip_norm_with_flag(force_world, cfg.max_force_n)

    damping_torque_world = -cfg.angular_damping_nms_per_rad * np.asarray(omega_world, dtype=float)
    damping_torque_world, torque_clipped = _clip_norm_with_flag(
        damping_torque_world,
        cfg.max_torque_nm,
    )

    diag = ShuttlecockAeroDiagnostics(
        speed_m_s=speed,
        angle_of_attack_rad=angle_of_attack,
        drag_constant_kg_m=k,
        effective_drag_constant_kg_m=k_eff,
        force_world_n=force_world.copy(),
        damping_torque_world_nm=damping_torque_world.copy(),
        center_of_pressure_world_m=cp_world.copy(),
        force_clipped=force_clipped,
        torque_clipped=torque_clipped,
    )
    return force_world, damping_torque_world, cp_world, diag
```

- [ ] **Step 4: Update `apply_shuttlecock_aero` to use the helper and return diagnostics**

Change the function signature and body tail in `src/shuttlecock_aero.py`:

```python
def apply_shuttlecock_aero(
    model,
    data,
    cfg: ShuttlecockAeroConfig | None = None,
) -> ShuttlecockAeroDiagnostics | None:
```

Replace the internal force computation from `v_rel = ...` through `damping_torque_world = ...` with:

```python
    wind = np.array(model.opt.wind, dtype=float) if cfg.use_model_wind else np.zeros(3)
    rot = np.array(data.xmat[body_id], dtype=float).reshape(3, 3)
    nose_axis_world = rot @ np.array([0.0, 0.0, 1.0])
    mass = float(model.body_mass[body_id])
    com_world = np.array(data.xipos[body_id], dtype=float)

    force_world, damping_torque_world, cp_world, diag = compute_shuttlecock_aero(
        mass_kg=mass,
        gravity=np.array(model.opt.gravity, dtype=float),
        wind=wind,
        v_world=v_world,
        omega_world=omega_world,
        nose_axis_world=nose_axis_world,
        com_world=com_world,
        cfg=cfg,
    )
    if diag.speed_m_s < 1e-8:
        return diag
```

At the end of the function, after `mj_applyFT(...)`, add:

```python
    return diag
```

- [ ] **Step 5: Run the focused aero tests**

Run:

```bash
pytest tests/test_shuttlecock_aero.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 2**

Run:

```bash
git add src/shuttlecock_aero.py tests/test_shuttlecock_aero.py
git commit -m "feat: add shuttlecock aero diagnostics"
```

---

### Task 3: Add Racket Impact Event Helpers

**Files:**
- Create: `src/shuttlecock_racket_impact.py`
- Create: `tests/test_shuttlecock_racket_impact.py`

- [ ] **Step 1: Write failing event rebound tests**

Create `tests/test_shuttlecock_racket_impact.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from src.shuttlecock_racket_impact import (
    ShuttlecockImpactConfig,
    compute_event_rebound_velocity,
    set_freejoint_linear_velocity,
    should_apply_event_rebound,
)


def test_compute_event_rebound_velocity_reflects_closing_normal_component():
    cfg = ShuttlecockImpactConfig(event_restitution_normal=0.5, event_tangential_velocity_scale=0.8)

    result = compute_event_rebound_velocity(
        shuttle_velocity_world=np.array([0.0, 0.0, -10.0]),
        racket_surface_velocity_world=np.array([0.0, 0.0, 2.0]),
        normal_world=np.array([0.0, 0.0, 1.0]),
        cfg=cfg,
    )

    assert result == pytest.approx(np.array([0.0, 0.0, 8.0]))


def test_compute_event_rebound_velocity_preserves_scaled_tangential_component():
    cfg = ShuttlecockImpactConfig(event_restitution_normal=0.5, event_tangential_velocity_scale=0.8)

    result = compute_event_rebound_velocity(
        shuttle_velocity_world=np.array([3.0, 0.0, -10.0]),
        racket_surface_velocity_world=np.array([1.0, 0.0, 2.0]),
        normal_world=np.array([0.0, 0.0, 1.0]),
        cfg=cfg,
    )

    assert result == pytest.approx(np.array([2.6, 0.0, 8.0]))


def test_compute_event_rebound_velocity_returns_input_when_already_separating():
    cfg = ShuttlecockImpactConfig()
    velocity = np.array([0.0, 0.0, 4.0])

    result = compute_event_rebound_velocity(
        shuttle_velocity_world=velocity,
        racket_surface_velocity_world=np.zeros(3),
        normal_world=np.array([0.0, 0.0, 1.0]),
        cfg=cfg,
    )

    assert result == pytest.approx(velocity)


def test_should_apply_event_rebound_requires_active_fast_closing_contact():
    cfg = ShuttlecockImpactConfig(min_speed_for_event_m_s=5.0)

    assert should_apply_event_rebound({"active": True, "relative_normal_velocity": -5.1}, cfg) is True
    assert should_apply_event_rebound({"active": True, "relative_normal_velocity": -4.9}, cfg) is False
    assert should_apply_event_rebound({"active": False, "relative_normal_velocity": -10.0}, cfg) is False


@dataclass
class FakeModel:
    body_name_to_id: dict[str, int]
    body_jntadr: np.ndarray
    jnt_dofadr: np.ndarray
    jnt_type: np.ndarray


@dataclass
class FakeData:
    qvel: np.ndarray


def test_set_freejoint_linear_velocity_writes_first_three_freejoint_dofs():
    model = FakeModel(
        body_name_to_id={"shuttle": 0},
        body_jntadr=np.array([0]),
        jnt_dofadr=np.array([2]),
        jnt_type=np.array([0]),
    )
    data = FakeData(qvel=np.zeros(8))

    set_freejoint_linear_velocity(
        model,
        data,
        body_name="shuttle",
        velocity_world=np.array([1.0, 2.0, 3.0]),
        free_joint_type_value=0,
    )

    assert data.qvel.tolist() == [0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_shuttlecock_racket_impact.py -v
```

Expected: import fails because `src/shuttlecock_racket_impact.py` does not exist.

- [ ] **Step 3: Create the impact helper module**

Create `src/shuttlecock_racket_impact.py`:

```python
"""Event-style racket impact helpers for the MuJoCo shuttlecock."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:
    import mujoco
except Exception as exc:  # pragma: no cover
    mujoco = None
    _MUJOCO_IMPORT_ERROR = exc
else:
    _MUJOCO_IMPORT_ERROR = None


@dataclass(frozen=True)
class ShuttlecockImpactConfig:
    min_speed_for_event_m_s: float = 5.0
    event_restitution_normal: float = 0.50
    event_tangential_velocity_scale: float = 0.85
    max_rebound_speed_m_s: float = 100.0


def _unit(vec: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        raise ValueError(f"{name} must be non-zero")
    return arr / norm


def compute_event_rebound_velocity(
    *,
    shuttle_velocity_world: np.ndarray,
    racket_surface_velocity_world: np.ndarray,
    normal_world: np.ndarray,
    cfg: ShuttlecockImpactConfig = ShuttlecockImpactConfig(),
) -> np.ndarray:
    """Return post-impact shuttle velocity from a short-duration string-bed impact."""
    n = _unit(normal_world, "normal_world")
    shuttle_v = np.asarray(shuttle_velocity_world, dtype=float)
    racket_v = np.asarray(racket_surface_velocity_world, dtype=float)
    v_rel = shuttle_v - racket_v
    v_n = float(np.dot(v_rel, n))
    if v_n >= 0.0:
        return shuttle_v.copy()

    v_t = v_rel - v_n * n
    v_rel_after = (-cfg.event_restitution_normal * v_n) * n + cfg.event_tangential_velocity_scale * v_t
    rebound = racket_v + v_rel_after
    speed = float(np.linalg.norm(rebound))
    if speed > cfg.max_rebound_speed_m_s > 0.0:
        rebound = rebound * (cfg.max_rebound_speed_m_s / speed)
    return rebound


def should_apply_event_rebound(
    contact_info: dict[str, object],
    cfg: ShuttlecockImpactConfig = ShuttlecockImpactConfig(),
) -> bool:
    """Return True when active contact is closing too fast for soft contact alone."""
    if not bool(contact_info.get("active", False)):
        return False
    relative_normal_velocity = float(contact_info.get("relative_normal_velocity", 0.0))
    return relative_normal_velocity < -cfg.min_speed_for_event_m_s


def _body_id(model, body_name: str) -> int:
    if hasattr(model, "body_name_to_id"):
        return int(model.body_name_to_id[body_name])
    if mujoco is None:  # pragma: no cover
        raise RuntimeError(f"mujoco Python package is not available: {_MUJOCO_IMPORT_ERROR}")
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body not found: {body_name!r}")
    return int(body_id)


def set_freejoint_linear_velocity(
    model,
    data,
    *,
    body_name: str,
    velocity_world: np.ndarray,
    free_joint_type_value: int | None = None,
) -> None:
    """Set the linear velocity part of a body's freejoint qvel.

    MuJoCo freejoint qvel stores translational velocity in the first three dofs
    of the joint and angular velocity in the next three dofs.
    """
    body_id = _body_id(model, body_name)
    joint_id = int(model.body_jntadr[body_id])
    if joint_id < 0:
        raise ValueError(f"Body has no joint: {body_name!r}")

    if free_joint_type_value is None:
        if mujoco is None:  # pragma: no cover
            raise RuntimeError(f"mujoco Python package is not available: {_MUJOCO_IMPORT_ERROR}")
        free_joint_type_value = int(mujoco.mjtJoint.mjJNT_FREE)

    if int(model.jnt_type[joint_id]) != int(free_joint_type_value):
        raise ValueError(f"Body joint is not a freejoint: {body_name!r}")

    dof_adr = int(model.jnt_dofadr[joint_id])
    data.qvel[dof_adr : dof_adr + 3] = np.asarray(velocity_world, dtype=float)
```

- [ ] **Step 4: Run the impact helper tests**

Run:

```bash
pytest tests/test_shuttlecock_racket_impact.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add src/shuttlecock_racket_impact.py tests/test_shuttlecock_racket_impact.py
git commit -m "feat: add shuttlecock racket impact helpers"
```

---

### Task 4: Add Randomization and Impact Parameters

**Files:**
- Modify: `params/shuttlecock_nominal.json`
- Modify: `tests/test_shuttlecock_asset.py`

- [ ] **Step 1: Run the params test to verify it still fails**

Run:

```bash
pytest tests/test_shuttlecock_asset.py::test_nominal_params_include_randomization_and_impact_sections -v
```

Expected: fail with `KeyError: 'randomization'`.

- [ ] **Step 2: Add randomization and impact sections to the JSON**

In `params/shuttlecock_nominal.json`, add this block after the existing `aerodynamics` object and before `mujoco`:

```json
  "randomization": {
    "mass_kg": [0.00474, 0.0055],
    "terminal_velocity_m_s": [6.5, 6.9],
    "center_of_pressure_offset_m": [0.025, 0.045],
    "angle_drag_gain": [0.0, 0.5],
    "angular_damping_nms_per_rad": [0.000005, 0.00008],
    "event_restitution_normal": [0.45, 0.6],
    "event_tangential_velocity_scale": [0.8, 0.9],
    "stringbed_center_stiffness_n_per_m": [8000.0, 12000.0],
    "wind_m_s": [-0.5, 0.5]
  },
  "racket_impact": {
    "racket_body_name": "racket",
    "shuttle_body_name": "shuttle",
    "shuttle_contact_site_name": "cork_contact_site",
    "stringbed_center_stiffness_n_per_m": 9600.0,
    "normal_damping_n_s_per_m": 3.0,
    "tangential_damping_n_s_per_m": 0.15,
    "tangential_mu": 0.08,
    "min_speed_for_event_m_s": 5.0,
    "event_restitution_normal": 0.5,
    "event_restitution_normal_range": [0.45, 0.6],
    "event_tangential_velocity_scale": 0.85,
    "event_tangential_velocity_scale_range": [0.8, 0.9],
    "max_rebound_speed_m_s": 100.0
  },
```

Keep valid JSON commas. Do not remove the existing `aerodynamics` or `mujoco` sections.

- [ ] **Step 3: Run all asset tests**

Run:

```bash
pytest tests/test_shuttlecock_asset.py -v
```

Expected: all asset and params tests pass.

- [ ] **Step 4: Commit Task 4**

Run:

```bash
git add params/shuttlecock_nominal.json tests/test_shuttlecock_asset.py
git commit -m "feat: add shuttlecock randomization parameters"
```

---

### Task 5: Expand Validation Protocol

**Files:**
- Modify: `validation_protocol.md`

- [ ] **Step 1: Replace the protocol with staged validation text**

Replace `validation_protocol.md` with:

```markdown
# Shuttlecock MuJoCo Validation Protocol

Run these tests after integrating `assets/shuttlecock_mujoco.xml`,
`src/shuttlecock_aero.py`, and the racket string-bed proxy.

## 1. Geometry and Contact Acceptance

- Feather count: 16.
- Feather length from base-top plane to tip: `0.065 m`; allowed rules range is `0.062-0.070 m`.
- Tip circle diameter: `0.065 m`; allowed rules range is `0.058-0.068 m`.
- Base/cork diameter: `0.027 m`; allowed rules range is `0.025-0.028 m`.
- Mass: `0.00519 kg`; allowed rules range is `0.00474-0.00550 kg`.
- `cork_collision` is the primary collision geom.
- `cork_contact_site` exists and is colocated with the cork collision sphere center.
- Feather and thread geoms are visual-only in the first RL version.

## 2. Stage 1: Free Flight

Initialize the shuttle at `z=18 m`, low or zero velocity, random orientation,
and enable `apply_shuttlecock_aero`.

Acceptance:

- Measured vertical speed approaches `6.5-6.9 m/s` before ground contact.
- Default `terminal_velocity_m_s=6.86` produces approximately `6.86 m/s` terminal speed.
- A `30 m/s`, `30 degree` launch lands around the current `9-10 m` horizontal scale.
- A launch with the nose initially `60-120 degrees` away from velocity aligns nose-first within about `0.05-0.20 s` for speeds above `15 m/s`.
- Sideways or reversed flight decays faster than nose-forward flight.
- Aerodynamic force opposes relative velocity.
- Pressure-center torque tends to align local `+Z` with relative velocity.

## 3. Stage 2: Ordinary Racket Impact

Use the racket string-bed proxy with:

```python
apply_stringbed_force(
    model,
    data,
    racket_body_name="racket",
    shuttle_body_name="shuttle",
    shuttle_contact_site_name="cork_contact_site",
)
```

Acceptance:

- Medium racket speeds produce outgoing velocity consistent with racket surface velocity and string-bed normal.
- Sweet-spot impacts are stable and repeatable.
- Edge impacts are less stable than center impacts but remain numerically bounded.
- No high-speed tunneling at `timestep <= 0.0005 s`.
- The impact path uses `cork_contact_site`, not the shuttle COM fallback.

## 4. Stage 3: High-Intensity Racket Impact

Use event rebound when active contact has high closing normal speed.

Acceptance:

- Fast smash or drive-like contacts do not miss the shuttle.
- Event rebound triggers only for active contact with closing normal speed above `min_speed_for_event_m_s`.
- Rebound velocity is bounded by `max_rebound_speed_m_s`.
- Aerodynamic torque restores nose-forward flight after impact.
- Force, torque, and event clipping diagnostics are logged during validation.

## 5. Parameter Randomization Smoke Test

Sample the configured randomization ranges:

- `mass_kg`
- `terminal_velocity_m_s`
- `center_of_pressure_offset_m`
- `angle_drag_gain`
- `angular_damping_nms_per_rad`
- `event_restitution_normal`
- `event_tangential_velocity_scale`
- `stringbed_center_stiffness_n_per_m`
- `wind_m_s`

Acceptance:

- The nominal model passes before randomization is widened.
- Randomized ordinary impacts do not produce NaN, infinite velocity, or unbounded force.
- Training should start with narrow randomization and widen only after nominal validation passes.
```

- [ ] **Step 2: Verify references exist**

Run:

```bash
rg -n "cork_contact_site|Stage 1|Stage 2|Stage 3|event rebound|randomization" validation_protocol.md
```

Expected: output includes all listed terms.

- [ ] **Step 3: Commit Task 5**

Run:

```bash
git add validation_protocol.md
git commit -m "docs: expand shuttlecock validation protocol"
```

---

### Task 6: End-to-End Verification

**Files:**
- No planned source edits.

- [ ] **Step 1: Run all local shuttlecock tests**

Run:

```bash
pytest tests/test_shuttlecock_asset.py tests/test_shuttlecock_aero.py tests/test_shuttlecock_racket_impact.py -v
```

Expected: all tests pass.

- [ ] **Step 2: Verify all Python files import without the larger project**

Run:

```bash
python - <<'PY'
from src.shuttlecock_aero import ShuttlecockAeroConfig, compute_shuttlecock_aero
from src.shuttlecock_racket_impact import ShuttlecockImpactConfig, compute_event_rebound_velocity
print(ShuttlecockAeroConfig())
print(ShuttlecockImpactConfig())
print(compute_event_rebound_velocity)
PY
```

Expected: command exits `0` and prints both config objects plus the function object.

- [ ] **Step 3: Validate JSON and XML parseability**

Run:

```bash
python - <<'PY'
import json
import xml.etree.ElementTree as ET
json.load(open("params/shuttlecock_nominal.json", encoding="utf-8"))
ET.parse("assets/shuttlecock_mujoco.xml")
print("ok")
PY
```

Expected: prints `ok`.

- [ ] **Step 4: Check git status**

Run:

```bash
git status --short
```

Expected: only unrelated pre-existing untracked files remain.

---

## Self-Review Notes

- Spec coverage: Task 1 implements the explicit cork contact site and confirms feathers remain visual-only. Task 2 implements aerodynamic diagnostics and force/torque clipping visibility. Task 3 implements the single event rebound helper path. Task 4 implements randomization and racket-impact parameter exposure. Task 5 implements staged validation documentation. Task 6 verifies local tests, imports, JSON, and XML.
- Scope check: This plan is a single implementation unit because it only modifies the shuttlecock package and a new local test suite. It does not modify the humanoid RL stack or the racket package.
- Type consistency: `cork_contact_site`, `ShuttlecockAeroConfig`, `ShuttlecockAeroDiagnostics`, `ShuttlecockImpactConfig`, `compute_event_rebound_velocity`, and `set_freejoint_linear_velocity` are named consistently across tasks.

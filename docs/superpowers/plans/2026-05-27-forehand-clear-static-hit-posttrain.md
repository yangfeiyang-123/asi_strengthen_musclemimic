# Forehand Clear Static-Hit PostTrain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the staged ForehandClear static-shuttle post-train pipeline: impact-target extraction, layered body/grip control, static shuttle freeze/release, flight diagnostics, grip swing-disturbance training hooks, and a reproducible experiment spec.

**Architecture:** Implement small pure-Python modules first, then connect them to the existing Overall and grip environments. Keep the first implementation diagnostic-first: physics chain validation and grip-stabilizer readiness are required before training the composed static-hit task.

**Tech Stack:** Python 3, NumPy, MuJoCo Python, PyTorch PPO grip trainer, Pytest, YAML experiment specs.

---

## File Structure

Create focused files:

- `environment/overall_environment/src/impact_target.py`
  - Pure helpers for extracting and regularizing a root-local ForehandClear impact target.
- `environment/overall_environment/src/layered_control.py`
  - Pure actuator-name router for merging body and grip policy actions.
- `environment/overall_environment/src/static_forehand_clear_env.py`
  - Overall task wrapper, state machine, freeze/release handling, impact diagnostics, and flight diagnostics.
- `environment/overall_environment/tests/test_impact_target.py`
  - Pure unit tests for reference extraction and body-scale regularization.
- `environment/overall_environment/tests/test_layered_control.py`
  - Pure unit tests for actuator routing and conflict detection.
- `environment/overall_environment/tests/test_static_forehand_clear_env.py`
  - Unit tests for state transitions, shuttle freeze behavior, release guard, and flight classification.
- `tests/test_right_hand_racket_grip.py`
  - Extend existing grip tests for swing-disturbance config parsing and deterministic perturbation helper behavior.
- `src/grip/right_hand_racket_grip_env.py`
  - Add optional swing-disturbance helper hooks that are off by default.
- `src/grip/train_right_hand_racket_grip_policy.py`
  - Record whether training used swing disturbance and expose stable checkpoint metadata.
- `configs/right_hand_racket_grip_training.yaml`
  - Add explicit swing-disturbance curriculum fields with zero/default-off behavior.
- `BadmintonMimic/experiments/posttrain/forehand_clear_static_hit_v1.yaml`
  - Reproducible experiment spec for the staged static-hit task.
- `docs/forehand_clear_static_hit_posttrain.md`
  - Short operational doc with commands and acceptance sequence.

Keep the initial environment implementation CPU MuJoCo oriented. Do not integrate this directly into the MJX fullbody PPO runner until the pure tests and Overall smoke path pass.

---

### Task 1: Add Impact Target Pure Utility

**Files:**
- Create: `environment/overall_environment/src/impact_target.py`
- Test: `environment/overall_environment/tests/test_impact_target.py`

- [ ] **Step 1: Write failing tests for virtual racket-head impact extraction**

Create `environment/overall_environment/tests/test_impact_target.py` with:

```python
from __future__ import annotations

import numpy as np
import pytest

from environment.overall_environment.src.impact_target import (
    BodyScale,
    ImpactTargetConfig,
    extract_impact_target_from_sites,
    regularize_impact_target,
)


def test_extract_impact_target_prefers_peak_virtual_racket_speed():
    right_hand_pos = np.array(
        [
            [0.20, 0.20, 1.40],
            [0.25, 0.25, 1.55],
            [0.35, 0.32, 1.75],
            [0.48, 0.36, 1.88],
            [0.55, 0.34, 1.82],
        ],
        dtype=float,
    )
    root_pos = np.zeros_like(right_hand_pos)
    forward_axis = np.tile(np.array([1.0, 0.0, 0.0]), (right_hand_pos.shape[0], 1))
    right_axis = np.tile(np.array([0.0, 1.0, 0.0]), (right_hand_pos.shape[0], 1))

    target = extract_impact_target_from_sites(
        right_hand_pos=right_hand_pos,
        root_pos=root_pos,
        forward_axis=forward_axis,
        right_axis=right_axis,
        dt=0.01,
        racket_length_m=0.67,
    )

    assert target.impact_frame == 3
    assert 0.0 < target.impact_phase < 1.0
    assert target.position_root_local[0] > 0.9
    assert target.position_root_local[1] > 0.30
    assert target.position_root_local[2] == pytest.approx(1.88)
    np.testing.assert_allclose(np.linalg.norm(target.racket_head_velocity_dir), 1.0)


def test_regularize_impact_target_clamps_to_forehand_comfort_zone():
    raw = np.array([0.05, -0.20, 2.80], dtype=float)
    scale = BodyScale(
        shoulder_height_m=1.42,
        arm_reach_up_m=0.72,
        racket_effective_length_m=0.52,
    )
    cfg = ImpactTargetConfig(
        min_forward_offset_m=0.28,
        max_forward_offset_m=0.85,
        min_racket_side_offset_m=0.18,
        max_racket_side_offset_m=0.65,
        reach_alpha=0.78,
        racket_beta=0.82,
        min_height_margin_m=-0.08,
        max_height_margin_m=0.06,
    )

    result = regularize_impact_target(raw, scale, cfg)

    assert result[0] == pytest.approx(0.28)
    assert result[1] == pytest.approx(0.18)
    expected_height = 1.42 + 0.72 * 0.78 + 0.52 * 0.82 + 0.06
    assert result[2] == pytest.approx(expected_height)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest environment/overall_environment/tests/test_impact_target.py -q
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError` because `impact_target.py` does not exist.

- [ ] **Step 3: Implement impact target utility**

Create `environment/overall_environment/src/impact_target.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BodyScale:
    shoulder_height_m: float
    arm_reach_up_m: float
    racket_effective_length_m: float


@dataclass(frozen=True)
class ImpactTargetConfig:
    min_forward_offset_m: float = 0.28
    max_forward_offset_m: float = 0.90
    min_racket_side_offset_m: float = 0.16
    max_racket_side_offset_m: float = 0.70
    reach_alpha: float = 0.78
    racket_beta: float = 0.82
    min_height_margin_m: float = -0.08
    max_height_margin_m: float = 0.08


@dataclass(frozen=True)
class ImpactTarget:
    impact_frame: int
    impact_phase: float
    position_root_local: np.ndarray
    racket_head_velocity_dir: np.ndarray
    racket_normal_hint: np.ndarray


def _as_array(name: str, value: np.ndarray, columns: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f"{name} must have shape (N, {columns}), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _normalize(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return np.asarray(fallback, dtype=float).copy()
    return np.asarray(vec, dtype=float) / norm


def extract_impact_target_from_sites(
    *,
    right_hand_pos: np.ndarray,
    root_pos: np.ndarray,
    forward_axis: np.ndarray,
    right_axis: np.ndarray,
    dt: float,
    racket_length_m: float,
) -> ImpactTarget:
    right_hand_pos = _as_array("right_hand_pos", right_hand_pos, 3)
    root_pos = _as_array("root_pos", root_pos, 3)
    forward_axis = _as_array("forward_axis", forward_axis, 3)
    right_axis = _as_array("right_axis", right_axis, 3)
    if right_hand_pos.shape[0] < 3:
        raise ValueError("at least three frames are required")
    if root_pos.shape != right_hand_pos.shape:
        raise ValueError("root_pos must match right_hand_pos shape")
    if forward_axis.shape != right_hand_pos.shape:
        raise ValueError("forward_axis must match right_hand_pos shape")
    if right_axis.shape != right_hand_pos.shape:
        raise ValueError("right_axis must match right_hand_pos shape")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if racket_length_m <= 0.0:
        raise ValueError("racket_length_m must be positive")

    forward_unit = np.vstack([_normalize(v, np.array([1.0, 0.0, 0.0])) for v in forward_axis])
    right_unit = np.vstack([_normalize(v, np.array([0.0, 1.0, 0.0])) for v in right_axis])
    virtual_head = right_hand_pos + racket_length_m * forward_unit
    velocity = np.gradient(virtual_head, dt, axis=0)
    speed = np.linalg.norm(velocity, axis=1)

    rel = virtual_head - root_pos
    forward_offset = np.einsum("ij,ij->i", rel, forward_unit)
    side_offset = np.einsum("ij,ij->i", rel, right_unit)
    candidate = (forward_offset > 0.0) & (side_offset > 0.0)
    if np.any(candidate):
        masked_speed = np.where(candidate, speed, -np.inf)
        impact_frame = int(np.argmax(masked_speed))
    else:
        impact_frame = int(np.argmax(speed))

    position_root_local = np.array(
        [
            forward_offset[impact_frame],
            side_offset[impact_frame],
            rel[impact_frame, 2],
        ],
        dtype=float,
    )
    phase_denominator = max(1, right_hand_pos.shape[0] - 1)
    impact_phase = float(impact_frame / phase_denominator)
    velocity_dir = _normalize(velocity[impact_frame], forward_unit[impact_frame])
    racket_normal_hint = _normalize(np.cross(velocity_dir, right_unit[impact_frame]), np.array([0.0, 0.0, 1.0]))
    return ImpactTarget(
        impact_frame=impact_frame,
        impact_phase=impact_phase,
        position_root_local=position_root_local,
        racket_head_velocity_dir=velocity_dir,
        racket_normal_hint=racket_normal_hint,
    )


def regularize_impact_target(
    position_root_local: np.ndarray,
    body_scale: BodyScale,
    config: ImpactTargetConfig = ImpactTargetConfig(),
) -> np.ndarray:
    position = np.asarray(position_root_local, dtype=float)
    if position.shape != (3,):
        raise ValueError(f"position_root_local must have shape (3,), got {position.shape}")
    if not np.isfinite(position).all():
        raise ValueError("position_root_local contains non-finite values")

    nominal_height = (
        body_scale.shoulder_height_m
        + body_scale.arm_reach_up_m * config.reach_alpha
        + body_scale.racket_effective_length_m * config.racket_beta
    )
    min_height = nominal_height + config.min_height_margin_m
    max_height = nominal_height + config.max_height_margin_m

    return np.array(
        [
            np.clip(position[0], config.min_forward_offset_m, config.max_forward_offset_m),
            np.clip(position[1], config.min_racket_side_offset_m, config.max_racket_side_offset_m),
            np.clip(position[2], min_height, max_height),
        ],
        dtype=float,
    )
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
pytest environment/overall_environment/tests/test_impact_target.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add environment/overall_environment/src/impact_target.py environment/overall_environment/tests/test_impact_target.py
git commit -m "feat: add forehand clear impact target helper"
```

---

### Task 2: Add Layered Actuator Router

**Files:**
- Create: `environment/overall_environment/src/layered_control.py`
- Test: `environment/overall_environment/tests/test_layered_control.py`

- [ ] **Step 1: Write failing tests for actuator routing**

Create `environment/overall_environment/tests/test_layered_control.py` with:

```python
from __future__ import annotations

import numpy as np
import pytest

from environment.overall_environment.src.layered_control import LayeredActuatorRouter


def test_router_merges_body_and_grip_actions_by_name():
    router = LayeredActuatorRouter(
        all_actuator_names=["hip", "shoulder", "FDS2", "FDP2"],
        body_actuator_names=["hip", "shoulder"],
        grip_actuator_names=["FDS2", "FDP2"],
    )

    merged = router.merge(
        body_action=np.array([0.1, 0.2]),
        grip_action=np.array([0.7, 0.8]),
    )

    np.testing.assert_allclose(merged, np.array([0.1, 0.2, 0.7, 0.8]))
    assert router.source_labels() == ["body", "body", "grip", "grip"]


def test_router_rejects_overlapping_actuator_ownership():
    with pytest.raises(ValueError, match="owned by both"):
        LayeredActuatorRouter(
            all_actuator_names=["hip", "FDS2"],
            body_actuator_names=["hip", "FDS2"],
            grip_actuator_names=["FDS2"],
        )


def test_router_validates_action_shapes():
    router = LayeredActuatorRouter(
        all_actuator_names=["hip", "FDS2"],
        body_actuator_names=["hip"],
        grip_actuator_names=["FDS2"],
    )

    with pytest.raises(ValueError, match="body_action"):
        router.merge(body_action=np.array([0.1, 0.2]), grip_action=np.array([0.3]))
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest environment/overall_environment/tests/test_layered_control.py -q
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError` because `layered_control.py` does not exist.

- [ ] **Step 3: Implement router**

Create `environment/overall_environment/src/layered_control.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LayeredActuatorRouter:
    all_actuator_names: list[str]
    body_actuator_names: list[str]
    grip_actuator_names: list[str]

    def __post_init__(self) -> None:
        all_names = list(self.all_actuator_names)
        if len(set(all_names)) != len(all_names):
            raise ValueError("all_actuator_names contains duplicates")
        body = set(self.body_actuator_names)
        grip = set(self.grip_actuator_names)
        overlap = sorted(body.intersection(grip))
        if overlap:
            raise ValueError(f"actuator(s) owned by both body and grip: {overlap}")
        unknown = sorted(body.union(grip).difference(all_names))
        if unknown:
            raise ValueError(f"owned actuator(s) missing from all_actuator_names: {unknown}")

    @property
    def body_size(self) -> int:
        return len(self.body_actuator_names)

    @property
    def grip_size(self) -> int:
        return len(self.grip_actuator_names)

    def source_labels(self) -> list[str]:
        body = set(self.body_actuator_names)
        grip = set(self.grip_actuator_names)
        labels: list[str] = []
        for name in self.all_actuator_names:
            if name in body:
                labels.append("body")
            elif name in grip:
                labels.append("grip")
            else:
                labels.append("zero")
        return labels

    def merge(self, *, body_action: np.ndarray, grip_action: np.ndarray) -> np.ndarray:
        body_action = np.asarray(body_action, dtype=float)
        grip_action = np.asarray(grip_action, dtype=float)
        if body_action.shape != (self.body_size,):
            raise ValueError(f"body_action must have shape ({self.body_size},), got {body_action.shape}")
        if grip_action.shape != (self.grip_size,):
            raise ValueError(f"grip_action must have shape ({self.grip_size},), got {grip_action.shape}")

        body_values = dict(zip(self.body_actuator_names, body_action, strict=True))
        grip_values = dict(zip(self.grip_actuator_names, grip_action, strict=True))
        merged = np.zeros(len(self.all_actuator_names), dtype=float)
        for index, name in enumerate(self.all_actuator_names):
            if name in body_values:
                merged[index] = body_values[name]
            elif name in grip_values:
                merged[index] = grip_values[name]
        return merged
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
pytest environment/overall_environment/tests/test_layered_control.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add environment/overall_environment/src/layered_control.py environment/overall_environment/tests/test_layered_control.py
git commit -m "feat: add layered actuator router"
```

---

### Task 3: Add Static-Hit State Machine And Flight Classifier

**Files:**
- Create: `environment/overall_environment/src/static_forehand_clear_env.py`
- Test: `environment/overall_environment/tests/test_static_forehand_clear_env.py`

- [ ] **Step 1: Write failing pure tests for freeze, release, and landing classification**

Create `environment/overall_environment/tests/test_static_forehand_clear_env.py` with:

```python
from __future__ import annotations

import numpy as np

from environment.overall_environment.src.static_forehand_clear_env import (
    FlightRegion,
    StaticHitState,
    StaticShuttleTarget,
    classify_landing_region,
    release_condition_met,
    should_transition_to_flight_evaluation,
)


def test_release_condition_requires_active_fast_closing_contact_in_phase_window():
    contact = {
        "active": True,
        "rho2": 0.4,
        "penetration": 0.003,
        "relative_normal_velocity": -6.0,
    }

    assert release_condition_met(contact, phase=0.52, impact_phase=0.50, phase_tolerance=0.08)
    assert not release_condition_met(contact, phase=0.80, impact_phase=0.50, phase_tolerance=0.08)
    assert not release_condition_met({**contact, "rho2": 1.2}, phase=0.52, impact_phase=0.50, phase_tolerance=0.08)
    assert not release_condition_met({**contact, "relative_normal_velocity": 1.0}, phase=0.52, impact_phase=0.50, phase_tolerance=0.08)


def test_static_shuttle_target_freeze_writes_qpos_and_qvel():
    qpos = np.zeros(10)
    qvel = np.ones(9)
    target = StaticShuttleTarget(
        qpos_adr=2,
        qvel_adr=3,
        qpos=np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]),
    )

    target.apply_freeze(qpos, qvel)

    np.testing.assert_allclose(qpos[2:9], target.qpos)
    np.testing.assert_allclose(qvel[3:9], np.zeros(6))


def test_landing_region_classifies_opponent_back_court():
    assert classify_landing_region(
        landing_xy=np.array([5.9, 0.2]),
        player_half_sign=-1,
        singles=True,
    ) == FlightRegion.OPPONENT_BACK
    assert classify_landing_region(
        landing_xy=np.array([-3.0, 0.2]),
        player_half_sign=-1,
        singles=True,
    ) == FlightRegion.OWN_SIDE
    assert classify_landing_region(
        landing_xy=np.array([7.2, 0.2]),
        player_half_sign=-1,
        singles=True,
    ) == FlightRegion.OUT


def test_transition_to_flight_evaluation_after_net_crossing_or_landing():
    assert should_transition_to_flight_evaluation(StaticHitState.IMPACT_RELEASED, crossed_net=True, landed=False)
    assert should_transition_to_flight_evaluation(StaticHitState.IMPACT_RELEASED, crossed_net=False, landed=True)
    assert not should_transition_to_flight_evaluation(StaticHitState.PRE_IMPACT_FREEZE, crossed_net=True, landed=False)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest environment/overall_environment/tests/test_static_forehand_clear_env.py -q
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Implement pure state helpers**

Create `environment/overall_environment/src/static_forehand_clear_env.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import numpy as np


class StaticHitState(str, Enum):
    RESET = "RESET"
    PRE_IMPACT_FREEZE = "PRE_IMPACT_FREEZE"
    IMPACT_RELEASED = "IMPACT_RELEASED"
    FLIGHT_EVALUATION = "FLIGHT_EVALUATION"
    TERMINATED = "TERMINATED"


class FlightRegion(str, Enum):
    OWN_SIDE = "own_side"
    NET_FRONT = "net_front"
    OPPONENT_MID = "opponent_mid"
    OPPONENT_BACK = "opponent_back"
    OUT = "out"


@dataclass(frozen=True)
class StaticShuttleTarget:
    qpos_adr: int
    qvel_adr: int
    qpos: np.ndarray

    def apply_freeze(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        target_qpos = np.asarray(self.qpos, dtype=float)
        if target_qpos.shape != (7,):
            raise ValueError(f"static shuttle qpos must have shape (7,), got {target_qpos.shape}")
        qpos[self.qpos_adr : self.qpos_adr + 7] = target_qpos
        qvel[self.qvel_adr : self.qvel_adr + 6] = 0.0


def release_condition_met(
    contact: Mapping[str, object],
    *,
    phase: float,
    impact_phase: float,
    phase_tolerance: float,
) -> bool:
    if not bool(contact.get("active", False)):
        return False
    if abs(float(phase) - float(impact_phase)) > float(phase_tolerance):
        return False
    if float(contact.get("rho2", 2.0)) > 1.0:
        return False
    if float(contact.get("penetration", 0.0)) <= 0.0:
        return False
    return float(contact.get("relative_normal_velocity", 0.0)) < 0.0


def should_transition_to_flight_evaluation(
    state: StaticHitState,
    *,
    crossed_net: bool,
    landed: bool,
) -> bool:
    return state == StaticHitState.IMPACT_RELEASED and (crossed_net or landed)


def classify_landing_region(
    landing_xy: np.ndarray,
    *,
    player_half_sign: int,
    singles: bool,
) -> FlightRegion:
    xy = np.asarray(landing_xy, dtype=float)
    if xy.shape != (2,):
        raise ValueError(f"landing_xy must have shape (2,), got {xy.shape}")
    x, y = float(xy[0]), float(xy[1])
    half_width = 2.59 if singles else 3.05
    if abs(x) > 6.70 or abs(y) > half_width:
        return FlightRegion.OUT
    if np.sign(x) == player_half_sign or abs(x) < 1e-9:
        return FlightRegion.OWN_SIDE
    opponent_depth = abs(x)
    if opponent_depth < 2.0:
        return FlightRegion.NET_FRONT
    if opponent_depth >= 5.35:
        return FlightRegion.OPPONENT_BACK
    return FlightRegion.OPPONENT_MID
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
pytest environment/overall_environment/tests/test_static_forehand_clear_env.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add environment/overall_environment/src/static_forehand_clear_env.py environment/overall_environment/tests/test_static_forehand_clear_env.py
git commit -m "feat: add static hit state helpers"
```

---

### Task 4: Add Overall Static-Hit Environment Skeleton

**Files:**
- Modify: `environment/overall_environment/src/static_forehand_clear_env.py`
- Test: `environment/overall_environment/tests/test_static_forehand_clear_env.py`

- [ ] **Step 1: Add tests for reset and pre-impact freeze with a fake backend**

Append to `environment/overall_environment/tests/test_static_forehand_clear_env.py`:

```python

class _FakeData:
    def __init__(self) -> None:
        self.qpos = np.zeros(12)
        self.qvel = np.ones(11)


class _FakeBaseEnv:
    def __init__(self) -> None:
        self.data = _FakeData()
        self.step_count = 0

    def reset(self):
        return np.zeros(3), {"base_reset": True}

    def step(self, ctrl=None, pose_servo=False):
        self.step_count += 1
        return np.array([float(self.step_count)]), {"base_step": self.step_count}


def test_static_env_reset_enters_pre_impact_freeze_and_freezes_shuttle():
    from environment.overall_environment.src.static_forehand_clear_env import StaticForehandClearEnv

    base = _FakeBaseEnv()
    target = StaticShuttleTarget(
        qpos_adr=1,
        qvel_adr=2,
        qpos=np.array([0.5, 0.6, 2.0, 1.0, 0.0, 0.0, 0.0]),
    )
    env = StaticForehandClearEnv(
        base_env=base,
        shuttle_target=target,
        impact_phase=0.5,
        phase_tolerance=0.1,
    )

    _obs, info = env.reset()

    assert info["state"] == "PRE_IMPACT_FREEZE"
    np.testing.assert_allclose(base.data.qpos[1:8], target.qpos)
    np.testing.assert_allclose(base.data.qvel[2:8], np.zeros(6))


def test_static_env_step_keeps_shuttle_frozen_before_release():
    from environment.overall_environment.src.static_forehand_clear_env import StaticForehandClearEnv

    base = _FakeBaseEnv()
    target = StaticShuttleTarget(
        qpos_adr=1,
        qvel_adr=2,
        qpos=np.array([0.5, 0.6, 2.0, 1.0, 0.0, 0.0, 0.0]),
    )
    env = StaticForehandClearEnv(
        base_env=base,
        shuttle_target=target,
        impact_phase=0.5,
        phase_tolerance=0.1,
    )
    env.reset()
    base.data.qpos[1:8] = 4.0
    base.data.qvel[2:8] = 5.0

    _obs, _reward, terminated, truncated, info = env.step(
        ctrl=None,
        phase=0.1,
        contact_info={"active": False},
    )

    assert not terminated
    assert not truncated
    assert info["state"] == "PRE_IMPACT_FREEZE"
    np.testing.assert_allclose(base.data.qpos[1:8], target.qpos)
    np.testing.assert_allclose(base.data.qvel[2:8], np.zeros(6))
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest environment/overall_environment/tests/test_static_forehand_clear_env.py -q
```

Expected: FAIL with `ImportError` for `StaticForehandClearEnv`.

- [ ] **Step 3: Add skeleton wrapper class**

Append this class to `environment/overall_environment/src/static_forehand_clear_env.py`:

```python

class StaticForehandClearEnv:
    def __init__(
        self,
        *,
        base_env,
        shuttle_target: StaticShuttleTarget,
        impact_phase: float,
        phase_tolerance: float,
    ) -> None:
        self.base_env = base_env
        self.shuttle_target = shuttle_target
        self.impact_phase = float(impact_phase)
        self.phase_tolerance = float(phase_tolerance)
        self.state = StaticHitState.RESET
        self.release_step: int | None = None
        self.step_index = 0

    def reset(self):
        obs, info = self.base_env.reset()
        self.step_index = 0
        self.release_step = None
        self.state = StaticHitState.PRE_IMPACT_FREEZE
        self.shuttle_target.apply_freeze(self.base_env.data.qpos, self.base_env.data.qvel)
        return obs, {**info, "state": self.state.value}

    def step(self, ctrl=None, *, phase: float, contact_info: Mapping[str, object] | None = None):
        if contact_info is None:
            contact_info = {"active": False}
        if self.state == StaticHitState.PRE_IMPACT_FREEZE:
            self.shuttle_target.apply_freeze(self.base_env.data.qpos, self.base_env.data.qvel)
            if release_condition_met(
                contact_info,
                phase=phase,
                impact_phase=self.impact_phase,
                phase_tolerance=self.phase_tolerance,
            ):
                self.state = StaticHitState.IMPACT_RELEASED
                self.release_step = self.step_index
        obs, info = self.base_env.step(ctrl)
        if self.state == StaticHitState.PRE_IMPACT_FREEZE:
            self.shuttle_target.apply_freeze(self.base_env.data.qpos, self.base_env.data.qvel)
        self.step_index += 1
        reward = 0.0
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, {**info, "state": self.state.value}
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
pytest environment/overall_environment/tests/test_static_forehand_clear_env.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add environment/overall_environment/src/static_forehand_clear_env.py environment/overall_environment/tests/test_static_forehand_clear_env.py
git commit -m "feat: add static forehand clear env skeleton"
```

---

### Task 5: Integrate Stringbed, Event Rebound, And Aero Hooks

**Files:**
- Modify: `environment/overall_environment/src/static_forehand_clear_env.py`
- Test: `environment/overall_environment/tests/test_static_forehand_clear_env.py`

- [ ] **Step 1: Add tests for release hook ordering**

Append to `environment/overall_environment/tests/test_static_forehand_clear_env.py`:

```python

def test_static_env_calls_physics_hooks_after_release_only():
    from environment.overall_environment.src.static_forehand_clear_env import StaticForehandClearEnv

    calls: list[str] = []

    def stringbed_hook(model, data):
        calls.append("stringbed")
        return {"active": True, "relative_normal_velocity": -6.0, "normal_world": np.array([0.0, 0.0, 1.0])}

    def rebound_hook(contact_info):
        calls.append("rebound")
        return True

    def aero_hook(model, data):
        calls.append("aero")
        return {"speed_m_s": 12.0}

    base = _FakeBaseEnv()
    base.model = object()
    target = StaticShuttleTarget(
        qpos_adr=1,
        qvel_adr=2,
        qpos=np.array([0.5, 0.6, 2.0, 1.0, 0.0, 0.0, 0.0]),
    )
    env = StaticForehandClearEnv(
        base_env=base,
        shuttle_target=target,
        impact_phase=0.5,
        phase_tolerance=0.1,
        stringbed_hook=stringbed_hook,
        rebound_hook=rebound_hook,
        aero_hook=aero_hook,
    )
    env.reset()

    env.step(ctrl=None, phase=0.1, contact_info={"active": False})
    assert calls == []

    env.step(
        ctrl=None,
        phase=0.5,
        contact_info={"active": True, "rho2": 0.2, "penetration": 0.002, "relative_normal_velocity": -3.0},
    )
    assert calls == ["stringbed", "rebound", "aero"]
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest environment/overall_environment/tests/test_static_forehand_clear_env.py::test_static_env_calls_physics_hooks_after_release_only -q
```

Expected: FAIL because `StaticForehandClearEnv.__init__` does not accept the hook arguments.

- [ ] **Step 3: Extend wrapper with hook support**

Modify `StaticForehandClearEnv.__init__` in `environment/overall_environment/src/static_forehand_clear_env.py` to accept and store hooks:

```python
    def __init__(
        self,
        *,
        base_env,
        shuttle_target: StaticShuttleTarget,
        impact_phase: float,
        phase_tolerance: float,
        stringbed_hook=None,
        rebound_hook=None,
        aero_hook=None,
    ) -> None:
        self.base_env = base_env
        self.shuttle_target = shuttle_target
        self.impact_phase = float(impact_phase)
        self.phase_tolerance = float(phase_tolerance)
        self.stringbed_hook = stringbed_hook
        self.rebound_hook = rebound_hook
        self.aero_hook = aero_hook
        self.state = StaticHitState.RESET
        self.release_step: int | None = None
        self.step_index = 0
```

Add this private method to the class:

```python
    def _apply_released_physics(self) -> dict[str, object]:
        diagnostics: dict[str, object] = {}
        if self.stringbed_hook is not None:
            contact_info = self.stringbed_hook(self.base_env.model, self.base_env.data)
            diagnostics["stringbed"] = contact_info
            if self.rebound_hook is not None:
                diagnostics["event_rebound_used"] = bool(self.rebound_hook(contact_info))
        if self.aero_hook is not None:
            diagnostics["aero"] = self.aero_hook(self.base_env.model, self.base_env.data)
        return diagnostics
```

In `step`, after a release transition and before `base_env.step(ctrl)`, add:

```python
        physics_info: dict[str, object] = {}
        if self.state == StaticHitState.IMPACT_RELEASED:
            physics_info = self._apply_released_physics()
```

Then change the return info merge to:

```python
        return obs, reward, terminated, truncated, {**info, **physics_info, "state": self.state.value}
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
pytest environment/overall_environment/tests/test_static_forehand_clear_env.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add environment/overall_environment/src/static_forehand_clear_env.py environment/overall_environment/tests/test_static_forehand_clear_env.py
git commit -m "feat: add static hit physics hooks"
```

---

### Task 6: Add Grip Swing-Disturbance Config And Helper

**Files:**
- Modify: `configs/right_hand_racket_grip_training.yaml`
- Modify: `src/grip/right_hand_racket_grip_env.py`
- Test: `tests/test_right_hand_racket_grip.py`

- [ ] **Step 1: Add failing tests for config defaults and deterministic disturbance helper**

Append to `tests/test_right_hand_racket_grip.py`:

```python

def test_grip_training_config_includes_default_off_swing_disturbance():
    from src.grip.right_hand_racket_grip_env import load_training_config

    cfg = load_training_config("configs/right_hand_racket_grip_training.yaml")

    assert cfg["swing_disturbance"]["enabled"] is False
    assert cfg["swing_disturbance"]["force_scale_n"] == 0.0
    assert cfg["swing_disturbance"]["torque_scale_nm"] == 0.0
    assert cfg["swing_disturbance"]["phase_start"] == 0.0
    assert cfg["swing_disturbance"]["phase_end"] == 1.0


def test_swing_disturbance_profile_is_zero_outside_phase_window():
    from src.grip.right_hand_racket_grip_env import swing_disturbance_profile

    force, torque = swing_disturbance_profile(
        phase=0.1,
        phase_start=0.4,
        phase_end=0.6,
        force_scale_n=2.0,
        torque_scale_nm=0.03,
    )

    assert force.tolist() == [0.0, 0.0, 0.0]
    assert torque.tolist() == [0.0, 0.0, 0.0]


def test_swing_disturbance_profile_peaks_inside_phase_window():
    from src.grip.right_hand_racket_grip_env import swing_disturbance_profile

    force, torque = swing_disturbance_profile(
        phase=0.5,
        phase_start=0.4,
        phase_end=0.6,
        force_scale_n=2.0,
        torque_scale_nm=0.03,
    )

    assert force.tolist() == [2.0, 0.0, 0.0]
    assert torque.tolist() == [0.0, 0.03, 0.0]
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_right_hand_racket_grip.py::test_grip_training_config_includes_default_off_swing_disturbance tests/test_right_hand_racket_grip.py::test_swing_disturbance_profile_is_zero_outside_phase_window tests/test_right_hand_racket_grip.py::test_swing_disturbance_profile_peaks_inside_phase_window -q
```

Expected: FAIL because the config section and helper do not exist.

- [ ] **Step 3: Add default-off YAML fields**

Append this top-level section to `configs/right_hand_racket_grip_training.yaml`:

```yaml
swing_disturbance:
  enabled: false
  force_scale_n: 0.0
  torque_scale_nm: 0.0
  phase_start: 0.0
  phase_end: 1.0
```

- [ ] **Step 4: Add config parsing defaults and pure helper**

In `src/grip/right_hand_racket_grip_env.py`, add near the default config constants:

```python
DEFAULT_SWING_DISTURBANCE = {
    "enabled": False,
    "force_scale_n": 0.0,
    "torque_scale_nm": 0.0,
    "phase_start": 0.0,
    "phase_end": 1.0,
}
```

Add this function near other pure helpers:

```python
def swing_disturbance_profile(
    *,
    phase: float,
    phase_start: float,
    phase_end: float,
    force_scale_n: float,
    torque_scale_nm: float,
) -> tuple[np.ndarray, np.ndarray]:
    if phase_end <= phase_start:
        raise ValueError("phase_end must be greater than phase_start")
    if phase < phase_start or phase > phase_end:
        return np.zeros(3, dtype=float), np.zeros(3, dtype=float)
    normalized = (phase - phase_start) / (phase_end - phase_start)
    envelope = float(np.sin(np.pi * normalized))
    force = np.array([force_scale_n * envelope, 0.0, 0.0], dtype=float)
    torque = np.array([0.0, torque_scale_nm * envelope, 0.0], dtype=float)
    return force, torque
```

Modify `load_training_config` so the returned dictionary includes the merged section:

```python
    swing_disturbance = {
        **DEFAULT_SWING_DISTURBANCE,
        **_mapping_value(loaded.get("swing_disturbance", {}), "swing_disturbance"),
    }
    return {
        "env": env,
        "reward": reward,
        "swing_disturbance": swing_disturbance,
    }
```

If the current `load_training_config` already returns additional keys, preserve those keys and add `"swing_disturbance": swing_disturbance` to the existing returned dictionary.

- [ ] **Step 5: Run focused tests and verify pass**

Run:

```bash
pytest tests/test_right_hand_racket_grip.py::test_grip_training_config_includes_default_off_swing_disturbance tests/test_right_hand_racket_grip.py::test_swing_disturbance_profile_is_zero_outside_phase_window tests/test_right_hand_racket_grip.py::test_swing_disturbance_profile_peaks_inside_phase_window -q
```

Expected: PASS.

- [ ] **Step 6: Run existing grip tests**

Run:

```bash
pytest tests/test_right_hand_racket_grip.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add configs/right_hand_racket_grip_training.yaml src/grip/right_hand_racket_grip_env.py tests/test_right_hand_racket_grip.py
git commit -m "feat: add grip swing disturbance config"
```

---

### Task 7: Add Trained Grip Policy Metadata

**Files:**
- Modify: `src/grip/train_right_hand_racket_grip_policy.py`
- Test: `tests/test_right_hand_racket_grip.py`

- [ ] **Step 1: Add a failing metadata test**

Append to `tests/test_right_hand_racket_grip.py`:

```python

def test_grip_policy_training_metadata_records_disturbance_config(tmp_path):
    from src.grip.train_right_hand_racket_grip_policy import build_training_metadata

    metadata = build_training_metadata(
        xml="assets/right_hand_racket_grip_scene.xml",
        targets="configs/right_hand_racket_grip_targets.json",
        reference="configs/right_hand_racket_grip_reference.json",
        training_config="configs/right_hand_racket_grip_training.yaml",
        swing_disturbance={"enabled": True, "force_scale_n": 1.5, "torque_scale_nm": 0.02},
        obs_size=301,
        action_size=31,
        global_step=1024,
    )

    assert metadata["mode"] == "ppo_right_hand_racket_grip"
    assert metadata["swing_disturbance"]["enabled"] is True
    assert metadata["swing_disturbance"]["force_scale_n"] == 1.5
    assert metadata["obs_size"] == 301
    assert metadata["action_size"] == 31
    assert metadata["global_step"] == 1024
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
pytest tests/test_right_hand_racket_grip.py::test_grip_policy_training_metadata_records_disturbance_config -q
```

Expected: FAIL with `ImportError` because `build_training_metadata` does not exist.

- [ ] **Step 3: Add metadata helper**

In `src/grip/train_right_hand_racket_grip_policy.py`, add:

```python
def build_training_metadata(
    *,
    xml: str | Path,
    targets: str | Path,
    reference: str | Path,
    training_config: str | Path,
    swing_disturbance: dict[str, Any],
    obs_size: int,
    action_size: int,
    global_step: int,
) -> dict[str, Any]:
    return {
        "mode": "ppo_right_hand_racket_grip",
        "xml": str(Path(xml)),
        "targets": str(Path(targets)),
        "reference": str(Path(reference)),
        "training_config": str(Path(training_config)),
        "swing_disturbance": _json_safe(swing_disturbance),
        "obs_size": int(obs_size),
        "action_size": int(action_size),
        "global_step": int(global_step),
    }
```

In `train_policy`, after `env = RightHandRacketGripEnv(...)`, add:

```python
    swing_disturbance_config = env.config.get("swing_disturbance", {})
```

When building `metrics`, add:

```python
        "swing_disturbance": _json_safe(swing_disturbance_config),
```

- [ ] **Step 4: Run metadata test and trainer unit smoke tests**

Run:

```bash
pytest tests/test_right_hand_racket_grip.py::test_grip_policy_training_metadata_records_disturbance_config -q
```

Expected: PASS.

Run:

```bash
pytest tests/test_right_hand_racket_grip.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/grip/train_right_hand_racket_grip_policy.py tests/test_right_hand_racket_grip.py
git commit -m "feat: record grip policy training metadata"
```

---

### Task 8: Add Static-Hit Experiment Spec

**Files:**
- Create: `BadmintonMimic/experiments/posttrain/forehand_clear_static_hit_v1.yaml`
- Test: `tests/unit/test_forehand_clear_static_hit_spec.py`

- [ ] **Step 1: Write failing spec test**

Create `tests/unit/test_forehand_clear_static_hit_spec.py` with:

```python
from __future__ import annotations

from pathlib import Path

import yaml


SPEC = Path("BadmintonMimic/experiments/posttrain/forehand_clear_static_hit_v1.yaml")


def test_static_hit_spec_declares_required_stages_and_checkpoints():
    data = yaml.safe_load(SPEC.read_text(encoding="utf-8"))

    assert data["action"] == "ForehandClearStaticHit"
    assert data["body_policy"]["resume_from"]
    assert data["grip_policy"]["required"] is True
    assert data["grip_policy"]["checkpoint"] == "outputs/right_hand_racket_grip/policy/policy_latest.pt"
    assert [stage["name"] for stage in data["curriculum"]] == [
        "physics_chain_validation",
        "static_grip_stabilizer",
        "swing_disturbance_grip",
        "hit_and_over_net",
        "high_clear_depth",
    ]


def test_static_hit_spec_uses_freeze_release_shuttle_mode():
    data = yaml.safe_load(SPEC.read_text(encoding="utf-8"))

    assert data["shuttle"]["mode"] == "pre_impact_freeze_release"
    assert data["shuttle"]["release"]["require_stringbed_contact"] is True
    assert data["shuttle"]["release"]["phase_tolerance"] == 0.08
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
pytest tests/unit/test_forehand_clear_static_hit_spec.py -q
```

Expected: FAIL with `FileNotFoundError`.

- [ ] **Step 3: Create experiment spec**

Create `BadmintonMimic/experiments/posttrain/forehand_clear_static_hit_v1.yaml` with:

```yaml
action: ForehandClearStaticHit
description: Static-shuttle ForehandClear post-train with real hand-racket contact.

body_policy:
  resume_from: /data3/yangfeiyang/WorkSpace/musclemimic/checkpoints/ForehandClear/forehand_clear_best/checkpoint_7812
  trainable: false
  owns_right_hand_fingers: false

grip_policy:
  required: true
  checkpoint: outputs/right_hand_racket_grip/policy/policy_latest.pt
  trainable: true
  owns_right_hand_fingers: true

scene:
  xml: environment/overall_environment/assets/overall_badminton_scene.xml
  player_half_sign: -1
  root_start_xy: [-2.5, 0.0]
  face_net: true

impact_target:
  source: forehand_clear_reference
  racket_length_m: 0.67
  regularization:
    min_forward_offset_m: 0.28
    max_forward_offset_m: 0.90
    min_racket_side_offset_m: 0.16
    max_racket_side_offset_m: 0.70
    reach_alpha: 0.78
    racket_beta: 0.82

shuttle:
  mode: pre_impact_freeze_release
  release:
    require_stringbed_contact: true
    phase_tolerance: 0.08
    max_rho2: 1.0
    require_closing_velocity: true

curriculum:
  - name: physics_chain_validation
    train: false
    success_metric: impact_detected
  - name: static_grip_stabilizer
    train: true
    policy: grip
    success_metric: grip_acceptance_pass
  - name: swing_disturbance_grip
    train: true
    policy: grip
    success_metric: swing_grip_acceptance_pass
  - name: hit_and_over_net
    train: true
    policy: layered
    success_metric: over_net_rate
  - name: high_clear_depth
    train: true
    policy: layered
    success_metric: opponent_back_court_rate

validation:
  require_finite: true
  grip:
    min_contacts: 4
    max_illegal_contacts: 0
    max_mean_site_error_m: 0.02
    max_handle_penetration_m: 0.003
  impact:
    min_racket_head_speed_m_s: 8.0
    max_phase_error: 0.08
  flight:
    min_net_clearance_m: 0.25
    target_landing_region: opponent_back
```

- [ ] **Step 4: Run spec tests**

Run:

```bash
pytest tests/unit/test_forehand_clear_static_hit_spec.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add BadmintonMimic/experiments/posttrain/forehand_clear_static_hit_v1.yaml tests/unit/test_forehand_clear_static_hit_spec.py
git commit -m "feat: add forehand clear static hit experiment spec"
```

---

### Task 9: Add Operational Documentation

**Files:**
- Create: `docs/forehand_clear_static_hit_posttrain.md`

- [ ] **Step 1: Create operations doc**

Create `docs/forehand_clear_static_hit_posttrain.md` with:

```markdown
# Forehand Clear Static-Hit PostTrain

This workflow trains a staged static-shuttle ForehandClear task in the Overall badminton scene.

## Stage Order

1. Physics chain validation: static shuttle freeze, impact release, string-bed force, event rebound, aero, net crossing, and landing diagnostics.
2. Static grip stabilizer: train the right hand to hold the racket from the current grip reference.
3. Swing-disturbance grip stabilizer: train the right hand under ForehandClear-like wrist, forearm, and racket inertia disturbances.
4. Hit-and-over-net post-train: compose body and grip policies and optimize valid impact plus net clearance.
5. High-clear depth post-train: add back-court landing and clear-arc objectives.

## Key Files

- Spec: `docs/superpowers/specs/2026-05-27-forehand-clear-static-hit-posttrain-design.md`
- Plan: `docs/superpowers/plans/2026-05-27-forehand-clear-static-hit-posttrain.md`
- Overall scene: `environment/overall_environment/assets/overall_badminton_scene.xml`
- Static-hit env: `environment/overall_environment/src/static_forehand_clear_env.py`
- Impact target helper: `environment/overall_environment/src/impact_target.py`
- Layered control helper: `environment/overall_environment/src/layered_control.py`
- Grip trainer: `src/grip/train_right_hand_racket_grip_policy.py`
- Experiment spec: `BadmintonMimic/experiments/posttrain/forehand_clear_static_hit_v1.yaml`

## Validation Commands

```bash
pytest environment/overall_environment/tests/test_impact_target.py -q
pytest environment/overall_environment/tests/test_layered_control.py -q
pytest environment/overall_environment/tests/test_static_forehand_clear_env.py -q
pytest tests/test_right_hand_racket_grip.py -q
pytest tests/unit/test_forehand_clear_static_hit_spec.py -q
```

## Current Gating Condition

Do not start composed Overall static-hit training until a trained grip policy exists at:

```text
outputs/right_hand_racket_grip/policy/policy_latest.pt
```

The current grip reference alone is not accepted for real-contact swing training.
```

- [ ] **Step 2: Commit**

```bash
git add docs/forehand_clear_static_hit_posttrain.md
git commit -m "docs: add static hit posttrain operations guide"
```

---

### Task 10: Run Focused Verification

**Files:**
- No source changes.

- [ ] **Step 1: Run all new focused tests**

Run:

```bash
pytest \
  environment/overall_environment/tests/test_impact_target.py \
  environment/overall_environment/tests/test_layered_control.py \
  environment/overall_environment/tests/test_static_forehand_clear_env.py \
  tests/unit/test_forehand_clear_static_hit_spec.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run grip tests**

Run:

```bash
pytest tests/test_right_hand_racket_grip.py -q
```

Expected: PASS.

- [ ] **Step 3: Run existing Overall environment tests**

Run:

```bash
pytest environment/overall_environment/tests/test_overall_environment.py -q
```

Expected: PASS.

- [ ] **Step 4: Record verification in git status**

Run:

```bash
git status --short
```

Expected: no uncommitted files from this plan. Pre-existing unrelated dirty files may still appear if they were present before execution; do not revert them.

---

## Self-Review

Spec coverage:

- Static shuttle freeze/release is covered by Tasks 3, 4, and 5.
- Reference plus body-scale impact target extraction is covered by Task 1.
- Layered body/grip control is covered by Task 2.
- Grip stabilizer swing-disturbance preparation is covered by Tasks 6 and 7.
- Experiment configuration and operational handoff are covered by Tasks 8 and 9.
- Focused verification is covered by Task 10.

Completion scan:

- No placeholder markers or incomplete task steps remain.
- Every task names exact files, commands, and expected results.

Type consistency:

- `ImpactTarget`, `BodyScale`, and `ImpactTargetConfig` are defined before use.
- `LayeredActuatorRouter.merge()` uses `body_action` and `grip_action` consistently.
- `StaticHitState`, `FlightRegion`, `StaticShuttleTarget`, and `StaticForehandClearEnv` names match across tests and implementation steps.
- Grip metadata uses `swing_disturbance` consistently across config, environment, and trainer metadata.

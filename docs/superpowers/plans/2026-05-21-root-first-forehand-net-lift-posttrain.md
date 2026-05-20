# Root-First Forehand Net Lift Post-Train Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a root-first ForehandNetLift post-train path that resumes from the best existing PPO checkpoint and measures whether root motion improves.

**Architecture:** Add default-off root tracking signals to the existing goal and reward code, plus pure metric utilities and a dedicated root-first config. Keep the first run on the eight checkpoint-compatible ForehandNetLift best motions, with explicit diagnostics before and after post-train.

**Tech Stack:** Python, NumPy, JAX/JAX NumPy, MuJoCo/MJX, Hydra/OmegaConf YAML, pytest, existing MuscleMimic PPO runner.

---

## File Structure

- Create `musclemimic/utils/root_tracking.py`: pure NumPy helpers for checkpoint discovery, root/site metrics, and cache diagnostics.
- Create `tests/unit/test_root_tracking_metrics.py`: unit tests for pure metrics and checkpoint selection.
- Modify `musclemimic/core/goals/trajectory.py`: add default-off current root error to `GoalTrajMimic`.
- Create `tests/unit/test_goal_traj_root_error.py`: minimal tests for goal dimension and root-error math.
- Modify `musclemimic/core/reward/trajectory_based.py`: add default-off absolute site reward to `MimicReward`.
- Extend `tests/unit/test_mimic_reward.py`: tests for absolute site reward initialization and monotonic reward behavior.
- Create `BadmintonMimic/scripts/diagnose_root_tracking.py`: CLI for reference cache metrics.
- Create `tests/unit/test_diagnose_root_tracking.py`: CLI/unit tests for path resolution and JSON output.
- Create `fullbody/config_specific_task/conf_fullbody_forehand_net_lift_root_first.yaml`: post-train config for the current ForehandNetLift best data.
- Create `tests/unit/test_forehand_net_lift_root_first_config.py`: config contract tests.

## Task 1: Pure Root Tracking Metrics

**Files:**
- Create: `musclemimic/utils/root_tracking.py`
- Create: `tests/unit/test_root_tracking_metrics.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_root_tracking_metrics.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from musclemimic.utils.root_tracking import (
    compute_root_reference_metrics,
    compute_rollout_root_metrics,
    select_latest_checkpoint,
)


def test_select_latest_checkpoint_uses_largest_numeric_suffix(tmp_path):
    root = tmp_path / "ckpts"
    root.mkdir()
    for name in ["checkpoint_10", "checkpoint_2", "checkpoint_100"]:
        (root / name).mkdir()
    (root / "manifest.json").write_text("{}", encoding="utf-8")

    assert select_latest_checkpoint(root) == root / "checkpoint_100"


def test_select_latest_checkpoint_rejects_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint root does not exist"):
        select_latest_checkpoint(tmp_path / "missing")


def test_select_latest_checkpoint_rejects_empty_checkpoint_root(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()

    with pytest.raises(ValueError, match="no checkpoint_\\* directories"):
        select_latest_checkpoint(root)


def test_compute_root_reference_metrics_reports_displacement_and_speed():
    qpos = np.array(
        [
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.3, 0.4, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.6, 0.8, 1.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    qvel = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0, 0.0, 0.0],
            [6.0, 8.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    metrics = compute_root_reference_metrics(qpos=qpos, qvel=qvel)

    assert metrics["reference_root_xy_total_displacement"] == pytest.approx(1.0)
    assert metrics["reference_root_xy_path_length"] == pytest.approx(1.0)
    assert metrics["reference_root_xy_peak_speed"] == pytest.approx(10.0)


def test_compute_rollout_root_metrics_compares_rollout_to_reference():
    reference_qpos = np.array(
        [
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    rollout_qpos = np.array(
        [
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    metrics = compute_rollout_root_metrics(reference_qpos=reference_qpos, rollout_qpos=rollout_qpos)

    assert metrics["reference_root_xy_total_displacement"] == pytest.approx(1.0)
    assert metrics["rollout_root_xy_total_displacement"] == pytest.approx(0.5)
    assert metrics["root_displacement_ratio"] == pytest.approx(0.5)
    assert metrics["root_xy_final_error"] == pytest.approx(0.5)
    assert metrics["root_xy_rmse"] == pytest.approx(np.sqrt((0.0**2 + 0.5**2) / 2.0))
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_root_tracking_metrics.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'musclemimic.utils.root_tracking'`.

- [ ] **Step 3: Implement the metrics module**

Create `musclemimic/utils/root_tracking.py`:

```python
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np


_CHECKPOINT_RE = re.compile(r"^checkpoint_(\d+)$")


def select_latest_checkpoint(checkpoint_root: str | Path) -> Path:
    root = Path(checkpoint_root)
    if not root.exists():
        raise FileNotFoundError(f"checkpoint root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"checkpoint root is not a directory: {root}")

    candidates: list[tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = _CHECKPOINT_RE.match(child.name)
        if match is not None:
            candidates.append((int(match.group(1)), child))
    if not candidates:
        raise ValueError(f"no checkpoint_* directories found under {root}")
    return max(candidates, key=lambda item: item[0])[1]


def _as_float_array(name: str, value: Any, ndim: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got shape {arr.shape}")
    return arr


def _root_xy(qpos: Any) -> np.ndarray:
    arr = _as_float_array("qpos", qpos, 2)
    if arr.shape[1] < 2:
        raise ValueError(f"qpos must have at least 2 columns for root XY, got shape {arr.shape}")
    return arr[:, :2]


def _root_xy_speed(qvel: Any | None, qpos: Any, frequency: float | None) -> np.ndarray:
    if qvel is not None:
        vel = _as_float_array("qvel", qvel, 2)
        if vel.shape[1] < 2:
            raise ValueError(f"qvel must have at least 2 columns for root XY velocity, got shape {vel.shape}")
        return np.linalg.norm(vel[:, :2], axis=-1)
    xy = _root_xy(qpos)
    if xy.shape[0] < 2:
        return np.zeros((xy.shape[0],), dtype=np.float64)
    dt_scale = 1.0 if frequency is None else float(frequency)
    step_speed = np.linalg.norm(np.diff(xy, axis=0), axis=-1) * dt_scale
    return np.concatenate([[0.0], step_speed])


def _path_length_xy(xy: np.ndarray) -> float:
    if xy.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=-1)))


def _yaw_from_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def compute_root_reference_metrics(
    *,
    qpos: Any,
    qvel: Any | None = None,
    site_xpos: Any | None = None,
    right_hand_site_index: int | None = None,
    frequency: float | None = None,
) -> dict[str, float]:
    qpos_arr = _as_float_array("qpos", qpos, 2)
    xy = _root_xy(qpos_arr)
    total_disp = float(np.linalg.norm(xy[-1] - xy[0])) if xy.shape[0] else 0.0
    speed = _root_xy_speed(qvel, qpos_arr, frequency)

    metrics = {
        "reference_root_xy_total_displacement": total_disp,
        "reference_root_xy_path_length": _path_length_xy(xy),
        "reference_root_xy_peak_speed": float(np.max(speed)) if speed.size else 0.0,
        "reference_root_yaw_change": 0.0,
        "right_hand_world_path_length": 0.0,
    }

    if qpos_arr.shape[1] >= 7:
        yaw = _yaw_from_quat_wxyz(qpos_arr[:, 3:7])
        yaw_delta = np.arctan2(np.sin(yaw[-1] - yaw[0]), np.cos(yaw[-1] - yaw[0]))
        metrics["reference_root_yaw_change"] = float(abs(yaw_delta))

    if site_xpos is not None and right_hand_site_index is not None:
        sites = _as_float_array("site_xpos", site_xpos, 3)
        if not 0 <= int(right_hand_site_index) < sites.shape[1]:
            raise IndexError(
                f"right_hand_site_index {right_hand_site_index} outside site axis with size {sites.shape[1]}"
            )
        metrics["right_hand_world_path_length"] = _path_length_xy(sites[:, int(right_hand_site_index), :2])

    return metrics


def compute_rollout_root_metrics(
    *,
    reference_qpos: Any,
    rollout_qpos: Any,
    reference_qvel: Any | None = None,
    rollout_qvel: Any | None = None,
    frequency: float | None = None,
) -> dict[str, float]:
    ref_qpos = _as_float_array("reference_qpos", reference_qpos, 2)
    out_qpos = _as_float_array("rollout_qpos", rollout_qpos, 2)
    n = min(ref_qpos.shape[0], out_qpos.shape[0])
    if n == 0:
        raise ValueError("reference_qpos and rollout_qpos must contain at least one frame")
    ref_qpos = ref_qpos[:n]
    out_qpos = out_qpos[:n]
    ref_xy = _root_xy(ref_qpos)
    out_xy = _root_xy(out_qpos)

    ref_disp = float(np.linalg.norm(ref_xy[-1] - ref_xy[0]))
    out_disp = float(np.linalg.norm(out_xy[-1] - out_xy[0]))
    xy_err = np.linalg.norm(out_xy - ref_xy, axis=-1)

    metrics = compute_root_reference_metrics(qpos=ref_qpos, qvel=None if reference_qvel is None else np.asarray(reference_qvel)[:n], frequency=frequency)
    metrics.update(
        {
            "rollout_root_xy_total_displacement": out_disp,
            "root_displacement_ratio": out_disp / max(ref_disp, 1.0e-8),
            "root_xy_rmse": float(np.sqrt(np.mean(np.square(xy_err)))),
            "root_xy_final_error": float(xy_err[-1]),
            "root_speed_rmse": 0.0,
        }
    )

    if reference_qvel is not None or rollout_qvel is not None:
        ref_speed = _root_xy_speed(None if reference_qvel is None else np.asarray(reference_qvel)[:n], ref_qpos, frequency)
        out_speed = _root_xy_speed(None if rollout_qvel is None else np.asarray(rollout_qvel)[:n], out_qpos, frequency)
        metrics["root_speed_rmse"] = float(np.sqrt(np.mean(np.square(out_speed - ref_speed))))

    return metrics
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_root_tracking_metrics.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add musclemimic/utils/root_tracking.py tests/unit/test_root_tracking_metrics.py
git commit -m "feat: add root tracking metric utilities"
```

## Task 2: Reference Cache Diagnostic CLI

**Files:**
- Create: `BadmintonMimic/scripts/diagnose_root_tracking.py`
- Create: `tests/unit/test_diagnose_root_tracking.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_diagnose_root_tracking.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_diagnose_root_tracking.py -q
```

Expected: FAIL with `FileNotFoundError` or `ModuleNotFoundError` because the script does not exist.

- [ ] **Step 3: Implement the CLI**

Create `BadmintonMimic/scripts/diagnose_root_tracking.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from musclemimic.utils.root_tracking import compute_root_reference_metrics


def _resolve_cache_file(cache_root: Path, motion: str) -> Path:
    rel = Path(motion)
    if rel.suffix != ".npz":
        rel = rel.with_suffix(".npz")
    path = cache_root / rel
    if not path.exists():
        raise FileNotFoundError(f"cache file not found: {path}")
    return path


def _float_scalar(value, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(np.asarray(value).reshape(-1)[0])


def _diagnose_cache_file(cache_file: Path, right_hand_site_index: int | None) -> dict[str, float | int | str]:
    data = np.load(cache_file, allow_pickle=True)
    qpos = data["qpos"]
    qvel = data["qvel"] if "qvel" in data else None
    site_xpos = data["site_xpos"] if "site_xpos" in data else None
    frequency = _float_scalar(data["frequency"] if "frequency" in data else None)
    metrics = compute_root_reference_metrics(
        qpos=qpos,
        qvel=qvel,
        site_xpos=site_xpos,
        right_hand_site_index=right_hand_site_index,
        frequency=frequency,
    )
    return {
        "motion": cache_file.with_suffix("").name,
        "path": str(cache_file),
        "frames": int(qpos.shape[0]),
        "frequency": float(frequency),
        **metrics,
    }


def _write_json_report(output: Path, rows: list[dict]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose root motion in retargeted cache files.")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("caches/AMASS/MyoFullBody/gmr"),
        help="Root directory containing retargeted cache motions.",
    )
    parser.add_argument("--motion", action="append", required=True, help="Motion path relative to --cache-root.")
    parser.add_argument("--right-hand-site-index", type=int, default=8, help="right_hand_mimic site index in cache.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    rows = []
    for motion in args.motion:
        cache_file = _resolve_cache_file(args.cache_root, motion)
        rows.append(_diagnose_cache_file(cache_file, args.right_hand_site_index))

    for row in rows:
        print(
            f"{row['motion']}: disp={row['reference_root_xy_total_displacement']:.3f}m "
            f"path={row['reference_root_xy_path_length']:.3f}m "
            f"peak_speed={row['reference_root_xy_peak_speed']:.3f}m/s"
        )
    if args.output is not None:
        _write_json_report(args.output, rows)
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_diagnose_root_tracking.py tests/unit/test_root_tracking_metrics.py -q
```

Expected: PASS.

- [ ] **Step 5: Run the ForehandNetLift reference diagnostic**

Run:

```bash
.venv/bin/python BadmintonMimic/scripts/diagnose_root_tracking.py \
  --cache-root caches/AMASS/MyoFullBody/gmr \
  --motion ForehandNetLift/best/video01_best_stage7_smpl \
  --motion ForehandNetLift/best/video02_best_stage7_smpl \
  --motion ForehandNetLift/best/video03_best_stage7_smpl \
  --motion ForehandNetLift/best/video04_best_stage7_smpl \
  --motion ForehandNetLift/best/video05_best_stage7_smpl \
  --motion ForehandNetLift/best/video06_best_stage7_smpl \
  --motion ForehandNetLift/best/video07_best_stage5_smpl \
  --motion ForehandNetLift/best/video08_best_stage5_smpl \
  --output outputs/root_first_forehand_net_lift/reference_metrics.json
```

Expected: prints one line per motion and writes `outputs/root_first_forehand_net_lift/reference_metrics.json`. If any displacement is below `0.300m`, pause execution and inspect whether the selected reference data can train forward root motion.

- [ ] **Step 6: Commit**

Run:

```bash
git add BadmintonMimic/scripts/diagnose_root_tracking.py tests/unit/test_diagnose_root_tracking.py
git commit -m "feat: add root tracking diagnostic CLI"
```

## Task 3: Goal Observation Current Root Error

**Files:**
- Modify: `musclemimic/core/goals/trajectory.py`
- Create: `tests/unit/test_goal_traj_root_error.py`

- [ ] **Step 1: Write failing tests for helper math and dimensions**

Create `tests/unit/test_goal_traj_root_error.py`:

```python
import numpy as np

from musclemimic.core.goals.trajectory import _root_error_components


def test_root_error_components_aligns_reference_xy_to_episode_origin():
    sim_qpos = np.array([0.2, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    sim_qvel = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    ref_qpos = np.array([2.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    ref_qvel = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    init_ref_qpos = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    error = _root_error_components(sim_qpos, sim_qvel, ref_qpos, ref_qvel, init_ref_qpos, np)

    np.testing.assert_allclose(error[:3], np.array([0.8, 0.0, 0.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(error[3:9], np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(error[9:], np.array([0.0], dtype=np.float32), atol=1e-6)


def test_root_error_components_wraps_yaw_to_pi_interval():
    sim_qpos = np.array([0.0, 0.0, 1.0, -0.9998477, 0.0, 0.0, 0.0174524], dtype=np.float32)
    sim_qvel = np.zeros(6, dtype=np.float32)
    ref_qpos = np.array([0.0, 0.0, 1.0, -0.9998477, 0.0, 0.0, -0.0174524], dtype=np.float32)
    ref_qvel = np.zeros(6, dtype=np.float32)
    init_ref_qpos = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    error = _root_error_components(sim_qpos, sim_qvel, ref_qpos, ref_qvel, init_ref_qpos, np)

    assert abs(float(error[-1])) < 0.08
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_goal_traj_root_error.py -q
```

Expected: FAIL with `ImportError: cannot import name '_root_error_components'`.

- [ ] **Step 3: Add root error helper to `trajectory.py`**

In `musclemimic/core/goals/trajectory.py`, add this helper near the imports:

```python
def _yaw_from_wxyz_quat(quat, backend):
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return backend.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _root_error_components(sim_qpos, sim_qvel, ref_qpos, ref_qvel, init_ref_qpos, backend):
    ref_xyz = ref_qpos[:3]
    init_xy = init_ref_qpos[:2]
    offset_xyz = backend.concatenate([init_xy, backend.zeros(1, dtype=init_xy.dtype)])
    aligned_ref_xyz = ref_xyz - offset_xyz
    root_pos_error = aligned_ref_xyz - sim_qpos[:3]
    root_vel_error = ref_qvel[:6] - sim_qvel[:6]
    yaw_error = _yaw_from_wxyz_quat(ref_qpos[3:7], backend) - _yaw_from_wxyz_quat(sim_qpos[3:7], backend)
    yaw_error = backend.arctan2(backend.sin(yaw_error), backend.cos(yaw_error))
    return backend.concatenate([root_pos_error, root_vel_error, backend.atleast_1d(yaw_error)])
```

- [ ] **Step 4: Wire the flag into `GoalTrajMimic`**

In `GoalTrajMimic.__init__`, after `use_concise_lookahead`, add:

```python
self.include_current_root_error = bool(
    kwargs.pop("include_current_root_error", info_props.get("include_current_root_error", False))
)
```

In `_init_from_mj`, after computing `motion_phase_dim`, add the root error dimension:

```python
root_error_dim = 10 if self.include_current_root_error else 0
self._dim += root_error_dim
```

In `get_obs_and_update_state`, before final `goal_components` assembly, compute:

```python
root_error_obs = None
if self.include_current_root_error:
    current_ref_data = env.th.get_current_traj_data(carry, backend)
    init_ref_data = env.th.get_init_traj_data(carry, backend)
    root_error_obs = _root_error_components(
        data.qpos[root_qpos_ind],
        data.qvel[root_qvel_ind],
        current_ref_data.qpos[root_qpos_ind],
        current_ref_data.qvel[root_qvel_ind],
        init_ref_data.qpos[root_qpos_ind],
        backend,
    )
```

Then append it immediately before `traj_goal_obs` in both goal assembly branches:

```python
if root_error_obs is not None:
    goal_components.append(root_error_obs)
```

For the branch where `len(self._rel_site_ids) == 0`, build `goal_components = []`, append `root_error_obs` when present, then append `traj_goal_obs`.

- [ ] **Step 5: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_goal_traj_root_error.py tests/test_n_step_lookahead.py -q
```

Expected: PASS. If `tests/test_n_step_lookahead.py` is absent in this checkout, run:

```bash
.venv/bin/python -m pytest tests/unit/test_n_step_wrapper.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add musclemimic/core/goals/trajectory.py tests/unit/test_goal_traj_root_error.py
git commit -m "feat: add current root error goal signal"
```

## Task 4: Absolute Site Reward

**Files:**
- Modify: `musclemimic/core/reward/trajectory_based.py`
- Modify: `tests/unit/test_mimic_reward.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/unit/test_mimic_reward.py`:

```python
def test_mimic_reward_absolute_site_reward_resolves_site_ids():
    model = mujoco.MjModel.from_xml_string(MINIMAL_MJCF)
    traj_data = make_traj_data([0.0] * 8)
    th = FakeTrajectoryHandler(traj_data)
    env = make_env(model, th)

    reward = MimicReward(
        env,
        absolute_site_reward_sites=["child_mimic"],
        absolute_site_w_sum=0.1,
        absolute_site_w_exp=10.0,
    )

    assert reward._absolute_site_w_sum == 0.1
    assert reward._absolute_site_ids.tolist() == [2]


def test_mimic_reward_absolute_site_reward_rejects_missing_site():
    model = mujoco.MjModel.from_xml_string(MINIMAL_MJCF)
    traj_data = make_traj_data([0.0] * 8)
    th = FakeTrajectoryHandler(traj_data)
    env = make_env(model, th)

    with pytest.raises(ValueError, match="absolute site reward site not found"):
        MimicReward(env, absolute_site_reward_sites=["missing_mimic"], absolute_site_w_sum=0.1)


def test_mimic_reward_absolute_site_reward_decreases_with_site_error():
    model = mujoco.MjModel.from_xml_string(MINIMAL_MJCF)
    qpos = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    qvel = np.zeros(7, dtype=np.float32)
    ref_data = make_traj_data(qpos, qvel=qvel)
    ref_data.site_xpos = np.array(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.1], [1.0, 0.0, 1.2]],
        dtype=np.float32,
    )
    th = FakeTrajectoryHandler(ref_data)
    env = make_env(model, th)
    reward = MimicReward(
        env,
        qpos_w_sum=0.0,
        qvel_w_sum=0.0,
        root_pos_w_sum=0.0,
        root_vel_w_sum=0.0,
        rpos_w_sum=0.0,
        rquat_w_sum=0.0,
        rvel_w_sum=0.0,
        absolute_site_reward_sites=["child_mimic"],
        absolute_site_w_sum=1.0,
        absolute_site_w_exp=10.0,
    )
    carry = make_carry()
    carry = carry.replace(reward_state=reward.init_state(env, None, model, make_sim_data(qpos, qvel=qvel), np))
    matched_data = make_sim_data(qpos, qvel=qvel)
    matched_data.site_xpos = ref_data.site_xpos.copy()
    offset_data = make_sim_data(qpos, qvel=qvel)
    offset_data.site_xpos = ref_data.site_xpos.copy()
    offset_data.site_xpos[2, 0] += 0.5

    matched_reward, _, matched_info = reward(None, np.zeros(3), None, False, {}, env, model, matched_data, carry, np)
    offset_reward, _, offset_info = reward(None, np.zeros(3), None, False, {}, env, model, offset_data, carry, np)

    assert matched_reward > offset_reward
    assert matched_info["reward_absolute_site"] > offset_info["reward_absolute_site"]
```

Also add `import pytest` near the imports if it is not already present.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_mimic_reward.py -q
```

Expected: FAIL because `MimicReward` has no absolute site reward fields.

- [ ] **Step 3: Implement initialization**

In `MimicReward.__init__`, after action/energy coefficients, add:

```python
self._absolute_site_w_sum = float(kwargs.get("absolute_site_w_sum", 0.0))
self._absolute_site_w_exp = float(kwargs.get("absolute_site_w_exp", 10.0))
self._absolute_site_names = list(kwargs.get("absolute_site_reward_sites", []))
self._absolute_site_ids = np.array([], dtype=int)
```

After `_rel_site_ids` is initialized, add:

```python
if self._absolute_site_w_sum > 0.0:
    if not self._absolute_site_names:
        raise ValueError("absolute_site_w_sum > 0 requires absolute_site_reward_sites")
    absolute_site_ids = []
    for site_name in self._absolute_site_names:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise ValueError(f"absolute site reward site not found: {site_name}")
        absolute_site_ids.append(site_id)
    self._absolute_site_ids = np.array(absolute_site_ids, dtype=int)
```

- [ ] **Step 4: Implement reward computation**

In `MimicReward.__call__`, before total reward calculation, add:

```python
absolute_site_reward = 0.0
raw_absolute_site_dist = 0.0
if self._absolute_site_w_sum > 0.0 and self._absolute_site_ids.size > 0:
    current_abs_sites = data.site_xpos[self._absolute_site_ids]
    if self._site_mapper.requires_mapping:
        ref_abs_indices = self._site_mapper.model_ids_to_traj_indices(self._absolute_site_ids)
        ref_abs_sites = traj_data_single.site_xpos[ref_abs_indices]
    else:
        ref_abs_sites = traj_data_single.site_xpos[self._absolute_site_ids]
    if xy_offset is not None:
        if offset_xyz is None:
            offset_xyz = backend.concatenate([xy_offset, backend.zeros(1, dtype=xy_offset.dtype)])
        ref_abs_sites = ref_abs_sites - offset_xyz
    raw_absolute_site_dist = backend.mean(backend.square(current_abs_sites - ref_abs_sites))
    absolute_site_reward = backend.exp(-self._absolute_site_w_exp * raw_absolute_site_dist)
```

Add the raw distance to `imitation_error_total`:

```python
+ self._absolute_site_w_sum * raw_absolute_site_dist
```

Add the reward to `total_reward`:

```python
total_reward = total_reward + self._absolute_site_w_sum * absolute_site_reward
```

Add diagnostics to `reward_info`:

```python
"reward_absolute_site": absolute_site_reward,
"err_absolute_site": backend.sqrt(raw_absolute_site_dist),
```

- [ ] **Step 5: Run focused reward tests**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_mimic_reward.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add musclemimic/core/reward/trajectory_based.py tests/unit/test_mimic_reward.py
git commit -m "feat: add optional absolute site mimic reward"
```

## Task 5: Root-First Config Contract

**Files:**
- Create: `fullbody/config_specific_task/conf_fullbody_forehand_net_lift_root_first.yaml`
- Create: `tests/unit/test_forehand_net_lift_root_first_config.py`

- [ ] **Step 1: Write failing config test**

Create `tests/unit/test_forehand_net_lift_root_first_config.py`:

```python
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "fullbody" / "config_specific_task" / "conf_fullbody_forehand_net_lift_root_first.yaml"


def test_root_first_config_exists_and_uses_current_data_paths():
    cfg = OmegaConf.load(CONFIG)
    paths = list(cfg.experiment.task_factory.params.amass_dataset_conf.rel_dataset_path)

    assert paths == [
        "ForehandNetLift/best/video01_best_stage7_smpl",
        "ForehandNetLift/best/video02_best_stage7_smpl",
        "ForehandNetLift/best/video03_best_stage7_smpl",
        "ForehandNetLift/best/video04_best_stage7_smpl",
        "ForehandNetLift/best/video05_best_stage7_smpl",
        "ForehandNetLift/best/video06_best_stage7_smpl",
        "ForehandNetLift/best/video07_best_stage5_smpl",
        "ForehandNetLift/best/video08_best_stage5_smpl",
    ]


def test_root_first_config_uses_root_heavy_reward_and_strict_validation():
    cfg = OmegaConf.load(CONFIG)
    reward = cfg.experiment.env_params.reward_params
    validation = cfg.experiment.validation

    assert reward.root_pos_w_sum > reward.qpos_w_sum
    assert reward.root_vel_w_sum > reward.qvel_w_sum
    assert reward.absolute_site_reward_sites == ["right_hand_mimic"]
    assert validation.terminal_state_type == "MeanRelativeSiteDeviationWithRootTerminalStateHandler"
    assert validation.terminal_state_params.root_deviation_threshold == 0.30


def test_root_first_config_points_to_existing_checkpoint_root():
    cfg = OmegaConf.load(CONFIG)
    checkpoint_root = Path(cfg.experiment.checkpoint_root)

    assert checkpoint_root == REPO_ROOT / "checkpoints" / "ForehandNetLift" / "forehand_net_lift_best_ppo"
    assert cfg.experiment.resume_from == str(checkpoint_root / "checkpoint_7812")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_forehand_net_lift_root_first_config.py -q
```

Expected: FAIL because the config file does not exist.

- [ ] **Step 3: Create the root-first config**

Create `fullbody/config_specific_task/conf_fullbody_forehand_net_lift_root_first.yaml`:

```yaml
# @package _global_

defaults:
  - /conf_fullbody_gmr
  - _self_

hydra:
  job:
    env_set:
      MUSCLEMIMIC_AMASS_PATH: /data3/yangfeiyang/WorkSpace/musclemimic/BadmintonMimic/data/amass_npz
      AMASS_PATH: /data3/yangfeiyang/WorkSpace/musclemimic/BadmintonMimic/data/amass_npz
      MUSCLEMIMIC_CONVERTED_AMASS_PATH: /data3/yangfeiyang/WorkSpace/musclemimic/caches/AMASS
      CONVERTED_AMASS_PATH: /data3/yangfeiyang/WorkSpace/musclemimic/caches/AMASS
      MUSCLEMIMIC_SMPL_MODEL_PATH: /data3/yangfeiyang/WorkSpace/musclemimic/smpl_models/smplh
      SMPL_MODEL_PATH: /data3/yangfeiyang/WorkSpace/musclemimic/smpl_models/smplh
      MPLCONFIGDIR: /tmp/matplotlib
      XDG_CACHE_HOME: /tmp
      XLA_PYTHON_CLIENT_PREALLOCATE: "false"

wandb:
  project: "musclemimic"
  mode: "online"
  tags: ["fullbody", "gmr", "badminton", "forehand_net_lift", "root_first", "posttrain"]

experiment:
  env_params:
    env_name: MjxMyoFullBody
    num_envs: 4096
    disable_fingers: true
    goal_params:
      include_current_root_error: false
    reward_params:
      qpos_w_sum: 0.05
      qvel_w_sum: 0.08
      root_pos_w_sum: 0.35
      root_vel_w_sum: 0.25
      rpos_w_sum: 0.30
      rquat_w_sum: 0.01
      rvel_w_sum: 0.06
      absolute_site_reward_sites:
        - right_hand_mimic
      absolute_site_w_sum: 0.10
      absolute_site_w_exp: 10.0
    terminal_state_type: MeanRelativeSiteDeviationWithRootTerminalStateHandler
    terminal_state_params:
      mean_site_deviation_threshold: 0.45
      root_deviation_threshold: 0.30
      root_orientation_threshold: 0.70
      enable_site_check: true

  checkpoint_root: /data3/yangfeiyang/WorkSpace/musclemimic/checkpoints/ForehandNetLift/forehand_net_lift_best_ppo
  resume_from: /data3/yangfeiyang/WorkSpace/musclemimic/checkpoints/ForehandNetLift/forehand_net_lift_best_ppo/checkpoint_7812
  reset_lr_schedule_on_resume: true
  reset_logging_timestep: false
  lr: 5e-5
  total_timesteps: 200000000

  ppo_config:
    num_steps: 64
    update_epochs: 2
    num_minibatches: 128
    init_std: 1.0
    ent_coef: 0.0

  adaptive_sampling:
    enabled: false
  adaptive_termination:
    enabled: false
  reward_curriculum:
    enabled: false
  asi:
    enabled: false

  task_factory:
    params:
      amass_dataset_conf:
        dataset_group: null
        rel_dataset_path:
          - "ForehandNetLift/best/video01_best_stage7_smpl"
          - "ForehandNetLift/best/video02_best_stage7_smpl"
          - "ForehandNetLift/best/video03_best_stage7_smpl"
          - "ForehandNetLift/best/video04_best_stage7_smpl"
          - "ForehandNetLift/best/video05_best_stage7_smpl"
          - "ForehandNetLift/best/video06_best_stage7_smpl"
          - "ForehandNetLift/best/video07_best_stage5_smpl"
          - "ForehandNetLift/best/video08_best_stage5_smpl"
        retargeting_method: gmr
        clear_cache: false
        gmr_config:
          src_human: smplh
          target_fps: 100
          solver: daqp
          damping: 1.0
          offset_to_ground: false
          use_velocity_limit: true
          use_fitted_shape: true
          shape_fitting_iterations: 500
          ik_config_path: "loco_mujoco/smpl/gmr_configs/smplh_to_myofullbody_smooth_train.json"

  validation:
    active: true
    deterministic: true
    num_steps: 500
    num_envs: 8
    num: 8
    video_length: 500
    video_frequency: 1
    terminal_state_type: MeanRelativeSiteDeviationWithRootTerminalStateHandler
    terminal_state_params:
      mean_site_deviation_threshold: 0.45
      root_deviation_threshold: 0.30
      root_orientation_threshold: 0.70
      enable_site_check: true
    amass_dataset_conf:
      dataset_group: null
      rel_dataset_path:
        - "ForehandNetLift/best/video01_best_stage7_smpl"
        - "ForehandNetLift/best/video02_best_stage7_smpl"
        - "ForehandNetLift/best/video03_best_stage7_smpl"
        - "ForehandNetLift/best/video04_best_stage7_smpl"
        - "ForehandNetLift/best/video05_best_stage7_smpl"
        - "ForehandNetLift/best/video06_best_stage7_smpl"
        - "ForehandNetLift/best/video07_best_stage5_smpl"
        - "ForehandNetLift/best/video08_best_stage5_smpl"
      retargeting_method: gmr
      clear_cache: false
      gmr_config:
        src_human: smplh
        target_fps: 100
        solver: daqp
        damping: 1.0
        offset_to_ground: false
        use_velocity_limit: true
        use_fitted_shape: true
        shape_fitting_iterations: 500
        ik_config_path: "loco_mujoco/smpl/gmr_configs/smplh_to_myofullbody_smooth_train.json"
```

- [ ] **Step 4: Run config test**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_forehand_net_lift_root_first_config.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add fullbody/config_specific_task/conf_fullbody_forehand_net_lift_root_first.yaml tests/unit/test_forehand_net_lift_root_first_config.py
git commit -m "config: add root-first ForehandNetLift post-train"
```

## Task 6: Checkpoint Evaluation Commands

**Files:**
- Modify: `docs/superpowers/plans/2026-05-21-root-first-forehand-net-lift-posttrain.md`

- [ ] **Step 1: Run the baseline checkpoint metrics**

Run:

```bash
.venv/bin/python fullbody/eval.py \
  --path /data3/yangfeiyang/WorkSpace/musclemimic/checkpoints/ForehandNetLift/forehand_net_lift_best_ppo/checkpoint_7812 \
  --metrics \
  --metrics_only \
  --motion_path \
    ForehandNetLift/best/video01_best_stage7_smpl \
    ForehandNetLift/best/video02_best_stage7_smpl \
    ForehandNetLift/best/video03_best_stage7_smpl \
    ForehandNetLift/best/video04_best_stage7_smpl \
    ForehandNetLift/best/video05_best_stage7_smpl \
    ForehandNetLift/best/video06_best_stage7_smpl \
    ForehandNetLift/best/video07_best_stage5_smpl \
    ForehandNetLift/best/video08_best_stage5_smpl \
  --terminal_state_type MeanRelativeSiteDeviationWithRootTerminalStateHandler \
  --mean_site_deviation_threshold 0.45 \
  --root_deviation_threshold 0.30 \
  --root_orientation_threshold 0.70 \
  --metrics_envs 8 \
  --metrics_deterministic \
  --eval_seed 0 \
  --no_render
```

Expected: evaluation completes and prints validation metrics. Save the terminal output path or W&B run ID in the execution notes before training.

- [ ] **Step 2: Run a single rollout video for visual baseline**

Run:

```bash
.venv/bin/python fullbody/eval.py \
  --path /data3/yangfeiyang/WorkSpace/musclemimic/checkpoints/ForehandNetLift/forehand_net_lift_best_ppo/checkpoint_7812 \
  --use_mujoco \
  --record \
  --record_dir outputs/root_first_forehand_net_lift/baseline_video \
  --motion_path ForehandNetLift/best/video01_best_stage7_smpl \
  --traj_index 0 \
  --traj_start_step 0 \
  --terminal_state_type MeanRelativeSiteDeviationWithRootTerminalStateHandler \
  --mean_site_deviation_threshold 0.45 \
  --root_deviation_threshold 0.30 \
  --root_orientation_threshold 0.70 \
  --n_steps 500
```

Expected: a baseline video is written under `outputs/root_first_forehand_net_lift/baseline_video`.

## Task 7: Short Root-First Post-Train Smoke Run

**Files:**
- No new files

- [ ] **Step 1: Run a short smoke train**

Run:

```bash
wandb.mode=disabled \
.venv/bin/python fullbody/experiment.py \
  --config-name=config_specific_task/conf_fullbody_forehand_net_lift_root_first \
  experiment.total_timesteps=4096000 \
  experiment.checkpoint_root=/data3/yangfeiyang/WorkSpace/musclemimic/checkpoints/ForehandNetLift/root_first_smoke \
  experiment.auto_resume=false \
  experiment.env_params.num_envs=512 \
  experiment.num_envs=512 \
  experiment.validation.num_envs=4 \
  experiment.validation.num=4
```

Expected: training starts from `checkpoint_7812`, completes at least one PPO update, and writes a new checkpoint under `/data3/yangfeiyang/WorkSpace/musclemimic/checkpoints/ForehandNetLift/root_first_smoke`. If checkpoint restore fails because `goal_params.include_current_root_error` changes shape, confirm this config has `include_current_root_error: false`; then rerun the same command.

- [ ] **Step 2: Run post-train metrics on the new checkpoint**

```bash
.venv/bin/python fullbody/eval.py \
  --path /data3/yangfeiyang/WorkSpace/musclemimic/checkpoints/ForehandNetLift/root_first_smoke \
  --metrics \
  --metrics_only \
  --motion_path \
    ForehandNetLift/best/video01_best_stage7_smpl \
    ForehandNetLift/best/video02_best_stage7_smpl \
    ForehandNetLift/best/video03_best_stage7_smpl \
    ForehandNetLift/best/video04_best_stage7_smpl \
    ForehandNetLift/best/video05_best_stage7_smpl \
    ForehandNetLift/best/video06_best_stage7_smpl \
    ForehandNetLift/best/video07_best_stage5_smpl \
    ForehandNetLift/best/video08_best_stage5_smpl \
  --terminal_state_type MeanRelativeSiteDeviationWithRootTerminalStateHandler \
  --mean_site_deviation_threshold 0.45 \
  --root_deviation_threshold 0.30 \
  --root_orientation_threshold 0.70 \
  --metrics_envs 8 \
  --metrics_deterministic \
  --eval_seed 0 \
  --no_render
```

Expected: metrics complete. Compare root-related metrics and video against the baseline checkpoint before scaling training.

## Task 8: Full Verification

**Files:**
- No new files

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/unit/test_root_tracking_metrics.py \
  tests/unit/test_diagnose_root_tracking.py \
  tests/unit/test_goal_traj_root_error.py \
  tests/unit/test_mimic_reward.py \
  tests/unit/test_forehand_net_lift_root_first_config.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run existing regression tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/unit/test_n_step_wrapper.py \
  tests/unit/test_ppo_config.py \
  tests/unit/test_enhanced_fullbody_terminal_handler.py \
  -q
```

Expected: PASS.

- [ ] **Step 3: Inspect git diff before final handoff**

Run:

```bash
git status --short
git log --oneline -5
```

Expected: only intentional changes from this plan are staged or committed. Existing unrelated untracked files such as `doc/present_and_todo_suggestions.md` remain untouched.

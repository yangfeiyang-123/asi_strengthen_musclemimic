# Stage5 Smooth Retarget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a smooth, training-oriented MyoFullBody retarget pipeline for the 10 stage5 forehand-clear demos without overwriting the current baseline caches.

**Architecture:** Keep SMPL/AMASS inputs unchanged and create a separate smooth motion namespace. Add small GMR configuration hooks so `run_retarget.py` can select a custom IK mapping, enable velocity limits, and use higher damping. Validate with unit tests plus end-to-end retarget, visualization, and discontinuity metrics.

**Tech Stack:** Python, argparse, NumPy, MuJoCo/GMR retargeting, pytest, existing BadmintonMimic scripts.

---

### Task 1: Add Retarget CLI Config Hooks

**Files:**
- Modify: `musclemimic/badminton/scripts/run_retarget.py`
- Test: `tests/unit/test_badminton_smooth_retarget_config.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_badminton_smooth_retarget_config.py`:

```python
from pathlib import Path

from BadmintonMimic.scripts.run_retarget import _build_gmr_config


def test_build_gmr_config_defaults_keep_baseline_behavior():
    config = _build_gmr_config(target_fps=60, damping=None, use_velocity_limit=False, ik_config_path=None)

    assert config["target_fps"] == 60
    assert config["damping"] == 0.5
    assert config["use_velocity_limit"] is False
    assert "ik_config_path" not in config


def test_build_gmr_config_accepts_smooth_overrides():
    config = _build_gmr_config(
        target_fps=60,
        damping=1.0,
        use_velocity_limit=True,
        ik_config_path=Path("loco_mujoco/smpl/gmr_configs/smplh_to_myofullbody_smooth_train.json"),
    )

    assert config["target_fps"] == 60
    assert config["damping"] == 1.0
    assert config["use_velocity_limit"] is True
    assert config["ik_config_path"] == "loco_mujoco/smpl/gmr_configs/smplh_to_myofullbody_smooth_train.json"
```

- [ ] **Step 2: Run failing test**

Run: `.venv/bin/python -m pytest tests/unit/test_badminton_smooth_retarget_config.py -q`

Expected: FAIL because `_build_gmr_config` does not exist.

- [ ] **Step 3: Implement config builder and CLI args**

In `musclemimic/badminton/scripts/run_retarget.py`, add `_build_gmr_config()` and parser args:

```python
def _build_gmr_config(
    target_fps: int,
    damping: float | None,
    use_velocity_limit: bool,
    ik_config_path: Path | None,
) -> dict:
    config = {
        "src_human": "smplh",
        "target_fps": target_fps,
        "solver": "daqp",
        "damping": 0.5 if damping is None else float(damping),
        "offset_to_ground": False,
        "use_velocity_limit": bool(use_velocity_limit),
        "use_fitted_shape": True,
        "shape_fitting_iterations": 500,
    }
    if ik_config_path is not None:
        config["ik_config_path"] = str(ik_config_path)
    return config
```

Parser args:

```python
parser.add_argument("--damping", type=float, default=None)
parser.add_argument("--use-velocity-limit", action="store_true")
parser.add_argument("--gmr-ik-config", type=Path, default=None)
```

Use `_build_gmr_config()` in `main()`.

- [ ] **Step 4: Run passing test**

Run: `.venv/bin/python -m pytest tests/unit/test_badminton_smooth_retarget_config.py -q`

Expected: PASS.

### Task 2: Support Custom MyoFullBody GMR IK Config

**Files:**
- Modify: `loco_mujoco/smpl/retargeting.py`

- [ ] **Step 1: Add config path resolution**

In `fit_gmr_motion()`, after `shape_fitting_iterations`, read:

```python
ik_config_path = gmr_config.get("ik_config_path")
```

Where `myofullbody_ik_config` is currently hard-coded, replace it with:

```python
if ik_config_path is None:
    myofullbody_ik_config = Path(__file__).parent / "gmr_configs" / "smplh_to_myofullbody.json"
else:
    myofullbody_ik_config = Path(ik_config_path)
    if not myofullbody_ik_config.is_absolute():
        myofullbody_ik_config = (Path.cwd() / myofullbody_ik_config).resolve()
```

Keep the existing existence check.

- [ ] **Step 2: Fix dict access for equality constraint weight**

Change:

```python
equality_constraint_weight = getattr(gmr_config, "equality_constraint_weight", 5.0)
```

to:

```python
equality_constraint_weight = gmr_config.get("equality_constraint_weight", 5.0)
```

- [ ] **Step 3: Syntax check**

Run: `.venv/bin/python -m py_compile loco_mujoco/smpl/retargeting.py musclemimic/badminton/scripts/run_retarget.py`

Expected: exit 0.

### Task 3: Add Smooth GMR Mapping

**Files:**
- Create: `loco_mujoco/smpl/gmr_configs/smplh_to_myofullbody_smooth_train.json`

- [ ] **Step 1: Copy baseline mapping**

Copy `loco_mujoco/smpl/gmr_configs/smplh_to_myofullbody.json` to `loco_mujoco/smpl/gmr_configs/smplh_to_myofullbody_smooth_train.json`.

- [ ] **Step 2: Lower upper-limb end-effector tracking weights**

In both `ik_match_table1` and `ik_match_table2`, set wrist and elbow position weights lower than baseline:

```json
"ulna_l": ["left_elbow", 15, 5, [0, 0, 0], [-0.5, -0.5, 0.5, -0.5]],
"lunate_l": ["left_wrist", 8, 3, [0, 0, 0], [-0.5, -0.5, 0.5, -0.5]],
"ulna_r": ["right_elbow", 15, 5, [0, 0, 0], [-0.5, 0.5, 0.5, 0.5]],
"lunate_r": ["right_wrist", 8, 3, [0, 0, 0], [-0.5, 0.5, 0.5, 0.5]]
```

Keep pelvis/head/trunk/lower-body weights unchanged.

- [ ] **Step 3: Validate JSON**

Run: `.venv/bin/python -m json.tool loco_mujoco/smpl/gmr_configs/smplh_to_myofullbody_smooth_train.json >/tmp/smooth_gmr_config.json`

Expected: exit 0.

### Task 4: Create Smooth AMASS Namespace and Manifest

**Files:**
- Create: `manifests/stage5_10demo_smooth_list.txt`
- Create files under: `musclemimic/badminton/data/amass_npz/forehand_clear/stage5_10demo_smooth/`

- [ ] **Step 1: Copy 10 AMASS NPZ files**

Copy:

```text
musclemimic/badminton/data/amass_npz/forehand_clear/stage5_10demo/video*_lower_body_full_poses.npz
```

to:

```text
musclemimic/badminton/data/amass_npz/forehand_clear/stage5_10demo_smooth/video*_lower_body_full_poses.npz
```

- [ ] **Step 2: Write smooth manifest**

`manifests/stage5_10demo_smooth_list.txt`:

```text
forehand_clear/stage5_10demo_smooth/video1_lower_body_full_poses
forehand_clear/stage5_10demo_smooth/video2_lower_body_full_poses
forehand_clear/stage5_10demo_smooth/video3_lower_body_full_poses
forehand_clear/stage5_10demo_smooth/video4_lower_body_full_poses
forehand_clear/stage5_10demo_smooth/video5_lower_body_full_poses
forehand_clear/stage5_10demo_smooth/video6_lower_body_full_poses
forehand_clear/stage5_10demo_smooth/video7_lower_body_full_poses
forehand_clear/stage5_10demo_smooth/video8_lower_body_full_poses
forehand_clear/stage5_10demo_smooth/video9_lower_body_full_poses
forehand_clear/stage5_10demo_smooth/video10_lower_body_full_poses
```

- [ ] **Step 3: Verify smooth input FPS**

Run a Python check that all 10 copied files have `mocap_framerate = mocap_frame_rate = 60`.

Expected: all pass.

### Task 5: Retarget, Visualize, and Compare

**Files:**
- Create caches under: `caches/AMASS/MyoFullBody/gmr/forehand_clear/stage5_10demo_smooth/`
- Create videos under: `visualize/msk_retarget_smooth/`
- Create report: `visualize/msk_retarget_smooth/discontinuity_report.md`

- [ ] **Step 1: Run smooth retarget**

Run:

```bash
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES="" \
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/matplotlib \
.venv/bin/python musclemimic/badminton/scripts/run_retarget.py \
  --manifest manifests/stage5_10demo_smooth_list.txt \
  --fps 60 \
  --clear-cache \
  --use-velocity-limit \
  --damping 1.0 \
  --gmr-ik-config loco_mujoco/smpl/gmr_configs/smplh_to_myofullbody_smooth_train.json
```

Expected: `[OK] Retarget cache generation complete`.

- [ ] **Step 2: Render smooth videos**

Run `musclemimic/badminton/scripts/render_retarget_cache.py` for all 10 `forehand_clear/stage5_10demo_smooth/video*_lower_body_full_poses` motions with `--sample-fps 60`, `--stride 1`, `--width 640`, `--height 480`, output dir `visualize/msk_retarget_smooth`.

Expected: 10 mp4 files.

- [ ] **Step 3: Compare discontinuity metrics**

Run the same metric script used for the baseline report, comparing baseline namespace `stage5_10demo` to smooth namespace `stage5_10demo_smooth`.

Expected: hand-site speed and shoulder qpos-step maxima are lower in smooth outputs.

- [ ] **Step 4: Report final status**

Summarize changed files, cache/video paths, and the metric comparison. If the smooth version does not improve the spikes, stop and report the actual metrics before trying another fix.

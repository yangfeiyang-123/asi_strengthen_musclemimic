# ForehandClear Grip-Hold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a no-shuttle ForehandClear grip-hold post-train path that uses the local ForehandClear checkpoint as the body prior and trains/validates a right-hand residual controller to keep the racket held during the swing.

**Architecture:** Add a new `forehand_clear_grip_hold` runner path separate from `static_hit_staging`. The first implementation layer produces a validated experiment spec and diagnostic reset/rollout artifacts; later tasks add the residual controller and PPO loop once the reset, checkpoint, and grip metrics are proven to work in isolation.

**Tech Stack:** Python, PyTorch for residual PPO, MuJoCo CPU rendering for diagnostics/video, existing fullbody checkpoint/eval utilities for ForehandClear policy loading, W&B for metrics/video.

---

### Task 1: Experiment Spec And Prepare Support

**Files:**
- Create: `experiments/posttrain/forehand_clear_grip_hold_v1.yaml`
- Modify: `musclemimic/badminton/scripts/run_posttrain_experiment.py`
- Test: `tests/unit/test_forehand_clear_grip_hold_spec.py`

- [ ] **Step 1: Write the failing spec loader test**

Add `tests/unit/test_forehand_clear_grip_hold_spec.py`:

```python
from pathlib import Path

import pytest
import yaml

from BadmintonMimic.scripts.run_posttrain_experiment import (
    load_spec,
    prepare,
    requires_dedicated_static_hit_runner,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = REPO_ROOT / "BadmintonMimic" / "experiments" / "posttrain" / "forehand_clear_grip_hold_v1.yaml"


def test_forehand_clear_grip_hold_spec_uses_existing_local_checkpoint():
    data = load_spec(SPEC)

    assert data["action"] == "ForehandClearGripHold"
    assert data["runner_type"] == "forehand_clear_grip_hold"
    assert data["resume_from"] == str(REPO_ROOT / "checkpoints" / "de63059b16c0" / "checkpoint_7812")
    assert Path(data["resume_from"]).is_dir()
    assert data["shuttle"]["enabled"] is False
    assert data["grip_seed"]["path"] == "outputs/right_hand_racket_grip/reference/right_hand_racket_grip_seed.json"
    assert requires_dedicated_static_hit_runner(data) is False


def test_prepare_writes_forehand_clear_grip_hold_handoff(tmp_path):
    spec = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    spec["output_root"] = str(tmp_path / "outputs")
    local_spec = tmp_path / "spec.yaml"
    local_spec.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

    result = prepare(local_spec)

    handoff = result.output_dir / "commands" / "README_forehand_clear_grip_hold.txt"
    assert handoff.is_file()
    text = handoff.read_text(encoding="utf-8")
    assert "forehand_clear_grip_hold" in text
    assert "no shuttle" in text.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_forehand_clear_grip_hold_spec.py -q
```

Expected: fails because the new spec file and runner handling do not exist.

- [ ] **Step 3: Add the YAML spec**

Create `experiments/posttrain/forehand_clear_grip_hold_v1.yaml`:

```yaml
experiment_id: v1
action: ForehandClearGripHold
runner_type: forehand_clear_grip_hold
description: No-shuttle ForehandClear post-train for holding a racket through the swing.
output_root: outputs/posttrain
resume_from: /data3/yangfeiyang/WorkSpace/musclemimic/checkpoints/de63059b16c0/checkpoint_7812

reference:
  train:
    - forehand_clear/stage5_10demo/video1_lower_body_full_poses
  validation:
    - forehand_clear/stage5_10demo/video2_lower_body_full_poses

scene:
  xml: environment/overall_environment/assets/overall_badminton_scene.xml
  face_net: true
  shuttle_enabled: false

grip_seed:
  path: outputs/right_hand_racket_grip/reference/right_hand_racket_grip_seed.json

body_policy:
  checkpoint: /data3/yangfeiyang/WorkSpace/musclemimic/checkpoints/de63059b16c0/checkpoint_7812
  trainable: false

residual_policy:
  trainable: true
  actuator_groups:
    stage1: [right_hand_fingers]
    stage2: [right_hand_fingers, right_wrist, right_forearm]

shuttle:
  enabled: false

training:
  total_steps: 50000
  rollout_steps: 512
  validation_video_interval_steps: 10000
  validation_video_steps: 120
  wandb_project: musclemimic
  wandb_mode: online

reward:
  mimic: 1.0
  root_stability: 1.0
  grip_site: 8.0
  contact: 2.0
  no_slip: 8.0
  no_penetration: 10.0
  racket_hand_pose: 4.0
  residual_effort: 0.01

validation:
  require_finite: true
  no_fall: true
  min_contacts_stage1: 2
  min_contacts_final: 4
  max_handle_penetration_m: 0.003
  max_grip_slip_m: 0.05

arms:
  - id: E0_diagnostic_replay
    type: baseline
    description: Reset Overall with the grip seed and record no-training diagnostics.
  - id: E1_stage1_short_grip_hold
    type: posttrain
    description: Train right-hand finger residuals on short no-shuttle ForehandClear windows.
    stage: stage1
```

- [ ] **Step 4: Add prepare handoff for the new runner type**

In `musclemimic/badminton/scripts/run_posttrain_experiment.py`, add a helper:

```python
FOREHAND_CLEAR_GRIP_HOLD_RUNNER = "forehand_clear_grip_hold"


def requires_dedicated_grip_hold_runner(spec: dict[str, Any]) -> bool:
    return spec.get("runner_type") == FOREHAND_CLEAR_GRIP_HOLD_RUNNER
```

Update `prepare()` so specs with this runner write `commands/README_forehand_clear_grip_hold.txt` instead of ordinary fullbody train commands:

```python
def _write_grip_hold_handoff(output_dir: Path, spec: dict[str, Any]) -> None:
    commands_dir = output_dir / "commands"
    commands_dir.mkdir(parents=True, exist_ok=True)
    text = (
        "# ForehandClear Grip-Hold runner handoff\n\n"
        "This experiment uses runner_type: forehand_clear_grip_hold.\n"
        "It is a no shuttle residual grip-hold experiment and must be run by the "
        "dedicated grip-hold runner, not by fullbody/experiment.py.\n\n"
        f"Base checkpoint: {spec['resume_from']}\n"
        f"Grip seed: {spec['grip_seed']['path']}\n"
        f"Scene: {spec['scene']['xml']}\n"
    )
    (commands_dir / "README_forehand_clear_grip_hold.txt").write_text(text, encoding="utf-8")
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_forehand_clear_grip_hold_spec.py tests/unit/test_forehand_clear_static_hit_spec.py -q
```

Expected: all selected tests pass.

Commit:

```bash
git add experiments/posttrain/forehand_clear_grip_hold_v1.yaml musclemimic/badminton/scripts/run_posttrain_experiment.py tests/unit/test_forehand_clear_grip_hold_spec.py
git commit -m "feat: stage forehand clear grip hold experiment"
```

### Task 2: Diagnostic Grip-Hold Reset Runner

**Files:**
- Create: `musclemimic/badminton/scripts/run_forehand_clear_grip_hold.py`
- Test: `tests/unit/test_forehand_clear_grip_hold_runner.py`

- [ ] **Step 1: Write failing tests for preflight and diagnostic output**

Add tests that call pure functions without loading the full policy:

```python
from pathlib import Path

import yaml

from BadmintonMimic.scripts.run_forehand_clear_grip_hold import (
    GripHoldPaths,
    load_grip_hold_spec,
    preflight,
)


def test_load_grip_hold_spec_resolves_paths():
    spec_path = Path("experiments/posttrain/forehand_clear_grip_hold_v1.yaml")
    paths = load_grip_hold_spec(spec_path)

    assert paths.runner_type == "forehand_clear_grip_hold"
    assert paths.resume_from.is_dir()
    assert paths.scene_xml.is_file()
    assert paths.grip_seed.is_file()


def test_preflight_writes_report(tmp_path):
    spec_path = Path("experiments/posttrain/forehand_clear_grip_hold_v1.yaml")
    paths = load_grip_hold_spec(spec_path)
    report = preflight(paths, out_dir=tmp_path)

    assert report["runner_type"] == "forehand_clear_grip_hold"
    assert report["checkpoint_exists"] is True
    assert report["scene_exists"] is True
    assert report["grip_seed_exists"] is True
    assert (tmp_path / "preflight_report.json").is_file()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_forehand_clear_grip_hold_runner.py -q
```

Expected: import fails because the runner module does not exist.

- [ ] **Step 3: Implement path loading and preflight**

Create `musclemimic/badminton/scripts/run_forehand_clear_grip_hold.py` with:

```python
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class GripHoldPaths:
    spec_path: Path
    runner_type: str
    resume_from: Path
    scene_xml: Path
    grip_seed: Path
    output_dir: Path


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def load_grip_hold_spec(spec_path: str | Path) -> GripHoldPaths:
    spec_path = _resolve(spec_path)
    data = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{spec_path} must contain a mapping")
    if data.get("runner_type") != "forehand_clear_grip_hold":
        raise ValueError(f"unsupported runner_type: {data.get('runner_type')!r}")
    output_dir = _resolve(data.get("output_root", "outputs/posttrain")) / data["action"] / data["experiment_id"]
    return GripHoldPaths(
        spec_path=spec_path,
        runner_type=str(data["runner_type"]),
        resume_from=_resolve(data["resume_from"]),
        scene_xml=_resolve(data["scene"]["xml"]),
        grip_seed=_resolve(data["grip_seed"]["path"]),
        output_dir=output_dir,
    )


def preflight(paths: GripHoldPaths, *, out_dir: str | Path | None = None) -> dict[str, Any]:
    out_path = Path(out_dir) if out_dir is not None else paths.output_dir
    out_path.mkdir(parents=True, exist_ok=True)
    report = {
        "runner_type": paths.runner_type,
        "spec_path": str(paths.spec_path),
        "resume_from": str(paths.resume_from),
        "scene_xml": str(paths.scene_xml),
        "grip_seed": str(paths.grip_seed),
        "checkpoint_exists": paths.resume_from.is_dir(),
        "scene_exists": paths.scene_xml.is_file(),
        "grip_seed_exists": paths.grip_seed.is_file(),
    }
    (out_path / "preflight_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
```

- [ ] **Step 4: Add CLI**

Append:

```python
def main() -> int:
    parser = argparse.ArgumentParser(description="Run ForehandClear grip-hold diagnostics and training.")
    parser.add_argument("--spec", default="experiments/posttrain/forehand_clear_grip_hold_v1.yaml")
    parser.add_argument("--stage", choices=("preflight",), default="preflight")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    paths = load_grip_hold_spec(args.spec)
    report = preflight(paths, out_dir=args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run tests, run preflight, commit**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_forehand_clear_grip_hold_runner.py -q
.venv/bin/python musclemimic/badminton/scripts/run_forehand_clear_grip_hold.py --stage preflight
```

Expected: test passes and `outputs/posttrain/ForehandClearGripHold/v1/preflight_report.json` reports all required files exist.

Commit:

```bash
git add musclemimic/badminton/scripts/run_forehand_clear_grip_hold.py tests/unit/test_forehand_clear_grip_hold_runner.py outputs/posttrain/ForehandClearGripHold/v1/preflight_report.json
git commit -m "feat: add forehand clear grip hold preflight"
```

### Task 3: Reset Diagnostic Video

**Files:**
- Modify: `musclemimic/badminton/scripts/run_forehand_clear_grip_hold.py`
- Test: `tests/unit/test_forehand_clear_grip_hold_runner.py`

- [ ] **Step 1: Write failing test with mocked renderer**

Add a test that monkeypatches `record_reset_video()` and verifies the CLI/report path records a video under `diagnostics/`.

- [ ] **Step 2: Implement `record_reset_video()`**

Use MuJoCo CPU rendering to load `paths.scene_xml`, set `MUJOCO_GL=egl` before importing `mujoco`, reset to keyframe `overall_ready` when present, and write:

```text
outputs/posttrain/ForehandClearGripHold/v1/diagnostics/reset_grip_hold.mp4
```

- [ ] **Step 3: Run real reset video smoke**

Run:

```bash
.venv/bin/python musclemimic/badminton/scripts/run_forehand_clear_grip_hold.py --stage preflight
```

Expected: an MP4 is created and `file` reports `ISO Media, MP4`.

- [ ] **Step 4: Commit**

Commit runner/test changes:

```bash
git add musclemimic/badminton/scripts/run_forehand_clear_grip_hold.py tests/unit/test_forehand_clear_grip_hold_runner.py outputs/posttrain/ForehandClearGripHold/v1/diagnostics/reset_grip_hold.mp4
git commit -m "feat: record forehand clear grip hold reset video"
```

### Task 4: Frozen Policy Replay Interface

**Files:**
- Modify: `musclemimic/badminton/scripts/run_forehand_clear_grip_hold.py`
- Test: `tests/unit/test_forehand_clear_grip_hold_runner.py`

- [ ] **Step 1: Add tests for checkpoint metadata extraction**

Add a test that reads `checkpoint/de63059b16c0/checkpoint_7812/config/metadata` and verifies the metadata includes `forehand_clear`.

- [ ] **Step 2: Add metadata extraction function**

Implement:

```python
def checkpoint_metadata(checkpoint_dir: Path) -> dict[str, Any]:
    metadata_path = checkpoint_dir / "config" / "metadata"
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    return data
```

- [ ] **Step 3: Add replay command that fails clearly until the action adapter exists**

Implement a `--stage replay` entry that loads metadata and writes a diagnostic report, but exits with a clear message if the full policy action adapter is not implemented. This prevents users from accidentally thinking training is running.

- [ ] **Step 4: Commit**

Commit:

```bash
git add musclemimic/badminton/scripts/run_forehand_clear_grip_hold.py tests/unit/test_forehand_clear_grip_hold_runner.py
git commit -m "feat: inspect forehand clear base checkpoint for grip hold"
```

### Task 5: Residual Controller Design Gate

**Files:**
- Create: `docs/forehand_clear_grip_hold.md`

- [ ] **Step 1: Document current runnable commands**

Include:

```bash
.venv/bin/python musclemimic/badminton/scripts/run_posttrain_experiment.py \
  --spec experiments/posttrain/forehand_clear_grip_hold_v1.yaml \
  --stage prepare

.venv/bin/python musclemimic/badminton/scripts/run_forehand_clear_grip_hold.py \
  --spec experiments/posttrain/forehand_clear_grip_hold_v1.yaml \
  --stage preflight
```

- [ ] **Step 2: State the remaining residual-policy blocker**

Document that full residual PPO requires connecting the ForehandClear checkpoint action output to the Overall racket scene and mapping right-hand residual actuator IDs.

- [ ] **Step 3: Commit**

```bash
git add docs/forehand_clear_grip_hold.md
git commit -m "docs: add forehand clear grip hold workflow"
```

---

## Self-Review Notes

- This plan covers the approved v1 scope through a runnable diagnostic/preflight path and avoids stopping the current pure grip run.
- It intentionally does not promise full residual PPO in the first implementation batch because the checkpoint-to-Overall action adapter must be verified before training.
- All code tasks include tests before implementation and concrete commands.

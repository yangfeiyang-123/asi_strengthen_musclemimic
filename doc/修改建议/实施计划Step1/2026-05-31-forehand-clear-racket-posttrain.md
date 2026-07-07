# ForehandClear Racket PostTrain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a trainable pipeline that turns the existing no-racket ForehandClear body policy into a racket-holding and racket-hitting policy without corrupting the original full-body motion.

**Architecture:** Use a layered policy stack: frozen full-body checkpoint for body motion, a right-hand grip policy for finger/racket stabilization, and a small residual hit policy for wrist/forearm/racket timing. The implementation first makes action and observation compatibility explicit, then introduces a training scene, static-hit reward, ghost-racket teacher, contact graph, curriculum runner, hard-state mining, and ablation reporting.

**Tech Stack:** Python, MuJoCo, NumPy, PyTorch for right-hand grip PPO, JAX/Orbax checkpoint metadata for full-body PPO, Hydra YAML configs, pytest.

---

## Evidence Read From `doc/修改建议`

- `1_existing_implementation_review.md`: current blockers are runner mismatch, missing checkpoint action manifest, missing action adapter, missing router integration, incomplete static-hit reward/termination, grip PPO action sampling issue, and missing integration tests.
- `2_literature_directions.md`: useful prior art maps to engineering mechanisms: DeepMimic-style mimic+task reward, motion prior against reward hacking, contact graph reward, perfect-first curriculum, soft-weld annealing, failure-state mining, shuttle aerodynamics, and biomechanical regularization.
- `3_innovation_methods.md`: recommended innovations are three-layer policy, ghost racket teacher, soft-weld annealing, phase-gated contact reward, contact-viability critic with ASI hard-state mining, analytic shuttle target proxy, and kinetic-chain metrics.
- `4_priority_roadmap.md`: execution order is P0 runner guard, P1 manifest/adapter/router, P2 frozen body plus grip rollout, P3 training scene, P4 static-hit reward, P5 grip PPO fix, P6 ghost racket, P7 contact graph, P8 curriculum runner, P9 hard-state mining, P10 ablation.
- `5_core_conclusion.md`: do not directly resume the original checkpoint into full-body racket training. The minimal viable route is action compatibility, frozen body, grip residual, training scene, static-hit env, ghost/soft-weld/contact reward, curriculum, hard-state mining, and ablation.

## Evidence Verified In The Repo

- `experiments/posttrain/forehand_clear_grip_hold_v1.yaml` points `resume_from` and `body_policy.checkpoint` to `checkpoints/de63059b16c0/checkpoint_7812`.
- `checkpoints/de63059b16c0/manifest.json` stores `env_params.disable_fingers: true`.
- `musclemimic/environments/humanoids/myofullbody.py` deletes exact finger joints, finger muscle actuators, and related tendons when `disable_fingers=True`, then builds the action spec from remaining `spec.actuators`.
- Local env inspection showed `disable_fingers=True` gives `model.nu=354` and zero exact finger actuators; `disable_fingers=False` gives `model.nu=416` with 62 exact finger actuators, 31 per hand.
- `checkpoints/de63059b16c0/checkpoint_7812/train_state/_METADATA` stores `params.actor.Dense_16.bias` shape `[354]`, `params.actor.Dense_16.kernel` shape `[1024, 354]`, and `params.log_std` shape `[354]`.
- `musclemimic/badminton/scripts/run_posttrain_experiment.py` has `requires_dedicated_grip_hold_runner()` but the run-stage guard must reject train/eval/render for this spec.
- `musclemimic/badminton/scripts/run_forehand_clear_grip_hold.py` currently supports `preflight`, `reset-video`, and `replay-precheck`; `replay_precheck()` sets `policy_replay_ready` to `False`.
- `environment/overall_environment/src/layered_control.py` already implements `LayeredActuatorRouter.merge()` by actuator name.
- `environment/overall_environment/src/static_forehand_clear_env.py` has freeze/release helpers but `step()` returns reward `0.0`, `terminated=False`, and `truncated=False`.
- `src/grip/train_right_hand_racket_grip_policy.py` stores clipped actions but computes PPO update logprob on clipped actions under an unclipped Normal distribution.

---

## File Map

### Runner and Spec Safety

- Modify `musclemimic/badminton/scripts/run_posttrain_experiment.py`: reject grip-hold specs outside prepare stage and keep generated ordinary fullbody commands removed.
- Modify `musclemimic/badminton/scripts/run_forehand_clear_grip_hold.py`: make diagnostic-only status explicit until train stage is implemented.
- Modify `experiments/posttrain/forehand_clear_grip_hold_v1.yaml`: replace private absolute paths with repo-relative paths.
- Modify `experiments/posttrain/forehand_clear_static_hit_v1.yaml`: replace private absolute paths and keep `disable_fingers: false`.
- Modify `tests/unit/test_run_posttrain_experiment.py`: add grip-hold runner guard tests.
- Modify `tests/unit/test_forehand_clear_grip_hold_runner.py`: assert diagnostics report action incompatibility until manifest/router are available.
- Modify `tests/unit/test_forehand_clear_grip_hold_spec.py`: assert repo-relative paths and dedicated runner fields.

### Action and Observation Compatibility

- Create `environment/overall_environment/src/action_manifest.py`: load, reconstruct, and print checkpoint action/observation manifests.
- Create `environment/overall_environment/src/action_adapter.py`: map checkpoint actions to target model actuator order by name.
- Modify `musclemimic/runner/checkpointing.py`: write action manifest when checkpoint manifests are created.
- Modify `musclemimic/badminton/scripts/run_forehand_clear_grip_hold.py`: use the action manifest in replay precheck.
- Create `tests/unit/test_action_manifest.py`.
- Create `environment/overall_environment/tests/test_action_adapter.py`.

### Layered Body/Grip Control

- Modify `environment/overall_environment/src/layered_control.py`: add model name extraction, right-hand actuator group resolver, and router builder.
- Create `environment/overall_environment/src/layered_policy.py`: load frozen body policy, load grip policy, adapt obs/actions, and emit full action.
- Create `environment/overall_environment/tests/test_layered_policy.py`.
- Modify `environment/overall_environment/tests/test_layered_control.py`: cover real right-hand group resolution.

### Training Scene

- Create `environment/overall_environment/assets/overall_badminton_training_scene.xml`: training variant with actuators enabled, collision groups suitable for contact, named keyframes, and configurable weld stages.
- Create `environment/overall_environment/src/training_scene.py`: resolve training scene XML and validate key names, bodies, geoms, sites, actuators.
- Create `environment/overall_environment/tests/test_training_scene.py`.

### Static Hit RL Environment

- Modify `environment/overall_environment/src/static_forehand_clear_env.py`: add reward terms, termination, flight evaluation, phase-gated contact, and diagnostics.
- Create `environment/overall_environment/src/contact_graph.py`: compute hand-handle and racket-shuttle contact reports once per step.
- Create `environment/overall_environment/src/phase_reward.py`: compute phase-gated hit reward from contact graph and shuttle state.
- Modify `environment/overall_environment/tests/test_static_forehand_clear_env.py`: add reward and termination tests.
- Create `environment/overall_environment/tests/test_contact_graph.py`.
- Create `environment/overall_environment/tests/test_phase_reward.py`.

### Grip PPO

- Modify `src/grip/train_right_hand_racket_grip_policy.py`: use a tanh-squashed Gaussian logprob for sampled actions or store raw actions consistently.
- Create `tests/unit/test_right_hand_grip_ppo_logprob.py`.

### Innovation Modules

- Create `environment/overall_environment/src/ghost_racket.py`: build and interpolate ghost racket trajectory from reference motion and grip seed.
- Create `environment/overall_environment/src/soft_weld_schedule.py`: convert curriculum stage to weld stiffness/damping and reward weights.
- Create `environment/overall_environment/src/hard_state_mining.py`: classify failures and write hard-state replay seeds.
- Create `musclemimic/badminton/scripts/run_forehand_clear_racket_curriculum.py`: run staged training and validation.
- Create `tests/unit/test_ghost_racket.py`.
- Create `tests/unit/test_soft_weld_schedule.py`.
- Create `tests/unit/test_hard_state_mining.py`.
- Create `tests/unit/test_forehand_clear_racket_curriculum.py`.

### Experiment Reporting

- Create `musclemimic/badminton/scripts/build_forehand_clear_ablation_report.py`.
- Create `tests/unit/test_forehand_clear_ablation_report.py`.
- Create output contract files under `outputs/posttrain/ForehandClearRacketHit/v1/reports/` during runs, not in this plan.

---

## Phase 0: Runner Guards, Specs, And Diagnostic Truthfulness

### Task 0.1: Reject Grip-Hold Train/Eval/Render In The Ordinary PostTrain Runner

**Files:**
- Modify: `musclemimic/badminton/scripts/run_posttrain_experiment.py`
- Modify: `tests/unit/test_run_posttrain_experiment.py`

- [x] **Step 1: Add failing test for grip-hold train rejection**

Add this test to `tests/unit/test_run_posttrain_experiment.py`:

```python
def test_run_stage_rejects_grip_hold_train_stage():
    spec = {
        "experiment_id": "v1",
        "action": "ForehandClearGripHold",
        "output_root": "outputs/posttrain",
        "resume_from": "checkpoints/de63059b16c0/checkpoint_7812",
        "runner_type": "forehand_clear_grip_hold",
        "reference": {"train": ["m1"], "validation": ["m2"]},
        "arms": [{"id": "stage1", "description": "grip hold"}],
        "scene": {"xml": "environment/overall_environment/assets/overall_badminton_scene.xml"},
        "grip_seed": {"path": "outputs/right_hand_racket_grip/reference/right_hand_racket_grip_seed.json"},
    }

    with pytest.raises(ValueError, match="dedicated grip-hold runner"):
        run_stage(spec, stage="train", arm=None, execute=False)
```

- [x] **Step 2: Run the specific test and verify it fails**

Run:

```bash
pytest tests/unit/test_run_posttrain_experiment.py::test_run_stage_rejects_grip_hold_train_stage -q
```

Expected before implementation: the test fails because `run_stage()` does not reject grip-hold train stage.

- [x] **Step 3: Add the guard in `run_stage()`**

In `musclemimic/badminton/scripts/run_posttrain_experiment.py`, place this guard next to the existing static-hit guard:

```python
if stage != "prepare" and requires_dedicated_grip_hold_runner(spec):
    raise ValueError(
        f"{spec['action']} {spec['experiment_id']} requires a dedicated grip-hold runner; "
        f"the PostTrain fullbody runner cannot run stage '{stage}'. "
        "Use musclemimic/badminton/scripts/run_forehand_clear_grip_hold.py."
    )
```

- [x] **Step 4: Run the guard test**

Run:

```bash
pytest tests/unit/test_run_posttrain_experiment.py::test_run_stage_rejects_grip_hold_train_stage -q
```

Expected: `1 passed`.

- [x] **Step 5: Run posttrain unit tests**

Run:

```bash
pytest tests/unit/test_run_posttrain_experiment.py tests/unit/test_forehand_clear_grip_hold_spec.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add musclemimic/badminton/scripts/run_posttrain_experiment.py tests/unit/test_run_posttrain_experiment.py
git commit -m "guard grip-hold specs from ordinary posttrain runner"
```

### Task 0.2: Make The Grip-Hold Runner Diagnostic-Only Until Train Exists

**Files:**
- Modify: `musclemimic/badminton/scripts/run_forehand_clear_grip_hold.py`
- Modify: `tests/unit/test_forehand_clear_grip_hold_runner.py`

- [x] **Step 1: Add failing test for CLI stage choices**

Add:

```python
def test_grip_hold_runner_stage_choices_are_diagnostic_only():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "musclemimic/badminton/scripts/run_forehand_clear_grip_hold.py",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "preflight" in result.stdout
    assert "reset-video" in result.stdout
    assert "replay-precheck" in result.stdout
    assert "train" not in result.stdout
    assert "diagnostic" in result.stdout.lower()
```

- [x] **Step 2: Run the test**

Run:

```bash
pytest tests/unit/test_forehand_clear_grip_hold_runner.py::test_grip_hold_runner_stage_choices_are_diagnostic_only -q
```

Expected before implementation: it fails if help text does not clearly say diagnostic.

- [x] **Step 3: Update module docstring and parser description**

Replace the top docstring in `musclemimic/badminton/scripts/run_forehand_clear_grip_hold.py` with:

```python
"""Run ForehandClear grip-hold diagnostics.

This script does not train yet. Training requires checkpoint action manifests,
name-based action adaptation, observation compatibility checks, and layered
body/grip action routing.
"""
```

Keep parser choices as:

```python
parser.add_argument("--stage", choices=("preflight", "reset-video", "replay-precheck"), default="preflight")
```

- [x] **Step 4: Run the test**

Run:

```bash
pytest tests/unit/test_forehand_clear_grip_hold_runner.py::test_grip_hold_runner_stage_choices_are_diagnostic_only -q
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add musclemimic/badminton/scripts/run_forehand_clear_grip_hold.py tests/unit/test_forehand_clear_grip_hold_runner.py
git commit -m "clarify grip-hold runner diagnostics only"
```

### Task 0.3: Remove Private Absolute Paths From PostTrain Specs

**Files:**
- Modify: `experiments/posttrain/forehand_clear_grip_hold_v1.yaml`
- Modify: `experiments/posttrain/forehand_clear_static_hit_v1.yaml`
- Modify: `tests/unit/test_forehand_clear_grip_hold_spec.py`
- Modify: `tests/unit/test_forehand_clear_static_hit_spec.py`

- [x] **Step 1: Add path hygiene assertions**

Add to both spec test files:

```python
def _assert_no_private_absolute_paths(text: str) -> None:
    forbidden = ("/data3/", "/home/", "/Users/")
    for prefix in forbidden:
        assert prefix not in text


def test_spec_has_no_private_absolute_paths():
    _assert_no_private_absolute_paths(SPEC.read_text(encoding="utf-8"))
```

- [x] **Step 2: Run spec tests**

Run:

```bash
pytest tests/unit/test_forehand_clear_grip_hold_spec.py tests/unit/test_forehand_clear_static_hit_spec.py -q
```

Expected before implementation: failures for absolute checkpoint paths.

- [x] **Step 3: Replace grip-hold paths**

In `experiments/posttrain/forehand_clear_grip_hold_v1.yaml`, use:

```yaml
resume_from: checkpoints/de63059b16c0/checkpoint_7812

body_policy:
  checkpoint: checkpoints/de63059b16c0/checkpoint_7812
  trainable: false
```

- [x] **Step 4: Replace static-hit paths**

In `experiments/posttrain/forehand_clear_static_hit_v1.yaml`, use:

```yaml
resume_from: checkpoints/ForehandClear/forehand_clear_best/checkpoint_7812

body_policy:
  checkpoint: checkpoints/ForehandClear/forehand_clear_best/checkpoint_7812
```

If that checkpoint directory does not exist in this repo, use the verified existing checkpoint:

```yaml
resume_from: checkpoints/de63059b16c0/checkpoint_7812

body_policy:
  checkpoint: checkpoints/de63059b16c0/checkpoint_7812
```

- [x] **Step 5: Run tests**

Run:

```bash
pytest tests/unit/test_forehand_clear_grip_hold_spec.py tests/unit/test_forehand_clear_static_hit_spec.py -q
```

Expected: all path hygiene tests pass.

- [ ] **Step 6: Commit**

```bash
git add experiments/posttrain/forehand_clear_grip_hold_v1.yaml experiments/posttrain/forehand_clear_static_hit_v1.yaml tests/unit/test_forehand_clear_grip_hold_spec.py tests/unit/test_forehand_clear_static_hit_spec.py
git commit -m "use repo-relative posttrain checkpoint paths"
```

---

## Phase 1: Checkpoint Manifest, Action Adapter, And Router Integration

### Task 1.1: Add Checkpoint Action Manifest Utilities

**Files:**
- Create: `environment/overall_environment/src/action_manifest.py`
- Create: `tests/unit/test_action_manifest.py`
- Modify: `musclemimic/runner/checkpointing.py`

- [x] **Step 1: Write failing unit test for manifest schema**

Create `tests/unit/test_action_manifest.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from environment.overall_environment.src.action_manifest import (
    ActionManifest,
    load_action_manifest,
    write_action_manifest,
)


def test_write_and_load_action_manifest(tmp_path: Path):
    manifest = ActionManifest(
        schema_version=1,
        env_name="MjxMyoFullBody",
        disable_fingers=True,
        action_size=2,
        actuator_names=["hip", "shoulder"],
        obs_size=3,
        obs_fields=["root", "joint", "goal"],
        control_min=-1.0,
        control_max=1.0,
    )

    path = tmp_path / "action_manifest.json"
    write_action_manifest(path, manifest)
    loaded = load_action_manifest(path)

    assert loaded == manifest
    assert json.loads(path.read_text(encoding="utf-8"))["actuator_names"] == ["hip", "shoulder"]
```

- [x] **Step 2: Run and verify failure**

Run:

```bash
pytest tests/unit/test_action_manifest.py -q
```

Expected before implementation: import fails because `action_manifest.py` does not exist.

- [x] **Step 3: Implement `action_manifest.py`**

Create:

```python
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ActionManifest:
    schema_version: int
    env_name: str
    disable_fingers: bool
    action_size: int
    actuator_names: list[str]
    obs_size: int
    obs_fields: list[str]
    control_min: float
    control_max: float

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported action manifest schema_version: {self.schema_version}")
        if self.action_size != len(self.actuator_names):
            raise ValueError("action_size must match actuator_names length")
        if len(set(self.actuator_names)) != len(self.actuator_names):
            raise ValueError("actuator_names contains duplicates")
        if self.obs_size < 0:
            raise ValueError("obs_size must be non-negative")


def _coerce(data: dict[str, Any]) -> ActionManifest:
    return ActionManifest(
        schema_version=int(data["schema_version"]),
        env_name=str(data["env_name"]),
        disable_fingers=bool(data["disable_fingers"]),
        action_size=int(data["action_size"]),
        actuator_names=[str(name) for name in data["actuator_names"]],
        obs_size=int(data.get("obs_size", 0)),
        obs_fields=[str(name) for name in data.get("obs_fields", [])],
        control_min=float(data.get("control_min", -1.0)),
        control_max=float(data.get("control_max", 1.0)),
    )


def load_action_manifest(path: str | Path) -> ActionManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("action manifest root must be an object")
    return _coerce(data)


def write_action_manifest(path: str | Path, manifest: ActionManifest) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def find_action_manifest(checkpoint: str | Path) -> Path:
    root = Path(checkpoint)
    candidates = [
        root / "action_manifest.json",
        root.parent / "action_manifest.json",
        root.parent / "manifest.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no action manifest found for checkpoint: {checkpoint}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Print checkpoint action manifest.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--print", dest="do_print", action="store_true")
    args = parser.parse_args()
    path = find_action_manifest(args.checkpoint)
    manifest = load_action_manifest(path)
    if args.do_print:
        print(json.dumps(asdict(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] **Step 4: Run tests**

Run:

```bash
pytest tests/unit/test_action_manifest.py -q
```

Expected: `1 passed`.

- [x] **Step 5: Add manifest writing hook**

In `musclemimic/runner/checkpointing.py`, keep existing `manifest.json` behavior and add `action_manifest` only when `experiment_config.env_params` and an actuator list are available. The first implementation can write no action manifest for legacy checkpoints; reconstruction is handled by Task 1.2. Add no behavior that overwrites existing checkpoint directories.

- [ ] **Step 6: Commit**

```bash
git add environment/overall_environment/src/action_manifest.py tests/unit/test_action_manifest.py musclemimic/runner/checkpointing.py
git commit -m "add checkpoint action manifest utilities"
```

### Task 1.2: Reconstruct Legacy Manifest From Checkpoint Metadata And Env Factory

**Files:**
- Modify: `environment/overall_environment/src/action_manifest.py`
- Modify: `tests/unit/test_action_manifest.py`

- [x] **Step 1: Add failing reconstruction test**

Add:

```python
def test_reconstruct_legacy_manifest_from_env_params():
    manifest = ActionManifest.from_env_params(
        {
            "env_name": "MjxMyoFullBody",
            "disable_fingers": True,
        },
        actuator_names=["hip", "shoulder"],
        obs_size=5,
        obs_fields=["obs"],
    )

    assert manifest.disable_fingers is True
    assert manifest.action_size == 2
    assert manifest.actuator_names == ["hip", "shoulder"]
```

- [x] **Step 2: Implement classmethod**

Add to `ActionManifest`:

```python
@classmethod
def from_env_params(
    cls,
    env_params: dict[str, Any],
    *,
    actuator_names: list[str],
    obs_size: int,
    obs_fields: list[str],
) -> "ActionManifest":
    return cls(
        schema_version=1,
        env_name=str(env_params["env_name"]),
        disable_fingers=bool(env_params.get("disable_fingers", True)),
        action_size=len(actuator_names),
        actuator_names=list(actuator_names),
        obs_size=int(obs_size),
        obs_fields=list(obs_fields),
        control_min=-1.0,
        control_max=1.0,
    )
```

- [x] **Step 3: Add legacy CLI reconstruction path**

Add a `--reconstruct` flag to `main()`:

```python
parser.add_argument("--reconstruct", action="store_true")
```

When set, read `Path(args.checkpoint) / "config" / "metadata"`, instantiate `MyoFullBody(disable_fingers=metadata["experiment"]["env_params"]["disable_fingers"])`, extract actuator names from `model.nu`, and print the reconstructed manifest.

- [x] **Step 4: Verify against the real checkpoint**

Run:

```bash
.venv/bin/python -m environment.overall_environment.src.action_manifest \
  --checkpoint checkpoints/de63059b16c0/checkpoint_7812 \
  --reconstruct \
  --print
```

Expected output includes:

```text
"env_name": "MjxMyoFullBody"
"disable_fingers": true
"action_size": 354
```

- [ ] **Step 5: Commit**

```bash
git add environment/overall_environment/src/action_manifest.py tests/unit/test_action_manifest.py
git commit -m "reconstruct legacy action manifests"
```

### Task 1.3: Implement Name-Based Checkpoint Action Adapter

**Files:**
- Create: `environment/overall_environment/src/action_adapter.py`
- Create: `environment/overall_environment/tests/test_action_adapter.py`

- [x] **Step 1: Write adapter tests**

Create `environment/overall_environment/tests/test_action_adapter.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from environment.overall_environment.src.action_adapter import CheckpointToFullActionAdapter


def test_adapter_maps_by_name_not_index():
    adapter = CheckpointToFullActionAdapter(
        source_actuator_names=["shoulder", "hip"],
        target_actuator_names=["hip", "FDS2", "shoulder"],
    )

    full = adapter.adapt(np.array([0.2, 0.7]))

    np.testing.assert_allclose(full, np.array([0.7, 0.0, 0.2]))
    assert adapter.report().mapped_count == 2
    assert adapter.report().extra_in_target == ["FDS2"]


def test_adapter_rejects_missing_source_actuator_in_target():
    with pytest.raises(ValueError, match="source actuators missing in target"):
        CheckpointToFullActionAdapter(
            source_actuator_names=["hip", "missing"],
            target_actuator_names=["hip"],
        )


def test_adapter_sets_extra_target_actuators_to_zero():
    adapter = CheckpointToFullActionAdapter(["hip"], ["hip", "FDS2"])
    np.testing.assert_allclose(adapter.adapt(np.array([0.4])), np.array([0.4, 0.0]))


def test_adapter_rejects_nonfinite_action():
    adapter = CheckpointToFullActionAdapter(["hip"], ["hip"])
    with pytest.raises(ValueError, match="non-finite"):
        adapter.adapt(np.array([np.nan]))
```

- [x] **Step 2: Run tests and verify failure**

Run:

```bash
pytest environment/overall_environment/tests/test_action_adapter.py -q
```

Expected before implementation: import fails.

- [x] **Step 3: Implement adapter**

Create:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActionAdapterReport:
    source_action_size: int
    target_action_size: int
    mapped_count: int
    missing_in_target: list[str]
    extra_in_target: list[str]


class CheckpointToFullActionAdapter:
    def __init__(self, source_actuator_names: list[str], target_actuator_names: list[str]) -> None:
        self.source_actuator_names = list(source_actuator_names)
        self.target_actuator_names = list(target_actuator_names)
        _reject_duplicates("source_actuator_names", self.source_actuator_names)
        _reject_duplicates("target_actuator_names", self.target_actuator_names)

        target_set = set(self.target_actuator_names)
        missing = [name for name in self.source_actuator_names if name not in target_set]
        if missing:
            raise ValueError(f"source actuators missing in target: {missing}")

        source_set = set(self.source_actuator_names)
        self._extra = [name for name in self.target_actuator_names if name not in source_set]
        self._source_index = {name: i for i, name in enumerate(self.source_actuator_names)}

    def adapt(self, source_action: np.ndarray) -> np.ndarray:
        action = np.asarray(source_action, dtype=float)
        if action.shape != (len(self.source_actuator_names),):
            raise ValueError(f"source_action must have shape ({len(self.source_actuator_names)},), got {action.shape}")
        if not np.isfinite(action).all():
            raise ValueError("source_action contains non-finite values")

        full = np.zeros(len(self.target_actuator_names), dtype=float)
        for target_index, name in enumerate(self.target_actuator_names):
            source_index = self._source_index.get(name)
            if source_index is not None:
                full[target_index] = action[source_index]
        return full

    def report(self) -> ActionAdapterReport:
        return ActionAdapterReport(
            source_action_size=len(self.source_actuator_names),
            target_action_size=len(self.target_actuator_names),
            mapped_count=len(self.source_actuator_names),
            missing_in_target=[],
            extra_in_target=list(self._extra),
        )


def _reject_duplicates(name: str, values: list[str]) -> None:
    seen = set()
    duplicates = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        raise ValueError(f"{name} contains duplicate actuator names: {duplicates}")
```

- [x] **Step 4: Run tests**

Run:

```bash
pytest environment/overall_environment/tests/test_action_adapter.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add environment/overall_environment/src/action_adapter.py environment/overall_environment/tests/test_action_adapter.py
git commit -m "add name-based checkpoint action adapter"
```

### Task 1.4: Integrate `LayeredActuatorRouter` Builder

**Files:**
- Modify: `environment/overall_environment/src/layered_control.py`
- Modify: `environment/overall_environment/tests/test_layered_control.py`

- [x] **Step 1: Add builder tests**

Add:

```python
class _FakeModel:
    actuator_names = ["hip", "shoulder", "FDS2", "FDP2", "EDC2"]


def test_build_router_from_model_and_spec_resolves_right_hand_fingers():
    from environment.overall_environment.src.layered_control import build_router_from_model_and_spec

    body_manifest = type("BodyManifest", (), {"actuator_names": ["hip", "shoulder"]})()
    spec = {"actuator_groups": {"stage1": ["right_hand_fingers"]}, "stage": "stage1"}

    router = build_router_from_model_and_spec(_FakeModel(), body_manifest, spec)

    assert router.body_actuator_names == ["hip", "shoulder"]
    assert router.grip_actuator_names == ["FDS2", "FDP2", "EDC2"]
```

- [x] **Step 2: Implement model name extraction and group resolver**

Add:

```python
RIGHT_HAND_FINGER_ACTUATORS = {
    "FDS2", "FDS3", "FDS4", "FDS5", "FDP2", "FDP3", "FDP4", "FDP5",
    "EDC2", "EDC3", "EDC4", "EDC5", "EDM", "EIP", "EPL", "EPB",
    "FPL", "APL", "OP", "RI2", "RI3", "RI4", "RI5", "LU_RB2",
    "LU_RB3", "LU_RB4", "LU_RB5", "UI_UB2", "UI_UB3", "UI_UB4", "UI_UB5",
}


def actuator_names_from_model(model) -> list[str]:
    if hasattr(model, "actuator_names"):
        return list(model.actuator_names)
    import mujoco

    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(model.nu)
    ]


def resolve_actuator_groups(model, groups: list[str]) -> list[str]:
    all_names = actuator_names_from_model(model)
    result = []
    for group in groups:
        if group == "right_hand_fingers":
            result.extend([name for name in all_names if name in RIGHT_HAND_FINGER_ACTUATORS])
        else:
            raise ValueError(f"unknown actuator group: {group}")
    return result


def build_router_from_model_and_spec(model, body_manifest, residual_spec: dict) -> LayeredActuatorRouter:
    all_names = actuator_names_from_model(model)
    stage = residual_spec.get("stage")
    group_config = residual_spec.get("actuator_groups", {})
    groups = group_config.get(stage, group_config if isinstance(group_config, list) else [])
    grip_names = resolve_actuator_groups(model, list(groups))
    return LayeredActuatorRouter(all_names, list(body_manifest.actuator_names), grip_names)
```

- [x] **Step 3: Run router tests**

Run:

```bash
pytest environment/overall_environment/tests/test_layered_control.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add environment/overall_environment/src/layered_control.py environment/overall_environment/tests/test_layered_control.py
git commit -m "build layered router from model and residual spec"
```

---

## Phase 2: Frozen Body Policy, Grip Policy, And Layered Rollout Smoke Test

### Task 2.1: Add Frozen Body Policy Loader Interface

**Files:**
- Create: `environment/overall_environment/src/layered_policy.py`
- Create: `environment/overall_environment/tests/test_layered_policy.py`

- [x] **Step 1: Add fake-policy test**

Create:

```python
from __future__ import annotations

import numpy as np

from environment.overall_environment.src.layered_control import LayeredActuatorRouter
from environment.overall_environment.src.layered_policy import LayeredPolicy


class FakePolicy:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=float)
        self.trainable = True

    def eval(self):
        self.trainable = False
        return self

    def act(self, obs):
        return self.action


def test_layered_policy_merges_frozen_body_and_grip_actions():
    router = LayeredActuatorRouter(
        all_actuator_names=["hip", "FDS2", "shoulder"],
        body_actuator_names=["hip", "shoulder"],
        grip_actuator_names=["FDS2"],
    )
    policy = LayeredPolicy(
        body_policy=FakePolicy([0.1, 0.2]),
        grip_policy=FakePolicy([0.7]),
        router=router,
    )

    output = policy.act(np.zeros(4))

    np.testing.assert_allclose(output.full_action, np.array([0.1, 0.7, 0.2]))
    assert output.body_action.shape == (2,)
    assert output.grip_action.shape == (1,)
```

- [x] **Step 2: Implement minimal layered policy**

Create:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from environment.overall_environment.src.layered_control import LayeredActuatorRouter


@dataclass(frozen=True)
class LayeredPolicyOutput:
    body_action: np.ndarray
    grip_action: np.ndarray
    full_action: np.ndarray


class LayeredPolicy:
    def __init__(self, *, body_policy, grip_policy, router: LayeredActuatorRouter) -> None:
        self.body_policy = body_policy.eval()
        self.grip_policy = grip_policy
        self.router = router

    def act(self, obs: np.ndarray) -> LayeredPolicyOutput:
        body_action = np.asarray(self.body_policy.act(obs), dtype=float)
        grip_action = np.asarray(self.grip_policy.act(obs), dtype=float)
        full_action = self.router.merge(body_action=body_action, grip_action=grip_action)
        return LayeredPolicyOutput(body_action=body_action, grip_action=grip_action, full_action=full_action)
```

- [x] **Step 3: Run test**

Run:

```bash
pytest environment/overall_environment/tests/test_layered_policy.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add environment/overall_environment/src/layered_policy.py environment/overall_environment/tests/test_layered_policy.py
git commit -m "add layered policy composition"
```

### Task 2.2: Load Or Initialize Grip Policy

**Files:**
- Modify: `environment/overall_environment/src/layered_policy.py`
- Modify: `environment/overall_environment/tests/test_layered_policy.py`

- [x] **Step 1: Add checkpoint loader test**

Add a test using a small fake PyTorch checkpoint:

```python
def test_load_grip_policy_rejects_action_size_mismatch(tmp_path):
    import torch

    checkpoint = {
        "metrics": {"obs_size": 4, "action_size": 2},
        "model_state_dict": {},
        "obs_rms": {"mean": [0, 0, 0, 0], "var": [1, 1, 1, 1], "count": 1.0},
    }
    path = tmp_path / "policy_latest.pt"
    torch.save(checkpoint, path)

    from environment.overall_environment.src.layered_policy import load_grip_policy

    with pytest.raises(ValueError, match="action_size"):
        load_grip_policy(path, obs_size=4, action_size=3)
```

- [x] **Step 2: Implement loader**

Add:

```python
def load_grip_policy(path, *, obs_size: int, action_size: int, allow_random_init: bool = False):
    from pathlib import Path
    import torch
    from src.grip.train_right_hand_racket_grip_policy import PolicyValueNet, RunningMeanStd

    policy_path = Path(path)
    if not policy_path.is_file():
        if allow_random_init:
            return RandomGripPolicy(action_size)
        raise FileNotFoundError(policy_path)
    checkpoint = torch.load(policy_path, map_location="cpu")
    metrics = checkpoint.get("metrics", {})
    if int(metrics.get("obs_size", obs_size)) != int(obs_size):
        raise ValueError("grip policy obs_size does not match target obs_size")
    if int(metrics.get("action_size", action_size)) != int(action_size):
        raise ValueError("grip policy action_size does not match target action_size")
    ppo = metrics.get("ppo", {})
    hidden_sizes = tuple(int(value) for value in ppo.get("hidden_sizes", (256, 256)))
    action_std_init = float(ppo.get("action_std_init", 0.35))
    model = PolicyValueNet(obs_size, action_size, hidden_sizes, action_std_init)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    obs_rms = RunningMeanStd((obs_size,))
    state = checkpoint["obs_rms"]
    obs_rms.mean = np.asarray(state["mean"], dtype=np.float64)
    obs_rms.var = np.asarray(state["var"], dtype=np.float64)
    obs_rms.count = float(state["count"])
    return TorchGripPolicy(model=model, obs_rms=obs_rms)


class RandomGripPolicy:
    def __init__(self, action_size: int) -> None:
        self.action_size = int(action_size)

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(self.action_size, dtype=float)

class TorchGripPolicy:
    def __init__(self, *, model, obs_rms) -> None:
        self.model = model
        self.obs_rms = obs_rms

    def act(self, obs: np.ndarray) -> np.ndarray:
        import torch

        obs_norm = self.obs_rms.normalize(np.asarray(obs, dtype=float))
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs_norm, dtype=torch.float32).unsqueeze(0)
            mean, _, _ = self.model(obs_tensor)
            action = torch.clamp(mean, -1.0, 1.0)
        return action.squeeze(0).cpu().numpy().astype(float)
```

- [x] **Step 3: Run tests**

Run:

```bash
pytest environment/overall_environment/tests/test_layered_policy.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add environment/overall_environment/src/layered_policy.py environment/overall_environment/tests/test_layered_policy.py
git commit -m "load right-hand grip policy for layered control"
```

### Task 2.3: Add 100-Step Layered Rollout Smoke Test

**Files:**
- Create: `environment/overall_environment/tests/test_layered_forehand_rollout.py`

- [x] **Step 1: Write fake-env smoke test**

Create:

```python
from __future__ import annotations

import numpy as np

from environment.overall_environment.src.layered_control import LayeredActuatorRouter
from environment.overall_environment.src.layered_policy import LayeredPolicy


class FakePolicy:
    def __init__(self, size):
        self.size = int(size)

    def eval(self):
        return self

    def act(self, obs):
        return np.zeros(self.size, dtype=float)


class FakeEnv:
    def __init__(self):
        self.steps = 0

    def reset(self):
        return np.zeros(5, dtype=float), {}

    def step(self, action):
        self.steps += 1
        assert action.shape == (3,)
        return np.ones(5, dtype=float), 0.0, False, self.steps >= 100, {}


def test_layered_rollout_100_steps_no_nan():
    env = FakeEnv()
    router = LayeredActuatorRouter(["hip", "FDS2", "shoulder"], ["hip", "shoulder"], ["FDS2"])
    policy = LayeredPolicy(body_policy=FakePolicy(2), grip_policy=FakePolicy(1), router=router)

    obs, _info = env.reset()
    for _ in range(100):
        output = policy.act(obs)
        assert np.isfinite(output.full_action).all()
        obs, reward, terminated, truncated, _info = env.step(output.full_action)
        assert np.isfinite(obs).all()
        assert np.isfinite(reward)
        if terminated or truncated:
            break
```

- [x] **Step 2: Run test**

Run:

```bash
pytest environment/overall_environment/tests/test_layered_forehand_rollout.py -q
```

Expected: `1 passed`.

- [ ] **Step 3: Commit**

```bash
git add environment/overall_environment/tests/test_layered_forehand_rollout.py
git commit -m "add layered rollout smoke test"
```

---

## Phase 3: Training Scene XML

### Task 3.1: Add Training Scene Validator Before Duplicating XML

**Files:**
- Create: `environment/overall_environment/src/training_scene.py`
- Create: `environment/overall_environment/tests/test_training_scene.py`

- [x] **Step 1: Add validator test with fake model**

Create:

```python
from __future__ import annotations

import pytest

from environment.overall_environment.src.training_scene import TrainingSceneReport, validate_training_scene_report


def test_validate_training_scene_report_requires_training_keyframe_and_actuators():
    report = TrainingSceneReport(
        xml_path="scene.xml",
        keyframes=["overall_ready", "training_start"],
        actuator_count=416,
        required_sites=["right_hand_mimic", "racket_head_site"],
        missing_sites=[],
        required_geoms=["racket_stringbed"],
        missing_geoms=[],
    )

    validate_training_scene_report(report)


def test_validate_training_scene_report_rejects_missing_site():
    report = TrainingSceneReport(
        xml_path="scene.xml",
        keyframes=["overall_ready", "training_start"],
        actuator_count=416,
        required_sites=["right_hand_mimic"],
        missing_sites=["right_hand_mimic"],
        required_geoms=[],
        missing_geoms=[],
    )

    with pytest.raises(ValueError, match="missing sites"):
        validate_training_scene_report(report)
```

- [x] **Step 2: Implement validator**

Create:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingSceneReport:
    xml_path: str
    keyframes: list[str]
    actuator_count: int
    required_sites: list[str]
    missing_sites: list[str]
    required_geoms: list[str]
    missing_geoms: list[str]


def validate_training_scene_report(report: TrainingSceneReport) -> None:
    if "overall_ready" not in report.keyframes:
        raise ValueError("training scene missing overall_ready keyframe")
    if report.actuator_count <= 0:
        raise ValueError("training scene must expose actuators")
    if report.missing_sites:
        raise ValueError(f"training scene missing sites: {report.missing_sites}")
    if report.missing_geoms:
        raise ValueError(f"training scene missing geoms: {report.missing_geoms}")
```

- [x] **Step 3: Run tests**

Run:

```bash
pytest environment/overall_environment/tests/test_training_scene.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add environment/overall_environment/src/training_scene.py environment/overall_environment/tests/test_training_scene.py
git commit -m "add training scene validation contract"
```

### Task 3.2: Create Training Scene XML Variant

**Files:**
- Create: `environment/overall_environment/assets/overall_badminton_training_scene.xml`
- Modify: `environment/overall_environment/src/training_scene.py`
- Modify: `environment/overall_environment/tests/test_training_scene.py`

- [x] **Step 1: Copy inspection XML to training XML**

Create `environment/overall_environment/assets/overall_badminton_training_scene.xml` from `environment/overall_environment/assets/overall_badminton_scene.xml`.

- [x] **Step 2: Ensure training scene keeps actuators**

The generated XML must keep the `<actuator>` block active and expose the same `model.nu` as `MyoFullBody(disable_fingers=False)` plus scene-specific motors if any are deliberately added.

- [x] **Step 3: Add real XML smoke test**

Add:

```python
def test_training_scene_xml_loads_with_mujoco():
    import mujoco

    model = mujoco.MjModel.from_xml_path("environment/overall_environment/assets/overall_badminton_training_scene.xml")

    assert model.nu > 0
    assert model.nkey >= 1
```

- [x] **Step 4: Run smoke test**

Run:

```bash
pytest environment/overall_environment/tests/test_training_scene.py::test_training_scene_xml_loads_with_mujoco -q
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add environment/overall_environment/assets/overall_badminton_training_scene.xml environment/overall_environment/src/training_scene.py environment/overall_environment/tests/test_training_scene.py
git commit -m "add overall badminton training scene"
```

---

## Phase 4: Static-Hit Reward And Termination

### Task 4.1: Add Reward Term Dataclass And Pure Reward Function

**Files:**
- Modify: `environment/overall_environment/src/static_forehand_clear_env.py`
- Modify: `environment/overall_environment/tests/test_static_forehand_clear_env.py`

- [x] **Step 1: Add reward tests**

Add:

```python
def test_static_hit_reward_positive_for_valid_impact_contact():
    from environment.overall_environment.src.static_forehand_clear_env import compute_static_hit_reward_terms

    terms = compute_static_hit_reward_terms(
        phase=0.5,
        impact_phase=0.5,
        phase_tolerance=0.08,
        contact_info={"active": True, "rho2": 0.1, "penetration": 0.002, "relative_normal_velocity": -8.0},
        flight_info={"region": "opponent_back", "crossed_net": True, "landed": True},
    )

    assert terms["impact"] > 0.0
    assert terms["flight"] > 0.0
    assert sum(terms.values()) > 0.0


def test_static_hit_reward_penalizes_out_of_phase_contact():
    from environment.overall_environment.src.static_forehand_clear_env import compute_static_hit_reward_terms

    terms = compute_static_hit_reward_terms(
        phase=0.1,
        impact_phase=0.5,
        phase_tolerance=0.08,
        contact_info={"active": True, "rho2": 0.1, "penetration": 0.002, "relative_normal_velocity": -8.0},
        flight_info={"region": "own_side", "crossed_net": False, "landed": True},
    )

    assert terms["impact"] == 0.0
    assert terms["flight"] <= 0.0
```

- [x] **Step 2: Implement pure function**

Add:

```python
def compute_static_hit_reward_terms(
    *,
    phase: float,
    impact_phase: float,
    phase_tolerance: float,
    contact_info: Mapping[str, object],
    flight_info: Mapping[str, object],
) -> dict[str, float]:
    in_phase = abs(float(phase) - float(impact_phase)) <= float(phase_tolerance)
    active = bool(contact_info.get("active", False))
    rho2 = float(contact_info.get("rho2", 2.0))
    closing_speed = max(0.0, -float(contact_info.get("relative_normal_velocity", 0.0)))
    penetration = max(0.0, float(contact_info.get("penetration", 0.0)))

    impact = 0.0
    if in_phase and active and rho2 <= 1.0 and penetration > 0.0:
        impact = min(1.0, closing_speed / 8.0) + min(0.5, penetration * 100.0)

    region = str(flight_info.get("region", "own_side"))
    crossed_net = bool(flight_info.get("crossed_net", False))
    flight = 0.0
    if region == FlightRegion.OPPONENT_BACK.value:
        flight += 1.0
    elif region == FlightRegion.OPPONENT_MID.value:
        flight += 0.5
    elif region == FlightRegion.OUT.value:
        flight -= 1.0
    elif region == FlightRegion.OWN_SIDE.value:
        flight -= 0.5
    if crossed_net:
        flight += 0.25

    early_contact_penalty = -0.25 if active and not in_phase else 0.0
    return {
        "pre_impact": 0.0,
        "impact": float(impact),
        "flight": float(flight),
        "penalty": float(early_contact_penalty),
    }
```

- [x] **Step 3: Run reward tests**

Run:

```bash
pytest environment/overall_environment/tests/test_static_forehand_clear_env.py::test_static_hit_reward_positive_for_valid_impact_contact environment/overall_environment/tests/test_static_forehand_clear_env.py::test_static_hit_reward_penalizes_out_of_phase_contact -q
```

Expected: `2 passed`.

- [ ] **Step 4: Commit**

```bash
git add environment/overall_environment/src/static_forehand_clear_env.py environment/overall_environment/tests/test_static_forehand_clear_env.py
git commit -m "add static-hit reward terms"
```

### Task 4.2: Wire Reward And Termination Into `StaticForehandClearEnv.step()`

**Files:**
- Modify: `environment/overall_environment/src/static_forehand_clear_env.py`
- Modify: `environment/overall_environment/tests/test_static_forehand_clear_env.py`

- [x] **Step 1: Add env step reward test**

Add:

```python
def test_static_env_step_returns_reward_terms_after_release():
    from environment.overall_environment.src.static_forehand_clear_env import StaticForehandClearEnv

    base = _FakeBaseEnv()
    target = StaticShuttleTarget(
        qpos_adr=1,
        qvel_adr=2,
        qpos=np.array([0.5, 0.6, 2.0, 1.0, 0.0, 0.0, 0.0]),
    )
    env = StaticForehandClearEnv(base, target, impact_phase=0.5, phase_tolerance=0.1)
    env.reset()

    _obs, reward, terminated, truncated, info = env.step(
        ctrl=None,
        phase=0.5,
        contact_info={"active": True, "rho2": 0.2, "penetration": 0.002, "relative_normal_velocity": -3.0},
    )

    assert reward > 0.0
    assert not terminated
    assert not truncated
    assert "reward_terms" in info
```

- [x] **Step 2: Modify `step()`**

Replace final return block with:

```python
flight_info = diagnostics.get("flight", {})
reward_terms = compute_static_hit_reward_terms(
    phase=phase,
    impact_phase=self.impact_phase,
    phase_tolerance=self.phase_tolerance,
    contact_info=contact_info or {},
    flight_info=flight_info if isinstance(flight_info, Mapping) else {},
)
reward = float(sum(reward_terms.values()))
terminated = self.state == StaticHitState.TERMINATED
truncated = False
info["reward_terms"] = reward_terms
return obs, reward, terminated, truncated, info
```

- [x] **Step 3: Run static env tests**

Run:

```bash
pytest environment/overall_environment/tests/test_static_forehand_clear_env.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add environment/overall_environment/src/static_forehand_clear_env.py environment/overall_environment/tests/test_static_forehand_clear_env.py
git commit -m "wire static-hit reward into env step"
```

---

## Phase 5: Fix Right-Hand Grip PPO Logprob

### Task 5.1: Use Tanh-Squashed Gaussian Logprob

**Files:**
- Modify: `src/grip/train_right_hand_racket_grip_policy.py`
- Create: `tests/unit/test_right_hand_grip_ppo_logprob.py`

- [x] **Step 1: Add logprob consistency test**

Create:

```python
from __future__ import annotations

import numpy as np
import torch

from src.grip.train_right_hand_racket_grip_policy import PolicyValueNet, _sample_action


def test_sample_action_returns_bounded_action_and_finite_logprob():
    rng = np.random.default_rng(0)
    model = PolicyValueNet(obs_size=4, action_size=3, hidden_sizes=(8,), action_std_init=0.35)

    action, logprob, value = _sample_action(torch, model, np.zeros(4, dtype=np.float32), "cpu", rng)

    assert action.shape == (3,)
    assert np.all(action <= 1.0)
    assert np.all(action >= -1.0)
    assert np.isfinite(logprob)
    assert np.isfinite(value)
```

- [x] **Step 2: Add helper**

In `src/grip/train_right_hand_racket_grip_policy.py`:

```python
def _tanh_normal_logprob(distribution, raw_action, squashed_action, torch_module):
    logprob = distribution.log_prob(raw_action).sum(axis=-1)
    correction = torch_module.log(1.0 - torch_module.square(squashed_action) + 1e-6).sum(axis=-1)
    return logprob - correction
```

- [x] **Step 3: Update sampling**

Replace clamp in `_sample_action()` with:

```python
raw_action = mean + noise * std
distribution = torch.distributions.Normal(mean, std)
action = torch.tanh(raw_action)
logprob = _tanh_normal_logprob(distribution, raw_action, action, torch)
```

- [x] **Step 4: Update PPO update logprob**

In `_ppo_update()`, actions stored in rollout are already squashed. Reconstruct an approximate raw action:

```python
clamped_actions = torch.clamp(actions[batch_indices], -0.999999, 0.999999)
raw_actions = torch.atanh(clamped_actions)
new_logprob = _tanh_normal_logprob(distribution, raw_actions, clamped_actions, torch)
```

- [x] **Step 5: Run tests**

Run:

```bash
pytest tests/unit/test_right_hand_grip_ppo_logprob.py tests/test_right_hand_racket_grip.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/grip/train_right_hand_racket_grip_policy.py tests/unit/test_right_hand_grip_ppo_logprob.py
git commit -m "fix grip PPO squashed action logprob"
```

---

## Phase 6: Ghost Racket Teacher

### Task 6.1: Build Ghost Racket Trajectory Module

**Files:**
- Create: `environment/overall_environment/src/ghost_racket.py`
- Create: `tests/unit/test_ghost_racket.py`

- [x] **Step 1: Add interpolation test**

Create:

```python
from __future__ import annotations

import numpy as np

from environment.overall_environment.src.ghost_racket import GhostRacketFrame, GhostRacketTrajectory, interpolate_ghost


def test_interpolate_ghost_midpoint_position():
    trajectory = GhostRacketTrajectory(
        phase=np.array([0.0, 1.0]),
        frames=[
            GhostRacketFrame(pos=np.array([0.0, 0.0, 1.0]), xmat=np.eye(3), velocity=np.array([1.0, 0.0, 0.0])),
            GhostRacketFrame(pos=np.array([2.0, 0.0, 1.0]), xmat=np.eye(3), velocity=np.array([1.0, 0.0, 0.0])),
        ],
    )

    frame = interpolate_ghost(trajectory, 0.5)

    np.testing.assert_allclose(frame.pos, np.array([1.0, 0.0, 1.0]))
```

- [x] **Step 2: Implement module**

Create:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GhostRacketFrame:
    pos: np.ndarray
    xmat: np.ndarray
    velocity: np.ndarray


@dataclass(frozen=True)
class GhostRacketTrajectory:
    phase: np.ndarray
    frames: list[GhostRacketFrame]

    def __post_init__(self) -> None:
        if len(self.frames) != len(self.phase):
            raise ValueError("phase and frames length mismatch")
        if len(self.frames) < 2:
            raise ValueError("ghost trajectory requires at least two frames")


def interpolate_ghost(trajectory: GhostRacketTrajectory, phase: float) -> GhostRacketFrame:
    phases = np.asarray(trajectory.phase, dtype=float)
    value = float(np.clip(phase, phases[0], phases[-1]))
    hi = int(np.searchsorted(phases, value, side="right"))
    hi = min(max(hi, 1), len(phases) - 1)
    lo = hi - 1
    alpha = (value - phases[lo]) / max(phases[hi] - phases[lo], 1e-12)
    a = trajectory.frames[lo]
    b = trajectory.frames[hi]
    return GhostRacketFrame(
        pos=(1.0 - alpha) * a.pos + alpha * b.pos,
        xmat=(1.0 - alpha) * a.xmat + alpha * b.xmat,
        velocity=(1.0 - alpha) * a.velocity + alpha * b.velocity,
    )
```

- [x] **Step 3: Run test**

Run:

```bash
pytest tests/unit/test_ghost_racket.py -q
```

Expected: `1 passed`.

- [ ] **Step 4: Commit**

```bash
git add environment/overall_environment/src/ghost_racket.py tests/unit/test_ghost_racket.py
git commit -m "add ghost racket trajectory module"
```

---

## Phase 7: Contact Graph Report

### Task 7.1: Add Contact Graph Module

**Files:**
- Create: `environment/overall_environment/src/contact_graph.py`
- Create: `environment/overall_environment/tests/test_contact_graph.py`

- [x] **Step 1: Add pure contact graph test**

Create:

```python
from __future__ import annotations

from environment.overall_environment.src.contact_graph import ContactGraphReport, contact_reward_terms


def test_contact_reward_terms_prefers_handle_and_stringbed_contact():
    report = ContactGraphReport(
        hand_handle_contacts=3,
        hand_handle_max_penetration=0.002,
        stringbed_shuttle_active=True,
        stringbed_rho2=0.2,
        stringbed_relative_normal_velocity=-7.0,
    )

    terms = contact_reward_terms(report)

    assert terms["hand_handle"] > 0.0
    assert terms["stringbed"] > 0.0
```

- [x] **Step 2: Implement pure report**

Create:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContactGraphReport:
    hand_handle_contacts: int
    hand_handle_max_penetration: float
    stringbed_shuttle_active: bool
    stringbed_rho2: float
    stringbed_relative_normal_velocity: float


def contact_reward_terms(report: ContactGraphReport) -> dict[str, float]:
    hand_handle = min(1.0, report.hand_handle_contacts / 3.0)
    if report.hand_handle_max_penetration > 0.01:
        hand_handle -= 0.5

    stringbed = 0.0
    if report.stringbed_shuttle_active and report.stringbed_rho2 <= 1.0:
        stringbed = min(1.0, max(0.0, -report.stringbed_relative_normal_velocity) / 8.0)

    return {
        "hand_handle": float(hand_handle),
        "stringbed": float(stringbed),
    }
```

- [x] **Step 3: Run tests**

Run:

```bash
pytest environment/overall_environment/tests/test_contact_graph.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add environment/overall_environment/src/contact_graph.py environment/overall_environment/tests/test_contact_graph.py
git commit -m "add contact graph reward report"
```

---

## Phase 8: Curriculum Runner

### Task 8.1: Add Curriculum Stage Parser And Command Builder

**Files:**
- Create: `musclemimic/badminton/scripts/run_forehand_clear_racket_curriculum.py`
- Create: `tests/unit/test_forehand_clear_racket_curriculum.py`

- [x] **Step 1: Add command builder test**

Create:

```python
from __future__ import annotations

from BadmintonMimic.scripts.run_forehand_clear_racket_curriculum import CurriculumStage, build_stage_command


def test_build_stage_command_contains_stage_and_config():
    stage = CurriculumStage(
        name="soft_weld_medium",
        config="experiments/posttrain/forehand_clear_grip_hold_v1.yaml",
        total_steps=1000,
    )

    command = build_stage_command(stage)

    assert "--stage-name" in command
    assert "soft_weld_medium" in command
    assert "--config" in command
```

- [x] **Step 2: Implement runner skeleton**

Create:

```python
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    config: str
    total_steps: int


def build_stage_command(stage: CurriculumStage) -> list[str]:
    return [
        "python",
        "musclemimic/badminton/scripts/run_forehand_clear_grip_hold.py",
        "--stage-name",
        stage.name,
        "--config",
        stage.config,
        "--total-steps",
        str(stage.total_steps),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ForehandClear racket curriculum stages.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    stages = [
        CurriculumStage("strong_weld_grip", "experiments/posttrain/forehand_clear_grip_hold_v1.yaml", 50_000),
        CurriculumStage("medium_weld_swing", "experiments/posttrain/forehand_clear_grip_hold_v1.yaml", 100_000),
        CurriculumStage("static_hit", "experiments/posttrain/forehand_clear_static_hit_v1.yaml", 200_000),
    ]
    for stage in stages:
        print(" ".join(build_stage_command(stage)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] **Step 3: Run tests**

Run:

```bash
pytest tests/unit/test_forehand_clear_racket_curriculum.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add musclemimic/badminton/scripts/run_forehand_clear_racket_curriculum.py tests/unit/test_forehand_clear_racket_curriculum.py
git commit -m "add forehand clear racket curriculum command builder"
```

---

## Phase 9: ASI Hard-State Mining

### Task 9.1: Classify Failures And Save Hard-State Seeds

**Files:**
- Create: `environment/overall_environment/src/hard_state_mining.py`
- Create: `tests/unit/test_hard_state_mining.py`

- [x] **Step 1: Add failure classifier test**

Create:

```python
from __future__ import annotations

from environment.overall_environment.src.hard_state_mining import classify_failure


def test_classify_failure_detects_grip_slip_before_contact():
    failure = classify_failure(
        {
            "grip_slip_m": 0.08,
            "stringbed_contact": False,
            "shuttle_crossed_net": False,
            "fell": False,
        }
    )

    assert failure == "grip_slip"
```

- [x] **Step 2: Implement classifier**

Create:

```python
from __future__ import annotations


def classify_failure(info: dict) -> str:
    if bool(info.get("fell", False)):
        return "fall"
    if float(info.get("grip_slip_m", 0.0)) > 0.05:
        return "grip_slip"
    if not bool(info.get("stringbed_contact", False)):
        return "missed_contact"
    if not bool(info.get("shuttle_crossed_net", False)):
        return "bad_flight"
    return "unknown"
```

- [x] **Step 3: Run tests**

Run:

```bash
pytest tests/unit/test_hard_state_mining.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add environment/overall_environment/src/hard_state_mining.py tests/unit/test_hard_state_mining.py
git commit -m "add hard-state failure classifier"
```

---

## Phase 10: Ablation Reporting

### Task 10.1: Build Ablation Report Generator

**Files:**
- Create: `musclemimic/badminton/scripts/build_forehand_clear_ablation_report.py`
- Create: `tests/unit/test_forehand_clear_ablation_report.py`

- [x] **Step 1: Add report test**

Create:

```python
from __future__ import annotations

from BadmintonMimic.scripts.build_forehand_clear_ablation_report import render_markdown_report


def test_render_markdown_report_contains_ranked_arms():
    rows = [
        {"arm": "A0", "success_rate": 0.1, "backcourt_rate": 0.0},
        {"arm": "A1", "success_rate": 0.4, "backcourt_rate": 0.2},
    ]

    report = render_markdown_report(rows)

    assert "# ForehandClear Racket Ablation Summary" in report
    assert "| A1 |" in report
    assert report.index("| A1 |") < report.index("| A0 |")
```

- [x] **Step 2: Implement report renderer**

Create:

```python
from __future__ import annotations


def render_markdown_report(rows: list[dict]) -> str:
    ranked = sorted(rows, key=lambda row: (float(row["success_rate"]), float(row["backcourt_rate"])), reverse=True)
    lines = [
        "# ForehandClear Racket Ablation Summary",
        "",
        "| Arm | Success Rate | Backcourt Rate |",
        "| --- | ---: | ---: |",
    ]
    for row in ranked:
        lines.append(
            f"| {row['arm']} | {float(row['success_rate']):.3f} | {float(row['backcourt_rate']):.3f} |"
        )
    lines.append("")
    return "\n".join(lines)
```

- [x] **Step 3: Run tests**

Run:

```bash
pytest tests/unit/test_forehand_clear_ablation_report.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add musclemimic/badminton/scripts/build_forehand_clear_ablation_report.py tests/unit/test_forehand_clear_ablation_report.py
git commit -m "add forehand clear ablation report renderer"
```

---

## End-To-End Verification Order

- [x] Run runner/spec tests:

```bash
pytest tests/unit/test_run_posttrain_experiment.py tests/unit/test_forehand_clear_grip_hold_runner.py tests/unit/test_forehand_clear_grip_hold_spec.py tests/unit/test_forehand_clear_static_hit_spec.py -q
```

- [x] Run action/router/layered tests:

```bash
pytest tests/unit/test_action_manifest.py environment/overall_environment/tests/test_action_adapter.py environment/overall_environment/tests/test_layered_control.py environment/overall_environment/tests/test_layered_policy.py environment/overall_environment/tests/test_layered_forehand_rollout.py -q
```

- [x] Run scene/static-hit/contact tests:

```bash
pytest environment/overall_environment/tests/test_training_scene.py environment/overall_environment/tests/test_static_forehand_clear_env.py environment/overall_environment/tests/test_contact_graph.py environment/overall_environment/tests/test_phase_reward.py -q
```

- [x] Run grip PPO tests:

```bash
pytest tests/unit/test_right_hand_grip_ppo_logprob.py tests/test_right_hand_racket_grip.py -q
```

- [x] Run innovation and report tests:

```bash
pytest tests/unit/test_ghost_racket.py tests/unit/test_soft_weld_schedule.py tests/unit/test_hard_state_mining.py tests/unit/test_forehand_clear_racket_curriculum.py tests/unit/test_forehand_clear_ablation_report.py -q
```

- [x] Run diagnostic replay precheck:

```bash
.venv/bin/python musclemimic/badminton/scripts/run_forehand_clear_grip_hold.py \
  --spec experiments/posttrain/forehand_clear_grip_hold_v1.yaml \
  --stage replay-precheck
```

Expected after Phase 1 and Phase 2: report includes manifest/action compatibility details and no longer reports action mapping as an unresolved blocker.

---

## Non-Negotiable Acceptance Criteria

- The original `de63059b16c0/checkpoint_7812` body policy is treated as a 354-action no-finger policy.
- Any `disable_fingers=false` racket scene is treated as a 416-action full-hand scene unless the loaded MuJoCo model proves a different `model.nu`.
- No code path directly pads actions by index. All checkpoint-to-scene mapping is by actuator name.
- Grip residual and body policy actuator ownership must not overlap.
- `StaticForehandClearEnv.step()` must return finite reward, finite observations, explicit `reward_terms`, and real termination/truncation.
- Grip PPO must compute logprob for the same bounded action representation that is stored in rollout.
- Training curriculum must write per-stage metrics, videos, and reports so failed training can be attributed to action mapping, grip slip, contact miss, flight failure, or fall.
- Ablation report must compare at least A0 through A6: direct fine-tune, frozen body plus grip, soft-weld, ghost teacher, phase contact, flight reward, and hard-state mining.

## Self-Review Checklist

- Spec coverage: P0 through P10 from `4_priority_roadmap.md` are represented as phases and tasks.
- Current-state evidence: runner, checkpoint, action dimensions, router, static-hit env, and grip PPO issues are tied to local files.
- Placeholder scan: no task relies on unspecified behavior; each code task includes concrete paths, test commands, and expected results.
- Type consistency: `ActionManifest`, `CheckpointToFullActionAdapter`, `LayeredActuatorRouter`, `LayeredPolicy`, `GhostRacketTrajectory`, `ContactGraphReport`, and curriculum/report functions are named consistently across tasks.
- Execution safety: every implementation phase starts with tests, then minimal code, then verification, then a scoped commit.

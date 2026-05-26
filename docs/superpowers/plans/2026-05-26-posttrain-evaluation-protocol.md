# PostTrain Evaluation Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a clean ForehandNetLift PostTrain evaluation protocol that compares baseline and PostTrain checkpoints from frame 0 without changing reward, PPO, terminal handlers, or model code.

**Architecture:** Extend the existing PostTrain spec generator to pass `validation_start_from_beginning` into generated Hydra validation configs. Add a focused offline comparison script that executes `fullbody/eval.py` per motion/checkpoint, parses validation metrics, writes CSV/Markdown reports, and optionally renders side-by-side videos. Keep the runner and comparison logic separate so training preparation remains small and evaluation batching can evolve independently.

**Tech Stack:** Python 3, PyYAML, subprocess, CSV, pytest, existing `fullbody/eval.py`, existing `BadmintonMimic/scripts/run_posttrain_experiment.py`.

---

### Task 1: Pass Start-From-Beginning Into Generated Validation Configs

**Files:**
- Modify: `BadmintonMimic/scripts/run_posttrain_experiment.py`
- Modify: `tests/unit/test_run_posttrain_experiment.py`

- [ ] **Step 1: Write the failing test**

Add this assertion to `test_prepare_experiment_writes_hydra_configs_and_report` after loading `config`:

```python
assert config["experiment"]["validation"]["start_from_beginning"] is True
```

Add this field to `_write_spec(...).spec["training"]`:

```python
"validation_start_from_beginning": True,
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_run_posttrain_experiment.py::test_prepare_experiment_writes_hydra_configs_and_report -v
```

Expected: FAIL with missing key `start_from_beginning`.

- [ ] **Step 3: Implement the config pass-through**

In `build_hydra_config`, add this key under `"experiment": {"validation": ...}`:

```python
"start_from_beginning": bool(training.get("validation_start_from_beginning", False)),
```

- [ ] **Step 4: Run the focused test**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_run_posttrain_experiment.py::test_prepare_experiment_writes_hydra_configs_and_report -v
```

Expected: PASS.

### Task 2: Add Offline Comparison Command Builder

**Files:**
- Create: `BadmintonMimic/scripts/evaluate_posttrain_protocol.py`
- Create: `tests/unit/test_evaluate_posttrain_protocol.py`

- [ ] **Step 1: Write command-builder tests**

Create tests for:

```python
from pathlib import Path

from BadmintonMimic.scripts.evaluate_posttrain_protocol import (
    build_metrics_command,
    latest_checkpoint,
    parse_validation_metrics,
)


def test_build_metrics_command_uses_start_from_beginning_and_evaluate_all(tmp_path: Path):
    command = build_metrics_command(
        checkpoint=tmp_path / "checkpoint_10",
        motion="ForehandNetLift/best/video07_best_stage5_smpl",
        eval_seed=0,
        metrics_envs=1,
    )

    assert command[:3] == ["uv", "run", "fullbody/eval.py"]
    assert "--start_from_beginning" in command
    assert "--evaluate_all" in command
    assert "--metrics_deterministic" in command
    assert "--metrics_only" in command
    assert "ForehandNetLift/best/video07_best_stage5_smpl" in command


def test_latest_checkpoint_searches_config_hash_dirs(tmp_path: Path):
    root = tmp_path / "arm"
    (root / "hash_a" / "checkpoint_7907").mkdir(parents=True)
    (root / "hash_a" / "checkpoint_8002").mkdir(parents=True)

    assert latest_checkpoint(root) == root / "hash_a" / "checkpoint_8002"


def test_parse_validation_metrics_reads_metric_block():
    output = '''
=== VALIDATION METRICS ===
val_early_termination_rate: 0.000000
val_frame_coverage: 1.000000
val_mean_episode_return: 123.500000
'''

    metrics = parse_validation_metrics(output)

    assert metrics["val_early_termination_rate"] == 0.0
    assert metrics["val_frame_coverage"] == 1.0
    assert metrics["val_mean_episode_return"] == 123.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_evaluate_posttrain_protocol.py -v
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement the command helpers**

Implement:

```python
def latest_checkpoint(root: Path) -> Path | None
def build_metrics_command(checkpoint: Path, motion: str, eval_seed: int, metrics_envs: int) -> list[str]
def parse_validation_metrics(output: str) -> dict[str, float]
```

The metrics command must call:

```text
uv run fullbody/eval.py
--path <checkpoint>
--motion_path <motion>
--use_mujoco
--start_from_beginning
--evaluate_all
--metrics
--metrics_only
--metrics_deterministic
--metrics_envs <metrics_envs>
--eval_seed <eval_seed>
```

- [ ] **Step 4: Run helper tests**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_evaluate_posttrain_protocol.py -v
```

Expected: PASS.

### Task 3: Add Report Generation

**Files:**
- Modify: `BadmintonMimic/scripts/evaluate_posttrain_protocol.py`
- Modify: `tests/unit/test_evaluate_posttrain_protocol.py`

- [ ] **Step 1: Write report tests**

Add tests that call:

```python
from BadmintonMimic.scripts.evaluate_posttrain_protocol import build_delta_rows, write_reports
```

Use rows for one baseline and one PostTrain motion. Assert:

```python
delta_rows[0]["delta_val_mean_episode_return"] == -10.0
delta_rows[0]["pass_hard_gates"] == "false"
```

Assert `metrics_table.csv`, `metrics_delta.csv`, and `comparison_report.md` are written.

- [ ] **Step 2: Run report tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_evaluate_posttrain_protocol.py -v
```

Expected: FAIL because report functions are missing.

- [ ] **Step 3: Implement report functions**

Implement:

```python
def build_delta_rows(rows: list[dict[str, str | float]]) -> list[dict[str, str | float]]
def write_reports(output_dir: Path, rows: list[dict[str, str | float]], delta_rows: list[dict[str, str | float]]) -> None
```

Hard gates:

```python
early_termination_rate == 0.0
frame_coverage >= 0.95
posttrain_return >= baseline_return
posttrain_joint_vel <= baseline_joint_vel + 0.10
posttrain_rpos <= baseline_rpos + 0.01
```

- [ ] **Step 4: Run report tests**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_evaluate_posttrain_protocol.py -v
```

Expected: PASS.

### Task 4: Add CLI and Regenerate Configs

**Files:**
- Modify: `BadmintonMimic/scripts/evaluate_posttrain_protocol.py`
- Modify: `BadmintonMimic/experiments/posttrain/forehand_net_lift_v1.yaml`
- Generated: `fullbody/config_specific_task/posttrain/ForehandNetLift/v1/E1c_fullbody_stability.yaml`
- Generated: `outputs/posttrain/ForehandNetLift/v1/configs/E1c_fullbody_stability.yaml`
- Generated: `outputs/posttrain/ForehandNetLift/v1/commands/*.sh`

- [ ] **Step 1: Add CLI arguments**

The script should accept:

```text
--spec
--arm
--checkpoint
--run-name
--splits train,validation,stress_test
--metrics-envs
--eval-seed
--execute
```

Default behavior without `--execute`: print commands and write no metrics.

- [ ] **Step 2: Add config flag**

In `BadmintonMimic/experiments/posttrain/forehand_net_lift_v1.yaml`, add:

```yaml
training:
  validation_start_from_beginning: true
```

- [ ] **Step 3: Regenerate configs**

Run:

```bash
.venv/bin/python BadmintonMimic/scripts/run_posttrain_experiment.py \
  --spec BadmintonMimic/experiments/posttrain/forehand_net_lift_v1.yaml \
  --stage prepare
```

Expected: generated E1c Hydra config contains `experiment.validation.start_from_beginning: true`.

### Task 5: Verification

**Files:**
- Verify only.

- [ ] **Step 1: Run unit tests**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_run_posttrain_experiment.py tests/unit/test_evaluate_posttrain_protocol.py -v
```

Expected: all tests pass.

- [ ] **Step 2: Dry-run comparison commands**

Run:

```bash
.venv/bin/python BadmintonMimic/scripts/evaluate_posttrain_protocol.py \
  --spec BadmintonMimic/experiments/posttrain/forehand_net_lift_v1.yaml \
  --arm E1c_fullbody_stability \
  --run-name dryrun_protocol_check
```

Expected: script prints baseline and PostTrain commands for every configured motion and does not execute them.

- [ ] **Step 3: Commit implementation**

Run:

```bash
git add BadmintonMimic/scripts/run_posttrain_experiment.py \
  BadmintonMimic/scripts/evaluate_posttrain_protocol.py \
  BadmintonMimic/experiments/posttrain/forehand_net_lift_v1.yaml \
  fullbody/config_specific_task/posttrain/ForehandNetLift/v1/E1c_fullbody_stability.yaml \
  outputs/posttrain/ForehandNetLift/v1/configs/E1c_fullbody_stability.yaml \
  outputs/posttrain/ForehandNetLift/v1/commands \
  outputs/posttrain/ForehandNetLift/v1/reports/posttrain_plan.md \
  outputs/posttrain/ForehandNetLift/v1/spec_snapshot.yaml \
  tests/unit/test_run_posttrain_experiment.py \
  tests/unit/test_evaluate_posttrain_protocol.py
git commit -m "feat: add posttrain evaluation protocol"
```

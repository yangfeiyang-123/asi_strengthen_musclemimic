# Forehand Net Lift PostTrain Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable YAML-driven PostTrain experiment interface for ForehandNetLift and future badminton actions.

**Architecture:** A single runner script parses an experiment spec, emits Hydra configs and reports, and builds train/eval/render commands. The runner is safe by default: stages dry-run unless `--execute` is passed.

**Tech Stack:** Python 3.11, PyYAML, pytest, existing `fullbody/experiment.py` and `fullbody/eval.py` entry points.

---

### Task 1: Runner Unit Tests

**Files:**
- Create: `tests/unit/test_run_posttrain_experiment.py`

- [x] **Step 1: Write failing tests**

Tests import `load_spec`, `prepare_experiment`, `build_train_command`, and `build_eval_command`, then verify motion normalization, generated Hydra config contents, and command construction.

- [x] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_run_posttrain_experiment.py -v`

Expected: failure because `BadmintonMimic.scripts.run_posttrain_experiment` does not exist.

### Task 2: Runner Implementation

**Files:**
- Create: `musclemimic/badminton/scripts/run_posttrain_experiment.py`

- [x] **Step 1: Implement spec loading**

Read YAML, require `experiment_id`, `action`, `output_root`, `resume_from`, `reference.train`, `reference.validation`, and `arms`; normalize motion paths by removing `.npz`.

- [x] **Step 2: Implement config generation**

Generate one Hydra config for each non-baseline arm, copy it to the output config directory and the Hydra config tree, and include env path overrides, reward overrides, train motions, and validation motions.

- [x] **Step 3: Implement command building**

Build train commands for generated configs and eval/render commands for baseline or arm checkpoints. Use `uv run fullbody/experiment.py` and `uv run fullbody/eval.py`.

- [x] **Step 4: Implement CLI stages**

Support `--stage prepare|train|eval|render|all`, `--arm`, and `--execute`. Dry-run prints and writes commands; execute runs subprocesses.

### Task 3: ForehandNetLift v1 Spec

**Files:**
- Create: `experiments/posttrain/forehand_net_lift_v1.yaml`

- [x] **Step 1: Add baseline and three posttrain arms**

Use `forehand_net_lift_best_asi_curriculum/checkpoint_7812` as the resume baseline, train on video05/video06, validate on video07/video08, and keep video01/video03/video04/video09 as stress tests.

### Task 4: Verification

**Files:**
- Test: `tests/unit/test_run_posttrain_experiment.py`

- [x] **Step 1: Run focused tests**

Run: `.venv/bin/python -m pytest tests/unit/test_run_posttrain_experiment.py -v`

Expected: all tests pass.

- [x] **Step 2: Run prepare dry-run for ForehandNetLift**

Run: `.venv/bin/python musclemimic/badminton/scripts/run_posttrain_experiment.py --spec experiments/posttrain/forehand_net_lift_v1.yaml --stage prepare`

Expected: generated configs and report under `outputs/posttrain/ForehandNetLift/v1/`.

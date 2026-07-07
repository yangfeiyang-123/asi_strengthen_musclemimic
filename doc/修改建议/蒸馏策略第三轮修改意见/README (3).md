# Distillation Readiness Audit — `68cba3c complete distillation review round two`

> Scope: this audit only covers **ForehandClear body-trajectory imitation distillation**. It does **not** cover racket, shuttle, impact, or shot outcome learning.

## 0. Executive conclusion

The repository at commit-like ref `68cba3c` is now **substantially closer to a usable distillation workflow** than the previous `eade4bc` state. It has the key library pieces needed for a no-future-lookahead student policy:

- student observation filtering: full teacher observation -> `state + motion phase`;
- teacher rollout dataset collection;
- NPZ shard dataset IO with train/val/test shard naming;
- offline BC/KD student training;
- DAgger-style student rollout + teacher relabeling;
- iterative DAgger orchestration;
- student PPO fine-tune config;
- teacher/student metric comparison helpers;
- registered console entrypoints for generic fullbody and ForehandClear wrappers.

However, I would still **not start a long production-scale distillation run before fixing several hidden integration risks**. The code is now suitable for a **small smoke test** after the P0 fixes below, but the current workflow still has packaging, wrapper-argument, path, and validation weaknesses that can waste GPU time or produce misleading experiments.

My current readiness estimate:

| Layer | Status | Estimate |
|---|---|---:|
| Core student observation filtering | Mostly ready | 85–90% |
| Dataset schema / shard IO | Mostly ready | 85% |
| Teacher rollout collection | Mostly ready | 80–85% |
| Offline BC/KD trainer | Mostly ready | 80% |
| DAgger correction collection | Mostly ready | 75–80% |
| Iterative DAgger loop | Functional but fragile | 65–70% |
| ForehandClear wrapper commands | Useful but misleading/incomplete | 55–65% |
| End-to-end reproducible distillation experiment | Not fully ready | 60–70% |

**Short recommendation:** fix the P0 issues, run the observation-filter smoke test, collect a tiny teacher dataset, train a tiny BC student, evaluate it, then run one tiny DAgger iteration. Only after that should you run larger jobs.

---

## 1. What is now implemented and appears aligned with the strategy

### 1.1 Student observation filter is now conceptually correct

`musclemimic/distill/obs_filter.py` implements the v1 student input design: keep all non-goal observations and keep only the final goal element, interpreted as motion phase. This matches the recommended first student policy:

```text
student_obs = joint state + muscle state + foot contact + motion phase
```

The filter now explicitly reads `drop_goal_lookahead`, and it rejects the invalid setting `keep_motion_phase=True` with `drop_goal_lookahead=False`. That is good because it prevents a misleading “student” that still sees the full goal/lookahead vector.

Important behavior:

```text
raw teacher obs = [state, full goal lookahead]
student obs     = [state, phase]
```

The wrapper still leaves environment state, trajectory handler, reward, done, and dynamics unchanged. Only the observation returned to the policy is filtered.

### 1.2 Dataset shard IO is stronger than before

`musclemimic/distill/dataset.py` now has:

- required fields: `student_obs`, `teacher_action`;
- `SCHEMA_VERSION = "distill_v1"`;
- `write_distill_shard()`;
- `write_split_shard()` for `train_*.npz`, `val_*.npz`, `test_*.npz`;
- `DistillDataset(..., split="train")` loading split-specific shards first, falling back to generic `shard_*.npz`.

This is enough for BC/KD training and for DAgger aggregation.

### 1.3 Teacher rollout collection now fixes earlier concerns

`musclemimic/distill/collect_teacher.py` now explicitly builds a teacher rollout config and disables `student_obs_filter` during teacher rollout. This is important because the teacher must act on the full lookahead observation.

It also saves useful fields:

```text
student_obs
teacher_action
teacher_mu
teacher_log_std
teacher_value
teacher_log_prob
reward
done
absorbing
traj_no
subtraj_step_no
phase
optional full_obs
```

It also exposes `freeze_run_stats`, `split`, and `save_full_obs`. This is a good improvement over the previous state.

### 1.4 BC/KD training is now closer to experiment-ready

`musclemimic/distill/train_bc.py` now:

- ensures `student_obs_filter.enabled=True`;
- validates `dataset.student_obs_dim` against the configured student env observation dimension;
- supports action MSE, optional value distillation, and optional diagonal Gaussian KL;
- saves a PPO-compatible checkpoint;
- writes `distill_metadata.json`.

The dimension validation is especially important because it catches the most common distillation failure: collecting dataset with one observation layout and training with another.

### 1.5 DAgger correction now includes the missing diagnostics

`musclemimic/distill/dagger.py` now stores:

```text
student_obs
teacher_action / teacher_mu
student_action
rollout_action
used_teacher_action
teacher_log_std
teacher_log_prob_teacher_mu
teacher_log_prob_student_action
teacher_log_prob_rollout_action
reward
done
absorbing
traj_no
subtraj_step_no
phase
optional full_obs
```

This is a meaningful improvement. It allows post-hoc inspection of:

- whether student or teacher action was used for rollout;
- whether teacher thinks the student action was likely;
- how DAgger states differ from teacher states;
- whether reward collapses on student-visited states.

### 1.6 Student PPO fine-tune config exists

There is now a specific no-future-lookahead student PPO config:

```text
fullbody/config_specific_task/conf_fullbody_badminton_student_gmr.yaml
```

which inherits from:

```text
fullbody/config_specific_task/distill/conf_fullbody_forehandclear_student_phase_ppo.yaml
```

The underlying config sets:

```yaml
student_obs_filter:
  enabled: true
  drop_goal_lookahead: true
  keep_motion_phase: true
  require_goal_group: true
  require_motion_phase: true
```

This is the correct direction for student PPO fine-tune: the policy does not receive future goal lookahead, but reward and trajectory handler can still use the reference trajectory internally.

### 1.7 CLI entrypoints are now registered

`pyproject.toml` now registers generic fullbody distill commands and ForehandClear wrappers, including:

```text
musclemimic-distill-collect-teacher
musclemimic-distill-train-bc
musclemimic-distill-collect-dagger
musclemimic-distill-run-dagger
musclemimic-distill-compare
forehand-clear-distill-collect-teacher
forehand-clear-distill-train-bc
forehand-clear-distill-collect-dagger
forehand-clear-distill-run-dagger
forehand-clear-distill-evaluate
forehand-clear-distill-inspect-obs
```

This is a major practical improvement, but there are still packaging and wrapper issues below.

---

## 2. Hidden errors / risks to fix before long runs

### P0-1. `BadmintonMimic` console entrypoints may not be installed correctly

`pyproject.toml` registers console scripts under `BadmintonMimic.scripts.*`, but the package discovery include list visible in the repository still does not include `BadmintonMimic*` in the default branch view I could fetch. The `68cba3c` URL content registers the `forehand-clear-*` scripts, and the repository now has `BadmintonMimic/__init__.py` and `musclemimic/badminton/scripts/__init__.py`, so the intended module path is valid. But if setuptools package discovery excludes `BadmintonMimic*`, installed console scripts can fail with:

```text
ModuleNotFoundError: No module named 'BadmintonMimic'
```

**Required fix:** in `pyproject.toml`, include `BadmintonMimic*`:

```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["musclemimic*", "loco_mujoco*", "bimanual*", "fullbody*", "BadmintonMimic*", "src*"]
```

If `src*` was intentionally removed, that is fine, but `BadmintonMimic*` should be present because pyproject registers console scripts that import it.

### P0-2. ForehandClear wrapper arguments are currently misleading

Several ForehandClear wrappers expose arguments that are not actually forwarded to the generic fullbody distill commands.

Examples:

- `collect_forehand_clear_teacher_dataset.py` accepts `--config-name`, `--motion-path`, and `--wandb`, but the constructed subprocess command does not use them.
- `collect_forehand_clear_dagger_dataset.py` also accepts `--config-name`, `--motion-path`, and `--wandb`, but does not forward them.
- `train_forehand_clear_student_bc.py` accepts `--resume-student` and `--wandb`, but does not use them.
- `evaluate_forehand_clear_student.py` accepts `--config-name`, but does not use it.

This is dangerous because a user may think they are overriding the ForehandClear motion path or config, but the actual subprocess ignores the request.

**Required fix options:**

Option A: remove unused arguments from wrappers until supported.

Option B: forward them properly. For example, `--motion-path` should either:

1. be passed to `fullbody/distill_collect.py`, after that generic script is extended to accept motion overrides, or
2. be applied directly inside the Forehand wrapper by loading and modifying the config instead of delegating blindly.

I recommend Option B eventually, but Option A is safer for immediate correctness.

### P0-3. Subprocess commands use repo-relative file paths

`dagger_loop.py` builds subprocess commands such as:

```text
python fullbody/distill_collect_dagger.py
python fullbody/distill_train_bc.py
```

The Forehand wrappers do the same. This only works reliably when the current working directory is the repository root. It can fail when called through installed console scripts or from another directory.

**Required fix:** build commands with module mode or absolute file paths:

```text
python -m fullbody.distill_collect_dagger
python -m fullbody.distill_train_bc
```

or resolve paths using `Path(__file__).resolve()`.

### P0-4. BC DAgger loop retraining is from scratch unless intentionally designed otherwise

The DAgger loop uses `initial_student_ckpt` for collection, then collects DAgger shards and calls `fullbody/distill_train_bc.py`. The BC trainer currently initializes a fresh student train state from the config each time. This is a valid DAgger variant if the aggregated dataset is large and training steps are sufficient, but it is expensive and can be surprising.

The Forehand wrapper even exposes `--resume-student`, but currently ignores it.

**Recommended fix:** add optional warm-start to `train_bc()`:

```text
--init_ckpt / --resume_student
```

Then DAgger iteration `k+1` can initialize from `student_ckpt_in` instead of always training from scratch.

### P0-5. `freeze_run_stats=True` may not mean what it sounds like

The collectors call the network with `mutable=["run_stats"]`. They then either persist or discard the updated `run_stats`. Discarding the updates prevents checkpoint state mutation, but the **forward pass itself may still use batch-updated running stats** inside the `RunningMeanStd` layer, depending on the layer implementation. This means `freeze_run_stats=True` is not necessarily equivalent to “evaluate under fixed checkpoint normalization statistics.”

**Recommended fix:** either:

- rename it to `persist_run_stats=False`, or
- implement true inference-mode normalization support in `RunningMeanStd`, e.g. a flag that uses stored mean/var without updating or using batch statistics.

For now, document that current `freeze_run_stats` only freezes persisted state, not necessarily the current forward normalization calculation.

### P1-1. Generic fullbody CLI does not expose motion/path overrides

`fullbody/distill_collect.py` loads the teacher checkpoint config and constructs the env from that config. There is no generic support for:

```text
--motion_path
--motion_group
--start_from_beginning
--traj_index
--traj_start_step
```

For ForehandClear this may be acceptable when the teacher checkpoint itself was trained on the correct ForehandClear config. But for validation splits or targeted replay, this is limiting.

**Recommended fix:** mirror a subset of `fullbody/eval.py` motion override arguments in distill collection scripts.

### P1-2. Evaluation parser should fail if required metrics are absent

`distill/eval_student.py` parses metrics from stdout using regex. This is useful, but it should enforce that required metrics are present. If `completion_rate` or key tracking metrics are not printed by `fullbody/eval.py`, the summary will silently leave blanks.

**Recommended fix:** add a required metric set and fail fast:

```python
required = {"mean_episode_return", "mean_episode_length", "early_termination_rate", "err_rpos"}
missing = required - metrics.keys()
if missing:
    raise RuntimeError(f"missing eval metrics: {sorted(missing)}")
```

Use `completion_rate` only if it is actually printed by current fullbody metrics. Otherwise compute it from `early_termination_rate` or remove it from required acceptance.

### P1-3. `StudentObsContainer.items()` is not fully compatible with the original obs container

`StudentObsContainer.items()` returns `(name, np.ndarray)` group entries, not full observation entries with attributes like `.obs_ind`. Some existing utility code in the repository expects `env.obs_container.items()` to yield objects with `.obs_ind`. This may break some export/diagnostic utilities when run under student configs.

**Recommended fix:** either implement a minimal object with `obs_ind` for each group, or avoid using `items()` in student-filtered envs. Add a unit test for trajectory export / touch index extraction under a student-filtered env if that workflow matters.

### P1-4. DAgger does not support observation history yet

The DAgger collector explicitly raises `NotImplementedError` if teacher or student uses `len_obs_history > 1`. This is okay because the current student config sets `len_obs_history: 1`, but it should remain documented. If later you want history-stacked student policies, DAgger must handle history buffers for both full teacher obs and filtered student obs.

### P2-1. Duplicate CLI aliases are useful but noisy

`pyproject.toml` registers both:

```text
musclemimic-distill-run-dagger
musclemimic-distill-dagger-loop
```

pointing to the same command, and both:

```text
musclemimic-distill-compare
musclemimic-distill-eval-student
```

pointing to the same command. This is not wrong, but it can confuse documentation.

---

## 3. Worthwhile improvements after P0 fixes

### 3.1 Add a formal smoke-test runbook

Create `docs/forehand_clear_distillation_runbook.md` with a minimal 10–30 minute test:

```bash
forehand-clear-distill-inspect-obs \
  --config-name config_specific_task/conf_fullbody_badminton_student_gmr

forehand-clear-distill-collect-teacher \
  --teacher-path /path/to/teacher/checkpoint_N \
  --output-dir /tmp/fc_distill_smoke \
  --num-envs 4 \
  --num-steps 32 \
  --shard-size 64 \
  --split train

forehand-clear-distill-train-bc \
  --dataset-dir /tmp/fc_distill_smoke \
  --output-dir /tmp/fc_student_bc_smoke \
  --num-steps 10 \
  --batch-size 16

forehand-clear-distill-collect-dagger \
  --teacher-path /path/to/teacher/checkpoint_N \
  --student-path /tmp/fc_student_bc_smoke/checkpoints/checkpoint_10 \
  --output-dir /tmp/fc_distill_smoke \
  --num-envs 4 \
  --num-steps 16 \
  --shard-size 64 \
  --split train \
  --append
```

### 3.2 Add integration tests for CLI commands

At minimum, add tests that check:

- pyproject console scripts import successfully;
- Forehand wrappers construct expected subprocess commands;
- DAgger loop `--dry_run` writes a manifest;
- `train_bc()` can produce a checkpoint from a tiny synthetic dataset;
- student config can instantiate env and network.

### 3.3 Add dataset inspection command

A small command like:

```bash
musclemimic-distill-inspect-dataset --dataset_dir ...
```

should report:

```text
schema_version
num_samples per split
student_obs_dim
action_dim
phase min/max
teacher_action range
student_action range if present
rollout_action range if present
used_teacher_action ratio if present
reward mean/std
traj_no counts
subtraj_step range
```

This will prevent training on malformed shards.

### 3.4 Add stronger metadata

BC checkpoints should include:

```text
distill_stage: teacher_bc / dagger_bc / ppo_finetune
teacher_ckpt
base_student_ckpt if warm-started
dataset_dir
dataset_schema_version
student_obs_dim
raw_teacher_obs_dim if available
student_obs_filter
forehand_clear_motion_paths
```

### 3.5 Consider a two-tier student policy evaluation

Use two evaluation modes:

1. **BC one-step metrics**: action MSE / Gaussian KL on held-out val shards.
2. **Closed-loop rollout metrics**: return, tracking errors, termination, completion.

Do not judge student quality only by action MSE.

---

## 4. What remains before actually doing distillation

### Step 1 — Fix P0 issues

Before large runs, fix:

1. package include for `BadmintonMimic*`;
2. unused/misleading Forehand wrapper arguments;
3. relative subprocess paths;
4. ignored `--resume-student` or explicitly remove it;
5. document or correct `freeze_run_stats` semantics.

### Step 2 — Verify observation filtering

Run:

```bash
forehand-clear-distill-inspect-obs \
  --config-name config_specific_task/conf_fullbody_badminton_student_gmr \
  --output-json outputs/distill/obs_filter.json
```

Expected behavior:

```text
student_obs_dim < raw_obs_dim
kept_goal_indices has length 1
phase_index_student is not null
dropped_goal_dim = goal_dim - 1
```

If this is not true, do not proceed.

### Step 3 — Confirm teacher checkpoint quality

Use existing evaluation to ensure the lookahead teacher is stable on ForehandClear:

```bash
uv run fullbody/eval.py \
  --path /path/to/teacher/checkpoint_N \
  --metrics --metrics_only \
  --metrics_envs 20 \
  --metrics_steps 500 \
  --metrics_deterministic
```

Proceed only if teacher tracking is stable enough. A bad teacher will produce a bad student.

### Step 4 — Create small train/val distillation shards

Create a small smoke train split and a val split:

```bash
forehand-clear-distill-collect-teacher \
  --teacher-path /path/to/teacher/checkpoint_N \
  --output-dir outputs/distill/fc_teacher_dataset \
  --num-envs 8 \
  --num-steps 1000 \
  --shard-size 4000 \
  --split train

forehand-clear-distill-collect-teacher \
  --teacher-path /path/to/teacher/checkpoint_N \
  --output-dir outputs/distill/fc_teacher_dataset \
  --num-envs 8 \
  --num-steps 200 \
  --shard-size 1600 \
  --seed 100 \
  --split val
```

### Step 5 — Train small BC student

```bash
forehand-clear-distill-train-bc \
  --dataset-dir outputs/distill/fc_teacher_dataset \
  --output-dir outputs/distill/fc_student_bc_smoke \
  --num-steps 1000 \
  --batch-size 512 \
  --lr 3e-4 \
  --gaussian-kl-weight 0.01
```

Check `distill_metadata.json` and ensure train/val action MSE is finite and decreasing.

### Step 6 — Evaluate closed-loop BC student

```bash
forehand-clear-distill-evaluate \
  --teacher-path /path/to/teacher/checkpoint_N \
  --student-path outputs/distill/fc_student_bc_smoke/checkpoints/checkpoint_1000 \
  --output-dir outputs/distill/fc_eval_bc_smoke \
  --num-envs 20 \
  --num-steps 500
```

If it immediately falls or return is near zero, DAgger or PPO fine-tune is mandatory.

### Step 7 — Run one small DAgger correction

```bash
forehand-clear-distill-collect-dagger \
  --teacher-path /path/to/teacher/checkpoint_N \
  --student-path outputs/distill/fc_student_bc_smoke/checkpoints/checkpoint_1000 \
  --output-dir outputs/distill/fc_teacher_dataset \
  --num-envs 8 \
  --num-steps 500 \
  --shard-size 4000 \
  --split train \
  --append
```

Then retrain BC, evaluate again, and compare.

### Step 8 — Run student PPO fine-tune

After BC/DAgger student can complete at least part of the motion, use PPO fine-tune:

```bash
uv run fullbody/experiment.py \
  --config-name=config_specific_task/conf_fullbody_badminton_student_gmr \
  experiment.resume_from=/path/to/student_bc_or_dagger/checkpoint_N \
  experiment.reset_std_on_resume=0.5 \
  wandb.mode=disabled
```

The policy input remains no-future-lookahead, while `MimicReward` still uses the reference trajectory internally.

---

## 5. Final readiness decision

### Can you start distillation now?

**Yes, after fixing P0-1 and preferably P0-2/P0-3.**

The core library functions now exist. You can start a small smoke-test distillation pipeline.

### Should you start a full-scale production run now?

**Not yet.**

I would first verify:

- console scripts import correctly after installation;
- Forehand wrappers do not silently ignore important CLI args;
- teacher dataset shards match student config dimension;
- BC checkpoint can be evaluated with `fullbody/eval.py`;
- DAgger loop can complete one tiny iteration;
- summary report contains non-empty key metrics.

### What is still missing for a robust paper-grade workflow?

- package / console script cleanup;
- true runbook and smoke tests;
- wrapper argument correctness;
- stronger metric validation;
- optional BC warm-start for DAgger;
- explicit dataset inspection command;
- documented acceptance thresholds and example commands.

Once those are fixed, the repository will be in good shape to run the real ForehandClear no-future-lookahead student distillation experiment.

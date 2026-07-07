# Distillation Review Round 4 Fix Plan

Goal: address the fourth review feedback for ForehandClear teacher-to-student distillation, prioritizing issues that block mixed teacher plus DAgger training.

## Current Verification

The fourth review contains both current blockers and items already addressed by the previous round.

Already addressed in the current repository:

- Forehand wrapper scripts use `python -m fullbody.*` for generic distillation entrypoints.
- `--resume-student` maps to `fullbody.distill_train_bc --init_ckpt`.
- Forehand teacher, DAgger, and evaluation wrappers forward `--motion-path` as `--motion_path`.
- Generic teacher and DAgger collectors accept `--motion_path`.
- Dataset inspect CLI exists.

Still needs work:

- Mixed teacher initialization shards and DAgger correction shards do not share the same schema.
- Dataset inspector depends on `DistillDataset`, so it can fail before showing shard-level diagnostics.
- DAgger shard metadata does not explicitly record per-iteration provenance when run from the loop.
- `fullbody/eval.py` has no machine-readable metrics output path; compare still parses stdout.
- Collectors only implement `motion_path`; `motion_group`, `traj_index`, and `traj_start_step` overrides are not implemented.
- BC checkpoint roundtrip coverage is not explicit enough.

## Task 1: Stabilize Distillation Shard Schema

Priority: P0.

Files:

- Modify: `musclemimic/distill/dataset.py`
- Modify: `musclemimic/distill/collect_teacher.py`
- Modify: `musclemimic/distill/dagger.py`
- Test: `tests/unit/test_distill_dataset.py`
- Test: `tests/unit/test_dagger_collect.py`

Plan:

- Define a single optional superset schema for distillation diagnostics:
  - `teacher_mu`
  - `teacher_log_std`
  - `teacher_value`
  - `teacher_log_prob`
  - `student_action`
  - `rollout_action`
  - `used_teacher_action`
  - `teacher_log_prob_teacher_mu`
  - `teacher_log_prob_student_action`
  - `teacher_log_prob_rollout_action`
  - `reward`
  - `done`
  - `absorbing`
  - `traj_no`
  - `subtraj_step_no`
  - `phase`
  - optional `full_obs`
- Add a helper such as `complete_distill_schema(data)` in `dataset.py`.
- Apply the helper before writing every teacher and DAgger shard.
- For teacher shards, fill DAgger-only fields with stable values:
  - `student_action = teacher_action`
  - `rollout_action = teacher_action`
  - `used_teacher_action = True`
  - `teacher_log_prob_teacher_mu = teacher_log_prob`
  - `teacher_log_prob_student_action = teacher_log_prob`
  - `teacher_log_prob_rollout_action = teacher_log_prob`
- Preserve `full_obs` only when explicitly requested; do not force it into every shard.
- Add a regression test that writes one teacher-like `train_000000.npz` and one DAgger-like `train_000001.npz`, then verifies `DistillDataset(..., split="train")` loads the combined split without field-length errors.
- Add assertions that the placeholder fields have the expected dtype and shape.

Validation:

```bash
uv run pytest tests/unit/test_distill_dataset.py tests/unit/test_dagger_collect.py -q
```

## Task 2: Add Shard-Level Dataset Inspection

Priority: P0.

Files:

- Modify: `musclemimic/distill/inspect_dataset.py`
- Test: `tests/unit/test_distill_dataset.py`
- Update docs: `docs/distillation/commands.md`
- Update docs: `docs/forehand_clear_student_distillation.md`

Plan:

- Add `--shard_level` to `musclemimic-distill-inspect-dataset`.
- Implement an inspector path that reads every `*.npz` directly without constructing `DistillDataset`.
- Report per shard:
  - filename
  - split inferred from filename
  - sample count
  - fields
  - shape and dtype per field
  - missing required fields
  - missing optional schema fields
- Keep the current split-level aggregate report as the default.
- If aggregate loading fails, include a clear error message and recommend rerunning with `--shard_level`.

Validation:

```bash
uv run pytest tests/unit/test_distill_dataset.py -q
uv run python -m musclemimic.distill.inspect_dataset --help
```

## Task 3: Record DAgger Iteration Provenance in Shards

Priority: P1.

Files:

- Modify: `musclemimic/distill/dagger_loop.py`
- Modify: `fullbody/distill_collect_dagger.py`
- Modify: `musclemimic/distill/dagger.py`
- Test: `tests/unit/test_dagger_loop.py`

Plan:

- Add optional CLI arguments to `fullbody.distill_collect_dagger`:
  - `--dagger_iteration`
  - `--rollout_policy`
- Make `dagger_loop.py` pass:
  - `--dagger_iteration <iteration>`
  - `--rollout_policy student_with_optional_teacher_mix`
- Add these fields to shard metadata:
  - `dagger_iteration`
  - `student_ckpt_in`
  - `teacher_ckpt`
  - `rollout_policy`
  - `collector = dagger_student_rollout_teacher_relabel`
- Keep existing `dagger_loop_manifest.json` and `dagger_loop_results.json`; do not duplicate large config blobs unnecessarily beyond existing metadata.

Validation:

```bash
uv run pytest tests/unit/test_dagger_loop.py -q
```

## Task 4: Add Machine-Readable Evaluation Metrics

Priority: P1.

Files:

- Modify: `fullbody/eval.py`
- Modify: `musclemimic/distill/eval_student.py`
- Test: add or update `tests/unit/test_bc_loss.py` or a focused eval metrics test file.

Plan:

- Add `fullbody/eval.py --metrics_output_json <path>`.
- When metrics are computed, write the metrics dict to that JSON path using sorted keys.
- Keep stdout printing for backward compatibility.
- Update `run_eval_metrics()` to request a temporary JSON file and parse it first.
- Keep stdout regex parsing as fallback only.
- Keep `validate_required_metrics()` unchanged except for tests that prove JSON and fallback modes both enforce required metrics.

Validation:

```bash
uv run pytest tests/unit/test_bc_loss.py -q
uv run python -m fullbody.eval --help
```

## Task 5: Expand Motion Override Support Where It Is Actually Needed

Priority: P1.

Files:

- Modify: `fullbody/distill_collect.py`
- Modify: `fullbody/distill_collect_dagger.py`
- Modify: Forehand wrappers only if the new overrides are useful for Forehand smoke tests.
- Test: add lightweight argument/command construction tests if feasible without building the heavy environment.

Plan:

- Keep existing `--motion_path`.
- Add `--motion_group`, `--traj_index`, and `--traj_start_step` only if the loaded config and environment factory can consume them through the same fields used by `fullbody/eval.py`.
- Reuse existing `fullbody/eval.py` override semantics instead of inventing a separate layout.
- If an override cannot be applied safely without constructing the environment, fail early with a clear CLI error rather than silently ignoring it.

Validation:

```bash
uv run python -m fullbody.distill_collect --help
uv run python -m fullbody.distill_collect_dagger --help
```

## Task 6: Add BC Checkpoint Roundtrip Coverage

Priority: P1.

Files:

- Add: `tests/unit/test_bc_checkpoint_roundtrip.py`
- Possibly modify: `tests/unit/test_train_bc_validation.py`

Plan:

- Prefer a lightweight test that creates a tiny synthetic dataset and runs `train_bc()` for one or two steps using an existing minimal/mock student config path.
- Verify:
  - a checkpoint path is written
  - `load_checkpoint()` can load it
  - loaded checkpoint contains `train_state.params` and `train_state.run_stats`
  - `PPOJax.init_agent_conf()` can rebuild the network from the checkpoint config
  - a zero observation with dataset student obs dim can run through the policy network
- If the real environment is too heavy for unit tests, mark the full environment roundtrip as integration and keep the unit test focused on checkpoint schema.

Validation:

```bash
uv run pytest tests/unit/test_bc_checkpoint_roundtrip.py tests/unit/test_train_bc_validation.py -q
```

## Task 7: Final Smoke and Packaging Checks

Priority: P0 for command help, P1 for full smoke run.

Files:

- Update docs: `docs/distillation/commands.md`
- Update docs: `docs/forehand_clear_distillation_runbook.md`

Plan:

- Verify all console/module entrypoints expose help:
  - `python -m fullbody.distill_collect --help`
  - `python -m fullbody.distill_train_bc --help`
  - `python -m fullbody.distill_collect_dagger --help`
  - `python -m fullbody.distill_run_dagger --help`
  - `python -m fullbody.distill_compare --help`
  - `python -m musclemimic.distill.inspect_dataset --help`
  - each Forehand wrapper `--help`
- Run the focused unit test set used in round 3 plus new round 4 tests.
- If a valid teacher checkpoint is available, run the minimum chain with `num_envs=2`, `num_steps=20`, `train_steps=5`, `batch_size=4`.

Validation:

```bash
uv run pytest \
  tests/unit/test_train_bc_validation.py \
  tests/unit/test_ppo.py \
  tests/unit/test_student_obs_filter.py \
  tests/unit/test_distill_dataset.py \
  tests/unit/test_bc_loss.py \
  tests/unit/test_dagger_collect.py \
  tests/unit/test_dagger_loop.py \
  tests/unit/test_distill_collect_teacher.py \
  tests/unit/test_distill_packaging.py \
  tests/unit/test_bc_checkpoint_roundtrip.py \
  -q

git diff --check -- \
  pyproject.toml \
  musclemimic/distill \
  fullbody \
  BadmintonMimic/scripts \
  BadmintonMimic/docs \
  docs/distillation \
  tests/unit
```

## Execution Order

1. Task 1: fix schema compatibility first; this is the only current full-chain blocker.
2. Task 2: make inspect usable even when schema is broken.
3. Task 3: add DAgger provenance while touching DAgger command paths.
4. Task 4: make eval compare robust with JSON metrics.
5. Task 5: add only safe motion overrides.
6. Task 6: add checkpoint roundtrip coverage.
7. Task 7: run focused verification and update docs.

# Distillation Audit for `0348709 add dagger distillation correction`

Date: 2026-06-02
Repository: `yangfeiyang-123/asi_strengthen_musclemimic`
Scope: only ForehandClear body-trajectory imitation. Racket, shuttle, impact physics, shot outcome, and grip/racket residual control are intentionally out of scope.

## 1. Executive conclusion

The current `0348709` state is **substantially closer** to the required no-future-lookahead student distillation pipeline than the previous `d0a32c9` state.

It now contains the following core building blocks:

1. Student observation filtering: keep non-goal state observations and keep only motion phase from the goal group.
2. Distillation shard dataset utilities: write/load `.npz` shards with metadata.
3. Teacher rollout collection: collect `student_obs -> teacher_action` pairs from a lookahead teacher.
4. Offline BC trainer: train a PPO-compatible student checkpoint from the distillation dataset.
5. DAgger-style correction core: roll out the student, relabel student-visited states with the teacher, and append new shards.

However, it still does **not yet fully satisfy** the complete implementation requirements for an end-to-end, reproducible distillation workflow. The remaining gaps are mostly integration, configuration, validation, and robustness work.

The short verdict is:

```text
Infrastructure status:       mostly present
Research pipeline status:    partial
End-to-end runnable workflow: not yet complete
Paper-quality evidence:      not yet complete
```

Recommended next work:

```text
1. Add official ForehandClear distillation configs.
2. Add command-line entrypoints or scripts for collect-teacher, train-BC, collect-DAgger, and evaluate-student.
3. Add an iterative DAgger driver.
4. Add student PPO fine-tune config and warm-start instructions.
5. Add teacher-vs-student evaluation reports and acceptance criteria.
6. Add tests for DAgger and BC checkpoint roundtrip.
```

## 2. What the current commit already satisfies

### 2.1 Student observation filtering is now implemented

The repository now has `musclemimic/distill/obs_filter.py`.

The implemented design matches the recommended first student policy:

```text
student_obs = all non-goal observations + motion phase
```

The file explicitly documents that the default student policy keeps all non-goal observations and only the final goal element, interpreted as the motion phase when `GoalTrajMimic.enable_motion_phase=True`.

This satisfies the key design requirement:

```text
Teacher observation:
  joint + muscle + foot contact + full goal lookahead

Student observation:
  joint + muscle + foot contact + phase
```

The implementation builds a `StudentObsSpec` containing:

```text
raw_obs_dim
goal_indices
state_indices
student_indices
phase_index
keep_motion_phase
```

It constructs `student_indices` as:

```text
state_indices + [phase_index]
```

when `keep_motion_phase=True`. The wrapper also updates the observation space and `obs_container`, and filters observations in `reset`, `reset_to`, and `step`.

Assessment: **satisfies the student observation filtering requirement.**

Remaining caveat: the implementation assumes that the final goal element is motion phase. This is true for the current `GoalTrajMimic` layout when `enable_motion_phase=True`, but should be protected by a stronger integration test using a real MyoFullBody/GoalTrajMimic environment, not only a mock env.

### 2.2 PPO training integration for filtered student observations is partially implemented

`musclemimic/algorithms/common/env_utils.py` now applies `StudentObservationFilterWrapper` inside `wrap_env()` when `config.student_obs_filter.enabled=True`.

The wrapper order is:

```text
StudentObservationFilterWrapper
  -> optional NStepWrapper
  -> VecEnv / LogWrapper / AutoResetWrapper
  -> NormalizeVecReward if enabled
```

This is the right order for the intended student design. Filtering should happen before history stacking, because the student should not accidentally stack or retain future lookahead components.

`PPOJax._create_network()` also wraps the environment when `student_obs_filter.enabled=True` before computing observation indices and constructing the network. This makes the network shape align with the filtered student observation shape.

Assessment: **mostly satisfies the student PPO input-shape requirement.**

Remaining caveat: because filtering happens both in `_create_network()` for network shape inference and in `wrap_env()` for actual training rollout, the two code paths must remain perfectly consistent. They currently both use `StudentObservationFilterWrapper`, which is good. A regression test should instantiate a real config and assert that:

```text
network input dimension == wrapped_env.info.observation_space.shape[0]
```

### 2.3 Distillation dataset IO is implemented

The repository now has `musclemimic/distill/dataset.py`.

It defines required fields:

```text
student_obs
teacher_action
```

and supports additional fields such as:

```text
teacher_mu
teacher_value
teacher_log_prob
reward
done
absorbing
traj_no
subtraj_step_no
phase
full_obs
student_action
```

It provides:

```text
write_distill_shard(path, data, metadata)
load_metadata(dataset_dir)
DistillDataset(dataset_dir)
DistillDataset.iter_batches(...)
```

This satisfies the minimum data format requirement for off-policy BC/KD distillation.

Assessment: **satisfies dataset IO requirement.**

Remaining caveat: the writer does not enforce a schema version. Add a metadata field such as:

```json
"schema_version": "distill_v1"
```

so future DAgger shards and BC shards remain compatible.

### 2.4 Off-policy teacher rollout collection is implemented

The repository now has `musclemimic/distill/collect_teacher.py`.

It collects teacher-generated data for student distillation. The function:

```python
collect_teacher_dataset(...)
```

builds a student observation spec, wraps the environment with the teacher config, runs the teacher policy, and saves shards containing:

```text
student_obs
teacher_action
teacher_mu
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

This is the correct first step for off-policy distillation.

Assessment: **satisfies the off-policy teacher dataset collection requirement at the library-function level.**

Remaining caveat: no official CLI/script appears to expose this functionality for ForehandClear experiments. Codex should add a runnable script with explicit teacher checkpoint, output directory, motion paths, num envs, num steps, and deterministic/stochastic teacher options.

### 2.5 Offline BC trainer is implemented

The repository now has `musclemimic/distill/train_bc.py`.

It provides:

```python
train_bc(...)
```

The trainer:

1. Ensures `student_obs_filter.enabled=True`.
2. Loads a `DistillDataset`.
3. Instantiates the environment and PPO student network.
4. Initializes a PPO-compatible `TrainState`.
5. Optimizes action MSE with optional value distillation.
6. Saves a PPO-compatible checkpoint using `UnifiedCheckpointManager`.
7. Writes `distill_metadata.json`.

This satisfies the minimum offline BC trainer requirement.

Assessment: **partially satisfies BC training requirement.**

Remaining caveats:

1. It needs an official command-line entrypoint or script.
2. It should verify that `dataset.student_obs_dim` exactly matches the filtered environment/network input dimension before training.
3. It should save the student filter config into the checkpoint metadata in a way that is unambiguous during evaluation and PPO fine-tuning.
4. It currently supports action MSE and optional value MSE, but not KL distillation against a full teacher Gaussian distribution. This is acceptable for v1, but KD should be added later if teacher uncertainty matters.

### 2.6 DAgger-style correction core is implemented

The repository now has `musclemimic/distill/dagger.py`.

The function:

```python
collect_dagger_dataset(...)
```

rolls out the **student policy**, labels the visited states using the **teacher policy mean action**, and writes additional distillation shards. This directly addresses the distribution-shift problem of pure off-policy BC.

It also supports:

```text
mix_teacher_action_prob
append
save_full_obs
metadata
```

This is the right conceptual direction.

Assessment: **partially satisfies DAgger requirement.**

Remaining caveats:

1. It is only a collection function, not an iterative DAgger driver.
2. It currently explicitly supports only `len_obs_history=1` for both teacher and student.
3. It needs a CLI/script that loads teacher checkpoint, student checkpoint, config, and output dataset directory.
4. It needs tests with mock teacher/student networks or a small fake environment.
5. It needs a documented loop:

```text
teacher dataset -> BC train student_0
for k in 1..K:
    collect DAgger dataset with student_k
    merge dataset
    retrain or continue BC -> student_{k+1}
```

## 3. What still does not satisfy the full requirements

### 3.1 No complete command-line workflow yet

At this commit, the core Python functions exist, but the workflow still needs user-facing scripts. The `pyproject.toml` project scripts only expose path/cache utilities and GMR cache download commands; it does not expose distillation commands.

Required additions:

```text
BadmintonMimic/scripts/collect_forehand_clear_teacher_dataset.py
BadmintonMimic/scripts/train_forehand_clear_student_bc.py
BadmintonMimic/scripts/collect_forehand_clear_dagger_dataset.py
BadmintonMimic/scripts/run_forehand_clear_dagger_loop.py
BadmintonMimic/scripts/evaluate_forehand_clear_student.py
```

or equivalent console scripts:

```toml
[project.scripts]
musclemimic-distill-collect-teacher = "musclemimic.distill.cli:collect_teacher_main"
musclemimic-distill-train-bc = "musclemimic.distill.cli:train_bc_main"
musclemimic-distill-collect-dagger = "musclemimic.distill.cli:collect_dagger_main"
musclemimic-distill-run-dagger = "musclemimic.distill.cli:run_dagger_main"
```

Without this, the implementation is usable by importing functions manually, but it is not yet a reproducible experimental pipeline.

### 3.2 No official ForehandClear student config found

The teacher config exists:

```text
fullbody/config_specific_task/conf_fullbody_badminton_gmr.yaml
```

But the distillation workflow needs at least one student config, for example:

```text
fullbody/config_specific_task/conf_fullbody_badminton_student_gmr.yaml
```

Recommended content:

```yaml
# @package _global_

defaults:
  - /config_specific_task/conf_fullbody_badminton_gmr
  - _self_

wandb:
  tags: ["fullbody", "gmr", "badminton", "student", "no_future_lookahead"]

experiment:
  student_obs_filter:
    enabled: true
    keep_motion_phase: true
    require_goal_group: true
    require_motion_phase: true

  # Recommended for student v1 unless history is explicitly supported in DAgger.
  len_obs_history: 1
  split_goal: false

  # Set for PPO fine-tune from BC checkpoint.
  resume_from: null
  reset_std_on_resume: 0.5
  total_timesteps: 20480000

  validation:
    active: true
    deterministic: true
```

### 3.3 BC trainer lacks a confirmed CLI and roundtrip test

The BC trainer function exists, but for reliability it needs a test proving:

```text
DistillDataset -> train_bc -> checkpoint -> fullbody/eval.py loads checkpoint -> student observation dimension matches env
```

Add an integration test with a tiny mock environment or a small demo configuration.

### 3.4 DAgger currently lacks iterative orchestration

The current `collect_dagger_dataset()` performs one relabeling collection pass.

A complete DAgger workflow should include an orchestration layer:

```python
for iteration in range(num_dagger_iters):
    collect_dagger_dataset(student_checkpoint=current_student)
    train_bc(dataset_dir=aggregated_dataset, resume_from=current_student or fresh)
    evaluate_student()
    select best checkpoint
```

Minimum metadata per DAgger iteration:

```json
{
  "dagger_iteration": 2,
  "teacher_checkpoint": "...",
  "student_checkpoint_in": "...",
  "student_checkpoint_out": "...",
  "num_samples_added": 123456,
  "mix_teacher_action_prob": 0.1,
  "mean_reward": 0.0,
  "done_rate": 0.0,
  "early_termination_rate": 0.0
}
```

### 3.5 Student PPO fine-tune is still not packaged as a workflow

The PPO infrastructure should work with `student_obs_filter.enabled=True`, but the repository still needs a documented fine-tune command:

```bash
uv run fullbody/experiment.py \
  --config-name=config_specific_task/conf_fullbody_badminton_student_gmr \
  experiment.resume_from=/path/to/bc/checkpoint \
  experiment.reset_std_on_resume=0.5 \
  wandb.tags='["student", "ppo_finetune", "no_future_lookahead"]'
```

Acceptance for this step:

```text
1. Student checkpoint loads.
2. PPO rollout does not crash due to observation shape mismatch.
3. Reward still uses reference trajectory.
4. Policy input excludes future lookahead.
5. Validation videos and metrics are generated.
```

### 3.6 Evaluation and acceptance criteria are not yet complete

Need a dedicated teacher-vs-student evaluation script/report.

Required metrics:

```text
Teacher vs Student:
  mean_episode_return
  completion_rate
  early_termination_rate
  mean_episode_length
  err_root_xyz
  err_root_yaw
  err_joint_pos
  err_joint_vel
  err_site_abs
  err_rpos
  reward_qpos
  reward_qvel
  reward_root_pos
  reward_root_vel
  reward_rpos
  reward_rquat
  reward_rvel_rot
  reward_rvel_lin
```

Suggested acceptance thresholds for initial student v1:

```text
BC one-step action MSE:             low and stable across train/val
Student rollout completion_rate:    >= 80% of teacher on the three ForehandClear clips
Student mean_episode_return:        >= 70% of teacher before PPO fine-tune
Student mean_episode_return:        >= 85% of teacher after PPO fine-tune
Early termination rate:             not worse than teacher by > 20 percentage points after PPO fine-tune
err_site_abs / err_rpos:            within acceptable tracking gap vs teacher
```

Exact thresholds can be adjusted after seeing baseline teacher metrics.

## 4. Potential correctness issues to address

### 4.1 `drop_goal_lookahead` is set but not consumed

`train_bc._ensure_student_filter()` sets:

```python
config.experiment.student_obs_filter.drop_goal_lookahead = True
```

But `obs_filter.py` does not currently consume a `drop_goal_lookahead` key. The current behavior is still correct because it drops all goal components except phase by construction, but the unused key is confusing.

Recommended fix:

1. Either remove `drop_goal_lookahead` from `_ensure_student_filter()`;
2. Or explicitly support it in `build_student_obs_indices()` and raise if `drop_goal_lookahead=False` is incompatible with `keep_motion_phase=True`.

### 4.2 `phase_index = goal_indices[-1]` should be validated in a real env test

The implementation assumes the final goal dimension is motion phase. This matches the current `GoalTrajMimic` construction when `enable_motion_phase=True`, but it is fragile if future goal layouts change.

Recommended fix:

Add an integration test with real `GoalTrajMimic` config:

```text
1. instantiate MyoFullBody ForehandClear env
2. get goal group indices
3. verify final goal index equals motion phase
4. verify student wrapper output final dimension is in [0, 1]
5. verify student obs excludes all future lookahead dimensions
```

### 4.3 DAgger supports only `len_obs_history=1`

This is acceptable for student v1, but it should be clearly documented in configs and scripts.

Recommended fix:

```yaml
experiment:
  len_obs_history: 1
  split_goal: false
```

for all DAgger v1 configs.

If history is later required, extend DAgger to use the same `ObservationHistoryBuffer` / `NStepWrapper(split_goal=True)` logic as inference and PPO training.

### 4.4 DAgger and teacher collection should use transition-state metadata where possible

Teacher collection uses `step_with_transition`, which is good. DAgger uses `rollout_env.step(...)` and then `info`. With `AutoResetWrapper`, current info may be post-autoreset or may contain wrapper-specific final info depending on wrapper behavior.

Recommended check:

Verify that `traj_no` and `subtraj_step_no` saved in DAgger shards refer to the state that was labeled, not the reset candidate after `done=True`. If ambiguous, use `step_with_transition()` in DAgger too.

### 4.5 BC evaluation updates running statistics during evaluation

`evaluate_bc_loss()` calls network with `mutable=["run_stats"]` but discards updates. This is pure in Flax, so it does not mutate the stored `train_state`, but it means loss is evaluated under batch-updated normalization behavior rather than frozen running statistics.

Recommended fix:

Add a helper for evaluation with fixed `run_stats`, or explicitly document that BC eval loss is measured with the same mutable normalization behavior as training.

## 5. Recommended next implementation plan

### Milestone 1: Make the existing functions runnable

Add CLI/scripts:

```text
collect teacher dataset
train BC student
collect DAgger correction dataset
run iterative DAgger loop
evaluate student
```

Each script should print and save:

```text
resolved config
teacher checkpoint
student checkpoint
student obs dim
action dim
num samples
output paths
metrics
```

### Milestone 2: Add official configs

Add:

```text
fullbody/config_specific_task/conf_fullbody_badminton_student_gmr.yaml
fullbody/config_specific_task/conf_fullbody_badminton_student_bc_eval.yaml  # optional
```

The first should be used for student PPO fine-tune and evaluation.

### Milestone 3: Add BC checkpoint roundtrip test

Test:

```text
mock dataset -> train_bc small steps -> checkpoint saved -> load checkpoint -> network action shape correct
```

### Milestone 4: Add DAgger unit test

Test with mock env and tiny teacher/student distributions:

```text
collect_dagger_dataset writes fields:
  student_obs
  teacher_action
  student_action
  teacher_mu
  reward
  done
  phase
```

and verify that student rollout action and teacher label differ when expected.

### Milestone 5: Add evaluation report generation

Add script:

```text
BadmintonMimic/scripts/evaluate_teacher_student_distill.py
```

It should generate:

```text
metrics.json
summary.md
optional videos
```

## 6. Final status matrix

| Requirement | Current status at `0348709` | Notes |
|---|---:|---|
| Lookahead teacher PPO exists | Yes | Existing ForehandClear GMR PPO setup remains valid. |
| Student obs = state + phase | Yes | Implemented by `StudentObservationFilterWrapper`. |
| Student PPO wrapper integration | Mostly yes | `wrap_env()` and `_create_network()` both integrate the wrapper. |
| Distillation dataset IO | Yes | `.npz` shard utilities exist. |
| Off-policy teacher collection | Yes, library function | Needs CLI/script and integration test. |
| BC student training | Yes, library function | Needs CLI/script, dimension checks, checkpoint roundtrip test. |
| DAgger correction | Partial | One-pass collection exists; iterative driver missing. |
| Student PPO fine-tune | Partial | PPO likely supports it, but official config and workflow missing. |
| Teacher-vs-student evaluation | No | Need report script and thresholds. |
| Paper-quality evidence | No | Need metrics, videos, ablations. |

## 7. Bottom-line recommendation

The `0348709` commit is a good implementation step. It likely satisfies the **core infrastructure** requirement for no-future-lookahead student distillation, including the newly requested DAgger correction primitive.

It does **not** yet satisfy the full experimental/research requirement, because the end-to-end workflow is not packaged or validated.

The next Codex task should not be another low-level wrapper. It should be an integration task:

```text
Create a complete ForehandClear distillation workflow:
1. collect teacher dataset
2. train BC student
3. collect DAgger correction data
4. retrain student
5. PPO fine-tune student
6. evaluate teacher vs student
7. write metrics/report artifacts
```

Once those are implemented and verified, the repository will meet the full requirement for a complete, reproducible student distillation pipeline for ForehandClear body-trajectory imitation.

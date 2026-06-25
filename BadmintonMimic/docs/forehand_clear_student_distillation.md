# ForehandClear Student Distillation

This workflow distills a full-lookahead ForehandClear teacher into a student
policy that observes only non-goal state features plus motion phase. The
environment still uses the reference trajectory for `MimicReward`; only the
policy observation is filtered.

## 1. Teacher Checkpoint

Start from a trained ForehandClear lookahead teacher checkpoint. The checkpoint
config should still point to the ForehandClear GMR clips:

```text
badminton/train/forehand_clear_clip1_merged_poses
badminton/train/forehand_clear_clip2_merged_poses
badminton/train/forehand_clear_clip3_merged_poses
```

## 2. Collect Teacher Dataset

```bash
uv run python BadmintonMimic/scripts/collect_forehand_clear_teacher_dataset.py \
  --teacher-path /path/to/teacher/checkpoint \
  --output-dir outputs/distill/forehand_clear/teacher_dataset \
  --num-envs 256 \
  --num-steps 200000 \
  --split train \
  --seed 0
```

Teacher collection forcibly disables `student_obs_filter` during rollout so the
teacher always receives full lookahead observations. Shards still store filtered
`student_obs` targets for student BC.

For latent posterior/decoder training, add:

```bash
--save-reference-features
```

This writes `reference_features` from the dropped goal lookahead terms and
records `reference_features_dim` in `metadata.json`. The motion phase remains in
`student_obs`; use `--include-reference-phase` only for experiments that
explicitly want phase duplicated in the posterior reference tensor.

Use `--motion-path`, `--motion-group`, `--traj-index`, and
`--traj-start-step` to override the checkpoint motion config for validation
splits or fixed-start smoke tests. `--motion-path` takes precedence over
`--motion-group`.

## 3. Train BC Student

```bash
uv run python BadmintonMimic/scripts/train_forehand_clear_student_bc.py \
  --config-name config_specific_task/conf_fullbody_badminton_student_gmr \
  --dataset-dir outputs/distill/forehand_clear/teacher_dataset \
  --output-dir outputs/distill/forehand_clear/bc_student \
  --num-steps 200000 \
  --batch-size 4096 \
  --value-distill-weight 0.1
```

The BC trainer validates `dataset.student_obs_dim` against the configured
student environment before training. If teacher shards include
`teacher_log_std`, `--gaussian-kl-weight` can add diagonal Gaussian KL
distillation.

Warm-start BC from an existing student checkpoint with:

```bash
--resume-student /path/to/student/checkpoints/checkpoint_N
```

## 4. DAgger Correction

One correction pass:

```bash
uv run python BadmintonMimic/scripts/collect_forehand_clear_dagger_dataset.py \
  --teacher-path /path/to/teacher/checkpoint \
  --student-path outputs/distill/forehand_clear/bc_student/checkpoints/checkpoint_200000 \
  --output-dir outputs/distill/forehand_clear/teacher_dataset \
  --num-envs 256 \
  --num-steps 50000 \
  --mix-teacher-action-prob 0.1 \
  --split train \
  --append
```

Use `--save-reference-features` on DAgger collection too when the aggregated
dataset will train a latent posterior/decoder rather than a direct-action BC
student.

Iterative DAgger:

```bash
uv run python BadmintonMimic/scripts/run_forehand_clear_dagger_loop.py \
  --teacher-path /path/to/teacher/checkpoint \
  --student-path outputs/distill/forehand_clear/bc_student/checkpoints/checkpoint_200000 \
  --config-name config_specific_task/conf_fullbody_badminton_student_gmr \
  --dataset-dir outputs/distill/forehand_clear/teacher_dataset \
  --output-dir outputs/distill/forehand_clear/dagger_loop \
  --num-iters 3 \
  --num-envs 256 \
  --num-steps 50000 \
  --train-steps 200000 \
  --mix-teacher-action-prob 0.1
```

DAgger shards include `rollout_action`, `used_teacher_action`,
`teacher_log_prob_student_action`, and `teacher_log_prob_rollout_action` for
diagnostics. Teacher rollout shards are written with compatible placeholder
fields, so initial teacher shards and appended DAgger shards can be loaded as
one BC training split.

`--freeze-run-stats` freezes persisted running-stat state during collection.
It does not implement a separate inference-normalization path.

## 5. PPO Fine-Tune

```bash
uv run python fullbody/experiment.py \
  --config-name config_specific_task/conf_fullbody_badminton_student_gmr \
  experiment.resume_from=outputs/distill/forehand_clear/dagger_loop/iter_002/checkpoints/checkpoint_200000 \
  experiment.reset_std_on_resume=0.5 \
  wandb.mode=online
```

The student policy input remains state plus phase. The reward remains
reference-trajectory `MimicReward`.

## 6. Evaluation

```bash
uv run python BadmintonMimic/scripts/evaluate_forehand_clear_student.py \
  --teacher-path /path/to/teacher/checkpoint \
  --student-path outputs/distill/forehand_clear/bc_student/checkpoints/checkpoint_200000 \
  --dagger-student-path outputs/distill/forehand_clear/dagger_loop/iter_002/checkpoints/checkpoint_200000 \
  --ppo-student-path /path/to/ppo_finetuned/checkpoint \
  --output-dir outputs/distill/forehand_clear/eval \
  --num-envs 20 \
  --num-steps 500
```

Outputs:

```text
comparison_metrics.json
comparison_table.csv
summary.md
```

Evaluation metrics are written through machine-readable JSON before the compare
report is assembled, which avoids depending on stdout formatting.

## 7. Observation Diagnostic

```bash
uv run python BadmintonMimic/scripts/inspect_student_obs_filter.py \
  --config-name config_specific_task/conf_fullbody_badminton_student_gmr
```

Confirm that `student_obs_dim = state_dim + 1`, that the kept goal index is the
raw phase index, and that all future lookahead goal features are dropped.

## Known Limitations

- DAgger v1 supports `len_obs_history=1` only.
- Real acceptance thresholds must be calibrated from teacher baseline metrics.
- Collection wrappers use checkpoint configs as the source of truth, with
  `--motion-path` and `--motion-group` available for dataset override.

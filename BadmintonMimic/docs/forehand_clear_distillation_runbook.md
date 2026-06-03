# ForehandClear Distillation Smoke Runbook

This runbook is the short validation pass to run before a long A100 job. It is
intended to take minutes, not hours.

## 1. Inspect Student Observation

```bash
forehand-clear-distill-inspect-obs \
  --config-name config_specific_task/conf_fullbody_badminton_student_gmr \
  --output-json /tmp/fc_obs_filter.json
```

Proceed only if `student_obs_dim < raw_obs_dim`, `kept_goal_indices` has length
1, and `dropped_goal_dim = goal_dim - 1`.

## 2. Collect Tiny Teacher Dataset

```bash
forehand-clear-distill-collect-teacher \
  --teacher-path /path/to/teacher/checkpoint_N \
  --output-dir /tmp/fc_distill_smoke \
  --num-envs 4 \
  --num-steps 32 \
  --shard-size 64 \
  --split train
```

Inspect the shards:

```bash
musclemimic-distill-inspect-dataset \
  --dataset_dir /tmp/fc_distill_smoke \
  --output_json /tmp/fc_distill_smoke/inspect.json
```

## 3. Train Tiny BC Student

```bash
forehand-clear-distill-train-bc \
  --dataset-dir /tmp/fc_distill_smoke \
  --output-dir /tmp/fc_student_bc_smoke \
  --num-steps 10 \
  --batch-size 16
```

## 4. Collect One Tiny DAgger Shard

```bash
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

After this smoke pass, inspect `used_teacher_action`, `rollout_action`,
`reward`, and `phase` in the dataset inspection output before scaling up.

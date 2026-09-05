# ForehandClear Distillation Smoke Runbook

This runbook is the short validation pass to run before a long A100 job. It is
intended to take minutes, not hours.

## 1. Inspect Student Observation

```bash
forehand-clear-distill-inspect-obs \
  --config-name config_specific_task/distill/conf_fullbody_forehandclear_racket_student_phase_bc \
  --output-json /tmp/fc_obs_filter.json
```

Proceed only if `student_obs_dim < raw_obs_dim`, `kept_goal_indices` has length
1, and `dropped_goal_dim = goal_dim - 1`.

## 2. Collect Tiny Teacher Dataset

```bash
forehand-clear-distill-collect-teacher \
  --teacher-path /path/to/teacher/checkpoint_N \
  --teacher-promotion-manifest /path/to/pipeline/stage2_promotion_manifest.json \
  --output-dir /tmp/fc_distill_smoke \
  --run-uid fc-distill-smoke-v1 \
  --num-envs 4 \
  --num-transitions 128 \
  --shard-size 64 \
  --split train
```

Inspect the shards:

```bash
musclemimic-distill-inspect-dataset \
  --dataset_dir /tmp/fc_distill_smoke \
  --output_json /tmp/fc_distill_smoke/inspect.json
```

If a mixed-schema or shard loading error appears, inspect individual files:

```bash
musclemimic-distill-inspect-dataset \
  --dataset_dir /tmp/fc_distill_smoke \
  --shard_level
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
  --teacher-promotion-manifest /path/to/pipeline/stage2_promotion_manifest.json \
  --student-path /tmp/fc_student_bc_smoke/checkpoints/checkpoint_10 \
  --output-dir /tmp/fc_distill_smoke \
  --run-uid fc-distill-smoke-v1 \
  --num-envs 4 \
  --num-transitions 64 \
  --shard-size 64 \
  --split train \
  --dagger-iteration 0 \
  --resume-dataset
```

## 5. Plan the Three EMG Arms

The `synergy_v3` launcher writes `pipeline_plan.json` without running anything,
so these three commands are safe to run locally and diff before dispatching a
GPU job. Each arm needs its own `--output_dir`.

Baseline (S2-B), no EMG anywhere in the sweep command:

```bash
python -m fullbody.run_forehand_clear_pipeline \
  --profile synergy_v3 --output_dir /tmp/arm_baseline
```

Privileged EMG arm (S2-C):

```bash
python -m fullbody.run_forehand_clear_pipeline \
  --profile synergy_v3 --output_dir /tmp/arm_peasd \
  --emg_reference_manifest /path/to/reference_tube.json \
  --emg_synergy_dim 3
```

Shuffled-context control (S2-D), the negative control the §884 gate compares
against:

```bash
python -m fullbody.run_forehand_clear_pipeline \
  --profile synergy_v3 --output_dir /tmp/arm_shuffled \
  --emg_reference_manifest /path/to/reference_tube.json \
  --emg_synergy_dim 3 \
  --emg_shuffle_context_ablation
```

Check the arms differ only in their EMG tail before spending GPU time:

```bash
python - <<'PY'
import json
for arm in ("baseline", "peasd", "shuffled"):
    plan = json.load(open(f"/tmp/arm_{arm}/pipeline_plan.json"))
    step = [s for s in plan["steps"] if s["name"] == "latent_dimension_sweep"][0]
    print(arm, [t for t in step["command"] if "emg" in t.lower()])
PY
```

The baseline must print an empty list. S2-C and S2-D land in different run
directories, so the gate can read both; if they collided, one arm would
silently overwrite the other's checkpoints.

`--emg_shuffle_context_ablation` takes an optional value, so `False`, `0`, and
`off` all disable it and an unparseable value fails at the command line. It is
rejected outright without `--emg_reference_manifest`, since shuffling a context
that was never supplied would produce a control identical to the baseline.

After this smoke pass, inspect `used_teacher_action`, `rollout_action`,
`reward`, and `phase` in the dataset inspection output before scaling up.
The teacher and DAgger shards share a superset schema, so this transactionally
committed dataset should still load as one `train` split for the next BC pass.
The smoke path still validates a real Stage-2 promotion artifact. The explicit
test-only unpromoted-teacher switch is reserved for isolated unit tests and its
outputs cannot pass production promotion.

# Distillation Commands

Collect teacher rollouts:

```bash
uv run python -m fullbody.distill_collect \
  --teacher_ckpt /path/to/teacher/checkpoint_123 \
  --output_dir datasets/distill/forehandclear_teacher_v1 \
  --num_envs 256 \
  --num_steps 200000 \
  --motion_group FOREHAND_CLEAR_TRAIN \
  --split train \
  --freeze_run_stats \
  --deterministic_teacher \
  --teacher_action_target mean
```

Use `--motion_path ...` for explicit clips, or `--motion_group ...` for a
dataset group. `--motion_path` takes precedence. `--traj_index` and
`--traj_start_step` can pin collection to a fixed trajectory start for smoke
tests and diagnostics.

Train BC student:

```bash
uv run python -m fullbody.distill_train_bc \
  --dataset_dir datasets/distill/forehandclear_teacher_v1 \
  --student_config fullbody/config_specific_task/distill/conf_fullbody_forehandclear_student_phase_bc.yaml \
  --output_dir outputs/distill/forehandclear_student_phase_bc \
  --batch_size 4096 \
  --num_steps 200000 \
  --lr 3e-4 \
  --gaussian_kl_weight 0.0 \
  --seed 0
```

Warm-start BC student:

```bash
uv run python -m fullbody.distill_train_bc \
  --dataset_dir datasets/distill/forehandclear_teacher_v1 \
  --student_config config_specific_task/conf_fullbody_badminton_student_gmr \
  --init_ckpt /path/to/student_bc/checkpoints/checkpoint_200000 \
  --output_dir outputs/distill/forehandclear_student_dagger_bc \
  --num_steps 100000
```

Collect DAgger student-rollout relabel shards:

```bash
uv run python -m fullbody.distill_collect_dagger \
  --teacher_ckpt /path/to/teacher/checkpoint_123 \
  --student_ckpt /path/to/student_bc/checkpoints/checkpoint_200000 \
  --output_dir datasets/distill/forehandclear_dagger_v1 \
  --num_envs 256 \
  --num_steps 50000 \
  --dagger_iteration 0 \
  --rollout_policy student_with_optional_teacher_mix \
  --split train \
  --freeze_run_stats \
  --append
```

Teacher and DAgger shards use a shared superset schema, so a single split can
mix initial teacher rollout shards and appended DAgger correction shards.

Run the iterative DAgger loop:

```bash
uv run python -m fullbody.distill_run_dagger \
  --teacher_ckpt /path/to/teacher/checkpoint_123 \
  --initial_student_ckpt /path/to/student_bc/checkpoints/checkpoint_200000 \
  --student_config fullbody/config_specific_task/conf_fullbody_badminton_student_gmr.yaml \
  --dataset_dir datasets/distill/forehandclear_teacher_v1 \
  --output_dir outputs/distill/forehandclear_dagger_loop \
  --num_iters 3 \
  --num_envs 256 \
  --num_steps 50000 \
  --train_steps 200000 \
  --split train \
  --freeze_run_stats \
  --gaussian_kl_weight 0.0 \
  --mix_teacher_action_prob 0.1
```

The loop appends DAgger relabel shards into `--dataset_dir`, retrains BC on the
aggregated dataset after each collection pass, and writes
`dagger_loop_manifest.json` plus `dagger_loop_results.json` under
`--output_dir`.

Fine-tune with PPO:

```bash
uv run python fullbody/experiment.py \
  --config-name=config_specific_task/conf_fullbody_badminton_student_gmr \
  experiment.resume_from=/path/to/student_bc_or_dagger/checkpoints/checkpoint_200000 \
  experiment.reset_std_on_resume=0.5 \
  wandb.tags='["student", "ppo_finetune", "no_future_lookahead"]'
```

Compare checkpoints:

```bash
uv run python -m fullbody.distill_compare \
  --teacher_ckpt /path/to/teacher/checkpoint_123 \
  --student_ckpt /path/to/student_bc/checkpoints/checkpoint_200000 \
  --student_dagger_ckpt /path/to/student_dagger/checkpoints/checkpoint_250000 \
  --student_ppo_ckpt /path/to/student_ppo/checkpoint_456 \
  --output_dir outputs/distill/comparison \
  --motion_path \
    badminton/train/forehand_clear_clip1_merged_poses \
    badminton/train/forehand_clear_clip2_merged_poses \
    badminton/train/forehand_clear_clip3_merged_poses \
  --metrics_envs 20 \
  --metrics_steps 1000 \
  --eval_seed 0
```

`fullbody/distill_compare.py` writes:

```text
comparison_metrics.json
comparison_table.csv
summary.md
```

Metric collection asks `fullbody.eval` for a machine-readable JSON metrics file
and falls back to stdout parsing only if the JSON file is unavailable.

The same report entrypoint is available as:

```bash
uv run python musclemimic/badminton/scripts/evaluate_teacher_student_distill.py ...
```

ForehandClear task-specific wrappers:

```bash
uv run python musclemimic/badminton/scripts/collect_forehand_clear_teacher_dataset.py ...
uv run python musclemimic/badminton/scripts/train_forehand_clear_student_bc.py ...
uv run python musclemimic/badminton/scripts/collect_forehand_clear_dagger_dataset.py ...
uv run python musclemimic/badminton/scripts/run_forehand_clear_dagger_loop.py ...
uv run python musclemimic/badminton/scripts/evaluate_forehand_clear_student.py ...
uv run python musclemimic/badminton/scripts/inspect_student_obs_filter.py ...
```

Dataset inspection:

```bash
uv run python -m musclemimic.distill.inspect_dataset \
  --dataset_dir datasets/distill/forehandclear_teacher_v1 \
  --output_json outputs/distill/forehandclear_teacher_v1_inspect.json
```

If aggregate loading fails or you need per-shard schema diagnostics:

```bash
uv run python -m musclemimic.distill.inspect_dataset \
  --dataset_dir datasets/distill/forehandclear_teacher_v1 \
  --shard_level
```

Dataset shard naming:

```text
shard_*.npz    generic unsplit shards
train_*.npz    training split
val_*.npz      validation split
test_*.npz     held-out diagnostic split
```

`--freeze_run_stats` freezes persisted checkpoint running-stat state during
collection. The current collector still uses the normal network apply path and
then discards updates; it is not a separate inference-normalization mode.

Unified reproducible direct BC / DAgger run:

```bash
uv run python -m fullbody.run_distill_experiment \
  --teacher_ckpt /path/to/teacher/checkpoint_123 \
  --student_config fullbody/config_specific_task/distill/conf_fullbody_forehandclear_student_phase_bc.yaml \
  --motion_path badminton/train/forehand_clear_clip1_merged_poses \
  --out_dir runs/distill/forehand_clear_v1 \
  --collect_train \
  --train_bc \
  --run_dagger 3 \
  --compare
```

This writes `manifest.json` and `final_report.md` under `--out_dir`, with fixed
subdirectories:

```text
dataset/
bc/
dagger/
compare/
```

Latent distillation requires strict shards with posterior reference features:

```bash
uv run python -m fullbody.distill_collect \
  --teacher_ckpt /path/to/teacher/checkpoint_123 \
  --output_dir datasets/distill/forehandclear_latent_v1 \
  --num_envs 256 \
  --num_steps 200000 \
  --split train \
  --teacher_action_target mean \
  --save_reference_features
```

Train posterior/prior/decoder:

```bash
uv run python -m fullbody.latent_train \
  --dataset_dir datasets/distill/forehandclear_latent_v1 \
  --output_dir outputs/latent/forehandclear_v1 \
  --latent_dim 32 \
  --horizon 8 \
  --kl_weight 0.001 \
  --kl_warmup_steps 10000 \
  --smooth_weight 0.1
```

Inspect the latent checkpoint:

```bash
uv run python -m fullbody.latent_eval \
  --checkpoint_dir outputs/latent/forehandclear_v1/latent_checkpoint
```

Prepare a LAB high-level PPO manifest:

```bash
uv run python -m fullbody.latent_run_lab_ppo \
  --latent_checkpoint_dir outputs/latent/forehandclear_v1/latent_checkpoint \
  --highlevel_config config_specific_task/distill/conf_fullbody_badminton_student_action_conditioned \
  --output_dir outputs/latent/forehandclear_v1/lab_ppo \
  --lambda_lab 1.0
```

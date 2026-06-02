# Distillation Commands

Collect teacher rollouts:

```bash
uv run python fullbody/distill_collect.py \
  --teacher_ckpt /path/to/teacher/checkpoint_123 \
  --output_dir datasets/distill/forehandclear_teacher_v1 \
  --num_envs 256 \
  --num_steps 200000 \
  --deterministic_teacher
```

Train BC student:

```bash
uv run python fullbody/distill_train_bc.py \
  --dataset_dir datasets/distill/forehandclear_teacher_v1 \
  --student_config fullbody/config_specific_task/distill/conf_fullbody_forehandclear_student_phase_bc.yaml \
  --output_dir outputs/distill/forehandclear_student_phase_bc \
  --batch_size 4096 \
  --num_steps 200000 \
  --lr 3e-4 \
  --seed 0
```

Collect DAgger student-rollout relabel shards:

```bash
uv run python fullbody/distill_collect_dagger.py \
  --teacher_ckpt /path/to/teacher/checkpoint_123 \
  --student_ckpt /path/to/student_bc/checkpoints/checkpoint_200000 \
  --output_dir datasets/distill/forehandclear_dagger_v1 \
  --num_envs 256 \
  --num_steps 50000 \
  --append
```

Continue BC/KD on the aggregated dataset by pointing `--dataset_dir` at the
directory containing both teacher rollout shards and DAgger relabel shards, or
by copying DAgger shards into the original dataset directory before rerunning
`fullbody/distill_train_bc.py`.

Fine-tune with PPO:

```bash
uv run python fullbody/experiment.py \
  --config-name=config_specific_task/distill/conf_fullbody_forehandclear_student_phase_ppo \
  experiment.resume_from=/path/to/student_bc/checkpoints/checkpoint_200000
```

Compare checkpoints:

```bash
uv run python fullbody/distill_compare.py \
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

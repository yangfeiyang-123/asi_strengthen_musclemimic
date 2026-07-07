# Badminton rollout biomechanics visualization

This directory contains an offline visualizer for an already trained MuscleMimic checkpoint.

## 1. Export a rollout

Use MuJoCo export because the exporter records `qpos`, `qvel`, `qacc`, muscle `ctrl`, and muscle `act`.

```bash
cd /data3/yangfeiyang/WorkSpace/musclemimic

export AMASS_PATH=/data3/yangfeiyang/WorkSpace/musclemimic/musclemimic/badminton/data/amass_npz
export MUSCLEMIMIC_AMASS_PATH=$AMASS_PATH
export CONVERTED_AMASS_PATH=/data3/yangfeiyang/WorkSpace/musclemimic/caches/AMASS
export MUSCLEMIMIC_CONVERTED_AMASS_PATH=$CONVERTED_AMASS_PATH
export SMPL_MODEL_PATH=/data3/yangfeiyang/WorkSpace/musclemimic/smpl_models/smplh
export MUSCLEMIMIC_SMPL_MODEL_PATH=$SMPL_MODEL_PATH

uv run python fullbody/eval.py \
  --path checkpoints/8f2862e76095/checkpoint_1000 \
  --use_mujoco \
  --no_render \
  --traj_index 0 \
  --traj_start_step 0 \
  --n_steps 350 \
  --export_trajectory \
  --trajectory_dir trajectory_data
```

If `uv run` is blocked by the local shell/snap environment, use `.venv/bin/python` in the same commands.

For other clips, change `--traj_index` to `1` or `2`. If you evaluate from a random start, the reference-error plot may be offset; prefer `--traj_start_step 0` for clean policy-vs-reference comparisons.

## 2. Plot key muscles and joints

```bash
uv run python visualize/analyze_rollout_biomechanics.py \
  --input trajectory_data/myofullbody_episodes_mujoco_YYYYMMDD_HHMMSS.npz \
  --outdir visualize/output/badminton_ckpt1000
```

The script writes per-episode outputs:

- `muscle_summary.csv`: ranked key muscle activation and command statistics.
- `joint_summary.csv`: ranked key joint range, velocity, and acceleration statistics.
- `tracking_error.csv`: policy-vs-reference root and qpos error when reference qpos is present.
- `muscle_activation_heatmap.png`
- `muscle_activation_dynamics_panel.png`: example-style combined panel. The left block is all muscle activations at one selected frame; the right stacked plots show full-sequence activation profiles for ranked key muscles.
- `key_muscle_activation_profiles.png`
- `key_muscle_command_profiles.png`
- `joint_position_heatmap.png`
- `joint_velocity_heatmap.png`
- `key_joint_position_profiles.png`
- `key_joint_velocity_profiles.png`
- `tracking_error.png`

Useful filters:

```bash
uv run python visualize/analyze_rollout_biomechanics.py \
  --input trajectory_data/myofullbody_episodes_mujoco_YYYYMMDD_HHMMSS.npz \
  --outdir visualize/output/right_arm_focus \
  --muscle-pattern 'delt|supsp|infsp|subsc|pecm|lat|tri|bic|bra|brd|ecr|ecu|fcr|fcu' \
  --joint-pattern 'shoulder|elbow|wrist|radioulnar|lumbar|pelvis'
```

To make the combined muscle dynamics panel focus on a specific frame or show more traces:

```bash
uv run python visualize/analyze_rollout_biomechanics.py \
  --input trajectory_data/myofullbody_episodes_mujoco_YYYYMMDD_HHMMSS.npz \
  --outdir visualize/output/right_arm_focus \
  --activation-snapshot-step 120 \
  --activation-profile-count 10
```

Use `--activation-snapshot-step -1` to keep the default middle-frame snapshot.

New trajectory exports include `muscle_names` in the NPZ. The visualizer uses those exported names first, so the plotted labels and `muscle_summary.csv` match the exact actuator order used by the evaluated environment. Older NPZ files without `muscle_names` fall back to model metadata when the actuator count matches, otherwise to numbered labels.

# ForehandClear Stand-Tail Post-Training

## Goal

The 27-video ForehandClear policy tracks the demonstration, but the target
trajectory ends before the body has a reason to recover and stand. The
post-training target is therefore:

1. preserve or improve the original action tracking;
2. after the original motion ends, hold a stable zero-velocity final stance for
   at least 2 seconds.

## Implemented Data Path

Do not overwrite the original `10trajectories_smooth` cache. Generate a new
cache namespace:

```bash
.venv/bin/python BadmintonMimic/scripts/extend_retarget_cache_with_stand_tail.py \
  --manifest BadmintonMimic/manifests/10trajectories_smooth_27_list.txt \
  --hold-seconds 2.0 \
  --settle-seconds 0.5 \
  --anchor-window-seconds 0.25
```

Outputs:

```text
caches/AMASS/MyoFullBody/gmr/10trajectories_smooth_stand_tail/*.npz
BadmintonMimic/manifests/10trajectories_smooth_27_stand_tail_list.txt
```

The script keeps the original frames intact, appends a 0.5 s smooth settle
segment, then appends a 2.0 s repeated hold segment. It recomputes `qvel`,
`xpos`, `xquat`, `cvel`, `subtree_com`, `site_xpos`, and `site_xmat` with the
MyoFullBody MuJoCo model.

Run stand-tail QC before training:

```bash
.venv/bin/python BadmintonMimic/scripts/qc_stand_tail_cache.py \
  --manifest BadmintonMimic/manifests/10trajectories_smooth_27_stand_tail_list.txt
```

Current QC result for this data release:

```text
27/27 passed
max_tail_qvel = 6.94e-16
max_tail_qpos_step = 0
min_root_height_hold = 0.939 m
max_com_support_margin = 0.041 m
```

## Training Config

Use:

```bash
MM_CUDA_VISIBLE_DEVICES=0 \
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp \
scripts/run_with_cuda_compat.sh .venv/bin/python fullbody/experiment.py \
  --config-name=config_specific_task/posttrain/ForehandClearStandTail/v1/E1_27demo_stand_tail
```

Training commands in this repo should use `scripts/run_with_cuda_compat.sh`
so CUDA 12.4 compat libraries are prepended to `LD_LIBRARY_PATH`. Select the GPU
with `MM_CUDA_VISIBLE_DEVICES`; do not rely on a bare `CUDA_VISIBLE_DEVICES=...`
training launch.

Default resume checkpoint:

```text
checkpoints/999d68245dd4/checkpoint_7812
```

The training dataset intentionally mixes:

- the original 27 `10trajectories_smooth/*` caches, to preserve action tracking;
- the 27 `10trajectories_smooth_stand_tail/*` caches, to teach the recovery and
  hold phase.

Validation uses only the 27 stand-tail caches so the final verdict measures the
new requirement directly.

If a newer 27-video checkpoint is preferred, override:

```bash
experiment.resume_from=/absolute/path/to/checkpoint_N
```

## Reward And Termination Changes

Compared with the initial tracking run, this post-training config:

- increases `qvel_w_sum`, `root_vel_w_sum`, and `rvel_w_sum` so the hold tail
  actually becomes still;
- keeps nonzero `root_pos_w_sum` and adds absolute rewards for pelvis,
  ankles, and toes so the final stance remains anchored in world space;
- adds small action-rate and activation-energy penalties to reduce oscillation;
- uses tighter root/site termination and validation thresholds.

## Quick Preview

Render one generated cache:

```bash
.venv/bin/python BadmintonMimic/scripts/render_retarget_cache.py \
  --motion 10trajectories_smooth_stand_tail/video1_best_smpl \
  --output-dir BadmintonMimic/outputs/vis/stand_tail \
  --width 640 \
  --height 480 \
  --stride 4 \
  --format mp4
```

Example output:

```text
BadmintonMimic/outputs/vis/stand_tail/10trajectories_smooth_stand_tail_video1_best_smpl.mp4
```

## Acceptance Checks

Run post-training evaluation from the beginning for the full 800 validation
steps. Accept a checkpoint only if:

- action phase still has low root/site tracking error relative to the 27-demo
  baseline;
- hold phase has root speed near zero and no body fall;
- last 2 seconds keep pelvis/ankle/toe absolute-site errors within the
  validation termination thresholds;
- rendered validation videos show no visible stepping drift or late collapse.

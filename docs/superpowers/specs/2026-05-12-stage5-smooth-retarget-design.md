# Stage5 Smooth Retarget Design

## Context

The latest stage5 forehand-clear data was converted from WHAM to AMASS-style NPZ at 60 Hz and then retargeted to MyoFullBody GMR caches. Visual inspection showed several bad musculoskeletal trajectories, mainly large arm and hand jumps.

The discontinuity check in `visualize/msk_retarget/discontinuity_report.md` indicates:

- Retarget-before AMASS root translation and SMPL rotations are smooth enough for this use case.
- `frame_ids` are continuous in the original stage5 WHAM files.
- Retarget-after caches have large `left_hand_mimic` site speed spikes in `video3` through `video10`.
- The largest related robot joint step is usually `shoulder_elv_l`.
- Root motion remains smooth, and GMR `pos_error` is moderate, so the issue is not a global root jump or complete IK failure.

The current GMR config uses `use_velocity_limit: false`, and the default MyoFullBody mapping strongly tracks upper-limb sites. For training, the priority is smooth and usable muscle trajectories, not exact hand-site matching.

## Goal

Generate a second, training-oriented MyoFullBody retarget dataset for the 10 stage5 demos that is smoother and safer for policy training, while preserving the original 60 Hz SMPL/AMASS inputs and the current baseline retarget outputs for comparison.

Success criteria:

- Keep the original AMASS-style files unchanged.
- Write new retarget caches under a separate motion namespace, for example `forehand_clear/stage5_10demo_smooth`.
- Reduce large hand-site spikes, especially `left_hand_mimic`.
- Reduce large `shoulder_elv_l` qpos steps.
- Keep root trajectory smooth.
- Keep retarget position errors within a reasonable range for imitation training.
- Produce side-by-side diagnostics and visualization so the smooth version can be compared against the current baseline.

## Recommended Approach

Use a training-oriented GMR configuration:

1. Enable GMR velocity limiting with `use_velocity_limit: true`.
2. Increase solver damping from `0.5` to a conservative value such as `1.0`.
3. Create a dedicated GMR mapping config, for example `smplh_to_myofullbody_smooth_train.json`.
4. Reduce tracking weights for upper-limb end-effectors, primarily wrist/hand tracking and, if needed, elbow tracking.
5. Keep pelvis, head, trunk, and lower-body tracking weights close to the current defaults.
6. Retarget into a new cache namespace rather than overwriting the existing stage5 outputs.

This directly addresses the observed failure mode: the solver is currently allowed to take aggressive joint steps to satisfy fast hand-site targets. Velocity limits and lower hand-site weights make the solution prefer smooth, trainable motion over exact wrist tracking.

## Alternatives Considered

### A. Only Enable Velocity Limit

This is the smallest change and should be tested first if we want a minimal ablation. It may reduce spikes, but if the wrist targets remain high weight, the solver can still create poor upper-limb motion around difficult frames.

### B. Velocity Limit Plus Upper-Limb Weight Reduction

This is the recommended path. It is still localized to GMR behavior, leaves SMPL data unchanged, and matches the training priority. The trade-off is less accurate hand tracking during fast swing phases.

### C. Post-process Retargeted Qpos

This can quickly smooth existing cache files, but it risks violating model constraints and producing inconsistent kinematics. It should not be the primary fix for training data.

## Implementation Shape

Add support for a custom GMR mapping config path if the current retarget pipeline cannot already receive one from `gmr_config`.

Expected configuration fields:

- `gmr_config.ik_config_path`: optional path to a GMR mapping JSON file.
- `gmr_config.use_velocity_limit: true`
- `gmr_config.damping: 1.0`
- `gmr_config.target_fps: 60`

Create a stage5 smooth config or script path that:

- Copies the 10 AMASS motion names to a new smooth manifest or namespace.
- Retargets to `forehand_clear/stage5_10demo_smooth/video*_lower_body_full_poses`.
- Uses `--clear-cache` only for the smooth namespace.
- Does not overwrite `forehand_clear/stage5_10demo`.

## Validation

Run the existing discontinuity analysis on both baseline and smooth caches.

Required checks:

- AMASS input remains `mocap_framerate = mocap_frame_rate = 60`.
- Smooth cache count is 10 main NPZ files plus 10 analysis NPZ files.
- `left_hand_mimic` and `right_hand_mimic` site speed maxima decrease compared with baseline.
- `shoulder_elv_l` qpos step maxima decrease compared with baseline.
- Root speed maxima stay comparable to baseline.
- Render all 10 smooth trajectories at original playback speed into a separate visualization directory.

The fix should be accepted only if the smooth outputs visibly remove the large arm jumps and the metrics confirm the reduction.

## Non-goals

- Do not modify or smooth the original WHAM pkl files.
- Do not modify the AMASS-style 60 Hz input files.
- Do not overwrite the current baseline GMR caches.
- Do not solve racket or shuttle tracking.
- Do not tune PPO rewards in this change.


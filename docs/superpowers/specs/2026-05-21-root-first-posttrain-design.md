# Root-First Forehand Net Lift Post-Train Design

## Context

The immediate target is the ForehandNetLift failure mode where the policy performs a plausible stroke posture but does not move the root forward enough. The first stage should optimize for visible and measurable root tracking improvement, not for a full SMPL or court-calibration redesign.

The approved direction is a root-first PPO post-train path using the existing trained ForehandNetLift policy:

- Checkpoint root: `/data3/yangfeiyang/WorkSpace/musclemimic/checkpoints/ForehandNetLift/forehand_net_lift_best_ppo`
- Preferred checkpoint: the largest numbered checkpoint under that directory, currently `checkpoint_7812`
- Source SMPL data: `/data3/yangfeiyang/WorkSpace/musclemimic/BadmintonMimic/data/ForehandNetLift/best`

The checkpoint manifest shows the original training dataset used `forehand_net_lift/best/video01` through `video08`, with GMR target FPS 100, damping 1.0, velocity limits enabled, and `smplh_to_myofullbody_smooth_train.json`. The source data directory also contains `video09` and `video10`. To keep the first post-train resume compatible and interpretable, stage one should keep the checkpoint's original eight-motion dataset. The ten-motion dataset can be a follow-up once the root-first objective is validated.

Code inspection supports the diagnosis:

- `fullbody/conf_fullbody.yaml` weights root position and root velocity at `0.1` each, while relative site position is `0.6`.
- `MimicReward` excludes the root free joint from joint qpos/qvel errors; root tracking only enters through separate root terms.
- `GoalTrajMimic` concise lookahead provides future reference root deltas but not the current policy-vs-reference root error.
- Default validation uses `MeanRelativeSiteDeviationTerminalStateHandler`, which is insensitive to global translation.

## Goal

Create a root-first post-train path that improves ForehandNetLift root motion from an existing checkpoint.

Success is measured by:

- Lower `root_xy_rmse`
- Lower `root_xy_final_error`
- Higher rollout/reference root displacement ratio
- Lower or stable `right_hand_abs_error`
- Rollout video visibly shows forward root motion rather than an in-place lift

Training reward alone is not an acceptance metric.

## Non-Goals

- Do not redesign WHAM, SMPL optimization, or court calibration in this stage.
- Do not overwrite existing checkpoints or retarget caches.
- Do not mix `video09` and `video10` into the first resume run.
- Do not change baseline config behavior unless the new options are explicitly enabled.
- Do not make ASI or curriculum the primary fix before the root objective is corrected.

## Recommended Approach

Implement a `RootFirstPostTrain` path with default-off code hooks:

1. Add current root error to `GoalTrajMimic`.
2. Add optional absolute site reward to `MimicReward`, initially for `right_hand_mimic`.
3. Add root-first diagnostics for reference cache and rollout.
4. Add a ForehandNetLift post-train config that resumes from `checkpoint_7812`, keeps the original eight motions, tightens root validation, and uses root-heavy reward weights.

This is the smallest path that directly attacks the observed failure while preserving an escape route if checkpoint restore is blocked by observation dimension changes.

## Architecture

### Goal Observation

Add an optional `goal_params.include_current_root_error` flag, default `false`.

When enabled in `GoalTrajMimic`, append:

- `ref_root_pos_current - sim_root_pos_current`, after applying the same per-episode XY origin alignment used by reward and terminal logic
- `ref_root_vel_current - sim_root_vel_current`
- `root_yaw_error`, wrapped to `[-pi, pi]`

This gives the policy a closed-loop correction signal. Future reference deltas remain useful, but the policy also sees whether it is currently behind the reference.

If this changes observation shape and prevents full checkpoint restoration, use the fallback config described below.

### Reward

Keep `MimicReward` backward compatible and add optional absolute site tracking:

- `absolute_site_reward_sites`: list of mimic sites, initially `["right_hand_mimic"]`
- `absolute_site_w_sum`: scalar weight, default `0.0`
- `absolute_site_w_exp`: exponential distance coefficient

The reward compares current world-space site positions against reference world-space site positions with the same initial XY offset alignment already used for root/site diagnostics. For ForehandNetLift, this prevents a local arm pose from scoring well when the right hand is spatially wrong because the root did not move.

The root-first starting reward should be:

- `qpos_w_sum: 0.05`
- `qvel_w_sum: 0.08`
- `root_pos_w_sum: 0.35`
- `root_vel_w_sum: 0.25`
- `rpos_w_sum: 0.30`
- `rquat_w_sum: 0.01`
- `rvel_w_sum: 0.06`
- `absolute_site_w_sum: 0.10`

These numbers should be treated as the first controlled post-train setting, not a universal final setting.

### Terminal And Validation

Use `MeanRelativeSiteDeviationWithRootTerminalStateHandler` for both training and validation.

Recommended first thresholds:

- `mean_site_deviation_threshold: 0.45`
- `root_deviation_threshold: 0.30`
- `root_orientation_threshold: 0.70`
- `enable_site_check: true`

This makes a policy that stands in place fail quickly instead of looking acceptable under root-relative validation.

### PPO Post-Train Config

Create a dedicated ForehandNetLift root-first config rather than modifying the existing base config.

The config should:

- Resume from `/data3/yangfeiyang/WorkSpace/musclemimic/checkpoints/ForehandNetLift/forehand_net_lift_best_ppo/checkpoint_7812`
- Use the original eight motion paths from the checkpoint manifest: `forehand_net_lift/best/video01` through `video08`
- Keep GMR settings from the checkpoint manifest: target FPS 100, damping 1.0, velocity limit enabled, fitted shape enabled, smooth-train IK config
- Set `reset_lr_schedule_on_resume: true`
- Use `lr: 5e-5` to `1e-4`
- Use `ppo_config.num_steps: 64` or `80`
- Use `ppo_config.update_epochs: 2`
- Use `ppo_config.init_std: 0.5` to `1.0` if the checkpoint restore path allows this safely
- Keep `ent_coef: 0.0` for conservative fine-tuning unless the policy collapses early

### Checkpoint Selection

Implementation should select the largest numeric `checkpoint_*` directory by default if the user does not override the checkpoint. For the current directory, this resolves to `checkpoint_7812`.

The run should log the selected checkpoint path before training starts.

### Fallback Path

If `include_current_root_error` changes observation shape and full checkpoint restore cannot proceed:

1. Run a reward-only post-train baseline with unchanged observation shape:
   - root-heavy reward
   - absolute right-hand reward
   - strict root terminal
   - strict validation metrics
2. If reward-only improves root displacement ratio, keep it as the first stage.
3. If reward-only does not improve enough, add partial checkpoint loading or a short warm-start run with the new observation shape.

## Data Flow

Reference data:

`ForehandNetLift/best SMPL` -> AMASS-style npz if needed -> GMR retarget cache -> trajectory handler.

Training:

1. `TrajInitialStateHandler` resets the simulator from a trajectory frame.
2. `GoalTrajMimic` emits current relative body data, future reference deltas, motion phase, and optionally current root error.
3. Policy outputs muscle/control actions.
4. `MimicReward` computes root-heavy imitation reward, relative body reward, and right-hand absolute reward.
5. Terminal handler ends episodes that drift too far in root or relative pose.
6. PPO fine-tunes from the selected checkpoint.

Evaluation:

1. Roll out deterministic policy on the same eight motions.
2. Save video and per-motion metrics.
3. Compare root displacement ratio and right-hand absolute error against the pre-post-train checkpoint.

## Diagnostics

Before training, run a reference-cache diagnostic. It should report per motion:

- `reference_root_xy_total_displacement`
- `reference_root_xy_peak_speed`
- `reference_root_yaw_change`
- `right_hand_world_path_length`

If a motion has `reference_root_xy_total_displacement < 0.30m`, it should be flagged as unsuitable for root-first policy post-train because the reference itself does not contain enough forward movement.

After rollout, report:

- `rollout_root_xy_total_displacement`
- `root_displacement_ratio`
- `root_xy_rmse`
- `root_xy_final_error`
- `root_speed_rmse`
- `root_yaw_error`
- `right_hand_abs_error`
- existing `err_root_xyz`, `err_site_abs`, `err_rpos`

The post-train is considered promising if root displacement ratio increases materially without a large right-hand error regression.

## Testing

Unit tests:

- `GoalTrajMimic` keeps the old observation dimension when `include_current_root_error=false`.
- `GoalTrajMimic` adds the expected root error dimension when enabled.
- Root error sign is correct in a simple constructed state.
- `MimicReward` absolute site reward is high when current and reference sites match and lower when the current site is offset.
- Missing configured absolute reward sites raise a clear error at initialization.
- Root-first config uses `MeanRelativeSiteDeviationWithRootTerminalStateHandler` for validation.

Integration checks:

- Reference diagnostic runs on the eight checkpoint-matched ForehandNetLift motions.
- Baseline checkpoint evaluation logs root metrics before post-train.
- A short post-train run starts from the selected checkpoint and writes a new checkpoint directory.
- Rollout evaluation after post-train logs the root metrics and renders a video.

Regression checks:

- Existing fullbody configs behave identically with default-off flags.
- Existing tests for n-step lookahead and mimic reward still pass.

## Implementation Order

1. Add checkpoint/data discovery and root diagnostics.
2. Evaluate the selected checkpoint on the eight ForehandNetLift motions and save baseline metrics.
3. Add optional absolute site reward.
4. Add optional current root error to goal observation.
5. Create the root-first post-train config.
6. Run the reward-only fallback first if observation shape restore fails.
7. Run a short post-train and compare metrics.
8. Only after improvement is confirmed, consider extending to `video09` and `video10`.

## Risks And Mitigations

Checkpoint restore may fail after observation dimension changes. The mitigation is the reward-only fallback, then partial loading if needed.

The reference cache may already have weak root displacement. The mitigation is the pre-train root diagnostic gate.

Root-heavy reward may hurt local pose quality. The mitigation is tracking `right_hand_abs_error`, `err_rpos`, and rollout video, then annealing root weight down after root tracking improves.

Right-hand absolute reward may overconstrain noisy wrist targets. The mitigation is to keep its first weight modest and disable it per motion if it destabilizes training.

Adding `video09` and `video10` immediately may confound resume behavior. The mitigation is to keep stage one on the original eight motions and run ten-motion expansion only after the root-first objective works.

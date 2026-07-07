# ForehandClear Grip-Hold Residual Design

## Goal

Build a no-shuttle ForehandClear post-train experiment that reuses the existing no-racket ForehandClear policy as the body swing prior, initializes the Overall badminton scene from the accepted right-hand grip seed, and trains only a small residual controller to keep the racket held correctly throughout the swing.

This stage deliberately does not train shuttle impact or flight. Success means the human keeps the ForehandClear motion stable while the right hand maintains a forehand grip on the racket without dropping, sliding, or penetrating the handle.

## Current Context

The standalone right-hand grip PPO experiment is still running and must not be stopped by this work. Its output directory remains `outputs/right_hand_racket_grip/policy`.

The existing static-hit spec points at:

```text
checkpoints/ForehandClear/forehand_clear_best/checkpoint_7812
```

That path is not present in the local workspace. The best local ForehandClear candidate is:

```text
checkpoints/de63059b16c0/checkpoint_7812
```

Its checkpoint metadata tags include `forehand_clear`, `10trajectories`, `smooth_filtered`, and `a100`, so this design uses it as the base body policy unless the user later supplies a more specific checkpoint.

The accepted grip seed source is:

```text
outputs/right_hand_racket_grip/reference/right_hand_racket_grip_seed.json
```

The Overall scene already consumes this seed during scene generation and faces the model toward the net.

## Non-Goals

- Do not stop or overwrite the current pure grip PPO run.
- Do not train hitting, shuttle release, net clearance, or landing depth.
- Do not require the model to learn a reaching/grasping motion from an empty hand.
- Do not fine-tune the whole ForehandClear body policy in the first experiment.
- Do not add a permanent hand-racket weld as the final solution.

## Recommended Approach

Use a layered policy:

1. Frozen ForehandClear base policy controls the whole-body swing.
2. A trainable residual controller owns right-hand fingers and, in later curriculum stages, small wrist/forearm corrections.
3. The environment starts from the accepted grip seed with the racket already in the hand.
4. The reward preserves ForehandClear imitation and adds grip-hold stability terms.
5. Validation videos are recorded early and frequently so failures are visually obvious.

This is safer than directly fine-tuning the full body policy because the existing body policy already solves the main standing and swing behavior. It is also more realistic than static right-hand grip PPO because the residual controller sees the swing-induced inertial loads that cause the racket to slip during ForehandClear.

## Curriculum

### Stage 1: Short Assisted Grip Hold

- Horizon: 0.3-0.5 seconds from the beginning or selected stable windows of the ForehandClear swing.
- Trainable actuators: right-hand fingers only.
- Base policy: frozen.
- Assistance: optional weak soft tether between palm/grip frames to prevent immediate state-distribution collapse.
- Objective: keep contact and hand shape while avoiding large residual actions.

### Stage 2: Swing-Phase Grip Hold

- Horizon: 1.0-1.5 seconds over backswing and acceleration windows.
- Trainable actuators: right-hand fingers plus small wrist/forearm residuals.
- Assistance: reduced tether strength.
- Objective: maintain grip during racket acceleration.

### Stage 3: Full No-Shuttle ForehandClear

- Horizon: full ForehandClear clip.
- Trainable actuators: same residual action space as Stage 2.
- Assistance: disabled or nearly zero.
- Objective: preserve the full body motion and hold the racket through follow-through.

## Reward Terms

The total reward should be a weighted sum of:

- `r_mimic`: preserve ForehandClear body imitation from the base environment reward.
- `r_root_stability`: penalize root drift, large root orientation deviation, and falling.
- `r_grip_site`: keep palm, thumb, index, middle, ring, and pinky grip sites near racket-local target points.
- `r_contact`: reward at least four right-hand handle contacts.
- `r_no_slip`: penalize change in grip-site to palm-site relative transform.
- `r_no_penetration`: penalize handle penetration into hand geoms.
- `r_racket_hand_pose`: keep racket pose stable relative to palm or wrist.
- `r_residual_effort`: penalize residual action magnitude and fast residual changes.

The first runnable version should keep rewards diagnostic and log each component separately to W&B.

## Validation

Validation must produce:

- JSON metrics for every validation rollout.
- MP4 videos under a new output directory such as:

```text
outputs/posttrain/ForehandClearGripHold/v1/validation_videos/
```

- W&B scalar logs for reward components and key grip metrics.
- W&B video logs when W&B is enabled.

Default validation interval should be much shorter than the pure grip PPO run:

```text
validation_video_interval_steps: 10000
```

Acceptance criteria for the first smoke experiment:

- finite rollout through at least one short validation horizon;
- no fall in the validation horizon;
- `max_handle_penetration_m <= 0.003`;
- `contact_count` reaches at least 2 in Stage 1 smoke, then 4 in later stages;
- `grip_slip_m` trends downward rather than upward;
- validation video shows the racket staying near the hand.

## Experiment Spec

Create a dedicated experiment spec, separate from static-hit:

```text
experiments/posttrain/forehand_clear_grip_hold_v1.yaml
```

Key fields:

```yaml
experiment_id: v1
action: ForehandClearGripHold
runner_type: forehand_clear_grip_hold
resume_from: /data3/yangfeiyang/WorkSpace/musclemimic/checkpoints/de63059b16c0/checkpoint_7812
reference:
  train:
    - forehand_clear/stage5_10demo/video1_lower_body_full_poses
  validation:
    - forehand_clear/stage5_10demo/video2_lower_body_full_poses
scene:
  xml: environment/overall_environment/assets/overall_badminton_scene.xml
grip_seed:
  path: outputs/right_hand_racket_grip/reference/right_hand_racket_grip_seed.json
shuttle:
  enabled: false
```

The existing `static_hit_staging` runner remains unchanged. This new runner type must not be routed into the static-hit fail-fast path.

## Implementation Shape

The implementation should be staged:

1. Spec and config generation only.
2. Environment adapter smoke test that resets Overall with the grip seed and no shuttle objective.
3. Diagnostic rollout that replays the frozen ForehandClear body policy and records grip metrics without training.
4. Residual action wrapper that applies right-hand residual controls on top of the base policy.
5. PPO training smoke with very small timesteps.
6. Full Stage 1 experiment with W&B and validation video.

The first useful result is not a full solved policy. It is a short video and metrics report proving the frozen ForehandClear policy can be replayed in the racket scene and that the residual action interface changes only the intended actuators.

## Risks

- The local ForehandClear checkpoint path may be correct but not the exact user-intended checkpoint. The plan should make this path configurable.
- Enabling fingers and adding the racket may change the action/observation interface enough that direct checkpoint restore cannot be used without an adapter.
- Real contact may still fail before the residual policy learns. The optional soft tether exists only as a curriculum aid and must be logged.
- Existing fullbody runners may not instantiate Overall/racket scenes directly; a dedicated runner or adapter is expected.

## Open Decisions Resolved for v1

- Base checkpoint: `checkpoints/de63059b16c0/checkpoint_7812`.
- No shuttle in v1.
- Start from accepted grip seed, not empty-hand grasping.
- Train residual first, not full-body finetune.
- Preserve current pure grip PPO process and output directory.

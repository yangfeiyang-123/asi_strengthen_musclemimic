# General Badminton Action Training Design

## Context

The current project has badminton motion data and manifests for four concrete groups:

- `ForehandClear`
- `Backhand`
- `ForehandNetLift`
- `Smash`

The broader goal is more general than these labels. The training strategy should support larger badminton motions such as clear, drop, smash, net lift, lunge, recovery, and footwork, while avoiding a dependency on fine hand, wrist, finger, or racket-face details that the current SMPL pipeline cannot reliably capture.

`doc/PostTrain_Advice.md` supports this direction: post-training should not simply minimize effort. It should keep root tracking, right-hand or racket endpoint tracking, foot contact, joint limits, and motion smoothness from degrading, then use muscle effort and activation smoothness as secondary realism constraints.

Recent root-tracking diagnostics also show that action names alone are not enough. Motions in the same category can have different root displacement, path length, peak speed, and yaw change. Training-stage assignment should therefore be metric-driven first, action-name-driven second.

## Goal

Define a general action-selection policy for badminton imitation training:

1. Which action types should be used in the initial/base training set.
2. Which action types should be reserved for post-training fine-tuning.
3. Which action types should be excluded or repaired before training.
4. Which metrics should decide the split when action labels are ambiguous.

The design should be practical for the existing MuscleMimic codebase and should not require reliable fine-grained hand or racket-face capture.

## Non-Goals

- Do not make tiny wrist, finger, or racket-face details a first-stage training target.
- Do not assume every action label maps to one fixed training stage.
- Do not replace PPO or the current GMR/SMPL retargeting pipeline.
- Do not require court calibration as a prerequisite for the first version.
- Do not mix every badminton action into a single from-scratch run without staging.

## Recommended Approach

Use a two-stage, metric-gated curriculum:

1. **Base training:** learn general musculoskeletal control, stable full-body imitation, basic root movement, torso rotation, arm swing, and foot contact patterns from clean and moderately difficult motions.
2. **Post-training:** start from a trained checkpoint and fine-tune movement-heavy, root-sensitive, contact-sensitive, or explosive badminton actions with stronger task and physical constraints.

This is preferable to a pure action-label split because "drop shot", "smash", or "backhand" can each be easy or hard depending on whether the motion includes large root travel, lunge, jump, rapid braking, or strong yaw rotation.

## Action Categories

### Base Training Candidates

Base training should use motions that are visible in SMPL, physically stable, and not dominated by fine hand details.

Recommended action types:

- Forehand clear / high clear.
- Backhand samples with moderate root displacement and stable foot contact.
- Standing or small-step drop shots.
- Standing smash without large jump or hard landing.
- Basic side-step, recovery, split-step, and small retreat motions when the retargeting quality is stable.
- ForehandClear best or smooth-filtered data if root diagnostics and videos look consistent.

These motions teach the policy the reusable parts of badminton movement: posture, shoulder and torso coordination, basic root control, leg support, and full-body timing.

### Post-Train Fine-Tuning Candidates

Post-training should handle motions whose main difficulty is root movement, lower-body contact, acceleration, braking, or large rotation.

Recommended action types:

- ForehandNetLift / front-court net lift.
- Net approach with lunge.
- Large forward step, cross step, chasse, and recovery footwork.
- Rear-court retreat plus drop or clear.
- Jump smash or smash with hard takeoff and landing.
- Large-yaw smash, large-yaw backhand, and fast direction-change actions.
- Any action where the right hand reaches the target only if the root moves correctly.

These actions should fine-tune from a competent base or action-family checkpoint. They should use root-heavy reward, stricter root-aware termination, absolute right-hand or racket endpoint reward when reliable, and contact/effort realism penalties after tracking is stable.

### Repair Or Exclude Before Training

Some motions should not be used directly, even if the action label is useful.

Repair or exclude when:

- The video clearly has court movement, but reference root displacement is too small.
- Root path is noisy, discontinuous, or inconsistent with foot motion.
- Feet slide heavily during apparent stance.
- Feet penetrate the ground or hover during stance.
- The right-hand path is not consistent with the torso/root movement.
- The clip contains only the final hit frame and misses preparation or recovery.
- The action depends mainly on finger, wrist, racket-face angle, or shuttle contact that SMPL does not observe.

These clips may become useful after retargeting repair, smoothing, better root reconstruction, or adding external task labels.

### Not Primary Targets For Now

The current method should not emphasize tiny net-play details as a main claim.

Examples:

- Fine net tumbling shots.
- Subtle wrist-only push or hold shots.
- Deceptive racket-face changes.
- Finger grip changes.
- Motions where the body stays nearly still and the discriminative information is mostly in the racket face.

They can remain auxiliary data only if they do not dominate the reward or cause the policy to overfit noisy hand targets.

## Metric Gates

Before assigning a motion to base training or post-training, run root/reference diagnostics from the retarget cache.

Recommended first metrics:

- `reference_root_xy_total_displacement`
- `reference_root_xy_path_length`
- `reference_root_xy_peak_speed`
- `reference_root_yaw_change`
- `right_hand_world_path_length`
- Later: foot slip, penetration, joint-limit violation, activation energy, and activation-rate diagnostics.

Initial thresholds:

- `root_xy_displacement < 0.25m`: treat as stationary or small-step. Use for base only if the video also looks stationary. If the real action should move, repair root first.
- `0.25m <= root_xy_displacement <= 0.60m`: medium movement. Use for base if contact is stable; use for light post-training if root tracking is task-critical.
- `root_xy_displacement > 0.60m`: movement-heavy. Prefer post-training.
- `root_xy_peak_speed > 1.2m/s`: likely acceleration/braking sensitive. Prefer post-training or a staged curriculum.
- `abs(root_yaw_change) > 0.8rad`: large rotation. Prefer post-training unless the policy already handles rotations well.
- jump, lunge, hard landing, rapid stop, or large direction change: prefer post-training even if displacement is moderate.

These thresholds are starting points, not universal truths. They should be revised after several training/evaluation runs.

## Training Stage Design

### Stage 1: General Base Policy

Use clean, visible, medium-scale actions:

- ForehandClear.
- Stable Backhand samples.
- Standing or small-step Smash samples.
- Standing or small-step Drop samples when available.
- Basic footwork clips if the lower-body/root tracking is reliable.

Reward should emphasize general imitation:

- Keep relative body/site tracking strong.
- Keep moderate root position and velocity terms.
- Avoid high absolute hand/racket reward if the endpoint is noisy.
- Keep effort penalties weak or off at first.
- Use validation metrics for root, site, and right-hand errors, but do not terminate too aggressively before the base controller is stable.

### Stage 2: Action-Family Post-Training

Create separate fine-tuning configs for difficult action families:

- Net/front-court family: ForehandNetLift, net approach, lunge, recovery.
- Rear-court family: retreat, drop, clear, recovery.
- Explosive family: smash, jump smash, hard landing.
- Rotation family: large-yaw backhand or turning smash.

Reward should be task and physics aware:

- Increase `root_pos_w_sum` and `root_vel_w_sum`.
- Use `MeanRelativeSiteDeviationWithRootTerminalStateHandler`.
- Add absolute `right_hand_mimic` or racket endpoint reward only when endpoint data is reliable.
- Add effort and smoothness terms after tracking is stable: `activation_energy_coeff`, `action_rate_coeff`, later `activation_rate_coeff`.
- Add foot slip/penetration and joint-limit penalties when implemented.

### Stage 3: Generalization Evaluation

Evaluate not just on the trained action family, but also on held-out motions from nearby families.

Examples:

- A base policy trained on clear/backhand should be evaluated on standing drop and standing smash.
- A net-lift post-train policy should be evaluated on unseen net-lift clips and approach/recovery clips.
- A smash post-train policy should be evaluated separately on standing smash and jump smash.

The policy is more general only if it improves task/root/contact realism without collapsing nearby actions.

## Data Flow

1. Motion files are listed in action manifests.
2. GMR retargeting builds or reuses cache files under `caches/AMASS/MyoFullBody/gmr`.
3. A diagnostic step computes root and hand-path metrics from each cache file.
4. A stage-assignment table labels each motion as `base`, `posttrain`, `repair`, or `exclude`.
5. Config generation builds per-stage or per-family training configs from those labels.
6. PPO trains or fine-tunes.
7. Evaluation reports tracking, root, right-hand, muscle, contact, and smoothness metrics.

## Proposed Documentation Changes

Add a section to `doc/PostTrain_Advice.md` or a companion document that states:

- Effort minimization is a post-training regularizer, not the main objective.
- Action selection should be based on observability and root/contact complexity.
- Fine hand-only badminton skills are outside the current primary scope.
- Large visible actions and footwork are the main generalization target.
- A motion can move between stages after diagnostics or data repair.

## Proposed Code/Config Changes For A Later Implementation Plan

This design does not implement code yet, but the likely implementation plan should include:

- A small action-stage manifest, for example `BadmintonMimic/manifests/action_stage_map.yaml`.
- A diagnostic report command that reads existing manifests and emits per-motion stage recommendations.
- Optional config templates for:
  - `base_general_badminton`
  - `posttrain_net_frontcourt`
  - `posttrain_rearcourt`
  - `posttrain_smash`
  - `posttrain_rotation`
- Logging fields for effort and naturalness metrics from `PostTrain_Advice.md`.

## Success Criteria

The design is successful when:

- Each motion has a clear reason for being in base training, post-training, repair, or exclusion.
- Large visible actions such as drop, clear, smash, net lift, lunge, and footwork are supported.
- Tiny hand-only skills are not overclaimed.
- Root-heavy post-training is used only when the reference contains meaningful root motion.
- Adding realism penalties does not reduce root or endpoint tracking.
- Held-out nearby actions remain stable after post-training.

## Risks And Mitigations

**Risk:** A label like "drop shot" mixes easy standing samples and hard rear-court retreat samples.  
**Mitigation:** Assign by metrics and video diagnostics, not by label alone.

**Risk:** Effort minimization makes athletic actions too small or too slow.  
**Mitigation:** Enable effort penalties only after root and endpoint tracking are stable, and track Pareto tradeoffs.

**Risk:** Fine hand action noise damages the general controller.  
**Mitigation:** Keep fine hand-only actions out of the main training set until a better racket/hand representation is available.

**Risk:** Post-training overfits one action family and hurts generality.  
**Mitigation:** Evaluate on nearby held-out actions and keep base checkpoints separate from family-specific fine-tunes.

**Risk:** Reference root is wrong, so root-heavy training reinforces bad data.  
**Mitigation:** Gate root-heavy post-training on reference diagnostics and repair weak root clips first.


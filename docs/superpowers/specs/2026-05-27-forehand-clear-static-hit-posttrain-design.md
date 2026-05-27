# Forehand Clear Static-Hit PostTrain Design

Date: 2026-05-27

## Goal

Build a staged post-train pipeline for a realistic badminton forehand clear hit in the composed Overall environment. The first version uses a static shuttle at the ideal forehand-clear contact point, real hand-racket contact, and the existing no-racket/no-shuttle ForehandClear policy as the body swing prior.

The target is not a full rally. The target is a debuggable training path that can first hold the racket through a swing, then hit a frozen shuttle, then gradually optimize the shuttle flight into a high clear toward the opponent back court.

## Current Repository Context

The repository already has the required physical pieces, but not the final task:

- `environment/overall_environment` builds a MuJoCo scene with court, net, MyoFullBody, racket, and shuttle.
- `environment/racket/src/racket_stringbed.py` provides a string-bed force proxy and high-speed event rebound helper.
- `environment/shuttlecock/src/shuttlecock_aero.py` provides aerodynamic force and torque for released shuttle flight.
- `src/grip` provides a right-hand grip scene, grip reference, grip environment, validation scripts, and a standalone PPO trainer.
- `fullbody/config_specific_task/conf_fullbody_badminton_gmr.yaml` and related configs provide ForehandClear imitation/post-train precedent.

The current grip status is not a trained policy. It is closer to an IK/reference baseline:

- `configs/right_hand_racket_grip_reference.json` has `mean_site_error_m = 0.021730698315818076`.
- It meets the loose IK mean threshold but fails the stricter training mean threshold.
- `docs/right_hand_racket_grip.md` records partial acceptance only: contact count is too low, zero-action racket drift is too high, and perturbation recovery fails.
- No `outputs/right_hand_racket_grip/policy/policy_latest.pt` or equivalent trained grip checkpoint is currently present.

Therefore this design includes a required grip stabilizer stage before full Overall static-hit post-training.

## Design Position

Use a staged, layered system:

1. Validate the static-hit physics chain without learning.
2. Train a right-hand grip stabilizer policy from the current grip reference.
3. Train the grip stabilizer under ForehandClear swing disturbances.
4. Compose the existing ForehandClear body policy with the grip policy inside the Overall static-shuttle task.

This is preferred over direct end-to-end post-training because the existing ForehandClear checkpoint was trained without racket, shuttle, or enabled fingers. Directly enabling fingers changes observation/action dimensions and couples network migration, grip contact, swing timing, impact, and shuttle flight into one failure surface.

## Architecture

### Grip Stabilizer Layer

The grip stabilizer controls the 31 right-hand actuator commands exposed by the current grip environment. It is trained from the current grip reference, then stress-tested with swing-like disturbances.

Responsibilities:

- keep at least four meaningful hand-handle contacts;
- reduce hand-racket slip;
- keep racket grip frame translation and orientation stable;
- avoid non-hand handle contacts and excessive penetration;
- recover from perturbations and impact-like impulses.

The stabilizer should become a checkpointed policy before being used by the static-hit task.

### Body/Swing Layer

The body layer reuses the existing ForehandClear PPO checkpoint as a swing prior.

Responsibilities:

- root, legs, trunk, shoulder, elbow, and wrist motion;
- ForehandClear phase structure: preparation, backswing, acceleration, impact, follow-through;
- preserving the learned no-racket motion frame during post-train.

The first version should freeze this policy or fine-tune it with a lower learning rate than the grip policy. It must not control the right-hand finger actuators reserved for the grip stabilizer.

### Overall Static-Shuttle Task Layer

The Overall task layer wraps the composed scene and owns:

- initial placement of the human, racket, and shuttle;
- static shuttle freeze before impact;
- impact detection and release;
- string-bed force, event rebound, gravity, and shuttle aerodynamics after release;
- task rewards for impact, net crossing, flight arc, and landing.

It should expose explicit diagnostics for grip, impact, and flight rather than only a scalar reward.

### Composition Layer

A small actuator router combines policy outputs by actuator name:

```text
body_action = body_policy(obs_body)
grip_action = grip_policy(obs_grip)
merged_action = actuator_router(body_action, grip_action)
```

The router must prevent actuator conflicts. The grip policy owns right-hand finger actuators. The body policy owns the non-finger body and arm actuators. The router should record which policy wrote each actuator for debugging.

Supported modes:

- body frozen, train grip;
- grip frozen, train body timing;
- both frozen for scripted evaluation;
- both trainable with separate learning-rate groups in a later version.

## Overall Environment State Machine

The static-hit task should use an explicit state machine.

### RESET

On reset:

- choose a ForehandClear reference episode or rollout;
- place the root on one half-court center, facing the net;
- initialize right hand and racket from the grip reference or grip policy reset state;
- compute a root-local ideal impact point;
- place the shuttle at the impact point with zero velocity;
- enter `PRE_IMPACT_FREEZE`.

### PRE_IMPACT_FREEZE

Before valid impact:

- force shuttle freejoint `qpos` to the target impact pose each step;
- force shuttle `qvel` to zero;
- do not apply shuttle aerodynamics;
- keep human, racket, and contacts physically simulated;
- do not disable gravity globally, because body and racket dynamics still need normal physics.

Impact release should require:

- cork/contact site projects inside the racket string-bed ellipse;
- distance to the string-bed plane is within the proxy contact region;
- relative normal velocity is closing;
- motion phase is inside or near the expected impact window.

When these conditions hold, enter `IMPACT_RELEASED`.

### IMPACT_RELEASED

After release:

- stop teleporting the shuttle;
- apply string-bed force from `environment/racket/src/racket_stringbed.py`;
- use event rebound only when the configured high-speed active-contact condition holds;
- apply shuttle aerodynamics from `environment/shuttlecock/src/shuttlecock_aero.py`;
- record impact diagnostics: step, phase, impact location, racket normal, racket speed, shuttle outgoing velocity, peak force, and event rebound usage.

### FLIGHT_EVALUATION

Track the released shuttle until landing, timeout, out-of-bounds, severe human failure, or racket drop.

Diagnostics:

- crossed net;
- net crossing height;
- maximum flight height;
- landing position;
- landing region;
- time to land;
- flight distance.

### TERMINATED

The terminal info must include:

- state sequence and terminal reason;
- grip metrics;
- impact metrics;
- flight metrics;
- reward breakdown.

## Impact Target Extraction

The shuttle impact target should not be a fixed world coordinate. It should be extracted from ForehandClear reference motion and then regularized using body scale.

### Reference Extraction

Extract candidate impact phase from ForehandClear reference, retarget cache, or rollout:

- if a racket site exists, use racket head position and velocity;
- otherwise estimate a virtual racket head from the right-hand site plus a racket-length offset;
- prefer frames where right-hand or virtual racket-head speed is near a local peak;
- require the candidate point to be in front of the body and on the racket-hand side;
- prefer high but comfortable reach points after backswing and before follow-through.

Output:

- `impact_phase`;
- `impact_frame`;
- `target_impact_pos_root_local`;
- `target_racket_normal`;
- `target_racket_head_velocity_dir`.

### Body-Scale Regularization

Project the extracted target into a plausible forehand clear contact region:

```text
height ~= shoulder_height + arm_reach_up * alpha + racket_effective_length * beta
```

Use conservative `alpha` and `beta` values instead of maximum extension. Horizontally, require:

- positive forward offset from the root;
- positive racket-side offset for right-hand forehand;
- not directly above the head;
- not across the body centerline.

Add narrow randomization first:

- a few centimeters in forward, lateral, and vertical offsets;
- small phase jitter;
- narrow racket-face and shuttle pose variation.

Widen randomization only after nominal static-hit training passes.

## Training Curriculum

### Stage 0: Physics Chain Validation

No RL update is required in this stage. Use a scripted or reference swing to validate:

- shuttle freeze and release;
- impact detection;
- string-bed force application;
- event rebound is not double-applied;
- post-release gravity and aero;
- net crossing and landing-region logic;
- diagnostics remain finite.

Acceptance:

- no NaN/Inf values;
- impact can be detected repeatably;
- the shuttle can be struck into flight;
- state transitions and terminal reasons are correct.

### Stage 1: Static Grip Stabilizer

Train only the right-hand grip policy in the grip environment or an Overall grip subset.

Reward terms:

- `r_grip_site`: hand pad sites remain near handle targets;
- `r_contact`: at least four meaningful handle contacts;
- `r_no_slip`: low palm/grip-frame slip;
- `r_racket_drift`: low racket grip-frame translation drift;
- `r_orient`: low handle-axis and face-normal orientation drift;
- `r_no_penetration`: low illegal penetration;
- `r_effort`: low actuator effort;
- `r_action_rate`: smooth controls.

Acceptance:

- finite rollouts;
- contact count reaches the configured threshold;
- racket drift and orientation drift are inside staged thresholds;
- no persistent non-hand handle contact.

### Stage 2: Grip Stabilizer With Swing Disturbances

Still train only the grip policy. Add ForehandClear-like disturbances:

- wrist and forearm acceleration disturbance;
- racket inertia disturbance;
- phase-conditioned disturbance from backswing, acceleration, impact, and follow-through;
- later, an impact-like impulse.

The reward remains grip-centered. No shuttle-flight objective is added in this stage.

Acceptance:

- grip remains stable through the ForehandClear disturbance profile;
- no racket drop;
- slip and drift stay within staged thresholds;
- perturbation recovery passes before entering static-hit post-train.

### Stage 3a: Static-Hit, Hit And Over-Net

Compose body policy and grip policy in Overall. Early post-train rewards:

- `r_mimic`: preserve ForehandClear body motion;
- `r_grip`: preserve trained grip;
- `r_impact_timing`: hit near the reference impact window;
- `r_impact_location`: cork lands near the string-bed sweet spot;
- `r_racket_speed`: sufficient racket-head speed at impact;
- `r_racket_face`: face points the shuttle up and toward the far court;
- `r_hit`: valid release and outgoing shuttle speed;
- `r_over_net`: shuttle crosses the net with safe clearance.

Acceptance:

- repeated valid impact;
- over-net rate passes a configured threshold;
- grip remains stable through impact.

### Stage 3b: Static-Hit, High Clear Depth

Add high-clear quality rewards:

- `r_clear_depth`: landing near opponent back court;
- `r_clear_arc`: reasonable high clear arc;
- `r_in_bounds`: legal landing;
- `r_recovery`: body and racket remain stable after follow-through.

Acceptance:

- valid over-net high-clear shots;
- landing-region distribution concentrated in opponent rear court;
- no systematic racket drop or body instability.

## Metrics And Failure Diagnosis

### Grip Metrics

Record:

- `contact_count`;
- `illegal_handle_contact_count`;
- `mean_grip_site_error_m`;
- `racket_translation_drift_m`;
- `racket_orientation_drift_deg`;
- `grip_slip_m`;
- `max_handle_penetration_m`;
- `drop_racket`.

Initial staged thresholds may be looser than final grip acceptance. Final targets should converge toward:

- `contact_count >= 4`;
- `illegal_handle_contact_count == 0`;
- `mean_grip_site_error_m < 0.02`;
- `racket_translation_drift_m < 0.01`;
- `racket_orientation_drift_deg < 8`;
- `max_handle_penetration_m < 0.003`.

### Impact Metrics

Record:

- `impact_detected`;
- `impact_phase_error`;
- `impact_point_rho2`;
- `impact_signed_z`;
- `relative_normal_velocity`;
- `racket_head_speed_m_s`;
- `racket_normal_world`;
- `shuttle_outgoing_velocity`;
- `event_rebound_used`;
- `stringbed_force_peak_n`.

Failure labels:

- `missed_shuttle`;
- `wrong_phase`;
- `edge_hit`;
- `weak_hit`;
- `bad_face_angle`;
- `double_rebound`.

### Flight Metrics

Record:

- `crossed_net`;
- `net_crossing_height_m`;
- `max_flight_height_m`;
- `landing_xy`;
- `landing_region`;
- `flight_distance_m`;
- `time_to_land_s`.

Landing regions should distinguish at least:

- own side;
- net/front court;
- opponent mid court;
- opponent back court;
- out of bounds.

## Code Organization

### New Static-Hit Environment

Add:

```text
environment/overall_environment/src/static_forehand_clear_env.py
```

Responsibilities:

- wrap `OverallBadmintonEnvironment`;
- implement the state machine;
- freeze and release the shuttle;
- call string-bed, rebound, aero, and flight evaluators;
- expose structured diagnostics and reward terms.

### New Impact Target Utility

Add:

```text
environment/overall_environment/src/impact_target.py
```

Responsibilities:

- extract impact phase and target from reference or rollout;
- estimate virtual racket head when no racket reference exists;
- regularize the target by body scale and forehand-side constraints;
- apply staged randomization.

### New Layered Control Utility

Add:

```text
environment/overall_environment/src/layered_control.py
```

Responsibilities:

- route actions by actuator name;
- prevent body/grip actuator conflicts;
- support freeze/fine-tune modes;
- log action source mapping.

### Grip Stabilizer Extensions

Extend:

```text
src/grip/right_hand_racket_grip_env.py
src/grip/train_right_hand_racket_grip_policy.py
configs/right_hand_racket_grip_training.yaml
```

Needed capabilities:

- swing-disturbance curriculum;
- impact-like impulse disturbance;
- trained policy checkpoint saving;
- trained-policy validation rather than only zero-action baseline validation.

### Experiment Spec

Add a future experiment spec:

```text
BadmintonMimic/experiments/posttrain/forehand_clear_static_hit_v1.yaml
```

Responsibilities:

- point to the ForehandClear body checkpoint;
- point to the grip policy checkpoint;
- define curriculum stages;
- define train/eval/render command generation parameters.

## Out Of Scope

The first version does not include:

- moving incoming shuttle;
- opponent or rally simulation;
- serving or receive-shot logic;
- a single monolithic full-body-plus-finger policy;
- permanent hand-racket weld as the main path;
- broad domain randomization before nominal behavior works;
- high success rate for back-court landing before hit and over-net training passes.

## Acceptance Summary

The first implementation is accepted when:

1. Stage 0 physics validation produces finite, repeatable hit/release/flight diagnostics.
2. A grip stabilizer checkpoint exists and passes static plus swing-disturbance grip validation.
3. The layered actuator router composes body and grip actions without actuator conflicts.
4. The Overall static-hit task can detect valid impacts and release the shuttle exactly once.
5. Stage 3a reaches reliable hit-and-over-net behavior while preserving grip stability.
6. Stage 3b adds measurable progress toward opponent back-court high-clear landing.

## Self-Review

No placeholders remain. The scope is intentionally limited to a staged static-shuttle forehand clear task, not a full rally. The design resolves the current grip-policy gap by requiring a grip stabilizer stage before Overall post-train. The architecture keeps body motion, grip control, impact physics, and flight evaluation separated so failures can be diagnosed independently.

# Right-Hand Racket Grip Design

Date: 2026-05-26

## Goal

Train an independent right-hand musculoskeletal controller to hold a badminton racket in a neutral forehand grip. This first version is a grip pretraining task only. It does not train full-body swing imitation, shuttlecock impact, landing accuracy, or forehand/backhand posttraining integration.

The output should provide a validated right-hand grip scene, a static reference grip state, and a grip-hold training/evaluation environment that can later be reused by full-body badminton tasks.

## Repository Context

The `environment/HoldRacket/right_hand_racket_grip_codex_package` documents a staged pipeline:

1. Audit the hand and racket MuJoCo models.
2. Add grip annotation sites.
3. Solve a static reference grip posture.
4. Train a grip-hold environment around that posture.
5. Validate grip stability, contacts, drift, and perturbation recovery.

This design follows that sequence. It deliberately avoids pure free-contact RL from scratch.

The existing racket package already provides `environment/racket/assets/badminton_racket_rigid.xml` with:

- `racket` body
- `racket_free` freejoint
- `grip_pose_site`
- `butt_site`
- `stringbed_center_site`
- `head_tip_site`
- `handle_grip` collision geom

The racket still needs grip-specific annotation sites for handle axis and face normal.

The MyoFullBody environment currently defaults to `disable_fingers: true` in full-body training configs. Grip pretraining must use a model/spec path where right-hand finger joints and muscles remain enabled.

## Scope

In scope:

- Right hand, wrist, and optionally a small right-arm reach adjustment for grip setup.
- Rigid badminton racket model with freejoint.
- Neutral/forehand grip geometry.
- CPU MuJoCo reference solving, smoke testing, and validation.
- A standalone `reset()` / `step(action)` grip-hold environment that can be wrapped by a trainer later.

Out of scope for this first version:

- Full-body swing imitation.
- Shuttlecock impact/contact reward.
- Court or landing target reward.
- End-to-end MJX/Warp training with the full-body task.
- Backhand thumb grip.
- Permanent hand-racket weld in the final scene.

## Architecture

Create a new `src/grip/` package with focused modules:

- `hand_racket_model_map.py`
  - Loads a MuJoCo model.
  - Discovers and validates right-hand body, joint, actuator, site, racket body, racket freejoint, and handle geom names.
  - Provides a typed map used by all later scripts.

- `visualize_grip_sites.py`
  - Loads the grip scene.
  - Fails loudly if required sites are missing.
  - Prints palm, finger pad, and racket annotation site world positions.
  - Optionally launches the MuJoCo viewer when available.

- `grip_objectives.py`
  - Implements reusable objective/reward helpers for target site errors, racket-palm frame error, finger wrapping, joint regularization, joint limit cost, contact readiness, penetration checks, and orientation error.

- `solve_right_hand_racket_grip.py`
  - Builds a static forehand grip reference using `scipy.optimize` when available.
  - Optimizes right-hand qpos and, if needed, wrist/right-arm qpos while placing the racket grip near the palm.
  - Saves `configs/right_hand_racket_grip_reference.json`.

- `right_hand_racket_grip_env.py`
  - Provides a CPU MuJoCo grip-hold task with `reset()` and `step(action)`.
  - Applies controls through right-hand actuators, not by teleporting qpos during steps.
  - Supports curriculum stages with optional soft weld assistance.

- `train_right_hand_racket_grip.py`
  - Loads XML, target config, and reference state.
  - Uses an existing training backend only if a clean fit exists.
  - Otherwise provides a baseline rollout/controller entrypoint and clear trainer integration hooks.

- `evaluate_right_hand_racket_grip.py`
  - Runs deterministic evaluation episodes.
  - Reports reward terms, site errors, contacts, racket drift, orientation drift, and perturbation recovery.

- `validate_right_hand_racket_grip.py`
  - Runs the acceptance checks listed below and prints PASS/FAIL reasons.

Add tests in `tests/test_right_hand_racket_grip.py` for loadability, site presence, target config parsing, reference JSON dimension checks, and finite environment rewards.

## Model And Scene Design

The grip scene should combine:

- MyoFullBody source model from `musclemimic_models.get_xml_path("myofullbody")`.
- Rigid racket from `environment/racket/assets/badminton_racket_rigid.xml`.

Preferred generated scene path:

```text
assets/right_hand_racket_grip_scene.xml
```

If the implementation finds that the repository convention is better served by generating the scene under `environment/HoldRacket/outputs/` or a similar path, it may do so, but must document the final path in `docs/right_hand_racket_grip.md` and all CLI defaults.

The scene must keep handle collision active. The final scene must not permanently weld the racket to the hand. Soft weld assistance is allowed only as an optional curriculum mechanism.

Required racket annotation sites:

- `grip_pose_site`, near local `[0, 0.09, 0]`
- `butt_site`
- `stringbed_center_site`
- `head_tip_site`
- `handle_axis_start_site`, local `[0, 0.02, 0]`
- `handle_axis_end_site`, local `[0, 0.16, 0]`
- `racket_face_normal_site`, local `[0, 0.09, 0.05]`

Required hand grip sites:

- `rh_palm_grip_site`
- `rh_thumb_pad_site`
- `rh_index_pad_site`
- `rh_middle_pad_site`
- `rh_ring_pad_site`
- `rh_pinky_pad_site`

If the hand model already has suitable anatomical sites, the model map may alias them. Otherwise the implementation should add small, non-colliding sites to the best matching right-hand bodies. Expected body candidates include `lunate_r`, `distal_thumb_r`, `2distph_r`, `3distph_r`, `4distph_r`, and `5distph_r`, but the implementation must validate actual names from the loaded model instead of hard-coding unverified indices.

## Forehand Grip Geometry

Use the racket coordinate convention from the racket package and HoldRacket config:

- origin: butt cap center
- `+Y`: butt cap toward racket head
- `+Z`: stringbed normal
- `+X`: lateral across stringbed

Use `configs/right_hand_racket_grip_targets.json` with this target layout:

```text
palm_target    y=0.085, theta=180 deg, weight=1.5
thumb_target   y=0.122, theta= 45 deg, weight=2.0
index_target   y=0.125, theta=-45 deg, weight=2.0
middle_target  y=0.098, theta=-115 deg, weight=1.6
ring_target    y=0.075, theta=-135 deg, weight=1.4
pinky_target   y=0.055, theta=-150 deg, weight=1.2
```

The handle radius is `0.014 m`, with `0.0015 m` contact clearance. Target coordinates use:

```text
x = radius * cos(theta)
z = radius * sin(theta)
```

This creates a neutral forehand grip: the handle passes diagonally across the palm through the thumb-index web, thumb and index oppose each other near the upper handle, and the middle/ring/pinky fingers wrap around the lower handle.

The implementation may mirror the theta signs if visual validation shows that the hand/racket coordinate convention is reversed. Any mirroring must be documented in the target config and audit doc.

## Static Reference Solver

The solver should optimize a static state before any RL/control training.

Optimized variables:

- Right-hand finger qpos.
- Wrist qpos if needed.
- Limited right-arm qpos if the racket cannot be placed near the palm without unreachable geometry.
- Racket freejoint pose during setup.

The objective should include:

- Weighted palm/finger site distance to handle target points.
- Racket `grip_pose_site` proximity to `rh_palm_grip_site`.
- Handle axis alignment through the palm/thumb-index web.
- Thumb/index opposition and lower-finger wrap plausibility.
- Deviation from a neutral or current hand pose.
- Joint-limit safety.
- Contact readiness near the handle surface.
- Penalty for obvious non-handle penetration.

Primary output:

```text
configs/right_hand_racket_grip_reference.json
```

The reference JSON must contain:

- XML path used.
- Keyframe/reference name.
- Full `qpos` and `qvel`.
- Racket freejoint qpos.
- Right-hand joint names.
- Site error report in meters.
- Objective breakdown.
- Notes for limitations or mirrored target conventions.

If XML keyframe editing is risky, the JSON loader is sufficient for the first version.

## Training Environment

The environment is a standalone CPU MuJoCo task. It should be deterministic and easy to debug before any accelerated full-body integration.

Observation should include:

- Right-hand qpos/qvel, normalized.
- Right-hand actuator activations/controls when available.
- Racket grip site pose relative to the palm grip site.
- Handle axis and face normal relative to the palm frame.
- Palm/finger site errors relative to handle targets.
- Contact indicators by palm/thumb/index/middle/ring/pinky with handle geoms.
- Racket linear and angular velocity.
- Curriculum stage id.

Action should control right-hand muscle activations or actuator controls. During `step`, the environment must not directly set qpos except for reset or explicit curriculum helpers.

Reward terms:

- `r_site_match`: grip sites stay close to handle targets.
- `r_racket_pose`: racket grip site stays near the palm grip site.
- `r_racket_orient`: handle axis and face normal stay close to reference.
- `r_contact`: at least four meaningful hand-handle contacts.
- `r_no_slip`: low relative hand-racket slip at the grip frame.
- `r_reference_pose`: bounded deviation from IK reference qpos.
- `r_effort`: penalize excessive controls or muscle activation.
- `r_joint_limits`: penalize unsafe joint positions.
- `r_no_penetration`: penalize illegal persistent penetration.
- `r_perturb_stable`: bonus for recovery after perturbation.

The `info` dictionary must report the reward breakdown every step.

## Curriculum

Implement stages:

0. Reference-pose tracking with racket fixed or very strong optional soft weld.
1. Soft weld to palm/grip frame, gravity enabled, no perturbation.
2. Weaker soft weld, high handle friction, small perturbations.
3. Free racket with contact, perturb handle or racket head.

Stage 4 full-body swing integration is intentionally excluded from this design. The implementation may document the handoff points needed for later integration.

Perturbation limits:

- Force: up to `2 N`.
- Torque: up to `0.03 N*m`.
- Duration: `0.05 s` to `0.15 s`.
- Recovery window: `0.5 s`.

## Validation And Acceptance

Validation commands should eventually include:

```bash
python -m src.grip.hand_racket_model_map --xml assets/right_hand_racket_grip_scene.xml
python src/grip/visualize_grip_sites.py --xml assets/right_hand_racket_grip_scene.xml --no-viewer
python src/grip/solve_right_hand_racket_grip.py --xml assets/right_hand_racket_grip_scene.xml --targets configs/right_hand_racket_grip_targets.json --out configs/right_hand_racket_grip_reference.json
python src/grip/right_hand_racket_grip_env.py --xml assets/right_hand_racket_grip_scene.xml --smoke-test
python src/grip/validate_right_hand_racket_grip.py --xml assets/right_hand_racket_grip_scene.xml --reference configs/right_hand_racket_grip_reference.json
pytest -q tests/test_right_hand_racket_grip.py
```

The first version is accepted when:

- Mean grip-site error after reset is below `0.02 m`.
- Racket grip frame translation drift over two seconds with gravity is below `0.01 m`.
- Racket orientation drift over two seconds with gravity is below `8 deg`.
- At least four meaningful handle contacts are present in the stable pose.
- After a `2 N` force or `0.03 N*m` torque perturbation, recovery reaches below `0.02 m` site error and `12 deg` orientation error within `0.5 s`.
- No NaN/Inf values appear.
- No severe joint-limit violations persist.
- No obvious persistent non-handle penetration is required for the grip.

For IK-only intermediate results, mean grip-site error below `0.03 m` is acceptable, but the final trained grip should use the stricter `0.02 m` threshold.

## Documentation

Create or update `docs/right_hand_racket_grip.md` during implementation with:

- Model files used.
- Discovered right-hand and racket naming map.
- Grip coordinate convention.
- Target site definitions.
- IK objective and current reference quality.
- Environment observation/action/reward definitions.
- Curriculum stages.
- Validation metrics and latest results.
- Known limitations.

## Risks

- Finger joints and muscles are disabled in existing full-body configs. The grip environment must avoid inheriting that default.
- MJX/Warp large-scale training may be brittle with hand-racket contact. First validation should use CPU MuJoCo.
- Hand pad sites may require visual tuning because the base model has limited pre-existing fingertip sites.
- Handle target theta signs may need mirroring after rendering.
- Muscle-actuated fingers may need staged assistance before they can hold a free racket under gravity.

## Future Integration

After the independent grip task passes validation, later work can:

- Use the reference state as reset initialization for full-body badminton posttraining.
- Add racket pose and grip preservation rewards to forehand/backhand motion imitation.
- Transfer or distill the right-hand grip policy into a larger full-body policy.
- Add shuttlecock impact and landing rewards only after the grip remains stable during swing-like wrist/arm motion.

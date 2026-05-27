# Right-Hand Racket Grip Pipeline

This document describes the standalone CPU MuJoCo pipeline for a right-hand forehand badminton racket grip. The pipeline is a grip pretraining and validation path only; it is not the full-body badminton swing task.

## Files

Primary model and configuration files:

- `assets/right_hand_racket_grip_scene.xml`: generated MuJoCo scene combining `musclemimic_models.get_xml_path("myofullbody")` with the rigid badminton racket.
- `environment/racket/assets/badminton_racket_rigid.xml`: source racket asset used by the scene builder.
- `configs/right_hand_racket_grip_targets.json`: grip target geometry, site candidates, coordinate convention, and acceptance thresholds.
- `configs/right_hand_racket_grip_reference.json`: solved static reference qpos/qvel and site error report.
- `configs/right_hand_racket_grip_training.yaml`: standalone environment reward and rollout settings.

Main entry points:

- `src.grip.build_right_hand_racket_grip_scene`: builds the combined XML scene and grip annotation sites.
- `src.grip.solve_right_hand_racket_grip`: solves the static forehand grip reference.
- `src.grip.right_hand_racket_grip_env`: CPU MuJoCo reset/step environment.
- `src.grip.evaluate_right_hand_racket_grip`: deterministic zero-action evaluation.
- `src.grip.validate_right_hand_racket_grip`: acceptance validation with optional strict exit behavior.

## Coordinate Convention

Racket-local grip targets use the convention from `configs/right_hand_racket_grip_targets.json`:

- origin: butt cap center
- `+Y`: along the handle toward the racket head
- `+Z`: stringbed normal
- `+X`: lateral axis across the stringbed

Target points are cylindrical coordinates around the handle:

```text
x = handle_radius_m * cos(theta)
z = handle_radius_m * sin(theta)
y = configured distance from the butt cap
```

The current handle radius is `0.014 m`; the configured contact clearance is `0.0015 m`. The neutral forehand layout places the palm near `y=0.085`, thumb and index near the upper handle, and middle/ring/pinky wrapping lower on the handle.

## Model Map

The generated scene currently resolves these MuJoCo names:

- right-hand bodies: `palm -> lunate_r`, `thumb -> distal_thumb_r`, `index -> 2distph_r`, `middle -> 3distph_r`, `ring -> 4distph_r`, `pinky -> 5distph_r`, `wrist -> lunate_r`
- hand grip sites: `palm -> rh_palm_grip_site` on `thirdmc_r`, `thumb -> rh_thumb_pad_site`, `index -> rh_index_pad_site`, `middle -> rh_middle_pad_site`, `ring -> rh_ring_pad_site`, `pinky -> rh_pinky_pad_site`
- right-hand joints: `cmc_flexion_r`, `cmc_abduction_r`, `mp_flexion_r`, `ip_flexion_r`, `mcp2_flexion_r`, `mcp2_abduction_r`, `pm2_flexion_r`, `md2_flexion_r`, `mcp3_flexion_r`, `mcp3_abduction_r`, `pm3_flexion_r`, `md3_flexion_r`, `mcp4_flexion_r`, `mcp4_abduction_r`, `pm4_flexion_r`, `md4_flexion_r`, `mcp5_flexion_r`, `mcp5_abduction_r`, `pm5_flexion_r`, `md5_flexion_r`
- right-hand actuators: `FDS5`, `FDS4`, `FDS3`, `FDS2`, `FDP5`, `FDP4`, `FDP3`, `FDP2`, `EDC5`, `EDC4`, `EDC3`, `EDC2`, `EDM`, `EIP`, `EPL`, `EPB`, `FPL`, `APL`, `OP`, `RI2`, `LU_RB2`, `UI_UB2`, `RI3`, `LU_RB3`, `UI_UB3`, `RI4`, `LU_RB4`, `UI_UB4`, `RI5`, `LU_RB5`, `UI_UB5`
- racket body/freejoint: `racket`, `racket_free`
- racket sites: `grip_pose_site`, `butt_site`, `stringbed_center_site`, `head_tip_site`, `handle_axis_start_site`, `handle_axis_end_site`, `racket_face_normal_site`
- handle contact geom: `handle_grip`

The environment maps the 31 right-hand actuator names directly to MuJoCo `data.ctrl` indices. Non-right-hand actuators are left untouched by the standalone grip action.

## Grip Targets

The six right-hand target points are racket-local cylindrical targets on or near the handle:

| target | hand site | y (m) | theta (deg) | weight |
| --- | --- | ---: | ---: | ---: |
| palm | `rh_palm_grip_site` | 0.085 | 180.0 | 1.5 |
| thumb | `rh_thumb_pad_site` | 0.122 | 45.0 | 2.0 |
| index | `rh_index_pad_site` | 0.125 | -45.0 | 2.0 |
| middle | `rh_middle_pad_site` | 0.098 | -115.0 | 1.6 |
| ring | `rh_ring_pad_site` | 0.075 | -135.0 | 1.4 |
| pinky | `rh_pinky_pad_site` | 0.055 | -150.0 | 1.2 |

The scene builder creates the hand pad sites and racket reference sites if they are absent from the source XML. The model-map audit is the quick check that every required site, joint, actuator, racket body, freejoint, and handle geom is present before solving or rollout.

## Static Reference

`solve_right_hand_racket_grip.py` optimizes right-hand `qpos` plus the racket freejoint translation and orientation. The objective is weighted site-target least squares, with a small regularization term toward the initial right-hand pose.

Current reference quality:

- optimizer success: `true`
- function evaluations: `63`
- least-squares cost: `0.005661604183419494`
- mean site error: `0.01813442561965174 m`
- max site error: `0.05020173478769403 m` at `palm`
- max handle penetration in the solved reference: `0.013639371367756514 m`
- IK mean threshold: `0.030 m`, pass
- training mean threshold: `0.020 m`, pass

The reference is adequate for a static site-matching starting pose, but it is not yet a physically accepted grasp. The palm remains the largest residual, and the handle penetration metric shows that the contact geometry still needs a contact-stable IK/control stage before final free-contact use.

## Environment

`RightHandRacketGripEnv` is a CPU MuJoCo environment with Gym-style `reset()` and `step(action)` methods.

- observation: concatenated full `qpos`, full `qvel`, and the current 31 right-hand actuator controls; the generated scene currently reports observation size `301`
- action: 31-dimensional right-hand actuator command vector, clipped to `[-1, 1]`
- reset state: solved reference `qpos`/`qvel`, zero controls, then `mj_forward`
- step: writes only the right-hand actuator controls, advances `control_substeps=10`, then reports reward and diagnostics
- episode length: `max_episode_steps=500`
- contacts: `contact_count` filters right-hand contact geoms against `handle_grip`; `illegal_handle_contact_count` reports non-hand handle contacts; `max_handle_penetration_m` reports the worst handle-related penetration seen over validation; `raw_contact_count` reports all MuJoCo contacts for debugging

Reward terms are reported with stable `r_*` names:

- `r_site_match`: negative weighted mean site error
- `r_contact`: positive reward when filtered hand-handle contact exists
- `r_effort`: negative mean squared action penalty
- `r_racket_pose`, `r_racket_orient`, `r_no_slip`, `r_reference_pose`, `r_joint_limits`, `r_no_penetration`: present as explicit terms and currently zero in this baseline environment

The training YAML records curriculum and reward settings. The implemented first stage is `curriculum_stage: 0`, a zero-action/reference-hold baseline for validating the scene, reference, reward terms, and acceptance metrics before adding policy optimization or staged perturbation curricula.

## Commands

Run commands from the repository root. The examples below use the project virtualenv explicitly:

```bash
/data3/yangfeiyang/WorkSpace/ENV/musclemimic/.venv/bin/python3 -m src.grip.build_right_hand_racket_grip_scene \
  --out assets/right_hand_racket_grip_scene.xml

/data3/yangfeiyang/WorkSpace/ENV/musclemimic/.venv/bin/python3 -m src.grip.hand_racket_model_map \
  --xml assets/right_hand_racket_grip_scene.xml

/data3/yangfeiyang/WorkSpace/ENV/musclemimic/.venv/bin/python3 -m src.grip.visualize_grip_sites \
  --xml assets/right_hand_racket_grip_scene.xml \
  --no-viewer

/data3/yangfeiyang/WorkSpace/ENV/musclemimic/.venv/bin/python3 -m src.grip.solve_right_hand_racket_grip \
  --xml assets/right_hand_racket_grip_scene.xml \
  --targets configs/right_hand_racket_grip_targets.json \
  --out configs/right_hand_racket_grip_reference.json

/data3/yangfeiyang/WorkSpace/ENV/musclemimic/.venv/bin/python3 -m src.grip.right_hand_racket_grip_env \
  --xml assets/right_hand_racket_grip_scene.xml \
  --targets configs/right_hand_racket_grip_targets.json \
  --reference configs/right_hand_racket_grip_reference.json \
  --smoke-test

/data3/yangfeiyang/WorkSpace/ENV/musclemimic/.venv/bin/python3 -m src.grip.evaluate_right_hand_racket_grip \
  --xml assets/right_hand_racket_grip_scene.xml \
  --targets configs/right_hand_racket_grip_targets.json \
  --reference configs/right_hand_racket_grip_reference.json \
  --episodes 1 \
  --steps 200

/data3/yangfeiyang/WorkSpace/ENV/musclemimic/.venv/bin/python3 -m src.grip.validate_right_hand_racket_grip \
  --xml assets/right_hand_racket_grip_scene.xml \
  --reference configs/right_hand_racket_grip_reference.json \
  --steps 1

/data3/yangfeiyang/WorkSpace/ENV/musclemimic/.venv/bin/python3 -m pytest \
  tests/test_right_hand_racket_grip.py -q
```

If the virtualenv is already active, the same entry points can be run with repo-root module syntax:

```bash
python -m src.grip.validate_right_hand_racket_grip \
  --xml assets/right_hand_racket_grip_scene.xml \
  --reference configs/right_hand_racket_grip_reference.json \
  --steps 1
```

Use `--strict` when the command should fail the process on configured acceptance failures:

```bash
/data3/yangfeiyang/WorkSpace/ENV/musclemimic/.venv/bin/python3 -m src.grip.validate_right_hand_racket_grip \
  --xml assets/right_hand_racket_grip_scene.xml \
  --reference configs/right_hand_racket_grip_reference.json \
  --steps 1 \
  --strict
```

## Current Validation Status

Current status: partial acceptance only. The handle contact filter now prevents non-hand body parts from contacting the handle, but the current reference still has excessive handle penetration and zero-action drift. Full acceptance does not pass.

Validation run on 2026-05-27:

```bash
/data3/yangfeiyang/WorkSpace/ENV/musclemimic/.venv/bin/python3 -m src.grip.validate_right_hand_racket_grip \
  --xml assets/right_hand_racket_grip_scene.xml \
  --reference configs/right_hand_racket_grip_reference.json \
  --steps 20
```

Observed metrics:

- `finite`: `true`
- `acceptance_pass`: `false`
- `mean_site_error_m`: `0.03788722945249839` against threshold `0.02` fail
- `contact_count`: `4` against threshold `4` pass
- `illegal_handle_contact_count`: `0` pass
- `max_handle_penetration_m`: `0.013639371367756514` against threshold `0.002` fail
- `translation_drift_m`: `0.3070762287095239` against threshold `0.01` fail
- `orientation_drift_deg`: `74.93382373015629` against threshold `8.0` fail
- `recovery_mean_site_error_m`: `0.11899683873745087` against threshold `0.02` fail
- `recovery_orientation_drift_deg`: `179.2779997611997` against threshold `12.0` fail

Default-horizon validation omits `--steps`, so it runs 200 zero-action steps. On the current reference it remains finite and exits `0` in non-strict mode, but `acceptance_pass` is still `false`; mean site error, contact count, translation drift, orientation drift, and recovery site error are all outside the configured thresholds at that horizon.

Exit behavior:

- non-strict validation exits `0` when metrics are finite, even if `acceptance_pass` is `false`.
- strict validation exits `2` when metrics are finite but any configured acceptance threshold fails.
- validation exits `1` when metrics are non-finite or simulation stepping fails.

## Interpreting Metrics

`acceptance_pass` is the conjunction of every item under the JSON `pass` object. A single failed configured threshold makes full acceptance false.

The current checks mean:

- `mean_site_error_m`: average distance from the six right-hand grip sites to their racket-local handle targets.
- `contact_count`: filtered right-hand contact count against configured handle geoms.
- `illegal_handle_contact_count`: number of handle contacts with non-right-hand geoms.
- `max_handle_penetration_m`: maximum handle-related penetration over the validation rollout.
- `translation_drift_m`: racket body translation drift during the zero-action reference hold.
- `orientation_drift_deg`: racket body orientation drift during the zero-action reference hold.
- `recovery_mean_site_error_m`: mean site error after applying the configured racket perturbation and recovery window.
- `recovery_orientation_drift_deg`: racket orientation drift after the perturbation recovery window.
- `finite`: observations, rewards, and reported metric values stayed finite.

The configured perturbation is `2.0 N` force and `0.03 N*m` torque with a `0.5 s` recovery window. The current reference can hold the static pose for the smoke validation, but it does not yet recover the grip within the recovery thresholds.

## Current Limitations

- The standalone environment is a CPU MuJoCo reference-hold task; it is not integrated into the full-body badminton training loop.
- The current rollout uses zero action around a solved reference state. It is a validation baseline, not a trained recovery controller.
- Perturbation recovery is outside the configured thresholds, so this pipeline should not be treated as fully accepted.
- The scene is generated without a permanent hand-racket weld. Any future curriculum assistance should remain explicit and optional.
- The reference has a relatively large palm site error compared with the finger site errors, even though the mean static site error is under the current threshold.
- The current reference is not contact-stable: handle penetration and zero-action drift remain outside acceptance thresholds.

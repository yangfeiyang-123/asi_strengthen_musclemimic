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

Current status: partial acceptance only. The one-step smoke validation is finite and passes the immediate mean site error, contact count, racket translation drift, and racket orientation drift checks. Perturbation recovery does not meet the configured acceptance thresholds. The default validation horizon is longer (`--steps 200`) and currently shows additional zero-action drift/contact failures, so full acceptance does not pass.

Validation run on 2026-05-26:

```bash
/data3/yangfeiyang/WorkSpace/ENV/musclemimic/.venv/bin/python3 -m src.grip.validate_right_hand_racket_grip \
  --xml assets/right_hand_racket_grip_scene.xml \
  --reference configs/right_hand_racket_grip_reference.json \
  --steps 1
```

Observed metrics:

- `finite`: `true`
- `acceptance_pass`: `false`
- `mean_site_error_m`: `0.01950480574865326` against threshold `0.02` pass
- `contact_count`: `14` against threshold `4` pass
- `translation_drift_m`: `0.005767179940576803` against threshold `0.01` pass
- `orientation_drift_deg`: `1.6279944228772285` against threshold `8.0` pass
- `recovery_mean_site_error_m`: `0.26983943850468134` against threshold `0.02` fail
- `recovery_orientation_drift_deg`: `66.88779275731056` against threshold `12.0` fail

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

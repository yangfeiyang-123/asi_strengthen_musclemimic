# Task 05 — Validate and document the full right-hand racket grip pipeline

Goal: add automated validation and a clear report so we can decide whether the hand has learned a usable grip.

## Required outputs

Create or update:

```text
tests/test_right_hand_racket_grip.py
src/grip/validate_right_hand_racket_grip.py
docs/right_hand_racket_grip.md
```

## Validation script

`src/grip/validate_right_hand_racket_grip.py` should run these tests:

1. XML/model load:
   - scene loads,
   - all required sites/geoms/actuators exist,
   - no duplicate critical names.
2. Reference pose:
   - reference JSON loads,
   - qpos/qvel dimensions match current model,
   - mean site error is printed.
3. Static hold:
   - reset to reference,
   - simulate 2 seconds with gravity,
   - report racket grip frame drift and orientation drift.
4. Contact report:
   - count handle contacts by finger/palm group,
   - report missing critical contacts.
5. Perturbation test:
   - apply force/torque perturbation,
   - report recovery over 0.5 seconds.
6. Safety:
   - detect NaN/Inf,
   - detect severe joint limit violations,
   - detect persistent illegal penetration if possible.

## Acceptance thresholds

Use configurable thresholds:

```text
mean grip-site error < 0.02 m after training; < 0.03 m for IK-only reference
racket translation drift < 0.01 m over 2 s
racket orientation drift < 8 deg over 2 s
min meaningful handle contacts >= 4
perturb recovery under 0.02 m and 12 deg within 0.5 s
```

If these are not achieved, the script should print FAIL with a detailed reason, not hide it.

## Documentation

`docs/right_hand_racket_grip.md` must include:

- model files used,
- discovered right-hand and racket naming map,
- grip coordinate convention,
- target site definitions,
- reference IK objective,
- training env observation/action/reward,
- curriculum stages,
- validation metrics and current results,
- known limitations.

## Final commands to verify

Make these pass, or document precisely why they cannot pass yet:

```bash
python -m src.grip.hand_racket_model_map --xml assets/right_hand_racket_grip_scene.xml
python src/grip/visualize_grip_sites.py --xml assets/right_hand_racket_grip_scene.xml --no-viewer
python src/grip/solve_right_hand_racket_grip.py --xml assets/right_hand_racket_grip_scene.xml --targets configs/right_hand_racket_grip_targets.json --out configs/right_hand_racket_grip_reference.json
python src/grip/right_hand_racket_grip_env.py --xml assets/right_hand_racket_grip_scene.xml --smoke-test
python src/grip/validate_right_hand_racket_grip.py --xml assets/right_hand_racket_grip_scene.xml --reference configs/right_hand_racket_grip_reference.json
pytest -q tests/test_right_hand_racket_grip.py
```

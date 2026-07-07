# Forehand Clear Static-Hit PostTrain

This workflow stages a static-shuttle ForehandClear task in the Overall badminton training scene. The current implementation provides the target calculation, layered control helpers, static-hit state logic, physics hook seams, grip disturbance configuration, experiment staging spec, and a dedicated preflight/physics-smoke runner.

Long-horizon static-hit PPO is still intentionally separate from the ordinary fullbody runner. The experiment spec is marked with `runner_type: static_hit_staging`, so `musclemimic/badminton/scripts/run_posttrain_experiment.py --stage prepare` writes config snapshots plus dedicated static-hit command files, while ordinary `train`, `eval`, `render`, and `all` still fail fast.

## Stage Order

1. Physics chain validation: static shuttle freeze, impact release, string-bed force, event rebound, aero, net crossing, and landing diagnostics.
2. Static grip stabilizer: train the right hand to hold the racket from the current grip reference.
3. Swing-disturbance grip stabilizer: train the right hand under ForehandClear-like wrist, forearm, and racket inertia disturbances.
4. Hit-and-over-net post-train: compose body and grip policies and optimize valid impact plus net clearance.
5. High-clear depth post-train: add back-court landing and clear-arc objectives.

## Key Files

- Spec: `docs/superpowers/specs/2026-05-27-forehand-clear-static-hit-posttrain-design.md`
- Plan: `docs/superpowers/plans/2026-05-27-forehand-clear-static-hit-posttrain.md`
- Overall training scene: `environment/overall_environment/assets/overall_badminton_training_scene.xml`
- Static-hit env: `environment/overall_environment/src/static_forehand_clear_env.py`
- Static-hit runner: `musclemimic/badminton/scripts/run_forehand_clear_static_hit.py`
- Impact target helper: `environment/overall_environment/src/impact_target.py`
- Layered control helper: `environment/overall_environment/src/layered_control.py`
- Grip trainer: `src/grip/train_right_hand_racket_grip_policy.py`
- Experiment spec: `experiments/posttrain/forehand_clear_static_hit_v1.yaml`

## Staging Command

```bash
python musclemimic/badminton/scripts/run_posttrain_experiment.py \
  --spec experiments/posttrain/forehand_clear_static_hit_v1.yaml \
  --stage prepare
```

The command writes outputs under:

```text
outputs/posttrain/ForehandClearStaticHit/v1/
```

The `commands/README_static_hit.txt` file in that directory records why
ordinary fullbody train/eval/render commands are intentionally not generated.
The prepare stage also writes:

```text
commands/static_hit_preflight.sh
commands/static_hit_physics_smoke.sh
```

You can run the dedicated runner directly:

```bash
uv run python musclemimic/badminton/scripts/run_forehand_clear_static_hit.py \
  --spec experiments/posttrain/forehand_clear_static_hit_v1.yaml \
  --stage preflight

uv run python musclemimic/badminton/scripts/run_forehand_clear_static_hit.py \
  --spec experiments/posttrain/forehand_clear_static_hit_v1.yaml \
  --stage physics-smoke \
  --steps 120
```

The static shuttle target is not read from the scene keyframe. The runner first
checks `shuttle.static_qpos`; if it is absent, it derives a seven-value free-joint
qpos from `scene.root_start_xy`, `scene.player_half_sign`, the
`impact_target.regularization` forward/side offsets, and
`shuttle.static_height_m`. The default spec freezes the shuttle at
`[-1.91, 0.43, 1.40, 1, 0, 0, 0]`, which keeps physics-smoke validation in the
pre-impact air state instead of immediately reporting a ground landing.

## Validation Commands

```bash
pytest environment/overall_environment/tests/test_impact_target.py -q
pytest environment/overall_environment/tests/test_layered_control.py -q
pytest environment/overall_environment/tests/test_static_forehand_clear_env.py -q
pytest tests/test_right_hand_racket_grip.py -q
pytest tests/unit/test_forehand_clear_static_hit_spec.py -q
pytest tests/unit/test_forehand_clear_static_hit_runner.py -q
```

## Current Gating Conditions

Do not start composed Overall static-hit training until a trained grip policy exists at:

```text
outputs/right_hand_racket_grip/policy/policy_latest.pt
```

Do not run the staged spec through `fullbody/experiment.py` or `fullbody/eval.py`. Those runners cannot instantiate `StaticForehandClearEnv` or apply `env_params.static_hit_params`; static-hit work must go through the dedicated runner until a full PPO adapter is implemented.

Acceptance gates reject pose-servo usage, body fall, racket drop, missing
impact, missing net crossing, or landing outside the configured target region.

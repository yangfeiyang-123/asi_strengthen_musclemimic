# Right-hand badminton racket grip integration package

Purpose: give Codex a concrete implementation plan for making an existing MuJoCo musculoskeletal right hand learn and hold a correct badminton racket grip.

Use this package from the root of your own repository. The prompts are intentionally self-contained and tell Codex to discover your actual XML/body/site/actuator names instead of assuming a fixed musculoskeletal model layout.

Recommended sequence:

```bash
# From the root of your repository
mkdir -p codex_tasks
cp /path/to/right_hand_racket_grip_codex_package/prompts/*.md codex_tasks/

codex exec "$(cat codex_tasks/01_audit_models.md)"
codex exec "$(cat codex_tasks/02_add_grip_sites_and_scene.md)"
codex exec "$(cat codex_tasks/03_implement_grip_ik_solver.md)"
codex exec "$(cat codex_tasks/04_implement_grip_training_env.md)"
codex exec "$(cat codex_tasks/05_validate_grip_pipeline.md)"
```

If `codex exec` is not available or your Codex version behaves differently, run `codex` interactively and paste each prompt in order.

Primary deliverables expected in your repo after Codex finishes:

- `docs/right_hand_racket_grip.md`
- `configs/right_hand_racket_grip_targets.json`
- `assets/right_hand_racket_grip_scene.xml` or equivalent merged scene
- `src/grip/hand_racket_model_map.py`
- `src/grip/solve_right_hand_racket_grip.py`
- `src/grip/right_hand_racket_grip_env.py`
- `src/grip/train_right_hand_racket_grip.py`
- `src/grip/validate_right_hand_racket_grip.py`
- `tests/test_right_hand_racket_grip.py`


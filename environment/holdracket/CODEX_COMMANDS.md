# Commands to give Codex

From the root of your actual repository:

```bash
mkdir -p codex_tasks
cp /path/to/right_hand_racket_grip_codex_package/prompts/*.md codex_tasks/
cp /path/to/right_hand_racket_grip_codex_package/configs/right_hand_racket_grip_targets.json configs/ 2>/dev/null || true
```

Then run these in order:

```bash
codex exec "$(cat codex_tasks/01_audit_models.md)"
codex exec "$(cat codex_tasks/02_add_grip_sites_and_scene.md)"
codex exec "$(cat codex_tasks/03_implement_grip_ik_solver.md)"
codex exec "$(cat codex_tasks/04_implement_grip_training_env.md)"
codex exec "$(cat codex_tasks/05_validate_grip_pipeline.md)"
```

After Codex changes the repo, run:

```bash
python -m src.grip.hand_racket_model_map --xml assets/right_hand_racket_grip_scene.xml
python src/grip/visualize_grip_sites.py --xml assets/right_hand_racket_grip_scene.xml --no-viewer
python src/grip/solve_right_hand_racket_grip.py --xml assets/right_hand_racket_grip_scene.xml --targets configs/right_hand_racket_grip_targets.json --out configs/right_hand_racket_grip_reference.json
python src/grip/right_hand_racket_grip_env.py --xml assets/right_hand_racket_grip_scene.xml --smoke-test
python src/grip/validate_right_hand_racket_grip.py --xml assets/right_hand_racket_grip_scene.xml --reference configs/right_hand_racket_grip_reference.json
pytest -q tests/test_right_hand_racket_grip.py
```

If `codex exec` is not supported by your Codex version, open interactive Codex:

```bash
codex
```

Then paste the contents of each prompt file in order.

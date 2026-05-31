# Task 03 — Implement a static reference grip-pose solver

Goal: compute a plausible right-hand badminton grip posture around the racket handle, then save it as a reference keyframe/state for training.

## Required output

Create:

```text
src/grip/solve_right_hand_racket_grip.py
src/grip/grip_objectives.py
configs/right_hand_racket_grip_reference.json
```

## Solver behavior

The solver should:

1. Load `assets/right_hand_racket_grip_scene.xml` or a user-provided `--xml`.
2. Load model map and grip target config.
3. Identify right-hand qpos variables only:
   - hand/finger joints,
   - optionally wrist joints,
   - optionally right arm joints if needed for reach.
4. Keep the racket initially fixed in a convenient pose near the right palm, or set its freejoint pose so `grip_pose_site` is close to `rh_palm_grip_site`.
5. Optimize qpos to minimize a weighted objective.

## Objective terms

Implement these terms with readable functions and tunable weights:

```text
site_target_loss:
  For palm/thumb/index/middle/ring/pinky sites, match the corresponding target point on the racket handle cylinder.

racket_palm_frame_loss:
  Keep racket grip_pose_site close to rh_palm_grip_site.
  Keep handle axis plausibly aligned through the palm/thumb-index web.

finger_wrap_loss:
  Encourage thumb and index on opposing sides of the handle; middle/ring/pinky curled around lower handle.

joint_regularization:
  Penalize deviation from a neutral hand pose but allow flexion.

joint_limit_loss:
  Penalize qpos outside limits or very close to unsafe extremes.

penetration_loss:
  Penalize obvious illegal hand-racket/body penetration outside the handle contact region.

contact_readiness_loss:
  Finger pad sites should be close to the handle surface with small clearance, not floating far away.
```

Use `scipy.optimize.least_squares` or `scipy.optimize.minimize` if SciPy is available. If SciPy is not available, implement a fallback random/CEM-style local search so the script still runs.

## Reference output

Save a JSON file:

```json
{
  "xml": "assets/right_hand_racket_grip_scene.xml",
  "keyframe_name": "right_hand_racket_grip_ref",
  "qpos": [...],
  "qvel": [...],
  "racket_freejoint_qpos": [...],
  "right_hand_joint_names": [...],
  "site_errors_m": {...},
  "objective_breakdown": {...},
  "notes": "..."
}
```

Also add a MuJoCo keyframe if the repository style supports it. If editing the XML keyframe is too risky, provide a loader that applies the JSON state.

## Validation command

This command should solve or load a solution, then print a clear report:

```bash
python src/grip/solve_right_hand_racket_grip.py \
  --xml assets/right_hand_racket_grip_scene.xml \
  --targets configs/right_hand_racket_grip_targets.json \
  --out configs/right_hand_racket_grip_reference.json
```

Acceptance for this stage:

- mean grip-site error should be under 3 cm if possible,
- no missing sites,
- no obvious extreme finger hyperextension,
- at least four finger/palm sites near the handle surface.

If the result is poor because the actual hand model lacks enough DOFs/sites, document the limitation and still save the best state.

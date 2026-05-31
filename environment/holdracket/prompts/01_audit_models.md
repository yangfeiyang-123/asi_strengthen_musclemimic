# Task 01 — Audit the musculoskeletal hand and badminton racket models

You are working in a repository that already contains:

- a MuJoCo musculoskeletal human/hand model with a right hand,
- a MuJoCo badminton racket model,
- likely Python simulation/training code.

Goal: inspect the repository and produce a precise model map for later grip-pose solving and training. Do not assume exact XML names; discover them.

## Required work

1. Search the repository for MuJoCo XML/MJCF files and identify:
   - the main scene/model file for the musculoskeletal model,
   - the file containing the right hand or full body,
   - the file containing the badminton racket.
2. Inspect right-hand kinematic/body names:
   - wrist/hand/palm body,
   - thumb distal/proximal bodies/geoms,
   - index/middle/ring/pinky distal or pad bodies/geoms,
   - existing sites that could serve as palm/fingertip/pad targets,
   - joint names and qpos addresses for right-hand joints,
   - actuator names and control ranges for right hand muscles/motors.
3. Inspect racket names:
   - racket body name,
   - freejoint name if present,
   - handle geoms,
   - `grip_pose_site`, `butt_site`, `stringbed_center_site`, `head_tip_site` if present.
4. Create `src/grip/hand_racket_model_map.py` with dataclasses or dictionaries containing all discovered names and a helper function:

```python
def load_model_map(model: mujoco.MjModel) -> HandRacketModelMap:
    """Validate all configured body/site/geom/actuator names against the loaded MuJoCo model."""
```

5. Create `docs/right_hand_racket_grip_audit.md` summarizing:
   - discovered XML paths,
   - right-hand body/site/joint/actuator map,
   - racket body/site/geom map,
   - missing annotations that Task 02 must add,
   - any risky assumptions.

## Constraints

- Do not modify core musculoskeletal biomechanics in this task.
- Do not hard-code qpos indices without validating via `mj_name2id` and model arrays.
- If multiple candidate files exist, choose the one most likely used by existing training/simulation scripts and document why.

## Validation

Add or update a test/script so this command succeeds:

```bash
python -m src.grip.hand_racket_model_map --xml <main_scene_or_model.xml>
```

The script should print a clear PASS/FAIL report and list unresolved names.

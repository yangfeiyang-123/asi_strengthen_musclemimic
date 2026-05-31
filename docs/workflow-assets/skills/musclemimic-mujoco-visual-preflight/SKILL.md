---
name: musclemimic-mujoco-visual-preflight
description: Verify MuscleMimic MuJoCo scenes before training or evaluation by checking reset screenshots/videos, local-viewer portability, anatomy visibility, grip pose, racket and shuttle placement, passive physics, and court orientation. Use when the user asks for a visual window, initialization screenshot, local MuJoCo viewing, scene display fixes, grip-pose alignment, object falling behavior, racket handle shape, shuttle orientation, or whether a training environment is visually ready.
---

# Musclemimic MuJoCo Visual Preflight

## Overview

Use this workflow before launching expensive RL runs or after modifying XML/assets. The goal is to catch scene-level errors that scalar tests miss: wrong anatomy, wrong orientation, bad grip seed transfer, object penetration, fixed bodies that should fall, or visual artifacts that make local inspection misleading.

## Workflow

1. Locate the scene entry point.
   - Prefer the current environment module or builder script named in the user request.
   - For overall badminton scenes, inspect `environment/overall_environment/`.
   - For grip-only references, inspect `environment/holdracket/`, `outputs/right_hand_racket_grip/`, and the current scene-generation script.

2. Produce noninteractive evidence first.
   - Generate reset screenshots or short videos when scripts support it.
   - Prefer fixed camera outputs: full scene, right-hand closeup, racket handle closeup, and shuttle/court view.
   - If a GUI viewer is needed, confirm `DISPLAY` or local MuJoCo availability before recommending it.

3. Check visual invariants.
   - Anatomy: musculoskeletal body, bones, and muscles are visible when expected; no missing head or placeholder body.
   - Orientation: athlete faces the net when the task requires it.
   - Grip: right-hand fingers are bent consistently with the seed/reference, with no obvious handle penetration.
   - Racket: handle dimensions and octagonal bevels match the current standard/reference files.
   - Shuttle: initial pose matches the task, such as on ground, in air, or frozen pre-impact.
   - Court/materials: floor/court colors are distinguishable and reflections do not hide contacts.

4. Check physics invariants.
   - On reset, controlled static scenes should stay stable.
   - On run/play, free objects should fall or move naturally unless intentionally frozen.
   - Passive fixtures should not inject forces into the body.
   - Contacts should be scoped to intended geoms, especially grip pads and handle regions.

5. Trace mismatches to source.
   - If a reference output looks correct but the training scene does not, compare qpos/qvel transfer, model name maps, keyframes, and XML includes.
   - If local MuJoCo differs from script output, check asset paths, generated XML path, mesh/material includes, and viewer camera defaults.
   - If hand pose is straight in one scene but bent in another, inspect whether finger joints are omitted, overwritten by a keyframe, or clipped by a reduced model map.

6. Gate training readiness.
   - Do not call a scene ready if reset screenshots are missing, the full-body orientation is wrong, or grip/shuttle/racket geometry cannot be visually verified.
   - For expensive RL runs, require a reset image plus either a short rollout video or a clear explanation of why physics is intentionally static.

## Useful Commands

Use these as starting points, adjusting module/script names to the current scene:

```bash
.venv/bin/python -m environment.overall_environment.src.overall_env \
  --xml environment/overall_environment/assets/overall_badminton_scene.xml
```

```bash
.venv/bin/python -m environment.overall_environment.src.overall_env \
  --xml environment/overall_environment/assets/overall_badminton_scene.xml \
  --viewer
```

If the viewer fails with missing `DISPLAY`, switch to offscreen screenshot/video generation or tell the user which XML/assets to open locally.

## Output Format

Return:

- Verdict: `ready`, `ready_with_risks`, `not_ready`, or `inconclusive`.
- Evidence produced: paths to screenshots/videos/XML and what each shows.
- Failed invariant: exact visual or physics issue.
- Source hypothesis: likely file/function/keyframe/config causing it.
- Next action: smallest fix or validation command before training.

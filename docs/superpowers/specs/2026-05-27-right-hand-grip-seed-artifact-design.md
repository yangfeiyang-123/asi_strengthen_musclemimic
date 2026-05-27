# Right-Hand Grip Seed Artifact Design

## Problem

`outputs/right_hand_racket_grip/visualization` contains the hand-racket images the user visually accepts as the correct starting grip, but that directory only contains PNG files. It has no qpos JSON, XML package, policy checkpoint, or reproducible generation metadata.

The current training and Overall scene initialization instead depend on `configs/right_hand_racket_grip_reference.json`. That reference is loadable, but it is not a reliable representation of the visually accepted grip:

- Several finger flexion joints are at or near extension: `mcp4_flexion_r = 0`, `pm5_flexion_r = 0`, and `md5_flexion_r = 0`.
- The palm and pinky target errors are large enough to make the hand look under-wrapped around the handle.
- Validation reports partial acceptance only; the reset is finite but contact count, recovery, and longer-horizon stability fail configured thresholds.
- The accepted PNGs cannot be loaded by MuJoCo, so they cannot directly serve as training initial state.

The fix is to introduce a first-class grip seed artifact: a versioned, loadable, visualized initial state that both the standalone grip environment and the Overall badminton environment use as the single source of truth.

## Goals

- Create a reproducible grip seed state that can be loaded into MuJoCo without relying on images or hidden temporary scripts.
- Make the Overall initial state and grip training reset use the same seed qpos/racket pose.
- Preserve the user-approved visual target by generating comparison renders next to the existing `outputs/right_hand_racket_grip/visualization` images.
- Add tests that fail if Overall or training silently drift away from the seed.
- Keep this scoped to initialization/reference quality. Do not implement full swing-hit RL in this change.

## Non-Goals

- Do not infer a unique qpos directly from PNG pixels.
- Do not add a permanent hand-racket weld to the final scene.
- Do not hide mesh geometry as the primary fix. Visualization can have diagnostic modes, but training must use a real qpos/racket pose.
- Do not require a trained grip policy checkpoint before the static seed is usable.

## Approach

Use a seed artifact plus a better seed builder.

The seed builder will generate a qpos/racket pose using the existing MuJoCo hand-racket scene, but it will no longer rely only on six surface site targets. It will also include explicit hand-shape priors for the accepted forehand V-grip:

- Keep thumb and index opposed around the handle.
- Require middle/ring/pinky to curl around the lower handle instead of remaining extended.
- Penalize finger joints sitting exactly at flexion lower bounds unless that is anatomically intended.
- Keep handle penetration below the current validation threshold.
- Keep the racket freejoint aligned to the solved hand pose.

The accepted PNGs remain the visual oracle, but the generated seed state is the actual artifact consumed by code.

## Artifact Layout

Add a new output directory:

```text
outputs/right_hand_racket_grip/reference/
  right_hand_racket_grip_seed.json
  right_hand_racket_grip_seed_scene.xml
  right_hand_racket_grip_seed_report.json
  visualization/
    seed_grip_closeup_front.png
    seed_grip_closeup_side.png
    seed_overall_initial_grip.png
    seed_comparison_montage.png
```

`right_hand_racket_grip_seed.json` contains:

- `schema_version`
- `source_xml`
- `target_config`
- `qpos`
- `qvel`
- `right_hand_joint_names`
- `racket_freejoint_name`
- `racket_freejoint_qpos`
- `site_errors_m`
- `joint_shape_metrics`
- `contact_metrics`
- `visualization_paths`
- `generation_command`

`right_hand_racket_grip_seed_scene.xml` is the MuJoCo scene with a named seed keyframe. It is not a separate hand model; it is a portable package of the exact scene used to validate the seed.

## Data Flow

1. The seed builder loads `assets/right_hand_racket_grip_scene.xml`, `configs/right_hand_racket_grip_targets.json`, and the existing reference as an optional warm start.
2. It solves or selects a better hand-racket pose and writes `right_hand_racket_grip_seed.json`.
3. It renders seed closeups and a montage next to the existing accepted PNGs.
4. `RightHandRacketGripEnv` accepts a `--reference` path pointing to the seed JSON and resets from it.
5. `environment/overall_environment/src/build_overall_environment.py` reads the same seed JSON and copies the named right-hand joints plus racket freejoint pose into `overall_ready`.
6. Overall package generation includes the seed JSON path in the README and renders a close-up proving the loaded state matches the seed.

## Components

### Seed Builder

Add a dedicated CLI, for example `src/grip/build_right_hand_racket_grip_seed.py`.

Responsibilities:

- Load the model, targets, and optional warm-start reference.
- Build an objective with surface site errors, finger curl priors, joint-boundary penalties, handle penetration penalty, and racket pose consistency.
- Save the seed JSON and report.
- Render deterministic verification images.

The builder should be deterministic for a fixed seed and model.

### Seed Loader

Add a small loader helper in `src/grip`, reused by both standalone grip code and Overall.

Responsibilities:

- Validate qpos/qvel dimensions against the current model.
- Validate that every named right-hand joint exists.
- Copy matching scalar joint values by joint name, not by raw index alone.
- Copy the racket freejoint by name.
- Report missing or mismatched fields with explicit errors.

### Overall Integration

Overall should use the seed JSON as its default grip reference source. It should keep the existing body placement, face-net rotation, shuttle placement, and scene packaging behavior, but the right-hand and racket pose must come from the seed artifact.

### Verification Renders

The renders are not the source of truth, but they are required diagnostics. They should include:

- Seed hand-racket close-up with the same visual style as the accepted PNGs.
- Overall initial close-up using the same seed.
- Optional mesh-visible and collision-only diagnostic views.

## Acceptance Criteria

The implementation is acceptable when:

- `outputs/right_hand_racket_grip/reference/right_hand_racket_grip_seed.json` exists and loads against the current grip scene.
- Overall `overall_ready` uses the same right-hand joint values and racket pose as the seed within numerical tolerance.
- New seed validation reports finite reset and no severe handle penetration.
- Ring and pinky are no longer trivially extended in the seed unless explicitly justified in the report.
- Generated seed closeups are available for visual review beside the existing accepted images.
- Tests cover seed loading, Overall seed copying, and dimension/name mismatch failures.

## Testing

Add targeted tests:

- Seed JSON schema and dimension validation.
- Seed loader copies right-hand joints by name.
- Seed loader rejects missing joints or wrong qpos length.
- Overall ready qpos matches the seed for all named right-hand joints and racket freejoint.
- Seed builder smoke test writes JSON/report/renders in a temp directory with a low iteration count.

Existing grip and Overall tests should continue to pass.

## Risks

- The PNGs cannot define the exact historical qpos, so the new seed will be an approximation validated by metrics and visual comparison.
- A visually curled grip may still be dynamically unstable without a trained controller. The seed only improves reset quality; training still needs grip stabilization.
- MuJoCo mesh/collision rendering can make the same qpos look different across views. Diagnostic renders should show both mesh-visible and collision-only modes to avoid confusing display issues with state issues.

## Rollout

1. Add seed schema and loader tests.
2. Implement seed loader.
3. Implement seed builder and deterministic renders.
4. Generate the seed artifact under `outputs/right_hand_racket_grip/reference`.
5. Switch Overall default grip source to the seed.
6. Regenerate the Overall visualization package.
7. Run targeted grip and Overall tests.


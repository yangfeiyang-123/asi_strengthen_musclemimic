# Overall Badminton Environment

This package builds a single MuJoCo scene containing:

- the existing BWF badminton court and net from `environment/court`
- the original `musclemimic_models.get_xml_path("myofullbody")` person model
- the rigid badminton racket from `environment/racket`
- the shuttlecock from `environment/shuttlecock`, placed cork-down on the court floor

All new code and generated files live under `environment/overall_environment`; the source court, racket, shuttlecock, and muscle model assets are not modified.
The generated scene keeps the MyoFullBody musculoskeletal mesh assets under `assets/mimic_msk_model`, so the bones/head/muscle-tendon visualization can be opened locally without the original model package.
The skybox/background image is intentionally removed.

## Build

Run from the repository root:

```bash
/data3/yangfeiyang/WorkSpace/ENV/musclemimic/.venv/bin/python3 \
  -m environment.overall_environment.src.build_overall_environment \
  --out environment/overall_environment/assets/overall_badminton_scene.xml
```

The generated XML contains an `overall_ready` keyframe and also writes the same pose into MuJoCo's initial `qpos0`: the person starts from the natural MyoFullBody standing pose with right-hand fingers in the grip reference, the racket handle starts at the right palm, and the shuttlecock starts cork-down on the court floor.

## Smoke Test

```bash
/data3/yangfeiyang/WorkSpace/ENV/musclemimic/.venv/bin/python3 \
  -m environment.overall_environment.src.overall_env \
  --xml environment/overall_environment/assets/overall_badminton_scene.xml
```

Expected output includes:

```json
{
  "has_court": true,
  "has_racket": true,
  "has_shuttlecock": true,
  "keyframe": "overall_ready"
}
```

## Visualize

Use `--viewer` to open an interactive MuJoCo window after the environment resets to `overall_ready`.
The viewer is static by default: it does not advance physics, so the person, racket, and shuttlecock stay in the configured inspection pose.
The viewer keeps the musculoskeletal display layers used by MuJoCo by default: bone meshes, skin, and tendons are visible, while joints, actuators, contacts, and high-numbered debug groups are hidden.

```bash
/data3/yangfeiyang/WorkSpace/ENV/musclemimic/.venv/bin/python3 \
  -m environment.overall_environment.src.overall_env \
  --xml environment/overall_environment/assets/overall_badminton_scene.xml \
  --viewer
```

For MuJoCo's built-in play/pause controls, use the managed native viewer:

```bash
/data3/yangfeiyang/WorkSpace/ENV/musclemimic/.venv/bin/python3 \
  -m environment.overall_environment.src.overall_env \
  --xml environment/overall_environment/assets/overall_badminton_scene.xml \
  --native-viewer
```

Add `--simulate` when you intentionally want the passive viewer to step raw MuJoCo physics forward:

```bash
/data3/yangfeiyang/WorkSpace/ENV/musclemimic/.venv/bin/python3 \
  -m environment.overall_environment.src.overall_env \
  --xml environment/overall_environment/assets/overall_badminton_scene.xml \
  --viewer \
  --simulate
```

Use `--pose-servo` only when you want a weak pose-stabilized preview. Without a trained policy or a hand-racket constraint, raw physics can make the person fall and the racket move away from the hand:

```bash
/data3/yangfeiyang/WorkSpace/ENV/musclemimic/.venv/bin/python3 \
  -m environment.overall_environment.src.overall_env \
  --xml environment/overall_environment/assets/overall_badminton_scene.xml \
  --viewer \
  --simulate \
  --pose-servo
```

This requires a working desktop display or X11 forwarding. If you are connected over SSH, use an X-enabled session such as `ssh -X`/`ssh -Y`, VNC, or a machine-local terminal.

Add `--debug-visuals` only when you intentionally want all MuJoCo visual groups, joints, tendons, sites, and actuators for debugging.

## Local Static Viewer

For local inspection, download the whole `environment/overall_environment` directory and keep its internal layout unchanged:

```text
overall_environment/
  assets/overall_badminton_scene.xml
  assets/mimic_msk_model/meshes/
  src/overall_env.py
```

Install only the viewer dependencies on the local machine:

```bash
python -m pip install mujoco numpy
```

Then run the portable static viewer from the directory that contains `overall_environment`:

```bash
python overall_environment/src/overall_env.py \
  --xml overall_environment/assets/overall_badminton_scene.xml \
  --viewer
```

For the MuJoCo play/pause UI locally, run:

```bash
python overall_environment/src/overall_env.py \
  --xml overall_environment/assets/overall_badminton_scene.xml \
  --native-viewer
```

Do not use `--pose-servo` for normal viewing; it is only a stabilization aid and can introduce visual jitter.

## Tests

```bash
/data3/yangfeiyang/WorkSpace/ENV/musclemimic/.venv/bin/python3 \
  -m pytest environment/overall_environment/tests/test_overall_environment.py -q
```

## Notes

- The person and racket pose comes from `configs/right_hand_racket_grip_reference.json`.
- The racket is still modeled as a free body; the keyframe places it in the hand, but it is not welded to the hand.
- The shuttlecock is initialized cork-down on the floor using its `overall_shuttle_free` freejoint.

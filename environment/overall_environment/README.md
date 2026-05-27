# Overall Badminton Environment

This package builds a single MuJoCo scene containing:

- the existing BWF badminton court and net from `environment/court`
- the original `musclemimic_models.get_xml_path("myofullbody")` person model
- the rigid badminton racket from `environment/racket`
- the shuttlecock from `environment/shuttlecock`, placed cork-down on the court floor

All new code and generated files live under `environment/overall_environment`; the source court, racket, shuttlecock, and muscle model assets are not modified.

## Build

Run from the repository root:

```bash
/data3/yangfeiyang/WorkSpace/ENV/musclemimic/.venv/bin/python3 \
  -m environment.overall_environment.src.build_overall_environment \
  --out environment/overall_environment/assets/overall_badminton_scene.xml
```

The generated XML contains an `overall_ready` keyframe. Resetting to that keyframe puts the person in the current right-hand grip reference, places the racket handle near the right hand, and places the shuttlecock on the court floor.

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
The viewer uses the same clean startup policy as the main MuscleMimic viewer: it shows the model/court visual geometry and hides MuJoCo debug layers such as tendons, joints, sites, actuators, contact overlays, and transparent groups.

```bash
/data3/yangfeiyang/WorkSpace/ENV/musclemimic/.venv/bin/python3 \
  -m environment.overall_environment.src.overall_env \
  --xml environment/overall_environment/assets/overall_badminton_scene.xml \
  --viewer
```

Add `--simulate` when you intentionally want the whole scene to step forward. This mode uses a pose servo by default, keeping the person and racket close to the `overall_ready` reference while physics advances:

```bash
/data3/yangfeiyang/WorkSpace/ENV/musclemimic/.venv/bin/python3 \
  -m environment.overall_environment.src.overall_env \
  --xml environment/overall_environment/assets/overall_badminton_scene.xml \
  --viewer \
  --simulate
```

Use `--free-simulate` only when you want raw MuJoCo physics with no pose stabilization. Without a trained policy or a hand-racket constraint, raw physics can make the person fall and the racket move away from the hand:

```bash
/data3/yangfeiyang/WorkSpace/ENV/musclemimic/.venv/bin/python3 \
  -m environment.overall_environment.src.overall_env \
  --xml environment/overall_environment/assets/overall_badminton_scene.xml \
  --viewer \
  --simulate \
  --free-simulate
```

This requires a working desktop display or X11 forwarding. If you are connected over SSH, use an X-enabled session such as `ssh -X`/`ssh -Y`, VNC, or a machine-local terminal.

Add `--debug-visuals` only when you intentionally want all MuJoCo visual groups, joints, tendons, sites, and actuators for debugging.

## Local Static Viewer

For local inspection, download the whole `environment/overall_environment` directory and keep its internal layout unchanged:

```text
overall_environment/
  assets/overall_badminton_scene.xml
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

Do not use `--simulate` for pose checking. Simulation advances physics, and the current scene does not weld the racket to the hand, so the body, racket, and shuttlecock can move away from the inspection pose.
When you do want local simulation, add `--simulate`; keep `--free-simulate` off for the stabilized whole-scene preview.

## Tests

```bash
/data3/yangfeiyang/WorkSpace/ENV/musclemimic/.venv/bin/python3 \
  -m pytest environment/overall_environment/tests/test_overall_environment.py -q
```

## Notes

- The person and racket pose comes from `configs/right_hand_racket_grip_reference.json`.
- The racket is still modeled as a free body; the keyframe places it in the hand, but it is not welded to the hand.
- The shuttlecock is initialized cork-down on the floor using its `overall_shuttle_free` freejoint.

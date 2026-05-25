# Badminton Court MuJoCo Design Package

This package contains a BWF-standard badminton court design dossier and generated MuJoCo assets.

## Files

```text
badminton_court_design_dossier.md
assets/badminton_court_bwf_visual.xml
assets/badminton_court_bwf_collision_net.xml
params/court_bwf_nominal.json
src/court_geometry.py
src/generate_court_mjcf.py
src/validate_court_params.py
docs/validation_protocol.md
docs/codex_tasks.md
```

## Quick start

```bash
python src/generate_court_mjcf.py
python src/validate_court_params.py
```

## Coordinate system

```text
x: court length, net at x=0
y: court width, centre service line at y=0
z: up, court surface at z=0
```

Use `assets/badminton_court_bwf_visual.xml` by default. Use `assets/badminton_court_bwf_collision_net.xml` only when testing net contact.

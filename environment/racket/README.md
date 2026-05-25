# Badminton Racket MuJoCo Design Package

This package contains a MuJoCo-ready badminton racket design dossier and implementation scaffold.

Start here:

1. Read `badminton_racket_design_dossier.md`.
2. Inspect/tune `params/racket_nominal.json`.
3. Load `assets/badminton_racket_rigid.xml` for the stable rigid model.
4. Use `src/racket_stringbed.py` to apply a tunable string-bed proxy force to the shuttlecock.
5. Use `assets/badminton_racket_flex_proxy.xml` only when a qualitative shaft-flex response is needed.

Run the sanity checks:

```bash
python src/validate_racket_params.py
```

Regenerate MJCF assets after editing parameters:

```bash
python src/generate_racket_mjcf.py
```

Coordinate frame:

```text
origin = butt cap center
+X     = lateral across string bed
+Y     = butt-to-tip
+Z     = normal to string-bed plane
```

Recommended attachment/control site: `grip_pose_site` at `[0, 0.09, 0]`.

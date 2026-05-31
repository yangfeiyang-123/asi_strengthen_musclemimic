# Task 02 — Add grip annotation sites and create a hand+racket grip scene

Use the audit from Task 01. Goal: add non-colliding sites that make the correct right-hand badminton grip observable and optimizable.

## Required work

1. Create or modify a scene XML that includes both:
   - the musculoskeletal model,
   - the badminton racket model.

Preferred output path:

```text
assets/right_hand_racket_grip_scene.xml
```

If the repository has a different asset convention, use that convention and document it.

2. Add or ensure the racket has these sites:

```text
grip_pose_site              existing is OK; expected near local [0, 0.09, 0]
butt_site                   existing is OK
stringbed_center_site        existing is OK
head_tip_site                existing is OK
handle_axis_start_site       local [0, 0.02, 0]
handle_axis_end_site         local [0, 0.16, 0]
racket_face_normal_site      local [0, 0.09, 0.05]
```

All sites must be visual/annotation only and must not affect collision/mass.

3. Add or ensure the right hand has sites for:

```text
rh_palm_grip_site
rh_thumb_pad_site
rh_index_pad_site
rh_middle_pad_site
rh_ring_pad_site
rh_pinky_pad_site
```

Place them on the best available corresponding palm/finger bodies. If the hand model already has better fingertip/pad sites, create aliases/config entries rather than duplicating geometry. Site placement should be approximate but anatomically plausible.

4. Create `configs/right_hand_racket_grip_targets.json` if missing. Use this default target schema:

```json
{
  "handle_radius_m": 0.014,
  "contact_clearance_m": 0.0015,
  "target_points_racket_local": {
    "palm":   {"y": 0.085, "theta_deg": 180.0,  "weight": 1.5},
    "thumb":  {"y": 0.122, "theta_deg": 45.0,   "weight": 2.0},
    "index":  {"y": 0.125, "theta_deg": -45.0,  "weight": 2.0},
    "middle": {"y": 0.098, "theta_deg": -115.0, "weight": 1.6},
    "ring":   {"y": 0.075, "theta_deg": -135.0, "weight": 1.4},
    "pinky":  {"y": 0.055, "theta_deg": -150.0, "weight": 1.2}
  }
}
```

where `x = r*cos(theta)` and `z = r*sin(theta)` in racket local coordinates around the handle axis.

5. Add a small visualization/debug script:

```text
src/grip/visualize_grip_sites.py
```

It should load the scene, print all grip sites world positions, and optionally launch the MuJoCo viewer if available.

## Modeling guidance

- Do not weld the racket permanently to the hand in the main model. If you need a helper constraint, make it soft and controllable by a separate training scene or option.
- Keep handle collision enabled. Finger/hand collision geoms must interact with the handle.
- If the racket model uses `racket_handle` and `racket_head` split bodies, attach grip sites to the handle body.
- Sites are suitable for targets because MuJoCo sites do not participate in collisions or mass/inertia; use them for optimization and sensors, not contact.

## Validation

Add a script or test so this command passes:

```bash
python src/grip/visualize_grip_sites.py --xml assets/right_hand_racket_grip_scene.xml --no-viewer
```

It must fail loudly if any required site is absent.

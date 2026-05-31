# Strategy: making a MuJoCo musculoskeletal right hand hold a badminton racket

## Main design decision

Do not start from pure free-contact RL. First solve a static reference posture, then train dynamics around it.

Recommended pipeline:

1. Model audit: discover musculoskeletal right-hand body names, phalanx geoms, fingertip/pad sites, wrist/palm sites, actuator names, qpos addresses, and the racket body/site names.
2. Add grip annotation sites: add or map palm, thumb pad, index pad, middle pad, ring pad, pinky pad; add racket handle axis and grip target sites.
3. Static reference posture: solve an IK/optimization problem that puts the hand around the handle with plausible contact sites and no illegal penetrations.
4. Keyframe generation: save the resulting qpos/qvel and racket freejoint pose as a named keyframe.
5. Grip-hold training: train muscle/actuator controls to maintain the reference posture while the racket is held, first with a soft weld, then with contact and perturbations.
6. Validation: verify contact coverage, racket slip, joint-limit safety, muscle effort, and stability under perturbation/swing.

## Recommended grasp geometry

Racket local convention from the previous racket design:

- origin: butt cap center
- local +Y: from butt cap toward racket head
- local +Z: stringbed normal
- local +X: stringbed lateral axis
- `grip_pose_site`: approximately `[0, 0.09, 0]`

For a right-hand neutral/forehand grip, the handle should lie diagonally across the palm and pass through the thumb-index web. Use target sites on the handle cylinder rather than trying to prescribe every joint angle.

Default handle radius: `0.014 m`.

Suggested target locations in racket local coordinates:

```text
palm_target    y=0.085, theta=180 deg, weight=1.5
thumb_target   y=0.122, theta= 45 deg, weight=2.0
index_target   y=0.125, theta=-45 deg, weight=2.0
middle_target  y=0.098, theta=-115 deg, weight=1.6
ring_target    y=0.075, theta=-135 deg, weight=1.4
pinky_target   y=0.055, theta=-150 deg, weight=1.2
```

where `x = r*cos(theta)`, `z = r*sin(theta)` around the handle axis. Codex may mirror signs after visual validation if your hand/racket coordinate convention differs.

## Reward terms for training

A good grip is not just site matching. It should remain stable under perturbation and not depend on impossible penetrations.

Use these reward components:

- `r_pose`: distance from current hand qpos to solved reference qpos.
- `r_sites`: palm/finger pad sites close to handle target sites.
- `r_contact`: at least thumb, index, middle, ring/pinky or palm in contact with handle geoms.
- `r_slip`: small relative velocity between racket handle and palm/finger contacts.
- `r_racket_pose`: racket grip site remains near palm grip frame.
- `r_effort`: penalize excessive muscle activation / actuator effort.
- `r_limits`: penalize joint limit violations, unnatural hyperextension, and large penetration.
- `r_perturb`: bonus for keeping racket stable after external force/torque perturbation.

## Curriculum

Stage 0: fixed racket, pose IK only.
Stage 1: racket welded/soft-welded to palm frame, fingers learn posture.
Stage 2: soften weld, keep high handle friction, add small perturbations.
Stage 3: free racket with contact, perturb handle and racket head.
Stage 4: integrate arm/wrist swing motion while preserving grip.

Acceptance targets:

- Mean grip-site error under 2 cm after reset.
- Racket grip frame translation drift under 1 cm for 2 seconds with gravity.
- Racket orientation drift under 8 degrees for 2 seconds with gravity.
- At least 4 meaningful hand-handle contacts in the stable pose.
- No persistent non-handle self-collision or illegal object penetration.
- After a 2 N handle perturbation or 0.03 N·m torque perturbation, recover to under 2 cm / 12 degrees within 0.5 s.

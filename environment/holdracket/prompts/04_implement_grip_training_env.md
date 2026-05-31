# Task 04 — Implement right-hand racket grip training environment

Goal: create an RL/control environment that makes the musculoskeletal right hand maintain a correct badminton racket grip under gravity and perturbations.

## Required outputs

Create:

```text
src/grip/right_hand_racket_grip_env.py
src/grip/train_right_hand_racket_grip.py
src/grip/evaluate_right_hand_racket_grip.py
configs/right_hand_racket_grip_training.yaml
```

Use Gymnasium-style API if the repository already uses it. If not, implement a minimal `reset()` / `step(action)` environment that can be wrapped later.

## Environment design

Observation should include:

```text
right-hand qpos/qvel, normalized
right-hand actuator activations/controls if available
relative pose: racket grip_pose_site to rh_palm_grip_site
relative orientation: racket handle axis and face normal in palm frame
finger/palm site positions relative to handle target points
contact indicators for thumb/index/middle/ring/pinky/palm with handle geoms
racket linear/angular velocity
curriculum stage id
```

Action should be:

- right-hand muscle activations or actuator controls if the model is muscle actuated;
- right-hand motor target/torque controls if the model is motor actuated;
- never directly teleport qpos during `step` except in reset/curriculum helpers.

## Reward

Implement a modular reward dictionary and scalar sum:

```text
r_site_match      high when grip sites stay close to handle targets
r_racket_pose     high when grip_pose_site stays close to palm grip site
r_racket_orient   high when handle axis and face normal stay aligned to reference
r_contact         high when at least 4 meaningful handle contacts exist
r_no_slip         penalizes relative hand-racket slip at grip frame
r_reference_pose  penalizes deviation from IK reference qpos
r_effort          penalizes excessive controls / muscle activations
r_joint_limits    penalizes unsafe joint positions
r_no_penetration  penalizes illegal persistent penetration
r_perturb_stable  bonus for recovering after perturbations
```

Log the reward breakdown in `info` every step.

## Curriculum

Implement stages:

```text
stage 0: reference-pose tracking, racket fixed or very strongly soft-welded to palm/grip frame
stage 1: soft weld to palm/grip frame, gravity on, no perturbation
stage 2: weaker soft weld, high handle friction, small perturbation
stage 3: free racket with contact, perturb handle and racket head
stage 4: optional swing trajectory while maintaining grip
```

Do not hard-code a permanent weld in the final task. If using MuJoCo equality weld, make it optional and adjustable by XML variant or runtime `eq_active`/softness settings.

## Perturbations

Apply random perturbations after the grip has settled:

```text
force range: 0 to 2 N at handle or head frame
 torque range: 0 to 0.03 N*m
 duration: 0.05 to 0.15 s
```

Use MuJoCo external force arrays or existing repo utilities. Reset them after each perturbation.

## Training script

`train_right_hand_racket_grip.py` should:

1. load XML, target config, and reference state,
2. construct the env,
3. choose an existing training backend if the repo has one,
4. otherwise provide a simple baseline controller / random rollout / placeholder with clear TODOs,
5. save checkpoints and evaluation metrics.

Do not silently invent a dependency. If stable-baselines3, rl-games, brax, mjx, or another framework is already used in the repo, integrate with that. If not, keep the environment independent.

## Validation commands

These commands should run without crashing:

```bash
python src/grip/right_hand_racket_grip_env.py --xml assets/right_hand_racket_grip_scene.xml --smoke-test
python src/grip/evaluate_right_hand_racket_grip.py --episodes 3 --steps 1000
```

Acceptance:

- The environment loads and steps.
- The reference reset places the hand around the racket.
- Reward terms are finite.
- Contacts are counted and reported.
- Racket drift and orientation error are reported.

# Primitive physical-control producer

`musclemimic-synergy-produce-primitive` turns a non-ChinaJump retargeted
trajectory into a raw primitive trial by running physical muscle control on the
exact ChinaJump TaskFactory model. It never treats `qpos`/`qvel`, normalized
policy action, signed Paper_Need action, or EMG as control evidence.

Production invariants:

- the model comes from the resolved ChinaJump Hydra config through
  `TaskFactory`, or from its strictly verified content-addressed MJB artifact;
  direct `MyoFullBody(...)` construction is diagnostic-only;
- a `num_envs=1` construction is accepted only after its complete model hash
  matches a second construction at the config's declared production width;
- the exact runtime MJB and a content-addressed controller/optimizer manifest
  are saved together;
- the source NPZ SHA-256 and half-open source frame interval are immutable;
- because retargeted kinematics do not contain a muscle state, the trajectory
  optimizer samples the contact-constrained MuJoCo forward-acceleration response
  at the first state and computes a bounded activation seed for
  `(qvel[1]-qvel[0])/dt`; it sets `data.act=data.ctrl`, records the exact initial
  integration state, and takes no hidden warm-up physics steps;
- `act=ctrl` claims only activation-dynamics steady state, not mechanical
  equilibrium: the linearized solve and instantaneous forward acceleration are
  diagnostic, while an exact constant-control one-transition shadow rollout
  must pass tracking and target-contact QC before the real trial can succeed;
  P01 additionally requires exact bilateral support at every shadow physics
  substep, not merely at the source-frame endpoint;
- an explicit phase plan labels every cropped transition without gaps or
  overlaps; and
- target preflight runs every `qpos`/`qvel` state through the exact runtime
  `mj_forward`, records exact foot-floor contact and a separate ankle/toe-site
  hysteresis proxy, and applies the task phase state machine before rollout;
- every rollout transition records post-transition left/right exact contact and
  summed positive normal force from `mj_contactForce`; replay must reproduce
  both booleans exactly and both force sums within the configured tolerance;
- each controller step forms a bounded physical-`ctrl` proposal by finite-
  differencing the exact multi-substep MuJoCo transition from the current full
  integration state, solving its position/velocity shooting linearization, and
  retaining that proposal as the deterministic baseline for CEM refinement;
- P01 target and actual semantics use root free-joint `z/vz`; P05, P06, P07,
  P11 and P12 use root-subtree COM `z` and the transition-aligned COM velocity
  `delta(z)/(transition_substeps*timestep)` for both target and actual evidence;
- `success=true` requires complete closed-loop tracking, actual-rollout exact
  contact/phase semantics, and exact physical-control/contact forward replay. A
  target proxy or static inverse-dynamics residual can never pass final QC.

## CPU-only verified runtime reuse

By default every `preflight`, `produce`, and `import-policy` command constructs
the ChinaJump Warp `TaskFactory` runtime and proves the production-width model
hash. After that default path has created a production-eligible,
fingerprint-named controller artifact, the same commands may load its exact MJB
without constructing another Warp environment:

```bash
cd /data3/yangfeiyang/WorkSpace/musclemimic
source configs/env.sh
export VERIFIED_RUNTIME_ARTIFACT='datasets/_global/primitive_synergy/controllers/<64-hex-fingerprint>'

.venv/bin/musclemimic-synergy-produce-primitive preflight \
  --verified-runtime-artifact "$VERIFIED_RUNTIME_ARTIFACT" \
  --config-name config_specific_task/stage1_body/conf_fullbody_chinajump_root_control_v2 \
  --source-npz 'datasets/_global/muscle_trajectory/gmr_cache/MyoFullBody/gmr/Transitions_mocap/mazen_c3d/sit_stand_poses.npz' \
  --source-motion-path '_global/muscle_trajectory/gmr_cache/MyoFullBody/gmr/Transitions_mocap/mazen_c3d/sit_stand_poses' \
  --start-frame 780 \
  --end-frame-exclusive 810 \
  --phase-schema fullbody/config_specific_task/stage1_body/primitive_catalog/phase_schemas/P01_natural_stance_v1.json \
  --phase-plan fullbody/config_specific_task/stage1_body/primitive_catalog/phase_plans/P01_amass_sit_stand_frames_780_810_v1.json \
  --controller-store datasets/_global/primitive_synergy/controllers
```

The reuse loader is CPU-only. It verifies the directory fingerprint, strict
optimizer manifest, MJB SHA-256, complete model hash, `nu=na=354`, actuator
order, unit control ranges and immutable runtime binding. It then recomposes
the recorded `config_name` and ordered Hydra overrides without constructing an
environment, and rejects any current resolved-config or model-parameter drift.
Repeat every original `--hydra-override` in its original order. Reports and
rollout manifests identify whether the model was freshly constructed or reused
and record the source artifact hashes; the runtime binding v1 schema is
unchanged.

This is only a reproducible primitive-producer CPU replay entry point. It does
not validate current Warp/CUDA construction and is not a replacement for the
GPU preflight or canonical launcher required by a training launch/restart. When
the model, config, overrides, or runtime construction code changes, create a
new verified artifact through the default TaskFactory path.

## P01 stable-stance candidates

The checked-in P01 candidates are three independent AMASS transitions with
exact bilateral target contact. Use two motions for train and a different
motion for validation:

| split | motion | frames | plan |
|---|---|---:|---|
| train | `sit_stand_poses` | `[780,810)` | `P01_amass_sit_stand_frames_780_810_v1.json` |
| train | `dance_stand_poses` | `[760,800)` | `P01_amass_dance_stand_frames_760_800_v1.json` |
| val | `airkick_stand_poses` | `[580,620)` | `P01_amass_airkick_stand_frames_580_620_v1.json` |

The former forehand-clear P01 crops (`6月2日-2 [26,46)`, `6月2日-4
[51,71)`, and `6月2日-1 [15,35)`) are rejected candidates. Exact-MJB
`mj_forward` found right-foot contact on every transition but left-foot contact
on `0/20`; low root vertical speed does not satisfy P01's “both feet stable
support” contract. Do not restore those plans or relax P01 to make them pass.

Run preflight on an explicit physical GPU. The Warp TaskFactory needs a visible
CUDA device even though the producer subsequently uses the CPU `MjModel`:

```bash
cd /data3/yangfeiyang/WorkSpace/musclemimic
source configs/env.sh
export CUDA_VISIBLE_DEVICES=1

scripts/run_with_cuda_compat.sh uv run \
  musclemimic-synergy-produce-primitive preflight \
  --config-name config_specific_task/stage1_body/conf_fullbody_chinajump_root_control_v2 \
  --source-npz 'datasets/_global/muscle_trajectory/gmr_cache/MyoFullBody/gmr/Transitions_mocap/mazen_c3d/sit_stand_poses.npz' \
  --source-motion-path '_global/muscle_trajectory/gmr_cache/MyoFullBody/gmr/Transitions_mocap/mazen_c3d/sit_stand_poses' \
  --start-frame 780 \
  --end-frame-exclusive 810 \
  --phase-schema fullbody/config_specific_task/stage1_body/primitive_catalog/phase_schemas/P01_natural_stance_v1.json \
  --phase-plan 'fullbody/config_specific_task/stage1_body/primitive_catalog/phase_plans/P01_amass_sit_stand_frames_780_810_v1.json' \
  --controller-store datasets/_global/primitive_synergy/controllers
```

The current reviewed Warp preflight produces model hash
`e7a12d2f1a7a8640ff7591d97e2fd9f873211f80dc856d35cc49b313da072837`.
Any different hash must be investigated, not overridden.

Run the first contact-forward transition-shooting/CEM trial with explicit,
predeclared pilot gates:

```bash
scripts/run_with_cuda_compat.sh uv run \
  musclemimic-synergy-produce-primitive produce \
  --config-name config_specific_task/stage1_body/conf_fullbody_chinajump_root_control_v2 \
  --source-npz 'datasets/_global/muscle_trajectory/gmr_cache/MyoFullBody/gmr/Transitions_mocap/mazen_c3d/sit_stand_poses.npz' \
  --source-motion-path '_global/muscle_trajectory/gmr_cache/MyoFullBody/gmr/Transitions_mocap/mazen_c3d/sit_stand_poses' \
  --start-frame 780 \
  --end-frame-exclusive 810 \
  --phase-schema fullbody/config_specific_task/stage1_body/primitive_catalog/phase_schemas/P01_natural_stance_v1.json \
  --phase-plan 'fullbody/config_specific_task/stage1_body/primitive_catalog/phase_plans/P01_amass_sit_stand_frames_780_810_v1.json' \
  --controller-store datasets/_global/primitive_synergy/controllers \
  --output-dir datasets/_global/primitive_synergy/raw/P01_natural_stance/train_amass_sit_stand_780_810_seed0 \
  --trial-id P01-train-amass-sit-stand-780-810-seed0 \
  --seed 0 \
  --max-position-rmse 0.08 \
  --max-velocity-rmse 2.0 \
  --max-position-abs 0.35 \
  --max-velocity-abs 12.0 \
  --max-saturation-fraction 0.10
```

These are pilot gates, not evidence that the first optimizer attempt will pass.
The rollout manifest and QC NPZ bind the initialization contract, solver
diagnostics, initial activation/control hashes or arrays, integration-state
hash, instantaneous forward evidence, and exact shadow-transition evidence.
This prevents a zero-activation cold start or an unrecorded settling interval
from being mistaken for the first primitive transition.
The command exits with status 2 and writes `success=false` when any gate fails.
Repeat with the other two checked-in plans and different `--trial-id`/output
paths. Do not enable P01 in a catalog until both train trials and the val trial
independently pass and their three manifests have been reviewed.

## P11 decomposed-jump candidates

Three checked-in P11 phase plans pass target preflight on the reviewed exact
MJB. All three pass **only** through the ankle/toe site proxy:
`exact.passed=false`, `gate_basis=site_xpos_hysteresis_proxy`, and
`target_exact_contact_incomplete=true`. This permits a controller attempt; it
is not physical-rollout success evidence.

| candidate | motion | frames | phase plan | split limitation |
|---|---|---:|---|---|
| AMASS transition | `jumpinplace_push_poses` | `[0,242)` | `P11_amass_jumpinplace_push_frames_0_242_v1.json` | independent motion candidate |
| KIT 425 | `walking_07_poses` | `[0,140)` | `P11_amass_kit425_walking07_frames_0_140_v1.json` | motion-level only |
| KIT 425 | `walking_09_poses` | `[0,147)` | `P11_amass_kit425_walking09_frames_0_147_v1.json` | motion-level only |

The two KIT clips share subject `425`. They are distinct motion-level examples
but **not** a subject-held-out train/validation pair; do not label either one as
subject-held-out validation. Preflight the first candidate with:

```bash
scripts/run_with_cuda_compat.sh uv run \
  musclemimic-synergy-produce-primitive preflight \
  --config-name config_specific_task/stage1_body/conf_fullbody_chinajump_root_control_v2 \
  --source-npz 'datasets/_global/muscle_trajectory/gmr_cache/MyoFullBody/gmr/Transitions_mocap/mazen_c3d/jumpinplace_push_poses.npz' \
  --source-motion-path '_global/muscle_trajectory/gmr_cache/MyoFullBody/gmr/Transitions_mocap/mazen_c3d/jumpinplace_push_poses' \
  --start-frame 0 \
  --end-frame-exclusive 242 \
  --phase-schema fullbody/config_specific_task/stage1_body/primitive_catalog/phase_schemas/P11_decomposed_jump_v1.json \
  --phase-plan fullbody/config_specific_task/stage1_body/primitive_catalog/phase_plans/P11_amass_jumpinplace_push_frames_0_242_v1.json \
  --controller-store datasets/_global/primitive_synergy/controllers
```

The default command above requires the same sourced environment and explicit
physical GPU shown for P01. Once it has created a verified artifact, the
CPU-only reuse option may be used for the remaining plans with the exact same
config identity.

Each published rollout directory contains:

- `primitive_trial.npz`: exact transition `data.ctrl`, post-transition
  `data.act`, force/length/velocity, integer phase and boolean success;
- `rollout_qc.npz`: target/actual/replay states, target exact and site-proxy
  contacts, actual/replay exact left/right contact booleans and summed normal
  forces, explicit target/actual root and COM vertical position/velocity,
  exact physics-substep schedule, and initial `mjSTATE_INTEGRATION`; and
- `rollout_manifest.json`: source crop, model/config/controller identities,
  thresholds, metrics, gates and artifact hashes.

## Exact contact and phase hard QC

The runtime binding resolves geom names to IDs once and records both in
preflight and the rollout manifest. The floor must be exactly `floor`. The
left/right inventories must be exactly:

```text
l/r_talus
l/r_foot
l/r_foot_col1
l/r_foot_col3
l/r_foot_col4
l/r_bofoot
l/r_bofoot_col1
l/r_bofoot_col2
```

A missing, extra, duplicated, or name/ID-mismatched foot geom rejects
production. For every post-transition state the producer visits each active
MuJoCo contact, selects floor-versus-one-of-these-foot-geom pairs, calls
`mj_contactForce`, sums positive `force[0]` by side, and sets the side's contact
boolean when the sum reaches `--min-contact-normal-force` (default `1e-6 N`).
Replay must reproduce each boolean exactly and each force sum within
`--replay-contact-force-atol` (default `1e-10 N`).

Target preflight keeps two distinct evidence tracks:

- `exact`: the target state is loaded into the same runtime model and evaluated
  with `mj_forward` plus the normal-force rule above;
- `site_proxy`: per-site 1% stance baselines for the stable
  `left/right_ankle_mimic` and `left/right_toes_mimic` sites, with contact-enter
  at `+0.035 m` and contact-exit at `+0.045 m` (hysteresis).

P01 always requires target exact bilateral support. Dynamic targets P05, P06,
P07, P11 and P12 may pass preflight on strict proxy semantics when retargeted
states do not produce complete exact support forces; this is reported as
`target_exact_contact_incomplete=true` and is only permission to attempt the
controller. It is not success evidence. The independently generated actual
rollout and replay must still pass the exact normal-force state machine.

The supported task gates are fail-closed:

- P01: both feet remain in exact contact for the complete primitive and root
  vertical speed stays at or below `0.20 m/s`.
- P05: phase order is ready, descent, reversal, extension; both feet remain in
  contact throughout; ready has at least 5 frames with COM vertical speed at or
  below `0.15 m/s`; descent drops at least `0.03 m`; the reversal phase contains
  the unique low and one negative-to-positive crossing; extension rises at
  least `0.03 m`.
- P06: preload and propulsion start with bilateral support; the only permitted
  contact sequence is `both -> single (at most 5 frames) -> air`, with no
  release/recontact; toe-off contains the loss events and low flight has at
  least 2 all-air frames.
- P07: pre-contact has at least 2 all-air frames; first contact is in the
  initial-contact phase; left/right touchdown lag is at most 5 frames; after
  bilateral landing neither foot may lose contact; stabilization is bilateral
  for at least 10 frames and its COM vertical speed stays at or below
  `0.20 m/s`.
- P11: ready is bilateral for at least 5 frames at `|vz| <= 0.15 m/s`;
  countermovement remains bilateral and contains the unique low/reversal;
  propulsion follows the same monotonic `both -> <=5 single -> air` rule;
  flight has at least 2 all-air frames; controlled landing has at most 5 frames
  lag, no later contact loss, and ends with at least 10 bilateral frames at
  `|vz| <= 0.20 m/s`.
- P12: both feet remain in contact throughout, the first post-impact vertical
  speed is at most `0.20 m/s`, and ready-hold is bilateral for at least 10
  frames at `|vz| <= 0.15 m/s`.

Every phase also has at least 2 transitions and must occur exactly once in the
schema order. Other production `Pxx` task IDs are rejected until an explicit
semantic state machine is implemented. `P00_synthetic_fixture` is the only
exception: it is an explicit toy, skips semantic gates, records empty/real
contact arrays as applicable, and is always marked `production_eligible=false`.

## Existing full-354 policy route

An independently trained full-action policy can provide a better proposal, but
it must first be executed by the existing physical teacher collector. For a
promoted teacher, collect one crop (29 transitions here):

```bash
scripts/run_with_cuda_compat.sh uv run fullbody/distill_collect.py \
  --teacher_ckpt "$FULL354_CHECKPOINT" \
  --teacher_promotion_manifest "$TEACHER_PROMOTION_MANIFEST" \
  --output_dir datasets/_global/primitive_synergy/policy_capture/P01_trial1 \
  --num_envs 1 \
  --num_transitions 29 \
  --motion_path '_global/muscle_trajectory/gmr_cache/MyoFullBody/gmr/Transitions_mocap/mazen_c3d/sit_stand_poses' \
  --traj_start_step 780 \
  --deterministic_teacher \
  --teacher_action_target mean \
  --save_physical_muscle_state \
  --split train
```

Then CPU-forward the captured physical sequence on the exact ChinaJump model:

```bash
scripts/run_with_cuda_compat.sh uv run \
  musclemimic-synergy-produce-primitive import-policy \
  --config-name config_specific_task/stage1_body/conf_fullbody_chinajump_root_control_v2 \
  --source-npz 'datasets/_global/muscle_trajectory/gmr_cache/MyoFullBody/gmr/Transitions_mocap/mazen_c3d/sit_stand_poses.npz' \
  --source-motion-path '_global/muscle_trajectory/gmr_cache/MyoFullBody/gmr/Transitions_mocap/mazen_c3d/sit_stand_poses' \
  --start-frame 780 \
  --end-frame-exclusive 810 \
  --phase-schema fullbody/config_specific_task/stage1_body/primitive_catalog/phase_schemas/P01_natural_stance_v1.json \
  --phase-plan 'fullbody/config_specific_task/stage1_body/primitive_catalog/phase_plans/P01_amass_sit_stand_frames_780_810_v1.json' \
  --physical-rollout-shard datasets/_global/primitive_synergy/policy_capture/P01_trial1/train_000000.npz \
  --physical-rollout-metadata datasets/_global/primitive_synergy/policy_capture/P01_trial1/metadata.json \
  --teacher-checkpoint "$FULL354_CHECKPOINT" \
  --controller-store datasets/_global/primitive_synergy/controllers \
  --output-dir datasets/_global/primitive_synergy/raw/P01_natural_stance/policy_train_amass_sit_stand \
  --trial-id P01-policy-train-amass-sit-stand \
  --seed 0 \
  --max-position-rmse 0.08 \
  --max-velocity-rmse 2.0 \
  --max-position-abs 0.35 \
  --max-velocity-abs 12.0 \
  --max-saturation-fraction 0.10
```

If a shard contains multiple rollout IDs, add `--rollout-uid ID`. The importer
verifies the checkpoint directory inventory, exact 354 actuator order and
control range, frame coordinates, stable motion UID and optional collected
phase IDs. It rejects shards with only normalized action and any physical
control outside `[0,1]`.

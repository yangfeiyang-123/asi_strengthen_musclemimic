# PostTrain Evaluation Protocol Design

## Context

ForehandNetLift PostTrain runs have repeatedly shown the same pattern: root/site-related signals can improve while full-body stability and completion degrade. The current training-time validation is hard to interpret because deterministic policy evaluation does not necessarily mean every rollout starts at frame 0, and validation videos can sample training trajectories rather than the validation split used for metrics.

This design focuses only on evaluation protocol. It does not change the training algorithm, reward implementation, environment dynamics, or model architecture.

## Goal

Build a clean, repeatable protocol for judging whether a PostTrain checkpoint is actually better than the baseline on complete badminton actions.

The protocol must answer:

- Can the policy start from frame 0 and complete the full action?
- Does PostTrain improve over the baseline on the same motion under the same evaluation settings?
- Are reward improvements hiding worse posture, velocity, or early termination?

## Non-Goals

- Do not redesign PPO.
- Do not change reward formulas.
- Do not add KL anchors, behavior cloning, freezing, or network changes yet.
- Do not delete existing checkpoint or video results.
- Do not treat training-time reward as the main selection criterion.

## Recommended Approach

Use a two-layer validation protocol.

### Layer 1: Training-Time Validation Alignment

Add support for generated PostTrain configs to set:

```yaml
experiment:
  validation:
    start_from_beginning: true
```

The PostTrain spec should expose this as:

```yaml
training:
  validation_start_from_beginning: true
```

The runner should copy this into the generated Hydra config. This only changes validation reset behavior. It does not change the training rollout distribution.

### Layer 2: Offline Checkpoint Comparison

For each checkpoint selected for inspection, run deterministic offline evaluation from the beginning of each motion.

The fixed motion groups are:

```text
train-seen:
  ForehandNetLift/best/video05_best_stage7_smpl
  ForehandNetLift/best/video06_best_stage7_smpl
  ForehandNetLift/best/video07_best_stage5_smpl
  ForehandNetLift/best/video08_best_stage5_smpl

heldout-validation:
  ForehandNetLift/best/video01_best_stage7_smpl
  ForehandNetLift/best/video03_best_stage7_smpl

stress-test:
  ForehandNetLift/best/video04_best_stage7_smpl
  ForehandNetLift/best/video09_best_stage7_smpl
```

Each motion should be evaluated separately for both:

- baseline: `checkpoints/ForehandNetLift/forehand_net_lift_best_asi_curriculum/checkpoint_7812`
- PostTrain candidate: latest or selected checkpoint under `outputs/posttrain/ForehandNetLift/v1/checkpoints/<arm>/...`

The offline evaluation must use:

```text
deterministic policy
start_from_beginning
same n_steps
same terminal settings
same motion path
```

## Metrics

For each motion and checkpoint, record:

- `mean_episode_return`
- `early_termination_rate`
- `frame_coverage`
- `err_joint_pos`
- `err_joint_vel`
- `err_root_xyz`
- `err_root_yaw`
- `err_rpos`
- `err_site_abs`
- `reward_total`

Derived comparison columns:

- PostTrain minus baseline for every error metric
- PostTrain minus baseline for return
- pass/fail status against the gates below

## Selection Gates

A PostTrain checkpoint is not eligible as an improved model unless it passes all hard gates:

```text
early_termination_rate == 0
frame_coverage >= 0.95
mean_episode_return >= baseline_return
err_joint_vel <= baseline_joint_vel + tolerance
err_rpos <= baseline_rpos + tolerance
```

Recommended tolerances:

```text
joint_vel tolerance: 0.10 absolute
rpos tolerance: 0.01 absolute
```

Root/site improvements should be treated as secondary. A checkpoint that improves `root_xyz` or `site_abs` but fails completion is not considered better.

## Video Protocol

Only generate side-by-side videos for checkpoints that either pass the hard gates or are needed for debugging a failure mode.

Video layout:

```text
left: baseline
right: PostTrain candidate
```

Priority motions:

```text
video07_best_stage5_smpl
video01_best_stage7_smpl
video03_best_stage7_smpl
```

The rendered command must include `--start_from_beginning`.

## Expected Outputs

For each evaluation batch:

```text
outputs/posttrain/ForehandNetLift/v1/metrics/<run_name>/metrics_table.csv
outputs/posttrain/ForehandNetLift/v1/metrics/<run_name>/metrics_delta.csv
outputs/posttrain/ForehandNetLift/v1/metrics/<run_name>/comparison_report.md
outputs/posttrain/ForehandNetLift/v1/videos/<run_name>/*.mp4
```

The report should clearly separate:

- train-seen results
- heldout-validation results
- stress-test results
- final recommendation: keep baseline, keep PostTrain, or continue debugging

## Error Handling

- If no PostTrain checkpoint exists, report that training has not reached a saved checkpoint.
- If multiple config-hash folders exist under an arm, select the latest checkpoint by numeric checkpoint suffix unless a checkpoint path is explicitly provided.
- If a motion fails to load, mark that motion failed and continue evaluating the remaining motions.
- If video rendering fails, keep the metrics report and mark video generation as failed.

## Test Plan

Minimum tests:

- Unit test that `validation_start_from_beginning: true` is copied from the spec into the generated Hydra config.
- Unit test that latest checkpoint discovery still works under nested config-hash folders.
- Dry-run command generation for baseline and PostTrain evaluation.

Manual verification:

- Run one baseline and one PostTrain offline evaluation on `video07_best_stage5_smpl`.
- Confirm the resulting report includes `frame_coverage` and `early_termination_rate`.
- Confirm the rendered video starts at the first frame of the action.

## Implementation Boundaries

Allowed changes after approval:

- Extend `musclemimic/badminton/scripts/run_posttrain_experiment.py` to pass `validation_start_from_beginning`.
- Add or extend scripts for offline checkpoint comparison.
- Add focused unit tests for config generation and checkpoint discovery.
- Add documentation for the exact evaluation commands.

Not allowed without separate approval:

- Reward function changes.
- PPO algorithm changes.
- Terminal handler changes.
- Deleting checkpoints, videos, or training logs.
- Changing training data again.

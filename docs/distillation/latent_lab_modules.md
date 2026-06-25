# Latent Muscle Skill + LAB Modules

This note maps `doc/DistillationAndLAB.md` onto the current repository.

## Why This Fits The Existing Code

The repository already has:

- teacher rollout collection in `musclemimic/distill/collect_teacher.py`
- offline BC and DAgger-compatible distillation shards in `musclemimic/distill/`
- PPO-compatible student checkpoints
- a layered body/grip action merge in `environment/overall_environment/src/layered_policy.py`

The missing part was the LATENT-style controller structure:

```text
posterior q(z | state, reference)
prior     p(z | state)
decoder   D(state, z) -> body action
LAB       z = mu + lambda * sigma * tanh(raw_latent)
```

The new `musclemimic.latent_muscle` package provides that structure without
replacing the existing teacher, BC, or DAgger pipeline.

## Added API

```python
from musclemimic.latent_muscle import (
    ActionMask,
    ConditionalPrior,
    LABActionWrapper,
    LatentDecoder,
    PosteriorEncoder,
    latent_distillation_loss,
)
```

Use `PosteriorEncoder`, `ConditionalPrior`, and `LatentDecoder` during latent
distillation. Use only `ConditionalPrior` and `LatentDecoder` at deployment or
high-level PPO time.

`LatentDecoder(bounded_action=True)` emits symmetric muscle actions in
`[-1, 1]`, matching the MyoFullBody `DefaultControl` action convention.

`LABActionWrapper` is the runtime transform for a high-level latent residual:

```python
body_action = lab_wrapper(state, raw_latent)
```

`ActionMask` keeps body decoder actuators separate from distal correction
actuators such as right wrist, hand, grip, and racket residual controls.

## Layered Policy Integration

`environment.overall_environment.src.layered_policy.LatentBodyPolicy` adapts a
high-level latent policy into the existing layered controller. If the high-level
observation contains task fields beyond the body state, pass a `state_adapter`
so LAB receives only the state representation used to train the prior/decoder:

```text
high-level policy -> raw_latent
state_adapter     -> state
LAB wrapper       -> body_action
grip policy       -> correction_action
LayeredPolicy     -> full actuator action
```

This preserves the current correction path and avoids forcing unreliable
right-wrist or grip tracking into the body latent skill space.

## Current Scope

This implementation adds the reusable math, networks, action interfaces, and
collector plumbing needed to write `reference_features` shards. It does not run
a full latent training job by itself. The next training step should collect
teacher/DAgger shards with `--save-reference-features`, then optimize
`latent_distillation_loss`.

## Known Integration Gaps

### reference_features collection

`PosteriorEncoder` requires `(state, reference_features)` as input. The dataset
writer/loader preserves a `reference_features` array and records
`reference_features_dim` in `metadata.json` when that field is present.
Teacher and DAgger collectors now support:

```bash
--save-reference-features
```

By default this saves the goal lookahead features dropped from the student
observation and excludes the motion phase, because phase is already appended to
`student_obs`. Add `--include-reference-phase` only when a latent experiment
explicitly wants phase duplicated in the posterior reference tensor.

Legacy shards without `reference_features` remain valid for direct BC/DAgger
student training, but they are insufficient for latent posterior training.

### ActionMask / LayeredActuatorRouter alignment

`ActionMask` (latent_muscle) and `LayeredActuatorRouter` (layered_control) both
partition actuators into body vs distal sets. Build masks from the runtime
router with:

```python
mask = ActionMask.from_layered_router(router)
```

or call `mask.assert_matches_partitions(...)` when a mask is loaded from disk.

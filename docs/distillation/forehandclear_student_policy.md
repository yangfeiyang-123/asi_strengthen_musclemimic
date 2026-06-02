# ForehandClear Student Policy Distillation

This distillation path converts a trajectory-conditioned ForehandClear body teacher into a student policy that does not consume future goal lookahead.

Student v1 input:

```text
joint state + muscle state + foot contact + motion phase
```

The environment still loads the reference trajectory and still computes `MimicReward` from the reference. Only the policy observation is filtered. The filter keeps every non-`goal` observation and appends the last element of the original `goal` group as phase.

Implementation points:

```text
musclemimic/distill/obs_filter.py
musclemimic/distill/dataset.py
musclemimic/distill/collect_teacher.py
musclemimic/distill/train_bc.py
fullbody/config_specific_task/distill/
```

The checkpoint remains PPO-compatible because the BC trainer builds the same `ActorCritic` module and saves through the existing Orbax checkpoint manager.

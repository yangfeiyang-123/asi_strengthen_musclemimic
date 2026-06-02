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
musclemimic/distill/dagger.py
musclemimic/distill/dagger_loop.py
musclemimic/distill/eval_student.py
fullbody/config_specific_task/distill/
fullbody/config_specific_task/conf_fullbody_badminton_student_gmr.yaml
```

The checkpoint remains PPO-compatible because the BC trainer builds the same `ActorCritic` module and saves through the existing Orbax checkpoint manager.

Current stages:

```text
1. Off-policy BC/KD
   teacher full-lookahead rollout -> student_obs + teacher mean action shards.

2. DAgger-style correction
   student rollout -> teacher relabels student-visited full states -> append shards.

3. Student PPO fine-tune
   student policy input remains state + phase; MimicReward still uses reference trajectory.
```

The DAgger collector is intentionally offline: it writes relabeled shards, then
the same BC/KD trainer is rerun on the aggregated dataset.

Student v1 constraints:

```text
len_obs_history: 1
split_goal: false
student_obs_filter.drop_goal_lookahead: true
student_obs_filter.keep_motion_phase: true
```

The BC trainer validates `dataset.student_obs_dim` against the configured
wrapped student environment before training. A mismatch fails before checkpoint
creation.

End-to-end workflow:

```text
1. fullbody/distill_collect.py
   collect lookahead teacher rollout shards.

2. fullbody/distill_train_bc.py
   train student_0 from off-policy shards.

3. fullbody/distill_run_dagger.py
   iteratively collect student-visited states, teacher relabel them, append
   shards, and retrain student_{k+1}.

4. fullbody/experiment.py with
   config_specific_task/conf_fullbody_badminton_student_gmr
   PPO fine-tune from the BC or BC+DAgger checkpoint while keeping policy input
   at state + phase and reward at MimicReward.

5. fullbody/distill_compare.py
   evaluate teacher, BC, BC+DAgger, and PPO-finetuned students and write
   JSON/CSV/Markdown report artifacts.
```

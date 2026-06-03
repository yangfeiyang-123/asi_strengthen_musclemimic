# Distillation Results Template

| Policy | Input | Return Up | Early Term Down | err_rpos Down | err_joint_pos Down | Completion Up |
|---|---|---:|---:|---:|---:|---:|
| Teacher PPO | state + lookahead | | | | | |
| Student BC | state + phase | | | | | |
| Student BC+DAgger | state + phase | | | | | |
| Student BC+PPO | state + phase | | | | | |
| Student no phase | state only | | | | | |

Required checks:

```text
student_obs_dim matches checkpoint network input
phase range is [0, 1]
teacher_action is actor mean unless intentionally testing sampled targets
student checkpoint loads in fullbody/eval.py
reward info still includes MimicReward terms
DAgger shards use student rollout states with teacher mean labels
dataset metadata schema_version is distill_v1
BC trainer rejects dataset/env student_obs_dim mismatches before training
teacher collection metadata collector_obs_mode is teacher_full_obs
collection metadata freeze_run_stats is recorded
DAgger shards include rollout_action and used_teacher_action
```

Initial acceptance signals:

```text
Student rollout completion_rate >= 80% of teacher on ForehandClear clips
Student BC mean_episode_return >= 70% of teacher before PPO fine-tune
Student PPO mean_episode_return >= 85% of teacher after fine-tune
Student PPO early_termination_rate <= teacher + 0.20
err_site_abs / err_rpos remain within the selected tracking gap
```

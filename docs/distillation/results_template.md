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
```

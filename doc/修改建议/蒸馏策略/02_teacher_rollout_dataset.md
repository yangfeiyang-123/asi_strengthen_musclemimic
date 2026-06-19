# 02 Teacher Rollout Dataset 采集设计

## 2.1 目标

用已经训练好的 lookahead teacher checkpoint 在环境中 rollout，采集 student 训练数据：

```text
student_obs_t -> teacher_action_t
```

teacher 仍然看到 full lookahead，但保存给 student 的输入只包含：

```text
joint + muscle + foot contact + motion phase
```

---

## 2.2 新增脚本建议

新增入口：

```text
fullbody/distill_collect.py
```

内部调用模块：

```text
musclemimic/distill/collect_teacher.py
musclemimic/distill/obs_filter.py
musclemimic/distill/dataset.py
```

命令示例：

```bash
uv run python fullbody/distill_collect.py \
  --teacher_ckpt /path/to/teacher/checkpoint \
  --config_name conf_fullbody_badminton_gmr \
  --output_dir datasets/distill/forehandclear_teacher_v1 \
  --num_envs 256 \
  --num_steps 200000 \
  --deterministic_teacher \
  --student_obs_mode phase_only_goal \
  --seed 0
```

---

## 2.3 Teacher policy 输出建议

默认保存 teacher actor mean，而不是 stochastic sampled action。

```text
teacher_action = mean(π_teacher(. | full_obs_t))
```

原因：

```text
1. BC 目标更干净；
2. 降低 teacher sampling noise；
3. student 更容易复现 teacher 的主动作模式。
```

可选同时保存：

```text
sampled_action
teacher_mu
teacher_log_std
teacher_log_prob
teacher_value
```

如果以后要做 KL distillation，需要 `teacher_mu` 和 `teacher_log_std`。

---

## 2.4 每条 transition 保存字段

建议 shard `.npz` 字段：

```text
student_obs          float32 [N, student_obs_dim]
teacher_action       float32 [N, action_dim]
teacher_mu           float32 [N, action_dim] optional
teacher_log_std      float32 [action_dim] or [N, action_dim] optional
teacher_value        float32 [N]
teacher_log_prob     float32 [N]
reward               float32 [N]
done                 bool    [N]
absorbing            bool    [N]
traj_no              int32   [N]
subtraj_step_no      int32   [N]
phase                float32 [N]
qpos                 float32 [N, nq] optional
qvel                 float32 [N, nv] optional
tracking_error       float32 [N] optional, if available from info
```

强烈建议默认保存：

```text
student_obs
teacher_action
teacher_value
reward
done
traj_no
subtraj_step_no
phase
```

不要默认保存完整 `full_obs`，因为会很大；可以加 `--save_full_obs` debug 开关。

---

## 2.5 数据采集循环

建议实现 JAX/MJX 并行采集，而不是 MuJoCo 单环境采集。

伪代码：

```python
config, teacher_state, metadata = load_checkpoint(teacher_ckpt)
base_env = instantiate env from config / overrides
teacher_env = wrap_env(base_env, teacher_config.experiment)

student_spec = build_student_obs_indices(base_env, student_filter_config)

obs, env_state = teacher_env.reset(keys)

for chunk in range(num_chunks):
    batch = scan_collect_teacher(
        teacher_state,
        env_state,
        obs,
        rng,
        num_steps_per_chunk,
    )

    # batch.obs is teacher/full observation in policy input space
    student_obs = filter_student_obs(batch.obs, student_spec)
    teacher_action = batch.action_mean or batch.action

    write_npz_shard(...)
```

如果复用现有 PPO rollout 逻辑，注意当前 `_collect_trajectories()` 会 sample action 并存 action/value/log_prob，但不一定返回 actor mean。第一版可以复制其核心逻辑并加 `deterministic_teacher` 分支。

---

## 2.6 与现有 `run_with_trajectory_export` 的关系

仓库已有 `fullbody/eval.py --export_trajectory`，内部 `run_with_trajectory_export()` 会保存 policy actions、muscle commands、qpos/qvel 等。

但是它目前不适合作为 distillation dataset 的唯一来源，原因：

```text
1. 默认注释掉 episode_observations；
2. 没有 student_obs；
3. 没有 teacher_mu/log_std；
4. MuJoCo path 更偏可视化/导出，不适合大规模并行采样。
```

可以参考它的数据导出结构，但应新建专门的 distill collector。

---

## 2.7 Trajectory 覆盖策略

为了让 student 覆盖完整动作轨迹，采集时建议：

```text
1. 覆盖所有 ForehandClear clips；
2. 既有 random_start，也有 start_from_beginning；
3. 保证 phase 从 0 到 1 分布均匀；
4. 保存 traj_no/subtraj_step_no，便于按轨迹划分 train/val。
```

推荐策略：

```text
70% random_start rollouts
30% start_from_beginning rollouts
```

如果 teacher 在某些阶段 tracking 很差，可以后续用质量过滤剔除。

---

## 2.8 数据质量过滤

建议第一版提供可选过滤：

```text
--min_episode_return
--max_early_termination_rate
--drop_done_transition
--max_tracking_error
```

默认可以不过滤，只记录统计。

后续可做两个 dataset：

```text
raw: 所有 transition
clean: 去掉 early termination 前后异常段
```

---

## 2.9 Sharding 与 metadata

每个 shard 建议：

```text
shard_000000.npz
shard_000001.npz
...
metadata.json
```

`metadata.json` 包含：

```json
{
  "teacher_ckpt": "...",
  "teacher_config_hash": "...",
  "student_obs_mode": "state_plus_phase",
  "student_obs_dim": 1950,
  "action_dim": 354,
  "num_samples": 200000,
  "num_envs": 256,
  "control_dt": 0.01,
  "motion_paths": [...],
  "fields": [...],
  "obs_filter": {...}
}
```

---

## 2.10 测试

新增：

```text
tests/unit/test_distill_dataset.py
```

测试：

1. shard 写入/读取字段一致；
2. `student_obs.shape[0] == teacher_action.shape[0]`；
3. `student_obs_dim` 与 metadata 一致；
4. `phase` 在 `[0, 1]`；
5. action 维度等于环境 action space；
6. dataset loader 可以跨多个 shard 迭代 batch。

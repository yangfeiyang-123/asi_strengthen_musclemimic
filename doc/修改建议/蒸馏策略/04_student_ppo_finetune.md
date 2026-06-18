# 04 Student PPO Fine-tuning 设计

## 4.1 目标

BC student 初始能模仿 teacher action，但 rollout 时可能出现 distribution shift。因此需要用 PPO 在 student observation 环境中 fine-tune。

关键原则：

```text
student policy input 不含 future lookahead
MimicReward 仍使用 reference trajectory
```

---

## 4.2 Fine-tune 配置

新增配置建议：

```text
fullbody/config_specific_task/distill/conf_fullbody_forehandclear_student_phase_ppo.yaml
```

示例：

```yaml
# @package _global_

defaults:
  - /config_specific_task/conf_fullbody_badminton_gmr
  - _self_

wandb:
  tags: ["fullbody", "badminton", "forehand_clear", "distill", "student_phase_ppo"]

experiment:
  resume_from: /path/to/student_bc_checkpoint
  reset_std_on_resume: 1.0
  reset_lr_schedule_on_resume: true

  student_obs_filter:
    enabled: true
    drop_goal_lookahead: true
    keep_motion_phase: true
    require_goal_group: true
    require_motion_phase: true

  total_timesteps: 102400000

  ppo_config:
    num_steps: 80
    update_epochs: 1
    num_minibatches: 128
    gamma: 0.99
    gae_lambda: 0.95
    clip_eps: 0.2
    clip_eps_vf: 0.2
    init_std: 1.0
    learnable_std: true
    ent_coef: 0.0
    vf_coef: 0.5
```

根据稳定性可调：

```yaml
experiment:
  lr: 1e-4
  max_grad_norm: 0.5
```

---

## 4.3 Fine-tune 环境

需要确保 `wrap_env()` 按顺序应用：

```text
base env with reference trajectory and full MimicReward
 -> StudentObservationFilterWrapper
 -> NStepWrapper(optional)
 -> VecEnv / LogWrapper / AutoResetWrapper
 -> NormalizeVecReward(optional)
```

这样：

```text
policy sees: student_obs
reward sees: full env state + reference trajectory
```

---

## 4.4 Fine-tune 的 Actor-Critic

第一版：actor 和 critic 都只看 student_obs。

优点：

```text
1. 最终 checkpoint 真正不需要 future lookahead；
2. inference 与 training 一致；
3. checkpoint 兼容性简单。
```

不建议第一版做 privileged critic。

---

## 4.5 Resume BC checkpoint 注意事项

BC checkpoint 必须与 student PPO config 的 observation/action shape 一致。

Codex 需要保证：

```text
student BC config.student_obs_filter == student PPO config.student_obs_filter
```

如果不一致，应在 resume 阶段报错，而不是 silent failure。

---

## 4.6 Fine-tune 流程

```bash
uv run fullbody/experiment.py \
  --config-name=config_specific_task/distill/conf_fullbody_forehandclear_student_phase_ppo \
  experiment.resume_from=/path/to/student_bc_checkpoint \
  wandb.mode=online
```

如果用 Hydra 路径不对，Codex 需要根据仓库 config path 调整。

---

## 4.7 PPO 训练时 GAE 的作用

Fine-tune 时 PPO 不再依赖 teacher action 标签，而是使用 environment reward：

```text
rollout -> reward/value/done -> GAE -> PPO update
```

GAE 输出：

```text
A_t: actor loss 使用
R_t: critic value loss 使用
```

这一步让 student 适应自己 rollout 时遇到的状态，而不仅仅是 teacher 访问过的状态。

---

## 4.8 稳定性建议

如果 student fine-tune 初期摔倒严重：

1. 降低 learning rate：`lr=1e-4` 或 `5e-5`。
2. 增大 entropy 不一定有益；先保持 `ent_coef=0.0`。
3. 从 `start_from_beginning: true` 做短训练，再恢复 random start。
4. 临时放宽 termination threshold。
5. 使用 reward curriculum 或 adaptive termination，但先不要引入太多变量。
6. 短期保留更多信息：例如 `student_obs_filter.keep_small_goal_anchor=true`，只保留 step0 site_rpos；等稳定后再去掉。

---

## 4.9 两阶段 fine-tune 建议

### Stage A: phase student stabilization

```text
input = body state + foot contact + phase
reward = full MimicReward
termination = slightly relaxed
```

目标：不摔倒，完成整段动作相位推进。

### Stage B: strict tracking fine-tune

```text
input = body state + foot contact + phase
reward = full MimicReward
termination = normal training/eval threshold
```

目标：tracking error 接近 teacher。

---

## 4.10 可选：DAgger-style iterative distillation

如果 BC student rollout 偏离 teacher 很多，可以做：

```text
1. student rollout -> collect states
2. 在这些 states 上查询 teacher action
3. append dataset
4. 重新 BC
5. 再 PPO fine-tune
```

但第一版先不要做 DAgger，避免系统过复杂。

---

## 4.11 测试

新增：

```text
tests/integration/test_student_ppo_config.py
```

测试：

1. student PPO config 可以 instantiate env；
2. env observation dim 与 checkpoint network 输入一致；
3. env reset + one policy action + step 不报错；
4. reward info 中仍包含 `reward_qpos/reward_qvel/reward_rpos` 等；
5. `student_obs_filter` 打开后，policy obs dim 小于 teacher obs dim；
6. `goal lookahead` 被删除，phase 被保留。

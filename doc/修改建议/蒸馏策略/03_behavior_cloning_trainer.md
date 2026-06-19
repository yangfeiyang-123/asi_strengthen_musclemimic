# 03 Behavior Cloning / Policy Distillation Trainer

## 3.1 目标

用 teacher rollout dataset 训练 student actor-critic checkpoint。

第一版训练目标：

```text
student_obs_t -> teacher_action_t
```

其中：

```text
student_obs_t = joint + muscle + foot contact + phase
teacher_action_t = teacher actor mean or teacher sampled action
```

---

## 3.2 新增入口

建议新增：

```text
fullbody/distill_train_bc.py
```

内部模块：

```text
musclemimic/distill/train_bc.py
musclemimic/distill/losses.py
musclemimic/distill/dataset.py
musclemimic/distill/checkpoint.py
```

命令示例：

```bash
uv run python fullbody/distill_train_bc.py \
  --dataset_dir datasets/distill/forehandclear_teacher_v1 \
  --student_config fullbody/config_specific_task/distill/conf_fullbody_forehandclear_student_phase_bc.yaml \
  --output_dir outputs/distill/forehandclear_student_phase_bc \
  --batch_size 4096 \
  --num_steps 200000 \
  --lr 3e-4 \
  --seed 0
```

---

## 3.3 Student config

新增配置建议：

```text
fullbody/config_specific_task/distill/conf_fullbody_forehandclear_student_phase_bc.yaml
```

内容应继承 ForehandClear GMR config，但启用 student obs filter：

```yaml
# @package _global_

defaults:
  - /config_specific_task/conf_fullbody_badminton_gmr
  - _self_

wandb:
  tags: ["fullbody", "badminton", "forehand_clear", "distill", "student_phase_bc"]

experiment:
  student_obs_filter:
    enabled: true
    drop_goal_lookahead: true
    keep_motion_phase: true
    require_goal_group: true
    require_motion_phase: true

  # BC trainer can reuse network architecture from PPO config.
  algorithm: PPOJax
  normalize_env: true
```

注意：BC config 的 observation shape 必须与 dataset 中的 `student_obs_dim` 一致。

---

## 3.4 Network 创建与 checkpoint 兼容性

为了让 BC 训练出的 checkpoint 能被 `fullbody/eval.py` 和 PPO fine-tune 直接加载，BC trainer 应复用 PPO 的 agent 类型：

```python
env = instantiate_env(student_config)
env = StudentObservationFilterWrapper(env, ...)
agent_conf = PPOJax.init_agent_conf(env, student_config)
train_state = TrainState.create(
    apply_fn=agent_conf.network.apply,
    params=network_params,
    run_stats=network_run_stats,
    tx=agent_conf.tx,
)
agent_state = PPOAgentState(train_state=train_state)
```

保存 checkpoint 时使用现有 checkpoint manager 格式，确保：

```text
fullbody/eval.py --path <student_ckpt>
```

能够恢复 student config、创建 student env、加载 student params。

---

## 3.5 Loss 设计

### 必做：action imitation loss

```text
L_action = mean(||mu_student(student_obs) - teacher_action||^2)
```

如果 action 维度是 354，建议对每维平均：

```python
loss_action = jnp.mean(jnp.square(student_mu - teacher_action))
```

### 推荐：value distillation loss

如果 dataset 存了 `teacher_value`：

```text
L_value_distill = mean((V_student(student_obs) - V_teacher)^2)
```

### 可选：Gaussian KL distillation

如果 dataset 存了 teacher mean/std：

```text
L_KL = KL(π_teacher(.|teacher_obs) || π_student(.|student_obs))
```

第一版可以只做 action MSE + value distillation：

```text
L_BC = L_action + α_value L_value_distill
```

推荐初始：

```yaml
loss:
  action_mse_weight: 1.0
  value_distill_weight: 0.1
  kl_weight: 0.0
```

---

## 3.6 Student action 输出

`ActorCritic` 输出 distribution `pi`。BC loss 应使用 actor mean，而不是 sample：

```python
pi, value = network.apply(...)
student_mu = pi.mean()  # or pi.mode(), depending on distrax API
```

Codex 实施时需要检查 `distrax.MultivariateNormalDiag` 的 API。建议写兼容函数：

```python
def distribution_mean(pi):
    if callable(getattr(pi, "mean", None)):
        return pi.mean()
    if hasattr(pi, "mean"):
        return pi.mean
    if callable(getattr(pi, "mode", None)):
        return pi.mode()
    raise TypeError("Unsupported distribution mean API")
```

---

## 3.7 Observation normalization / RunningMeanStd

当前 `ActorCritic` 内部使用 `RunningMeanStd` 归一化 observation。

BC trainer 前向时需要允许更新 `run_stats`：

```python
y, updates = network.apply(
    {"params": params, "run_stats": run_stats},
    student_obs_batch,
    mutable=["run_stats"],
)
```

训练时更新 `run_stats`；验证时不更新或固定。

注意：如果 BC 训练完后用 PPO fine-tune，student checkpoint 的 `run_stats` 必须和 student obs 分布一致。

---

## 3.8 Data loader

实现：

```python
class DistillDataset:
    def __init__(dataset_dir, split="train", shuffle=True, seed=0): ...
    def iter_batches(batch_size): ...
```

建议第一版用 NumPy mmap/np.load 加载 shard，转 JAX batch。

支持：

```text
train split
val split
shuffle
repeat
```

如果数据量不大，可以先全部载入内存；但接口要保留 shard 扩展。

---

## 3.9 训练日志

每隔若干 step 记录：

```text
train/action_mse
train/value_mse
train/total_loss
val/action_mse
val/value_mse
student/action_std_mean
student/action_abs_mean
```

如果有 rollout eval，可记录：

```text
eval/mean_episode_return
eval/early_termination_rate
eval/err_rpos
eval/err_joint_pos
```

---

## 3.10 保存内容

BC checkpoint 应保存：

```text
PPO-compatible config
PPO-compatible agent_state
student_obs_filter config
dataset metadata
training metadata
```

推荐额外写：

```text
outputs/distill/.../distill_metadata.json
```

包括：

```json
{
  "teacher_ckpt": "...",
  "dataset_dir": "...",
  "student_obs_dim": 1950,
  "action_dim": 354,
  "bc_steps": 200000,
  "best_val_action_mse": 0.0123
}
```

---

## 3.11 测试

新增：

```text
tests/unit/test_bc_loss.py
tests/unit/test_distill_trainer_checkpoint.py
```

测试点：

1. action MSE 在 toy batch 上可计算；
2. value loss 可选；
3. `distribution_mean()` 支持当前 distrax distribution；
4. BC 训练 3 step 后 loss 下降；
5. 保存 checkpoint 后可以通过 `load_checkpoint()` 读回；
6. 读回 checkpoint 后 env observation dim 与 network init dim 一致。

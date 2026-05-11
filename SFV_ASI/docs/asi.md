# SFV ASI 方法与 MuscleMimic 配合方案

本文针对 Peng et al. 2018, *SFV: Reinforcement Learning of Physical Skills from Videos* 中的 Adaptive State Initialization (ASI) 做工程化拆解，并说明如何接入当前 `musclemimic` 代码。

## 0. 信心边界

不能对该策略给出数学或工程意义上的 100% 保证。原因很直接：SFV 的 ASI 在论文中是在 Bullet + 低维 humanoid + 单技能 policy 上验证的，而 `musclemimic` 使用 MuJoCo/MJX、肌肉骨骼模型、大规模并行环境、自动 reset wrapper、多轨迹数据集和额外的 reward/termination curriculum。这里的方案应被视为“高可行性的第一版工程适配”，不是论文方法的等价复现。

因此，实现前必须满足两个条件：

- 文档中明确哪些部分忠实于 SFV，哪些是为 `musclemimic` 做的近似。
- 第一版实现必须包含防作弊、防分布塌缩、credit assignment 和 checkpoint 恢复设计，否则 ASI 可能让训练指标变好但真实动作覆盖变差。

## 1. 论文中 ASI 解决的问题

SFV 的输入参考动作来自单目视频 pose estimation 和 motion reconstruction，而不是干净的 mocap。视频重建动作会有几个典型问题：

- 局部姿态错误，例如倒立、旋转、遮挡导致关节预测错位。
- 高频抖动，导致由有限差分得到的速度很不可靠。
- 真实演员和模拟角色形态不同，某些参考状态在模拟器中不可恢复。
- 动作很短但很动态，例如 backflip/frontflip/handspring，一旦从动作开头训练，策略很难探索到中后段状态。

DeepMimic 常用的 Reference State Initialization (RSI) 是直接从参考轨迹任意帧初始化。对干净 mocap 很有效，但对视频重建动作不稳，因为 RSI 会把策略初始化到一些“参考里存在、物理上很差”的状态。ASI 的目标就是学习一个更适合训练的初始状态分布，让策略多从有价值、可恢复、能产生高回报的状态开始。

## 2. 论文中的 ASI 形式化方法

论文把训练看成两个协作智能体：

- 控制策略 `pi_theta(a | s)`：控制模拟角色。
- 初始状态分布 `rho_omega(s0)`：给每个 episode 采样初始状态。

联合目标是最大化从 `rho_omega` 初始化、由 `pi_theta` rollout 得到的期望折扣回报：

```text
J(theta, omega) = E_{tau ~ p_{theta,omega}(tau)} [sum_t gamma^t r_t]
```

其中 trajectory 分布为：

```text
p_{theta,omega}(tau)
  = rho_omega(s0) * prod_t p(s_{t+1} | s_t, a_t) * pi_theta(a_t | s_t)
```

对初始状态分布的 policy gradient 是：

```text
grad_omega J
  = E[ grad_omega log rho_omega(s0) * sum_t gamma^t r_t ]
```

也就是说：

- 如果从某个初始状态开始能得到高回报，就提高这个初始状态的概率。
- 如果某个初始状态经常导致失败或低回报，就降低它的概率。
- 这个更新只在 episode 的第一步对 `rho_omega` 做一次，不像 actor policy 那样每个 timestep 都更新。

## 3. 论文中的初始状态分布参数化

SFV 把状态拆成：

```text
s = [s_hat, phi]
```

- `phi` 是 motion phase，离散取 `k` 个均匀 phase 点。
- `s_hat` 是除 phase 以外的角色状态，例如 root、关节姿态、线速度、角速度等。

初始状态分布被分解为：

```text
rho_omega(s) = p_omega(s_hat | phi) * p(phi)
```

其中：

- `p(phi)` 是离散均匀分布。
- 每个 phase 对应一个独立高斯：

```text
p_omega(s_hat | phi_i) = N(mu_i, Sigma_i)
```

实现细节：

- 论文实验中通常用 `k = 10` 个 Gaussian component。
- 每个 `mu_i` 初始化为参考动作在对应 phase 的状态。
- 每个 `Sigma_i` 初始化为参考动作所有状态的样本协方差，实际使用对角协方差。
- `mu_i` 和 `Sigma_i` 都用上面的 policy gradient 更新。
- phase component 的位置固定，不学习 phase 本身。

论文给出的训练流程是：

1. 从 `rho_omega(s0)` 采样初始状态。
2. 用当前策略 rollout 一个 episode。
3. 用 TD(lambda) 估计 value target，用 GAE(lambda) 估计 actor advantage。
4. PPO 更新 `pi_theta`。
5. 用 episode return `R0` 更新 `rho_omega`。

论文使用的关键超参：

```text
k = 10 Gaussian components
gamma = 0.95
lambda = 0.95
policy optimizer: PPO, clip threshold = 0.2
rho optimizer: SGD, alpha_rho = 0.001
rho update batch: 2000 episodes
```

## 4. ASI 与 MuscleMimic 当前实现的关系

当前项目已经有几个相关入口：

- `loco_mujoco/trajectory/handler.py`
  - `TrajectoryHandler.reset_state()` 负责选 `traj_idx` 和 `subtraj_step_idx`。
  - 当前可以均匀随机选轨迹、随机选起始帧。
  - 如果 `carry.sampling_weights` 存在，会按轨迹权重采样轨迹。

- `loco_mujoco/core/initial_state_handler/traj_init_state.py`
  - `TrajInitialStateHandler.reset()` 读取当前 `traj_state` 对应帧，并把模拟状态设为参考轨迹状态。

- `musclemimic/environments/base.py`
  - `LocoCarry` 已有 `traj_state`、`sampling_weights`、`ema_done_counts`、`ema_early_counts`。

- `musclemimic/algorithms/common/adaptive_sampling.py`
  - 当前已有“按轨迹 early termination 率提高采样概率”的机制。
  - 这是 trajectory-level adaptive sampling，不是论文里的 full-state ASI。

- `musclemimic/algorithms/ppo/runner.py`
  - rollout 后会计算 early termination rate。
  - adaptive sampling 已在 update boundary 更新 `sampling_weights`。

因此，MuscleMimic 已经有 ASI 的一部分基础设施：可以控制 reset 时采哪个轨迹、哪个帧，也可以在 PPO 外维护额外的采样状态。缺失的是：按“初始状态/初始帧”学习 `rho_omega`，以及用 episode return 或 imitation quality 更新它。

## 5. 推荐实现路线

不要第一步就照搬论文的 full-state Gaussian ASI。原因是 MuscleMimic 的状态是 MuJoCo/MJX 状态，直接对高维 `qpos/qvel` 采高斯噪声容易破坏 quaternion 归一化、接触一致性、关节范围、肌肉状态和地面穿透。更稳妥的路线是分两阶段：

### 5.1 第一阶段：frame-level ASI

把 ASI 的 `rho_omega(s0)` 简化成“轨迹-起始帧”的离散分布：

```text
rho_omega(i, b) = categorical(logits[i, b])
```

其中：

- `i` 是 trajectory id。
- `b` 是 phase bucket 或起始帧 bucket。
- 每个 trajectory 分成 `K` 个 bucket，例如 `K = 10` 或 `K = 20`。
- bucket 对应参考轨迹上的一个起始帧或一小段帧范围。

初始化：

- `logits[i, b] = 0`，即均匀分布。
- 或者保留当前 trajectory-level adaptive sampling，先采 trajectory，再在该 trajectory 内按 learned frame logits 采 bucket。

采样：

```text
traj_id ~ categorical(traj_weights)
bucket_id ~ categorical(frame_logits[traj_id])
start_step ~ bucket_to_frame(traj_id, bucket_id)
```

更新：

```text
logits[traj_id, bucket_id] += alpha_rho * (R0 - baseline[traj_id, bucket_id])
```

上式只表示“高分 bucket 增加概率”的直觉，不应作为最终实现公式。第一版实现应使用后文的严格 categorical policy-gradient update。

其中：

- `R0` 可用 episode return、normalized return、或者 `1 - early_terminated` 与 tracking reward 的组合。
- `baseline` 用 EMA 减小方差。
- 更新后对 logits 做 clip，例如 `[-5, 5]`，避免分布塌缩。
- 加 uniform floor，例如最终概率 `p = (1 - eps) * softmax(logits) + eps / K`，`eps = 0.05` 到 `0.2`。

优点：

- 不破坏物理状态，只从已有 reference frame reset。
- 与当前 `TrajectoryHandler` 非常接近。
- 适合 JAX/MJX 批量训练。
- 足以解决“哪些动作段应该多训练”的核心问题。

重要偏差：

- 这不是 SFV 原文的 Gaussian ASI。SFV 固定均匀 phase 分布，学习的是每个 phase 条件下的连续状态分布；这里学习的是离散 phase/bucket 的采样概率。
- 因为它会改变 phase 采样概率，存在偏向容易片段、降低完整动作覆盖率的风险。
- 如果研究目标是复现 SFV，应保留一个更忠实的 variant：`bucket/phase` 均匀采样，只学习每个 bucket 的局部状态扰动或质量权重；如果目标是工程训练稳定性，才使用 categorical frame-level ASI。

第一版建议把它命名为 `FrameCategoricalASI`，避免和论文 ASI 混淆。

### 5.2 第二阶段：local-state ASI

在 frame-level ASI 稳定后，再加入小扰动：

```text
s0 = ref_state(i, frame) + epsilon
epsilon ~ N(0, sigma_i,b^2)
```

建议只对安全变量加扰动：

- root XY 小扰动：例如 `sigma = 0.02m`。
- root yaw 小扰动：例如 `sigma = 0.03rad`。
- 非 root hinge qpos 小扰动：例如 `sigma = 0.01rad`。
- qvel 小扰动：例如 `sigma = 0.05` 到 `0.1`。

不建议直接扰动：

- quaternion 四元数的 4 个原始分量。
- 肌肉 activation/excitation 内部状态，除非已有明确 reset 接口。
- contact-rich 片段的 root height。

如果扰动 quaternion，应在 tangent space 采样小旋转，再左乘/右乘到 reference quaternion，并重新 normalize。

局部扰动必须在设置 `qpos/qvel` 后重新执行 forward dynamics，使 `xpos/site_xpos/cvel` 与新状态一致。否则 reward、goal lookahead 和 terminal check 会看到不一致的缓存状态。

## 6. MuscleMimic 中的具体改动点

### 6.1 扩展 carry

在 `musclemimic/environments/base.py` 的 `LocoCarry` 增加：

```python
asi_frame_logits: jax.Array | None = None      # (num_envs, n_traj, K)
asi_frame_probs: jax.Array | None = None       # (num_envs, n_traj, K)
asi_frame_baseline: jax.Array | None = None    # (num_envs, n_traj, K)
asi_start_bucket: jax.Array = int32 scalar     # 当前 episode 采到的 bucket
asi_start_step: jax.Array = int32 scalar       # 当前 episode 采到的真实起始帧
```

如果只在 runner 维护 ASI 状态，也可以不把 logits 放进 carry，只把当前 reset 需要的采样概率放进去。但 reset 在 env 内执行，最简单的方式还是让 carry 持有可被 `TrajectoryHandler.reset_state()` 读取的分布。

需要区分两种 shape：

- runner/env_state 中的 batched carry 形状应是 `(num_envs, n_traj, K)`。
- 被 `vmap` 后进入单个 env reset 的 carry 逻辑形状应是 `(n_traj, K)`。

实现时不要在 `TrajectoryHandler.reset_state()` 中假设第一维一定是 `num_envs`。应复用类似 `update_carry_weights_normalized()` 的 wrapper-state 更新方式写入 batched carry，让 vmap 自动切片。

### 6.2 改 `TrajectoryHandler.reset_state()`

当前位置已经支持：

```python
weights = getattr(carry, "sampling_weights", None)
traj_idx = jax.random.choice(_k1, self.n_trajectories, p=weights)
subtraj_step_idx = jax.random.randint(...)
```

需要增加：

```text
if carry.asi_frame_probs is not None:
    bucket = categorical(carry.asi_frame_probs[traj_idx])
    subtraj_step_idx = bucket_to_step(traj_idx, bucket)
else:
    使用现有 random step
```

`bucket_to_step` 必须 JIT 友好：

- 预先构造 `(n_traj, K)` 的 `bucket_start_steps`。
- 对每条轨迹按长度计算 bucket 对应帧：

```text
bucket_start_steps[i, b] = floor(b / K * len_trajectory(i))
```

需要注意每条轨迹长度不同，建议在 `TrajectoryHandler.__init__` 里预计算 `traj_lengths` 和 `bucket_start_steps`。

必须额外处理以下边界：

- 不要采到最后一帧或过于靠近结尾的帧，否则 episode 立即结束，ASI 会错误地奖励“短 episode”。建议设 `min_remaining_steps = max(num_steps, n_step_lookahead * n_step_stride + 1)`，使 `start_step <= len_i - min_remaining_steps`；太短的轨迹退化为从 0 开始或进入禁用 ASI 的 mask。
- 如果 `len_i < K`，多个 bucket 会映射到同一帧。应允许重复但在日志中记录有效 bucket 数，或把 `K_i = min(K, len_i)` 做成 mask。
- 对 contact-rich 或明显坏数据帧，不应只靠 ASI 自己学会避开。建议支持 `valid_start_mask[i, b]`，由数据预处理阶段屏蔽 NaN、速度爆炸、穿地、root 跳变等 bucket。
- goal、reward、terminal 都必须使用 `subtraj_step_no_init` 计算 root XY offset。当前 `MimicReward` 和 `MeanRelativeSiteDeviationWithRootTerminalStateHandler` 已有类似逻辑，但实现 ASI 前要确认 `GoalTrajMimic` 的 lookahead 也不会因为随机起点产生世界坐标偏置。

### 6.3 在 rollout batch 中记录 episode 初始信息

ASI 更新需要知道每个完成 episode 的初始 `(traj_id, bucket_id)` 和回报。当前 adaptive sampling 用 `traj_batch.info["final_traj_no"]` 统计每条轨迹的 early termination。ASI 需要更细：

```text
init_traj_no:    (num_steps, num_envs)
init_bucket_no:  (num_steps, num_envs)
episode_return:  (num_steps, num_envs) 或在 log wrapper 中取 episode return
done:            (num_steps, num_envs)
absorbing:       (num_steps, num_envs)
```

这是第一版最容易出错的地方。因为训练环境通常有 `AutoResetWrapper`、`LogWrapper`、可能还有 `NStepWrapper`，一个 rollout batch 内某个 env 可能经历多个 episode。不能用当前 timestep 的 `traj_state` 或 `final_traj_no` 反推 episode 初始 bucket；那会把 credit 分给错误起点。

最低要求：

- reset 时把 `init_traj_no`、`init_bucket_no`、`init_start_step` 写进 carry。
- step info/metrics 中在 episode done 的那个 timestep 输出这三个初始字段。
- episode return 必须来自同一个 episode 的累计值，而不是 rollout window 的局部 return。
- 对 auto-reset 后的首步，必须保证新 episode 的 init 信息不会覆盖刚结束 episode 的 logging 信息。

如果当前 `info` 不方便存 episode return，可以先用 rollout 内从 done 前累计的 discounted return 或 undiscounted return。第一版可以用更低方差的成功信号：

```text
score = 1.0 - early_terminated
```

再逐步换成：

```text
score = normalized_episode_return
score = normalized_episode_return - lambda_fail * early_terminated
```

不要直接使用未归一化 episode return。不同 start frame 的剩余 horizon 不同，靠近结尾的起点天然更短、更容易得到较高平均表现。推荐：

```text
score = completed_fraction * mean_step_reward - early_penalty * early_terminated
```

其中 `completed_fraction = actual_episode_len / available_len_from_start`，并且 `available_len_from_start` 至少要经过上面的 `min_remaining_steps` 过滤。

### 6.4 新增 ASI 更新函数

建议新增文件：

```text
musclemimic/algorithms/common/asi.py
```

核心纯函数：

```python
def update_frame_asi(
    logits,             # (n_traj, K)
    baseline,           # (n_traj, K)
    init_traj_ids,      # (N,)
    init_bucket_ids,    # (N,)
    scores,             # (N,)
    alpha=0.01,
    baseline_beta=0.1,
    logit_clip=5.0,
):
    ...
```

实现逻辑：

1. 用 `segment_sum` 聚合每个 `(traj, bucket)` 的 score sum 和 count。
2. 计算该 bucket 的平均 score。
3. 更新 baseline：

```text
baseline_new = (1 - beta) * baseline + beta * mean_score
```

4. advantage：

```text
adv = mean_score - baseline
```

5. 更新 logits：

```text
logits_new = logits + alpha * adv
logits_new = clip(logits_new, -logit_clip, logit_clip)
```

6. 未被采样的 bucket 不更新。

这不是严格的 categorical policy gradient，因为严格形式还会包含 `grad log softmax` 的全类归一化项。工程上可以先用这个 bandit-style rank update；如果要更贴近论文，应使用：

```text
grad_logits[c] = (one_hot(c) - probs) * advantage
```

对每个完成 episode 聚合后更新。

第一版更推荐使用严格 categorical update：

```text
grad_logits[i, :] += (one_hot(bucket) - probs[i, :]) * advantage
```

并附加：

```text
probs = (1 - uniform_mix) * softmax(logits / temperature) + uniform_mix * valid_mask_normalized
```

这样更新会显式降低同一 trajectory 内未被选 bucket 的相对概率，也更接近 policy-gradient 形式。`temperature` 建议从 `1.0` 起步，不要小于 `0.5`。

### 6.5 接入 PPO runner

在 `musclemimic/algorithms/ppo/runner.py`：

1. 初始化 ASI state：

```text
asi.enabled
asi.num_buckets
asi.alpha
asi.baseline_beta
asi.uniform_mix
asi.logit_clip
```

2. reset env 后，把 `asi_frame_probs` 写入 carry。
3. 每个 update rollout 结束后，取完成 episode 的 `(init_traj, init_bucket, score)`。
4. 调 `update_frame_asi()` 得到新 logits/probs。
5. broadcast 到 `(num_envs, n_traj, K)` 写回 carry。
6. 记录日志：

```text
asi/prob_entropy
asi/prob_min
asi/prob_max
asi/top_traj
asi/top_bucket
asi/score_mean
asi/early_rate_by_selected_bucket
```

还必须把 ASI state 纳入 checkpoint/resume：

- `logits`
- `baseline`
- `valid_start_mask`
- `bucket_start_steps`
- 当前 temperature/uniform_mix 之类静态或半静态参数

如果训练从 checkpoint 恢复但 ASI state 被重置，采样分布会突然变化，导致复现实验和续训指标不可信。

### 6.6 配置示例

建议在 `fullbody/conf_fullbody.yaml` 和具体 SFV/Badminton 配置中加：

```yaml
experiment:
  asi:
    enabled: true
    mode: frame_categorical
    num_buckets: 20
    alpha: 0.01
    baseline_beta: 0.1
    uniform_mix: 0.1
    logit_clip: 5.0
    score_type: normalized_return_minus_early
    early_penalty: 0.5
```

如果同时用现有 adaptive trajectory sampling：

```yaml
experiment:
  adaptive_sampling:
    enabled: true
    beta: 0.2
    alpha: 1.0
    floor_mix: 0.2
```

建议组合方式：

- `adaptive_sampling` 控制采哪条轨迹。
- `asi` 控制在该轨迹内从哪个 phase/bucket 开始。

## 7. 是否预计有效

预计会有效，尤其是在以下场景：

- 使用 WHAM/HMR/GMR/视频重建动作，参考轨迹噪声比 mocap 大。
- 动作包含击球、跳跃、转体、快速步法、落地等高动态片段。
- 当前训练出现大量 early termination，且失败集中在少数轨迹或少数 phase。
- 策略能学会动作开头，但中后段长期覆盖不到。

但不建议期待 ASI 单独解决所有问题。它主要改善“从哪里开始训练”的分布，不会修复错误参考动作。如果参考轨迹本身脚穿地、root 漂移、肢体方向错，ASI 只会降低这些状态的采样概率，最终可能导致某些动作段被绕开。对 SFV 风格视频模仿，ASI 应与以下组件一起使用：

- motion reconstruction / smoothing：先降低参考动作噪声。
- 较宽松的初期 termination threshold：避免一开始所有片段都立即失败。
- reward curriculum：先学姿态和关键点，再逐步加强速度项。
- validation coverage：防止 ASI 过度偏向容易片段。

## 8. 关键漏洞与修复措施

| 漏洞 | 后果 | 最低修复 |
|---|---|---|
| 把 frame-level ASI 当作 SFV ASI 等价复现 | 论文解释不严谨，复现实验结论站不住 | 明确命名为工程近似；保留 uniform phase + local perturb 的 faithful variant |
| episode return 归因到错误的 start bucket | ASI 学到随机噪声或反向分布 | 在 reset carry 和 done info 中显式记录 init traj/bucket/start step 和 episode return |
| 采到 near-terminal frame | 短 episode 作弊，概率集中到动作末尾 | 加 `min_remaining_steps` 和 horizon-normalized score |
| ASI 偏向容易片段 | 平均 reward 上升但完整动作覆盖下降 | uniform floor、entropy 监控、coverage bonus、per-phase validation |
| adaptive_sampling 与 ASI 目标冲突 | 一个采困难轨迹，一个避开低回报状态，训练振荡 | 分阶段启用；先 trajectory sampling，稳定后加 ASI；或降低二者学习率/温度 |
| adaptive_termination 改变 early termination 定义 | ASI score 非平稳，前后不可比 | ASI scoring 使用固定诊断阈值或 tracking metric；termination curriculum 只影响环境终止 |
| reward_curriculum 改变 reward 标尺 | ASI baseline 失效，logits 因 reward scale 漂移 | ASI score 使用归一化 reward 或固定 diagnostic score；reward 权重变化时重置/慢更新 baseline |
| 坏数据被 hard mining 反复采样 | 训练被不可学片段拖垮 | 预处理质量 mask；对长期高 early rate 的 bucket/trajectory quarantine |
| JAX batched carry shape 错误 | vmap/reset 编译或采样失败 | 文档和实现区分 batched env_state shape 与 per-env carry shape |
| ASI state 未 checkpoint | 续训不可复现，采样分布突变 | 把 logits/baseline/mask 写入 checkpoint state |
| local perturb 后缓存不一致 | reward/terminal 使用旧 site_xpos/cvel | 扰动后强制 forward dynamics，并测试 qpos/qvel/site consistency |

## 9. 推荐实验顺序

1. Baseline：当前随机轨迹、随机帧初始化。
2. 开启现有 `adaptive_sampling`，看 hard trajectory 是否被更多采样。
3. 加 frame-level ASI，只学习 trajectory 内起始 bucket。
4. 加 reward curriculum，提高速度 tracking 权重。
5. 如果稳定，再尝试 local-state ASI 小扰动。

核心对比指标：

- training early termination rate。
- validation early termination rate。
- validation coverage：episode 能覆盖参考轨迹长度的比例。
- `err_site_abs`、`err_rpos`、`err_joint_pos`、`err_joint_vel`。
- 每个 trajectory 和 bucket 的采样概率熵，防止分布塌缩。

新增必须通过的 prototype gate：

- 单元测试：bucket mapping 不会采到 invalid/near-terminal frame。
- 单元测试：ASI categorical update 在固定 toy score 下提高高分 bucket 概率，并保持 uniform floor。
- 集成测试：AutoResetWrapper 下 done timestep 的 init bucket 不会被下一 episode 覆盖。
- checkpoint 测试：保存/恢复后 ASI 概率完全一致。
- 小规模训练 smoke test：启用 ASI 后 validation coverage 不低于 baseline。

## 10. 结论

SFV 的 ASI 本质是学习 reset distribution。对 MuscleMimic，最务实的实现不是直接采样全状态 Gaussian，而是先做 frame-level categorical ASI：学习每条轨迹哪些 phase 更适合作为起点。它能复用当前 `TrajectoryHandler`、`TrajInitialStateHandler`、PPO runner 和 adaptive sampling 框架，改动范围小，风险低，也更适合视频重建动作和 MJX 批量训练。

但该方案不是无条件安全。只有在完成上述 credit assignment、near-terminal 防作弊、非平稳 score 隔离、checkpoint 和 coverage 验证后，才应进入正式训练实验。

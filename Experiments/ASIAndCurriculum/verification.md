# ASI And Curriculum Verification

## Goal

验证 ASI 是否真的提升 MuscleMimic 的学习效果，而不仅仅是改变了采样分布。

当前使用的数据是单条 retarget 后的 MyoFullBody 轨迹：

```text
caches/AMASS/MyoFullBody/gmr/ablation/04_lower_body_full_poses.npz
```

对应训练配置：

```text
fullbody/config_specific_task/Ablation/conf_fullbody_ablation_04_lower_body_full_gmr.yaml
```

## Important Distinction

ASI 自身日志只能说明“采样策略是否在变化”，不能直接证明“学习效果更好”。

例如：

```text
asi/frame_entropy
asi/prob_min
asi/prob_max
```

这些指标只能说明 ASI 是否开始偏向某些起始帧。真正证明 ASI 有效，需要和 baseline 对比训练效果。

## Experiment Groups

至少跑下面三组：

| Group | ASI | Adaptive Termination | Reward Curriculum | Purpose |
| --- | --- | --- | --- | --- |
| A. Baseline PPO | off | off | off | 原始 mimic PPO 对照组 |
| B. ASI Only | on | off | off | 单独验证 ASI 是否有效 |
| C. ASI + Curriculum | on | on | on | 验证完整增强策略 |

最关键的对比是：

```text
A vs B
```

它回答：ASI 本身是否有效。

```text
B vs C
```

它回答：curriculum 是否在 ASI 基础上进一步提升。

## Commands

所有命令从仓库根目录执行：

```bash
cd /data3/yangfeiyang/WorkSpace/musclemimic
```

### A. Baseline PPO

```bash
MM_CUDA_VISIBLE_DEVICES=2 \
scripts/run_with_cuda_compat.sh \
.venv/bin/python fullbody/experiment.py \
  --config-name=config_specific_task/Ablation/conf_fullbody_ablation_04_lower_body_full_gmr \
  wandb.mode=online \
  experiment.asi.enabled=false \
  experiment.adaptive_termination.enabled=false \
  experiment.reward_curriculum.enabled=false
```

### B. ASI Only

```bash
MM_CUDA_VISIBLE_DEVICES=2 \
scripts/run_with_cuda_compat.sh \
.venv/bin/python fullbody/experiment.py \
  --config-name=config_specific_task/Ablation/conf_fullbody_ablation_04_lower_body_full_gmr \
  wandb.mode=online \
  experiment.asi.enabled=true \
  experiment.adaptive_termination.enabled=false \
  experiment.reward_curriculum.enabled=false
```

### C. ASI + Curriculum

```bash
MM_CUDA_VISIBLE_DEVICES=2 \
scripts/run_with_cuda_compat.sh \
.venv/bin/python fullbody/experiment.py \
  --config-name=config_specific_task/Ablation/conf_fullbody_ablation_04_lower_body_full_gmr \
  wandb.mode=online \
  experiment.asi.enabled=true \
  experiment.adaptive_termination.enabled=true \
  experiment.reward_curriculum.enabled=true
```

## Primary Metrics

这些是判断学习是否更好的核心指标。

### 1. Mean Episode Return

WandB key:

```text
mean_episode_return
```

越高越好。表示 imitation reward 更高。

判断方式：

```text
同样 timesteps 下，ASI 组比 baseline 更高。
```

### 2. Mean Episode Length

WandB key:

```text
mean_episode_length
```

越高越好。表示策略能活得更久，不容易 early terminate。

### 3. Early Termination Rate

WandB key:

```text
ppo/early_termination_rate
```

越低越好。这是当前任务里非常关键的稳定性指标。

理想现象：

```text
ASI only 的 early termination rate 下降速度快于 baseline。
```

### 4. Validation Tracking Error

使用 validation 里导出的 tracking metrics。重点看：

```text
JointPosition / EuclideanDistance
JointVelocity / EuclideanDistance
RelSitePosition / EuclideanDistance
RelSiteVelocity / EuclideanDistance
RelSiteOrientation / EuclideanDistance
```

越低越好。这个指标比训练 reward 更能说明是否真的跟上参考动作。

### 5. Time-To-Threshold

定义一个稳定阈值，例如：

```text
ppo/early_termination_rate < 0.1
```

比较不同实验组达到该阈值需要多少 environment timesteps。

越少越好。

### 6. Learning Curve AUC

固定训练步数内，比较：

```text
mean_episode_return 曲线下面积
mean_episode_length 曲线下面积
```

AUC 越大，说明整体学习效率越高。

## ASI Diagnostic Metrics

这些指标用来确认 ASI 机制有没有在工作。

### 1. ASI Entropy

WandB key:

```text
asi/frame_entropy
```

含义：

```text
ASI 起始 bucket 分布的熵。
```

如果 entropy 下降，说明 ASI 从接近均匀采样变成偏向某些起始片段。

### 2. ASI Probability Range

WandB keys:

```text
asi/prob_min
asi/prob_max
```

含义：

```text
prob_max 上升：某些起始 bucket 被更频繁采样。
prob_min 下降：某些起始 bucket 被降低采样。
```

注意：这只能说明采样分布发生变化，不等于学习效果提升。

## Success Criteria

ASI 有效的证据应该是：

```text
在 A vs B 对比中：
  B 的 mean_episode_return 更高，或更早达到同等 return；
  B 的 mean_episode_length 更长；
  B 的 ppo/early_termination_rate 更低；
  B 的 validation tracking error 更低；
  B 的 time-to-threshold 更短。
```

如果只看到：

```text
asi/prob_max 上升
asi/frame_entropy 下降
```

但 reward、episode length、termination、validation error 没有改善，则不能说 ASI 有效。

## Current Limitation

当前实验只有一条 motion：

```text
ablation/04_lower_body_full_poses
```

所以 ASI 只能在同一条轨迹内部选择不同起始帧，不能验证“选择更难 motion”的效果。

这意味着：

```text
ASI 的效果可能比较弱；
如果单条 motion 很短或难度分布不明显，ASI 的优势不一定显著。
```

更强的验证方式是加入多条动作，或者至少加入同一动作的多个阶段片段，例如：

```text
准备阶段
引拍阶段
击球阶段
随挥阶段
回位阶段
```

## Recommended Reading Of Results

优先级：

```text
1. ppo/early_termination_rate
2. mean_episode_length
3. mean_episode_return
4. validation tracking error
5. time-to-threshold
6. ASI diagnostic metrics
```

推荐先跑较短训练，确认趋势：

```text
2M - 5M environment timesteps
```

如果趋势清楚，再跑完整：

```text
20.48M environment timesteps
```

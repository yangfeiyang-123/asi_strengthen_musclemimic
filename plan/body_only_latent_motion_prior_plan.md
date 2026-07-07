# Body-Only Denoising Latent Motion Prior 完整计划

日期：2026-06-22  
分支：`body-only-latent-motion-prior`  
目标仓库：`/data3/yangfeiyang/WorkSpace/musclemimic`

## 0. 一句话目标

从不完美的人体视频先验（WHAM/SMPL/retarget cache）中，学习一个不依赖手部、不追求击球任务的 **body-only 合理人体动作框架**；再把后续学习限制在这个新学到的身体动作流形中，使肌骨模型动作更稳定、更平滑、更符合人体协同。

这里的“新的流行”按上下文理解为“新的动作流形 / latent body-motion manifold”：

```text
不完美人体先验
  -> body-only denoising teacher
  -> latent body-motion manifold
  -> 在该 manifold 内学习/采样/微调
```

## 1. Problem Anchor

### 1.1 需要解决的问题

当前仓库可以从视频/WHAM/SMPL 经 GMR retarget 到 MyoFullBody，再用 PPO 做 mimic tracking。但这条路线的核心问题是：WHAM/SMPL 和 retarget 轨迹并不完美，尤其手、腕、末端姿态噪声很大；如果直接逐帧追踪，会把噪声和不合理末端动作也学进肌骨控制器。

本项目真正要解决的是：

> 如何从不完美人体先验中提取稳定的身体动作框架，让肌骨模型学会合理的人体 body coordination，而不是机械追踪 noisy trajectory。

### 1.2 非目标

本阶段明确不做：

- 不做击球、落点、球拍、shuttlecock 任务。
- 不追求 right hand / wrist / fingers 的精确重建。
- 不把手部轨迹作为主要 reward、reference feature 或 latent supervision。
- 不先做 high-level task PPO。
- 不以“看起来更像视频中的手部姿态”为成功条件。

### 1.3 成功条件

一个方法可以被认为成功，至少要满足：

- 在不含手部 site 的 body-only tracking 上不明显弱于 direct tracker。
- 动作更平滑：root jerk、joint acceleration、action rate 下降。
- 肌肉控制更合理：activation energy、activation saturation、high-frequency activation 下降。
- 接触更稳定：foot slip 和 early termination rate 下降。
- prior-only rollout 可以在短时间内自然延续动作相位，而不是立刻崩溃或冻结。
- latent 空间能表达身体动作阶段，例如准备、转体、蹬转、挥臂大链条、回位，而不是编码手部噪声。

## 2. 核心假设

### 2.1 主假设

即使 WHAM/SMPL 轨迹不完美，其大身体链条仍包含有价值的人体先验：

- root / pelvis 的位移和朝向趋势
- torso rotation
- shoulder-elbow 的大臂运动趋势
- hip-knee-ankle 的支撑和蹬转
- 动作 phase 的连续结构

如果训练目标只保留这些相对可靠的 body signals，并显式排除 hand/wrist/finger 噪声，就可以学到一个更稳健的 body latent manifold。

### 2.2 方法假设

直接训练 PPO tracker 会把“逐帧跟踪”当成目标；而 latent distillation 可以把 teacher 的 body action 压缩成：

```text
posterior q(z | body_state, body_reference)
prior     p(z | body_state)
decoder   D(body_state, z) -> body_action
```

其中 `prior p(z | body_state)` 描述当前身体状态下合理的动作延续空间；后续学习不再自由探索全部 muscle action，而是在这个 latent body-motion manifold 内学习。

## 3. 总体路线

完整路线分为两大阶段。

### 3.1 阶段 A：从不完美中学习

目标：从 noisy WHAM/SMPL/retarget 数据中提取 body-only 动作框架。

```text
video / WHAM / SMPL
  -> AMASS-style npz
  -> MyoFullBody GMR retarget cache
  -> body-only QC and filtering
  -> body-only denoising tracker teacher
  -> teacher rollout shards
  -> latent body prior distillation
```

### 3.2 阶段 B：在新动作流形中学习

目标：把后续学习和采样限制在已学到的 body latent manifold 内。

```text
body_state
  -> prior p(z | body_state)
  -> latent sample / latent residual / constrained update
  -> decoder D(body_state, z)
  -> body muscle action
  -> MyoFullBody simulation
```

在当前阶段，阶段 B 不做击球任务，只做：

- prior-only rollout
- phase continuation
- short-horizon motion refinement
- robust recovery from noisy states

## 4. Body-Only 数据定义

### 4.1 保留的身体信号

用于 tracking reward、reference features、evaluation 的 body sites 建议包括：

```text
pelvis_mimic
upper_body_mimic
head_mimic
left_shoulder_mimic
left_elbow_mimic
right_shoulder_mimic
right_elbow_mimic
left_hip_mimic
left_knee_mimic
left_ankle_mimic
left_toes_mimic
right_hip_mimic
right_knee_mimic
right_ankle_mimic
right_toes_mimic
```

### 4.2 排除的信号

第一阶段明确排除：

```text
left_hand_mimic
right_hand_mimic
right wrist
finger joints
grip/racket sites
```

排除不代表这些信号永远无用，而是避免它们污染 body latent manifold。

### 4.3 数据命名约定

不要覆盖已有 baseline cache。建议新建 namespace：

```text
forehand_clear/stage5_10demo_body_only
forehand_clear/stage5_10demo_body_only_filtered
```

manifest 建议：

```text
manifests/stage5_10demo_body_only_list.txt
manifests/stage5_10demo_body_only_filtered_list.txt
```

训练配置建议：

```text
fullbody/config_specific_task/body_only/conf_fullbody_forehandclear_body_teacher.yaml
fullbody/config_specific_task/body_only/conf_fullbody_forehandclear_body_latent.yaml
```

## 5. 阶段 0：数据和 retarget 基线冻结

### 5.1 目标

先冻结一个可复现实验数据版本，避免后续每次结果变化都无法归因。

### 5.2 输入

优先使用当前较平滑的 retarget cache：

```text
caches/AMASS/MyoFullBody/gmr/forehand_clear/stage5_10demo_smooth_filtered
```

如果要重新生成，必须保持统一 FPS，并写入新的 namespace。

### 5.3 检查项

- AMASS npz 中 `mocap_framerate` 和 `mocap_frame_rate` 一致。
- GMR config 的 `target_fps` 与源数据一致。
- cache frequency、训练 control frequency、渲染 fps 不混淆。
- 不覆盖 `stage5_10demo`、`stage5_10demo_smooth`、`stage5_10demo_smooth_filtered` 这些已有结果。

### 5.4 产物

- body-only manifest
- body-only teacher config
- body-only evaluation config
- cache QC report

### 5.5 Stop / Go Gate

只有在以下条件满足时进入 teacher 训练：

- root speed 无明显断裂
- hand speed spike 不再作为排除标准，但不能通过 shoulder/elbow 传播成大身体异常
- qpos 大步跳变不集中出现在 torso/hip/knee/ankle/shoulder/elbow
- 视频/metric 都说明数据足够训练 body-only teacher

## 6. 阶段 1：Body-Only Denoising Teacher

### 6.1 目标

训练一个 body-only teacher tracker。它可以看 reference lookahead，但只追踪可靠身体链条。

### 6.2 Teacher 输入

保留当前 full observation 中的身体状态和肌肉状态：

- joint position / velocity
- root orientation / velocity
- muscle length / velocity / force
- muscle excitation / activation
- touch sensor
- body-only reference lookahead
- phase

### 6.3 Teacher 输出

短期保持当前 MyoFullBody action convention：

```text
body_action in [-1, 1]^{N_body_actuators}
```

如果当前环境仍输出全 actuator action，第一版可以先让 teacher 输出 full action，但训练 latent 时用 action mask 只取 body actuator subset。

### 6.4 Reward 设计

原始 mimic reward 需要从 exact tracking 改成 robust body imitation：

```text
R_teacher =
    w_root      * root reward
  + w_torso     * torso / upper body reward
  + w_leg       * hip-knee-ankle-toes reward
  + w_arm_chain * shoulder-elbow reward
  + w_vel       * body velocity reward
  - c_rate      * action rate
  - c_energy    * activation energy
  - c_acc       * joint acceleration
  - c_slip      * foot slip
```

建议第一版权重方向：

- 降低 qpos 逐维精确追踪。
- 降低或移除 hand site reward。
- 保留 root / torso / leg / shoulder-elbow 的相对位置奖励。
- 打开 `action_rate_coeff` 和 `activation_energy_coeff`。
- 如果 foot slip 指标当前只是离线 metric，先作为 eval；后续再接入 reward。

### 6.5 Teacher 训练输出

保存：

```text
checkpoints/body_only_teacher/...
outputs/body_only_teacher/train_metrics.json
outputs/body_only_teacher/eval_body_only_metrics.json
```

### 6.6 Teacher 成功标准

必须满足：

- body-only tracking error 可接受。
- 不含 hand 后 early termination rate 下降或不升高。
- action rate / activation energy 不显著恶化。
- 渲染视频中身体重心、躯干、下肢、肩肘链条自然。

如果 teacher 本身不稳定，不进入 latent distillation。

## 7. 阶段 2：收集 Latent Distillation 数据

### 7.1 目标

从 body-only teacher 收集训练 latent prior 所需的数据：

```text
body_state
body_reference_features
teacher_body_action
phase
traj_no
subtraj_step_no
diagnostics
```

### 7.2 Collector 要求

使用当前 distillation collector 的 `--save-reference-features`，但必须确保 reference feature 是 body-only 的 dropped goal lookahead，不含 hand。

### 7.3 Shard schema

每个 shard 至少包含：

```text
student_obs
reference_features
teacher_action
teacher_mu
teacher_log_std
teacher_value
reward
phase
traj_no
subtraj_step_no
done
absorbing
```

如果 teacher 输出 full action，需要额外记录 action mask 或 body action subset：

```text
body_action_indices
teacher_body_action
```

### 7.4 数据 split

建议：

```text
train: 70%
val:   15%
test:  15%
```

如果只有 10 条 demo，不建议按随机 frame 完全打散。更稳妥：

- train 使用 7 条 motions
- val 使用 1-2 条 motions
- test 使用 1-2 条 motions

这样能检验 latent prior 是否跨 motion 泛化，而不是记住单条轨迹。

### 7.5 数据质量 gate

进入 latent training 前检查：

- `reference_features_dim` 一致。
- `student_obs_dim` 一致。
- `teacher_action` 和 action mask 维度一致。
- phase 覆盖完整动作周期。
- 每个 motion 至少有足够 samples。
- done/absorbing 过高的 shard 不进入主训练。

## 8. 阶段 3：Latent Body Prior Distillation

### 8.1 目标

训练 body-only latent 模型：

```text
q_phi(z | s_body, r_body)
p_psi(z | s_body)
D_theta(s_body, z) -> a_body
```

### 8.2 输入输出

训练时：

```text
state = body_state
reference = body_reference_features
posterior_mu, posterior_sigma = q_phi(state, reference)
z = posterior sample
pred_body_action = D_theta(state, z)
prior_mu, prior_sigma = p_psi(state)
```

部署/评估时：

```text
state = body_state
prior_mu, prior_sigma = p_psi(state)
z ~ p_psi(state) 或 z = prior_mu
body_action = D_theta(state, z)
```

### 8.3 Loss

建议第一版：

```text
L =
    lambda_action * MSE(pred_body_action, teacher_body_action)
  + lambda_KL     * KL(q_phi(z|s,r) || p_psi(z|s))
  + lambda_smooth * ||a_t - a_{t-1}||^2
  + lambda_bound  * action bound penalty
```

后续可加入：

```text
lambda_energy * predicted activation energy proxy
lambda_phase  * phase consistency auxiliary loss
```

### 8.4 KL 策略

避免 posterior collapse：

- 使用 KL warmup。
- 记录 per-dim KL。
- 如果 KL 很快接近 0，同时 action MSE 不好，说明 latent 没起作用。
- 如果 KL 过高且 prior rollout 崩，说明 prior 没学会状态条件分布。

建议起点：

```text
latent_dim: 16
kl_weight: warmup from 1e-5 to 1e-3
sigma_min: 0.05
sigma_max: 2.0
batch_size: 4096
hidden_dims: [512, 256]
```

latent_dim 后续 ablation：

```text
8, 16, 32
```

### 8.5 保存格式

保存：

```text
outputs/body_latent_prior/
  latent_config.yaml
  posterior_params.npz
  prior_params.npz
  decoder_params.npz
  normalization_stats.npz
  action_mask.json
  train_metrics.json
  val_metrics.json
```

### 8.6 Latent training 成功标准

必须同时满足：

- validation action MSE 接近 BC student baseline。
- KL 不 collapse。
- prior sigma 不全变成 `sigma_min` 或 `sigma_max`。
- decoder output 不频繁 saturate 到 action bound。
- posterior rollout 可复现 teacher 的 body motion。
- prior-mean rollout 可以短时稳定延续 phase。

## 9. 阶段 4：在新动作流形中学习

### 9.1 本阶段含义

这里不是击球任务学习，而是在新学到的 latent body-motion manifold 中进行动作延续、微调和稳健性学习。

第一版不使用自由 muscle action PPO，而只允许以下形式：

```text
z = prior_mu(s)
```

或：

```text
z = prior_mu(s) + alpha * prior_sigma(s) * tanh(u)
```

其中 `u` 可以来自一个很小的 residual policy 或 optimizer，但不是高维 muscle action。

### 9.2 三种学习层级

#### Level 1：Prior Mean Rollout

不训练新 policy：

```text
z = prior_mu(s)
a = D(s, z)
```

目的：检查 manifold 自身是否可滚动。

#### Level 2：Constrained Latent Residual

训练一个小 residual：

```text
u = pi_residual(s, phase)
z = prior_mu(s) + alpha * prior_sigma(s) * tanh(u)
```

reward 仍然是 body-only imitation + smoothness，不含 hand。

#### Level 3：Latent Recovery Policy

从扰动状态恢复到合理动作流形：

```text
s_noisy -> pi_recovery -> z -> decoder -> stable body action
```

这可以证明 latent prior 不只是 reconstruction，而是有实际稳定化作用。

### 9.3 本阶段成功标准

- residual policy 的动作质量优于直接 full-action PPO 微调。
- 在扰动初始状态下，latent recovery 的 early termination 更低。
- latent residual 不导致 action energy 飙升。
- phase continuation 更平滑。

## 10. Baselines 和 Ablations

### 10.1 必须比较的系统

| ID | 系统 | 目的 |
|----|------|------|
| B0 | Noisy retarget replay / reference only | 数据质量下界 |
| B1 | Direct body-only PPO tracker | 强 tracking baseline |
| B2 | BC student without latent | 检查 latent 是否真的有用 |
| M1 | Posterior latent decoder | 检查 latent reconstruction |
| M2 | Prior mean latent decoder | 检查 manifold rollout |
| M3 | Prior + constrained residual | 检查在新流形内学习是否有效 |

### 10.2 必须做的 ablation

| Ablation | 问题 |
|----------|------|
| with hand vs without hand | 排除手是否真的减少噪声污染 |
| latent_dim 8/16/32 | latent 容量是否足够 |
| KL weight sweep | prior 是否学到有效分布 |
| no action smoothness | 平滑项是否必要 |
| no activation regularization | 肌肉合理性是否来自正则 |
| prior mean vs posterior sample | prior 是否真正可部署 |

### 10.3 不建议第一阶段做的 ablation

- 球拍/击球任务。
- 手腕 residual。
- 大规模 motion diffusion。
- LLM/VLM planner。
- 复杂 multi-skill hierarchy。

这些会稀释主问题。

## 11. 指标体系

### 11.1 Body tracking 指标

不含手部：

```text
err_root_xyz
err_root_yaw
err_joint_pos_body
err_joint_vel_body
err_site_abs_body
err_rpos_body
```

### 11.2 平滑性指标

```text
root_jerk
joint_acceleration
qpos_step_norm
qvel_step_norm
action_rate
```

### 11.3 肌肉合理性指标

```text
activation_energy
activation_rate
excitation_rate
action_saturation_ratio
high_frequency_activation_power
```

### 11.4 接触和稳定性指标

```text
foot_slip
foot_contact_consistency
early_termination_rate
episode_length
fall_rate
root_height_outlier_rate
```

### 11.5 Latent 指标

```text
posterior_action_mse
prior_action_mse
KL_per_dim
active_latent_dims
prior_sigma_mean
prior_sigma_min_ratio
prior_sigma_max_ratio
latent_temporal_smoothness
latent_phase_separability
```

### 11.6 定性输出

每个关键系统至少渲染：

- same motion / same seed 的 side-by-side video
- root/torso/leg/shoulder-elbow overlay
- activation heatmap
- foot contact timeline
- latent trajectory PCA/t-SNE plot

## 12. 实验块设计

### Block 1：Body-only teacher 是否比 full noisy tracking 更适合作为先验

Claim：去掉手部并加入鲁棒身体约束后，teacher 更适合提取人体动作框架。

比较：

- full-site tracker
- body-only tracker
- body-only tracker + smooth/energy penalties

指标：

- body-only tracking error
- action rate
- activation energy
- foot slip
- early termination

成功标准：

- body-only tracking 不显著变差。
- smoothness/muscle/contact 指标明显改善。
- 视频中身体链条更自然。

优先级：MUST-RUN

### Block 2：Latent prior 是否能重建 teacher body action

Claim：latent posterior/decoder 可以压缩 teacher 的 body control。

比较：

- BC student
- posterior latent decoder
- posterior latent decoder without KL

指标：

- action MSE
- body rollout tracking
- latent KL
- action smoothness

成功标准：

- posterior latent decoder 接近 BC student。
- 加 KL 后 MSE 可接受，且 prior 不 collapse。

优先级：MUST-RUN

### Block 3：Prior-only rollout 是否形成可用动作流形

Claim：`p(z|s)` 不只是训练正则，而是可部署的状态条件动作流形。

比较：

- posterior rollout
- prior mean rollout
- prior sample rollout
- BC student rollout

指标：

- short-horizon body tracking
- early termination
- action energy
- latent sigma diagnostics

成功标准：

- prior mean rollout 能稳定延续 1-3 秒。
- prior sample 不产生大量异常动作。

优先级：MUST-RUN

### Block 4：在 latent manifold 内学习是否优于直接 muscle action 学习

Claim：后续学习限制在 latent manifold 内，比直接在 muscle action 空间学习更稳定。

比较：

- direct PPO fine-tune on body-only reward
- latent residual policy
- latent residual without prior scaling

指标：

- sample efficiency
- early termination
- action rate
- activation energy
- body tracking

成功标准：

- latent residual 达到相似 body tracking，但更平滑、更少 early termination。

优先级：MUST-RUN after Blocks 1-3 pass

### Block 5：手部噪声是否污染 body latent

Claim：排除 hand 是必要设计，而不是任意选择。

比较：

- latent trained with hand reference
- latent trained without hand reference

指标：

- body tracking
- action smoothness
- prior rollout stability
- latent active dims
- shoulder/elbow jerk around hand spike frames

成功标准：

- without-hand 的 prior rollout 更稳。
- with-hand 在 noisy frames 更容易引入 shoulder/elbow 或 torso artifacts。

优先级：MUST-RUN for method defense

## 13. 执行里程碑

| Milestone | 目标 | 主要任务 | Decision Gate |
|-----------|------|----------|---------------|
| M0 | 冻结数据 | body-only manifest, cache QC, FPS 检查 | cache 可训练 |
| M1 | teacher | body-only PPO tracker | teacher 稳定 |
| M2 | shards | 收集 latent distill dataset | schema 和质量通过 |
| M3 | latent | 训练 posterior/prior/decoder | posterior 和 prior 指标通过 |
| M4 | manifold | prior-only rollout | 短时稳定 |
| M5 | residual | constrained latent residual learning | 优于 direct PPO |
| M6 | ablation | hand/no-hand, KL, latent dim | 支撑核心 claim |

## 14. 推荐运行顺序

### R001：Body-only config smoke test

目的：确认去掉 hand sites 后环境、reward、validation 不崩。

输出：

```text
outputs/body_only/smoke_config_check.json
```

### R002：Retarget cache QC

目的：确认当前 `stage5_10demo_smooth_filtered` 是否可直接作为 body-only 数据。

输出：

```text
outputs/body_only/cache_qc_report.md
```

### R003：Tiny body-only teacher

目的：用小步数验证 reward 和 termination。

预算：

```text
num_envs: 256
total_timesteps: 1M - 5M
```

### R004：Full body-only teacher

目的：训练可用 teacher。

预算：

```text
num_envs: 1024+
total_timesteps: 20M - 100M
```

### R005：Collect teacher shards

目的：收集 latent distillation 数据。

输出：

```text
datasets/distill/body_only_teacher_v1/
```

### R006：Latent overfit one motion

目的：检查 latent trainer 是否能在单条 motion 上过拟合。

成功标准：

- train action MSE 快速下降。
- KL 不立即 collapse。

### R007：Latent train/val split

目的：训练正式 latent prior。

### R008：Prior-only rollout eval

目的：验证 `p(z|s)` 是否可部署。

### R009：Latent residual tiny PPO

目的：验证在 manifold 内学习是否比 direct action 更稳。

### R010：Ablation sweep

目的：支撑论文/报告 claim。

## 15. 文件和代码改动计划

### 15.1 配置文件

新增：

```text
fullbody/config_specific_task/body_only/conf_fullbody_forehandclear_body_teacher.yaml
fullbody/config_specific_task/body_only/conf_fullbody_forehandclear_body_student.yaml
fullbody/config_specific_task/body_only/conf_fullbody_forehandclear_body_latent_residual.yaml
```

### 15.2 数据脚本

新增或扩展：

```text
musclemimic/badminton/scripts/build_body_only_manifest.py
musclemimic/badminton/scripts/evaluate_body_only_cache.py
musclemimic/badminton/scripts/collect_body_only_teacher_dataset.py
```

### 15.3 Latent 训练

新增：

```text
musclemimic/latent_muscle/train_latent_distillation.py
musclemimic/latent_muscle/eval_latent_rollout.py
musclemimic/latent_muscle/checkpoint.py
musclemimic/latent_muscle/config.py
```

### 15.4 Metrics

新增：

```text
musclemimic/utils/body_only_metrics.py
visualize/analyze_body_latent_prior.py
```

### 15.5 Tests

新增：

```text
tests/unit/test_body_only_site_filter.py
tests/unit/test_latent_distillation_dataset.py
tests/unit/test_latent_checkpoint.py
tests/unit/test_body_only_metrics.py
```

## 16. 风险和修复措施

### 风险 1：去掉 hand 后动作变得过于宽松

现象：

- teacher 学会大致站稳，但挥臂链条不明显。

修复：

- 保留 shoulder/elbow。
- 增加 torso-shoulder-elbow 相对关系 reward。
- 加 phase-conditioned shoulder/elbow velocity reward。

### 风险 2：teacher 仍然追踪 noisy retarget artifact

现象：

- body-only tracker 中 torso/shoulder/elbow 出现突然 jerk。

修复：

- 对 source retarget cache 做 body qpos velocity clamp。
- 在 reward 中提高 action rate 和 joint acceleration penalty。
- 剔除问题 frames 或降低其 sampling 权重。

### 风险 3：latent posterior collapse

现象：

- KL 接近 0。
- active latent dims 很少。
- decoder 只靠 state 输出平均动作。

修复：

- KL warmup。
- free-bits。
- 降低 decoder capacity。
- 增加 phase/reference diversity。

### 风险 4：prior 学不好，posterior 好但 prior rollout 崩

现象：

- posterior reconstruction 好。
- prior mean rollout 1 秒内崩。

修复：

- 增加 KL 权重。
- 使用 scheduled posterior-to-prior sampling。
- DAgger 式收集 prior visited states 后 relabel。
- 增加 short rollout consistency loss。

### 风险 5：latent residual 又学出 muscle hacking

现象：

- body tracking 好，但 activation/action_rate 飙升。

修复：

- 限制 residual scale `alpha`。
- 使用 LAB-style bound。
- reward 中加入 activation/action smoothness。
- 直接比较 residual vs direct PPO，若没有优势则不进入主 claim。

### 风险 6：指标不能证明“合理人体动作框架”

现象：

- tracking 数字好，但视频不自然。

修复：

- 加入 qualitative panel。
- 加 foot contact timeline。
- 加 muscle activation heatmap。
- 加 root/torso/leg/arm-chain 分组指标。

## 17. Paper / Report Claim Map

### Claim 1

从 noisy human prior 中训练 body-only denoising teacher，比 full noisy tracking 更适合提取肌骨身体动作框架。

证据：

- body-only tracking 不差。
- smoothness、activation、foot contact 更好。
- hand spike frames 不再污染 shoulder/torso。

### Claim 2

latent prior `p(z|s)` 和 decoder `D(s,z)` 能形成可部署的 body-motion manifold。

证据：

- posterior reconstruction 成功。
- prior-only rollout 短时稳定。
- latent diagnostics 不 collapse。

### Claim 3

后续在该 manifold 内学习，比直接在 high-dimensional muscle action 空间学习更稳。

证据：

- latent residual policy sample efficiency 更高或相当。
- action energy / action rate / termination 更低。
- body tracking 不明显牺牲。

## 18. 最小可交付版本

如果时间有限，最小版本只做：

1. body-only teacher config。
2. teacher rollout collection with body-only reference features。
3. offline latent distillation trainer。
4. posterior reconstruction eval。
5. prior mean rollout eval。
6. hand/no-hand ablation。

不做：

- latent residual PPO
- DAgger prior relabel
- 多动作类别泛化
- 复杂论文图

## 19. 最终 Checklist

- [ ] 数据 namespace 不覆盖已有 baseline。
- [ ] FPS 在 AMASS、GMR、config、render 中一致。
- [ ] hand sites 不进入 body-only reward。
- [ ] hand reference 不进入 posterior reference features。
- [ ] action mask 明确区分 body actuator 和 excluded actuator。
- [ ] teacher 通过 body-only QC。
- [ ] distill shards 记录 `reference_features_dim`。
- [ ] latent trainer 有 train/val/test split。
- [ ] posterior reconstruction 通过。
- [ ] prior rollout 通过。
- [ ] KL/active dims/sigma diagnostics 通过。
- [ ] 与 direct tracker、BC student、with-hand latent 做比较。
- [ ] 所有主结论都有对应指标和视频支持。

## 20. 当前最推荐的下一步

第一步不要直接写 PPO residual，也不要做 LAB policy。先完成：

```text
body-only config + body-only teacher smoke test
```

原因：

如果 teacher 本身不是一个更干净的 body-motion source，后面 latent distillation 学到的只是更复杂的噪声压缩器。teacher 通过后，再收集 shards 和训练 latent prior。

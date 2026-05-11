# SFV Curriculum 方法与 MuscleMimic 配合方案

本文说明 Peng et al. 2018 的 SFV 中 curriculum 的实际含义，并给出与当前 `musclemimic` 已有 curriculum/adaptive sampling 机制的配合方案。

## 0. 信心边界

不能声称对该 curriculum 组合有 100% 信心。SFV 原文的 curriculum 主要是 ASI 自动产生的，而本文建议额外组合 `adaptive_sampling`、`adaptive_termination` 和 `reward_curriculum`。这些机制在工程上合理，但会引入非平稳训练分布、非平稳终止条件和非平稳 reward 标尺。它们必须分阶段启用，并用固定诊断指标评估，否则可能出现“训练 reward 变好但动作覆盖变差”的假进展。

## 1. 论文里是否有独立 curriculum 模块

SFV 论文没有设计一个手工分阶段 curriculum，例如“先学短片段、再学长片段、再调 reward”。论文中最明确的 curriculum 是 ASI 的副作用：

```text
learning an initial state distribution can be interpreted as automatic curriculum generation
```

也就是说，论文的 curriculum 主要来自学习初始状态分布 `rho_omega(s0)`：

- 策略当前能处理、并能产生较高 return 的初始状态，会被提高采样概率。
- 太难、噪声太大、不可恢复的初始状态，会被降低采样概率。
- 随着策略变强，`rho_omega` 会逐渐把训练分布推向更完整、更有挑战的动作状态。

这是一种自动课程生成，而不是人工写死的阶段表。

## 2. 论文中与 curriculum 相关的训练设计

SFV 对比了三种初始化方式：

### 2.1 Fixed State Initialization (FSI)

每个 episode 都从动作开头或固定初始状态开始。

特点：

- 实现最简单。
- 对长时序和高动态动作很差。
- backflip、frontflip、handspring 等动作中，策略很难从开头探索到后半段。

论文结果中 FSI 对动态技能表现最差。

### 2.2 Reference State Initialization (RSI)

每个 episode 从参考轨迹随机帧开始。

特点：

- 这是 DeepMimic 风格训练常用方法。
- 能让策略接触动作中后段，大幅降低长时序探索难度。
- 依赖参考动作质量。对干净 mocap 有效，对视频重建轨迹会受到错误状态和速度抖动影响。

### 2.3 Adaptive State Initialization (ASI)

学习一个初始状态分布，而不是固定或均匀随机。

特点：

- 初始状态分布根据 rollout return 更新。
- 高 return 初始状态概率上升，低 return 初始状态概率下降。
- 对视频重建动作更稳，因为它会自动避开明显不可恢复的重建帧。
- 论文实验中 ASI 在 backflip、cartwheel、frontflip、handspring 等任务上都优于 FSI 和 RSI。

因此，SFV 的 curriculum 不是单独模块，而是：

```text
FSI -> RSI -> ASI
```

这个初始化分布越来越能根据学习进展调整难度。

## 3. 论文中的训练终止与奖励难度

SFV 使用 early termination：

- 如果角色摔倒，例如 torso link 接触地面，则 episode 结束。
- episode 提前结束后，剩余 timestep reward 记为 0。
- 对 rolling 等接触丰富动作会关闭或调整这类终止条件。

这会让 return 自然反映“从某个初始状态开始是否可恢复”。ASI 正是利用这个 return 更新初始状态分布。

奖励方面，SFV 使用固定的 imitation reward 权重：

```text
r = 0.65 * pose
  + 0.10 * velocity
  + 0.15 * end_effector
  + 0.10 * center_of_mass
```

论文没有报告逐步增加 reward 权重的 curriculum。对 MuscleMimic 来说，reward curriculum 是可以额外加的工程增强，但它不是 SFV 原文的核心方法。

## 4. MuscleMimic 当前已有 curriculum 机制

当前项目已经有三类与 curriculum 相关的机制。

### 4.1 Adaptive termination threshold

文件：

```text
musclemimic/algorithms/common/curriculum.py
musclemimic/algorithms/ppo/runner.py
musclemimic/core/terminal_state_handler/enhanced_fullbody.py
```

逻辑：

- 统计 rollout 的 early termination rate。
- 对 early termination rate 做 EMA。
- 如果 EMA 长期低于 `low_band`，说明策略稳定，降低 termination threshold，让任务变难。
- 如果 EMA 高于 `high_band`，说明太难，放宽 threshold，但不会超过初始阈值。

当前配置入口：

```yaml
experiment:
  adaptive_termination:
    enabled: false
    init_threshold: 0.5
    low_band: 0.10
    high_band: 0.20
    adjust_factor: 0.95
    consecutive_k: 5
    min_threshold: 0.1
    ema_alpha: 0.1
```

这与 SFV 的 early termination 很契合：SFV 用摔倒终止来影响 return，MuscleMimic 可以进一步把终止阈值作为难度旋钮。

但 adaptive termination 会改变 done/absorbing 的含义。如果 ASI、adaptive_sampling 和 reward_curriculum 都直接用当前 absorbing 统计，它们看到的是一个随时间变化的失败定义。最低修复：

- 环境可以继续用动态 threshold 控制训练终止。
- 日志和 curriculum scoring 额外计算一套固定阈值 diagnostic failure，例如固定 `mean_site_deviation_threshold = 0.5` 或任务指定值。
- ASI 和 adaptive_sampling 优先使用固定 diagnostic failure 与 tracking metric，而不是只用动态 absorbing。

### 4.2 Reward curriculum

文件：

```text
musclemimic/algorithms/common/curriculum.py
musclemimic/core/reward/trajectory_based.py
```

逻辑：

- 训练初期先降低速度 tracking 权重，减小视频重建速度噪声的影响。
- 当 termination rate 持续低于阈值时，逐步提高 `qvel_w_sum` 和 `root_vel_w_sum`。
- `MimicReward` 已经从 `carry.qvel_w_sum` 和 `carry.root_vel_w_sum` 读取动态权重。

配置入口：

```yaml
experiment:
  reward_curriculum:
    enabled: false
    qvel_w_sum_init: 0.1
    root_vel_w_sum_init: 0.1
    qvel_w_sum_max: 0.4
    root_vel_w_sum_max: 0.4
    eta: 0.02
    ema_alpha: 0.2
    term_rate_threshold: 0.08
    consecutive_k: 5
```

这不是 SFV 原文方法，但对视频重建动作很有意义。视频 pose 的位置通常比速度可靠，速度由帧差分得到，更容易抖。先弱化速度项，等姿态稳定后再增强速度项，是合理的工程 curriculum。

### 4.3 Adaptive trajectory sampling

文件：

```text
musclemimic/algorithms/common/adaptive_sampling.py
musclemimic/algorithms/ppo/runner.py
loco_mujoco/trajectory/handler.py
```

逻辑：

- 按 trajectory 统计 done count 和 early termination count。
- 估计每条轨迹的 early termination rate。
- early termination rate 高的轨迹提高采样权重。
- 使用 `floor_mix` 保证所有轨迹仍有最低采样概率。

配置入口：

```yaml
experiment:
  adaptive_sampling:
    enabled: false
    beta: 0.2
    alpha: 1.0
    floor_mix: 0.1
    ema_done_init: 10.0
    ema_early_init: 5.0
```

这更像 hard-example mining，不是 SFV 的 ASI。它提高难轨迹出现频率，而 ASI 是学习初始状态分布，可能会降低不可恢复状态概率。两者可以组合，但目标不同。

关键风险：如果某条轨迹 early termination 高是因为数据坏、重建错、速度爆炸或坐标不一致，adaptive sampling 会反复采它；ASI 又可能在该轨迹内部避开低回报 phase。两者叠加后，训练可能既浪费样本，又降低完整动作覆盖。

## 5. 与 MuscleMimic 配合的推荐 curriculum 设计

针对 SFV 风格视频模仿，推荐使用四层课程：

```text
Layer 1: trajectory-level sampling
Layer 2: frame-level ASI
Layer 3: termination threshold curriculum
Layer 4: reward weight curriculum
```

这四层不应该默认同时从第 0 步全部打开。推荐以“一个主旋钮 + 固定诊断指标”的方式逐层加入：

- `adaptive_termination` 可以最先开，因为它解决初期全摔导致无信号的问题。
- `adaptive_sampling` 和 `frame-level ASI` 不要同时从零启动，至少先让 baseline policy 产生有意义的 done/return 统计。
- `reward_curriculum` 会改变 ASI 的 score 标尺，建议在 ASI 分布稳定后再开，或者让 ASI 使用与 reward curriculum 解耦的 diagnostic score。

### 5.1 Layer 1：轨迹级采样

使用现有 `adaptive_sampling`。

目标：

- 多训练 early termination 率高的轨迹。
- 防止训练集里容易动作主导 gradient。

建议：

```yaml
adaptive_sampling:
  enabled: true
  beta: 0.2
  alpha: 0.5        # 初期建议别太激进
  floor_mix: 0.2    # 保证动作覆盖
  ema_done_init: 10.0
  ema_early_init: 5.0
```

如果发现采样高度集中在少数坏轨迹，增大 `floor_mix` 或降低 `alpha`。

额外限制：

- 设置每条 trajectory 最大采样概率，例如 `p_i <= 5 / n_traj` 或通过 temperature 控制。
- 对连续多个评估周期仍高 early termination 且 tracking error 异常的轨迹进入 quarantine，不再通过 hard mining 提权，转入数据修复列表。
- adaptive sampling 的统计应使用固定诊断 early termination，而不是被 `adaptive_termination` 动态阈值改变后的唯一指标。

### 5.2 Layer 2：帧级 ASI

新增 frame-level ASI，详见 `asi.md`。

目标：

- 在每条轨迹内部学习哪些 phase 适合作为 episode 起点。
- 避开视频重建中明显错误或速度爆炸的帧。
- 提高中后段动作覆盖率。

建议配置：

```yaml
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

与 `adaptive_sampling` 组合：

```text
先按 adaptive_sampling 选 trajectory，
再按 ASI frame distribution 选起始 bucket。
```

为了避免 ASI 只选择容易 bucket，score 不应只等于 return。建议第一版使用：

```text
score = completed_fraction * mean_step_reward
      + coverage_bonus * unseen_or_low_prob_bucket
      - early_penalty * early_terminated
```

并记录 per-phase coverage。如果某些 bucket 长期采样概率接近 uniform floor，应区分两种情况：

- bucket 数据质量差：进入 data-quality mask 或 reconstruction 修复。
- bucket 只是困难但合理：增加 curriculum support，而不是永久避开。

### 5.3 Layer 3：终止阈值 curriculum

启用已有 `adaptive_termination`。

目标：

- 初期放宽终止，让策略有机会从坏姿态恢复，获得非零学习信号。
- 稳定后逐渐收紧 site deviation threshold，提高跟踪精度。

建议对视频/GMR 轨迹从较宽松阈值开始：

```yaml
adaptive_termination:
  enabled: true
  init_threshold: 1.0
  low_band: 0.08
  high_band: 0.25
  adjust_factor: 0.95
  consecutive_k: 5
  min_threshold: 0.3
  ema_alpha: 0.1
```

如果动作是羽毛球这类快速移动，`min_threshold` 不应太小。过早收紧会导致训练被 early termination 主导。

### 5.4 Layer 4：奖励权重 curriculum

启用已有 `reward_curriculum`。

目标：

- 初期重视相对关键点/姿态。
- 后期加强 joint velocity 和 root velocity，使动作更有动态一致性。

建议：

```yaml
reward_curriculum:
  enabled: true
  qvel_w_sum_init: 0.05
  root_vel_w_sum_init: 0.05
  qvel_w_sum_max: 0.2
  root_vel_w_sum_max: 0.2
  eta: 0.02
  ema_alpha: 0.2
  term_rate_threshold: 0.08
  consecutive_k: 5
```

如果参考速度来自高质量 mocap，可以把 max 提高；如果来自视频重建，速度项不要太大。

reward curriculum 会改变 reward scale，导致 ASI 的 return baseline 变旧。最低修复：

- ASI score 使用归一化或固定 diagnostic reward，不直接用当前 total reward。
- 如果必须用 total reward，当 `qvel_w_sum/root_vel_w_sum` 增长时，对 ASI baseline 使用较快 EMA 或分段重置。
- 记录 reward 权重变化点，validation 曲线按阶段解释，不把不同 reward 标尺下的 training return 直接比较。

## 6. 训练阶段建议

### Stage A：稳定性启动

目标：策略不要一开始全摔。

建议：

```yaml
adaptive_sampling.enabled: false
asi.enabled: false
adaptive_termination.enabled: true
reward_curriculum.enabled: false
```

使用宽松 termination，例如 `init_threshold = 1.0`。

判断进入下一阶段：

- training early termination rate 低于 30% 到 40%。
- validation 能覆盖动作前半段。
- 固定诊断 tracking error 没有持续恶化。

### Stage B：轨迹级 hard mining

目标：让训练关注困难轨迹。

建议：

```yaml
adaptive_sampling.enabled: true
adaptive_sampling.floor_mix: 0.2
adaptive_sampling.alpha: 0.5
```

判断风险：

- 如果 top-k trajectory 权重过高，说明分布塌缩。
- 如果 validation 平均变好但部分动作完全不会，说明覆盖不足。
- 如果 top-k 都是同一批明显坏数据，停止 hard mining，先修数据或 mask。

### Stage C：帧级 ASI

目标：改善每条轨迹内部的 phase 覆盖和可恢复性。

建议：

```yaml
asi.enabled: true
asi.uniform_mix: 0.1
asi.num_buckets: 20
```

判断有效：

- episode coverage 增加。
- 动作中后段 tracking error 下降。
- ASI probability entropy 逐步下降但不接近 0。
- near-terminal bucket 概率没有异常升高。
- per-bucket success 和 per-bucket 采样概率相关，但不是只集中到最短或最容易片段。

### Stage D：加强速度和精度

目标：从“能完成动作”变成“动作动态更像参考”。

建议：

```yaml
reward_curriculum.enabled: true
adaptive_termination.min_threshold: 0.3
```

判断风险：

- `err_joint_vel` 下降但 early termination 飙升，说明速度权重加太快。
- `root_vel_w_sum` 增大后 root drift 变大，说明 root velocity 参考可能有噪声。
- ASI logits 在 reward 权重变化后快速塌缩，说明 score 标尺耦合太强。

## 7. 是否预计有效

### 7.1 对当前 MuscleMimic 会有效的部分

预计最有效的是：

1. **adaptive termination**
   - 已经接入 terminal handler 的 `carry.termination_threshold`。
   - 对初期训练稳定性有直接帮助。

2. **reward curriculum**
   - 已经接入 `MimicReward` 的动态速度权重。
   - 对视频/GMR 重建轨迹尤其合理，因为速度噪声通常大于位置噪声。

3. **frame-level ASI**
   - 当前尚未实现，但和现有 `TrajectoryHandler` 结构高度匹配。
   - 对高动态动作、短视频动作、失败集中在特定 phase 的任务预计有明显收益。

### 7.2 可能无效或有副作用的部分

需要警惕：

- 过强 adaptive sampling 会一直采最坏轨迹，导致整体动作覆盖下降。
- ASI 如果只奖励高 return，可能偏向容易 phase，反而忽略困难 phase。
- termination threshold 收太紧会让奖励信号变稀疏。
- velocity reward 加太快会放大视频重建速度噪声。
- dynamic termination 和 dynamic reward 同时改变，会让 ASI/adaptive sampling 的统计不可比。
- trajectory-level hard mining 可能把样本集中到不可学习的坏轨迹。
- frame-level ASI 可能通过选择动作末尾或短 horizon 片段刷分。

缓解方法：

- 所有采样分布都保留 uniform floor。
- ASI score 中加入 coverage bonus。
- validation 按 trajectory 和 phase 分桶统计，不只看平均 reward。
- 速度权重 max 对视频数据设小一些。
- 使用固定 diagnostic failure/tracking metric 做 curriculum scoring。
- 对 start frame 设置 `min_remaining_steps`，并使用 horizon-normalized score。
- 加数据质量 mask 和 quarantine 机制。

## 8. 配置模板

下面是“全功能模板”，用于说明各模块的相对保守参数。它不应作为第 0 步默认配置；正式训练应优先使用后面的保守首轮模板，再逐个打开模块。

```yaml
experiment:
  adaptive_sampling:
    enabled: true
    beta: 0.2
    alpha: 0.5
    floor_mix: 0.2
    ema_done_init: 10.0
    ema_early_init: 5.0

  adaptive_termination:
    enabled: true
    init_threshold: 1.0
    low_band: 0.08
    high_band: 0.25
    adjust_factor: 0.95
    consecutive_k: 5
    min_threshold: 0.3
    ema_alpha: 0.1

  reward_curriculum:
    enabled: true
    qvel_w_sum_init: 0.05
    root_vel_w_sum_init: 0.05
    qvel_w_sum_max: 0.2
    root_vel_w_sum_max: 0.2
    eta: 0.02
    ema_alpha: 0.2
    term_rate_threshold: 0.08
    consecutive_k: 5

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

如果全功能模板训练不稳定，优先关闭 `reward_curriculum`，保留 `adaptive_termination`。如果动作覆盖不足，增大 `adaptive_sampling.floor_mix` 和 `asi.uniform_mix`。

更保守的首轮模板是：

```yaml
experiment:
  adaptive_sampling:
    enabled: false
  reward_curriculum:
    enabled: false
  adaptive_termination:
    enabled: true
    init_threshold: 1.0
    min_threshold: 0.3
  asi:
    enabled: false
```

只有 baseline 产生稳定 episode 统计后，再依次打开 `adaptive_sampling`、`asi`、`reward_curriculum`。

## 9. 实验对照

建议至少做以下 ablation：

```text
A: baseline random trajectory + random frame
B: A + adaptive_termination
C: B + adaptive_sampling
D: C + frame-level ASI
E: D + reward_curriculum
```

每个实验记录：

- training early termination rate。
- validation early termination rate。
- validation episode coverage。
- per-trajectory success rate。
- per-phase success rate。
- reward components：`reward_qpos`、`reward_qvel`、`reward_rpos`、`reward_root_vel`。
- error metrics：`err_site_abs`、`err_rpos`、`err_joint_pos`、`err_joint_vel`。
- start bucket distribution entropy。
- near-terminal start frequency。
- fixed-threshold diagnostic failure rate。
- data-quality masked/quarantined trajectory 数量。

判断标准：

- 如果 D 相比 C 提高 coverage 且不降低 validation success，说明 ASI 有效。
- 如果 E 降低 velocity error 但 success 不降，说明 reward curriculum 有效。
- 如果 C 提高训练速度但 validation coverage 下降，说明 hard mining 过强。

失败判据也要明确：

- validation coverage 低于 baseline，即使 reward 更高，也判为失败。
- ASI top bucket 集中到最后 10% 动作区间，判为 horizon-cheating。
- 固定诊断 failure rate 不降反升，判为 curriculum 指标失真。
- 某些 trajectory 长期 0 success 且被高频采样，判为数据或任务不可学问题，不继续提高采样权重。

## 10. 关键漏洞与修复措施

| 漏洞 | 后果 | 最低修复 |
|---|---|---|
| 四层 curriculum 同时启动 | 训练分布、终止和 reward 同时非平稳，无法归因 | 分阶段启用；每阶段只打开一个新机制 |
| adaptive termination 改变 failure 定义 | ASI/adaptive sampling 的 early rate 前后不可比 | 增加固定 diagnostic failure metric |
| reward curriculum 改变 reward scale | ASI baseline 和采样 logits 漂移 | ASI 使用归一化/固定 diagnostic score，或在权重变化时重置 baseline |
| hard mining 追逐坏数据 | 样本浪费，策略被不可学轨迹拖垮 | data-quality mask、max sampling cap、quarantine |
| ASI 选择短 horizon 起点作弊 | reward/成功率虚高，动作覆盖下降 | `min_remaining_steps`、horizon-normalized score、near-terminal logging |
| ASI 和 adaptive_sampling 目标相反 | 轨迹级采难样本，帧级避难 phase，造成振荡 | 降低学习率，分阶段打开，或只在 trajectory sampling 稳定后启用 ASI |
| 只看平均 validation | 少数动作完全失败被平均值掩盖 | per-trajectory、per-phase、coverage 指标必须作为 gate |

## 11. 结论

SFV 的 curriculum 核心是 ASI 带来的自动初始状态课程，而不是独立的手写阶段训练。对 MuscleMimic，建议把论文思想落成 frame-level ASI，并与现有 adaptive termination、reward curriculum、adaptive trajectory sampling 组合使用。最可能有效的方向是：用宽松 termination 启动训练，用 trajectory-level sampling 找困难动作，用 frame-level ASI 学每条动作内部的可训练起点，最后逐步增强速度奖励提高动态质量。

修订后的策略更适合被当作第一版 prototype，但不是可以直接全量开启的默认训练方案。可靠执行顺序是：先固定诊断指标，再单独打开一个 curriculum 旋钮，只有通过 coverage 和 fixed-failure gate 后才进入下一阶段。

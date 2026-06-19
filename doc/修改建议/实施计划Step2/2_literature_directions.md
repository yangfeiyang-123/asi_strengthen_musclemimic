# 2. 文献中值得借鉴和修改的方向

本文档整理与“肌骨 full-body 正手高远球动作 + 握拍 + 击球”任务最相关的文献方向，并给出具体应该如何迁移到当前仓库。重点不是罗列论文，而是把论文思想转成你的代码和训练路线中的可执行改动。

---

## 一、核心任务拆解

你的任务不是普通 humanoid 控制，也不是普通刚体球拍击球。它同时包含以下难点：

```text
1. 肌骨 full-body policy：高维 muscle actuation，控制难度高。
2. 已有无拍正手高远球动作：已有 body prior，不能轻易破坏。
3. 手-球拍交互：右手手指、掌心、球拍 handle 之间需要稳定 contact。
4. 球拍-羽毛球击球：需要 stringbed contact、impact timing、拍面法向、拍头速度。
5. 羽毛球飞行：shuttlecock 高阻力、速度快速衰减、落点非普通抛体。
6. 稀疏任务奖励：直接从落点 reward 学习非常困难。
```

因此最合理的文献借鉴方向是：

```text
motion imitation prior
+ human-object interaction contact graph
+ curriculum / teacher-student
+ musculoskeletal control regularization
+ shuttlecock flight modeling
+ impact teacher signal
```

---

## 二、DeepMimic：保留动作模仿，同时加入任务目标

### 文献要点

DeepMimic 的核心思想是：通过 motion imitation objective，让物理角色模仿参考动作；同时可以加入 task objective，让角色完成额外任务。例如角色既模仿动作，又朝指定方向移动或把球扔向目标。

论文：

```text
DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills
Xue Bin Peng, Pieter Abbeel, Sergey Levine, Michiel van de Panne, 2018
https://arxiv.org/abs/1804.02717
```

### 对当前任务的启发

你已经有一个无拍正手高远球策略，这个策略包含了重要的全身动力链：

```text
下肢 / 髋 / 躯干稳定
肩关节旋转
肘 / 前臂 / 腕部配合
正手高远球整体节奏
```

如果后续只用击球 reward，例如：

```text
shuttle over net
landing in back court
```

策略很容易学出投机动作：

```text
身体姿态不自然
手臂突然抽动
靠拍框或非真实接触蹭球
牺牲原来的正手高远球动作来追求落点
```

所以后续训练必须保留 body imitation reward。

推荐 reward 结构：

```text
r_total =
    w_mimic_body     * r_mimic_body
  + w_mimic_sites    * r_mimic_sites
  + w_grip           * r_grip
  + w_impact         * r_impact
  + w_flight         * r_flight
  + w_landing        * r_landing
  - w_effort         * effort_penalty
  - w_slip           * slip_penalty
  - w_penetration    * penetration_penalty
  - w_fall           * fall_penalty
```

其中 `r_mimic_body` 不应在训练初期过快降权。建议：

```text
Stage 1: w_mimic_body 高，w_task 低
Stage 2: w_mimic_body 中高，w_grip 高，w_impact 低
Stage 3: w_mimic_body 中，w_impact 中高
Stage 4: w_mimic_body 中低但不为 0，w_landing 高
```

### 建议在仓库中的修改

1. 在 grip-hold runner 中加入 body mimic reward。
2. 在 static-hit runner 中加入 body mimic reward，而不是只看 shuttle。
3. 对肩、肘、躯干、root 的 imitation 权重高于手指，因为手指需要为握拍适配。
4. right-hand fingers 的 reference tracking 不应该完全来自无拍 motion，而应来自 grip seed / hand-racket target。

推荐分层：

```text
body imitation: root, torso, shoulders, elbows, wrist coarse pose
hand grip imitation: grip seed / target sites
racket task: ghost racket / impact / shuttle
```

---

## 三、AMP：用 motion prior 防止任务 reward 破坏动作风格

### 文献要点

AMP 使用 adversarial motion prior，从非结构化 motion clips 中学习 style reward。策略可以通过简单 task reward 完成任务，同时通过 adversarial prior 保持动作风格自然。

论文：

```text
AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control
Xue Bin Peng, Ze Ma, Pieter Abbeel, Sergey Levine, Angjoo Kanazawa, 2021
https://arxiv.org/abs/2104.02180
```

### 对当前任务的启发

你的击球任务非常容易 reward hacking。比如：

```text
提前把球拍伸到 shuttle 附近
用身体异常姿势换取拍头速度
用拍框碰球也算成功
为了落点破坏高远球动作节奏
```

所以除了 explicit imitation reward，还可以训练一个 motion prior：

```text
D(s_t, s_{t+1}) 判断当前动作是否像 forehand clear reference
```

奖励：

```text
r_style = -log(1 - D(s_t, s_{t+1}))
```

或者先用更简单的 discriminator-free 版本：

```text
r_style_proxy = exp(-body_pose_error) + exp(-body_velocity_error) + exp(-site_error)
```

### 建议在仓库中的修改

短期不建议马上实现完整 AMP，因为你的系统还没有训练闭环。建议分两步：

#### 第一阶段：reference tracking prior

在 dedicated runner 中直接实现：

```text
root pose / velocity tracking
upper-body joint tracking
right shoulder / elbow / wrist site tracking
phase tracking
```

#### 第二阶段：motion prior discriminator

当 static-hit runner 稳定后，再考虑：

```text
收集成功 / reference body transition
训练 discriminator
将 r_style 加入 RL reward
```

这样不会过早增加系统复杂度。

---

## 四、PhysHOI：显式 contact graph 对人-物交互很关键

### 文献要点

PhysHOI 研究 dynamic human-object interaction imitation。它强调人体与物体的耦合很复杂，并提出 contact graph 来显式建模 body part 与 object 的接触关系。论文指出 contact graph reward 对 HOI imitation 很关键。

论文：

```text
PhysHOI: Physics-Based Imitation of Dynamic Human-Object Interaction
Yinhuai Wang et al., 2023
https://arxiv.org/abs/2312.04393
```

### 对当前任务的启发

你的任务是典型 HOI：

```text
右手 ↔ 球拍 handle
球拍 stringbed ↔ shuttlecock
身体 ↔ 地面
```

因此不能只奖励：

```text
racket head near shuttle
```

而应该奖励接触结构：

```text
thumb_pad ↔ handle
index_pad ↔ handle
middle_pad ↔ handle
ring_pad ↔ handle
pinky_pad ↔ handle
palm ↔ handle
stringbed_center ↔ shuttle cork
```

并惩罚：

```text
handle ↔ forearm / torso illegal contact
frame ↔ shuttle fake hit
过深 penetration
contact slip
early contact
late contact
```

### 建议在仓库中的修改

你当前 `RightHandRacketGripEnv` 已经有 filtered contact count、illegal handle contact、max penetration。下一步建议把这部分抽象成 reusable contact graph 模块：

```text
environment/overall_environment/src/contact_graph.py
```

推荐 API：

```python
@dataclass
class GripContactGraph:
    thumb_handle: bool
    index_handle: bool
    middle_handle: bool
    ring_handle: bool
    pinky_handle: bool
    palm_handle: bool
    illegal_handle_contact_count: int
    max_handle_penetration_m: float
    slip_m: float

@dataclass
class HitContactGraph:
    stringbed_shuttle: bool
    frame_shuttle: bool
    rho2: float
    penetration_m: float
    relative_normal_velocity: float
    contact_point_world: np.ndarray
    normal_world: np.ndarray
```

然后 grip env、grip-hold runner、static-hit runner 共用这个模块。

---

## 五、InterMimic：先 perfect，再 scale up

### 文献要点

InterMimic 面向复杂 whole-body human-object interaction。它强调 HOI 难点包括人-物耦合、物体几何变化、MoCap 接触伪影、手部细节不足。它采用 curriculum 思想：先在更理想、更可控的条件下训练 teacher，再扩展到更复杂的交互和更大数据。

论文：

```text
InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions
Sirui Xu et al., 2025
https://arxiv.org/abs/2502.20390
```

### 对当前任务的启发

这与当前任务非常贴合。你现在没有真实球拍轨迹，也没有真实击球数据，如果直接端到端训练：

```text
free racket + free shuttle + high-dimensional muscles + sparse landing reward
```

探索会非常难。

因此应该先构造更容易的 teacher / curriculum：

```text
1. 静态 grip teacher
2. soft-weld racket teacher
3. ghost racket teacher
4. frozen body replay teacher
5. static shuttle contact teacher
6. real shuttle flight task
```

### 建议在仓库中的修改

将 `forehand_clear_static_hit_v1.yaml` 中的 curriculum 真正实现为 runner scheduler：

```text
physics_chain_validation
static_grip_stabilizer
swing_disturbance_grip
hit_and_over_net
high_clear_depth
```

每一阶段设置进入下一阶段的指标，而不是只按训练步数推进。

推荐 gating metrics：

```text
static_grip_stabilizer:
  contact_count >= 4
  grip_slip_m <= 0.01
  max_penetration_m <= 0.003

swing_disturbance_grip:
  racket_drop_rate <= 5%
  grip_slip_m <= 0.02

hit_and_over_net:
  valid_stringbed_contact_rate >= 40%
  over_net_rate >= 20%

high_clear_depth:
  opponent_back_rate >= 20%
  out_rate <= 30%
```

---

## 六、MyoSuite / MuscleMimic：肌骨系统需要稳定 curriculum 和生物力学约束

### 文献要点

MyoSuite 提供了 contact-rich musculoskeletal motor control benchmark，强调肌骨控制中的 proprioception、contact-rich manipulation、muscle dynamics、non-stationary conditions 等挑战。

论文：

```text
MyoSuite -- A contact-rich simulation suite for musculoskeletal motor control
Vittorio Caggiano et al., 2022
https://arxiv.org/abs/2205.13600
```

MuscleMimic 则强调 full-body musculoskeletal motion imitation 的规模化训练、SMPL retargeting、GPU 并行和肌肉驱动策略的重要性。

论文：

```text
Towards Embodied AI with MuscleMimic: Unlocking full-body musculoskeletal motor learning at scale
Chengkun Li et al., 2026
https://arxiv.org/abs/2603.25544
```

### 对当前任务的启发

你的系统不是刚体 humanoid，而是肌肉驱动。肌肉系统会带来：

```text
activation dynamics
force-length-velocity properties
高维 action
延迟与耦合
energy / effort 问题
手部肌肉控制复杂
```

所以训练时应该避免过大 residual 和突然 reward 切换。

### 建议在仓库中的修改

#### 1. 对 residual action 加小幅限制

尤其在 wrist / forearm 解冻阶段：

```python
residual = residual_scale(phase, stage) * policy_output
```

建议 scale：

```text
stage1 fingers: 1.0
stage2 wrist: 0.2 -> 0.5
stage3 forearm: 0.1 -> 0.3
stage4 shoulder: 0.05 -> 0.2
```

#### 2. 加 muscle effort 和 smoothness penalty

```text
r_effort = -mean(a^2)
r_smooth = -mean((a_t - a_{t-1})^2)
```

#### 3. 加 activation / tendon safety diagnostics

如果环境能读 muscle activation / force，建议记录：

```text
mean_activation
max_activation
activation_smoothness
muscle_force_outlier_rate
```

#### 4. 不要过早解冻全身

推荐解冻顺序：

```text
right hand fingers
right wrist
right forearm
right elbow / shoulder small residual
```

不要一开始让 full-body policy 全部 trainable。

---

## 七、KINESIS / negative mining：失败状态重采样适合你的任务

### 文献方向

KINESIS 这类肌骨 imitation / RL 方法的一个重要启发是：复杂肌骨任务不能只依赖均匀采样 reference motion，而应重视失败状态、困难状态和 negative examples。

相关方向：

```text
KINESIS: musculoskeletal imitation learning / negative mining
```

### 对当前任务的启发

你的失败会高度集中在特定时刻：

```text
挥拍加速最大时球拍掉落
击球前 0.1 秒手指打滑
impact phase 错过 shuttle
contact 在拍框而不是 stringbed
球过网但落点太短
身体为了击球失稳
```

这些状态比普通 reference reset 更有训练价值。

### 建议在仓库中的修改

实现 failure-type ASI hard-state buffer：

```python
class HardStateBuffer:
    def add(state, phase, failure_type, diagnostics): ...
    def sample(batch_size, failure_type_weights): ...
```

failure type：

```text
racket_drop
grip_slip
illegal_penetration
miss_shuttle
frame_hit
no_net_clearance
short_landing
out_of_bounds
body_fall
```

训练采样比例：

```text
初期：80% reference reset, 20% hard states
中期：50% reference reset, 50% hard states
后期：30% reference reset, 70% hard states
```

注意 hard-state replay 要保留随机性，避免过拟合同一批失败片段。

---

## 八、羽毛球空气阻力：不能用普通抛体替代完整 flight reward

### 文献要点

羽毛球飞行高度依赖空气阻力。最新 shuttlecock velocity decay 研究报告了羽毛球速度在飞行中快速指数衰减；羽毛球轨迹应按高阻力 projectile 处理，而不是普通无阻力抛体。

论文：

```text
Shuttlecock velocity decay after smash and slice shots in badminton
Eric Collet, 2026
https://arxiv.org/abs/2601.01412
```

### 对当前任务的启发

如果你用普通抛体模型反推：

```text
desired outgoing velocity -> landing point
```

会严重高估 shuttle 的飞行距离，导致策略学到错误的出球速度和方向。

### 建议在仓库中的修改

#### 短期 proxy

早期可以先用简化 drag：

```text
F_drag = -k * |v| * v
```

或直接用速度指数衰减近似：

```text
v(t + dt) = v(t) * exp(-lambda * dt)
```

#### 中期 flight reward

把 reward 分成：

```text
r_outgoing_velocity
r_net_clearance
r_apex_height
r_landing_region
```

不要一开始只用 landing reward。

#### 长期 domain randomization

随机化：

```text
shuttle mass
drag coefficient
stringbed restitution
impact normal noise
```

这样策略对模拟误差更稳。

---

## 九、事件相机 / 高速视频 impact teacher：可作为未来数据增强方向

### 文献要点

有最新研究使用同步事件相机估计羽毛球 smash 的 impact time、impact location 和 shuttlecock speed。研究报告了 impact time、impact location、speed 的自动估计流程。

论文：

```text
Automated Estimation of Impact Time, Impact Location, and Shuttlecock Speed in Badminton Smashes Using Event Cameras
Yudai Washida et al., 2026
https://arxiv.org/abs/2605.28011
```

### 对当前任务的启发

如果你后续能从真实高远球视频或事件相机中估计：

```text
impact time
impact location on racket face
racket face ellipse
post-impact shuttle speed
```

就可以替换当前 `impact_target.py` 中基于右手位置的 pseudo impact target。

### 建议在仓库中的修改

先把 `impact_target.py` 的输出 schema 设计得更通用：

```python
@dataclass
class ImpactTeacher:
    impact_phase: float
    impact_time_s: float | None
    impact_point_world: np.ndarray
    racket_face_normal_world: np.ndarray
    racket_head_velocity_world: np.ndarray
    outgoing_shuttle_velocity_world: np.ndarray | None
    confidence: float
    source: str  # pseudo_hand, ghost_racket, video, event_camera
```

这样未来可以无缝替换 teacher source。

---

## 十、综合建议：文献思想对应代码修改表

| 文献方向 | 应用于当前任务 | 对应代码修改 |
|---|---|---|
| DeepMimic | 保留无拍正手高远球动作先验 | grip/static-hit runner 中加入 body imitation reward |
| AMP | 防止击球 reward hacking | 后期加入 motion prior / style reward |
| PhysHOI | 显式建模手-拍、拍-球 contact | 新增 reusable contact_graph.py |
| InterMimic | 先易后难，teacher-student/curriculum | 把 YAML curriculum 变成 runner scheduler |
| MyoSuite | 肌骨 contact-rich 控制要重视扰动与安全 | 加 effort/smoothness/activation diagnostics |
| MuscleMimic | full-body muscle policy 迁移需保留 motion prior | frozen body + small residual，不要直接全量解冻 |
| Shuttlecock drag | 羽毛球不能用普通抛体 | 加 drag / velocity decay flight model |
| Event camera impact estimation | impact teacher 可由视频/传感器校准 | 扩展 ImpactTeacher schema |

---

## 十一、优先级建议

当前不建议马上实现完整 AMP 或复杂视频 teacher。优先级应该是：

```text
P0: training scene + frozen body replay + grip-hold runner
P1: contact graph module + swing disturbance grip
P2: ghost racket teacher + static-hit runner
P3: shuttle drag / rebound / landing reward
P4: AMP / video impact teacher / advanced ASI
```

原因是：没有训练闭环时，复杂文献方法只会增加工程复杂度；先让最小任务能跑起来，后续再加高级方法。

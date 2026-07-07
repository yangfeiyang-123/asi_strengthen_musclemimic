# 2. 文献中值得借鉴的方向

> 目标：把物理角色运动模仿、肌骨控制、人-物交互、羽毛球生物力学和 MuJoCo 约束机制中的可借鉴思想，转化为本项目可执行的修改方向。

---

## 0. 总体结论

你的任务不是单纯的“让手抓住球拍”，也不是单纯的“让球拍碰到羽毛球”。它同时包含：

```text
1. 肌骨全身动作模仿
2. 右手细粒度握拍
3. 手-球拍接触稳定性
4. 球拍-羽毛球瞬时碰撞
5. 羽毛球高阻力飞行
6. 正手高远球的动力链约束
7. 稀疏任务奖励下的 curriculum / ASI
```

因此最值得借鉴的文献方向可以归纳为：

```text
DeepMimic / motion imitation:
  保留原始正手高远球动作质量，不要被击球 reward 带偏。

AMP / motion prior:
  用动作风格先验防止策略 reward hacking。

PhysHOI / InterMimic:
  显式建模人-物接触图，采用 teacher/curriculum 处理复杂 HOI。

MyoSuite / KINESIS / MuscleMimic:
  肌骨系统需要 curriculum、negative mining、GPU-parallel imitation 和可验证的生理合理性。

Badminton biomechanics / shuttle aerodynamics:
  击球任务必须尊重正手高远球的近端到远端动力链和羽毛球高阻力飞行。

MuJoCo equality / weld:
  训练初期可以用 soft weld 降低自由接触探索难度，再逐渐退火到纯 contact。
```

---

## 1. DeepMimic：模仿 reward 和任务 reward 必须并存

### 1.1 相关思想

DeepMimic 的核心思想是：用参考动作定义物理角色的运动风格和技能形态，同时允许加入 task objective，使角色在保持动作质量的同时完成目标任务。

论文明确强调可以将 motion-imitation objective 与 task objective 结合，例如让角色既模仿动作又完成指定目标。

参考：

- DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills
- https://arxiv.org/abs/1804.02717
- https://dl.acm.org/doi/10.1145/3197517.3201311

### 1.2 对本项目的启发

你的基础策略已经会做“无球拍正手高远球动作”。新增任务是：

```text
拿球拍、保持握拍、击中羽毛球、让羽毛球过网并落到后场。
```

不应该把训练目标完全切换为：

```text
只奖励球飞得远 / 击中球
```

否则策略很可能学出：

```text
身体姿态不自然
手臂乱甩
躯干过度扭曲
用拍框/异常接触蹭球
靠数值漏洞把球打出去
```

推荐 reward 结构：

```python
reward = (
    w_mimic * r_body_mimic
  + w_style * r_motion_prior
  + w_grip * r_grip_stability
  + w_racket * r_racket_trajectory
  + w_impact * r_valid_impact
  + w_flight * r_shuttle_flight
  - w_effort * cost_muscle_effort
  - w_slip * cost_grip_slip
  - w_fall * cost_fall
)
```

### 1.3 落地建议

#### 建议 1：冻结或半冻结 body policy

训练初期不要 full-body 全量微调。使用：

```text
π_body: frozen
π_grip: trainable
π_residual: small, phase-gated, trainable
```

只在击球前后打开 residual。

#### 建议 2：分阶段降低 mimic 权重

```text
Stage 1: w_mimic = 1.0, w_hit = 0.0
Stage 2: w_mimic = 1.0, w_hit = 0.2
Stage 3: w_mimic = 0.7, w_hit = 0.5
Stage 4: w_mimic = 0.5, w_hit = 1.0
Stage 5: w_mimic = 0.3, w_hit = 1.0, 解冻少量 wrist/forearm/shoulder
```

#### 建议 3：使用 phase-conditioned mimic

正手高远球不是全程都需要强约束。可以按 phase 设置不同 mimic 权重：

```text
准备期 / 引拍期：强 mimic，确保姿态稳定。
击球窗口：允许 wrist/forearm 小幅 residual。
随挥期：恢复 mimic，防止动作崩坏。
```

---

## 2. AMP：用 motion prior 防止 reward hacking

### 2.1 相关思想

AMP 使用 adversarial motion prior，从动作数据中学习 style reward，让物理角色在完成任务的同时保持自然运动风格，而不必为每个动作手写复杂 imitation reward。

参考：

- AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control
- https://arxiv.org/abs/2104.02180
- https://dl.acm.org/doi/10.1145/3450626.3459670

### 2.2 对本项目的启发

你的任务里存在强 reward hacking 风险：

```text
1. 击球 reward 稀疏且高权重。
2. 球拍是外部物体，接触复杂。
3. 肌骨模型有大量肌肉 actuator，探索空间大。
4. 如果只看 shuttle landing，策略可能牺牲动作自然性。
```

AMP 的启发是：可以从已有正手高远球动作片段中训练一个 discriminator：

```text
D(s_t, s_{t+1}) -> 当前运动是否像正手高远球
```

然后加入：

```python
r_amp = -log(1 - D(s_t, s_{t+1}))
```

或者使用更稳定的 least-squares GAN / hinge loss 风格 reward。

### 2.3 简化版落地方案

短期不一定要完整实现 AMP。可以先实现一个“motion prior critic”：

```python
input = [
    root orientation velocity,
    torso orientation velocity,
    shoulder/elbow/wrist qpos/qvel,
    racket-free version phase,
    selected sites velocity,
]
output = probability_like_forehand_clear
```

训练数据：

```text
positive: 原始 forehand_clear reference / base policy successful rollout
negative: 当前 posttrain 中 fall / miss / abnormal jerk / racket drop 前后的片段
```

reward：

```python
r_prior = clip(logit_prior, -5, 5)
```

### 2.4 Codex 任务

新增模块：

```text
BadmintonMimic/prior/forehand_motion_prior.py
```

最小实现：

```python
class ForehandMotionPrior(nn.Module):
    def forward(self, features):
        return logits
```

新增数据导出：

```text
outputs/posttrain/.../motion_prior/positive_segments.npz
outputs/posttrain/.../motion_prior/negative_segments.npz
```

新增 reward hook：

```python
r_motion_prior = prior_reward(prior_model, obs_features)
```

验收：

```text
1. 能从 reference motion 导出 positive segments。
2. 能从失败 rollout 导出 negative segments。
3. prior forward 不影响 env step 稳定性。
4. 开关 reward.motion_prior.weight=0 时行为与旧环境一致。
```

---

## 3. PhysHOI：接触图比单个距离 reward 更重要

### 3.1 相关思想

PhysHOI 针对动态人-物交互 imitation，指出难点在于身体与物体之间的复杂 coupling，并引入 contact graph 显式建模 body-object 接触关系。论文摘要中强调 contact graph reward 对精确 HOI imitation 很关键。

参考：

- PhysHOI: Physics-Based Imitation of Dynamic Human-Object Interaction
- https://arxiv.org/abs/2312.04393
- https://wyhuai.github.io/physhoi-page/

### 3.2 对本项目的启发

握拍击球的关键不是“手离球拍近”，而是：

```text
哪些手部 geoms 与 handle geoms 接触？
接触是否在正确部位？
接触是否稳定？
接触是否造成非法 penetration？
球拍是否相对手掌滑动？
击球时是否是 stringbed 接触 shuttle，而不是拍框或手指接触？
```

因此应该把 contact graph 当作 observation 和 reward 的核心，而不是只用 site distance。

### 3.3 建议的 contact graph

#### Hand-handle graph

```text
nodes:
  palm
  thumb_pad
  index_pad
  middle_pad
  ring_pad
  pinky_pad
  handle_top
  handle_mid
  handle_bottom

edges:
  palm -> handle_mid
  thumb_pad -> handle_top
  index_pad -> handle_top
  middle_pad -> handle_mid
  ring_pad -> handle_mid/bottom
  pinky_pad -> handle_bottom
```

每条 edge 记录：

```python
edge_feature = [
    active_contact,
    penetration_depth,
    normal_force_proxy,
    tangential_relative_velocity,
    distance_to_target_contact_site,
]
```

#### Racket-shuttle graph

```text
nodes:
  stringbed_center
  stringbed_normal
  frame_edge
  shuttle_head
  shuttle_skirt

edges:
  stringbed_center -> shuttle
  frame_edge -> shuttle
```

希望奖励：

```text
stringbed_center contact high
frame_edge contact low / penalty
shuttle contact in impact phase high
out-of-phase contact penalty
```

### 3.4 落地 reward

```python
r_contact_graph = (
    w_thumb * active_thumb_handle
  + w_index * active_index_handle
  + w_middle * active_middle_handle
  + w_palm * active_palm_handle
  + w_coverage * min(contact_count / target_count, 1.0)
  - w_illegal * illegal_handle_contact_count
  - w_penetration * max_handle_penetration
  - w_slip * tangential_slip
)
```

击球接触：

```python
r_stringbed = (
    w_active * stringbed_shuttle_contact
  + w_center * exp(-rho2 / sigma)
  + w_normal * alignment(contact_normal, stringbed_normal)
  + w_phase * exp(-phase_error^2 / sigma_phase)
  - w_frame * frame_shuttle_contact
)
```

### 3.5 Codex 任务

新增：

```text
environment/overall_environment/src/contact_graph.py
```

接口：

```python
@dataclass
class ContactGraphReport:
    hand_handle_edges: dict[str, ContactEdge]
    stringbed_shuttle: ContactEdge | None
    illegal_contacts: list[IllegalContact]
    max_penetration_m: float
    mean_slip_mps: float


def compute_contact_graph(model, data, model_map) -> ContactGraphReport:
    ...
```

验收：

```text
1. reset 后 contact graph finite。
2. no-contact case 返回 active=false，不报错。
3. 手-柄合法接触和非法接触可区分。
4. stringbed 与 frame contact 可区分。
5. reward term 能从 ContactGraphReport 计算。
```

---

## 4. InterMimic：复杂 HOI 适合 “perfect first, then scale up”

### 4.1 相关思想

InterMimic 处理复杂人-物交互，指出 HOI imitation 难在人体-物体耦合、物体几何差异、MoCap 接触伪影和手部细节不足。其关键思想是 curriculum：先训练 subject-specific teacher，再 distill 到 student，最后 RL fine-tune。

参考：

- InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions
- https://arxiv.org/abs/2502.20390
- https://sirui-xu.github.io/InterMimic/

### 4.2 对本项目的启发

你的任务同样有 “imperfect reference” 问题：

```text
原始动作没有球拍。
原始动作没有羽毛球。
原始动作没有手-柄真实 contact。
原始动作没有真实 impact label。
```

所以不要一开始就让 policy 从 free-contact 中自己探索。应当先造一个“更完美的 teacher”：

```text
1. 静态 IK 得到合理握拍姿态。
2. soft weld 让球拍跟手，得到稳定挥拍轨迹。
3. ghost racket 根据无球拍动作生成伪球拍轨迹。
4. teacher policy 先跟踪 ghost racket。
5. student / residual policy 再学习真实 contact 和 shuttle flight。
```

### 4.3 建议 curriculum

```text
Stage 0: static grip IK teacher
  输出：right_hand_racket_grip_seed.json

Stage 1: soft-weld grip teacher
  球拍半绑定到手掌，训练手指和腕部不破坏握拍。

Stage 2: ghost racket tracking teacher
  在完整挥拍动作中跟踪 ghost racket pose / head velocity。

Stage 3: static shuttle impact
  羽毛球固定在击球点，只有满足 stringbed contact + phase window 才 release。

Stage 4: shuttle flight target
  奖励过网、apex、后场落点。

Stage 5: weak residual full-body fine-tune
  解冻 wrist/forearm/shoulder 的小幅 residual，不破坏 body prior。
```

### 4.4 Codex 任务

新增：

```text
experiments/posttrain/curricula/forehand_clear_racket_hit.yaml
```

字段：

```yaml
curriculum:
  - name: static_grip_ik
    train: false
    output: right_hand_racket_grip_seed.json
  - name: soft_weld_grip_hold
    train: true
    policies: [grip]
    weld_strength: strong
  - name: ghost_racket_tracking
    train: true
    policies: [grip, residual]
    ghost_weight: 1.0
  - name: static_shuttle_hit
    train: true
    policies: [layered]
    shuttle_mode: pre_impact_freeze_release
  - name: high_clear_flight
    train: true
    policies: [layered]
    landing_target: opponent_backcourt
```

验收：

```text
每个 stage 都能单独 dry-run。
每个 stage 输出 metrics.json。
后一 stage 读取前一 stage checkpoint。
失败时生成 failure_examples.npz。
```

---

## 5. MyoSuite / MyoChallenge：肌骨任务需要 curriculum

### 5.1 相关思想

MyoSuite 是面向 musculoskeletal motor control 的 contact-rich simulation suite。MyoChallenge / Neuron 2024 的 Baoding balls 任务展示了用 curriculum-based RL 控制 39 块手部肌肉完成复杂手内操作的可行性。

参考：

- MyoSuite papers: https://sites.google.com/view/myosuite/papers
- MyoSuite GitHub: https://github.com/MyoHub/myosuite
- Acquiring musculoskeletal skills with curriculum-based reinforcement learning, Neuron 2024: https://www.sciencedirect.com/science/article/pii/S0896627324006500
- PubMed: https://pubmed.ncbi.nlm.nih.gov/39357519

### 5.2 对本项目的启发

右手握拍是典型肌骨手控制问题。复杂点包括：

```text
1. 多肌肉冗余。
2. 接触丰富且不连续。
3. 手指姿态可行域窄。
4. 稳定握拍要抵抗球拍惯性和挥拍扰动。
5. 过强 effort 会导致不自然肌肉激活。
```

因此训练不应只设一个最终 reward，而应分课程：

```text
pose -> contact -> anti-slip -> perturbation -> swing disturbance -> hit impact
```

### 5.3 建议指标

每个 curriculum stage 都应有明确指标：

| Stage | 指标 | 目标 |
|---|---|---:|
| Static grip | mean grip-site error | < 2 cm |
| Static grip | contact count | >= 4 |
| Static grip | max penetration | < 3 mm |
| Gravity hold | racket drift | < 1 cm / 2 s |
| Perturbation | recovery | < 2 cm / 0.5 s |
| Swing hold | drop rate | < 5% |
| Static hit | valid stringbed contact | > 60% |
| Flight | over-net rate | > 50% |
| High clear | backcourt landing | > 25% initially, then increase |

### 5.4 Codex 任务

在 trainer 中加入 curriculum gate：

```python
@dataclass
class StageGate:
    metric: str
    threshold: float
    window: int
    direction: Literal["above", "below"]


def should_advance(metrics_history, gate: StageGate) -> bool:
    ...
```

不要只按 training step 前进，而是：

```text
只有连续 N 个 eval window 达标，才进入下一 stage。
```

---

## 6. KINESIS / MuscleMimic：利用 negative mining 和肌骨 motion prior

### 6.1 相关思想

KINESIS 是用于 physiologically plausible musculoskeletal motor control 的 model-free motion imitation 框架。相关资料提到其可以在肌骨模型中学习 imitation prior，并通过 negative mining 增强 robustness，再 fine-tune 到下游任务。

参考：

- KINESIS arXiv: https://arxiv.org/abs/2503.14637
- KINESIS GitHub: https://github.com/amathislab/Kinesis

MuscleMimic 是本项目基础所基于的方向：JAX-based motion imitation benchmark，面向 muscle-actuated models，支持 MyoBimanualArm 和 MyoFullBody，并强调 GPU-parallel training、collision support、GMR retargeting 等。

参考：

- MuscleMimic arXiv: https://arxiv.org/abs/2603.25544
- MuscleMimic GitHub: https://github.com/amathislab/musclemimic
- 当前 fork README: https://github.com/yangfeiyang-123/asi_strengthen_musclemimic

### 6.2 对本项目的启发

你的失败样本非常有价值。不要只看最终 reward，而要按失败类型采样：

```text
racket_dropped
finger_slip
illegal_penetration
missed_shuttle
bad_contact_frame
shuttle_hits_net
shuttle_lands_own_side
body_fall
muscle_effort_explosion
```

这些失败前 10~30 帧可以进入 ASI buffer：

```python
failure_state_buffer.add(
    state=state_t_minus_k,
    failure_type="racket_dropped",
    phase=phase,
    contact_graph=contact_graph,
)
```

下一轮训练从这些 hard states 初始化，重点修复失败。

### 6.3 Codex 任务

新增：

```text
environment/overall_environment/src/failure_mining.py
```

接口：

```python
@dataclass
class FailureEvent:
    kind: str
    step: int
    phase: float
    state_snapshot: dict[str, np.ndarray]
    metrics: dict[str, float]

class FailureReplayBuffer:
    def add_episode(self, episode_trace): ...
    def sample_initial_state(self, kind: str | None = None): ...
```

训练时：

```text
initial_state_source:
  70% reference reset / normal rollout
  20% failure buffer
  10% randomized perturbation states
```

验收：

```text
1. failure buffer 可保存/加载 npz。
2. 每类 failure 有计数。
3. reset 可以从 failure state 恢复。
4. hard-state mining 开关关闭时训练行为不变。
```

---

## 7. 羽毛球空气阻力：不能用普通抛体近似替代最终评估

### 7.1 相关思想

羽毛球飞行具有强空气阻力。研究指出 shuttlecock 的空气阻力与速度平方相关，击球角度和击球强度都会影响轨迹。

参考：

- A Study of Shuttlecock's Trajectory in Badminton: https://pmc.ncbi.nlm.nih.gov/articles/PMC3761540/
- Aerodynamics of badminton shuttlecocks: https://www.sciencedirect.com/science/article/abs/pii/S0889974613000315
- The physics of badminton: https://rainbow.ldeo.columbia.edu/~alexeyk/Papers/Cohen_2015_New_J._Phys._17_063001.pdf

### 7.2 对本项目的启发

正手高远球的目标不是“球飞得越快越好”，而是：

```text
高弧线
过网
到对方后场
尽可能不出界
```

如果 early-stage 只用普通无阻力 projectile 反推目标速度，会导致：

```text
出球速度 / 角度 proxy 错误
策略过度追求水平速度
高远球轨迹不像真实羽毛球
```

### 7.3 建议两阶段 flight modeling

#### Stage A：解析 proxy

早期训练可以用简化 drag proxy：

```python
F_drag = -0.5 * rho_air * Cd * A * ||v|| * v
```

然后根据目标落点近似反推 desired outgoing velocity。

这只是训练辅助 reward，不是最终评价。

#### Stage B：MuJoCo / custom shuttle flight

最终用真实 simulation：

```text
MuJoCo contact impulse + shuttle freejoint + aerodynamic force hook
```

reward 根据真实 flight 计算：

```python
r_flight = (
    w_cross_net * crossed_net
  + w_apex * exp(-(apex_height - target_apex)^2 / sigma)
  + w_depth * backcourt_region_reward
  + w_inbounds * inbounds
)
```

---

## 8. 羽毛球生物力学：不要让 wrist/finger residual 代替全身动力链

### 8.1 相关思想

羽毛球 power strokes 的研究强调肩部旋转、前臂旋转等上肢动力学因素。综述中也提到精英击球速度来自 sequential proximo-distal joint action chain，并与球拍 deflection 有关。

参考：

- Biomechanical principles applied to badminton power strokes: https://ojs.ub.uni-konstanz.de/cpa/article/download/2233/2089/
- The Science of Badminton Game: https://www.worldbadminton.com/reference/research/documents/The_Science_of_Badminton_Game.pdf
- Biomechanical analysis of clear strokes in badminton: https://projekter.aau.dk/projekter/files/42678547/articledone.pdf

### 8.2 对本项目的启发

如果只训练右手 residual，策略可能会把任务变成：

```text
手腕/手指乱甩来补偿整个击球速度
```

这会违背正手高远球的 biomechanical structure。更合理的是：

```text
body policy 负责动力链主体；
grip policy 负责球拍不掉；
phase-gated residual 只做击球窗口微调；
```

### 8.3 建议 reward

增加 kinetic chain consistency：

```python
r_chain = (
    w_shoulder * exp(-||shoulder_vel - ref_shoulder_vel||^2 / sigma)
  + w_forearm * exp(-||forearm_pronation_timing - ref||^2 / sigma)
  + w_wrist * exp(-||wrist_timing - ref||^2 / sigma)
  + w_sequence * phase_order_bonus
)
```

phase order bonus 可以简单定义为：

```text
shoulder peak velocity occurs before forearm/wrist peak velocity
forearm/wrist peak velocity occurs before or near impact
racket head velocity peaks near impact
```

不需要一开始就做得很复杂，但至少要监控：

```text
shoulder angular velocity peak phase
elbow angular velocity peak phase
wrist angular velocity peak phase
racket head speed peak phase
impact phase
```

---

## 9. MuJoCo weld / equality：soft-weld annealing 是合理工程技巧

### 9.1 相关文档

MuJoCo 的 `equality/weld` 会把两个 body 连接起来，移除两者之间的相对自由度，但像 MuJoCo 里的其他约束一样是 soft constraint。

参考：

- MuJoCo XML reference, equality/weld: https://mujoco.readthedocs.io/en/3.2.6/XMLreference.html
- MuJoCo stable XML reference: https://mujoco.readthedocs.io/en/stable/XMLreference.html

### 9.2 对本项目的启发

一开始就训练“纯 contact 手握 free racket”探索难度过高。建议课程：

```text
Stage A: strong soft weld，球拍几乎跟手。
Stage B: weaker weld，手指 contact 开始承担稳定。
Stage C: weak weld + perturbation。
Stage D: remove weld，只靠 contact grip。
Stage E: swing + impact。
```

这与仓库已有 `grip_strategy.md` 的建议一致：先静态参考姿态，再 grip-hold training，先 soft weld，再 contact/perturbation，最后接入 arm/wrist swing。

参考：

- https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/environment/holdracket/docs/grip_strategy.md#L241-L252
- https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/environment/holdracket/docs/grip_strategy.md#L288-L298

### 9.3 实现建议

XML 中定义多个 weld preset，或通过不同 XML 文件区分：

```xml
<equality>
  <weld name="palm_racket_weld_strong" body1="right_palm" body2="racket" solref="0.002 1" solimp="0.95 0.99 0.001"/>
</equality>
```

不同 stage 使用不同 solref/solimp。注意：MJX 中动态开关 equality 可能需要验证，最好先采用“每个 stage 一个 XML”而不是运行时频繁修改约束。

---

## 10. 总结：文献到工程的映射表

| 文献/方向 | 关键思想 | 本项目落地 |
|---|---|---|
| DeepMimic | imitation + task objective | 保留 body mimic reward，同时加 grip/hit/flight reward |
| AMP | learned style reward | 加 forehand motion prior，防止击球 reward 带偏 |
| PhysHOI | contact graph reward | 手-柄、拍面-球接触图作为 obs/reward |
| InterMimic | perfect first, then scale up | 先 teacher / ghost / soft weld，再真实 contact |
| MyoSuite / Neuron | curriculum for muscle hand | pose -> contact -> perturb -> swing -> hit |
| KINESIS | negative mining / downstream fine-tune | failure-state ASI buffer |
| MuscleMimic | scalable muscle imitation | 旧 body policy 作为 frozen prior |
| Shuttle aerodynamics | strong drag, nonlinear flight | flight reward 不用普通抛体替代最终评价 |
| Badminton biomechanics | proximal-distal chain | 不让手腕 residual 代替全身动力链 |
| MuJoCo weld | soft equality constraint | soft-weld annealing 降低探索难度 |

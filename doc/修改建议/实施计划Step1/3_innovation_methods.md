# 3. 我认为最值得尝试、并且可能有效的创新方案

> 目标：提出适合“肌骨正手高远球从无拍动作扩展到拿拍击球”的创新训练方案，并且写成可交给 Codex 拆任务实现的工程规格。

---

## 0. 总体方案

我最推荐的组合不是单一技巧，而是一套分层、分阶段、可验证的系统：

```text
Frozen body prior
+ Right-hand grip policy
+ Phase-gated residual policy
+ Ghost racket teacher
+ Soft-weld annealing
+ Contact graph reward
+ Static shuttle freeze-release
+ Shuttle flight curriculum
+ Contact-viability ASI hard-state mining
```

核心目标是：

```text
利用你已经训练好的无球拍正手高远球动作，
但不要让模型直接从 free-contact + sparse hit reward 中盲目探索。
```

如果把策略结构写成公式：

```python
a_body = π_body(obs_body)                  # frozen old policy

a_grip = π_grip(obs_grip)                  # trainable right-hand grip

gate = phase_gate(phase)                   # only active near impact

a_res = π_residual(obs_task, phase)         # small residual for wrist/forearm/fingers

a_full = router.merge(
    body_action=a_body,
    grip_action=a_grip + gate * a_res,
)
```

训练逻辑：

```text
先让球拍稳定在手里；
再让球拍轨迹像由原始动作推断出的 ghost racket；
再让球拍在正确相位击中静态羽毛球；
最后让羽毛球真实飞行并落到后场。
```

---

## 1. 创新方案一：三层策略结构

### 1.1 动机

旧策略已经会做正手高远球大动作，但它：

```text
1. 没有拿球拍。
2. 可能是 disable_fingers=True 训练出来的。
3. 没有手指握拍控制。
4. 没有球拍/羽毛球 observation。
```

如果直接 fine-tune 全策略，会有两个问题：

```text
1. 灾难性遗忘：原来学好的全身挥拍动作被破坏。
2. 探索过难：手指、球拍、羽毛球同时学，reward 很稀疏。
```

所以建议分成三层：

```text
π_body: 冻结的全身正手高远球策略，负责主体动力链。
π_grip: 右手握拍策略，负责手指/拇指/掌心接触稳定。
π_residual: 相位门控小残差，负责击球窗口微调球拍姿态和速度。
```

### 1.2 控制范围

推荐 actuator ownership：

```yaml
body_policy:
  owns:
    - lower_body
    - torso
    - neck
    - left_arm
    - right_shoulder
    - right_elbow
  trainable: false

grip_policy:
  owns:
    - right_hand_fingers
  trainable: true

residual_policy:
  owns:
    stage1:
      - right_hand_fingers
    stage2:
      - right_hand_fingers
      - right_wrist
    stage3:
      - right_hand_fingers
      - right_wrist
      - right_forearm
    stage4:
      - right_hand_fingers
      - right_wrist
      - right_forearm
      - small_right_shoulder_residual
  trainable: true
  residual_scale:
    right_hand_fingers: 0.25
    right_wrist: 0.15
    right_forearm: 0.10
    right_shoulder: 0.05
```

注意：不要让 residual 一开始控制整条右臂，否则它会学会绕过 body prior。

### 1.3 Phase gate

定义：

```python
def phase_gate(phase, center, width, sharpness=10.0):
    # center = pseudo impact phase
    # width  = allowed impact neighborhood
    x = abs(phase - center) / max(width, 1e-6)
    return np.exp(-sharpness * x * x)
```

推荐：

```yaml
phase_gate:
  center_source: impact_target.impact_phase
  width_stage1: 0.20
  width_stage2: 0.12
  width_stage3: 0.08
  min_gate_outside_window: 0.0
```

### 1.4 Reward

```python
reward = (
    w_mimic * r_body_mimic
  + w_grip * r_grip
  + w_ghost * r_ghost_racket
  + w_impact * r_impact
  + w_flight * r_flight
  - w_residual * ||a_residual||^2
  - w_delta * ||a_residual_t - a_residual_t_minus_1||^2
)
```

### 1.5 实现任务

新增：

```text
environment/overall_environment/src/layered_policy.py
```

建议接口：

```python
@dataclass
class LayeredPolicyOutput:
    body_action: np.ndarray
    grip_action: np.ndarray
    residual_action: np.ndarray
    full_action: np.ndarray
    phase_gate: float

class LayeredForehandPolicy:
    def __init__(self, body_policy, grip_policy, residual_policy, router, config):
        ...

    def act(self, obs, deterministic: bool = False) -> LayeredPolicyOutput:
        ...
```

验收：

```text
1. body policy 可 freeze。
2. grip/residual 有独立 optimizer。
3. full action size 与 full model actuator size 一致。
4. phase gate outside impact window 时 residual 接近 0。
5. 100-step rollout 不出现 NaN。
```

---

## 2. 创新方案二：Ghost Racket Teacher

### 2.1 动机

原始动作没有球拍，但你需要让模型学会：

```text
球拍头位置
球拍头速度
拍面朝向
击球相位
```

直接从真实 shuttle contact 学这些会很稀疏。Ghost racket teacher 的思想是：

```text
从已有无球拍动作推断一个“虚拟球拍轨迹”，
先让真实 free racket 跟踪它，
再逐渐转向真实击球 reward。
```

仓库已有 `impact_target.py`，它已经根据右手位置和 racket length 构造 virtual head，并估计 impact phase。建议把它扩展成完整的 ghost racket trajectory，而不只是单个 impact target。

参考：

- https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/environment/overall_environment/src/impact_target.py#L648-L708

### 2.2 Ghost racket 定义

```python
@dataclass
class GhostRacketTrajectory:
    phase: np.ndarray                         # shape (T,)
    grip_pos_world: np.ndarray                # shape (T, 3)
    grip_quat_world: np.ndarray               # shape (T, 4)
    head_pos_world: np.ndarray                # shape (T, 3)
    head_vel_world: np.ndarray                # shape (T, 3)
    stringbed_normal_world: np.ndarray         # shape (T, 3)
    impact_frame: int
    impact_phase: float
```

构造逻辑：

```text
1. 从 reference motion 取 right hand / palm frame。
2. 定义 grip transform：racket handle 相对 palm 的固定变换。
3. 根据 racket length 计算 head_pos。
4. 根据 head_pos 时间差分计算 head_vel。
5. 根据 palm / forearm / velocity 估计 stringbed normal。
6. 选择候选 impact frame。
```

### 2.3 Reward

```python
def ghost_racket_reward(real, ghost, phase):
    ghost_t = interpolate(ghost, phase)

    r_grip = exp(-norm(real.grip_pos - ghost_t.grip_pos)**2 / sigma_grip)
    r_head = exp(-norm(real.head_pos - ghost_t.head_pos)**2 / sigma_head)
    r_vel = exp(-norm(real.head_vel - ghost_t.head_vel)**2 / sigma_vel)
    r_face = exp(-angle(real.normal, ghost_t.normal)**2 / sigma_face)

    return w_grip*r_grip + w_head*r_head + w_vel*r_vel + w_face*r_face
```

推荐初始权重：

```yaml
ghost_reward:
  grip_pose: 2.0
  head_pos: 4.0
  head_velocity: 2.0
  stringbed_normal: 2.0
```

### 2.4 权重退火

```text
Stage 1: ghost_weight = 1.0, shuttle_weight = 0.0
Stage 2: ghost_weight = 0.7, shuttle_weight = 0.2
Stage 3: ghost_weight = 0.4, shuttle_weight = 0.6
Stage 4: ghost_weight = 0.1, shuttle_weight = 1.0
Stage 5: ghost_weight = 0.0 or only diagnostic
```

### 2.5 风险和修正

| 风险 | 表现 | 修正 |
|---|---|---|
| palm frame 不等于真实握拍 frame | 球拍轨迹偏移 | 用 grip IK seed 校正 palm-to-handle transform |
| impact phase 错 | 击球窗口错过 | 可视化每条 motion 的 pseudo impact frame |
| stringbed normal 估计差 | 拍面朝向不合理 | 加 learnable/calibrated normal offset |
| ghost reward 太强 | 只跟踪 ghost，不学真实击球 | 后期退火 ghost weight |

### 2.6 实现任务

新增：

```text
environment/overall_environment/src/ghost_racket.py
```

接口：

```python
def build_ghost_racket_trajectory(reference_motion, grip_seed, config) -> GhostRacketTrajectory:
    ...

def interpolate_ghost(trajectory, phase) -> GhostRacketFrame:
    ...

def compute_ghost_reward(model, data, trajectory, phase, weights) -> dict[str, float]:
    ...
```

新增 CLI：

```bash
python -m environment.overall_environment.src.ghost_racket \
  --reference forehand_clear/stage5_10demo/video1_lower_body_full_poses \
  --grip-seed outputs/right_hand_racket_grip/reference/right_hand_racket_grip_seed.json \
  --out outputs/posttrain/ghost_racket/video1_ghost.npz \
  --render outputs/posttrain/ghost_racket/video1_ghost.mp4
```

验收：

```text
1. 生成 npz。
2. 生成可视化视频。
3. impact phase 在挥拍最快区域附近。
4. head velocity finite。
5. ghost reward 可独立单元测试。
```

---

## 3. 创新方案三：Soft-weld annealing

### 3.1 动机

纯 contact 握拍是高难度接触控制：

```text
手指肌肉维度高。
接触切换不连续。
球拍 free body 容易掉。
挥拍时惯性大。
RL reward 稀疏。
```

MuJoCo 的 weld equality 是 soft constraint，可以先把球拍“软绑定”到手掌，再逐渐减弱绑定，让手指 contact 接管稳定性。

MuJoCo 文档说明 `equality/weld` 会把两个 body 连接起来，移除相对自由度，并且像 MuJoCo 其他约束一样是 soft constraint。

参考：

- https://mujoco.readthedocs.io/en/3.2.6/XMLreference.html
- https://mujoco.readthedocs.io/en/stable/XMLreference.html

### 3.2 课程

```text
Weld Stage 0: inspection only
  强约束，检查球拍 frame 与手掌 frame 是否正确。

Weld Stage 1: strong soft weld
  球拍几乎跟手。训练 finger pose 和 contact coverage。

Weld Stage 2: medium soft weld
  降低 weld，加入小扰动。训练 no-slip。

Weld Stage 3: weak weld
  手指 contact 承担主要稳定。训练 swing disturbance。

Weld Stage 4: no weld
  纯 contact。进入 static-hit。
```

### 3.3 XML 策略

建议每个 stage 一个 XML，而不是运行时动态改 equality：

```text
overall_badminton_training_weld_strong.xml
overall_badminton_training_weld_medium.xml
overall_badminton_training_weld_weak.xml
overall_badminton_training_contact_only.xml
```

原因：

```text
1. 便于复现实验。
2. 避免 MJX / MuJoCo runtime 修改 equality active 的不确定性。
3. 每个 stage 的 contact / solref / solimp 可单独测试。
```

### 3.4 Reward 配合

不要在 strong weld 时给很高手-柄 contact reward，否则模型可能依赖 weld 而不学 contact。推荐：

```yaml
stage_strong_weld:
  reward:
    racket_pose: 8.0
    hand_pose: 4.0
    contact: 0.5
    no_slip: 0.5

stage_medium_weld:
  reward:
    racket_pose: 4.0
    hand_pose: 3.0
    contact: 2.0
    no_slip: 2.0

stage_weak_weld:
  reward:
    racket_pose: 2.0
    hand_pose: 2.0
    contact: 4.0
    no_slip: 6.0

stage_contact_only:
  reward:
    racket_pose: 1.0
    hand_pose: 1.0
    contact: 6.0
    no_slip: 8.0
```

### 3.5 实现任务

新增 XML generator：

```text
environment/overall_environment/src/make_training_scene.py
```

接口：

```python
def generate_training_scene(base_xml, weld_mode, contact_mode, out_xml):
    ...
```

YAML：

```yaml
scene_variants:
  strong_weld:
    xml: environment/overall_environment/assets/overall_badminton_training_weld_strong.xml
    weld: strong
  medium_weld:
    xml: environment/overall_environment/assets/overall_badminton_training_weld_medium.xml
    weld: medium
  weak_weld:
    xml: environment/overall_environment/assets/overall_badminton_training_weld_weak.xml
    weld: weak
  contact_only:
    xml: environment/overall_environment/assets/overall_badminton_training_contact_only.xml
    weld: none
```

验收：

```text
1. 每个 XML 可被 MjModel.from_xml_path 加载。
2. reset 后 qpos/qvel finite。
3. strong weld 下球拍 2 秒内不掉。
4. contact_only 下 contact geoms 启用。
5. debug report 输出 weld constraint names 和 contact pairs。
```

---

## 4. 创新方案四：Phase-gated contact reward

### 4.1 动机

如果全程奖励球拍靠近羽毛球，策略会 reward hack：

```text
提前把球拍停在球附近。
用不自然姿态等待球。
用拍框或手碰球。
破坏正手高远球动作，只追求接触。
```

所以所有击球相关 reward 必须相位门控。

### 4.2 三段 reward

#### Pre-impact

目标：准备正确击球条件，不要求碰球。

```python
r_pre = (
    w_grip * r_grip_stability
  + w_face * r_stringbed_face_target
  + w_head_pos * r_racket_head_near_shuttle
  + w_head_vel * r_head_velocity_toward_shuttle
)
```

只在：

```text
phase in [impact_phase - 0.15, impact_phase]
```

开启。

#### Impact

目标：正确相位、正确区域、正确方向地碰球。

```python
r_impact = (
    w_contact * stringbed_contact
  + w_center * exp(-rho2 / sigma_rho)
  + w_phase * exp(-(phase - impact_phase)^2 / sigma_phase)
  + w_closing * positive_closing_velocity
  + w_normal * contact_normal_alignment
  - w_frame * frame_contact
)
```

只在：

```text
phase in [impact_phase - 0.08, impact_phase + 0.08]
```

开启。

#### Post-impact

目标：球的飞行结果。

```python
r_post = (
    w_cross_net * crossed_net
  + w_apex * high_clear_apex_reward
  + w_depth * opponent_backcourt_reward
  + w_inbounds * inbounds_reward
)
```

只在 release 后开启。

### 4.3 Static shuttle freeze-release

仓库已有 static-hit YAML 的 release 条件：

```yaml
shuttle:
  mode: pre_impact_freeze_release
  release:
    require_stringbed_contact: true
    phase_tolerance: 0.08
    max_rho2: 1.0
    require_closing_velocity: true
```

参考：

- https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/experiments/posttrain/forehand_clear_static_hit_v1.yaml#L544-L557

`StaticForehandClearEnv` 也已有 release condition，但缺 reward/done。建议保留这个机制并补齐 reward。

### 4.4 实现任务

新增：

```text
environment/overall_environment/src/phase_reward.py
```

接口：

```python
@dataclass
class PhaseRewardTerms:
    pre_impact: float
    impact: float
    post_impact: float
    penalties: float


def compute_phase_gated_hit_reward(state, contact_graph, shuttle_state, config) -> PhaseRewardTerms:
    ...
```

验收：

```text
1. phase outside impact window 时 stringbed contact 不给正奖励，甚至可罚。
2. valid contact in window 给明显正奖励。
3. frame contact 不等于 stringbed contact。
4. missed impact window 会 terminated。
5. release condition 与 reward condition 使用同一套 contact report。
```

---

## 5. 创新方案五：Contact-viability critic + ASI hard-state mining

### 5.1 动机

训练失败最有信息量的时刻通常发生在失败前：

```text
球拍刚要滑出手。
击球前拍面偏了。
手指 contact coverage 下降。
身体开始失稳。
球拍头速度方向错了。
```

如果每次都从 episode 开始训练，策略很少反复看到这些“临界状态”。ASI hard-state mining 可以主动从失败前状态重新初始化。

### 5.2 Failure taxonomy

定义失败类型：

```python
FAILURE_TYPES = [
    "body_fall",
    "racket_drop",
    "grip_slip",
    "illegal_penetration",
    "missed_impact_window",
    "wrong_contact_surface",
    "low_racket_head_speed",
    "shuttle_hit_net",
    "shuttle_own_side_landing",
    "shuttle_out",
]
```

### 5.3 Viability critic

训练一个 critic 预测未来短时间内是否会失败：

```python
V_contact = P(success within next 0.5s | current state)
```

输入：

```python
features = [
    phase,
    grip_contact_count,
    per_finger_contact_flags,
    max_penetration,
    grip_slip,
    racket_pose_relative_to_palm,
    racket_head_velocity,
    stringbed_normal,
    shuttle_relative_position,
    body_balance_features,
]
```

输出：

```python
p_no_drop
p_valid_impact
p_over_net
```

可以先不用 neural critic，只用 heuristic viability score：

```python
score = (
    + contact_count_score
    - slip_score
    - penetration_score
    - racket_pose_error
    - face_error
    - body_fall_risk
)
```

### 5.4 ASI buffer

```python
@dataclass
class HardState:
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    phase: float
    failure_type: str
    steps_before_failure: int
    metrics: dict[str, float]
```

采样策略：

```yaml
initial_state_sampling:
  reference_reset: 0.60
  successful_midphase: 0.15
  hard_failure_states: 0.20
  randomized_perturbation: 0.05
```

课程推进时逐渐提高 hard-state 比例。

### 5.5 实现任务

新增：

```text
environment/overall_environment/src/asi_hard_state.py
```

接口：

```python
class HardStateBuffer:
    def add_trace(self, episode_trace, failure_event): ...
    def sample(self, failure_type: str | None = None) -> HardState: ...
    def save(self, path): ...
    def load(self, path): ...
```

训练 loop 增加：

```python
if done and failure_event is not None:
    hard_state_buffer.add_trace(trace, failure_event)

if rng.random() < hard_state_prob:
    env.reset_from_state(hard_state_buffer.sample())
else:
    env.reset()
```

验收：

```text
1. 能保存 failure states。
2. reset_from_state 后 MuJoCo state finite。
3. 每类 failure 计数可见。
4. hard-state sampling 可关闭。
5. 启用后不会改变 observation/action shape。
```

---

## 6. 创新方案六：解析 shuttle target proxy，再切到真实碰撞

### 6.1 动机

真实击球 reward 稀疏：

```text
要在正确相位碰到球，
还要拍面方向正确，
还要出球速度合适，
还要过网落后场。
```

早期可以用解析 proxy 降低难度：从目标落点反推出 desired outgoing velocity，再奖励球拍头速度和拍面方向。

### 6.2 Proxy

```python
desired_outgoing_velocity = solve_shuttle_velocity_to_target(
    impact_pos,
    target_landing_pos,
    drag_model,
)
```

然后奖励：

```python
r_proxy = (
    w_speed * exp(-||v_racket_head - v_desired||^2 / sigma_v)
  + w_face * exp(-angle(stringbed_normal, desired_direction)^2 / sigma_n)
  + w_impact_pos * exp(-||stringbed_center - shuttle||^2 / sigma_pos)
)
```

### 6.3 注意

不能最终用普通抛体作为评价，因为羽毛球强阻力很明显。研究表明 shuttlecock air drag 与速度平方有关，轨迹受击球角度和力量影响。

参考：

- https://pmc.ncbi.nlm.nih.gov/articles/PMC3761540/
- https://www.sciencedirect.com/science/article/abs/pii/S0889974613000315

### 6.4 课程

```text
Stage Proxy-1:
  只奖励 desired racket head velocity，不要求真实球飞。

Stage Proxy-2:
  加 static shuttle release，但 landing reward 权重低。

Stage Real-1:
  用真实 shuttle flight，proxy 权重减半。

Stage Real-2:
  只用真实 flight metrics，proxy 仅做 diagnostic。
```

---

## 7. 创新方案七：动力链一致性 regularizer

### 7.1 动机

正手高远球不是纯手腕动作。羽毛球生物力学研究强调肩、前臂、腕和球拍的序列协同。若只让 residual 控制手腕/手指，策略可能会靠局部抖动击球。

参考：

- https://ojs.ub.uni-konstanz.de/cpa/article/download/2233/2089/
- https://www.worldbadminton.com/reference/research/documents/The_Science_of_Badminton_Game.pdf

### 7.2 指标

每个 episode 记录：

```python
metrics = {
    "phase_peak_shoulder_angvel": ...,
    "phase_peak_elbow_angvel": ...,
    "phase_peak_wrist_angvel": ...,
    "phase_peak_racket_head_speed": ...,
    "impact_phase": ...,
}
```

期望：

```text
shoulder peak <= forearm/wrist peak <= racket head peak near impact
```

### 7.3 Reward

```python
r_chain_timing = exp(-max(0, shoulder_peak_phase - wrist_peak_phase)**2 / sigma)
              * exp(-(racket_peak_phase - impact_phase)**2 / sigma_impact)
```

简化版：先不作为训练 reward，只作为 validation metric。

---

## 8. 推荐组合实验

### 8.1 Ablation matrix

```text
A0: 直接 full-body fine-tune + hit reward
A1: frozen body + grip policy
A2: A1 + soft-weld annealing
A3: A2 + ghost racket teacher
A4: A3 + phase-gated static hit reward
A5: A4 + shuttle flight reward
A6: A5 + ASI hard-state mining
A7: A6 + partial unfreeze wrist/forearm/shoulder
```

### 8.2 主要指标

```text
racket_drop_rate
mean_grip_slip_m
contact_count_mean
max_penetration_m
valid_stringbed_contact_rate
wrong_surface_contact_rate
impact_phase_error
racket_head_speed_at_impact
over_net_rate
opponent_backcourt_landing_rate
body_fall_rate
body_mimic_error
muscle_effort_mean
residual_action_norm
```

### 8.3 成功判据

初期不要要求 landing rate 很高。建议阶段目标：

```text
Stage grip:
  racket_drop_rate < 5%
  mean_grip_slip_m < 0.05
  contact_count >= 4

Stage static hit:
  valid_stringbed_contact_rate > 50%
  wrong_surface_contact_rate < 20%

Stage over-net:
  over_net_rate > 30%

Stage high-clear:
  opponent_backcourt_landing_rate > 15% 起步，随后提高
```

---

## 9. 最推荐先做的创新 MVP

如果只做一个最小可行创新版本，我建议：

```text
MVP = Frozen body policy
    + right-hand grip policy
    + LayeredActuatorRouter
    + soft-weld medium scene
    + ghost racket head tracking
    + phase-gated static shuttle release
```

不要一开始就实现 full AMP / full ASI / full shuttle aerodynamics。MVP 要解决三个问题：

```text
1. 旧 body policy 能在带球拍 scene 里 replay。
2. 球拍能稳定留在手中。
3. 在正确相位能让 stringbed 接触静态羽毛球。
```

一旦 MVP 成功，再加：

```text
shuttle flight
hard-state mining
partial unfreeze
motion prior
```

---

## 10. 参考链接

- DeepMimic: https://arxiv.org/abs/1804.02717
- AMP: https://arxiv.org/abs/2104.02180
- PhysHOI: https://arxiv.org/abs/2312.04393
- InterMimic: https://arxiv.org/abs/2502.20390
- KINESIS: https://arxiv.org/abs/2503.14637
- MuscleMimic: https://arxiv.org/abs/2603.25544
- MyoSuite: https://github.com/MyoHub/myosuite
- Neuron curriculum musculoskeletal hand: https://www.sciencedirect.com/science/article/pii/S0896627324006500
- Shuttle trajectory: https://pmc.ncbi.nlm.nih.gov/articles/PMC3761540/
- Badminton power strokes: https://ojs.ub.uni-konstanz.de/cpa/article/download/2233/2089/
- Science of Badminton Game: https://www.worldbadminton.com/reference/research/documents/The_Science_of_Badminton_Game.pdf
- MuJoCo equality/weld: https://mujoco.readthedocs.io/en/3.2.6/XMLreference.html

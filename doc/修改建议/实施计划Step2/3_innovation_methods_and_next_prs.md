# 3. 对任务值得创新且可能有效的方法 + 4. 下一轮最优先做的 4 个 PR

本文档把当前任务可尝试的创新方法和下一轮工程 PR 计划合并说明。目标是让 Codex 能直接根据文档拆任务、改代码、写测试、跑 smoke check。

---

## 一、当前最适合的创新方向总览

你的任务最难的地方是：

```text
无拍正手高远球动作 -> 带拍握拍 -> 静态击球 -> 过网 -> 后场高远球落点
```

这是一个从 motion imitation 到 contact-rich object interaction 再到 projectile task 的连续迁移问题。最有效的方法不应该是“直接端到端训练完整击球”，而应该是把难点拆开。

推荐创新组合：

```text
1. Inspection scene / Training scene 双 XML
2. Soft-weld annealing，从绑住球拍逐渐过渡到真实握拍
3. Frozen body policy + grip policy + phase-gated residual
4. Ghost racket teacher，把无拍动作转成球拍伪监督
5. Contact-graph reward，显式奖励手-拍和拍-球接触
6. Outgoing velocity proxy，先学击球条件再学真实落点
7. Failure-type ASI hard-state mining
8. Grip viability critic，自适应控制 residual 解冻
```

这些方法的共同目标是：

```text
降低探索难度
保留已有无拍正手高远球动作
逐步引入真实接触
减少 reward hacking
提高训练稳定性
```

---

## 二、创新方法 1：Inspection scene / Training scene 双 XML + soft-weld annealing

### 2.1 问题

当前 `overall_badminton_scene.xml` 是 inspection scene：

```text
actuation disabled
person-racket contact excluded
racket 是 free body
racket 没有 welded 到手
```

这对可视化很有用，但不能直接用于训练握拍或击球。

如果直接在 free racket + free hand contact + high-dimensional muscle actuation 上训练，探索难度极高，早期几乎都会掉拍或失稳。

### 2.2 方法

新增 training scene，并使用 soft-weld annealing：

```text
Stage A: racket 近似 soft-weld 到 palm，几乎不会掉
Stage B: 降低 weld stiffness，hand-handle contact 承担部分约束
Stage C: 加入扰动和挥拍惯性
Stage D: 关闭 weld，只靠手指 / 掌心 contact 握拍
Stage E: 加入 shuttle impact
```

### 2.3 代码设计

新增或修改：

```text
environment/overall_environment/src/build_overall_environment.py
```

建议接口：

```python
def build_overall_scene(
    output_xml: str | Path | None = None,
    *,
    grip_seed: str | Path | None = None,
    mode: str = "inspection",  # inspection | training
    enable_actuation: bool | None = None,
    enable_person_racket_contact: bool | None = None,
    enable_soft_weld: bool = False,
    soft_weld_solref: str = "0.02 1",
    soft_weld_solimp: str = "0.8 0.95 0.001",
) -> Path:
    ...
```

建议输出：

```text
environment/overall_environment/assets/overall_badminton_scene.xml
environment/overall_environment/assets/overall_badminton_training_scene.xml
```

### 2.4 soft-weld 实现建议

MuJoCo 中可以用 equality weld 或 tendon / constraint 辅助。早期可以先做一个简单版本：

```xml
<equality>
  <weld name="racket_palm_soft_weld"
        body1="thirdmc_r"
        body2="overall_racket"
        solref="0.02 1"
        solimp="0.8 0.95 0.001" />
</equality>
```

后续通过不同 XML 或 runtime model 参数调整 constraint softness。

### 2.5 验收指标

```text
training XML 中 actuation enabled
training XML 中 Full Body 与 overall_racket 不再 body-level exclude
right-hand geoms 与 handle bevel geoms 可以 contact
soft weld 可以通过 config 开关
reset 后 qpos/qvel finite
100 steps zero ctrl / pose_servo 不 NaN
```

---

## 三、创新方法 2：Frozen body policy + grip policy + phase-gated residual

### 3.1 问题

你已有的无拍正手高远球策略很有价值，不应该直接被全量 finetune 破坏。但击球又需要额外控制右手、手腕、前臂，甚至少量肩肘修正。

### 3.2 方法

使用三层策略：

```text
π_body: 已训练好的无拍正手高远球策略，冻结
π_grip: 右手握拍策略，主要控制手指
π_residual: 相位门控的小幅 wrist/forearm/finger residual
```

动作合成：

```python
a_body = π_body(obs_body)
a_grip = π_grip(obs_grip)
a_residual = phase_gate(phase) * π_residual(obs_task)

a_full = router.merge(
    body_action=a_body,
    grip_action=a_grip + a_residual,
)
```

其中 `phase_gate` 只在击球窗口附近打开：

```text
impact_phase - 0.12 <= phase <= impact_phase + 0.08
```

### 3.3 为什么有效

优点：

```text
保留已有全身动作
避免 residual 在整段 motion 中乱改身体
把握拍和击球微调从全身运动中拆出来
降低探索空间
适配当前 ActionManifest / LayeredActuatorRouter 设计
```

### 3.4 代码设计

新增：

```text
environment/overall_environment/src/frozen_body_policy.py
environment/overall_environment/src/body_obs_adapter.py
environment/overall_environment/src/phase_gate.py
BadmintonMimic/scripts/run_forehand_clear_grip_hold.py --stage train
```

核心 class：

```python
class FrozenBodyPolicy:
    @classmethod
    def load(cls, checkpoint_path: Path, manifest: ActionManifest): ...
    def act(self, body_obs: np.ndarray) -> np.ndarray: ...

class BodyObsAdapter:
    def __init__(self, obs_manifest, normalizer): ...
    def build_obs(self, model, data, reference_state, phase) -> np.ndarray: ...

class PhaseGate:
    def __call__(self, phase: float, impact_phase: float) -> float: ...
```

### 3.5 训练顺序

```text
Stage 1: 只训练 right_hand_fingers residual
Stage 2: 加入 right_wrist residual，小幅 scale
Stage 3: 加入 right_forearm residual，小幅 scale
Stage 4: 击球任务稳定后，少量解冻 shoulder/elbow residual
```

不要一开始解冻全身。

---

## 四、创新方法 3：Ghost Racket Teacher

### 4.1 问题

原始 reference motion 没有球拍轨迹。直接训练真实球拍击球会非常稀疏，因为策略需要同时探索：

```text
如何握拍
球拍应该在哪里
什么时候击球
拍面朝向哪里
拍头速度多大
```

### 4.2 方法

把无拍动作转换成 ghost racket trajectory：

```text
ghost_grip_pose(t)
ghost_stringbed_center(t)
ghost_racket_head_velocity(t)
ghost_stringbed_normal(t)
ghost_impact_phase
```

当前 `impact_target.py` 已经有基础：它根据右手位置和 forward axis 估计 virtual racket head，并选择 peak virtual racket speed 作为 impact frame。下一步应该把“单点 target”扩展成“完整 trajectory teacher”。

### 4.3 训练 reward

早期 reward：

```text
r_ghost_pos = exp(-||real_stringbed_center - ghost_stringbed_center||^2 / sigma_pos^2)
r_ghost_vel = exp(-||real_head_velocity - ghost_head_velocity||^2 / sigma_vel^2)
r_ghost_normal = exp(-angle(real_stringbed_normal, ghost_normal)^2 / sigma_ang^2)
r_impact_phase = exp(-|phase - ghost_impact_phase|^2 / sigma_phase^2)
```

后期逐渐降低 ghost reward，增加真实 shuttle reward：

```text
Stage early: ghost 80%, shuttle 20%
Stage middle: ghost 50%, shuttle 50%
Stage late: ghost 20%, shuttle 80%
```

### 4.4 代码设计

新增：

```text
environment/overall_environment/src/ghost_racket_teacher.py
```

数据结构：

```python
@dataclass
class GhostRacketFrame:
    phase: float
    grip_pos_root: np.ndarray
    grip_quat_root: np.ndarray
    stringbed_center_root: np.ndarray
    stringbed_normal_root: np.ndarray
    racket_head_velocity_root: np.ndarray

@dataclass
class GhostRacketTrajectory:
    frames: list[GhostRacketFrame]
    impact_phase: float
    impact_frame: int
    confidence: float
```

API：

```python
def build_ghost_racket_trajectory(reference_motion, racket_geometry, grip_transform): ...
def sample_ghost_at_phase(trajectory, phase): ...
def compute_ghost_reward(real_racket_state, ghost_frame): ...
```

### 4.5 验收标准

```text
给定 5 帧 mock right_hand_pos，可以输出 finite trajectory
impact_phase 在 [0, 1]
stringbed_center 连续
racket_head_velocity finite
reward 对真实 racket 贴近 ghost 时更高
```

---

## 五、创新方法 4：Contact-graph reward

### 5.1 问题

只奖励距离会导致 reward hacking：

```text
球拍靠近 shuttle 但不合理接触
拍框碰球也算成功
手和 handle 穿透
用身体其他部位卡住球拍
```

### 5.2 方法

建立 hand-racket 和 racket-shuttle contact graph。

推荐 graph：

```text
thumb_pad_handle
index_pad_handle
middle_pad_handle
ring_pad_handle
pinky_pad_handle
palm_handle
stringbed_shuttle
frame_shuttle
illegal_handle_body
```

### 5.3 reward 设计

握拍：

```text
+ target finger-handle contacts
+ no slip
+ stable palm-to-grip transform
- illegal handle contact
- excessive penetration
```

击球：

```text
+ stringbed-shuttle contact
+ contact inside stringbed ellipse
+ closing velocity
+ desired normal alignment
- frame contact
- early / late contact
```

### 5.4 代码设计

新增：

```text
environment/overall_environment/src/contact_graph.py
```

并让这些模块复用它：

```text
RightHandRacketGripEnv
GripHoldRunner
StaticForehandClearEnv
StaticHitRunner
```

---

## 六、创新方法 5：Outgoing velocity proxy -> real shuttle flight

### 6.1 问题

直接用后场落点 reward 太稀疏。早期训练可能几乎没有任何成功落点。

### 6.2 方法

先训练 impact 条件：

```text
racket head velocity
stringbed normal
impact point
outgoing shuttle velocity proxy
```

然后再切换到真实 shuttle flight。

### 6.3 proxy 设计

根据目标落点，反推期望出球速度 `v_desired`。注意羽毛球高 drag，不要用普通无阻力抛体。

早期可以用简化模型：

```text
v(t + dt) = v(t) * exp(-lambda * dt)
```

或：

```text
F_drag = -k * |v| * v
```

reward：

```text
r_outgoing = exp(-||v_shuttle_after_impact - v_desired||^2 / sigma_v^2)
r_normal = exp(-angle(stringbed_normal, desired_normal)^2 / sigma_n^2)
r_center = exp(-rho2 / sigma_rho^2)
```

### 6.4 切换策略

```text
Stage 1: no shuttle, ghost racket
Stage 2: shuttle contact + outgoing velocity proxy
Stage 3: over-net reward
Stage 4: apex + landing region reward
Stage 5: full high-clear landing objective
```

---

## 七、创新方法 6：Failure-type ASI hard-state mining

### 7.1 问题

失败集中在关键时刻：

```text
挥拍加速最大时掉拍
击球前手指打滑
impact 窗口错过 shuttle
碰到拍框
球没过网
过网但太短
身体失稳
```

普通从 reference 起点 rollout 会浪费大量训练时间。

### 7.2 方法

存储失败前的 hard states，下轮训练优先从这些状态开始。

数据结构：

```python
@dataclass
class HardState:
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    phase: float
    failure_type: str
    diagnostics: dict
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

采样策略：

```text
初期: 80% reference reset, 20% hard state
中期: 50% reference reset, 50% hard state
后期: 30% reference reset, 70% hard state
```

### 7.3 代码设计

新增：

```text
environment/overall_environment/src/hard_state_buffer.py
```

API：

```python
class HardStateBuffer:
    def add(self, state: HardState): ...
    def sample(self, batch_size: int, weights: dict[str, float]): ...
    def save(self, path: Path): ...
    def load(self, path: Path): ...
```

---

## 八、创新方法 7：Grip viability critic

### 8.1 问题

什么时候允许 wrist / forearm residual 更大？什么时候应该只让手指先稳住拍子？固定 curriculum 不一定足够。

### 8.2 方法

训练一个 critic：

```text
C(s_t) = 未来 0.3 秒内是否会掉拍 / 打滑 / miss impact
```

输入：

```text
phase
finger qpos/qvel
hand-handle contact graph
palm-to-grip transform
racket pose/velocity
muscle activation
```

输出：

```text
failure probability
```

用它控制 residual：

```python
if grip_failure_prob > threshold:
    residual_scope = "fingers_only"
else:
    residual_scope = "fingers_wrist_forearm"
```

### 8.3 为什么可能有效

它可以避免策略在握拍不稳时还大幅调整腕/前臂，降低掉拍概率。也可以把训练注意力集中在最容易失败的局部状态。

---

# 九、下一轮最优先做的 4 个 PR

下面 4 个 PR 是建议的最小闭环路线。顺序非常重要。

---

## PR 1：新增 training Overall XML builder

### 目标

把当前 inspection scene 与 training scene 分开。

当前 inspection scene 保留：

```text
actuation disabled
person-racket contact excluded
用于可视化 / reset smoke test
```

新增 training scene：

```text
actuation enabled
person-racket contact enabled
可选 soft weld
可选 shuttle
用于 grip-hold / static-hit 训练
```

### 涉及文件

```text
environment/overall_environment/src/build_overall_environment.py
environment/overall_environment/src/paths.py
environment/overall_environment/README.md
environment/overall_environment/tests/test_overall_environment.py
```

### 建议实现

新增参数：

```python
mode: Literal["inspection", "training"] = "inspection"
enable_actuation: bool | None = None
enable_person_racket_contact: bool | None = None
enable_soft_weld: bool = False
```

默认行为：

```text
mode="inspection" 保持当前行为，不破坏已有测试
mode="training" 不调用 _disable_actuation，不调用 _exclude_person_racket_contacts
```

新增 output helper：

```python
def default_overall_training_scene_path() -> Path:
    return OVERALL_ROOT / "assets" / "overall_badminton_training_scene.xml"
```

### 测试

新增：

```text
test_build_training_scene_has_actuation_enabled
test_build_training_scene_does_not_exclude_person_racket_contact
test_build_training_scene_can_enable_soft_weld
test_training_scene_reset_reports_expected_objects
test_training_scene_100_steps_finite_with_pose_servo
```

### 验收标准

```text
pytest environment/overall_environment/tests/test_overall_environment.py -q 通过
inspection scene 行为不变
training scene model.opt.disableflags 不含 mjDSBL_ACTUATION
training scene hand-handle contact 没有 body-level exclude
```

---

## PR 2：实现 `run_forehand_clear_grip_hold.py --stage train`

### 目标

打通无 shuttle 的最小训练闭环：

```text
frozen body policy replay
+ right-hand residual grip policy
+ training Overall scene
+ grip-hold reward
```

这是当前最重要的 PR。

### 涉及文件

```text
BadmintonMimic/scripts/run_forehand_clear_grip_hold.py
environment/overall_environment/src/action_manifest.py
environment/overall_environment/src/layered_control.py
environment/overall_environment/src/overall_grip_hold_env.py  # 新增
environment/overall_environment/src/frozen_body_policy.py     # 新增
environment/overall_environment/src/body_obs_adapter.py       # 新增
tests/unit/test_forehand_clear_grip_hold_spec.py
```

### 建议实现阶段

#### Stage 2.1：只做 replay，不训练

目标：旧 body checkpoint 可以输出 finite action，并通过 router 控制 training scene。

```bash
python BadmintonMimic/scripts/run_forehand_clear_grip_hold.py \
  --stage replay-smoke \
  --steps 100
```

输出：

```json
{
  "policy_replay_ready": true,
  "steps": 100,
  "finite": true,
  "fall": false,
  "racket_drop": false,
  "mean_body_mimic_error": ...,
  "mean_grip_slip_m": ...
}
```

#### Stage 2.2：训练 right-hand fingers residual

新增：

```bash
--stage train
--residual-stage stage1
```

reward：

```text
r_mimic_body
r_grip_site
r_contact
r_no_slip
r_no_penetration
r_racket_hand_pose
r_residual_effort
```

#### Stage 2.3：保存 residual checkpoint

输出：

```text
outputs/posttrain/ForehandClearGripHold/v1/checkpoints/stage1/policy_latest.pt
outputs/posttrain/ForehandClearGripHold/v1/metrics/stage1_metrics.json
```

### 关键实现注意

最难的是 body obs compatibility。不能只适配 action。

需要：

```text
old checkpoint obs_size
old normalizer
body observation fields
current overall scene state -> old body obs
```

如果无法完全构造旧 obs，至少在 runner 中明确 fail fast：

```text
body_obs_adapter_ready: false
blocked_reason: obs schema mismatch
```

不要假装可以 replay。

### 测试

新增：

```text
test_grip_hold_replay_precheck_reports_adapter_ready
test_grip_hold_replay_smoke_fails_fast_without_obs_adapter
test_grip_hold_train_smoke_with_fake_body_policy
test_grip_hold_router_controls_only_residual_groups
test_grip_hold_reward_terms_are_finite
```

### 验收标准

```text
--stage replay-precheck 输出 action_adapter_ready=true
--stage replay-smoke 在 fake policy 下能跑 100 steps finite
--stage train 在 fake policy / tiny steps 下能写 metrics 和 checkpoint
真实 checkpoint 若 obs 不兼容，必须明确报错而不是 silent wrong replay
```

---

## PR 3：把 `StaticForehandClearEnv` 补成真正 RL environment

### 目标

当前 static-hit env 有 reward skeleton，但 phase/contact/flight/termination 仍不完整。PR 3 要让它成为真正可训练环境。

### 涉及文件

```text
environment/overall_environment/src/static_forehand_clear_env.py
environment/overall_environment/src/contact_graph.py       # 可新增
environment/overall_environment/src/shuttle_flight.py      # 可新增
environment/overall_environment/src/stringbed_contact.py   # 可新增
environment/overall_environment/tests/test_static_forehand_clear_env.py
```

### 需要补齐的功能

#### 1. 内部 phase tracker

不要要求外部传入 phase。

```python
phase = self.phase_tracker.phase(self.step_index)
```

支持：

```text
reference_len
loop / no-loop
impact_phase
```

#### 2. stringbed-shuttle contact detector

检测：

```text
active
contact point
normal
penetration
relative normal velocity
rho2 是否在 stringbed ellipse 内
frame contact or stringbed contact
```

#### 3. rebound model

早期可以简化：

```text
如果 valid stringbed contact：
  设置 shuttle qvel = function(racket velocity, normal, restitution)
```

注意要明确是直接写 qvel，还是通过 impulse / qfrc。不要被 `qfrc_applied` 清零吞掉。

#### 4. aero / drag model

早期简化：

```text
v <- v * exp(-lambda * dt)
```

或用 quadratic drag。

#### 5. flight tracker

记录：

```text
crossed_net
net_clearance_m
apex_height_m
landed
landing_xy
landing_region
out_of_bounds
```

#### 6. termination

必须支持：

```text
body_fall
racket_drop
miss_timeout
landed
episode_length
invalid_nan
```

### 推荐 state machine

```text
RESET
  -> PRE_IMPACT_FREEZE
  -> IMPACT_RELEASED
  -> FLIGHT_EVALUATION
  -> TERMINATED
```

当前 helper `should_transition_to_flight_evaluation()` 要真正接入 `step()`。

### 观测设计

```text
obs = [
  base proprioception,
  phase_sin, phase_cos,
  racket pose in root frame,
  racket velocity in root frame,
  handle-palm relative pose,
  grip contact flags,
  shuttle pose in root frame,
  shuttle velocity in root frame,
  impact target,
  landing target
]
```

### 测试

新增：

```text
test_static_hit_internal_phase_progresses
test_stringbed_contact_detector_identifies_valid_contact
test_stringbed_contact_detector_rejects_frame_contact
test_rebound_model_changes_shuttle_velocity
test_drag_model_reduces_speed
test_flight_tracker_detects_net_crossing
test_flight_tracker_classifies_landing_region
test_static_env_transitions_to_flight_evaluation
test_static_env_terminates_on_landing
test_static_env_terminates_on_miss_timeout
test_static_env_terminates_on_racket_drop
```

### 验收标准

```text
mock valid contact -> release -> rebound -> flight -> terminated
mock miss -> terminated with miss reason
reward_terms 包含 impact / flight / landing / penalties
step 不再强制要求外部 phase/contact_info
```

---

## PR 4：实现 Ghost Racket Teacher 并接入 grip-hold/static-hit reward

### 目标

把无拍 reference motion 转换成球拍伪监督，降低击球训练探索难度。

### 涉及文件

```text
environment/overall_environment/src/impact_target.py
environment/overall_environment/src/ghost_racket_teacher.py  # 新增
environment/overall_environment/tests/test_impact_target.py
environment/overall_environment/tests/test_ghost_racket_teacher.py
BadmintonMimic/experiments/posttrain/forehand_clear_static_hit_v1.yaml
```

### 建议实现

保留现有 `extract_impact_target_from_sites()`，在其基础上新增：

```python
def build_ghost_racket_trajectory(
    right_hand_pos: np.ndarray,
    root_pos: np.ndarray,
    forward_axis: np.ndarray,
    right_axis: np.ndarray,
    dt: float,
    racket_geometry: RacketGeometry,
    grip_transform: GripTransform,
) -> GhostRacketTrajectory:
    ...
```

`RacketGeometry`：

```python
@dataclass
class RacketGeometry:
    length_m: float
    stringbed_center_offset_m: np.ndarray
    head_offset_m: np.ndarray
    nominal_normal_axis: np.ndarray
```

`GhostRacketTrajectory`：

```python
@dataclass
class GhostRacketTrajectory:
    phases: np.ndarray
    grip_pos_root: np.ndarray
    stringbed_center_root: np.ndarray
    stringbed_normal_root: np.ndarray
    head_velocity_root: np.ndarray
    impact_frame: int
    impact_phase: float
```

### Reward 接入

在 grip-hold 阶段：

```text
r_ghost_grip_pose
r_ghost_racket_pose
```

在 static-hit 阶段：

```text
r_ghost_stringbed_center
r_ghost_head_velocity
r_ghost_normal
r_impact_phase
```

### 配置建议

在 YAML 中新增：

```yaml
ghost_racket:
  enabled: true
  racket_length_m: 0.67
  position_sigma_m: 0.08
  velocity_sigma_m_s: 4.0
  normal_sigma_deg: 25.0
  weight_schedule:
    early: 1.0
    middle: 0.5
    late: 0.2
```

### 测试

新增：

```text
test_ghost_racket_trajectory_shapes
test_ghost_racket_trajectory_finite
test_ghost_impact_phase_matches_peak_speed
test_ghost_reward_higher_when_real_matches_ghost
test_ghost_teacher_handles_degenerate_forward_axis
```

### 验收标准

```text
能从 mock reference 生成完整 ghost trajectory
impact_phase 与 peak speed 一致
reward 可被 static-hit env 调用
```

---

# 十、PR 之间的依赖关系

推荐顺序：

```text
PR 1 training scene
  ↓
PR 2 grip-hold train runner
  ↓
PR 4 ghost racket teacher
  ↓
PR 3 static-hit full env
```

也可以并行：

```text
PR 4 ghost teacher 可与 PR 1/2 并行
PR 3 依赖 PR 1，部分依赖 PR 4
```

最不建议的顺序是先做 static-hit runner，因为没有 training scene 和 frozen body replay 时，static-hit runner 会缺少底层闭环。

---

# 十一、建议的实验 ablation

当 PR 1～4 初步完成后，建议做以下 ablation：

```text
A0: frozen body + no grip pretrain + direct static-hit
A1: frozen body + grip pretrain
A2: A1 + soft-weld annealing
A3: A2 + ghost racket teacher
A4: A3 + phase-gated residual
A5: A4 + contact graph reward
A6: A5 + hard-state mining
```

主要指标：

```text
racket_drop_rate
grip_slip_m
valid_hand_handle_contact_count
valid_stringbed_contact_rate
frame_hit_rate
over_net_rate
opponent_back_landing_rate
out_rate
body_mimic_error
fall_rate
mean_effort
residual_action_norm
```

推荐不要只看 reward。必须看诊断指标，否则 reward hacking 很难发现。

---

# 十二、Codex 可直接执行的任务提示模板

下面是可以直接放给 Codex 的任务拆分模板。

## Codex Task 1

```text
Implement a training mode for environment/overall_environment/src/build_overall_environment.py.
Keep the current inspection scene behavior unchanged. Add a training scene mode that enables muscle actuation, does not exclude Full Body vs overall_racket contact, and optionally adds a soft weld between the right palm/hand body and the racket body. Add tests verifying that the inspection scene still disables actuation, while the training scene enables actuation and can run finite smoke steps.
```

## Codex Task 2

```text
Extend BadmintonMimic/scripts/run_forehand_clear_grip_hold.py with a train or replay-smoke stage. The runner should load the training Overall scene, reconstruct or load an ActionManifest, verify body observation compatibility, route frozen body actions and right-hand residual actions through LayeredActuatorRouter, and write JSON metrics. Implement a fake-policy smoke test first; fail fast if real checkpoint observation compatibility is not available.
```

## Codex Task 3

```text
Refactor StaticForehandClearEnv so it no longer requires external phase/contact_info for normal operation. Add internal phase tracking, stringbed-shuttle contact detection hooks with a default implementation, flight tracking, and termination on landing, miss timeout, racket drop, or non-finite state. Preserve existing tests and add new tests for state transitions and reward terms.
```

## Codex Task 4

```text
Add environment/overall_environment/src/ghost_racket_teacher.py. Build a ghost racket trajectory from reference right-hand/root/axis arrays, including stringbed center, normal, head velocity, and impact phase. Add reward helpers and tests. Integrate config fields into forehand_clear_static_hit_v1.yaml, but keep the runner fail-fast until static-hit training is implemented.
```

---

# 十三、最终建议

下一轮不要把精力放在继续写更复杂 YAML，也不要立刻做完整 high-clear landing reward。

优先目标应该是：

```text
让旧无拍 body checkpoint 能在 training Overall scene 中 replay，
并让右手 residual policy 在无 shuttle 情况下学会握拍不掉。
```

这个闭环一旦打通，后面所有创新方法都会更容易接入。

最小成功标准：

```text
1. training scene 有效
2. frozen body replay finite
3. grip residual action routing 正确
4. no-shuttle grip-hold reward 非零且可训练
5. 输出 racket_drop_rate / grip_slip_m / contact_count / body_mimic_error
```

达到这个标准之后，再进入 static shuttle impact 和 high-clear landing。

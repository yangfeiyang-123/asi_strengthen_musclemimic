# Step2 方法可行性检查与推荐实施路线

**日期:** 2026-06-01  
**检查对象:** `/data3/yangfeiyang/WorkSpace/musclemimic/doc/修改建议/实施计划Step2`  
**结论一句话:** Step2 建议的方法方向是有效且可实施的，但当前仓库还不能直接进入完整正手高远球 static-hit / landing 训练；最合理可行的选择是先完成 `training scene builder -> frozen body replay-smoke -> grip-hold train runner`，再做 ghost teacher 和 static-hit full env。

---

## 1. 本次读取的 Step2 建议文件

已检查以下文件：

- `doc/修改建议/实施计划Step2/1_latest_implementation_review_and_overall_judgment.md`
- `doc/修改建议/实施计划Step2/2_literature_directions.md`
- `doc/修改建议/实施计划Step2/3_innovation_methods_and_next_prs.md`

三份文件的共同判断一致：

```text
不要直接端到端训练完整击球；
先保留无拍正手高远球 body prior；
用 training scene + frozen body + grip/residual + curriculum 降低探索难度；
再逐步加入 ghost racket、contact graph、static shuttle release、flight reward、hard-state mining。
```

这个方向与当前项目状态匹配。

---

## 2. 当前仓库事实检查

### 2.1 已经基本具备的基础

当前仓库已经具备以下基础模块：

| 模块 | 当前状态 | 证据 |
|---|---|---|
| 防误跑 guard | 已有 | `BadmintonMimic/scripts/run_posttrain_experiment.py` 已拒绝 grip-hold/static-hit 误走 ordinary runner |
| action manifest / adapter | 已有 | `environment/overall_environment/src/action_manifest.py`, `action_adapter.py` |
| layered router | 已有 | `environment/overall_environment/src/layered_control.py` |
| grip policy loader / layered policy skeleton | 已有 | `environment/overall_environment/src/layered_policy.py` |
| static-hit reward skeleton | 已有 | `environment/overall_environment/src/static_forehand_clear_env.py` |
| contact reward skeleton | 已有 | `environment/overall_environment/src/contact_graph.py` |
| phase reward skeleton | 已有 | `environment/overall_environment/src/phase_reward.py` |
| soft-weld schedule skeleton | 已有 | `environment/overall_environment/src/soft_weld_schedule.py` |
| ghost racket interpolation skeleton | 已有 | `environment/overall_environment/src/ghost_racket.py` |
| grip PPO tanh logprob 修正 | 已有 | `src/grip/train_right_hand_racket_grip_policy.py` |

关键 replay-precheck 已能确认旧 checkpoint 与 full scene action space 的差异：

```json
{
  "checkpoint_action_size": 354,
  "scene_action_size": 416,
  "adapter_mapped_count": 354,
  "adapter_extra_in_target_count": 62,
  "action_adapter_ready": true,
  "policy_replay_ready": false
}
```

这说明 action-name adapter 路线是必要且已经部分可用的。

### 2.2 当前仍不满足训练闭环的事实

#### 事实 1：grip-hold runner 仍不是 train runner

`BadmintonMimic/scripts/run_forehand_clear_grip_hold.py` 文件头仍明确写着：

```text
This script does not train yet.
```

当前 parser 只有：

```text
preflight
reset-video
replay-precheck
```

没有 `train` 或 `replay-smoke`。

因此现在不能说已经能训练：

```text
frozen body + grip residual + racket hold
```

#### 事实 2：当前 training XML 只是部分满足

当前存在：

```text
environment/overall_environment/assets/overall_badminton_training_scene.xml
```

实测 MuJoCo 加载结果：

```json
{
  "nu": 416,
  "actuation_disabled": false,
  "nexclude": 11,
  "has_fullbody_racket_exclude": true,
  "neq": 51
}
```

这说明：

- 肌肉 actuator 已启用：`nu=416`, `actuation_disabled=false`
- 但 `Full Body` 与 `overall_racket` 仍有 body-level contact exclude：

```xml
<exclude body1="Full Body" body2="overall_racket" />
```

所以它还不是合格的 grip/contact training scene。  
它能用于 action-size / loading smoke test，但不能用于真实 hand-racket contact 学习。

#### 事实 3：builder 仍只会生成 inspection scene

`environment/overall_environment/src/build_overall_environment.py` 当前 `build_overall_scene()` 无 `mode="training"` 参数，并且固定调用：

```python
_disable_actuation(raw_xml)
_exclude_person_racket_contacts(raw_xml)
```

这与 Step2 PR 1 的建议完全一致：必须先把 inspection scene 和 training scene 分开。

#### 事实 4：StaticForehandClearEnv 还不是完整 RL env

`StaticForehandClearEnv.step()` 当前仍要求外部传入：

```python
phase
contact_info
```

并且真实 stringbed detector / rebound / flight tracker / termination 还没有默认实现。

当前已有的是 reward skeleton，不是完整 static-hit training environment。

#### 事实 5：hook 与 applied force 顺序确实有隐患

`environment/overall_environment/src/overall_env.py` 中：

```python
self.data.qfrc_applied[:] = 0.0
```

发生在 `step()` 内部。  
而 Step2 文档指出，如果未来 rebound/aero hook 通过 `qfrc_applied` 写 force，可能被清零吞掉。这个判断成立。

---

## 3. Step2 方法是否有效

结论：有效。

### 3.1 为什么有效

你的任务不是单纯“让 humanoid 打球”，而是：

```text
已有无拍肌骨正手高远球策略
+ 新增右手握拍
+ 球拍-手真实接触
+ 球拍-羽毛球 impact
+ 羽毛球飞行落点
```

如果直接 full-body fine-tune 或直接 static-hit，会同时面对：

```text
416 维 full-hand muscle action
旧 checkpoint 354 维 action 不兼容
旧 obs schema 未明确
free racket 接触探索难
shuttle contact 稀疏
landing reward 极稀疏
肌骨系统容易 reward hacking
```

所以 Step2 推荐的组合是合理的：

```text
DeepMimic / MuscleMimic 思路: 保留 body prior
PhysHOI 思路: contact graph
InterMimic 思路: perfect first, then scale up
MyoSuite 思路: curriculum for contact-rich muscle control
MuJoCo weld 思路: soft-weld annealing 降低早期探索难度
Shuttle drag 思路: 不把羽毛球当普通抛体
```

### 3.2 哪些方法当前最适合马上实施

最适合马上实施的是：

```text
PR 1: training Overall XML builder
PR 2: grip-hold replay-smoke / train runner
```

原因：

1. 当前 action adapter / router / grip policy / PPO 修正都已经有基础。
2. 当前最大 blocker 是 scene 和 runner，不是 reward 细节。
3. 没有 training scene，contact graph / static-hit / ghost reward 都无法真实闭环验证。
4. 没有 frozen body replay-smoke，就无法确认旧 checkpoint 的 obs/action 能否进入 full scene。

### 3.3 哪些方法暂时不应该优先做

暂时不建议优先做：

```text
完整 AMP / motion prior discriminator
事件相机 impact teacher
复杂 shuttle aerodynamic domain randomization
完整 hard-state replay buffer
完整 static-hit high-clear landing runner
```

原因：当前最小训练闭环还没打通，过早加这些会增加复杂度，不能解决当前 blocker。

---

## 4. 推荐选择的最合理可行路线

我建议选择 Step2 中的以下路线，并略微调整顺序：

```text
PR 1 training scene builder
  -> PR 2A grip-hold replay-smoke
  -> PR 2B fake-policy tiny train
  -> PR 2C real checkpoint fail-fast obs compatibility
  -> PR 4 ghost racket teacher
  -> PR 3 static-hit full env
```

理由：Step2 原文推荐 `PR1 -> PR2 -> PR4 -> PR3`，这个依赖关系基本正确。但 PR2 内部应再拆成 replay-smoke、fake train、real checkpoint compatibility 三步，避免一次性把真实 checkpoint actor loading、obs adapter、PPO training 都塞进一个 PR。

---

## 5. 具体实施路线

### 阶段 A：PR 1，真正实现 training scene builder

**目标:** 保留 inspection scene，同时用 builder 生成 training scene。

当前不应继续手工维护复制版 XML，因为它已经暴露出问题：

```text
actuation enabled 了，但 Full Body - overall_racket exclude 仍存在。
```

应修改：

```text
environment/overall_environment/src/build_overall_environment.py
environment/overall_environment/src/paths.py
environment/overall_environment/tests/test_overall_environment.py
environment/overall_environment/README.md
```

推荐 API：

```python
def build_overall_scene(
    output_xml: str | Path | None = None,
    *,
    grip_seed: str | Path | None = None,
    mode: str = "inspection",
    enable_actuation: bool | None = None,
    enable_person_racket_contact: bool | None = None,
    enable_soft_weld: bool = False,
    soft_weld_solref: str = "0.02 1",
    soft_weld_solimp: str = "0.8 0.95 0.001",
) -> Path:
    ...
```

验收必须包括：

```text
inspection scene 行为不变
training scene actuation enabled
training scene 不再有 Full Body - overall_racket exclude
training scene 可选 soft weld
training scene 100 steps finite
```

这是当前最应该先做的选择。

### 阶段 B：PR 2A，新增 replay-smoke，不训练

**目标:** 先证明 frozen body action 能进 training scene。

新增 stage：

```bash
python BadmintonMimic/scripts/run_forehand_clear_grip_hold.py \
  --spec BadmintonMimic/experiments/posttrain/forehand_clear_grip_hold_v1.yaml \
  --stage replay-smoke \
  --steps 100
```

先支持 fake body policy，输出：

```json
{
  "policy_replay_ready": false,
  "fake_policy_replay_ready": true,
  "steps": 100,
  "finite": true,
  "scene_action_size": 416,
  "adapter_mapped_count": 354,
  "racket_drop": false
}
```

真实 checkpoint 如果 obs adapter 未完成，应明确 fail-fast：

```json
{
  "policy_replay_ready": false,
  "blocked_reason": "body observation adapter is not implemented"
}
```

不要 silent wrong replay。

### 阶段 C：PR 2B，新增 overall_grip_hold_env

**目标:** 用 training scene 做无 shuttle grip-hold 小闭环。

新增：

```text
environment/overall_environment/src/overall_grip_hold_env.py
```

它应提供：

```text
reset()
step(ctrl)
reward_terms
grip_slip_m
hand_handle_contact_count
racket_drop
body_fall
finite diagnostics
```

短期 reward：

```text
r_mimic_body
r_grip_site
r_contact
r_no_slip
r_no_penetration
r_racket_hand_pose
-r_residual_effort
```

### 阶段 D：PR 2C，body obs compatibility

**目标:** 处理旧 checkpoint actor 真正 replay 的最大风险。

当前 replay-precheck 已知：

```text
checkpoint_obs_size = 2418
checkpoint_action_size = 354
scene_action_size = 416
```

但还没有：

```text
obs_manifest.json
normalization_manifest.json
BodyObsAdapter
FrozenBodyPolicy.load()
```

因此下一步要新增：

```text
environment/overall_environment/src/frozen_body_policy.py
environment/overall_environment/src/body_obs_adapter.py
```

要求：

```text
obs_size 对不上必须报错
normalizer 缺失必须报错
actor load 失败必须报错
不能用随机 obs 假装 replay 成功
```

### 阶段 E：PR 4，Ghost Racket Teacher

**目标:** 在 static-hit 前先给球拍轨迹一个可学习 teacher。

当前 `environment/overall_environment/src/ghost_racket.py` 只有插值 skeleton，不够。

建议新增：

```text
environment/overall_environment/src/ghost_racket_teacher.py
```

实现：

```text
build_ghost_racket_trajectory()
sample_ghost_at_phase()
compute_ghost_reward()
```

输入先用 mock/reference arrays，不要一开始绑定复杂 runner。

验收：

```text
5 帧 mock right_hand_pos 能生成 finite trajectory
impact_phase 在 [0, 1]
impact frame 与 peak head speed 一致
真实 racket 越接近 ghost，reward 越高
```

### 阶段 F：PR 3，StaticForehandClearEnv full env

**目标:** 把 static-hit 从 reward skeleton 变成可训练环境。

在 PR 1/2/4 后再做，原因是它依赖：

```text
training scene contact
layered action
ghost teacher
grip-hold 稳定性
```

应补：

```text
internal phase tracker
stringbed contact detector
rebound model
drag / aero model
flight tracker
termination
structured observation
```

验收：

```text
mock valid contact -> release -> rebound -> flight -> terminated
mock miss -> terminated with miss reason
step 不再强制要求外部 phase/contact_info
reward_terms 包含 impact / flight / landing / penalties
```

---

## 6. 不推荐的选择

### 不推荐 1：直接实现 static-hit full runner

原因：

```text
training scene 当前仍排除了 Full Body - racket contact
frozen body replay 还未打通
obs adapter 还未实现
grip-hold 还未证明能稳定不掉拍
```

直接做 static-hit runner 会把所有问题混在一起，训练失败无法定位。

### 不推荐 2：直接全身 fine-tune

原因：

```text
旧无拍 body policy 是最有价值的 body prior
full-body fine-tune 会破坏已有正手高远球动力链
416 维 action 高维且稀疏 reward 下非常不稳定
```

应冻结 body，先只训练 right-hand fingers，再逐步 wrist/forearm。

### 不推荐 3：先做复杂 AMP / video teacher

原因：

```text
当前训练闭环未打通
复杂 teacher 不能替代 scene / runner / contact / obs compatibility
```

这些应放在后期。

---

## 7. 当前模型学习正手高远球击打的推荐整体流程

最终目标流程应是：

```text
1. 原始无拍 ForehandClear body policy
   - checkpoint action size 354
   - disable_fingers=True
   - 提供全身正手高远球动作 prior

2. Training Overall scene
   - action size 416
   - actuation enabled
   - hand-racket contact enabled
   - optional soft weld

3. Frozen body replay
   - 加载旧 body actor
   - BodyObsAdapter 构造旧 obs
   - ActionManifest / LayeredActuatorRouter 保证 action name 对齐

4. Grip-hold training
   - π_body frozen
   - π_grip / residual 控制 right_hand_fingers
   - 目标：挥拍过程中球拍不掉、contact 稳定、penetration 小

5. Swing disturbance
   - 给 racket 施加近似挥拍惯性扰动
   - 让 grip policy 从静态握拍迁移到动态握拍

6. Ghost racket teacher
   - 从无拍动作生成 ghost stringbed / head velocity / normal / impact phase
   - 先学球拍轨迹，再学真实击球

7. Static shuttle hit
   - shuttle freeze at impact target
   - correct phase + stringbed contact + closing velocity 才 release
   - reward impact / outgoing velocity proxy

8. Flight curriculum
   - over-net
   - net clearance
   - apex height
   - opponent back landing
   - out penalty

9. Hard-state mining
   - grip_slip / missed_contact / frame_hit / bad_flight / fall
   - 失败状态重采样

10. Ablation
   - A0 direct static-hit
   - A1 grip pretrain
   - A2 soft-weld
   - A3 ghost teacher
   - A4 phase-gated residual
   - A5 contact graph
   - A6 hard-state mining
```

---

## 8. 最终决策

Step2 的方法可以实施，且方向正确。

但当前最合理的实施选择不是“开始完整击球训练”，而是：

```text
先实施 PR 1 training scene builder；
再实施 PR 2 replay-smoke / grip-hold train runner；
确认 frozen body + grip residual 能在 training scene 中 100-step finite；
然后再做 PR 4 ghost teacher；
最后做 PR 3 static-hit full env。
```

当前最优先任务：

```text
PR 1: training Overall XML builder
```

理由：

```text
当前 copied training_scene 虽然启用了 actuation，但仍排除了 Full Body - racket contact；
没有合格 training scene，grip-hold / static-hit / contact graph 都无法真实训练；
PR 1 风险最低、收益最高、可测试性最好，是后续所有训练闭环的前置条件。
```


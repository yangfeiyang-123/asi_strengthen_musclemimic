# 5. 最核心的结论

> 这是给项目决策和 Codex 执行优先级用的压缩版结论。它总结了为什么当前不能直接端到端训练、应该先补什么、以及最值得采用的整体技术路线。

---

## 1. 一句话结论

你现在最不应该做的是：

```text
直接把球拍和羽毛球加进环境，然后 resume 原来的无球拍正手高远球 checkpoint 端到端训练。
```

更稳、更可解释、也更可能成功的路线是：

```text
已有无球拍正手高远球策略
  -> 冻结 body policy
  -> 用 actuator-name adapter 接入带球拍 full scene
  -> 单独训练右手握拍 policy
  -> 用 soft-weld annealing 保证球拍先不掉
  -> 用 ghost racket teacher 给球拍轨迹伪监督
  -> 用 phase-gated static shuttle release 学会正确相位击球
  -> 再加入真实 shuttle flight 和后场落点 reward
  -> 最后只小幅解冻 wrist / forearm / shoulder residual
```

---

## 2. 当前实现的真实状态

当前仓库不是“已经实现完整拿拍击球训练”，而是处在：

```text
诊断脚本 + 局部握拍环境 + overall inspection scene + static-hit staging wrapper
```

这些部分是有价值的，但还没有完整打通。

关键证据：

```text
1. grip-hold runner 只支持 preflight / reset-video / replay-precheck，没有 train。
2. replay_precheck 明确写着 policy_replay_ready=False。
3. blocked_reason 指出还缺 disable_fingers checkpoint 到 overall racket scene 的 action adapter，以及 right-hand residual merge。
4. grip-hold YAML 虽然写了 frozen body policy 和 trainable residual policy，但 runner 没有真正训练这些字段。
5. fullbody badminton config 使用 disable_fingers=true，与握拍所需手指控制冲突。
6. overall scene README 说明球拍是 free body，未 welded，person-racket contacts 被过滤，raw physics 下会离手。
7. StaticForehandClearEnv.step() 当前返回 reward=0，terminated=False，truncated=False。
```

参考：

- `run_forehand_clear_grip_hold.py`: https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/musclemimic/badminton/scripts/run_forehand_clear_grip_hold.py#L850-L858
- `run_forehand_clear_grip_hold.py` stages: https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/musclemimic/badminton/scripts/run_forehand_clear_grip_hold.py#L879-L882
- `forehand_clear_grip_hold_v1.yaml`: https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/experiments/posttrain/forehand_clear_grip_hold_v1.yaml#L406-L455
- `disable_fingers: true`: https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/fullbody/config_specific_task/conf_fullbody_badminton_gmr.yaml#L381-L390
- overall scene notes: https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/tree/main/environment/overall_environment#L313-L322
- static env returns zero reward: https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/environment/overall_environment/src/static_forehand_clear_env.py#L767-L808

---

## 3. 最大风险不是 reward 调得不好，而是系统没有接起来

最容易导致训练失败的不是某个 reward 权重不合适，而是这些结构性问题：

```text
旧 checkpoint action space 与新环境 action space 不一致。
旧 checkpoint disable_fingers=True，但新任务需要 right-hand fingers。
旧 checkpoint obs normalization 与新环境 observation 不一致。
body action 和 grip action 没有按 actuator name 安全 merge。
overall scene 是 inspection scene，不是 training scene。
static-hit env 没有 reward / done。
球拍 free body 一开始就可能掉。
shuttle contact reward 稀疏，直接训练很难探索到。
```

因此第一优先级应该是：

```text
manifest
adapter
router
training scene
smoke tests
```

而不是先跑大规模 PPO。

---

## 4. 最推荐的策略架构

采用三层策略：

```text
π_body: frozen full-body forehand clear policy
π_grip: trainable right-hand grip policy
π_residual: small phase-gated residual policy
```

动作合成：

```python
a_body = π_body(obs_body)
a_grip = π_grip(obs_grip)
gate = phase_gate(phase, impact_phase)
a_res = π_residual(obs_task, phase)

a_full = router.merge(
    body_action=a_body,
    grip_action=a_grip + gate * a_res,
)
```

约束：

```text
1. π_body 初期完全冻结。
2. π_grip 先只控制 right_hand_fingers。
3. π_residual 初期只在 impact window 附近打开。
4. residual scale 要小，尤其 wrist/forearm/shoulder。
5. 所有 action mapping 必须按 actuator name，不按 index。
```

理由：

```text
旧 body policy 已经学会正手高远球的大动作；
握拍是局部手部接触任务；
击球微调只应发生在击球相位附近；
分层能防止 full-body 动作被击球 reward 带崩。
```

---

## 5. 最推荐的训练课程

```text
Stage 0: 静态握拍 IK / grip seed
  目标：得到 plausible qpos/qvel 和 racket pose。

Stage 1: right-hand static grip policy
  目标：手指闭合、接触覆盖、无穿透、球拍不掉。

Stage 2: soft-weld grip hold under swing
  目标：replay body forehand motion 时，手保持球拍稳定。

Stage 3: ghost racket tracking
  目标：真实球拍跟踪由无拍动作推断出的 ghost racket head / face / velocity。

Stage 4: static shuttle freeze-release
  目标：只在正确 phase、正确 stringbed contact、正确 closing velocity 时 release shuttle。

Stage 5: shuttle over-net
  目标：羽毛球过网，轨迹合理。

Stage 6: high clear landing
  目标：高弧线、落到对方后场。

Stage 7: partial unfreeze fine-tune
  目标：小幅解冻 wrist/forearm/shoulder，提高击球效果，但保持 body mimic。
```

不要按固定步数强行推进。建议用指标推进：

```text
contact_count >= 4
mean_grip_slip_m < 0.05
racket_drop_rate < 5%
valid_stringbed_contact_rate > 50%
over_net_rate > 30%
opponent_backcourt_landing_rate 逐步提高
```

---

## 6. 最值得创新的组合

最有创新性、并且我认为最可能有效的是：

```text
Ghost racket teacher
+ Soft-weld annealing
+ Phase-gated contact reward
+ Contact graph reward
+ ASI hard-state mining
+ Frozen body prior / small residual control
```

### 6.1 Ghost racket teacher

从无球拍动作推断：

```text
球拍 grip pose
球拍 head pose
球拍 head velocity
拍面 normal
impact phase
```

先让真实球拍跟踪 ghost，后期退火 ghost reward。

价值：把“没有球拍的动作数据”转成“有球拍的伪监督”。

### 6.2 Soft-weld annealing

先让球拍通过 soft weld 稳定在手上，再逐渐减弱 weld，最后纯 contact。

价值：降低自由接触握拍的探索难度。

### 6.3 Phase-gated contact reward

只有在 impact window 附近奖励 stringbed-shuttle contact；其他相位不奖励甚至惩罚。

价值：避免模型提前把球拍伸到球旁边、破坏正手高远球动作。

### 6.4 Contact graph reward

显式记录：

```text
thumb/index/middle/palm 与 handle 的接触
stringbed 与 shuttle 的接触
frame 与 shuttle 的错误接触
illegal penetration
slip velocity
```

价值：人-物交互任务不能只用距离 reward。

### 6.5 ASI hard-state mining

把失败前的状态保存下来：

```text
掉拍前
打滑前
miss impact 前
身体失稳前
球打不过网前
```

下一轮从这些 hard states 初始化。

价值：让策略反复练最容易失败的临界状态。

---

## 7. 最小可行版本 MVP

如果要最快验证路线是否可行，MVP 不需要一次性实现所有创新。建议先做：

```text
MVP-1:
  checkpoint action manifest
  action adapter
  LayeredActuatorRouter integration
  overall training scene with medium soft weld
  frozen body replay smoke test

MVP-2:
  load right-hand grip policy
  train grip under no-shuttle swing
  metrics: racket_drop_rate, grip_slip, contact_count

MVP-3:
  ghost racket teacher
  reward: racket head / face / velocity tracking

MVP-4:
  static shuttle freeze-release
  reward: phase-gated stringbed contact
```

MVP 成功标准：

```text
1. 旧 body policy 能在带球拍场景 replay 100~300 steps，不 NaN。
2. 球拍在挥拍过程中不立即掉。
3. right hand contact count 能稳定达到 3~4 个以上。
4. static shuttle 能在 impact phase 附近被 stringbed 接触 release。
5. body mimic error 不显著恶化。
```

MVP 失败时优先排查：

```text
action manifest / actuator mapping
obs normalization
training XML contact/weld
grip seed qpos/qvel
right-hand actuator group
contact geom filtering
phase / impact target
```

---

## 8. 不建议的路线

### 不建议 1：直接 full-body 端到端 fine-tune

风险：

```text
灾难性遗忘
动作变形
reward hacking
训练不稳定
失败原因不可解释
```

### 不建议 2：只训练手腕/手指去打球

风险：

```text
破坏正手高远球动力链
学出不真实的局部甩动
球拍速度不足或方向异常
```

### 不建议 3：全程奖励 racket-shuttle 距离

风险：

```text
模型提前伸拍等球
不再做完整挥拍动作
用错误接触蹭球
```

### 不建议 4：忽略 shuttle aerodynamics

风险：

```text
训练出的 high clear 轨迹不像真实羽毛球
普通抛体 proxy 会误导出球速度和角度
```

早期可以用解析 proxy，但最终评价必须用真实或近似真实的 shuttle drag / flight。

---

## 9. Codex 执行时的优先级

请让 Codex 按这个顺序改：

```text
1. runner guard：阻止 grip-hold 误走 fullbody runner。
2. diagnostic doc：明确当前 grip-hold runner 还不能 train。
3. action manifest：记录旧 checkpoint action/obs/disable_fingers。
4. action adapter：按 actuator name 映射。
5. router integration：body + grip action merge。
6. training scene XML：soft weld / contact variants。
7. layered rollout smoke test：fake policy + real env。
8. grip PPO logprob 修正。
9. StaticForehandClearEnv reward / termination。
10. contact graph。
11. ghost racket teacher。
12. curriculum runner。
13. ASI hard-state mining。
14. partial unfreeze fine-tune。
15. ablation report。
```

每一步都要有测试。没有测试就不要进入下一步大规模训练。

---

## 10. 最后判断

这个任务是可行的，但关键不是“再调一个 reward”或“直接加一个球拍”。真正关键是：

```text
把原来的无拍全身动作策略当成 body prior，
把握拍当成独立的人-物接触控制问题，
把击球当成 phase-gated 事件，
把羽毛球飞行当成后期 curriculum，
用 ghost/soft-weld/ASI 降低探索难度。
```

最终路线：

```text
无拍正手高远球 imitation
  + 握拍 contact graph
  + 球拍 ghost trajectory
  + static impact event
  + shuttle flight target
  = 可解释、可调试、可逐步提高成功率的拿拍击球策略
```

这是比“直接端到端训练”更稳、更科学、也更容易发表/展示创新点的方案。

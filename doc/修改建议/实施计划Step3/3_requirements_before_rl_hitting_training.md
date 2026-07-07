# 3. 要开始 RL 击球训练前必须完成的事情

## 当前明确判断

当前系统**还不建议直接开始 RL 击球训练**。

更准确的判断是：

```text
可以开始：no-shuttle grip-hold residual RL 的短程调试
不建议开始：shuttle impact / over-net / landing RL
不建议直接跑长 PPO
```

原因是：当前 pipeline 虽然已经能完成 frozen body replay smoke 和 tiny train，但仍存在会污染学习信号的问题：

```text
pose_servo 默认参与 step
body mimic 可能奖励 reset pose
overall reset 与 trajectory goal 可能没有 phase 对齐
Frozen NumPy actor 缺少 Flax 等价性测试
10-step smoke 不足以证明长程稳定
```

因此，开始击球 RL 前必须先通过下面的阶段性 gate。

---

# Stage 0：Artifact / scene / cache 准备完成

## 目标

确保所有训练依赖都可复现、可校验、与同一个 checkpoint 一致。

## 必须存在

```text
checkpoints/de63059b16c0/checkpoint_7812
outputs/frozen_body_policy/de63059b16c0_7812/manifest.json
outputs/frozen_body_policy/de63059b16c0_7812/params.npz
outputs/frozen_body_policy/de63059b16c0_7812/run_stats.npz
environment/overall_environment/assets/overall_badminton_training_scene.xml
outputs/right_hand_racket_grip/reference/right_hand_racket_grip_seed.json
caches/AMASS/MyoFullBody/gmr/10trajectories/video1_lower_body_full_poses.npz
```

## 必须自动化检查

新增或使用：

```bash
python musclemimic/badminton/scripts/prepare_forehand_clear_grip_hold_artifacts.py \
  --spec experiments/posttrain/forehand_clear_grip_hold_v1.yaml \
  --build-training-scene \
  --export-frozen-policy \
  --check-trajectory-cache \
  --check-grip-seed
```

## 验收标准

```json
{
  "artifact_source_checkpoint_matches_spec": true,
  "params_npz_exists": true,
  "run_stats_npz_exists": true,
  "training_scene_ok": true,
  "actuation_enabled": true,
  "hand_racket_contact_allowed": true,
  "trajectory_cache_ok": true,
  "body_obs_size": 2418,
  "goal_size": 469,
  "body_action_size": 354,
  "overall_action_size": 416
}
```

只有 Stage 0 通过后，才能继续。

---

# Stage 1：Frozen body policy 数值等价验证

## 目标

证明导出的 NumPy frozen policy 与原 Flax ActorCritic deterministic mean action 等价。

## 必须验证

```text
FrozenBodyPolicy.act(obs)
≈
Flax ActorCritic.apply(...).pi.mean()
```

## 检查点

```text
RunningMeanStd
LayerNorm
silu
Dense kernel/bias
actor_obs_ind
final Dense linear output
log_std 不参与 mean action
```

## 验收标准

```json
{
  "numpy_vs_flax_actor_mean_checked": true,
  "max_abs_action_diff": "<= 1e-5",
  "mean_abs_action_diff": "<= 1e-6",
  "cosine_similarity": ">= 0.99999"
}
```

如果不通过，不能进入任何长训练或击球训练。

---

# Stage 2：Trajectory-aligned no-servo frozen body replay

## 目标

验证 frozen body policy 在 overall training scene 中能真实 replay，而不是依赖 pose servo。

## 必须关闭

```text
pose_servo = False
shuttle_hit_reward = disabled
residual_action = zero
```

## 必须开启

```text
training scene
trajectory-aligned reset
trajectory GoalTrajMimic
frozen body policy artifact
354 → 416 actuator name mapping
```

## 推荐命令

```bash
python musclemimic/badminton/scripts/run_forehand_clear_grip_hold.py \
  --spec experiments/posttrain/forehand_clear_grip_hold_v1.yaml \
  --stage replay-smoke \
  --policy-source real \
  --steps 1000 \
  --no-pose-servo
```

## 验收标准

```json
{
  "policy_replay_ready": true,
  "steps_completed": 1000,
  "finite": true,
  "body_fall": false,
  "body_obs_size": 2418,
  "body_action_size": 354,
  "goal_obs_source": "trajectory_cache",
  "pose_servo_enabled": false,
  "full_ctrl_saturation_rate": "< 0.1"
}
```

## 视频验收

必须保存并人工检查：

```text
frozen_body_replay_no_servo.mp4
```

人工检查标准：

```text
动作像 ForehandClear
身体没有明显倒地或抽搐
root/torso/shoulder/elbow 动作连续
球拍没有靠明显非物理约束悬浮
```

---

# Stage 3：No-shuttle grip-hold residual short PPO

## 目标

只训练右手手指 residual，让模型在无球阶段学会持拍。

## 训练设置

```text
shuttle hit disabled
pose_servo=False
body policy frozen
residual groups = right_hand_fingers
trajectory-aligned reset
reward = grip hold only
```

## 推荐训练规模

先不要长训：

```text
2k steps
5k steps
20k steps
```

每个阶段都要看视频和 metrics。

## 关键 reward

```text
contact
no_slip
no_penetration
racket_hand_pose(reference transform)
residual_effort
trajectory body mimic
```

## 验收指标

```json
{
  "racket_drop_rate": "< 5%",
  "body_fall_rate": "< 5%",
  "mean_grip_slip_m": "< 0.05",
  "max_handle_penetration_m": "< 0.003",
  "mean_hand_handle_contact_count": ">= 2 for stage1",
  "finite_rate": "> 99%"
}
```

## 不能接受的现象

```text
contact_count 提升但 penetration 很大
grip_slip 下降但球拍穿进手里
body mimic reward 高但没有挥拍
reward 上升但视频中球拍靠 servo 或 weld 悬浮
```

---

# Stage 4：No-shuttle swing-disturbance grip training

## 目标

让右手 residual 能抵抗挥拍过程中的惯性扰动，而不仅是静态握拍。

## 做法

在无球情况下加入：

```text
racket inertial disturbance
wrist/forearm acceleration disturbance
phase-dependent disturbance
```

可以先使用简化外力：

```python
force, torque = swing_disturbance_profile(
    phase=phase,
    phase_start=0.2,
    phase_end=0.8,
    force_scale_n=...,
    torque_scale_nm=...,
)
data.xfrc_applied[racket_body_id] = [force, torque]
```

## residual groups

先：

```text
right_hand_fingers
```

稳定后再扩展：

```text
right_hand_fingers + right_wrist + right_forearm
```

但 wrist/forearm residual 必须有小 scale：

```yaml
right_wrist: 0.2
right_forearm: 0.15
```

## 验收标准

```json
{
  "racket_drop_rate_under_disturbance": "< 10%",
  "mean_grip_slip_m": "< 0.05",
  "mean_contact_count": ">= 4",
  "penetration_pass": true,
  "body_motion_still_forehand_clear": true
}
```

通过 Stage 4 后，才说明“持拍/挥拍”阶段基本可用。

---

# Stage 5：Ghost racket / virtual stringbed tracking

## 为什么需要

直接加 shuttle 会非常稀疏。应该先让球拍在正确相位到达合理击球位置和拍面方向。

## 目标

从无拍 ForehandClear reference 生成：

```text
ghost_racket_pose(t)
ghost_stringbed_center(t)
ghost_stringbed_normal(t)
ghost_racket_head_velocity(t)
impact_phase
```

## reward

```text
r_ghost_center
r_ghost_normal
r_ghost_head_velocity
r_impact_phase
```

## 验收标准

```json
{
  "stringbed_center_error_m_at_impact": "< 0.10",
  "stringbed_normal_error_deg_at_impact": "< 20",
  "racket_head_speed_m_s_at_impact": "> threshold",
  "impact_phase_error": "< 0.08"
}
```

如果没有 Ghost racket 过渡，直接击球 RL 很可能学不到。

---

# Stage 6：Static shuttle contact，不要求过网

## 目标

开始引入 shuttle，但只要求在合理 phase 用 stringbed 接触到 shuttle。

## 环境要求

必须已经实现：

```text
phase 内部维护
stringbed-shuttle contact detector
frame-hit detector
contact point inside stringbed ellipse
relative normal velocity
early/late contact penalty
miss timeout termination
racket drop termination
body fall termination
```

## reward

```text
+ stringbed contact
+ contact inside stringbed ellipse
+ closing velocity
+ correct phase
- early contact
- late contact
- frame contact
- penetration / illegal contact
```

## 验收标准

```json
{
  "valid_stringbed_contact_rate": "> 50%",
  "frame_hit_rate": "< 10%",
  "early_contact_rate": "< 10%",
  "miss_rate": "< 30%",
  "body_fall_rate": "< 5%",
  "racket_drop_rate": "< 10%"
}
```

通过 Stage 6 后，才进入真正击球飞行目标。

---

# Stage 7：Over-net RL

## 目标

让 shuttle 被击出后过网，不要求落点很准。

## 必须实现

```text
shuttle rebound model
aero / drag model
net crossing detector
net clearance detector
landing detector
flight termination
```

## reward

```text
+ crossed_net
+ net_clearance
+ outgoing velocity direction
+ high arc
- net hit
- own side landing
- out of bounds
```

## 验收标准

```json
{
  "over_net_rate": "> 50%",
  "net_hit_rate": "< 20%",
  "own_side_landing_rate": "< 30%",
  "valid_grip_rate_during_hit": "> 80%"
}
```

---

# Stage 8：High-clear depth RL

## 目标

最终让 shuttle 形成正手高远球：高弧线、过网、落到对方后场。

## reward

```text
+ opponent_back_court_landing
+ apex_height_in_range
+ net_clearance
+ valid stringbed contact
+ body motion prior
- out
- too short
- too flat
- body fall
- racket drop
```

## 验收标准

```json
{
  "opponent_back_landing_rate": "> 30% initially, then > 60%",
  "over_net_rate": "> 70%",
  "out_rate": "< 30%",
  "body_motion_quality_pass": true,
  "grip_stability_pass": true
}
```

---

# 开始 RL 击球前的硬性 Checklist

下面每一项都应该是 `true`，否则不建议开始击球 RL。

```text
[ ] training scene actuation enabled
[ ] Full Body - overall_racket contact allowed
[ ] frozen body policy artifact has params.npz and run_stats.npz
[ ] artifact source checkpoint matches spec checkpoint
[ ] trajectory cache exists and goal_size=469
[ ] BodyObsAdapter builds 2418-dim obs from overall MuJoCo state
[ ] FrozenBodyPolicy NumPy forward matches Flax actor mean
[ ] trajectory-aligned reset implemented
[ ] pose_servo=False replay stable for >=300 steps
[ ] pose_servo=False replay video looks like ForehandClear
[ ] body mimic reward tracks trajectory, not reset pose
[ ] right-hand grip residual short PPO improves contact / slip metrics
[ ] no-shuttle grip-hold success rate high enough
[ ] swing disturbance grip training passes
[ ] ghost racket target exists and is trackable
[ ] stringbed-shuttle contact detector exists
[ ] rebound / aero / flight / landing logic exists
[ ] static-hit env has real termination
```

---

# 可以开始哪些训练？

## 现在可以开始

```text
1. no-shuttle frozen body replay debug
2. no-shuttle grip residual tiny train
3. no-shuttle short PPO, 2k~20k steps
```

前提是最好先关闭 pose servo。

## 暂时不建议开始

```text
1. 长 PPO
2. 多 motion 正式 schedule
3. shuttle impact RL
4. over-net RL
5. high-clear landing RL
```

---

# 最终决策

## 是否可以进入多 motion schedule？

```text
暂不建议正式进入。
```

可以先写 schedule 代码，但不要正式训练。单 motion 在 no-servo 下稳定后再开。

## 是否可以进入更长 PPO？

```text
可以做短程 no-shuttle debug PPO；
不建议直接长训。
```

建议顺序：

```text
2k → 5k → 20k → 50k → 200k
```

每一步都看视频和 reward terms。

## 是否可以加球与击球目标？

```text
暂不建议。
```

先完成无球持拍 / 挥拍稳定。

## 是否可以开始 RL 击球？

```text
现在不建议开始 RL 击球。
```

当前最合理的下一步是：

```text
修正 pose_servo
实现 trajectory-aligned reset
修正 body mimic reward
验证 frozen actor 数值等价
完成 no-shuttle grip-hold short PPO
```

完成这些后，再进入：

```text
ghost racket → static shuttle contact → over-net → high-clear depth
```

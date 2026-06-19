# 1. 阻塞问题：ForehandClear Frozen Body + Grip Residual 当前必须先修的风险

## 目标背景

当前项目目标是：

```text
已训练好的无拍 / 无球 ForehandClear body policy checkpoint
        ↓
作为 frozen body policy 接入 overall badminton training scene
        ↓
actor 输入 checkpoint-compatible 2418 维 body observation
        ↓
frozen actor 输出 354 维 body action
        ↓
按 actuator name 映射到 416 维 overall scene ctrl
        ↓
右手 grip residual action 叠加
        ↓
MuJoCo step
        ↓
只训练右手 grip residual policy，使模型学习正手高远球中的持拍 / 挥拍阶段
```

这条主线现在已经基本打通：`BodyObsAdapter`、`TrajectoryGoalProvider`、`FrozenBodyPolicy` artifact、`CheckpointToFullActionAdapter`、`OverallGripHoldEnv`、`run_forehand_clear_grip_hold.py` 都已经具备雏形，并且已经能完成 real replay smoke 和 tiny train。

但是，目前仍有几个会导致“短 smoke 能跑，但长 PPO 学到错误策略”的阻塞问题。下面的问题建议在正式长训练、加球、击球 RL 之前优先处理。

---

## Blocker 1：`pose_servo=True` 会污染真实学习信号

### 当前风险

`OverallGripHoldEnv.step()` 当前使用 frozen body ctrl + residual ctrl 后，会调用 base env 的 MuJoCo step，并且训练 / replay 路径中使用了 `pose_servo=True`。

这会带来两个核心问题：

```text
1. frozen body policy 与 pose servo 抢控制权
2. pose servo 可能直接或间接帮助稳定球拍，使 residual policy 学到错误依赖
```

如果 `pose_servo` 对全身和 freejoint 都施加稳定力，那么模型可能并不是靠：

```text
手指肌肉激活 + 手柄接触 + 摩擦
```

来持拍，而是靠：

```text
外部 qfrc_applied pose servo
```

来维持姿态 / 球拍位置。

这会导致一个危险假象：

```text
real replay smoke finite
tiny train finite
但去掉 pose servo 后策略立即失败
```

### 为什么这是阻塞问题

当前目标是训练真实物理接触下的 grip residual。`pose_servo` 是外部辅助控制，如果训练时默认开启，reward 可能会把外部伺服稳定性错误归因给 residual action。

这会破坏后续阶段：

```text
多 motion schedule
长 PPO
加球
击球
```

因为这些阶段都要求策略在真实物理下稳定持拍。

### 建议修改

将 `pose_servo` 改成显式 debug 参数，训练默认关闭：

```python
class OverallGripHoldEnv:
    def __init__(
        ...,
        pose_servo: bool = False,
        servo_scope: str = "none",  # none/root/torso/all_debug
    ):
        ...
```

训练 runner 区分：

```bash
# 仅调试用
python BadmintonMimic/scripts/run_forehand_clear_grip_hold.py \
  --stage replay-smoke \
  --pose-servo-debug

# 正式 replay / train 默认
python BadmintonMimic/scripts/run_forehand_clear_grip_hold.py \
  --stage replay-smoke \
  --no-pose-servo

python BadmintonMimic/scripts/run_forehand_clear_grip_hold.py \
  --stage train-tiny \
  --no-pose-servo
```

### 验收标准

必须新增 report 字段：

```json
{
  "pose_servo_enabled": false,
  "servo_scope": "none",
  "servo_force_norm_mean": 0.0,
  "servo_force_norm_max": 0.0
}
```

必须通过：

```text
frozen body policy + zero residual + pose_servo=False
300~1000 steps finite
body_fall=false
body_obs/action finite
goal phase 正常推进
```

---

## Blocker 2：`mimic_body` 当前像 reset-pose reward，不是 trajectory mimic reward

### 当前风险

`OverallGripHoldEnv` 中的 body mimic error 当前是当前 `qpos` 与 reset 时保存的 `_reference_qpos` 之间的差异。也就是说，它更像：

```text
保持 overall_ready reset pose
```

而不是：

```text
跟随 ForehandClear 轨迹
```

如果 frozen body policy 真的开始挥拍，那么 body qpos 会离开 reset pose，此时 `r_mimic_body` 反而会惩罚正确的挥拍动作。

### 另一个隐患：使用 `[-14:]` 忽略 racket/shuttle 过于脆弱

如果当前代码通过：

```python
delta[-14:-7] = 0.0
delta[-7:] = 0.0
```

来忽略 racket / shuttle freejoint，那么它依赖 qpos 排列顺序。一旦 XML 中 body / joint 顺序变化，这个逻辑可能悄悄失效。

### 建议修改

把 body mimic 改成基于 trajectory phase 的 reference tracking：

```python
target_qpos, target_qvel = trajectory_provider.reference_state(
    traj_no=current_traj_no,
    traj_step=current_traj_step,
)

r_mimic_body = compute_body_tracking_error_by_joint_name(
    current_model=overall_model,
    current_data=overall_data,
    target_legacy_qpos=target_qpos,
    target_legacy_qvel=target_qvel,
    exclude_joint_groups=[
        "right_hand_fingers",
        "overall_racket_free",
        "overall_shuttle_free",
    ],
)
```

关键原则：

```text
1. 用 joint name 映射，不用 qpos index 假设
2. 跟随 ForehandClear trajectory，不跟随 reset pose
3. 对右手手指放松 mimic，让 residual policy 有空间修正握拍
4. 对 root/torso/shoulder/elbow/wrist 按阶段设置不同权重
```

### 验收标准

新增测试：

```text
test_body_mimic_reward_uses_trajectory_phase_not_reset_pose
test_body_mimic_excludes_racket_shuttle_by_joint_name
test_body_mimic_excludes_right_hand_fingers_when_residual_controls_fingers
```

训练 report 中输出：

```json
{
  "body_mimic_reference": "trajectory_phase",
  "body_mimic_traj_step": 123,
  "excluded_joint_names": ["overall_racket_free", "overall_shuttle_free", "..."]
}
```

---

## Blocker 3：overall reset 与 trajectory goal 可能没有 phase 对齐

### 当前风险

`TrajectoryGoalProvider` 已经从 trajectory cache 生成真实 `GoalTrajMimic`，这比 zero padding 正确很多。但如果 overall env reset 仍然使用 `overall_ready` keyframe，而 goal provider 从 `video1_lower_body_full_poses` 的第 0 帧开始提供 future goal，就可能出现：

```text
当前 body state = overall_ready 人为初始姿态
goal obs = video1 trajectory step 0 的未来目标
```

这两者不一定是同一个 phase。

结果是：

```text
2418 维 observation shape 正确
goal 非零
actor 输出 finite
但语义不等价于原训练环境中的 observation
```

短程 smoke 可能仍然通过，但长程行为会偏。

### 建议修改

新增 trajectory-aligned reset：

```python
env.reset(
    traj_no=0,
    traj_step=0,
    reset_mode="trajectory_aligned",
)
```

它应该执行：

```text
1. 从 trajectory cache 取 legacy MyoFullBody qpos/qvel
2. 按 joint name 拷贝到 overall model
3. 叠加 right-hand grip seed
4. 根据 palm/grip reference 放置 racket
5. 设置 goal_provider.traj_step = same traj_step
6. mj_forward
```

### 验收标准

新增对齐测试：

```text
test_overall_reset_matches_legacy_trajectory_state_by_joint_name
test_goal_provider_phase_matches_reset_phase
test_body_obs_kinematic_part_matches_legacy_env_at_same_phase
```

至少报告：

```json
{
  "reset_mode": "trajectory_aligned",
  "traj_no": 0,
  "traj_step": 0,
  "legacy_to_overall_joint_copy_count": 100,
  "missing_legacy_joints": [],
  "missing_overall_joints": []
}
```

---

## Blocker 4：FrozenBodyPolicy NumPy forward 缺少真实 Flax 等价性测试

### 当前状态

当前测试已经确认：

```text
obs_size = 2418
action_size = 354
actor_hidden_layers = 12 × 1024 + 3 × 2048 + 1024
activation = silu
use_layernorm = true
actor output kernel shape = (1024, 354)
log_std shape = (354,)
run_stats mean shape = (2418,)
```

这说明 shape 和 metadata 是正确的。

### 当前风险

但 shape 正确不等于 forward 等价。必须验证：

```text
FrozenBodyPolicy.act(obs)
≈
原 Flax ActorCritic.apply(...).pi.mean()
```

尤其是这些细节容易出错：

```text
RunningMeanStd 是否使用 frozen mean/var 还是更新后的 new_mean/new_var
LayerNorm epsilon 是否一致
Dense kernel/bias 命名是否一致
silu 是否完全一致
actor_obs_ind 是否完全等价
是否误包含 critic 参数
输出是否是 actor_mean，而不是 sample / tanh / clipped action
```

### 建议修改

新增真实 checkpoint 数值等价测试：

```python
def test_numpy_frozen_policy_matches_flax_actor_mean_on_real_checkpoint():
    obs = build_real_checkpoint_compatible_obs(...)
    np_action = FrozenBodyPolicy.load_from_export(...).act(obs)

    flax_action = flax_actor_mean_from_original_actorcritic(
        checkpoint=CHECKPOINT,
        obs=obs,
    )

    np.testing.assert_allclose(
        np_action,
        flax_action,
        rtol=1e-5,
        atol=1e-5,
    )
```

如果由于 RunningMeanStd mutable 行为无法完全一致，则至少输出并约束：

```text
max_abs_diff < 1e-4
mean_abs_diff < 1e-5
cosine_similarity > 0.9999
```

### 验收标准

长 PPO 前必须有：

```json
{
  "numpy_vs_flax_actor_mean_checked": true,
  "max_abs_action_diff": 0.00001,
  "mean_abs_action_diff": 0.000001,
  "cosine_similarity": 0.99999
}
```

---

## Blocker 5：10-step real replay smoke 不足以证明可训练

### 当前状态

你已经验证：

```text
policy_replay_ready = true
steps_completed = 10
body_obs_size = 2418
body_action_size = 354
goal_obs_source = trajectory_cache
finite = true
```

这证明 pipeline 能跑通，但不能证明策略可以长时间 replay，也不能证明 reward 信号正确。

### 建议提高验收门槛

至少新增：

```text
100-step replay smoke
300-step replay smoke
1000-step replay smoke
```

并且必须包含：

```text
pose_servo=False
zero residual
trajectory-aligned reset
frozen body action only
```

### 验收指标

```json
{
  "steps_completed": 1000,
  "finite": true,
  "body_fall": false,
  "racket_drop_rate": 0.0,
  "body_obs_size": 2418,
  "body_action_size": 354,
  "goal_obs_source": "trajectory_cache",
  "body_goal_traj_step_final": 999,
  "raw_body_action_max_abs": 2.1,
  "full_ctrl_saturation_rate": 0.03
}
```

同时必须人工检查视频：

```text
动作是否仍像 ForehandClear
是否真的有挥拍
球拍是否靠手接触保持
是否出现抖动 / 漂移 / 穿模
```

---

## Blocker 6：`r_racket_hand_pose` 可能鼓励错误的“越近越好”

### 当前风险

如果 `r_racket_hand_pose` 是：

```python
-racket_hand_pose_weight * palm_to_grip_m
```

那么它会鼓励 `palm_to_grip_m` 越小越好。这样可能导致：

```text
手掌把球拍吸进去
handle 与手部穿透
牺牲真实握拍姿态
reward hacking
```

更合理的是相对 reference transform：

```text
当前 palm→grip transform
vs
参考 palm→grip transform
```

### 建议修改

改成：

```python
r_racket_hand_pose = -w * norm(
    current_palm_to_grip_vector - reference_palm_to_grip_vector
)
```

可进一步加入 orientation：

```text
palm frame 下 grip pose 的位置误差
palm frame 下 racket orientation 误差
```

### 验收标准

新增测试：

```text
test_racket_hand_pose_reward_is_reference_transform_error
test_racket_hand_pose_reward_does_not_prefer_zero_distance
```

---

## Blocker 7：artifact / cache 仍是隐式依赖

### 当前风险

当前训练依赖：

```text
outputs/frozen_body_policy/de63059b16c0_7812
caches/AMASS/MyoFullBody/gmr/10trajectories/video1_lower_body_full_poses.npz
environment/overall_environment/assets/overall_badminton_training_scene.xml
outputs/right_hand_racket_grip/reference/right_hand_racket_grip_seed.json
```

这些 artifact 如果缺失、过期或来自不同 checkpoint，训练可能出现：

```text
shape mismatch
goal mismatch
actor 输出异常
动作语义错误
```

### 建议修改

新增统一准备脚本：

```bash
python BadmintonMimic/scripts/prepare_forehand_clear_grip_hold_artifacts.py \
  --spec BadmintonMimic/experiments/posttrain/forehand_clear_grip_hold_v1.yaml \
  --build-training-scene \
  --export-frozen-policy \
  --check-trajectory-cache \
  --check-grip-seed
```

### 验收标准

输出：

```json
{
  "training_scene_ok": true,
  "frozen_policy_artifact_ok": true,
  "artifact_source_checkpoint_matches_spec": true,
  "params_npz_exists": true,
  "run_stats_npz_exists": true,
  "trajectory_cache_ok": true,
  "goal_size": 469,
  "body_obs_size": 2418,
  "body_action_size": 354,
  "overall_action_size": 416,
  "grip_seed_ok": true
}
```

---

## 当前是否可以绕过这些 blocker？

建议不要。
可以继续做非常短的 debug run，但不建议：

```text
长 PPO
多 motion 正式训练
加球
击球 RL
```

在这些 blocker 修完前，训练结果很可能不可解释。

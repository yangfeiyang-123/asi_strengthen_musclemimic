# 2. 优先建议与其次建议：下一轮工程修改计划

## 总体策略

当前系统已经从 “precheck/staging” 进入 “frozen body replay + grip residual tiny train” 阶段。下一轮工程重点不应继续堆 YAML 或复杂 reward，而应保证：

```text
1. frozen body policy 输入语义正确
2. frozen actor forward 数值等价
3. 训练物理信号不被 pose servo 污染
4. residual grip policy 真正通过手柄接触学习持拍
5. artifact/cache/scene 可复现
```

下面按优先级拆分为高优先级建议和中低优先级建议。

---

# A. 高优先级建议

## P0-1：关闭训练默认 `pose_servo`

### 修改目标

训练默认：

```text
pose_servo = False
```

调试可显式开启：

```text
pose_servo_debug = True
```

### 推荐实现

在 `OverallGripHoldEnv.__init__` 加：

```python
pose_servo: bool = False
servo_scope: str = "none"
```

在 `step()` 中：

```python
obs, base_info = self.base_env.step(
    ctrl=full_ctrl,
    pose_servo=self.pose_servo,
)
```

在 runner CLI 中加：

```bash
--pose-servo-debug
--no-pose-servo
```

训练 report 中写：

```json
{
  "pose_servo_enabled": false,
  "servo_scope": "none"
}
```

### 验收测试

```text
test_train_tiny_defaults_to_no_pose_servo
test_replay_smoke_can_enable_pose_servo_debug_explicitly
test_pose_servo_reported_in_metrics
```

---

## P0-2：实现 trajectory-aligned reset

### 修改目标

让 overall state 与 `GoalTrajMimic` 的 phase 对齐。

### 推荐实现

新增：

```python
class OverallGripHoldEnv:
    def reset(
        self,
        *,
        traj_no: int = 0,
        traj_step: int = 0,
        reset_mode: str = "trajectory_aligned",
    ):
        ...
```

核心逻辑：

```text
1. goal_provider.reset(traj_step=traj_step)
2. 从 trajectory handler 取 legacy qpos/qvel
3. 按 joint name 拷贝 legacy body state 到 overall body state
4. 应用 right-hand grip seed
5. 放置 racket 到手柄参考位置
6. mj_forward
```

### 验收测试

```text
test_trajectory_aligned_reset_sets_goal_provider_same_step
test_trajectory_aligned_reset_copies_legacy_joints_by_name
test_trajectory_aligned_reset_preserves_grip_seed_hand_pose
test_trajectory_aligned_reset_places_racket_near_palm
```

### 验收 report

```json
{
  "reset_mode": "trajectory_aligned",
  "traj_no": 0,
  "traj_step": 0,
  "legacy_joint_copy_count": 120,
  "missing_joints": [],
  "goal_traj_step": 0
}
```

---

## P0-3：把 body mimic reward 改成 trajectory-phase mimic

### 修改目标

当前 `mimic_body` 不应奖励 reset pose，而应奖励跟随当前 trajectory phase。

### 推荐实现

把：

```python
_body_mimic_error = mean_square(data.qpos - reference_reset_qpos)
```

替换为：

```python
_body_mimic_error = trajectory_body_tracking_error(
    overall_model,
    overall_data,
    legacy_reference_qpos,
    legacy_reference_qvel,
    phase=current_traj_step,
    joint_weights=...
)
```

### 推荐权重分组

```yaml
body_mimic_weights:
  root: 1.0
  torso: 1.0
  shoulder: 0.8
  elbow: 0.8
  wrist: 0.2
  right_hand_fingers: 0.0
  racket: 0.0
  shuttle: 0.0
```

### 验收测试

```text
test_body_mimic_tracks_trajectory_not_reset
test_body_mimic_uses_joint_names_not_tail_indices
test_right_hand_fingers_excluded_from_body_mimic_when_residual_controls_fingers
```

---

## P0-4：增加 FrozenBodyPolicy NumPy vs Flax 等价性测试

### 修改目标

证明：

```text
FrozenBodyPolicy.act(obs) == 原 ActorCritic deterministic actor_mean
```

### 推荐测试

```python
def test_numpy_frozen_policy_matches_flax_actor_mean_real_checkpoint():
    obs = build_checkpoint_compatible_obs(...)
    np_action = frozen_policy.act(obs)
    flax_action = original_flax_actor_mean(checkpoint, obs)

    assert max_abs_diff < 1e-5
    assert mean_abs_diff < 1e-6
```

### 需要特别检查

```text
RunningMeanStd
LayerNorm
silu
Dense kernel/bias
actor_obs_ind
output layer linear
log_std 不参与 deterministic mean action
```

### 验收 report

```json
{
  "numpy_vs_flax_checked": true,
  "max_abs_action_diff": 0.00001,
  "mean_abs_action_diff": 0.000001,
  "cosine_similarity": 0.99999
}
```

---

## P0-5：统一 reward 配置命名并写入 effective reward report

### 当前问题

YAML 中有：

```yaml
reward:
  mimic: 1.0
  root_stability: 1.0
```

但 env 中使用的是：

```text
mimic_body
grip_site
contact
no_slip
no_penetration
racket_hand_pose
residual_effort
```

这可能导致配置没有真正生效。

### 推荐修改

统一为：

```yaml
reward:
  mimic_body: 0.2
  root_stability: 0.0
  grip_site: 8.0
  contact: 2.0
  no_slip: 8.0
  no_penetration: 10.0
  racket_hand_pose: 4.0
  residual_effort: 0.01
```

在 runner 中显式传入：

```python
env = OverallGripHoldEnv(
    ...,
    reward_weights=spec["reward"],
)
```

### 验收测试

```text
test_spec_reward_weights_are_passed_to_env
test_unknown_reward_key_fails_fast
test_effective_reward_weights_written_to_metrics
```

---

## P1-1：显式处理 residual 与 body action 的 actuator overlap

### 当前风险

stage1 只控制 fingers，通常不与 body checkpoint actuator 重叠。
stage2 控制 fingers + wrist + forearm，其中 wrist/forearm 很可能已经由 frozen body policy 控制。

### 推荐策略

新增配置：

```yaml
residual_policy:
  mode: additive
  scale:
    right_hand_fingers: 1.0
    right_wrist: 0.2
    right_forearm: 0.15
  allow_overlap_with_body_policy:
    right_hand_fingers: false
    right_wrist: true
    right_forearm: true
```

在 metrics 中报告：

```json
{
  "residual_overlap_actuators": ["ECRL", "ECRB", "SUP", "..."],
  "residual_extra_only_actuators": ["FDS2", "FDP2", "..."],
  "residual_scale_by_group": {
    "right_hand_fingers": 1.0,
    "right_wrist": 0.2
  }
}
```

### 验收测试

```text
test_stage1_residual_has_no_body_overlap
test_stage2_residual_overlap_is_explicitly_allowed
test_residual_group_scale_applied_before_ctrl_clip
```

---

## P1-2：增强 residual policy observation

### 当前问题

如果 residual policy 只看到 overall qpos/qvel，学习会比较困难。
建议加入 grip-specific features：

```text
phase / traj_step normalized
palm_to_grip vector
palm_to_grip distance
grip_slip
hand-handle contact count
contact flags per hand geom
racket pose relative to palm
racket velocity relative to palm
previous residual action
body action on overlapping actuators
```

### 推荐实现

```python
def _residual_observation(self):
    return np.concatenate([
        base_qpos_qvel,
        phase_features,
        grip_features,
        contact_features,
        previous_action,
    ])
```

### 验收测试

```text
test_residual_obs_contains_phase_and_grip_features
test_residual_obs_is_finite
test_residual_obs_size_reported
```

---

## P1-3：修正 `r_racket_hand_pose`

### 当前问题

如果 reward 是：

```python
-racket_hand_pose_weight * palm_to_grip_m
```

它会鼓励 palm 与 grip 距离越小越好，可能造成穿模或错误持拍。

### 推荐修改

改成 reference transform error：

```python
current = current_palm_to_grip_vector
target = reference_palm_to_grip_vector
r_racket_hand_pose = -w * ||current - target||
```

可进一步使用 palm local frame：

```python
current_local = palm_xmat.T @ (grip_pos - palm_pos)
target_local = reference_palm_to_grip_local
```

### 验收测试

```text
test_racket_hand_pose_uses_reference_transform
test_reward_does_not_increase_when_grip_distance_collapses_to_zero
```

---

## P1-4：增强 training scene validator

### 当前 validator 已检查

```text
overall_ready keyframe
actuator_count > 0
required sites/geoms
Full Body - overall_racket contact exclude 不存在
```

### 建议新增检查

```text
actuation_disabled = false
racket_freejoint exists
shuttle_freejoint exists
hand-handle contact possible
handle bevel geoms contype/conaffinity valid
right-hand contact geoms conaffinity valid
soft_weld exists or not, depending on config
```

### 验收 report

```json
{
  "actuation_enabled": true,
  "racket_freejoint_exists": true,
  "hand_handle_contact_possible": true,
  "soft_weld_enabled": false,
  "handle_contact_geom_count": 8,
  "right_hand_contact_geom_count": 12
}
```

---

## P1-5：自动化 artifact / cache / scene 准备

### 推荐新增脚本

```bash
python BadmintonMimic/scripts/prepare_forehand_clear_grip_hold_artifacts.py \
  --spec BadmintonMimic/experiments/posttrain/forehand_clear_grip_hold_v1.yaml \
  --build-training-scene \
  --export-frozen-policy \
  --check-trajectory-cache \
  --check-grip-seed
```

### 功能

```text
1. 读取 spec
2. 检查 checkpoint 是否存在
3. 构建 training scene
4. 导出 frozen body policy artifact
5. 检查 params.npz / run_stats.npz
6. 检查 trajectory cache
7. 检查 grip seed
8. 跑 body obs / action / ctrl shape smoke
9. 输出 JSON report
```

### 验收测试

```text
test_prepare_artifacts_writes_complete_report
test_prepare_artifacts_fails_if_cache_missing
test_prepare_artifacts_fails_if_artifact_checkpoint_mismatch
```

---

# B. 中低优先级建议

## P2-1：让 export CLI 必须显式指定导出模式

当前 CLI 容易生成 metadata-only artifact。建议改成：

```text
必须指定 --metadata-only 或 --restore-tensors
```

如果不指定，直接报错：

```text
Please choose --metadata-only or --restore-tensors.
```

这样避免用户误以为 artifact 可训练。

---

## P2-2：增加 action/obs 分布统计

训练和 replay metrics 建议加入：

```text
body_obs_norm_mean
body_obs_norm_std
body_obs_norm_max_abs
goal_obs_mean_abs
raw_body_action_mean_abs
raw_body_action_max_abs
full_ctrl_saturation_rate
residual_action_mean_abs
residual_action_max_abs
```

这有助于发现：

```text
normalizer 错
actor 输出尺度异常
goal 语义错
ctrl 长期饱和
```

---

## P2-3：增加视频验证脚本

建议加：

```bash
python BadmintonMimic/scripts/run_forehand_clear_grip_hold.py \
  --stage replay-video \
  --steps 300 \
  --no-pose-servo
```

输出：

```text
frozen_body_replay_no_servo.mp4
grip_residual_train_eval.mp4
```

人工检查比纯 metric 更重要，因为可能出现：

```text
finite=true 但动作不像 ForehandClear
身体被锁在 reset pose
球拍靠非物理约束稳定
```

---

## P2-4：多 motion schedule 参数化

先不要正式开启多 motion 长训，但可以提前让 spec 支持：

```yaml
goal_provider:
  motion_schedule:
    - motion_index: 0
    - motion_index: 1
    - motion_index: 2
  random_start: false
  random_phase_after_stable: true
```

等待单 motion 稳定后再启用。

---

## P2-5：文档化训练前 checklist

新增：

```text
docs/forehand_clear_grip_hold_training_checklist.md
```

内容包括：

```text
1. 构建 training scene
2. 导出 frozen policy artifact
3. 检查 actor 数值等价
4. 检查 trajectory goal cache
5. 无 servo replay
6. tiny train
7. short no-shuttle train
8. long no-shuttle train
9. 才能加 shuttle
```

---

# 推荐下一轮 PR 划分

## PR-1：No-servo replay correctness

包含：

```text
pose_servo 默认关闭
replay-smoke 支持 --pose-servo-debug
300-step no-servo replay
metrics 增加 servo 信息
```

## PR-2：Trajectory-aligned reset + trajectory body mimic

包含：

```text
reset 与 trajectory phase 对齐
body mimic 使用 trajectory reference
joint-name based exclusion
```

## PR-3：Frozen actor numerical equivalence

包含：

```text
NumPy actor vs Flax actor mean 测试
export artifact 校验增强
```

## PR-4：Grip residual training signal cleanup

包含：

```text
reward key 统一
racket_hand_pose 改 reference transform error
residual overlap scale
grip-specific observation
```

## PR-5：Artifact automation

包含：

```text
prepare_forehand_clear_grip_hold_artifacts.py
cache/artifact/scene/grip seed 一键检查
```

---

# 当前建议执行顺序

```text
1. PR-1 No-servo replay correctness
2. PR-2 Trajectory-aligned reset + mimic reward
3. PR-3 Frozen actor equivalence
4. PR-4 Reward / residual signal cleanup
5. PR-5 Artifact automation
6. 2k~20k no-shuttle debug PPO
7. 50k~200k no-shuttle grip-hold PPO
8. 再考虑加球和击球
```

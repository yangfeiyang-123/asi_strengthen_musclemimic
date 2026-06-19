# Step3 下一步实施计划：No-Servo Trajectory-Aligned Grip-Hold Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把当前“能短跑”的 frozen body policy + grip residual pipeline，提升到可以可信地进行 no-shuttle grip-hold 短 PPO 的状态，避免 pose servo、reset-pose reward、phase 不对齐等问题污染学习信号。

**Architecture:** 保留当前已打通的主线：`BodyObsAdapter -> TrajectoryGoalProvider -> FrozenBodyPolicy -> CheckpointToFullActionAdapter -> OverallGripHoldEnv`。下一步不急于加球，而是先把 replay/training 变成 no-servo、trajectory-aligned、trajectory-phase reward，并补齐 artifact/actor 数值校验。

**Tech Stack:** Python, MuJoCo, NumPy, JAX/Flax/Orbax, PyTorch PPO tiny trainer, pytest, existing `musclemimic` / `loco_mujoco` trajectory cache。

---

## 0. GPT Pro 建议与当前代码核验结论

### 已经完成，可以保留

- `FrozenBodyPolicy` 已可从 `outputs/frozen_body_policy/de63059b16c0_7812` 离线加载 `params.npz/run_stats.npz`，并输出 354 维动作。
- `TrajectoryGoalProvider` 已从真实 GMR cache 生成 469 维 `GoalTrajMimic`，不是 zero padding。
- `run_forehand_clear_grip_hold.py --stage replay-smoke --policy-source real` 已能构造 2418 维 body obs，并把 354 维 action 映射到 416 维 overall ctrl。
- `OverallGripHoldEnv` 已支持 frozen body policy artifact，并能跑 tiny train。

### 必须优先修正

- `OverallGripHoldEnv.step()` 仍硬编码 `pose_servo=True`，`run_forehand_clear_grip_hold.py` 的 real/fake replay 也仍硬编码 `pose_servo=True`。
- `_body_mimic_error()` 当前比较当前 qpos 与 reset qpos，并用 `delta[-14:-7]` / `delta[-7:]` 忽略 racket/shuttle；这既不是 trajectory mimic，也依赖 qpos 尾部顺序。
- `OverallGripHoldEnv.reset()` 仍只从 `overall_ready` keyframe reset，没有把 body phase 对齐到 trajectory cache 的同一帧。
- `FrozenBodyPolicy.act()` 只有 shape/finite 测试，缺少真实 checkpoint 的 NumPy actor forward vs Flax ActorCritic deterministic mean 等价测试。
- YAML 中 `reward.mimic`、`reward.root_stability` 与 env 内部 `mimic_body` 等 key 不一致，且 runner 没有把 spec reward 传入 env。

---

## PR-1 / Task 1: No-Servo Replay Correctness

**目标:** 训练和 real replay 默认不使用 pose servo；pose servo 只能作为显式 debug 开关。所有 report/metrics 必须写出 servo 状态和 servo force norm。

**Files:**
- Modify: `environment/overall_environment/src/overall_grip_hold_env.py`
- Modify: `BadmintonMimic/scripts/run_forehand_clear_grip_hold.py`
- Modify: `environment/overall_environment/tests/test_overall_grip_hold_env.py`
- Modify: `tests/unit/test_forehand_clear_grip_hold_runner.py`

- [ ] **Step 1: 写 failing tests**

在 `environment/overall_environment/tests/test_overall_grip_hold_env.py` 增加：

```python
def test_overall_grip_hold_env_defaults_to_no_pose_servo():
    env = OverallGripHoldEnv(
        default_training_scene_path(),
        residual_groups=["right_hand_fingers"],
        body_policy_artifact="outputs/frozen_body_policy/de63059b16c0_7812",
    )

    env.reset()
    _, _, _, _, info = env.step(np.zeros(env.action_size, dtype=float))

    assert info["pose_servo_enabled"] is False
    assert info["servo_scope"] == "none"
    assert info["servo_force_norm_max"] == 0.0
```

在 `tests/unit/test_forehand_clear_grip_hold_runner.py` 增加：

```python
def test_replay_smoke_real_policy_defaults_to_no_pose_servo(tmp_path: Path):
    paths = load_grip_hold_spec(SPEC)

    report = replay_smoke(paths, out_dir=tmp_path, steps=5, policy_source="real")

    assert report["pose_servo_enabled"] is False
    assert report["servo_scope"] == "none"
    assert report["policy_replay_ready"] is True
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
env MPLCONFIGDIR=/data3/yangfeiyang/WorkSpace/ENV/tmp/matplotlib XDG_CACHE_HOME=/data3/yangfeiyang/WorkSpace/ENV/tmp/fontcache \
/data3/yangfeiyang/WorkSpace/ENV/musclemimic/.venv/bin/python3 -m pytest \
  environment/overall_environment/tests/test_overall_grip_hold_env.py::test_overall_grip_hold_env_defaults_to_no_pose_servo \
  tests/unit/test_forehand_clear_grip_hold_runner.py::test_replay_smoke_real_policy_defaults_to_no_pose_servo -q
```

Expected: FAIL，因为当前没有 `pose_servo_enabled` report，且代码硬编码 `pose_servo=True`。

- [ ] **Step 3: 实现 no-servo 参数**

在 `OverallGripHoldEnv.__init__` 增加：

```python
pose_servo: bool = False,
servo_scope: str = "none",
```

保存：

```python
self.pose_servo = bool(pose_servo)
self.servo_scope = str(servo_scope)
if self.servo_scope not in {"none", "all_debug"}:
    raise ValueError(f"unsupported servo_scope: {self.servo_scope!r}")
if self.pose_servo and self.servo_scope == "none":
    self.servo_scope = "all_debug"
self._last_servo_force_norm_max = 0.0
```

把 `step()` 中的：

```python
obs, base_info = self.base_env.step(ctrl=full_ctrl, pose_servo=True)
```

改为：

```python
obs, base_info = self.base_env.step(ctrl=full_ctrl, pose_servo=self.pose_servo)
self._last_servo_force_norm_max = (
    float(np.linalg.norm(self.data.qfrc_applied))
    if self.pose_servo
    else 0.0
)
```

在 `_body_policy_info()` 或新的 `_servo_info()` 中写：

```python
"pose_servo_enabled": bool(self.pose_servo),
"servo_scope": self.servo_scope,
"servo_force_norm_max": float(self._last_servo_force_norm_max),
```

- [ ] **Step 4: runner 增加 CLI/debug 参数**

在 `replay_smoke()` 增加参数：

```python
pose_servo_debug: bool = False,
```

real/fake replay step 使用：

```python
pose_servo=pose_servo_debug
```

report 写入：

```python
"pose_servo_enabled": bool(pose_servo_debug),
"servo_scope": "all_debug" if pose_servo_debug else "none",
```

在 argparse 增加：

```python
parser.add_argument("--pose-servo-debug", action="store_true")
```

调用 `replay_smoke(..., pose_servo_debug=args.pose_servo_debug)`。

- [ ] **Step 5: 验证**

Run:

```bash
env MPLCONFIGDIR=/data3/yangfeiyang/WorkSpace/ENV/tmp/matplotlib XDG_CACHE_HOME=/data3/yangfeiyang/WorkSpace/ENV/tmp/fontcache \
/data3/yangfeiyang/WorkSpace/ENV/musclemimic/.venv/bin/python3 -m pytest \
  environment/overall_environment/tests/test_overall_grip_hold_env.py \
  tests/unit/test_forehand_clear_grip_hold_runner.py -q
```

Expected: PASS。

- [ ] **Step 6: 提交**

```bash
git add environment/overall_environment/src/overall_grip_hold_env.py \
  BadmintonMimic/scripts/run_forehand_clear_grip_hold.py \
  environment/overall_environment/tests/test_overall_grip_hold_env.py \
  tests/unit/test_forehand_clear_grip_hold_runner.py
git commit -m "Disable pose servo by default for grip hold"
```

---

## PR-2 / Task 2: Trajectory-Aligned Reset

**目标:** `OverallGripHoldEnv.reset()` 支持 `reset_mode="trajectory_aligned"`，把 overall body qpos/qvel 对齐到 `TrajectoryGoalProvider` 当前 trajectory frame，而不是固定 `overall_ready`。

**Files:**
- Modify: `environment/overall_environment/src/trajectory_goal_provider.py`
- Modify: `environment/overall_environment/src/overall_grip_hold_env.py`
- Test: `environment/overall_environment/tests/test_overall_grip_hold_env.py`
- Test: `environment/overall_environment/tests/test_body_obs_adapter.py`

- [ ] **Step 1: 写 failing tests**

在 `test_overall_grip_hold_env.py` 增加：

```python
def test_trajectory_aligned_reset_sets_goal_provider_same_step():
    env = OverallGripHoldEnv(
        default_training_scene_path(),
        residual_groups=["right_hand_fingers"],
        body_policy_artifact="outputs/frozen_body_policy/de63059b16c0_7812",
        reset_mode="trajectory_aligned",
    )

    _, info = env.reset(traj_step=12)

    assert info["reset_mode"] == "trajectory_aligned"
    assert info["body_goal_next_traj_step"] == 12
    assert info["legacy_to_overall_joint_copy_count"] > 80
    assert info["missing_legacy_joints"] == []
```

在 `test_body_obs_adapter.py` 增加：

```python
def test_trajectory_aligned_reset_body_obs_kinematics_match_legacy_phase():
    env = OverallGripHoldEnv(
        default_training_scene_path(),
        residual_groups=["right_hand_fingers"],
        body_policy_artifact="outputs/frozen_body_policy/de63059b16c0_7812",
        reset_mode="trajectory_aligned",
    )
    env.reset(traj_step=5)
    goal_obs = env.body_goal_provider.build(env.model, env.data)
    body_obs = env.body_obs_adapter.build_from_mujoco(env.model, env.data, goal_obs=goal_obs)

    assert body_obs.shape == (2418,)
    assert np.isfinite(body_obs).all()
    assert env.body_goal_provider.last_built_step == 5
```

- [ ] **Step 2: 运行测试确认失败**

Expected: FAIL，因为当前 `OverallGripHoldEnv.reset()` 不接受 `traj_step/reset_mode`。

- [ ] **Step 3: 给 `TrajectoryGoalProvider` 暴露 reference state**

新增 dataclass：

```python
@dataclass(frozen=True)
class TrajectoryReferenceState:
    qpos: np.ndarray
    qvel: np.ndarray
    traj_step: int
    traj_len: int
```

新增方法：

```python
def reference_state(self, traj_step: int | None = None) -> TrajectoryReferenceState:
    step = self._clamp_step(self._traj_step if traj_step is None else traj_step)
    data = self._trajectory_handler.traj.data.get(0, step, np)
    return TrajectoryReferenceState(
        qpos=np.asarray(data.qpos, dtype=float).copy(),
        qvel=np.asarray(data.qvel, dtype=float).copy(),
        traj_step=step,
        traj_len=self.traj_len,
    )
```

- [ ] **Step 4: 实现 joint-name copy 到 overall**

在 `OverallGripHoldEnv` 新增：

```python
def _copy_legacy_reference_to_overall(self, reference) -> dict[str, Any]:
    copy_count = 0
    missing = []
    legacy_model = self.body_goal_provider._legacy_model
    for legacy_joint_id in range(legacy_model.njnt):
        name = mujoco.mj_id2name(legacy_model, mujoco.mjtObj.mjOBJ_JOINT, legacy_joint_id)
        overall_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if overall_joint_id < 0:
            missing.append(name)
            continue
        # qpos/qvel width checks use existing helper widths or local helpers.
        ...
        copy_count += 1
    mujoco.mj_forward(self.model, self.data)
    return {
        "legacy_to_overall_joint_copy_count": copy_count,
        "missing_legacy_joints": missing,
    }
```

实现时必须用 joint name，不允许再用 `[-14:]` 或 qpos 尾部假设。

- [ ] **Step 5: `reset()` 增加 reset mode**

签名改为：

```python
def reset(
    self,
    *,
    traj_step: int = 0,
    reset_mode: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
```

默认：

```python
self.reset_mode = reset_mode or ("trajectory_aligned" if self.body_goal_provider else "keyframe")
```

流程：

```python
obs, base_info = self.base_env.reset()
if self.body_goal_provider is not None:
    self.body_goal_provider.reset(traj_step=traj_step)
if active_reset_mode == "trajectory_aligned":
    ref = self.body_goal_provider.reference_state(traj_step)
    alignment_info = self._copy_legacy_reference_to_overall(ref)
    obs = self.base_env._observation()
```

先保留现有 racket keyframe 位置作为稳定基线；如果 palm-to-grip 初始误差明显变差，再单独开一个后续任务做 racket placement by reference transform。

- [ ] **Step 6: 验证**

Run:

```bash
env MPLCONFIGDIR=/data3/yangfeiyang/WorkSpace/ENV/tmp/matplotlib XDG_CACHE_HOME=/data3/yangfeiyang/WorkSpace/ENV/tmp/fontcache \
/data3/yangfeiyang/WorkSpace/ENV/musclemimic/.venv/bin/python3 -m pytest \
  environment/overall_environment/tests/test_overall_grip_hold_env.py \
  environment/overall_environment/tests/test_body_obs_adapter.py -q
```

Expected: PASS。

---

## PR-3 / Task 3: Trajectory-Phase Body Mimic Reward

**目标:** `r_mimic_body` 从 reset-pose error 改为 trajectory reference tracking error，且按 joint name 排除 racket/shuttle/right hand fingers。

**Files:**
- Create: `environment/overall_environment/src/body_tracking_reward.py`
- Modify: `environment/overall_environment/src/overall_grip_hold_env.py`
- Test: `environment/overall_environment/tests/test_overall_grip_hold_env.py`

- [ ] **Step 1: 写 failing tests**

新增：

```python
def test_body_mimic_reward_uses_trajectory_phase_not_reset_pose():
    env = OverallGripHoldEnv(
        default_training_scene_path(),
        residual_groups=["right_hand_fingers"],
        body_policy_artifact="outputs/frozen_body_policy/de63059b16c0_7812",
        reset_mode="trajectory_aligned",
    )
    env.reset(traj_step=0)
    _, _, _, _, info0 = env.step(np.zeros(env.action_size, dtype=float))
    env.reset(traj_step=20)
    _, _, _, _, info20 = env.step(np.zeros(env.action_size, dtype=float))

    assert info0["body_mimic_reference"] == "trajectory_phase"
    assert info20["body_mimic_traj_step"] == 20
    assert info0["excluded_joint_names"]
```

新增：

```python
def test_body_mimic_excludes_racket_shuttle_and_right_hand_fingers_by_name():
    env = OverallGripHoldEnv(
        default_training_scene_path(),
        residual_groups=["right_hand_fingers"],
        body_policy_artifact="outputs/frozen_body_policy/de63059b16c0_7812",
        reset_mode="trajectory_aligned",
    )
    env.reset()
    _, _, _, _, info = env.step(np.zeros(env.action_size, dtype=float))

    excluded = set(info["excluded_joint_names"])
    assert "overall_racket_free" in excluded
    assert "overall_shuttle_free" in excluded
    assert any("MCP" in name or "PIP" in name or "DIP" in name for name in excluded)
```

- [ ] **Step 2: 实现 `body_tracking_reward.py`**

提供：

```python
@dataclass(frozen=True)
class BodyTrackingReport:
    error: float
    reference: str
    traj_step: int
    compared_joint_count: int
    excluded_joint_names: tuple[str, ...]
```

提供：

```python
def trajectory_body_tracking_error(
    overall_model: mujoco.MjModel,
    overall_data: mujoco.MjData,
    legacy_model: mujoco.MjModel,
    reference_qpos: np.ndarray,
    reference_qvel: np.ndarray,
    *,
    traj_step: int,
    residual_actuator_names: tuple[str, ...],
) -> BodyTrackingReport:
    ...
```

排除规则：

```python
def _excluded_joint(name: str) -> bool:
    return (
        name in {"overall_racket_free", "overall_shuttle_free"}
        or name.lower().startswith(("mcp", "pip", "dip"))
        or "thumb" in name.lower()
        or "finger" in name.lower()
    )
```

比较 qpos/qvel 时都按 joint name 找 adr/width，不能用数组尾部假设。

- [ ] **Step 3: 接入 env**

在 `_body_mimic_error()` 改为返回 report 或缓存 report：

```python
if self.body_goal_provider is None:
    return reset_pose_report
reference = self.body_goal_provider.reference_state(self.body_goal_provider.last_built_step)
report = trajectory_body_tracking_error(...)
self._last_body_tracking_report = report
return report.error
```

`_info()` 写入：

```python
"body_mimic_reference": report.reference,
"body_mimic_traj_step": report.traj_step,
"excluded_joint_names": list(report.excluded_joint_names),
"body_mimic_compared_joint_count": report.compared_joint_count,
```

- [ ] **Step 4: 验证**

Run:

```bash
env MPLCONFIGDIR=/data3/yangfeiyang/WorkSpace/ENV/tmp/matplotlib XDG_CACHE_HOME=/data3/yangfeiyang/WorkSpace/ENV/tmp/fontcache \
/data3/yangfeiyang/WorkSpace/ENV/musclemimic/.venv/bin/python3 -m pytest environment/overall_environment/tests/test_overall_grip_hold_env.py -q
```

Expected: PASS。

---

## PR-4 / Task 4: Frozen NumPy Actor vs Flax Actor Mean Equivalence

**目标:** 证明 `FrozenBodyPolicy.act(obs)` 与原 Flax `ActorCritic` deterministic actor mean 数值一致。

**Files:**
- Modify: `environment/overall_environment/src/frozen_body_policy.py`
- Modify: `environment/overall_environment/tests/test_frozen_body_policy.py`

- [ ] **Step 1: 写 failing test**

新增：

```python
def test_numpy_frozen_policy_matches_flax_actor_mean_real_checkpoint():
    from environment.overall_environment.src.frozen_body_policy import (
        FrozenBodyPolicy,
        restore_flax_actor_mean_for_verification,
    )

    policy = FrozenBodyPolicy.load_from_export("outputs/frozen_body_policy/de63059b16c0_7812")
    obs = np.zeros(policy.actor_spec.obs_size, dtype=np.float32)

    numpy_action = policy.act(obs)
    flax_action = restore_flax_actor_mean_for_verification(CHECKPOINT, obs)

    diff = np.abs(numpy_action - flax_action)
    assert float(diff.max()) <= 1e-5
    assert float(diff.mean()) <= 1e-6
```

- [ ] **Step 2: 实现 verification helper**

在 `frozen_body_policy.py` 增加：

```python
def restore_flax_actor_mean_for_verification(checkpoint: str | Path, obs: np.ndarray) -> np.ndarray:
    spec = reconstruct_actor_checkpoint_spec(checkpoint)
    params, run_stats = _restore_checkpoint_policy_tensors(Path(checkpoint))
    network = _network_from_spec(spec)
    variables = {"params": params, "run_stats": run_stats}
    dist, _value = network.apply(variables, jnp.asarray(obs, dtype=jnp.float32), mutable=False)
    return np.asarray(dist.mean(), dtype=np.float32)
```

如果 `dist.mean()` API 不存在，检查 `musclemimic/algorithms/common/networks.py` 中 ActorCritic 返回对象的字段，改用实际字段名。验收必须记录真实 API，不允许猜。

- [ ] **Step 3: 验证**

Run:

```bash
env MPLCONFIGDIR=/data3/yangfeiyang/WorkSpace/ENV/tmp/matplotlib XDG_CACHE_HOME=/data3/yangfeiyang/WorkSpace/ENV/tmp/fontcache \
/data3/yangfeiyang/WorkSpace/ENV/musclemimic/.venv/bin/python3 -m pytest environment/overall_environment/tests/test_frozen_body_policy.py -q
```

Expected: PASS，且输出中不能有 action diff 超阈值。

---

## PR-5 / Task 5: Reward Config Cleanup + Racket-Hand Reference Transform

**目标:** 让 spec reward 真正传入 env；修正 `r_racket_hand_pose`，避免鼓励 palm/grip 距离越小越好。

**Files:**
- Modify: `BadmintonMimic/experiments/posttrain/forehand_clear_grip_hold_v1.yaml`
- Modify: `BadmintonMimic/scripts/run_forehand_clear_grip_hold.py`
- Modify: `environment/overall_environment/src/overall_grip_hold_env.py`
- Test: `environment/overall_environment/tests/test_overall_grip_hold_env.py`
- Test: `tests/unit/test_forehand_clear_grip_hold_runner.py`

- [ ] **Step 1: 修改 YAML key**

把：

```yaml
reward:
  mimic: 1.0
  root_stability: 1.0
```

改为：

```yaml
reward:
  mimic_body: 0.2
  grip_site: 8.0
  contact: 2.0
  no_slip: 8.0
  no_penetration: 10.0
  racket_hand_pose: 4.0
  residual_effort: 0.01
```

- [ ] **Step 2: runner 解析 reward**

扩展 `GripHoldPaths` 或新增 `GripHoldSpec`。推荐从 `GripHoldPaths` 升级为：

```python
@dataclass(frozen=True)
class GripHoldSpec:
    paths: GripHoldPaths
    reward_weights: dict[str, float]
```

如果改动过大，先在 `load_grip_hold_spec()` 返回的 dataclass 增加：

```python
reward_weights: dict[str, float]
```

创建 env 时传入：

```python
reward_weights=paths.reward_weights,
```

- [ ] **Step 3: 修正 racket-hand pose reward**

在 reset 时记录 palm-local reference：

```python
self._reference_palm_to_grip_local = self._palm_local_grip_vector()
```

新增：

```python
def _palm_local_grip_vector(self) -> np.ndarray:
    palm_xmat = np.asarray(self.data.site_xmat[self.palm_site_id], dtype=float).reshape(3, 3)
    world_vec = self.data.site_xpos[self.grip_site_id] - self.data.site_xpos[self.palm_site_id]
    return palm_xmat.T @ world_vec
```

reward 改为：

```python
"r_racket_hand_pose": -self.reward_weights["racket_hand_pose"] * float(info["racket_hand_pose_error_m"])
```

`_info()` 增加：

```python
"racket_hand_pose_error_m": float(np.linalg.norm(
    self._palm_local_grip_vector() - self._reference_palm_to_grip_local
)),
```

- [ ] **Step 4: 测试**

新增：

```python
def test_racket_hand_pose_reward_uses_reference_transform_not_zero_distance():
    env = OverallGripHoldEnv(default_training_scene_path(), residual_groups=["right_hand_fingers"])
    env.reset()
    _, _, _, _, info = env.step(np.zeros(env.action_size, dtype=float))

    assert "racket_hand_pose_error_m" in info
    assert abs(info["reward_terms"]["r_racket_hand_pose"]) < 1.0
```

新增：

```python
def test_spec_reward_weights_are_passed_to_env(tmp_path: Path):
    paths = load_grip_hold_spec(SPEC)
    report = train_tiny(paths, out_dir=tmp_path, total_steps=8, rollout_steps=4)

    assert report["effective_reward_weights"]["contact"] == 2.0
    assert "mimic" not in report["effective_reward_weights"]
```

- [ ] **Step 5: 验证**

Run:

```bash
env MPLCONFIGDIR=/data3/yangfeiyang/WorkSpace/ENV/tmp/matplotlib XDG_CACHE_HOME=/data3/yangfeiyang/WorkSpace/ENV/tmp/fontcache \
/data3/yangfeiyang/WorkSpace/ENV/musclemimic/.venv/bin/python3 -m pytest \
  environment/overall_environment/tests/test_overall_grip_hold_env.py \
  tests/unit/test_forehand_clear_grip_hold_runner.py -q
```

Expected: PASS。

---

## PR-6 / Task 6: Long No-Servo Replay Gate + Metrics

**目标:** 把 10-step smoke 提升为 300/1000-step gate，并记录 action/ctrl saturation、obs/goal 分布。

**Files:**
- Modify: `BadmintonMimic/scripts/run_forehand_clear_grip_hold.py`
- Modify: `tests/unit/test_forehand_clear_grip_hold_runner.py`

- [ ] **Step 1: 增加统计字段**

real replay loop 中累计：

```python
body_obs_max_abs = max(body_obs_max_abs, float(np.max(np.abs(body_obs))))
goal_obs_mean_abs_values.append(float(np.mean(np.abs(goal_obs))))
raw_body_action_mean_abs_values.append(float(np.mean(np.abs(body_action))))
full_ctrl_saturation_count += int(np.count_nonzero(np.isclose(np.abs(clipped_full_action), 1.0, atol=1e-6)))
full_ctrl_count += int(clipped_full_action.size)
```

report：

```python
"body_obs_max_abs": body_obs_max_abs,
"goal_obs_mean_abs": float(np.mean(goal_obs_mean_abs_values)),
"raw_body_action_mean_abs": float(np.mean(raw_body_action_mean_abs_values)),
"full_ctrl_saturation_rate": full_ctrl_saturation_count / max(full_ctrl_count, 1),
```

- [ ] **Step 2: 增加 CLI 验证命令**

手工 gate：

```bash
env MPLCONFIGDIR=/data3/yangfeiyang/WorkSpace/ENV/tmp/matplotlib XDG_CACHE_HOME=/data3/yangfeiyang/WorkSpace/ENV/tmp/fontcache \
/data3/yangfeiyang/WorkSpace/ENV/musclemimic/.venv/bin/python3 BadmintonMimic/scripts/run_forehand_clear_grip_hold.py \
  --stage replay-smoke \
  --policy-source real \
  --steps 300 \
  --out-dir outputs/posttrain/ForehandClearGripHold/v1/real_policy_no_servo_300
```

验收：

```text
policy_replay_ready=true
steps_completed=300
finite=true
pose_servo_enabled=false
goal_obs_source=trajectory_cache
full_ctrl_saturation_rate < 0.1
```

- [ ] **Step 3: 1000-step gate**

300-step 通过后再跑：

```bash
env MPLCONFIGDIR=/data3/yangfeiyang/WorkSpace/ENV/tmp/matplotlib XDG_CACHE_HOME=/data3/yangfeiyang/WorkSpace/ENV/tmp/fontcache \
/data3/yangfeiyang/WorkSpace/ENV/musclemimic/.venv/bin/python3 BadmintonMimic/scripts/run_forehand_clear_grip_hold.py \
  --stage replay-smoke \
  --policy-source real \
  --steps 1000 \
  --out-dir outputs/posttrain/ForehandClearGripHold/v1/real_policy_no_servo_1000
```

如果 1000-step 失败但 300-step 通过，本阶段不能进入长 PPO，只能进入视频诊断。

---

## PR-7 / Task 7: Artifact / Cache / Scene Preparation Script

**目标:** 把训练前依赖检查自动化，避免 artifact/cache/scene/grip seed 版本不一致。

**Files:**
- Create: `BadmintonMimic/scripts/prepare_forehand_clear_grip_hold_artifacts.py`
- Test: `tests/unit/test_forehand_clear_grip_hold_runner.py` 或新增 `tests/unit/test_prepare_forehand_clear_artifacts.py`

- [ ] **Step 1: 写准备脚本**

脚本参数：

```python
--spec
--build-training-scene
--export-frozen-policy
--check-trajectory-cache
--check-grip-seed
--out-dir
```

report 必须包含：

```json
{
  "artifact_source_checkpoint_matches_spec": true,
  "params_npz_exists": true,
  "run_stats_npz_exists": true,
  "training_scene_ok": true,
  "actuation_enabled": true,
  "hand_racket_contact_allowed": true,
  "trajectory_cache_ok": true,
  "grip_seed_ok": true,
  "body_obs_size": 2418,
  "goal_size": 469,
  "body_action_size": 354,
  "overall_action_size": 416
}
```

- [ ] **Step 2: 失败条件**

必须 fail-fast：

```text
checkpoint missing
artifact missing params.npz/run_stats.npz
artifact source checkpoint != spec body_policy.checkpoint
trajectory cache missing
grip seed missing
training scene lacks actuators
Full Body - overall_racket contact exclude exists
```

- [ ] **Step 3: 验证命令**

```bash
env MPLCONFIGDIR=/data3/yangfeiyang/WorkSpace/ENV/tmp/matplotlib XDG_CACHE_HOME=/data3/yangfeiyang/WorkSpace/ENV/tmp/fontcache \
/data3/yangfeiyang/WorkSpace/ENV/musclemimic/.venv/bin/python3 BadmintonMimic/scripts/prepare_forehand_clear_grip_hold_artifacts.py \
  --spec BadmintonMimic/experiments/posttrain/forehand_clear_grip_hold_v1.yaml \
  --check-trajectory-cache \
  --check-grip-seed \
  --out-dir outputs/posttrain/ForehandClearGripHold/v1/prepare_report
```

Expected: JSON report 全部关键字段为 true。

---

## PR-8 / Task 8: Short No-Shuttle PPO Gate

**目标:** 在前 7 个任务通过后，只做 no-shuttle grip residual 短训练，不加 shuttle，不做击球。

**Files:**
- Modify: `BadmintonMimic/scripts/run_forehand_clear_grip_hold.py`
- No required code if `train-tiny` 已足够；主要是 gate 命令和 metrics。

- [ ] **Step 1: 2k steps**

```bash
env JAX_PLATFORM_NAME=cpu MPLCONFIGDIR=/data3/yangfeiyang/WorkSpace/ENV/tmp/matplotlib XDG_CACHE_HOME=/data3/yangfeiyang/WorkSpace/ENV/tmp/fontcache \
/data3/yangfeiyang/WorkSpace/ENV/musclemimic/.venv/bin/python3 BadmintonMimic/scripts/run_forehand_clear_grip_hold.py \
  --stage train-tiny \
  --total-steps 2000 \
  --rollout-steps 128 \
  --out-dir outputs/posttrain/ForehandClearGripHold/v1/no_servo_traj_aligned_2k
```

验收：

```text
finite=true
policy_source=frozen_artifact
body_goal_obs_source=trajectory_cache
pose_servo_enabled=false
racket_drop rate low
body_fall rate low
```

- [ ] **Step 2: 5k / 20k**

2k 通过并检查视频后，再跑 5k、20k。若 contact_count 仍为 0，需要先修 reward/contact/initial grip，不进入更长训练。

---

## 不在本轮实施的内容

以下内容先不做，避免在基础信号未清理前扩大问题面：

- 多 motion schedule 正式训练。可以先写接口，但不要正式长训。
- static shuttle contact、over-net、landing reward。
- ghost racket 可以在 no-servo 300/1000 replay 通过后启动。
- right_wrist/right_forearm residual stage2。当前先稳定 right_hand_fingers。

---

## 总验证命令

每个 PR 完成后至少运行：

```bash
env MPLCONFIGDIR=/data3/yangfeiyang/WorkSpace/ENV/tmp/matplotlib XDG_CACHE_HOME=/data3/yangfeiyang/WorkSpace/ENV/tmp/fontcache \
/data3/yangfeiyang/WorkSpace/ENV/musclemimic/.venv/bin/python3 -m pytest \
  environment/overall_environment/tests/test_body_obs_adapter.py \
  environment/overall_environment/tests/test_frozen_body_policy.py \
  environment/overall_environment/tests/test_overall_grip_hold_env.py \
  environment/overall_environment/tests/test_ghost_racket_teacher.py \
  environment/overall_environment/tests/test_overall_environment.py \
  environment/overall_environment/tests/test_static_forehand_clear_env.py \
  environment/overall_environment/tests/test_training_scene.py \
  tests/unit/test_forehand_clear_grip_hold_runner.py -q
```

当前 baseline 已经通过：

```text
70 passed, 27 warnings in 356.94s
```

---

## 执行顺序

1. PR-1：No-servo replay correctness。
2. PR-2：Trajectory-aligned reset。
3. PR-3：Trajectory-phase body mimic reward。
4. PR-4：Frozen NumPy actor vs Flax actor mean 等价性测试。
5. PR-5：Reward config cleanup + racket-hand reference transform。
6. PR-6：300/1000-step no-servo replay gate + 分布统计。
7. PR-7：Artifact/cache/scene preparation script。
8. PR-8：2k -> 5k -> 20k no-shuttle grip residual short PPO。

只有 PR-1 到 PR-6 通过后，才建议进入 PR-8 的 2k/5k/20k 短 PPO。只有 PR-8 指标和视频都通过后，才考虑 ghost racket；在 ghost racket 通过前，不建议加 shuttle 或做正手高远球击球 RL。

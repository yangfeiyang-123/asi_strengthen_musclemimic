# 01 Student Policy 设计与 Observation Filter

## 1.1 目标

实现一个 student policy，它完成同样的 ForehandClear body trajectory imitation，但 policy 输入不包含 future lookahead。

第一版目标：

```text
student_obs = joint state + muscle state + foot contact + motion phase
student_action = 354-D normalized muscle controls
```

其中 motion phase 是动作进度，不是未来轨迹目标。

---

## 1.2 Teacher 与 Student 对比

| 项目 | Teacher | Student v1 |
|---|---|---|
| 输入 | full observation | filtered observation |
| joint state | 保留 | 保留 |
| muscle state | 保留 | 保留 |
| foot contact | 保留 | 保留 |
| future goal lookahead | 保留 | 删除 |
| motion phase | 保留 | 保留 |
| 输出 | 354-D muscle action | 354-D muscle action |
| reward | MimicReward(reference trajectory) | MimicReward(reference trajectory) |
| 训练方式 | PPO | BC/KD + PPO fine-tune |

---

## 1.3 推荐不要修改 reward

Student 不看 future lookahead，不代表 environment 不再使用 reference trajectory。

训练时应保持：

```text
policy input: no future lookahead
reward input: simulation state + reference trajectory
```

即：

```text
r_t = R_mimic(s_{t+1}, τ_t)
```

这样 student 被迫把 ForehandClear 动作模式内化到网络参数中。

---

## 1.4 Observation Filter Wrapper 方案

推荐实现一个 wrapper，而不是直接修改 `GoalTrajMimic`。

新增文件建议：

```text
musclemimic/distill/obs_filter.py
```

实现：

```python
class StudentObservationFilterWrapper(BaseWrapper):
    """Filter policy observations for student policies.

    Default mode:
      keep all non-goal observations;
      keep only motion phase from the goal group;
      drop future lookahead components.
    """
```

### 输入

Wrapper 接收 base env，它的 raw observation 包含：

```text
[state observations, goal observations]
```

通过：

```python
goal_indices = env.obs_container.get_obs_ind_by_group("goal")
```

识别 goal group。

### 默认 keep indices

```python
raw_obs_dim = env.info.observation_space.shape[0]
goal_indices = env.obs_container.get_obs_ind_by_group("goal")
state_indices = all_indices_except(goal_indices)
phase_index = goal_indices[-1]   # 因为 GoalTrajMimic 在 enable_motion_phase=True 时把 phase append 到 goal 末尾
student_indices = concat(state_indices, [phase_index])
```

注意：如果 `enable_motion_phase=False`，应报错，除非配置显式允许 `keep_phase=False`。

---

## 1.5 Wrapper 输出布局

推荐输出：

```text
student_obs = [state_obs, phase]
```

其中：

```text
state_obs = raw_obs[non_goal_indices]
phase = raw_obs[goal_indices[-1]]
```

Wrapper 需要更新：

```python
self.info.observation_space
self.obs_container
```

最小 obs_container 支持：

```text
state group -> indices of state_obs in student_obs
goal group  -> index of phase in student_obs
```

这样如果 `len_obs_history > 1` 且 `split_goal=True`，现有 `NStepWrapper` 仍可以做到：

```text
[state history, current phase]
```

---

## 1.6 与现有 wrap_env 的集成点

修改：

```text
musclemimic/algorithms/common/env_utils.py
```

在 `wrap_env(env, config)` 中，在 `NStepWrapper` 之前加入：

```python
student_cfg = config.get("student_obs_filter", {})
if student_cfg.get("enabled", False):
    env = StudentObservationFilterWrapper(env, **student_cfg)
```

顺序必须是：

```text
base env
 -> StudentObservationFilterWrapper
 -> NStepWrapper(optional)
 -> VecEnv / LogWrapper / AutoResetWrapper
 -> NormalizeVecReward(optional)
```

原因：如果先做 history stacking，再过滤 goal，index mapping 会复杂很多。

---

## 1.7 配置字段建议

在 experiment config 下加入：

```yaml
student_obs_filter:
  enabled: true
  drop_goal_lookahead: true
  keep_motion_phase: true
  motion_phase_from_goal_last_index: true
  require_goal_group: true
  require_motion_phase: true
```

可选扩展：

```yaml
student_obs_filter:
  keep_goal_indices: []       # advanced explicit override
  keep_obs_groups: null       # optional group-based filtering
  expose_goal_group_as_phase: true
```

---

## 1.8 离线数据集也必须复用同一套过滤逻辑

不要在 teacher rollout collection 中手写另一套 index 规则。

建议 `obs_filter.py` 同时提供纯函数：

```python
def build_student_obs_indices(env, config) -> StudentObsSpec:
    ...

def filter_student_obs(raw_obs, spec):
    ...
```

这样：

```text
训练时 wrapper 使用同一套 StudentObsSpec
数据采集时 collector 使用同一套 StudentObsSpec
测试时也验证二者输出一致
```

---

## 1.9 单元测试

新增：

```text
tests/unit/test_student_obs_filter.py
```

测试点：

1. mock env with state_dim=6, goal_dim=5。
2. keep phase 时输出维度为 `6 + 1`。
3. phase index 等于原始 goal 最后一维。
4. `goal` group 在过滤后只有一个 phase index。
5. 与 `NStepWrapper(split_goal=True, n_steps=3)` 组合后输出为：

```text
3 * state_dim + 1
```

6. 若没有 goal group 且 require_goal_group=True，应报错。
7. 若 goal_dim=0，应报错。

---

## 1.10 不建议第一版使用 privileged critic

可以考虑 actor 不看 lookahead、critic 看 full lookahead 的 asymmetric actor-critic，但第一版不建议。

原因：

```text
1. 现有 checkpoint / inference 更简单；
2. 最终 student checkpoint 应该完全不需要 lookahead；
3. 使用 privileged critic 会让网络输入/obs index 更难解释和部署。
```

第一版建议：

```text
actor 和 critic 都使用 student_obs。
```

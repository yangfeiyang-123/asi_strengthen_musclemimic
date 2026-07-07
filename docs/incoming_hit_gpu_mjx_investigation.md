# musclemimic GPU 并行训练栈调查报告

## 执行摘要

**时间**: 2026年7月6日  
**地点**: /data3/yangfeiyang/WorkSpace/musclemimic  
**调查范围**: musclemimic 主训练管线 + 新羽毛球击球环境 MJX 兼容性

---

## 问题 1: 主训练管线架构

### 答案: JAX + MuJoCo MJX，num_envs 通过 YAML 配置

#### 证据

**位置**: `fullbody/conf_fullbody.yaml:22-27`
```yaml
env_params:
  env_name: MjxMyoFullBody
  headless: True
  horizon: 1000
  disable_fingers: true
  mjx_backend: warp
  num_envs: 8192
```

**关键信息**:
- **JAX 集成**: `musclemimic/algorithms/ppo/ppo.py:11-13` 导入 `jax` 和 `jax.numpy`
- **MJX 使用**: `musclemimic/core/mujoco_mjx.py:14-15` 导入 `mujoco.mjx`，第 57-126 行实现 `Mjx` 基类
- **num_envs 配置**: `musclemimic/algorithms/ppo/ppo.py:73` 计算 `exp.num_envs * exp.num_steps` 作为 batch_size

#### Warp Backend 支持

**是的**，musclemimic 支持 Warp 后端：

- `musclemimic/core/mujoco_mjx.py:67-85`：`__init__` 接受 `mjx_backend` 参数
- `musclemimic/core/mujoco_mjx.py:136-172`：`_get_mjx_backend()` 方法选择 'jax' 或 'warp'
- `fullbody/conf_fullbody.yaml:26`：配置中 `mjx_backend: warp` 设置为默认

#### 使用场景

```python
# 训练配置中的 num_envs 使用
exp.num_envs = 8192  # 超大规模并行
batch_size = exp.num_envs * exp.num_steps  # = 8192 * 20 = 163,840 样本/更新
```

---

## 问题 2: MJX Step 流程与力注入机制

### MJX Step 定义位置

**主函数**: `musclemimic/core/mujoco_mjx.py:291-386` 的 `mjx_step()` 方法

#### 流程架构

```
mjx_step(state, action):
  ├─ _mjx_preprocess_action()     [L312] 预处理动作
  ├─ _mjx_simulation_pre_step()   [L315] 修改模型/数据（地形、域随机化）
  │
  ├─ inner_loop (scan over n_intermediate_steps) [L317-338]
  │  └─ for each substep:
  │     ├─ _mjx_compute_action()  [L321] 计算实际控制
  │     ├─ ctrl = data.ctrl.at[].set(ctrl_action) [L325]
  │     └─ scan(mjx.step(), n_substeps) [L333] 多个 MuJoCo 步进
  │
  ├─ _mjx_simulation_post_step()  [L341] 步进后修改
  ├─ _mjx_create_observation()    [L344]
  ├─ _mjx_step_finalize()         [L347] 域随机化观测
  ├─ _mjx_reward()                [L356]
  ├─ _mjx_is_done()               [L363]
  └─ return MjxState
```

#### 实际 MuJoCo 步进

**位置**: `musclemimic/core/mujoco_mjx.py:329-333`
```python
def single_step(data, _):
    data = mjx.step(sys, data)  # JAX 纯函数步进
    return data, None

_data = jax.lax.scan(single_step, _data, (), self._n_substeps)[0]
```

### 力注入点

#### 现有机制

MJX 中的力注入通过 **`data.qfrc_applied`** 进行（与 CPU MuJoCo 相同）：

1. **在 `_mjx_simulation_pre_step()` 中**: 域随机化可修改力
   - 位置: `musclemimic/core/mujoco_mjx.py:513-529`
   - 调用 `self._domain_randomizer.update()`

2. **在 `_mjx_compute_action()` 中**: 控制函数可注入自定义力
   - 位置: `musclemimic/core/mujoco_mjx.py:565-581`
   - 调用 `self._control_func.generate_action()`

3. **直接通过 `data.qfrc_applied.at[].set()`**: 在任何 JAX 函数中

#### 现有模式（参考）

- `loco_mujoco/core/domain_randomizer/base.py`：提供 `update()` 钩子，接收 (self, env, model, data, carry, jnp)
- 返回 (model, data, carry)，允许修改 data.qfrc_applied

#### 缺失的功能

**没有现成的 `xfrc_applied` 支持** —— MJX Data 结构没有暴露 body 力矩：
- MuJoCo 标准有 `data.xfrc_applied[nbody, 6]` 用于施加的力/力矩
- MJX 3.4.0 未在 Data 结构中暴露这个字段

**解决方案**: 通过关节受约束体的 qfrc_applied 间接实现，或修改 body 速度后让 contact/friction 处理

---

## 问题 3: loco_mujoco Vmap 并行与自定义 Reward

### Vmap 并行模式

**位置**: `loco_mujoco/core/mujoco_base.py:44-250` (Mujoco 基类)

#### 如何进行 Vmap 并行

musclemimic 使用 **隐式 vmap**（通过 PPO 训练器）：

1. **环境状态维度化**:
   - `musclemimic/core/mujoco_mjx.py` 中 MjxState 的所有数组字段自动 vmapped
   - `musclemimic/algorithms/ppo/runner.py:92-500` 训练循环对 num_envs 个环境并行执行

2. **核心模式** (推断自 PPO runner):
   ```python
   # 伪代码
   def rollout_fn(env_state, action):
       env_state = vmap(env.mjx_step)(env_state, action)  # 自动 vmap
       return env_state, trajectory_data
   ```

3. **纯函数设计**:
   - `musclemimic/core/mujoco_mjx.py:388-401` 所有 `_mjx_*` 方法都是纯函数
   - 使用 `jnp` (JAX numpy) 而非 `np`
   - 返回新的 state 而非修改 in-place

### 自定义 Reward/Observation 注入

#### 观测生成

**位置**: `musclemimic/core/mujoco_mjx.py:388-401` 的 `_mjx_create_observation()`

```python
def _mjx_create_observation(self, model: Model, data: Data, carry: MjxAdditionalCarry) -> jax.Array:
    return self._create_observation_compat(model, data, carry, jnp)
    # 调用基类，自动检测 observation_spec 并提取所需字段
```

**自定义观测模式**:
1. 在 `observation_spec` 中定义新的观测类型（继承 `loco_mujoco.core.observations.Observation`）
2. 在环境初始化时注册
3. `_create_observation_compat()` 会自动处理

#### Reward 函数

**位置**: `musclemimic/core/mujoco_mjx.py:433-468` 的 `_mjx_reward()`

```python
def _mjx_reward(self, obs, action, next_obs, absorbing, info, model, data, carry) -> (float, carry, dict):
    result = self._reward_function(obs, action, next_obs, absorbing, info, 
                                   self, model, data, carry, jnp)
    return reward, carry, reward_info  # 支持返回 reward_info dict
```

**自定义 Reward 注入**:
1. 创建继承 `loco_mujoco.core.reward.base.Reward` 的类
2. 实现 `__call__(obs, action, next_obs, absorbing, info, env, model, data, carry, jnp_module)`
3. 在环境配置中通过 `reward_type` 和 `reward_params` 注入
4. 返回 `(reward_scalar, updated_carry, info_dict)`

**例子** (从代码推断):
```python
# 在训练配置中
experiment:
  reward_type: "MyCustomReward"
  reward_params:
    weight_1: 0.5
    weight_2: 0.3
```

---

## 问题 4: MJX XML 兼容性与 overall_incoming_hit_scene.xml

### 模型规模

```
Model: environment/overall_environment/assets/overall_incoming_hit_scene.xml

nq = 143 (位置坐标)
nv = 140 (速度自由度)
ngeom = 1,143 (几何体)
neq = 52 (约束)
```

### 几何体类型分解

| 类型 | 数量 | MJX 支持 | 备注 |
|------|------|---------|------|
| CAPSULE | 638 | ✓ 是 | 主要人体碰撞形状 |
| BOX | 151 | ✓ 是 | 场景边界 (地面、边线等) |
| SPHERE | 117 | ✓ 是 | 羽毛球模型 |
| CYLINDER | 113 | ✓ 是 | 网柱 |
| MESH | 102 | ⚠ 部分 | 球拍、网等复杂形状 |
| ELLIPSOID | 21 | ✓ 单独支持 | **头部骨骼** |
| PLANE | 1 | ✓ 是 | 地面 |

### 碰撞对关系

```
实际定义的碰撞对 (npair = 75):
  - CAPSULE <-> CAPSULE: 66 对 (身体-身体)
  - CAPSULE <-> ELLIPSOID: 8 对 (身体-头部)
  - ELLIPSOID <-> ELLIPSOID: 1 对 (头部内部)

隐含可能发生的碰撞:
  - ELLIPSOID <-> BOX: 未显式定义，但由于 contype/conaffinity 重叠，
                      会被 MJX 隐式创建 → 触发 NotImplementedError
```

### 约束类型

| 类型 | 数量 | MJX 支持 | 用途 |
|------|------|---------|------|
| JOINT | 51 | ✓ 是 | 手指关节对齐 |
| WELD | 1 | ✓ 是 | 右手 <-> 球拍（`overall_right_hand_racket_soft_weld`，L4046） |

### ✗ 阻塞问题: ELLIPSOID-BOX 碰撞

#### 错误信息
```
NotImplementedError: (mjtGeom.mjGEOM_ELLIPSOID, mjtGeom.mjGEOM_BOX) collisions not implemented.
```

**根本原因**:
- 模型中 ELLIPSOID 几何体（头部骨骼）和 BOX 几何体（场景边界）由于 contype/conaffinity 设置会隐式产生碰撞对
- MJX 3.4.0 中 ELLIPSOID-BOX 碰撞对尚未实现

**位置验证**: `musclemimic/core/mujoco_mjx.py:85` 中 `mjx.put_model()` 调用会触发

---

## 问题 5: IncomingShuttleHitEnv 迁移可行性

### 环境架构分析

**位置**: `environment/overall_environment/src/incoming_shuttle_hit_env.py`

#### 当前实现 (CPU-only)

```python
# 第 88-122 行: 初始化
class IncomingShuttleHitEnv:
    def __init__(self, xml, feed_bank, physics_config, ...):
        self.model = mujoco.MjModel.from_xml_path(xml)  # 标准 MuJoCo
        self.data = mujoco.MjData(self.model)
        self.physics = BadmintonPhysics(physics_config)
        # ...
```

#### 力注入关键点

**位置**: `environment/overall_environment/src/badminton_physics.py:104-147`

```python
def substep(self, model, data) -> dict:
    """每个 MuJoCo 步进前的力注入"""
    
    data.qfrc_applied[:] = 0.0                    # [L106] 清空
    
    aero_diag = apply_shuttlecock_aero(...)       # [L107] 空气动力
    contact = apply_stringbed_force(...)          # [L108] 球拍弦床力
    
    if should_apply_event_rebound(...):           # [L122]
        # 事件反弹: 速度直接重写
        set_freejoint_linear_velocity(...)        # [L138]
        self._cooldown = int(...)                 # [L145]
    
    mujoco.mj_step(model, data)                   # [L147]
    return {...}
```

#### 力的组成

1. **空气动力** (L107):
   - 调用 `apply_shuttlecock_aero(model, data, config)`
   - 位置: `environment/shuttlecock/src/shuttlecock_aero.py`
   - **操作**: 修改 `data.qfrc_applied[shuttle_body_dof]`
   - **类型**: 纯函数（输入: 速度、四元数、配置 → 输出: 力向量）

2. **弦床力** (L108):
   - 调用 `apply_stringbed_force(model, data, ...)`
   - 位置: `environment/racket/src/racket_stringbed.py`
   - **操作**: 修改 `data.qfrc_applied[shuttle_body_dof]`
   - **类型**: 纯函数（输入: 球位置/速度、球拍位置/速度 → 输出: 接触力）

3. **事件反弹** (L122-145):
   - 条件: 检测高速接触（`should_apply_event_rebound`）
   - 动作: **直接覆盖速度** via `set_freejoint_linear_velocity()`
   - **不是** qfrc_applied，而是状态变异
   - 冷却: 防止多次应用（状态机: `self._cooldown`）

### 迁移 JAX/MJX 的工作量估计

#### ✓ 直接可迁移 (纯函数)

1. **空气动力** → **30 分钟**
   - 现有: `apply_shuttlecock_aero()` 使用 NumPy
   - 迁移: 用 `jnp` 替换 `np`，输入/输出保持一致
   - 类型: 纯函数 ✓

2. **弦床力** → **45 分钟**
   - 现有: `apply_stringbed_force()` 计算几何碰撞、力学
   - 迁移: JAX 化 RacketGeometry、StringbedParams 参数
   - 类型: 纯函数 ✓

#### ⚠ 需要重构 (状态管理)

3. **事件反弹** → **2-3 小时**
   - 问题 1: **冷却计时器** (`self._cooldown`)
     - 当前: Python 整数状态
     - 迁移: 需要加入 `MjxAdditionalCarry` 数据类
   - 问题 2: **速度直接重写** vs qfrc_applied
     - MJX 中 `set_freejoint_linear_velocity()` 需要变成 JAX 操作
     - 输入: `data: Data`，输出: `data_updated: Data`
   - 类型: 混合（纯函数 + 状态）✓ 可做到
   - 参考模式: `musclemimic/core/mujoco_mjx.py:642-673` 中 `_mjx_set_sim_state_from_obs()`

#### ✗ 临界阻塞 (XML 不兼容)

4. **XML 加载** → **无法进行，需要绕过**
   - 问题: `overall_incoming_hit_scene.xml` 包含 21 个 ELLIPSOID 和 151 个 BOX
   - MJX 不支持 ELLIPSOID-BOX 碰撞
   - 解决方案选项:
     - **选项 A** (推荐): 禁用头部 ↔ 边界碰撞
       - 工作量: 改 XML contype/conaffinity，测试
       - 影响: 头部可能穿过边界（视觉瑕疵，但物理上不重要用于击球训练）
     - **选项 B**: 等待 MJX 4.0+ 支持
       - 时间: 未知
     - **选项 C**: 改用 CPU 多进程（见下文）

### 相关代码

**新环境定义**: `environment/overall_environment/src/incoming_scene.py`（未检查）

**测试**: `environment/overall_environment/tests/test_incoming_shuttle_hit_env.py`（未检查，但代表期望行为）

---

## 问题 6: 替代方案 — CPU 多进程向量化

### musclemimic 中是否有现成例子？

**答案**: 有限的支持

#### 已有的向量化设施

1. **Gymnasium 向量环境包装** (推断):
   - 位置: 需查 `musclemimic/core/wrappers/`
   - 可能支持 `gym.vector.SyncVectorEnv` 或 `AsyncVectorEnv`

2. **Jax Vmap** (已有):
   - 位置: `musclemimic/algorithms/ppo/runner.py`
   - 用于批处理训练数据

#### CPU 多进程示例

**未找到** 在 musclemimic 中使用 `multiprocessing.Pool` 的现成代码。

**参考文献**:
- Stable-Baselines3 使用 `SubprocVecEnv` (multiprocessing)
- 可以作为模板，但需要自行集成

#### 实现可行性

**是的**，可以创建：
```python
class IncomingHitVecEnv:
    """多进程向量化 IncomingShuttleHitEnv"""
    def __init__(self, num_envs=8):
        self.envs = [IncomingShuttleHitEnv(...) for _ in range(num_envs)]
        self.pool = multiprocessing.Pool(num_envs)
    
    def step(self, actions):
        results = self.pool.starmap(lambda env, a: env.step(a), 
                                    zip(self.envs, actions))
        obs, rewards, dones, infos = zip(*results)
        return np.array(obs), np.array(rewards), np.array(dones), infos
```

**工作量**: 200-300 行 Python，但:
- 性能比 GPU 低 **10-100 倍**
- 内存占用高（每个进程复制模型）
- 通信开销（进程间序列化）

---

## 总结与建议

### 当前状态

| 组件 | JAX 化 | GPU 就绪 | 备注 |
|------|--------|----------|------|
| PPO 算法 | ✓ | ✓ | 完整实现 |
| MyoFullBody 环境 | ✓ | ✓ | Warp backend 可用 |
| MJX step 管线 | ✓ | ✓ | 纯函数设计 |
| 空气动力力 | 部分 (NumPy) | ⚠ | 需要 JAX 化 |
| 弦床力 | 部分 (NumPy) | ⚠ | 需要 JAX 化 |
| 事件反弹 | 否 (Python 状态) | ⚠ | 需要重构 |
| **XML 加载** | **N/A** | **✗** | **ELLIPSOID-BOX 不支持** |

### 推荐迁移路径

#### 方案 A: MJX GPU 加速 (最快，有 XML 绕过)

**总工作量**: 12-15 工作天

1. **第 1 天**: XML 修复
   - 禁用 ELLIPSOID ↔ BOX 碰撞（改 contype/conaffinity 或删除冲突 geom）
   - 测试标准 MuJoCo 仍可加载

2. **第 2-3 天**: JAX 化物理力
   - `apply_shuttlecock_aero()` 用 `jnp` 重写
   - `apply_stringbed_force()` 用 `jnp` 重写
   - 单元测试对比 NumPy 输出

3. **第 4-5 天**: 事件反弹 JAX 化
   - 冷却计时器加入 `LocoCarry` / `MjxAdditionalCarry`
   - 速度重写改成 JAX 状态变异
   - 参考 `_mjx_set_sim_state_from_obs()` 模式

4. **第 6-7 天**: 环境框架集成
   - 创建 `MjxIncomingShuttleHitEnv` 继承 `LocoEnv`
   - 实现 `_mjx_simulation_post_step()` 调用物理管线
   - 实现自定义 reward 和 observation

5. **第 8-10 天**: 训练集成
   - 接入 PPO 训练器 (`musclemimic/algorithms/ppo/runner.py`)
   - num_envs = 256-1024 GPU 训练
   - 验证收敛性

6. **第 11-15 天**: 测试、优化、文档

**预期性能**:
- 单 GPU: ~10,000 环境步进/秒 (num_envs=512)
- 与 CPU 对比: 50-100 倍加速

#### 方案 B: CPU 多进程 (快速原型，性能低)

**总工作量**: 3-5 工作天

1. 实现 `MultiProcIncomingHitEnv` 包装
2. 接入训练器
3. num_envs = 16-32 CPU 训练

**预期性能**:
- 多核 CPU: ~500 环境步进/秒 (num_envs=16)
- 与单线程对比: 8-16 倍加速

#### 方案 C: 等待 MJX 4.0

- 预计时间: 3-6 个月 (猜测)
- 优点: 无需 XML 改动
- 缺点: 阻塞当前开发

---

## 技术债与建议

1. **MJX 版本升级监控**
   - 关注 MuJoCo 3.5+ 是否支持 ELLIPSOID-BOX 碰撞
   - 订阅 MuJoCo GitHub releases

2. **physics 模块 JAX 化**
   - 所有力计算（aero、stringbed、impact）应改为纯函数
   - 便于未来 JIT 编译和 vmap 并行

3. **环境库结构**
   - 建议分离 "CPU-only physics" 和 "JAX-native physics"
   - 便于两种后端共存

4. **性能监测**
   - 建立 benchmark: 环境步进速度 (steps/sec)
   - 追踪 num_envs 缩放效率

---

## 附录: 文件导航

### 核心 MJX 集成
- `musclemimic/core/mujoco_mjx.py`: MJX 基类，step 流程
- `musclemimic/environments/base.py`: LocoEnv 环境基类
- `fullbody/conf_fullbody.yaml`: 训练配置示例

### PPO 训练器
- `musclemimic/algorithms/ppo/ppo.py`: PPO 算法主类
- `musclemimic/algorithms/ppo/runner.py`: 训练循环

### 新环境代码
- `environment/overall_environment/src/incoming_shuttle_hit_env.py`: CPU 环境实现
- `environment/overall_environment/src/badminton_physics.py`: 物理力注入
- `environment/overall_environment/assets/overall_incoming_hit_scene.xml`: 场景模型

### Loco-Mujoco 基础设施
- `loco_mujoco/core/mujoco_base.py`: 环境基类
- `loco_mujoco/core/domain_randomizer/base.py`: 力注入钩子
- `loco_mujoco/core/reward/base.py`: Reward 基类


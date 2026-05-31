# 1. 对现有实现的检查与建议修正

> 适用场景：把已经训练好的“无球拍、无击球”的肌骨正手高远球策略，扩展到“右手握拍 + 球拍动力学 + 羽毛球击打”。
>
> 本文是给 Codex / 开发代理使用的代码审查说明。它不是论文式总结，而是面向下一步改代码、补测试、跑实验的工程任务规格。

---

## 0. 审查结论

当前仓库已经包含了很多与“握拍击球”有关的初始实现，包括：

- `BadmintonMimic/scripts/run_forehand_clear_grip_hold.py`
- `BadmintonMimic/experiments/posttrain/forehand_clear_grip_hold_v1.yaml`
- `BadmintonMimic/experiments/posttrain/forehand_clear_static_hit_v1.yaml`
- `environment/overall_environment/src/overall_env.py`
- `environment/overall_environment/src/static_forehand_clear_env.py`
- `environment/overall_environment/src/layered_control.py`
- `environment/overall_environment/src/impact_target.py`
- `src/grip/right_hand_racket_grip_env.py`
- `src/grip/train_right_hand_racket_grip_policy.py`

但从静态代码看，这些模块还没有形成一个完整可训练的 post-train 系统。当前更准确的状态是：

```text
已有：
  1. 无球拍正手高远球 full-body policy checkpoint
  2. 右手握拍局部环境
  3. overall badminton scene / inspection scene
  4. static-hit staging wrapper
  5. layered actuator router utility
  6. impact target pseudo-label utility

缺失或未打通：
  1. frozen body policy 到 overall racket scene 的 action adapter
  2. body action + right-hand residual action 的训练集成
  3. static-hit env 的真实 reward / termination / observation
  4. checkpoint action manifest / obs normalization manifest
  5. dedicated grip-hold train stage
  6. body / grip / hit 三阶段课程学习 runner
```

所以当前最重要的判断是：**不要直接 resume 原始 checkpoint 端到端训练拿拍击球**。应该先补齐 runner、action mapping、握拍 residual、static-hit reward 和 integration tests。

---

## 1. `run_forehand_clear_grip_hold.py` 当前不是训练 runner

### 1.1 现象

文件：

```text
BadmintonMimic/scripts/run_forehand_clear_grip_hold.py
```

脚本 docstring 写的是：

```python
"""Run ForehandClear grip-hold diagnostics and training stages."""
```

但 argparse 里实际只有三个 stage：

```python
parser.add_argument(
    "--stage",
    choices=("preflight", "reset-video", "replay-precheck"),
    default="preflight",
)
```

也就是说，当前没有：

```text
train
stage1
stage2
eval
render
all
```

更关键的是，`replay_precheck()` 里直接把：

```python
"policy_replay_ready": False
```

写死，并且 blocked reason 说明：

```text
Frozen policy replay still needs an action adapter from the checkpoint's
 disable_fingers=True MjxMyoFullBody action space into the Overall racket scene
 plus a right-hand residual action merge.
```

这说明脚本自己已经记录了核心阻塞项：**旧策略不能直接 replay 到带球拍的 overall scene，必须先做 action adapter 和 right-hand residual merge。**

参考：

- https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/BadmintonMimic/scripts/run_forehand_clear_grip_hold.py#L850-L858
- https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/BadmintonMimic/scripts/run_forehand_clear_grip_hold.py#L879-L882

### 1.2 风险

如果用户或自动化脚本看到 `grip-hold diagnostics and training stages`，可能误以为已经可以训练：

```bash
python BadmintonMimic/scripts/run_forehand_clear_grip_hold.py --stage train
```

但实际会失败，因为 `train` 不是合法 stage。

更危险的是，用户可能改走 `run_posttrain_experiment.py --stage train --execute`，让普通 fullbody runner 去跑 grip-hold spec，导致：

```text
YAML 描述的是 frozen body + residual right-hand policy，
但实际 runner 仍然是普通 MjxMyoFullBody posttrain command。
```

这会造成“看似跑起来，实则训练目标不对”的实验污染。

### 1.3 建议修正

有两种可选修法。

#### 修法 A：把它明确降级成诊断脚本

如果短期内不准备实现训练逻辑，应当把文件命名、docstring 和 README 都改清楚：

```python
"""Run ForehandClear grip-hold diagnostics only.

This runner currently supports:
- preflight
- reset-video
- replay-precheck

It does not execute policy training. Training requires a future action adapter
from the base disable_fingers checkpoint to the full racket scene plus a
right-hand residual action merge.
"""
```

并把 argparse 保持为：

```python
choices=("preflight", "reset-video", "replay-precheck")
```

这样能避免误导。

#### 修法 B：真正实现 train stage

如果要进入训练，建议增加如下 stage：

```python
choices=(
    "preflight",
    "reset-video",
    "replay-precheck",
    "train-grip-stage1",
    "train-grip-stage2",
    "eval-grip",
    "render-grip",
)
```

训练阶段必须做这些事情：

```text
1. 加载 base full-body checkpoint。
2. 加载 checkpoint action manifest / obs manifest。
3. 构建 overall racket scene 的完整 actuator name list。
4. 用 actuator name 做 body action 到 full action 的映射。
5. 加载或初始化 right-hand grip residual policy。
6. 用 LayeredActuatorRouter 合并：
      full_action = merge(body_action, grip_action)
7. 对 body policy freeze，只训练 grip residual 或少量 wrist/forearm residual。
8. rollout 时保存：
      grip slip、contact count、racket drift、fall、NaN、reward terms。
```

建议新增入口函数结构：

```python
def train_grip_hold(paths: GripHoldPaths, stage: str, out_dir: Path) -> dict[str, Any]:
    spec = load_raw_spec(paths.spec_path)
    base_policy = load_body_policy(spec["body_policy"]["checkpoint"])
    manifest = load_action_manifest(spec["body_policy"]["checkpoint"])
    env = make_overall_training_env(spec["scene"]["xml"])
    router = build_layered_router_from_model_and_manifest(env.model, manifest, spec)
    grip_policy = make_or_load_grip_policy(spec)
    trainer = GripHoldResidualTrainer(env, base_policy, grip_policy, router, spec)
    return trainer.train(stage=stage, out_dir=out_dir)
```

### 1.4 验收标准

完成后至少应满足：

```bash
python BadmintonMimic/scripts/run_forehand_clear_grip_hold.py \
  --spec BadmintonMimic/experiments/posttrain/forehand_clear_grip_hold_v1.yaml \
  --stage replay-precheck
```

输出：

```json
{
  "policy_replay_ready": true,
  "action_adapter_ready": true,
  "router_ready": true,
  "obs_adapter_ready": true
}
```

并且：

```bash
python BadmintonMimic/scripts/run_forehand_clear_grip_hold.py \
  --spec ... \
  --stage train-grip-stage1
```

可以执行至少一个短 rollout，不出现：

```text
NaN
shape mismatch
action size mismatch
missing actuator name
racket immediately drops due to disabled contact / missing weld
```

---

## 2. `forehand_clear_grip_hold_v1.yaml` 描述了训练意图，但 runner 没有实现

### 2.1 现象

文件：

```text
BadmintonMimic/experiments/posttrain/forehand_clear_grip_hold_v1.yaml
```

里面已经写了完整的训练意图：

```yaml
runner_type: forehand_clear_grip_hold
body_policy:
  checkpoint: /data3/yangfeiyang/WorkSpace/musclemimic/checkpoints/de63059b16c0/checkpoint_7812
  trainable: false
residual_policy:
  trainable: true
  actuator_groups:
    stage1: [right_hand_fingers]
    stage2: [right_hand_fingers, right_wrist, right_forearm]
reward:
  mimic: 1.0
  root_stability: 1.0
  grip_site: 8.0
  contact: 2.0
  no_slip: 8.0
  no_penetration: 10.0
  racket_hand_pose: 4.0
  residual_effort: 0.01
```

参考：

- https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/BadmintonMimic/experiments/posttrain/forehand_clear_grip_hold_v1.yaml#L373-L469

这些配置本身是合理的：它们表达了“冻结 body policy、只训练右手 residual”的方向。但目前没有 runner 真正消费这些字段。

### 2.2 风险

YAML 中的字段会给人一种“训练 pipeline 已经实现”的错觉，但实际只是 spec。尤其是这些字段：

```yaml
training:
  total_steps: 50000
  rollout_steps: 512
reward:
  grip_site: 8.0
  no_slip: 8.0
validation:
  min_contacts_final: 4
```

如果没有 env reward 和 trainer 消费它们，它们只是文档，不会影响训练。

### 2.3 建议修正

为每个字段建立“被谁消费”的检查表：

| YAML 字段 | 应由哪个模块消费 | 当前状态 | 建议 |
|---|---|---:|---|
| `runner_type` | `run_posttrain_experiment.py` / dedicated runner | 部分消费 | 继续保留 |
| `body_policy.checkpoint` | dedicated runner | 未训练消费 | 加载 body checkpoint |
| `body_policy.trainable` | trainer | 未消费 | 明确 freeze 参数 |
| `residual_policy.trainable` | trainer | 未消费 | 控制优化器参数 |
| `residual_policy.actuator_groups` | router builder | 未消费 | 映射 actuator group 到 actuator names |
| `reward.*` | env reward function | 未消费或未集成 | 实现 reward term registry |
| `validation.*` | eval / acceptance test | 未消费或未集成 | 生成 validation report |
| `shuttle.enabled` | env factory | 未消费或弱消费 | 选择 no-shuttle / static-hit env |

建议添加一个 spec validator：

```python
def validate_grip_hold_spec(spec: dict[str, Any]) -> None:
    assert spec["runner_type"] == "forehand_clear_grip_hold"
    assert spec["body_policy"]["trainable"] is False
    assert spec["residual_policy"]["trainable"] is True
    assert "right_hand_fingers" in spec["residual_policy"]["actuator_groups"]["stage1"]
    assert spec["shuttle"]["enabled"] is False
```

再添加一个 consumption report：

```json
{
  "spec_fields_consumed": {
    "body_policy.checkpoint": true,
    "body_policy.trainable": true,
    "residual_policy.actuator_groups.stage1": true,
    "reward.no_slip": true,
    "validation.max_grip_slip_m": true
  }
}
```

---

## 3. `run_posttrain_experiment.py` 对 grip-hold 的 guard 不完整

### 3.1 现象

`run_posttrain_experiment.py` 已经知道有两类 dedicated runner：

```python
def requires_dedicated_static_hit_runner(spec: dict[str, Any]) -> bool:
    ...

def requires_dedicated_grip_hold_runner(spec: dict[str, Any]) -> bool:
    return spec.get("runner_type") == FOREHAND_CLEAR_GRIP_HOLD_RUNNER
```

prepare 阶段也会对 grip-hold 写专用 README，并删除普通 fullbody command 文件。

但是 `run_stage()` 里只拦截了 static-hit：

```python
if stage != "prepare" and requires_dedicated_static_hit_runner(spec):
    raise ValueError(...)
```

没有同样拦截 grip-hold。

参考：

- https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/BadmintonMimic/scripts/run_posttrain_experiment.py#L1609-L1613
- https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/BadmintonMimic/scripts/run_posttrain_experiment.py#L2007-L2022
- https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/BadmintonMimic/scripts/run_posttrain_experiment.py#L2248-L2257

### 3.2 风险

这会导致：

```bash
python BadmintonMimic/scripts/run_posttrain_experiment.py \
  --spec BadmintonMimic/experiments/posttrain/forehand_clear_grip_hold_v1.yaml \
  --stage train \
  --execute
```

有机会落入普通 fullbody train command，而不是 dedicated grip-hold runner。

### 3.3 建议补丁

在 `run_stage()` 中增加 grip-hold guard：

```python
def run_stage(spec: dict[str, Any], *, stage: str, arm: str | None, execute: bool) -> int:
    if stage != "prepare" and requires_dedicated_static_hit_runner(spec):
        raise ValueError(
            f"{spec['action']} {spec['experiment_id']} requires a dedicated static-hit runner; "
            f"the PostTrain fullbody runner cannot run stage '{stage}'."
        )

    if stage != "prepare" and requires_dedicated_grip_hold_runner(spec):
        raise ValueError(
            f"{spec['action']} {spec['experiment_id']} requires a dedicated grip-hold runner; "
            f"the PostTrain fullbody runner cannot run stage '{stage}'. "
            "Use BadmintonMimic/scripts/run_forehand_clear_grip_hold.py, "
            "and implement its train stage before launching training."
        )

    result = prepare_experiment(spec)
    ...
```

### 3.4 测试

新增测试：

```python
def test_grip_hold_train_stage_rejected_by_posttrain_runner():
    spec = load_spec("BadmintonMimic/experiments/posttrain/forehand_clear_grip_hold_v1.yaml")
    with pytest.raises(ValueError, match="dedicated grip-hold runner"):
        run_stage(spec, stage="train", arm=None, execute=False)
```

如果后续真的实现了 dedicated train，可以把 guard 改为 dispatch：

```python
if requires_dedicated_grip_hold_runner(spec):
    return run_grip_hold_stage(spec, stage=stage, arm=arm, execute=execute)
```

---

## 4. `disable_fingers=True` 与握拍任务冲突

### 4.1 现象

基础羽毛球 fullbody config 里：

```yaml
experiment:
  env_params:
    env_name: MjxMyoFullBody
    num_envs: 256
    disable_fingers: true
```

参考：

- https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/BadmintonMimic/experiments/fullbody/config_specific_task/conf_fullbody_badminton_gmr.yaml#L381-L390

这意味着你已经训练好的正手高远球策略大概率没有手指控制 action。握拍任务却需要：

```text
thumb / index / middle / ring / pinky
palm-handle contact
handle slip control
finger muscle activation
```

### 4.2 风险

如果直接把旧 checkpoint 放进 full model：

```text
旧 checkpoint action size != full racket scene action size
旧 checkpoint observation normalization != full racket scene observation
旧 checkpoint 不知道 right-hand finger actuators
旧 checkpoint 没有 racket/shuttle state
```

会出现：

```text
action shape mismatch
obs shape mismatch
隐式 index 对错 actuator
手指没有闭合能力
球拍接触失效
```

### 4.3 推荐方案：manifest + adapter + residual

不要按 index 复用 action，必须按 actuator name 映射。

建议在 checkpoint 保存目录里新增：

```text
checkpoint_dir/
  config/
    metadata
    action_manifest.json
    obs_manifest.json
    normalization_manifest.json
```

`action_manifest.json` 示例：

```json
{
  "env_name": "MjxMyoFullBody",
  "disable_fingers": true,
  "action_size": 354,
  "actuator_names": [
    "hip_flexion_r", "hip_extension_r", "..."
  ],
  "excluded_actuator_groups": ["right_hand_fingers", "left_hand_fingers"],
  "control_range": [-1.0, 1.0]
}
```

Full racket scene 也要能生成：

```json
{
  "env_name": "OverallBadmintonTrainingEnv",
  "disable_fingers": false,
  "action_size": 416,
  "actuator_names": [
    "hip_flexion_r", "...", "right_thumb_flexor", "right_index_flexor"
  ]
}
```

adapter 逻辑：

```python
class CheckpointActionAdapter:
    def __init__(self, checkpoint_actuator_names, full_actuator_names):
        self.src_names = list(checkpoint_actuator_names)
        self.dst_names = list(full_actuator_names)
        self.src_index = {name: i for i, name in enumerate(self.src_names)}
        self.dst_index = {name: i for i, name in enumerate(self.dst_names)}
        missing = sorted(set(self.src_names) - set(self.dst_names))
        if missing:
            raise ValueError(f"checkpoint actuators missing in target model: {missing}")

    def adapt(self, body_action):
        full = np.zeros(len(self.dst_names), dtype=np.float32)
        for name, i_src in self.src_index.items():
            full[self.dst_index[name]] = body_action[i_src]
        return full
```

然后 residual 只覆盖右手 actuator：

```python
full_body_action = adapter.adapt(a_body)
full_action = full_body_action.copy()
full_action[right_hand_actuator_indices] = a_grip
```

或者使用已有的 `LayeredActuatorRouter`。

---

## 5. `LayeredActuatorRouter` 是正确方向，但需要被真正集成

### 5.1 现状

文件：

```text
environment/overall_environment/src/layered_control.py
```

`LayeredActuatorRouter` 已经支持：

- all actuator names
- body actuator names
- grip actuator names
- duplicate check
- overlap check
- missing actuator check
- body action / grip action 按 name merge 成 full action

参考：

- https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/environment/overall_environment/src/layered_control.py

这是非常重要的正确设计，因为它避免了 actuator index 对齐错误。

### 5.2 当前缺口

它目前只是 utility，还没有进入：

```text
checkpoint loader
obs adapter
rollout runner
PPO trainer
static-hit env
validation renderer
```

### 5.3 建议集成路径

新增：

```text
environment/overall_environment/src/action_adapter.py
```

内容包含：

```python
@dataclass(frozen=True)
class PolicyActionSpace:
    env_name: str
    disable_fingers: bool
    actuator_names: list[str]
    action_size: int


def load_policy_action_space(checkpoint_dir: Path) -> PolicyActionSpace:
    ...


def make_layered_router(model: mujoco.MjModel, body_manifest: PolicyActionSpace, grip_groups: list[str]) -> LayeredActuatorRouter:
    all_names = actuator_names_from_model(model)
    body_names = body_manifest.actuator_names
    grip_names = actuator_group_to_names(model, grip_groups)
    return LayeredActuatorRouter(all_names, body_names, grip_names)
```

新增测试：

```python
def test_layered_router_rejects_overlap(): ...
def test_layered_router_rejects_missing_body_actuator(): ...
def test_layered_router_merge_shape_and_values(): ...
def test_checkpoint_disable_fingers_adapter_to_full_racket_scene(): ...
```

验收：

```text
给定 disable_fingers=True checkpoint action，router 可以生成 full model action。
右手手指 actuator 由 grip policy 控制。
未被 body/grip 控制的 actuator 显式置零或保持 baseline，而不是随机未初始化。
```

---

## 6. Overall scene 当前更适合 inspection，不适合直接训练

### 6.1 现象

`environment/overall_environment` 的 README 明确说明：

- viewer 默认是静态 inspection。
- 如果加 `--simulate` 跑 raw physics，没有训练策略或 hand-racket constraint 时，人可能倒，球拍会离开手。
- racket visually initialized at the hand，但 person-racket contacts 被过滤，以避免初始 contact explosion。
- racket 是 free body，没有 welded 到 hand。

参考：

- https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/tree/main/environment/overall_environment#L313-L322
- https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/tree/main/environment/overall_environment#L353-L357

### 6.2 风险

如果直接拿这个 XML 跑训练：

```text
球拍可能一开始就掉；
如果启用手-球拍 contact，可能接触爆炸；
如果没有 soft constraint，探索极其稀疏；
如果没有握拍策略，body policy 不知道如何保持球拍；
如果没有 observation/reward，RL 无法学击球。
```

### 6.3 建议拆分 XML / env

建议把 overall 环境拆成：

```text
overall_badminton_inspection_scene.xml
  用于可视化和 reset 检查；可以过滤 contact；可以无训练 reward。

overall_badminton_training_scene.xml
  用于 RL；启用 right-hand actuators、handle contacts、可选 soft weld、shuttle contact、sensor sites。
```

训练 XML 必须具备：

```text
1. right hand / finger actuators enabled
2. racket freejoint enabled
3. handle collision geoms enabled
4. palm / thumb / finger contact geoms enabled
5. optional soft weld palm-to-handle
6. stringbed contact geoms / sites
7. shuttle freejoint and collision geoms
8. racket frame sensors / shuttle sensors
```

训练 env 必须返回：

```python
obs = concat([
    body_proprioception,
    phase,
    racket_pose_root_frame,
    racket_velocity_root_frame,
    handle_pose_relative_to_palm,
    grip_contact_flags,
    grip_slip_metrics,
    shuttle_pose_root_frame,
    shuttle_velocity_root_frame,
    target_landing_position,
])
```

---

## 7. `StaticForehandClearEnv` 是 staging wrapper，不是完整 RL env

### 7.1 现象

文件：

```text
environment/overall_environment/src/static_forehand_clear_env.py
```

当前实现已经有有价值的状态机：

```python
class StaticHitState(str, Enum):
    RESET = "RESET"
    PRE_IMPACT_FREEZE = "PRE_IMPACT_FREEZE"
    IMPACT_RELEASED = "IMPACT_RELEASED"
    FLIGHT_EVALUATION = "FLIGHT_EVALUATION"
    TERMINATED = "TERMINATED"
```

也有 release condition：

```python
active contact
phase within tolerance
rho2 <= 1.0
penetration > 0
relative_normal_velocity < 0
```

参考：

- https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/environment/overall_environment/src/static_forehand_clear_env.py#L624-L653

但是 `step()` 返回：

```python
return obs, 0.0, False, False, info
```

参考：

- https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/environment/overall_environment/src/static_forehand_clear_env.py#L767-L808

也就是说：

```text
reward = 0
terminated = False
truncated = False
```

它还没有完成 RL env 必需的 reward 和 termination。

### 7.2 风险

如果把它直接接入 PPO：

```text
策略拿不到正反馈；
episode 不会因失败结束；
hit success / miss / drop racket / fall 都不会形成学习信号；
训练只会变成随机探索或依赖外部 wrapper。
```

### 7.3 建议实现 reward

建议把 reward 分成四段。

#### A. Pre-impact reward

```python
r_pre = (
    w_mimic * r_body_mimic
  + w_grip * r_grip_stability
  + w_face * r_stringbed_orientation
  + w_center * r_stringbed_center_to_shuttle
  + w_velocity * r_racket_head_velocity_toward_shuttle
)
```

#### B. Impact reward

```python
r_impact = (
    w_contact * r_stringbed_contact
  + w_phase * r_phase_window
  + w_rho * r_centered_contact_rho
  + w_closing * r_closing_velocity
  + w_normal * r_contact_normal_alignment
)
```

#### C. Post-impact flight reward

```python
r_flight = (
    w_cross_net * r_cross_net
  + w_apex * r_high_clear_apex
  + w_depth * r_opponent_backcourt_landing
  + w_inbounds * r_inbounds
)
```

#### D. Penalty

```python
penalty = (
    w_fall * body_fall
  + w_drop * racket_dropped
  + w_slip * grip_slip
  + w_penetration * illegal_penetration
  + w_effort * residual_effort
  + w_action_smooth * action_delta
)
```

最终：

```python
reward = r_pre + r_impact + r_flight - penalty
```

### 7.4 建议 termination

```python
terminated = any([
    body_fallen,
    racket_dropped,
    illegal_penetration_too_large,
    missed_impact_window,
    shuttle_landed,
    shuttle_out_of_bounds,
])

truncated = step_index >= max_episode_steps
```

### 7.5 测试

```python
def test_static_hit_reward_positive_on_mock_valid_contact(): ...
def test_static_hit_no_release_outside_phase_window(): ...
def test_static_hit_release_with_valid_stringbed_contact(): ...
def test_static_hit_terminates_on_missed_impact_window(): ...
def test_static_hit_classifies_backcourt_landing(): ...
def test_static_hit_penalizes_racket_drop(): ...
```

---

## 8. `RightHandRacketGripEnv` 是最成熟模块，但还只是局部握拍

### 8.1 现状

文件：

```text
src/grip/right_hand_racket_grip_env.py
```

它已经做了很多正确的事情：

- 加载 MuJoCo model。
- 加载 grip target config 和 reference。
- 查找 right-hand actuator ids。
- step 时只控制 right-hand actuators。
- observation 包含 qpos、qvel、right-hand ctrl。
- info 里包含 site error、V-shape、racket pose error、grip slip、contact count、penetration 等。
- reward term 包括 site match、V-shape、anti-panhandle、anti-thumb、racket pose、contact、no-slip、reference pose、effort、joint limits、no-penetration。

参考：

- https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/src/grip/right_hand_racket_grip_env.py#L1664-L1817
- https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/src/grip/right_hand_racket_grip_env.py#L1873-L1928

这部分可以作为 `π_grip` 的预训练环境。

### 8.2 当前限制

但它现在还不是“全身挥拍中保持握拍”的环境。主要限制：

```text
1. 它只训练右手局部控制。
2. 它没有 replay full-body forehand clear。
3. 它没有相位 phase。
4. 它没有 racket head velocity / stringbed center / shuttle。
5. 它没有 body fall / whole-body balance。
6. 它没有 swing disturbance curriculum。
```

所以它适合做：

```text
Stage 0 / Stage 1: static grip policy pretrain
```

不应该直接代表最终击球任务。

### 8.3 PPO action clipping 问题

文件：

```text
src/grip/train_right_hand_racket_grip_policy.py
```

采样时：

```python
raw_action = mean + noise * std
logprob = Normal(mean, std).log_prob(raw_action).sum(axis=-1)
action = torch.clamp(raw_action, -1.0, 1.0)
```

更新时：

```python
new_logprob = Normal(mean, std).log_prob(actions[batch_indices]).sum(axis=-1)
```

参考：

- https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/src/grip/train_right_hand_racket_grip_policy.py#L2355-L2379
- https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/src/grip/train_right_hand_racket_grip_policy.py#L2577-L2584

这里存在 PPO ratio 不一致风险：old logprob 是 raw action 的概率，env 执行的是 clipped action，update 又用 clipped action 当作 Normal 样本计算 new logprob。

建议改成 tanh-squashed Gaussian：

```python
def sample_squashed_action(mean, log_std, eps=1e-6):
    std = torch.exp(log_std)
    dist = torch.distributions.Normal(mean, std)
    raw = dist.rsample()
    action = torch.tanh(raw)
    logprob = dist.log_prob(raw).sum(-1)
    logprob -= torch.log(1.0 - action.pow(2) + eps).sum(-1)
    entropy = None  # 可用近似或 Monte Carlo
    return action, raw, logprob
```

rollout buffer 同时保存：

```python
raw_actions
squashed_actions
old_logprobs
```

update 时用 raw action 重算 corrected logprob：

```python
new_logprob = dist.log_prob(raw_actions).sum(-1)
new_logprob -= torch.log(1.0 - squashed_actions.pow(2) + 1e-6).sum(-1)
```

### 8.4 验收

```text
1. 零动作 smoke test 仍然通过。
2. PPO rollout buffer shape 不变或明确迁移。
3. 训练 1k step 不出现 NaN。
4. clipped action 与 logprob 不再不一致。
5. validation video 仍能生成。
```

---

## 9. `impact_target.py` 思路正确，但要视为伪标签

### 9.1 现状

`impact_target.py` 会根据右手位置、root frame、forward/right axis 和 racket length 构造 virtual racket head，并用速度最大候选帧估计 impact frame / impact phase。

参考：

- https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/environment/overall_environment/src/impact_target.py#L648-L708

这对于“原始动作没有球拍轨迹”非常有用。

### 9.2 风险

它不是 ground truth，而是 pseudo-label。可能出错的地方：

```text
1. right_hand frame 不等于真实握拍 frame。
2. racket effective length 取值错误会导致 impact point 偏移。
3. velocity maximum 不一定是击球帧。
4. 没有真实 stringbed normal。
5. 没有视频中的球拍弯曲/滞后。
```

### 9.3 建议

将它扩展为 `GhostRacketTeacher`，而不只是 impact target：

```python
@dataclass
class GhostRacketFrame:
    phase: float
    grip_pos_world: np.ndarray
    grip_quat_world: np.ndarray
    head_pos_world: np.ndarray
    head_vel_world: np.ndarray
    stringbed_normal_world: np.ndarray
```

训练早期奖励：

```python
r_ghost = (
    w_grip_pose * exp(-||racket_grip - ghost_grip||^2 / sigma_grip)
  + w_head_pos * exp(-||racket_head - ghost_head||^2 / sigma_head)
  + w_head_vel * exp(-||racket_head_vel - ghost_head_vel||^2 / sigma_vel)
  + w_face * exp(-angle(stringbed_normal, ghost_normal)^2 / sigma_face)
)
```

后期逐步降低：

```text
w_ghost: 1.0 -> 0.5 -> 0.2 -> 0.0
w_shuttle_contact / w_landing: 0.0 -> 0.5 -> 1.0
```

---

## 10. 需要补的 integration tests

当前最容易出错的不是单个 reward，而是多个子系统没有对齐。因此优先补这些测试：

```text
tests/test_checkpoint_action_manifest.py
  - checkpoint metadata contains action_manifest
  - disable_fingers flag matches expected old policy
  - actuator names are unique

tests/test_layered_policy_integration.py
  - body action from old checkpoint can be mapped into overall full action
  - grip action controls only right hand / selected wrist/forearm
  - overlap raises error
  - missing actuator raises error

tests/test_overall_training_env.py
  - reset finite
  - 100 zero-action steps finite under soft weld
  - racket does not immediately disappear from hand in stage0
  - contacts are enabled in training XML

tests/test_static_forehand_clear_env_reward.py
  - valid mock contact produces positive impact reward
  - out-of-window contact does not release shuttle
  - missed impact terminates
  - deep landing gives larger reward than own-side landing

tests/test_grip_hold_runner.py
  - preflight writes report
  - replay-precheck detects adapter readiness
  - train stage rejects missing manifest clearly
  - train stage runs smoke rollout with fake policies
```

---

## 11. 建议 Codex 第一批改动任务

### Task 1：给 grip-hold 加 guard

文件：

```text
BadmintonMimic/scripts/run_posttrain_experiment.py
```

目标：避免 grip-hold spec 误走 fullbody runner。

验收：测试能捕获 ValueError。

### Task 2：新增 action manifest 和 adapter

新增文件：

```text
environment/overall_environment/src/action_adapter.py
```

目标：按 actuator name 映射旧 body action 到 full model action。

验收：旧 checkpoint manifest + full model actuator names 可以生成合法 full action。

### Task 3：集成 LayeredActuatorRouter

目标：body policy action 和 grip policy action 合并。

验收：无 overlap、无 missing、full action finite。

### Task 4：把 `StaticForehandClearEnv.step()` 改成真实 reward / done

目标：不再返回常数 reward 和永不结束。

验收：mock contact tests 通过。

### Task 5：修 grip PPO 的 action logprob

目标：避免 clipped action 与 logprob 不一致。

验收：rollout/update 使用同一套 raw/squashed action。

---

## 12. 参考链接

仓库文件：

- `run_forehand_clear_grip_hold.py`: https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/BadmintonMimic/scripts/run_forehand_clear_grip_hold.py
- `forehand_clear_grip_hold_v1.yaml`: https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/BadmintonMimic/experiments/posttrain/forehand_clear_grip_hold_v1.yaml
- `run_posttrain_experiment.py`: https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/BadmintonMimic/scripts/run_posttrain_experiment.py
- `conf_fullbody_badminton_gmr.yaml`: https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/BadmintonMimic/experiments/fullbody/config_specific_task/conf_fullbody_badminton_gmr.yaml
- `overall_environment`: https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/tree/main/environment/overall_environment
- `static_forehand_clear_env.py`: https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/environment/overall_environment/src/static_forehand_clear_env.py
- `layered_control.py`: https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/environment/overall_environment/src/layered_control.py
- `impact_target.py`: https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/environment/overall_environment/src/impact_target.py
- `right_hand_racket_grip_env.py`: https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/src/grip/right_hand_racket_grip_env.py
- `train_right_hand_racket_grip_policy.py`: https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/src/grip/train_right_hand_racket_grip_policy.py

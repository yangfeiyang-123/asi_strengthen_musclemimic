# 4. 建议你接下来按这个优先级修改

> 目标：把“从无拍正手高远球策略到拿拍击球策略”的开发工作，拆成可以按优先级执行的工程路线图。本文适合直接放进 Codex，让它按 P0/P1/P2/P3 顺序实现。

---

## 0. 总体原则

不要一开始就做：

```text
full-body checkpoint + racket + shuttle + end-to-end PPO
```

应该按以下顺序推进：

```text
P0: 防止误跑，补 manifest / adapter / tests。
P1: 打通 frozen body policy + grip policy + overall training scene。
P2: 把 static-hit wrapper 改成真正 RL env。
P3: 加 ghost racket、soft-weld annealing、phase reward、flight reward。
P4: 做 ASI hard-state mining、motion prior、partial unfreeze 和 ablation。
```

每一步都要有 smoke test 和 acceptance metrics。否则训练失败时无法判断是：

```text
scene 错
actuator mapping 错
checkpoint obs/action 错
握拍策略错
static-hit reward 错
contact filtering 错
shuttle physics 错
```

---

## P0：先修实验入口，避免误跑

### P0.1 给 grip-hold spec 加 runner guard

#### 背景

`run_posttrain_experiment.py` 已经有 `requires_dedicated_grip_hold_runner()`，prepare 阶段也会为 grip-hold 写专用 README，但 `run_stage()` 只阻止 static-hit 的非 prepare stage，没有阻止 grip-hold。

参考：

- https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/musclemimic/badminton/scripts/run_posttrain_experiment.py#L1609-L1613
- https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/musclemimic/badminton/scripts/run_posttrain_experiment.py#L2248-L2257

#### 修改

在：

```text
musclemimic/badminton/scripts/run_posttrain_experiment.py
```

加入：

```python
if stage != "prepare" and requires_dedicated_grip_hold_runner(spec):
    raise ValueError(
        f"{spec['action']} {spec['experiment_id']} requires a dedicated grip-hold runner; "
        f"the PostTrain fullbody runner cannot run stage '{stage}'. "
        "Use musclemimic/badminton/scripts/run_forehand_clear_grip_hold.py."
    )
```

#### 测试

```python
def test_posttrain_runner_rejects_grip_hold_train_stage():
    spec = load_spec("experiments/posttrain/forehand_clear_grip_hold_v1.yaml")
    with pytest.raises(ValueError, match="dedicated grip-hold runner"):
        run_stage(spec, stage="train", arm=None, execute=False)
```

#### 验收

```bash
python musclemimic/badminton/scripts/run_posttrain_experiment.py \
  --spec experiments/posttrain/forehand_clear_grip_hold_v1.yaml \
  --stage train
```

必须清楚报错，而不是生成/执行 fullbody training command。

---

### P0.2 明确 `run_forehand_clear_grip_hold.py` 当前能力

#### 背景

该 runner 目前只支持：

```text
preflight
reset-video
replay-precheck
```

且 `replay_precheck()` 写死：

```python
"policy_replay_ready": False
```

参考：

- https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/musclemimic/badminton/scripts/run_forehand_clear_grip_hold.py#L850-L858
- https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/musclemimic/badminton/scripts/run_forehand_clear_grip_hold.py#L879-L882

#### 修改

把 docstring 改成 diagnostic-only，或者新增 train stage。

短期建议先 diagnostic-only：

```python
"""Run ForehandClear grip-hold diagnostics.

This script does not train yet. Training requires:
- checkpoint action manifest
- action adapter
- overall training scene
- layered body/grip action merge
"""
```

#### 验收

README / docstring / CLI help 不再暗示当前已经能训练。

---

### P0.3 清理 YAML 中的本地绝对路径

#### 背景

当前 YAML 包含：

```yaml
resume_from: /data3/yangfeiyang/WorkSpace/musclemimic/checkpoints/...
body_policy:
  checkpoint: /data3/yangfeiyang/WorkSpace/musclemimic/checkpoints/...
```

这对 Codex、CI、其他机器不可复现。

#### 修改

改为：

```yaml
paths:
  checkpoint_root: ${oc.env:MUSCLEMIMIC_CHECKPOINT_ROOT,checkpoints}

resume_from: ${paths.checkpoint_root}/de63059b16c0/checkpoint_7812
body_policy:
  checkpoint: ${paths.checkpoint_root}/de63059b16c0/checkpoint_7812
```

或者 repo-relative：

```yaml
resume_from: checkpoints/forehand_clear/checkpoint_7812
```

#### 测试

```python
def test_posttrain_spec_has_no_private_absolute_paths():
    ...
```

检查不出现：

```text
/data3/
/home/
/Users/
```

---

## P1：补 checkpoint manifest 和 action adapter

### P1.1 保存 action manifest

#### 背景

基础 full-body badminton config 使用：

```yaml
disable_fingers: true
```

参考：

- https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/fullbody/config_specific_task/conf_fullbody_badminton_gmr.yaml#L381-L390

握拍任务需要右手手指 actuator，所以旧 checkpoint action space 与 full racket scene action space 不一致。

#### 新增文件

```text
musclemimic/utils/action_manifest.py
```

或：

```text
environment/overall_environment/src/action_manifest.py
```

#### manifest 示例

```json
{
  "schema_version": 1,
  "env_name": "MjxMyoFullBody",
  "disable_fingers": true,
  "action_size": 354,
  "actuator_names": ["..."],
  "obs_size": 1234,
  "obs_fields": ["..."],
  "control_min": -1.0,
  "control_max": 1.0
}
```

#### Codex 任务

```text
1. 在训练 checkpoint 保存时输出 action_manifest.json。
2. 如果旧 checkpoint 没有 manifest，提供一个从 metadata + env factory 重建 manifest 的脚本。
3. 在 replay-precheck 中读取 manifest。
```

#### 验收

```bash
python -m environment.overall_environment.src.action_manifest \
  --checkpoint checkpoints/forehand_clear/checkpoint_7812 \
  --print
```

输出包含：

```text
env_name
disable_fingers
action_size
actuator_names
```

---

### P1.2 实现 action adapter

#### 新增文件

```text
environment/overall_environment/src/action_adapter.py
```

#### 接口

```python
@dataclass(frozen=True)
class ActionAdapterReport:
    source_action_size: int
    target_action_size: int
    mapped_count: int
    missing_in_target: list[str]
    extra_in_target: list[str]

class CheckpointToFullActionAdapter:
    def __init__(self, source_actuator_names, target_actuator_names):
        ...

    def adapt(self, source_action: np.ndarray) -> np.ndarray:
        ...

    def report(self) -> ActionAdapterReport:
        ...
```

#### 要求

```text
1. 必须按 actuator name 映射。
2. 不允许按 index 直接拼接。
3. source 中的 actuator 如果 target 没有，必须报错。
4. target 中 source 没有的 actuator 置 0 或交给 residual policy。
5. 所有 action 必须 finite。
```

#### 测试

```python
def test_adapter_maps_by_name_not_index(): ...
def test_adapter_rejects_missing_source_actuator_in_target(): ...
def test_adapter_sets_extra_target_actuators_to_zero(): ...
def test_adapter_rejects_nonfinite_action(): ...
```

---

### P1.3 集成 `LayeredActuatorRouter`

#### 背景

仓库已有：

```text
environment/overall_environment/src/layered_control.py
```

它可以按 actuator name 合并 body action 和 grip action。

参考：

- https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/environment/overall_environment/src/layered_control.py

#### 修改

新增：

```python
def build_router_from_model_and_spec(model, body_manifest, residual_spec) -> LayeredActuatorRouter:
    all_names = actuator_names_from_model(model)
    body_names = body_manifest.actuator_names
    grip_names = resolve_actuator_groups(model, residual_spec["actuator_groups"])
    return LayeredActuatorRouter(all_names, body_names, grip_names)
```

#### 验收

```text
1. body/grip overlap 会报错。
2. grip actuator 不在 full model 会报错。
3. merge 输出长度等于 model.nu。
4. grip action 只覆盖 right hand / selected residual groups。
```

---

## P2：打通 old body policy + right-hand grip policy

### P2.1 加载 body checkpoint 并冻结

#### 目标

实现：

```python
body_policy = load_body_policy(checkpoint)
body_policy.eval()
body_policy.requires_grad_(False)
```

#### 注意

必须同时加载：

```text
obs normalization
action manifest
policy architecture config
```

如果 obs 不一致，需要实现 obs adapter：

```python
obs_body = extract_body_obs(full_obs, manifest.obs_fields)
```

#### 验收

```text
1. body policy 对同一 obs 输出 deterministic action。
2. no grad 更新 body policy 参数。
3. body action size 与 manifest 一致。
```

---

### P2.2 加载 / 初始化 grip policy

#### 背景

已有局部握拍环境：

```text
src/grip/right_hand_racket_grip_env.py
```

参考：

- https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/src/grip/right_hand_racket_grip_env.py#L1664-L1817

#### 修改

定义标准 checkpoint：

```text
outputs/right_hand_racket_grip/policy/policy_latest.pt
```

并提供 loader：

```python
grip_policy = load_grip_policy(path, obs_size, action_size)
```

#### 验收

```text
1. grip policy action size == right-hand actuator count。
2. checkpoint missing 时可选择随机初始化。
3. deterministic eval 可生成 validation video。
```

---

### P2.3 实现 layered rollout smoke test

#### 新增测试

```text
tests/test_layered_forehand_rollout.py
```

#### 测试内容

```python
def test_layered_rollout_100_steps_no_nan():
    env = make_overall_training_env(stage="strong_weld")
    body_policy = FakeBodyPolicy(...)
    grip_policy = FakeGripPolicy(...)
    router = build_router(...)
    obs, info = env.reset()
    for _ in range(100):
        action = layered_policy.act(obs).full_action
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(obs).all()
        assert np.isfinite(reward)
        if terminated or truncated:
            break
```

#### 验收

```text
1. Fake policies 可跑。
2. 真实 checkpoint 可 replay-precheck。
3. full action finite。
4. env 不出现 qpos/qvel NaN。
```

---

## P3：把 overall scene 拆成 inspection 和 training

### P3.1 新增 training scene XML

#### 背景

当前 README 说明 racket 是 free body，未 welded，person-racket contacts 被过滤，raw physics 下球拍会离手。

参考：

- https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/tree/main/environment/overall_environment#L313-L322
- https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/tree/main/environment/overall_environment#L353-L357

#### 新增文件

```text
environment/overall_environment/assets/overall_badminton_training_weld_strong.xml
environment/overall_environment/assets/overall_badminton_training_weld_medium.xml
environment/overall_environment/assets/overall_badminton_training_weld_weak.xml
environment/overall_environment/assets/overall_badminton_training_contact_only.xml
```

#### 要求

```text
1. right-hand/finger actuators enabled。
2. palm/finger-handle contact enabled。
3. stringbed-shuttle contact enabled。
4. optional weld palm-to-racket grip frame。
5. 有 racket head / stringbed / handle sites。
6. 有 shuttle freejoint。
```

#### 验收

```python
def test_training_scene_loads_all_variants(): ...
def test_training_scene_has_expected_contacts(): ...
def test_training_scene_has_racket_sites(): ...
def test_training_scene_has_shuttle_freejoint(): ...
```

---

### P3.2 不要让 inspection XML 承担训练职责

保留：

```text
overall_badminton_scene.xml
```

作为 inspection scene。

新增 README：

```text
environment/overall_environment/README_training.md
```

写清楚：

```text
inspection scene 不用于 RL；
training scene 才启用 hand-racket contact / weld / shuttle contact；
不同 weld stage 使用不同 XML。
```

---

## P4：把 `StaticForehandClearEnv` 改成真正 RL env

### P4.1 实现 reward

当前：

```python
return obs, 0.0, False, False, info
```

参考：

- https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/environment/overall_environment/src/static_forehand_clear_env.py#L767-L808

改为：

```python
reward_terms = self._compute_reward_terms(...)
reward = sum(reward_terms.values())
terminated = self._compute_terminated(...)
truncated = self.step_index >= self.max_episode_steps
info["reward_terms"] = reward_terms
return obs, reward, terminated, truncated, info
```

### P4.2 Reward terms

```text
r_body_mimic
r_grip_stability
r_stringbed_center
r_stringbed_contact
r_phase
r_closing_velocity
r_cross_net
r_backcourt_landing
p_fall
p_racket_drop
p_slip
p_penetration
p_effort
```

### P4.3 Termination

```text
body_fallen
racket_dropped
missed_impact_window
invalid_contact
shuttle_landed
shuttle_out
max_steps
```

### P4.4 Tests

```python
def test_reward_zero_or_negative_without_contact(): ...
def test_valid_contact_releases_shuttle(): ...
def test_valid_contact_gets_positive_impact_reward(): ...
def test_out_of_phase_contact_does_not_release(): ...
def test_missed_impact_terminates(): ...
def test_backcourt_landing_reward_gt_own_side(): ...
```

---

## P5：修 right-hand grip PPO action sampling

### 背景

当前采样阶段对 `raw_action` 算 logprob，但对 env 执行 `torch.clamp(raw_action, -1, 1)`；PPO update 又用 clipped action 计算 new logprob。

参考：

- https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/src/grip/train_right_hand_racket_grip_policy.py#L2355-L2379
- https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/blob/main/src/grip/train_right_hand_racket_grip_policy.py#L2577-L2584

### 修改

改成 tanh-squashed Gaussian，保存 raw action 和 squashed action。

```python
raw_action = mean + std * noise
action = torch.tanh(raw_action)
logprob = dist.log_prob(raw_action).sum(-1)
logprob -= torch.log(1.0 - action.pow(2) + 1e-6).sum(-1)
```

### 验收

```text
1. rollout buffer 有 raw_actions。
2. update 使用 raw_actions 计算 logprob。
3. env 执行 squashed action。
4. 训练不 NaN。
5. deterministic_action 仍输出 [-1, 1]。
```

---

## P6：实现 Ghost Racket Teacher

### P6.1 新增模块

```text
environment/overall_environment/src/ghost_racket.py
```

### P6.2 功能

```text
1. 从 reference motion 生成 ghost racket trajectory。
2. 从 grip seed 读取 palm-to-handle transform。
3. 输出 head position / velocity / stringbed normal。
4. 输出 impact phase。
5. 可视化 ghost racket。
```

### P6.3 验收

```bash
python -m environment.overall_environment.src.ghost_racket \
  --reference ... \
  --grip-seed ... \
  --out outputs/posttrain/ghost/video1.npz \
  --render outputs/posttrain/ghost/video1.mp4
```

必须生成：

```text
video1.npz
video1.mp4
impact_report.json
```

---

## P7：实现 ContactGraphReport

### P7.1 新增模块

```text
environment/overall_environment/src/contact_graph.py
```

### P7.2 功能

```text
hand_handle_edges
stringbed_shuttle_contact
frame_shuttle_contact
illegal_contacts
max_penetration
slip_velocity
contact_count
```

### P7.3 验收

```text
1. no-contact 时不报错。
2. 有 hand-handle contact 时 contact_count 正确。
3. frame contact 和 stringbed contact 可区分。
4. reward 使用 ContactGraphReport，而不是重复遍历 contact。
```

---

## P8：实现 curriculum runner

### P8.1 新增 runner

```text
musclemimic/badminton/scripts/run_forehand_clear_racket_hit.py
```

或扩展：

```text
musclemimic/badminton/scripts/run_forehand_clear_grip_hold.py
```

### P8.2 Stage

```text
preflight
reset-video
replay-precheck
train-static-grip
train-swing-grip
train-ghost-racket
train-static-hit
train-flight
eval
render
```

### P8.3 每个 stage 输出

```text
metrics.json
validation_video.mp4
checkpoint_latest.pt
failure_examples.npz
stage_report.md
```

### P8.4 验收

```text
1. 每个 stage 可 dry-run。
2. 每个 stage 可 smoke train 1000 steps。
3. 每个 stage 有明确成功指标。
4. 下一 stage 自动读取上一 stage checkpoint。
```

---

## P9：实现 ASI hard-state mining

### P9.1 新增模块

```text
environment/overall_environment/src/asi_hard_state.py
```

### P9.2 功能

```text
1. 捕获 failure event。
2. 保存失败前 K 帧 state。
3. 按 failure type 采样 initial state。
4. 支持开关和比例。
```

### P9.3 验收

```text
1. buffer 可 save/load。
2. reset_from_state finite。
3. failure type 统计正确。
4. hard-state sampling 关闭时行为不变。
```

---

## P10：实验 ablation

### P10.1 实验组合

```text
A0: baseline no racket replay
A1: frozen body + grip policy
A2: A1 + strong/medium/weak weld curriculum
A3: A2 + ghost racket
A4: A3 + static shuttle hit
A5: A4 + shuttle flight
A6: A5 + ASI hard-state mining
A7: A6 + partial unfreeze wrist/forearm/shoulder
```

### P10.2 指标

```text
racket_drop_rate
mean_grip_slip_m
contact_count
max_penetration_m
valid_stringbed_contact_rate
wrong_surface_contact_rate
over_net_rate
opponent_backcourt_landing_rate
body_mimic_error
fall_rate
muscle_effort
```

### P10.3 报告

每次实验输出：

```text
reports/ablation_summary.md
metrics/ablation_metrics.csv
videos/best_success.mp4
videos/common_failure.mp4
```

---

## 总体执行顺序

```text
Week / Phase 1:
  P0 runner guard + doc correction
  P1 manifest + adapter + router tests

Week / Phase 2:
  P2 layered rollout smoke test
  P3 training scene XML variants
  P5 grip PPO fix

Week / Phase 3:
  P4 static-hit reward / termination
  P6 ghost racket teacher
  P7 contact graph

Week / Phase 4:
  P8 curriculum runner
  P9 hard-state mining
  P10 ablation
```

不要跳过 P0-P2。只要 action/obs/actuator mapping 没有打通，后面的训练结果都不可解释。

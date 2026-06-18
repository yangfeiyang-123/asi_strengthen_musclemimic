# eade4bc 蒸馏系统审查报告

审查对象：`yangfeiyang-123/asi_strengthen_musclemimic`，用户说明的最新主线更新为：

```text
main: 0348709 -> eade4bc
```

审查目标：

1. 判断当前整体仓库是否存在隐藏错误；
2. 判断当前实现还有哪些值得改进的地方；
3. 判断现在离“可以正式进行 ForehandClear no-future-lookahead student policy 蒸馏”还差什么。

> 结论先行：  
> **eade4bc 已经具备“底层蒸馏能力”的主要模块：student observation filter、distillation shard IO、teacher off-policy collection、BC trainer、DAgger-style student rollout relabeling、student PPO 配置。**  
> 但它仍然没有完全达到“端到端可复现实验流程”的标准。现在可以开始做库函数级调试和小规模蒸馏 smoke test，但还需要补齐 CLI / scripts、评估对比、DAgger loop driver、测试覆盖和若干隐藏稳定性修正，才适合跑正式实验。

---

## 1. 当前已经完成的内容

### 1.1 Student observation filter 已经基本满足要求

当前 `musclemimic/distill/obs_filter.py` 已实现：

```python
StudentObservationFilterWrapper
StudentObsSpec
build_student_obs_indices
filter_student_obs
```

它的设计意图是：

```text
Teacher observation:
joint + muscle + foot contact + full GoalTrajMimic goal lookahead

Student observation:
all non-goal observations + motion phase
```

也就是：

```text
student_obs = state_obs + phase
```

其中 `state_obs` 包含非 goal 的观测部分，通常对应 joint state、muscle state、touch/contact 等；`phase` 来自 goal group 的最后一维。

这个设计方向是正确的。它符合之前建议的第一版 student：

```text
Student v1:
joint + muscle + foot contact + motion phase
    -> 354-D muscle controls
```

不建议第一版完全移除 phase，因为 ForehandClear 是长时序动作。没有 phase，student 容易混淆引拍、挥拍、击球附近和随挥阶段。

当前 wrapper 的优点：

- 只过滤返回给 policy 的 observation；
- 不改变底层环境状态；
- 不改变 trajectory handler；
- 不改变 MimicReward；
- 支持 `reset / reset_to / step`；
- 支持与 `NStepWrapper(split_goal=True)` 组合；
- 已有单元测试覆盖基本行为。

### 1.2 `wrap_env()` 已经接入 student filter

`musclemimic/algorithms/common/env_utils.py` 里已经在标准 wrapper 链前加入：

```python
student_cfg = config.get("student_obs_filter", {})
if student_cfg.get("enabled", False):
    env = StudentObservationFilterWrapper(env, student_cfg)
```

然后再做：

```python
NStepWrapper -> VecEnv -> LogWrapper -> AutoResetWrapper -> NormalizeVecReward
```

这个顺序是正确的。原因是：

```text
先做 student observation filtering
再做 history stacking / split_goal
```

这样可以避免把 full goal lookahead 历史重复堆叠到 student 输入里。

### 1.3 `PPOJax._create_network()` 已经按 student observation 维度建网

`musclemimic/algorithms/ppo/ppo.py` 中 `_create_network()` 已经检查：

```python
if exp.get("student_obs_filter", {}).get("enabled", False):
    env = StudentObservationFilterWrapper(env, exp.student_obs_filter)
```

这可以保证：

```text
ActorCritic 网络初始化时使用 student observation dim
训练环境 wrap_env 后也返回 student observation dim
```

这是必要的。否则会出现：

```text
network input dim != env observation dim
```

这一点现在基本满足。

### 1.4 Distillation dataset IO 已经实现

当前 `musclemimic/distill/dataset.py` 已经实现：

```python
write_distill_shard()
load_metadata()
DistillDataset
```

最小必需字段是：

```python
REQUIRED_FIELDS = ("student_obs", "teacher_action")
```

这足够支持第一版 behavior cloning：

```text
student_obs -> teacher_action
```

并且可以额外保存：

```text
teacher_mu
teacher_value
teacher_log_prob
reward
done
absorbing
traj_no
subtraj_step_no
phase
full_obs
student_action
```

这是合理的。

### 1.5 Teacher off-policy collection 已经实现

当前 `musclemimic/distill/collect_teacher.py` 已经实现：

```python
collect_teacher_dataset()
```

它会：

1. 使用 lookahead teacher policy rollout；
2. 构造 student observation；
3. 保存 `student_obs -> teacher_action` 样本；
4. 保存 teacher mean、value、log_prob、reward、done、trajectory id、phase 等信息。

这已经覆盖了 off-policy distillation 的第一阶段需求。

### 1.6 BC trainer 已经实现

当前 `musclemimic/distill/train_bc.py` 已经实现：

```python
train_bc()
evaluate_bc_loss()
```

它会：

1. 加载 distillation shards；
2. 强制开启 `student_obs_filter`；
3. 初始化 PPO-compatible `ActorCritic` student network；
4. 用 action MSE + optional value distillation 训练 student；
5. 保存 UnifiedCheckpointManager checkpoint；
6. 保存 `distill_metadata.json`。

这已经满足：

```text
teacher rollout dataset -> BC student checkpoint
```

的核心需求。

### 1.7 DAgger-style correction 已经实现核心库函数

当前 `musclemimic/distill/dagger.py` 已经实现：

```python
collect_dagger_dataset()
```

它会：

1. 使用 student policy rollout；
2. 在 student 访问到的 full observation 上用 teacher relabel；
3. 保存 `student_obs -> teacher_mu`；
4. 保存 student action、reward、done、absorbing、trajectory id、phase 等信息；
5. 支持 `mix_teacher_action_prob`，允许 rollout action 混入 teacher action。

这已经覆盖了“on-policy correction / DAgger-style relabeling”的核心思想：

```text
student rollout state distribution
    -> teacher relabel
    -> aggregate dataset
    -> continue BC/KD training
```

### 1.8 Student PPO fine-tune config 已经出现

当前新增了：

```text
fullbody/config_specific_task/conf_fullbody_badminton_student_gmr.yaml
fullbody/config_specific_task/distill/conf_fullbody_forehandclear_student_phase_ppo.yaml
```

其中 student PPO 配置继承 ForehandClear badminton GMR teacher 配置，并开启：

```yaml
student_obs_filter:
  enabled: true
  drop_goal_lookahead: true
  keep_motion_phase: true
  require_goal_group: true
  require_motion_phase: true
```

这是正确方向。它意味着：

```text
policy input: no future lookahead
reward: still MimicReward against reference trajectory
```

这一点非常重要：  
**student 不看 future lookahead，但训练环境仍然可以使用 reference trajectory 计算 MimicReward。**

---

## 2. 当前可能存在的隐藏错误

下面这些不是全部都会立刻报错，但它们属于正式跑蒸馏前应该优先排查的隐患。

---

### 2.1 `collect_teacher_dataset()` 没有显式关闭 `student_obs_filter`

`collect_dagger_dataset()` 里有这个保护：

```python
rollout_cfg = OmegaConf.create(OmegaConf.to_container(teacher_exp, resolve=True))
rollout_cfg.num_envs = int(num_envs)
if "student_obs_filter" in rollout_cfg:
    rollout_cfg.student_obs_filter.enabled = False
```

但是 `collect_teacher_dataset()` 里目前没有同样的保护。它直接：

```python
exp_cfg = OmegaConf.create(OmegaConf.to_container(agent_conf.config.experiment, resolve=True))
exp_cfg.num_envs = int(num_envs)
teacher_env = wrap_env(env, exp_cfg)
```

如果调用者不小心把 student config 或带 `student_obs_filter.enabled=True` 的 config 传给 teacher collection，就会发生：

```text
teacher_env 返回 filtered student obs
teacher network 可能期待 full teacher obs
```

从而导致维度错误，或者更隐蔽地采集到错误分布。

#### 建议修改

在 `collect_teacher_dataset()` 里加入和 DAgger 一样的保护：

```python
if "student_obs_filter" in exp_cfg:
    exp_cfg.student_obs_filter.enabled = False
```

并在 metadata 里写入：

```json
"collector_obs_mode": "teacher_full_obs"
```

---

### 2.2 `drop_goal_lookahead` 配置项目前没有实际被读取

配置里写了：

```yaml
student_obs_filter:
  drop_goal_lookahead: true
  keep_motion_phase: true
```

但是 `obs_filter.py` 实际只读取：

```python
keep_motion_phase
require_goal_group
require_motion_phase
```

没有读取 `drop_goal_lookahead`。

这不一定会影响当前默认行为，因为当前逻辑本来就是“drop all goal except phase”。但从工程语义上看，这是一个容易误导的配置项。

#### 风险

用户可能以为：

```yaml
drop_goal_lookahead: false
```

会保留 lookahead，但实际上不会。

#### 建议修改

方案 A：明确实现该字段。

```python
drop_goal_lookahead = bool(_cfg_get(config, "drop_goal_lookahead", True))
if not drop_goal_lookahead:
    student_indices = np.arange(raw_obs_dim)
else:
    student_indices = state_indices + optional phase
```

方案 B：如果暂时不支持，就删除这个字段，只保留：

```yaml
keep_motion_phase: true
```

或者在代码里发现 `drop_goal_lookahead` 不是 `true` 时直接 raise：

```python
if _cfg_get(config, "drop_goal_lookahead", True) is not True:
    raise NotImplementedError("Only drop_goal_lookahead=True is supported.")
```

---

### 2.3 `StudentObsContainer` 可能缺少兼容接口

当前 `StudentObsContainer` 有：

```python
get_obs_ind_by_group()
get_all_group_names()
filter_by_group()
entries()
keys()
```

但是很多环境相关代码可能使用：

```python
env.obs_container.items()
env.obs_container.get(...)
env.obs_container["name"]
```

例如旧的 trajectory export 逻辑曾经遍历：

```python
for name, obs in env.obs_container.items():
    ...
```

如果 student policy 在某些 evaluation/export 路径上使用 filtered env，就可能因为 `StudentObsContainer` 没有 `.items()` 报错。

#### 建议修改

至少补齐：

```python
def items(self):
    return self._group_indices.items()

def __contains__(self, key):
    return key in self._group_indices

def __getitem__(self, key):
    return self._group_indices[key]

def get(self, key, default=None):
    return self._group_indices.get(key, default)
```

如果以后需要按单个 observation name 提取 touch/joint/muscle，更建议把原始 `obs_container` 映射到 filtered obs index，而不是只保留 group 级别。

---

### 2.4 Teacher / student collection 默认会更新 RunningMeanStd

`collect_teacher_dataset()` 和 `collect_dagger_dataset()` 都调用：

```python
network.apply(..., mutable=["run_stats"])
ts = ts.replace(run_stats=updates["run_stats"])
```

这意味着 collection 过程中 policy 的 normalization running stats 会继续变化。

这和现有 inference 代码风格一致，但在 distillation 语境下可能不是最稳妥的。原因是 teacher checkpoint 中的 running stats 本来就是训练好的归一化统计，采集过程中继续更新可能导致：

```text
同一个 checkpoint 在不同 collection 顺序 / num_envs / seed 下产生略有不同的 labels
```

#### 建议修改

给 collection 增加参数：

```python
freeze_run_stats: bool = True
```

默认冻结。实现上可以：

- 继续 `mutable=["run_stats"]` 但不写回 updates；
- 或者如果网络允许，使用非 mutable apply。

建议：

```python
if freeze_run_stats:
    # discard updates
    next_ts = ts
else:
    next_ts = ts.replace(run_stats=updates["run_stats"])
```

对 BC training 来说，student 的 run_stats 可以训练更新；对 teacher labeling 来说，teacher run_stats 更建议冻结。

---

### 2.5 DAgger 数据缺少 `rollout_action` 和 `used_teacher_mask`

`collect_dagger_dataset()` 支持：

```python
mix_teacher_action_prob
```

实际执行动作为：

```python
rollout_action = where(use_teacher, teacher_mu, student_action)
```

但是 shard 里目前只保存：

```text
teacher_action
teacher_mu
student_action
reward
done
...
```

没有保存：

```text
rollout_action
used_teacher_mask
```

#### 风险

后续诊断时无法知道该状态是由 student action 访问到的，还是由 teacher mixed action 访问到的。这样 DAgger 数据质量分析不完整。

#### 建议修改

在 `build_dagger_shard_data()` 里保存：

```python
"rollout_action": rollout_action
"used_teacher_action": use_teacher
```

并在 metadata 中保存：

```json
"mix_teacher_action_prob": ...
```

metadata 已经有 `mix_teacher_action_prob`，但单步 mask 仍然有价值。

---

### 2.6 `teacher_log_prob` 当前是 teacher 对 teacher mean 的 log-prob

当前 DAgger 中：

```python
teacher_mu = distribution_mean(teacher_pi)
teacher_log_prob = teacher_pi.log_prob(teacher_mu)
```

这对 “teacher confidence” 诊断有一点用，但它不是：

```text
teacher log_prob(student_action)
teacher log_prob(rollout_action)
```

如果后续想做 KL / advantage-weighted relabel / action filtering，这个字段不够。

#### 建议修改

额外保存：

```python
teacher_log_prob_teacher_mu = teacher_pi.log_prob(teacher_mu)
teacher_log_prob_student_action = teacher_pi.log_prob(student_action)
teacher_log_prob_rollout_action = teacher_pi.log_prob(rollout_action)
```

如果只做 MSE BC，可以暂时忽略。

---

### 2.7 当前 DAgger 只支持 `len_obs_history=1`

`collect_dagger_dataset()` 里明确：

```python
if teacher_exp.get("len_obs_history", 1) > 1:
    raise NotImplementedError(...)
if student_exp.get("len_obs_history", 1) > 1:
    raise NotImplementedError(...)
```

这和当前 student phase config 中 `len_obs_history: 1` 一致，因此不是立即错误。

但是需要在文档和 CLI 中明确：

```text
目前 DAgger only supports len_obs_history=1
```

否则后续开启 history stack 会直接报错。

---

### 2.8 `__init__.py` 没有导出新模块

当前 `musclemimic/distill/__init__.py` 只导出：

```python
DistillDataset
StudentObservationFilterWrapper
StudentObsSpec
build_student_obs_indices
filter_student_obs
load_metadata
write_distill_shard
```

没有导出：

```python
collect_teacher_dataset
collect_dagger_dataset
train_bc
bc_loss
distribution_mean
```

这不影响直接 import 具体模块，但不利于 API 使用和测试。

#### 建议修改

补上：

```python
from musclemimic.distill.collect_teacher import collect_teacher_dataset
from musclemimic.distill.dagger import collect_dagger_dataset
from musclemimic.distill.losses import bc_loss, distribution_mean
from musclemimic.distill.train_bc import train_bc
```

---

### 2.9 没有 distillation CLI / pyproject entrypoints

当前 `pyproject.toml` 的 `[project.scripts]` 只注册了 dataset path / cache / GMR 相关命令，尚未注册 distillation 相关入口。

这意味着现在虽然库函数存在，但使用者还必须手写 Python 脚本加载 checkpoint、实例化 env、调用 collector、调用 BC trainer。

#### 建议新增命令

```toml
[project.scripts]
musclemimic-distill-collect-teacher = "musclemimic.distill.cli:collect_teacher_cli"
musclemimic-distill-train-bc = "musclemimic.distill.cli:train_bc_cli"
musclemimic-distill-collect-dagger = "musclemimic.distill.cli:collect_dagger_cli"
musclemimic-distill-dagger-loop = "musclemimic.distill.cli:dagger_loop_cli"
musclemimic-distill-eval-student = "musclemimic.distill.cli:evaluate_student_cli"
```

或者在 `BadmintonMimic/scripts/` 下提供任务专用脚本。

---

### 2.10 缺少 DAgger / BC checkpoint round-trip 的测试

当前已有 student obs filter 和 dataset IO 测试，但还需要：

```text
test_collect_teacher_dataset_smoke
test_collect_dagger_dataset_smoke
test_train_bc_writes_reloadable_checkpoint
test_student_config_instantiates_env_and_agent
test_student_ppo_one_update_smoke
```

否则隐藏维度错误、checkpoint 兼容错误、run_stats 错误，可能要到正式跑才暴露。

---

## 3. 值得改进的地方

---

### 3.1 增加 ForehandClear 专用 distillation scripts

建议新增：

```text
BadmintonMimic/scripts/collect_forehand_clear_teacher_dataset.py
BadmintonMimic/scripts/train_forehand_clear_student_bc.py
BadmintonMimic/scripts/collect_forehand_clear_dagger_dataset.py
BadmintonMimic/scripts/run_forehand_clear_dagger_loop.py
BadmintonMimic/scripts/evaluate_forehand_clear_student.py
```

这些脚本不需要写很多算法逻辑，只需要做：

```text
load checkpoint
restore config
apply motion path / config overrides
instantiate env
build agent_conf
load agent_state
call distill library function
write metadata
```

这样 Codex / 你本人可以直接从命令行跑。

---

### 3.2 增加 DAgger loop driver

现在有 `collect_dagger_dataset()`，但还没有一个完整 loop：

```text
Dataset_0 = teacher rollout dataset
student_0 = BC(Dataset_0)

for k in 1..K:
    collect DAgger dataset with student_{k-1}
    aggregate Dataset_0 + ... + Dataset_k
    student_k = BC / continue-BC(aggregate dataset)
    evaluate student_k
```

建议实现：

```python
run_dagger_loop(
    teacher_checkpoint,
    initial_student_checkpoint,
    config,
    output_root,
    iterations,
    num_envs,
    num_steps_per_iter,
    train_steps_per_iter,
    mix_schedule,
)
```

输出目录建议：

```text
outputs/distill/forehand_clear/
  teacher_dataset/
  bc_student_0/
  dagger_iter_001/
    dataset/
    student/
    eval/
  dagger_iter_002/
    ...
  final_student/
```

---

### 3.3 增加 teacher-vs-student evaluation report

正式判断 student 是否成功，不能只看 BC loss。必须看闭环 rollout：

```text
teacher rollout metrics
student BC rollout metrics
student after DAgger metrics
student after PPO fine-tune metrics
```

至少输出：

```text
mean_episode_return
mean_episode_length
early_termination_rate
completion_rate
err_root_xyz
err_root_yaw
err_joint_pos
err_joint_vel
err_site_abs
err_rpos
reward_qpos
reward_qvel
reward_root_pos
reward_root_vel
reward_rpos
reward_rquat
reward_rvel_rot / lin
action_mse_to_teacher_on_eval_states
```

建议生成：

```text
metrics.json
summary.md
comparison.csv
optional video
```

---

### 3.4 增加 train / val split 支持

`DistillDataset` 支持寻找 `train_*.npz` / `val_*.npz`，但 writer 当前主要写 `shard_*.npz`。

建议在 collector 增加：

```python
split="train" | "val"
```

并写：

```text
train_000000.npz
val_000000.npz
```

或者提供一个 split 脚本：

```text
musclemimic-distill-split-dataset
```

这样 BC 训练的 val_action_mse 才稳定可用。

---

### 3.5 增加 KL / Gaussian distillation

当前 `bc_loss()` 主要是 action MSE：

```text
student_mu vs teacher_action
```

这足够跑第一版。但如果想更像 policy distillation，可以保存 teacher log_std / std，并加入 KL：

```text
KL(π_teacher || π_student)
```

因为当前 ActorCritic 是 diagonal Gaussian，分布 KL 很容易实现。

建议字段：

```text
teacher_mu
teacher_log_std
teacher_std
```

loss：

```python
action_mse + alpha * value_mse + beta * gaussian_kl
```

---

### 3.6 增加 freeze_run_stats 策略

建议在以下函数都支持：

```python
collect_teacher_dataset(..., freeze_run_stats=True)
collect_dagger_dataset(..., freeze_run_stats=True)
evaluate_student(..., freeze_run_stats=True)
```

并且默认冻结 teacher run_stats。student BC 训练时可以更新 student run_stats。

---

### 3.7 增加 observation diagnostic 工具

需要一个小工具确认 student observation 真的不含 future lookahead：

```text
raw_obs_dim
goal_dim
student_obs_dim
kept_goal_indices
phase_index
dropped_goal_indices
```

命令示例：

```bash
uv run python BadmintonMimic/scripts/inspect_student_obs_filter.py \
  --config-name config_specific_task/conf_fullbody_badminton_student_gmr
```

输出应类似：

```text
raw_obs_dim: 2418
goal_dim: 469
state_dim: 1949
phase_index_raw: 2417
student_obs_dim: 1950
kept: state + phase
dropped: 468 future/current goal features
```

具体数值以实际环境为准。

---

## 4. 现在离“可以进行蒸馏”还差什么？

这里要分两个层级回答。

---

### 4.1 如果是“开发者手写 Python 调用库函数”，现在基本可以开始小规模 smoke test

现在已经有：

```text
student obs filter
dataset IO
teacher collector
BC trainer
DAgger collector
student PPO config
```

所以从库函数能力看，已经可以做：

```text
1. load teacher checkpoint
2. instantiate ForehandClear env
3. call collect_teacher_dataset()
4. call train_bc()
5. load student checkpoint
6. call collect_dagger_dataset()
7. aggregate dataset
8. call train_bc() again
9. run fullbody/eval.py with student config
```

但是这仍然需要你手写 glue code。

---

### 4.2 如果是“正式可复现实验流程”，还差以下内容

必须补齐：

```text
A. 命令行入口 / task scripts
B. checkpoint loading + env creation glue
C. dataset split / aggregation / metadata规范
D. DAgger loop driver
E. student closed-loop evaluation
F. PPO fine-tune command/config验证
G. test coverage
H. README 使用说明
```

也就是说，目前还不是“一条命令可以跑完蒸馏”的状态。

---

## 5. 推荐的最小下一步任务清单

下面是给 Codex 的优先级任务。

---

### P0：修正隐藏错误

#### P0.1 collect_teacher_dataset 禁用 student filter

文件：

```text
musclemimic/distill/collect_teacher.py
```

修改：

```python
if "student_obs_filter" in exp_cfg:
    exp_cfg.student_obs_filter.enabled = False
```

#### P0.2 StudentObsContainer 增加兼容接口

文件：

```text
musclemimic/distill/obs_filter.py
```

新增：

```python
def items(self):
    return self._group_indices.items()

def __contains__(self, key):
    return key in self._group_indices

def __getitem__(self, key):
    return self._group_indices[key]

def get(self, key, default=None):
    return self._group_indices.get(key, default)
```

#### P0.3 DAgger 保存 rollout_action 和 used_teacher_mask

文件：

```text
musclemimic/distill/dagger.py
```

保存：

```text
rollout_action
used_teacher_action
teacher_log_prob_student_action
teacher_log_prob_rollout_action
```

#### P0.4 collection 默认 freeze teacher run_stats

文件：

```text
musclemimic/distill/collect_teacher.py
musclemimic/distill/dagger.py
```

新增参数：

```python
freeze_run_stats: bool = True
```

---

### P1：补齐可运行入口

新增：

```text
BadmintonMimic/scripts/collect_forehand_clear_teacher_dataset.py
BadmintonMimic/scripts/train_forehand_clear_student_bc.py
BadmintonMimic/scripts/collect_forehand_clear_dagger_dataset.py
BadmintonMimic/scripts/run_forehand_clear_dagger_loop.py
BadmintonMimic/scripts/evaluate_forehand_clear_student.py
```

每个脚本都要支持：

```text
--teacher-path
--student-path
--config-name
--output-dir
--num-envs
--num-steps
--seed
--motion-path optional
--wandb disabled/online
```

---

### P2：补齐 student evaluation

新增评估脚本输出：

```text
metrics.json
summary.md
comparison.csv
optional video path
```

评价对象：

```text
teacher
BC student
DAgger student
PPO-finetuned student
```

---

### P3：补齐测试

新增测试：

```text
tests/unit/test_distill_collect_teacher.py
tests/unit/test_distill_dagger.py
tests/unit/test_distill_train_bc_checkpoint.py
tests/unit/test_student_config_smoke.py
```

其中 `test_student_config_smoke` 应能做到：

```text
load config
instantiate env
init PPOJax agent_conf
assert obs dim == student obs dim
run one reset/step
```

---

### P4：补齐文档

新增：

```text
BadmintonMimic/docs/forehand_clear_student_distillation.md
```

内容包括：

```text
1. teacher checkpoint 准备
2. collect teacher dataset
3. train BC student
4. DAgger correction
5. PPO fine-tune
6. evaluation
7. known limitations
```

---

## 6. 推荐端到端流程

目标目录：

```text
outputs/distill/forehand_clear/
```

推荐流程：

### Step 1：收集 teacher off-policy dataset

```bash
uv run python BadmintonMimic/scripts/collect_forehand_clear_teacher_dataset.py \
  --teacher-path /path/to/teacher/checkpoint \
  --config-name config_specific_task/conf_fullbody_badminton_gmr \
  --output-dir outputs/distill/forehand_clear/teacher_dataset \
  --num-envs 256 \
  --num-steps 2000 \
  --seed 0
```

### Step 2：训练 BC student

```bash
uv run python BadmintonMimic/scripts/train_forehand_clear_student_bc.py \
  --config-name config_specific_task/conf_fullbody_badminton_student_gmr \
  --dataset-dir outputs/distill/forehand_clear/teacher_dataset \
  --output-dir outputs/distill/forehand_clear/bc_student \
  --num-steps 200000 \
  --batch-size 4096
```

### Step 3：收集 DAgger dataset

```bash
uv run python BadmintonMimic/scripts/collect_forehand_clear_dagger_dataset.py \
  --teacher-path /path/to/teacher/checkpoint \
  --student-path outputs/distill/forehand_clear/bc_student/checkpoints/... \
  --config-name config_specific_task/conf_fullbody_badminton_student_gmr \
  --output-dir outputs/distill/forehand_clear/dagger_iter_001/dataset \
  --num-envs 256 \
  --num-steps 2000 \
  --mix-teacher-action-prob 0.1
```

### Step 4：聚合数据继续 BC

```bash
uv run python BadmintonMimic/scripts/train_forehand_clear_student_bc.py \
  --config-name config_specific_task/conf_fullbody_badminton_student_gmr \
  --dataset-dir outputs/distill/forehand_clear/aggregated_dataset \
  --resume-student outputs/distill/forehand_clear/bc_student/checkpoints/... \
  --output-dir outputs/distill/forehand_clear/dagger_iter_001/student \
  --num-steps 100000
```

### Step 5：PPO fine-tune student

```bash
uv run fullbody/experiment.py \
  --config-name config_specific_task/conf_fullbody_badminton_student_gmr \
  experiment.resume_from=outputs/distill/forehand_clear/dagger_iter_001/student/checkpoints/... \
  experiment.reset_std_on_resume=0.5 \
  wandb.mode=online
```

### Step 6：评估

```bash
uv run python BadmintonMimic/scripts/evaluate_forehand_clear_student.py \
  --teacher-path /path/to/teacher/checkpoint \
  --student-path /path/to/student/checkpoint \
  --config-name config_specific_task/conf_fullbody_badminton_student_gmr \
  --output-dir outputs/distill/forehand_clear/eval \
  --num-envs 20 \
  --num-steps 500
```

---

## 7. 当前状态评级

| 模块 | 当前状态 | 是否满足正式蒸馏 |
|---|---:|---:|
| Lookahead teacher PPO | 已有 | 是 |
| Student obs filter | 已有 | 基本是 |
| Dataset IO | 已有 | 基本是 |
| Off-policy teacher collection | 已有 | 基本是，但需小修 |
| BC trainer | 已有 | 基本是 |
| DAgger correction collector | 已有 | 基本是，但需 diagnostics / tests |
| Student PPO config | 已有 | 基本是 |
| CLI / scripts | 缺少 | 否 |
| DAgger loop driver | 缺少 | 否 |
| Evaluation comparison | 缺少 | 否 |
| Unit / smoke tests | 不足 | 否 |
| Documentation | 不足 | 否 |

综合判断：

```text
库函数级别：70% - 80% 已满足
端到端实验级别：50% - 60% 已满足
正式可复现实验：仍需 1-2 轮集成和测试
```

---

## 8. 最终结论

eade4bc 这次更新已经完成了关键底层能力：

```text
student_obs_filter
distill dataset IO
teacher off-policy collection
BC trainer
DAgger-style student rollout relabeling
student PPO config
```

所以现在不是“还不能做蒸馏”，而是：

> **可以开始小规模 smoke-test 蒸馏，但还不建议直接跑大规模正式实验。**

正式蒸馏前必须完成：

```text
1. 修复 collect_teacher_dataset 不强制关闭 student filter 的隐患；
2. 给 DAgger 数据增加 rollout_action / used_teacher_mask；
3. 增加 freeze_run_stats 选项；
4. 补齐 CLI / BadmintonMimic scripts；
5. 补齐 DAgger loop driver；
6. 补齐 teacher-vs-student evaluation；
7. 补齐单元测试和 smoke tests。
```

完成这些之后，项目就可以比较稳地进入：

```text
teacher lookahead PPO
    -> off-policy BC student
    -> DAgger correction
    -> no-future-lookahead PPO fine-tune
    -> teacher-vs-student evaluation
```

这一完整蒸馏流程。

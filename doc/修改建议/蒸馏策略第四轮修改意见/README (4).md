# ForehandClear Student Distillation 最新工程审查与方法说明

审查对象：`yangfeiyang-123/asi_strengthen_musclemimic` 当前 `main` 最新可见内容。
审查范围：只考虑 **ForehandClear body-trajectory imitation** 的 teacher → student 蒸馏；不考虑球拍、羽毛球、击球接触或落点任务。

> 说明：这份报告基于远端仓库静态代码审查；我没有在你的 GPU 环境里实际运行 smoke test、训练或评估。因此本文对“隐藏错误”的判断是代码路径、数据 schema、命令链路和配置一致性的风险分析。

---

## 0. 总体结论

当前仓库已经从“底层模块可用”推进到了“接近可执行蒸馏工程”的状态。现在已经具备：

1. student observation filter：把 teacher 的 full observation 裁成 `state + motion phase`，去掉 future lookahead；
2. teacher rollout dataset collector；
3. distillation NPZ shard 数据结构；
4. BC/KD trainer；
5. DAgger-style student rollout + teacher relabel collector；
6. DAgger loop orchestration；
7. student no-future-lookahead PPO config；
8. teacher-vs-student evaluation compare；
9. console entrypoints 和 ForehandClear wrapper scripts；
10. dataset inspect 工具。

所以，**现在可以开始小规模 smoke-test 蒸馏**，例如 `num_envs=2`、`num_steps=10~100`、`BC num_steps=5~20` 的链路测试。

但是，**现在还不建议直接开始正式完整 distillation 实验**。原因是还有一个 P0 级别隐藏风险：**teacher shards 和 DAgger shards 的字段 schema 不完全一致，而 `DistillDataset` 当前会把所有字段跨 shard 拼接并要求每个字段样本数一致。只要把 teacher 初始 shard 和 DAgger correction shard append 到同一个 `train` split，BC retrain 很可能因为字段长度不一致而失败。**

---

## 1. 当前已经满足的部分

### 1.1 Student observation filter 已经基本正确

`StudentObservationFilterWrapper` 当前实现了 v1 student 目标：保留所有 non-goal observation，并默认只保留 goal 最后一维作为 motion phase。它现在读取了：

```yaml
student_obs_filter:
  enabled: true
  drop_goal_lookahead: true
  keep_motion_phase: true
  require_goal_group: true
  require_motion_phase: true
```

其核心逻辑是：

```text
teacher/full obs:
  [state obs, full goal lookahead]

student obs:
  [state obs, phase]
```

这正符合当前最合理的 student v1 设计：**去掉 future lookahead，但保留 motion phase，避免 ForehandClear 长时序动作发生相位混淆。**

当前实现还加入了对 `drop_goal_lookahead` 的检查：当 `keep_motion_phase=True` 但 `drop_goal_lookahead=False` 时会直接报错。这可以避免误把“仍然看到完整 lookahead 的 policy”当作 no-lookahead student。

### 1.2 package discovery 和 console scripts 已明显改善

`pyproject.toml` 当前已经包含：

```toml
include = ["musclemimic*", "loco_mujoco*", "bimanual*", "fullbody*", "BadmintonMimic*", "src*"]
```

这解决了上一轮审查中 `BadmintonMimic.scripts.*` 可能无法被安装后 entrypoint import 的问题。

同时已经注册了通用命令和 ForehandClear 专用命令，例如：

```text
musclemimic-distill-collect-teacher
musclemimic-distill-train-bc
musclemimic-distill-collect-dagger
musclemimic-distill-run-dagger
musclemimic-distill-compare
musclemimic-distill-inspect-dataset

forehand-clear-distill-collect-teacher
forehand-clear-distill-train-bc
forehand-clear-distill-collect-dagger
forehand-clear-distill-run-dagger
forehand-clear-distill-evaluate
forehand-clear-distill-inspect-obs
```

这说明工程已经从“只能 import 库函数”推进到“有命令行工作流入口”。

### 1.3 teacher dataset collector 已经更安全

当前 `collect_teacher_dataset()` 已经有这些关键改进：

```text
build_teacher_rollout_config()
freeze_run_stats
teacher_log_std
split shard support
metadata 中记录 collector 类型、student obs dim、action dim 等
```

其中 `build_teacher_rollout_config()` 会显式把 teacher rollout config 里的 `student_obs_filter.enabled` 关掉，避免 teacher dataset collection 错误使用 student observation。

这点非常重要，因为 teacher 必须使用完整 lookahead observation：

```text
teacher input:
  state + full goal lookahead

student supervised target:
  teacher mean action / sampled teacher action
```

### 1.4 BC/KD trainer 已经进入可用状态

`train_bc()` 当前支持：

```text
DistillDataset
student env observation dimension validation
action MSE
optional value distillation
optional diagonal Gaussian KL distillation
PPO-compatible checkpoint writing
optional init_ckpt warm-start
distill_metadata.json
```

其中 `validate_dataset_matches_student_env()` 会实际 wrap student env 并比较：

```text
dataset.student_obs_dim == wrapped student env observation_space.shape[0]
```

这可以防止一个常见严重错误：dataset 是一种 observation layout，student policy 是另一种 observation layout，训练时维度不一致但难以定位。

### 1.5 DAgger collector 已经具备核心功能

当前 `collect_dagger_dataset()` 的核心逻辑已经正确：

```text
1. 用 student policy 访问环境状态；
2. 在 student-visited full observation 上调用 teacher；
3. 保存 student_obs -> teacher_mu；
4. 保存 student_action、rollout_action、used_teacher_action、teacher log-prob diagnostics；
5. 支持 mix_teacher_action_prob；
6. 支持 freeze_run_stats；
7. 支持 split 和 append。
```

这已经覆盖了 DAgger-style correction 的本质：

```text
off-policy teacher data 解决初始化；
on-policy student-visited data 解决闭环 distribution shift。
```

### 1.6 DAgger loop 已经支持 warm-start

当前 `dagger_loop.py` 的 plan 中，BC retrain 命令会传入：

```text
--init_ckpt current_student
```

因此每轮 DAgger 不再只是“聚合数据后从头训练”，而是可以从上一轮 student checkpoint warm-start。这是正确的工程方向。

### 1.7 evaluation compare 已经有基础报告

`distill_compare.py` / `eval_student.py` 会调用 `fullbody/eval.py --metrics --metrics_only`，并输出：

```text
comparison_metrics.json
comparison_table.csv
summary.md
```

报告中也包含一些初始验收阈值，例如：

```text
BC student return ratio before PPO fine-tune >= 0.70
BC+PPO return ratio after fine-tune >= 0.85
completion ratio >= 0.80 of teacher
early termination rate after PPO fine-tune <= teacher + 0.20
```

这些阈值可以作为第一版工程验收标准。

---

## 2. 仍然存在的隐藏错误或高风险点

### P0-1. teacher shard 和 DAgger shard 混合后可能导致 `DistillDataset` 失败

这是当前最重要的问题。

当前 `DistillDataset` 加载时会：

```python
for shard_path in self.shard_paths:
    with np.load(shard_path) as shard:
        for field in shard.files:
            loaded.setdefault(field, []).append(np.asarray(shard[field]))
self.arrays = {field: np.concatenate(parts, axis=0) for field, parts in loaded.items()}
self.num_samples = _validate_data(self.arrays)
```

`_validate_data()` 会要求所有字段第一维都等于 `student_obs` 的样本数。

但是当前 teacher collector 保存的字段大致是：

```text
student_obs
teacher_action
teacher_mu
teacher_log_std
teacher_value
teacher_log_prob
reward
done
absorbing
traj_no
subtraj_step_no
phase
optional full_obs
```

当前 DAgger collector 保存的字段更多：

```text
student_obs
teacher_action
teacher_mu
student_action
rollout_action
used_teacher_action
reward
done
absorbing
traj_no
subtraj_step_no
phase
teacher_value
teacher_log_prob
teacher_log_std
teacher_log_prob_teacher_mu
teacher_log_prob_student_action
teacher_log_prob_rollout_action
optional full_obs
```

如果先收集 teacher `train_000000.npz`，再 append DAgger `train_000001.npz`，那么加载所有 `train_*.npz` 时：

```text
student_obs: N_teacher + N_dagger
teacher_action: N_teacher + N_dagger
student_action: only N_dagger
rollout_action: only N_dagger
used_teacher_action: only N_dagger
teacher_log_prob_student_action: only N_dagger
...
```

这会导致 `_validate_data()` 报错，因为部分字段只有 DAgger shard 的长度。

#### 推荐修复

二选一即可。

**方案 A：统一 shard superset schema。**

让 teacher collector 也写入 DAgger 字段的 placeholder：

```text
student_action = teacher_action
rollout_action = teacher_action
used_teacher_action = True
teacher_log_prob_teacher_mu = teacher_log_prob
teacher_log_prob_student_action = teacher_log_prob
teacher_log_prob_rollout_action = teacher_log_prob
```

这样 teacher shard 和 DAgger shard 字段完全一致，后续 dataset concat 最简单。

**方案 B：让 `DistillDataset` 支持 missing optional fields。**

加载前先统计所有 shard 的字段 union，然后对缺失字段补齐：

```text
float field: NaN 或 0
bool field: False
int field: -1
```

这个方案更灵活，但需要严格维护 field dtype 和 shape。

工程上我更推荐 **方案 A**，因为蒸馏 dataset schema 更稳定，也更容易调试。

---

### P0-2. `inspect_distill_dataset` 也可能被混合 schema 卡住

当前 `inspect_distill_dataset()` 内部直接构造 `DistillDataset(dataset_path, split=split)`。如果出现 P0-1 的混合 schema 问题，inspect 本身也会失败，无法给出诊断。

#### 推荐修复

增加一个不依赖 `DistillDataset` 的 shard-level inspector：

```text
for each npz shard:
  print fields
  print shape per field
  print dtype per field
  check required fields
  warn missing optional fields
```

命令可以叫：

```bash
musclemimic-distill-inspect-dataset --dataset_dir <dir> --shard_level
```

---

### P0-3. Forehand wrapper 里仍有未使用参数

这些 wrapper 当前声明了一些参数，但没有实际传递到下游：

#### `collect_forehand_clear_teacher_dataset.py`

声明但基本未使用：

```text
--config-name
--motion-path
--wandb
```

它最终只是调用：

```text
fullbody/distill_collect.py --teacher_ckpt ...
```

而 generic collector 使用 checkpoint 里的 config 创建环境，不会使用 wrapper 的 `--config-name` 或 `--motion-path`。

#### `train_forehand_clear_student_bc.py`

声明但未使用：

```text
--resume-student
--wandb
```

虽然 generic `fullbody/distill_train_bc.py` 已经支持 `--init_ckpt`，但 wrapper 没有把 `--resume-student` 映射过去。

#### `evaluate_forehand_clear_student.py`

声明但未使用：

```text
--config-name
```

这个可以接受，因为 eval 主要依赖 checkpoint config，但最好不要保留误导参数。

#### 推荐修复

要么删除这些参数，要么真正实现：

```text
--resume-student -> fullbody/distill_train_bc.py --init_ckpt
--motion-path -> generic collector config override
--config-name -> generic collector 或 explicit config loading
--wandb -> 设置 wandb.mode 或删除
```

---

### P0-4. 部分 Forehand wrapper 仍使用 repo-relative script 路径

`run_forehand_clear_dagger_loop.py` 已经改成：

```text
python -m fullbody.distill_run_dagger
```

这是更稳的方式。

但其他 Forehand wrapper 仍然调用类似：

```text
python fullbody/distill_collect.py
python fullbody/distill_train_bc.py
python fullbody/distill_collect_dagger.py
python fullbody/distill_compare.py
```

如果用户从 repository root 运行没问题；如果是安装后的 console script，在其他 CWD 运行，就可能找不到 `fullbody/distill_collect.py`。

#### 推荐修复

全部统一成：

```text
python -m fullbody.distill_collect
python -m fullbody.distill_train_bc
python -m fullbody.distill_collect_dagger
python -m fullbody.distill_compare
```

或者直接在 wrapper 中 import 相应 `main()`，但 `python -m` 改动最小。

---

### P1-1. collection 没有真正支持 `motion_path` override

完整 ForehandClear distillation 最好能分别采集：

```text
train split
validation split
specific trajectory
specific start step
```

当前 collector 基本依赖 checkpoint config 中的 motion dataset。这样如果 teacher checkpoint config 指向 train motions，那么即使 wrapper 参数有 `--motion-path`，实际仍会采集 train motions。

#### 推荐修复

在 generic `fullbody/distill_collect.py` 和 `fullbody/distill_collect_dagger.py` 中添加：

```text
--motion_path
--motion_group
--traj_index
--traj_start_step
```

并复用 `fullbody/eval.py` 中已有的 trajectory override 逻辑。

---

### P1-2. DAgger loop 训练数据集会持续聚合，但缺少 dataset manifest 的 iteration 信息

当前 DAgger loop 会写：

```text
dagger_loop_manifest.json
dagger_loop_results.json
```

但是 dataset 的 `metadata.json` 不一定清楚记录：

```text
which shard belongs to teacher initialization
which shard belongs to dagger iteration 0
which shard belongs to dagger iteration 1
which student checkpoint generated each shard
```

#### 推荐修复

每轮 DAgger collector 的 metadata 中加：

```text
dagger_iteration
student_ckpt_in
teacher_ckpt
rollout_policy
collector = dagger_student_rollout_teacher_relabel
```

这样后续如果某一轮数据污染，可以定位。

---

### P1-3. 还缺少 BC checkpoint roundtrip test

你现在需要一个测试证明：

```text
train_bc 保存的 checkpoint
  -> load_checkpoint()
  -> PPOJax.init_agent_conf()
  -> policy forward
  -> fullbody/eval.py 或 compare 脚本可加载
```

否则 BC 训练看起来成功，但后续 DAgger / PPO / eval 可能在 checkpoint metadata 或 config 对齐处失败。

#### 推荐修复

新增：

```text
tests/unit/test_bc_checkpoint_roundtrip.py
```

最低限度可以 mock 小网络或用小 env；若真实 MyoFullBody 太重，则做 integration marker。

---

### P1-4. `run_eval_metrics()` 依赖 stdout 正则解析，可能漏掉关键指标

`eval_student.py` 通过正则解析 `fullbody/eval.py` stdout：

```python
METRIC_RE = re.compile(...)
```

如果 eval 输出改格式，或者某些 metrics 只在 wandb/log dict 中没有打印，compare 报告可能缺字段。

#### 推荐修复

优先让 `fullbody/eval.py --metrics_only --output_json <path>` 直接输出机器可读 JSON。
`distill_compare.py` 读取 JSON，不要依赖 stdout。

---

### P1-5. 缺少正式流程文档

现在命令已经基本有了，但还缺少一个工程级 README，例如：

```text
docs/forehand_clear_distillation.md
```

里面应包含：

```text
0. 需要的 teacher checkpoint
1. inspect student observation
2. collect teacher train/val dataset
3. inspect dataset
4. BC train
5. evaluate BC student
6. collect DAgger
7. run DAgger loop
8. PPO fine-tune
9. final compare
10. common failures
```

---

## 3. 现在离“可以进行完整 distillation”还差什么？

按照可实施工程标准，我建议分成三步：

---

### Step A：必须先修 P0

在正式跑完整 distillation 前，至少完成：

```text
A1. 解决 teacher shards + DAgger shards 混合 schema 问题
A2. wrapper 统一使用 python -m
A3. 删除或实现 wrapper 中未使用参数
A4. 确认所有 console entrypoints 可以 --help
```

其中 **A1 是最关键的阻塞点**。不修 A1，DAgger loop 可能第一轮 collection 成功，但随后的 BC retrain 在加载聚合数据集时失败。

---

### Step B：跑通 smoke test

完成 P0 后，按下面顺序跑最小链路：

```bash
forehand-clear-distill-inspect-obs \
  --config-name config_specific_task/conf_fullbody_badminton_student_gmr

forehand-clear-distill-collect-teacher \
  --teacher-path <TEACHER_CKPT> \
  --output-dir outputs/distill_smoke/dataset \
  --num-envs 2 \
  --num-steps 20 \
  --shard-size 20 \
  --split train

musclemimic-distill-inspect-dataset \
  --dataset_dir outputs/distill_smoke/dataset

forehand-clear-distill-train-bc \
  --dataset-dir outputs/distill_smoke/dataset \
  --output-dir outputs/distill_smoke/bc \
  --num-steps 5 \
  --batch-size 4

forehand-clear-distill-collect-dagger \
  --teacher-path <TEACHER_CKPT> \
  --student-path outputs/distill_smoke/bc/checkpoints/checkpoint_5 \
  --output-dir outputs/distill_smoke/dataset \
  --num-envs 2 \
  --num-steps 20 \
  --shard-size 20 \
  --split train \
  --append

forehand-clear-distill-run-dagger \
  --teacher-path <TEACHER_CKPT> \
  --student-path outputs/distill_smoke/bc/checkpoints/checkpoint_5 \
  --config-name config_specific_task/conf_fullbody_badminton_student_gmr \
  --dataset-dir outputs/distill_smoke/dataset \
  --output-dir outputs/distill_smoke/dagger \
  --num-iters 1 \
  --num-envs 2 \
  --num-steps 20 \
  --train-steps 5 \
  --batch-size 4

forehand-clear-distill-evaluate \
  --teacher-path <TEACHER_CKPT> \
  --student-path outputs/distill_smoke/bc/checkpoints/checkpoint_5 \
  --dagger-student-path outputs/distill_smoke/dagger/iter_000/checkpoints/checkpoint_5 \
  --output-dir outputs/distill_smoke/eval \
  --num-envs 2 \
  --num-steps 20
```

验收标准：

```text
每个命令能正常退出
dataset metadata 存在
BC checkpoint 存在
DAgger loop manifest 存在
evaluation summary.md 存在
无 NaN / Inf
student_obs_dim 与 env 一致
```

---

### Step C：正式小规模实验

Smoke test 后再跑：

```text
teacher train dataset:
  num_envs=256
  num_steps=100k~300k

BC:
  num_steps=50k~200k
  batch_size=4096

DAgger:
  num_iters=2~3
  num_steps=50k~100k per iter
  mix_teacher_action_prob=0.05~0.2

PPO fine-tune:
  resume_from=<best BC/DAgger student checkpoint>
  config=conf_fullbody_badminton_student_gmr
  total_timesteps=20M~100M
```

这才算进入“完整 distillation 实验”。

---

## 4. 你的 distillation 方法详细介绍

你的方法可以命名为：

> Trajectory-Conditioned Teacher to Phase-Conditioned Student Distillation

或者更具体：

> ForehandClear Lookahead-Teacher to Phase-Student Distillation with DAgger and PPO Fine-tuning

整体目标是：

```text
把依赖 GoalTrajMimic future lookahead 的 teacher policy，
蒸馏成一个不需要 future lookahead、只依赖当前身体状态和 motion phase 的 student policy。
```

---

### 4.1 Teacher policy

Teacher 是当前已有的 PPO imitation policy：

```text
teacher input:
  joint state
  muscle state
  foot contact
  goal lookahead from reference trajectory

teacher output:
  354-D muscle controls
```

Teacher 的优势是它看得到未来目标，因此非常适合做稳定的 trajectory tracking。
Teacher 的缺点是它依赖 reference lookahead，不能算真正“内化”的运动技能。

数学上：

```text
a_t^T ~ pi_T(a_t | o_t^T)

o_t^T = [x_t, g_t^{lookahead}]
```

其中：

```text
x_t = joint + muscle + foot contact
g_t^{lookahead} = reference future trajectory goal
```

---

### 4.2 Student observation

Student 的输入被裁剪为：

```text
o_t^S = [x_t, phase_t]
```

也就是：

```text
joint state
muscle state
foot contact
motion phase
```

它不再看到：

```text
future root delta
future site rpos
future target trajectory sequence
```

这一步由 `StudentObservationFilterWrapper` 完成。

这里保留 `phase_t` 是合理的，因为 ForehandClear 是长时序动作。如果完全不提供 phase，同样的局部身体状态可能对应引拍、挥拍、随挥等不同阶段，student 容易相位混乱。

---

### 4.3 Off-policy teacher dataset collection

第一阶段让 teacher 在原始 lookahead 环境中 rollout，收集：

```text
student_obs_t = filter(o_t^T)
teacher_action_t = mean(pi_T(. | o_t^T))
teacher_mu_t
teacher_log_std_t
teacher_value_t
reward_t
done_t
traj_no
subtraj_step_no
phase_t
```

形成 supervised dataset：

```text
D_0 = {(o_t^S, a_t^T)}
```

这一阶段是 off-policy distillation，因为 student 还没有参与环境。

---

### 4.4 BC/KD student training

用 dataset 训练 student policy：

```text
pi_S(a_t | o_t^S)
```

最基础 loss 是 action MSE：

```text
L_BC = || mean(pi_S(.|o_t^S)) - a_t^T ||^2
```

当前实现还支持：

```text
value distillation:
  || V_S(o_t^S) - V_T(o_t^T) ||^2

Gaussian KL distillation:
  KL( N_T(mu_T, sigma_T) || N_S(mu_S, sigma_S) )
```

因此总 loss 可以理解为：

```text
L = L_action_mse
  + lambda_v L_value
  + lambda_kl KL(pi_T || pi_S)
```

BC 阶段的作用是快速获得一个可用的 no-lookahead student 初始化。

---

### 4.5 DAgger-style on-policy correction

纯 off-policy BC 有 distribution shift 问题：

```text
BC 训练数据来自 teacher states
部署时 student 看到的是 student 自己造成的 states
```

因此第二阶段让 student 自己 rollout：

```text
student action -> simulation -> student-visited state
```

然后在这些 student-visited states 上，让 teacher 重新给 label：

```text
teacher_mu = mean(pi_T(. | full_obs_student_visited))
```

生成 DAgger correction dataset：

```text
D_1 = {(filter(full_obs_visited_by_student), teacher_mu)}
```

再把它 append 到原 dataset 中继续训练 student。

这是解决 covariate shift 的关键步骤。

---

### 4.6 Student PPO fine-tuning

BC/DAgger 之后，student 仍然只是模仿 teacher action，不一定最大化 long-horizon imitation reward。因此最后用 PPO fine-tune：

```text
policy input:
  state + phase

reward:
  MimicReward(sim state, reference trajectory)
```

注意这里非常重要：

```text
student policy 不看 future lookahead
但 reward 仍然可以使用 reference trajectory
```

这没有矛盾。policy 输入限制的是部署时能看到什么；reward 是训练信号，可以使用 reference 做监督。

PPO fine-tune 的作用是让 student 不仅在 one-step action 上接近 teacher，而且在 closed-loop rollout 中真正完成 ForehandClear body trajectory。

---

### 4.7 Evaluation

最终评估比较：

```text
teacher
student_bc
student_bc_dagger
student_bc_dagger_ppo
```

指标包括：

```text
mean_episode_return
completion_rate
early_termination_rate
mean_episode_length
err_root_xyz
err_joint_pos
err_joint_vel
err_site_abs
err_rpos
reward subterms
```

这能回答：

```text
student 去掉 future lookahead 后，动作还能完成多少？
DAgger 是否改善闭环稳定性？
PPO fine-tune 是否进一步提高 tracking?
```

---

## 5. 这个方法的好处

### 5.1 避免 future lookahead 依赖

Teacher 是一个 tracking controller，依赖未来 reference。
Student 被限制为 `state + phase`，因此更像一个“内化”的 motor skill。

这对后续研究更有意义：

```text
teacher: follow this future trajectory
student: given current body state and phase, produce ForehandClear motor command
```

---

### 5.2 训练难度分解合理

直接训练 no-lookahead student PPO 很难，因为肌肉控制长时序、延迟大、动作相位复杂。

你的分阶段方法更稳：

```text
lookahead teacher PPO
  -> off-policy BC 初始化
  -> DAgger 修正 distribution shift
  -> PPO fine-tune 闭环优化
```

每一步解决一个具体问题。

---

### 5.3 DAgger 弥补 BC 的闭环漂移

纯 BC 容易出现：

```text
teacher states 上 action 很准
student rollout 时逐渐偏离
```

DAgger 用 student-visited states 重新让 teacher label，可以显著减少这个问题。

---

### 5.4 保留 phase 而不是完全无条件，工程上更稳

完全去掉 lookahead 和 phase 会造成相位不确定性。
保留 phase 是合理折中：

```text
不看未来轨迹
但知道当前处于 ForehandClear 的哪个阶段
```

这更适合第一版 student。

---

### 5.5 PPO-compatible checkpoint 设计正确

BC 输出的是 PPO-compatible checkpoint，后续可以直接接入：

```text
fullbody/eval.py
DAgger collector
PPO fine-tune
compare evaluation
```

这避免了“BC model 和 RL/eval 系统割裂”的问题。

---

## 6. 这个方法的不足

### 6.1 Student 仍然依赖 motion phase

Student 不看 future lookahead，但仍然需要 `phase_t`。
因此它还不是完全 autonomous policy。

如果未来要做到完全自主，需要进一步研究：

```text
phase-free student
phase predictor
latent skill state
recurrent policy
task-conditioned policy
```

---

### 6.2 Teacher 在 student OOD states 上的 label 不一定最优

DAgger 假设 teacher 可以在 student-visited states 上提供好动作。
但如果 student 进入很差的状态，例如重心大幅偏离、姿态崩坏，teacher 虽有 lookahead，也不一定能救回来。

这时 teacher label 可能只是“在坏状态下的局部补救动作”，并不一定是全局最优。

---

### 6.3 BC/KD 优化的是 action imitation，不直接优化长期 reward

BC loss 低不代表 rollout 一定稳定。
动作误差很小也可能累积成姿态偏差。

因此 PPO fine-tune 不是可选装饰，而是最终闭环能力的重要阶段。

---

### 6.4 当前方法仍然依赖 reference trajectory 作为 reward

即使 student 输入不含 future lookahead，PPO fine-tune 的 reward 仍然使用 reference trajectory。
这对“动作轨迹完成”任务是合理的，但如果未来转向真实击球任务，还需要引入：

```text
task goal
racket state
shuttle state
shot outcome reward
```

---

### 6.5 数据 schema 稳定性目前还需要修

如上面 P0 所述，teacher shard 和 DAgger shard 混合时字段不一致。
这是当前从“可 smoke-test”进入“完整 distillation”的主要工程阻塞点。

---

## 7. 推荐的最终 Definition of Done

我建议把下面作为“完整 distillation 工程”的完成标准：

```text
给定一个 trained lookahead teacher checkpoint，
用户可以按文档运行一组命令，
自动完成：

1. inspect student observation
2. collect teacher train/val distillation dataset
3. inspect dataset and schema
4. train BC/KD student
5. evaluate BC student
6. collect DAgger correction dataset
7. run DAgger loop
8. PPO fine-tune no-lookahead student
9. compare teacher / BC / DAgger / PPO student
10. output json/csv/markdown report

并且：
- 所有 command line entrypoints 能 --help；
- 所有 shard schema 一致；
- student checkpoint 能被 fullbody/eval.py 加载；
- final student 不输入 future lookahead；
- evaluation summary 中给出 return、completion、early termination、tracking errors。
```

---

## 8. 下一步最小修改清单

### 必做

1. **修复混合 shard schema 问题。**
   - 推荐让 teacher collector 写入 DAgger superset fields。
2. **把所有 Forehand wrapper 的 subprocess 调用改成 `python -m ...`。**
3. **删除或实现 wrapper 中未使用参数。**
   - `--motion-path`
   - `--config-name`
   - `--wandb`
   - `--resume-student`
4. **给 `inspect_dataset` 增加 shard-level mode，避免 DistillDataset schema 失败时无法诊断。**
5. **跑通最小 smoke test。**

### 建议做

6. 添加 `test_bc_checkpoint_roundtrip.py`。
7. 添加 `test_eval_metrics_parser.py`。
8. 添加 `docs/forehand_clear_distillation.md`。
9. 支持 `--motion_path` / `--motion_group` override，方便 train/val 数据分开采集。
10. 让 `distill_compare.py` 优先读取 eval JSON，而不是依赖 stdout regex。

---

## 9. 当前是否可以开始？

我的判断：

```text
可以开始：小规模 smoke-test
不建议开始：大规模正式 distillation
```

如果先修 P0-1，然后 smoke test 全过，就可以开始正式的第一轮：

```text
teacher dataset collection
BC student training
BC evaluation
DAgger correction
student PPO fine-tune
final comparison
```

如果不修 P0-1，最可能出现的问题是：

```text
teacher dataset collection 成功
BC 成功
DAgger collection 成功
DAgger retrain 失败：字段长度不一致
```

因此，当前最重要的不是再增加新算法，而是把数据 schema 和 end-to-end smoke test 打牢。

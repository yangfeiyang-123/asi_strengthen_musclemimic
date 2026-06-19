# 06 Codex Implementation Tasks

下面按 milestone 拆分，适合逐个 PR 实施。每个 milestone 都应包含测试。

---

## Milestone 1: Student Observation Filter

### 新增文件

```text
musclemimic/distill/__init__.py
musclemimic/distill/obs_filter.py
tests/unit/test_student_obs_filter.py
```

### 修改文件

```text
musclemimic/algorithms/common/env_utils.py
```

### 功能

实现：

```python
StudentObservationFilterWrapper
StudentObsSpec
build_student_obs_indices(env, config)
filter_student_obs(obs, spec)
```

默认行为：

```text
keep non-goal observations
keep goal[-1] as motion phase
drop all other goal lookahead values
```

### 集成

在 `wrap_env()` 中，在 `NStepWrapper` 前应用：

```python
if config.get("student_obs_filter", {}).get("enabled", False):
    env = StudentObservationFilterWrapper(env, config.student_obs_filter)
```

### 测试

```bash
uv run pytest tests/unit/test_student_obs_filter.py
uv run pytest tests/unit/test_split_goal_integration.py
```

### 验收

```text
student filter 输出维度正确；
phase 保留；
goal lookahead 删除；
与 NStepWrapper(split_goal=True) 组合正常。
```

---

## Milestone 2: Distillation Dataset Loader/Writer

### 新增文件

```text
musclemimic/distill/dataset.py
tests/unit/test_distill_dataset.py
```

### 功能

实现：

```python
write_distill_shard(path, data, metadata)
DistillDataset(dataset_dir, split="train")
DistillDataset.iter_batches(batch_size, shuffle=True)
load_metadata(dataset_dir)
```

支持字段：

```text
student_obs
teacher_action
teacher_value
teacher_log_prob
reward
done
absorbing
traj_no
subtraj_step_no
phase
```

### 验收

```text
可以写入多个 shard；
可以跨 shard 迭代 batch；
metadata 与数组 shape 一致。
```

---

## Milestone 3: Teacher Rollout Collector

### 新增文件

```text
fullbody/distill_collect.py
musclemimic/distill/collect_teacher.py
```

### 依赖现有能力

复用：

```text
fullbody/eval.py 的 checkpoint load / env instantiate 思路
PPO policy apply
StudentObsSpec / filter_student_obs
```

### 功能

命令：

```bash
uv run python fullbody/distill_collect.py \
  --teacher_ckpt /path/to/checkpoint \
  --output_dir datasets/distill/forehandclear_v1 \
  --num_envs 256 \
  --num_steps 200000 \
  --deterministic_teacher
```

采集：

```text
teacher full obs -> teacher pi/value -> teacher_action
full obs -> student_obs via StudentObsSpec
env.step -> reward/done/traj_state/info
write shard
```

### 验收

```text
可采集至少 2 个 shard；
每个 shard student_obs/action/reward 长度一致；
phase 范围正确；
metadata 记录 teacher checkpoint 和 obs filter。
```

---

## Milestone 4: BC Trainer

### 新增文件

```text
fullbody/distill_train_bc.py
musclemimic/distill/train_bc.py
musclemimic/distill/losses.py
musclemimic/distill/checkpoint.py
tests/unit/test_bc_loss.py
```

### 新增配置

```text
fullbody/config_specific_task/distill/conf_fullbody_forehandclear_student_phase_bc.yaml
```

### 功能

训练：

```text
student_obs -> student actor mean
loss = action MSE + optional value distill
```

保存：

```text
PPO-compatible checkpoint
student config
run_stats
metadata
```

### 验收

```text
toy batch loss 下降；
训练 100 steps 不报错；
checkpoint 可 load；
fullbody/eval.py 能用 student checkpoint 创建 env 和 policy。
```

---

## Milestone 5: Student PPO Fine-tune Config

### 新增配置

```text
fullbody/config_specific_task/distill/conf_fullbody_forehandclear_student_phase_ppo.yaml
```

### 修改/验证

确认：

```text
resume_from BC checkpoint
student_obs_filter.enabled = true
reward_type 仍为 MimicReward
goal trajectory 仍加载
policy obs 不含 future lookahead
```

### 命令

```bash
uv run fullbody/experiment.py \
  --config-name=config_specific_task/distill/conf_fullbody_forehandclear_student_phase_ppo \
  experiment.resume_from=/path/to/student_bc_checkpoint
```

### 验收

```text
能够跑过若干 PPO updates；
reward info 正常；
checkpoint 正常保存；
student eval 正常。
```

---

## Milestone 6: Evaluation Scripts / Reports

### 新增文件

```text
fullbody/distill_compare.py
musclemimic/distill/eval_student.py
```

### 功能

输入 teacher ckpt 和 student ckpt，输出：

```text
metrics JSON
CSV summary
可选 videos
```

比较：

```text
teacher lookahead PPO
student BC
student BC+PPO
```

### 验收

```text
生成 comparison_metrics.json
生成 comparison_table.csv
打印主要指标
```

---

## Milestone 7: Documentation

### 新增文档

```text
docs/distillation/forehandclear_student_policy.md
docs/distillation/commands.md
docs/distillation/results_template.md
```

内容：

```text
方法说明
命令
配置说明
评估表格模板
常见错误排查
```

---

## Suggested PR Order

```text
PR1: StudentObservationFilterWrapper + tests
PR2: DistillDataset + tests
PR3: Teacher rollout collector
PR4: BC trainer + checkpoint compatibility
PR5: Student PPO fine-tune config + smoke test
PR6: Evaluation comparison scripts
PR7: Documentation and command examples
```

---

## Non-goals for this phase

不要做：

```text
racket model
shuttle dynamics
impact reward
shot outcome reward
residual racket policy
privileged critic
DAgger loop
fully phase-free student
```

这些可以作为后续阶段。

# 05 Evaluation 与通过标准

## 5.1 目标

比较：

```text
Teacher policy with full goal lookahead
vs
Student policy with body state + foot contact + phase
```

评估范围只包括：

```text
ForehandClear body trajectory imitation
```

不评估球拍/羽毛球。

---

## 5.2 必做评估

### A. Tracking metrics

使用现有 validation metrics / reward info，比较：

```text
mean_episode_return
mean_episode_length
early_termination_rate
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
reward_rvel_rot
reward_rvel_lin
```

### B. Completion metrics

```text
trajectory completion rate
average completed phase
episode length distribution
fall / early termination count
```

### C. Action behavior

```text
action_mse(student, teacher) on held-out teacher rollouts
action_abs_mean
action_rate / smoothness
muscle activation energy
```

### D. Robustness

```text
random starts
start from beginning
different eval_seed
small initial pose perturbations optional
```

---

## 5.3 命令模板

Teacher metrics：

```bash
uv run python fullbody/eval.py \
  --path /path/to/teacher/checkpoint \
  --metrics --metrics_only \
  --motion_path badminton/train/forehand_clear_clip1_merged_poses \
  --metrics_envs 20 \
  --metrics_steps 500 \
  --eval_seed 0
```

Student metrics：

```bash
uv run python fullbody/eval.py \
  --path /path/to/student/checkpoint \
  --metrics --metrics_only \
  --motion_path badminton/train/forehand_clear_clip1_merged_poses \
  --metrics_envs 20 \
  --metrics_steps 500 \
  --eval_seed 0
```

Evaluate all three clips：

```bash
uv run python fullbody/eval.py \
  --path /path/to/student/checkpoint \
  --metrics --metrics_only \
  --motion_path \
    badminton/train/forehand_clear_clip1_merged_poses \
    badminton/train/forehand_clear_clip2_merged_poses \
    badminton/train/forehand_clear_clip3_merged_poses \
  --metrics_envs 20 \
  --metrics_steps 1000 \
  --eval_seed 0
```

---

## 5.4 关键消融实验

### Ablation 1: teacher with lookahead vs student with phase

目的：证明 student 不依赖 future lookahead 仍能完成动作。

```text
Teacher: joint + muscle + foot contact + full lookahead
Student: joint + muscle + foot contact + phase
```

### Ablation 2: student without phase

可选，不作为第一版目标。

```text
Student-no-phase: joint + muscle + foot contact
```

预期：可能相位混淆、completion 降低。这个实验可以证明 phase 的必要性。

### Ablation 3: BC only vs BC + PPO fine-tune

```text
Student-BC
Student-BC-PPO
```

预期：PPO fine-tune 改善 rollout distribution shift，提高 completion 和 reward。

### Ablation 4: deterministic teacher action vs sampled teacher action

```text
BC target = teacher mean
BC target = sampled teacher action
```

预期：teacher mean 更稳定。

---

## 5.5 推荐通过标准

第一阶段 BC student 通过标准：

```text
held-out action_mse 明显低于随机策略 baseline；
rollout 能完成至少部分 ForehandClear 动作；
没有维度/加载/推理错误；
student checkpoint 可被 fullbody/eval.py 加载。
```

第二阶段 PPO fine-tuned student 通过标准：

```text
early_termination_rate <= teacher 的 1.5x，或绝对值低于 20%；
mean_episode_return >= teacher 的 70% 起步，目标 85%+；
err_rpos <= teacher 的 1.5x，目标 1.2x；
能从 start_from_beginning 完成完整 ForehandClear trajectory；
禁用/置零 future lookahead 不影响 student policy，因为它根本不使用该输入。
```

最终论文/报告级目标：

```text
student 不看 future lookahead，仅凭 body state + phase 完成轨迹模仿，tracking performance 接近 teacher。
```

---

## 5.6 可视化输出

建议保存：

```text
teacher rollout video
student BC rollout video
student PPO fine-tuned rollout video
reference ghost overlay video
tracking error curves
phase vs time curves
action magnitude curves
```

---

## 5.7 报告表格模板

| Policy | Input | Return ↑ | Early Term ↓ | err_rpos ↓ | err_joint_pos ↓ | Completion ↑ |
|---|---|---:|---:|---:|---:|---:|
| Teacher PPO | state + lookahead | | | | | |
| Student BC | state + phase | | | | | |
| Student BC+PPO | state + phase | | | | | |
| Student no phase | state only | | | | | |

---

## 5.8 Debug checklist

如果 student 表现差，按顺序检查：

1. `student_obs_dim` 是否与 checkpoint network 输入一致。
2. phase 是否正确保留，范围是否 `[0,1]`。
3. teacher_action 是 mean 还是 sample。
4. action scale 是否仍为 `[-1,1]` normalized controls。
5. BC checkpoint 的 RunningMeanStd 是否正确保存。
6. PPO fine-tune 是否真的使用 student filter。
7. reward 是否仍使用 reference trajectory。
8. evaluation 是否加载了 student config，而不是 teacher config。
9. `done` 是否过早由 strict termination 触发。
10. random_start 是否导致 student 没见过的 phase/state 组合过多。

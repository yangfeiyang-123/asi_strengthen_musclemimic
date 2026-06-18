# ForehandClear MyoFullBody PPO 蒸馏实施文档索引

这些文档面向 Codex 实施，目标是在**不考虑球拍/羽毛球击球**的前提下，把当前依赖 `goal lookahead` 的 ForehandClear trajectory-conditioned teacher policy，蒸馏成一个更少依赖参考轨迹输入的 student body policy。

推荐第一版 student：

```text
student input = joint state + muscle state + foot contact + motion phase
student output = 354-D normalized muscle controls
```

也就是说：**去掉未来 reference lookahead，但保留一个 phase 标量**。phase 不是未来轨迹目标，而是动作进度，用来避免正手高远球这种长时序动作发生相位混淆。

文档顺序：

1. [`00_current_system_grounding.md`](00_current_system_grounding.md)  
   说明仓库当前 PPO/环境/奖励/观测/评估机制，以及为什么 teacher policy 依赖 lookahead。

2. [`01_student_policy_design.md`](01_student_policy_design.md)  
   定义 teacher/student 观测、推荐 student 版本、observation filter/wrapper 设计。

3. [`02_teacher_rollout_dataset.md`](02_teacher_rollout_dataset.md)  
   设计 teacher rollout 数据采集器，定义 `.npz` shard 数据格式、采样策略、质量过滤。

4. [`03_behavior_cloning_trainer.md`](03_behavior_cloning_trainer.md)  
   设计 offline behavior cloning / policy distillation trainer，包含 loss、checkpoint 兼容性、配置和测试。

5. [`04_student_ppo_finetune.md`](04_student_ppo_finetune.md)  
   设计 student PPO fine-tuning：policy 输入无 future lookahead，但 reward 仍用 reference trajectory。

6. [`05_evaluation_and_acceptance.md`](05_evaluation_and_acceptance.md)  
   定义 teacher/student 对比评估、指标、命令、通过标准和消融实验。

7. [`06_codex_implementation_tasks.md`](06_codex_implementation_tasks.md)  
   按 PR / milestone 拆分的实施任务清单，适合直接交给 Codex 执行。

总体实施路线：

```text
1. 稳定训练/选择 lookahead teacher checkpoint
2. 实现 StudentObservationFilterWrapper
3. 用 teacher rollout 采集 student_obs -> teacher_action 数据集
4. 用 BC/KD 训练 student checkpoint
5. 用 student checkpoint 初始化 PPO，去掉 future lookahead 后 fine-tune
6. 用 validation metrics、trajectory tracking errors、early termination rate 比较 teacher/student
```

重要边界：

```text
当前蒸馏只针对 body trajectory imitation。
不引入球拍、不引入羽毛球、不引入击球接触、不引入落点 reward。
```

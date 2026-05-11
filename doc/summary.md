# MuscleMimic 项目结构总结

生成时间：2026-04-28

## 1. 项目总体定位

`musclemimic` 是一个基于 JAX + MuJoCo/MJX 的肌肉骨骼模仿学习项目。它围绕两类肌肉驱动模型展开：

| 模型 | 主要入口 | 目的 |
| --- | --- | --- |
| `MyoFullBody` / `MjxMyoFullBody` | `fullbody/experiment.py`, `fullbody/eval.py` | 全身肌肉骨骼模型的运动模仿、训练、评估和可视化 |
| `MyoBimanualArm` / `MjxMyoBimanualArm` | `bimanual/experiment.py`, `bimanual/eval.py` | 双臂肌肉骨骼模型的操作/上肢动作模仿训练与评估 |

核心流程可以概括为：

```text
AMASS/SMPL 动作数据
  -> retarget 到肌肉骨骼 MuJoCo 模型
  -> 构造 imitation 环境和目标观测
  -> PPO/JAX 大规模并行训练
  -> checkpoint 保存与自动恢复
  -> 评估、指标统计、视频/Viewer 可视化
```

## 2. 顶层文件和目录

| 路径 | 类型 | 作用 |
| --- | --- | --- |
| `README.md` | 文档 | 项目主说明，包含功能介绍、安装、demo cache、GMR/AMASS retarget、训练和评估命令 |
| `TUTORIAL.md` | 文档 | 中文上手流程，解释环境部署、demo 跑通、数据/配置/checkpoint/log 位置和研究开发建议 |
| `CONTRIBUTING.md` | 文档 | 开发规范，说明 `uv`、`make`、lint/test、pre-commit 和 PR 要求 |
| `LICENSE` | 许可证 | Apache-2.0 项目许可证 |
| `pyproject.toml` | Python 项目配置 | 包元数据、依赖、optional extras、命令行入口、ruff/pytest/setuptools 配置 |
| `uv.lock` | 依赖锁文件 | `uv` 解析出的精确依赖版本 |
| `Makefile` | 开发命令 | `make install`, `make lint`, `make test`, `make smoke`, `make ci`, `make clean` 等 |
| `.python-version` | 环境文件 | 指定 Python 版本 |
| `.gitignore` | Git 配置 | 忽略缓存、构建产物、训练输出等 |
| `.ruffignore` | Ruff 配置 | Ruff 忽略路径 |
| `.pre-commit-config.yaml` | 开发配置 | pre-commit hooks，当前只覆盖一组迁移中的文件 |
| `.codex` | 本地文件 | Codex/本地代理状态或配置文件 |
| `tea_debug.log` | 日志 | 本地调试日志，不属于核心源码 |

## 3. 主要运行入口

| 路径 | 目的 |
| --- | --- |
| `fullbody/experiment.py` | 全身模型训练入口。通过 Hydra 加载 `fullbody/conf_*.yaml`，调用 `musclemimic.runner.engine.run_experiment` |
| `bimanual/experiment.py` | 双臂模型训练入口。通过 Hydra 加载 `bimanual/conf_*.yaml` |
| `fullbody/eval.py` | 全身模型评估入口，支持 MJX/MuJoCo、checkpoint/Hugging Face、viewer、视频、指标评估和轨迹导出 |
| `bimanual/eval.py` | 双臂模型评估入口，功能与 fullbody eval 类似但面向双臂环境 |
| `scripts/retarget_dataset.py` | 批量 retarget AMASS 动作数据到目标肌肉骨骼环境 |
| `scripts/run_with_cuda_compat.sh` | 在私有 CUDA compat 环境下运行命令，解决部分服务器 CUDA/driver 兼容问题 |

## 4. `fullbody/`

全身模型的训练、评估和 Hydra 配置目录。

| 文件/目录 | 作用 |
| --- | --- |
| `experiment.py` | 全身训练入口，加载配置后进入统一 runner |
| `eval.py` | 全身策略评估、可视化、录制和指标计算入口 |
| `_eval_terminal.py` | eval 时 terminal handler 默认值和 CLI 覆盖逻辑 |
| `conf_fullbody.yaml` | 全身基础训练配置，定义环境、目标观测、奖励、terminal、PPO、checkpoint、validation 等 |
| `conf_fullbody_gmr.yaml` | 基于 `conf_fullbody.yaml` 的 GMR retarget 配置，训练/验证使用 GMR cache 或 GMR retarget 参数 |
| `conf_fullbody_demo.yaml` | 小规模 demo 配置，使用少量 KIT 轨迹和更小并行环境数，适合环境验收 |
| `conf_fullbody_moe.yaml` | 开启 Soft Mixture of Experts 的全身配置 |
| `conf_fullbody_gmr_resnet.yaml` | GMR + residual network baseline 配置 |
| `config_specific_task/conf_fullbody_turnleft_gmr.yaml` | turn-left 专项实验配置，显式列出训练/验证轨迹 |
| `__pycache__/` | Python 字节码缓存，可删除 |

## 5. `bimanual/`

双臂模型的训练、评估和 Hydra 配置目录。

| 文件/目录 | 作用 |
| --- | --- |
| `experiment.py` | 双臂训练入口，加载配置后进入统一 runner |
| `eval.py` | 双臂策略评估、可视化和指标计算入口 |
| `conf_bimanual.yaml` | 双臂基础训练配置，定义 `MjxMyoBimanualArm`、目标观测、奖励、terminal、PPO 等 |
| `conf_bimanual_gmr.yaml` | 双臂 GMR retarget 配置，使用 AMASS bimanual 数据组 |
| `conf_bimanual_demo.yaml` | 双臂 demo 配置，使用少量抛接、挥手、网球等动作 |
| `conf_bimanual_moe.yaml` | 双臂 Soft Mixture of Experts 配置 |
| `__pycache__/` | Python 字节码缓存，可删除 |

## 6. `musclemimic/`

项目自己的核心包，包含算法、环境、MJX wrapper、reward、goal、runner、工具和 viewer。

| 文件/目录 | 作用 |
| --- | --- |
| `__init__.py` | 包初始化和延迟导出 |
| `utils.py` | 顶层 utility 导出，主要用于路径/cache 配置入口 |
| `algorithms/` | JAX RL 算法实现，目前核心是 PPO |
| `core/` | MuscleMimic 对 MuJoCo/MJX、goal、reward、terminal、wrapper 的扩展 |
| `environments/` | MyoFullBody 和 MyoBimanualArm 环境定义 |
| `rl_core/` | 通用 on-policy rollout buffer 和 GAE/minibatch 工具 |
| `runner/` | 训练/评估编排、checkpoint、日志、validation video |
| `utils/` | 指标、日志、cache、显示、debug、retarget 分析等工具 |
| `viewer/` | Viser web viewer 相关代码 |
| `__pycache__/` | Python 字节码缓存，可删除 |

### 6.1 `musclemimic/algorithms/common/`

| 文件 | 作用 |
| --- | --- |
| `adaptive_sampling.py` | 根据 early termination 统计计算轨迹采样权重，训练时偏向更难轨迹 |
| `base_algorithm.py` | 算法、agent config、agent state 的抽象基类 |
| `checkpoint_hooks.py` | JAX host callback 风格的 checkpoint hook 配置与创建 |
| `checkpoint_manager.py` | Orbax/统一 checkpoint manager，负责保存、恢复、元数据管理 |
| `checkpoint_utils.py` | optimizer step、update、global timestep、resume state 的换算工具 |
| `curriculum.py` | adaptive termination threshold 和 reward curriculum 状态/更新逻辑 |
| `dataclasses.py` | Transition、ValidationData、TrainState 等训练数据结构 |
| `env_state_utils.py` | 在 wrapper 层级中访问/更新底层 MJX carry 的工具 |
| `env_utils.py` | 环境 wrapping 和 observation history/split-goal 处理 |
| `moe_networks.py` | Soft MoE 网络层和 ActorCritic 变体 |
| `networks.py` | MLP、Residual MLP、ActorCritic、RunningMeanStd 等网络组件 |
| `optimizer.py` | linear/warmup-cosine schedule、Muon/Optax optimizer 构造 |
| `__init__.py` | 子包初始化 |

### 6.2 `musclemimic/algorithms/ppo/`

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | PPO public API 导出 |
| `checkpoint.py` | PPO checkpoint 读取和 `PPOAgentState` 恢复 |
| `config.py` | PPO 配置合并和 agent config/state dataclass |
| `inference.py` | PPO policy 推理、obs history buffer、MuJoCo/MJX 播放 |
| `loss.py` | PPO value loss、actor loss、KL、advantage normalization、总 loss |
| `moe.py` | PPO 中 MoE 指标聚合和空指标工具 |
| `ppo.py` | `PPOJax` 算法类，连接 config、network、optimizer、train loop |
| `runner.py` | JAX scan-based PPO 训练主循环，包括 rollout、update、validation、checkpoint |

### 6.3 `musclemimic/core/`

| 文件/目录 | 作用 |
| --- | --- |
| `mujoco_mjx.py` | 在基础 MuJoCo 环境上扩展 MJX state/carry/reset/step 行为 |
| `goals/trajectory.py` | 全身轨迹模仿目标观测 `GoalTrajMimic` 和 v2 |
| `goals/bimanual.py` | 双臂轨迹模仿目标观测 `GoalBimanualTrajMimic` 和 v2 |
| `reward/trajectory_based.py` | 轨迹相关 reward，核心是 `MimicReward` |
| `terminal_state_handler/enhanced_fullbody.py` | 全身增强 terminal handler，包括 site deviation、relative site、root deviation 判定 |
| `terminal_state_handler/enhanced_bimanual.py` | 双臂增强 terminal handler |
| `utils/site_mapping.py` | 模型 site id 和 trajectory site index 的映射，支持 reduced trajectory storage |
| `wrappers/mjx.py` | MJX wrapper 集合：log、n-step obs、vectorization、auto-reset、reward normalization |
| 各级 `__init__.py` | 子包初始化 |

### 6.4 `musclemimic/environments/`

| 文件/目录 | 作用 |
| --- | --- |
| `base.py` | MuscleMimic 环境基类 `LocoEnv` 和额外 carry 数据结构 |
| `humanoids/base_robot_humanoid.py` | 机器人 humanoid 环境基类 |
| `humanoids/base_bimanual.py` | 固定 root 和 bimanual skeleton 基类 |
| `humanoids/bimanual.py` | `MyoBimanualArm` 和 `MjxMyoBimanualArm` 环境 |
| `humanoids/myofullbody.py` | `MyoFullBody` 和 `MjxMyoFullBody` 环境 |
| 各级 `__init__.py` | 子包初始化 |

### 6.5 `musclemimic/runner/`

| 文件 | 作用 |
| --- | --- |
| `engine.py` | 统一实验编排：JAX cache、wandb、环境构建、算法选择、训练函数构造、run id/checkpoint 处理 |
| `checkpointing.py` | checkpoint 路径解析、自动恢复、Hugging Face 下载、manifest 写入和兼容性检查 |
| `eval_utils.py` | eval 通用参数、checkpoint 加载、轨迹选择、viewer/video/metrics 执行 |
| `logging.py` | 训练日志 hook、wandb/console 统一回调 |
| `validation_video_recorder.py` | training validation 阶段录制视频 |
| `__init__.py` | 子包初始化 |

### 6.6 `musclemimic/utils/`

| 文件/目录 | 作用 |
| --- | --- |
| `__init__.py` | utilities public API，包含 GMR cache 下载入口 |
| `debug_tools.py` | 训练时 debug/profiling callback |
| `demo_cache.py` | 从 Hugging Face 下载 demo retarget cache |
| `display.py` | headless 渲染检测和 `MUJOCO_GL` 配置 |
| `gmr_cache.py` | GMR cache 路径解析、下载、CLI |
| `logging.py` | logger adapter 和 timestep tracker |
| `metrics.py` | validation metrics handler，计算 joint/site/body/trajectory 误差 |
| `model.py` | 参数量统计和网络结构日志 |
| `runtime_env.py` | CUDA library path 自动配置和必要时 re-exec |
| `utd.py` | update-to-data ratio 计算 |
| `retarget/msk_metrics.py` | GMR/mimic retarget 后的肌骨约束指标，如 joint violation、ground penetration、tendon jump |
| `retarget/massive_compare.py` | 批量比较 GMR 与 mimic retarget 结果，统计失败原因 |

### 6.7 `musclemimic/rl_core/`

| 文件 | 作用 |
| --- | --- |
| `rollout_buffer.py` | on-policy rollout buffer、GAE、minibatch 生成 |
| `__init__.py` | RL core public API |

### 6.8 `musclemimic/viewer/`

| 文件 | 作用 |
| --- | --- |
| `viser_viewer.py` | 基于 Viser 的 web viewer，当前主要面向 MuJoCo CPU 环境 |
| `viser_utils.py` | 从 MuJoCo model 构造 trimesh mesh、颜色、body mesh |
| `__init__.py` | viewer public API |

## 7. `loco_mujoco/`

这是项目内置并深度修改过的 `loco_mujoco` fork，提供 MuJoCo 环境基础设施、观测/reward/terminal/control、trajectory 数据结构、SMPL retargeting 和任务工厂。

| 文件/目录 | 作用 |
| --- | --- |
| `__init__.py` | 包初始化、路径配置读取、注册环境延迟导出 |
| `core/` | MuJoCo 环境核心抽象与组件 |
| `datasets/` | 数据加载和数据生成辅助工具 |
| `smpl/` | SMPL/SMPL-H/MANO parser、retargeting、robot/GMR 配置 |
| `task_factories/` | 从配置构建 imitation/RL 任务环境 |
| `trajectory/` | 轨迹数据结构、插值、速度重算、trajectory handler |
| `utils/` | dataset path 配置、running stats、video 工具 |
| `__pycache__/` | Python 字节码缓存，可删除 |

### 7.1 `loco_mujoco/core/`

| 文件/目录 | 作用 |
| --- | --- |
| `mujoco_base.py` | 基础 `Mujoco` 环境和 `AdditionalCarry` |
| `stateful_object.py` | 有状态组件统一接口 |
| `control_functions/base.py` | control function 抽象基类 |
| `control_functions/default.py` | 默认 action 到 MuJoCo ctrl 的映射 |
| `control_functions/pd.py` | PD control 实现 |
| `domain_randomizer/base.py` | domain randomization 抽象基类 |
| `domain_randomizer/default.py` | link mass 等默认随机化 |
| `domain_randomizer/no_randomization.py` | 空随机化实现 |
| `initial_state_handler/base.py` | 初始状态 handler 抽象 |
| `initial_state_handler/default.py` | 默认初始状态 |
| `initial_state_handler/traj_init_state.py` | 从轨迹初始化环境状态 |
| `observations/base.py` | 大量基础观测类型：body、joint、site、muscle、touch、height matrix 等 |
| `observations/goals.py` | 通用 goal 观测，如随机 root velocity、trajectory root velocity |
| `observations/visualizer.py` | goal arrow 可视化辅助 |
| `reward/base.py` | reward 抽象基类 |
| `reward/default.py` | NoReward、目标速度、locomotion reward |
| `reward/utils.py` | action 越界 cost 等 reward 工具 |
| `terminal_state_handler/base.py` | terminal handler 抽象 |
| `terminal_state_handler/bimanual.py` | 双臂 terminal 判定 |
| `terminal_state_handler/height.py` | 基于高度的跌倒/终止判定 |
| `terminal_state_handler/no_terminal.py` | 永不终止 handler |
| `terminal_state_handler/traj.py` | 基于 root pose 轨迹误差的终止判定 |
| `terrain/base.py` | terrain 抽象 |
| `terrain/static.py` | 静态地形 |
| `terrain/dynamic.py` | 动态地形基类 |
| `terrain/rough.py` | rough terrain 生成与采样 |
| `utils/backend.py` | backend 支持检查 |
| `utils/decorators.py` | info property 装饰器 |
| `utils/env.py` | Box、MDPInfo 等环境元数据结构 |
| `utils/math.py` | 坐标变换、相对位置/速度/四元数、site/body velocity 等数学工具 |
| `utils/mujoco.py` | MuJoCo joint/site/geom id 和 collision 工具 |
| `visuals/scene.py` | 简化的 scene/geom 数据结构 |
| `visuals/video_recorder.py` | 视频帧记录器 |
| `visuals/viewer.py` | MuJoCo viewer/headless renderer |
| 各级 `__init__.py` | 子包初始化 |

### 7.2 `loco_mujoco/datasets/`

| 文件/目录 | 作用 |
| --- | --- |
| `data_generation/utils.py` | replay callback、mocap body 添加、collision extension、qvel 计算、robot/dataset yaml 加载 |
| `humanoids/LAFAN1/load.py` | LAFAN1 轨迹加载与扩展 |
| `humanoids/LAFAN1/const.py` | LAFAN1 常量 |
| 各级 `__init__.py` | 子包初始化 |

### 7.3 `loco_mujoco/smpl/`

| 文件/目录 | 作用 |
| --- | --- |
| `parser.py` | SMPL、SMPL-H、MANO parser |
| `retargeting.py` | AMASS/SMPL 读取、SMPL shape fitting、GMR fitting、motion retargeting 主逻辑 |
| `generate_smplh_model.py` | 生成 SMPL-H 模型资源 |
| `install_smplh.sh` | 安装/准备 SMPL-H 资源脚本 |
| `const.py` | SMPL/retarget 相关常量 |
| `utils/smoothing.py` | Gaussian smoothing 工具 |
| `robot_confs/defaults.yaml` | 默认 SMPL joint 到 robot mimic site 的匹配和优化参数 |
| `robot_confs/MyoFullBody.yaml` | MyoFullBody 专用 SMPL joint 到 mimic site 匹配 |
| `gmr_configs/smplh_to_myofullbody.json` | GMR 从 SMPL-H 到 MyoFullBody 的 IK match table 和缩放配置 |
| `__init__.py` | 子包初始化 |

### 7.4 `loco_mujoco/task_factories/`

| 文件 | 作用 |
| --- | --- |
| `base.py` | 任务工厂抽象 |
| `dataset_confs.py` | AMASS/LAFAN1/custom dataset config 和 dataset group 展开 |
| `imitation_factory.py` | 根据数据集配置构造 imitation 环境 |
| `rl_factory.py` | 构造一般 RL 环境 |
| `__init__.py` | public API |

### 7.5 `loco_mujoco/trajectory/`

| 文件 | 作用 |
| --- | --- |
| `dataclasses.py` | Trajectory、TrajectoryInfo、TrajectoryData、SingleData、插值和速度重算 |
| `handler.py` | `TrajectoryHandler`，训练/评估时按 trajectory/subtrajectory 管理参考动作 |
| `__init__.py` | 延迟导出 |

### 7.6 `loco_mujoco/utils/`

| 文件 | 作用 |
| --- | --- |
| `dataset.py` | 设置 AMASS、SMPL、converted AMASS、LAFAN1 cache 路径 |
| `running_stats.py` | running standardization 和滑动平均统计 |
| `video.py` | video 转 gif |
| `__init__.py` | 子包初始化 |

## 8. `scripts/`

| 文件 | 作用 |
| --- | --- |
| `retarget_dataset.py` | 并行批量 retarget AMASS 数据，支持 MyoBimanualArm 和 MyoFullBody |
| `upload_checkpoint.py` | 将 checkpoint 上传到 Hugging Face model repo |
| `run_with_cuda_compat.sh` | 下载/解包 CUDA compat RPM，并用指定 `LD_LIBRARY_PATH` 和 `CUDA_VISIBLE_DEVICES` 运行命令 |
| `sync_public_repo.sh` | 将当前提交树以 snapshot commit 同步到 public remote，要求工作区干净 |

## 9. `examples/`

| 文件/目录 | 作用 |
| --- | --- |
| `retargeting/retarget_visualize.py` | retarget 后动作的可视化和视频生成示例 |
| `runai/doc.md` | RunAI 使用说明 |
| `runai/image.png` | RunAI 文档图片 |
| `slurm/job.sh` | Slurm 作业脚本示例 |

## 10. `tests/`

测试目录覆盖算法、wrapper、环境、retarget 指标、checkpoint/resume、validation video 等。

| 文件/目录 | 作用 |
| --- | --- |
| `test_engine_preempt_resume.py` | runner 在本地 auto-resume 和 explicit resume 场景下的行为测试 |
| `test_resume_checkpoint_scenarios.py` | checkpoint/resume timestep、optimizer step、LR/std reset 等复杂场景测试 |
| `test_muscle_observations.py` | MyoBimanualArm/MyoFullBody 的肌肉观测 flag 和 MuJoCo/MJX 一致性测试 |
| `test_n_step_lookahead.py` | goal observation 的 n-step lookahead 维度、内容、stride、JAX/NumPy 一致性测试 |
| `unit/conftest.py` | pytest fixture |
| `unit/test_adaptive_sampling.py` | adaptive trajectory sampling 单元测试 |
| `unit/test_auto_resume.py` | auto-resume checkpoint 路径、manifest、兼容性测试 |
| `unit/test_auto_resume_integration.py` | Orbax checkpoint auto-resume 集成测试 |
| `unit/test_curriculum.py` | adaptive termination/reward curriculum 测试 |
| `unit/test_dataset_group_spec.py` | AMASS dataset group 展开测试 |
| `unit/test_default_control.py` | 默认 control action->ctrl 映射测试 |
| `unit/test_enhanced_fullbody_terminal_handler.py` | fullbody terminal handler 单元测试 |
| `unit/test_enhanced_fullbody_terminal_handler_integration.py` | fullbody terminal handler 与真实环境/轨迹的集成测试 |
| `unit/test_inference_obs_history.py` | inference observation history 和 split-goal 测试 |
| `unit/test_metrics.py` | validation metrics 和 site mapping 测试 |
| `unit/test_mimic_reward.py` | `MimicReward` 的 root XY offset 修正测试 |
| `unit/test_mjx_reset.py` | MJX reset、auto-reset、terrain/domain/carry swap 等测试 |
| `unit/test_model_mass.py` | 模型质量和 collision geom 相关测试 |
| `unit/test_model_utils.py` | 参数量统计工具测试 |
| `unit/test_msk_metrics.py` | retarget 肌骨指标测试 |
| `unit/test_n_step_wrapper.py` | `NStepWrapper` 和 split-goal 行为测试 |
| `unit/test_ppo.py` | PPO loss、optimizer schedule、MoE、wrapper order 等综合测试 |
| `unit/test_ppo_config.py` | PPO 配置合并和兼容性测试 |
| `unit/test_ppo_early_termination_metrics.py` | PPO 训练指标中 early termination 统计测试 |
| `unit/test_rollout_buffer.py` | rollout buffer、GAE、minibatch 测试 |
| `unit/test_split_goal_integration.py` | split-goal 从 env 到 wrapper 的集成测试 |
| `unit/test_terrain_height_sampling.py` | HeightMatrix 和 rough/static terrain 采样测试 |
| `unit/test_trajectory_handler_boundary.py` | trajectory 边界和最后一帧处理测试 |
| `unit/test_validation_video_recorder.py` | validation video recorder 配置、headless、录制行为测试 |
| `unit/test_video_recorder.py` | video recorder stop 幂等性测试 |
| `unit/test_warp_backend.py` | Warp backend 下 MJX env、VecEnv、AutoResetWrapper 测试 |
| `__pycache__/`, `unit/__pycache__/` | pytest/Python 字节码缓存，可删除 |

## 11. `assets/`

| 文件 | 作用 |
| --- | --- |
| `banner.jpg` | README 顶部 banner |
| `teaser.gif` | README 中展示项目效果的 teaser 动图 |
| `retargeting.gif` | README 中展示 retargeting 效果的动图 |

## 12. `smpl_models/`

SMPL-H/MANO 模型资源目录，主要服务于 SMPL/GMR retargeting。这里包含较大模型文件，通常受外部模型许可约束。

| 文件/目录 | 作用 |
| --- | --- |
| `smplh/LICENSE.txt`, `smplh/info.txt` | SMPL-H 资源许可证和说明 |
| `smplh/female/model.npz` | female SMPL-H 模型 |
| `smplh/male/model.npz` | male SMPL-H 模型 |
| `smplh/neutral/model.npz` | neutral SMPL-H 模型 |
| `mano_v1_2/LICENSE.txt`, `mano_v1_2/models/LICENSE.txt`, `mano_v1_2/models/info.txt` | MANO/SMPL-H 手部模型许可证和说明 |
| `mano_v1_2/models/MANO_LEFT.pkl` | 左手 MANO 模型 |
| `mano_v1_2/models/MANO_RIGHT.pkl` | 右手 MANO 模型 |
| `mano_v1_2/models/SMPLH_female.pkl` | female SMPL-H pkl 模型 |
| `mano_v1_2/models/SMPLH_male.pkl` | male SMPL-H pkl 模型 |
| `mano_v1_2/webuser/*.py` | MANO 官方/兼容 webuser 工具，如 LBS、pose mapper、serialization、verts、hand PCA wrapper |
| `mano_v1_2/webuser/hello_world/*.py` | MANO/SMPL+H hello world 和 render 示例 |
| `.DS_Store`, `._.DS_Store` | macOS 元数据文件，不属于核心逻辑 |

## 13. 数据、checkpoint 和训练输出目录

这些目录主要是运行产生或下载得到的产物，不是核心源码。

| 路径 | 作用 |
| --- | --- |
| `caches/AMASS/` | 本地 converted AMASS/GMR/demo cache 根目录 |
| `caches/AMASS/MyoFullBody/gmr/` | MyoFullBody 的 GMR retarget cache 和 shape metadata |
| `checkpoints/` | 训练保存的 checkpoint 根目录，按 run/config hash 分组 |
| `checkpoints/*/checkpoint_*` | Orbax checkpoint，包含 `train_state/`, `config/`, `metadata/`, `_CHECKPOINT_METADATA` |
| `checkpoints/*/manifest.json` | checkpoint run manifest，记录配置 hash、git sha、resume 信息等 |
| `outputs/YYYY-MM-DD/HH-MM-SS/` | Hydra 运行输出目录，包含 `experiment.log`、`.hydra/config.yaml` 等 |
| `outputs/**/validation_*` | validation 阶段生成的视频或评估输出 |
| `wandb/` | Weights & Biases 本地 run、日志、metadata、summary 和 `.wandb` 二进制记录 |
| `.pytest_cache/` | pytest 缓存，可删除 |
| `.venv/` | Python 虚拟环境，可重建 |
| `musclemimic.egg-info/` | editable/install 后生成的 Python package metadata，可重建 |

## 14. `.github/`

| 文件/目录 | 作用 |
| --- | --- |
| `.github/workflows/ci.yml` | GitHub Actions CI，分 lint 和 pytest 两个 job |
| `.github/ISSUE_TEMPLATE/bug_report.md` | bug report issue 模板 |
| `.github/ISSUE_TEMPLATE/feature_request.md` | feature request issue 模板 |
| `.github/pull_request_template.md` | PR 模板 |

## 15. `.git/`, `.omc/`, 运行缓存

| 路径 | 作用 |
| --- | --- |
| `.git/` | Git 仓库历史、索引、refs 等内部数据 |
| `.omc/state/hud-stdin-cache.json` | 本地 OMC/Codex UI 状态缓存 |
| `**/__pycache__/` | Python 字节码缓存 |

## 16. 推荐阅读顺序

如果目的是理解代码，建议按这个顺序：

1. `README.md` 和 `TUTORIAL.md`：先理解使用方式和主链路。
2. `fullbody/conf_fullbody_demo.yaml`、`bimanual/conf_bimanual_demo.yaml`：理解最小可运行配置。
3. `fullbody/experiment.py`、`bimanual/experiment.py`：看训练如何进入统一 runner。
4. `musclemimic/runner/engine.py`：理解环境、算法、日志、checkpoint 如何被组装。
5. `musclemimic/environments/humanoids/*.py`：理解具体肌肉骨骼环境。
6. `musclemimic/core/goals/*.py`、`musclemimic/core/reward/trajectory_based.py`、`musclemimic/core/terminal_state_handler/*.py`：理解模仿学习任务定义。
7. `musclemimic/algorithms/ppo/runner.py`、`loss.py`、`networks.py`：理解 PPO 训练细节。
8. `loco_mujoco/smpl/retargeting.py`、`scripts/retarget_dataset.py`：理解数据 retargeting。

## 17. 哪些目录通常不应该手动改

| 路径 | 原因 |
| --- | --- |
| `.git/` | Git 内部目录 |
| `.venv/` | 依赖环境，可重建 |
| `.pytest_cache/`, `__pycache__/` | 自动缓存 |
| `wandb/`, `outputs/`, `checkpoints/` | 运行产物，除非需要清理或分析实验 |
| `smpl_models/` | 第三方模型资源，涉及许可和大文件 |
| `musclemimic.egg-info/` | 安装元数据，可重建 |

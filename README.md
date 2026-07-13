# MuscleMimic Badminton

面向羽毛球全身动作学习的肌肉骨骼强化学习与模仿学习代码库。

本项目基于 [MuscleMimic](https://github.com/amathislab/musclemimic)，围绕 MyoFullBody 肌肉驱动模型扩展了羽毛球动作重定向、来球击打环境、分阶段策略训练、教师—学生蒸馏、DAgger、latent muscle policy、训练恢复以及发布前质量门控。

> 本仓库采用“只发布代码”的策略。数据集、模型权重、训练输出、视频、普通文档以及 YAML 配置不会提交到 Git；根目录 `README.md` 是唯一跟踪的 Markdown 文件。

## 项目目标

项目希望建立从人体运动参考到可训练羽毛球策略的完整代码链路：

1. 将 WHAM/SMPL/AMASS 动作转换并重定向到 MyoFullBody。
2. 检查帧率、轨迹连续性、手部速度、关节跳变和重定向误差。
3. 在 MuJoCo/MJX 中构建球场、球拍、羽毛球和来球击打任务。
4. 通过模仿学习、PPO、BC、DAgger 和 latent policy 学习全身击球控制。
5. 使用数据契约、训练门控、checkpoint 恢复和鲁棒性评估保证实验可复现。

## 主要能力

- **肌肉骨骼控制**：支持 MyoFullBody 等高维肌肉驱动模型。
- **GPU 并行训练**：基于 JAX、MJX、Flax、Optax 和 Warp。
- **羽毛球环境**：包含球拍握持、来球生成、碰撞、击球目标与分层控制逻辑。
- **动作重定向**：提供 WHAM/SMPL/AMASS 到 GMR/MyoFullBody 的处理脚本。
- **轨迹质量控制**：检查 FPS 一致性、根节点与手部运动连续性、缓存质量及训练准入条件。
- **三阶段训练**：身体动作、球拍控制和击球技能可按阶段训练与验收。
- **策略蒸馏**：支持 teacher rollout、行为克隆、DAgger、student evaluation 与 provenance 记录。
- **Latent muscle policy**：支持 latent action、归一化、闭环评估和高层策略接口。
- **可靠训练**：支持自动恢复、checkpoint 限额、验证指标、视频记录接口和 early stop。

## 代码结构

```text
musclemimic/
├── musclemimic/
│   ├── algorithms/          # PPO、推理、checkpoint 与通用训练组件
│   ├── badminton/           # 羽毛球数据 QC、训练门控、重定向和评估脚本
│   ├── core/                # 奖励、wrapper 与环境公共逻辑
│   ├── distill/             # BC/DAgger、数据契约、动作与观测 schema
│   ├── environments/        # 肌肉骨骼环境
│   ├── latent_muscle/       # latent policy 训练与运行时
│   └── runner/              # 训练引擎、恢复、日志和验证
├── environment/
│   ├── overall_environment/ # 羽毛球综合场景与 MJX 来球训练环境
│   ├── court/               # 球场模型
│   ├── racket/              # 球拍模型
│   └── shuttlecock/         # 羽毛球动力学与碰撞测试
├── fullbody/                # 全身训练、评估、蒸馏和三阶段流程入口
├── bimanual/                # 双臂模型训练与评估入口
├── loco_mujoco/             # 模型、数据加载与 SMPL/GMR 支撑代码
├── scripts/                 # 数据与动作扫描工具
├── tests/                   # 单元测试和集成测试
└── pyproject.toml           # Python 包与命令行入口
```

## 环境要求

- Python 3.11+
- Linux（训练推荐）或 macOS（CPU 推理/部分评估）
- 使用 GPU 训练时需要兼容的 NVIDIA 驱动和 CUDA 12 环境
- 推荐使用 [uv](https://docs.astral.sh/uv/) 管理依赖

## 安装

```bash
git clone https://github.com/yangfeiyang-123/asi_strengthen_musclemimic.git
cd asi_strengthen_musclemimic

# 基础依赖与开发测试工具
uv sync --extra dev
```

Linux GPU 环境：

```bash
uv sync --extra dev --extra cuda
```

如果需要 SMPL/GMR 动作重定向：

```bash
uv sync --extra dev --extra smpl --extra gmr
```

安装完成后可先检查包导入和命令行入口：

```bash
uv run python -c "import musclemimic; print('musclemimic import OK')"
uv run forehand-clear-three-stage --help
uv run forehand-clear-data-qc --help
```

## 本地配置与资产

为了避免把实验配置、授权数据和大文件提交到代码仓库，以下内容需要在本地准备：

- Hydra/YAML 训练配置；
- WHAM、SMPL、AMASS 或其他动作数据；
- Myo/MuJoCo、SMPL-H、MANO 等模型资产；
- GMR 重定向缓存；
- teacher/student checkpoint；
- W&B 日志、评估结果、渲染图片与视频。

常用本地目录包括：

```text
data/
datasets/
caches/
checkpoints/
ckpts/
outputs/
wandb/
smpl_models/
```

这些文件受 `.gitignore` 保护。YAML 配置同样不随仓库发布，因此运行训练命令前需要准备与任务对应的本地配置。

## 常用入口

### 来球击打环境

```bash
uv run python environment/overall_environment/src/train_incoming_hit_mjx.py --help
uv run python environment/overall_environment/src/overall_env.py --help
```

### 羽毛球三阶段流程

```bash
uv run forehand-clear-three-stage --help
uv run forehand-clear-promotion-gate --help
uv run forehand-clear-finger-robustness --help
```

### 教师—学生蒸馏

```bash
uv run musclemimic-distill-collect-teacher --help
uv run musclemimic-distill-train-bc --help
uv run musclemimic-distill-run-dagger --help
uv run musclemimic-distill-compare --help
```

### Latent muscle policy

```bash
uv run musclemimic-latent-train --help
uv run musclemimic-latent-eval --help
uv run musclemimic-latent-closed-loop-eval --help
```

### 重定向与数据质量检查

```bash
uv run python -m musclemimic.badminton.scripts.run_retarget --help
uv run python -m musclemimic.badminton.scripts.render_retarget_cache --help
uv run forehand-clear-data-qc --help
uv run forehand-clear-visual-review --help
```

## 测试

完整本地配置和模型资产准备好后运行：

```bash
uv run pytest
```

在尚未准备本地 YAML 和训练资产时，可以先执行源码级 smoke tests：

```bash
uv run pytest -q tests/unit/test_metrics.py tests/unit/test_collection_budget.py
```

当前代码提交前的本地测试结果为：

```text
1146 passed, 41 skipped
```

部分集成测试会根据 GPU、Warp、模型文件或数据是否存在自动跳过。

## 数据与提交策略

仓库只跟踪源代码、测试、必要的 JSON/XML/TOML 代码资产和根目录 README。下列内容不会跟踪：

- `*.md`（根目录 `README.md` 除外）
- `*.yaml`、`*.yml`
- `*.gif`、本地生成的 PNG 和视频
- `*.pt`、`*.pth`、`*.ckpt`、`*.pkl`、`*.npz`
- 数据集、缓存、checkpoint、日志和训练输出目录

提交前建议检查：

```bash
git status --short
git diff --cached --check
git ls-files | rg -i '\.(yaml|yml|gif|pt|pth|ckpt|npz|mp4)$'
```

## 上游项目与许可

本项目基于 MuscleMimic 进行羽毛球方向扩展。原始项目、模型和第三方数据分别受其各自许可约束。

本仓库代码遵循 [Apache License 2.0](LICENSE)。

# MuscleMimic Badminton

基于 MyoFullBody 的羽毛球研究：轨迹跟踪 → 技能蒸馏 → 自适应击球，以人体 sEMG 约束肌肉激活与协同。

| 要做什么 | 从这里开始 |
| --- | --- |
| 看本机训练状态 | [实验主表](experiments/EXPERIMENT_LOG.md) · 服务器 9 交接 `docs/runbooks/server9/当前实验状态.md`（仅本机） |
| 理解项目与安装环境 | [项目总览与环境说明](docs/narrative/00_项目总览与环境说明.md) |
| 找方法、计划、合同 | [文档索引](docs/README.md) |
| 找实验配置 | [三阶段实验](experiments/README.md) · [Hydra 配置导航](src/fullbody/config_specific_task/README.md) |
| 修改或启动训练 | [协作约定](AGENTS.md) · 本机训练规范 `docs/runbooks/server9/智能体训练执行规范.md`（仅本机） |
| 找目录与整理备份 | 本机目录导航 `docs/runbooks/server9/仓库目录导航.md`（仅本机） |

> 标“仅本机”的文档在 `docs/runbooks/server9/`，被 `.gitignore` 排除，只存在于服务器 9 的工作目录；clean clone 里没有这些文件。

## 代码放在哪里

| 目录 | 职责 |
| --- | --- |
| `src/musclemimic/` | 核心 Python 包：算法、奖励、肌肉环境、蒸馏与评估 |
| `src/loco_mujoco/` | 轨迹、重定向与环境基础组件 |
| `src/fullbody/` | 训练与评估入口、Hydra 配置 |
| `src/environment/` | 球场、球拍、羽毛球、单人击球及双人对打场景 |
| `src/musclemimic/grip/` | 独立的手指握拍实验 |
| `src/analysis/`、`jidian_measurement/` | 离线分析与独立肌电采集子项目 |
| `scripts/`、`tests/` | 通用命令与分层测试 |
| `configs/`、`assets/` | 公共配置、合同模板与小型公共资产 |

## 文件归位规则

- 文档统一在 `docs/`，实验定义统一在 `experiments/stage1/`、`stage2/`、`stage3/`。
- 本机只有一份实验主表：`experiments/EXPERIMENT_LOG.md`；另一工作站记录进入 `experiments/archive/`。
- 数据与模型在 `datasets/`、`smpl_models/`；运行输出在 `outputs/`；证据、报告、视频和迁移包在 `artifacts/`。
- 本机工具、缓存、整理备份和冻结源码副本在 `.local/`。根目录两份私有资产清单入口仍被训练合同引用，保持原路径。
- 私有数据、模型、清单和训练产物不进入 Git。新增内容使用规范路径；根目录不再保留 `doc`、`Experiments` 的重复入口。

当前开发目录在 `74bf895` 分类基础上采用统一的 `src/` 源码布局。运行中的训练继续使用各自冻结 worktree；开发目录 HEAD 不代表已有训练的源码版本。

## 本机开发检查

```bash
source configs/env.sh
make source-only
uv run --locked python -c 'import musclemimic; print(musclemimic.__file__)'
```

本机开发入口使用 `.local/dev-venv/`，与已有训练环境分开；新 clone 仍可按环境说明安装常规 `.venv/`。生产训练按对应 family 的冻结 worktree、专用 venv 和启动合同执行。

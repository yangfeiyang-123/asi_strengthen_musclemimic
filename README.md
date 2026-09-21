# MuscleMimic Badminton

[![CI](https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/actions/workflows/ci.yml/badge.svg)](https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](pyproject.toml)

![MuscleMimic banner](assets/banner.jpg)

用少量表面肌电（sEMG）锚定的三阶段学习框架，让 354 肌肉的 MyoFullBody 全身肌骨模型学会正手高远球，并根据来球状态自适应击球。

本项目基于上游 [MuscleMimic](https://github.com/amathislab/musclemimic)，在 MyoFullBody 上扩展了从视频重定向、肌电奖励跟踪、隐空间技能蒸馏到冻结技能的自适应击球的完整链路，并附带一个双人对打环境。

> [!IMPORTANT]
> 这是研究代码。源码和可移植配置随仓库发布；动作数据、SMPL/AMASS 资产、真实 sEMG、checkpoint、W&B 记录、视频和训练结果默认不进入 Git。没有对应 artifact 时，代码可验证接口和实验合同，但不能据此声称已经获得训练效果或人体生理结论。

## 主线：三阶段学习

类比一个人学球：教练带着挥拍（轨迹跟踪）→ 丢掉教练（技能蒸馏）→ 多球训练（自适应击球）。

| 阶段 | 学什么 | 输入 → 输出 | 肌电的作用 | 仓库中的组成 |
|---|---|---|---|---|
| **S1 轨迹跟踪** | 跟着人体参考完成正手高远球，全身 354 肌肉 PPO | 当前状态 + 未来参考 → 354-D 肌肉激活 | 15 通道 sEMG 投影为奖励：单肌肉激活 tube + 协同形状 `1-cos` | Stage 1 body-only matched family（T0–T4）→ 25/50/75/100% 球拍质量课程 |
| **S2 技能蒸馏** | 去掉未来参考和在线肌电，压成低维技能 | 当前状态 → 隐变量 z → 354-D | 训练期 posterior 看 `q(z\|s,r,e)`，prior 只看 `p(z\|s)`；`L_KL + L_π + L_EMG`，context dropout | direct BC/DAgger/PPO（S2-A）+ latent family（S2-B/C/D/E） |
| **S3 自适应击球** | 冻结 decoder，按来球调整技能 | 状态 + 球/拍状态 → `u`，`z = μ + λσ⊙tanh(u)` → 354-D | 不直接输入，由冻结技能隐式继承 | LAB 高层 PPO、固定站位多球喂送、H1/H2/H3 |

消融命名以代码为准：Stage 1 为 T0–T4，Stage 2 为 S2-A..E，Stage 3 为 H1–H3。与汇报 PPT 标签的对应表见 [三阶段方法与肌电参与机制](docs/narrative/02_三阶段方法与肌电参与机制.md) 第 24 节。

**保留的扩展环境**：[双人对打](environment/double_play/README.md)。两个 MyoFullBody 在 BWF 球场两侧互击高远球，P2 为 P1 的 180° 镜像副本，镜像观测后可用单一共享策略自博弈。定位是 S3 之后的评估场，不是主线训练目标。

### 当前状态（2026-09-18）

| 阶段 | 代码 | 正式实验 |
|---|---|---|
| S0 数据与肌电 | 完成 | aug100 80/20 release、verified tube 已落盘；impact 事件独立证据为 0，肌电为单 subject |
| S1 | 完成，有测试 | 80/20 matched family 15 个 run 中 12 个到达 800M endpoint（`checkpoints/stage1/`），未晋级；缺 T0 s1（本机中断于 300M）、T0 s2、T1 s2 |
| 球拍课程 | 完成 | 未开始 |
| S2 | 完成，有测试 | 未开始 |
| S3 LAB | 完成，可运行 | 未开始 |
| 双人对打 | 完成（CPU MuJoCo） | 不排期 |

进度记录：[experiments/EXPERIMENT_LOG.md](experiments/EXPERIMENT_LOG.md)。待办：[当前状态与待办](docs/narrative/03_当前状态与待办.md)。

## 核心设计

### 1. 统一的物理肌肉控制语义

当前生产合同是 excitation v2：

~~~text
policy action [-1, 1]
    → DefaultControl
physical muscle control [0, 1]
    → MuJoCo muscle actuator
effective excitation = clip(raw data.ctrl, 0, 1)
~~~

- 354 个 body muscle runtime 必须具有物理 <code>ctrlrange=[0,1]</code>，policy ABI 保持 <code>[-1,1]</code>。
- activation 必须通过 <code>model.actuator_actadr</code> 读取 <code>data.act</code>。
- 肌电奖励与生理评估都作用在 activation 经名称安全投影得到的 15 通道空间上，不对 354 个肌肉逐个约束。

详见 [肌肉生理约束实施契约 v2](docs/contracts/肌肉生理约束实施契约_v2.md)。

### 2. 肌电如何进入三个阶段

~~~text
S1：直接约束输出      r = r_track − λ_A·tube(P·a) − λ_S·[(1−cos(ĥ, Ĥ)) + 0.25·tube(intensity)]
S2：约束技能表征      posterior q(z|s, r, e_EMG)，prior p(z|s)，SynergyHead G(z)，context dropout 0.30
S3：隐式继承          冻结 prior/decoder，λ_EMG = 0
~~~

- `P` 是 16 采集通道中 15 个有同源 actuator 的通道到 354 肌肉的观测投影；tube 是按动作相位统计的 median ± 1.4826·MAD 范围，只惩罚超出人体自然变化的部分。
- 奖励路径 fail-closed：tube 必须 `review_status=verified` 且 `training_enabled=true`，mapping 哈希必须与 tube 绑定一致。
- S1 的 T4 与 S2 的 S2-D 是负对照：真实肌电先验必须优于相位平移或 shuffled 先验，才能说模型利用了生理结构。

### 3. 球拍与场景

- 球拍是右手 <code>thirdmc_r</code> 的 jointless exact child，没有 free joint 和手指 actuator；质量与惯量由肩肘腕肌肉链承担。
- Stage 2 与 Stage 3 复用同一版本化 attachment contract，见 [354 维动作模式与刚性球拍主线](docs/contracts/body_action_modes_and_rigid_racket.md)。
- 羽毛球空气动力学与拍球接触为 v2 模型，双人对打环境复用同一套。

## 仓库结构

~~~text
musclemimic/
├── musclemimic/
│   ├── algorithms/          # PPO、网络、推理和训练公共组件
│   ├── badminton/           # 数据、event、promotion、Stage-1 gate、Stage-2 family、Stage-3 脚本
│   ├── core/                # 奖励（含 EMG 一致性项）、terminal、wrapper、MJX 公共逻辑
│   ├── distill/             # teacher dataset、BC/DAgger、direct lifecycle
│   ├── environments/        # MyoFullBody 环境
│   ├── latent_muscle/       # posterior/prior/decoder、SynergyHead、LAB、闭环评估
│   ├── physiology/          # EMG tube、anchor loss、runtime binding、taxonomy
│   ├── evaluation/          # EMG、cohort、physiology 和 Stage-3 signal export
│   ├── synergy/             # NMF 与 basis 工具（主线只用于人体侧协同提取）
│   └── runner/              # 训练引擎、checkpoint、自动恢复和日志
├── environment/
│   ├── court/               # BWF 球场几何与 MJCF
│   ├── racket/              # 刚性球拍、stringbed 和参数验证
│   ├── shuttlecock/         # 空气动力学与拍球碰撞
│   ├── overall_environment/ # 单人来球击打场景、LAB、CPU/MJX 环境
│   └── double_play/         # 双人对打环境（保留）
├── fullbody/                # Hydra 训练入口、蒸馏、latent、pipeline planner
│   └── config_specific_task/  # 主线配置 + archive/{chinajump,continuity_graph_nmf,forehand_lift,fixed_synergy,legacy_22_5,subset_40_10,strokes,skill}，导航见其 README.md
├── loco_mujoco/             # 模型、数据加载、SMPL/GMR 和环境基座
├── configs/                 # 环境绑定、公开 JSON 模板、physiology 合同、球拍/握拍资产合同
├── experiments/             # 按三阶段组织：EXPERIMENT_LOG.md、stage1/ 记录与分析脚本、stage2/、stage3/{lab,direct_residual,early_tasks} spec
├── docs/                    # narrative/ plans/ runbooks/ contracts/ archive/，索引见 docs/README.md
├── analysis/                # latent_synergy/（S2/S3 分析）、physiology_synergy/（归档的连续性/Graph-NMF 分析）
├── jidian_measurement/      # 独立的 Delsys Trigno 16 通道采集与预处理子项目
├── scripts/                 # 生产 launcher、CUDA compat、数据/肌电工具；legacy/ 为不再引用的旧脚本
├── src/grip/                # 手指级握拍：CPU 环境、IK、torch PPO、渲染（与主线刚性挂接不连通）
├── assets/                  # banner 与右手握拍场景 MJCF
└── tests/                   # source_only/、unit/、integration/、asset/、gpu/
~~~

本地私有目录（gitignored）：`datasets/`（release、训练产物）、`artifacts/`（verified tube 等证据）、`outputs/`、`wandb/`、`进展/`（汇报 PDF）。

## 环境要求

- Linux x86_64；Python 3.11；NVIDIA GPU 与兼容驱动；<code>uv</code>；<code>git</code>、<code>wget</code>、<code>bsdtar</code>；
- 可选的 W&B 账号；合法获取的 SMPL-H/AMASS 等外部资产。

CPU 工具、source-only 测试和部分分析可以在其他系统运行。Delsys Trigno 采集子项目通常运行在装有 Trigno Control Utility 的 Windows 主机。

## 安装

~~~bash
git clone https://github.com/yangfeiyang-123/asi_strengthen_musclemimic.git
cd asi_strengthen_musclemimic

uv sync --locked --extra dev                                   # 源码、CPU 工具、开发测试
uv sync --locked --extra dev --extra cuda                      # Linux CUDA 训练
uv sync --locked --extra dev --extra cuda --extra smpl --extra gmr   # 还需要 SMPL/GMR 重定向时
~~~

验证干净源码：

~~~bash
make source-only
make lint
uv run --locked forehand-clear-three-stage --help
uv run --locked musclemimic-emg-eval --dry-run
~~~

## 本地数据、模型与环境变量

需要把当前工作迁移到另一台 GPU 服务器时，按 [服务器部署与私有资产迁移](docs/runbooks/server_deployment.md) 生成逐文件 SHA-256 资产清单、私下同步数据，并在目标机运行 fail-closed preflight。Git clone 本身不包含训练所需的 release、SMPL-H、verified EMG tube 或 checkpoint。

~~~text
datasets/
├── _global/{amass_npz, muscle_trajectory/gmr_cache}
└── forehandClear_standard/{muscle_trajectory, manifests, distill, training_*}
artifacts/emg_human_review_v2/verified_tubes/forehand_high_clear/   # verified tube + mapping（gitignored）
smpl_models/smplh/
outputs/  wandb/
~~~

从仓库根目录 <code>source configs/env.sh</code> 统一绑定 <code>MUSCLEMIMIC_DATASETS_ROOT</code>、<code>MUSCLEMIMIC_AMASS_PATH</code>、<code>MUSCLEMIMIC_GMR_CACHE_PATH</code>、<code>MUSCLEMIMIC_SMPL_MODEL_PATH</code>、<code>MUSCLEMIMIC_JAX_CACHE_ROOT</code> 等变量。正式训练 launcher 会自动 source 此文件。

## 正式训练

所有本地 fullbody 生产训练必须从仓库根目录通过 [scripts/run_fullbody_training.sh](scripts/run_fullbody_training.sh) 启动，一张物理 GPU 一个进程：

~~~bash
export CUDA_VISIBLE_DEVICES=<one_physical_gpu>
export MUSCLEMIMIC_JAX_CACHE_KEY=forehand_stage1_t0_s1
export MUSCLEMIMIC_TRAIN_LOG=datasets/forehandClear_standard/training/logs/forehand_stage1_t0_s1.log

# 1. 只检查 launcher 环境
MUSCLEMIMIC_DRY_RUN=1 scripts/run_fullbody_training.sh \
  --config-name=config_specific_task/stage1_body/peasd_lite_v1/conf_fullbody_forehand_clear_aug100_peasd_t0 \
  wandb.mode=disabled
# 2. 解析 Hydra 配置但不训练
scripts/run_fullbody_training.sh --config-name=<同上> --cfg job --resolve wandb.mode=disabled
# 3. 启动
scripts/run_fullbody_training.sh --config-name=<同上> wandb.mode=online
~~~

启动后只有以下条件全部满足才算成功：所有轨迹从本地 retarget cache 加载、run manifest 已写入、W&B 显示 live run、<code>nvidia-smi</code> 在预期 GPU 上显示新 PID、日志到达 <code>Starting training...</code> 且没有 fatal traceback。完整合同见 [AGENTS.md](AGENTS.md) 与 [训练启动合同](docs/runbooks/workstation_training_contract.md)。

## 主要工作流

### 1. 数据重定向与 QC（S0）

~~~bash
uv run --locked python -m musclemimic.badminton.scripts.run_retarget --help
uv run --locked forehand-clear-data-qc --help
uv run --locked forehand-clear-visual-review --help
~~~

正手高远球数据为 22 条 train、5 条 held-out validation 的源划分，经 yaw 增强为 aug100 后按源分组切成 80/20；source 60 Hz，环境 cache 100 Hz。训练前重新校验 release、逐文件哈希、split、FPS 和视觉签字。

### 2. Stage 1：T0–T4 matched family（S1）

配置在 <code>fullbody/config_specific_task/stage1_body/peasd_lite_v1/</code>，五个 arm 由 <code>presets/stage1_peasd_lite_t{0..4}_v1.yaml</code> 决定，只差 EMG treatment：

| Arm | anchor | synergy | 含义 |
|---|---:|---:|---|
| T0 | 0 | 0 | 无 EMG 基线 |
| T1 | 0.02 | 0 | 仅激活约束 |
| T2 | 0 | 0.05 | 仅协同约束 |
| T3 | 0.02 | 0.05 | PEASD-Lite 主方法 |
| T4 | 0.02 | 0.05 + 相位半周期平移 | 负对照 |

每个 arm 3 seeds、800M 步 fixed endpoint、fresh optimizer。晋级门要求 T3 每个 seed 的真实协同损失相对 T4 至少改善 5%，且 tracking 与安全指标不退化（<code>musclemimic/badminton/stage1_peasd_gate.py</code>）。T3 晋级后进入 <code>stage2_racket_v2/</code> 的 25→50→75→100% 球拍质量课程，该课程不再重复加肌电奖励。

### 3. Stage 2：蒸馏（S2）

~~~bash
uv run --locked musclemimic-distill-collect-teacher --help     # 一次且仅一次的 shared teacher collection
uv run --locked musclemimic-distill-train-bc --help
uv run --locked musclemimic-distill-run-dagger --help
uv run --locked musclemimic-latent-train --help                # posterior/prior/decoder + SynergyHead
uv run --locked musclemimic-latent-synergy-sweep --help        # --stage2-arm S2-B/C/D/E
uv run --locked musclemimic-latent-closed-loop-eval --help
~~~

S2-A 是完整 direct lifecycle（BC → 3 轮 DAgger → fresh-optimizer PPO）。S2-B 在无肌电条件下选择并锁定一次 latent architecture；S2-C（真实 EMG context，dropout 0.30）、S2-D（shuffled context）、S2-E（dropout 0）共享同一 shared inputs 与 architecture lock。部署时只保留 prior 和 decoder，不需要未来参考或在线肌电；student 仍接收动作相位标量。

### 4. Stage 3：LAB 自适应击球（S3）

~~~bash
uv run --locked python -m musclemimic.badminton.scripts.run_incoming_shuttle_hit \
  --spec experiments/stage3/lab/incoming_shuttle_hit_impact_recovery_v2.yaml \
  --stage preflight --out-dir outputs/stage3_preflight
~~~

LAB spec 为 <code>incoming_shuttle_hit_v1.yaml</code> 与 <code>incoming_shuttle_hit_impact_recovery_v2.yaml</code>（<code>stage3_lab.enabled: true</code>）；训练入口 <code>fullbody/latent_run_lab_ppo.py</code>。正式训练前必须依次通过 preflight、base-only-check、feed-check 和单 feed 物理可达性门（真实 stringbed 接触且 outgoing z 为正）。H1/H2/H3 分别使用 S2-B、S2-C、S2-C + 有界右臂 residual（alpha ≤ 0.10）。<code>experiments/stage3/direct_residual/</code> 是已归档的直接残差探索线。

### 5. Jidian 16 通道 sEMG

采集子项目位于 [jidian_measurement](jidian_measurement/README.md)。进入主项目前必须做全有或全无导入，并由 <code>scripts/build_emg_reference_tube.py</code> 生成按动作相位统计的 verified tube：

~~~bash
uv run --locked musclemimic-jidian-emg-import outputs/emg/jidian_selection.json outputs/emg/jidian_strict.npz \
  --audit-json outputs/emg/jidian_import.audit.json
uv run --locked musclemimic-emg-eval --dry-run
uv run --locked musclemimic-physiology-eval --dry-run
~~~

16 通道采集、15 通道可比；仓库内 mapping 为 provisional，训练读取与 tube 绑定的 verified 副本。paired 与 unpaired 评估设计必须在采集前固定，"动作名称相同"不等于 paired。详见 [Jidian sEMG 集成合同](docs/contracts/jidian_emg_integration.md) 与 [MVC 小于动作信号时如何处理](docs/contracts/MVC小于动作信号时如何处理.md)。

### 6. 双人对打环境

~~~bash
.venv/bin/python -m environment.double_play.src.build_double_play_scene
MUJOCO_GL=egl .venv/bin/python -m environment.double_play.src.run_double_play_demo --video outputs/double_play/double_play_demo.mp4
.venv/bin/python -m pytest environment/double_play/tests/ -q
~~~

<code>DoublePlayRallyEnv</code> 为回合制拉吊环境：双方必须交替合法击球，落地结分，错序击球或倒地判负。当前为 CPU MuJoCo，MJX 批量版未移植，尚无 Stage-3 策略桥。详见 [environment/double_play/README.md](environment/double_play/README.md)。

### 7. 归档与历史模块

fixed-synergy / W+R 动作空间、ChinaJump primitive 协同、Graph-NMF、肌束连续性（IMR）、握拍 preset 编辑器等支线的代码与测试仍在仓库中，但不再排期、不进入主方法。文档见 [docs/archive/](docs/archive/README.md)。<code>src/grip/</code> 为手指级握拍支线；上游的 <code>bimanual/</code> 入口与 <code>examples/</code> 已于 2026-09-20 删除（git 历史可查）。

## 配置导航

| 路径 | 用途 |
|---|---|
| <code>fullbody/conf_fullbody.yaml</code> | 通用全身 PPO/MJX 基座 |
| <code>fullbody/config_specific_task/stage1_body/peasd_lite_v1/</code> | Stage-1 T0–T4 |
| <code>fullbody/config_specific_task/presets/stage1_peasd_lite_*</code> | 五个 arm 的 treatment preset |
| <code>fullbody/config_specific_task/stage2_racket_v2/</code> | 25/50/75/100% 球拍质量课程 |
| <code>fullbody/config_specific_task/distill/</code> | direct student 与 latent/LAB |
| <code>experiments/stage3/lab/</code> | Stage-3 LAB spec（v1、impact_recovery_v2）；<code>direct_residual/</code> 为归档线 |
| <code>configs/physiology/</code> | 354-muscle taxonomy 与 16→15 mapping |
| <code>configs/public/</code> | event、signal identity 和 physiology 模板 |
| <code>environment/double_play/params/</code> | 双人对打场景与物理参数 |

Hydra 的 <code>--config-name</code> 相对于 <code>fullbody/</code>，不带 <code>.yaml</code>。标记为 legacy/experimental/nonproduction 的配置默认失败关闭。

## Artifact 与可复现性

| Artifact | 作用 |
|---|---|
| <code>run_manifest.json</code> | 固定 config hash、run ID、训练预算、reward、terminal、promotion 和 action contract |
| <code>stage1_peasd_validation_history.json</code> | 每次 validation 的 tracking 与 EMG 诊断，绑定 checkpoint 身份 |
| <code>dataset_manifest.json</code> | 固定 teacher、split、collection、shard、样本数和内容哈希 |
| <code>emg_reference_tube.npz</code> + manifest | verified 人体 tube，绑定 mapping 哈希与 trial QC review |
| <code>pipeline_plan.json</code> | 固定 profile、步骤、命令和所需 artifact |
| <code>preflight_report.json</code> / <code>feed_check_report.json</code> | Stage-3 场景、prior、feed 和 target 前置证据 |
| promotion manifest / family gate | 把 checkpoint、指标、人工 review 和 parent lineage 封装为可验证晋级证据 |

关键原则：train 与 validation 按 motion 隔离；配置和数据内容改变时旧 checkpoint 不按 shape 静默恢复；promotion 读取原始指标和内容哈希，不接受手写 <code>passed: true</code>；失败 run 也保留在消融报告中。

## 测试与开发

~~~bash
make lint
make source-only     # 不需要数据、checkpoint、SMPL 或 GPU
make test            # tests/unit 中部分用例依赖本地私有资产
make ci
~~~

生产训练前不要只运行全局 smoke test；还要执行与本次 reward、terminal、checkpoint contract 和目标 config 对应的 focused tests。

## 数据、隐私与发布策略

仓库不会发布 WHAM、AMASS、SMPL-H/MANO 等受许可约束的原始资产；本地 retarget cache 和动作视频；真实受试者身份、原始 sEMG 或可识别证据视频；checkpoint、W&B、训练日志和大体积评估输出；机器专用路径、token 和私密配置。公开 JSON/YAML 只提供 schema、模板和可移植实验合同。

## 已知限制

1. 数据、verified tube 和 checkpoint 不随源码发布，clone 后不能直接复现。
2. 生产 GPU 栈面向 Linux x86_64 + NVIDIA。
3. 球拍是 exact-child rigid tool，不是可滑移、可摩擦的真实手指抓握。
4. 15 通道 sEMG 是对 354 肌肉的部分观测锚点，不是全身人体效度验证；当前肌电为单 subject 单 session，与视频受试者非配对，impact 事件独立证据为 0。
5. 正手高远球的重定向未启用下肢接触优化。
6. Stage 3 的 LAB 路径尚无训练结果；双人对打环境尚无策略桥与 MJX 版。
7. 任何实验结果都必须与具体 commit、resolved config、数据 release、seed、checkpoint 和 promotion artifact 一起解释。

## 进一步文档

| 文档 | 内容 |
|---|---|
| [研究故事与论文叙事主线](docs/narrative/01_研究故事与论文叙事主线.md) | 三阶段叙事、论文问题定义与贡献 |
| [三阶段方法与肌电参与机制](docs/narrative/02_三阶段方法与肌电参与机制.md) | 每阶段的输入、损失、肌电作用、验收门与消融命名 |
| [当前状态与待办](docs/narrative/03_当前状态与待办.md) | 就绪度、立即待办、已决定事项、工程债 |
| [PEASD 正式实验计划](docs/plans/PEASD正式实验计划.md) | claim map、实验块、Go/Stop 与 Definition of Done |
| [PEASD 实施指南](docs/runbooks/peasd_implementation_guide.md) | 逐步可执行的正式流程与证据门 |
| [蒸馏 runbook](docs/runbooks/forehand_clear_distillation_runbook.md) | teacher collection、BC/DAgger、latent 训练操作 |
| [Jidian sEMG 集成合同](docs/contracts/jidian_emg_integration.md) | 16→15 mapping、事件证据、strict import、paired/unpaired |
| [肌肉生理约束实施契约 v2](docs/contracts/肌肉生理约束实施契约_v2.md) | excitation/activation 语义与迁移边界 |
| [双人对打环境](environment/double_play/README.md) | 场景、物理、回合规则与已知边界 |
| [AGENTS.md](AGENTS.md) / [训练启动合同](docs/runbooks/workstation_training_contract.md) / [服务器部署](docs/runbooks/server_deployment.md) | 生产训练、迁移与验收 |
| [docs/archive/](docs/archive/README.md) | 已归档的支线文档 |

## 上游、许可与引用

- 上游项目：[amathislab/musclemimic](https://github.com/amathislab/musclemimic)
- 本项目仓库：[yangfeiyang-123/asi_strengthen_musclemimic](https://github.com/yangfeiyang-123/asi_strengthen_musclemimic)
- 代码许可：[Apache License 2.0](LICENSE)

如果在研究中使用本项目，请同时引用上游 MuscleMimic 工作，并记录本仓库的具体 commit SHA、实验配置和数据/模型来源。第三方模型、人体数据和数据集仍受各自许可、知情同意和伦理要求约束。

# MuscleMimic Badminton

[![CI](https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/actions/workflows/ci.yml/badge.svg)](https://github.com/yangfeiyang-123/asi_strengthen_musclemimic/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](pyproject.toml)

![MuscleMimic banner](assets/banner.jpg)

面向羽毛球全身动作学习、肌肉协同控制和生理信号验证的肌肉骨骼模仿学习与强化学习平台。

本项目基于上游 [MuscleMimic](https://github.com/amathislab/musclemimic)，围绕 MyoFullBody 肌肉驱动人体扩展了一条从动作参考、重定向和质量控制，到分阶段训练、教师—学生蒸馏、低维肌肉协同、真实来球击打，再到 sEMG/生理评估的完整研究链路。

> [!IMPORTANT]
> 这是研究代码，不是即装即用的游戏或临床工具。源码和可移植配置会随仓库发布；动作数据、SMPL/AMASS 资产、真实 sEMG、checkpoint、W&B 记录、视频和训练结果默认不进入 Git。没有对应 artifact 时，代码可验证接口和实验合同，但不能据此声称已经获得训练效果或人体生理结论。

## 项目定位

项目要回答的核心问题是：

1. 如何把视频或人体运动参考可靠地转换为 MyoFullBody 可训练的肌肉骨骼轨迹？
2. 如何让 354 个非手指肌肉执行器完成正手高远球、ChinaJump 和来球击打等复杂动作？
3. 固定低维肌肉协同能否在保持任务能力的同时，提高探索效率、控制紧凑性和可解释性？
4. 如何在 direct 354-D、固定协同和协同加结构化残差之间做真正公平、可复现的比较？
5. 如何把仿真的 excitation、activation、关节、球拍、击球和落点信号，与真实 sEMG 在严格证据合同下对齐？

这个仓库同时关注“能训练”和“能相信”。因此，数据划分、物理控制语义、父 checkpoint、配置哈希、训练恢复、人工视觉验收、promotion gate 和最终科学结论都采用失败关闭设计。

## 我完成了什么

相对于上游 MuscleMimic，本项目的主要工作如下。

| 方向 | 本项目新增或强化的内容 |
|---|---|
| 羽毛球数据链路 | WHAM/SMPL/AMASS 转换、GMR/MyoFullBody 重定向、60→100 Hz cache、轨迹连续性检查、train/validation 防泄漏、版本化 release 和人工视觉 QC。 |
| 羽毛球物理环境 | 标准球场、刚性球拍、羽毛球空气动力学、stringbed/event 击球、来球 feeder、impact target、落点和 recovery 任务。 |
| 分阶段训练 | Stage 1 身体模仿、Stage 2 球拍质量课程、教师 rollout、BC、DAgger、PPO/latent 蒸馏和 Stage 3 来球击打。 |
| 肌肉动作表示 | 公平的 <code>full_354</code>、<code>fixed_synergy</code>、<code>fixed_synergy_residual</code> 三种控制坐标，以及跨阶段可验证的 <code>BodySynergyContractV2</code>。 |
| 肌肉协同分析 | physical excitation/activation 分离、global/regional/hybrid NMF、held-out VAF、初始化/分半/bootstrap/cross-trial 稳定性、rank gate 和结构化残差拟合。 |
| ChinaJump 研究线 | primitive rollout 入库、目标动作隔离、bootstrap/formal readiness、static/dynamic coverage、ASI 成对实验和 640M Stage-1 配置。 |
| 策略蒸馏与 LAB | teacher provenance、不可变数据集、精确 transition 预算、BC/DAgger、低维 latent policy、闭环评估和 causal intervention artifact。 |
| 训练可靠性 | Hydra 配置哈希、唯一 run ID、Orbax 完整性检查、自动恢复、父 checkpoint lineage、W&B step 连续性、append-only 日志和 promotion manifest。 |
| 生理约束基础 | 正确的 excitation v2 控制语义、activation address 读取、肌肉 taxonomy、intra-muscle/IMR 诊断和饱和/裁剪审计。 |
| 真实 sEMG | 独立的 16 通道 Delsys Trigno 采集与预处理工具、MVC、事件证据审计、全有或全无导入，以及 paired/unpaired 两类评估。 |

## 整体系统

~~~mermaid
flowchart LR
    A["视频 / WHAM / SMPL / AMASS"] --> B["GMR 重定向与轨迹 QC"]
    B --> C["Stage 1<br/>身体轨迹模仿"]
    P["Primitive physical rollouts"] --> Q["NMF: W / R / coefficient stats"]
    Q --> C
    C --> D["Stage 2<br/>球拍质量 25% → 50% → 75% → 100%"]
    D --> E["Teacher rollout"]
    E --> F["BC / DAgger / latent distillation"]
    F --> G["Stage 3<br/>来球、impact、落点、recovery"]
    G --> H["任务指标与因果干预"]
    G --> I["仿真 excitation / activation / joint / racket signals"]
    J["Delsys 16-channel sEMG"] --> K["MVC、QC、事件审计、strict import"]
    I --> L["paired 或 independent-cohort 评估"]
    K --> L
~~~

主线由多个可独立审计的阶段组成。一个阶段通过，并不自动证明下一个阶段成立；每次晋级都必须携带原始指标、配置、数据和 checkpoint 的内容指纹。

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

- 354 个 body muscle runtime 必须具有物理 <code>ctrlrange=[0,1]</code>。
- policy ABI 仍保持 <code>[-1,1]</code>，便于 PPO、BC 和 checkpoint 接口统一。
- raw <code>data.ctrl</code> 原样保留；不能再把旧的 ctrlrange 仿射坐标称为 physical excitation。
- activation 必须通过 <code>model.actuator_actadr</code> 读取 <code>data.act</code>，不能把 actuator ID 当作 activation-state 地址。
- excitation 与 activation 分开拟合、分开报告；NMF 不接收归一化后的有符号 policy action。

旧 physical dataset、basis、decoder 或 checkpoint 如果不满足 v2 合同，会被拒绝，而不是按 shape 静默恢复。详细合同见 [肌肉生理约束实施契约 v2](docs/肌肉生理约束实施契约_v2.md)。

### 2. 三种可比较的动作表示

同一成对实验中，三种模式共享相同的 354 个有序 actuator、观测、奖励、终止条件、数据、球拍模型和评估样本，只改变策略输出坐标。

| 模式 | 策略输出 | 最终物理输出 | 用途 |
|---|---:|---:|---|
| <code>full_354</code> | 354 | 354 | 独立端到端 direct 基线。 |
| <code>fixed_synergy</code> | <code>rank(W)</code> | 354 | 固定非负协同字典、bounded coefficient 和 tonic baseline。 |
| <code>fixed_synergy_residual</code> | <code>rank(W)+rank(R)</code> | 354 | 固定协同加小维、有 mask 的结构化残差。 |

冻结解码器的统一形式为：

~~~text
c   = cmax * sigmoid(raw_c / temperature + logit(center / cmax))
rho = alpha * tanh(raw_rho)
u   = clip(tonic + W c + R rho, excitation_bounds)
a   = physical_to_normalized(u)
~~~

正式协同模型不允许 learned 354-D baseline 或 354-D residual 绕过 <code>W</code>。动作 ABI、<code>W/R</code>、coefficient 变换、tonic、bounds、actuator 顺序、模型绑定和 coverage 证据会进入版本化合同。详见 [354 维动作模式与刚性球拍主线](docs/body_action_modes_and_rigid_racket.md)。

### 3. 正手高远球的阶段化训练

| 阶段 | 学习目标 | 关键输入 | 晋级证据 |
|---|---|---|---|
| Stage 0 | 数据 release、轨迹 QC、event/reference bank | 22 条 train、5 条 held-out validation 的本地 retarget cache | release manifest、数值 QC、人工视觉 QC、内容哈希 |
| Stage 1 | 徒手或统一动作接口下的全身轨迹模仿 | canonical <code>raw_smooth_v1</code> | validation 指标、连续 gate、5 条不同 validation 视频 |
| Stage 2 | 在保持挥拍质量的同时适应真实球拍惯量 | Stage-1 promoted checkpoint | 25%→50%→75%→100% 球拍质量、逐档新 run、新 optimizer、parent lineage |
| Direct distill | 压缩完整 lookahead teacher | promoted Stage-2 teacher rollout | BC、DAgger、held-out rollout comparison、promotion evidence |
| Latent distill | 学习低维 muscle policy / LAB prior | 相同 teacher physical rollout | train-only normalizer、motion-level split、闭环 gate、decoder contract |
| Stage 3 | 来球击打、过网、落点和恢复 | frozen body/latent policy、固定 feed/target bank | preflight、base-only、feed gate、128-feed held-out evaluation |
| Phase 3 physiology | 与真实 sEMG 或生理指标比较 | Stage-3 physical signal、strict EMG artifact | paired 或 unpaired 设计、mapping/taxonomy/policy evidence |

默认 <code>legacy_v2</code> 保持原 Stage 1→Stage 2→蒸馏→Stage 3 顺序；<code>synergy_v3</code> 是显式 opt-in 的研究 profile，不会被默认流程自动启用。

### 4. 当前球拍的物理语义

生产 Stage-2/Stage-3 使用 rigid-tool baseline：

- 球拍作为右手 <code>thirdmc_r</code> 的 jointless exact child；
- 没有球拍 free joint，也没有 hand–racket weld/equality；
- 生产 354-D 场景没有手指 joint、actuator、hand policy 或 finger observation；
- 保留 racket–shuttle 接触和 event rebound；
- 球拍的质量与惯量由肩、肘、前臂和手腕的肌肉链承担；
- Stage-2 与 Stage-3 复用同一个版本化 attachment contract。

这可以研究“身体如何驱动刚性工具”，但不能称为 learned physical grip，也不能支持握力、拍柄滑移或手指协同结论。

## 当前实现状态与结论边界

| 模块 | 当前状态 | 可以声称什么 |
|---|---|---|
| JAX/MJX/Warp 全身训练 | 已接通并有 source/asset 测试 | 可以进行大规模并行肌肉骨骼训练；性能仍依赖具体 GPU、数据和配置。 |
| Forehand Clear 生产合同 | 数据、训练、恢复、gate 和 pipeline 代码已实现 | 可以生成和验证完整实验链；本仓库不随源码发布训练数据或最终 benchmark。 |
| Synergy v3 | action contract、NMF、sweep、closed-loop 和 causal artifact 已实现 | 可以执行预注册比较；没有 sealed artifact 时不能宣称协同比 direct 更好。 |
| ChinaJump early synergy | bootstrap/formal release 和公平配置已实现 | 可以比较 full-354、fixed-W、W+R 及 ASI；bootstrap 通过不等于 formal target coverage 通过。 |
| 生理/IMR | Phase 0–1 安全基础和诊断已实现 | 可以审计 excitation、activation、taxonomy、饱和和 intra-muscle 指标；当前没有默认启用未经验证的生理 reward。 |
| Jidian sEMG | 采集、MVC、预处理、事件审计、strict import、paired/unpaired evaluator 已接通 | 按 2026-07-26 审计，已有 98 个 raw trial，但 evidence-backed impact 和 official eligible trial 都是 0，因此尚无正式 Phase-3 结果。 |

当前 sEMG profile 采集 16 个通道；模型比较只使用有显式同源映射的 S2–S16，共 15 个通道。S1 上斜方肌保留在采集与 QC 中，但不会被猜测映射到 MyoFullBody actuator。现有 mapping 标记为 provisional，显式放行后也只能生成 exploratory report。完整边界见 [Jidian sEMG 集成合同](docs/jidian_emg_integration.md)。

## 仓库结构

~~~text
musclemimic/
├── musclemimic/
│   ├── algorithms/          # PPO、网络、推理和训练公共组件
│   ├── badminton/           # 数据、event、promotion、Stage-3 和专项脚本
│   ├── core/                # 奖励、terminal、wrapper、MJX 公共逻辑
│   ├── distill/             # teacher dataset、BC/DAgger、provenance
│   ├── environments/        # MyoFullBody 和专项肌肉骨骼环境
│   ├── latent_muscle/       # latent/LAB、decoder、闭环和因果分析
│   ├── synergy/             # NMF、basis、primitive、coverage 和 action contract
│   ├── physiology/          # taxonomy、effective excitation、intra-muscle 指标
│   ├── evaluation/          # EMG、cohort、physiology 和 Stage-3 signal export
│   └── runner/              # 训练引擎、checkpoint、自动恢复和日志
├── environment/
│   ├── court/               # BWF 球场几何与 MJCF
│   ├── racket/              # 刚性球拍、stringbed 和参数验证
│   ├── shuttlecock/         # 空气动力学与拍球碰撞
│   └── overall_environment/ # 完整来球击打场景与 CPU/MJX 环境
├── fullbody/                # Hydra 训练入口、蒸馏、latent 和流程规划器
├── bimanual/                # 上游双臂训练/评估入口
├── loco_mujoco/             # 模型、数据加载、SMPL/GMR 和环境基座
├── analysis/                # latent/synergy 表征和干预分析
├── configs/                 # 环境绑定、公开 JSON 模板、physiology 合同
├── experiments/             # Stage-3/post-train 与 synergy 实验定义
├── jidian_measurement/      # 独立的 Delsys Trigno 采集与预处理子项目
├── src/grip/                # 独立 CPU 手指握拍原型与几何验证
├── rl_training_environment/ # 球拍/握拍场景的离线渲染辅助工具
├── assets/                  # 仓库级图片与可发布场景资产
├── scripts/                 # 生产 launcher、CUDA compat 和数据工具
├── tests/                   # source-only、unit、integration 和资产测试
├── Makefile
├── pyproject.toml
└── uv.lock
~~~

## 环境要求

生产 GPU 训练的推荐环境：

- Linux x86_64；
- Python 3.11；
- NVIDIA GPU 与兼容驱动；
- <code>uv</code>；
- <code>git</code>、<code>wget</code> 和 <code>bsdtar</code>；
- 可选的 W&B 账号；
- 合法获取的 SMPL-H/AMASS 等外部资产。

CPU 工具、source-only 测试和部分分析可以在其他系统运行，但本项目不把 macOS 或 Windows 视为生产 GPU 训练平台。Delsys Trigno 采集子项目通常运行在装有 Trigno Control Utility 的 Windows 主机。

## 安装

### 1. Clone

~~~bash
git clone https://github.com/yangfeiyang-123/asi_strengthen_musclemimic.git
cd asi_strengthen_musclemimic
~~~

### 2. 安装依赖

仅源码、CPU 工具和开发测试：

~~~bash
uv sync --locked --extra dev
~~~

Linux CUDA 训练环境：

~~~bash
uv sync --locked --extra dev --extra cuda
~~~

还需要 SMPL/GMR 重定向时：

~~~bash
uv sync --locked --extra dev --extra cuda --extra smpl --extra gmr
~~~

<code>smpl</code> 和 <code>gmr</code> extra 含外部 Git 依赖；SMPL 模型文件本身不会通过 Python 包自动下载。

### 3. 验证干净源码

~~~bash
make source-only
make lint
~~~

快速检查公开 CLI：

~~~bash
uv run --locked forehand-clear-three-stage --help
uv run --locked musclemimic-chinajump-synergy-pipeline --help
uv run --locked musclemimic-emg-eval --dry-run
uv run --locked musclemimic-physiology-eval --dry-run
~~~

## 本地数据、模型与环境变量

### 推荐目录

~~~text
datasets/
├── _global/
│   ├── amass_npz/
│   └── muscle_trajectory/gmr_cache/
└── <action>/
    ├── muscle_trajectory/<variant>/*.npz
    ├── manifests/
    ├── distill/
    └── training/
smpl_models/
└── smplh/
outputs/
wandb/
~~~

这些目录默认受 <code>.gitignore</code> 保护。

### 统一环境绑定

从仓库根目录运行：

~~~bash
source configs/env.sh
~~~

主要变量如下。

| 变量 | 默认位置 | 作用 |
|---|---|---|
| <code>MUSCLEMIMIC_DATASETS_ROOT</code> | <code>datasets/</code> | 所有本地动作和训练 artifact 的根目录 |
| <code>MUSCLEMIMIC_AMASS_PATH</code> | <code>datasets/_global/amass_npz</code> | AMASS-style 输入 |
| <code>MUSCLEMIMIC_CONVERTED_AMASS_PATH</code> | <code>datasets/_global/muscle_trajectory/gmr_cache</code> | 通用 GMR cache |
| <code>MUSCLEMIMIC_GMR_CACHE_PATH</code> | <code>datasets/</code> | 按 action 直读已重定向轨迹 |
| <code>MUSCLEMIMIC_SMPL_MODEL_PATH</code> | <code>smpl_models/smplh</code> | 授权 SMPL-H 模型 |
| <code>MM_CUDA_COMPAT_ROOT</code> | <code>.local/cuda-compat-12.4</code> | 用户态 CUDA compatibility library |

正式训练 launcher 会自动 source 此文件。其他重定向、QC、评估和数据工具如果依赖本地资产，应在同一 shell 中先 source。

## 正式训练

### 生产启动合同

所有本地 fullbody 生产训练必须从仓库根目录通过 [scripts/run_fullbody_training.sh](scripts/run_fullbody_training.sh) 启动。不要直接运行 <code>python fullbody/experiment.py</code>、<code>.venv/bin/python</code> 或 <code>uv run fullbody/experiment.py</code>。

启动前必须显式设置：

~~~bash
export CUDA_VISIBLE_DEVICES=0
export MUSCLEMIMIC_JAX_CACHE_KEY=forehand_stage1
export MUSCLEMIMIC_TRAIN_LOG=datasets/forehandClear_standard/training/logs/forehand_stage1.log
export JAX_COMPILATION_CACHE_DIR=/data3/yangfeiyang/WorkSpace/ENV/jax-cache/forehand_stage1
~~~

在其他服务器上，<code>JAX_COMPILATION_CACHE_DIR</code> 应改为该机器上按任务隔离、可写且空间充足的路径。

### 1. 只检查 launcher 环境

~~~bash
MUSCLEMIMIC_DRY_RUN=1 scripts/run_fullbody_training.sh \
  --config-name=config_specific_task/stage1_body/conf_fullbody_forehand_clear_body_local \
  wandb.mode=disabled
~~~

### 2. 解析 Hydra 配置但不训练

~~~bash
scripts/run_fullbody_training.sh \
  --config-name=config_specific_task/stage1_body/conf_fullbody_forehand_clear_body_local \
  --cfg job --resolve \
  wandb.mode=disabled
~~~

在 resolved config 中至少核对：

- <code>total_timesteps</code>；
- 唯一 <code>run_id</code>；
- train/validation motion split；
- reward 权重；
- terminal 类型和阈值；
- promotion 行为；
- action representation 和父 checkpoint lineage。

### 3. 启动生产训练

~~~bash
scripts/run_fullbody_training.sh \
  --config-name=config_specific_task/stage1_body/conf_fullbody_forehand_clear_body_local \
  wandb.mode=online
~~~

launcher 会：

- 自动 source <code>configs/env.sh</code>；
- 限定一个物理 GPU；
- 设置 Orbax save/restore 并发预算；
- 使用任务独立的 JAX compilation cache；
- 通过 CUDA compatibility wrapper 调用锁定的 uv 环境；
- 把 stdout/stderr 追加到同一个日志文件。

### ChinaJump 当前工作站示例

~~~bash
export CUDA_VISIBLE_DEVICES=2
export MUSCLEMIMIC_JAX_CACHE_KEY=chinajump_stage1
export MUSCLEMIMIC_TRAIN_LOG=datasets/ChinaJump/training/logs/chinajump_root_control_v2_stage1_body_640m.log
export JAX_COMPILATION_CACHE_DIR=/data3/yangfeiyang/WorkSpace/ENV/jax-cache/chinajump_stage1

scripts/run_fullbody_training.sh \
  --config-name=config_specific_task/stage1_body/conf_fullbody_chinajump_root_control_v2 \
  wandb.mode=online
~~~

### 长任务与停止

当前工作站的 ChinaJump 长任务使用显式 tmux socket：

~~~bash
tmux -S /data3/yangfeiyang/tmp/tmux_chinajump.sock new-session -d \
  -s chinajump_root_v2_640m \
  -c /data3/yangfeiyang/WorkSpace/musclemimic
~~~

不要复用显示 <code>Pane is dead</code> 的 pane。停止训练时只向对应 pane 发送一次 Ctrl-C，等待 Python PID 和 CUDA context 消失，并保留最新的 finalized checkpoint。

完整的本机启动合同见 [AGENTS.md](AGENTS.md)。

### 启动前后必须检查

启动前：

1. 运行与本次 config、reward、terminal 和 action contract 对应的 focused tests。
2. 完整解析 Hydra 配置。
3. 使用 <code>nvidia-smi</code> 检查物理 GPU 上已有进程。
4. reward、termination 或 action contract 改变时，使用新 run ID 和 fresh optimizer，不恢复不兼容 checkpoint。

启动后，只有以下条件全部满足才算成功：

1. 日志中的 train/validation trajectory 全部从已有本地 retarget 文件加载；出现 Hugging Face 下载尝试通常说明环境绑定错误，应停止。
2. checkpoint run manifest 存在，并记录预期 config hash、run ID、训练预算、promotion、reward 和 terminal。
3. W&B 已显示 live run ID 和 URL。
4. <code>nvidia-smi</code> 在预期物理 GPU 上显示新 Python PID。
5. 日志到达 <code>Starting training...</code>，且没有 fatal traceback。

## 主要工作流

### 1. 动作重定向与数据 QC

查看重定向和可视化入口：

~~~bash
uv run --locked python -m musclemimic.badminton.scripts.run_retarget --help
uv run --locked python -m musclemimic.badminton.scripts.render_retarget_cache --help
uv run --locked forehand-clear-data-qc --help
uv run --locked forehand-clear-visual-review --help
~~~

canonical Forehand Clear 数据采用 22 条 train、5 条 held-out validation，source 为 60 Hz，环境 cache 为 100 Hz。训练前会重新校验 release、逐文件哈希、split、FPS、修复 recipe 和视觉签字，不能用旧 QC report 绕过数据漂移。

### 2. Forehand Clear pipeline plan

默认主线：

~~~bash
uv run --locked forehand-clear-three-stage \
  --profile legacy_v2 \
  --output_dir outputs/plans/forehand_clear_legacy_v2
~~~

显式协同研究线：

~~~bash
uv run --locked forehand-clear-three-stage \
  --profile synergy_v3 \
  --output_dir outputs/plans/forehand_clear_synergy_v3
~~~

不传 <code>--execute_step</code> 时只写 <code>pipeline_plan.json</code>，不会启动训练。计划中的数据处理和 gate 可以逐步执行；任何 fullbody 长训练仍必须改用前述生产 launcher。

<code>synergy_v3</code> 的权威流程由 plan 生成，阶段为：

1. event/reference cache、train/validation bank 和 QC；
2. promoted 100% racket-mass teacher 的 physical train/validation rollout；
3. 同一 rollout 上的 direct-BC 对照；
4. excitation/activation 的 held-out NMF；
5. <code>2/4/8/16/32 × decoder × seed</code> latent sweep；
6. 每个注册 run 的 Stage-2 closed-loop 与 causal diagnostics；
7. direct/synergy 使用同一 feed、target、seed 的 Stage-3 成对训练与评估；
8. 任务级因果干预与 simulation signal export；
9. EMG/physiology 评估和包含失败 run 的消融报告。

查看核心工具：

~~~bash
uv run --locked musclemimic-physical-rollout-qc --help
uv run --locked musclemimic-synergy-fit --help
uv run --locked musclemimic-latent-synergy-sweep --help
uv run --locked forehand-clear-promotion-gate --help
~~~

### 3. ChinaJump primitive synergy

ChinaJump 流程把 primitive source、NMF、coverage 和训练 release 分开：

~~~bash
uv run --locked musclemimic-chinajump-synergy-pipeline --help

uv run --locked musclemimic-chinajump-synergy-pipeline plan \
  --primitive-catalog artifacts/primitive_rollouts/catalog.json \
  --readiness bootstrap \
  --output-root artifacts/primitive_synergy/chinajump_bootstrap
~~~

- <code>B0/B1</code> 是 primitive bootstrap，分别关闭/打开 ASI。
- <code>S0/S1</code> 还要求与 primitive 独立的 ChinaJump target-control coverage。
- <code>full354</code>、fixed <code>W</code> 和 fixed <code>W+R</code> 使用不同 run ID 和 fresh optimizer。
- action representation 之外的数据、reward、terminal、预算和 ASI 开关保持成对一致。

完整流程见 [ChinaJump Primitive 协同 runbook](docs/chinajump_primitive_synergy_runbook.md) 和 [Stage-1 早期协同说明](docs/stage1_early_synergy.md)。

### 4. Stage-3 来球击打

Stage-3 v2 同时提供 direct full-354 和 latent/synergy 两个正式入口：

- [incoming_shuttle_hit_full354_v1.yaml](experiments/posttrain/incoming_shuttle_hit_full354_v1.yaml)：354-D 端到端 direct；
- [incoming_shuttle_hit_impact_recovery_v2.yaml](experiments/posttrain/incoming_shuttle_hit_impact_recovery_v2.yaml)：固定 decoder/latent 路径。

二者共享 scene、train/eval feed、impact/recovery target、reward、episode 和刚性球拍合同。先运行非训练前置检查：

~~~bash
uv run --locked python -m musclemimic.badminton.scripts.run_incoming_shuttle_hit \
  --spec experiments/posttrain/incoming_shuttle_hit_impact_recovery_v2.yaml \
  --stage preflight \
  --out-dir outputs/stage3_preflight
~~~

随后还必须通过 <code>base-only-check</code> 和 <code>feed-check</code>。runner 支持 <code>train-gpu</code> 和 <code>evaluate</code>，但生产训练必须服从 [AGENTS.md](AGENTS.md) 的 GPU、日志、环境和恢复合同，不能把一个 preflight 通过误报为训练成功。

### 5. Teacher–student 与 latent policy

~~~bash
uv run --locked musclemimic-distill-collect-teacher --help
uv run --locked musclemimic-distill-train-bc --help
uv run --locked musclemimic-distill-run-dagger --help
uv run --locked musclemimic-distill-compare --help
uv run --locked musclemimic-latent-train --help
uv run --locked musclemimic-latent-closed-loop-eval --help
~~~

生产 collection 只接受 promoted teacher，并写入不可变 <code>dataset_manifest.json</code>。manifest 固定 teacher checkpoint、motion split、shard 集合、样本数和内容 SHA-256。DAgger iteration 使用 staging 和幂等提交，防止重跑时重复 append。

### 6. Jidian 16 通道 sEMG

采集子项目位于 [jidian_measurement](jidian_measurement/README.md)，具有独立依赖和测试：

~~~bash
cd jidian_measurement
uv run --extra test pytest
uv run python -m emg.cli --help
~~~

它覆盖：

- 版本化 16 通道 sensor profile；
- Delsys Trigno TCP 采集；
- MVC、动作协议、休息和断点恢复；
- 原始数据非破坏性保存；
- band-pass、notch、rectify、envelope 和 MVC normalization；
- preprocessing QC；
- 带 evidence SHA-256 的人工事件补标；
- NMF 协同提取和跨 trial 稳定性。

进入主项目前，必须使用显式 selection manifest 做全有或全无导入：

~~~bash
uv run --locked musclemimic-jidian-emg-import \
  outputs/emg/jidian_selection.json \
  outputs/emg/jidian_strict.npz \
  --audit-json outputs/emg/jidian_import.audit.json
~~~

正式 importer 只接受 <code>badminton_synergy_16_v2</code>、MVC-normalized envelope、<code>preprocessing_qc.analysis_ready=true</code> 的 trial，以及带独立 <code>evidence_sha256</code> 审计的真实 <code>racket_contact</code>。任一 trial 不合格时，只写 rejection audit，不保留部分 NPZ。

仿真侧必须在导出前选择并填写对应 identity：

- 默认 independent cohort：[stage3_signal_trial_identity_template.json](configs/public/stage3_signal_trial_identity_template.json)；
- 真正逐 trial paired：[stage3_signal_trial_identity_paired_template.json](configs/public/stage3_signal_trial_identity_paired_template.json)；
- 当前唯一的 16→15 observation mapping：[emg_badminton_synergy_16_v2_myofullbody_observation_v1.json](configs/physiology/emg_badminton_synergy_16_v2_myofullbody_observation_v1.json)。

比较设计必须在采集/导入前固定。

| 设计 | 逐 trial reference 必须相同 | 允许报告 |
|---|---:|---|
| <code>paired_same_reference_v1</code> | 是 | envelope correlation、DTW、onset/peak timing、paired NMF/phase |
| <code>unpaired_action_cohort_v1</code> | 否 | 两个 cohort 分别 NMF 后的 W geometry、matching 和各自 VAF |

“动作名称相同”不等于 paired。没有完全一致的 <code>reference_trial_fingerprint</code> 时，禁止报告逐 trial 或 H/时序指标。

输入合同可在无数据时检查：

~~~bash
uv run --locked musclemimic-emg-eval --dry-run
uv run --locked musclemimic-emg-cohort-eval --dry-run
uv run --locked musclemimic-physiology-eval --dry-run
~~~

### 7. 附加与历史模块

- <code>src/grip/</code> 提供独立 CPU MuJoCo 手指握拍建模、静态姿态求解、seed、训练和验收工具。它用于几何实验和历史 grip 研究，不是当前 354-D exact-child 生产主线。
- <code>bimanual/</code> 保留上游双臂训练与评估入口，便于回归通用 MuscleMimic 能力。
- <code>forehand-clear-rag-export-csv</code> 可把 rollout 导出为外部 BadmintonRAG 所需的关节/肌肉时序表。
- <code>rl_training_environment/</code> 用于球拍、手指和挥拍场景的离线视角渲染，不承担生产 PPO。
- legacy/negative-ablation 配置仍可用于历史复现，但默认不能进入 production promotion。

## 配置导航

| 路径 | 用途 |
|---|---|
| <code>fullbody/conf_fullbody.yaml</code> | 通用全身 PPO/MJX 基座 |
| <code>fullbody/config_specific_task/base/</code> | badminton body/racket 共享配置 |
| <code>fullbody/config_specific_task/stage1_body/</code> | Forehand、ChinaJump 和 action-mode Stage-1 |
| <code>fullbody/config_specific_task/stage2_racket_v2/</code> | 25/50/75/100% 球拍质量课程 |
| <code>fullbody/config_specific_task/distill/</code> | direct student、latent/LAB 和 synergy v3 |
| <code>experiments/posttrain/</code> | static hit、incoming hit 和 Stage-3 v2 |
| <code>configs/public/</code> | event、causal、signal identity 和 physiology 模板 |
| <code>configs/physiology/</code> | 354-muscle taxonomy 与 Jidian mapping |
| <code>experiments/synergy/</code> | 354 actuator 的 region grouping |

Hydra 的 <code>--config-name</code> 相对于 <code>fullbody/</code>，不带 <code>.yaml</code>。标记为 legacy/experimental/nonproduction 的配置默认失败关闭；隔离复现实验必须显式 opt-in，并且其 artifact 不能进入生产 promotion。

## Artifact 与可复现性

| Artifact | 作用 |
|---|---|
| <code>run_manifest.json</code> | 固定 config hash、run ID、训练预算、reward、terminal、promotion 和 action contract |
| <code>promotion_progress.json</code> | 保存每次 validation、连续通过次数和实际停止位置 |
| <code>dataset_manifest.json</code> | 固定 teacher、split、collection、shard、样本数和内容哈希 |
| <code>BodySynergyContractV2</code> | 固定 portable decoder core 与 stage runtime binding |
| <code>pipeline_plan.json</code> | 固定 profile、步骤、命令和所需 artifact |
| <code>preflight_report.json</code> / <code>feed_check_report.json</code> | Stage-3 场景、prior、feed 和 target 前置证据 |
| promotion manifest | 把 checkpoint、指标、人工 review 和 parent lineage 封装为可验证晋级证据 |
| EMG import audit | 记录 selection 的接受或拒绝；失败时不留下部分 NPZ |

关键原则：

- train 与 validation 按 motion/trial 隔离，不从训练 motion 随机切 validation frame；
- 配置和数据内容改变时，旧 checkpoint 不按 shape 静默恢复；
- 跨阶段初始化允许 portable decoder core 一致，但同一 optimizer 恢复要求 exact runtime 一致；
- promotion 读取原始指标和内容哈希，不接受手写 <code>passed: true</code>；
- 失败 run 也保留在消融报告中，避免只报告成功实验。

## 测试与开发

~~~bash
make help
make install-dev
make lint
make source-only
make test
make ci
~~~

- <code>make ci</code> 运行 scoped Ruff 和 clean-clone source contract。
- <code>make source-only</code> 不需要数据、checkpoint、SMPL 或 GPU。
- <code>make asset-test</code> 面向已经准备 MuJoCo/model/data 资产的 runner。
- <code>tests/integration/</code> 可能需要模型、图形后端或 GPU。
- <code>jidian_measurement</code> 在自己的目录内运行独立测试。

生产训练前不要只运行全局 smoke test；还要执行与本次 reward、terminal、checkpoint contract 和目标 config 对应的 focused tests。

## 数据、隐私与发布策略

仓库不会发布：

- WHAM、AMASS、SMPL-H/MANO 等受许可约束的原始资产；
- 本地 retarget cache 和动作视频；
- 真实受试者身份、原始 sEMG 或可识别证据视频；
- checkpoint、W&B、训练日志和大体积评估输出；
- 机器专用路径、token 和私密配置。

公开 JSON/YAML 只提供 schema、模板和可移植实验合同。真实 artifact 应使用伪名 subject/session UID、内容哈希和独立私有存储；不要把姓名、设备凭据或原始证据路径提交到 Git。

## 已知限制

1. 数据和 checkpoint 不随源码发布，clone 后不能直接复现最终策略。
2. 生产 GPU 栈主要面向 Linux x86_64 + NVIDIA；其他平台仅覆盖部分工具。
3. 当前球拍是 exact-child rigid tool，不是可滑移、可摩擦的真实手指抓握。
4. 当前 shuttle event 模型重点闭合线动量；不能把它解释为完整的羽毛球旋转碰撞辨识。
5. provisional 15-channel sEMG mapping 不是对 354 个 actuator 的全身人体效度验证。
6. 当前 evidence-backed impact 数为 0，因此没有正式 paired/unpaired EMG 结论。
7. intra-muscle/IMR 当前以诊断和审计为主，未经验证的 hard physiological constraint 没有默认进入训练 reward；训练侧不保留 IMR 指标字段。checked-in taxonomy 的合法 hard group 是空集，因此所有 IMR loss 恒为 0，必须先读报告里的 `coverage` 才能区分"未测量"和"已测量且一致"。
8. 任何实验结果都必须与具体 commit、resolved config、数据 release、seed、checkpoint 和 promotion artifact 一起解释。

## 进一步文档

| 文档 | 内容 |
|---|---|
| [AGENTS.md](AGENTS.md) | 本机生产训练启动、tmux、preflight 和成功判据 |
| [354 维动作模式与刚性球拍主线](docs/body_action_modes_and_rigid_racket.md) | full-354、fixed synergy、residual、跨阶段合同和 rigid-tool 语义 |
| [ChinaJump Primitive 协同](docs/chinajump_primitive_synergy_runbook.md) | primitive 采集、NMF、coverage、bootstrap/formal release |
| [ChinaJump Stage-1 早期协同](docs/stage1_early_synergy.md) | 公平实验矩阵、action wrapper、配置和结论规则 |
| [肌肉生理约束实施契约 v2](docs/肌肉生理约束实施契约_v2.md) | excitation/activation、taxonomy、IMR 和迁移边界 |
| [Jidian sEMG 严格集成合同](docs/jidian_emg_integration.md) | 16→15 channel mapping、事件证据、strict import、paired/unpaired 评估 |
| [Jidian 采集工具](jidian_measurement/README.md) | 现场采集、MVC、QC、预处理和 NMF 操作手册 |

## 上游、许可与引用

- 上游项目：[amathislab/musclemimic](https://github.com/amathislab/musclemimic)
- 本项目仓库：[yangfeiyang-123/asi_strengthen_musclemimic](https://github.com/yangfeiyang-123/asi_strengthen_musclemimic)
- 代码许可：[Apache License 2.0](LICENSE)

如果在研究中使用本项目，请同时引用上游 MuscleMimic 工作，并记录本仓库的具体 commit SHA、实验配置和数据/模型来源。第三方模型、人体数据和数据集仍受各自许可、知情同意和伦理要求约束。

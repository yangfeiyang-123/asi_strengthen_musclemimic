# MuscleMimic 上手与研究开发流程

这份文档的目标不是重复 `README.md`，而是给出一条更容易照着执行的路线：

1. 先把环境部署起来。
2. 用最小 demo 跑通一次训练和评估。
3. 明白数据、配置、检查点、日志分别在哪里。
4. 在现有 PPO 流程上开展自己的研究。
5. 最后把训练出的策略用于自己的展示、评估和部署。

---

## 1. 先理解这个仓库在做什么

`musclemimic` 是一个基于 JAX + MuJoCo/MJX 的肌肉骨骼模仿学习项目。你可以把它理解成下面这条链路：

`动作数据 -> retarget 到肌肉骨骼模型 -> 构造 imitation 环境 -> 用 PPO 训练策略 -> 保存 checkpoint -> 评估 / 可视化 / 二次研究`

在这个仓库里，最重要的入口文件是：

- `fullbody/experiment.py`：全身模型训练入口。
- `bimanual/experiment.py`：双臂模型训练入口。
- `fullbody/eval.py`：全身模型评估、录视频、可视化入口。
- `bimanual/eval.py`：双臂模型评估入口。
- `scripts/retarget_dataset.py`：批量 retarget AMASS 数据。
- `fullbody/conf_*.yaml`、`bimanual/conf_*.yaml`：训练配置。

如果你只准备先跑通全流程，建议优先从 `MyoFullBody` 开始，因为 `README` 和默认配置对它更完整。

---

## 2. 推荐部署方式

### 2.1 推荐硬件和系统

这个项目的实际训练要求比较明确：

- 训练：推荐 `Linux + NVIDIA GPU`。
- 评估和可视化：Linux 更稳，macOS 也支持一部分 CPU MuJoCo 工作流。
- Windows 原生环境不建议直接承担训练。

如果你现在在 Windows 上开发，最稳妥的方式是：

1. 在 Windows 上看代码、改配置、写文档。
2. 在 `WSL2 Ubuntu` 或远程 Linux 服务器上安装环境并训练。
3. 数据缓存、checkpoint、输出目录都尽量放在 Linux 文件系统里。

原因很简单：JAX/CUDA、MJX、MuJoCo、Warp 这一套在 Linux 上最稳定，训练时的问题会少很多。

### 2.2 你至少要准备什么

按工作目标分成三档：

- 只想先跑通 demo：`uv + Python 3.11 + 基础依赖 + Hugging Face 账号`
- 想正式训练：再加 `NVIDIA 驱动 + CUDA 对应 JAX`
- 想自己做数据 retarget：再加 `AMASS + SMPL-H/MANO + smpl/gmr 扩展依赖`

---

## 3. 环境配置

### 3.1 克隆仓库并安装 `uv`

```bash
git clone <your_repo_or_fork_url>
cd musclemimic
```

安装 `uv` 后同步依赖：

```bash
uv sync
```

如果你在 Linux x86_64 上用 NVIDIA GPU 训练，再安装 CUDA 版 JAX：

```bash
uv sync --extra cuda
```

如果你还要做 retarget，再补装：

```bash
uv sync --extra smpl --extra gmr
```

如果你也要做代码开发、测试和格式检查：

```bash
uv sync --extra dev
```

更常见的完整安装方式是：

```bash
uv sync --extra cuda --extra smpl --extra gmr --extra dev
```

### 3.2 最小自检

先确认基础导入没坏：

```bash
make smoke
```

如果你装了开发依赖，也可以运行：

```bash
make test
```

注意：测试通过不代表训练环境完全可用，尤其不代表 GPU/MJX/retarget 链路已经全部可用。

### 3.3 W&B 和 Hugging Face

这个仓库默认会把训练日志打到 Weights & Biases。

如果你不想配置 W&B，可以在运行时显式关闭：

```bash
wandb.mode=disabled
```

另外，demo 数据和官方 checkpoint 走的是 Hugging Face，所以你通常还需要：

```bash
uv run hf auth login
```

---

## 4. 第一件事：先跑通最小 demo

不要一上来就下载 AMASS、配置 SMPL、批量 retarget。先用作者提供的 demo cache 跑通一次，这样你可以先确认：

- 依赖是否安装正确；
- 训练入口是否能启动；
- checkpoint 是否能保存；
- eval 是否能读取模型。

### 4.1 下载 demo cache

全身模型：

```bash
uv run python -c "from musclemimic.utils.demo_cache import setup_demo_for_myo_fullbody; setup_demo_for_myo_fullbody()"
```

双臂模型：

```bash
uv run python -c "from musclemimic.utils.demo_cache import setup_demo_for_bimanual; setup_demo_for_bimanual()"
```

如果这里报 `403 Forbidden` 或 `you are not in the authorized list`，说明当前 Hugging Face 账号还没有拿到这个 gated dataset 的访问权限，或者当前环境还没有执行过：

```bash
uv run hf auth login
```

这些 demo 数据的路径解析现在和 GMR cache 一致，优先级是：

1. 显式传入 `cache_dir`
2. 环境变量 `MUSCLEMIMIC_CONVERTED_AMASS_PATH`
3. `musclemimic-set-all-caches` 写入的配置
4. 默认回退到：

```text
~/.musclemimic/caches/AMASS/
```

如果你希望它们放到当前仓库下，先设置：

### 4.2 固定使用私有 CUDA compat 环境

如果服务器的系统 CUDA / driver 组合偏老，但你不想改动全局环境，可以用仓库内的 wrapper 固定走私有 `cuda-compat-12-4`：

```bash
cd /data3/yangfeiyang/WorkSpace/musclemimic
scripts/run_with_cuda_compat.sh uv run fullbody/experiment.py --config-name=conf_fullbody_demo wandb.mode=disabled
```

默认行为：

1. 在 `/data3/yangfeiyang/WorkSpace/CUDA/12.4` 下缓存 compat RPM 和解包后的 `compat/`
2. 自动把该目录加到 `LD_LIBRARY_PATH` 前面
3. 默认使用 `CUDA_VISIBLE_DEVICES=1`

如果要换 GPU：

```bash
MM_CUDA_VISIBLE_DEVICES=0 scripts/run_with_cuda_compat.sh uv run fullbody/experiment.py --config-name=conf_fullbody_demo wandb.mode=disabled
```

```bash
mkdir -p /data3/yangfeiyang/WorkSpace/musclemimic/caches/AMASS
export MUSCLEMIMIC_CONVERTED_AMASS_PATH=/data3/yangfeiyang/WorkSpace/musclemimic/caches/AMASS
```

### 4.2 跑一个最小训练

全身 demo：

```bash
uv run fullbody/experiment.py --config-name=conf_fullbody_demo wandb.mode=disabled
```

双臂 demo：

```bash
uv run bimanual/experiment.py --config-name=conf_bimanual_demo wandb.mode=disabled
```

你应该把这一步当作“环境验收”。只要它能正常启动、进入训练、周期性写 checkpoint，说明主链路已经通了。

### 4.3 输出目录在哪里

训练时有两类输出要分清：

- Hydra 运行输出：通常在 `outputs/YYYY-MM-DD/HH-MM-SS/`
- 检查点目录：默认在仓库根目录下的 `checkpoints/<experiment_id>/`

这里的 `<experiment_id>` 默认来自配置哈希，所以：

- 同一份配置重复启动时，`auto_resume=true` 会自动续训；
- 你一旦改了关键配置，哈希会变，新的训练通常会进入新的 checkpoint 目录。

如果你想手动控制实验名字和 checkpoint 根目录，建议显式传：

```bash
experiment.run_id=my_first_fullbody
experiment.checkpoint_root=/data/musclemimic/checkpoints
```

例如：

```bash
uv run fullbody/experiment.py --config-name=conf_fullbody_demo \
  wandb.mode=disabled \
  experiment.run_id=my_first_fullbody \
  experiment.checkpoint_root=/data/musclemimic/checkpoints
```

---

## 5. 数据准备有两条路线

你后续做研究，基本就是在这两条路线里选一条。

### 5.1 路线 A：直接用官方已经 retarget 好的数据

这是最推荐的方式，适合：

- 先复现实验；
- 快速训练自己的 baseline；
- 在 PPO、reward、observation、network 结构上做研究。

先设置缓存目录：

```bash
uv run musclemimic-set-all-caches --path /data/musclemimic/caches
```

然后下载 GMR retarget 好的数据缓存。

全身训练集：

```bash
uv run musclemimic-download-gmr-caches --dataset-group KIT_KINESIS_TRAINING_MOTIONS
```

双臂训练集：

```bash
uv run musclemimic-download-gmr-caches --dataset-group AMASS_BIMANUAL_TRAIN_MOTIONS --env-name MyoBimanualArm
```

之后直接使用对应的 `gmr` 配置训练即可。

### 5.2 路线 B：自己下载 AMASS 并做 retarget

适合：

- 你要加入新的动作数据；
- 你要研究 retarget 质量；
- 你要控制 retarget 参数，而不是只使用现成缓存。

#### 第一步：下载 AMASS

从 AMASS 官网注册并下载，假设你放到：

```text
/data/amass
```

目录结构大致应是：

```text
/data/amass/
  ACCAD/
  KIT/
  ...
```

#### 第二步：下载 SMPL-H 和 MANO

从 MANO/SMPL 相关网站下载后，假设放到：

```text
/data/smpl
```

需要至少包含：

```text
/data/smpl/
  mano_v1_2/
  smplh/
```

#### 第三步：配置路径

```bash
uv run musclemimic-set-amass-path --path /data/amass
uv run musclemimic-set-smpl-model-path --path /data/smpl
uv run musclemimic-set-all-caches --path /data/musclemimic/caches
```

这些命令会把路径写入用户配置，默认位置一般是：

```text
~/.musclemimic/MUSCLEMIMIC_VARIABLES.yaml
```

#### 第四步：生成 `SMPLH_neutral.pkl`

```bash
cd loco_mujoco/smpl
bash install_smplh.sh
```

#### 第五步：批量 retarget

全身模型：

```bash
uv run scripts/retarget_dataset.py \
  --model MyoFullBody \
  --retargeting-method gmr \
  --dataset KIT_KINESIS_TRAINING_MOTIONS \
  --workers 8
```

双臂模型：

```bash
uv run scripts/retarget_dataset.py \
  --model MyoBimanualArm \
  --retargeting-method gmr \
  --dataset AMASS_BIMANUAL_MARGINAL_MOTIONS \
  --workers 8
```

这一步完成后，你自己的缓存就和官方提供的 GMR cache 一样，后续训练只需要读取缓存，不需要每次重新 retarget。

---

## 6. 配置文件怎么读

这一节很重要，因为你后续的大多数研究工作，本质上都是“改配置 + 少量改代码”。

### 6.1 训练入口

训练入口非常简单：

- 全身：`fullbody/experiment.py`
- 双臂：`bimanual/experiment.py`

它们都调用同一个训练引擎 `musclemimic.runner.engine.run_experiment`。

### 6.2 你最常改的配置项

先看 `experiment` 下面这些块：

- `task_factory.params.amass_dataset_conf`
  作用：决定训练用哪些动作数据，是否使用 `gmr` retarget。
- `env_params`
  作用：决定环境类型、并行环境数、观测项、奖励、终止条件。
- `actor_hidden_layers` / `critic_hidden_layers`
  作用：决定策略网络和价值网络结构。
- `ppo_config`
  作用：决定 PPO 的 rollout 长度、minibatch、GAE、clip、std 等。
- `validation`
  作用：决定验证集、验证频率、验证指标和视频录制。
- `resume_from`
  作用：从已有 checkpoint 或 Hugging Face checkpoint 继续训练。

### 6.3 最关键的几类配置

#### 数据配置

用整组数据：

```yaml
experiment:
  task_factory:
    params:
      amass_dataset_conf:
        dataset_group: "KIT_KINESIS_TRAINING_MOTIONS"
```

只用几个指定动作：

```yaml
experiment:
  task_factory:
    params:
      amass_dataset_conf:
        dataset_group: null
        rel_dataset_path:
          - "KIT/314/walking_medium09_poses"
          - "KIT/348/turn_right03_poses"
```

#### 环境并行度

```yaml
experiment:
  env_params:
    num_envs: 2048
```

这是最敏感的吞吐参数之一。GPU 显存不够、JAX 编译太慢、训练刚启动就 OOM，优先先降它。

#### 奖励和终止条件

全身配置里你最应该先理解的是：

- `reward_params`
- `terminal_state_type`
- `terminal_state_params`

如果策略刚开始就频繁摔倒、训练不稳定、验证指标很差，先不要急着改 PPO，先看奖励权重和 early termination 阈值是不是过于激进。

#### PPO 超参数

最重要的是：

```yaml
experiment:
  total_timesteps: ...
  ppo_config:
    num_steps: ...
    update_epochs: ...
    num_minibatches: ...
    gamma: ...
    gae_lambda: ...
    clip_eps: ...
    init_std: ...
```

其中有个关系要记住：

```text
num_updates = total_timesteps / (num_envs * num_steps)
```

也就是说，当你增加 `num_envs` 或 `num_steps` 时，每次更新采样到的数据会更多，但总更新次数会变少。

---

## 7. 先学会评估和看结果

训练不是只看 loss。你至少要学会三件事：

1. 找到最新 checkpoint。
2. 用 eval 脚本跑回放。
3. 跑验证指标并导出视频。

### 7.1 用 checkpoint 做可视化

假设你已经有一个 checkpoint：

```text
checkpoints/my_first_fullbody/checkpoint_400
```

用 MuJoCo viewer 回放：

```bash
uv run python fullbody/eval.py \
  --path checkpoints/my_first_fullbody/checkpoint_400 \
  --use_mujoco \
  --mujoco_viewer \
  --stochastic \
  --n_steps 1000
```

如果想指定某个 motion：

```bash
uv run python fullbody/eval.py \
  --path checkpoints/my_first_fullbody/checkpoint_400 \
  --use_mujoco \
  --motion_path KIT/314/walking_medium09_poses \
  --stochastic \
  --n_steps 1000
```

### 7.2 计算验证指标

```bash
uv run python fullbody/eval.py \
  --path checkpoints/my_first_fullbody/checkpoint_400 \
  --metrics \
  --metrics_only \
  --motion_group KIT_KINESIS_TESTING_MOTIONS \
  --eval_seed 0
```

### 7.3 录视频或开 Viser

录视频：

```bash
uv run python fullbody/eval.py \
  --path checkpoints/my_first_fullbody/checkpoint_400 \
  --use_mujoco \
  --record \
  --no_render
```

用 Viser 做交互可视化：

```bash
uv run python fullbody/eval.py \
  --path checkpoints/my_first_fullbody/checkpoint_400 \
  --use_mujoco \
  --viser_viewer
```

---

## 8. 正式进入 PPO 训练

这一节就是你之后最常执行的主流程。

### 8.1 建议的训练顺序

建议按下面顺序，而不是一步到位：

1. `conf_fullbody_demo` 或 `conf_bimanual_demo` 跑通。
2. 官方 GMR cache + 官方主配置跑一个小规模实验。
3. 缩小 `num_envs`，确认自己机器稳定。
4. 固定数据集后，再改 reward / observation / PPO。
5. 最后再尝试大规模长时训练。

### 8.2 全身 PPO 训练

如果你已经下载好官方 GMR cache：

```bash
uv run fullbody/experiment.py --config-name=conf_fullbody_gmr wandb.mode=disabled
```

如果你想用 residual 网络版本：

```bash
uv run fullbody/experiment.py --config-name=conf_fullbody_gmr_resnet wandb.mode=disabled
```

如果你的显卡没那么大，建议从缩小并行环境开始：

```bash
uv run fullbody/experiment.py --config-name=conf_fullbody_gmr_resnet \
  wandb.mode=disabled \
  experiment.env_params.num_envs=1024
```

### 8.3 双臂 PPO 训练

```bash
uv run bimanual/experiment.py --config-name=conf_bimanual wandb.mode=disabled
```

或者 demo 版本：

```bash
uv run bimanual/experiment.py --config-name=conf_bimanual_demo wandb.mode=disabled
```

### 8.4 从已有 checkpoint 继续训练

从 Hugging Face 官方模型继续训练：

```bash
uv run fullbody/experiment.py --config-name=conf_fullbody_gmr_resnet \
  experiment.resume_from="hf://amathislab/mm-fullbody-base" \
  wandb.mode=disabled
```

针对单个动作做 finetune：

```bash
uv run fullbody/experiment.py --config-name=conf_fullbody_gmr_resnet \
  experiment.resume_from="hf://amathislab/mm-fullbody-base" \
  experiment.reset_std_on_resume=3 \
  experiment.task_factory.params.amass_dataset_conf.dataset_group=null \
  experiment.task_factory.params.amass_dataset_conf.rel_dataset_path='["KIT/200/Handstand01_poses"]' \
  wandb.mode=disabled
```

### 8.5 什么时候算“PPO 已经开始正常工作”

至少满足这几条：

- 程序能稳定进入训练，不频繁崩溃或 OOM。
- 能周期性写出 `checkpoint_*`。
- `eval.py` 能成功加载 checkpoint。
- 验证指标不是完全随机。
- 回放时策略至少开始学会模仿一部分 motion，而不是立刻发散。

到这里，才算真正完成“把 MuscleMimic 的 PPO 训练链路部署起来”。

---

## 9. 在这个基础上做自己的研究

如果你的目标是科研，不要直接改主配置文件。更推荐下面这套方法。

### 9.1 给自己的实验建立独立配置

建议从现有配置复制一份，再开始改。例如：

- 从 `fullbody/conf_fullbody_gmr_resnet.yaml` 复制出 `fullbody/conf_fullbody_myproj.yaml`
- 或从 `bimanual/conf_bimanual.yaml` 复制出 `bimanual/conf_bimanual_myproj.yaml`

这样做的好处是：

- 你能保留作者 baseline；
- 你的实验更容易复现；
- 你不会把基础配置改乱。

### 9.2 最值得研究的方向

这个项目最适合从下面几类改动切入：

- 数据层：改 `dataset_group`、替换成你自己 retarget 的 motion、做 curriculum 或分阶段训练。
- 观测层：增减 muscle observations、goal lookahead、motion phase。
- 奖励层：调整 `reward_params` 中各项权重，特别是 site tracking、root、velocity、action regularization。
- 终止机制：修改 `terminal_state_type` 和阈值，看训练稳定性如何变化。
- 网络层：改 `actor_hidden_layers`、`critic_hidden_layers`，或尝试 `use_moe`、`use_residual`。
- PPO 层：改 `num_steps`、`num_minibatches`、`update_epochs`、`init_std`、学习率调度。

### 9.3 推荐的研究迭代方式

不要同时改太多。建议每次只改一层：

1. 固定数据和环境。
2. 只改一种因素，比如 reward 或 PPO。
3. 记录 run_id、配置文件名、checkpoint 路径。
4. 用同一套验证动作和指标比较结果。

最怕的是一口气同时改数据、奖励、网络、并行度，最后无法判断到底是什么起作用。

### 9.4 推荐保留的实验记录

每个实验至少保存这些信息：

- 使用的配置文件
- 命令行 override
- `run_id`
- checkpoint 路径
- 训练/验证数据集
- 核心指标
- 最终演示视频

这样后面写论文、做汇报、复现实验时会省很多时间。

---

## 10. 如何把它用于自己的部署和项目开发

这里的“部署”，我建议分成三种情况理解。

### 10.1 部署成你的训练基线

这是最常见的用法：

1. 选定一个基础配置，比如 `conf_fullbody_gmr_resnet`。
2. 下载官方 GMR cache 或自己 retarget 数据。
3. 固定 checkpoint 根目录和 `run_id`。
4. 开始持续训练、评估和迭代。

这是最适合科研开发的模式。

### 10.2 部署成一个可演示的策略

如果你已经有训练好的模型，可以用：

- `eval.py --mujoco_viewer` 做本地演示；
- `eval.py --viser_viewer` 做网页可视化；
- `eval.py --record` 输出视频；
- `eval.py --export_trajectory` 导出轨迹数据，给下游系统使用。

这适合：

- 项目汇报；
- 对外展示；
- 把策略输出接到你自己的分析管线里。

### 10.3 部署成你自己的研究代码库

如果你准备长期在这个基础上做研究，建议尽快形成自己的目录规范：

- `conf_*_myproj.yaml`：你自己的实验配置
- `scripts/`：你自己的批处理训练和评估脚本
- 独立的 `checkpoints/` 根目录
- 独立的 `logs/`、`videos/`、`metrics/` 输出目录

核心原则是：

- 不要直接覆盖官方 baseline；
- 不要把一次性实验命令只留在 shell history 里；
- 不要让数据路径、checkpoint 路径、可视化输出路径混在一起。

---

## 11. 一个建议的实际执行顺序

如果你现在要从零开始，我建议你照着这个顺序做：

1. 在 Linux 或 WSL2 里完成 `uv sync`。
2. 用 `conf_fullbody_demo` 跑通一次最小训练。
3. 用 `fullbody/eval.py` 成功加载 checkpoint。
4. 下载官方 GMR cache，跑 `conf_fullbody_gmr` 或 `conf_fullbody_gmr_resnet`。
5. 固定你的 `run_id` 和 `checkpoint_root`。
6. 复制一份自己的配置文件，比如 `conf_fullbody_myproj.yaml`。
7. 只改一个研究因素，跑第一组对照实验。
8. 用统一指标和视频做结果比较。

---

## 12. 常见问题和直接建议

### 12.1 训练刚启动就 OOM

优先按这个顺序排查：

1. 降 `experiment.env_params.num_envs`
2. 减小网络层数或宽度
3. 先关闭大规模验证
4. 先用 demo 或更小数据集

### 12.2 跑不起来，不确定是环境问题还是数据问题

最简单的判断方法：

- `conf_*_demo` 都跑不通：先修环境。
- demo 能跑，GMR/AMASS 跑不通：优先查缓存路径、数据路径、retarget 配置。

### 12.3 训练能跑，但结果很差

先不要立刻重写算法。先检查：

- 数据是否真的读到了你想要的 motions
- reward 权重是否合理
- early termination 是否太严格
- `init_std` 是否过小或过大
- 验证集和训练集是否一致或差异过大

---

## 13. 一句话总结

对大多数研究者来说，最合理的起点不是“自己先 retarget 全 AMASS”，而是：

`先用 demo 跑通 -> 再用官方 GMR cache 复现 -> 再复制配置开始自己的 PPO 实验 -> 最后再扩展到自己的数据和部署方式`

这条路线最稳，也最省时间。

# Remote Training Setup

这份清单用于把当前 BadmintonMimic 训练代码迁移到另一台 Linux GPU 服务器。仓库只提交代码、配置、manifest 和文档；数据、retarget cache、SMPL/MANO 模型、checkpoint 和实验输出都需要在训练服务器单独准备。

## 1. Clone

```bash
git clone -b badminton-remote-training https://github.com/yangfeiyang-123/badminton_mimic.git
cd badminton_mimic
```

如果你暂时推到了其它分支名，把 `badminton-remote-training` 替换成实际分支。

## 2. System Requirements

推荐环境：

- Linux x86_64
- NVIDIA GPU
- Python 3.11
- `uv`
- `wget` and `bsdtar` if you use `scripts/run_with_cuda_compat.sh`
- Optional: Weights & Biases account for online logging
- Optional: Hugging Face account/token for official MuscleMimic datasets and checkpoints

安装 `uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

建议把虚拟环境和缓存放到数据盘：

```bash
export ENV_ROOT=/data/musclemimic_env
mkdir -p "$ENV_ROOT"/{uv-cache,uv-data,tmp,venv}
export UV_CACHE_DIR="$ENV_ROOT/uv-cache"
export UV_PROJECT_ENVIRONMENT="$ENV_ROOT/venv"
export XDG_DATA_HOME="$ENV_ROOT/uv-data"
export TMPDIR="$ENV_ROOT/tmp"
```

安装训练依赖：

```bash
uv sync --extra cuda --extra smpl --extra gmr --extra dev
```

如果只训练已经 retarget 好的 cache，通常仍建议保留 `--extra smpl --extra gmr`，因为 badminton workflow 中包含 retarget 和 cache 检查脚本。

## 3. Files You Must Prepare Separately

### 3.1 Badminton AMASS-style Motions

当前 badminton 配置读取 manifest 中的 motion path。默认 manifest 位于：

```text
BadmintonMimic/manifests/train_list.txt
BadmintonMimic/manifests/val_list.txt
```

这些 path 是相对于 `MUSCLEMIMIC_AMASS_PATH` 的路径，并且不带 `.npz` 后缀。例如：

```text
badminton/train/forehand_clear_clip1_merged_poses
```

对应文件需要放在：

```text
$MUSCLEMIMIC_AMASS_PATH/badminton/train/forehand_clear_clip1_merged_poses.npz
```

每个 `.npz` 至少需要包含：

- `poses`
- `trans`
- `betas`
- `gender`
- `mocap_framerate`

本地 `BadmintonMimic/data/**/*.npz` 不提交到 Git。把这些文件用 `rsync/scp` 或对象存储传到新服务器。

### 3.2 SMPL-H Models

如果要运行 GMR retarget，需要下载 licensed SMPL-H 模型。当前脚本默认读取：

```text
$MUSCLEMIMIC_SMPL_MODEL_PATH
```

建议目录结构：

```text
/data/models/smplh/
  female/model.npz
  male/model.npz
  neutral/model.npz
  SMPLH_NEUTRAL.pkl
```

如果还需要重新生成 `SMPLH_NEUTRAL.pkl`，先准备包含 `smplh/` 和 `mano_v1_2/` 的上级目录，然后运行：

```bash
uv run musclemimic-set-smpl-model-path --path /data/models
cd loco_mujoco/smpl
bash install_smplh.sh
cd ../..
```

训练/retarget 时再把 `MUSCLEMIMIC_SMPL_MODEL_PATH` 指到实际 `smplh` 目录：

```bash
export MUSCLEMIMIC_SMPL_MODEL_PATH=/data/models/smplh
```

### 3.3 Retarget Cache

GMR retarget cache 默认读取/写入：

```text
$MUSCLEMIMIC_CONVERTED_AMASS_PATH
```

如果已有本机 retarget cache，可以直接同步到新服务器：

```text
/data/badminton_mimic/caches/AMASS/
```

如果没有 cache，在新服务器上重跑：

```bash
source BadmintonMimic/configs/env.sh
uv run python BadmintonMimic/scripts/run_retarget.py --split train
uv run python BadmintonMimic/scripts/run_retarget.py --split val
```

### 3.4 Optional Official Checkpoints and Datasets

官方 checkpoint 可直接用 Hugging Face URI，不需要提交到仓库：

- Full body checkpoint: `hf://amathislab/mm-fullbody-base`
- Bimanual checkpoint: `hf://amathislab/mm-bimanual-v0`

如果要下载官方 demo/GMR resources，先登录 Hugging Face：

```bash
uv run hf auth login
```

常用资源：

- Gated demo dataset: `amathislab/demo_dataset`
- Full body GMR dataset: `amathislab/musclemimic-retargeted`
- Bimanual GMR dataset: `amathislab/musclemimic-bimanual-retargeted`

## 4. Configure Paths on the Training Server

推荐先显式设置路径，再 source project env：

```bash
export MUSCLEMIMIC_AMASS_PATH=/data/badminton_mimic/amass_npz
export MUSCLEMIMIC_CONVERTED_AMASS_PATH=/data/badminton_mimic/caches/AMASS
export MUSCLEMIMIC_SMPL_MODEL_PATH=/data/models/smplh
export MM_CUDA_COMPAT_ROOT=/data/badminton_mimic/cuda-compat-12.4
export MM_CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled

source BadmintonMimic/configs/env.sh
```

`BadmintonMimic/configs/env.sh` 会保留上面这些显式环境变量；没有设置时才回退到仓库内默认路径。

也可以写入 MuscleMimic 用户配置：

```bash
uv run musclemimic-set-amass-path --path "$MUSCLEMIMIC_AMASS_PATH"
uv run musclemimic-set-conv-amass-path --path "$MUSCLEMIMIC_CONVERTED_AMASS_PATH"
uv run musclemimic-set-smpl-model-path --path "$MUSCLEMIMIC_SMPL_MODEL_PATH"
```

## 5. Generate and Install the Hydra Config

```bash
uv run python BadmintonMimic/scripts/build_config_from_manifests.py
bash BadmintonMimic/scripts/install_configs.sh
```

默认生成：

```text
BadmintonMimic/experiments/fullbody/config_specific_task/conf_fullbody_badminton_gmr.yaml
```

并安装到：

```text
fullbody/config_specific_task/conf_fullbody_badminton_gmr.yaml
```

## 6. Start Training

最简单入口：

```bash
bash BadmintonMimic/scripts/train_fullbody.sh
```

等价展开：

```bash
source BadmintonMimic/configs/env.sh
uv run python BadmintonMimic/scripts/build_config_from_manifests.py
bash BadmintonMimic/scripts/install_configs.sh

MM_CUDA_VISIBLE_DEVICES=0 \
WANDB_MODE=disabled \
scripts/run_with_cuda_compat.sh \
uv run fullbody/experiment.py \
  --config-name=config_specific_task/conf_fullbody_badminton_gmr \
  wandb.mode=disabled
```

如果服务器 CUDA/driver 环境已经可用，也可以不用 compat wrapper：

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run fullbody/experiment.py \
  --config-name=config_specific_task/conf_fullbody_badminton_gmr \
  wandb.mode=disabled
```

显存不足时优先降低并行环境数：

```bash
uv run fullbody/experiment.py \
  --config-name=config_specific_task/conf_fullbody_badminton_gmr \
  experiment.env_params.num_envs=128 \
  wandb.mode=disabled
```

## 7. Resume or Finetune from Checkpoint

从官方 fullbody checkpoint 继续：

```bash
uv run fullbody/experiment.py \
  --config-name=config_specific_task/conf_fullbody_badminton_gmr \
  experiment.resume_from="hf://amathislab/mm-fullbody-base" \
  wandb.mode=disabled
```

从本地 checkpoint 继续：

```bash
uv run fullbody/experiment.py \
  --config-name=config_specific_task/conf_fullbody_badminton_gmr \
  experiment.resume_from=/data/badminton_mimic/checkpoints/<run_id>/checkpoint_<step> \
  wandb.mode=disabled
```

训练输出和 checkpoint 不提交到 Git。建议放在数据盘，例如：

```bash
uv run fullbody/experiment.py \
  --config-name=config_specific_task/conf_fullbody_badminton_gmr \
  experiment.checkpoint_root=/data/badminton_mimic/checkpoints \
  wandb.mode=disabled
```

## 8. Smoke Checks

基础导入：

```bash
uv run python -c "import jax, mujoco, musclemimic; print(jax.devices())"
```

检查 manifest 对应的 `.npz` 是否存在：

```bash
while read -r item; do
  [[ -z "$item" || "$item" == \#* ]] && continue
  test -f "$MUSCLEMIMIC_AMASS_PATH/${item%.npz}.npz" || echo "missing: $item"
done < BadmintonMimic/manifests/train_list.txt
```

运行单元测试中的轻量级 badminton contract：

```bash
uv run pytest tests/unit/test_badminton_fps_contract.py tests/unit/test_badminton_smooth_retarget_config.py
```

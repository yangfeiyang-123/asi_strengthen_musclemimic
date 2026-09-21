# 在另一台 GPU 服务器部署与训练

## 适用范围

仅在远程部署或私有资产迁移时使用本手册。用户指定的运行位置优先，本机训练按
[训练启动合同](workstation_training_contract.md) 执行，不要求登录 7/9 号机。

| 环境 | SSH | 仓库路径 |
|---|---|---|
| 历史工作服务器 7 | `yangfeiyang@172.18.22.7` | `/data3/yangfeiyang/WorkSpace/asi_strengthen_musclemimic` |
| 实验服务器 9 | `yangfeiyang@172.18.22.9` | `/data/yangfeiyang/WorkSpace/asi_strengthen_musclemimic` |

远程部署前在工作端冻结 Git SHA、source fingerprint、配置和 run id，使用目标机独立干净
worktree；不在活动 checkout 中开发、pull、清理或覆盖配置。原迁移引用的
`AGENTS.experiment-server9.md` 未随当前快照提供；9 号机部署前从原机取得并读取，
该文件缺失不阻塞本机开发/训练。历史迁入 HEAD 为 `c1ccd9328e40b01d92ac1a41697b941e589b8f5a`。

下文固定 SHA、22/5 release 与顺序属于 **2026-08-11 legacy Clear PEASD family**。
其他 family 应使用其实际 release、配置和 checkpoint 合同，不照搬本例的固定身份/检查器。

这份流程把 Git 中的源码/配置与私有训练资产分开迁移。GitHub 仓库不包含 `datasets/`、
`artifacts/`、SMPL-H、checkpoint、W&B、日志和视频；这些内容必须通过私有通道传输。

## 1. 先冻结实验身份

PEASD Stage1 的 matched family 绑定 source-tree snapshot，不只绑定 commit。当前 2026-08-11
运行的 Clear T0-S0 绑定：

```text
git SHA: 103f0b1538ff
source-tree fingerprint: 5882a7b35f663911281419435922b5577efdb0546171687a6dcc432c2d37c45a
```

最新开发树已经不是这个 fingerprint。另一服务器如果 checkout 最新发布提交，必须建立新的
source snapshot；不能把它作为旧 family 的 T0-S1/T0-S2，也不能恢复旧 family checkpoint。
若要在新源码上做 matched T0–T4 比较，应从 T0 seed 0 开始完整重跑。

## 2. 在源服务器生成私有资产清单

从仓库根目录运行：

```bash
uv run --locked python scripts/build_training_asset_manifest.py \
  --action forehand_clear \
  --output /tmp/forehand_clear_assets.json \
  --files-from /tmp/forehand_clear_assets.files
```

生成器先验证 22/5 release、source/cache hash、visual QC、verified PEASD tube，并确认
SMPL-H neutral model 存在，
再生成逐文件 SHA-256 清单。默认不包含 checkpoint；需要迁移某个合法 immutable leaf 时，
显式加 `--extra-path <checkpoint_leaf>`，不要同步整个历史 checkpoint 目录。

把 Git 以外的资产传到目标仓库同名相对路径：

```bash
rsync -a --info=progress2 --files-from=/tmp/forehand_clear_assets.files \
  ./ <user>@<target-host>:<target-repo>/
rsync -a /tmp/forehand_clear_assets.json \
  <user>@<target-host>:<target-repo>/private_asset_manifest.json
```

该清单包含授权/私有数据的路径和哈希，不提交 GitHub。

## 3. 在目标服务器安装

```bash
git clone https://github.com/yangfeiyang-123/asi_strengthen_musclemimic.git
cd asi_strengthen_musclemimic
git checkout <published-commit-or-branch>

uv sync --locked --extra dev --extra cuda --extra smpl --extra gmr
wandb login
```

目标服务器需要 Linux x86_64、Python 3.11、NVIDIA driver、`uv`、`tmux`、`wget`、
`bsdtar`。`run_with_cuda_compat.sh` 首次运行会准备用户态 CUDA compatibility 包。

设置该服务器自己的缓存盘；不要修改源码中的路径：

```bash
export MUSCLEMIMIC_JAX_CACHE_ROOT=/path/to/large-writable-volume/jax-cache
```

## 4. 运行 fail-closed preflight

```bash
source configs/env.sh
uv run --locked python scripts/server_training_preflight.py \
  --action forehand_clear \
  --physical-gpu 0 \
  --jax-cache-root "$MUSCLEMIMIC_JAX_CACHE_ROOT" \
  --asset-manifest private_asset_manifest.json \
  --output server_preflight.json
```

只有 `passed: true` 才能训练。该检查会默认拒绝有源码修改的 checkout，并验证 Git/source snapshot、工具、GPU、可写 JAX
cache、SMPL-H、完整 action release/visual QC、verified tube 和私有资产逐文件 hash。

如果需要与某个预注册 snapshot 精确匹配，额外传：

```bash
--expected-git-sha <40-char-sha> \
--expected-source-tree-fingerprint <64-char-sha256>
```

## 5. 解析配置与启动

新源码应使用新的 run ID 和 fresh optimizer。以下示例只做 T0-S0 新 family 的启动模板：

```bash
export CUDA_VISIBLE_DEVICES=0
export MUSCLEMIMIC_JAX_CACHE_KEY=forehand_clear_peasd_new_snapshot_t0_s0
export MUSCLEMIMIC_TRAIN_LOG=artifacts/forehand_clear_peasd_new_snapshot/stage1_family/logs/t0_s0.log
export MUSCLEMIMIC_ORBAX_SAVE_CONCURRENT_GB=4
export MUSCLEMIMIC_ORBAX_RESTORE_CONCURRENT_GB=4

MUSCLEMIMIC_DRY_RUN=1 scripts/run_fullbody_training.sh \
  --config-name=config_specific_task/archive/legacy_22_5/peasd_lite_v1/conf_fullbody_forehand_clear_peasd_t0 \
  experiment.run_id=forehand_clear_peasd_new_snapshot_t0_s0 \
  wandb.name=forehand_clear_peasd_new_snapshot_t0_s0 \
  experiment.seeds=[0] \
  wandb.mode=disabled
```

核对 resolved `total_timesteps`、run ID、seed、reward、terminal、promotion、22/5 split 和
source-tree fingerprint 后，把 `MUSCLEMIMIC_DRY_RUN=1` 去掉并改成 `wandb.mode=online`。
长任务必须放进显式 socket/name 的新 tmux session。

启动成功仍以 `AGENTS.md` 的五项后验条件为准：本地 trajectory、run manifest、W&B URL、
正确物理 GPU PID、`Starting training...` 且无 fatal traceback。

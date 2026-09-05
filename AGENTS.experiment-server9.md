# MuscleMimic 新服务器 Codex 执行规范

本文件用于新服务器 `yangfeiyang@172.18.22.9`。在该服务器的仓库根目录中，
本文件的同内容副本必须命名为 `AGENTS.md`，Codex 进入仓库后应先读取它。

## 1. 固定服务器路径与代码身份

```text
repository: /data/yangfeiyang/WorkSpace/asi_strengthen_musclemimic
user tools: /data/yangfeiyang/WorkSpace/asi_strengthen_musclemimic/.local/workspace/.tools/bin
JAX cache:  /data/yangfeiyang/WorkSpace/asi_strengthen_musclemimic/.local/workspace/jax-cache
Git SHA:    250aa53777af7b19da748c60846a2665f0a1fdca
source FP:  0a61388a6ac0bde64f8e1d9c1f0572e90f4c13df26a50927bb454660a1b05954
```

每个新 shell、tmux pane 或 Codex 执行会话在运行项目命令前必须执行：

```bash
export PATH=/data/yangfeiyang/WorkSpace/asi_strengthen_musclemimic/.local/workspace/.tools/bin:$PATH
export UV_CACHE_DIR=/data/yangfeiyang/WorkSpace/asi_strengthen_musclemimic/.local/workspace/.cache/uv
export UV_PYTHON_INSTALL_DIR=/data/yangfeiyang/WorkSpace/asi_strengthen_musclemimic/.local/workspace/.tools/uv-python
export MUSCLEMIMIC_JAX_CACHE_ROOT=/data/yangfeiyang/WorkSpace/asi_strengthen_musclemimic/.local/workspace/jax-cache
export TMPDIR=/data/yangfeiyang/WorkSpace/asi_strengthen_musclemimic/.local/workspace/runtime-tmp
export MPLCONFIGDIR=/data/yangfeiyang/WorkSpace/asi_strengthen_musclemimic/.local/workspace/mpl-cache
cd /data/yangfeiyang/WorkSpace/asi_strengthen_musclemimic
```

正式 PEASD Stage-1 matched family 固定使用上面的代码快照。不要在该 checkout 上执行
`git pull`，不要让未验证的源码或配置覆盖它。`doc/*.md`、根目录 `AGENTS.md`、
`private_asset_manifest.json` 和私有数据可以不受 Git 跟踪，但 scoped source/config 必须干净。

## 2. 项目主要实验内容

项目主方法是 Partial-EMG Anchored Synergy Distillation（PEASD）：

1. Stage 1：以视频重定向轨迹训练 full-354 肌肉 tracking teacher，完成 T0--T4
   matched comparison。少量 sEMG 只作为 measured-subspace anchor，不是 354 维肌肉真值。
2. Stage 2：把 teacher 蒸馏成不依赖未来 reference、部署时不需要在线 EMG 的
   direct/latent skill。
3. Stage 3：冻结 prior/decoder，用 LAB 与受限 residual 处理来球并评估真实击球。

正手高远球 `forehand_clear` 是 P0 主结果。当前发布任务是新建 matched family，从
T0 seed 0 和 fresh optimizer 开始；不得恢复旧 family checkpoint，也不得把 dry-run、
单元测试或旧结果写成新实验结果。详细顺序与验收标准见：

- `doc/新服务器部署与训练执行手册.md`
- `docs/peasd_implementation_guide.md`
- `doc/实验计划/PEASD正式实验计划.md`
- `doc/实验计划/实验运行跟踪表.md`
- `docs/server_deployment.md`
- `doc/新服务器Aug100增广数据说明.md`

## 3. 私有数据边界

正式训练不能只复制 `datasets/`。当前服务器还必须保留：

- `datasets/`：22 条训练轨迹、5 条 held-out validation 轨迹及动作数据；
- `artifacts/`：release/QC 与 sealed verified EMG tube；
- `outputs/`：被 release 或 QC 显式绑定的派生结果；
- `smpl_models/`：SMPL-H 模型；
- `jidian_measurement/data/`：原始肌电测量数据，只用于重做预处理、QC、tube 构建与审计，
  当前 PPO 训练不会直接读取；
- `private_asset_manifest.json`：上述私有资产的逐文件 SHA-256 清单。

当前扩展清单必须满足：

```text
action: forehand_clear
files: 1303
total bytes: 1778934913
manifest fingerprint: b17076050ec4a9c010ea5b7ea2047bfa1917f00468b6d10f2d055b6e41919bd2
```

这些数据、模型和清单不得提交到公开 GitHub，也不得使用 `rsync --delete` 同步。

2026-08-13 另行同步了 Forehand Clear、Forehand Lift、China Jump 各 100 条的
`_aug100` 新 namespace。它们的路径、来源分组、SHA 清单和防止 train/validation
近重复泄漏的规则见 `doc/新服务器Aug100增广数据说明.md`。现有 configs 不会自动切换
到这批数据；启用时必须建立新的分组 split、config、run id 和 fresh optimizer。

## 4. 唯一合法的训练启动入口

正式训练只能从仓库根目录通过 `scripts/run_fullbody_training.sh` 启动。禁止直接运行
`python`、`.venv/bin/python` 或 `uv run fullbody/experiment.py` 作为 production run。
launcher 必须：

- 自动 `source configs/env.sh`；
- 通过 `CUDA_VISIBLE_DEVICES` 选择一张明确的物理 GPU；
- 使用稳定的 task-specific `MUSCLEMIMIC_JAX_CACHE_KEY`；
- 使用 append-only `MUSCLEMIMIC_TRAIN_LOG`；
- 保持 `MUSCLEMIMIC_ORBAX_SAVE_CONCURRENT_GB=4` 和
  `MUSCLEMIMIC_ORBAX_RESTORE_CONCURRENT_GB=4`，除非明确批准修改；
- 最终经 `scripts/run_with_cuda_compat.sh uv run` 启动。

基本形式：

```bash
export CUDA_VISIBLE_DEVICES=<空闲物理GPU编号>
export MUSCLEMIMIC_JAX_CACHE_KEY=<稳定且任务唯一的key>
export MUSCLEMIMIC_TRAIN_LOG=<append-only日志路径>
export MUSCLEMIMIC_ORBAX_SAVE_CONCURRENT_GB=4
export MUSCLEMIMIC_ORBAX_RESTORE_CONCURRENT_GB=4

scripts/run_fullbody_training.sh \
  --config-name=config_specific_task/<stage>/<config> \
  wandb.mode=online
```

长任务必须使用命名 tmux session 和显式 socket。不要复用 `Pane is dead` 的 pane。
reward、termination、split、tube、mapping 或源码变化后，必须更换 run id 并使用 fresh
optimizer；禁止恢复不兼容 checkpoint。

## 5. 每次启动前的硬门禁

先检查 GPU 进程，不能把进程内的可见 device 0 误认为物理 GPU 0。然后运行 focused tests、
解析 Hydra config 并核对 `total_timesteps`、唯一 `run_id`、reward weights、terminal limits、
promotion 与 resume 行为。

服务器完整预检命令：

```bash
export PATH=/data/yangfeiyang/WorkSpace/asi_strengthen_musclemimic/.local/workspace/.tools/bin:$PATH
export UV_CACHE_DIR=/data/yangfeiyang/WorkSpace/asi_strengthen_musclemimic/.local/workspace/.cache/uv
export UV_PYTHON_INSTALL_DIR=/data/yangfeiyang/WorkSpace/asi_strengthen_musclemimic/.local/workspace/.tools/uv-python
export MUSCLEMIMIC_JAX_CACHE_ROOT=/data/yangfeiyang/WorkSpace/asi_strengthen_musclemimic/.local/workspace/jax-cache
export TMPDIR=/data/yangfeiyang/WorkSpace/asi_strengthen_musclemimic/.local/workspace/runtime-tmp
export MPLCONFIGDIR=/data/yangfeiyang/WorkSpace/asi_strengthen_musclemimic/.local/workspace/mpl-cache
cd /data/yangfeiyang/WorkSpace/asi_strengthen_musclemimic
source configs/env.sh

uv run --locked python scripts/server_training_preflight.py \
  --action forehand_clear \
  --physical-gpu <空闲物理GPU编号> \
  --jax-cache-root "$MUSCLEMIMIC_JAX_CACHE_ROOT" \
  --asset-manifest private_asset_manifest.json \
  --expected-git-sha 250aa53777af7b19da748c60846a2665f0a1fdca \
  --expected-source-tree-fingerprint \
    0a61388a6ac0bde64f8e1d9c1f0572e90f4c13df26a50927bb454660a1b05954 \
  --output server_preflight.json
```

`server_preflight.json` 必须是 `"passed": true`。不得用 `--skip-gpu`、`--skip-tube`、
`--allow-dirty-source` 绕过错误。正式启动前先设置 `MUSCLEMIMIC_DRY_RUN=1` 走同一个
launcher，确认训练没有启动且 resolved config 完全符合实验合同。

## 6. 启动后的五项验收与停止规则

不能凭 tmux 存活就报告成功，必须同时确认：

1. 22+5 条 trajectory 全部命中本地 retargeted file；任何 Hugging Face 下载尝试都必须停止；
2. checkpoint run manifest 存在，并记录正确 config hash、run id、timesteps、promotion、reward、
   terminal 与 fresh/no-resume 合同；
3. W&B 有真实 live run id 和 URL；
4. 新 Python PID 位于所选物理 GPU；
5. 日志达到 `Starting training...` 且没有 fatal traceback、NaN 或 OOM。

停止任务时，只向对应 tmux pane 发送一次 Ctrl-C，等待 Python PID 和 CUDA context 消失，
保留最新 finalized checkpoint 与完整日志。不要直接 `kill -9` 或删除 checkpoint。

## 7. Codex 的默认行为

Codex 接到训练任务后，先读本文件和部署手册，再检查现状；未得到用户明确的“启动训练”指令
时只做 preflight、tests 与 dry-run，不自行开启收费或长时间 GPU 任务。任何硬门禁失败都应报告
具体错误并停止，而不是降低检查标准。

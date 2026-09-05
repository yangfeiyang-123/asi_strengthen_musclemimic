# MuscleMimic 工作服务器 Codex 执行规范

本仓库位于工作服务器 `yangfeiyang@172.18.22.7`：

```text
repository: /data3/yangfeiyang/WorkSpace/asi_strengthen_musclemimic
imported Git HEAD: c1ccd9328e40b01d92ac1a41697b941e589b8f5a
experiment server: yangfeiyang@172.18.22.9
experiment repository: /data/yangfeiyang/WorkSpace/asi_strengthen_musclemimic
```

## 1. 服务器职责

- 7 号机是默认工作服务器。代码开发、Git 操作、文档、数据准备、结果分析、checkpoint
  检查、评估、迁移和实验部署准备都在这里完成。
- 9 号机是实验服务器，只运行经过确认的正式实验及其必要 preflight。不要在 9 号机的活动
  checkout 中开发、`git pull`、清理文件或覆盖配置。
- 未得到用户明确的“启动训练”指令时，不在任何服务器启动收费或长时间 GPU 任务。
- 需要把变更用于 9 号机实验时，先在 7 号机形成可审计的 Git SHA、source fingerprint、配置
  和 run id，再部署到 9 号机的独立干净 worktree；不得修改正在运行任务使用的 worktree。

9 号机迁移时的完整训练规则原样保存在 `AGENTS.experiment-server9.md`。凡是登录 9 号机、
部署或管理实验，必须同时遵守该文件和相关部署手册；其中固定快照要求只适用于对应的
PEASD 正式实验 family，不能误用于 7 号机日常开发分支。

## 2. 迁入状态保护

本仓库是从 9 号机整体迁入的工作副本，包含原有分支、tracked 修改、untracked 文件、私有数据
和 checkpoint。它们都属于用户：

- 开始修改前先检查 `git status --short --branch`，不得擅自 reset、clean、覆盖或删除迁入内容；
- 不要假设 dirty worktree 可以丢弃；需要隔离工作时新建分支或 worktree；
- `.local/`、虚拟环境和硬件相关缓存应在本机重建，不从 9 号机复制；
- 迁移验收记录和校验清单保存在 `.local/migration/`，不要提交到公开仓库。

## 3. 私有资产边界

以下内容不得提交或上传到公开 GitHub：

- `datasets/`
- `artifacts/`
- `outputs/`
- `smpl_models/`
- `jidian_measurement/data/`
- `private_asset_manifest.json`
- checkpoint、W&B 本地文件、原始或派生人体数据

同步私有资产时不得使用 `rsync --delete`。删除、覆盖或重新生成任何 checkpoint 前，必须先明确
目标、确认没有活动写入，并获得用户授权。

## 4. 本机环境

项目命令从仓库根目录运行。优先使用锁文件重建本机环境和缓存，不复制 9 号机的 `.venv`、
JAX cache 或编译产物。推荐的工作区变量为：

```bash
export PATH=/data3/yangfeiyang/WorkSpace/asi_strengthen_musclemimic/.local/workspace/.tools/bin:$PATH
export UV_CACHE_DIR=/data3/yangfeiyang/WorkSpace/asi_strengthen_musclemimic/.local/workspace/.cache/uv
export UV_PYTHON_INSTALL_DIR=/data3/yangfeiyang/WorkSpace/asi_strengthen_musclemimic/.local/workspace/.tools/uv-python
export MUSCLEMIMIC_JAX_CACHE_ROOT=/data3/yangfeiyang/WorkSpace/asi_strengthen_musclemimic/.local/workspace/jax-cache
export TMPDIR=/data3/yangfeiyang/WorkSpace/asi_strengthen_musclemimic/.local/workspace/runtime-tmp
export MPLCONFIGDIR=/data3/yangfeiyang/WorkSpace/asi_strengthen_musclemimic/.local/workspace/mpl-cache
cd /data3/yangfeiyang/WorkSpace/asi_strengthen_musclemimic
```

不要把密码、私钥或访问 token 写入仓库。跨服务器访问使用权限受限的 SSH key 或 ssh-agent。

## 5. 实验部署底线

正式训练仍只能从 9 号机目标 worktree 的仓库根目录，通过
`scripts/run_fullbody_training.sh` 启动，并遵守 preflight、dry-run、明确物理 GPU、唯一 run id、
fresh/resume 合同、W&B、checkpoint manifest 和启动后五项验收。不得以目标机文件已存在、tmux
存活或 checkpoint 目录出现作为训练成功的依据。

活动实验的停止只能对相应 tmux pane 发送一次 Ctrl-C，等待 Python PID 与 CUDA context 消失，
保留 finalized checkpoint 和完整日志；不要直接 `kill -9`。

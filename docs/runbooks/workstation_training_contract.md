# MuscleMimic 训练启动合同

仅在准备、启动、恢复或停止训练时读取。用户提出训练请求后，自主完成以下流程；日常开发不必逐项执行。

## 环境

正式训练从实际仓库根目录通过 `scripts/run_fullbody_training.sh` 启动。启动器加载
`configs/env.sh`，调用 `scripts/run_with_cuda_compat.sh uv run --locked`，以 `tee -a` 追加日志。
不要绕过启动器直接运行 trainer Python。依赖从锁文件恢复，不复制其他主机的虚拟环境或编译缓存。

本机环境模板（替换任务占位符）：

```bash
cd /home/msc/fyTmp/WorkSpace/asi_strengthen_musclemimic
export CUDA_VISIBLE_DEVICES=<physical_gpu_index>
export MUSCLEMIMIC_JAX_CACHE_KEY=<stable_task_cache_key>
export MUSCLEMIMIC_TRAIN_LOG=<unique_append_only_log_path>
export MUSCLEMIMIC_JAX_CACHE_ROOT="$PWD/.local/workspace/jax-cache"
export UV_CACHE_DIR="$PWD/.local/workspace/.cache/uv"
export TMPDIR="$PWD/.local/workspace/runtime-tmp"
export MPLCONFIGDIR="$PWD/.local/workspace/mpl-cache"
export MUSCLEMIMIC_ORBAX_SAVE_CONCURRENT_GB=4
export MUSCLEMIMIC_ORBAX_RESTORE_CONCURRENT_GB=4
mkdir -p "$UV_CACHE_DIR" "$TMPDIR" "$MPLCONFIGDIR"
```

缓存目录按目标机磁盘调整。若继承其他任务的 `JAX_COMPILATION_CACHE_DIR`，取消该覆盖后由
启动器按 task key 分配。Orbax 并发默认 4 GB，若有意调整应记录。

## 启动前

1. 从实际配置/checkpoint 核对 arm、seed、split、预算与初始化来源。80/20、40/10 与 legacy
   22/5 不能混为一个 family；历史默认预算不代替实际实验合同。
2. 检查轨迹和模型的本地路径。迁移时优先复用已有清单、内容哈希和 checkpoint 证据。
   Aug100 可显式配置 `training_source.checkpoint_evidence.config_path` 与 `config_sha256`，
   重验实际缓存与历史 split/数据哈希、继承原 QC；必须记录历史审核没有在本机重做。
3. 对相关改动运行 config/reward/terminal 和受影响的验证测试。报告已有私有资产缺项；
   不把跳过测试说成通过，不伪造 gate 或通过删除校验器来掩盖失败。
4. 用 canonical launcher 执行 dry-run，核对完整 resolved 配置：

   ```bash
   MUSCLEMIMIC_DRY_RUN=1 scripts/run_fullbody_training.sh \
     --config-name=<hydra_config> <exact_overrides> wandb.mode=online
   ```

   检查 total_timesteps、run id、seed、网络/优化器、reward、terminal、validation 与 promotion。
   正式启动使用相同参数；若 dry-run 设了 `JAX_PLATFORMS=cpu`，训练前取消。
5. 保存可审计的 Git SHA、source fingerprint、最终配置和独立 run id。
   reward/termination 改变时使用新 run id 与 fresh optimizer，不恢复不兼容 checkpoint。
   resume 必须核对 exact finalized leaf 和配置兼容性；不能覆盖原有 checkpoint。
6. 启动前查询 `nvidia-smi` 的物理 GPU 与进程，一个物理 GPU 同时只运行一个正式 PPO。
   低显存或短暂 0% 利用率不是独占空闲的充分证据，不停止或挤占其他用户的任务。

## 启动与验收

使用显式 tmux socket 和新的具名 session，不复用 dead pane：

```bash
mkdir -p .local/tmux
tmux -S "$PWD/.local/tmux/training.sock" new-session -d \
  -s <unique_session> -c "$PWD"
```

把完整环境导出和 canonical launcher 命令送入该 pane，记录 socket/session/pane、GPU、run id、
启动时间、源码身份和日志路径。下列五项全部满足后才报告启动成功并写“训练中”：

1. 所有 train/validation 轨迹从已有本地 retargeted 文件加载。若训练尝试从 Hugging Face
   下载轨迹，停止该 run 并修复环境/缓存绑定。准备阶段授权的资产下载不属于此异常。
2. checkpoint run manifest 已生成，config hash、run id、预算、promotion、reward 和 terminal 正确。
3. W&B 在线 run id 与 URL 已生成且服务连接成功。
4. 新 Python PID 位于指定物理 GPU；进程内部 visible index 0 不等于物理 GPU 0。
5. 日志到达 `Starting training...`，没有 fatal traceback。

记录失败原因并保留失败 run。用户要求远端同步时，更新适用记录、推送并核对远端分支 SHA；
私有日志、数据和 checkpoint 不公开提交。

## 停止

对对应 tmux pane 发送一次 Ctrl-C，等待 Python PID 与 CUDA context 消失，保留完整日志与最新
finalized checkpoint；不要直接 `kill -9`。用户取消尚未启动的训练时，不再启动。

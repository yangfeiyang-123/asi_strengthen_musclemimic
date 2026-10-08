# MuscleMimic 协作约定

开发源码在 `src/{musclemimic,loco_mujoco,environment,fullbody,analysis}/`；握拍工具导入 `musclemimic.grip`，`src` 本身不是包。
文档在 `docs/`，实验在 `experiments/stage{1,2,3}/`；导航见 [docs/README.md](docs/README.md) 和 [配置索引](src/fullbody/config_specific_task/README.md)。不要恢复根目录旧 `doc`、`Experiments` 入口。

## 按任务读取

- 启动、恢复或检查训练：先读 [当前状态](docs/runbooks/server9/当前实验状态.md)、[训练规范](docs/runbooks/server9/智能体训练执行规范.md)，再读对应 family 合同和脚本。
- 新服务器与原始 22/5 split 的 320M 基线：[部署手册](docs/runbooks/server9/新服务器部署与训练执行手册.md)。Aug100 来源、分组与防泄漏：[增广说明](docs/runbooks/server9/新服务器Aug100增广数据说明.md)。
- 方法、实验设计或 teacher promotion：[实现指南](docs/runbooks/peasd_implementation_guide.md)、[正式计划](docs/plans/PEASD正式实验计划.md)。
- 普通代码/文档修改：只读相关实现和局部说明；本机 shell 环境见 [开发环境](docs/runbooks/server9/本机开发环境.md)。

## 开发与验收

- 本机先加载上述基础环境，再 `source configs/env.sh`；开发检查用 `.local/dev-venv/bin/python`。按改动运行对应 pytest；源码布局或跨模块合同改动运行 `make source-only`。文档修改检查路径/命令即可。
- 用户要求修复即完成实现、必要验证和交接；可逆本地操作不逐步确认。用户明确要求启动训练即授权对应范围的启动与五项验收，已确认 GPU、seed、family 和范围不反复询问。无训练授权只做 preflight、测试和 dry-run。
- 门禁失败停止受影响启动，说明具体错误；其他独立工作继续。区分预检通过、运行中和训练完成，长期状态写入交接并链接日志、manifest、W&B。
- 本机实验主表是 `experiments/EXPERIMENT_LOG.md`；`experiments/archive/msclab/` 是另一工作站的历史记录。

## 训练与数据硬边界

- 冻结 worktree 禁止 `git pull`、覆盖或同步开发源码。用目标 family 已批准的 SHA、fingerprint、split、预算与专用环境；开发目录 HEAD 和旧部署手册不能替代其身份。
- reward、termination、split、tube、mapping 或源码改变：新 run id、fresh optimizer；不恢复不兼容 checkpoint。
- 生产训练只走训练 checkout 根目录 `scripts/run_fullbody_training.sh`：单张明确物理 GPU、独立日志/cache key、命名 tmux session、显式 socket。Orbax save/restore 并发均为 4 GB，除非用户批准更改。
- 停训只向对应 pane 发送一次 Ctrl-C，等待 Python PID/CUDA context 消失；保留完整日志和最新 finalized checkpoint，不用 kill -9 或删 checkpoint。
- 私有数据、SMPL、资产清单不提交公开 GitHub；不用 rsync --delete。清理限当前开发目录，不递归清理冻结副本、环境、checkpoint 或编译缓存。

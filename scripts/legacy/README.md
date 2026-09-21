# scripts/legacy

仓库其余部分不再引用的历史脚本，按来源分组保留，便于复现旧结果；不进入主线 runbook。

| 目录 / 文件 | 来源 | 说明 |
|---|---|---|
| `data_pipeline_v0/` | 2026-06/07 视频→WHAM/DPVO→重定向的第一版数据管线 | `run_initial_wham_dpvo_pipeline.py`、`regen_all_optimized.sh`、`retarget_dataset.py`、`scan_*.py`（三条重定向分支的健康扫描）、`visualize_muscle_trajectory_check.py`。现行数据合同是 `musclemimic/badminton/data_release.py` + aug100 release，见 `docs/runbooks/peasd_implementation_guide.md` §4 |
| `stage3_direct_residual/` | 已归档的 Stage-3 direct-residual 线（v3–v45） | 击球高点时序校准、apex 对齐审计、旧 baseline demo 重渲染、warp 可行性探针。主线 Stage 3 是 LAB，见 `docs/narrative/03` §3 |
| `wandb_global_timestep_mirror.py` | 2026-08 W&B step 修复 | 把 Current Timestep 镜像为 history step 的一次性工具 |

这些脚本移动时未改内容；路径以仓库根为基准的调用方式不变（`python scripts/legacy/<dir>/<file>`）。

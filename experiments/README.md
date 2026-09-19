# experiments/

按 `进展/工作汇报0816.pdf` 的三阶段组织。每个 stage 目录放该阶段的实验定义（spec）、单次 run 记录和离线分析脚本；训练用的 Hydra 配置仍在 `fullbody/config_specific_task/`（导航见其 README），因为 Hydra 的搜索路径固定在 `fullbody/`。

| 目录 | 阶段 | 内容 |
|---|---|---|
| `EXPERIMENT_LOG.md` | 全部 | 唯一的进度记录（append-only） |
| `stage1/` | 阶段一 轨迹跟踪 | T0–T4 run 记录、320M/640M 离线分析脚本、`racket_curriculum/` 球拍质量课程的远端启动脚本 |
| `stage2/` | 阶段二 技能蒸馏 | 目前只有 README（如何启动 S2-A 与 S2-B/C/D/E），run 记录待产生 |
| `stage3/` | 阶段三 自适应击球 | `lab/` 主线 spec；`direct_residual/` 已归档的 v1–v31 直接残差线；`early_tasks/` 早期握拍/静态击球/挑球 |

以前的 `posttrain/`（Stage-3 spec 的旧名）和 `synergy/`（两个 NMF 分区 JSON，现在在 `configs/synergy/`）已取消。

# 归档文档

2026-09-18 起，项目主线收敛为 `进展/工作汇报0816.pdf` 定义的三阶段：
**S1 轨迹跟踪 + 肌电奖励 → S2 隐空间蒸馏 → S3 冻结 decoder 的 LAB 自适应击球**，加上保留的双人对打环境。
下列文档描述的是主线之外的研究支线或已被取代的计划。代码与测试仍在仓库中，只是不再排期、不进入论文主方法。

| 文件 | 原位置 | 归档原因 |
|---|---|---|
| `chinajump_primitive_synergy_runbook.md` | `docs/` | ChinaJump primitive/NMF/coverage 支线，不在主线动作范围 |
| `stage1_early_synergy.md` | `docs/` | ChinaJump Stage-1 早期 fixed-synergy 消融矩阵 |
| `primitive_physical_control_producer.md` | `docs/` | fixed synergy `W/R` 的 primitive rollout 生产合同；主线 decoder 为 direct MLP |
| `肌肉协同与肌肉内部约束.md` | `docs/` | fascicle continuity / IMR / Graph-NMF 专项，不进入主方法 |
| `racket_pose_editor.rst` | `docs/` | 20 关节握拍 preset 编辑器；生产 354-D 场景无手指关节，只用其产出的 attachment/grip preset |
| `stage3_return_training_findings_20260805.md` | `doc/` | Stage 3 direct-residual 线（v3–v45）的复盘；主线 S3 为 LAB，只保留其 reachability gate 与 aero 校准结论 |
| `新服务器部署与训练执行手册.md` | `doc/` | 旧服务器专用手册；现行为 `docs/runbooks/server_deployment.md` + `docs/runbooks/workstation_training_contract.md` + `AGENTS.md` |
| `PEASD实验运行跟踪表_20260811.md` | `doc/实验计划/实验运行跟踪表.md` | 2026-08-11 的状态快照；现行进度记录为 `experiments/EXPERIMENT_LOG.md` |
| `03_仓库改进与待办路线图_20260812.md` | `docs/narrative/03_…` | 1256 行旧路线图，含已过时判断（如"EMG latent 未实现"）；现行为同目录 `03_当前状态与待办.md` |

仍在主线中的文档：`README.md`、`AGENTS.md`、`docs/narrative/01–03`、`docs/plans/PEASD正式实验计划.md`、
`docs/contracts/MVC小于动作信号时如何处理.md`、`docs/runbooks/peasd_implementation_guide.md`、`docs/runbooks/forehand_clear_distillation_runbook.md`、
`docs/contracts/jidian_emg_integration.md`、`docs/contracts/emg_human_review_wizard.md`、`docs/contracts/肌肉生理约束实施契约_v2.md`、
`docs/contracts/body_action_modes_and_rigid_racket.md`（刚性球拍合同部分）、`docs/runbooks/workstation_training_contract.md`、
`docs/runbooks/server_deployment.md`、`environment/double_play/README.md`。

## 代码与配置层面的归档（2026-09-20）

- 训练配置：支线配置已物理搬到 `fullbody/config_specific_task/archive/{chinajump,continuity_graph_nmf,forehand_lift,fixed_synergy,legacy_22_5,subset_40_10,strokes,skill}/`，
  `action_registry.py`、测试与配置内部 `defaults:` 的路径同步改写，导航见 `fullbody/config_specific_task/README.md`。
- 脚本：仓库其余部分不再引用的旧脚本搬到 `scripts/legacy/`（第一版 WHAM/DPVO 数据管线、Stage-3 direct-residual 的 CEM/时序工具、W&B step 镜像），说明见其 README。
- 删除（git 历史可查）：上游 `bimanual/` 训练入口（配置早已不存在）、上游 `examples/`、个人 IDE 文件 `.cursorrules` 与 `musclemimic.code-workspace`。
- 迁移：`rl_training_environment/` 的握拍渲染脚本并入 `src/grip/render/`；`tests/` 顶层零散测试并入 `tests/unit/`。

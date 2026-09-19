# 文档索引

主线以 `进展/工作汇报0816.pdf` 为准：**S1 轨迹跟踪 + 肌电奖励 → S2 隐空间蒸馏 → S3 LAB 自适应击球**，外加保留的双人对打环境。

| 目录 | 放什么 | 文件 |
|---|---|---|
| `narrative/` | 研究故事、方法、当前状态 | `01_研究故事与论文叙事主线.md`、`02_三阶段方法与肌电参与机制.md`（含消融命名对照表 §24）、`03_当前状态与待办.md` |
| `plans/` | 正式实验合同 | `PEASD正式实验计划.md`（claim map、实验块、Go/Stop、Definition of Done） |
| `runbooks/` | 怎么跑 | `peasd_implementation_guide.md`（逐步正式流程）、`forehand_clear_distillation_runbook.md`、`workstation_training_contract.md`、`server_deployment.md` |
| `contracts/` | 数据与物理合同 | `肌肉生理约束实施契约_v2.md`、`body_action_modes_and_rigid_racket.md`、`jidian_emg_integration.md`、`emg_human_review_wizard.md`、`MVC小于动作信号时如何处理.md` |
| `archive/` | 归档支线 | ChinaJump、fixed synergy、Graph-NMF/肌束连续性、Stage-3 direct-residual 复盘、旧路线图等，见其 `README.md` |

其他位置：

- 实验目录：`experiments/README.md`（按阶段一/二/三组织；`EXPERIMENT_LOG.md` 为 append-only 进度记录）。
- 训练配置导航：`fullbody/config_specific_task/README.md`。
- 双人对打环境：`environment/double_play/README.md`。
- 肌电采集子项目：`jidian_measurement/README.md`。
- 协作规范与训练启动合同：根目录 `AGENTS.md`。

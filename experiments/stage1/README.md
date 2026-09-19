# 阶段一：轨迹跟踪

阶段一 = 身体轨迹模仿（仓库 Stage 1，T0–T4 matched family）+ 球拍质量课程（仓库 Stage 2，25→50→75→100%）。

## 训练入口

- 配置：`fullbody/config_specific_task/stage1_body/peasd_lite_v1/conf_fullbody_forehand_clear_aug100_peasd_t{0..4}.yaml`，treatment 由 `presets/stage1_peasd_lite_t{0..4}_v1.yaml` 决定（T0 无 EMG，T1 仅激活，T2 仅协同，T3 主方法，T4 相位平移负对照）。
- 启动：`scripts/run_fullbody_training.sh --config-name=<上面的配置>`，一张物理 GPU 一个 run，800M 步 fixed endpoint。
- 球拍课程：`fullbody/config_specific_task/stage2_racket_v2/conf_fullbody_forehand_clear_racket_mass_{025,050,075,100}.yaml`，从晋级的 T3 teacher 起步；`racket_curriculum/` 是 2026-09-01 在远端 7 号机的启动脚本（引用的 `*.env` 未随快照回传）。
- 晋级门：`musclemimic/badminton/stage1_peasd_gate.py`。

## 本目录内容

| 文件 | 说明 |
|---|---|
| `T0_SEED1_GPU3_20260907.md`、`T0_SEED1_GPU6_20260908.md` | 本机 T0 seed1 的启动准备与启动记录 |
| `extract_stage1_320m_metrics.py`、`extract_stage1_640m_metrics.py` | 40/10 子集 T2/T3/T4 seed0 在 320M/640M 的五维验证指标（探索性，不进 formal family） |
| `capture_t3_320m_activation_rollout.py`、`analyze_synergy_chain.py`、`compare_arms_s2.py`、`synergy_similarity_matrix.py` | 从 rollout 激活重建 15 通道协同链并与实测 tube 对比 |
| `build_metrics_dashboard.py`、`make_motion_filmstrip.py` | 汇报用图表与胶片图 |
| `compare_endpoints_80_20.py`、`endpoint_comparison_80_20.md` | 12 个 800M endpoint 在同一 tube 下的 held-out 生理/跟踪对比（探索性） |
| `capture_endpoint_kinematics.py` | 12 个 endpoint 的 held-out rollout，另存 qpos/qvel/17 个 mimic 站点与人体参考，供渲染与生物力学分析（`outputs/stage1_endpoint_compare/kin/`） |
| `biomech_metrics.py`、`biomech_metrics_80_20.md` | 发力时序、运动学动力链、拮抗肌共收缩、募集分布与有效维度、平滑度、逐区域跟踪误差 |
| `render_endpoint_videos.py` | 人体参考 ｜ 各 arm 并排、肌肉按激活着色的视频（`outputs/stage1_endpoint_compare/videos/`） |
| `rank_trajectories.py`、`trajectory_table_100.md` | 全部 100 条动作（80 train + 20 held-out，`capture_endpoint_kinematics.py --split train`）逐条对比、胜负计数与分布图 |
| `plot_trajectory_showcase.py` | 单条 held-out 轨迹的三条协同系数 + 右手速度（人体 tube vs 五臂），demo 用；轨迹 4（285 帧）是 T3 相对全部对照优势最大的一条 |
| `plot_endpoint_comparison.py` | 上述对比的五张图：相位热图、协同系数曲线、逐通道时序相关、逐通道 anchor loss、包络频谱（输出在 `outputs/stage1_endpoint_compare/figures/`） |
| `racket_curriculum/` | 球拍课程远端启动/对比脚本 |

**anchor v2（2026-09-18）后要重跑的**：T3v2 s0（P0）→ T4v2、T1v2 s0（P1）→ 三者 seed 1/2（P2）；T0 s1 续训/重跑与 T0 s2 照旧；T0、T2 定义未变不重跑。配置在 `fullbody/config_specific_task/stage1_body/peasd_lite_v1/conf_fullbody_forehand_clear_aug100_peasd_t{1,3,4}_anchor_v2.yaml`，细节见 `docs/narrative/03_当前状态与待办.md`。

已完成的 800M endpoint checkpoint 在 `checkpoints/stage1/seed{0,1,2}/T*/checkpoint_39063`（gitignored；哈希与 W&B 见同目录 `COMPLETED_20260918.json`）。正式状态以 `../EXPERIMENT_LOG.md` 为准。

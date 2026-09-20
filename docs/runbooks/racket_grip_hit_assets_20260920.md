# 球拍 / 羽毛球 / 握拍资产迁移记录（2026-09-20）

来源：Hugging Face 数据集 `yangfy0627/musclemimic-racket-grip-hit-20260920`，文件 `racket_grip_hit_20260920_100904.tar.gz`
（249,945,599 B，sha256 `1ab19b36f5333c21d588a39c578eb4e89cc515602de7d4bccd76d4c5f451caac`）。
包是源服务器 `/data/yangfeiyang/WorkSpace/asi_strengthen_musclemimic` 在 HEAD `c1ccd93` 上的工作目录快照，1275 个文件，包内 SHA256SUMS 全部校验通过。
本机保留：`datasets/incoming/racket_grip_hit_20260920_100904.tar.gz` 与 `datasets/incoming/racket_grip_hit_20260920_100904_meta/`（FILES.tsv、TRANSFER_*、SHA256SUMS），解压树已删除。

## 与本地仓库逐文件对比的结论

| 类别 | 数量 | 处理 |
|---|---:|---|
| 与本地字节一致 | 1118 | 不动。球拍 MJCF/网格（`environment/racket/`、`environment/overall_environment/assets/`）、握拍场景 `assets/right_hand_racket_grip_scene.xml`、握拍预设 `configs/racket_grip/forehand_clear_grip_v2_custom.json`、挂接合同 `configs/racket_attachment/forehand_clear_rigid_v4_custom.json`、`configs/racket_handle_params.json`、`src/grip/`、羽毛球气动/碰撞代码全在此列，**本地早已具备** |
| 内容不同 | 47 | 全部是包内版本 **早于** 本地 git 历史（anchor v2、aug100 release 校验、气动 v2 等本地更新），**未采用包内代码**。仅 `AGENTS.md` 为源服务器的本地版本，忽略 |
| 本地缺失、已复制 | 53 | 见下表 |
| 本地路径已改名 | 57 | `experiments/posttrain/*` 已在本地搬到 `experiments/stage3/{direct_residual,early_tasks}/`，`docs/*.md` 已搬到 `docs/{runbooks,contracts,archive}/`；内容相同，不复制 |

复制进仓库的 53 个文件（均在 gitignored 目录，复制后按 FILES.tsv 逐个核对 SHA256 通过）：

| 路径 | 内容 | 大小 |
|---|---|---:|
| `datasets/forehandClear_standard/muscle_trajectory/raw/` | 6 条原始 60 Hz GMR 轨迹（`6月2日-1..5,7`） | 5.8 MB |
| `datasets/forehandClear_standard/muscle_trajectory/raw_smooth_v1/` | legacy 22/5 数据的 27 条 100 Hz cache + 6 个 cache_provenance | 42 MB |
| `datasets/forehandClear_standard/manifests/raw_smooth_v1/` | 22/5 的 train/val/all list、release_manifest、qc_report、visual_qc_report | 140 KB |
| `artifacts/stage3_demo_sources/frozen_base_ckpt156/` | Stage-3 demo 的冻结 base 策略（354 维、obs 1950、16 层 MLP），来自 legacy distill ckpt156 | 95 MB |
| `artifacts/stage3_demo_sources/v24c_checkpoint_491520/` | Stage-3 direct-residual v24c 演示策略（`incoming_hit_training_v3`） | 4.9 MB |

包内说明源服务器也没有的旧产物：握拍 policy `policy_latest.pt`、握拍 seed json、`checkpoints/de63059b16c0/checkpoint_7812`、
来球 feed bank `outputs/incoming_shuttle_hit_reference_graded_v3/feed_bank_{train,eval}.npz`。这些仍然缺失，S3 direct-residual 演示不能原样复现。

## 本机验证（mujoco 3.4.0）

以下五个场景在本地全部加载并可 `mj_step`：`assets/right_hand_racket_grip_scene.xml`（nq 136 / nu 416）、
`environment/racket/assets/badminton_racket_{rigid,flex_proxy}.xml`、`environment/overall_environment/assets/overall_badminton_{scene,training_scene}.xml`（nq 143 / nu 416）。
`MjxMyoFullBodyRacket` 默认读取 grip v2 预设并绑定 attachment v4 合同（`musclemimic/badminton/racket_grip_preset.py`），指纹不一致会 fail-closed。

Stage-3 demo 策略的使用方式（只配置路径，不代表已通过本机训练兼容性验收）：

```bash
export MUSCLEMIMIC_STAGE3_SOURCE_CHECKPOINT="$PWD/artifacts/stage3_demo_sources/v24c_checkpoint_491520/policy.npz"
export MUSCLEMIMIC_STAGE3_BASE_POLICY="$PWD/artifacts/stage3_demo_sources/frozen_base_ckpt156"
```

## 对主线的意义

1. **球拍质量课程现在不缺资产**。`stage2_racket/conf_fullbody_badminton_racket_local.yaml` 的 `RacketMimicReward` 用轨迹右手 site 位姿加刚性握拍变换按步推导球拍参考（`racket_reference_source: derived_rigid`），不需要 event bank；只有 `stage2_racket_v2/*event_bank*` 才需要 `event_reference_v2` 清单，而那两个清单本机和源服务器都没有。
2. 该配置仍写在 legacy 22/5 数据上；正式课程应把 `task_factory.params.amass_dataset_conf` 与 `validation.amass_dataset_conf` 换成 `stage1_body/conf_fullbody_forehand_clear_aug100_body_local.yaml` 的 80/20 名单，并 `resume_from` T3 checkpoint、`racket_mass_scale` 取 0.25 → 1.0。
3. 22/5 数据只用于复现旧的握拍保持 / 静态击球实验（`experiments/stage3/early_tasks/`），不进 formal family（`docs/narrative/03` §3）。

## 零样本持拍探针（2026-09-20，`experiments/stage2/racket_zero_shot_t3.py`）

把徒手 T3 seed0（`checkpoints/stage1/seed0/T3/checkpoint_39063`）原样放进 `MjxMyoFullBodyRacket`（grip v2 / attachment v4、`derived_rigid` 球拍参考、无肌电项），在 20 条 held-out aug100 动作上确定性 rollout。观测维度与徒手一致（2418），checkpoint 直接可载。

| 球拍质量档 | 提前终止 | held-out 稳定集 12 条存活 | 球拍位置误差 mean / p90 (m) | 球拍姿态误差 mean / p90 (rad) |
|---:|---:|---:|---|---|
| 25% | 8 / 20 | 12 / 12 | 0.246 / 0.464 | 0.511 / 0.834 |
| 50% | 8 / 20 | 12 / 12 | 0.252 / 0.469 | 0.525 / 0.869 |
| 75% | 8 / 20 | 12 / 12 | 0.256 / 0.467 | 0.535 / 0.897 |
| 100% | 7 / 20 | 12 / 12 | 0.260 / 0.474 | 0.545 / 0.914 |

结论：

- 提前终止的是徒手 T3 本来就倒的那 8 条 `video*` 族动作（index 4, 11, 14–19），球拍质量 25%→100% 没有新增一条倒地。球拍课程要解决的**不是平衡**，是球拍位姿。
- 球拍位置误差 0.25 m、姿态误差 0.5 rad，距晋级门（≤0.05 m、≤0.20 rad）差一个数量级，来源是前臂旋前/腕关节在徒手奖励里几乎无约束（`rquat_w_sum` 仅 0.01），拍面误差被杠杆放大。质量每加 25% 误差只涨约 0.005 m，所以四档课程可以压缩为 25%→100% 两档，把预算留给拍面姿态。
- 结果文件：`outputs/stage2_racket_zero_shot/t3_seed0_zero_shot.json`（gitignored，可用上面脚本 2 分钟重建）。

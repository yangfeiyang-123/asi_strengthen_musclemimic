# PEASD 实施指南（三动作统一接口：Forehand Clear / ForehandLift / ChinaJump）

> 目标：按 `doc/整体故事框架与思路/03_仓库改进与待办路线图.md` 的 P1 路线，把
> **Full-354 tracking teacher → reference-free PEASD latent → LAB hitting** 这条主链路
> 实际跑通。本文档只写「现在能直接执行的步骤」，每一步的命令都已在本仓库实测。

## 0. 总览

```
数据(轨迹+EMG) → ① 构建 EMG 参考 tube → ② Stage1/2 训练(PEASD 臂) → ③ Stage3 击球(H2/H3)
```

| 阶段 | 产物 | 入口 |
|---|---|---|
| ① EMG tube | `artifacts/<slug>_peasd_v1/data/emg_reference/<emg_trial_action>/` | `scripts/build_emg_reference_tube.py --action <emg_trial_action>` |
| ② 四臂 latent | sweep plan + 每臂 checkpoint | `run_forehand_clear_pipeline --profile synergy_v3 --action <slug>` |
| ③ 击球 | Stage3 策略 + 评估报告 | `run_incoming_shuttle_hit` |

三动作现在共用同一套入口，动作由 `--action` 选择，取值为
`forehand_clear` / `forehand_lift` / `chinajump`。所有 per-action 事实
（dataset root、retarget variant、clip 划分、config 路径、Stage3 spec）集中在
**唯一事实源** `musclemimic/badminton/action_registry.py`，不再散落在各脚本的字面量里。

> ⚠️ 接口已对齐，**资产尚未齐备**。`--action` 能让三动作走同一条代码路径，但
> forehand_lift / chinajump 还缺一部分 config 与 Stage3 spec。缺失时 launcher
> **fail-closed**：抛出错误并点名缺哪个 registry 字段，**绝不**回退到正手高远球的资产
> （静默回退会让你以为在跑 lift、其实在跑 clear）。要补什么见 §2.5。

### 作用域：主结果 vs 泛化结果

`doc/整体故事框架与思路/01_研究故事与论文叙事主线.md` §5/§16 的方框主线
（Full-354 tracking teacher → reference-free latent → LAB hitting）以
**正手高远球**为主动作，`.../03_仓库改进与待办路线图.md` §5 把 chinajump 标为
「另一任务或后续论文」。这个叙事定位不变：

- **正手高远球（forehandClear_standard）= PEASD 主线**：完整四臂 latent + Stage3，
  §1–§4 的命令为它写、已实测，所有 gate 证据都已封存；
- **forehandlift（forehandLift）/ chinajump（ChinaJump）= 泛化验证**：用**同一套**
  代码路径与同一组超参跑，用来回答「PEASD 是否只在一个动作上成立」。它们进论文的
  形式是泛化表/消融，不是并列主结果。

03 §11 的红线是「不要再建立一套平行 EMG 框架」。本次对齐**遵守**这条红线：没有为
lift/chinajump 复制任何 pipeline，而是把正手高远球那一套里硬编码的动作字面量抽成
registry 参数，三动作共用同一份 plan builder、同一个 tube builder、同一套 gate。
这与「复制一套平行框架」是相反的方向 —— 代码路径数量从 2 条（forehand launcher +
chinajump wrapper）收敛，而不是扩张。

> 泛化实验的说服力取决于**共用代码**。若为某个动作单独改超参或放宽 gate，那条结果
> 就不再是泛化证据。三动作必须跑同一 profile、同一 gate 阈值。

### 三动作就绪度（动手前对照）

| 就绪项 | 正手高远球 | forehandlift | chinajump |
|---|---|---|---|
| registry slug | `forehand_clear` | `forehand_lift` | `chinajump` |
| `training_action` | `forehandClear_standard` | `forehandLift` | `ChinaJump` |
| retarget variant | `raw_smooth_v1` | `optimized_root_smooth_v2` | `optimized_qc10`（cache `optimized`） |
| 轨迹缓存 | ✅ 22 train + 5 val | ✅ 12 train + 4 val | ✅ 8 train + 2 val |
| clip 全部落盘 | ✅ 缺 0 | ✅ 缺 0 | ✅ 缺 0 |
| data QC | ✅ `--action forehand_clear` | ✅ `--action forehand_lift` | ✅ `--action chinajump` |
| EMG trial | ✅ 11 trial | ✅ 13 trial | ✅ 9 trial |
| EMG tube | ✅ `forehand_high_clear` | ✅ `forehand_lift_footwork` | ✅ `china_jump_high_clear` |
| Stage1 body config | ✅ | ✅ | ✅ |
| Stage1R 指扰动 | ✅ | ❌ 待建 | ❌ 待建 |
| Stage2 racket | ✅ | ❌ 待建 | ⚠️ 无球拍动作，见下 |
| latent 四臂 config | ✅ | ❌ 待建 | ❌ 待建 |
| Stage3 spec | ✅ | ❌ 待建 | ⚠️ 不适用 |
| PEASD 可跑到哪 | Step ①②③ 全程 | Step ① 全部 + Step ② 待补 config | Step ① 全部 + Stage1 协同 |

上表 ✅ 项均由本次对齐实测通过（QC 与 tube 三动作各跑一遍）。clip 数是 registry 里
**声明的划分**，不是缓存目录的文件数 —— 缓存目录里还有未纳入划分的 clip 与
`*_analysis.npz` 伴生文件，声明划分才是训练真正读的集合，且已逐一核验存在。

> **chinajump 的球拍问题（不要当成 bug）**：ChinaJump 是起跳动作，没有击球环节，所以
> Stage2 racket 与 Stage3 击球对它**不适用**，不是「缺资产」。chinajump 的泛化验证
> 应停在 Step ① + Stage1 协同（PEASD 的核心主张——privileged EMG 能改善 latent——在
> Stage1/latent 层就可被检验）。把 Stage3 强行套到 chinajump 上是伪泛化。
> forehandlift 有球拍、有击球，是**真正可做全链路泛化**的第二动作。

> 缓存的真实位置是 `datasets/<action>/muscle_trajectory/<variant>/`（由 `configs/env.sh` 把
> `MUSCLEMIMIC_GMR_CACHE_PATH` 指到 `datasets/`）。`caches/AMASS/MyoFullBody/gmr/...` 是空的
> 历史遗留目录（0 文件），**不要**指望它。三动作的 EMG 都来自
> `jidian_measurement/data/P002/S20260721_A`。

### 关于通道数：16 采集 / 15 可比

采集是完整的 **16 通道**，`channel_profile.json` 里每个 sensor 都有真实数据，本流程
不丢弃、不修改任何原始采集。

「15」指的是**能与仿真活性对比**的通道数。差的是 **S1 右斜方肌上束**：MyoFullBody 354
肌肉模型里没有斜方肌 —— 实际上整个肩带稳定肌群（trapezius / rhomboid / serratus /
levator scapulae）都不存在，模型只有附着到肱骨的 `DELT1-3`、`PECM1-3`、`LAT1-3`。
把 S1 强行映射到 DELT 或 LAT 会编造一个生理上不成立的对应（斜方肌稳定肩带，
三角肌抬臂，功能不同），所以 mapping 里它是
`mapping_status="excluded_no_verified_model_homolog"`。

影响范围仅限「仿真 vs 实测」这一个对比维度。NMF 协同分解和 tube 构建在这 15 个通道上
进行；S1 的原始数据完好保留在 `mvc_normalized_emg.npz` 中。

若要让 S1 参与对比，需要给模型补一个斜方肌 actuator —— 那是模型层面的工作，
不属于本数据流程。

---

## 1. 构建 EMG 参考 tube（Step ①）

输入：P002 逐 trial `mvc_normalized_emg.npz` + reviewed mapping JSON。
输出：`EmgPhaseReferenceTube`（`emg_reference_manifest.json` + `emg_reference_tube.npz`）。

```bash
# 正手高远球（11 trial）
.venv/bin/python scripts/build_emg_reference_tube.py --action forehand_high_clear

# 正手挑球 / 上网步法（13 trial）
.venv/bin/python scripts/build_emg_reference_tube.py --action forehand_lift_footwork

# 中国跳高远球（9 trial）
.venv/bin/python scripts/build_emg_reference_tube.py --action china_jump_high_clear
```

三条命令均已实测通过，各自产出 `15 通道 / 20 相位 bin / rank-3` 的 tube。
`--action` 取的是 **EMG trial 目录名**（不是 registry slug）；可选值由
`emg_trial_action_choices()` 从 registry 汇总，因此新增动作只需改 registry。
`--output-dir` 默认按动作分流：正手高远球仍写 `artifacts/forehand_clear_peasd_v1/...`
（保持历史路径不变），其余动作写 `artifacts/<slug>_peasd_v1/...`。

> **每个动作各自拟合 NMF basis，不共用。** 实测三动作的 `synergy_binding.basis_sha256`
> 与 `reference_fingerprint` 两两不同，即 lift/chinajump 的 tube 不含正手高远球的协同
> 信息。这一点必须保持：若让三动作共用一个 basis，泛化实验就变成了「把 clear 的先验
> 灌给别的动作」，PEASD 的主张会被这条捷径污染。
> 相关边界见 memory 里「基本动作→完整动作协同复用」的结论（跨动作复用只能达到自身
> 上限的 61–74%）。

脚本做的事：

- 从 `jidian_measurement/data/P002/S20260721_A/trials/<action>/trial_XXX/mvc_normalized_emg.npz`
  读取全部 16 通道 MVC 归一化包络；建 tube 时只取 15 个有模型同源肌的通道
  （`excluded_sensor_ids=[1]`，理由见 §0）—— 原始文件不改，S1 数据仍在；
- 每个 trial 按相位均匀分 `--phase-bins`（默认 20）个 bin；
  - `forehand_high_clear`：用 101 样本时间归一化（`software_cue_exploratory`）的相位轴；
  - `forehand_lift_footwork` / `china_jump_high_clear`：对原始时域序列做**时长归一化**
    到同一 0–100% 相位轴（走同一段代码，无 per-action 分支）；
- 在 15 通道数据上拟合 NMF basis（`--synergy-rank 3`，多 seed 最佳初始化）；
- 用 `build_phase_reference_tube` 写出 median/MAD tube，`review_status=provisional`、
  `training_enabled=false` —— 这是**fail-closed 默认**，未 review 前任何下游都不能拿它做训练。

验证：

```bash
.venv/bin/python - <<'PY'
from musclemimic.physiology.emg_reference import (
    load_emg_phase_reference_tube, resolve_emg_reference_reward_gate)
ROOTS = {
    "forehand_high_clear": "artifacts/forehand_clear_peasd_v1",
    "forehand_lift_footwork": "artifacts/forehand_lift_peasd_v1",
    "china_jump_high_clear": "artifacts/chinajump_peasd_v1",
}
for action, root in ROOTS.items():
    t = load_emg_phase_reference_tube(f"{root}/data/emg_reference/{action}")
    print(action, t.channel_count, t.synergy_count, t.phase_bin_count,
          t.anchor_valid.mean(), t.review_status)
    try:
        resolve_emg_reference_reward_gate(t, enabled=True)  # 应当抛错
    except ValueError as e:
        print("  gate:", str(e)[:60])
PY
```

预期：三行都是 `15 3 20 1.0 provisional`，gate 各报「mapping review must complete」。

> mapping review 是**三动作共享**的一道门：15 可比通道与 354 肌肉的对应关系与动作无关，
> 所以 review 一次解冻全部三个动作。反过来说，在 review 完成前，三个动作的正式结果
> 都出不来 —— 这是当前泛化实验的**共同前置**，不是某个动作的局部阻塞。

### 何时解冻（review 门）

tube 从 `provisional` 升到 `verified` 需要：

1. 人工复核 `configs/physiology/emg_badminton_synergy_16_v2_myofullbody_observation_v1.json`
   的 mapping（多 compartment 聚合、biceps/triceps、腕屈伸组、左右侧、等权是否合理）；
2. 在 mapping JSON 里把 `review_status` 改成 `verified`、填 `review_evidence`；
3. 重建 tube 时 `review_status="verified", training_enabled=True`（脚本加 `--verified` 开关，
   或改 `build_phase_reference_tube` 调用）。

在那之前，所有训练都应使用**未解冻 tube 也能跑的代码路径**（见下）。

---

## 2. Stage1/2 训练：四条臂（Step ②）

> 本节命令对三动作同构，动作由 `--action` 选择（默认 `forehand_clear`，
> 省略时行为与本次对齐前**完全一致**）。lift/chinajump 目前会在缺 config 的步骤
> fail-closed 并点名缺哪个字段，见 §2.5。

主入口是 `fullbody/run_forehand_clear_pipeline.py --profile synergy_v3`。
它一次只规划/执行一个阶段（`--execute_step`），不改动作空间、不新造框架。四个 EMG flag
（`--emg_reference_manifest` / `--emg_synergy_dim` / `--emg_shuffle_context_ablation` /
`--emg_no_context_dropout`）由 `PipelineArtifacts` dataclass 字段在
`run_forehand_clear_pipeline.py:3100` 动态注册为 CLI，已实测存在。

### 2.1 规划四臂（不启动训练）

> 先看一个前提：**三动作共同的公共地板是 Stage 1 前六步**（data_release_validate /
> data_qc / stage1_train / stage1_gate / stage1_visual_gate / stage1_promote），
> 三动作都能跑，用 profile `stage1_aligned`：
>
> ```bash
> for a in forehand_clear forehand_lift chinajump; do
>   python -m fullbody.run_forehand_clear_pipeline --profile stage1_aligned --action $a \
>     --output_dir /tmp/s1_$a
> done
> ```
>
> 这个 profile 已实测三动作各产出 6 步 plan，且每步都指向各自的 dataset / variant /
> config（不是 clear 的）。它是「三动作目前都能跑的共同证据」——在 config 补齐前，
> 用它作为泛化主线的**进度基线**，比等全量 latent 更早暴露问题。
> `stage1_aligned` **不包含** stage1r（指扰动）——那是 lift/chinajump 还没建的步骤。

把动作与 tube 路径提成变量，同一段脚本对三动作复用：

```bash
OUT=/tmp/peasd_arms
ACTION=forehand_clear          # 或 forehand_lift / chinajump
TUBE=artifacts/forehand_clear_peasd_v1/data/emg_reference/forehand_high_clear/emg_reference_manifest.json
# forehand_lift: artifacts/forehand_lift_peasd_v1/data/emg_reference/forehand_lift_footwork/emg_reference_manifest.json
# chinajump:     artifacts/chinajump_peasd_v1/data/emg_reference/china_jump_high_clear/emg_reference_manifest.json

python -m fullbody.run_forehand_clear_pipeline --profile synergy_v3 --action $ACTION \
  --output_dir $OUT/${ACTION}_s2b_baseline
python -m fullbody.run_forehand_clear_pipeline --profile synergy_v3 --action $ACTION \
  --output_dir $OUT/${ACTION}_s2c_peasd \
  --emg_reference_manifest $TUBE --emg_synergy_dim 3
python -m fullbody.run_forehand_clear_pipeline --profile synergy_v3 --action $ACTION \
  --output_dir $OUT/${ACTION}_s2d_shuffled \
  --emg_reference_manifest $TUBE --emg_synergy_dim 3 --emg_shuffle_context_ablation
python -m fullbody.run_forehand_clear_pipeline --profile synergy_v3 --action $ACTION \
  --output_dir $OUT/${ACTION}_s2e_nodropout \
  --emg_reference_manifest $TUBE --emg_synergy_dim 3 --emg_no_context_dropout
```

`--emg_synergy_dim 3` 对三动作都用 3：tube 各自拟合、但 rank 相同，这样「四臂之间只差
EMG context」的对照关系在三动作间保持一致。**不要**为某个动作单独调 rank。

对应文档 §26.2 消融矩阵：

| 臂 | 命令差异 | 含义 |
|---|---|---|
| S2-B baseline | 无任何 EMG flag | EMG-free 对照 |
| S2-C PEASD | 3 个 flag | 训练期 posterior 读 EMG 协同 |
| S2-D shuffled | 多 `--emg_shuffle_context_ablation` | 负对照：打乱 context |
| S2-E no-dropout | 多 `--emg_no_context_dropout` | 去掉 context dropout |

检查四臂命令只在 EMG 尾部不同（基线必须零 EMG token）：

```bash
for d in s2b_baseline s2c_peasd s2d_shuffled s2e_nodropout; do
  python - <<PY
import json
plan = json.load(open("$OUT/$d/pipeline_plan.json"))
step = [s for s in plan["steps"] if s["name"] == "latent_dimension_sweep"][0]
print("$d", [t for t in step["command"] if "emg" in t.lower()])
PY
done
```

### 2.2 执行

按依赖顺序逐步推进（每步都是 gate 保护的，先跑前置的）：

```bash
python -m fullbody.run_forehand_clear_pipeline --profile synergy_v3 \
  --output_dir $OUT/s2c_peasd --execute_step data_qc
python -m fullbody.run_forehand_clear_pipeline --profile synergy_v3 \
  --output_dir $OUT/s2c_peasd --execute_step stage1_train
# ... stage1_gate → stage1_visual_gate → stage1_promote → ...
# stage2_train → ... → stage2_promote
# latent_dimension_sweep（只生成计划）→ latent_dimension_execute（真正训练）
```

执行到 `latent_dimension_execute` 时，sweep 会在
`$OUT/s2c_peasd/synergy_v3/latent_synergy/<run_name>/` 下为每个
`d<dim>_<decoder><_peasd|_peasd_shuffled|_peasd_nodropout>_seed<N>` 训练 latent。

> 注意：`--emg_reference_manifest` 指向**未解冻** tube 时，训练会通过（tube 只在
> Stage 1 的 EMG reward 里 fail-closed；privileged latent 路径读取 tube 的
> mean/scale/valid 数组即可训练）。若你想先跑通主链路，可先用未解冻 tube 跑
> S2-C 验证管线，等 mapping review 完成后再用 verified tube 出正式结果。

### 2.3 动作注册表：唯一事实源

`musclemimic/badminton/action_registry.py` 是三动作全部差异的**唯一**声明处。
每个动作一条 `ActionSpec`，字段分三类：

- **必备且已填**：`action_id`（dataset 目录名）、`slug`、`data_variant`、
  `source_namespace`（`temp/<variant>` 或 `wham/<variant>`）、`cache_namespace`、
  `train_motions` / `val_motions`、`release_manifest`、`emg_trial_actions`、
  `env_prefix`、`stage1_config`、`synergy_grouping`；
- **可选、缺则该步骤 fail-closed**：`stage1r_config`、`stage1r005_config`、
  `stage2_config`、`stage2_extend_config`、`student_bc_config`、`student_ppo_config`、
  `latent_lab_config`、`latent_synergy_config`、`stage3_spec` /
  `stage3_v2_spec` / `stage3_direct_spec`、`coverage_phase_schema`、`synergy_preset`、
  `racket_attachment`；
- **构造即校验**：模块导入时对每条 `validate()`（split 不重叠、无重复、非空），
  import 失败比运行到一半才发现 split 写错要早。

用法：

```python
from musclemimic.badminton.action_registry import resolve, action_choices
p = resolve("forehand_lift")
p.action_id            # 'forehandLift'
p.motion_path("forehandLift-1")
#    'forehandLift/muscle_trajectory/optimized_root_smooth_v2/forehandLift-1'
p.source_namespace     # 'temp/optimized_root_smooth_v2'
action_choices()       # ('chinajump', 'forehand_clear', 'forehand_lift')
```

`resolve()` 同时接受 slug、dataset 目录名、EMG trial 名（`forehand_lift_footwork` 会
解析到 lift），launcher 与 QC 都走它。`emg_trial_action_choices()` 汇总三动作的
EMG trial 名，tube builder 的 `--action` 用它。

**为什么必须 fail-closed**：可选字段缺失时，`require(field)` 抛 `ValueError`，消息里
点名动作 + 字段 + 写在哪。若改成回退默认值，lift 的 Stage2 会静默用正手高远球的
球拍 config，跑出来的「泛化结果」其实是 clear 的资产 —— 这类错误不会崩、只会污染
结论，因此宁可早崩。

历史入口 `fullbody/run_chinajump_synergy_pipeline.py` 保持不变：它做的是
**Stage1 协同 release 的准备/校验**（primitive catalog、coverage gate、
phase schema），与本 pipeline 的 latent 四臂是不同层的工作，不是重复路径。
chinajump 的协同准备仍走它：

```bash
python -m fullbody.run_chinajump_synergy_pipeline plan \
  --primitive-catalog fullbody/config_specific_task/stage1_body/primitive_catalog/chinajump_primitives_p01_canonical_tonic_v5.json
```

### 2.4 三动作 data QC

QC 现在按动作解析 dataset root、source/cache 变体与 clip 划分，三条命令都已实测通过：

```bash
python -m musclemimic.badminton.data_qc --action forehand_clear   # 22 train / 5 val
python -m musclemimic.badminton.data_qc --action forehand_lift    # 12 train / 4 val
python -m musclemimic.badminton.data_qc --action chinajump        # 8 train / 2 val
```

注意 chinajump 的 source 与 cache 变体名**不同**（source `wham/optimized_wham`，
cache `muscle_trajectory/optimized`，变体标识 `optimized_qc10`）；另两个动作
source/cache 同名。这个差异由 registry 的 `source_namespace` / `cache_variant`
两个字段分别承载，QC 不再假设两者相等 —— 原先写死 `temp/<variant>` 是
chinajump 过不了 QC 的直接原因。

原先 `data_qc.py` 里 `len(TRAIN_MOTIONS) != 22 or len(VAL_MOTIONS) != 5` 的硬断言
已改为按 registry 声明的划分校验，因此仍然守着「顺序与集合必须完全匹配」这条，
只是期望值来自动作而非常量。

### 2.5 还缺什么资产（lift / chinajump）

接口已通，以下 config 需要你或 codex 补齐；每一项都会在对应步骤 fail-closed 报出：

**forehandlift（可做全链路泛化，优先补）**

| registry 字段 | 要建的东西 |
|---|---|
| `stage1r_config` / `stage1r005_config` | 指扰动 Stage1R 两级（照 clear 的 `finger_qpos_perturb_scale` 0.1/0.05 建） |
| `stage2_config` / `stage2_extend_config` | 球拍 Stage2（lift 有球拍，可参照 `conf_fullbody_badminton_racket_local`） |
| `student_bc_config` / `student_ppo_config` | distill student |
| `latent_lab_config` / `latent_synergy_config` | latent 四臂 |
| `synergy_grouping` | lift 的 354 分区 json（可否复用 clear 的分区需人工判断，勿默认复用） |
| `stage3_spec` 等 | 击球 Stage3 spec |

**chinajump**

只需补 `latent_lab_config` / `latent_synergy_config` / `synergy_grouping` 即可做
latent 四臂泛化。Stage2 racket 与 Stage3 击球**不要补** —— ChinaJump 是起跳动作、
无击球环节，registry 里这些字段留空是**正确状态**，不是待办。

> `experiments/posttrain/forehand_net_lift_v1.yaml` 是**另一条** ForehandNetLift 回球
> 调参轨道（含 7 处 `/data3` 硬编码路径、指向不存在的 `gmr_cache`/`amass_npz`），
> 与本 PEASD lift 无关，**不要**拿它当 lift 的 `stage3_spec`。

---

## 3. Stage3 击球：H2 / H3（Step ③）

> Stage3 只对**有击球环节**的动作成立：正手高远球（已就绪）与 forehandlift（待补 spec）。
> chinajump 无击球，跳过 Stage3 是设计意图，不是缺口 —— 详见 §2.5。
> 下面命令加 `--action` 即可切到 lift（spec 补齐后）。

文档 §26.3：

| ID | 含义 | 怎么选 |
|---|---|---|
| H1 | latent baseline + LAB | 用 S2-B 选出的 latent checkpoint |
| H2 | PEASD latent + LAB | 用 S2-C 选出的 latent checkpoint |
| H3 | PEASD latent + LAB + grouped right-arm correction | H2 + `--bounded-residual-groups-json` |

### 3.1 准备 grouped residual 配置（H3）

```bash
cat > /tmp/h3_groups.json <<'JSON'
{
  "wrist_forearm": {"alpha": 0.05},
  "elbow_forearm": {"alpha": 0.03},
  "shoulder": {"alpha": 0.02}
}
JSON
```

可用的组名与 actuator 名单（`environment/overall_environment/src/stage3_lab.py`）：
`wrist_forearm`（10 个腕/前臂）、`elbow_forearm`、`shoulder`（后两者默认空，需显式列
`actuator_names` 才能启用）。每组独立 alpha，`[0, 0.10]` 内。

### 3.2 launcher 传入（H3）

在 canonical pipeline 上叠加 residual：

```bash
python -m fullbody.run_forehand_clear_pipeline --profile synergy_v3 --action $ACTION \
  --output_dir $OUT/${ACTION}_h3 \
  --emg_reference_manifest $TUBE \
  --emg_synergy_dim 3 \
  --stage3_bounded_residual_groups_json /tmp/h3_groups.json
```

这会往 `stage3_v2_base_only`、`stage3_static_target_train`、`stage3_v2_train`、
`stage3_v2_evaluate` 四个步骤注入 `--bounded-residual-groups-json`。H2 则不传该字段，
用同一个 latent checkpoint。

### 3.3 直接运行（不走 launcher）

```bash
# H2 评估（用 S2-C 选出的 latent）
python -m musclemimic.badminton.scripts.run_incoming_shuttle_hit \
  --spec experiments/posttrain/incoming_shuttle_hit_impact_recovery_v2.yaml \
  --stage evaluate \
  --latent-checkpoint $OUT/s2c_peasd/synergy_v3/latent_synergy/selected/best_synergy \
  --checkpoint $OUT/h2/stage3_lab/policy_latest.json \
  --episodes 128 \
  --target-bank <train_bank> --eval-target-bank <eval_bank> \
  --out-dir $OUT/h2/evaluate

# H3 评估（加 grouped residual）
... 同样命令 + --bounded-residual-groups-json /tmp/h3_groups.json
```

---

## 4. 对照与指标

### 4.1 关键 gate（文档 §884）

> **若真实 prior 不优于 shuffled prior，停止进入 privileged distillation。**

S2-C 必须优于 S2-D，否则说明模型没有真正使用 EMG 结构，先修数据/mapping 再继续。

### 4.2 必报指标（文档 §27）

- **动作**：joint RMSE、keypoint error、root error、racket trajectory、fall/early termination；
- **肌肉**：activation energy、saturation、M-channel correlation、peak phase、onset/offset、co-contraction；
- **协同**：held-out VAF、per-channel VAF、W cosine、subspace angle、H correlation、bootstrap stability；
- **蒸馏**：train/val action MSE、student closed-loop return、DAgger improvement、prior/posterior gap、active latent dims；
- **击球**：hit rate、positive outgoing-z rate、cross-net rate、legal return、net clearance、landing accuracy、no-fall、held-out feed generalization。

### 4.3 统计

- 统计单位是 trial/subject/seed，不是帧；
- 每个 RL 组至少多个相同 seeds；
- 报告均值、标准差、失败率、effect size、置信区间；
- 单一 subject/session（P002）不报告 population-level 结论；
- paired / unpaired 设计严格分开。

---

## 5. 当前已知限制（诚实记录）

- **接口已对齐，但泛化实验尚未跑过**：本次工作让三动作共用同一条代码路径，并实测
  Step ①（tube）与 data QC 三动作全通。Step ② 的 latent 四臂**只在正手高远球上真正跑过**；
  lift/chinajump 还缺 §2.5 列的 config，且没有任何一次 latent 训练产出。因此现在**不能**
  声称「方法在三动作上成立」——能声称的是「三动作已具备用同一套代码验证的条件」；
- **lift / chinajump 的 latent 与 Stage3 结果为零**：不要在论文里预告尚未存在的数字；
- **tube 是 `provisional`（三动作同样）**：mapping 未人工复核，`training_enabled=false`。
  S2-C/S2-D 的训练仍可跑（privileged latent 不 fail-closed），但**正式论文结果**必须等
  review 后用 verified tube 重跑。review 一次解冻三动作，是共同前置；
- **事件对齐是 `software_cue_exploratory`**：不是独立视频/硬件证据的 impact 对齐，
  tube 的相位轴来自软件 cue 归一化。论文中如实写 exploratory、不宣称 impact-aligned；
- **lift / ChinaJump 的 tube 相位轴是时长归一化**：与 forehand 的 101 样本归一化同构，
  但不是同一套事件标注。跨动作比较 tube 形状时须说明这一点；
- **lift 的可比通道集沿用了 clear 的 15 通道 mapping**：mapping 本身与动作无关
  （354 肌肉 ↔ 传感器的解剖对应），但 lift 是**下肢步法为主**的动作，15 个上肢/躯干
  通道对它的覆盖是否充分**未经人工核验**。这会限制 lift 的「仿真 vs 实测」维度，
  不影响 privileged latent 训练。相关先例见 memory「基本动作→完整动作协同复用」：
  采样偏下肢时结论有边界；
- **chinajump 无 Stage3 是设计而非缺口**：起跳动作没有击球环节。若论文需要「三动作
  全链路」的说法，只有 clear + lift 两个动作能支撑，chinajump 的泛化证据止于 latent 层；
- **P002 单 subject / 单 session**：不能外推到人群，不能报告 population CI；
- **S9 渐进性电极失效**：P002 中 S9 右腹外斜肌 near-flatline（monotonic decay），
  位于 15 个可比通道内，会影响与仿真的对比。不能通过放宽阈值批量转 valid；
- **MVC >200% 的通道**（S2 最大 ~10.33×MVC）：正式纳入前必须人工复核，
  不能把归一化值悄悄截到 [0,1]。

---

## 6. 参考

- 方法叙事：`doc/整体故事框架与思路/01_研究故事与论文叙事主线.md`
- 三阶段方法：`doc/整体故事框架与思路/02_三阶段方法与肌电参与机制.md`
- 路线图（P0–P2）：`doc/整体故事框架与思路/03_仓库改进与待办路线图.md`
- EMG 数据契约：`docs/jidian_emg_integration.md`
- 蒸馏 runbook：`docs/forehand_clear_distillation_runbook.md`
- 代码入口：
  - **动作注册表（改动作先看这里）**：`musclemimic/badminton/action_registry.py`
  - tube：`scripts/build_emg_reference_tube.py`、
    `musclemimic/physiology/emg_reference.py`、`emg_anchor.py`
  - data QC：`musclemimic/badminton/data_qc.py`
  - latent：`musclemimic/latent_muscle/train_latent.py`、`networks.py`
  - launcher：`fullbody/run_forehand_clear_pipeline.py`（`--action` 选动作）
  - chinajump Stage1 协同 release：`fullbody/run_chinajump_synergy_pipeline.py`
  - Stage3：`environment/overall_environment/src/stage3_lab.py`、
    `musclemimic/badminton/scripts/run_incoming_shuttle_hit.py`

---

## 7. 给 codex 的落实清单

按依赖顺序，前两项**已完成**（本次对齐），从第 3 项开始做：

1. ~~registry + data QC + tube builder 三动作对齐~~ ✅ 已实测通过；
2. ~~launcher `--action` 参数化，正手高远球 plan 保持字节一致，`stage1_aligned` profile~~ ✅；
3. 补 forehandlift 的 config（§2.5 表格），每补一个跑一次
   `--action forehand_lift --profile synergy_v3` 看是否还 fail-closed；
4. 补 chinajump 的 latent 三件（`latent_lab_config` / `latent_synergy_config` /
   `synergy_grouping`）；**不要**给它补 Stage2 racket 或 Stage3；
5. 完成 mapping review，用 `--verified` 重建三动作 tube（解冻 `training_enabled`）；
6. 三动作各跑四臂 latent，同 rank、同 gate 阈值，产泛化表。

> 进度基线：目前三动作能跑的就是 `stage1_aligned` 六步。任何动作、任何新 config 的
> 改动，都先用 `--profile stage1_aligned` 验证 6 步仍通，再往 `synergy_v3` 走。

回归基线（任何改动后都应通过）：

```bash
# 正手高远球 plan 必须与对齐前字节一致
python -m fullbody.run_forehand_clear_pipeline --profile synergy_v3 --output_dir /tmp/chk
# 三动作 QC
for a in forehand_clear forehand_lift chinajump; do
  python -m musclemimic.badminton.data_qc --action $a || echo "FAIL $a"
done
# 单测
.venv/bin/python -m pytest tests/test_action_registry.py \
  tests/unit/test_forehand_clear_data_qc.py -q
```

# Jidian 16 通道 sEMG 与 MuscleMimic 的严格集成契约

> 状态日期：2026-07-26  
> 适用采集 profile：`badminton_synergy_16_v2`（右手持拍）  
> 当前结论：采集、审计、严格导入和两类评估入口已经接通；现有真实数据没有任何一个带独立证据审计的 `racket_contact`，因此 **official eligible trial 数仍为 0**，尚未产生正式 Phase 3 结论，也没有因此启动或重启训练。

本文定义从 `jidian_measurement` 到 MuscleMimic 的唯一正式入口，并回答三个容易混淆的问题：

1. 采集到的 16 个通道中，哪些可以与 354 actuator 模型比较；
2. 什么时候允许做逐 trial paired 指标，什么时候只能做 independent-cohort 指标；
3. 为什么“已有可读数据”不等于“已有可发表的 impact 对齐证据”。

## 1. 当前真实库存与结论边界

以下数字来自对仓库内现有采集目录的只读盘点，不表示 trial 已通过正式导入：

| 数据 | Profile | Raw | 人工 valid | 已预处理且 `analysis_ready` | 人工 valid 与 ready 交集 | 有证据的 impact | Official eligible |
|---|---|---:|---:|---:|---:|---:|---:|
| `P001/PILOT01` | `badminton_synergy_16_legacy_actual_v1` | 35 | 33 | 0 | 0 | 0 | 0 |
| `P002/S20260721_A` | `badminton_synergy_16_v2` | 63 | 58 | 31 | 30 | 0 | 0 |

`P001/PILOT01` 没有同配置完整 MVC 和正式 processed artifact，只能保留为 legacy/exploratory 数据，不能进入本页定义的 V2 strict importer。`P002` 的信号与处理文件虽已存在，但当前事件表中的物理事件没有通过 `events.annotation.audit.jsonl` 绑定独立证据。软件 cue 不能替代 `racket_contact`、`takeoff` 或 `landing`。

因此当前可以诚实声称的是：

- 已获得 98 个 16 通道 raw trial，并建立了非破坏性、可审计的数据处理链；
- `P002` 有 30 个“人工 valid 且 preprocessing-ready”的候选 trial；
- 所有候选仍因缺少证据化 impact 而不能进入正式对齐与评估；
- 当前数据不能证明仿真与真人逐 trial 一致，也不能证明 IMR 改善真实人体协同。

## 2. 固定的 16 采集 / 15 可比契约

采集必须永久保留 16 个通道。模型观测层只比较 S2–S16，共 15 个通道：

- S1 `right:upper_trapezius`：采集、QC、归一化和来源哈希均保留；354 actuator inventory 中没有经确认的上斜方肌同源 actuator，因此状态固定为 `excluded_no_verified_model_homolog`。
- S2–S16：通过显式 actuator 名称和权重投影到模型 activation；不能按字符串相似度自动猜测。
- 严禁把 S1 猜成 `LTpT` 或任何其他 actuator，也不能为了得到 16/16 指标而静默删除或替代通道。

唯一 mapping 是：

[`configs/physiology/emg_badminton_synergy_16_v2_myofullbody_observation_v1.json`](../configs/physiology/emg_badminton_synergy_16_v2_myofullbody_observation_v1.json)

其 schema 是 `emg_observation_mapping_v2`，并精确绑定：

- `badminton_synergy_16_v2` profile 及其 SHA-256；
- `myofullbody_354_muscle_taxonomy_audit_v2` 及其 fingerprint；
- runtime model hash 和 ordered actuator schema hash；
- `acquired_channel_count=16`、`comparable_channel_count=15`、`excluded_sensor_ids=[1]`。

该 mapping 当前仍为 `review_status=provisional`、`training_enabled=false`。使用 `--allow-provisional-mapping` 只能生成探索性报告；在解剖映射完成人工复核前，不能把它写成正式人体效度，也不能作为训练监督或 reward。

## 3. 数据流与两个互斥的比较设计

```text
jidian_measurement 原始记录
  -> V2 MVC 预处理 + preprocessing QC
  -> 独立证据补标 impact + 哈希审计
  -> jidian_emg_selection_v1（预注册选择集）
  -> musclemimic-jidian-emg-import（全有或全无）
  -> strict Jidian EMG NPZ
       |-> paired_same_reference_v1 -> musclemimic-emg-eval
       `-> unpaired_action_cohort_v1 -> musclemimic-emg-cohort-eval

Stage3 成功 rollout
  -> stage3_signal_trial_identity_v2
  -> Stage3 physical-signal NPZ
       |-> 与 paired EMG 共用逐 trial reference fingerprint
       `-> 与 unpaired EMG 仅共用 action/hand/split/comparison_set_uid
```

两个设计不能在运行后互相改名：

| 设计 | 需要相同 trial 数 | 需要逐 trial `reference_trial_fingerprint` | 允许指标 |
|---|---:|---:|---|
| `paired_same_reference_v1` | 是 | 是，且 EMG/仿真逐 trial 完全相同 | envelope correlation、DTW、onset/peak timing、paired NMF/phase 等 |
| `unpaired_action_cohort_v1` | 否 | 禁止出现 | 两个 cohort 分别 NMF 后的 W 几何、matching 与各自 VAF；逐 trial/H/timing 指标显式 unavailable |

“动作名称相同”不是 pairing 证据；“都在 impact 附近裁剪”也不是 pairing 证据。只有同一个外部 reference trial 被两侧 artifact 以相同 SHA-256 绑定，才允许进入 paired evaluator。

## 4. 先把 impact 变成可审计事实

### 4.1 证据要求

每个拟纳入 trial 必须依据保留的视频、同步硬件记录或其他可复核证据补标 `racket_contact`。`annotate-event` 会原子更新 `events.csv`，并向 `events.annotation.audit.jsonl` 写入 prepared/committed 两阶段记录。strict importer 会重新验证：

- 证据对象的 `evidence_sha256` 是 64 位小写 SHA-256；
- committed 记录有且只有一个完全相同的 prepared 前驱；
- annotation manifest hash 正确；
- `events.csv` 当前字节 hash 等于最新一次全局 committed 记录的 `after_sha256`；
- 当前目标 event 行与该 event 最新一次 committed 记录的 `after_event` 完全一致（允许之后合法补标其他 event）；
- confidence 达到 selection manifest 的阈值；
- impact 前后窗口没有越过 trial 边界。

### 4.2 补标命令

在 `jidian_measurement` 目录中运行；sample index 和证据引用必须来自逐帧核对，下面只是格式示例：

```powershell
$TrialPath = '.\data\P002\S20260721_A\trials\forehand_high_clear\trial_001'
$Evidence = 'D:\EMG_Evidence\P002_S20260721_A_camA.mp4'
$EvidenceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Evidence).Hash.ToLower()

python -m emg.cli annotate-event `
  --trial-path $TrialPath `
  --event-name racket_contact `
  --sample-index 7124 `
  --source manual_video `
  --confidence 0.95 `
  --annotator OPERATOR_01 `
  --evidence-reference 'P002_S20260721_A_camA.mp4#frame=1842' `
  --evidence-sha256 $EvidenceHash `
  --notes 'Racket-shuttle contact confirmed by frame review'
```

不得把 `movement_cue` 改名为 impact，不得根据 EMG 峰值反推接触点，也不得只编辑 `events.csv` 而绕过审计。修改已有人工事件时必须显式使用 `--overwrite`；多人复核时可用 `--expected-before-sha256` 防止覆盖并发修改。

## 5. Strict Jidian importer

### 5.1 Selection manifest

从以下模板复制后填写，不要修改模板本身：

[`configs/physiology/jidian_emg_selection_example.json`](../configs/physiology/jidian_emg_selection_example.json)

关键字段：

- `session_path`：一个明确 session，不递归扫描整个数据根目录；
- `subject_uid` / `session_uid`：来自伪名 registry，不要直接使用姓名；
- `trial_ids`：预先选择的完整集合，格式必须为 `<action_id>_trial_NNN`；
- `dataset_split`：只能是 `heldout`、`validation` 或 `test`；
- `training_session_uids`：必须非空，且不得包含当前 held-out session；
- `comparison_design` 与 `comparison_set.comparison_set_uid`：必须与 Stage3 identity 完全一致；
- `alignment.mode`：只能是 `impact`，没有 cue 或 full-trial fallback。

paired 选择还必须增加一个覆盖所有且仅覆盖已选 trial 的映射：

```json
{
  "comparison_design": "paired_same_reference_v1",
  "comparison_set": {
    "comparison_set_uid": "forehand-clear-heldout-v1",
    "reference_trial_fingerprints": {
      "forehand_high_clear_trial_001": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    }
  }
}
```

unpaired 设计禁止包含 `reference_trial_fingerprints`。

### 5.2 导入命令

从 MuscleMimic 根目录运行：

```bash
export JIDIAN_SELECTION=outputs/emg/jidian_selection.json
export JIDIAN_EMG_NPZ=outputs/emg/jidian_strict.npz
export JIDIAN_IMPORT_AUDIT=outputs/emg/jidian_import.audit.json

uv run musclemimic-jidian-emg-import \
  "$JIDIAN_SELECTION" "$JIDIAN_EMG_NPZ" \
  --audit-json "$JIDIAN_IMPORT_AUDIT"
```

导入器不会重新滤波、平滑、裁剪幅度或重新归一化；它只读取经过 V2 MVC 处理的 `normalized_envelope`，按证据化 impact 裁剪并封存全部来源哈希。选择集采用全有或全无规则：任意一个 trial 不合格，命令返回非零、写 rejection audit，并且不留下部分 NPZ。这防止失败 trial 被事后静默删掉而改变预注册 cohort。

当前 `P002` 直接运行 strict import 的预期结果是拒绝，因为 0 个 trial 有符合要求的事件审计；这是正确的 fail-closed 行为，不是 importer 故障。

## 6. Stage3 identity v2 与信号导出

Stage3 必须按比较设计选择一个模板：

- unpaired cohort：[`configs/public/stage3_signal_trial_identity_template.json`](../configs/public/stage3_signal_trial_identity_template.json)
- 真正 paired：[`configs/public/stage3_signal_trial_identity_paired_template.json`](../configs/public/stage3_signal_trial_identity_paired_template.json)

`stage3_signal_trial_identity_v2` 强制记录 `action_id`、`handedness`、`comparison_design`、`comparison_set_uid` 和 `model_taxonomy_path`。导出器会从 taxonomy 解析并验证 taxonomy fingerprint、base-body runtime model hash、actuator schema hash，并把这些绑定写入 simulation NPZ；Stage-3 含球拍/羽球的完整场景哈希另存为 `scene_runtime_model_hash`，避免把 base taxonomy hash 冒充场景本身的 hash。

两个模板是互斥的，不能用同一个 simulation NPZ 先跑 unpaired 再改名为 paired。unpaired identity 的 trial 行不含 `reference_trial_fingerprint`。paired identity 的每一行必须增加该字段，并与 strict Jidian EMG NPZ 中对应 trial 完全相同。两侧还必须满足：

- 同一 `action_id`、handedness、split 和 `comparison_set_uid`；
- held-out session 不出现在 `training_session_uids`；
- paired 模式有相同 trial UID 集合和 trial 数；
- policy checkpoint、promotion artifact 与 formal synergy basis 的 fingerprint 可由 sealed Stage3 policy evidence 验证。

Stage3 rollout/信号导出仍使用根 README 的 canonical Stage3 evaluate 命令。这里不启动训练；修改 identity 也不授权恢复或覆盖任何 checkpoint。

## 7. 两个评估入口

统一使用 V2 mapping：

```bash
export EMG_MAPPING=configs/physiology/emg_badminton_synergy_16_v2_myofullbody_observation_v1.json
export STAGE3_SIGNAL_NPZ=outputs/synergy_v3/stage3_signal/simulation_signals.npz
export POLICY_EVIDENCE=outputs/synergy_v3/stage3_paired/paired_comparison.json
```

### 7.1 真正 paired 的逐 trial 评估

只有 selection 和 Stage3 identity 都声明 `paired_same_reference_v1`，且每个 reference fingerprint 完全相同时，才运行：

```bash
uv run musclemimic-emg-eval \
  --simulation-npz "$STAGE3_SIGNAL_NPZ" \
  --emg-npz "$JIDIAN_EMG_NPZ" \
  --mapping-json "$EMG_MAPPING" \
  --policy-evidence-json "$POLICY_EVIDENCE" \
  --synergy-rank 2 \
  --allow-provisional-mapping \
  --output-json outputs/synergy_v3/emg/paired_report.exploratory.json
```

`--allow-provisional-mapping` 明确把当前报告限制为 exploratory。移除该参数后，当前 provisional mapping 会按设计拒绝，而不是自动升级为正式结论。

### 7.2 Independent action cohort 评估

真人与仿真不是同一批 trial 时，selection 和 Stage3 identity 必须都声明 `unpaired_action_cohort_v1`，并复用同一个预注册 `comparison_set_uid`：

```bash
uv run musclemimic-emg-cohort-eval \
  --simulation-npz "$STAGE3_SIGNAL_NPZ" \
  --emg-npz "$JIDIAN_EMG_NPZ" \
  --mapping-json "$EMG_MAPPING" \
  --policy-evidence-json "$POLICY_EVIDENCE" \
  --synergy-rank 2 \
  --allow-provisional-mapping \
  --output-json outputs/synergy_v3/emg/unpaired_cohort_report.exploratory.json
```

此入口允许两侧 trial 数不同，分别拟合 NMF 后只比较 channel-space basis W、matching 和 cohort 内 VAF。H 相关、逐 trial 波形相关、DTW、onset/peak timing 和共享 phase 会在报告中明确标记 unavailable，不能事后另行计算来暗示 pairing。

可先检查两个 evaluator 的输入契约，而不读数据或写报告：

```bash
uv run musclemimic-emg-eval --dry-run
uv run musclemimic-emg-cohort-eval --dry-run
```

## 8. 仍未解决的真实数据风险

### 8.1 Impact 证据：当前硬阻塞

现有数据的 `racket_contact` / takeoff / landing 等物理事件均未形成可验证 annotation audit，因此 official eligible 为 0。补标必须回到原始独立证据；如果证据不存在，就保持 unavailable，不能估计或补造。

### 8.2 P002 的 S9 near-flatline

P002 共 63 个 processed trial，其中 32 个未通过 preprocessing-ready；审计显示 27 个 rejected trial 的主要问题涉及 S9 右侧腹外斜肌 near-flatline。不得通过放宽 flatline 阈值或删除 S9 来批量转成 valid。应检查电极接触、传感器编号、原始波形、MVC 波形和贴片记录；未来 session 若重贴，必须新建 session 并重采 MVC。

### 8.3 MVC 幅值异常

P002 审计中有 40 个 channel-trial peak 超过 200% MVC，S2 的最大值约为 `10.33 × MVC`。超过 100% MVC 不必然是软件错误，但如此集中或极端的超限会影响通道间尺度、NMF W 和仿真比较。正式纳入前必须人工复核 MVC 动作是否充分、通道/贴片是否一致、是否有运动伪迹或 clipping；不能把归一化后的值悄悄截到 `[0,1]`。

### 8.4 受试者与统计外推

现有元数据不足以证明 P001 与 P002 是两个独立 biological subject；当前 V2 可用候选主要来自一个已知 session。因此不能报告 population confidence interval，也不能把单 session 结果外推到运动员群体。正式设计需使用稳定的 biological-subject registry UID、多个独立被试和真正 held-out session。

### 8.5 Observation mapping 仍是 provisional

15 通道 projection 是明确、可复现的候选 observation model，但还不是经专家确认的解剖真值。特别是 S5、S8、S9/S10 将一个表面通道聚合到多个模型 compartment/line，权重目前是占位的均匀权重。它们只能用于探索性验证，不能进入 IMR hard group，也不能被解释为神经驱动等价。

## 9. 隐私、版本控制与不可变原始数据

仓库只应提交代码、配置模板、伪造的 example metadata、测试和不含个人信息的统计摘要。禁止提交：

- `jidian_measurement/data/` 下的真实 raw/processed 数据；
- 真实 `session_metadata.*.json`、姓名、联系方式、精确出生日期或可识别视频路径；
- 视频、音频和硬件同步原件；
- 含真实 subject registry 映射的 selection manifest。

真实证据保存在受控数据存储中，仓库 artifact 只记录伪名 UID、受控引用和 SHA-256。不要重写 raw NPZ；预处理、事件补标和导入都生成新的、有来源哈希的 artifact。`session_metadata.example.json` 只能包含虚构值或 `null`。

## 10. 完成条件

只有同时满足以下条件，才能把 Phase 3 从“链路已实现”升级为“有正式结果”：

- 16 通道 profile、15 通道 projection 和 S1 排除经过人工/专家复核；
- 每个纳入 trial 有独立证据绑定的 impact 和完整 committed audit；
- strict importer 对整个预注册 selection 一次性通过；
- Stage3 identity v2 与 strict EMG NPZ 的设计、action、hand、split、comparison UID 和模型绑定完全相同；
- paired 报告有逐 trial shared reference fingerprint；否则只报告 unpaired cohort 指标；
- S9、MVC >200%、单被试/单 session 风险已解决或在结论中明确限制；
- 报告同时保留失败 trial、QC warnings、mapping 状态和 unavailable 指标，不进行事后删选。

截至本文状态日期，上述条件尚未全部满足。代码和数据契约已经为下一步补标、复核和正式评估做好准备，但 **没有启动训练，也没有生成可宣称完成 Phase 3 的 official EMG 结果**。

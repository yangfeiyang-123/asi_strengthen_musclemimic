# EMG 人工审查向导

这个向导用于完成 PEASD 正式训练前的两类人工证据：

1. 16 个采集通道到 MyoFullBody actuator 的 observation mapping 审查；
2. Clear、Lift、ChinaJump 每个真实 trial、15 个 comparable channel 和 S9 风险审查。

脚本只整理证据并记录你的决定。它不会替你判断解剖学、不会自动把 provisional 改成
verified，也不会启动训练。

## 1. 先生成审查包

从仓库根目录运行：

```bash
.venv/bin/python scripts/review_emg_for_training.py prepare \
  --output-dir artifacts/emg_human_review_v2
```

输出包括：

- `review_packet.json`：所有源文件路径、SHA-256、自动 QC 和问题清单；
- `review_answers.json`：可恢复的人工答案草稿；
- `review_report.md`：人类可读总览；
- `plots/session_s9_chronology.png`：全 session 63 个 processed trial 的 S9 时间趋势；
- `plots/<action>/<trial>.png`：每个正式 trial 的 16 通道未截断 `%MVC` 波形。

先阅读 `review_report.md` 并查看所有图，再开始作答。橙色 `1.0` 和红色 `2.0` 横线只表示
相对当前 MVC reference 的比例。超过这些线不是排除理由，也不允许裁剪波形。

当前真实数据的机器预览会显示全 session S9 有 28 个旧 QC critical record，后段/前段
filtered-RMS 比约为 `0.29`。这只是必须人工复核的证据，不是自动排除结论。

## 2. 开始或继续向导

```bash
.venv/bin/python scripts/review_emg_for_training.py wizard \
  --packet artifacts/emg_human_review_v2/review_packet.json
```

每答完一题立即写回 `review_answers.json`。输入 `q` 可安全退出；之后运行同一命令从未完成
处继续。源 NPZ、QC、mapping 或图片被修改后，旧 packet 会因哈希不一致而停止，必须在新目录
重新准备，不能把旧答案套到新数据上。

### 2.1 Mapping 题怎么选

Mapping 判断的是“该表面电极是否能作为这些仿真 actuator 的 measured-subspace
observation”，不是说电极精确测到了深层肌肉，也不是判断 MVC 是否足够大。

- `high`：贴片位置和一对一同源关系均已核对；
- `medium`：总体同源，但存在 compartment 聚合或可解释串扰；
- `low`：仍可用于有限的 measured-subspace observation，但不确定性明显；
- `defer`：当前知识或记录不足，请解剖/肌电人员确认；
- `reject`：当前映射不成立，需要新 mapping/profile。

S1 右斜方肌上束当前没有已核验的模型同源 actuator。确认排除只会把它排除在仿真比较之外，
不会删除原始 S1 数据。

如果你无法依据贴片记录、肌肉功能和模型 actuator 清单判断，必须选 `defer`，不能按脚本提示
猜一个可信度。`defer/reject` 会保留 provisional 状态。

### 2.2 Trial 题怎么选

每个 trial 都要看对应的 16 通道图、采集 metadata 和 preprocessing QC：

- `include`：波形与采集记录足以进入该动作的 training cohort；
- `exclude`：存在具体采集/信号问题，并写明原因；
- `defer`：需要查看原始记录或请实验员确认。

不能仅因为 `P99/max > 1×MVC` 选择 exclude。若旧 QC 或 metadata 已标记异常而你仍选择
include，向导会额外要求可追踪的复核证据。NaN/Inf 或负包络不能由人工覆盖。

### 2.3 Channel 与 S9 题怎么选

动作级 channel 审查要结合该动作的所有 trial。当前 ABI 固定为 15 个 comparable channels：

- 审查后可保留就选 `include_after_review`；
- 若某通道确实不能用，不得在这里静默删除，选择“需要新 mapping/profile”；
- 暂时无法判断就 defer。

S9 必须同时查看：

- 全 session chronology；
- 当前动作逐 trial P99；
- trial 内前/后窗口；
- 原始采集/MVC 记录与旧 QC flags。

只有确认“任务相关低激活而非贴片/采集失败”，或采取并记录了明确缓解措施后，才能选择
`accepted_after_review`/`mitigated`。证据字段要写真实图、记录或复核说明。

## 3. 最终确认与输出

所有问题完成后，向导要求逐字输入：

```text
我已逐项审查并对这些决定负责
```

随后写入 `artifacts/emg_human_review_v2/final/`：

- `reviewed_mapping.json`；
- `<action>/emg_trial_qc_review.json`，共三份；
- `review_validation.json`。

任何 pending、证据为空、源哈希变化、少于 3 个纳入 trial、通道静默删除或 S9 未裁决都会
阻止 finalization。可随时重新验证：

```bash
.venv/bin/python scripts/review_emg_for_training.py validate \
  --packet artifacts/emg_human_review_v2/review_packet.json \
  --review-dir artifacts/emg_human_review_v2/final
```

## 4. 审查通过后构建 verified v2 tubes

```bash
REVIEW_ROOT=artifacts/emg_human_review_v2
TUBE_ROOT="$REVIEW_ROOT/verified_tubes"

for action in forehand_high_clear forehand_lift_footwork china_jump_high_clear; do
  .venv/bin/python scripts/build_emg_reference_tube.py \
    --action "$action" \
    --mapping "$REVIEW_ROOT/final/reviewed_mapping.json" \
    --verified \
    --trial-qc-review "$REVIEW_ROOT/final/$action/emg_trial_qc_review.json" \
    --output-dir "$TUBE_ROOT"
done
```

这一步仍不会启动 PPO。三份 verified tube 重新加载并通过 Stage1 tube gate 后，才进入
T0–T4 × seeds 0/1/2 的 PPO preflight。

# 羽毛球 16 通道 Delsys Trigno 表面肌电采集与肌肉协同工具

本仓库现在提供一条从原始采集到 NMF 协同先验导出的完整、版本化数据链路。主格式为 NPZ/JSON/CSV 事件表；原始 EMG 永久保留，滤波与归一化只写入新文件。真实采集的默认对象是**右手持拍运动员**，左手持拍者不会被自动镜像。

> 目前仅实现并验证了软件声音/视觉提示，元数据固定写入 `sync_method: software_cue_only`。软件提示不等价于 TTL 硬件同步，不能据此声称 EMG、视频或击球接触达到硬件级同步精度。

## 0. 先看这里：本研究的实际执行流程

本节是现场采集和后续分析的主流程。后续章节用于解释参数、文件格式和异常处理。新受试者与历史 `P001/PILOT01` 必须走不同路径：

- 今后新受试者：只用 `badminton_synergy_16_v2`。
- 已有 `data/P001/PILOT01`：只用 `badminton_synergy_16_legacy_actual_v1`。
- 禁止采集和分析误声明的 `badminton_synergy_16_v1`。
- 不得把 V2 数据写进 `P001/PILOT01`，也不得把两套 profile 的数据共同做 NMF。

### 0.1 新受试者正式采集前

在 PowerShell 进入仓库并设置本次实验变量。下面以新受试者 `P002` 为例；不要复用已有的 `P001/PILOT01`：

```powershell
Set-Location 'D:\Main\Research\IEProgram\jidian_measurement'

$DataRoot = (Resolve-Path .\data).Path
$ParticipantId = 'P002'
$SessionId = 'S001'
$PilotSessionId = 'PILOT_V2'
$ChannelProfileId = 'badminton_synergy_16_v2'
$ProtocolId = 'badminton_primitive_protocol_v1'
$MetadataFile = ".\session_metadata.$ParticipantId.json"

Copy-Item -LiteralPath .\session_metadata.example.json -Destination $MetadataFile
notepad $MetadataFile
Get-Content -Raw $MetadataFile | ConvertFrom-Json | Out-Null
```

开始前逐项确认：

1. 受试者完成知情同意、健康/伤病筛查和统一热身；记录持拍手、优势腿、球拍与人体测量信息。
2. 按第 3 节 V2 表格贴好 16 个传感器，逐个核对“传感器编号—侧别—肌肉”，不能只看无线传感器位置。
3. 打开 Trigno Control Utility，确认 16 个传感器在线、实时波形可见并启用 TCP 服务。
4. 对每块目标肌肉做轻度抗阻，确认只有对应通道出现合理的宽带 EMG 响应；发现平线、固定正弦、脱落或明显串扰时先重新贴片。
5. 同一 session 内不得拆贴、交换或重新编号传感器。若贴片发生变化，应新建 session，并重新采 MVC。

运行不保存 trial 的短时检查：

```powershell
python -m emg.cli sensor-check `
  --profile $ChannelProfileId `
  --duration 3
```

### 0.2 先采 pilot，再开始正式 session

先在独立 pilot session 采 1 次安静站立，不要把调试 trial 混入正式 session：

```powershell
python -m emg.cli collect `
  --participant $ParticipantId `
  --session $PilotSessionId `
  --profile $ChannelProfileId `
  --protocol $ProtocolId `
  --action quiet_stance `
  --dataset-root $DataRoot `
  --handedness right --dominant-leg right `
  --session-metadata $MetadataFile `
  --trials 1 `
  --show-preview
```

只有在以下条件全部满足后才能开始正式 session：16 个子图均有信号；`received_samples == expected_samples`；`dropped_samples == 0`；传感器顺序为 1–16；没有平线、大面积削顶、明显固定 50 Hz 或错误肌肉响应。自动 `qc_pass` 不能替代人工检查。

### 0.3 正式采集顺序

本仓库推荐固定为：传感器检查 → MVC → 充分休息 → 安静站立 → 基础动作 → 影子挥拍 → 完整动作。若实验方案决定把 MVC 放在动作之后，必须对所有受试者保持相同顺序并在实验记录中注明。

先在正式 session、贴片不变的条件下采完整 MVC。每块肌肉需要 3 次有效重复，每次 4 s，重复间休息 60 s：

```powershell
python -m emg.cli mvc `
  --participant $ParticipantId --session $SessionId `
  --profile $ChannelProfileId `
  --protocol $ProtocolId `
  --dataset-root $DataRoot `
  --handedness right --dominant-leg right `
  --session-metadata $MetadataFile `
  --repetitions 3 `
  --contraction-duration 4 `
  --rest 60 `
  --window-ms 500
```

完成 MVC 后让受试者充分恢复，再按当前研究的固定动作顺序采集。命令会使用协议中各动作自己的有效 trial 目标和休息时间：

```powershell
python -m emg.cli collect-protocol `
  --participant $ParticipantId --session $SessionId `
  --profile $ChannelProfileId `
  --protocol $ProtocolId `
  --dataset-root $DataRoot `
  --handedness right --dominant-leg right `
  --session-metadata $MetadataFile `
  --actions `
    quiet_stance `
    forehand_lunge_and_return `
    single_leg_countermovement_jump_land `
    trunk_rotation `
    shadow_forehand_high_clear `
    china_jump_high_clear `
  --show-preview
```

每个 trial 的实际操作是：检查场地和传感器后按 Enter 开始；完成动作；查看 4×4 波形；关闭图窗；标记有效性和错误原因；进入休息。已经恢复且实验员确认安全时，可按 Enter 跳过剩余休息。无效 trial 会保留，程序继续到协议规定的有效数量。中断后使用完全相同的 participant、session、profile、protocol 和 action 重新运行即可续采。

### 0.4 正式 session 采完后的现场检查

不要拆传感器前才检查数据。每完成一个动作，至少确认 trial 数、人工有效性、丢样、预览图和传感器响应。全部采完后运行：

```powershell
$SessionPath = Join-Path $DataRoot "$ParticipantId\$SessionId"

python -m emg.cli qc `
  --session-path $SessionPath `
  --profile $ChannelProfileId

python -m emg.cli action-stats `
  --session-path $SessionPath `
  --profile $ChannelProfileId `
  --config .\config\semg_preprocessing.json
```

检查每个动作目录中的 `preview.png`、`qc.json`、`action_mean_variance.png` 和 `action_statistics.json`。不要删除无效 trial，也不要手工重编号。

### 0.5 新 V2 数据的离线处理与肌肉协同

完整 MVC 已采集时，正式预处理使用 MVC 归一化：

```powershell
python -m emg.cli preprocess `
  --session-path $SessionPath `
  --profile $ChannelProfileId `
  --config .\config\semg_preprocessing.json `
  --normalization mvc
```

逐 trial 检查 `processing.json`、`preprocessing_qc.json`、`preprocessing_comparison.png` 和 `normalized_envelope.png`。只有人工标记有效且 `preprocessing_qc.analysis_ready=true` 的 trial 才进入正式协同分析。

正式数据集默认不再把软件 `movement_cue` 当成真实动作开始。每个拟纳入 trial 必须先依据视频、音频或硬件证据补标唯一的 `movement_start_manual`。证据文件本身必须保留，并把其 SHA-256 绑定到审计记录；下面的 sample 仅为示例，必须替换为逐帧核对后的值：

```powershell
$TrialPath = Join-Path $SessionPath 'trials\forehand_high_clear\trial_001'
$EvidenceFile = 'D:\EMG_Evidence\P002_S001_camA.mp4'
$EvidenceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $EvidenceFile).Hash.ToLower()

python -m emg.cli annotate-event `
  --trial-path $TrialPath `
  --event-name movement_start_manual `
  --sample-index 3120 `
  --source manual_video `
  --confidence 0.9 `
  --annotator OPERATOR_01 `
  --evidence-reference 'P002_S001_camA.mp4#frame=1842' `
  --evidence-sha256 $EvidenceHash
```

同一命令也可补标预留的 `racket_contact`、`foot_contact`、`takeoff` 或 `landing`。`movement_cue`、`recording_start` 和 `recording_stop` 是不可改的采集事实，尤其不能把 `movement_cue` 改名或冒充触球。

构建 `V=[channels,time]` 数据集并提取 NMF 协同：

```powershell
$DatasetFile = Join-Path $DataRoot "analysis\${ParticipantId}_${SessionId}_v2.npz"
$SynergyDir = Join-Path $DataRoot "analysis\${ParticipantId}_${SessionId}_v2_synergy"

python -m emg.cli build-synergy-dataset `
  --dataset-root $DataRoot `
  --output $DatasetFile `
  --profile $ChannelProfileId `
  --protocol $ProtocolId `
  --participant $ParticipantId `
  --scope all `
  --only-valid

python -m emg.cli extract-synergy `
  --dataset $DatasetFile `
  --output $SynergyDir `
  --k-min 1 --k-max 8 `
  --n-init 50 `
  --seed 20260720
```

最终至少检查 `metrics.json` 中各 K 的 global/local VAF、选择的 K 和稳定性，以及 `synergy_weights.png`。跨被试比较 W 时必须先按 profile 保证通道含义完全一致，再用余弦相似度和 Hungarian 匹配处理协同排列不唯一的问题。

### 0.6 当前 `P001/PILOT01` 的后续处理

该 session 已完成非破坏性语义迁移，目前是 `badminton_synergy_16_legacy_actual_v1`；原始 NPZ 未改变，当前没有完整 MVC，也没有正式 processed 数据。它不能使用 V2，也不能继续使用误声明 V1。

先确认迁移后状态：

```powershell
$DataRoot = (Resolve-Path .\data).Path
$ProtocolId = 'badminton_primitive_protocol_v1'
$LegacySessionPath = (Resolve-Path .\data\P001\PILOT01).Path
$LegacyProfileId = 'badminton_synergy_16_legacy_actual_v1'

python -m emg.cli audit-session-profile `
  --session-path $LegacySessionPath `
  --actual-profile $LegacyProfileId

python -m emg.cli qc `
  --session-path $LegacySessionPath `
  --profile $LegacyProfileId

python -m emg.cli action-stats `
  --session-path $LegacySessionPath `
  --profile $LegacyProfileId `
  --config .\config\semg_preprocessing.json
```

在没有历史同配置 MVC 的情况下，只能把动态归一化明确标为探索性方案，不能写成 MVC 归一化：

```powershell
python -m emg.cli preprocess `
  --session-path $LegacySessionPath `
  --profile $LegacyProfileId `
  --config .\config\semg_preprocessing.json `
  --normalization dynamic_p95
```

预处理 QC 通过后，单独构建历史 profile 数据集：

```powershell
$LegacyDatasetFile = Join-Path $DataRoot 'analysis\P001_PILOT01_legacy_dynamic_p95.npz'
$LegacySynergyDir = Join-Path $DataRoot 'analysis\P001_PILOT01_legacy_dynamic_p95_synergy'

python -m emg.cli build-synergy-dataset `
  --dataset-root $DataRoot `
  --output $LegacyDatasetFile `
  --profile $LegacyProfileId `
  --protocol $ProtocolId `
  --participant P001 `
  --scope all `
  --crop-mode software_cue_exploratory `
  --only-valid

python -m emg.cli extract-synergy `
  --dataset $LegacyDatasetFile `
  --output $LegacySynergyDir `
  --k-min 1 --k-max 8 --n-init 50 --seed 20260720
```

这里显式使用 `software_cue_exploratory`，因为历史 trial 尚无可绑定证据的人工动作起点；生成的 dataset 元数据会标为 `exploratory=true`。它不能被描述为基于真实动作起点的正式裁剪。若后续取得并哈希绑定原始视频，应先用 `annotate-event` 补标，再改用默认的 `annotated_movement_events`。

如果 `--only-valid` 后没有可用 trial，应回到 `preprocessing_qc.json` 和原始波形查原因，不能为了让 NMF 运行而把坏数据统一改成有效。只有在实验人员能重新确认历史 16 个位置完全一致时，才允许用 `--allow-retrospective-profile-mvc` 补采历史配置 MVC；补采后应重新以 `--normalization mvc` 处理全部 PILOT01 trial。

## 1. 安装与快速检查

推荐 Python 3.10 以上。在仓库根目录执行：

```powershell
python -m pip install -r requirements.txt
python -m emg.cli list-profiles
python -m emg.cli list-actions
python -m pytest
```

采集、传感器检查和事件补标命令不会导入 NMF 的 scikit-learn 依赖；`extract-synergy` 才需要 scikit-learn。完整研究环境仍建议一次安装 `requirements.txt`，避免到分析阶段才发现依赖缺失。

无需硬件的最小链路：

```powershell
python -m emg.cli collect `
  --participant TEST001 --session DRYRUN `
  --profile badminton_synergy_16_v2 `
  --action forehand_high_clear `
  --dataset-root .\dataset_root --dry-run --trials 1
```

`--dry-run` 生成固定随机种子的 2000 Hz 合成 EMG，包括静止基线和各通道不同的动作激活。还可用 `--synthetic-50hz-mV`、`--synthetic-clipping-mV`、`--synthetic-dropped-samples` 注入异常。dry-run 默认不询问 trial 有效性并跳过休息；用 `--interactive-review` 可测试动作后标注流程。

## 2. Trigno Control Utility 与连接

1. 打开 Delsys Trigno Control Utility，确认 16 个传感器已经配对并显示实时信号。
2. 按 Control Utility 当前协议启用 TCP 数据服务。
3. 本仓库保留了旧代码已使用的协议：控制端 `127.0.0.1:50040`，EMG 数据端 `127.0.0.1:50041`，little-endian float32、每个采样点固定 16 个流通道。
4. 流值继续乘以 `1000.0` 转为 mV；该值现在由 `--stream-scale-to-mV` 显式控制。没有设备证据时不要修改。
5. 正式采集前运行：

```powershell
python -m emg.cli sensor-check --profile badminton_synergy_16_v2
```

采集异常、Socket 断开和 `KeyboardInterrupt` 路径都会尽量发送 `STOP` 并关闭控制/数据连接。trial 中保存实际接收、期望及缺失样本数；不完整 trial 仍保存，但自动设为无效。

## 3. 版本化通道配置

新受试者正式采集的默认配置 ID 为 `badminton_synergy_16_v2`。顺序就是 Trigno 传感器编号，不做运行时隐式重排：

| Sensor | Side | muscle_slug | 中文 | Abbr |
|---:|---|---|---|---|
| 1 | right | upper_trapezius | 右斜方肌上束 | UT |
| 2 | right | anterior_deltoid | 右三角肌前束 | AD |
| 3 | right | posterior_deltoid | 右三角肌后束 | PD |
| 4 | right | pectoralis_major_clavicular | 右胸大肌锁骨部 | PM |
| 5 | right | latissimus_dorsi | 右背阔肌 | LD |
| 6 | right | triceps_lateral | 右肱三头肌外侧头 | TRI |
| 7 | right | pronator_teres | 右旋前圆肌 | PT |
| 8 | right | extensor_carpi_radialis | 右桡侧腕伸肌 | ECR |
| 9 | right | external_oblique | 右腹外斜肌 | EO |
| 10 | left | external_oblique | 左腹外斜肌 | EO |
| 11 | right | vastus_lateralis | 右股外侧肌 | VL |
| 12 | left | vastus_lateralis | 左股外侧肌 | VL |
| 13 | right | biceps_femoris_long_head | 右股二头肌长头 | BF |
| 14 | left | biceps_femoris_long_head | 左股二头肌长头 | BF |
| 15 | right | gastrocnemius_medialis | 右腓肠肌内侧头 | GM |
| 16 | left | gastrocnemius_medialis | 左腓肠肌内侧头 | GM |

registry 中三套 16 通道语义不能混用：

- `badminton_synergy_16_v1`：保留原错误声明，状态 `deprecated_misdeclared`；禁止采集和分析，不得原地修改定义。
- `badminton_synergy_16_legacy_actual_v1`：P001/PILOT01 的已确认真实贴片含义，状态 `legacy_actual`；允许历史分析，禁止作为新受试者默认采集配置。补采同配置 MVC 必须显式传入 `--allow-retrospective-profile-mvc`。
- `badminton_synergy_16_v2`：状态 `active`，允许采集和分析；删除双侧臀大肌，新增右旋前圆肌和右桡侧腕伸肌，并保留双侧腹外斜肌、股外侧肌、股二头肌长头和腓肠肌内侧头。

历史 1～6 号传感器配置另保留为 `legacy_high_clear_6ch`。每个 session 和 trial 都保存完整配置快照，而不只是 ID。构建数据集会校验传感器 ID、顺序、侧别和肌肉 slug，配置不同或顺序变化会直接报错。

左手持拍受试者使用 `badminton_synergy_16_v2` 会终止；请先建立经过实验人员确认的显式左手版本，不能静默镜像右手配置。

### 3.1 PILOT01 历史配置审查与迁移

先运行只读审查：

```powershell
python -m emg.cli audit-session-profile `
  --session-path data\P001\PILOT01 `
  --actual-profile badminton_synergy_16_legacy_actual_v1
```

迁移命令不带 `--apply` 时也是 dry-run。确认 `Migration safe: True` 后执行：

```powershell
python -m emg.cli migrate-session-profile `
  --session-path data\P001\PILOT01 `
  --actual-profile badminton_synergy_16_legacy_actual_v1 `
  --apply
```

迁移会备份将修改的 session/trial metadata、兼容 CSV、旧预览、动作统计和孤立草稿，纠正 metadata 快照与 CSV 表头，然后按真实配置重建派生图和动作统计。`raw_emg.npz` 只读；manifest 保存迁移前后逐文件 SHA-256，并验证 CSV 首行以后的数值字节完全一致。`channel_profile_v2.json` 若无人引用，会保留并登记为 `unregistered_draft`，不能替代代码 registry 中的正式 V2。

## 4. 贴片与正式采集前要求

- 传感器编号必须与上表和实际贴片逐一核对。
- 电极长轴需要沿肌纤维方向贴附，避开肌腱和明显运动边缘。
- 清洁皮肤并固定导线/传感器；正式采集前用对应抗阻动作确认肌腹与信号响应。
- 对双侧同名肌肉必须核对 side，不能只看 muscle slug。
- EMG 幅值反映电活动，不能直接等价为肌肉力。

## 5. 动作协议

正式协议 ID 为 `badminton_primitive_protocol_v1`。每个 trial 都包含固定的 **1.0 s 静止基线 + 0.5 s 提示等待 + 动作时长 + 1.0 s 动作后保留**；“总记录”列包含这 2.5 s 的前后阶段。

| action ID | 中文 | 类型 | 动作时长 | 总记录 | 有效 trial 目标 | block | trial 间休息 | 器材/事件要求 |
|---|---|---|---:|---:|---:|---:|---:|---|
| **`quiet_stance`** | 安静站立 | primitive | 4 s | 6.5 s | 5 | 1 | 30 s | 无 |
| `split_step` | 垫步启动 | primitive | 3 s | 5.5 s | 5 | 1 | 30 s | 足部事件槽 |
| **`forehand_lunge_and_return` **| 正手跨步并回位 | primitive | 4 s | 6.5 s | 5 | 1 | 30 s | 足部事件槽 |
| **`single_leg_countermovement_jump_land`** | 单腿反向跳与落地 | primitive | 4 s | 6.5 s | 5 | 1 | 60 s | 起跳/落地/足部事件槽 |
| **`trunk_rotation`** | 躯干旋转 | primitive | 4 s | 6.5 s | 5 | 1 | 30 s | 无 |
| **`shadow_forehand_high_clear`** | 正手高远球影子挥拍 | primitive | 4 s | 6.5 s | 5 | 1 | 30 s | 球拍 |
| `shadow_forehand_lift` | 正手挑球影子挥拍 | primitive | 4 s | 6.5 s | 5 | 1 | 30 s | 球拍 |
| **`china_jump_shadow`** | 中国跳影子动作 | primitive | 5 s | 7.5 s | 5 | 1 | 75 s | 球拍、起跳/落地/足部事件槽 |
| `forehand_lift_footwork_shadow` | 正手挑球步伐影子动作 | primitive | 5 s | 7.5 s | 5 | 1 | 45 s | 球拍、足部事件槽 |
| `forehand_high_clear` | 正手高远球 | complete | 4 s | 6.5 s | 10 | 2 | 45 s | 球拍、球、击球/足部事件槽 |
| `china_jump_high_clear` | 中国跳高远球 | complete | 5 s | 7.5 s | 8 | 2 | 75 s | 球拍、球、击球/起跳/落地/足部事件槽 |
| `forehand_lift_footwork` | 正手挑球步伐 | complete | 5 s | 7.5 s | 10 | 2 | 45 s | 球拍、球、击球/足部事件槽 |

完整动作的 block 间默认休息 180 s。事件槽只保留待人工或视频标注的位置，不会从软件提示伪造击球、足触地、起跳或落地时间。

> 历史现场备注：第一次测量曾记录“十号有问题”。该备注只作为 `P001/PILOT01` 的待核查事项，不能用于推断通道映射，也不能自动外推到后续受试者；复核时应结合原始波形、`qc.json` 和贴片记录。

随时用以下命令核对代码中的当前协议值：

```powershell
python -m emg.cli list-actions --protocol badminton_primitive_protocol_v1
```

## 6. 正式采集流程

### 6.1 一次正式采集的推荐顺序

1. 在同一批贴片不拆除的条件下完成 Control Utility 检查、逐肌肉抗阻响应检查和短时 pilot。
2. 创建本次 session 的元数据文件，固定 participant、session、profile、持拍手和优势腿。
3. 采 1 个 `quiet_stance` pilot，检查 16 通道波形、工频、平线、丢样和通道顺序。
4. 按预先确定的实验顺序采集 primitive、完整动作和 MVC；顺序必须在所有受试者间保持一致。
5. 每个动作完成后立刻复核 trial 数量、`metadata.json` 和 `qc.json`，不要等到拆除传感器后才发现坏通道。
6. 保留原始文件，确认采集完成后再运行离线预处理。

同一台电脑上不要同时运行两个采集命令；它们会竞争相同的 Trigno TCP 端口。PowerShell 多行命令中的反引号 `` ` `` 必须是该行最后一个字符，后面不能有空格。

### 6.2 创建并检查 session 元数据

从模板复制一份受试者专用文件，不建议直接修改模板：

```powershell
Copy-Item -LiteralPath .\session_metadata.example.json `
  -Destination .\session_metadata.P002.json

notepad .\session_metadata.P002.json

# 只验证 JSON 语法，不开始采集
Get-Content -Raw .\session_metadata.P002.json | ConvertFrom-Json | Out-Null
```

模板中的值均为 `DEMO_*_NOT_REAL` 或 `null`，不能原样当作真实实验信息。至少核对年龄、性别、身高、体重、训练年限、竞赛水平、近期伤病、球拍参数、操作员和贴片备注。V2 还可固定以下溯源信息：内部伪名 `biological_subject_uid` 与知情同意方案、采集 site、设备型号、Control Utility 版本、固件、逐 sensor 序列号、贴片协议、二维/三维电极坐标、方向、极间距及视频引用。不要在这些字段写姓名、身份证号或其他直接标识符；视频引用也应使用受控存储中的相对或逻辑 ID。

`--participant`、`--session`、`--profile`、`--protocol`、`--handedness`、`--dominant-leg` 和 `schema_version` 由程序写入。元数据文件若重复提供这些受保护字段，必须与命令行/canonical 值完全一致，否则立即报错；`collection_date` 一律不接受外部提供。首次创建后，除自动生成的本次调用时间外，所有标准 session 字段在恢复时都必须与 `session.json` 精确一致。旧 session 没有 V2 `schema_version` 时会发出 legacy warning；仅允许把旧记录中缺失且本次仍为 `null` 的新增可选溯源字段视为等价。若要给旧 session 新增非空 subject/device/placement/video 绑定，必须新建 session，不能事后静默改写历史快照。

participant/session ID 只能使用 ASCII 字母、数字、点、横线和下划线，例如 `P001`、`S20260721_A`。首次运行后 `session.json` 即成为该 session 的固定快照；需要改变受试者、持拍手、人体/设备/贴片/视频溯源或通道配置时应创建新 session，不要复用原目录。

为减少长命令中的路径错误，可以先在 PowerShell 设置本次采集变量：

```powershell
$DataRoot = 'D:\Main\Research\IEProgram\jidian_measurement\data'
$ParticipantId = 'P002'
$SessionId = 'S20260721_A'
$MetadataFile = '.\session_metadata.P002.json'
```

### 6.3 连接 Trigno 并执行传感器检查

先在 Delsys Trigno Control Utility 中确认 16 个传感器在线、编号正确、实时曲线可见并已启用 TCP 数据服务，然后执行：

```powershell
python -m emg.cli sensor-check `
  --profile badminton_synergy_16_v2 `
  --duration 3 `
  --host 127.0.0.1 `
  --command-port 50040 `
  --emg-port 50041 `
  --fs 2000 `
  --stream-scale-to-mV 1000
```

默认连接参数就是上面的值，因此通常可以省略 host/端口/fs/scale。`sensor-check` 会录制短时数据，并在全零、平线或 NaN/Inf 通道出现时返回失败；它只打印结果，不创建正式 trial。

必须另外在 Control Utility 中逐个做对应肌肉的轻度抗阻动作，确认目标通道随收缩出现不规则宽带 EMG。**固定正弦波、多个通道高度同步、明显 50 Hz 或近乎恒定负偏置，即使通过了 `sensor-check` 也不能开始正式采集。** 此时应检查贴片接触、传感器编号、电源/充电器和周围交流设备。

### 6.4 先采 1 个 pilot trial

建议 pilot 使用独立 session，避免把调试 trial 混入正式 session：

```powershell
python -m emg.cli collect `
  --participant P002 `
  --session PILOT_V2 `
  --profile badminton_synergy_16_v2 `
  --protocol badminton_primitive_protocol_v1 `
  --action quiet_stance `
  --dataset-root D:\Main\Research\IEProgram\jidian_measurement\data `
  --handedness right `
  --dominant-leg right `
  --session-metadata .\session_metadata.P002.json `
  --trials 1 `
  --show-preview
```

pilot 结束后至少确认：16 个通道都存在、无丢样、无平线、无大面积削顶、无固定 50 Hz 正弦、贴片通道与肌肉表一致。`--trials 1` 表示该 action 在该 session 中的**累计有效目标为 1**，不是“在已有数量上再加 1”。

### 6.5 采集一个正式动作

确认 pilot 正常后，在正式 session 中运行。下面命令不传 `--trials`，因此使用协议内置目标数量：

```powershell
python -m emg.cli collect `
  --participant P002 `
  --session S001 `
  --profile badminton_synergy_16_v2 `
  --protocol badminton_primitive_protocol_v1 `
  --action quiet_stance `
  --dataset-root D:\Main\Research\IEProgram\jidian_measurement\data `
  --handedness right `
  --dominant-leg right `
  --session-metadata .\session_metadata.P002.json `
  --show-preview
```

把 `--action quiet_stance` 替换为第 5 节表中的任一 action ID 即可采其他动作。例如采正手高远球：

```powershell
python -m emg.cli collect `
  --participant $ParticipantId --session $SessionId `
  --profile badminton_synergy_16_v2 `
  --protocol badminton_primitive_protocol_v1 `
  --action forehand_high_clear `
  --dataset-root $DataRoot `
  --handedness right --dominant-leg right `
  --session-metadata $MetadataFile `
  --show-preview
```

### 6.6 每个 trial 中程序会做什么

1. 每次 `collect` 命令启动时先显示通道表并执行一次短时传感器检查。
2. 显示动作中文名与口令；操作者检查场地、球和传感器后按 Enter。
3. 自动记录 1.0 s 静止基线；受试者此时保持起始姿势，不提前动作。
4. 再等待 0.5 s，在约 1.5 s 处发出软件声音/视觉动作提示。
5. 记录协议规定的动作阶段和 1.0 s 动作后阶段。
6. 先原子保存原始数据、事件、初始元数据和 QC，再显示 4×4 的 16 通道波形总览。
7. 关闭图窗后回答“该 trial 是否有效”。有效 trial 记录 `correct`；无效 trial 输入一个或多个英文错误标签，多个标签用英文逗号分隔，并必须填写非空原因。非交互模式下 `invalid/other` 同样不能省略 `--operator-notes`。
8. 无效 trial 不删除、不覆盖；程序继续采集，直到达到有效 trial 目标或达到尝试次数上限。
9. 进入休息倒计时；确实已准备好提前继续时可按 Enter 跳过剩余休息。新 trial 的 `post_trial_rest` 会记录 trial/block 类型、计划秒数、实际秒数和是否跳过；达到目标、dry-run 或协议零休息也会显式记录原因。

`QC PASS` 只表示没有触发当前自动规则，不能代替人工复核。反过来，安静站立可能因动作/基线比接近 1 而显示 QC warning；应结合具体 warning、波形和实验记录判断。丢样、中断或接收错误会自动将 trial 设为无效。

旧 trial 或在休息倒计时完成前异常终止的新 trial 若没有 `post_trial_rest`，只能解释为“休息信息未知”，不能反推为 0 秒、完整休息或操作者跳过。

错误标签：`correct`、`missed_shuttle`、`wrong_footwork`、`wrong_takeoff`、`wrong_landing`、`incomplete_motion`、`sensor_motion_artifact`、`sensor_detached`、`signal_clipping`、`sync_failed`、`other`。

### 6.7 覆盖 trial 数、动作时长和休息时间

只在实验方案明确要求时覆盖协议值：

```powershell
python -m emg.cli collect `
  --participant $ParticipantId --session $SessionId `
  --profile badminton_synergy_16_v2 `
  --protocol badminton_primitive_protocol_v1 `
  --action forehand_high_clear `
  --dataset-root $DataRoot `
  --handedness right --dominant-leg right `
  --session-metadata $MetadataFile `
  --trials 10 `
  --duration 4 `
  --rest 45 `
  --block-rest 180 `
  --show-preview
```

- `--trials`：该 session/action 的累计有效 trial 目标，不是本次新增数量。
- `--duration`：只覆盖动作阶段；1.0 s 基线、0.5 s 提示等待和 1.0 s 动作后阶段仍会保留。
- `--rest`：trial 间休息秒数。
- `--block-rest`：仅 `collect` 支持，覆盖多 block 动作的 block 间休息秒数。
- `--no-legacy-csv`：不生成体积较大的兼容 CSV；原始 NPZ、JSON、事件和预览仍保存。
- `--show-preview` / `--no-show-preview`：显式开启或关闭 trial 后波形窗。正式人工采集建议开启。

覆盖值会影响本次运行行为，但 protocol 快照仍表示原始版本，因此正式研究中必须在实验记录中注明所有覆盖参数，并对同一分析批次保持一致。

### 6.8 一次采集多个动作

推荐显式列出动作，避免误采整个协议：

```powershell
python -m emg.cli collect-protocol `
  --participant $ParticipantId `
  --session $SessionId `
  --profile badminton_synergy_16_v2 `
  --protocol badminton_primitive_protocol_v1 `
  --dataset-root $DataRoot `
  --handedness right `
  --dominant-leg right `
  --session-metadata $MetadataFile `
  --actions split_step trunk_rotation shadow_forehand_high_clear `
  --show-preview
```

不传 `--actions` 会按协议定义顺序采集全部动作。`--trials-per-action` 会把每个所选动作的累计有效目标改成同一个值；`--duration` 和 `--rest` 也会统一覆盖所有所选动作，异质动作通常不建议这样做。`collect-protocol` 会为每个动作重新执行传感器检查，并沿用相同的 participant/session/profile。

### 6.9 中断、恢复和补采

按 `Ctrl+C` 时程序会尽量发送 `STOP`、关闭 Socket，并保留已经接收的完整帧；中断 trial 标为无效。之后使用**完全相同**的 participant、session、profile、protocol 和 action 重新运行原命令即可恢复：

```powershell
python -m emg.cli collect `
  --participant $ParticipantId --session $SessionId `
  --profile badminton_synergy_16_v2 `
  --protocol badminton_primitive_protocol_v1 `
  --action quiet_stance `
  --dataset-root $DataRoot `
  --handedness right --dominant-leg right `
  --session-metadata $MetadataFile `
  --show-preview
```

程序从当前最大 `trial_###` 继续编号，并按 `metadata.json` 重新统计有效数量。已达到目标时会显示 `nothing to collect`，不会重复采集。不要删除无效 trial 后重编号，也不要用更小的 `--trials` 期待“追加若干次”。若要补到累计 7 个有效 trial，应显式传 `--trials 7`。

### 6.10 非交互和 dry-run

真实人体采集优先使用默认交互模式。只有外部流程已经承担 trial 判定时才使用：

```powershell
python -m emg.cli collect `
  --participant $ParticipantId --session $SessionId `
  --profile badminton_synergy_16_v2 `
  --action quiet_stance `
  --dataset-root $DataRoot `
  --handedness right `
  --non-interactive `
  --error-label correct `
  --operator-notes 'Externally supervised trial' `
  --no-show-preview
```

`--non-interactive` 会把每个完整 trial 直接按 `--error-label` 判定，不会询问操作者；不要为了省事在无人监督的正式采集中统一标成 `correct`。

无需硬件测试采集链路：

```powershell
python -m emg.cli collect `
  --participant TEST001 --session DRYRUN `
  --profile badminton_synergy_16_v2 `
  --action forehand_high_clear `
  --dataset-root .\dataset_root `
  --dry-run --trials 1 --interactive-review --show-preview
```

### 6.11 采集完成后的立即检查

先列出 trial，再重新计算一次原始信号 QC：

```powershell
$SessionPath = Join-Path $DataRoot "$ParticipantId\$SessionId"

Get-ChildItem -Path (Join-Path $SessionPath 'trials') -Directory -Recurse |
  Where-Object { $_.Name -like 'trial_*' }

python -m emg.cli qc `
  --session-path $SessionPath `
  --profile badminton_synergy_16_v2
```

然后逐动作抽查 `preview.png`、`metadata.json` 和 `qc.json`。至少确认 `received_samples == expected_samples`、`dropped_samples == 0`、通道 1～16 顺序正确，并处理工频、平线、脱落和削顶告警。采集 QC 未通过的数据可以保留作故障证据，但不得仅靠后续滤波改标为有效。

### 6.12 多次 trial 的动作统计图

当一个动作达到目标有效 trial 数后，采集程序会自动汇总该动作的全部 `valid_for_analysis=true` trial。每次先使用与离线流程一致的去异常点、去均值、30–300 Hz 带通、50 Hz 陷波、全波整流和 4 Hz 包络，再逐采样点计算跨 trial 均值与样本方差（`ddof=1`）。原始双极 EMG 不直接求均值，避免正负相位抵消。

每个动作目录生成：

- `action_mean_variance.png`：4×4 的 16 传感器图；实线为包络均值，阴影为均值 ±1 标准差，每个子图标注动作阶段的平均逐点方差。
- `action_statistics.npz`：逐点 `mean_envelope_mV`、`variance_envelope_mV2`、`std_envelope_mV`、时间轴、N 值和纳入的 trial ID。
- `action_statistics.json`：统计口径、预处理参数、纳入/排除 trial、动作时间和逐通道摘要。

对已有 session 重新生成全部动作统计：

```powershell
python -m emg.cli action-stats `
  --session-path $SessionPath `
  --profile badminton_synergy_16_v2 `
  --config .\config\semg_preprocessing.json
```

可重复传 `--action action_id` 只处理指定动作。默认排除人工判定无效的 trial；只有用于故障比较时才显式传 `--include-invalid`。统计图按软件 cue 对齐，如果真实动作起始、触球、起跳或落地没有采样点标注，阴影会同时包含动作时序偏差和生理幅值差异，不能解释成纯生理变异。

## 7. MVC

MVC 必须在传感器位置不变、通道编号确认无误且受试者完成热身后采集。MVC 放在动态动作之前还是之后应由实验设计统一规定，并安排充分休息；代码不替代实验员判断疲劳、疼痛或代偿动作。

正式默认每块肌肉 3 次有效重复、每次有效收缩 4 s、重复间休息 60 s。采集现场计算 500 ms 包络滑动均值的最大稳定窗口，供操作者判断本次重复是否有效；正式离线归一化会从保留的原始 MVC 时序按当前预处理配置重新计算 4 Hz 平滑包络峰值，不取原始 EMG 的单个采样尖峰：

```powershell
python -m emg.cli mvc `
  --participant $ParticipantId `
  --session $SessionId `
  --profile badminton_synergy_16_v2 `
  --protocol badminton_primitive_protocol_v1 `
  --dataset-root $DataRoot `
  --handedness right `
  --dominant-leg right `
  --session-metadata $MetadataFile `
  --repetitions 3 `
  --contraction-duration 4 `
  --rest 60 `
  --window-ms 500
```

程序按 profile 顺序逐块肌肉显示传感器编号、肌肉名称和 MVC 口令。每次按 Enter 后开始记录，结束时询问该重复是否有效并允许填写备注；主动判为无效时必须填写非空原因。无效重复保留，但不计入 3 次有效目标。每次重复均保存原始时序、4 Hz 显示包络、有效性、稳定窗口值、硬 QC 摘要和 `post_repetition_rest`。

MVC 的最小硬门不可由人工“有效”覆盖：目标 sensor 流必须完整且无 dropped/interruption/receive error，raw 必须非空、shape 正确、全部 finite、非全零且非 flatline。失败原因写入 rep `metadata.json.qc.hard_failures` 并自动作废。这里不以普通幅值高低或动作/基线比自动判有效，动作姿势、代偿、疼痛与用力程度仍由实验员判断。每次 MVC 后的计划/实际/跳过休息也会记录；历史 rep 缺少该字段表示未知，而不是未休息。

只采或补采指定肌肉时重复使用 `--muscle`：

```powershell
python -m emg.cli mvc `
  --participant $ParticipantId --session $SessionId `
  --profile badminton_synergy_16_v2 `
  --dataset-root $DataRoot `
  --handedness right `
  --muscle right_upper_trapezius `
  --muscle left_external_oblique
```

可使用纯 `muscle_slug`，例如 `--muscle upper_trapezius`；双侧同名肌肉必须使用 `right_external_oblique`、`left_external_oblique` 这种 `side_muscle_slug`，防止左右混淆。再次运行会从已有 `rep_###` 继续。

MVC 采集汇总保留现场稳定窗口统计；`preprocess` 从同一 participant 的有效原始 MVC 重复重新计算逐肌肉包络峰值并记录来源。缺少任何通道的有效 MVC 时，正式 MVC 归一化会停止，不会静默换用动态归一化。

PILOT01 若要补采与历史贴片语义一致的 MVC，必须确认仍按历史真实配置贴片，并显式授权；不能用正式 V2 的前臂通道含义替代：

```powershell
python -m emg.cli mvc `
  --participant P001 --session PILOT01 `
  --profile badminton_synergy_16_legacy_actual_v1 `
  --protocol badminton_primitive_protocol_v1 `
  --dataset-root .\data `
  --handedness right --dominant-leg right `
  --allow-retrospective-profile-mvc
```

## 8. 数据目录和 schema

```text
dataset_root/
  P001/S001/
    session.json
    channel_profile.json
    protocol.json
    mvc/
      mvc_results.json
      right_upper_trapezius/rep_001/
        mvc_timeseries.npz
        metadata.json
    trials/forehand_high_clear/trial_001/
      raw_emg.npz
      metadata.json
      events.csv
      events.annotation.audit.jsonl  # 仅在补标事件后生成，append-only
      qc.json
      preview.png
      legacy_raw_emg.csv       # 可用 --no-legacy-csv 关闭
      filtered_emg.npz         # 去偏置、带通和陷波
      rectified_emg.npz        # 全波整流
      envelope_emg.npz         # 4 Hz 包络
      mvc_normalized_emg.npz   # MVC 归一化包络
      processed_emg.npz        # preprocess 后生成
      preprocessing_qc.json
      preprocessing_comparison.png
      normalized_envelope.png
      processing.json
    trials/forehand_high_clear/
      action_mean_variance.png  # 多次有效 trial 的 4×4 均值/方差图
      action_statistics.npz     # 逐点均值、样本方差和标准差
      action_statistics.json    # 统计口径、纳入 trial 与通道摘要
```

`raw_emg.npz`：`emg_mV [T,C]`、`time_s`、`sample_index`、`stream_channel_ids`、`fs_hz`。trial `metadata.json` 含 participant/session/trial/action、协议与通道配置快照、持拍手、优势腿、采样计数、有效性、错误、器材、同步、休息记录、软件版本和 Git commit（仓库不是 Git worktree 时为 `null`）。MVC rep 的 metadata 另含不可覆盖的硬 QC 和重复后休息。缺失历史休息字段一律表示 unknown。文件通过同目录临时文件加 `os.replace` 原子提交。

`events.csv` 的固定列为 `event_name,sample_index,emg_time_s,monotonic_time_ns,wall_clock_iso,source,confidence,notes`。没有可靠注释的动作起点、击球、足部接触、起跳和落地事件保留为空样本的 annotation slot，`source=unannotated`、`confidence=0`，绝不伪造硬件时间。

`annotate-event` 只允许填写预声明的 `movement_start_manual`、`racket_contact`、`foot_contact`、`takeoff` 和 `landing`，要求 sample/time 在半个采样周期内一致、证据来源在白名单、confidence 位于 `(0,1]`、伪名 annotator、evidence reference 与证据文件 64-hex SHA-256。SHA 不写入固定 CSV schema，而写入 `events.annotation.audit.jsonl`。每次修改先 fsync 一条 `transaction_state=prepared`，原子替换并复核整份 `events.csv` 的 `after_sha256` 后再 fsync `committed`；两条记录还保留 before/after event、整表 before/after hash 和可复算的 `annotation_manifest_sha256`。默认禁止覆盖已有补标；确需纠正时先核对历史和 `--expected-before-sha256`，再显式传 `--overwrite`。

## 9. QC

每个 trial 自动检查：全零、flatline、NaN/Inf、样本数/短流、可能削顶、极端尖峰、基线 RMS、动作 RMS、动作/基线比、50 Hz 功率占比、低频晃动伪迹和异常高通道相关。阈值位于 `emg.qc.QCThresholds`，并完整写入 `qc.json`。自动 QC 只给出 warning、`qc_pass` 和人工复核建议，不自动删除或覆盖数据。

动作 trial 的近乎平线 warning 不能单凭一个 trial 就永久删除该通道，也不能在这里静默改写 QC 口径。应同时复核该通道在整个 session 的原始波形、传感器/贴片记录，以及同一肌肉的 MVC raw/QC：若 session 与 MVC 都持续近乎平线，应把它作为采集/贴片失败处理并说明分析范围；若仅特定动作低激活，则保留通道并报告这一生理/任务差异。当前历史数据不会因新增说明被自动改标。

重新计算：

```powershell
python -m emg.cli qc --session-path D:\EMG_Dataset\P001\S001 --profile badminton_synergy_16_v2
```

## 10. 离线预处理与归一化

默认正式参数位于 `config/semg_preprocessing.json`：逐通道修复可插值缺失点和孤立尖峰、减去通道均值、四阶 Butterworth 30–300 Hz 带通、中心 50 Hz/带宽 5 Hz 陷波、全波整流、四阶 Butterworth 4 Hz 低通包络，以及逐被试逐肌肉 MVC 包络峰值归一化。离线默认使用前后向零相位滤波；长度和时间戳不变，首尾各 0.25 s 标为滤波边界 guard，不参与 QC 核心统计。

```powershell
python -m emg.cli preprocess `
  --session-path D:\EMG_Dataset\P001\S001 `
  --profile badminton_synergy_16_v2 `
  --config .\config\semg_preprocessing.json
```

MVC 默认在同一 participant 的各 session 中查找原始 `mvc_timeseries.npz`，用与动作 trial 完全一致的滤波参数重新计算每块肌肉的 4 Hz 包络峰值，并取有效重复中的最大值。不会把旧版不同滤波参数的汇总值静默混入。MVC 缺失或不完整时命令报错；只有显式传入 `--fallback-normalization dynamic_p95`（或 `none`）才继续，并在 `processing.json` 记录 fallback。`dynamic_p95` 参考值由同一 session 的全部 trial 合并计算。

批量处理多个被试和 session：

```powershell
python .\scripts\preprocess_semg.py `
  --dataset-root D:\EMG_Dataset `
  --config .\config\semg_preprocessing.json `
  --profile badminton_synergy_16_v2
```

可重复使用 `--participant P001` 和 `--session S001` 筛选。每个 trial 分别写出滤波、整流、包络和 MVC 归一化 NPZ，同时继续生成兼容 NMF 构建器的 `processed_emg.npz`。`preprocessing_qc.json` 区分“处理成功”和“可安全分析”；纯 50 Hz、近乎平线或大段缺失即使完成滤波也不会被标为可分析。原始 `raw_emg.npz`、采集元数据和事件文件不会被覆盖。完整字段与 QC 说明见 `docs/semg_preprocessing.md`。

## 11. 协同数据集与 NMF

构建 NMF 输入 `V=[channels,time]`。默认 `--crop-mode annotated_movement_events` 严格要求唯一、已用证据补标且 `source` 非 software/unannotated 的 `movement_start_manual`，还会验证其 latest committed 审计、matching prepared、manifest hash、证据 SHA、当前事件行和当前整份 `events.csv` hash，再裁到唯一 `recording_stop`；事件缺失、重复、手工绕过审计、未补标或边界非法都会停止，绝不 fallback 到 full trial。`--only-valid` 同时要求人工 `valid_for_analysis=true` 和必须存在的 `preprocessing_qc.analysis_ready=true`，防止把未做 preprocessing QC 或处理成功但信号不可用的 trial 纳入：

```powershell
python -m emg.cli build-synergy-dataset `
  --dataset-root D:\EMG_Dataset `
  --output D:\EMG_Dataset\analysis\primitive_v1.npz `
  --profile badminton_synergy_16_v2 `
  --protocol badminton_primitive_protocol_v1 `
  --scope primitive --only-valid
```

可用 `--action`、`--participant` 重复筛选；`--crop-mode full_trial` 仅在明确需要完整记录窗时使用；`--crop-mode software_cue_exploratory` 才允许从软件 cue 裁剪，并在 dataset 和逐 trial 元数据明确写入 exploratory，仍不允许事件缺失时 fallback。此模式不能用于 impact 对齐或基于真实动作时刻的正式结论。`--time-normalize-samples 101` 将各段归一化到 0–100%。通道配置或预处理参数不一致会报错。

处理兼容性按方法与 scope 校验，而不是错误地要求所有被试共享同一 MVC 数值。MVC denominator 必须在同一 `participant:*` reference 内逐通道完全一致，不同 participant 可有各自有限正值；dynamic-p95 则按 `session:participant/session` 隔离。每个 denominator、scope、participant/session 和 MVC manifest 都保留在 dataset JSON 的 `normalization_references` 与 trial provenance 中。

提取协同：

```powershell
python -m emg.cli extract-synergy `
  --dataset D:\EMG_Dataset\analysis\primitive_v1.npz `
  --output D:\EMG_Dataset\analysis\synergy_artifact_v1 `
  --k-min 1 --k-max 8 --n-init 50 --seed 20260720
```

定义 `V ≈ W @ H`，`W=[16,K]`、`H=[K,T]`。默认选择 global VAF ≥ 0.90 且至少 80% 肌肉 local VAF ≥ 0.75 的最小 K，保存各 K 重建误差、global/local VAF、split-half 和跨 trial 稳定性。W 列做 L2 单位范数并对 H 逆缩放。NMF 协同排列不唯一，比较时使用余弦相似度加 Hungarian 最佳匹配。

artifact 包含 `synergy_basis.npz`（W、可选 H、通道、side、slug、K、fs）、`synergy_metadata.json`、`metrics.json` 和图。W 仅处于实验表面 EMG 通道空间；本仓库不猜测它到肌骨模型全部肌肉的一对一映射，后续仓库必须显式定义映射矩阵。

## 12. 旧 6 通道数据

`delysis_measure.py` 和 `delysis_measure_6ch.py` 保留为 deprecated wrapper，导入时不创建目录、不生成时间戳、不连接硬件。新采集请使用：

```powershell
python -m emg.cli collect `
  --participant P001 --session LEGACY01 `
  --profile legacy_high_clear_6ch --action forehand_high_clear
```

历史 CSV 仍可由 `visualize.py` 的 `read_trial_csv`/`visualize_folder` 读取和绘图；新兼容 CSV 的列名包含 sensor、side 和 muscle slug。旧 CSV 没有完整 profile 快照，不能直接与 16 通道数据共同做 NMF；必须先经过人工确认的显式迁移，绝不能只按列数猜配置。

## 13. 常见错误

- `Connection refused`：确认 Control Utility 数据服务开启、host/端口一致，并先跑 `sensor-check`。
- `sample_count_mismatch_or_short_stream`：保留该 trial，检查无线连接、CPU/网络、Control Utility 状态；不要手工补零后标成有效。
- `No complete compatible raw MVC set...`：先完成同一 participant/profile 的全部逐肌肉 MVC，或显式选择并记录 fallback。
- `different or reordered channel profile`：数据来自不同配置/顺序；不要绕过校验，回到原始文件做显式转换。
- `Preprocessing parameters differ`：用同一参数重新处理所有拟合 trial。
- 左手受试者被拒绝：这是防止静默镜像的保护；先建立并审核左手版本配置。
- QC warning：自动规则不等于剔除。结合贴片记录、波形、视频和实验员备注人工决定。

## 14. 当前硬件假设边界

代码级测试覆盖帧顺序、精确读取、STOP/关闭、配置校验、原子保存、MVC 稳定窗口、滤波、profile 隔离、dry-run、NMF 和旧入口。但本仓库当前没有记录到真实 Trigno 连接测试、传感器满量程、丢包指示机制、TTL 接口、视频/音频输入延迟或各肌肉 MVC 姿势的人体实验确认。正式使用前必须由实验员在目标机器和目标 Control Utility 版本上完成小规模 pilot。

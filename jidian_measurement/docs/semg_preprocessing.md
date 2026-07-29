# sEMG 离线预处理

本流程面向仓库现有的 Delsys Trigno `raw_emg.npz + metadata.json + events.csv` 数据。采集模块和原始文件保持不变，所有处理结果均以新文件写入 trial 目录。

## 1. 默认处理顺序

每个肌肉通道独立执行：

1. 检查二维形状、通道顺序、采样率、时间轴和最短信号时长。
2. 对 NaN/Inf 按原采样网格逐通道线性插值；长缺失段会被记录为 QC failure。
3. 用 MAD 检测极端幅值，仅插值不超过配置长度的孤立尖峰；持续高幅值保留并告警，避免误删真实肌肉激活。
4. 减去每个通道自己的均值，去除直流偏置。
5. 四阶 Butterworth 30–300 Hz 带通。
6. 中心频率 50 Hz、带宽 5 Hz 的 IIR notch；默认 Q = 50 / 5 = 10。
7. 全波整流。
8. 四阶 Butterworth 4 Hz 低通，得到非负平滑包络。
9. 使用同一被试、同一 side+muscle 的原始 MVC 重复重新计算 4 Hz 包络峰值，再执行 `envelope / MVC_peak`。

默认使用 `sosfiltfilt/filtfilt` 前后向零相位滤波。这样不会引入系统性相位延迟，但首尾仍可能有反射填充造成的边界效应。因此输出不裁剪、时间戳不改变，同时在 QC 和图中将默认首尾 0.25 s 标为 edge guard。

## 2. 适用边界

50 Hz 陷波适合去除叠加在有效 EMG 上的有限工频污染，不能从近乎纯 50 Hz 正弦或近乎平线信号中恢复不存在的生理信息。当 `notch_removed_variance_fraction` 超过配置阈值时，处理可以完成，但该通道的 `analysis_ready` 为 `false`。

安静站立的动作/基线 RMS 比接近 1 是合理的，不能单独作为坏通道依据。预处理 QC 重点检查缺失段、平线、滤波后有效幅度、工频占比和陷波去除的方差比例。

## 3. 配置

默认配置为 `config/semg_preprocessing.json`。主要字段：

- `sample_rate_hz`: 2000；必须与原始文件一致。
- `bandpass_low_hz`, `bandpass_high_hz`, `filter_order`: 30、300、4。
- `notch_hz`, `notch_bandwidth_hz`: 50、5；设 `notch_hz` 为 `null` 可禁用。
- `envelope_lowpass_hz`: 4。
- `zero_phase`: 离线正式处理保持 `true`。
- `minimum_duration_s`: 过短信号直接拒绝并写失败日志。
- `edge_guard_s`: QC 和图中不作为核心判定区间的首尾时长。
- `max_missing_fraction`, `max_interpolation_gap_s`: 缺失值风险阈值。
- `outlier_mad_threshold`, `max_outlier_run_samples`: 孤立尖峰检测与替换条件。
- `normalization`: 默认 `mvc`。

未知配置字段会报错，防止拼写错误被静默忽略。

## 4. 命令

单 session：

```powershell
python -m emg.cli preprocess `
  --session-path D:\EMG_Dataset\P001\S001 `
  --profile badminton_synergy_16_v2 `
  --config .\config\semg_preprocessing.json
```

上例适用于新采集的 active V2 session。已迁移的 `data/P001/PILOT01` 必须改用
`--profile badminton_synergy_16_legacy_actual_v1`；误声明的
`badminton_synergy_16_v1` 已禁止分析，不能用来绕过快照校验。

批量：

```powershell
python .\scripts\preprocess_semg.py `
  --dataset-root D:\EMG_Dataset `
  --profile badminton_synergy_16_v2 `
  --config .\config\semg_preprocessing.json
```

批量筛选示例：

```powershell
python -m emg.cli preprocess-dataset `
  --dataset-root D:\EMG_Dataset `
  --participant P001 --participant P002 `
  --session S001 `
  --profile badminton_synergy_16_v2
```

默认缺 MVC 即停止。仅做诊断或明确接受非 MVC 归一化时，必须显式使用：

```powershell
--fallback-normalization none
# 或
--fallback-normalization dynamic_p95
```

这些 fallback 会写入每个 trial 的 `processing.json`，不会伪装成 MVC。

## 5. 输出

原始 `raw_emg.npz` 不改动。每个 trial 新增：

- `filtered_emg.npz`: 修复后、去均值、带通、陷波和最终滤波信号。
- `rectified_emg.npz`: 全波整流信号。
- `envelope_emg.npz`: 4 Hz 包络。
- `mvc_normalized_emg.npz`: 归一化包络；数值 1.0 表示 100% MVC。
- `processed_emg.npz`: 汇总所有阶段，兼容现有 `build-synergy-dataset`。
- `processing.json`: 参数、步骤、MVC 来源、边界策略、输出路径和状态。
- `preprocessing_qc.json`: 每通道质量指标、warning、critical failure 和 `analysis_ready`。
- `preprocessing_comparison.png`: 原始、滤波、整流和包络对比，300 DPI。
- `normalized_envelope.png`: 各通道 `%MVC` 包络，300 DPI。

所有阶段 NPZ 均保留 `time_s`、`sample_index`、`participant_id`、`session_id`、`action_id`、`trial_id`、`trial_index`、传感器 ID、通道名称、side 和 muscle slug。

session 根目录新增 `preprocessing.log.jsonl`、`preprocessing_session_summary.json` 和 MVC 来源清单；数据集根目录批处理后新增 `preprocessing_batch_summary.json`。

后续 `build-synergy-dataset --only-valid` 会同时检查人工 `valid_for_analysis` 与现有的 `preprocessing_qc.analysis_ready`。QC failed trial 只有在操作者显式使用 `--include-invalid` 时才可能进入下游数据集。

## 6. QC 判读

- `qc_pass/analysis_ready=true`: 未检测到配置定义的关键失败；仍应结合贴片记录和动作视频复核。
- `signal_dominated_by_notched_powerline`: 大部分带通信号能量被 notch 去除，不能把剩余波形当作恢复后的有效 EMG。
- `filtered_signal_near_flatline`: 滤波后有效幅度过低，检查贴片、传感器配对和 Control Utility。
- `missing_gap_too_long` 或 `missing_fraction_too_high`: 插值仅为保证数值处理完成，该通道不应进入正式分析。
- `normalized_envelope_exceeds_200pct_mvc`: 可能是 MVC 未充分激活、贴片变化或动作伪迹，需要复核 MVC。

## 7. 测试

```powershell
python -m pytest --basetemp .\pytest_run_local -p no:cacheprovider
```

测试覆盖滤波频响、50 Hz 抑制、零相位峰值时刻、缺失/尖峰处理、短序列拒绝、逐被试 MVC 重算、阶段文件与元数据、批处理和绘图接口。

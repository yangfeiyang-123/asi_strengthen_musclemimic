# MVC 小于动作期 sEMG 时的处理规范

> 适用场景：已经完成 sEMG 采集，但发现某些肌肉在实际动作中的 EMG 包络峰值或高分位值高于对应 MVC 参考值，即归一化后出现 `>100% MVC`。
> 本文只回答一个问题：**这种数据应该怎么处理。**

---

## 1. 最重要的原则

当出现：

\[
EMG_{task} > EMG_{MVC}
\]

时，不要直接认为动作数据错误，也不要直接删除该 trial。

正确的处理原则是：

\[
oxed{
	ext{保留原始动作信号}
ightarrow
	ext{重新评估 MVC 参考是否可靠}
ightarrow
	ext{根据用途决定是否继续使用 MVC 归一化}
}
\]

最重要的三条规则：

1. **不要把超过 100% MVC 的值截断到 100%。**
2. **不要仅因为超过 100% MVC 就删除动作 trial。**
3. **如果 MVC 明显偏小，应把问题标记为“MVC reference unreliable”，而不是“task EMG invalid”。**

---

## 2. 偶尔略微超过 MVC：直接保留

例如：

```text
MVC reference = 1.00

动作 trial：
peak = 1.06
peak = 1.12
peak = 0.95
peak = 1.08
```

或者：

```text
96%
103%
112%
108% MVC
```

这种情况一般不需要特殊处理。

推荐：

```text
保留原 MVC
保留 >100% MVC 的原始比例
不 clip
不删 trial
```

也就是说，`112% MVC` 就保留成 `1.12`，不要改成 `1.00`。

---

## 3. 很多 trial 都明显高于 MVC：优先怀疑 MVC 参考偏低

例如某通道：

```text
MVC = 0.20 mV

动作 trial peak：
0.28
0.31
0.30
0.34
0.29
0.33
```

对应：

```text
140%
155%
150%
170%
145%
165% MVC
```

且这种情况在多个 trial 中反复出现。

这时应把：

```text
mvc_quality = unreliable
```

或：

```text
mvc_reference_status = underestimated
```

但仍可以保持：

```text
task_signal_quality = valid
```

也就是说：

> “动作信号是否可用”和“MVC 分母是否可靠”必须分开判断。

---

## 4. 不要只看单个最大值：使用 P99 / MVC

推荐计算：

\[
R_{99}
=
rac{P_{99}(EMG_{task})}{MVC}
\]

其中：

- \(P_{99}\)：该通道所有有效动作数据的 99% 分位值；
- MVC：当前 MVC reference。

例如：

```text
MVC = 0.20 mV
Task P99 = 0.29 mV
```

则：

\[
R_{99}=1.45
\]

这比单纯看一次最大 spike 更稳健。

---

## 5. 推荐项目级分级

> 以下阈值是项目数据 QC 规则，不是通用生理学标准。

| P99(Task) / MVC | 状态 | 推荐处理 |
|---:|---|---|
| `<= 1.20` | Good | 正常保留；允许少量 >100% |
| `1.20–1.50` | Questionable | 动作保留；建议复核/补采 MVC |
| `1.50–2.00` | Unreliable | MVC 不再适合做强 absolute amplitude reference |
| `> 2.00` | Invalid for absolute amplitude | 必须人工检查；不能再强解释 `%MVC` 幅值 |

关键是：

\[
R_{99}>2
\]

并不自动等价于“动作数据错误”。

如果动作波形稳定、多个 trial 可重复出现，更可能是 MVC reference 明显偏小。

---

## 6. 如果还能重新采 MVC：怎么做

这是最佳方案。

不要修改已有 task EMG，只重新补采 MVC。

建议：

```text
MVC repetition 1
MVC repetition 2
MVC repetition 3
...
```

如果单一姿势容易低估，可增加多个有效 MVC 条件。

最终：

\[
MVC_i^{final}
=
\max_j MVC_{i,j}^{valid}
\]

其中每个 MVC 值必须经过和动作数据完全相同的处理流程。

然后重新计算：

\[
EMG_{\%MVC}
=
rac{EMG_{task}}{MVC_i^{final}}
\]

同时保留：

```text
mvc_original
mvc_repeat_1
mvc_repeat_2
mvc_repeat_3
mvc_final_reference
```

不要覆盖原始 MVC。

---

## 7. 如果无法重新采 MVC：使用“双轨处理”

这是历史数据最推荐的方案。

### 7.1 轨道 A：原始 `%MVC` 永久保留

仍然计算：

\[
E_i^{MVC}(t)=rac{E_i(t)}{MVC_i}
\]

允许：

```text
1.2
1.5
1.8
2.1
```

不要 clip。

这套数据用于：

- 审计；
- 查看 MVC 偏低程度；
- 与传统 `%MVC` 结果兼容。

但不要解释为“肌肉真实激活达到人体最大值的 180%”。

更准确的说法是：

> “动作期 sEMG 为所选 MVC reference 的 180%。”

---

### 7.2 轨道 B：为下游分析重新定义稳健尺度

若 MVC 明显偏低，推荐在 **训练集** 中定义：

\[
S_i=P_{99}(E_i^{clean,train})
\]

然后：

\[
E_i^{robust}(t)=rac{E_i(t)}{S_i+\epsilon}
\]

例如：

```text
MVC = 0.20
Task train P99 = 0.32
```

则：

```text
%MVC:
0.30 / 0.20 = 1.50

robust normalization:
0.30 / 0.32 = 0.94
```

这里的 `0.94` 不能再叫 `%MVC`，应命名为：

```text
train-P99 normalized EMG
```

或：

```text
robust-normalized EMG
```

---

## 8. 为什么使用 P99，而不是 task 最大值

如果使用：

\[
S_i=\max(E_i)
\]

一个异常 peak 可能决定整条通道尺度。

例如：

```text
正常 peak ≈ 0.30
偶发 spike = 0.81
```

使用 max 会把所有正常动作压得过低。

因此推荐：

\[
oxed{
S_i=P_{99}(E_i^{clean,train})
}
\]

而不是使用单个最大采样点。

---

## 9. P99 必须只用训练集计算

正确流程：

```text
所有 trial
   ↓
先划分 train / val / test
   ↓
只使用 train 计算 P99
   ↓
冻结 normalization scale
   ↓
应用到 train / val / test
```

不要：

```text
先使用全部数据计算 P99
再划分 train/test
```

否则会发生数据泄漏。

---

## 10. 做肌肉协同时怎么处理

若目的是 NMF / muscle synergy，而某些 MVC 明显偏小：

### 主分析

推荐：

\[
E_i^{syn}
=
rac{E_i}{P99_i^{train}}
\]

然后：

\[
E^{syn}pprox WH
\]

用于：

- synergy rank；
- \(W\)；
- \(H\)；
- VAF；
- 跨 trial stability。

### 敏感性分析

另做一套：

\[
E_i^{MVC}
=
rac{E_i}{MVC_i}
\]

仍然**不 clip**。

最后比较两种 normalization 下：

- rank \(K\)；
- W cosine；
- H correlation；
- VAF；
- bootstrap stability。

如果两套结果接近，说明协同结论对 MVC 偏小比较稳健。

---

## 11. MVC 不可靠，不代表整个通道都没用

即使：

```text
mvc_quality = poor
```

只要任务 EMG 波形本身稳定，该通道仍可用于：

- onset；
- offset；
- peak timing；
- waveform shape；
- activation duration；
- muscle synergy；
- 跨肌肉时序关系。

因此要区分：

```text
signal_quality
```

和：

```text
mvc_quality
```

例如：

```text
signal_quality = good
mvc_quality = poor
```

这是完全合理的。

---

## 12. 推荐增加 amplitude confidence

可按项目规则设置：

```text
R99 <= 1.20        -> amplitude_confidence = 1.0
1.20 < R99 <= 1.50 -> 0.7
1.50 < R99 <= 2.00 -> 0.4
R99 > 2.00         -> 0.2
```

这样：

> MVC 差的通道不是被删除，而是在 absolute amplitude 监督中自动降权。

如果以后用于 EMG loss：

\[
L=\sum_i c_i L_i
\]

其中：

\[
c_i=amplitude\_confidence_i
\]

即可。

---

## 13. 推荐保存的字段

每个通道至少保存：

```text
mvc_original
mvc_final_reference
mvc_quality

task_p95
task_p99
task_max

task_p99_over_mvc
task_max_over_mvc

normalization_report
normalization_synergy

robust_scale
amplitude_confidence
```

例如：

```json
{
  "muscle": "right_anterior_deltoid",
  "mvc_original": 0.20,
  "mvc_final_reference": 0.20,
  "mvc_quality": "unreliable",
  "task_p99": 0.31,
  "task_max": 0.35,
  "task_p99_over_mvc": 1.55,
  "normalization_report": "percent_mvc_unclipped",
  "normalization_synergy": "train_p99",
  "robust_scale": 0.31,
  "amplitude_confidence": 0.4
}
```

---

## 14. 推荐实现逻辑

```python
ratio = task_p99 / mvc_reference

percent_mvc = task_envelope / mvc_reference
# 不 clip

if ratio <= 1.20:
    mvc_quality = "good"
    amplitude_confidence = 1.0
elif ratio <= 1.50:
    mvc_quality = "questionable"
    amplitude_confidence = 0.7
elif ratio <= 2.00:
    mvc_quality = "unreliable"
    amplitude_confidence = 0.4
else:
    mvc_quality = "invalid_for_absolute_amplitude"
    amplitude_confidence = 0.2

robust_scale = percentile(clean_train_task_envelope, 99)
robust_emg = task_envelope / robust_scale
```

关键是：

```python
percent_mvc
```

和：

```python
robust_emg
```

同时保存，不能互相覆盖。

---

## 15. 如果重新采 MVC 后仍然有动作 >100%

不需要要求：

\[
EMG_{task}\le MVC
\]

永远成立。

例如重新采后：

```text
MVC1 = 0.25
MVC2 = 0.27
MVC3 = 0.28
Task P99 = 0.34
```

仍然：

\[
0.34/0.28=1.21
\]

如果：

- MVC 重复稳定；
- 动作信号稳定；

那么保留 `121% MVC` 即可。

不要无限补采直到所有动作都被“压”到 100% 以下。

---

## 16. 不建议的处理方式

### 不要直接 clip

```python
emg_norm = np.minimum(emg / mvc, 1.0)
```

错误。

### 不要把超过 MVC 的 trial 全删掉

会产生选择偏差。

### 不要偷偷把 MVC 改成最大 task EMG

如果使用 task-based scale，必须明确叫：

```text
train-P99 normalization
```

而不能继续称为 MVC。

### 不要使用 train + test 一起计算 P99

会产生数据泄漏。

### 不要因为 MVC 差就删除整块肌肉

如果波形质量好，时序和协同信息仍然有价值。

---

## 17. 最终推荐方案

### A. `%MVC` 审计层

\[
oxed{
E^{MVC}=E/MVC
}
\]

- 永久保存；
- 不 clip；
- 允许 >1。

### B. MVC 质量层

使用：

\[
R_{99}=P99(Task)/MVC
\]

标记：

```text
good
questionable
unreliable
invalid_for_absolute_amplitude
```

### C. 肌肉协同层

MVC 不可靠时使用：

\[
oxed{
E^{syn}=E/P99_{train}
}
\]

### D. 模型监督层

\[
oxed{
MVC\ unreliable
\Rightarrow
amplitude\ down-weight
}
\]

而不是：

\[
oxed{
MVC\ unreliable
\Rightarrow
delete\ channel
}
\]

---

## 18. 最终决策表

| 情况 | 处理 |
|---|---|
| 偶尔略高于 100% MVC | 正常保留，不 clip |
| 多个 trial 120–150% | MVC questionable，动作保留 |
| 多个 trial 150–200% | MVC unreliable；推荐重采；下游用 robust scale |
| 多个 trial >200%，但波形稳定 | MVC 不再用于强 absolute amplitude 解释；动作保留 |
| 可以补采 MVC | 重采多个有效 MVC，更新 reference |
| 无法补采 MVC | `%MVC` 保留审计；主协同分析用 train-P99 |
| MVC 不可靠但时序稳定 | 保留 timing / synergy |
| 重新采后仍稍高于 100% | 可以接受，不要求 MVC 覆盖所有动态峰值 |

---

## 19. 一句话操作规范

> **当 MVC 小于动作期 sEMG 时，不删除动作数据、不截断超过 100% MVC 的值；首先计算 P99(Task)/MVC 判断 MVC 是否系统性偏低。若可以补采，则用多个有效 MVC 更新 reference；若不能补采，则保留未截断的 `%MVC` 作为审计结果，同时在肌肉协同等下游分析中采用仅由训练集估计的稳健通道尺度（推荐 train-P99），并将 MVC 不可靠转化为“绝对幅值降权”，而不是删除该通道。**

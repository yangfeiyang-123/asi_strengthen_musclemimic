# ChinaJump Primitive 协同：从采集到训练

> [!IMPORTANT]
> 本 runbook 现在只接受 physical signal v2。旧的 ctrlrange affine
> excitation 不能进入 NMF；必须保留 raw `data.ctrl`，验证所有 channel 为 scalar
> MuJoCo muscle，并计算 `clip(raw_ctrl,0,1)`。迁移后仍须重新拟合 basis 和使用
> fresh optimizer，详见
> [`肌肉生理约束实施契约_v2.md`](肌肉生理约束实施契约_v2.md)。

## 1. 现在的结论

采集到合格的基础动作数据后，仓库已经可以完成：

```text
真实 applied data.ctrl
→ 严格入库与 train/val 防泄漏 QC
→ primitive source manifest
→ global/regional NMF、held-out VAF 与稳定性门
→ 固定 W 和 coefficient statistics
→ 354D 模型/actuator/ctrlrange 离线 preflight
→ B0/B1 bootstrap 训练绑定
→ canonical launcher 启动训练
```

这里的“可以训练”指独立标记的 primitive bootstrap：

- `B0`：固定 `Wc`，ASI 关闭；
- `B1`：固定 `Wc`，ASI 打开。

它们能用于验证“基础动作协同是否让 ChinaJump 开始可学”，但不会声称已证明
`W` 覆盖完整 ChinaJump。正式 `S0/S1` 还必须有一份与 primitive 数据独立的
ChinaJump full-action target-control proxy，并通过 phase-conditioned static coverage gate。
两种 readiness 被代码、配置、release manifest 和 run id 分开，不能混用。

## 2. 采集前必须固定的合同

### 2.1 使用实际训练模型

基础动作必须在与 ChinaJump 相同的完整编译模型上采集：

```text
MjxMyoFullBody
disable_fingers=True
mjx_backend=warp（与当前 ChinaJump Hydra config 一致）
354 个 actuator
exact ordered actuator names
exact actuator ctrlrange
```

不要把 `musclemimic_models` 中尚含手指 actuator 的原始 416D XML 直接填进
catalog。采集器拿到真实 `env._model` 后，应先保存无损的 `.mjb`：

```python
from musclemimic.synergy.primitive_ingest import save_compiled_model_artifact

save_compiled_model_artifact(
    env._model,
    "artifacts/primitive_capture/chinajump_body_354_runtime.mjb",
)
```

必须从采集 rollout 实际使用的环境实例保存，不要另行构造 backend/contact 选项不同的
CPU 或 MJX-JAX 模型。catalog 的兼容字段 `model_xml_path` 应指向这份 `.mjb`。builder 会回读二进制模型，
重新计算完整 `MjModel.__getstate__()` SHA256；训练 wrapper 还会再次与当前运行模型核对。

### 2.2 每个 trial 保存实际物理控制

每一帧保存从 `s_t` 实际施加到 `s_{t+1}` 的 `data.ctrl`，不能只保存 policy 的
normalized action。推荐直接使用采集 writer：

```python
from musclemimic.synergy.primitive_recording import write_primitive_trial_npz

write_primitive_trial_npz(
    "artifacts/primitive_capture/raw/P06_train_01.npz",
    model=env._model,
    actuator_names=env.policy_actuator_names,
    teacher_ctrl_physical=recorded_applied_ctrl,  # [T, 354]
    phase_id=recorded_event_phase,                # integer [T]
    success=True,
    muscle_activation=recorded_activation,        # optional [T, 354]
)
```

writer 会从编译模型验证 `dyntype=muscle`、`actnum=1`、`actadr` 与
`ctrlrange=[0,1]`，再按 MuJoCo 语义计算 `clip(raw data.ctrl,0,1)`，并写入
model、actuator 顺序和 v2 signal contract hash。默认拒绝覆盖已有 trial。

以下数据不能进入 `W`：

- 只有 `qpos/qvel` 的运动学文件；
- 只有 normalized policy action；
- 把 activation 或 EMG 当作 excitation；
- 失败、提前终止或缺少关键阶段的 rollout；
- 完整 ChinaJump target rollout，或现有 train8/val2 validation motion；
- 由同一个 early-synergy policy 生成、再用于证明自身 coverage 的循环证据。

## 3. P01–P12 数据组织

模板位于：

```text
fullbody/config_specific_task/stage1_body/primitive_catalog/
├── chinajump_primitives_p01_p12_v1.json
├── trial_entries_template_v1.json
├── raw_trial_npz_contract_v1.json
└── phase_schemas/
```

动作库包括站立、准备姿态、慢蹲、蹲起、countermovement、低幅双脚跳、双脚落地、
轴向转体、split-step、上肢摆动、低速分解跳和落地恢复。应复制整个
`primitive_catalog/` 目录以保留相对的 `phase_schemas/`，再填写工作副本；若工作副本
换了目录层级，同时把 `regional_grouping_path` 改成正确路径。不要把模板中的空路径
当作默认值继续运行。

每个启用的 task 至少需要：

- 两个相互独立的 train trials；
- 一个独立 val trial；
- 每个 trial 覆盖该 task phase schema 的所有 phase；
- 每个 phase 至少两个有效 transition；
- 一个真实存在且内容会被逐文件哈希的 controller/optimizer artifact；
- 唯一的 `trial_id` 和不跨 split 重用的 `motion_path`。

三次 trial 只是代码允许的最低值。正式拟合建议每个动作采更多独立重复，尤其是
起跳、落地、转体和 split-step。

先检查模板或已填写 catalog：

```bash
uv run musclemimic-synergy-validate-primitive-catalog \
  --catalog artifacts/primitive_capture/catalog.json
```

采集完成后执行 build-ready 检查：

```bash
uv run musclemimic-synergy-validate-primitive-catalog \
  --catalog artifacts/primitive_capture/catalog.json \
  --require-build-ready
```

如需单独检查入库结果：

```bash
uv run musclemimic-synergy-ingest-primitives \
  --catalog artifacts/primitive_capture/catalog.json \
  --output-dir artifacts/primitive_capture/fit_ready
```

输出包含 `train_*.npz`、`val_*.npz`、`metadata.json`、`dataset_qc.json`、
`source_checkpoints.json` 和已封存的 `regional_grouping.json`。输入不变时可幂等复用；
任何 catalog、raw NPZ、controller、phase schema、model 或 grouping 字节变化都会改变
build fingerprint，已有不同产物不会被静默覆盖。

## 4. 一键构建 bootstrap 训练 release

流水线默认只做计划；`plan` 不写文件，也不会启动 PPO：

```bash
source configs/env.sh
uv run musclemimic-chinajump-synergy-pipeline plan \
  --readiness bootstrap \
  --primitive-catalog artifacts/primitive_capture/catalog.json \
  --output-root artifacts/stage1_synergy/chinajump_v1
```

确认计划后显式执行：

```bash
source configs/env.sh
uv run musclemimic-chinajump-synergy-pipeline apply \
  --readiness bootstrap \
  --primitive-catalog artifacts/primitive_capture/catalog.json \
  --output-root artifacts/stage1_synergy/chinajump_v1
```

`apply` 会把所有派生产物写入：

```text
<output-root>/.objects/<input-fingerprint>/
```

只有完整对象才会原子发布 `release.json`、`READY.json`、release pointer 和
`bindings/bootstrap.env`。最终 JSON 中必须同时满足：

```text
readiness = training_ready_s
ready_for_training = true
release_mode = bootstrap
formal_target_coverage = false
evidence_limitations = [no_independent_chinajump_target_control_coverage]
```

若 NMF 没有 rank 同时通过 held-out VAF、local VAF、initialization、split-half、
bootstrap 和 cross-trial 稳定性门，流水线只保留诊断产物，不会发布训练 bindings。

发布后再做一次 release preflight；加入 `--real-env-smoke` 时会真实构造环境并加载
354D wrapper，但不会创建 W&B run、checkpoint 或训练进程：

```bash
source configs/env.sh
export CUDA_VISIBLE_DEVICES=<physical_gpu_index>
uv run musclemimic-chinajump-synergy-pipeline preflight \
  --release <apply 输出中的 release_pointer_path> \
  --real-env-smoke
```

## 5. 启动 B0/B1

先 source `apply` 返回的 `training_bindings.shell_path`。实际训练只能通过根目录
`AGENTS.md` 规定的 launcher；下面以 B0 为例：

```bash
source <absolute_path_to_bindings/bootstrap.env>
export CUDA_VISIBLE_DEVICES=<physical_gpu_index>
export MUSCLEMIMIC_JAX_CACHE_KEY=chinajump_stage1_b0_primitive_bootstrap
export MUSCLEMIMIC_TRAIN_LOG=datasets/ChinaJump/training/logs/chinajump_stage1_b0_primitive_bootstrap.log
scripts/run_fullbody_training.sh \
  --config-name=config_specific_task/stage1_body/conf_fullbody_chinajump_early_synergy_bootstrap \
  config_status.allow_nonproduction_runtime=true \
  wandb.mode=online
```

B1 使用新的 run id、JAX cache key 和日志，并换成：

```text
config_specific_task/stage1_body/conf_fullbody_chinajump_early_synergy_bootstrap_asi
```

正式启动前仍需按 `AGENTS.md` 完成聚焦测试、Hydra resolve、GPU 进程检查；启动后要
确认本地 retarget cache、run manifest、W&B URL、目标物理 GPU/PID 和
`Starting training...`。动作 ABI 不同，B0/B1 不得恢复 F0/F1 或 S0/S1 checkpoint。

## 6. 从 bootstrap 升级到正式 S0/S1

正式 coverage 数据必须来自成功的 full-354D ChinaJump teacher，或经过同模型 forward
replay 的 full-action trajectory optimizer。仓库提供：

- `write_target_control_source_manifest(...)`；
- `write_target_control_qc(...)`；
- `build_coverage_proxy(...)` / `musclemimic-synergy-build-coverage-proxy`；
- 自动绑定 producer manifest 的 `musclemimic-synergy-static-coverage`；
- `load_coverage_proxy_artifact(...)` 的全量重新校验。

当前正式 proxy 是 v2 合同。source builder 不接受手填的 checkpoint/optimizer hash，
必须传仍然存在的 `checkpoint_artifact_path=...` 或
`optimizer_artifact_path=...`，由 builder 现场封存完整文件清单和内容哈希；以后每次加载
还会重新审计该产物。QC 也没有“默认通过”：tracking、forward replay、完整轨迹覆盖、
early termination、frame coverage、episode success、excitation saturation 及对应阈值都要
显式提供，`passed` 由代码重算。旧 v1 proxy 不能改 JSON 冒充 v2，必须从原始控制数据和
仍存在的 controller/optimizer 产物重新生成。

它不会从 WHAM、SMPL、`qpos/qvel` 或失败的 F0 rollout 猜出 354D excitation。拿到合法
proxy 后，用同一个 primitive catalog 重新执行：

```bash
source configs/env.sh
uv run musclemimic-chinajump-synergy-pipeline apply \
  --readiness formal \
  --primitive-catalog artifacts/primitive_capture/catalog.json \
  --coverage-proxy-artifact artifacts/chinajump_proxy/v2/proxy_manifest.json \
  --output-root artifacts/stage1_synergy/chinajump_v1
```

只有 producer provenance、phase 1–4、static reconstruction、rank/condition/saturation
门全部通过，才会发布正式 S0/S1 bindings。短时 dynamics oracle 仍是把方法提升为
canonical、并声称覆盖起跳/飞行转体/落地动力学之前的后续增强门；首轮正式 Phase-A
比较可以先使用严格的 static gate，但结论只能写成“目标 excitation proxy 可静态重建”。

## 7. 应怎样比较

第一轮建议按两个层次报告：

```text
Bootstrap 诊断：F0, F1, B0, B1
正式主比较：  F0, F1, S0, S1；必要时再加 SR0, SR1
```

所有组保持 ChinaJump train8/val2、reward、terminal、640M steps、validation 和 promotion
合同一致；使用相同 PPO seed 列表和 fresh optimizer。主指标包括 full-motion success、
frame coverage、early termination、逐阶段 tracking error、sample efficiency、物理 excitation
能量/饱和，以及 synergy coefficient 的使用率和饱和率。

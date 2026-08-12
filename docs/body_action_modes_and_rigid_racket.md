# 354 维动作模式与刚性球拍主线

> [!IMPORTANT]
> 当前生产动作合同是 excitation v2：所有 body muscle runtime
> `ctrlrange=[0,1]`，policy ABI 仍为 `[-1,1]`，effective excitation 为
> `clip(raw data.ctrl,0,1)`。`full_354_action_v1`、
> `early_synergy_action_v1` 和 `frozen_body_synergy_decoder_v1` 均为
> checkpoint-incompatible legacy artifact；参见
> [`肌肉生理约束实施契约_v2.md`](肌肉生理约束实施契约_v2.md)。

本文记录当前生产契约。在同一阶段的成对实验内，三种身体动作表示共享完全相同的
354 个非手指 actuator、顺序、ctrlrange、观测、奖励、终止条件和球拍模型；实验间只
改变策略输出坐标。

| 模式 | 策略输出 | 物理输出 | 允许的固定 artifact |
|---|---:|---:|---|
| `full_354` | 354 | 354 | 无 |
| `fixed_synergy` | `rank(W)` | 354 | `W`、系数变换、tonic baseline |
| `fixed_synergy_residual` | `rank(W)+rank(R)` | 354 | 上述 artifact 加固定小维 `R` 与 `alpha` |

`full_354` 是独立的端到端科学基线，不是协同 decoder 内部的 354 维
state-only baseline。后者会绕过 `W`，所以正式 latent synergy 配置默认关闭
`synergy_include_baseline`；只有显式命名的消融实验可以重新开启。

## 动作配置

直接动作模式：

```yaml
experiment:
  action_representation:
    mode: full_354
    enabled: false
    expected_underlying_action_dim: 354
```

固定协同模式：

```yaml
experiment:
  action_representation:
    mode: fixed_synergy
    enabled: true
    basis_path: ${oc.env:MUSCLEMIMIC_SYNERGY_BASIS}
    expected_basis_fingerprint: ${oc.env:MUSCLEMIMIC_SYNERGY_BASIS_FINGERPRINT}
    expected_underlying_action_dim: 354
```

带固定低维残差的协同模式把 `mode` 改为
`fixed_synergy_residual`，并配置已有的 residual basis、允许肌肉 mask 和
`alpha`。不能使用 354 维 residual。

旧的、没有 `mode` 的 `enabled` 开关仍可读取；新配置必须写显式 `mode`。
若显式 `mode` 与 `enabled` 冲突，加载会报错，不会静默切换框架。

运行时会生成：

- `action_manifest`：绑定动作模式、有序 actuator、ctrlrange、MuJoCo model、
  `W/R` 和物理动作接口；
- `body_synergy_contract`：版本化 `BodySynergyContractV2`，同时保存下面两层
  独立、可重算的 SHA-256：
  - `portable_decoder_core_fingerprint` 绑定 mode、354 个 actuator 的有序 ABI、
    正式 ctrlrange、`W`、coefficient transform/statistics、tonic baseline、
    结构化 `R`/mask/fit、`alpha` 和字典来源；
  - `stage_runtime_binding_fingerprint` 绑定上述 portable fingerprint、当前
    MuJoCo model、runtime ctrlrange、完整物理动作接口和该阶段 coverage 证据。

跨 Stage-1、不同球拍质量的 Stage-2、蒸馏和 Stage-3 传递同一 decoder 时，使用
`assert_portable_compatible`：运行模型和每阶段 coverage 可以不同，但 `W` 及其坐标
语义必须完全相同。同一阶段恢复 optimizer/checkpoint 时使用
`assert_exact_runtime_compatible`（历史 `assert_compatible` 是它的别名），此时 model、
interface 和 coverage 也必须完全相同。通用 artifact 读取器
`load_compatible_body_synergy_contract` 要求调用者显式选择
`portable_decoder_core` 或 `exact_runtime`，不会自行猜测消费阶段。

不同 mode、不同 `W`、重新拟合/旋转后的 `W` 或不同 residual 会改变 portable
fingerprint，必须使用新的 decoder/checkpoint family。不同物理模型或 coverage 会保留
portable fingerprint、改变 stage-runtime fingerprint，因此允许显式的跨阶段初始化，
但禁止当作同一 runtime 继续恢复旧 optimizer。禁止跨模式恢复 checkpoint。
只有扩展字典严格满足 `W_new=[W_old,W_extra]` 时，旧系数才能显式迁移为
`[c_old,0]`；普通重拟合不能这样迁移。

序列化采用失败关闭：两个子 fingerprint 和完整 `contract_fingerprint` 都是必填字段；
缺字段、未知字段、重复 JSON key，或任一层重算不一致都会拒绝加载。合同已经接入
Stage-1 action wrapper、Stage-2 teacher/DAgger collector、latent checkpoint/runtime 和
Stage-3 消费链路，而不是只存在于离线 helper 中。

协同主线只允许网络预测 raw `c/rho` 坐标。唯一的纯 JAX 冻结解码器位于
`musclemimic/synergy/frozen_decoder.py`，在 early PPO、teacher collection、latent
训练、checkpoint restore 和 Stage-3 runtime 中共用同一数值定义：

```text
c   = cmax * sigmoid(raw_c / temperature + logit(center / cmax))
rho = alpha * tanh(raw_rho)
u   = clip(tonic + W c + R rho, excitation_bounds)
a   = physical_to_normalized(u)
```

正式 `fixed_synergy`/`fixed_synergy_residual` 不允许 learned 354-D baseline 或
354-D residual 绕过 `W`；旧的独立 `W` latent decoder 只保留为显式命名的
`legacy_synergy_decoder_ablation`。冻结 artifact 同时绑定全部数值数组、actuator 顺序、
portable contract 和文件内容哈希，篡改任一项都会在加载时失败。

teacher/DAgger shard 采用双动作语义：`teacher_action` 与 decoded `teacher_mu` 保存实际
354-D body-action target；`teacher_policy_action`、`teacher_policy_mu` 和
`teacher_policy_log_std` 保存低维 raw `c/rho` 高斯坐标，并另存解码后的 synergy/residual
coefficients。Gaussian log-std 与 KL 只在真实的低维 `c/rho` 坐标中定义；非线性
decoded-354 分布明确标记为 unavailable，若请求其对角 Gaussian KL 会失败关闭。

协同 rank 选择采用失败关闭：选择通过所有离线门控以及所要求动态门控的最小
候选 rank；不存在合格候选、区域总 rank 超预算或动态证据缺失/不匹配时直接
报错，不回退到“VAF 最好”的不合格字典，也不静默截断。

`--mode both` 的 primary decoder 现在是
`physical_excitation_unit/hybrid_global_regional`：它完整保留 regional columns，只追加
不能被 regional 非负锥解释、与已有列不重复、且在 held-out 上带来严格正 marginal
VAF 的原始非负 global columns。联合 rank 上限为 64，并再次要求 global VAF、local
VAF q10、condition number 和 effective-rank 全部通过；global-only 和 regional
composite 保留为可追溯 comparator。要求 dynamics evidence 时先发布 content-bound
candidate inventory，只有对应 candidate fingerprint 的精确动态证据通过后才会写
primary manifest、coefficient statistics 和 preferred pointer；失败/缺证据会清除旧的
primary 文件，不会复用 stale artifact。

## 可比较的训练入口

ChinaJump Stage-1 的显式、公平 direct 基线为：

- `conf_fullbody_chinajump_full354_fair`：`full_354`，ASI 关闭；
- `conf_fullbody_chinajump_full354_fair_asi`：`full_354`，ASI 打开；
- `conf_fullbody_chinajump_early_synergy{,_asi}`：冻结 `W`；
- `conf_fullbody_chinajump_early_synergy_residual{,_asi}`：冻结 `W+R`。

direct 与 synergy 组都把初始扰动标定为 effective excitation RMS 0.08。只有在
runtime 已验证全部 muscle `ctrlrange=[0,1]` 后，direct 的 normalized-action
Jacobian 才是 0.5，因此等价初始标准差为 0.16；synergy 组从冻结 decoder
零点 Jacobian 逐维求解。动作表示以外的 motion split、reward、terminal、训练步数和
ASI 开关保持一致。每个模式使用独立 `run_id` 和 fresh optimizer。

Stage-3 的正式成对入口为：

- `experiments/posttrain/incoming_shuttle_hit_full354_v1.yaml`：策略直接输出 354-D，
  `stage3_lab.enabled=false`，不需要 latent checkpoint；
- `experiments/posttrain/incoming_shuttle_hit_impact_recovery_v2.yaml`：固定协同/latent
  路径，必须绑定选中的 frozen decoder 与 latent checkpoint。

两者共用同一 scene、train/eval feed bank、impact/recovery targets、reward、episode 和
刚性球拍。direct prerequisite 明确记录 `latent_checkpoint_fingerprint=null`，并绑定
spec/scene/target/feed/control/policy ABI；协同分支额外绑定 latent selection。promotion
只比较共同可定义的任务、能量、饱和、有限控制和 attachment 指标；raw-latent、OOD 与
prior-naturalness 等 LAB-only 指标在 direct 报告中为 N/A，且不参与 direct gate。

正式 task-causal 干预只对选中的 `fixed_synergy` 分支执行，因为只有该分支定义了可干预
latent 坐标；`full_354` 明确记录为 N/A，不会被错误送入 latent intervention。历史
latent decoder 的 direct head 仅保留为显式 `latent_direct_ablation`，不属于正式
`full_354` 对照。

checkpoint restore 同样失败关闭：同一 run 的自动恢复必须匹配完整 runtime contract；
带显式 parent lineage 的跨阶段初始化只允许 portable core 一致。新动作模式不会对缺少
合同的旧 checkpoint 做 shape-only restore。

Forehand Clear 的 early-unified 协同链路使用一个共享 preset：

```text
conf_fullbody_forehand_clear_early_unified_synergy_v4
  -> conf_fullbody_forehand_clear_racket_mass_025_early_unified_synergy_v4
  -> conf_fullbody_forehand_clear_racket_mass_050_early_unified_synergy_v4
  -> conf_fullbody_forehand_clear_racket_mass_075_early_unified_synergy_v4
  -> conf_fullbody_forehand_clear_racket_mass_100_early_unified_synergy_v4
```

五个配置都 compose
`config_specific_task/presets/forehand_early_unified_action_v4`，因此 `W`、primitive source、
coefficient statistics、hybrid thresholds、exact dynamic thresholds 和 actuator ABI 完全
相同。preset 显式使用 `primitive_runtime_model_compatibility=portable_body_action_abi`：
允许 exact-child 球拍使完整 model hash 改变，但仍要求非空 runtime model binding、有序
body actuator 与 ctrlrange 完全一致；默认兼容模式仍是 `exact_runtime_model`。每个
Stage-2 rung 的 `resume_from` 必须由对应 promoted-parent 环境变量提供，
lineage 明确串为 Stage1→25%→50%→75%→100%；每档保留 policy 权重但重置 optimizer 与
learning-rate schedule，且使用新 `run_id`、`auto_resume=false`。缺少 parent path 会在
Hydra resolve 时失败，不能偷偷从错误动作族或旧质量档开始。

对应的 Forehand direct 对照仍是同名 Stage-1 body 配置与不带
`_early_unified_synergy_v4` 后缀的四个 `racket_mass_{025,050,075,100}` 配置；这些
354-D runtime 会生成 `full_354` action contract。synergy variants 直接继承各自 direct
配置后只覆盖共享 action preset 与 run/lineage identity，因此 reward、mass scale、数据与
terminal contract 保持逐档可比。

## 手驱动球拍的生产语义

Stage-3 主线为：

```text
354-D fingerless body
  -> 肩 / 肘 / 前臂 / 手腕驱动 thirdmc_r
  -> 球拍作为 thirdmc_r 的 jointless exact child 同步运动
  -> 保留 racket-shuttle contact
```

球拍附件由
`configs/racket_attachment/forehand_clear_rigid_v4_custom.json` 唯一约束，Stage-2 和
Stage-3 共用其父体、局部位姿、质量、惯量、拍面变换和碰撞语义。生产场景中：

- 无球拍 freejoint；
- 无 hand-racket weld/equality；
- 无手指 joint、actuator、tendon；
- 无 hand provider、hand policy 或 finger observation filter；
- 人体—球拍 collision mask 兼容对为 0；
- stringbed 使用自定义连续接触力与高速 event rebound，native proxy—shuttle 接触关闭，
  避免同一击球被计算两次；
- native racket-frame—shuttle contact 保留，stringbed ground proxy 也保留独立碰撞位；
- event 给羽毛球的线性冲量同时以等量反向的 point impulse 施加到球拍，并通过
  jointless exact-child 的 wrist/forearm/elbow/shoulder/root ancestor DOF 传播；CPU 与 MJX
  使用相同语义。

这称为 rigid-tool control，不应描述为 learned physical grip。若以后研究真实抓握，
应作为单独分支引入无穿透握姿、拍柄接触/摩擦和独立手部控制器，不能与当前
rigid-tool baseline 混写。

当前 event 模型严格闭合线动量，并通过球拍侧施力点产生人体/球拍链上的力矩；它尚未按
cork 偏心冲量显式重置羽毛球角速度。因此若后续研究击球后的 shuttle spin，应新增带角
冲量的独立验证模型，不能把当前线性 rebound 结果解释为完整旋转碰撞辨识。

重新生成生产场景：

```bash
source configs/env.sh
PYTHONPATH="$PWD" uv run python -m \
  environment.overall_environment.src.incoming_scene \
  --out environment/overall_environment/assets/overall_incoming_hit_scene.xml \
  --racket-attachment-contract \
  configs/racket_attachment/forehand_clear_rigid_v4_custom.json
```

Stage-3 runner 会检查已有场景；旧的 416 维 weld 场景在
`build_if_missing: true` 时会按上述契约重建，否则失败关闭。

注意：当前生成的手指 actuator（仅旧 416 维兼容场景存在）实际 ctrlrange 为
`[-1,1]`，normalized `0` 对应 physical `0`。方案文档中关于 `[0,1]` 下映射到
`0.5` 的描述不适用于当前生成模型；生产 exact-child 场景则完全没有这些通道。

本文不授权启动训练。所有正式训练仍必须遵守仓库根目录 `AGENTS.md`，通过
`scripts/run_fullbody_training.sh` 和显式物理 GPU 启动。

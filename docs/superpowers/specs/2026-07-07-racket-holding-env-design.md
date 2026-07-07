# 手持球拍轨迹模仿环境 (Racket-Holding Trajectory Imitation Env) 设计

日期: 2026-07-07
状态: 待评审

## 背景与目标

第一阶段用 `MjxMyoFullBody` + `ImitationFactory` 让**徒手**肌肉人形跟随 retarget 后的挥拍轨迹，已跑通 GPU+JAX(MJX) 加速训练。

现在需要一个**人手持球拍**的仿真环境，用于"持拍学轨迹"（第一阶段的持拍版），并**必须兼容 musclemimic 现有 GPU+JAX(MJX) PPO 训练链路**，作为通往下游"击球残差 / soft-weld 课程 / ghost_racket / impact_target"模块的桥梁。

## 建模决策：球拍以**刚性子 body**固连到右手 `thirdmc_r`

球拍来自 `environment/racket/assets/badminton_racket_rigid.xml`（0.09 kg，拍柄八棱 + 拍头框 + 杆）。将其 **freejoint 去掉**，作为右手第三掌骨 `thirdmc_r`（掌心，下游 soft-weld 的 `SOFT_WELD_BODY1`）的**无关节固定子 body**挂入 MyoFullBody spec。

选此方案的依据（对齐下游）：

- 下游 `soft_weld_schedule` 的**起始档 `strong_weld`（weld_strength=1.0, solref=0.002）物理上≈刚性**。刚性持拍环境正是下游课程的**起点状态**，训好的 body 策略无缝落到 overall 击球场景第一档。
- 刚性 = **零新增 DOF**：qpos/qvel/nu 与现有 `MjxMyoFullBody`(disable_fingers) 完全一致（89/88/354）。徒手 retarget 轨迹**零改动复用**；obs/action 不变 → 预训 body 策略与现有 forehand_clear runner、以及下游 `LatentBodyPolicy`（只作用 body 肌肉）**checkpoint 兼容**。
- **无 equality 约束** → jax/warp 双后端 `mjx.put_model` 稳过（已用探针验证：ngeom=514, nq=129, nu=416, put_model OK）。
- 挂到 `thirdmc_r` 而非 `lunate_r`，与 overall 场景 weld 的 body1 一致，抓握位姿可精确对齐。

**为何不用自由体+weld / 真实手指抓握**：weld 版仅在课程"弱化 weld"后才需要，而该表示 overall 场景已具备；在此再造会平白多 7 维 DOF、需可 jit 的 reset 重定位、打乱 obs。真实手指×拍柄接触 MJX 3.4 不支持，只能 CPU，与 GPU 需求冲突。二者对"持拍学轨迹"目标全面更差。

## 组件与文件

1. **新环境类** `musclemimic/environments/humanoids/myofullbody_racket.py`
   - `MyoFullBodyRacket(MyoFullBody)`：`mjx_enabled=False`（CPU）。
   - `MjxMyoFullBodyRacket(MjxMyoFullBody)`：`mjx_enabled=True`（jax/warp）。
   - 关键覆写：`_apply_spec_changes(spec)` = `super()._apply_spec_changes(spec)` 后调用 `self._inject_racket(spec)`。
   - `_inject_racket(spec)`：
     - 载入 racket rigid spec，删除 `racket_free` freejoint。
     - 把球拍碰撞 geom 的 contype/conaffinity 移到独立 bit（4/4），使其**不与人体（bit1）碰撞**——持拍学轨迹阶段无需拍-体接触。string_visual 保持 0/0。
     - 计算 `thirdmc_r → racket` 相对位姿并作为 attach frame 的 pos/quat，使球拍抓握位姿**等于 overall 场景**：优先从 `configs/right_hand_racket_grip_reference.json` 的 `qpos`/`racket_freejoint_qpos` 在 grip 参考场景里前向解算得到；文件缺失时回退到内置默认 forehand 变换。变换在**构造期**（numpy/CPU）算好后固化进 spec，运行期无额外计算。
     - `base.attach(racket_spec, frame=frame, prefix="racket_")`。
   - 参数（均有默认值，可 override）：`enable_racket=True`、`racket_attach_body="thirdmc_r"`、`racket_grip_pos`/`racket_grip_quat`（None=自动解算）、`racket_collision_bit=4`、`racket_xml_path=None`（默认用仓库 rigid 资产）。
   - 新增 mimic 站点：**不改** body2sites_for_mimic（保持轨迹关节/站点匹配不变）。可选加一个 `racket_head_site` 供后续 ghost/impact 用，但**不进** mimic/obs 集合。

2. **注册** `musclemimic/environments/humanoids/__init__.py` 追加：
   ```python
   from .myofullbody_racket import MyoFullBodyRacket, MjxMyoFullBodyRacket
   MyoFullBodyRacket.register()
   MjxMyoFullBodyRacket.register()
   ```

3. **可跑训练配置** `BadmintonMimic/experiments/fullbody/config_specific_task/conf_fullbody_badminton_racket_gmr.yaml`
   - `defaults: [/conf_fullbody_badminton_gmr, _self_]`，仅 override：`experiment.env_params.env_name: MjxMyoFullBodyRacket`。
   - 轨迹数据、reward、goal、mimic 站点、验证配置全部继承（与徒手 badminton 配置一致），确保零改动复用现有 forehand_clear 轨迹。

4. **烟测脚本** `BadmintonMimic/scripts/smoke_racket_env.py`（或 `tests/test_myofullbody_racket.py`）
   - CPU：构造 `MyoFullBodyRacket`，断言 nq/nv/nu == 徒手版；`reset()`/`step(zeros)` 有限；球拍 body 挂在 `thirdmc_r`、mass≈0.09。
   - MJX：`mjx.put_model` 成功（jax 后端）。
   - 断言球拍 geom 与人体 geom 不产生碰撞对（碰撞组隔离生效）。

## MJX 兼容处理

- **jax 后端**：父类 `_modify_spec_for_mjx` 已把非 floor geom 的 contype/conaffinity 置 0，球拍随之禁碰撞，`put_model` 稳过。
- **warp 后端**：父类保留全部接触；球拍 geom 已在 `_inject_racket` 中置于独立 bit(4/4)，只会彼此（同一刚体，无自碰）"碰"，实际不与人体/地面产生接触对，nconmax 预算不受额外拍-体接触冲击。
- 无 equality/weld → 不占 njmax 约束预算，warp 全 batch 无新增限制。

## 验证标准（完成判据）

1. `MyoFullBodyRacket()` CPU：nq/nv/nu == `MyoFullBody(disable_fingers=True)`（89/88/354）；`thirdmc_r` 下有 `racket_racket` body，mass≈0.09；reset/step 有限。
2. `MjxMyoFullBodyRacket()`：`mjx.put_model` 成功（jax）。
3. 用新 config 起一小段 GPU 训练（少量 timesteps）无异常、loss 有限、能出 checkpoint。
4. 球拍-人体无碰撞对；obs/action 维度与徒手 badminton 配置一致。

## 影响面与回滚

- 纯**新增**：新文件 + `__init__.py` 两行注册 + 新 config + 烟测。不改现有环境/训练代码路径。
- 回滚：删除新文件与注册两行即可，无副作用。

## 后续（不在本次范围）

- weld 版持拍环境（自由体+软 weld，用于课程弱化 weld 阶段）——overall 场景已具备，按需再桥接。
- 抓握位姿精修（复用 grip seed 迭代解），本次用参考变换/默认即可。

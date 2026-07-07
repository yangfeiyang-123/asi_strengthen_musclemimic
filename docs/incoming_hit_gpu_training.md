# Incoming Shuttle Hit — GPU 训练指南

来球击打任务（人站本方半场中心、右手 weld 持拍、羽毛球从对面飞来）的 GPU 并行 RL 训练。
与 musclemimic 轨迹跟踪管线完全独立。

## 一、环境准备（每个 shell 一次）

```bash
cd /data3/yangfeiyang/WorkSpace/musclemimic
source configs/env.sh   # 清洗 LD_LIBRARY_PATH（剔除系统 CUDA 路径）+ 挂 cuda-compat 12.4
```

- 系统 CUDA 未被修改；GPU JAX 用的是 venv 内 pip 自带的 CUDA 12.6 库。
- Warp 后端需要 CUDA 驱动 API ≥ 12.4，由仓库自带的 `.local/cuda-compat-12.4/compat`（用户态前向兼容库）提供，env.sh 会自动挂载。
- 选卡：`export CUDA_VISIBLE_DEVICES=0`（写本文时 GPU 1/2/3 被其他训练占用）。

## 二、开始训练

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python musclemimic/badminton/scripts/run_incoming_shuttle_hit.py \
  --stage train-gpu \
  --num-envs 2048 --rollout-steps 64 \
  --total-env-steps 20000000
```

- 后端默认 `--impl warp`（生产路径）；`--impl jax` 仅用于调试（慢 ~15×）。
- PPO 超参（update_epochs / hidden_sizes / learning_rate / action_std_init）从 spec 的
  `ppo:` 段读取：`experiments/posttrain/incoming_shuttle_hit_v1.yaml`。
- 输出：`outputs/posttrain/IncomingShuttleHit/v1/train_gpu/`
  - `metrics.jsonl` — 每次迭代一行（reward / hit_rate / crossed_net_rate / landing_score / sps）
  - `policy_latest.npz` + `.json` — checkpoint（MLP 参数 + obs 归一化统计 + 元数据）
  - `train_report.json` — 最终汇总

实测吞吐（A100，num_envs=2048，含 PPO 更新）：**~1,100 env-steps/s**（≈14× 单 CPU 环境；
纯物理 rollout 12,189 substeps/s）。2000 万步 ≈ 5 小时。首次编译约 1–2 分钟。

## 三、训练前自检（可选但推荐）

```bash
.venv/bin/python musclemimic/badminton/scripts/run_incoming_shuttle_hit.py --stage preflight
.venv/bin/python musclemimic/badminton/scripts/run_incoming_shuttle_hit.py --stage feed-check
.venv/bin/python musclemimic/badminton/scripts/run_incoming_shuttle_hit.py --stage physics-smoke --record-video
```

## 四、物理保真度保证

GPU 物理与 CPU 参考实现（`badminton_physics.py`）逐公式对齐，测试兜底：

| 测试 | 内容 |
|------|------|
| `test_badminton_physics_mjx.py::test_aero_parity_with_numpy` | 气动力/力矩 vs numpy，1e-8 |
| `::test_stringbed_parity_with_numpy` | 弦床力 vs numpy，1e-6 |
| `::test_event_rebound_parity_with_numpy` | 事件回弹速度，1e-8 |
| `::test_batched_qfrc_formula_matches_mj_applyFT` | 批量力映射 vs mj_applyFT，1e-12 |
| `::test_flight_trajectory_parity_cpu_vs_mjx`（`RUN_MJX_TESTS=1`） | 全栈 150 substep 轨迹 <5cm |

场景 MJX 兼容化（椭球→球体裙撑、人体椭球碰撞位重分配）保持 CPU 行为等价：
旧/新场景零动作对拍轨迹差毫米级；场地地板箱顶面与地面 plane 共面（z=0）。

## 五、架构速览

```
shuttle_feeder.py            喂球弹道采样（离线，npz bank）
badminton_physics.py         CPU 参考物理（单环境，调试/验证用）
badminton_physics_mjx.py     JAX 移植：单环境版（对拍）+ 批量版 make_batched_substep_fn（训练）
incoming_shuttle_hit_env.py  CPU RL 环境（单环境，语义参考）
incoming_shuttle_hit_mjx_env.py  批量 GPU 环境（warp 半批量 / jax 全批量）
train_incoming_hit_mjx.py    JAX PPO（rollout+GAE+更新整体 jit）
run_incoming_shuttle_hit.py  统一 runner（preflight/feed-check/physics-smoke/train-tiny/train-gpu）
```

Warp 批量语义：接触池（naconmax 总预算）跨 world 共享不加批量维，其余字段带前导
world 维；`mjx.step` 直接调用（warp 核隐式 vmap），自定义力用 cdof 映射成 qfrc（与
`mj_applyFT` 数值一致）。8192 env 会在 warp graph 创建时 OOM（接触池过大），当前上限
建议 2048–4096。

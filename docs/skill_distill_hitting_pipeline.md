# Skill → Distill → Hitting Pipeline

三阶段方案的完整实现与运行手册：**分动作跟踪专家 → 蒸馏成一个基础模型 → 基础模型（无未来轨迹 + 球位置）拿拍学击球**。所有阶段都已跑通机械冒烟。

## 环境（每个 shell 一次）

```bash
cd /data3/yangfeiyang/WorkSpace/musclemimic
source configs/env.sh                          # 修 GPU CUDA 路径
export MM_CUDA_COMPAT_DIR="$(pwd)/.local/cuda-compat-12.4/compat"   # warp 需要
export MUSCLEMIMIC_GMR_CACHE_PATH="$(pwd)/datasets/_global/muscle_trajectory/skill_cache"
export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Stage 1 — 分动作跟踪专家

本地 `datasets/<action>/muscle_trajectory/optimized/*.npz` 已是合法 MyoFullBody 跟踪轨迹（无需 SMPL 重定向）。搬进 skill 缓存 + 生成专家配置，然后用主线 PPO 训练。

```bash
P=BadmintonMimic/skill_pipeline
# 1) 搬运本地轨迹 → skill 缓存（symlink），生成 train/val manifest
.venv/bin/python $P/stage_local_trajectories.py --action forehandClear_standard \
  --emit-manifest datasets/forehandClear_standard/manifests/skill_pipeline

# 2) 生成专家跟踪配置（--racket 让专家在带拍惯量下挥拍，与 Stage3 一致，推荐）
.venv/bin/python $P/generate_expert_config.py --action forehandClear_standard --num-envs 256

# 3) 主线 PPO 训练（GPU，长时；checkpoint 落 datasets/<action>/training/checkpoints/<hash>/）
CUDA_VISIBLE_DEVICES=0 .venv/bin/python fullbody/experiment.py \
  --config-name=config_specific_task/skill/conf_expert_forehandClear_standard \
  wandb.mode=disabled
```

关键：主线 `ImitationFactory` 解析 `rel_dataset_path` 时缓存命中就直接 load，所以 `MUSCLEMIMIC_GMR_CACHE_PATH` 指向 skill 缓存即可绕过缺失的 SMPL/HF 重定向。

## Stage 2 — 蒸馏成基础模型

teacher rollout → distill 分片（obs 已去未来轨迹、只留 motion phase）→ BC → **frozen base 导出**（`FrozenBodyPolicy` 布局 + `skill_manifest.json`）。

```bash
# 单动作：采集 train/val 分片
.venv/bin/python $P/run_skill_distill.py collect --action forehandClear_standard \
  --teacher-path datasets/forehandClear_standard/training/checkpoints/<hash>/checkpoint_<N> \
  --num-envs 64 --num-steps 4000 --split train
.venv/bin/python $P/run_skill_distill.py collect --action forehandClear_standard \
  --teacher-path <same> --num-envs 32 --num-steps 2000 --split val

# 蒸馏成 base（单动作：一个 skill，无 one-hot）
.venv/bin/python $P/run_skill_distill.py distill --actions forehandClear_standard \
  --schema-from datasets/forehandClear_standard/training/checkpoints/<hash>/checkpoint_<N> \
  --output-dir outputs/skill_pipeline/base_forehandClear --steps 50000
```

## Stage 3 — 基础模型 + 球位置学击球

frozen base 驱动身体（motion phase 由来球 time-to-intercept 合成），PPO 只学**残差**：
`a = clip(base(body_obs) + residual_scale · δ, -1, 1)`。球的位置进残差头的观测，不进 base 观测。

```bash
# GPU 残差训练（warp 后端）
.venv/bin/python musclemimic/badminton/scripts/run_incoming_shuttle_hit.py --stage train-gpu \
  --num-envs 2048 --rollout-steps 64 --total-env-steps 20000000 \
  --base-policy-artifact outputs/skill_pipeline/base_forehandClear \
  --residual-scale 0.3

# 评估（CPU 参考环境 + 录像）
MUJOCO_GL=egl .venv/bin/python musclemimic/badminton/scripts/run_incoming_shuttle_hit.py \
  --stage evaluate --episodes 8 --record-video
```

奖励里可加 `residual` 权重（`incoming_shuttle_hit_v1.yaml` 的 `reward:` 段）惩罚 δ 幅度，鼓励贴近专家挥拍。

## 多动作（skill-conditioned base）

多个专家的分片一起蒸馏，每个 skill 追加一个 one-hot，得到**一个 conditioned base**。Stage 3 用 `--base-skill` 选择动作。

```bash
# 每个动作各自 Stage1+collect 后：
.venv/bin/python $P/train_multi_skill_bc.py \
  --dataset forehandClear_standard=datasets/_global/distill/forehandClear_standard \
  --dataset smash=datasets/_global/distill/smash \
  --schema-from <teacher_ckpt> --output-dir outputs/skill_pipeline/base_multi --steps 50000

.venv/bin/python musclemimic/badminton/scripts/run_incoming_shuttle_hit.py --stage train-gpu \
  --base-policy-artifact outputs/skill_pipeline/base_multi --base-skill smash ...
```

## 一键机械冒烟（验证全链路连通，非训练质量）

```bash
.venv/bin/python $P/run_skill_distill.py full-check --action forehandClear_standard \
  --teacher-path datasets/forehandClear_standard/training/checkpoints/<hash>/checkpoint_<N>
```

## 组件清单

| 文件 | 作用 |
|------|------|
| `skill_pipeline/stage_local_trajectories.py` | 本地 optimized npz → skill 缓存 + manifest |
| `skill_pipeline/generate_expert_config.py` | 生成单动作专家跟踪配置（可选 --racket） |
| `skill_pipeline/train_multi_skill_bc.py` | 单/多动作 BC → frozen base 导出（多动作加 one-hot） |
| `skill_pipeline/run_skill_distill.py` | 编排器：stage-data/gen-config/collect/distill/full-check |
| `environment/overall_environment/src/base_swing_bridge.py` | frozen base 桥接：obs 适配 + 残差合成 + phase 合成 + skill 选择 + JAX 导出 |
| `incoming_shuttle_hit_env.py` / `_mjx_env.py` | 击球环境（CPU/GPU）残差模式接入 |

## 已知边界

- **数据依赖**：只能用本地已有 optimized 轨迹；新采动作需要修复 SMPL/GMR 重定向（`smpl_models/SMPLH_NEUTRAL.pkl` 缺失）。
- **skill_id 是数据集级**：主线 `multi_action`/`skill_id` 配置键在代码里没有消费者，所以多动作条件化在蒸馏时用 one-hot 追加到 obs 实现（离线蒸馏下等价），不改主线 env。
- **带拍一致性**：建议 Stage1 就用 `--racket`（拍 weld 到手），否则 base 的挥拍在 Stage3 带拍惯量下会变形，需残差补偿或额外 fine-tune。
- **策略质量需真训练**：本文档验证的是流水线连通性；专家收敛、蒸馏保真、击球学会都需要各自数小时的 GPU 训练。

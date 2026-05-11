# BadmintonMimic 实现思路

目标：利用 MuscleMimic 框架训练一个面向羽毛球动作的肌肉骨骼模仿学习策略。整体路线是：

```text
羽毛球视频
  -> WHAM 提取人体 SMPL/SMPL-H 参数
  -> 转成 MuscleMimic 可读取的 AMASS-style .npz
  -> 使用 MuscleMimic/GMR retarget 到 MyoFullBody
  -> 生成 retarget cache
  -> 用 PPO/JAX/MJX 进行高速模仿学习训练
  -> checkpoint 评估、视频可视化、指标分析
```

当前第一阶段建议只训练人体全身动作模仿，不直接建模球拍、羽毛球、击球接触和球路。等全身动作可以稳定模仿后，再扩展 object tracking、racket site、contact reward 或 task reward。

## 1. 使用的模型与任务选择

羽毛球动作包含下肢移动、躯干旋转、肩肘腕快速挥拍，因此优先使用 `MyoFullBody` / `MjxMyoFullBody`，而不是只使用 `MyoBimanualArm`。

推荐路线：

| 阶段 | 环境 | 目的 |
| --- | --- | --- |
| 原型验证 | `MjxMyoFullBody` + 少量动作 | 验证 WHAM 数据格式、retarget、训练入口是否能跑通 |
| 小规模训练 | `MjxMyoFullBody` + 10-50 条动作 | 验证羽毛球动作是否可被稳定 retarget 和模仿 |
| 扩展训练 | `MjxMyoFullBody` + 更大动作库 | 训练泛化策略 |
| 任务增强 | 自定义 site/object/reward | 加入球拍、击球点、动作类别或目标落点 |

## 2. WHAM 输出到 AMASS-style 数据

MuscleMimic 当前的 AMASS 读取路径在 `loco_mujoco/smpl/retargeting.py` 的 `load_amass_data()`。它期望从 `AMASS_PATH` 下找到相对路径对应的 `.npz` 文件，并读取以下字段：

| 字段 | 形状/类型 | 说明 |
| --- | --- | --- |
| `poses` | `[T, >=66]` | axis-angle pose。当前代码读取 `poses[:, :66]`，再补 6 维 0 |
| `trans` | `[T, 3]` | root translation |
| `betas` | `[10]` 或更多 | body shape 参数 |
| `gender` | string/array | 通常可设为 `"neutral"` |
| `mocap_framerate` 或 `mocap_frame_rate` | scalar | 帧率 |

因此需要写一个转换脚本，把 WHAM 输出保存为 AMASS-style `.npz`。建议路径结构如下：

```text
/data3/yangfeiyang/WorkSpace/musclemimic/data/badminton_wham_amass/
  badminton/
    train/
      clip_0001_poses.npz
      clip_0002_poses.npz
    val/
      clip_0101_poses.npz
```

对应的 `rel_dataset_path` 就写成：

```yaml
rel_dataset_path:
  - "badminton/train/clip_0001_poses"
  - "badminton/train/clip_0002_poses"
```

转换时需要注意：

1. 坐标系要统一。WHAM 的 world/root translation 和 AMASS/MuJoCo 的朝向、高度可能不同，第一版先保留 WHAM world 坐标，再通过可视化检查是否倒置、漂移或离地。
2. 帧率要明确。GMR 配置中通常会重采样到 `target_fps: 30`，但源文件仍应保存真实 WHAM FPS。
3. `poses` 至少要有 66 维。如果 WHAM 输出是 SMPL 的 24 joints、72 维 axis-angle，可以直接保存；如果是 rotation matrix 或 quaternion，需要先转 axis-angle。
4. `betas` 可以先使用 WHAM 每段估计的 shape；如果一段视频内 shape 逐帧变化，建议取时间平均。
5. `gender` 第一版设为 `neutral`，减少模型资源分支差异。
6. 对明显失败的 WHAM 片段要先过滤，例如人体翻转、root 高度跳变、关节抖动严重、多人混淆、遮挡导致的突然瞬移。

建议新增脚本：

```text
scripts/convert_wham_to_amass.py
```

脚本职责：

```text
读取 WHAM 结果
  -> 选取单人 track
  -> 转换 pose 为 axis-angle
  -> 整理 poses/trans/betas/gender/fps
  -> 可选平滑 root translation 和 pose
  -> 保存 AMASS-style .npz
  -> 生成 train/val 路径清单
```

## 3. 配置路径

运行前设置数据和 cache 路径：

```bash
cd /data3/yangfeiyang/WorkSpace/musclemimic

export MUSCLEMIMIC_AMASS_PATH=/data3/yangfeiyang/WorkSpace/musclemimic/data/badminton_wham_amass
export AMASS_PATH=$MUSCLEMIMIC_AMASS_PATH

export MUSCLEMIMIC_CONVERTED_AMASS_PATH=/data3/yangfeiyang/WorkSpace/musclemimic/caches/AMASS
export CONVERTED_AMASS_PATH=$MUSCLEMIMIC_CONVERTED_AMASS_PATH

export MUSCLEMIMIC_SMPL_MODEL_PATH=/data3/yangfeiyang/WorkSpace/musclemimic/smpl_models/smplh
export SMPL_MODEL_PATH=$MUSCLEMIMIC_SMPL_MODEL_PATH
```

也可以用项目提供的入口持久化 cache 路径：

```bash
uv run musclemimic-set-all-caches --path /data3/yangfeiyang/WorkSpace/musclemimic/caches/AMASS
```

## 4. Retarget 路线

当前项目支持两类 retarget：

| 方法 | 配置值 | 特点 |
| --- | --- | --- |
| SMPL optimization | `retargeting_method: smpl` | 使用项目内优化流程，依赖 `smpl` extra |
| GMR | `retargeting_method: gmr` | 更快，当前 README 推荐用于 MuscleMimic 数据链路，依赖 `gmr` extra |

羽毛球动作速度快、旋转大，第一版建议使用 GMR：

```yaml
retargeting_method: gmr
gmr_config:
  src_human: smplh
  target_fps: 30
  solver: daqp
  damping: 0.5
  offset_to_ground: false
  use_velocity_limit: false
  use_fitted_shape: true
  shape_fitting_iterations: 500
```

Retarget 会在构造 imitation 环境时自动触发，也可以先单独预生成 cache。对于自定义路径，最直接的方式是创建一个 badminton 专用 Hydra 配置，显式写 `rel_dataset_path`。这样不用修改 `loco_mujoco/smpl/const.py` 中的 dataset group。

预期 cache 输出：

```text
caches/AMASS/MyoFullBody/gmr/badminton/train/clip_0001_poses.npz
caches/AMASS/MyoFullBody/gmr/badminton/train/clip_0001_poses_analysis.npz
```

`*_analysis.npz` 中会保存 retarget 误差等分析数据。需要用它筛掉失败片段。

## 5. 新建 badminton 训练配置

建议新增：

```text
fullbody/config_specific_task/conf_fullbody_badminton_gmr.yaml
```

配置从 `conf_fullbody_gmr` 或 `conf_fullbody` 继承。第一版不要直接上全量数据，先用 3-5 条高质量动作跑通。

示例思路：

```yaml
# @package _global_

defaults:
  - /conf_fullbody_gmr
  - _self_

wandb:
  project: "musclemimic"
  mode: "online"
  tags: ["fullbody", "gmr", "badminton"]

experiment:
  env_params:
    env_name: MjxMyoFullBody
    num_envs: 512
    disable_fingers: true

  task_factory:
    params:
      amass_dataset_conf:
        dataset_group: null
        rel_dataset_path:
          - "badminton/train/clip_0001_poses"
          - "badminton/train/clip_0002_poses"
          - "badminton/train/clip_0003_poses"
        retargeting_method: gmr
        gmr_config:
          src_human: smplh
          target_fps: 30
          solver: daqp
          damping: 0.5
          offset_to_ground: false
          use_velocity_limit: false
          use_fitted_shape: true
          shape_fitting_iterations: 500

  validation:
    active: true
    amass_dataset_conf:
      dataset_group: null
      rel_dataset_path:
        - "badminton/val/clip_0101_poses"
      retargeting_method: gmr
      gmr_config:
        src_human: smplh
        target_fps: 30
        solver: daqp
        damping: 0.5
        offset_to_ground: false
        use_velocity_limit: false
        use_fitted_shape: true
        shape_fitting_iterations: 500
```

小规模原型建议：

```yaml
experiment:
  total_timesteps: 20480000
  env_params:
    num_envs: 256
  ppo_config:
    num_steps: 80
```

稳定后再增大：

```yaml
experiment:
  total_timesteps: 2048000000
  env_params:
    num_envs: 4096 或 8192
  ppo_config:
    num_steps: 20
```

## 6. 运行命令

安装依赖：

```bash
uv sync --extra cuda --extra smpl --extra gmr --extra dev
```

先跑 retarget/训练小样本：

```bash
uv run fullbody/experiment.py \
  --config-name=config_specific_task/conf_fullbody_badminton_gmr \
  wandb.mode=disabled
```

如果服务器 CUDA/driver 兼容性有问题，可以用仓库里的 wrapper：

```bash
scripts/run_with_cuda_compat.sh uv run fullbody/experiment.py \
  --config-name=config_specific_task/conf_fullbody_badminton_gmr \
  wandb.mode=disabled
```

训练后评估：

```bash
uv run fullbody/eval.py \
  --path checkpoints/<run_id>/checkpoint_<step> \
  --motion_path badminton/val/clip_0101_poses \
  --use_mujoco \
  --stochastic \
  --eval_seed 0 \
  --n_steps 1000
```

如果需要 viewer：

```bash
uv run mjpython fullbody/eval.py \
  --path checkpoints/<run_id>/checkpoint_<step> \
  --motion_path badminton/val/clip_0101_poses \
  --use_mujoco \
  --mujoco_viewer \
  --n_steps 1000
```

## 7. 数据质量检查

WHAM->retarget 是整个项目成败的关键。训练前必须做数据筛选。

建议对每条 clip 检查：

| 检查项 | 目的 |
| --- | --- |
| WHAM 重建视频叠加 | 确认人体 track、关节、root translation 基本正确 |
| root 高度曲线 | 排除漂浮、穿地、突然跳变 |
| root 速度/角速度 | 排除跟踪失败导致的瞬移 |
| 肩、肘、腕轨迹 | 羽毛球动作依赖上肢，重点看挥拍侧 |
| retarget `pos_error` | 筛掉 GMR 对不上 mimic site 的片段 |
| MuJoCo playback | 检查 retarget 后是否穿地、翻转、抖动 |
| 训练 early termination rate | 过高说明轨迹太难、retarget 有问题或 terminal 阈值太严 |

可以复用或扩展：

```text
musclemimic/utils/retarget/msk_metrics.py
musclemimic/utils/retarget/massive_compare.py
examples/retargeting/retarget_visualize.py
```

## 8. 训练策略

第一版目标不是追求最高性能，而是验证完整链路。

推荐训练顺序：

1. 单条 clip overfit：只训练一个清晰的挥拍动作，确认 reward 能上升、termination 降低。
2. 小集合训练：加入 3-5 条同类动作，例如正手高远球或杀球。
3. 分类扩展：按动作类别训练，例如 `serve`、`clear`、`smash`、`drop`、`footwork`。
4. 混合泛化：合并多类动作，加入 adaptive sampling。
5. 强化验证：held-out clip 评估，不只看训练动作。

配置层面建议：

| 问题 | 调整方向 |
| --- | --- |
| 频繁 early terminate | 放宽 `mean_site_deviation_threshold`、`root_deviation_threshold`，或先减少动作难度 |
| 上肢跟不上 | 增大上肢 mimic sites 的 reward 权重，或增加挥拍侧 site |
| 脚底穿地/漂浮 | 检查 WHAM root translation 和 GMR `offset_to_ground`，必要时预处理高度 |
| 训练不稳定 | 减小 learning rate、减少 init std、先用更少动作 |
| 泛化差 | 增加动作多样性，开启 adaptive sampling，分动作类别训练再混合 |

## 9. 羽毛球专项扩展

当基础全身模仿稳定后，可以逐步加任务结构。

### 9.1 球拍 site

如果要显式模仿球拍，需要在 MuJoCo XML 或环境中增加 racket 相关 site。第一版可以先用右手/左手 site 近似挥拍端，后续再加：

```text
racket_handle_site
racket_head_site
shuttle_contact_target_site
```

然后扩展：

```text
goal_params.sites_for_mimic
reward_params.sites_for_mimic
validation.rel_site_names
```

### 9.2 动作阶段信息

羽毛球动作有准备、引拍、挥拍、随挥、回位。当前 goal 已支持 `enable_motion_phase: true`。后续可以进一步加入动作类别 embedding 或 phase label。

### 9.3 任务 reward

基础 imitation reward 只保证动作像。若要“打球任务”成立，需要额外 reward：

| Reward | 目的 |
| --- | --- |
| racket head velocity | 鼓励击球瞬间拍头速度 |
| contact timing | 对齐击球帧 |
| target pose/site tracking | 强化关键帧姿态 |
| footwork stability | 降低下肢滑步和失衡 |
| energy/activation penalty | 控制肌肉激活代价 |

## 10. 当前最小可执行计划

1. 从 WHAM 选 3 条质量最高的羽毛球视频结果。
2. 写 `scripts/convert_wham_to_amass.py`，输出 AMASS-style `.npz`。
3. 设置 `MUSCLEMIMIC_AMASS_PATH` 指向转换后的数据根目录。
4. 新建 `fullbody/config_specific_task/conf_fullbody_badminton_gmr.yaml`。
5. 跑一次小规模训练，让 retarget 自动生成 cache。
6. 检查 `caches/AMASS/MyoFullBody/gmr/badminton/...` 和 `*_analysis.npz`。
7. 用 `fullbody/eval.py --use_mujoco` 回放 checkpoint。
8. 如果 retarget 质量不稳定，先修 WHAM 数据和转换脚本，不急着调 PPO。
9. 单条动作能 overfit 后，再扩展到多条动作。
10. 多条动作稳定后，再考虑球拍、击球点和任务 reward。

## 11. 风险点

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| WHAM 输出坐标系与 AMASS/GMR 不一致 | 人体倒置、转向错误、root 漂移 | 先做单 clip 可视化，必要时加坐标变换 |
| WHAM 只有 SMPL 而不是 SMPL-H | 手部细节不足 | 第一版 `disable_fingers: true`，只模仿身体和主要肢体 |
| 快速挥拍动作 blur/遮挡严重 | 上肢 retarget 失败 | 筛选高质量片段，必要时手工裁剪动作区间 |
| GMR 对高动态动作产生穿地/关节异常 | 训练 early termination 高 | 检查 `*_analysis.npz`，过滤失败片段或调 GMR 参数 |
| 数据量太少 | 策略只会记忆动作 | 先接受 overfit，链路跑通后再扩数据 |
| 数据量太杂 | PPO 初期难以收敛 | 按动作类别 curriculum，从单类到混合 |

## 12. 需要最终落地的文件

建议后续新增或维护：

```text
scripts/convert_wham_to_amass.py
fullbody/config_specific_task/conf_fullbody_badminton_gmr.yaml
data/badminton_wham_amass/badminton/train/*.npz
data/badminton_wham_amass/badminton/val/*.npz
doc/badmintonmimic.md
```

其中 `data/` 下的大文件不建议直接提交到 Git。可以只提交转换脚本、配置文件和路径清单，例如：

```text
data/badminton_wham_amass/train_list.txt
data/badminton_wham_amass/val_list.txt
```

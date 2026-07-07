# AblationForOptimize 流程命令

目标：把最优的 SMPL 时序文件：

```text
musclemimic/badminton/data/output/ablation/_ablation/5_1_-2_43e0a6ee/04_lower_body_full/corrected_smpl.pkl
```

转换成 MuscleMimic 可训练的数据链路：

```text
SMPL pkl
  -> AMASS-style npz
  -> MyoFullBody GMR retarget cache
  -> fullbody imitation training
```

所有命令都在仓库根目录执行：

```bash
cd /data3/yangfeiyang/WorkSpace/musclemimic
```

## 1. 生成 AMASS-Style npz

作用：把 SMPL/WHAM 风格的 `.pkl` 时序文件转换成 MuscleMimic AMASS loader 能读取的 `.npz`。

```bash
.venv/bin/python musclemimic/badminton/scripts/convert_wham_to_amass.py \
  --input musclemimic/badminton/data/output/ablation/_ablation/5_1_-2_43e0a6ee/04_lower_body_full/corrected_smpl.pkl \
  --output musclemimic/badminton/data/amass_npz/ablation/04_lower_body_full_poses.npz \
  --fps 30 \
  --gender neutral
```

生成文件：

```text
musclemimic/badminton/data/amass_npz/ablation/04_lower_body_full_poses.npz
```

已确认字段：

```text
poses: (202, 72) float32
trans: (202, 3) float32
betas: (10,) float32
fps: 30.0
gender: neutral
```

## 2. 生成 Retarget Manifest

作用：告诉 retarget 脚本要处理哪一个 AMASS-style 动作。

```bash
printf 'ablation/04_lower_body_full_poses\n' > /tmp/ablation_04_lower_body_full_manifest.txt
```

这里的路径是相对于：

```text
musclemimic/badminton/data/amass_npz
```

所以不要写 `.npz` 后缀。

## 3. Retarget 到 MyoFullBody

作用：把 SMPL/AMASS 动作重定向成 MyoFullBody 肌骨模型的 imitation trajectory。

```bash
.venv/bin/python musclemimic/badminton/scripts/run_retarget.py \
  --split train \
  --manifest /tmp/ablation_04_lower_body_full_manifest.txt \
  --target-fps 30
```

输出文件：

```text
caches/AMASS/MyoFullBody/gmr/ablation/04_lower_body_full_poses.npz
caches/AMASS/MyoFullBody/gmr/ablation/04_lower_body_full_poses_analysis.npz
```

已确认 retarget 结果：

```text
qpos:       (667, 89) float32
qvel:       (667, 88) float32
site_xpos:  (667, 17, 3) float32
frequency:  100.0
split:      [0, 667]
pos_error mean/max: 0.0159 / 0.0782
```

## 4. 训练配置文件

新建的单动作训练配置：

```text
fullbody/config_specific_task/Ablation/conf_fullbody_ablation_04_lower_body_full_gmr.yaml
```

这个配置只使用一个参考动作：

```text
ablation/04_lower_body_full_poses
```

训练和验证都用这一个 motion，适合先测试这条最优轨迹能不能被 MuscleMimic 学起来。

## 5. Baseline PPO 训练

作用：先跑原始 mimic PPO，不打开 ASI/curriculum，用来判断 retarget 数据本身是否可学。

```bash
wandb.mode=disabled \
.venv/bin/python fullbody/experiment.py \
  --config-name=config_specific_task/Ablation/conf_fullbody_ablation_04_lower_body_full_gmr
```

默认状态：

```text
ASI: disabled
adaptive termination curriculum: disabled
reward curriculum: disabled
```

## 6. ASI + Curriculum 训练

作用：在同一条 retarget 轨迹上打开增强训练策略。

```bash
MM_CUDA_VISIBLE_DEVICES=2 \
scripts/run_with_cuda_compat.sh \
.venv/bin/python fullbody/experiment.py \
  --config-name=config_specific_task/Ablation/conf_fullbody_10demo \
  wandb.mode=online \
  experiment.asi.enabled=true \
  experiment.adaptive_termination.enabled=true \
  experiment.reward_curriculum.enabled=true
```

三个开关含义：

```text
experiment.asi.enabled=true
  自适应选择 trajectory 的起始帧。

experiment.adaptive_termination.enabled=true
  根据 early termination rate 调整终止阈值。

experiment.reward_curriculum.enabled=true
  策略稳定后逐步提高速度跟踪 reward 权重。
```

## 7. 推荐实验顺序

1. 先跑 Baseline PPO。
2. 如果 baseline 能学，说明 retarget 轨迹基本可用。
3. 再跑 ASI + curriculum，看是否提升稳定性和收敛速度。
4. 如果这条 motion 效果好，再把更多 ablation 或 badminton motion 加进 `rel_dataset_path`。

# BadmintonMimic

BadmintonMimic 是基于仓库根目录 MuscleMimic 框架的羽毛球动作模仿学习研究项目。目标流程：

```text
badminton video
  -> WHAM SMPL/SMPL-H parameters
  -> AMASS-style .npz
  -> MuscleMimic GMR retarget
  -> MyoFullBody imitation training
  -> checkpoint evaluation and visualization
```

第一阶段只做全身动作模仿，不直接建模球拍、羽毛球、击球接触和落点。基础链路稳定后，再增加 racket site、contact timing、task reward。

## Directory Layout

```text
BadmintonMimic/
  configs/                         # 环境变量和路径配置
  data/
    wham_raw/                       # WHAM 原始输出，通常不提交
    amass_npz/badminton/train/      # 转换后的训练 .npz
    amass_npz/badminton/val/        # 转换后的验证 .npz
    retarget_cache/                 # 可选的项目级 retarget cache
  docs/                             # 研究流程和记录
  experiments/fullbody/config_specific_task/
                                   # MuscleMimic Hydra 配置模板
  manifests/                        # train/val 动作清单
  scripts/                          # 转换、retarget、训练、评估脚本
  tools/                            # 后续分析/可视化工具
  logs/                             # 本项目运行日志
  outputs/                          # 本项目临时输出
```

## Minimal Workflow

1. 准备 WHAM 输出到 `data/wham_raw/`。
2. 把 WHAM 输出转换成 AMASS-style `.npz`：

```bash
python BadmintonMimic/scripts/convert_wham_to_amass.py \
  --input BadmintonMimic/data/wham_raw/example_wham.npz \
  --output BadmintonMimic/data/amass_npz/badminton/train/clip_0001_poses.npz \
  --fps 30 \
  --gender neutral
```

3. 在 `manifests/train_list.txt` 和 `manifests/val_list.txt` 写入相对路径，例如：

```text
badminton/train/clip_0001_poses
badminton/val/clip_0101_poses
```

4. 加载环境变量：

```bash
source BadmintonMimic/configs/env.sh
```

5. 安装 Hydra 配置到 MuscleMimic 默认搜索路径：

```bash
bash BadmintonMimic/scripts/install_configs.sh
```

6. 预生成 retarget cache：

```bash
python BadmintonMimic/scripts/run_retarget.py --split train
python BadmintonMimic/scripts/run_retarget.py --split val
```

7. 小规模训练：

```bash
bash BadmintonMimic/scripts/train_fullbody.sh
```

8. 评估 checkpoint：

```bash
bash BadmintonMimic/scripts/eval_fullbody.sh checkpoints/<run_id>/checkpoint_<step> badminton/val/clip_0101_poses
```

## Data Contract

转换后的 `.npz` 需要包含 MuscleMimic 当前 AMASS loader 会读取的字段：

| key | expected value |
| --- | --- |
| `poses` | shape `[T, >=66]`, axis-angle |
| `trans` | shape `[T, 3]`, root translation |
| `betas` | shape `[10]` or compatible |
| `gender` | usually `neutral` |
| `mocap_framerate` | scalar fps |

## Notes

WHAM 输出格式可能随版本不同而不同。`scripts/convert_wham_to_amass.py` 已兼容常见 key，但如果你的 WHAM 文件字段不同，应先用 `python -c "import numpy as np; print(np.load(path, allow_pickle=True).files)"` 检查，再补充 key mapping。

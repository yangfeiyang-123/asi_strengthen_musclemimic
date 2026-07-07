# BadmintonMimic Workflow

## 1. WHAM Stage

输入是羽毛球视频，输出是每段 clip 的 SMPL/SMPL-H 参数。建议先人工筛选高质量视频片段：

| 检查项 | 标准 |
| --- | --- |
| 人体完整性 | 挥拍侧肩、肘、腕尽量可见 |
| root 稳定性 | 没有明显瞬移、翻转、漂移 |
| 动作完整性 | 包含准备、引拍、挥拍、随挥、回位 |
| 单人跟踪 | 避免多人混淆 |

## 2. Conversion Stage

把 WHAM 输出转为 AMASS-style `.npz`。输出放在：

```text
musclemimic/badminton/data/amass_npz/badminton/train/
musclemimic/badminton/data/amass_npz/badminton/val/
```

文件名建议使用：

```text
clip_0001_poses.npz
clip_0002_poses.npz
```

manifest 中不要写 `.npz` 后缀：

```text
badminton/train/clip_0001_poses
```

## 3. Retarget Stage

使用 GMR retarget 到 `MyoFullBody`。cache 默认写到仓库级目录：

```text
caches/AMASS/MyoFullBody/gmr/badminton/...
```

每个 motion 会生成：

```text
*_poses.npz
*_poses_analysis.npz
```

重点检查 `*_analysis.npz` 里的误差，失败片段不要进入训练集。

## 4. Training Stage

第一步只用少量动作验证 overfit：

```text
3-5 clips, 256 envs, short training
```

稳定后再扩大到：

```text
10-50 clips, 512-2048 envs
```

最后再考虑：

```text
multi-action dataset, 4096-8192 envs, adaptive sampling
```

## 5. Evaluation Stage

至少评估三类 motion：

| split | 目的 |
| --- | --- |
| train seen | 检查是否学会基本 imitation |
| val same action | 检查同类动作泛化 |
| val different action | 检查跨动作泛化 |

## 6. Extension Stage

基础 imitation 稳定后，再增加羽毛球专项结构：

| 扩展 | 目的 |
| --- | --- |
| racket handle/head site | 让策略显式跟踪球拍 |
| contact phase label | 对齐击球时刻 |
| racket velocity reward | 鼓励击球速度 |
| footwork reward | 降低滑步、失衡 |
| action embedding | 区分发球、杀球、高远球、步法 |

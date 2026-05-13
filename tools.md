# Tools

## Render Retargeted MyoFullBody Trajectory

作用：把已经 retarget 到 MyoFullBody 的 GMR cache 轨迹渲染成 MuJoCo 肌骨模型视频。

当前使用的 retarget cache：

```text
caches/AMASS/MyoFullBody/gmr/ablation/04_lower_body_full_poses.npz
```

生成的视频：

```text
Experiments/AblationForOptimize/vis/ablation_04_lower_body_full_poses.mp4
```

命令：

```bash
cd /data3/yangfeiyang/WorkSpace/musclemimic

JAX_PLATFORMS=cpu MUJOCO_GL=egl MPLCONFIGDIR=/tmp/matplotlib \
.venv/bin/python BadmintonMimic/scripts/render_retarget_cache.py \
  --motion ablation/04_lower_body_full_poses \
  --output-dir Experiments/AblationForOptimize/vis \
  --width 640 \
  --height 480 \
  --stride 4 \
  --format mp4
```

参数含义：

```text
JAX_PLATFORMS=cpu
  渲染只需要 MuJoCo，不需要 JAX GPU，避免 CUDA 初始化干扰。

MUJOCO_GL=egl
  使用 EGL 离屏渲染。

MPLCONFIGDIR=/tmp/matplotlib
  避免 matplotlib/fontconfig 写 home 目录失败。

--motion ablation/04_lower_body_full_poses
  轨迹名，不带 .npz 后缀。实际读取：
  caches/AMASS/MyoFullBody/gmr/ablation/04_lower_body_full_poses.npz

--output-dir Experiments/AblationForOptimize/vis
  视频输出目录。

--width 640 --height 480
  渲染分辨率。当前模型默认 offscreen framebuffer 最大宽度是 640，
  所以不要直接用 960 宽度，除非修改模型 XML 的 offwidth。

--stride 4
  每 4 帧渲染 1 帧。脚本默认从 cache 的 frequency 自动计算输出 fps：
  output_fps = cache_frequency / stride，所以默认保持真实时间。

--fps
  可选。手动指定输出视频帧率；如果指定值和 cache_frequency / stride 不一致，
  视频时间轴会被改变。一般预览时可以不传。

--format mp4
  输出 mp4 视频。
```

成功输出示例：

```text
[OK] caches/AMASS/MyoFullBody/gmr/ablation/04_lower_body_full_poses.npz
  -> Experiments/AblationForOptimize/vis/ablation_04_lower_body_full_poses.mp4
  (167 frames)
```

## WHAM pkl Retarget to MyoFullBody

作用：把当前 WHAM 版本导出的 `.pkl` 文件先转成 AMASS-style `.npz`，再用 GMR 重定向到肌骨模型 `MyoFullBody`。

对应脚本：

```text
BadmintonMimic/scripts/convert_wham_to_amass.py
  WHAM .pkl/.npz/.npy -> AMASS-style .npz

BadmintonMimic/scripts/run_retarget.py
  AMASS-style .npz manifest -> caches/AMASS/MyoFullBody/gmr/*.npz

BadmintonMimic/scripts/build_config_from_manifests.py
  manifest -> BadmintonMimic 训练配置，并同步 gmr_config.target_fps

BadmintonMimic/scripts/render_retarget_cache.py
  retarget cache -> 真实时间预览视频
```

### 统一帧率工作流

原则：用同一个 `FPS` 变量贯穿 WHAM 转换、GMR retarget、训练配置生成。当前所有原始信号是 30Hz 时：

```bash
FPS=30
```

如果后续换成 60Hz 视频/信号，只改成：

```bash
FPS=60
```

然后下面所有命令都继续使用 `${FPS}`。

### 1. WHAM pkl 转 AMASS-style npz

当前仓库里可用的 WHAM pkl 示例：

```text
BadmintonMimic/dataset/forehand_clear/stage5_lower_body_full/video1_lower_body_full.pkl
BadmintonMimic/dataset/forehand_clear/stage5_lower_body_full/video2_lower_body_full.pkl
BadmintonMimic/dataset/forehand_clear/stage5_lower_body_full/video3_lower_body_full.pkl
BadmintonMimic/dataset/forehand_clear/video1/wham_output.pkl
BadmintonMimic/data/output/ablation/5月1日-2/wham_output.pkl
```

单个已合并/已处理的 WHAM pkl：

```bash
cd /data3/yangfeiyang/WorkSpace/musclemimic

.venv/bin/python BadmintonMimic/scripts/convert_wham_to_amass.py \
  --input BadmintonMimic/dataset/forehand_clear/stage5_lower_body_full/video1_lower_body_full.pkl \
  --output BadmintonMimic/data/amass_npz/badminton/train/forehand_clear_clip1_merged_poses.npz \
  --fps ${FPS} \
  --force-fps \
  --gender neutral
```

原始多 track `wham_output.pkl` 可直接合并：

```bash
.venv/bin/python BadmintonMimic/scripts/convert_wham_to_amass.py \
  --input BadmintonMimic/dataset/forehand_clear/video1/wham_output.pkl \
  --output BadmintonMimic/data/amass_npz/badminton/train/forehand_clear_clip1_merged_poses.npz \
  --fps ${FPS} \
  --force-fps \
  --gender neutral \
  --merge-tracks
```

转换脚本默认使用 `pose_world/trans_world`，并做 WHAM Y-up 到 AMASS/MuJoCo Z-up 的坐标轴对齐；不要加 `--local` 或 `--no-up-align`，除非明确要调试局部相机坐标或跳过坐标对齐。

`--force-fps` 是推荐项：无论 WHAM pkl 内部有没有 `fps/frame_rate` metadata，都强制写入命令行传入的 `${FPS}`。

### 2. 写 manifest

把生成的 `.npz` 加入 manifest，路径不带 `.npz` 后缀：

```text
BadmintonMimic/manifests/train_list.txt

badminton/train/forehand_clear_clip1_merged_poses
```

当前 train manifest 已包含：

```text
badminton/train/forehand_clear_clip1_merged_poses
badminton/train/forehand_clear_clip2_merged_poses
badminton/train/forehand_clear_clip3_merged_poses
```

### 3. GMR 重定向到 MyoFullBody

```bash
cd /data3/yangfeiyang/WorkSpace/musclemimic

JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES="" \
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/matplotlib \
.venv/bin/python BadmintonMimic/scripts/run_retarget.py \
  --split train \
  --fps ${FPS}
```

如果要强制覆盖已有 cache，加：

```bash
--clear-cache
```

`run_retarget.py` 会先检查 manifest 里每个 `.npz` 的 `mocap_framerate/mocap_frame_rate` 必须等于 `${FPS}`。如果不一致会直接报错，例如：

```text
fps=60; expected 30. Reconvert with --fps 30 --force-fps.
```

生成的肌骨 retarget cache：

```text
caches/AMASS/MyoFullBody/gmr/badminton/train/forehand_clear_clip1_merged_poses.npz
caches/AMASS/MyoFullBody/gmr/badminton/train/forehand_clear_clip1_merged_poses_analysis.npz
```

### 4. 生成/更新训练配置

训练配置里的 `gmr_config.target_fps` 也要和 `${FPS}` 一致：

```bash
.venv/bin/python BadmintonMimic/scripts/build_config_from_manifests.py \
  --fps ${FPS}
```

默认输出：

```text
BadmintonMimic/experiments/fullbody/config_specific_task/conf_fullbody_badminton_gmr.yaml
```

### 5. 渲染真实时间预览

```bash
.venv/bin/python BadmintonMimic/scripts/render_retarget_cache.py \
  --motion badminton/train/forehand_clear_clip1_merged_poses \
  --stride 1 \
  --format mp4
```

不传 `--fps` 时，脚本会用 cache 的 `frequency / stride` 作为视频 fps，默认保持真实时间。只有想故意改变播放速度时才手动传 `--fps`。

### Stage5 10-demo 完整例子

目标：只针对

```text
BadmintonMimic/dataset/forehand_clear/stage5_lower_body_full
```

里的 10 个 stage5 WHAM pkl 做 MyoFullBody GMR retarget。

输入 pkl：

```text
BadmintonMimic/dataset/forehand_clear/stage5_lower_body_full/video1_lower_body_full.pkl
BadmintonMimic/dataset/forehand_clear/stage5_lower_body_full/video2_lower_body_full.pkl
BadmintonMimic/dataset/forehand_clear/stage5_lower_body_full/video3_lower_body_full.pkl
BadmintonMimic/dataset/forehand_clear/stage5_lower_body_full/video4_lower_body_full.pkl
BadmintonMimic/dataset/forehand_clear/stage5_lower_body_full/video5_lower_body_full.pkl
BadmintonMimic/dataset/forehand_clear/stage5_lower_body_full/video6_lower_body_full.pkl
BadmintonMimic/dataset/forehand_clear/stage5_lower_body_full/video7_lower_body_full.pkl
BadmintonMimic/dataset/forehand_clear/stage5_lower_body_full/video8_lower_body_full.pkl
BadmintonMimic/dataset/forehand_clear/stage5_lower_body_full/video9_lower_body_full.pkl
BadmintonMimic/dataset/forehand_clear/stage5_lower_body_full/video10_lower_body_full.pkl
```

第 1 步：转换为 AMASS-style `.npz`。输出到：

```text
BadmintonMimic/data/amass_npz/forehand_clear/stage5_10demo/video*_lower_body_full_poses.npz
```

命令：

```bash
cd /data3/yangfeiyang/WorkSpace/musclemimic

FPS=30
for i in 1 2 3 4 5 6 7 8 9 10
do
  .venv/bin/python BadmintonMimic/scripts/convert_wham_to_amass.py \
    --input BadmintonMimic/dataset/forehand_clear/stage5_lower_body_full/video${i}_lower_body_full.pkl \
    --output BadmintonMimic/data/amass_npz/forehand_clear/stage5_10demo/video${i}_lower_body_full_poses.npz \
    --fps ${FPS} \
    --force-fps \
    --gender neutral
done
```

第 2 步：manifest。已写入：

```text
BadmintonMimic/manifests/stage5_10demo_list.txt
```

内容是：

```text
forehand_clear/stage5_10demo/video1_lower_body_full_poses
forehand_clear/stage5_10demo/video2_lower_body_full_poses
forehand_clear/stage5_10demo/video3_lower_body_full_poses
forehand_clear/stage5_10demo/video4_lower_body_full_poses
forehand_clear/stage5_10demo/video5_lower_body_full_poses
forehand_clear/stage5_10demo/video6_lower_body_full_poses
forehand_clear/stage5_10demo/video7_lower_body_full_poses
forehand_clear/stage5_10demo/video8_lower_body_full_poses
forehand_clear/stage5_10demo/video9_lower_body_full_poses
forehand_clear/stage5_10demo/video10_lower_body_full_poses
```

第 3 步：GMR retarget。输出到：

```text
caches/AMASS/MyoFullBody/gmr/forehand_clear/stage5_10demo/video*_lower_body_full_poses.npz
caches/AMASS/MyoFullBody/gmr/forehand_clear/stage5_10demo/video*_lower_body_full_poses_analysis.npz
```

命令：

```bash
cd /data3/yangfeiyang/WorkSpace/musclemimic

JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES="" \
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/matplotlib \
.venv/bin/python BadmintonMimic/scripts/run_retarget.py \
  --manifest BadmintonMimic/manifests/stage5_10demo_list.txt \
  --fps ${FPS}
```

如果需要强制重算已有 cache：

```bash
.venv/bin/python BadmintonMimic/scripts/run_retarget.py \
  --manifest BadmintonMimic/manifests/stage5_10demo_list.txt \
  --fps ${FPS} \
  --clear-cache
```

第 4 步：训练配置。已写入：

```text
fullbody/config_specific_task/Ablation/conf_fullbody_10demo.yaml
```

该配置使用 10 条 motion：

```text
forehand_clear/stage5_10demo/video1_lower_body_full_poses
...
forehand_clear/stage5_10demo/video10_lower_body_full_poses
```

并设置：

```text
gmr_config.target_fps: 30
```

第 5 步：可视化其中一条 cache：

```bash
.venv/bin/python BadmintonMimic/scripts/render_retarget_cache.py \
  --motion forehand_clear/stage5_10demo/video1_lower_body_full_poses \
  --output-dir BadmintonMimic/outputs/vis/stage5_10demo \
  --width 640 \
  --height 480 \
  --stride 4 \
  --format mp4
```

如果要和原始 30Hz 视频对比，推荐输出 30fps 对比版，而不是 100fps：

```bash
.venv/bin/python BadmintonMimic/scripts/render_retarget_cache.py \
  --motion forehand_clear/stage5_10demo/video1_lower_body_full_poses \
  --output-dir BadmintonMimic/outputs/vis/stage5_10demo_30fps \
  --width 640 \
  --height 480 \
  --stride 1 \
  --sample-fps 30 \
  --format mp4
```

`--sample-fps 30` 会按真实时间从 100Hz cache 采样成 30fps 视频；输出时长不变，播放器兼容性更好。

当前已生成并校验：

```text
AMASS npz: 10 个，mocap_framerate=30, mocap_frame_rate=30
GMR cache: 10 个，frequency=100Hz，均有 *_analysis.npz
30fps 对比视频:
  BadmintonMimic/outputs/vis/stage5_10demo_30fps/forehand_clear_stage5_10demo_video1_lower_body_full_poses.mp4
```

### 帧率注意事项

```text
WHAM pkl -> AMASS-style npz:
  --fps ${FPS} --force-fps 写入 mocap_framerate 和 mocap_frame_rate。

GMR retarget:
  --fps ${FPS} 等价于 --target-fps ${FPS}。
  输入 npz metadata 不等于 ${FPS} 会直接报错。

PPO cache:
  extend_motion() 会把 GMR 结果重采样到环境控制频率 1 / env.dt。
  当前 MyoFullBody 默认 timestep=0.002, n_substeps=5，所以 PPO cache
  frequency 通常是 100Hz。这个重采样保持真实时间，不代表原始信号变成 100Hz。
  PPO 训练读取的是 cache frequency，因此训练端和 cache 控制步长保持一致。

30Hz -> 60Hz 切换：
  只需要把 FPS=30 改成 FPS=60，并重新执行转换、retarget、配置生成。
  不要只改其中一个阶段。
```

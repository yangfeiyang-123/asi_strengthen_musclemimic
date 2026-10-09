#!/usr/bin/env python3
"""从 T3@320M 验证视频提取关键帧，拼三阶段运动过程胶片图。

视频来源：训练引擎在 320M 验证时录制的 traj0（近全程轨迹，172/173 帧）回放。
输出：figures/motion_filmstrip_t3_320m.png
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
VIDEO = (
    REPO
    / "datasets/forehandClear_standard/training_aug100_40train10val/validation_videos"
    / "validation_30_traj0_t320000000_20260815_094815/MyoFullBody.mp4"
)
ROLLOUT_NPZ = HERE / "t3_320m_activation_rollout.npz"
OUT = HERE / "figures/motion_filmstrip_t3_320m.png"

STAGE_BOUNDS = (0.0, 0.45, 0.65, 1.0)
STAGE_NAMES = ["阶段一·准备引拍", "阶段二·挥拍击球", "阶段三·随挥恢复"]
# 7 个关键进度点：起始 / 阶段一中期 / S3→S1 交接 / 击球高峰 / S1→S2 交接 / 阶段三中期 / 结束
PROGRESS_POINTS = [0.0, 0.225, 0.45, 0.55, 0.65, 0.80, 1.0]

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> int:
    cap = cv2.VideoCapture(str(VIDEO))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    n_video = len(frames)
    # 视频帧率 == env 控制频率（100Hz，1 帧/步）；traj0 的 episode 长度来自采集 npz，
    # 视频尾部超出 episode 的帧（reset 后继续录制）不使用。
    roll = np.load(ROLLOUT_NPZ, allow_pickle=True)
    ep_len = int(roll["episode_lengths"][list(roll["trajectory_indices"]).index(0)])
    print(f"video: {n_video} frames @ {fps:.1f} fps; traj0 episode = {ep_len} steps")

    picks = [min(int(round(p * (ep_len - 1))), n_video - 1) for p in PROGRESS_POINTS]
    fig, axes = plt.subplots(1, len(picks), figsize=(2.6 * len(picks), 4.2))
    for ax, p, fi in zip(axes, PROGRESS_POINTS, picks):
        ax.imshow(frames[fi])
        ax.set_title(f"{p:.2f}\nframe {fi}", fontsize=9)
        ax.axis("off")
    # 阶段着色标注
    for si, (a, b) in enumerate(zip(STAGE_BOUNDS[:-1], STAGE_BOUNDS[1:])):
        covered = [i for i, p in enumerate(PROGRESS_POINTS) if a <= p < b or (si == 2 and p == 1.0)]
        if not covered:
            continue
        x0 = covered[0] / len(picks)
        x1 = (covered[-1] + 1) / len(picks)
        fig.text((x0 + x1) / 2, 0.02, STAGE_NAMES[si], ha="center", fontsize=11, color="dimgray")
    fig.suptitle("T3@320M 策略执行 forehand clear 的运动全过程（traj0 验证回放关键帧）", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

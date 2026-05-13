#!/usr/bin/env python3
"""Analyze exported MuscleMimic rollout data.

Input is the NPZ produced by:
  uv run python fullbody/eval.py --path CHECKPOINT --use_mujoco --no_render \
    --export_trajectory --trajectory_dir trajectory_data ...

The script ranks and plots:
  - muscle activations and muscle commands
  - joint position/velocity/acceleration statistics
  - policy-vs-reference qpos tracking error, when reference qpos is available
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
from matplotlib import gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_JOINT_PATTERN = (
    "pelvis|lumbar|shoulder|elbow|wrist|radioulnar|hip|knee|ankle|subtalar|mtp"
)
DEFAULT_MUSCLE_PATTERN = (
    "delt|supsp|infsp|subsc|pecm|lat|tri|bic|bra|brd|ecr|ecu|fcr|fcu|fds|fdp|edc|"
    "epl|epb|fpl|apl|rect|iliacus|psoas|add|bflh|bfsh|gas|glmax|glmed|glmin|"
    "recfem|semimem|semiten|soleus|tfl|tib|vas|eo|io|ql|mf"
)


def _as_str_list(value: np.ndarray | Iterable[str]) -> list[str]:
    arr = np.asarray(value)
    return [str(x) for x in arr.tolist()]


def _load_model_names() -> tuple[list[str], list[str], dict[str, tuple[int, int]]]:
    """Return actuator names, joint names, and qpos/qvel indices for scalar joints."""
    try:
        import mujoco
        from musclemimic.utils.retarget.msk_metrics import load_model
    except Exception as exc:  # pragma: no cover - best effort fallback
        print(f"Warning: could not load MuJoCo model metadata: {exc}")
        return [], [], {}

    model = load_model()
    actuator_names = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        actuator_names.append(name or f"actuator_{i:03d}")

    joint_names = []
    scalar_joint_indices: dict[str, tuple[int, int]] = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i:03d}"
        joint_names.append(name)
        joint_type = model.jnt_type[i]
        if joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            scalar_joint_indices[name] = (int(model.jnt_qposadr[i]), int(model.jnt_dofadr[i]))

    return actuator_names, joint_names, scalar_joint_indices


def _episode_ids(npz: np.lib.npyio.NpzFile) -> list[int]:
    ids = []
    for key in npz.files:
        m = re.fullmatch(r"episode_(\d+)_joint_positions", key)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def _select_indices(names: list[str], scores: np.ndarray, pattern: str | None, top_n: int) -> np.ndarray:
    if len(scores) == 0:
        return np.array([], dtype=int)

    mask = np.ones(len(scores), dtype=bool)
    if pattern:
        rx = re.compile(pattern, flags=re.IGNORECASE)
        mask = np.array([bool(rx.search(name)) for name in names[: len(scores)]], dtype=bool)
        if not np.any(mask):
            print(f"Warning: pattern matched no names, falling back to all entries: {pattern}")
            mask = np.ones(len(scores), dtype=bool)

    candidate = np.where(mask)[0]
    order = candidate[np.argsort(scores[candidate])[::-1]]
    return order[: min(top_n, len(order))]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "unnamed"


def _resolve_muscle_names(
    npz: np.lib.npyio.NpzFile,
    fallback_names: list[str],
    n_muscles: int,
    episode: int | None = None,
) -> list[str]:
    """Prefer names exported with the rollout, falling back to model metadata or indices."""
    candidate_keys = []
    if episode is not None:
        candidate_keys.append(f"episode_{episode}_muscle_names")
    candidate_keys.append("muscle_names")

    for key in candidate_keys:
        if key not in npz.files:
            continue
        names = _as_str_list(npz[key])
        if len(names) == n_muscles:
            return names
        print(f"Warning: {key} has {len(names)} names but rollout has {n_muscles} muscles; ignoring exported names")

    if fallback_names and len(fallback_names) == n_muscles:
        return fallback_names
    return [f"muscle_{i:03d}" for i in range(n_muscles)]


def _plot_heatmap(path: Path, data: np.ndarray, ylabels: list[str], title: str, cbar_label: str) -> None:
    if data.size == 0:
        return
    fig_h = max(4.0, min(14.0, 0.32 * len(ylabels) + 1.8))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    im = ax.imshow(data.T, aspect="auto", interpolation="nearest", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Signal")
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_profiles(path: Path, time: np.ndarray, series: np.ndarray, labels: list[str], title: str, ylabel: str) -> None:
    if series.size == 0:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, label in enumerate(labels):
        ax.plot(time[: series.shape[0]], series[:, i], linewidth=1.4, label=label)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _activation_snapshot_grid(activations: np.ndarray, snapshot_step: int) -> np.ndarray:
    """Arrange one frame of all muscle activations into a compact near-square grid."""
    if activations.ndim != 2 or activations.shape[0] == 0 or activations.shape[1] == 0:
        return np.empty((0, 0), dtype=float)

    step = int(np.clip(snapshot_step, 0, activations.shape[0] - 1))
    values = np.asarray(activations[step], dtype=float)
    cols = int(np.ceil(np.sqrt(values.size)))
    rows = int(np.ceil(values.size / cols))
    grid = np.full(rows * cols, np.nan, dtype=float)
    grid[: values.size] = values
    return grid.reshape(rows, cols)


def _plot_activation_dynamics_panel(
    path: Path,
    time: np.ndarray,
    activations: np.ndarray,
    muscle_names: list[str],
    profile_indices: np.ndarray,
    snapshot_step: int,
    title: str,
) -> None:
    """Plot a one-frame activation map beside full-sequence muscle activation traces."""
    if activations.size == 0:
        return

    grid = _activation_snapshot_grid(activations, snapshot_step)
    if grid.size == 0:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    step = int(np.clip(snapshot_step, 0, activations.shape[0] - 1))
    x = time[: activations.shape[0]] if len(time) >= activations.shape[0] else np.arange(activations.shape[0])
    profile_indices = np.asarray(profile_indices, dtype=int)
    profile_indices = profile_indices[(profile_indices >= 0) & (profile_indices < activations.shape[1])]
    if profile_indices.size == 0:
        profile_indices = np.arange(min(8, activations.shape[1]))

    fig = plt.figure(figsize=(13.5, 5.2), facecolor="white")
    panel = FancyBboxPatch(
        (0.015, 0.035),
        0.97,
        0.91,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        transform=fig.transFigure,
        linewidth=0.0,
        facecolor="#d8d8d8",
        zorder=-10,
    )
    fig.add_artist(panel)
    outer = gridspec.GridSpec(
        2,
        3,
        figure=fig,
        height_ratios=[0.18, 0.82],
        width_ratios=[1.0, 0.12, 1.38],
        left=0.075,
        right=0.955,
        bottom=0.13,
        top=0.91,
        wspace=0.05,
        hspace=0.02,
    )

    title_ax = fig.add_subplot(outer[0, :])
    title_ax.axis("off")
    title_ax.text(0.5, 0.52, title, ha="center", va="center", fontsize=24, color="black")

    heat_ax = fig.add_subplot(outer[1, 0])
    cmap = plt.cm.Reds.copy()
    cmap.set_bad("#f5f5f5")
    finite_activations = activations[np.isfinite(activations)]
    vmax = float(np.max(finite_activations)) if finite_activations.size else 1.0
    im = heat_ax.imshow(grid, cmap=cmap, vmin=0.0, vmax=max(vmax, 1e-6), interpolation="nearest", aspect="equal")
    heat_ax.set_xticks([])
    heat_ax.set_yticks([])
    for spine in heat_ax.spines.values():
        spine.set_color("white")
        spine.set_linewidth(2.0)
    heat_ax.set_title(f"Frame {step}", fontsize=11, pad=8)
    heat_ax.set_xlabel("All muscle activations", fontsize=18, labelpad=12)

    fig.colorbar(im, ax=heat_ax, fraction=0.045, pad=0.025).ax.tick_params(labelsize=7)

    trace_spec = gridspec.GridSpecFromSubplotSpec(len(profile_indices), 1, subplot_spec=outer[1, 2], hspace=0.18)
    for row, idx in enumerate(profile_indices):
        ax = fig.add_subplot(trace_spec[row, 0])
        ax.plot(x, activations[: len(x), idx], color="#ef6a6a", linewidth=1.2)
        if 0 <= step < len(x):
            ax.axvline(x[step], color="#444444", linewidth=0.7, alpha=0.35)
        ax.set_xlim(float(x[0]), float(x[-1])) if len(x) > 1 else ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(0.0, max(float(np.nanmax(activations[:, idx])) * 1.08, 1e-3))
        ax.set_xticks([])
        ax.set_yticks([])
        label = muscle_names[idx] if idx < len(muscle_names) else f"muscle_{idx:03d}"
        ax.text(0.01, 0.82, label, transform=ax.transAxes, fontsize=7, color="#444444", ha="left", va="top")
        for spine in ax.spines.values():
            spine.set_color("#eeeeee")
            spine.set_linewidth(0.9)
        ax.set_facecolor("white")

    fig.text(0.76, 0.072, "Full activation time series", ha="center", va="center", fontsize=18, color="black")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _analyze_muscles(
    npz: np.lib.npyio.NpzFile,
    episode: int,
    outdir: Path,
    actuator_names: list[str],
    top_n: int,
    pattern: str | None,
    snapshot_step: int,
    profile_count: int,
) -> None:
    prefix = f"episode_{episode}"
    activations = np.asarray(npz[f"{prefix}_muscle_activations"], dtype=float)
    commands = np.asarray(npz[f"{prefix}_muscle_commands"], dtype=float)
    time = np.asarray(npz[f"{prefix}_timesteps"], dtype=float)

    n = activations.shape[1] if activations.ndim == 2 else 0
    actuator_names = _resolve_muscle_names(npz, actuator_names, n, episode)

    max_activation = np.nanmax(activations, axis=0)
    mean_activation = np.nanmean(activations, axis=0)
    activation_auc = np.trapezoid(
        np.nan_to_num(activations),
        dx=float(np.nanmedian(np.diff(time))) if len(time) > 1 else 1.0,
        axis=0,
    )
    max_command = np.nanmax(commands, axis=0)
    command_abs_mean = np.nanmean(np.abs(commands), axis=0)
    score = 0.5 * max_activation + 0.3 * mean_activation + 0.2 * command_abs_mean

    keep = _select_indices(actuator_names, score, pattern, top_n)

    rows = []
    for idx in np.argsort(score)[::-1]:
        rows.append(
            {
                "rank": len(rows) + 1,
                "muscle": actuator_names[idx],
                "score": float(score[idx]),
                "max_activation": float(max_activation[idx]),
                "mean_activation": float(mean_activation[idx]),
                "activation_auc": float(activation_auc[idx]),
                "max_command": float(max_command[idx]),
                "mean_abs_command": float(command_abs_mean[idx]),
            }
        )
    _write_csv(outdir / "muscle_summary.csv", rows)

    labels = [actuator_names[i] for i in keep]
    _plot_heatmap(
        outdir / "muscle_activation_heatmap.png",
        activations[:, keep],
        labels,
        f"Episode {episode}: key muscle activation",
        "activation",
    )
    _plot_profiles(
        outdir / "key_muscle_activation_profiles.png",
        time,
        activations[:, keep[: min(8, len(keep))]],
        labels[: min(8, len(labels))],
        f"Episode {episode}: top muscle activation profiles",
        "activation",
    )
    _plot_profiles(
        outdir / "key_muscle_command_profiles.png",
        time[: commands.shape[0]],
        commands[:, keep[: min(8, len(keep))]],
        labels[: min(8, len(labels))],
        f"Episode {episode}: top muscle command profiles",
        "ctrl",
    )
    resolved_snapshot = activations.shape[0] // 2 if snapshot_step < 0 else snapshot_step
    _plot_activation_dynamics_panel(
        outdir / "muscle_activation_dynamics_panel.png",
        time,
        activations,
        actuator_names,
        keep[: min(max(profile_count, 1), len(keep))],
        resolved_snapshot,
        f"Episode {episode}: Comprehensive Muscle Activation Dynamics",
    )


def _analyze_joints(
    npz: np.lib.npyio.NpzFile,
    episode: int,
    outdir: Path,
    scalar_joint_indices: dict[str, tuple[int, int]],
    top_n: int,
    pattern: str | None,
) -> None:
    prefix = f"episode_{episode}"
    qpos = np.asarray(npz[f"{prefix}_joint_positions"], dtype=float)
    qvel = np.asarray(npz[f"{prefix}_joint_velocities"], dtype=float)
    qacc = np.asarray(npz[f"{prefix}_joint_accelerations"], dtype=float)
    time = np.asarray(npz[f"{prefix}_timesteps"], dtype=float)

    names = []
    qpos_cols = []
    qvel_cols = []
    for name, (qpos_idx, qvel_idx) in scalar_joint_indices.items():
        if qpos_idx < qpos.shape[1] and qvel_idx < qvel.shape[1]:
            names.append(name)
            qpos_cols.append(qpos_idx)
            qvel_cols.append(qvel_idx)

    if not names:
        names = _as_str_list(npz["joint_names"]) if "joint_names" in npz.files else [f"qpos_{i:03d}" for i in range(qpos.shape[1])]
        qpos_cols = list(range(min(len(names), qpos.shape[1])))
        qvel_cols = list(range(min(len(names), qvel.shape[1])))

    qpos_key = qpos[:, qpos_cols]
    qvel_key = qvel[:, qvel_cols]
    qacc_key = qacc[:, qvel_cols] if qacc.ndim == 2 and qacc.shape[1] >= max(qvel_cols, default=-1) + 1 else np.gradient(qvel_key, axis=0)

    range_abs = np.nanmax(qpos_key, axis=0) - np.nanmin(qpos_key, axis=0)
    mean_abs_vel = np.nanmean(np.abs(qvel_key), axis=0)
    max_abs_vel = np.nanmax(np.abs(qvel_key), axis=0)
    mean_abs_acc = np.nanmean(np.abs(qacc_key), axis=0)
    max_abs_acc = np.nanmax(np.abs(qacc_key), axis=0)
    score = range_abs + 0.1 * mean_abs_vel + 0.01 * mean_abs_acc

    keep = _select_indices(names, score, pattern, top_n)
    rows = []
    for idx in np.argsort(score)[::-1]:
        rows.append(
            {
                "rank": len(rows) + 1,
                "joint": names[idx],
                "qpos_index": qpos_cols[idx],
                "qvel_index": qvel_cols[idx],
                "score": float(score[idx]),
                "position_range_rad_or_m": float(range_abs[idx]),
                "mean_abs_velocity": float(mean_abs_vel[idx]),
                "max_abs_velocity": float(max_abs_vel[idx]),
                "mean_abs_acceleration": float(mean_abs_acc[idx]),
                "max_abs_acceleration": float(max_abs_acc[idx]),
            }
        )
    _write_csv(outdir / "joint_summary.csv", rows)

    labels = [names[i] for i in keep]
    _plot_heatmap(
        outdir / "joint_position_heatmap.png",
        qpos_key[:, keep],
        labels,
        f"Episode {episode}: key joint positions",
        "qpos",
    )
    _plot_heatmap(
        outdir / "joint_velocity_heatmap.png",
        qvel_key[:, keep],
        labels,
        f"Episode {episode}: key joint velocities",
        "qvel",
    )
    _plot_profiles(
        outdir / "key_joint_position_profiles.png",
        time,
        qpos_key[:, keep[: min(8, len(keep))]],
        labels[: min(8, len(labels))],
        f"Episode {episode}: top joint position profiles",
        "qpos",
    )
    _plot_profiles(
        outdir / "key_joint_velocity_profiles.png",
        time,
        qvel_key[:, keep[: min(8, len(keep))]],
        labels[: min(8, len(labels))],
        f"Episode {episode}: top joint velocity profiles",
        "qvel",
    )


def _analyze_tracking(npz: np.lib.npyio.NpzFile, episode: int, outdir: Path, ref_offset: int) -> None:
    prefix = f"episode_{episode}"
    ref_key = f"{prefix}_traj_qpos"
    if ref_key not in npz.files:
        return

    qpos = np.asarray(npz[f"{prefix}_joint_positions"], dtype=float)
    ref = np.asarray(npz[ref_key], dtype=float)
    time = np.asarray(npz[f"{prefix}_timesteps"], dtype=float)
    if ref_offset < 0 or ref_offset >= len(ref):
        print(f"Warning: invalid ref offset {ref_offset}; skipping tracking plot")
        return

    n = min(len(qpos), len(ref) - ref_offset)
    dim = min(qpos.shape[1], ref.shape[1])
    if n <= 0 or dim <= 0:
        return

    err = qpos[:n, :dim] - ref[ref_offset : ref_offset + n, :dim]
    root_xyz_err = np.linalg.norm(err[:, :3], axis=1) if dim >= 3 else np.zeros(n)
    joint_rmse = np.sqrt(np.nanmean(err[:, 7:] ** 2, axis=1)) if dim > 7 else np.sqrt(np.nanmean(err**2, axis=1))
    total_rmse = np.sqrt(np.nanmean(err**2, axis=1))

    rows = []
    for i in range(n):
        rows.append(
            {
                "step": i,
                "time_s": float(time[i]) if i < len(time) else float(i),
                "root_xyz_error_m": float(root_xyz_err[i]),
                "joint_qpos_rmse": float(joint_rmse[i]),
                "total_qpos_rmse": float(total_rmse[i]),
            }
        )
    _write_csv(outdir / "tracking_error.csv", rows)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = time[:n] if len(time) >= n else np.arange(n)
    ax.plot(x, root_xyz_err, label="root xyz error")
    ax.plot(x, joint_rmse, label="joint qpos RMSE")
    ax.plot(x, total_rmse, label="total qpos RMSE")
    ax.set_title(f"Episode {episode}: policy vs reference qpos error")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "tracking_error.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Trajectory export NPZ from fullbody/eval.py")
    parser.add_argument("--outdir", default="visualize/output", help="Directory for figures and CSV files")
    parser.add_argument("--episode", default="all", help="Episode id or 'all'")
    parser.add_argument("--top-muscles", type=int, default=24, help="Number of muscle rows to plot")
    parser.add_argument("--top-joints", type=int, default=18, help="Number of joint rows to plot")
    parser.add_argument("--muscle-pattern", default=DEFAULT_MUSCLE_PATTERN, help="Regex for candidate key muscles; empty uses all")
    parser.add_argument("--joint-pattern", default=DEFAULT_JOINT_PATTERN, help="Regex for candidate key joints; empty uses all")
    parser.add_argument(
        "--activation-snapshot-step",
        type=int,
        default=-1,
        help="Frame used for the all-muscle activation snapshot; -1 uses the middle frame",
    )
    parser.add_argument(
        "--activation-profile-count",
        type=int,
        default=8,
        help="Number of ranked muscle activation traces in the combined dynamics panel",
    )
    parser.add_argument("--ref-offset", type=int, default=0, help="Reference qpos start offset for tracking error")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    npz = np.load(in_path, allow_pickle=True)
    available = _episode_ids(npz)
    if not available:
        raise ValueError(f"No episode_*_joint_positions arrays found in {in_path}")

    if args.episode == "all":
        episodes = available
    else:
        episodes = [int(args.episode)]
        missing = [ep for ep in episodes if ep not in available]
        if missing:
            raise ValueError(f"Episodes not found: {missing}. Available: {available}")

    actuator_names, _joint_names, scalar_joint_indices = _load_model_names()
    muscle_pattern = args.muscle_pattern or None
    joint_pattern = args.joint_pattern or None

    print(f"Input: {in_path}")
    print(f"Episodes: {episodes}")
    for episode in episodes:
        ep_out = out_root / f"episode_{episode:03d}"
        ep_out.mkdir(parents=True, exist_ok=True)
        _analyze_muscles(
            npz,
            episode,
            ep_out,
            actuator_names,
            args.top_muscles,
            muscle_pattern,
            args.activation_snapshot_step,
            args.activation_profile_count,
        )
        _analyze_joints(npz, episode, ep_out, scalar_joint_indices, args.top_joints, joint_pattern)
        _analyze_tracking(npz, episode, ep_out, args.ref_offset)
        print(f"Wrote {ep_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

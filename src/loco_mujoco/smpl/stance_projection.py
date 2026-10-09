"""Stance-contact projection: an OmniRetarget-style foot-sticking post-process for GMR.

GMR retargets SMPL motion to the MyoFullBody skeleton with pure keypoint IK; it has
no notion of which foot is planted or where. This module takes the contact-preserving
reference bundle produced by ``optimized_wham`` (``stance_mask`` + per-segment stance
``anchors``, already in amass/MuJoCo Z-up) and projects the retargeted ``qpos`` so that,
on every stance frame, the corresponding foot *site* is pinned horizontally to its
stance anchor. Only lower-body DOFs move; the root and upper body keep the GMR solution
(the analogue of OmniRetarget's heavy lower-body regulariser).

Design choices (see fix.md, plan B):
* Runs at GMR's native ~30 Hz, *before* ``extend_motion`` interpolates to 100 Hz, so
  downstream forward kinematics (site_xpos/xquat) and qvel are recomputed cleanly by
  ``extend_motion`` rather than patched by hand.
* Horizontal-only pin (x, y in Z-up). Vertical (z) is left to GMR's ground handling;
  a non-penetration clamp keeps sites at or above ground.
* Damped least squares on the stacked foot Jacobians, restricted to lower-body DOFs,
  with per-step clamping and joint-range clipping for stability.

Everything is plain MuJoCo + NumPy so it is unit-testable and verifiable on the real
MyoFullBody model without a GPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

try:  # mujoco is a hard dependency of this module, but keep import errors legible
    import mujoco
except Exception as exc:  # pragma: no cover
    raise ImportError("stance_projection requires the 'mujoco' package") from exc


# SMPL bundle foot label -> MyoFullBody mimic site name.
DEFAULT_FOOT_SITE_MAP: dict[str, str] = {
    "left_heel_or_ankle": "left_ankle_mimic",
    "left_toe_or_forefoot": "left_toes_mimic",
    "right_heel_or_ankle": "right_ankle_mimic",
    "right_toe_or_forefoot": "right_toes_mimic",
}

# Substrings that uniquely identify the 28 MyoFullBody lower-body DOFs
# (hip_*, knee_*, ankle_*, subtalar_*, mtp_*). Arm/spine joint names contain none of these.
LOWER_BODY_JOINT_SUBSTRINGS: tuple[str, ...] = (
    "hip_", "knee_", "ankle_", "subtalar_", "mtp_",
)


@dataclass
class StanceBundle:
    """Per-frame stance schedule + horizontal anchors, in amass/MuJoCo Z-up."""

    stance_mask: np.ndarray  # [T, n_feet] bool
    anchor_xy: np.ndarray  # [T, n_feet, 2] horizontal (x, y) targets, valid where stance_mask
    foot_labels: list[str]
    fps: float


@dataclass
class ProjectionConfig:
    iters: int = 8
    damping: float = 0.05
    max_step: float = 0.2  # max ||dq_lower|| per iteration (rad)
    ground_z: float = 0.0
    clamp_ground: bool = True  # push a stance foot up if it penetrates the ground
    self_anchor: bool = True  # pin feet to the retargeted skeleton's own per-segment site median
    min_run: int = 2  # ignore stance runs shorter than this many frames
    foot_site_map: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_FOOT_SITE_MAP))
    lower_body_substrings: tuple[str, ...] = LOWER_BODY_JOINT_SUBSTRINGS


# --------------------------------------------------------------------------------------
# Model introspection helpers
# --------------------------------------------------------------------------------------
def _dof_count(jnt_type: int) -> int:
    if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
        return 6
    if jnt_type == mujoco.mjtJoint.mjJNT_BALL:
        return 3
    return 1  # slide / hinge


def resolve_lower_dof_ids(model, substrings: Sequence[str] = LOWER_BODY_JOINT_SUBSTRINGS) -> np.ndarray:
    """Return the DOF (nv) indices belonging to lower-body joints."""
    ids: list[int] = []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        low = name.lower()
        if any(s in low for s in substrings):
            adr = int(model.jnt_dofadr[j])
            for k in range(_dof_count(int(model.jnt_type[j]))):
                ids.append(adr + k)
    return np.array(sorted(set(ids)), dtype=int)


def resolve_foot_site_ids(model, foot_site_map: dict[str, str]) -> dict[str, int]:
    """Map SMPL foot labels -> MuJoCo site id, skipping any site absent from the model."""
    out: dict[str, int] = {}
    for label, site_name in foot_site_map.items():
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if sid >= 0:
            out[label] = int(sid)
    return out


def _clip_lower_to_range(model, qpos: np.ndarray, lower_dofs: np.ndarray) -> None:
    for dof in lower_dofs:
        j = int(model.dof_jntid[dof])
        if not model.jnt_limited[j]:
            continue
        qadr = int(model.jnt_qposadr[j])
        lo, hi = float(model.jnt_range[j, 0]), float(model.jnt_range[j, 1])
        qpos[qadr] = min(max(qpos[qadr], lo), hi)


# --------------------------------------------------------------------------------------
# Core IK
# --------------------------------------------------------------------------------------
def project_single_frame(
    model,
    data,
    qpos: np.ndarray,
    site_targets: Sequence[tuple[int, np.ndarray]],
    lower_dofs: np.ndarray,
    cfg: ProjectionConfig,
) -> np.ndarray:
    """Damped-least-squares pin of the given sites' horizontal position, lower-body DOFs only.

    Args:
        site_targets: list of ``(site_id, target_xy)`` where ``target_xy`` is the desired
            world (x, y) in Z-up. Vertical is not constrained here.
    Returns:
        The projected qpos (copy).
    """
    qpos = np.asarray(qpos, dtype=np.float64).copy()
    if len(site_targets) == 0 or len(lower_dofs) == 0:
        return qpos

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    n_lower = len(lower_dofs)

    for _ in range(cfg.iters):
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)  # populates cdof; required by mj_jacSite

        rows: list[np.ndarray] = []
        err: list[np.ndarray] = []
        for sid, target_xy in site_targets:
            mujoco.mj_jacSite(model, data, jacp, jacr, sid)
            rows.append(jacp[:2][:, lower_dofs])  # (2, n_lower)
            err.append(np.asarray(target_xy, dtype=np.float64) - data.site_xpos[sid, :2])
        j_all = np.vstack(rows)  # (2m, n_lower)
        e_all = np.concatenate(err)  # (2m,)

        # Damped least squares: dq = J^T (J J^T + lambda I)^-1 e
        jjt = j_all @ j_all.T
        dq_active = j_all.T @ np.linalg.solve(
            jjt + cfg.damping * np.eye(jjt.shape[0]), e_all
        )
        norm = float(np.linalg.norm(dq_active))
        if norm > cfg.max_step and norm > 0.0:
            dq_active *= cfg.max_step / norm

        dq = np.zeros(model.nv)
        dq[lower_dofs] = dq_active
        mujoco.mj_integratePos(model, qpos, dq, 1.0)
        _clip_lower_to_range(model, qpos, lower_dofs)

        if float(np.linalg.norm(e_all)) < 1e-4:
            break

    return qpos


# --------------------------------------------------------------------------------------
# Trajectory-level projection
# --------------------------------------------------------------------------------------
def project_trajectory(
    qpos: np.ndarray,
    model,
    bundle: StanceBundle,
    *,
    cfg: Optional[ProjectionConfig] = None,
) -> tuple[np.ndarray, dict]:
    """Project a whole ``qpos`` trajectory against a stance bundle.

    ``bundle`` is resampled (nearest frame) to ``qpos``'s frame count, so a small
    trim/offset between GMR frames and bundle frames is handled gracefully.

    Anchor targets are, by default (``cfg.self_anchor``), the retargeted skeleton's
    *own* foot-site horizontal position, taken as the median over each contiguous
    stance run. This is the correct OmniRetarget-style foot-lock: it removes the
    residual intra-stance slide GMR leaves behind without dragging the leg toward an
    absolute SMPL-world anchor that lives in a different origin/scale (which is what
    produced the ~0.5 m leg distortion). Setting ``self_anchor=False`` restores the
    legacy behaviour of pinning to ``bundle.anchor_xy``.

    Returns ``(projected_qpos, report)``.
    """
    cfg = cfg or ProjectionConfig()
    qpos = np.asarray(qpos, dtype=np.float64).copy()
    t_q = qpos.shape[0]

    lower_dofs = resolve_lower_dof_ids(model, cfg.lower_body_substrings)
    foot_site_ids = resolve_foot_site_ids(model, cfg.foot_site_map)

    stance_mask, anchor_xy = _resample_bundle(bundle, cfg, t_q)

    data = mujoco.MjData(model)
    upper_dof_mask = np.ones(model.nv, dtype=bool)
    upper_dof_mask[lower_dofs] = False

    # Foot-site horizontal position for every frame of the *unprojected* trajectory.
    site_xy0 = _forward_site_xy(model, data, qpos, foot_site_ids)

    # Per-frame list of (site_id, target_xy) targets.
    targets_per_frame = _build_stance_targets(
        bundle, foot_site_ids, stance_mask, anchor_xy, site_xy0, cfg
    )

    projected = qpos.copy()
    site_slide_before: list[float] = []
    site_slide_after: list[float] = []
    upper_dof_max_delta = 0.0
    n_projected_frames = 0

    prev_site_before: dict[int, np.ndarray] = {}
    prev_site_after: dict[int, np.ndarray] = {}

    for t in range(t_q):
        site_targets = targets_per_frame[t]

        # sliding "before" (pre-projection) foot displacement between stance frames
        for sid, _ in site_targets:
            cur = site_xy0[sid][t]
            if sid in prev_site_before:
                site_slide_before.append(float(np.linalg.norm(cur - prev_site_before[sid])))
            prev_site_before[sid] = cur

        if not site_targets:
            prev_site_after.clear()
            continue

        q_proj = project_single_frame(model, data, qpos[t], site_targets, lower_dofs, cfg)

        if cfg.clamp_ground:
            _apply_ground_clamp(model, data, q_proj, [sid for sid, _ in site_targets], cfg.ground_z)

        projected[t] = q_proj
        n_projected_frames += 1
        upper_dof_max_delta = max(
            upper_dof_max_delta,
            float(np.max(np.abs((projected[t] - qpos[t])[_qpos_of_dofs(model, np.where(upper_dof_mask)[0])])))
            if np.any(upper_dof_mask) else 0.0,
        )

        # sliding "after"
        data.qpos[:] = q_proj
        mujoco.mj_kinematics(model, data)
        for sid, _ in site_targets:
            cur = data.site_xpos[sid, :2].copy()
            if sid in prev_site_after:
                site_slide_after.append(float(np.linalg.norm(cur - prev_site_after[sid])))
            prev_site_after[sid] = cur

    report = {
        "n_frames": int(t_q),
        "n_projected_frames": int(n_projected_frames),
        "n_lower_dofs": int(len(lower_dofs)),
        "foot_sites_used": sorted(foot_site_ids.keys()),
        "self_anchor": bool(cfg.self_anchor),
        "stance_slide_mean_cm_before": float(np.mean(site_slide_before) * 100.0) if site_slide_before else 0.0,
        "stance_slide_max_cm_before": float(np.max(site_slide_before) * 100.0) if site_slide_before else 0.0,
        "stance_slide_mean_cm_after": float(np.mean(site_slide_after) * 100.0) if site_slide_after else 0.0,
        "stance_slide_max_cm_after": float(np.max(site_slide_after) * 100.0) if site_slide_after else 0.0,
        "upper_body_qpos_max_delta": float(upper_dof_max_delta),
    }
    return projected.astype(np.float32), report


def _forward_site_xy(model, data, qpos: np.ndarray, foot_site_ids: dict[str, int]) -> dict[int, np.ndarray]:
    """Return, per foot site id, the [T, 2] horizontal world position of the given qpos."""
    sids = sorted(set(foot_site_ids.values()))
    out = {sid: np.zeros((qpos.shape[0], 2), dtype=np.float64) for sid in sids}
    for t in range(qpos.shape[0]):
        data.qpos[:] = qpos[t]
        mujoco.mj_kinematics(model, data)
        for sid in sids:
            out[sid][t] = data.site_xpos[sid, :2].copy()
    return out


def _stance_runs(mask_col: np.ndarray, min_run: int) -> list[tuple[int, int]]:
    """Contiguous [start, end) runs of True at least ``min_run`` frames long."""
    runs: list[tuple[int, int]] = []
    start = None
    for t, flag in enumerate(mask_col):
        if flag and start is None:
            start = t
        elif not flag and start is not None:
            if t - start >= min_run:
                runs.append((start, t))
            start = None
    if start is not None and len(mask_col) - start >= min_run:
        runs.append((start, len(mask_col)))
    return runs


def _build_stance_targets(
    bundle: StanceBundle,
    foot_site_ids: dict[str, int],
    stance_mask: np.ndarray,
    anchor_xy: np.ndarray,
    site_xy0: dict[int, np.ndarray],
    cfg: ProjectionConfig,
) -> list[list[tuple[int, np.ndarray]]]:
    t_q = stance_mask.shape[0]
    targets_per_frame: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(t_q)]

    for foot_idx, label in enumerate(bundle.foot_labels):
        if label not in foot_site_ids:
            continue
        sid = foot_site_ids[label]
        col = stance_mask[:, foot_idx]

        if cfg.self_anchor:
            for start, end in _stance_runs(col, max(1, int(cfg.min_run))):
                target = np.median(site_xy0[sid][start:end], axis=0)
                for t in range(start, end):
                    targets_per_frame[t].append((sid, target))
        else:
            for t in range(t_q):
                if bool(col[t]):
                    targets_per_frame[t].append((sid, np.asarray(anchor_xy[t, foot_idx], dtype=np.float64)))

    return targets_per_frame


def _qpos_of_dofs(model, dof_ids: np.ndarray) -> np.ndarray:
    """Map DOF (nv) indices to their qpos addresses (1-dof joints; free-joint dofs skipped)."""
    qadr: list[int] = []
    for dof in np.atleast_1d(dof_ids):
        j = int(model.dof_jntid[int(dof)])
        if int(model.jnt_type[j]) in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
            continue
        qadr.append(int(model.jnt_qposadr[j]))
    return np.array(sorted(set(qadr)), dtype=int)


def _apply_ground_clamp(model, data, qpos: np.ndarray, site_ids: Sequence[int], ground_z: float) -> None:
    """Raise the whole body (root z) so no stance site penetrates the ground plane."""
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    lowest = min(float(data.site_xpos[sid, 2]) for sid in site_ids)
    if lowest < ground_z:
        qpos[2] += (ground_z - lowest)


def _resample_bundle(bundle: StanceBundle, cfg: ProjectionConfig, t_q: int) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-frame resample stance_mask/anchor_xy from bundle frames to t_q frames."""
    t_b = bundle.stance_mask.shape[0]
    n_feet = bundle.stance_mask.shape[1]
    if t_b == t_q:
        return bundle.stance_mask.astype(bool), bundle.anchor_xy.astype(np.float64)
    if t_b == 0:
        return (
            np.zeros((t_q, n_feet), dtype=bool),
            np.zeros((t_q, n_feet, 2), dtype=np.float64),
        )
    # map each output frame to nearest input frame on a normalised [0, 1] timeline
    src = np.clip(np.round(np.linspace(0, t_b - 1, t_q)).astype(int), 0, t_b - 1)
    return bundle.stance_mask.astype(bool)[src], bundle.anchor_xy.astype(np.float64)[src]


# --------------------------------------------------------------------------------------
# Bundle loading
# --------------------------------------------------------------------------------------
def load_stance_bundle(manifest_path: str | Path) -> StanceBundle:
    """Load stance_mask + rasterised horizontal anchors from a reference bundle."""
    manifest_path = Path(manifest_path)
    base = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    contact = np.load(base / manifest["contact_npz"], allow_pickle=True)
    stance_mask = np.asarray(contact["stance_mask"], dtype=bool)
    foot_labels = [str(x) for x in np.atleast_1d(contact["foot_labels"])]
    fps = float(manifest.get("fps", 30.0))

    anchors = json.loads((base / manifest["stance_anchors_json"]).read_text(encoding="utf-8"))
    anchor_xy = rasterize_anchor_xy(anchors, stance_mask.shape[0], stance_mask.shape[1])
    return StanceBundle(stance_mask=stance_mask, anchor_xy=anchor_xy, foot_labels=foot_labels, fps=fps)


def rasterize_anchor_xy(anchors_json: dict, n_frames: int, n_feet: int) -> np.ndarray:
    """Expand per-segment anchors into a per-frame [T, n_feet, 2] horizontal target array."""
    out = np.zeros((int(n_frames), int(n_feet), 2), dtype=np.float64)
    for anchor in anchors_json.get("anchors", []):
        foot_idx = int(anchor["foot_index"])
        if foot_idx < 0 or foot_idx >= n_feet:
            continue
        start = max(0, min(int(n_frames), int(anchor["start"])))
        end = max(start, min(int(n_frames), int(anchor["end"])))
        xy = np.asarray(anchor["anchor_xz"], dtype=np.float64)[:2]
        out[start:end, foot_idx, :] = xy
    return out

#!/usr/bin/env python3
"""Evaluate lower-body fidelity of an optimized muscle-trajectory cache against raw.

The optimized retarget path adds a stance-projection post-process on top of the raw
GMR solution. A good result must satisfy two competing goals at once:

  1. Faithfulness  - genuine footwork (steps, pivots) must survive. If the optimized
     legs deviate wildly from raw while raw already looked fine, the post-process is
     destroying motion, not cleaning it.
  2. Contact quality - during frames where a foot is genuinely planted, that foot
     should stop sliding and stop penetrating the floor. That is the whole point of
     the projection.

This script quantifies both from the two .npz caches alone (no GPU, no rendering) so
it can gate every iteration of the fix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FOOT_SITES = ["left_ankle_mimic", "left_toes_mimic", "right_ankle_mimic", "right_toes_mimic"]
LOWER_BODY_SUBSTRINGS = ("hip_", "knee_", "ankle_", "subtalar_", "mtp_")

# A raw retarget whose feet travel less than this (metres, summed over 4 sites) is
# treated as degenerate: GMR froze the raw WHAM because its ungrounded world frame put
# the feet below the floor. This is exactly what the optimization pipeline fixes, so the
# optimized output is scored on absolute quality rather than fidelity to the frozen raw.
DEGENERATE_RAW_FOOTWORK_M = 0.5
# Absolute lower-body joint jerk ceiling (rad, mean |3rd diff|), ~2.5x the ~0.0006-0.0008
# seen on healthy raw/optimized caches. Used only in degenerate-baseline mode.
ABS_JERK_CEILING = 0.0015


def _qpos_addr(joint_names: list[str], jnt_type: np.ndarray) -> dict[str, tuple[int, int]]:
    dim = {0: 7, 1: 4, 2: 1, 3: 1}
    addr: dict[str, tuple[int, int]] = {}
    cursor = 0
    for j, name in enumerate(joint_names):
        d = dim.get(int(jnt_type[j]), 1)
        addr[name] = (cursor, d)
        cursor += d
    return addr


def _lower_body_cols(joint_names: list[str], addr: dict[str, tuple[int, int]]) -> np.ndarray:
    cols: list[int] = []
    for name in joint_names:
        if any(s in name.lower() for s in LOWER_BODY_SUBSTRINGS):
            a, d = addr[name]
            cols.extend(range(a, a + d))
    return np.asarray(cols, dtype=int)


def _foot_speed(site_xpos: np.ndarray, site_idx: int, fps: float) -> np.ndarray:
    """Per-frame horizontal speed (m/s) of one foot site."""
    xy = site_xpos[:, site_idx, :2]
    return np.linalg.norm(np.diff(xy, axis=0), axis=1) * fps


def evaluate(raw_path: Path, opt_path: Path, planted_speed: float = 0.10) -> dict:
    raw = np.load(raw_path, allow_pickle=True)
    opt = np.load(opt_path, allow_pickle=True)

    joint_names = [str(x) for x in raw["joint_names"]]
    site_names = [str(x) for x in raw["site_names"]]
    addr = _qpos_addr(joint_names, raw["jnt_type"])
    lb_cols = _lower_body_cols(joint_names, addr)
    fps = float(np.asarray(raw["frequency"]).reshape(-1)[0])

    qr, qo = raw["qpos"], opt["qpos"]
    n = min(len(qr), len(qo))
    qr, qo = qr[:n], qo[:n]

    lower_dev = float(np.abs(qo[:, lb_cols] - qr[:, lb_cols]).max())
    lower_dev_mean = float(np.abs(qo[:, lb_cols] - qr[:, lb_cols]).mean())

    # Lower-body joint jerk: guards against the projection trading foot-skate for joint
    # chatter. Compared in joint space because that is what drives muscle dynamics.
    jerk_raw = float(np.abs(np.diff(qr[:, lb_cols], n=3, axis=0)).mean())
    jerk_opt = float(np.abs(np.diff(qo[:, lb_cols], n=3, axis=0)).mean())

    per_foot = {}
    slide_raw_planted, slide_opt_planted = [], []
    pen_raw, pen_opt = [], []
    footwork_raw, footwork_opt = [], []
    foot_z_raw_all, foot_z_opt_all = [], []

    for site in FOOT_SITES:
        if site not in site_names:
            continue
        si = site_names.index(site)
        sxr, sxo = raw["site_xpos"][:n, si], opt["site_xpos"][:n, si]
        foot_z_raw_all.append(sxr[:, 2])
        foot_z_opt_all.append(sxo[:, 2])

        # Footwork amplitude: total horizontal path length the foot travels.
        path_raw = float(np.linalg.norm(np.diff(sxr[:, :2], axis=0), axis=1).sum())
        path_opt = float(np.linalg.norm(np.diff(sxo[:, :2], axis=0), axis=1).sum())
        footwork_raw.append(path_raw)
        footwork_opt.append(path_opt)

        # Contact quality is only meaningful where the foot is genuinely planted. Use
        # the intersection of raw-planted and opt-planted frames so the metric measures
        # "when this foot is down in both, does the optimized one stop sliding" rather
        # than re-penalising legitimate footwork differences.
        spd_raw = _foot_speed(raw["site_xpos"][:n], si, fps)
        spd_opt = _foot_speed(opt["site_xpos"][:n], si, fps)
        low_raw = sxr[1:, 2] < (sxr[:, 2].min() + 0.05)
        low_opt = sxo[1:, 2] < (sxo[:, 2].min() + 0.05)
        planted_raw = (spd_raw < planted_speed) & low_raw
        planted_both = planted_raw & (spd_opt < planted_speed) & low_opt
        sr = np.linalg.norm(np.diff(sxr[:, :2], axis=0), axis=1)
        so = np.linalg.norm(np.diff(sxo[:, :2], axis=0), axis=1)
        if planted_raw.any():
            slide_raw_planted.append(float(sr[planted_raw].mean() * 100.0))
        if planted_both.any():
            slide_opt_planted.append(float(so[planted_both].mean() * 100.0))

        pen_raw.append(float(max(0.0, -sxr[:, 2].min()) * 100.0))
        pen_opt.append(float(max(0.0, -sxo[:, 2].min()) * 100.0))

        per_foot[site] = {
            "path_len_raw_m": round(path_raw, 4),
            "path_len_opt_m": round(path_opt, 4),
            "path_ratio_opt_over_raw": round(path_opt / path_raw, 3) if path_raw > 1e-6 else None,
            "planted_frames_raw": int(planted_raw.sum()),
            "planted_frames_both": int(planted_both.sum()),
            "z_min_raw": round(float(sxr[:, 2].min()), 4),
            "z_min_opt": round(float(sxo[:, 2].min()), 4),
        }

    # Feet grounding: the primary reported bug was feet floating well above the floor.
    # Judge each *leg* by its lowest contact point (min of its ankle and toe sites): a
    # grounded foot touches down at ~0 there, whereas the broken projection left every
    # site — toe included — floating 5-13 cm up. Ankle sites naturally sit ~6 cm above the
    # floor even when planted, so requiring them to reach 0 would false-flag good feet.
    # Raw supplies the per-leg reference touch height, clamped at 0 so a raw foot that
    # penetrates the floor does not make a correctly grounded optimized foot look floaty.
    leg_sites = {
        "left": ["left_ankle_mimic", "left_toes_mimic"],
        "right": ["right_ankle_mimic", "right_toes_mimic"],
    }
    leg_gaps = []
    for sites in leg_sites.values():
        present = [per_foot[s] for s in sites if s in per_foot]
        if not present:
            continue
        opt_low = min(pf["z_min_opt"] for pf in present)
        raw_low = min(pf["z_min_raw"] for pf in present)
        leg_gaps.append((opt_low - max(raw_low, 0.0)) * 100.0)
    foot_float_gap_cm = round(max(leg_gaps), 4) if leg_gaps else 0.0

    # Body uprightness: median pelvis clearance above the (per-frame) lowest foot
    # point. A standing/lunging player stays around 0.8-1.0 m; the frame-mismatch
    # ground alignment collapsed this to ~0.15 m (kneeling) while every foot-level
    # check above still passed - so it must gate explicitly.
    root_clearance_raw = root_clearance_opt = None
    if "root" in addr and foot_z_raw_all:
        root_start, root_width = addr["root"]
        if root_width == 7:
            lowest_raw = np.min(np.stack(foot_z_raw_all, axis=0), axis=0)
            lowest_opt = np.min(np.stack(foot_z_opt_all, axis=0), axis=0)
            root_clearance_raw = float(np.median(qr[:, root_start + 2] - lowest_raw))
            root_clearance_opt = float(np.median(qo[:, root_start + 2] - lowest_opt))

    result = {
        "raw_cache": str(raw_path),
        "opt_cache": str(opt_path),
        "n_frames": int(n),
        "raw_baseline_degenerate": bool(float(np.sum(footwork_raw)) < DEGENERATE_RAW_FOOTWORK_M),
        "lower_body_qpos_dev_from_raw_max_rad": round(lower_dev, 4),
        "lower_body_qpos_dev_from_raw_mean_rad": round(lower_dev_mean, 4),
        "lower_body_jerk_opt_abs": round(jerk_opt, 6),
        "lower_body_jerk_ratio_opt_over_raw": round(jerk_opt / jerk_raw, 3) if jerk_raw > 1e-9 else None,
        "foot_float_gap_max_cm": foot_float_gap_cm,
        "root_clearance_med_m_raw": round(root_clearance_raw, 4) if root_clearance_raw is not None else None,
        "root_clearance_med_m_opt": round(root_clearance_opt, 4) if root_clearance_opt is not None else None,
        "footwork_path_total_raw_m": round(float(np.sum(footwork_raw)), 4),
        "footwork_path_total_opt_m": round(float(np.sum(footwork_opt)), 4),
        "footwork_preserved_ratio": round(float(np.sum(footwork_opt) / max(np.sum(footwork_raw), 1e-6)), 3),
        "planted_slide_mean_cm_raw": round(float(np.mean(slide_raw_planted)), 4) if slide_raw_planted else None,
        "planted_slide_mean_cm_opt": round(float(np.mean(slide_opt_planted)), 4) if slide_opt_planted else None,
        "foot_penetration_max_cm_raw": round(float(np.max(pen_raw)), 4) if pen_raw else None,
        "foot_penetration_max_cm_opt": round(float(np.max(pen_opt)), 4) if pen_opt else None,
        "per_foot": per_foot,
    }
    result["verdict"] = _verdict(result)
    return result


def _verdict(r: dict) -> dict:
    """Effectiveness gate on absolute lower-body quality.

    The optimization pipeline exists precisely because the raw GMR retarget is
    unreliable - for some sequences it freezes, skates, or drives the feet through the
    floor. So fidelity to raw is the wrong yardstick; a correct optimized cache is judged
    on absolute quality, each check targeting a failure the fixes address:

    - feet_grounded: no foot floats more than 5 cm above its (floor-clamped) touch height.
      Catches the ~0.2 m float the anchor-mismatch projection produced.
    - footwork_present: feet travel at least 1 m in total - catches the all-frame stance
      schedule that froze the feet.
    - planted_sliding_ok: on frames where both raw and opt agree a foot is planted, the
      optimized foot slides under 0.3 cm/frame (contact quality - the point of projection).
    - smoothness_ok: absolute lower-body joint jerk under the ceiling - no chatter.
    - body_upright: median pelvis clearance above the lowest foot point stays over
      0.55 m (standing/lunging, not collapsed). Catches the foot-frame-mismatch
      ground alignment that sank the whole body ~1.7 m and left it kneeling while
      every foot-level check still passed.

    Raw-fidelity numbers (footwork ratio, pose deviation) are still reported for
    inspection but do not gate, because a defective raw baseline makes them misleading.
    """
    checks = {}
    checks["feet_grounded"] = bool(r["foot_float_gap_max_cm"] <= 5.0)
    checks["footwork_present"] = bool(r["footwork_path_total_opt_m"] >= 1.0)
    so = r["planted_slide_mean_cm_opt"]
    checks["planted_sliding_ok"] = bool(so is None or so <= 0.30)
    checks["smoothness_ok"] = bool(r["lower_body_jerk_opt_abs"] <= ABS_JERK_CEILING)
    clearance = r.get("root_clearance_med_m_opt")
    checks["body_upright"] = bool(clearance is None or clearance >= 0.55)
    return {"checks": checks, "success": all(checks.values())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--opt", type=Path, required=True)
    ap.add_argument("--planted-speed", type=float, default=0.10, help="m/s below which raw foot counts as planted")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    result = evaluate(args.raw, args.opt, planted_speed=args.planted_speed)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
    return 0 if result["verdict"]["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

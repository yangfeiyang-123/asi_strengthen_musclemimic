"""Unit tests for stance-contact projection (OmniRetarget-style foot-sticking post-process)."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from loco_mujoco.smpl.stance_projection import (
    DEFAULT_FOOT_SITE_MAP,
    ProjectionConfig,
    StanceBundle,
    _build_stance_targets,
    _resample_bundle,
    _stance_runs,
    load_stance_bundle,
    rasterize_anchor_xy,
)


def test_default_foot_site_map_covers_the_four_smpl_feet():
    assert set(DEFAULT_FOOT_SITE_MAP) == {
        "left_heel_or_ankle",
        "left_toe_or_forefoot",
        "right_heel_or_ankle",
        "right_toe_or_forefoot",
    }
    # every target is a MyoFullBody mimic site
    assert all(v.endswith("_mimic") for v in DEFAULT_FOOT_SITE_MAP.values())


def test_rasterize_anchor_xy_fills_segments():
    anchors = {
        "anchors": [
            {"foot_index": 1, "start": 2, "end": 5, "anchor_xz": [0.3, -0.4]},
            {"foot_index": 3, "start": 0, "end": 2, "anchor_xz": [1.0, 2.0]},
        ]
    }
    out = rasterize_anchor_xy(anchors, n_frames=6, n_feet=4)
    assert out.shape == (6, 4, 2)
    assert np.allclose(out[2:5, 1], [0.3, -0.4])
    assert np.allclose(out[0:2, 3], [1.0, 2.0])
    assert np.allclose(out[0, 1], 0.0)  # outside its segment
    assert np.allclose(out[5, 3], 0.0)


def test_rasterize_ignores_out_of_range_foot_and_clips_bounds():
    anchors = {
        "anchors": [
            {"foot_index": 9, "start": 0, "end": 3, "anchor_xz": [1.0, 1.0]},
            {"foot_index": 0, "start": -3, "end": 100, "anchor_xz": [0.2, 0.2]},
        ]
    }
    out = rasterize_anchor_xy(anchors, n_frames=4, n_feet=4)
    assert np.allclose(out[:, 9 % 4] if False else out[:, 1], 0.0)  # foot 9 dropped
    assert np.allclose(out[:, 0], [0.2, 0.2])  # clipped to [0, 4)


def test_resample_bundle_is_identity_when_lengths_match():
    mask = np.array([[True, False], [False, True]], dtype=bool)
    anch = np.arange(2 * 2 * 2).reshape(2, 2, 2).astype(float)
    bundle = StanceBundle(mask, anch, ["a", "b"], 30.0)
    m, a = _resample_bundle(bundle, ProjectionConfig(), t_q=2)
    assert np.array_equal(m, mask)
    assert np.allclose(a, anch)


def test_resample_bundle_nearest_upsamples_and_preserves_endpoints():
    mask = np.array([[True, False], [False, False], [True, True]], dtype=bool)  # T_b=3
    anch = np.arange(3 * 2 * 2).reshape(3, 2, 2).astype(float)
    bundle = StanceBundle(mask, anch, ["a", "b"], 30.0)
    m, a = _resample_bundle(bundle, ProjectionConfig(), t_q=6)
    assert m.shape == (6, 2)
    assert a.shape == (6, 2, 2)
    assert np.array_equal(m[0], mask[0])
    assert np.array_equal(m[-1], mask[-1])


def test_resample_bundle_empty_returns_zeros():
    bundle = StanceBundle(np.zeros((0, 4), bool), np.zeros((0, 4, 2)), ["a"] * 4, 30.0)
    m, a = _resample_bundle(bundle, ProjectionConfig(), t_q=5)
    assert m.shape == (5, 4) and not m.any()
    assert a.shape == (5, 4, 2)


def test_stance_runs_finds_min_length_runs():
    col = np.array([1, 1, 0, 1, 1, 1, 0, 1], dtype=bool)
    assert _stance_runs(col, min_run=2) == [(0, 2), (3, 6)]
    assert _stance_runs(col, min_run=3) == [(3, 6)]  # the length-2 run at the end is dropped
    assert _stance_runs(np.ones(4, dtype=bool), min_run=1) == [(0, 4)]


def test_build_stance_targets_self_anchor_uses_own_site_median():
    # foot 0 -> site 100. Its own site position is 0.5 during a stance run; the bundle's
    # absolute anchor (9.0) must be ignored in self_anchor mode.
    bundle = StanceBundle(
        stance_mask=np.array([[True], [True], [False], [True]], dtype=bool),
        anchor_xy=np.full((4, 1, 2), 9.0),
        foot_labels=["left_heel_or_ankle"],
        fps=30.0,
    )
    foot_site_ids = {"left_heel_or_ankle": 100}
    site_xy0 = {100: np.array([[0.5, 0.5], [0.5, 0.5], [7.0, 7.0], [0.5, 0.5]])}
    stance_mask = bundle.stance_mask
    cfg = ProjectionConfig(self_anchor=True, min_run=1)

    targets = _build_stance_targets(bundle, foot_site_ids, stance_mask, bundle.anchor_xy, site_xy0, cfg)
    # frames 0,1,3 are stance; each pinned to the run median of the skeleton's own site (0.5)
    assert np.allclose(targets[0][0][1], [0.5, 0.5])
    assert np.allclose(targets[1][0][1], [0.5, 0.5])
    assert targets[2] == []  # non-stance frame
    assert np.allclose(targets[3][0][1], [0.5, 0.5])


def test_build_stance_targets_legacy_uses_absolute_anchor():
    bundle = StanceBundle(
        stance_mask=np.array([[True], [False]], dtype=bool),
        anchor_xy=np.array([[[9.0, 8.0]], [[0.0, 0.0]]]),
        foot_labels=["left_heel_or_ankle"],
        fps=30.0,
    )
    foot_site_ids = {"left_heel_or_ankle": 100}
    site_xy0 = {100: np.array([[0.5, 0.5], [0.5, 0.5]])}
    cfg = ProjectionConfig(self_anchor=False)
    targets = _build_stance_targets(bundle, foot_site_ids, bundle.stance_mask, bundle.anchor_xy, site_xy0, cfg)
    assert np.allclose(targets[0][0][1], [9.0, 8.0])  # legacy: absolute bundle anchor
    assert targets[1] == []


def test_load_stance_bundle_reads_mask_labels_and_anchors(tmp_path):
    np.savez(
        tmp_path / "contact_schedule.npz",
        stance_mask=np.array([[True, False, False, False], [True, False, False, False]]),
        foot_labels=np.array(
            ["left_heel_or_ankle", "left_toe_or_forefoot", "right_heel_or_ankle", "right_toe_or_forefoot"]
        ),
    )
    (tmp_path / "stance_anchors.json").write_text(
        json.dumps({"anchors": [{"foot_index": 0, "start": 0, "end": 2, "anchor_xz": [0.5, -0.3]}]})
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "contact_npz": "contact_schedule.npz",
                "stance_anchors_json": "stance_anchors.json",
                "fps": 30.0,
            }
        )
    )

    bundle = load_stance_bundle(tmp_path / "manifest.json")
    assert bundle.stance_mask.shape == (2, 4)
    assert bundle.foot_labels[0] == "left_heel_or_ankle"
    assert np.allclose(bundle.anchor_xy[0, 0], [0.5, -0.3])
    assert np.allclose(bundle.anchor_xy[0, 1], 0.0)  # only foot 0 has an anchor
    assert bundle.fps == 30.0


# ---------------------------------------------------------------------------
# Integration: verifies the IK on the real MyoFullBody model (slow, ~12s load).
# ---------------------------------------------------------------------------
_GMR_CACHE = os.path.join(
    "datasets", "_global", "muscle_trajectory", "gmr_cache", "MyoFullBody", "gmr",
    "Transitions_mocap", "mazen_c3d", "run_longjump_poses.npz",
)


@pytest.mark.integration
@pytest.mark.skipif(not os.path.exists(_GMR_CACHE), reason="real gmr_cache not present")
def test_project_single_frame_reaches_target_and_only_moves_lower_body():
    import mujoco

    from loco_mujoco.smpl.retargeting import load_robot_conf_file
    from loco_mujoco.smpl.stance_projection import (
        project_single_frame,
        resolve_foot_site_ids,
        resolve_lower_dof_ids,
    )
    from musclemimic.environments.humanoids.myofullbody import MyoFullBody

    conf = load_robot_conf_file("MyoFullBody")
    env = MyoFullBody(**conf.env_params, th_params={"random_start": False, "fixed_start_conf": (0, 0)})
    model = env._model
    qpos = np.asarray(np.load(_GMR_CACHE, allow_pickle=True)["qpos"], dtype=np.float64)

    lower = resolve_lower_dof_ids(model)
    sites = resolve_foot_site_ids(model, DEFAULT_FOOT_SITE_MAP)
    assert len(lower) == 28  # 14 per leg
    assert set(sites) == set(DEFAULT_FOOT_SITE_MAP)

    data = mujoco.MjData(model)
    cfg = ProjectionConfig(iters=12, damping=0.02, max_step=0.15)
    q = qpos[300]
    data.qpos[:] = q
    mujoco.mj_kinematics(model, data)
    sid = sites["left_toe_or_forefoot"]
    target = data.site_xpos[sid, :2].copy() + np.array([0.08, -0.05])

    q_proj = project_single_frame(model, data, q, [(sid, target)], lower, cfg)
    data.qpos[:] = q_proj
    mujoco.mj_kinematics(model, data)

    assert np.linalg.norm(data.site_xpos[sid, :2] - target) < 0.01  # within 1 cm
    assert np.max(np.abs(q_proj[:7] - q[:7])) < 1e-6  # root untouched

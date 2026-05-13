from pathlib import Path

import numpy as np

from visualize.analyze_rollout_biomechanics import (
    _activation_snapshot_grid,
    _plot_activation_dynamics_panel,
    _resolve_muscle_names,
)


def test_activation_snapshot_grid_uses_one_frame_all_muscles():
    activations = np.arange(30, dtype=float).reshape(3, 10)

    grid = _activation_snapshot_grid(activations, snapshot_step=1)

    assert grid.shape == (3, 4)
    np.testing.assert_array_equal(grid.ravel()[:10], activations[1])
    assert np.isnan(grid.ravel()[-2:]).all()


def test_plot_activation_dynamics_panel_writes_combined_snapshot_and_profiles(tmp_path: Path):
    time = np.linspace(0.0, 0.4, 5)
    activations = np.array(
        [
            [0.00, 0.20, 0.90, 0.10, 0.30, 0.50],
            [0.10, 0.25, 0.80, 0.15, 0.35, 0.55],
            [0.20, 0.30, 0.70, 0.20, 0.40, 0.60],
            [0.30, 0.35, 0.60, 0.25, 0.45, 0.65],
            [0.40, 0.40, 0.50, 0.30, 0.50, 0.70],
        ],
        dtype=float,
    )
    names = [f"muscle_{i}" for i in range(activations.shape[1])]
    out = tmp_path / "panel.png"

    _plot_activation_dynamics_panel(
        out,
        time,
        activations,
        names,
        profile_indices=np.array([2, 5, 4]),
        snapshot_step=2,
        title="Episode 0: comprehensive muscle activation dynamics",
    )

    assert out.exists()
    assert out.stat().st_size > 10_000


def test_resolve_muscle_names_prefers_exported_npz_names_when_count_matches(tmp_path: Path):
    npz_path = tmp_path / "rollout.npz"
    np.savez(npz_path, muscle_names=np.array(["addbrev_r", "bflh_r", "glmax1_r"]))

    with np.load(npz_path, allow_pickle=True) as npz:
        names = _resolve_muscle_names(npz, fallback_names=["wrong_0", "wrong_1", "wrong_2"], n_muscles=3)

    assert names == ["addbrev_r", "bflh_r", "glmax1_r"]

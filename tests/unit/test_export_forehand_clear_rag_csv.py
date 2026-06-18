import csv

import numpy as np
import pytest

from BadmintonMimic.scripts.export_forehand_clear_rag_csv import (
    CSV_COLUMNS,
    export_npz_to_samples,
    write_samples_csv,
)


def _write_named_rollout(path):
    joint_names = np.array(
        ["root", "axial_rotation", "shoulder_rot_r", "elbow_flex_r", "pro_sup_r", "flexion_r"],
        dtype=str,
    )
    muscle_names = np.array(["rect_abd_r", "rect_abd_l", "DELT1", "TRIlong", "TRIlat", "TRImed", "PT", "PQ"], dtype=str)
    qpos = np.zeros((3, 12), dtype=np.float64)
    qpos[:, 7] = np.deg2rad([0.0, 10.0, 20.0])
    qpos[:, 8] = np.deg2rad([1.0, 2.0, 3.0])
    qpos[:, 9] = np.deg2rad([4.0, 5.0, 6.0])
    qpos[:, 10] = np.deg2rad([7.0, 8.0, 9.0])
    qpos[:, 11] = np.deg2rad([10.0, 11.0, 12.0])
    activations = np.array(
        [
            [0.10, 0.30, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80],
            [0.20, 0.40, 0.30, 0.40, 0.50, 0.60, 0.70, 0.90],
            [0.30, 0.50, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00],
        ],
        dtype=np.float64,
    )
    np.savez(
        path,
        n_episodes=np.array(2),
        joint_names=joint_names,
        muscle_names=muscle_names,
        episode_0_joint_positions=qpos,
        episode_0_muscle_activations=activations,
        episode_0_timesteps=np.array([0.0, 0.1, 0.2]),
        episode_1_joint_positions=qpos + 0.1,
        episode_1_muscle_activations=activations,
        episode_1_timesteps=np.array([0.0, 0.1, 0.2]),
    )


def test_export_named_rollout_matches_rag_csv_contract_columns(tmp_path):
    source = tmp_path / "rollout.npz"
    output = tmp_path / "rag.csv"
    _write_named_rollout(source)

    samples = export_npz_to_samples(source, impact_frame=1, include_all_muscles=False)
    write_samples_csv(samples, output)

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0].keys() == set(CSV_COLUMNS)
    assert [sample.split for sample in samples] == ["correct", "eval"]
    assert rows[0]["sample_id"] == "rollout_episode_000"
    assert rows[0]["action_type"] == "forehand_clear"
    assert rows[0]["event_impact"] == "0.1"
    assert float(rows[1]["joint_trunk_rotation"]) == 10.0
    assert float(rows[1]["joint_forearm_pronation"]) == 8.0
    assert float(rows[1]["muscle_external_oblique"]) == pytest.approx(0.30)
    assert float(rows[1]["muscle_triceps_brachii"]) == pytest.approx(0.50)
    assert float(rows[1]["muscle_forearm_pronator_group"]) == pytest.approx(0.80)


def test_export_legacy_354_activation_rollout_uses_disable_fingers_fallback(tmp_path):
    source = tmp_path / "legacy.npz"
    joint_names = np.array(
        ["root", "axial_rotation", "shoulder_rot_r", "elbow_flex_r", "pro_sup_r", "flexion_r"],
        dtype=str,
    )
    qpos = np.zeros((2, 12), dtype=np.float64)
    activations = np.zeros((2, 354), dtype=np.float64)
    activations[:, 22] = 0.2
    activations[:, 23] = 0.4
    activations[:, 210] = 0.5
    activations[:, 225] = 0.3
    activations[:, 226] = 0.6
    activations[:, 227] = 0.9
    activations[:, 240] = 0.1
    activations[:, 241] = 0.7
    np.savez(
        source,
        n_episodes=np.array(2),
        joint_names=joint_names,
        episode_0_joint_positions=qpos,
        episode_0_muscle_activations=activations,
        episode_0_timesteps=np.array([0.0, 0.1]),
        episode_1_joint_positions=qpos,
        episode_1_muscle_activations=activations,
        episode_1_timesteps=np.array([0.0, 0.1]),
    )

    samples = export_npz_to_samples(source, impact_frame=0, include_all_muscles=False)
    first_row = samples[0].rows[0]

    assert first_row["muscle_external_oblique"] == 0.30000000000000004
    assert first_row["muscle_anterior_deltoid"] == 0.5
    assert first_row["muscle_triceps_brachii"] == 0.6
    assert first_row["muscle_forearm_pronator_group"] == 0.39999999999999997


def test_export_includes_individual_muscle_activation_columns_by_default(tmp_path):
    source = tmp_path / "rollout.npz"
    output = tmp_path / "rag.csv"
    _write_named_rollout(source)

    samples = export_npz_to_samples(source, impact_frame=1)
    write_samples_csv(samples, output)

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert "muscle_myo_delt1" in rows[0]
    assert "muscle_myo_pt" in rows[0]
    assert "muscle_myo_pq" in rows[0]
    assert float(rows[1]["muscle_myo_delt1"]) == pytest.approx(0.30)
    assert float(rows[1]["muscle_myo_pt"]) == pytest.approx(0.70)
    assert float(rows[1]["muscle_myo_pq"]) == pytest.approx(0.90)
